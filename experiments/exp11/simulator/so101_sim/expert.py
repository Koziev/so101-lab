"""
Scripted grasping expert. Uses privileged object poses to plan, then replays the plan
open-loop through the same env.step() interface a learned policy will use.

Phases: approach (to pre-grasp above the object, opening the gripper), descend (straight
line along the approach axis), close, lift (straight up), hold.
"""
import math
from dataclasses import dataclass

import numpy as np

from .kinematics import unit

PHASES = ["approach", "descend", "close", "lift", "hold"]


def min_jerk(s):
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def sample_path(waypoints, duration, hz):
    """Min-jerk timing along a piecewise-linear joint-space path (all joints in sync)."""
    wp = np.asarray(waypoints, float)
    n = max(2, int(round(duration * hz)))
    out = []
    for i in range(1, n + 1):
        u = min_jerk(i / n) * (len(wp) - 1)
        j = min(int(u), len(wp) - 2)
        f = u - j
        out.append(wp[j] * (1 - f) + wp[j + 1] * f)
    return out


def smooth_offsets(rng, n_active, n_total, hz, peak, n_bumps, width_s, fade):
    """
    Horizontal offset curve (n_total x 2, metres). Sum of Gaussian bumps in random
    directions over the first n_active steps, scaled to the given peak, ramped in at the
    start and faded to zero `fade[0]`..`fade[1]` steps before n_active; zero afterwards.
    """
    t = np.arange(n_total, dtype=float)
    off = np.zeros((n_total, 2))
    for _ in range(n_bumps):
        c = rng.uniform(0.15, 0.85) * n_active
        sig = rng.uniform(*width_s) * hz
        ang = rng.uniform(0, 2 * math.pi)
        off += rng.uniform(0.5, 1.0) * np.outer(np.exp(-0.5 * ((t - c) / sig) ** 2), [math.cos(ang), math.sin(ang)])
    win = np.clip(t / 10.0, 0, 1)                                         # ramp in
    win *= np.clip((n_active - fade[1] - t) / max(1, fade[0] - fade[1]), 0, 1)  # fade out
    off *= win[:, None]
    m = np.max(np.linalg.norm(off, axis=1))
    return off * (peak / m) if m > 1e-9 else off


@dataclass
class ExpertConfig:
    joint_speed: tuple = (0.6, 1.2)       # rad/s, peak velocity of the largest joint move
    cart_speed: tuple = (0.04, 0.08)      # m/s, average speed of descend / lift
    pregrasp_height: tuple = (0.05, 0.09)
    lift_height: tuple = (0.08, 0.12)
    open_pct: tuple = (60.0, 100.0)
    close_time: float = 0.8
    hold_time: float = 0.6
    tilts_deg: tuple = (0, 15, 30, 45)    # approach tilt from vertical, tried in order
    grasp_depth: tuple = (-0.003, 0.003)  # TCP height jitter around the object centre
    p_offset: float = 0.0                 # probability of a deliberately offset grasp
    offset_range: tuple = (0.015, 0.04)
    table_clearance: float = 0.003
    path_points: int = 6
    # Perturbed demonstrations (DART-style): the executed motion is pushed sideways while
    # the recorded actions stay the clean plan, so the data shows how to correct offsets.
    p_perturb: float = 0.0               # share of episodes with perturbation
    perturb_peak: tuple = (0.01, 0.03)   # peak horizontal gripper offset, m
    perturb_bumps: tuple = (1, 3)        # number of smooth pushes (inclusive range)
    perturb_width_s: tuple = (0.25, 0.6) # duration scale (std) of one push, s
    perturb_fade_steps: tuple = (12, 4)  # offset fades to 0 between these step counts before closing
    # Early lift: during "close", watch the measured gripper; once the jaws are blocked by the
    # object (not moving, and clearly behind the command), wait settle_steps and start lifting.
    # Removes the long motionless wait whose frames are ambiguous for a memoryless policy.
    early_lift: bool = False
    blocked_steps: int = 2          # gripper unchanged for this many steps ...
    blocked_tol_pct: float = 0.5    # ... within this tolerance (gripper % units)
    blocked_gap_pct: float = 5.0    # and the command is at least this much below the measured value
    settle_steps: int = 2           # steps to keep squeezing before the lift starts


class ScriptedGraspExpert:
    def __init__(self, env, cfg=None, rng=None):
        self.env, self.cfg = env, cfg or ExpertConfig()
        self.rng = rng or np.random.default_rng()
        self.actions, self.phases, self.t = [], [], 0
        self.plan_info = {}
        self.perturb = None

    def reset(self, obs, info):
        """Plan an episode. Returns False if the target is unreachable."""
        self.t = 0
        plan = self._plan(obs, info)
        if plan is None:
            self.actions, self.phases = [], []
            return False
        self.actions, self.phases = plan
        self.perturb = self._make_perturbation(info)
        self._grip_hist, self._blocked_at = [], None
        self._comp, self._prev_state, self._last_desired = np.zeros(6), None, None
        self._ff = np.zeros(6)
        return True

    def executed(self, action):
        """Action to send to the robot for the step just returned by act() (clean if unperturbed)."""
        out = np.asarray(action) if self.perturb is None else np.asarray(action) + self.perturb[self.t - 1]
        if self.perturb is not None and self._last_desired is not None:
            self._last_desired = self._last_desired + self.perturb[self.t - 1]   # where the joint should go
        return out

    def act(self, obs):
        """Next action in LeRobot units and its phase label, or (None, None) when done."""
        if self.t >= len(self.actions):
            return None, None
        if self.cfg.early_lift and self.phases[self.t] == "close":
            self._check_blocked(obs)
        a, ph = np.array(self.actions[self.t], float), self.phases[self.t]
        desired = a.copy()
        if getattr(self.env, "v8", False) and self.env.act_params.get("active"):
            a = self._compensate(np.asarray(obs["state"], float), a)
        self._last_desired = desired
        self.t += 1
        return a, ph

    def _feedforward(self, q, a, eps=0.05):
        """Privileged anticipation of the simulated dead band (the expert may know it, the policy may not):
        a joint approaching its target in the lagging (negative) direction will stop `lag` short, so command
        `lag` further; approaching from below it stops `pos_deg` short. Inside the band: keep the last value."""
        env, p = self.env, self.env.act_params
        q_rad = env.from_lerobot(q)
        zeros = [0.0] * len(q_rad)
        tau = np.array(env.pb.calculateInverseDynamics(env.robot, [float(v) for v in q_rad], zeros, zeros))
        for k, j in enumerate((1, 2)):
            lag = p["k_deg_per_nm"][k] * abs(tau[j])
            if q[j] > a[j] + eps:
                self._ff[j] = -lag
            elif q[j] < a[j] - eps:
                self._ff[j] = p["pos_deg"]
        return self._ff

    def _compensate(self, q, a, gain=0.3, settled_deg=0.3, limit=12.0):
        """Correction on shoulder_lift / elbow_flex: privileged anticipation of the dead band plus an
        integral term for what the anticipation misses. The recorded labels contain the compensation, so
        the policy learns commands that work on a lagging arm."""
        ff = self._feedforward(q, a)
        if self._prev_state is not None and self._last_desired is not None:
            for j in (1, 2):
                if abs(q[j] - self._prev_state[j]) < settled_deg:
                    err = q[j] - self._last_desired[j]       # settled away from where it should be
                    self._comp[j] = float(np.clip(self._comp[j] - gain * err, -limit, limit))
        self._prev_state = q.copy()
        return a + ff + self._comp

    def _check_blocked(self, obs):
        """Skip the rest of the close phase once the jaws have been blocked for settle_steps."""
        c = self.cfg
        g = float(obs["state"][5])
        self._grip_hist.append(g)
        last_cmd = float(self.actions[self.t - 1][5]) if self.t > 0 else 100.0
        h = self._grip_hist[-(c.blocked_steps + 1):]
        if self._blocked_at is None:
            if (len(h) == c.blocked_steps + 1 and max(h) - min(h) < c.blocked_tol_pct
                    and g - last_cmd > c.blocked_gap_pct):
                self._blocked_at = self.t
        elif self.t - self._blocked_at >= c.settle_steps:
            first_lift = self.phases.index("lift")
            self.plan_info["early_lift"] = {"blocked_step": self._blocked_at, "lift_step": self.t,
                                            "skipped_close_steps": first_lift - self.t,
                                            "gripper_pct_blocked": round(g, 1)}
            self.t = first_lift

    # ----------------------------------------------------------------- planning
    def _plan(self, obs, info):
        env, kin, c, r = self.env, self.env.kin, self.cfg, self.rng
        tgt = info["target"]
        p = np.array(tgt["pos"], float)
        offset = np.zeros(3)
        if r.random() < c.p_offset:
            ang = r.uniform(0, 2 * math.pi)
            offset[:2] = r.uniform(*c.offset_range) * np.array([math.cos(ang), math.sin(ang)])
        grasp = p + offset
        grasp[2] = p[2] + r.uniform(*c.grasp_depth)
        radial = unit(np.r_[p[:2] - env.pan_axis[:2], 0.0])
        local = kin.grasp_local(tgt["size"])
        min_pct = kin.min_open_pct(tgt["size"])
        if min_pct is None:  # object wider than the open gripper
            return None
        open_pct = r.uniform(max(c.open_pct[0], min_pct), max(c.open_pct[1], min_pct))
        open_rad = kin.gripper_pct_to_rad(open_pct)
        closed_rad = kin.gripper_closed
        h_pre, h_lift = r.uniform(*c.pregrasp_height), r.uniform(*c.lift_height)
        seeds = [env.q_home[:5], obs["q"][:5], np.zeros(5)]

        for tilt in c.tilts_deg:
            t = math.radians(tilt)
            a = unit(radial * math.sin(t) + np.array([0, 0, -math.cos(t)]))
            res = self._grasp_pose(grasp, a, tgt["yaw"], seeds, open_rad, local)
            if res is None:
                continue
            q_grasp, grasp_pt = res
            down = self._line(grasp_pt, -a, h_pre, a, q_grasp, local)      # grasp -> pre-grasp
            up = self._line(grasp_pt, np.array([0, 0, 1.0]), h_lift, a, q_grasp, local) if down else None
            if down and up:
                break
        else:
            return None

        self.plan_info = {"tilt_deg": tilt, "offset_m": offset.tolist(), "grasp_tcp": grasp_pt.tolist(),
                          "perturbed": bool(np.any(offset)), "roll_deg": float(np.degrees(q_grasp[4])),
                          "open_pct": float(open_pct), "grasp_local": local.tolist()}

        hz = env.cfg.control_hz
        seq, phases = [], []

        def add(waypoints, duration, phase):
            for q in sample_path(waypoints, duration, hz):
                seq.append(env.to_lerobot(q))
                phases.append(phase)

        start = obs["q"].copy()
        pre = np.r_[down[-1], open_rad]
        dist = np.max(np.abs(pre[:5] - start[:5]))
        add([start, pre], max(0.8, 1.875 * dist / r.uniform(*c.joint_speed)), "approach")
        v = r.uniform(*c.cart_speed)
        add([np.r_[q, open_rad] for q in reversed(down)], max(0.5, h_pre / v), "descend")
        add([np.r_[q_grasp, open_rad], np.r_[q_grasp, closed_rad]], c.close_time, "close")
        lift = [np.r_[q, closed_rad] for q in up]
        add(lift, max(0.5, h_lift / v), "lift")
        add([lift[-1], lift[-1]], c.hold_time, "hold")
        return seq, phases

    def _grasp_pose(self, pt, a, yaw, seeds, open_rad, local):
        kin, c = self.env.kin, self.cfg
        q5, ok, *_ = kin.solve(pt, a, 0.0, seeds, local)
        if not ok:
            return None
        roll = kin.best_roll(q5, yaw, 90.0)
        q5, ok, *_ = kin.solve(pt, a, roll, [q5], local)
        if not ok:
            return None
        low = kin.min_gripper_z(np.r_[q5, open_rad])
        if low < c.table_clearance:  # open jaws would hit the table: raise the grasp point
            pt = pt + np.array([0, 0, c.table_clearance - low])
            q5, ok, *_ = kin.solve(pt, a, roll, [q5], local)
            if not ok:
                return None
        return q5, pt

    def _line(self, start, direction, length, approach, q_start, local):
        """IK along a straight Cartesian segment, warm-started point by point."""
        qs, q = [q_start], q_start
        for s in np.linspace(0, 1, self.cfg.path_points)[1:]:
            q, ok, *_ = self.env.kin.solve(start + direction * length * s, approach, q_start[4], [q], local)
            if not ok:
                return None
            qs.append(q)
        return qs

    # ----------------------------------------------------------------- perturbation
    def _make_perturbation(self, info):
        """Joint-space offsets (LeRobot units) that move the grasp point horizontally."""
        c, r, env = self.cfg, self.rng, self.env
        self.plan_info["perturbed"] = False
        if c.p_perturb <= 0 or r.random() >= c.p_perturb:
            return None
        n_total = len(self.actions)
        n_active = sum(p in ("approach", "descend") for p in self.phases)
        peak = r.uniform(*c.perturb_peak)
        off = smooth_offsets(r, n_active, n_total, env.cfg.control_hz, peak,
                             int(r.integers(c.perturb_bumps[0], c.perturb_bumps[1] + 1)), c.perturb_width_s,
                             c.perturb_fade_steps)
        local = np.asarray(self.plan_info["grasp_local"])
        kin = env.kin
        dq = np.zeros((n_total, 6))
        for t in range(n_active):
            if np.linalg.norm(off[t]) < 1e-5:
                continue
            q = env.from_lerobot(self.actions[t])
            p0 = kin.points(q[:5], local)[0]
            J = np.empty((3, 4))
            for j in range(4):  # pan, lift, elbow, wrist_flex; roll and gripper untouched
                qj = q[:5].copy()
                qj[j] += 1e-4
                J[:, j] = (kin.points(qj, local)[0] - p0) / 1e-4
            d = np.linalg.pinv(J) @ np.array([off[t, 0], off[t, 1], 0.0])  # keep height on plan
            dq[t, :4] = np.degrees(d)
        self.plan_info.update({"perturbed": True, "perturb_peak_m": float(peak)})
        return dq
