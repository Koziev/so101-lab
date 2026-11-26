"""
Atomic commands of experiment 14: approach, grasp, lift, lower, release. Definitions shared by the trajectory
generator (dataset/record_commands.py) and, later, the evaluation.

The commands chain: approach -> grasp -> lift -> lower -> release (-> lift -> lower ...). An episode records ONE
command; the commands before it are played by the same expert without rendering, so the episode starts exactly
where the previous command left the arm (and the object). The expert keeps one nominal joint path across the
chain and one integral sag correction, so the commands of consecutive episodes join without jumps.

  approach  as in experiments 12-13 (approach.py); v14: the wrist roll turns the jaws onto a face of a cube or
            hex prism (approach.face_roll), so that the grasp closes on two parallel faces.
  grasp     from the hover pose of an approach (gripper closed, tip 1.5 cm above the object, pointing at its
            centre): shift   the gripper moves sideways (away from the moving jaw) until the fixed jaw's gripping
                             face is GRASP_MARGIN outside the object, so that the fixed jaw passes beside it;
            open    the gripper opens (OPEN_PCT, at least wide enough for the object);
            lower   the gripper descends along its axis until the fixed jaw's tip is GRASP_TIP_Z above the
                    board (higher if the object is taller than the fingers are long);
            close   the gripper closes; once the jaws are blocked by the object, SETTLE_STEPS more steps.
  lift      the tip rises vertically to LIFT_Z above the board, the gripper as it is (holding or open/empty).
  lower     holding: down until the object's lowest point is LOWER_OBJ_GAP above the board;
            empty: down until the tip is LOWER_TIP_Z above the board.
  release   the gripper opens (OPEN_PCT); the arm does not move. A held object falls.
Every command ends with a hold (HOLD_S, or HOLD_LONG_S in a share of the episodes), so that the policy learns to
stand still when the command is done. Phases (data.npz "phase") index PHASES.
"""
import math

import numpy as np

import approach as ap

COMMANDS = ("approach", "grasp", "lift", "lower", "release")
PHASES = ["transit", "descend", "hold", "shift", "open", "lower", "close", "rise", "release"]
PH = {p: i for i, p in enumerate(PHASES)}

# chains that lead to the start of each command; the variants and their shares
CHAINS = {
    "approach": [("from_start", 1.0, [])],
    "grasp":    [("after_approach", 1.0, ["approach"])],
    "lift":     [("holding", 0.7, ["approach", "grasp"]),
                 ("empty_open", 0.3, ["approach", "grasp", "lift", "lower", "release"])],
    "lower":    [("holding", 0.7, ["approach", "grasp", "lift"]),
                 ("empty_open", 0.3, ["approach", "grasp", "lift", "lower", "release", "lift"])],
    "release":  [("low", 0.7, ["approach", "grasp", "lift", "lower"]),
                 ("lifted", 0.3, ["approach", "grasp", "lift"])],
}

GRASP_MARGIN = (0.004, 0.007)   # m, fixed jaw's face to the object surface while descending
GRASP_TIP_Z = (0.006, 0.010)    # m, fixed jaw's tip above the board at the grasp
GRASP_BOARD_CLEAR = 0.003       # m, exp14: gripper links (open and closed) above the board at the grasp pose
GRASP_MIN_DEPTH = 0.010         # m, ... raising the grasp for that stops this far below the object's top
GRASP_RAISE_TRIES = 4
GRASP_SHIFT_CLEAR = 0.004       # m, exp14: arm to object during the shift and the opening (the shift is not
GRASP_SHIFT_RAISE = (0.0, 0.004, 0.008, 0.012, 0.016)   # sag-corrected); shift target moved back along the axis
LOWER_GRIPPER_CLEAR = 0.004     # m, exp14: a lower stops with the gripper links at least this far above the board
VERTICAL_SIDESTEP = (0.0, 0.003, 0.006, 0.009, 0.012)   # m, exp14: an empty lower moves this far away from the object
VERTICAL_OBJ_CLEAR = 0.003      # m, exp14: an empty lift / lower keeps the arm this far from the objects
PROBE_BIG = 1.0
FINGER_SPARE = 0.005            # m, the object's top stays this far below the end of the jaws' gap
OPEN_PCT = (55.0, 85.0)         # gripper opening for grasp and release (at least wide enough for the object)
OPEN_CLEARANCE = 0.008          # m, free gap beyond object + margin when the open gripper descends
SHIFT_S = (0.4, 0.6)
OPEN_S = (0.3, 0.5)
CLOSE_S = (0.4, 0.6)            # ramp of the gripper command to closed
CLOSE_MAX_S = 1.0               # then at most this long until the jaws are blocked ...
SETTLE_STEPS = 3                # ... and this many steps after they are
MOVE_SPEED = (0.04, 0.07)       # m/s, Cartesian moves (grasp descent, lift, lower)
LIFT_Z = 0.10                   # m, tip above the board after a lift
LOWER_TIP_Z = 0.0125            # m, empty gripper: tip above the board after a lower
LOWER_OBJ_GAP = 0.004           # m, holding: object's lowest point above the board after a lower
RELEASE_S = (0.3, 0.5)
PREFIX_HOLD_S = (0.2, 0.5)      # hold at the end of a command played before the recorded one

# success of a command (its end state)
LIFT_TOL = 0.015                # m, tip height
LOWER_TIP_RANGE = (0.007, 0.020)
LOWER_OBJ_RANGE = (0.0, 0.012)
RELEASE_MIN_PCT = 30.0
STILL_TOL = 0.01                # m, release: the tip may move this much
TILT_MAX_DEG = 25.0             # grasp: the object may tilt this much in the jaws

OBJECT_VERBS = {
    "grasp":   ["grasp {np}", "grab {np}", "take hold of {np}", "close the gripper on {np}", "seize {np}"],
    "lift":    ["lift {np}", "raise {np}", "lift {np} up"],
    "lower":   ["lower {np}", "bring {np} down", "move {np} down"],
    "release": ["release {np}", "drop {np}", "let go of {np}"],
}
GENERIC = {
    "lift":    ["lift", "lift the gripper", "raise the gripper", "move up", "go up"],
    "lower":   ["lower the gripper", "lower", "move down", "go down"],
    "release": ["release", "open the gripper", "let go", "open"],
}
HELD_GENERIC = {"lift": ["lift it", "raise it"], "lower": ["lower it", "bring it down"],
                "release": ["drop it", "let it go", "release it"]}


# ------------------------------------------------------------------------------------------------ helpers
def board_z(env):
    return float(env.pb.getAABB(env.table)[1][2])


def chain_for(command, seed):
    """(variant, prefix commands) of an episode, from its own random stream."""
    u = np.random.default_rng([int(seed), 14, 2]).random()
    acc = 0.0
    for name, share, prefix in CHAINS[command]:
        acc += share
        if u < acc:
            return name, list(prefix)
    name, _, prefix = CHAINS[command][-1]
    return name, list(prefix)


def instruction(env, command, holding, rng):
    """Prompt of a command. approach: the scene's prompt; grasp: names the object; lift / lower / release: the
    object (when held, half of the time), "it" or a plain gripper command."""
    import tasks
    if command == "approach":
        return env.task["instruction"]
    tgt = env.task["specs"][0]
    if command == "grasp" or (holding and rng.random() < 0.5):
        return tasks.phrase(env.task, tgt, rng, True, verbs=OBJECT_VERBS[command])["instruction"]
    words = GENERIC[command] + (HELD_GENERIC[command] if holding else [])
    return words[int(rng.integers(len(words)))]


HOLD_TOUCH = 0.0005    # m, a jaw counts as touching the object within this contactDistance (a firm grip
                       # alternates between slight penetration and contact)


def contacts(env, obj):
    """(fixed jaw touches obj, moving jaw touches obj)."""
    pb = env.pb
    return (ap.touching(pb.getContactPoints(env.robot, obj["id"], linkIndexA=env.g_link), tol=HOLD_TOUCH),
            ap.touching(pb.getContactPoints(env.robot, obj["id"], linkIndexA=env.jaw_link), tol=HOLD_TOUCH))


def holding(env, obj):
    f, m = contacts(env, obj)
    return f and m


def gripper_board_contact(env):
    links = (env.g_link, env.jaw_link) + tuple(getattr(env, "cam_links", ()))
    return ap.touching(env.pb.getContactPoints(env.robot, env.table), links)


def object_tilt_deg(env, obj):
    _, orn = env.pb.getBasePositionAndOrientation(obj["id"])
    R = np.array(env.pb.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))


def object_low_z(env, obj):
    return float(ap.object_vertices(env, obj)[1][:, 2].min())


def finger_length(kin):
    return float(kin.tip_depth - np.min(kin.probe_depths))


# ------------------------------------------------------------------------------------------------ expert
class ChainExpert:
    """One nominal joint path through a chain of commands; act() adds the integral sag correction (all phases
    but the approach's transit), as ApproachExpert does."""

    def __init__(self, env, rng, ready_deg5=None, long_hold_p=0.0, align_roll=True):
        self.env, self.rng, self.ready = env, rng, ready_deg5
        self.long_hold_p, self.align_roll = long_hold_p, align_roll
        self.steps, self.t = [], 0
        self.q_nom, self.grip, self.corr = None, ap.GRIPPER_CLOSED_PCT, np.zeros(5)
        self.info, self.goal, self.long_hold = {}, None, False

    # -------------------------------------------------------------- state
    def start(self, q_deg5, grip_pct=ap.GRIPPER_CLOSED_PCT):
        self.q_nom, self.grip, self.corr = np.asarray(q_deg5, float).copy(), float(grip_pct), np.zeros(5)

    def done(self):
        return self.t >= len(self.steps)

    def _pose(self, q_deg5):
        """Tip, axis (world) and wrist roll (rad) of a nominal pose, from the kinematic model."""
        env = self.env
        q = env.from_lerobot(np.r_[q_deg5, ap.GRIPPER_CLOSED_PCT])[:5]
        tip, _, rot = env.kin.points(q, ap.tip_local(env))
        return np.asarray(tip), rot @ env.kin.approach_local, float(q[4]), rot

    def _line(self, q_deg5, tip_to, n=8):
        """Joint waypoints (deg) moving the tip in a straight line, axis and roll kept; None if IK fails."""
        env = self.env
        tip0, a, roll, _ = self._pose(q_deg5)
        q = env.from_lerobot(np.r_[q_deg5, ap.GRIPPER_CLOSED_PCT])[:5]
        out, local = [np.asarray(q_deg5, float)], ap.tip_local(env)
        for s in np.linspace(0, 1, n + 1)[1:]:
            q, ok, _, _ = env.kin.solve(tip0 + (tip_to - tip0) * s, a, roll, [q], local)
            if not ok:
                return None
            out.append(env.to_lerobot(np.r_[q, env.kin.gripper_closed])[:5])
        return out

    def _timed(self, waypoints, duration):
        from so101_sim.expert import sample_path
        return sample_path(waypoints, duration, self.env.cfg.control_hz)

    def _hold_steps(self, prefix):
        hz = self.env.cfg.control_hz
        if prefix:
            self.long_hold = False
            return int(round(self.rng.uniform(*PREFIX_HOLD_S) * hz))
        self.long_hold = bool(self.rng.random() < self.long_hold_p)
        return int(round(self.rng.uniform(*(ap.HOLD_LONG_S if self.long_hold else ap.HOLD_S)) * hz))

    # -------------------------------------------------------------- planning
    def plan(self, command, target, prefix=False, direct=False):
        """Plan `command` from the current nominal state; False if it cannot be planned. prefix: a command played
        before the recorded one (short hold)."""
        self.steps, self.t, self.info = [], 0, {"command": command}
        self._blocked_at, self._grip_hist, self._close_end = None, [], None
        ok = getattr(self, "_plan_" + command)(target, direct) if command == "approach" else \
            getattr(self, "_plan_" + command)(target)
        if not ok:
            return False
        q_last, g_last = self.steps[-1][0], self.steps[-1][1]
        self.steps += [(q_last, g_last, PH["hold"])] * self._hold_steps(prefix)
        return True

    def _plan_approach(self, target, direct=False):
        ex = ap.ApproachExpert(self.env, self.rng, self.ready, 0.0, align_roll=self.align_roll)
        if not ex.reset(self.q_nom, target, direct=direct):
            return False
        self.goal = ex.goal
        moving = [(q, ph) for q, ph in ex.plan if ph != 2]          # the hold is added by plan()
        self.steps = [(np.asarray(q, float), ap.GRIPPER_CLOSED_PCT, ph) for q, ph in moving]
        self.info.update({"tilt_deg": ex.goal["tilt"]})
        return True

    def _plan_grasp(self, target):
        env, r = self.env, self.rng
        q0 = self.q_nom
        tip0, a, roll, rot = self._pose(q0)
        m = env.kin.grasp_sign * (rot @ env.kin.closing_local)             # fixed jaw -> moving jaw
        m = m - a * float(m @ a)
        m /= np.linalg.norm(m) + 1e-12
        centre, verts = ap.object_vertices(env, target)
        w_fixed = float(np.max((verts - centre) @ (-m)))
        w_moving = float(np.max((verts - centre) @ m))
        margin = r.uniform(*GRASP_MARGIN)
        d0 = float((tip0 - centre) @ m)
        tip_shift = tip0 + m * (-(w_fixed + margin) - d0)
        z_b = board_z(env)
        top = float(verts[:, 2].max())
        tip_z = max(z_b + r.uniform(*GRASP_TIP_Z), top - finger_length(env.kin) + FINGER_SPARE)
        if a[2] > -0.3:
            return False
        min_pct = env.kin.min_open_pct(w_fixed + w_moving + margin, clearance=OPEN_CLEARANCE)
        if min_pct is None:
            return False
        open_pct = r.uniform(max(OPEN_PCT[0], min_pct), max(OPEN_PCT[1], min_pct))
        # exp14: the shift (closed) and the opening must keep GRASP_SHIFT_CLEAR between the arm and the object;
        # otherwise the shift target is raised back along the gripper axis (which keeps the jaw's offset to the
        # object), by GRASP_SHIFT_RAISE steps.
        shift = None
        for k in range(len(GRASP_SHIFT_RAISE)):
            back = GRASP_SHIFT_RAISE[k]
            cand = self._line(q0, tip_shift - a * back, n=6)
            if cand is None:
                return False
            with ap.pose_probe(env) as p:
                ok = True
                for q in cand[1:]:
                    p.set(q, ap.GRIPPER_CLOSED_PCT)
                    ok = ok and p.body(target["id"]) >= GRASP_SHIFT_CLEAR
                for pct in (0.5 * (ap.GRIPPER_CLOSED_PCT + open_pct), open_pct):
                    p.set(cand[-1], pct)
                    ok = ok and p.body(target["id"]) >= GRASP_SHIFT_CLEAR
            if ok:
                shift, tip_shift = cand, tip_shift - a * back
                break
        if shift is None:
            return False
        self.info["shift_back_m"] = float(back)
        raised = 0.0
        for _ in range(GRASP_RAISE_TRIES):
            s = (tip_shift[2] - tip_z) / (-a[2])
            if s <= 0.005:
                return False
            tip_grasp = tip_shift + a * s
            down = self._line(shift[-1], tip_grasp, n=8)
            if down is None:
                return False
            with ap.pose_probe(env) as p:
                clear = PROBE_BIG
                for q in down[-3:]:
                    p.set(q, open_pct)
                    clear = min(clear, p.board())
                p.set(down[-1], ap.GRIPPER_CLOSED_PCT)
                clear = min(clear, p.board())
            if clear >= GRASP_BOARD_CLEAR:
                break
            dz = GRASP_BOARD_CLEAR - clear + 0.0005
            if tip_z + dz > top - GRASP_MIN_DEPTH:
                return False
            tip_z += dz
            raised += dz
        else:
            return False
        hz = env.cfg.control_hz
        closed = ap.GRIPPER_CLOSED_PCT
        steps = [(q, closed, PH["shift"]) for q in self._timed(shift, r.uniform(*SHIFT_S))]
        q_s = steps[-1][0]
        n_open = max(2, int(round(r.uniform(*OPEN_S) * hz)))
        steps += [(q_s, closed + (open_pct - closed) * (k + 1) / n_open, PH["open"]) for k in range(n_open)]
        steps += [(q, open_pct, PH["lower"]) for q in self._timed(down, max(0.5, s / r.uniform(*MOVE_SPEED)))]
        q_g = steps[-1][0]
        n_close = max(2, int(round(r.uniform(*CLOSE_S) * hz)))
        steps += [(q_g, open_pct + (closed - open_pct) * (k + 1) / n_close, PH["close"]) for k in range(n_close)]
        steps += [(q_g, closed, PH["close"])] * int(round(CLOSE_MAX_S * hz))
        self._close_ramp_end = len(steps) - int(round(CLOSE_MAX_S * hz))
        self._close_end = len(steps)
        self.steps = steps
        self.info.update({"margin_m": margin, "open_pct": open_pct, "tip_grasp_z": tip_z - z_b,
                          "raised_m": raised, "board_clear_m": clear,
                          "width_m": w_fixed + w_moving, "descent_m": s})
        return True

    def _vertical(self, target_tip_z, phase, exclude=None, side=None):
        """Vertical tip move to target_tip_z. exp14: the nominal path keeps VERTICAL_OBJ_CLEAR between the arm and
        every object but `exclude` (a held object), or no less than at the start (an empty lift starts beside the
        object it released); if not, and `side` (unit horizontal vector) is given, the end point is moved
        sideways by VERTICAL_SIDESTEP steps until it is; False otherwise."""
        env, r = self.env, self.rng
        tip0, _, _, _ = self._pose(self.q_nom)
        dz = target_tip_z - tip0[2]
        if abs(dz) < 0.01:
            return False
        ids = [o["id"] for o in env.objects if exclude is None or o["id"] != exclude]
        steps = VERTICAL_SIDESTEP if side is not None else (0.0,)
        path = None
        for st in steps:
            end = tip0 + np.array([0, 0, dz]) + (np.asarray(side, float) * st if side is not None else 0.0)
            cand = self._line(self.q_nom, end, n=8)
            if cand is None:
                continue
            with ap.pose_probe(env) as p:
                p.set(cand[0], self.grip)
                need = {i: min(VERTICAL_OBJ_CLEAR, p.body(i) - 0.001) for i in ids}
                ok = True
                for q in cand[1:]:
                    p.set(q, self.grip)
                    if any(p.body(i) < need[i] for i in ids):
                        ok = False
                        break
            if ok:
                path = cand
                self.info["sidestep_m"] = float(st)
                break
        if path is None:
            return False
        dur = max(0.6, abs(dz) / r.uniform(*MOVE_SPEED))
        self.steps = [(q, self.grip, PH[phase]) for q in self._timed(path, dur)]
        self.info.update({"dz_m": float(dz)})
        return True

    def _plan_lift(self, target):
        held = holding(self.env, target)
        self.info["holding"] = held
        return self._vertical(board_z(self.env) + LIFT_Z, "rise", exclude=target["id"] if held else None)

    def _plan_lower(self, target):
        env = self.env
        held = holding(env, target)
        self.info["holding"] = held
        tip_now = ap.tip_world(env)                    # physical
        tip_nom, _, _, _ = self._pose(self.q_nom)
        if held:
            drop = object_low_z(env, target) - (board_z(env) + LOWER_OBJ_GAP)
        else:
            drop = tip_now[2] - (board_z(env) + LOWER_TIP_Z)
        # exp14: the gripper links (physical, now) stop LOWER_GRIPPER_CLEAR above the board
        drop = min(drop, ap.gripper_board_distance(env) - LOWER_GRIPPER_CLEAR)
        self.info["gripper_limited"] = bool(drop == ap.gripper_board_distance(env) - LOWER_GRIPPER_CLEAR)
        # exp14: the drop is measured on the physical arm, so it is applied to the physical tip: the sag correction
        # brings the readings onto the nominal pose, and a nominal target taken from the nominal tip came out lower
        # by whatever the arm had not yet caught up (3-4 mm: held objects pressed into the board).
        side = None
        if not held:                                   # empty: step sideways, away from the object below, if needed
            c, _ = ap.object_vertices(env, target)
            away = np.r_[tip_now[:2] - c[:2], 0.0]
            side = away / (np.linalg.norm(away) + 1e-9)
        return self._vertical(tip_now[2] - drop, "lower", exclude=target["id"] if held else None, side=side)

    def _plan_release(self, target):
        r, hz = self.rng, self.env.cfg.control_hz
        self.info["holding"] = holding(self.env, target)
        open_pct = r.uniform(*OPEN_PCT)
        n = max(2, int(round(r.uniform(*RELEASE_S) * hz)))
        g0 = self.grip
        self.steps = [(self.q_nom.copy(), g0 + (open_pct - g0) * (k + 1) / n, PH["release"]) for k in range(n)]
        self.info.update({"open_pct": open_pct, "tip_start": ap.tip_world(self.env).tolist()})
        return True

    # -------------------------------------------------------------- execution
    def act(self, obs):
        """Command (6 values, LeRobot units) and phase index for this step."""
        q_nom, grip, ph = self.steps[self.t]
        state = np.asarray(obs["state"], float)
        if ph != PH["transit"]:
            err = ap.comp_reference([st[0] for st in self.steps], self.t) - state[:5]
            err[0] = err[4] = 0.0                                       # pan and roll do not sag
            self.corr = np.clip(self.corr + ap.COMP_GAIN * err, -ap.COMP_LIMIT, ap.COMP_LIMIT)
        self.q_nom, self.grip = np.asarray(q_nom, float), float(grip)
        if ph == PH["close"] and self._close_end is not None and self.t >= self._close_ramp_end:
            self._grip_hist.append(float(state[5]))
            h = self._grip_hist[-3:]
            if self._blocked_at is None and len(h) == 3 and max(h) - min(h) < 0.5 and h[-1] - grip > 5.0:
                self._blocked_at = self.t
            if self._blocked_at is not None and self.t - self._blocked_at >= SETTLE_STEPS:
                self.info["blocked_pct"] = h[-1]
                self.t = self._close_end                               # on to the hold
                return np.r_[q_nom + self.corr, grip], ph
        self.t += 1
        return np.r_[q_nom + self.corr, grip], ph


# ------------------------------------------------------------------------------------------------ success
def evaluate(env, command, target, expert, start):
    """(success, metrics) of a command at the end of its episode. start: what was measured when it began
    (tip, holding)."""
    tip = ap.tip_world(env)
    z_b = board_z(env)
    state_grip = float(env.observe(render=False)["state"][5])
    held = holding(env, target)
    board = gripper_board_contact(env)
    m = {"tip_z_cm": 100 * (tip[2] - z_b), "holding": held, "gripper_pct": state_grip, "board_contact": board}
    if command == "approach":
        err = ap.evaluate_pose(env, expert.goal)
        c_obj, c_board = ap.arm_contact(env)
        m.update(err)
        return bool(ap.success(err, c_obj or c_board, state_grip)), m
    if command == "grasp":
        tilt = object_tilt_deg(env, target)
        m.update({"object_tilt_deg": tilt})
        return bool(held and not board and tilt <= TILT_MAX_DEG and state_grip > ap.GRIPPER_CLOSED_PCT + 2), m
    if command == "lift":
        ok = abs(tip[2] - z_b - LIFT_Z) <= LIFT_TOL and not board
        if start["holding"]:
            low = object_low_z(env, target) - z_b
            m["object_low_cm"] = 100 * low
            ok = ok and held and low > 0.03
        else:
            ok = ok and state_grip > RELEASE_MIN_PCT
        return bool(ok), m
    if command == "lower":
        if start["holding"]:
            low = object_low_z(env, target) - z_b
            m["object_low_cm"] = 100 * low
            ok = held and LOWER_OBJ_RANGE[0] <= low <= LOWER_OBJ_RANGE[1]
        else:
            ok = LOWER_TIP_RANGE[0] <= tip[2] - z_b <= LOWER_TIP_RANGE[1] and state_grip > RELEASE_MIN_PCT
        return bool(ok and not board), m
    if command == "release":
        moved = float(np.linalg.norm(tip - np.asarray(start["tip"])))
        f, mv = contacts(env, target)
        m.update({"tip_moved_cm": 100 * moved})
        return bool(state_grip >= RELEASE_MIN_PCT and not (f and mv) and moved <= STILL_TOL and not board), m
    raise ValueError(command)
