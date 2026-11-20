"""
Closed-loop control, shared definitions (experiment 11, second stage).

One call of the model = one small step: it sees the stereo pair with the arm where it is NOW, gets the
instruction and the current servo angles, and answers with the CHANGE of the four servo angles for the next step
and with the gripper decision:

    {"shoulder_pan": 1.6, "shoulder_lift": -0.4, "elbow_flex": 2.1, "wrist_flex": 0.0, "gripper": "open"}

The controller (simulated in evaluation/evaluate_cl.py, real in real/run_closed_loop.py) keeps a COMMAND per
joint and adds the model's change to it:   command <- command + change.   It does NOT set command = reading +
change: the servos give way under load (the shoulder up to 4 deg at full reach), and reading + change would
settle short of the target by exactly that amount. Adding to the command integrates the remaining error away.

The expert that makes the training labels (stateless, like the model), version 2:
  - far from the target (grasp point more than NEAR_XY away horizontally): head for the pre-grasp angles
    q_pre (the one-shot label's pose, PRE_UP higher, same tilt) on a straight line in joint space, so that the
    gripper arrives above the object instead of ploughing through the scene;
  - near the target: move the GRASP POINT straight to the target position, keeping the gripper's pitch as it
    is (lift + elbow + wrist unchanged). The change is the unique joint change that does this;
  - the largest joint change of a step is limited to STEP_MAX;
  - gripper = "close" when the grasp point is within CLOSE_XY (horizontally) and CLOSE_Z (vertically) of the
    target, otherwise "open".
Changes are written with one decimal: finer than the servos resolve.

Why version 2: version 1 defined "arrived" and the last corrections in JOINT space (every joint within 1 deg of
one particular set of target angles). Many joint combinations put the grasp point in nearly the same place, so
near the target those labels asked for corrections no camera can see; the model could not learn them, settled on
its own first estimate and closed there (median 4.4 deg from the target angles, 0.8 cm from the target).
Version 2 asks only for what is visible: the offset between the gripper and the object.
"""
import json
import re

import numpy as np

import common

STEP_MAX = 8.0          # deg, largest change of one joint per step
CLOSE_XY = 0.006        # m, grasp point this close to the target horizontally ...
CLOSE_Z = 0.003         # m, ... and vertically -> close the gripper (the height follows from the joint
                        # angles alone, so it can be held more tightly than the horizontal position)
NEAR_XY = 0.02          # m, horizontally this close to the target -> move the grasp point straight to it
VERSION = 2
PRE_UP = 0.05           # m, the pre-grasp point is this far above the grasp point
READ_NOISE = 0.3        # deg, the angles in the prompt differ from the true pose by this (play, encoder)
MIN_TIP_Z = 0.004       # m, states with the fixed jaw's tip lower than this above the board are not generated

PROMPT = ("The two images are the left and right views of a stereo camera looking at an SO-101 robot arm. "
          "Instruction: {instruction}. The servo angles are now " +
          ", ".join(k + " {" + k + "}" for k in common.JOINT_KEYS) + " degrees. "
          "Return the change of each servo angle for the next step of moving the gripper's grasp point to the "
          "grasp position of the object the instruction refers to, and whether the gripper stays open or closes, "
          "as JSON with the keys " + ", ".join(common.JOINT_KEYS) + ", gripper.")


def one_decimal(v):
    return round(float(v), 1) + 0.0                     # never "-0.0"


def prompt(instruction, joints_deg):
    return PROMPT.format(instruction=instruction,
                         **{k: f"{one_decimal(v):.1f}" for k, v in zip(common.JOINT_KEYS, joints_deg)})


def answer_json(delta, gripper):
    d = {k: one_decimal(v) for k, v in zip(common.JOINT_KEYS, delta)}
    d["gripper"] = gripper
    return json.dumps(d)


def parse_answer(text):
    """(change of the 4 angles, "open" | "close") or None."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        g = str(d["gripper"]).strip().lower()
        if g not in ("open", "close"):
            return None
        return np.array([float(d[k]) for k in common.JOINT_KEYS]), g
    except (ValueError, KeyError, TypeError):
        return None


def tip_point(env, deg4):
    """World position of the fixed jaw's tip for 4 joint angles (deg, simulator convention), wrist roll 0."""
    kin = env.kin
    d = kin.tip_depth
    u_f = float(np.interp(d, kin.probe_depths, kin.probe_fixed_u))
    local = kin.tcp_local + kin.approach_local * d + kin.closing_local * u_f
    q = env.from_lerobot(np.r_[np.asarray(deg4, float), 0.0, 50.0])
    return kin.points(q[:5], local)[0]


class Expert:
    """Goal angles of one scene; action() gives the label for any current pose."""

    def __init__(self, env, target_point):
        self.env = env
        self.target = np.asarray(target_point, float)
        self.q_final = self.q_pre = self.tilt = None
        lab = common.label_joints(env, self.target)
        if lab is None:
            return
        self.q_final, self.tilt = np.asarray(lab[0], float), lab[1]
        up = self.target + np.array([0, 0, PRE_UP])
        pre = common.label_joints(env, up, tilts=(self.tilt,)) or common.label_joints(env, up)
        if pre is not None:
            self.q_pre = np.asarray(pre[0], float)

    @property
    def ok(self):
        return self.q_final is not None

    def offset(self, q_now):
        """Target minus grasp point (m, 3 values)."""
        return self.target - common.grasp_point(self.env, q_now)

    def arrived(self, q_now):
        e = self.offset(q_now)
        return float(np.linalg.norm(e[:2])) <= CLOSE_XY and abs(float(e[2])) <= CLOSE_Z

    def jacobian(self, q, h=0.5):
        """d(grasp point) / d(joint angle), m per deg, 3 x 4 (central differences)."""
        J = np.zeros((3, 4))
        for j in range(4):
            d = np.zeros(4)
            d[j] = h
            J[:, j] = (common.grasp_point(self.env, q + d) - common.grasp_point(self.env, q - d)) / (2 * h)
        return J

    def null_direction(self, q):
        """Joint change (largest component 1 deg) that leaves the grasp point where it is: it only changes
        the gripper's pitch."""
        n = np.linalg.svd(self.jacobian(q))[2][-1]
        return n / np.max(np.abs(n))

    def action(self, q_now):
        """(change (4, deg, not rounded), gripper, phase) for the true current angles q_now."""
        q = np.asarray(q_now, float)
        e = self.offset(q)
        near = self.q_pre is None or float(np.linalg.norm(e[:2])) <= NEAR_XY
        if near:
            A = np.vstack([self.jacobian(q), [0.0, 1.0, 1.0, 1.0]])      # 3 x position + "pitch unchanged"
            try:
                delta = np.linalg.solve(A, np.r_[e, 0.0])
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, np.r_[e, 0.0], rcond=None)[0]
        else:
            delta = self.q_pre - q
        m = float(np.max(np.abs(delta)))
        if m > STEP_MAX:
            delta = delta * STEP_MAX / m
        return delta, ("close" if self.arrived(q) else "open"), ("final" if near else "pre")


# ------------------------------------------------------------------------------------------------ states
STATE_KINDS = ("start", "path", "descent", "near")
STATE_WEIGHTS = (0.15, 0.25, 0.2, 0.4)


def valid_state(env, q4):
    """Inside the joint limits and with the jaws above the board."""
    q = env.from_lerobot(np.r_[np.asarray(q4, float), 0.0, 50.0])
    if np.any(q[:4] < env.kin.lower[:4]) or np.any(q[:4] > env.kin.upper[:4]):
        return False
    return tip_point(env, q4)[2] >= MIN_TIP_Z


def sample_state(rng, env, ex, q_start, tries=30):
    """A pose the arm can be in during an approach: (angles (4, deg), kind) or None.
      start    the episode's ready pose
      path     between the ready pose and the pre-grasp pose, off the straight line by a few degrees
      descent  between the pre-grasp pose and the target
      near     around the target: small errors (down to "arrived") and larger ones, for the fine corrections;
               half of them with the gripper pitched differently from the label's pose (same grasp point, other
               joint angles), because that is how an approach really ends"""
    q_start = np.asarray(q_start, float)
    via = ex.q_pre if ex.q_pre is not None else ex.q_final
    for _ in range(tries):
        kind = STATE_KINDS[int(rng.choice(len(STATE_KINDS), p=STATE_WEIGHTS))]
        if kind == "start":
            q = q_start.copy()
        elif kind == "path":
            q = q_start + rng.uniform(0, 1) * (via - q_start) + rng.normal(0, 2.0, 4)
        elif kind == "descent":
            q = via + rng.uniform(0, 1) * (ex.q_final - via) + rng.normal(0, 1.2, 4)
        else:
            sigma = 0.25 if rng.random() < 0.25 else float(rng.choice([1.0, 1.8, 3.0]))
            q = ex.q_final + rng.normal(0, sigma, 4)
            if rng.random() < 0.5:
                q = q + ex.null_direction(ex.q_final) * rng.uniform(-8.0, 8.0)
        if valid_state(env, q):
            return q, kind
    return None


# ------------------------------------------------------------------------------------------------ servos
class ServoModel:
    """What the arm does with a command, as measured on the 12 V arm (real/probe_table.py, calib/servo_stats):
    the shoulder_lift servo gives way under the weight of the stretched arm (reading - command 0 deg at 14 cm
    reach, up to ~4.3 deg at 26 cm, not repeatable), and every joint has a dead band inside which it does not
    follow a small change of the command."""
    DEAD_BAND = np.array([0.29, 0.63, 0.93, 0.34])

    def __init__(self, rng, sag=True, dead_band=True):
        self.rng = rng
        self.sag_scale = float(rng.uniform(0.4, 1.1)) if sag else 0.0
        self.db = self.DEAD_BAND * rng.uniform(0.5, 1.0) if dead_band else np.zeros(4)

    def sag(self, env, command):
        reach = float(np.linalg.norm((common.grasp_point(env, command) - env.pan_axis)[:2]))
        return self.sag_scale * 4.3 * float(np.clip((reach - 0.14) / 0.12, 0.0, 1.3))

    def settle(self, env, command, reading):
        """The reading after the arm has settled on a command."""
        target = np.asarray(command, float) + np.array([0.0, self.sag(env, command), 0.0, 0.0])
        new = np.asarray(reading, float).copy()
        for j in range(4):
            if abs(target[j] - new[j]) >= self.db[j]:
                new[j] = target[j] + self.rng.normal(0, self.db[j] / 3)
        return new
