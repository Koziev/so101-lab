"""
Qwen3.5-0.8B as an "Approach the object" policy (experiment 13): the answer format, shared by the data
preparation, the training and the closed-loop evaluation.

One call of the model plans the next HORIZON control steps (30 Hz), like one SmolVLA action chunk. Spelling out 50
steps as text would be slow, so the answer gives WAYPOINTS: the servo command every STRIDE steps, at offsets
0, STRIDE, ..., HORIZON (11 waypoints for 50 / 5), each as the CHANGE of the five joint angles from the angles the
prompt reports, plus one gripper command (%) for the whole chunk:

    {"waypoints":[[0.0,0.1,-0.2,0.0,0.0],[3.1,-2.4,1.8,0.6,0.0],...],"gripper":1}

The controller turns the waypoints back into one command per step by linear interpolation (chunk_commands).

Labels: the expert's commands of the recorded episode, a[t + k * STRIDE] - state[t] for the five joints.
Past the end of an episode the last command is repeated (the arm holds the goal), so the model also learns to
stand still. The commands include the expert's sag correction near the goal, as in the SmolVLA labels.
Values have one decimal (finer than the servos resolve); a JSON parse failure is returned as None.
"""
import json
import re

import numpy as np

HORIZON = 50
STRIDE = 5
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

PROMPT = ("The two images are the left and right views of a stereo camera looking at an SO-101 robot arm. "
          "Instruction: {instruction}. The servo angles are now "
          + ", ".join(k + " {" + k + "}" for k in JOINTS) + " degrees, gripper {gripper} %. "
          "Return the servo commands for the next {horizon} control steps (30 per second) as JSON: "
          "\"waypoints\" lists, every {stride} steps from now to step {horizon}, the change of "
          + ", ".join(JOINTS) + " from the current angles; \"gripper\" is the gripper command in %.")


def one_decimal(v):
    return round(float(v), 1) + 0.0                     # never "-0.0"


def offsets(horizon=HORIZON, stride=STRIDE):
    return list(range(0, horizon + 1, stride))


def prompt(instruction, state6, horizon=HORIZON, stride=STRIDE):
    s = np.asarray(state6, float)
    return PROMPT.format(instruction=instruction, horizon=horizon, stride=stride,
                         gripper=f"{s[5]:.0f}",
                         **{k: f"{one_decimal(v):.1f}" for k, v in zip(JOINTS, s[:5])})


def label(actions, state6, t, horizon=HORIZON, stride=STRIDE):
    """Waypoints (K x 5, deg, relative to state[t]) and the gripper command for the chunk starting at step t
    of an episode with the commands `actions` (n x 6)."""
    a = np.asarray(actions, float)
    idx = [min(t + o, len(a) - 1) for o in offsets(horizon, stride)]
    wp = a[idx, :5] - np.asarray(state6, float)[:5]
    return wp, float(a[idx[-1], 5])


def answer_text(wp, gripper):
    return json.dumps({"waypoints": [[one_decimal(v) for v in row] for row in wp],
                       "gripper": int(round(gripper))}, separators=(",", ":"))


def parse_answer(text, horizon=HORIZON, stride=STRIDE):
    """(waypoints K x 5, gripper) or None. A short list (the generation stopped early) is padded with its last
    waypoint; malformed rows make the whole answer invalid."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        wp = np.array([[float(v) for v in row] for row in d["waypoints"]], float)
        g = float(d.get("gripper", 1.0))
    except (ValueError, KeyError, TypeError):
        return None
    k = len(offsets(horizon, stride))
    if wp.ndim != 2 or wp.shape[1] != 5 or len(wp) == 0:
        return None
    if len(wp) < k:
        wp = np.vstack([wp, np.repeat(wp[-1:], k - len(wp), axis=0)])
    return wp[:k], g


def chunk_commands(wp, gripper, state6, n_steps, stride=STRIDE):
    """Commands (n_steps x 6, LeRobot units) for the steps 0 .. n_steps-1 of a chunk: the waypoints, added to the
    angles of the prompt, linearly interpolated between their offsets (held after the last one)."""
    wp = np.asarray(wp, float)
    offs = np.arange(len(wp)) * stride
    base = np.asarray(state6, float)[:5]
    steps = np.arange(n_steps)
    q = np.stack([np.interp(steps, offs, wp[:, j]) for j in range(5)], axis=1) + base
    return np.hstack([q, np.full((n_steps, 1), float(gripper))])
