"""
Joint-angle convention between the real follower arm and the simulator (and the model trained on it).

    simulator angle = real angle + offset        (degrees; gripper percentage unchanged)

The offsets correct the arm's calibration zeros (and whatever constant geometric error was absorbed by the
fit). They are stored in a JSON file {joint name: offset}; missing joints count as 0. For the 7.4 V follower
calibrated as "my_follower" with the current camera: real/arm_convention.json (pan measured physically,
the rest fitted with evaluation/fit_offsets.py and confirmed on new samples).
"""
import json

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load(path):
    d = json.load(open(path))
    d = d.get("offsets", d)
    return np.array([float(d.get(j, 0.0)) for j in JOINTS])


def real_to_sim(q_real, offsets):
    return np.asarray(q_real, float) + offsets[: len(q_real)]


def sim_to_real(q_sim, offsets):
    return np.asarray(q_sim, float) - offsets[: len(q_sim)]
