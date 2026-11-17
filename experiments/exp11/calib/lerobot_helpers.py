"""Minimal LeRobot arm helpers for the calibration (degrees mode, slow smooth moves, torque handling)."""
import time

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def make_robot(port, rid, max_step=20.0):
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=port, id=rid, use_degrees=True, max_relative_target=max_step,
                                              disable_torque_on_disconnect=True))
    robot.connect(calibrate=False)
    return robot


def read_joints(robot):
    obs = robot.get_observation()
    return np.array([float(obs[f"{j}.pos"]) for j in JOINTS])


def send(robot, q):
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})


def move_to(robot, target, seconds=3.0, hz=30, settle=True):
    start = read_joints(robot)
    target = np.asarray(target, float)
    if target.size == 5:
        target = np.r_[target, start[5]]
    n = max(2, int(seconds * hz))
    for i in range(1, n + 1):
        s = i / n
        s = s ** 3 * (10 - 15 * s + 6 * s ** 2)
        send(robot, start + (target - start) * s)
        time.sleep(1.0 / hz)
    if settle:
        time.sleep(0.3)
