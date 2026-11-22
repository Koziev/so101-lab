"""
The gripper's own axes in the gripper_link frame: approach (where the fingers point), closing (from the fixed
jaw towards the moving one) and the grasp point. Taken from the simulator's kinematics (needs PyBullet);
without it, a toy convention (approach = x, closing = y) is returned.

Used to define REALISTIC test cameras: the real wrist camera sits beside the fixed jaw and looks roughly along
the approach direction, tilted ~50 deg towards the space between the jaws.
"""
import math
import os
import sys

import numpy as np


def axes(urdf):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulator"))
        from so101_sim.env import EnvConfig, SO101GraspEnv
        env = SO101GraspEnv(urdf, EnvConfig(scene_preset="v1"))
        env.reset(seed=0, render=False)
        a = np.asarray(env.kin.approach_local, float)
        c = np.asarray(env.kin.closing_local, float)
        g = np.asarray(env.kin.grasp_local(0.03), float)
        env.close()
        return a / np.linalg.norm(a), c / np.linalg.norm(c), g, "simulator"
    except Exception as e:                                       # no PyBullet or a toy URDF
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0.08, 0, 0]), f"toy convention ({e!r})"


def realistic_wrist_camera(urdf, tilt_deg=50.0, back_m=0.05, side_m=0.03, roll_deg=10.0):
    """camera -> gripper_link transform of a camera beside the fixed jaw, behind the grasp point, looking along
    the approach direction tilted `tilt_deg` towards the closing direction (into the space between the jaws)."""
    a, c, g, src = axes(urdf)
    c = c - a * (c @ a)
    c /= np.linalg.norm(c)
    eye = g - a * back_m - c * side_m
    t = math.radians(tilt_deg)
    z = a * math.cos(t) + c * math.sin(t)                         # viewing direction
    x = np.cross(z, a if abs(z @ a) < 0.99 else c)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = math.radians(roll_deg)
    R = np.c_[x, y, z] @ np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, eye
    return T, src


def jaw_points(urdf, jaw_length_beyond_grasp=0.035, half_width=0.02, body_back=0.03):
    """Points of the gripper that can come closest to the table, in the gripper_link frame: both jaw tips
    (beyond the grasp point along the approach direction, on either side), the grasp point itself, and a point
    of the gripper body behind it. Conservative: the tips are placed a little beyond the real ones."""
    a, c, g, src = axes(urdf)
    c = c - a * (c @ a)
    c /= np.linalg.norm(c)
    tip = g + a * jaw_length_beyond_grasp
    return np.array([tip - c * half_width, tip + c * half_width, g, g - a * body_back]), src
