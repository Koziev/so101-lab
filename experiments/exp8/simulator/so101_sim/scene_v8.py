"""
v8: the v7 scene seen from widely randomised stereo cameras, with a description of the camera
that a policy can be given as an extra input (experiment "approach 1": explicit calibration).

Camera sampling (per episode):
  azimuth around the front of the robot  -45..+45 deg (+ = towards the robot's left)
  horizontal distance from the pan axis   0.45..0.95 m
  height above the board                  0.15..0.60 m
  aim point                               on/just above the object area, so pitch is 15..55 deg down
  roll                                    -4..+4 deg
  horizontal field of view                42..68 deg (current lenses ~44, 2.8 mm ~60, 2.1 mm ~68)
Held-out viewpoints ("unseen"): azimuth > +30 deg or distance > 0.85 m. split "train" never
samples them, "unseen" samples only them.
A camera is accepted only if the robot base (up to 4 cm), the ready-pose gripper and the centre
of the object area are visible in both eyes; objects are then placed only where both eyes see them (plus 10 cm
of approach height above them).

Camera description (10 numbers, robot frame = forward / left / up from the pan axis):
  stereo-centre position (m, 3) | viewing direction (unit, 3) | image-up direction (unit, 3) | HFOV / 100
"""
import math

import numpy as np

from .kinematics import unit
from .scene_v7 import _fits

DESC_NAMES = ["cam_x", "cam_y", "cam_z", "cam_fx", "cam_fy", "cam_fz", "cam_ux", "cam_uy", "cam_uz", "cam_hfov"]


def _robot_frame(env):
    return np.stack([env.fwd, env.left, np.array([0.0, 0.0, 1.0])])


def is_unseen(cfg, az_deg, dist):
    return az_deg > cfg.v8_unseen_az_deg or dist > cfg.v8_unseen_dist_m


def sample_camera_v8(env, max_tries=400):
    c, r = env.cfg, env.rng
    # the robot's base (its foot is enough as a reference; the real camera cuts its top off),
    # the ready-pose gripper and the centre of the object area must be visible in both eyes
    must = np.array([env.pan_axis + [0, 0, 0.01], env.pan_axis + [0, 0, 0.04],
                     env.pan_axis + env.fwd * 0.10 + [0, 0, 0.09],
                     env.pan_axis + env.fwd * 0.18, env.pan_axis + env.fwd * 0.18 + [0, 0, 0.10]])
    for _ in range(max_tries):
        az, dist, h = r.uniform(*c.v8_azimuth_deg), r.uniform(*c.v8_distance_m), r.uniform(*c.v8_height_m)
        unseen = is_unseen(c, az, dist)
        if (c.v8_camera_split == "train" and unseen) or (c.v8_camera_split == "unseen" and not unseen):
            continue
        a = math.radians(az)
        center = env.pan_axis + dist * (math.cos(a) * env.fwd + math.sin(a) * env.left) + np.array([0, 0, h])
        target = (env.pan_axis + env.fwd * r.uniform(0.08, 0.22) + env.left * r.uniform(-0.06, 0.06)
                  + np.array([0, 0, r.uniform(0.0, 0.08)]))
        fwd = unit(target - center)
        pitch = math.degrees(math.asin(-fwd[2]))
        if not (c.v8_pitch_deg[0] <= pitch <= c.v8_pitch_deg[1]):
            continue
        right = unit(np.cross(fwd, [0, 0, 1]))
        up = np.cross(right, fwd)
        roll = math.radians(r.uniform(-c.v8_roll_deg, c.v8_roll_deg))
        right, up = right * math.cos(roll) + up * math.sin(roll), -right * math.sin(roll) + up * math.cos(roll)
        w, hh = c.front_size
        hfov = math.radians(r.uniform(*c.v8_hfov_deg))
        vfov = 2 * math.atan(math.tan(hfov / 2) * hh / w)
        baseline = r.uniform(*c.stereo_baseline_m)
        if not _fits(must, center, fwd, right, up, baseline, hfov, vfov, 0.95):
            continue
        env.stereo_cam = {"center": center.tolist(), "fwd": fwd.tolist(), "right": right.tolist(), "up": up.tolist(),
                          "baseline": float(baseline), "hfov_deg": math.degrees(hfov), "vfov_deg": math.degrees(vfov),
                          "distance_m": float(dist), "height_m": float(h), "azimuth_deg": float(az),
                          "pitch_deg": float(pitch), "roll_deg": math.degrees(roll), "unseen": bool(unseen)}
        return True
    return False


def object_visible(env, xy, margin=0.92):
    cam = env.stereo_cam
    pts = np.array([[xy[0], xy[1], 0.0], [xy[0], xy[1], 0.10]])
    return _fits(pts, np.array(cam["center"]), np.array(cam["fwd"]), np.array(cam["right"]), np.array(cam["up"]),
                 cam["baseline"], math.radians(cam["hfov_deg"]), math.radians(cam["vfov_deg"]), margin)


def camera_desc(env, cam=None):
    cam = cam or env.stereo_cam
    R = _robot_frame(env)
    return np.r_[R @ (np.array(cam["center"]) - env.pan_axis), R @ np.array(cam["fwd"]), R @ np.array(cam["up"]),
                 cam["hfov_deg"] / 100.0].astype(np.float32)


def perturb_desc(desc, rng, pos_err_m, ang_err_deg):
    """A deliberately wrong description: position off by pos_err_m in a random direction, viewing and
    up directions rotated by ang_err_deg about a random axis (for the calibration-sensitivity test)."""
    d = np.array(desc, float).copy()
    v = rng.normal(size=3)
    d[:3] += unit(v) * pos_err_m
    axis = unit(rng.normal(size=3))
    a = math.radians(ang_err_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rm = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    d[3:6], d[6:9] = Rm @ d[3:6], Rm @ d[6:9]
    return d.astype(np.float32)


def camera_desc_from_real_pose(pose, hfov_deg, baseline_m=0.062):
    """Description of the REAL camera from calibrate_stereo.py's pose_left_eye (robot frame, the camera
    assumed on the pan-axis line, i.e. in front of the robot), for model B on the real arm."""
    p, y, rho = (math.radians(pose[k]) for k in ("pitch_down_deg", "yaw_deg", "roll_deg"))
    left_eye = np.array([pose["distance_to_pan_axis_m"], 0.0, pose["height_above_table_m"]])
    f = np.array([-math.cos(p) * math.cos(y), -math.cos(p) * math.sin(y), -math.sin(p)])
    r0 = unit(np.cross(f, [0, 0, 1]))
    u0 = np.cross(r0, f)
    right = r0 * math.cos(rho) + u0 * math.sin(rho)
    up = np.cross(right, f)
    center = left_eye + right * baseline_m / 2
    return np.r_[center, f, up, hfov_deg / 100.0].astype(np.float32)
