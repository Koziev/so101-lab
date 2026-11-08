"""
v9: the v8 scene and servo model, with the stereo camera drawn from a normal distribution around the
measured pose of the real camera (3.6 mm lenses). One width factor s scales all standard deviations:
s = 0 is a fixed camera at the real pose, s = 1 the base spread, s = 2 and s = 4 wider.

Centre (calibrate_stereo.py pose of the real camera)   standard deviation at s = 1
  horizontal distance from the pan axis  0.525 m       0.03 m
  height above the board                 0.312 m       0.025 m
  pitch (looking down)                   38.7 deg      3 deg
  azimuth around the robot's front       0 deg         4 deg   (+ = robot's left)
  yaw (heading offset from the robot)    0 deg         2 deg
  roll                                   2.5 deg       1.5 deg
Every value is cut off at +-2.5 sigma. The horizontal field of view is 44.1 deg (rectified 3.6 mm lens)
with a fixed 0.5 deg standard deviation (calibration uncertainty), independent of s.
"""
import math

import numpy as np

from .kinematics import unit
from .scene_v7 import _fits


def _tn(r, mean, sigma, s, clip=2.5):
    """Normal sample truncated at +-clip standard deviations (always one draw: reproducible streams)."""
    z = float(np.clip(r.normal(), -clip, clip))
    return mean + z * sigma * s


def sample_camera_v9(env, max_tries=200):
    c, r = env.cfg, env.rng
    s = c.v9_scale
    must = np.array([env.pan_axis + [0, 0, 0.01], env.pan_axis + [0, 0, 0.04],
                     env.pan_axis + env.fwd * 0.10 + [0, 0, 0.09],
                     env.pan_axis + env.fwd * 0.18, env.pan_axis + env.fwd * 0.18 + [0, 0, 0.10]])
    w, hh = c.front_size
    for attempt in range(max_tries):
        dist = _tn(r, c.v9_center["distance_m"], c.v9_sigma["distance_m"], s)
        height = _tn(r, c.v9_center["height_m"], c.v9_sigma["height_m"], s)
        pitch = _tn(r, c.v9_center["pitch_deg"], c.v9_sigma["pitch_deg"], s)
        az = _tn(r, c.v9_center["azimuth_deg"], c.v9_sigma["azimuth_deg"], s)
        yaw = _tn(r, c.v9_center["yaw_deg"], c.v9_sigma["yaw_deg"], s)
        roll = _tn(r, c.v9_center["roll_deg"], c.v9_sigma["roll_deg"], s)
        hfov_deg = _tn(r, c.v9_hfov_deg[0], c.v9_hfov_deg[1], 1.0)
        baseline = r.uniform(*c.stereo_baseline_m)
        a = math.radians(az)
        radial = math.cos(a) * env.fwd + math.sin(a) * env.left            # pan axis -> camera, horizontal
        center = env.pan_axis + dist * radial + np.array([0, 0, height])
        heading = -radial                                                   # towards the pan axis ...
        y = math.radians(yaw)                                               # ... turned by the yaw offset
        heading = heading * math.cos(y) + np.cross([0, 0, 1.0], heading) * math.sin(y)
        p = math.radians(pitch)
        fwd = unit(heading * math.cos(p) - np.array([0, 0, math.sin(p)]))
        right = unit(np.cross(fwd, [0, 0, 1]))
        up = np.cross(right, fwd)
        rho = math.radians(roll)
        right, up = right * math.cos(rho) + up * math.sin(rho), -right * math.sin(rho) + up * math.cos(rho)
        hfov = math.radians(hfov_deg)
        vfov = 2 * math.atan(math.tan(hfov / 2) * hh / w)
        if _fits(must, center, fwd, right, up, baseline, hfov, vfov, 0.95):
            env.stereo_cam = {"center": center.tolist(), "fwd": fwd.tolist(), "right": right.tolist(),
                              "up": up.tolist(), "baseline": float(baseline), "hfov_deg": hfov_deg,
                              "vfov_deg": math.degrees(vfov), "distance_m": float(dist), "height_m": float(height),
                              "azimuth_deg": float(az), "pitch_deg": float(pitch), "yaw_deg": float(yaw),
                              "roll_deg": float(roll), "unseen": False, "v9_scale": float(s)}
            return True
    return False
