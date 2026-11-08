"""
Shared definitions of experiment 10: predict the target object's position in the arm's joint space.

Target ("joint-space coordinates" of the object): the angles of shoulder_pan, shoulder_lift, elbow_flex
and wrist_flex (degrees, LeRobot convention: 0 = middle of the range) that put the gripper's grasp
point at the centre of the named object, with
  - wrist_roll fixed at 0 deg (the grasp point is off the roll axis, so roll would otherwise move it),
  - one fixed grasp point between the jaws (the one used for a 30 mm object),
  - the approach as steep as possible: vertical if reachable, else tilted 15 / 30 / 45 deg towards
    the base (the same preference as the scripted expert).
"""
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "simulator"))     # the copied simulator package (no pip install)

JOINT_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
REF_SIZE = 0.03                     # grasp point = the one for a 30 mm object
TILTS_DEG = (0, 15, 30, 45)
DEFAULT_URDF = os.path.join(ROOT, "assets_so101", "so101_sim.urdf")

PROMPT = ("The two images are the left and right views of a stereo camera looking at an SO-101 robot arm. "
          "Target object: {descr}. Return the servo angles in degrees that place the gripper's grasp point "
          "at the centre of the target object, as JSON with the keys "
          + ", ".join(JOINT_KEYS) + ".")


SYNONYMS = {"cube": ["cube", "box", "block"], "gray": ["gray", "grey"]}


def vary_description(descr, rng):
    """Random synonyms for training prompts: 'gray cube' -> 'grey box', ... (evaluation keeps the plain form)."""
    return " ".join(SYNONYMS[w][rng.integers(len(SYNONYMS[w]))] if w in SYNONYMS else w for w in descr.split())


def answer_json(deg):
    return json.dumps({k: round(float(v), 1) for k, v in zip(JOINT_KEYS, deg)})


def parse_answer(text):
    """Model output -> array of 4 angles, or None if no valid JSON with all keys is found."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return np.array([float(d[k]) for k in JOINT_KEYS])
    except (ValueError, KeyError, TypeError):
        return None


def make_env(scale=1.0, n_distractors=(1, 3), urdf=DEFAULT_URDF, center=None):
    from so101_sim.env import EnvConfig, SO101GraspEnv
    kw = {"v9_center": center} if center else {}
    cfg = EnvConfig(scene_preset="v9", v9_scale=scale, front_size=(640, 360), record_wrist=False,
                    n_distractors=n_distractors, **kw)
    return SO101GraspEnv(urdf, cfg)


def center_from_calib(calib_path, baseline_m=0.062):
    """v9 camera centre from a calibration file's pose_left_eye (calibrate_stereo.py pose).

    The pose describes the LEFT eye; the simulator's camera position is the centre between the eyes,
    so the centre is shifted by half the baseline along the camera's right direction. Returns the v9
    parameters (distance, azimuth, height, pitch, yaw, roll) relative to the pan axis."""
    from so101_sim.scene_v8 import camera_desc_from_real_pose
    c = json.load(open(calib_path))
    pose = c["pose_left_eye"]
    baseline = float(c.get("stereo", {}).get("baseline_m", baseline_m))
    d = camera_desc_from_real_pose(pose, 44.1, baseline)
    cen, f, u = np.array(d[:3], float), np.array(d[3:6], float), np.array(d[6:9], float)
    dist, az = float(np.hypot(cen[0], cen[1])), math.atan2(cen[1], cen[0])
    h0 = -np.array([math.cos(az), math.sin(az)])                       # heading towards the pan axis
    fh = f[:2] / np.linalg.norm(f[:2])
    yaw = math.atan2(h0[0] * fh[1] - h0[1] * fh[0], float(h0 @ fh))
    pitch = math.asin(-f[2])
    r0 = np.cross(f, [0, 0, 1.0]); r0 /= np.linalg.norm(r0)
    u0 = np.cross(r0, f)
    roll = math.atan2(-float(u @ r0), float(u @ u0))
    return {"distance_m": dist, "height_m": float(cen[2]), "pitch_deg": math.degrees(pitch),
            "azimuth_deg": math.degrees(az), "yaw_deg": math.degrees(yaw), "roll_deg": math.degrees(roll)}


def grasp_local(env):
    return env.kin.grasp_local(REF_SIZE)


def label_joints(env, obj_pos, size=None):
    """Joint angles (deg, 4 values) putting the grasp point at obj_pos with roll 0; None if unreachable.
    size: object size the grasp point is planned for (default REF_SIZE, the training labels' convention);
    a larger size moves the fixed jaw further away from obj_pos."""
    kin, local = env.kin, (grasp_local(env) if size is None else env.kin.grasp_local(size))
    p = np.asarray(obj_pos, float)
    radial = p[:2] - env.pan_axis[:2]
    radial = np.r_[radial / (np.linalg.norm(radial) + 1e-9), 0.0]
    seeds = [env.q_home[:5], np.zeros(5)]
    for tilt in TILTS_DEG:
        t = math.radians(tilt)
        a = radial * math.sin(t) - np.array([0, 0, math.cos(t)])
        a /= np.linalg.norm(a)
        q5, ok, pos_err, _ = kin.solve(p, a, 0.0, seeds, local)
        if ok:
            deg = env.to_lerobot(np.r_[q5, kin.gripper_closed])[:4]
            return deg, tilt, float(pos_err)
    return None


def grasp_point(env, deg4):
    """World position of the fixed grasp point for 4 joint angles (deg) with wrist_roll = 0."""
    q = env.from_lerobot(np.r_[np.asarray(deg4, float), 0.0, 50.0])
    return env.kin.points(q[:5], grasp_local(env))[0]
