"""
Shared definitions of experiment 11 (localization with the new tasks): stereo pair + instruction -> the joint
angles (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex; degrees, simulator convention) that put the
gripper's grasp point at the grasp position of the object the instruction refers to.

Same label convention as exp10 (wrist_roll 0, the grasp point of a 30 mm object, steepest reachable approach),
with one change: the target point is the object's footprint centre at ONE height for all objects (2.0 cm above
the board, CellEnvV11.grasp_z), which keeps the fingertips about 8 mm above the board.
The board's height for the arm comes from the setup's table.json (real/probe_table.py).
"""
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "simulator"))
sys.path.insert(0, ROOT)

JOINT_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
REF_SIZE = 0.03
TILTS_DEG = (0, 15, 30, 45)
DEFAULT_URDF = os.path.join(ROOT, "assets_so101", "so101_sim.urdf")

PROMPT = ("The two images are the left and right views of a stereo camera looking at an SO-101 robot arm. "
          "Instruction: {instruction}. Return the servo angles in degrees that place the gripper's grasp point "
          "at the grasp position of the object the instruction refers to, as JSON with the keys "
          + ", ".join(JOINT_KEYS) + ".")


def answer_json(deg):
    """One decimal per angle, always written with the decimal ("2.0", never "2" or "-0.0"), so every number in
    the labels has the same token pattern."""
    return json.dumps({k: round(float(v), 1) + 0.0 for k, v in zip(JOINT_KEYS, deg)})


def parse_answer(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return np.array([float(d[k]) for k in JOINT_KEYS])
    except (ValueError, KeyError, TypeError):
        return None


def cell_config(setup_dir):
    """Camera centre (v9 parameters, needs an env for the robot geometry), rectified HFOV and baseline from a
    setup folder's stereo_calib.json. Returns a dict for make_env(cell=...)."""
    c = json.load(open(os.path.join(setup_dir, "stereo_calib.json")))
    from so101_sim.env_v11 import rectified_hfov_deg, rectified_left_pose
    cell = {"T_left2base": rectified_left_pose(c), "baseline_m": float(c["stereo"]["baseline_m"]),
            "hfov_deg": rectified_hfov_deg(c), "board_z_m": 0.0, "setup_dir": os.path.abspath(setup_dir)}
    t = os.path.join(setup_dir, "table.json")
    if os.path.exists(t):                      # board height probed with the torque on (real/probe_table.py)
        cell["board_z_m"] = float(json.load(open(t))["board_z_read_m"])
    return cell


def load_cell(data_dir):
    """The cell a dataset was made for (dataset_info.json), or None: evaluation must use the same geometry."""
    f = os.path.join(data_dir, "dataset_info.json")
    return json.load(open(f)).get("cell") if os.path.exists(f) else None


# What the model sees. "frame" is the size of the whole rectified view (60.9 deg wide), "crop" the window
# of it (x0, y0, x1, y1 in pixels of that frame) that is given to the model, None = the whole frame.
#   full    the whole view at 640x360: an object of 25 mm is about 25 pixels wide
#   crop2x  the workspace window (arm's base, board up to the lower edge) at twice the pixel density,
#           1024x576: 2.6 times the pixels of "full"
#   crop15  the same window at 1.5 times the density, 768x432: 1.4 times the pixels of "full"
# The camera's sensor resolves about 1.35 times the density of "full" (focal length 734 px against 544 px);
# beyond that the rectification interpolates. crop15 is therefore about sensor-native, crop2x gives the model
# larger objects but no more optical detail than crop15.
IMAGE_MODES = {"full": {"frame": (640, 360), "crop": None},
               "crop2x": {"frame": (1280, 720), "crop": (192, 144, 1216, 720)},
               "crop15": {"frame": (960, 540), "crop": (144, 108, 912, 540)}}


def model_image(rect_bgr_or_rgb, image="full"):
    """A rectified eye image at the camera's 1280x720 -> the image the model gets in that mode."""
    import cv2
    m = IMAGE_MODES[image]
    w, h = m["frame"]
    img = rect_bgr_or_rgb
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if m["crop"]:
        x0, y0, x1, y1 = m["crop"]
        img = img[y0:y1, x0:x1]
    return img


R_RANGE = (0.12, 0.30)        # object distance from the pan axis, m (exp10 and the first exp11 run: 0.12-0.24)


def make_env(scale=1.0, urdf=DEFAULT_URDF, cell=None, task_kwargs=None, half_angle_deg=45.0, r_range=R_RANGE,
             real_background=True, image="full", tip_paint_mm=20.0):
    """The exp11 cell: v9 scene, printed objects placed by a task; camera centred on the calibrated pose."""
    from so101_sim.env import EnvConfig
    from so101_sim.env_v11 import CellEnvV11, v9_center_from_pose_base
    kw = {}
    if cell:
        b = cell["baseline_m"]
        kw = {"v9_hfov_deg": (cell["hfov_deg"], 0.5), "stereo_baseline_m": (b - 0.0005, b + 0.0005),
              "board_z": float(cell.get("board_z_m", 0.0))}
    mode = IMAGE_MODES[image]
    cfg = EnvConfig(scene_preset="v9", v9_scale=scale, front_size=tuple(mode["frame"]), record_wrist=False,
                    v7_half_angle_deg=half_angle_deg, r_range=tuple(r_range), **kw)
    env = CellEnvV11(urdf, cfg)
    env.task_kwargs = dict(task_kwargs or {})
    env.crop = tuple(mode["crop"]) if mode["crop"] else None
    env.tip_paint_m = float(tip_paint_mm) / 1000.0     # the real arm's fixed jaw: last 20 mm painted black (16 Nov)
    if cell:                                   # the robot geometry is known now: set the camera centre
        env.cfg.v9_center = v9_center_from_pose_base(env, cell["T_left2base"], cell["baseline_m"])
        if real_background and cell.get("setup_dir") and os.path.isdir(cell["setup_dir"]):
            from so101_sim import scene_real                 # the real cell's picture as the scene, if captured
            env.real_bg = scene_real.load(cell["setup_dir"], env.kin.base_offset)
    if env.crop and real_background and env.real_bg is None:
        raise SystemExit(f"image mode '{image}' needs the setup's real background picture (stereo_background.png)")
    return env


def grasp_local(env):
    return env.kin.grasp_local(REF_SIZE)


def label_joints(env, point, tilts=None):
    """Joint angles (deg, 4 values) putting the grasp point at `point` with roll 0; None if unreachable.
    tilts: approach tilts to try, steepest first (default TILTS_DEG)."""
    kin, local = env.kin, grasp_local(env)
    p = np.asarray(point, float)
    radial = p[:2] - env.pan_axis[:2]
    radial = np.r_[radial / (np.linalg.norm(radial) + 1e-9), 0.0]
    seeds = [env.q_home[:5], np.zeros(5)]
    for tilt in (TILTS_DEG if tilts is None else tilts):
        t = math.radians(tilt)
        a = radial * math.sin(t) - np.array([0, 0, math.cos(t)])
        a /= np.linalg.norm(a)
        q5, ok, pos_err, _ = kin.solve(p, a, 0.0, seeds, local)
        if ok:
            deg = env.to_lerobot(np.r_[q5, kin.gripper_closed])[:4]
            return deg, tilt, float(pos_err)
    return None


def grasp_point(env, deg4):
    """World position of the grasp point for 4 joint angles (deg) with wrist_roll = 0."""
    q = env.from_lerobot(np.r_[np.asarray(deg4, float), 0.0, 50.0])
    return env.kin.points(q[:5], grasp_local(env))[0]
