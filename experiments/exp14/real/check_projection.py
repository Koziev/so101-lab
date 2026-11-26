"""
Does the geometry behind the real samples agree with the photos? Draws, on the left image of every real sample,
what the SIMULATOR'S camera model expects to see:

    python real/check_projection.py --setup setups/<name> --data data_real_v11 --eval outputs/eval_v11_real.json

  green cross    the recorded reference point (hand-guided grasp point, offsets applied, kinematics)
  green dot      the same point dropped onto the simulator's table plane
  red cross      the model's predicted grasp point (with --eval)
  yellow         the simulator's table plane around the robot: arcs 10 / 20 / 30 cm from the pan axis and the
                 rays at -45 / 0 / +45 deg; blue dot = the pan axis on the table

How to read it:
  - green cross on the target object           -> recording, offsets and camera pose agree with the photo
  - green cross beside the target              -> the geometry chain is off (camera pose or offsets)
  - yellow arcs not centred on the robot's base,
    or floating above / sunk into the board    -> the simulator's robot / table placement differs from reality
  - green fine, red beside the object          -> the model itself mislocates (appearance gap)
Writes <data>/projection/<n>.png and sheet_*.png (9 samples each), and prints the reference points' heights
above the simulator's table (expected: the grasp height, about 1.0-1.5 cm).
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
W_OUT, H_OUT = 640, 360


class SimCameraModel:
    """The left image as the simulator assumes it: a pinhole at the calibrated left-eye pose (pose_base),
    rotated by the rectification, with the rectified zoomed focal length and the principal point centred."""

    def __init__(self, calib, base_offset):
        c = calib
        W, H = c.get("image_size", (1280, 720))
        K1, D1 = np.array(c["left"]["K"]), np.array(c["left"]["dist"])
        K2, D2 = np.array(c["right"]["K"]), np.array(c["right"]["dist"])
        R, T = np.array(c["stereo"]["R"]), np.array(c["stereo"]["T_m"]).reshape(3, 1)
        self.R1, _, P1, *_ = cv2.stereoRectify(K1, D1, K2, D2, (W, H), R, T, alpha=0)
        self.f = P1[0, 0] * c.get("rectify_zoom", 1.0) * W_OUT / W
        Tlb = np.array(c["pose_base"]["T_left2base"], float)
        self.Rlb, self.t = Tlb[:3, :3], Tlb[:3, 3]
        self.base_offset = np.asarray(base_offset, float)

    def project(self, X_world):
        X = np.atleast_2d(np.asarray(X_world, float)) - self.base_offset
        xc = (self.R1 @ (self.Rlb.T @ (X - self.t).T)).T
        z = np.where(xc[:, 2] > 1e-6, xc[:, 2], np.nan)
        return np.c_[self.f * xc[:, 0] / z + W_OUT / 2, self.f * xc[:, 1] / z + H_OUT / 2]


def cross(img, uv, colour, s, size=7):
    if not np.all(np.isfinite(uv)):
        return
    u, v = int(round(uv[0] * s)), int(round(uv[1] * s))
    cv2.line(img, (u - size, v), (u + size, v), colour, 2)
    cv2.line(img, (u, v - size), (u, v + size), colour, 2)


def polyline(img, uv, colour, s):
    pts = [(int(round(a * s)), int(round(b * s))) for a, b in uv if np.isfinite(a) and np.isfinite(b)]
    for p, q in zip(pts[:-1], pts[1:]):
        cv2.line(img, p, q, colour, 1, cv2.LINE_AA)


def table_overlay(img, cam, pan_axis, fwd, left, s, table_z=0.0):
    ang = np.radians(np.linspace(-90, 90, 61))
    for r in (0.10, 0.20, 0.30):
        pts = pan_axis + r * (np.cos(ang)[:, None] * fwd + np.sin(ang)[:, None] * left)
        pts[:, 2] = table_z
        polyline(img, cam.project(pts), (0, 220, 255), s)
    for a in np.radians([-45.0, 0.0, 45.0]):
        rr = np.linspace(0.05, 0.32, 12)
        pts = pan_axis + rr[:, None] * (math.cos(a) * fwd + math.sin(a) * left)
        pts[:, 2] = table_z
        polyline(img, cam.project(pts), (0, 220, 255), s)
    p = pan_axis.copy()
    p[2] = table_z
    uv = cam.project(p)[0]
    if np.all(np.isfinite(uv)):
        cv2.circle(img, (int(uv[0] * s), int(uv[1] * s)), 4, (255, 80, 0), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--data", default="data_real_v11")
    ap.add_argument("--eval", default=None, help="evaluate.py output with the real split (adds the predictions)")
    ap.add_argument("--split", default="real")
    ap.add_argument("--scale", type=int, default=2, help="output images are 640x360 times this")
    args = ap.parse_args()

    import common
    from setup_io import Setup
    setup = Setup(args.setup)
    env = common.make_env(0.0, cell=common.cell_config(args.setup))
    env.reset(seed=0, render=False)
    base_offset = np.asarray(env.kin.base_offset, float)
    pan_axis, fwd, left = np.asarray(env.pan_axis, float), np.asarray(env.fwd, float), np.asarray(env.left, float)
    env.close()
    cam = SimCameraModel(setup.load("stereo_calib.json"), base_offset)
    print(f"simulator: the base frame's origin is {100 * base_offset[2]:+.2f} cm above the table plane; "
          f"camera (left eye) {100 * (cam.t[2] + base_offset[2]):.1f} cm above it")
    pred = {}
    if args.eval:
        for r in json.load(open(args.eval))["splits"][args.split]["samples"]:
            if "pred_point" in r:
                pred[int(r["seed"])] = (np.array(r["pred_point"]), r["dist_cm"])
    out = os.path.join(args.data, "projection")
    os.makedirs(out, exist_ok=True)
    s = args.scale
    tiles, heights = [], []
    files = sorted(glob.glob(os.path.join(args.data, "samples", "*.json")), key=lambda p: int(os.path.basename(p)[:-5]))
    for f in files:
        m = json.load(open(f))
        img = cv2.imread(f[:-5] + "_left.png")
        if img is None:
            continue
        img = cv2.resize(img, (W_OUT * s, H_OUT * s), interpolation=cv2.INTER_CUBIC)
        table_overlay(img, cam, pan_axis, fwd, left, s)
        ref = np.array(m["ref_point"], float)
        heights.append(100 * ref[2])
        foot = ref.copy()
        foot[2] = 0.0
        uv_foot = cam.project(foot)[0]
        if np.all(np.isfinite(uv_foot)):
            cv2.circle(img, (int(uv_foot[0] * s), int(uv_foot[1] * s)), 3, (0, 255, 0), -1)
        cross(img, cam.project(ref)[0], (0, 255, 0), s)
        text = f"#{m['index']} {m['instruction']} | ref {100 * ref[2]:.1f} cm above the sim table"
        if m["index"] in pred:
            cross(img, cam.project(pred[m["index"]][0])[0], (0, 0, 255), s)
            text += f" | miss {pred[m['index']][1]:.1f} cm"
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out, f"{m['index']}.png"), img)
        tiles.append(cv2.resize(img, (W_OUT, H_OUT), interpolation=cv2.INTER_AREA))
    for k in range(0, len(tiles), 9):
        chunk = tiles[k:k + 9] + [np.zeros_like(tiles[0])] * (-len(tiles[k:k + 9]) % 3)
        rows = [np.hstack(chunk[i:i + 3]) for i in range(0, len(chunk), 3)]
        cv2.imwrite(os.path.join(out, f"sheet_{k // 9}.png"), np.vstack(rows))
    if heights:
        h = np.array(heights)
        print(f"{len(h)} samples: reference points {np.median(h):.2f} cm above the simulator's table (median), "
              f"range {h.min():.2f} to {h.max():.2f} cm; the labels in training are at 1.0-1.5 cm")
        print(f"images in {out}/ (sheet_*.png: 9 samples each)")


if __name__ == "__main__":
    main()
