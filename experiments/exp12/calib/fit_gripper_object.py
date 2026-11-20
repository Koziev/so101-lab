"""
Fit the stereo camera to the arm from samples of calib/collect_gripper_object.py: the object held in the
gripper is a fixed (unknown) point p in the gripper frame; for joint readings q its image position in each raw
eye is predicted through the kinematics (with the joint offsets d) and the stereo camera (pose S of the left
eye in the base frame; right eye from the stereo calibration). Least squares over S, d (shoulder_lift,
elbow_flex, wrist_flex; pan fixed) and p.

    python calib/fit_gripper_object.py setups/<name>/gripper_object/<time> --setup setups/<name>
    ... --detector llmdet --llmdet-model /path/to/llmdet_tiny          (optional: find the box first)
    ... --write                                                        (store the result in the setup)

Measurement: around the position predicted by the current calibration (or inside the LLMDet box), the pixels of
the object's colour (from the description: red, orange, yellow, green, cyan, blue, purple, pink; otherwise any
strongly saturated colour) are segmented; the centroid of the largest region is the object's image position.
Report: residuals per eye (px), the error of the triangulated 3D position (mm), held-out errors (fit on even
samples, test on odd ones and vice versa), and the difference to the board-based stereo pose and offsets.
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
from calib.estimate import from_T, inv, rot_angle_deg, to_T  # noqa: E402
from calib.urdf_fk import UrdfChain  # noqa: E402
from setup_io import JOINTS, Setup  # noqa: E402

HUES = {"red": [(0, 10), (170, 180)], "orange": [(8, 24)], "yellow": [(22, 38)], "green": [(38, 85)],
        "cyan": [(85, 100)], "blue": [(100, 130)], "purple": [(128, 155)], "pink": [(150, 172)]}


def colour_mask(bgr, description, s_min=110, v_min=70):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s >= s_min) & (v >= v_min)
    words = description.lower().split()
    ranges = [r for w in words for r in HUES.get(w, [])]
    if not ranges:
        return sat
    hue = np.zeros_like(sat)
    for lo, hi in ranges:
        hue |= (h >= lo) & (h <= hi)
    return sat & hue


def centroid_in(bgr, box, description, min_area=40):
    """Centroid (x, y) of the largest object-coloured region inside box (x0, y0, x1, y1); None if none."""
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    m = colour_mask(bgr[y0:y1, x0:x1], description).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return None
    return np.array([cents[k][0] + x0, cents[k][1] + y0]), int(stats[k, cv2.CC_STAT_AREA])


class Model:
    """Prediction of the held object's image position in both raw eyes."""

    def __init__(self, fk, stereo, d0, free=("shoulder_lift", "elbow_flex", "wrist_flex")):
        self.fk = fk
        self.KL, self.DL = np.array(stereo["left"]["K"]), np.array(stereo["left"]["dist"])
        self.KR, self.DR = np.array(stereo["right"]["K"]), np.array(stereo["right"]["dist"])
        self.R = np.array(stereo["stereo"]["R"])
        self.T = np.array(stereo["stereo"]["T_m"]).reshape(3)
        self.d0 = np.asarray(d0, float)
        self.free = [JOINTS.index(j) for j in free]

    def unpack(self, x):
        S = to_T(x[0:3], x[3:6])
        d = self.d0.copy()
        d[self.free] = x[6:6 + len(self.free)]
        p = x[6 + len(self.free):9 + len(self.free)]
        return S, d, p

    def pack(self, S, d, p):
        return np.concatenate([*from_T(S), d[self.free], p])

    def point_base(self, q, d, p):
        G = self.fk.pose(np.radians(np.asarray(q)[:5] + d))
        return G[:3, :3] @ p + G[:3, 3]

    def project(self, x, q):
        S, d, p = self.unpack(x)
        Pb = self.point_base(q, d, p)
        Sinv = inv(S)
        Xl = Sinv[:3, :3] @ Pb + Sinv[:3, 3]
        Xr = self.R @ Xl + self.T
        zl = np.zeros(3)
        uvl = cv2.projectPoints(Xl.reshape(1, 3), zl, zl, self.KL, self.DL)[0].reshape(2)
        uvr = cv2.projectPoints(Xr.reshape(1, 3), zl, zl, self.KR, self.DR)[0].reshape(2)
        return uvl, uvr, Xl

    def pair_error(self, uvl, uvr):
        """How well a measured pair agrees with the stereo geometry alone: triangulate, reproject into both eyes,
        largest pixel error. A correct pair gives ~1-2 px; one eye detecting something else gives far more."""
        X = self.triangulate(uvl, uvr)
        zl = np.zeros(3)
        if X[2] <= 0:
            return 1e9
        a = cv2.projectPoints(X.reshape(1, 3), zl, zl, self.KL, self.DL)[0].reshape(2)
        b = cv2.projectPoints((self.R @ X + self.T).reshape(1, 3), zl, zl, self.KR, self.DR)[0].reshape(2)
        return float(max(np.linalg.norm(a - uvl), np.linalg.norm(b - uvr)))

    def triangulate(self, uvl, uvr):
        """3D point (left-eye frame) from the two measured pixels."""
        a = cv2.undistortPoints(np.asarray(uvl, float).reshape(1, 1, 2), self.KL, self.DL).reshape(2)
        b = cv2.undistortPoints(np.asarray(uvr, float).reshape(1, 1, 2), self.KR, self.DR).reshape(2)
        P1 = np.c_[np.eye(3), np.zeros(3)]
        P2 = np.c_[self.R, self.T]
        X = cv2.triangulatePoints(P1, P2, a.reshape(2, 1), b.reshape(2, 1))
        return (X[:3] / X[3]).reshape(3)


def fit(model, samples, x0, prior_d_deg=10.0):
    from scipy.optimize import least_squares

    def res(x):
        r = []
        for s in samples:
            uvl, uvr, _ = model.project(x, s["q"])
            r += [uvl - s["uvl"], uvr - s["uvr"]]
        _, d, _ = model.unpack(x)
        r.append((d[model.free] - model.d0[model.free]) / prior_d_deg)
        return np.concatenate(r)
    sol = least_squares(res, x0, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=500)
    J = sol.jac
    sigma2 = float(np.sum(sol.fun ** 2) / max(1, J.shape[0] - J.shape[1]))
    try:
        std = np.sqrt(np.clip(np.diag(np.linalg.inv(J.T @ J) * sigma2), 0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(x0), np.nan)
    return sol.x, std


def errors(model, x, samples):
    """Per-sample pixel errors (left, right) and 3D error (mm) of the triangulated measurement."""
    S, d, p = model.unpack(x)
    el, er, e3 = [], [], []
    for s in samples:
        uvl, uvr, Xl = model.project(x, s["q"])
        el.append(np.linalg.norm(uvl - s["uvl"]))
        er.append(np.linalg.norm(uvr - s["uvr"]))
        e3.append(1000 * np.linalg.norm(model.triangulate(s["uvl"], s["uvr"]) - Xl))
    return np.array(el), np.array(er), np.array(e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="+", help="one or more session folders (fitted together)")
    ap.add_argument("--setup", required=True)
    ap.add_argument("--urdf", default=os.path.join(ROOT, "assets_so101", "so101_sim.urdf"))
    ap.add_argument("--window", type=int, default=120, help="search half-width around the prediction (px)")
    ap.add_argument("--detector", choices=["predicted", "llmdet"], default="predicted")
    ap.add_argument("--llmdet-model", default="iSEE-Laboratory/llmdet_tiny")
    ap.add_argument("--rounds", type=int, default=2, help="measure -> fit rounds (the window follows the fit)")
    ap.add_argument("--write", action="store_true", help="store the fitted stereo pose and offsets in the setup")
    ap.add_argument("--pair-px", type=float, default=3.0,
                    help="reject a sample whose two eye measurements disagree with the stereo geometry by more (px)")
    ap.add_argument("--outlier-factor", type=float, default=3.0,
                    help="after fitting, drop samples whose error exceeds this x the median (and 10 px), refit")
    args = ap.parse_args()

    setup = Setup(args.setup)
    stereo = setup.load("stereo_calib.json")
    if "pose_base" not in stereo:
        raise SystemExit("stereo_calib.json has no pose_base yet (run the automated calibration first)")
    S0 = np.array(stereo["pose_base"]["T_left2base"])
    d0 = np.asarray(setup.offsets()[:5])
    fk = UrdfChain(args.urdf, "gripper_link")
    model = Model(fk, stereo, d0)
    try:
        from calib.gripper_axes import axes
        _, _, g, _ = axes(args.urdf)
        p0 = np.asarray(g, float)
    except Exception:
        p0 = np.zeros(3)
    x = model.pack(S0, d0, p0)
    files = []
    for folder in args.session:
        found = sorted(glob.glob(os.path.join(folder, "[0-9][0-9][0-9].json")))
        print(f"{folder}: {len(found)} samples")
        files += found
    if not files:
        raise SystemExit("no samples in these folders")
    out_dir = args.session[0] if len(args.session) == 1 else os.path.join(args.session[0], "combined")
    os.makedirs(out_dir, exist_ok=True)
    tag = lambda f: (os.path.basename(os.path.dirname(f))[-6:] + "/" if len(args.session) > 1 else "") + os.path.basename(f)[:3]
    finder = None
    if args.detector == "llmdet":
        finder = LlmDet(args.llmdet_model)
    raw = [(f, json.load(open(f))) for f in files]
    samples = []
    for rnd in range(args.rounds):
        samples, rejected = [], {}
        for f, info in raw:
            img = cv2.imread(f[:-5] + ".png")
            W = img.shape[1] // 2
            eyes = (img[:, :W], img[:, W:])
            descr = info.get("description", "")
            uvl_p, uvr_p, _ = model.project(x, info["q_real"])
            meas = []
            for eye, pred in zip(eyes, (uvl_p, uvr_p)):
                if finder is not None:
                    box = finder.box(eye, descr)
                    if box is None:
                        meas.append(None)
                        continue
                else:
                    w = args.window
                    box = (pred[0] - w, pred[1] - w, pred[0] + w, pred[1] + w)
                meas.append(centroid_in(eye, box, descr))
            if all(m is not None for m in meas):
                pe = model.pair_error(meas[0][0], meas[1][0])
                if pe > args.pair_px:
                    rejected[tag(f)] = f"eyes disagree ({pe:.1f} px)"
                    continue
                samples.append({"name": tag(f), "file": f[:-5] + ".png", "q": np.asarray(info["q_real"])[:5],
                                "uvl": meas[0][0], "uvr": meas[1][0]})
            else:
                rejected[tag(f)] = "not found in " + ("both eyes" if all(m is None for m in meas)
                                                                       else ("the left eye" if meas[0] is None else "the right eye"))
        if len(samples) < 12:
            raise SystemExit(f"only {len(samples)} samples with the object found in both eyes "
                             f"(try --window 200, or --detector llmdet)")
        x, std = fit(model, samples, x)
        for _ in range(3):                                      # outliers: far above the median -> drop, refit
            el, er, e3 = errors(model, x, samples)
            e = np.maximum(el, er)
            lim = max(args.outlier_factor * float(np.median(e)), 10.0)
            bad = [k for k in range(len(samples)) if e[k] > lim]
            if not bad or len(samples) - len(bad) < 12:
                break
            for k in bad:
                rejected[samples[k]["name"]] = f"outlier after the fit ({e[k]:.0f} px)"
            samples = [s_ for k, s_ in enumerate(samples) if k not in bad]
            x, std = fit(model, samples, x)
        el, er, e3 = errors(model, x, samples)
        kinds = {}
        for r in rejected.values():
            kinds[r.split(" (")[0]] = kinds.get(r.split(" (")[0], 0) + 1
        print(f"round {rnd + 1}: {len(samples)} of {len(raw)} samples used ({', '.join(f'{v} {k}' for k, v in kinds.items())}); "
              f"error left {np.median(el):.1f} px, right {np.median(er):.1f} px (median), 3D {np.median(e3):.1f} mm")

    # held-out check: fit on even, test on odd, and vice versa
    held = []
    for a, b in ((0, 1), (1, 0)):
        xa, _ = fit(model, samples[a::2], model.pack(S0, d0, p0))
        el, er, e3 = errors(model, xa, samples[b::2])
        held.append((np.median(np.r_[el, er]), np.median(e3)))
    S, d, p = model.unpack(x)
    dS_mm = 1000 * np.linalg.norm(S[:3, 3] - S0[:3, 3])
    dS_deg = rot_angle_deg(S0[:3, :3].T @ S[:3, :3])
    print(f"\nheld-out (fit on half, test on the other half): {held[0][0]:.1f} / {held[1][0]:.1f} px, "
          f"{held[0][1]:.1f} / {held[1][1]:.1f} mm (median)")
    print(f"stereo pose vs the pose stored in the setup ({stereo['pose_base'].get('source', 'board-based')}): "
          f"{dS_mm:.1f} mm, {dS_deg:.2f} deg "
          f"(fit uncertainty {1000 * np.linalg.norm(std[3:6]):.1f} mm, {math.degrees(np.linalg.norm(std[0:3])):.2f} deg)")
    print("joint offsets: " + ", ".join(f"{JOINTS[j]} {d[j]:+.2f} (was {d0[j]:+.2f}, +-{std[6 + k]:.2f})"
                                        for k, j in enumerate(model.free)))
    print(f"object in the gripper frame: {np.round(1000 * p, 1)} mm (start: {np.round(1000 * p0, 1)} mm)")
    report = {"samples": len(samples), "held_out_px": [h[0] for h in held], "held_out_mm": [h[1] for h in held],
              "stereo_pose_change_mm": dS_mm, "stereo_pose_change_deg": dS_deg, "T_left2base": S.tolist(),
              "offsets": dict(zip(JOINTS[:5], d.tolist())), "object_in_gripper_m": p.tolist(),
              "residuals": {s["name"]: [float(a), float(b), float(c)] for s, a, b, c in zip(samples, *errors(model, x, samples))}}
    json.dump(report, open(os.path.join(out_dir, "fit_report.json"), "w"), indent=1)
    debug = os.path.join(out_dir, "debug")
    os.makedirs(debug, exist_ok=True)
    for s in samples[:: max(1, len(samples) // 8)]:                     # a few images: measured (green), fitted (red)
        img = cv2.imread(s["file"])
        W = img.shape[1] // 2
        uvl, uvr, _ = model.project(x, s["q"])
        for (u, v), off, col in ((s["uvl"], 0, (0, 255, 0)), (s["uvr"], W, (0, 255, 0)), (uvl, 0, (0, 0, 255)), (uvr, W, (0, 0, 255))):
            cv2.circle(img, (int(u + off), int(v)), 9, col, 2)
        cv2.imwrite(os.path.join(debug, s["name"].replace("/", "_") + ".png"), img)
    paths = {tag(f): f for f in files}
    for name, why in rejected.items():                                   # every rejected sample, with the reason
        img = cv2.imread(paths[name][:-5] + ".png")
        if img is None:
            continue
        info = json.load(open(paths[name]))
        W = img.shape[1] // 2
        uvl, uvr, _ = model.project(x, info["q_real"])
        for (u, v), off in ((uvl, 0), (uvr, W)):
            cv2.circle(img, (int(u + off), int(v)), 9, (0, 0, 255), 2)
            cv2.rectangle(img, (int(u + off - args.window), int(v - args.window)), (int(u + off + args.window), int(v + args.window)), (255, 255, 0), 1)
        cv2.putText(img, why, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug, f"rejected_{name.replace('/', '_')}.png"), img)
    report["rejected"] = rejected
    report["sessions"] = args.session
    json.dump(report, open(os.path.join(out_dir, "fit_report.json"), "w"), indent=1)
    print(f"report: {out_dir}/fit_report.json; debug images (green = measured, red = fitted; rejected_*.png "
          f"with the reason, red = predicted, box = search window): {debug}/")
    if args.write:
        st = setup.load("stereo_calib.json")
        prev = st.get("pose_base")
        if prev and prev.get("source") != "gripper object fit":
            st["pose_base_board"] = prev                      # the board-based calibration is kept once
        st["pose_base_previous"] = prev                       # whatever was stored before this fit
        st["pose_base"] = {"T_left2base": S.tolist(), "source": "gripper object fit",
                           "session": [os.path.basename(os.path.normpath(f)) for f in args.session],
                           "std_mm": float(1000 * np.linalg.norm(std[3:6])), "std_deg": float(math.degrees(np.linalg.norm(std[0:3])))}
        setup.save("stereo_calib.json", st, note="stereo pose from the gripper-object fit (previous pose kept as pose_base_previous)")
        arm = setup.arm
        for j in model.free:
            arm["offsets"][JOINTS[j]] = round(float(d[j]), 2)
        arm.setdefault("notes", {})["fitted"] = ("lift/elbow/wrist_flex fitted in session(s) "
                                                 + ", ".join(os.path.basename(os.path.normpath(f)) for f in args.session))
        setup.save("arm.json", arm, note="joint offsets from the gripper-object fit")
        print("written into the setup (stereo_calib.json pose_base, arm.json offsets)")


class LlmDet:
    """The user's LLMDet wrapper: the best-scoring box for a text description (pixel coordinates)."""

    def __init__(self, model_id):
        import torch
        import transformers
        self.torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        self.model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(dev)

    def box(self, bgr, descr, threshold=0.4, pad=10):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, text=[[descr]], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        r = self.processor.post_process_grounded_object_detection(out, threshold=threshold,
                                                                  target_sizes=[(img.height, img.width)])[0]
        if len(r["scores"]) == 0:
            return None
        b = r["boxes"][int(r["scores"].argmax())].cpu().numpy()
        return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


if __name__ == "__main__":
    main()
