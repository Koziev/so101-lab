"""
Calibrate the USB stereo camera (side-by-side 1280x480 MJPEG: two 640x480 eyes) so the
simulator can reproduce it: field of view, lens distortion, baseline, and the camera's
position/tilt relative to the robot.

Print a checkerboard (default 9x6 inner corners = 10x7 squares) on A4, glue it to
something flat, and MEASURE the printed square size (e.g. 25 mm).

1. Capture ~20 views of the checkerboard, held at different distances/angles, filling
   different parts of the image (press SPACE to save a pair, Q to quit):
       python calibrate_stereo.py capture --device 0 --out calib_frames

2. Intrinsics + stereo calibration:
       python calibrate_stereo.py calibrate --frames calib_frames --square-mm 25

3. Camera pose. For a low camera (yours, ~8 cm) stand the checkerboard UPRIGHT on the
   plywood, facing the camera, centred on the line from the shoulder-pan axis to the
   camera, long side horizontal. Measure the horizontal distance from the pan axis to the
   checkerboard (D, cm) and the height of the checkerboard's centre above the plywood
   (H, cm). Capture one pair (step 1 into another folder) and run:
       python calibrate_stereo.py pose --image pose_frames/pair_000.png --square-mm 25 \
           --upright --center-cm D --center-height-cm H
   (A checkerboard lying flat works only for a camera looking down steeply; omit
   --upright and --center-height-cm then.)

Everything is written to stereo_calib.json; send me that file.
"""
import argparse
import glob
import json
import math
import os

import cv2
import numpy as np


def split(img):
    h, w = img.shape[:2]
    return img[:, :w // 2], img[:, w // 2:]


def board_points(cols, rows, square):
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def find_corners(gray, cols, rows):
    ok, c = cv2.findChessboardCorners(gray, (cols, rows),
                                      cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    else:
        # the sector-based detector copes better with oblique views and uneven lighting; its corners are
        # already sub-pixel accurate. (The pose step does not depend on the corners' start/order.)
        ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            return None
    # OpenCV versions differ (N,1,2) vs (N,2): normalise to (N,1,2) float32
    return np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)


# ---------------------------------------------------------------------------- capture
def capture(args):
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    os.makedirs(args.out, exist_ok=True)
    n = len(glob.glob(os.path.join(args.out, "pair_*.png")))
    print("SPACE: save pair, Q: quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        view = frame.copy()
        for k, eye in enumerate(split(frame)):
            c = find_corners(cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY), args.cols, args.rows)
            if c is not None:
                c = c.copy()
                c[:, 0, 0] += k * eye.shape[1]
                cv2.drawChessboardCorners(view, (args.cols, args.rows), c, True)
        if view.shape[1] > args.preview_width:  # shrink only the preview; saved pairs keep full resolution
            scale = args.preview_width / view.shape[1]
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(view, f"saved: {n}  ({frame.shape[1]}x{frame.shape[0]})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("stereo", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite(os.path.join(args.out, f"pair_{n:03d}.png"), frame)
            n += 1
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------- calibrate
def calibrate(args):
    obj = board_points(args.cols, args.rows, args.square_mm / 1000.0)
    objs, pts_l, pts_r, size = [], [], [], None
    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    for f in files:
        L, R = split(cv2.imread(f))
        size = (L.shape[1], L.shape[0])
        cl = find_corners(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), args.cols, args.rows)
        cr = find_corners(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), args.cols, args.rows)
        if cl is not None and cr is not None:
            objs.append(obj); pts_l.append(cl); pts_r.append(cr)
    print(f"{len(objs)} of {len(files)} pairs usable (checkerboard found in both eyes)")
    if len(objs) < 8:
        raise SystemExit("need at least 8 usable pairs; capture more views")
    res = {}
    Ks, Ds = [], []
    for name, pts in (("left", pts_l), ("right", pts_r)):
        rms, K, D, _, _ = cv2.calibrateCamera(objs, pts, size, None, None)
        Ks.append(K); Ds.append(D)
        fx, fy = K[0, 0], K[1, 1]
        res[name] = {"rms_px": rms, "K": K.tolist(), "dist": D.ravel().tolist(),
                     "hfov_deg": math.degrees(2 * math.atan(size[0] / (2 * fx))),
                     "vfov_deg": math.degrees(2 * math.atan(size[1] / (2 * fy)))}
        print(f"{name}: rms {rms:.2f} px, fx {fx:.1f} fy {fy:.1f}, cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, "
              f"HFOV {res[name]['hfov_deg']:.1f} deg, VFOV {res[name]['vfov_deg']:.1f} deg, "
              f"distortion k1 {D.ravel()[0]:+.3f} k2 {D.ravel()[1]:+.3f}")
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objs, pts_l, pts_r, Ks[0], Ds[0], Ks[1], Ds[1], size, flags=cv2.CALIB_FIX_INTRINSIC)
    ang = np.degrees(cv2.Rodrigues(R)[0].ravel())
    res["stereo"] = {"rms_px": rms, "R": R.tolist(), "T_m": T.ravel().tolist(),
                     "baseline_m": float(np.linalg.norm(T)), "rotation_deg": ang.tolist()}
    res["image_size"] = list(size)
    print(f"stereo: rms {rms:.2f} px, baseline {100 * np.linalg.norm(T):.2f} cm "
          f"(T = {np.round(100 * T.ravel(), 2)} cm), relative rotation {np.round(ang, 2)} deg")
    save(args.out, res)


# ---------------------------------------------------------------------------- pose
def pose(args):
    res = load(args.out)
    if "left" not in res:
        raise SystemExit("run 'calibrate' first")
    K, D = np.array(res["left"]["K"]), np.array(res["left"]["dist"])
    L, _ = split(cv2.imread(args.image))
    c = find_corners(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), args.cols, args.rows)
    if c is None:
        raise SystemExit("checkerboard not found in the left eye")
    obj = board_points(args.cols, args.rows, args.square_mm / 1000.0)
    obj -= obj.mean(axis=0)  # origin at the checkerboard centre
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, D)
    Rm = cv2.Rodrigues(rvec)[0]
    cam = (-Rm.T @ tvec).ravel()                    # camera centre in checkerboard coordinates
    view = Rm.T @ np.array([0.0, 0.0, 1.0])          # optical axis
    right = Rm.T @ np.array([1.0, 0.0, 0.0])         # image x axis
    img_down = Rm.T @ np.array([0.0, 1.0, 0.0])      # image y axis (points down in the image)
    if args.upright:   # board standing vertically: world "up" is +-y of the board, opposite to image-down
        up = np.array([0.0, 1.0, 0.0])
        up = -up if img_down @ up > 0 else up
        base_h = args.center_height_cm / 100.0
    else:              # board flat on the table: world "up" is the board normal on the camera's side
        up = np.array([0.0, 0.0, 1.0])
        up = -up if cam @ up < 0 else up
        base_h = 0.0
    h_rel = cam @ up
    horiz = cam - h_rel * up
    dist = np.linalg.norm(horiz)
    toward = -horiz / dist                            # horizontal direction camera -> checkerboard centre
    view_h = view - (view @ up) * up
    pitch = math.degrees(math.asin(np.clip(-(view @ up), -1, 1)))   # positive = looking down
    yaw = math.degrees(math.atan2(np.cross(toward, view_h) @ up, view_h @ toward))
    roll = math.degrees(math.asin(np.clip(right @ up, -1, 1)))
    res["pose_left_eye"] = {
        "height_above_table_m": float(base_h + h_rel),
        "distance_to_pan_axis_m": float(dist + args.center_cm / 100.0),
        "pitch_down_deg": pitch, "yaw_deg": yaw, "roll_deg": roll,
        "mode": "upright" if args.upright else "flat",
        "assumes": "checkerboard centred on the pan-axis/camera line, facing the camera",
    }
    print(f"left eye: {100 * (base_h + h_rel):.1f} cm above the board, {100 * (dist + args.center_cm / 100):.1f} cm "
          f"from the pan axis (horizontally), pitch {pitch:.1f} deg down, yaw {yaw:+.1f} deg, roll {roll:+.1f} deg")
    save(args.out, res)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save(path, res):
    old = load(path)
    old.update(res)
    with open(path, "w") as f:
        json.dump(old, f, indent=2)
    print("written", path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "calibrate", "pose"):
        p = sub.add_parser(name)
        p.add_argument("--cols", type=int, default=9, help="inner corners per row")
        p.add_argument("--rows", type=int, default=6, help="inner corners per column")
        p.add_argument("--out", default="stereo_calib.json" if name != "capture" else "calib_frames")
    sub.choices["capture"].add_argument("--device", type=int, default=0)
    sub.choices["capture"].add_argument("--width", type=int, default=1280)
    sub.choices["capture"].add_argument("--height", type=int, default=480)
    sub.choices["capture"].add_argument("--preview-width", type=int, default=1280,
                                        help="max width of the preview window (saved frames are not resized)")
    sub.choices["calibrate"].add_argument("--frames", default="calib_frames")
    sub.choices["calibrate"].add_argument("--square-mm", type=float, required=True)
    sub.choices["pose"].add_argument("--image", required=True)
    sub.choices["pose"].add_argument("--square-mm", type=float, required=True)
    sub.choices["pose"].add_argument("--center-cm", type=float, required=True,
                                     help="horizontal distance from the pan axis to the checkerboard centre")
    sub.choices["pose"].add_argument("--upright", action="store_true",
                                     help="checkerboard standing vertically, facing the camera (for a low camera)")
    sub.choices["pose"].add_argument("--center-height-cm", type=float, default=0.0,
                                     help="with --upright: height of the checkerboard centre above the board")
    args = ap.parse_args()
    {"capture": capture, "calibrate": calibrate, "pose": pose}[args.cmd](args)


if __name__ == "__main__":
    main()
