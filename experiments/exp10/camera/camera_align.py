"""
Restore the stereo camera's position by aligning the live view with a reference ("etalon") image.

    python camera_align.py --etalon etalon.png               # align against a saved reference
    python camera_align.py --etalon etalon.png --save-new    # 'n' may overwrite the reference

The reference is a full-resolution side-by-side frame (2560x720) saved by capture_stereocamera_stream.py
with the camera in its calibrated position. Take it WITHOUT checkerboard and objects and with the arm
folded in its rest pose: only things that stay put can be matched later (wood grain, board edges, base).

The window shows both eyes at half size in one of four modes (key m):
  edges   reference edges in red over the live image (default): move the camera until they coincide
  blend   50/50 mix of reference and live image
  diff    absolute difference: aligned = dark, misaligned edges light up
  live    the live image only
Per eye, the remaining misalignment is measured by matching features between reference and live image
(ORB + RANSAC homography): average shift (px, full resolution), rotation (deg) and scale. "ALIGNED"
appears when the average displacement is below --tolerance pixels (default 2 px ~ 0.08 deg).
Keys: m = next mode, s = save the current frame, n = save it as the new reference (with --save-new), q = quit.
"""
import argparse
import math
import time

import cv2
import numpy as np

MODES = ["edges", "blend", "diff", "live"]


class Matcher:
    """Estimates how the live eye image is displaced relative to the reference eye image."""

    def __init__(self, ref_gray):
        self.orb = cv2.ORB_create(3000)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(ref_gray, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        h, w = ref_gray.shape
        xs, ys = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 5), np.linspace(0.1 * h, 0.9 * h, 4))
        self.grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
        self.center = np.array([w / 2, h / 2])

    def measure(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or self.des_ref is None or len(kp) < 20:
            return None
        pairs = self.bf.knnMatch(self.des_ref, des, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            return None
        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or inl.sum() < 15:
            return None
        moved = cv2.perspectiveTransform(self.grid, H).reshape(-1, 2) - self.grid.reshape(-1, 2)
        A = H[:2, :2] / H[2, 2]
        return {"dx": float(moved[:, 0].mean()), "dy": float(moved[:, 1].mean()),
                "mean_px": float(np.linalg.norm(moved, axis=1).mean()),
                # positive = counter-clockwise as seen on screen (image y axis points down)
                "rot_deg": -math.degrees(math.atan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])),
                "scale": float(math.sqrt(abs(np.linalg.det(A)))), "inliers": int(inl.sum())}


def overlay(ref, live, mode):
    if mode == "blend":
        return cv2.addWeighted(ref, 0.5, live, 0.5, 0)
    if mode == "diff":
        return cv2.absdiff(ref, live)
    if mode == "edges":
        out = live.copy()
        edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 60, 160) > 0
        out[edges] = (0, 0, 255)
        return out
    return live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etalon", required=True, help="reference frame (2560x720 side-by-side PNG)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=2.0, help="ALIGNED below this mean displacement (px)")
    ap.add_argument("--every", type=int, default=3, help="measure every N frames")
    ap.add_argument("--save-new", action="store_true", help="allow 'n' to overwrite the reference image")
    args = ap.parse_args()

    ref = cv2.imread(args.etalon)
    if ref is None:
        raise SystemExit(f"cannot read {args.etalon}")
    H, W = ref.shape[:2]
    cam = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cam.set(cv2.CAP_PROP_FPS, 30)
    w, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (W, H):
        raise SystemExit(f"camera delivers {w}x{h}, the reference is {W}x{H}")
    half = W // 2
    matchers = [Matcher(cv2.cvtColor(ref[:, k * half:(k + 1) * half], cv2.COLOR_BGR2GRAY)) for k in (0, 1)]
    print(f"reference {args.etalon}: {len(matchers[0].kp_ref)} / {len(matchers[1].kp_ref)} features (left / right)")
    print("keys: m = mode, s = save frame, n = new reference" + ("" if args.save_new else " (disabled)") + ", q = quit")

    mode, n, res = 0, 0, [None, None]
    while True:
        ok, frame = cam.read()
        if not ok:
            print("cannot receive frame")
            break
        if n % args.every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = [matchers[k].measure(gray[:, k * half:(k + 1) * half]) for k in (0, 1)]
        n += 1
        view = cv2.resize(overlay(ref, frame, MODES[mode]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        vw = view.shape[1]
        cv2.line(view, (vw // 2, 0), (vw // 2, view.shape[0]), (255, 255, 255), 2)
        for k, name in enumerate(("left", "right")):
            x0, r = k * vw // 2 + 10, res[k]
            if r is None:
                lines, colour = [f"{name}: no match (too different / too few features)"], (0, 0, 255)
            else:
                aligned = r["mean_px"] < args.tolerance
                colour = (0, 200, 0) if aligned else (0, 165, 255)
                lines = [f"{name}: {'ALIGNED' if aligned else 'off by'} {r['mean_px']:.1f} px",
                         f"shift x {r['dx']:+.1f} y {r['dy']:+.1f} px  rot {r['rot_deg']:+.2f} deg",
                         f"scale {r['scale']:.4f}  inliers {r['inliers']}"]
            for i, t in enumerate(lines):
                cv2.putText(view, t, (x0, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        cv2.putText(view, f"mode: {MODES[mode]} (m)", (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 2)
        cv2.imshow("camera alignment (half size)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mode = (mode + 1) % len(MODES)
        if key == ord("s"):
            name = time.strftime("align_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, frame)
            print("saved", name)
        if key == ord("n") and args.save_new:
            cv2.imwrite(args.etalon, frame)
            ref = frame.copy()
            matchers = [Matcher(cv2.cvtColor(ref[:, k * half:(k + 1) * half], cv2.COLOR_BGR2GRAY)) for k in (0, 1)]
            print("new reference saved to", args.etalon)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
