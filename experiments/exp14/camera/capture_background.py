"""
The empty scene as the stereo camera sees it, without the arm: several raw stereo frames with the arm in
different places, combined by a per-pixel median (each part of the scene is hidden by the arm in only a few
frames). The simulator projects this image back onto its scene from the calibrated camera pose.

    python camera/capture_background.py --setup setups/<name> [--frames 9]

Before: remove the calibration boards and all objects; keep the lighting as during normal operation.
For each frame: move the arm by hand (torque off, or with the leader arm) to a clearly different place, e.g.
folded left, folded right, raised high, stretched over the far corners; keep it still, press Enter.
Writes <setup>/stereo_background.png (full raw 2560x720 frame, both eyes) and the individual frames.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from setup_io import Setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--device", type=int, default=None, help="stereo camera index (default: found by USB port)")
    ap.add_argument("--add", default=None, metavar="NAME",
                    help="keep the existing picture and write a further one, stereo_background_<NAME>.png (same "
                         "camera position; other light, other day, other things around the board): the simulator "
                         "draws one of the setup's background pictures per episode")
    args = ap.parse_args()
    target = f"stereo_background_{args.add}.png" if args.add else "stereo_background.png"
    setup = Setup(args.setup)
    idx = args.device if args.device is not None else setup.device("stereo")
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    out_dir = setup.file("background_frames" + (f"_{args.add}" if args.add else ""))
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    print("Remove the boards and objects. For each frame: move the arm to a clearly different place, keep it still.")
    for k in range(args.frames):
        input(f"[{k + 1}/{args.frames}] arm in a new place and still? Enter = take the frame...")
        for _ in range(5):                                       # flush stale frames
            cap.read()
        acc = []
        for _ in range(5):                                       # average 5 frames: less sensor noise
            ok, f = cap.read()
            if ok:
                acc.append(f.astype(np.float32))
            time.sleep(0.05)
        if not acc:
            raise SystemExit("camera read failed")
        f = np.mean(acc, axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{k:02d}.png"), f)
        frames.append(f)
    cap.release()
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(setup.file(target), med)
    # how clean is it? pixels where the frames disagree a lot are probably still showing parts of the arm
    spread = np.median(np.abs(np.stack(frames).astype(np.int16) - med[None].astype(np.int16)), axis=0).max(axis=2)
    bad = float(np.mean(spread > 25))
    cv2.imwrite(os.path.join(out_dir, "disagreement.png"), np.clip(spread * 4, 0, 255).astype(np.uint8))
    setup.info.setdefault("files", []).append(target)
    setup.save("setup.json", setup.info, note=f"stereo background from {len(frames)} frames (median)")
    print(f"written {setup.file(target)}; {100 * bad:.1f}% of pixels with large disagreement "
          f"(see {os.path.basename(out_dir)}/disagreement.png; bright = arm still present in many frames: add frames "
          f"with the arm elsewhere)")


if __name__ == "__main__":
    main()
