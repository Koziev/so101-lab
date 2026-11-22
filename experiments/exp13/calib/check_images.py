"""
Diagnose a calibration session's saved images: how many ArUco markers and ChArUco corners of each board are
detected in every wrist (or stereo) image, with image size and sharpness.

    python calib/check_images.py setups/<name>/calibration/<time>
    python calib/check_images.py setups/<name>/calibration/<time> --which stereo
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "camera"))
import charuco_board  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--which", choices=["wrist", "stereo"], default="wrist")
    ap.add_argument("--boards", default="2xA4")
    args = ap.parse_args()
    boards = charuco_board.make_board_set(args.boards)
    d = boards[0][1].getDictionary()
    det = cv2.aruco.ArucoDetector(d)
    files = sorted(glob.glob(os.path.join(args.session, "images", f"*_{args.which}.png")))
    if not files:
        raise SystemExit("no images found")
    n_marker_ids = [len(b.getIds()) for _, b in boards]
    rows = []
    for f in files:
        img = cv2.imread(f)
        if args.which == "stereo":
            img = img[:, : img.shape[1] // 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, ids, _ = det.detectMarkers(gray)
        ids = [] if ids is None else ids.ravel().tolist()
        per_board = []
        for k, (label, b) in enumerate(boards):
            lo = sum(n_marker_ids[:k])
            m = sum(1 for i in ids if lo <= i < lo + n_marker_ids[k])
            obj, _ = charuco_board.detect(b, gray)
            per_board.append((label, m, 0 if obj is None else len(obj)))
        sharp = cv2.Laplacian(gray[gray.shape[0] // 4: 3 * gray.shape[0] // 4, gray.shape[1] // 4: 3 * gray.shape[1] // 4],
                              cv2.CV_64F).var()
        other = [i for i in ids if i >= sum(n_marker_ids)]
        rows.append((os.path.basename(f), img.shape[1], img.shape[0], sharp, per_board, other))
        print(f"{os.path.basename(f):18s} {img.shape[1]}x{img.shape[0]}  sharpness {sharp:6.1f}  "
              + "  ".join(f"board {lab}: {m:2d} markers, {c:2d} corners" for lab, m, c in per_board)
              + (f"  (unknown ids {other[:5]})" if other else ""))
    best = [max(c for _, _, c in r[4]) for r in rows]
    print(f"\n{len(rows)} images; corners of the best-seen board per image: "
          f"0: {sum(b == 0 for b in best)}, 1-5: {sum(0 < b < 6 for b in best)}, 6-9: {sum(5 < b < 10 for b in best)}, "
          f">=10: {sum(b >= 10 for b in best)}; images with any marker: {sum(any(m for _, m, _ in r[4]) for r in rows)}")


if __name__ == "__main__":
    main()
