"""
Live view of the side-by-side USB stereo camera, for positioning the camera and the checkerboard.

    python capture_stereocamera_stream.py                  # wide mode 2560x720 (two 1280x720 eyes)
    python capture_stereocamera_stream.py --device 2 --cols 9 --rows 6

Shown at half size: both eyes with a vertical centre line each (align the checkerboard / the robot's
forward line with them) and, per eye, whether the full checkerboard is currently detected (green = yes).
Keys: s = save the full-resolution frame as PNG (lossless, timestamped), q = quit.
"""
import argparse
import time

import cv2


def detect(gray, pattern):
    """Fast check on a half-size image: is the whole pattern visible in this eye?"""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ok, _ = cv2.findChessboardCorners(small, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                                      | cv2.CALIB_CB_FAST_CHECK)
    if not ok:                                   # the sector-based detector copes better with oblique views
        ok, _ = cv2.findChessboardCornersSB(small, pattern)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=2560, help="full side-by-side width (2560 = wide mode)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--check-every", type=int, default=5, help="run the checkerboard check every N frames")
    args = ap.parse_args()

    cam = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # before the size: needed for 2560x720 @ 30
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cam.set(cv2.CAP_PROP_FPS, 30)
    if not cam.isOpened():
        raise SystemExit("could not open the camera")
    w, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cam.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"camera delivers {w}x{h} @ {cam.get(cv2.CAP_PROP_FPS):.0f} fps, format {fourcc}")
    if (w, h) != (args.width, args.height):
        print(f"WARNING: requested {args.width}x{args.height}; the calibration and all tools expect 2560x720")
    print("keys: s = save PNG, q = quit")

    pattern, status, n = (args.cols, args.rows), [False, False], 0
    while True:
        ok, frame = cam.read()
        if not ok:
            print("cannot receive frame")
            break
        half = frame.shape[1] // 2
        if n % args.check_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            status = [detect(gray[:, :half], pattern), detect(gray[:, half:], pattern)]
        n += 1
        view = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        vh, vw = view.shape[:2]
        for k, name in enumerate(("left", "right")):
            x0 = k * vw // 2
            cv2.line(view, (x0 + vw // 4, 0), (x0 + vw // 4, vh), (255, 255, 0), 1)       # eye centre line
            colour = (0, 200, 0) if status[k] else (0, 0, 255)
            cv2.putText(view, f"{name}: {'board OK' if status[k] else 'no board'}", (x0 + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.line(view, (vw // 2, 0), (vw // 2, vh), (255, 255, 255), 2)                   # border between the eyes
        cv2.imshow("stereo camera (half size)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = time.strftime("stereo_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, frame)
            print(f"saved {name} ({frame.shape[1]}x{frame.shape[0]}, both eyes: "
                  f"{'board OK' if all(status) else 'board NOT detected in both eyes'})")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
