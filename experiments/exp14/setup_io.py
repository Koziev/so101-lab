"""
A setup folder holds everything that describes one hardware configuration, so that tools take a single
--setup argument and experiments can record which setup they used.

    setups/2026-11_12v_stereo28mm_wrist/
        setup.json            description, date, arm id, camera USB ports, history of changes
        arm.json              joint signs and offsets (simulator = real + offset), per-joint notes
        stereo_calib.json     stereo intrinsics, rectification crop/alignment, pose (pose_left_eye, pose_base)
        stereo_reference.png  reference image for camera_align.py
        wrist_calib.json      wrist-camera intrinsics
        wrist_mount.json      wrist camera -> gripper_link (from the automated calibration)
        wrist_reference.png   jaws at a fixed opening (bracket check)
        calibration/          automated-calibration sessions (observations, images, reports)

Create one from the files you have:
    python setup_io.py create --dir setups/2026-11_12v_stereo28mm_wrist --arm-id my_follower_12v \
        --stereo-calib stereo_calib_28mm.json --stereo-ref stereo_20261111_092915.png \
        --wrist-calib wrist_calib.json --joint-signs joint_signs.json --pan-offset -0.5 \
        --stereo-usb 1-13.3 --wrist-usb 3-1
    python setup_io.py show --dir setups/2026-11_12v_stereo28mm_wrist
"""
import argparse
import glob
import json
import os
import re
import shutil
import time

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class Setup:
    def __init__(self, path):
        self.path = path
        f = os.path.join(path, "setup.json")
        if not os.path.exists(f):
            raise FileNotFoundError(f"{path} is not a setup folder (no setup.json)")
        self.info = json.load(open(f))

    def file(self, name):
        return os.path.join(self.path, name)

    def load(self, name, default=None):
        f = self.file(name)
        return json.load(open(f)) if os.path.exists(f) else default

    def save(self, name, data, note=""):
        json.dump(data, open(self.file(name), "w"), indent=2)
        self.info.setdefault("history", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "file": name, "note": note})
        json.dump(self.info, open(self.file("setup.json"), "w"), indent=2)

    @property
    def arm(self):
        return self.load("arm.json", {"signs": {j: 1 for j in JOINTS}, "offsets": {j: 0.0 for j in JOINTS}})

    def offsets(self):
        a = self.arm
        return [float(a["offsets"].get(j, 0.0)) for j in JOINTS]

    def device(self, which):
        """Video device index of 'stereo' or 'wrist', found by its USB port (indexes change, ports do not)."""
        port = self.info.get("cameras", {}).get(which, {}).get("usb")
        idx = find_device_by_usb(port) if port else None
        if idx is None:
            idx = self.info.get("cameras", {}).get(which, {}).get("index")
        if idx is None:
            raise RuntimeError(f"{which} camera not found (usb port {port}); check the connection or setup.json")
        return idx


def find_device_by_usb(port):
    """First /dev/videoN on the given USB port that actually delivers frames (Linux)."""
    import cv2
    for node in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        idx = int(re.sub(r"\D", "", node))
        try:
            real = os.path.realpath(f"/sys/class/video4linux/video{idx}/device")
            usb = re.findall(r"/(\d+-[\d.]+):", real)[-1]
        except (OSError, IndexError):
            continue
        if usb != port:
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return idx
    return None


def create(args):
    os.makedirs(args.dir, exist_ok=True)
    copies = {"stereo_calib.json": args.stereo_calib, "stereo_reference.png": args.stereo_ref,
              "wrist_calib.json": args.wrist_calib, "wrist_reference.png": args.wrist_ref}
    files = []
    for name, src in copies.items():
        if src:
            shutil.copy(src, os.path.join(args.dir, name))
            files.append(name)
    signs = json.load(open(args.joint_signs)) if args.joint_signs else {j: 1 for j in JOINTS}
    arm = {"arm_id": args.arm_id, "convention": "simulator angle = real reading + offset (deg)",
           "signs": signs, "offsets": {j: 0.0 for j in JOINTS},
           "notes": {"shoulder_pan": f"physical check (pointing along the board's centre line): {args.pan_offset:+.1f}"}}
    arm["offsets"]["shoulder_pan"] = args.pan_offset
    json.dump(arm, open(os.path.join(args.dir, "arm.json"), "w"), indent=2)
    info = {"name": os.path.basename(os.path.normpath(args.dir)), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": args.description, "arm_id": args.arm_id,
            "cameras": {"stereo": {"usb": args.stereo_usb, "mode": "2560x720 MJPG side by side"},
                        "wrist": {"usb": args.wrist_usb, "mode": "1280x720 MJPG"}},
            "files": files + ["arm.json"], "history": []}
    json.dump(info, open(os.path.join(args.dir, "setup.json"), "w"), indent=2)
    print(f"setup folder {args.dir}: " + ", ".join(info["files"]))


def show(args):
    s = Setup(args.dir)
    print(json.dumps(s.info, indent=2))
    print("offsets (simulator = real + offset):", dict(zip(JOINTS, s.offsets())))
    for name in ("stereo_calib.json", "wrist_calib.json", "wrist_mount.json"):
        print(f"{name:20s} {'present' if os.path.exists(s.file(name)) else 'missing'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--dir", required=True)
    c.add_argument("--arm-id", required=True)
    c.add_argument("--description", default="")
    c.add_argument("--stereo-calib", default="")
    c.add_argument("--stereo-ref", default="")
    c.add_argument("--wrist-calib", default="")
    c.add_argument("--wrist-ref", default="")
    c.add_argument("--joint-signs", default="")
    c.add_argument("--pan-offset", type=float, default=0.0)
    c.add_argument("--stereo-usb", default="")
    c.add_argument("--wrist-usb", default="")
    sh = sub.add_parser("show")
    sh.add_argument("--dir", required=True)
    args = ap.parse_args()
    {"create": create, "show": show}[args.cmd](args)


if __name__ == "__main__":
    main()
