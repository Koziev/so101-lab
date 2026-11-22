"""
Record REAL test samples for experiment 11 with the follower arm and the stereo camera of a setup folder.

    python real/record_real.py --setup setups/2026-11_12v_stereo28mm_wrist --out data_real_v11

Everything hardware-specific comes from the setup folder: arm id, joint offsets (simulator = real + offset),
stereo calibration and rectification, the stereo camera's USB port, the taught workspace.
At start the live view is compared with the setup's stereo_reference.png (arm at rest, board clear); if the
camera is displaced by more than --align-px, the tool stops before recording.

Per sample:
  1. You arrange the objects and type the INSTRUCTION as the model should get it ("pick up the cone",
     "take the biggest cube", "grab the white thing"). The task type and the target's shape and colour are
     read from the instruction; you are asked only for what cannot be read from it.
  2. The arm leaves its rest pose along the taught recording, moves to the simulator's ready pose (jaws
     8.5 cm high, 10 cm in front of the pan axis, torque on), waits until the image is still, and the
     rectified stereo pair is captured: the images look like the training images.
  3. You SUPPORT THE ARM and press Enter; only then the torque goes off. You guide the jaws by hand around
     the target, the same way for every object,
     wrist roll close to 0, from above as steeply as possible, close them gently, press Enter.
     Keep the JAW TIPS ABOUT 1 CM ABOVE THE BOARD for every object: that is where the training labels are.
  4. Torque goes back on at that pose, the gripper opens, and the arm returns (ready pose, taught recording)
     to its rest pose, where it waits with torque on while you arrange the next scene.
Commands at the instruction prompt: "q" = finish (writes the Parquet split), "u" = undo the last sample.

Reference point of a sample = the grasp point placed with the RECORDED angles (offsets applied, recorded
wrist roll included) in the simulator's kinematics. evaluate.py compares the grasp point placed with the
PREDICTED angles against it.

Output: <out>/samples/<n>.json + <n>_left.png + <n>_right.png (whole view, 640x360) + <n>_raw.png (the raw
camera frame) and <out>/real/real-00000.parquet (same columns as the simulated test splits). With --image
crop2x (or crop15) the split real_crop2x is written too, for models trained on the high-density workspace
window; for samples already recorded with raw frames: --parquet-only --image crop2x. Evaluate with
--splits real_crop2x.
    python evaluation/evaluate.py --model outputs/qwen35_08b_v11/final --data data_real_v11 --splits real \
        --out outputs/eval_v11_real.json
    python evaluation/analyze_real.py --eval outputs/eval_v11_real.json
Run in the LeRobot environment (robot control, PyBullet for the kinematics).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402
import tasks  # noqa: E402
from setup_io import Setup  # noqa: E402
from so101_sim import objects_v11 as ob  # noqa: E402

OUT_W, OUT_H = 640, 360
READY_GRIPPER_PCT = 25.0


# ----------------------------------------------------------------------------------------- stereo camera
class RectifiedStereo:
    """Side-by-side USB stereo camera -> the rectified, zoomed eye pair the model was trained on."""

    def __init__(self, calib, device):
        c = calib
        self.W, self.H = c.get("image_size", (1280, 720))
        zoom, right_dy = c.get("rectify_zoom", 1.0), c.get("rectify_right_dy_px", 0.0)
        K1, D1 = np.array(c["left"]["K"]), np.array(c["left"]["dist"])
        K2, D2 = np.array(c["right"]["K"]), np.array(c["right"]["dist"])
        R, T = np.array(c["stereo"]["R"]), np.array(c["stereo"]["T_m"]).reshape(3, 1)
        R1, R2, P1, _, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (self.W, self.H), R, T, alpha=0)
        f = P1[0, 0] * zoom
        dy = right_dy * self.H / OUT_H
        PL = np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1.0]])
        PR = np.array([[f, 0, self.W / 2], [0, f, self.H / 2 + dy], [0, 0, 1.0]])
        self.maps = [cv2.initUndistortRectifyMap(K, D, Rr, P, (self.W, self.H), cv2.CV_32FC1)
                     for K, D, Rr, P in ((K1, D1, R1, PL), (K2, D2, R2, PR))]
        self.cap = None
        if device is None:                               # only rectification (rebuilding images from raw frames)
            return
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self.cap.read()
        if not ok or frame.shape[1] != 2 * self.W or frame.shape[0] != self.H:
            raise RuntimeError(f"stereo camera (device {device}) does not deliver {2 * self.W}x{self.H} frames")

    def wait_still(self, settle_s=1.0, still=1.5, timeout=4.0):
        time.sleep(settle_s)
        t0, prev = time.time(), None
        while time.time() - t0 < timeout:
            ok, f = self.cap.read()
            if not ok:
                return
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 90)).astype(np.float32)
            if prev is not None and float(np.mean(np.abs(g - prev))) < still:
                return
            prev = g

    def raw(self, flush=5):
        """The current unrectified side-by-side frame."""
        for _ in range(flush):
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def rectify(self, frame):
        """Raw side-by-side frame -> [left, right] rectified eye images at the camera's resolution (BGR)."""
        return [cv2.remap(frame[:, k * self.W:(k + 1) * self.W], mx, my, cv2.INTER_LINEAR)
                for k, (mx, my) in enumerate(self.maps)]

    def read(self, flush=5):
        """(left, right) as the model's whole-view images (640x360, RGB) and the raw frame they came from."""
        frame = self.raw(flush)                         # drops buffered frames: we want the current scene
        eyes = [cv2.cvtColor(common.model_image(r, "full"), cv2.COLOR_BGR2RGB) for r in self.rectify(frame)]
        return eyes, frame


def camera_moved(cam, reference_png, tolerance_px):
    """Compare the live view (arm at rest, as in the reference) with the setup's reference image.
    Returns (moved, text): moved is True if either eye is displaced by more than the tolerance or cannot be
    matched, None if there is no reference image."""
    ref = cv2.imread(reference_png)
    if ref is None:
        return None, f"no reference image ({reference_png}): the camera position was NOT checked"
    from camera.camera_align import Matcher
    frame = cam.raw()
    if frame.shape != ref.shape:
        return True, f"the reference image is {ref.shape[1]}x{ref.shape[0]}, the camera delivers {frame.shape[1]}x{frame.shape[0]}"
    half = ref.shape[1] // 2
    moved, parts = False, []
    for k, name in enumerate(("left", "right")):
        m = Matcher(cv2.cvtColor(ref[:, k * half:(k + 1) * half], cv2.COLOR_BGR2GRAY))
        r = m.measure(cv2.cvtColor(frame[:, k * half:(k + 1) * half], cv2.COLOR_BGR2GRAY))
        if r is None:
            moved = True
            parts.append(f"{name}: no match with the reference")
        else:
            moved = moved or r["mean_px"] > tolerance_px
            parts.append(f"{name}: {r['mean_px']:.1f} px (shift x {r['dx']:+.1f} y {r['dy']:+.1f}, "
                         f"rotation {r['rot_deg']:+.2f} deg)")
    return moved, "camera against the reference image: " + "; ".join(parts)


# ----------------------------------------------------------------------------------------- instruction
def _find(text, vocab):
    """Key of the longest vocabulary word found in the text as a whole word (vocab: key -> list of words)."""
    best = (0, None)
    for key, words in vocab.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", text) and len(w) > best[0]:
                best = (len(w), key)
    return best[1]


SHAPE_VOCAB = {k: list(v) + [tasks.PLURALS[w] for w in v if w in tasks.PLURALS] for k, v in ob.NAMES.items()}
COLOUR_VOCAB = {c: tasks.COLOUR_WORDS.get(c, [c]) for c in ob.COLOURS}


def read_instruction(text):
    """(task type or None, target shape or None, target colour or None) as far as the wording tells."""
    t = text.lower()
    shape, colour = _find(t, SHAPE_VOCAB), _find(t, COLOUR_VOCAB)
    if _find(t, tasks.SUPERLATIVES):
        return "superlative", shape, colour
    if shape and colour:
        return "full", shape, colour
    if shape:
        return "shape", shape, None
    if colour:
        return "colour", None, colour
    return None, None, None


def ask(prompt, default=None, choices=None):
    while True:
        v = input(f"  {prompt}" + (f" [{default}]" if default else "") + ": ").strip().lower() or (default or "")
        if v and (choices is None or v in choices):
            return v
        if choices:
            print("    one of: " + ", ".join(choices))


# ----------------------------------------------------------------------------------------- arm travel
class Travel:
    """Rest pose <-> the simulator's ready pose. With a taught workspace in the setup, the arm leaves and
    reaches the rest pose along the recording (as the calibration tools do); otherwise it moves directly."""

    def __init__(self, robot, setup, urdf, q_ready_real, clearance_mm):
        from calib.lerobot_helpers import move_to, read_joints
        self.robot, self.move_to, self.read = robot, move_to, read_joints
        self.q_ready = np.asarray(q_ready_real, float)
        self.rest = read_joints(robot)
        self.path = None
        ws = setup.load("workspace.json")
        if ws is None:
            print("no workspace.json in the setup: the arm will move DIRECTLY between rest and ready pose")
            return
        from calib.gripper_axes import jaw_points
        from calib.plan import Planner
        from calib.session import thin
        from calib.urdf_fk import UrdfChain
        P = np.asarray(ws["points"], float)[:, :5]
        d = np.asarray(setup.offsets()[:5])
        planner = Planner(UrdfChain(urdf, "gripper_link"), np.eye(3), np.zeros(5))
        pts, src = jaw_points(urdf)
        if "toy" in src:
            raise SystemExit("the gripper geometry could not be read from the URDF (needed for the clearance rule)")
        planner.set_jaw_rule(pts, [q + d for q in P], clearance_mm / 1000)
        jaw_ok = lambda q: planner.jaw_height(np.asarray(q)[:5] + d) >= planner.min_jaw_z
        ok = np.array([jaw_ok(p) for p in P])
        start = int(np.argmin(np.max(np.abs(P - self.rest[:5]), axis=1)))
        after = next((i for i in range(start, len(P)) if ok[i]), None)
        before = next((i for i in range(start, -1, -1) if ok[i]), None)
        cands = [i for i in (after, before) if i is not None]
        if not cands:
            raise SystemExit("no taught pose keeps the jaws above the clearance limit")
        high = min(cands, key=lambda i: abs(i - start))
        steps = [P[high] + (self.q_ready[:5] - P[high]) * t for t in np.linspace(0, 1, 16)]
        if not all(jaw_ok(q) for q in steps):
            raise SystemExit("the move from the taught workspace to the ready pose would come too close to the "
                             "table; lower --clearance-mm or re-teach the workspace")
        self.path = (thin(P, start, high), thin(P, high, start), P[high])
        print(f"rest = taught pose {start}; leaving along the recording to taught pose {high} "
              f"({abs(high - start)} poses), then to the ready pose")

    def _along(self, waypoints, grip):
        for w in waypoints:
            cur = self.read(self.robot)
            dist = float(np.max(np.abs(np.asarray(w)[:5] - cur[:5])))
            self.move_to(self.robot, np.r_[np.asarray(w)[:5], grip], seconds=max(0.4, dist / 25.0), settle=False)

    def to_ready(self):
        if self.path:
            self._along(self.path[0], float(self.rest[5]))
        self.move_to(self.robot, self.q_ready, seconds=3.0)

    def to_rest(self):
        """From anywhere above the board: up to the ready pose first, then back to rest."""
        self.move_to(self.robot, self.q_ready, seconds=3.0)
        if self.path:
            self.move_to(self.robot, np.r_[self.path[2], self.rest[5]], seconds=2.5, settle=False)
            self._along(self.path[1], float(self.rest[5]))
        self.move_to(self.robot, self.rest, seconds=4.0 if self.path else 5.0)


def torque_on_here(robot):
    """Re-enable torque WITHOUT a jump: first make the current pose the goal."""
    from calib.lerobot_helpers import read_joints, send
    send(robot, read_joints(robot))
    robot.bus.enable_torque()


def open_gripper(robot, pct=45.0, seconds=0.8):
    """Open the jaws while the arm holds still, so that the object is not dragged along."""
    from calib.lerobot_helpers import read_joints, send
    q = read_joints(robot)
    for g in np.linspace(q[5], max(q[5], pct), 8):
        send(robot, np.r_[q[:5], g])
        time.sleep(seconds / 8)
    time.sleep(0.3)


def ready_pose_sim(env):
    """The simulator's central ready pose (jaws 8.5 cm high, 10 cm in front of the pan axis), degrees."""
    target = env.pan_axis + env.fwd * 0.10 + np.array([0, 0, 0.085])
    q5, ok, _, _ = env.kin.solve(target, -np.array([0, 0, 1.0]), 0.0, seeds=[env.q_home[:5], np.zeros(5)])
    if not ok:
        raise SystemExit("the simulator's ready pose could not be solved")
    return np.r_[env.to_lerobot(np.r_[q5, env.kin.gripper_closed])[:5], READY_GRIPPER_PCT]


# ----------------------------------------------------------------------------------------- dataset
def write_parquet(out, image="full", setup_dir=None):
    """The samples as a Parquet split: "real" with the whole-view images, or "real_<image>" with the images of
    another mode (common.IMAGE_MODES), cut from the stored raw frames with the setup's rectification."""
    from datasets import Dataset, Features, Image as HFImage, Sequence, Value
    split = "real" if image == "full" else f"real_{image}"
    rect = None
    if image != "full":
        rect = RectifiedStereo(Setup(setup_dir).load("stereo_calib.json"), None)
    rows, no_raw = [], 0
    for f in sorted(glob.glob(os.path.join(out, "samples", "*.json")), key=lambda p: int(os.path.basename(p)[:-5])):
        m = json.load(open(f))
        stem = f[:-5]
        if rect is None:
            eyes = [open(stem + "_left.png", "rb").read(), open(stem + "_right.png", "rb").read()]
        else:
            raw = cv2.imread(stem + "_raw.png")
            if raw is None:                               # recorded before raw frames were kept
                no_raw += 1
                continue
            eyes = [cv2.imencode(".png", common.model_image(r, image))[1].tobytes() for r in rect.rectify(raw)]
        rows.append({"seed": m["index"], "split": split, "camera_scale": 0.0, "task_type": m["task_type"],
                     "instruction": m["instruction"], "canonical": m["instruction"], "target": m["target"],
                     "target_colour": m["target_colour"], "target_shape": m["target_shape"], "prompt": m["prompt"],
                     "answer": common.answer_json(m["joints_deg"][:4]), "joints_deg": m["joints_deg"][:4],
                     "target_pos": m["ref_point"], "approach_tilt_deg": float("nan"), "ik_error_m": 0.0,
                     "n_objects": m["n_objects"], "objects": "[]", "camera": json.dumps({"real": True}),
                     "left": {"bytes": eyes[0], "path": None}, "right": {"bytes": eyes[1], "path": None}})
    if no_raw:
        print(f"{no_raw} samples have no raw frame (<n>_raw.png) and cannot be given in image mode '{image}'")
    if not rows:
        print("no samples for split", split)
        return
    s = lambda: Value("string")
    feats = Features({"seed": Value("int64"), "split": s(), "camera_scale": Value("float32"),
                      "task_type": s(), "instruction": s(), "canonical": s(), "target": s(),
                      "target_colour": s(), "target_shape": s(), "prompt": s(), "answer": s(),
                      "joints_deg": Sequence(Value("float32")), "target_pos": Sequence(Value("float32")),
                      "approach_tilt_deg": Value("float32"), "ik_error_m": Value("float32"),
                      "n_objects": Value("int32"), "objects": s(), "camera": s(),
                      "left": HFImage(), "right": HFImage()})
    os.makedirs(os.path.join(out, split), exist_ok=True)
    path = os.path.join(out, split, f"{split}-00000.parquet")
    Dataset.from_list(rows, features=feats).to_parquet(path)
    counts = {}
    for r in rows:
        counts[r["task_type"]] = counts.get(r["task_type"], 0) + 1
    print(f"wrote {len(rows)} samples to {path}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


def recompute(args):
    """Reference points from the stored readings with the setup's current offsets and cell geometry."""
    setup = Setup(args.setup)
    offsets = np.asarray(setup.offsets(), float)
    cell = common.cell_config(args.setup)
    env = common.make_env(0.0, urdf=args.urdf, cell=cell)
    env.reset(seed=0, render=False)
    moved = []
    for f in glob.glob(os.path.join(args.out, "samples", "*.json")):
        m = json.load(open(f))
        q_sim = np.asarray(m["q_real"], float)[:5] + offsets[:5]
        ref = env.kin.points(env.from_lerobot(np.r_[q_sim, 50.0])[:5], common.grasp_local(env))[0]
        moved.append(100 * np.linalg.norm(ref - np.asarray(m["ref_point"], float)))
        m.update({"offsets": offsets.tolist(), "joints_deg": q_sim.tolist(), "ref_point": [float(v) for v in ref]})
        json.dump(m, open(f, "w"), indent=1)
    env.close()
    if moved:
        print(f"{len(moved)} samples: reference points moved by {np.median(moved):.2f} cm (median), "
              f"{np.max(moved):.2f} cm at most")
    json.dump({"cell": cell, "setup": setup.path, "joint_keys": common.JOINT_KEYS},
              open(os.path.join(args.out, "dataset_info.json"), "w"), indent=1)
    write_parquet(args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--out", default="data_real_v11")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    ap.add_argument("--clearance-mm", type=float, default=20.0)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--max-step", type=float, default=20.0)
    ap.add_argument("--max-roll", type=float, default=8.0, help="warn if the recorded wrist roll exceeds this (deg)")
    ap.add_argument("--align-px", type=float, default=3.0,
                    help="stop if the stereo camera is displaced from stereo_reference.png by more than this")
    ap.add_argument("--recompute", action="store_true",
                    help="no recording: recompute the reference points of the samples in <out>/samples from their "
                         "stored arm readings with the setup's CURRENT offsets and board height, rebuild the Parquet")
    ap.add_argument("--image", choices=list(common.IMAGE_MODES), default="full",
                    help="also write the split real_<image> with the images of that mode (cut from the raw frames, "
                         "which are always stored); with --parquet-only: for samples recorded earlier")
    ap.add_argument("--parquet-only", action="store_true", help="only rebuild the Parquet file(s) from <out>/samples")
    args = ap.parse_args()
    if args.parquet_only:
        write_parquet(args.out)
        if args.image != "full":
            write_parquet(args.out, args.image, args.setup)
        return

    if args.recompute:
        recompute(args)
        return

    from calib.lerobot_helpers import make_robot, read_joints
    setup = Setup(args.setup)
    offsets = np.asarray(setup.offsets(), float)                 # simulator = real + offset
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    cell = common.cell_config(args.setup)
    env = common.make_env(0.0, urdf=args.urdf, cell=cell, real_background=False)
    env.reset(seed=0, render=False)
    q_ready = ready_pose_sim(env)
    q_ready_real = np.r_[q_ready[:5] - offsets[:5], q_ready[5]]
    cam = RectifiedStereo(setup.load("stereo_calib.json"), setup.device("stereo"))
    robot = make_robot(args.port, setup.info["arm_id"], args.max_step)
    input("The arm should be folded in its REST pose now, the board clear around it. Press Enter to continue...")
    moved, text = camera_moved(cam, setup.file("stereo_reference.png"), args.align_px)
    print(text)
    if moved and input(f"The camera seems to have MOVED (tolerance {args.align_px:g} px; the arm must be at rest and "
                       "the board clear for this check). Realign it with camera/camera_align.py. "
                       "Record anyway? [y/N]: ").strip().lower() != "y":
        robot.disconnect()
        env.close()
        return
    travel = Travel(robot, setup, args.urdf, q_ready_real, args.clearance_mm)
    existing = [int(os.path.basename(p)[:-5]) for p in glob.glob(os.path.join(args.out, "samples", "*.json"))]
    n = max(existing) + 1 if existing else 0
    print(f"offsets (deg): {dict(zip(common.JOINT_KEYS, np.round(offsets[:4], 2).tolist()))}")
    print(f"ready pose (arm readings): {np.round(q_ready_real, 1)} | next sample #{n}")
    try:
        while True:
            text = input("\nInstruction (e.g. 'pick up the cone'), 'u' = undo last, 'q' = finish: ").strip()
            if text.lower() == "q":
                break
            if text.lower() == "u":
                if n > 0:
                    for f in glob.glob(os.path.join(args.out, "samples", f"{n - 1}.json")) + \
                            glob.glob(os.path.join(args.out, "samples", f"{n - 1}_*.png")):
                        os.remove(f)
                    n -= 1
                    print(f"removed sample #{n}")
                continue
            if not text:
                continue
            text = text.rstrip(".")
            ttype, shape, colour = read_instruction(text)
            if ttype is None:
                ttype = ask("task type", choices=list(tasks.DEFAULT_WEIGHTS))
            if shape is None:
                shape = ask("shape of the target", choices=list(ob.NAMES))
            if colour is None:
                colour = ask("colour of the target", default="white", choices=list(ob.COLOURS))
            n_obj = ask("number of objects on the board, target included", default="3")
            n_obj = int(n_obj) if n_obj.isdigit() else 3
            print(f"  -> {ttype} task, target: {colour} {ob.NAMES[shape][0]}, {n_obj} object(s)")

            travel.to_ready()
            cam.wait_still(args.settle_s)
            (left, right), raw_frame = cam.read()
            cv2.imshow("captured (left | right) - check that the target is visible",
                       cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_RGB2BGR))
            cv2.waitKey(500)
            input("  SUPPORT THE ARM with your hand (the torque will be released), then press Enter...")
            robot.bus.disable_torque()
            input("  torque OFF: guide the jaws around the target, jaw tips about 1 cm above the board (wrist roll ~0, from above), "
                  "close them gently on it, hold still, press Enter...")
            q = np.mean([read_joints(robot) for _ in range(5)], axis=0)
            torque_on_here(robot)
            if abs(q[4] + offsets[4]) > args.max_roll:
                print(f"  wrist roll is {q[4] + offsets[4]:+.1f} deg (> {args.max_roll}): the reference point is "
                      "still exact, but please keep the roll near 0 where possible")
            keep = input(f"  recorded {np.round(q[:5], 1)}  keep this sample? [Y/n]: ").strip().lower() != "n"
            if keep:
                q_sim = q[:5] + offsets[:5]
                ref = env.kin.points(env.from_lerobot(np.r_[q_sim, 50.0])[:5], common.grasp_local(env))[0]
                stem = os.path.join(args.out, "samples", str(n))
                cv2.imwrite(stem + "_left.png", cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
                cv2.imwrite(stem + "_right.png", cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
                cv2.imwrite(stem + "_raw.png", raw_frame)        # any other image mode can be cut from this later
                json.dump({"index": n, "instruction": text, "task_type": ttype,
                           "target": f"{colour} {ob.NAMES[shape][0]}", "target_shape": shape, "target_colour": colour,
                           "prompt": common.PROMPT.format(instruction=text), "n_objects": n_obj,
                           "q_real": q.tolist(), "offsets": offsets.tolist(), "joints_deg": q_sim.tolist(),
                           "ref_point": [float(v) for v in ref], "setup": setup.path,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(stem + ".json", "w"), indent=1)
                print(f"  saved sample #{n}")
                n += 1
            open_gripper(robot)
            travel.to_rest()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            torque_on_here(robot)
            print("returning to the rest pose...")
            travel.to_rest()
        except Exception as e:
            print("could not park the arm:", e)
        robot.disconnect()
        cv2.destroyAllWindows()
        env.close()
    json.dump({"cell": cell, "setup": setup.path, "joint_keys": common.JOINT_KEYS},
              open(os.path.join(args.out, "dataset_info.json"), "w"), indent=1)
    write_parquet(args.out)
    if args.image != "full":
        write_parquet(args.out, args.image, args.setup)


if __name__ == "__main__":
    main()
