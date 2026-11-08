"""
Record REAL validation samples for experiment 10 with the SO-101 follower and the stereo camera.

    python real/record_real.py --port /dev/ttyACM0 --id my_follower --calib ../stereo_calib_wide.json --out data_real

Per sample:
  1. You place the target and 1-3 other objects on the board and type the target's description
     ("blue ball", "red cube", ...).
  2. The arm moves slowly from its rest pose to the ready pose used in simulation (torque on), and the
     rectified stereo pair is captured: the images look like the training images (arm away from the object).
  3. You SUPPORT THE ARM and press Enter; only then the torque goes off. You guide the jaws by hand around
     the CENTRE of the target, wrist roll close to 0, approaching from above as steeply as possible, and
     press Enter. The joint angles are recorded.
  4. Torque goes back ON at the current pose and the arm returns slowly (via the ready pose) to its rest
     pose, where it waits with torque ON while you arrange the next scene.
The torque is never switched off without warning, and the arm never waits in a raised pose.
Commands at the description prompt: "q" = finish (writes the Parquet split), "u" = undo the last sample.

Reference point of a sample = the grasp point placed with the RECORDED angles (including the recorded
wrist roll) in the simulator's kinematics. evaluate.py compares the grasp point placed with the
PREDICTED angles (roll 0) against it, so differences in approach tilt do not count as errors.

Output: data_real/samples/<n>.json + <n>_left.png + <n>_right.png (kept for inspection) and
data_real/real/real-00000.parquet (same schema as the simulated test splits):
    python evaluation/evaluate.py --model outputs/qwen35_08b_full/final --data data_real --splits real --out outputs/eval_real.json
Run in the LeRobot environment (robot control); evaluation runs in the Qwen environment as before.
"""
import argparse
import glob
import io
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OUT_W, OUT_H = 640, 360


# ----------------------------------------------------------------------------------------- stereo camera
class RectifiedStereo:
    """Side-by-side USB stereo camera -> rectified eye pair, exactly as in the policy runner (real_check.py)."""

    def __init__(self, calib_path, device=0):
        c = json.load(open(calib_path))
        self.W, self.H = c["image_size"]
        zoom, right_dy = c.get("rectify_zoom", 1.05), c.get("rectify_right_dy_px", 0.0)
        K1, D1 = np.array(c["left"]["K"]), np.array(c["left"]["dist"])
        K2, D2 = np.array(c["right"]["K"]), np.array(c["right"]["dist"])
        R, T = np.array(c["stereo"]["R"]), np.array(c["stereo"]["T_m"]).reshape(3, 1)
        if abs(T[2, 0]) > 0.05 * np.linalg.norm(T) or abs(T[1, 0]) > 0.05 * np.linalg.norm(T):
            R, T = np.eye(3), np.array([[-np.linalg.norm(T)], [0.0], [0.0]])     # assume parallel eyes
        R1, R2, P1, _, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (self.W, self.H), R, T, alpha=0)
        f = P1[0, 0] * zoom
        dy = right_dy * self.H / OUT_H
        PL = np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1.0]])
        PR = np.array([[f, 0, self.W / 2], [0, f, self.H / 2 + dy], [0, 0, 1.0]])
        self.maps = [cv2.initUndistortRectifyMap(K, D, Rr, P, (self.W, self.H), cv2.CV_32FC1)
                     for K, D, Rr, P in ((K1, D1, R1, PL), (K2, D2, R2, PR))]
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self, flush=5):
        for _ in range(flush):                          # drop buffered frames: we want the current scene
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        eyes = []
        for k, (mx, my) in enumerate(self.maps):
            r = cv2.remap(frame[:, k * self.W:(k + 1) * self.W], mx, my, cv2.INTER_LINEAR)
            r = cv2.resize(r, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            eyes.append(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        return eyes


# ----------------------------------------------------------------------------------------- robot
def make_robot(port, rid, max_step):
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=port, id=rid, use_degrees=True, max_relative_target=max_step,
                                              disable_torque_on_disconnect=True))
    robot.connect(calibrate=False)
    return robot


def read_joints(robot):
    obs = robot.get_observation()
    return np.array([float(obs[f"{j}.pos"]) for j in JOINTS])


def send(robot, q):
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})


def move_to(robot, target, seconds=3.0, hz=30):
    start = read_joints(robot)
    n = max(2, int(seconds * hz))
    for i in range(1, n + 1):
        s = i / n
        s = s ** 3 * (10 - 15 * s + 6 * s ** 2)
        send(robot, start + (np.asarray(target) - start) * s)
        time.sleep(1.0 / hz)
    time.sleep(0.5)


def park(robot, q_ready, rest):
    """Back to the folded rest pose, slowly and with torque ON (via the ready pose, away from the objects)."""
    move_to(robot, q_ready, seconds=3.0)
    move_to(robot, rest, seconds=5.0)


def torque_on_here(robot):
    """Re-enable torque WITHOUT a jump: first make the current pose the goal."""
    send(robot, read_joints(robot))
    robot.bus.enable_torque()


def ready_pose(env):
    """The simulator's central ready pose (jaws 8.5 cm high, 10 cm in front of the pan axis), LeRobot degrees."""
    target = env.pan_axis + env.fwd * 0.10 + np.array([0, 0, 0.085])
    q5, ok, _, _ = env.kin.solve(target, -np.array([0, 0, 1.0]), 0.0, seeds=[env.q_home[:5], np.zeros(5)])
    return np.r_[env.to_lerobot(np.r_[q5, env.kin.gripper_closed])[:5], 25.0]


# ----------------------------------------------------------------------------------------- dataset
def write_parquet(out):
    from datasets import Dataset, Features, Image as HFImage, Sequence, Value
    rows = []
    for f in sorted(glob.glob(os.path.join(out, "samples", "*.json")), key=lambda p: int(os.path.basename(p)[:-5])):
        m = json.load(open(f))
        stem = f[:-5]
        rows.append({"seed": m["index"], "split": "real", "camera_scale": 0.0, "target": m["target"],
                     "prompt": m["prompt"], "answer": common.answer_json(m["joints_deg"][:4]),
                     "joints_deg": m["joints_deg"][:4], "target_pos": m["ref_point"], "target_size": 0.0,
                     "approach_tilt_deg": float("nan"), "ik_error_m": 0.0, "n_objects": m.get("n_objects", 0),
                     "camera": json.dumps({"real": True}),
                     "left": {"bytes": open(stem + "_left.png", "rb").read(), "path": None},
                     "right": {"bytes": open(stem + "_right.png", "rb").read(), "path": None}})
    if not rows:
        print("no samples recorded")
        return
    feats = Features({"seed": Value("int64"), "split": Value("string"), "camera_scale": Value("float32"),
                      "target": Value("string"), "prompt": Value("string"), "answer": Value("string"),
                      "joints_deg": Sequence(Value("float32")), "target_pos": Sequence(Value("float32")),
                      "target_size": Value("float32"), "approach_tilt_deg": Value("float32"),
                      "ik_error_m": Value("float32"), "n_objects": Value("int32"), "camera": Value("string"),
                      "left": HFImage(), "right": HFImage()})
    os.makedirs(os.path.join(out, "real"), exist_ok=True)
    path = os.path.join(out, "real", "real-00000.parquet")
    Dataset.from_list(rows, features=feats).to_parquet(path)
    print(f"wrote {len(rows)} samples to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="my_follower")
    ap.add_argument("--calib", default="stereo_calib_wide.json")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="data_real")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    ap.add_argument("--max-step", type=float, default=20.0)
    ap.add_argument("--max-roll", type=float, default=8.0, help="warn if the recorded wrist roll exceeds this (deg)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    env = common.make_env(0.0, urdf=args.urdf)
    env.reset(seed=0, render=False)
    q_ready = ready_pose(env)
    cam = RectifiedStereo(args.calib, args.device)
    robot = make_robot(args.port, args.id, args.max_step)
    rest = read_joints(robot)
    input("The arm should be folded in its REST pose now (it returns there between samples, torque on). "
          "Press Enter to continue...")
    existing = [int(os.path.basename(p)[:-5]) for p in glob.glob(os.path.join(args.out, "samples", "*.json"))]
    n = max(existing) + 1 if existing else 0
    print(f"rest pose remembered: {np.round(rest, 1)} | ready pose: {np.round(q_ready, 1)} | next sample #{n}")
    try:
        while True:
            descr = input("\nTarget description (e.g. 'blue ball'), 'u' = undo last, 'q' = finish: ").strip().lower()
            if descr == "q":
                break
            if descr == "u":
                if n > 0:
                    for f in glob.glob(os.path.join(args.out, "samples", f"{n - 1}*")):
                        os.remove(f)
                    n -= 1
                    print(f"removed sample #{n}")
                continue
            if not descr:
                continue
            n_obj = input("  number of objects on the board, target included [2]: ").strip()
            n_obj = int(n_obj) if n_obj.isdigit() else 2
            move_to(robot, q_ready, seconds=3.0)
            left, right = cam.read()
            preview = cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_RGB2BGR)
            cv2.imshow("captured (left | right) - check that the target is visible", preview)
            cv2.waitKey(500)
            input("  SUPPORT THE ARM with your hand (the torque will be released), then press Enter...")
            robot.bus.disable_torque()
            input("  torque OFF: guide the jaws around the CENTRE of the target (wrist roll ~0, from above), "
                  "close them gently on it, hold still, press Enter...")
            q = np.mean([read_joints(robot) for _ in range(5)], axis=0)
            torque_on_here(robot)
            if abs(q[4]) > args.max_roll:
                print(f"  wrist roll is {q[4]:+.1f} deg (> {args.max_roll}): the reference point is still exact, "
                      "but please keep the roll near 0 where possible")
            ok = input(f"  recorded {np.round(q[:5], 1)}  keep this sample? [Y/n]: ").strip().lower()
            if ok == "n":
                park(robot, q_ready, rest)
                continue
            qr = env.from_lerobot(np.r_[q[:5], 50.0])
            ref = env.kin.points(qr[:5], common.grasp_local(env))[0]       # recorded roll included
            stem = os.path.join(args.out, "samples", str(n))
            cv2.imwrite(stem + "_left.png", cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(stem + "_right.png", cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
            json.dump({"index": n, "target": descr, "prompt": common.PROMPT.format(descr=descr),
                       "joints_deg": q.tolist(), "ref_point": ref.tolist(), "n_objects": n_obj,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(stem + ".json", "w"), indent=1)
            print(f"  saved sample #{n}")
            n += 1
            park(robot, q_ready, rest)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            torque_on_here(robot)
            print("returning slowly to the rest pose...")
            park(robot, q_ready, rest)
        except Exception as e:
            print("could not park the arm:", e)
        robot.disconnect()
        cv2.destroyAllWindows()
        env.close()
    write_parquet(args.out)


if __name__ == "__main__":
    main()
