"""
First real pointing / grasping with the exp10 model (run in the LeRobot environment).

    # terminal 1, Qwen environment:
    python evaluation/predict_server.py --model outputs/qwen35_08b_v2/final
    # terminal 2, LeRobot environment:
    python real/grasp_real.py --mode point --port /dev/ttyACM0 --id my_follower --calib ../stereo_calib_wide.json
    python real/grasp_real.py --mode grasp ...

Per trial: you place the objects and type the target's description. The arm goes from its rest pose to the
ready pose (as in the training images), the stereo pair is captured, and the model predicts the target in
joint space (simulator convention). From the predicted grasp point the tool plans, in the simulator's
kinematics, a pre-grasp pose 6 cm above it (steepest reachable approach, wrist roll 0) and then:
  point: goes down to --point-height above the predicted object centre and stops, so you can judge the aim;
  grasp: opens the jaws, descends to the predicted centre (+ --grasp-dz), closes the jaws step by step
         while the arm holds still, waits for the servos to settle, and only then lifts back to the
         pre-grasp height and checks whether the jaws are held open by the object.
Afterwards the arm puts the object back (grasp mode) and returns to its rest pose; torque stays on throughout.
All angles go through real/arm_convention.json (simulator = real + offset). After every move the arm-carrying
joints are corrected until they are within --tolerance degrees (servo dead band).
Each trial is logged in <log>/<n>.json with the images.
"""
import argparse
import base64
import glob
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
from real import arm_convention  # noqa: E402
from real.record_real import (RectifiedStereo, make_robot, move_to, read_joints, ready_pose, send,  # noqa: E402
                              torque_on_here)

CLOSED_PCT = 0.0


def ask_model(url, target, left, right, timeout=60):
    def png(rgb):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf.tobytes()).decode()
    body = json.dumps({"target": target, "left_png": png(left), "right_png": png(right)}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def move_precise(robot, target, seconds=3.0, tolerance=0.6, iters=4, max_corr=12.0):
    """Smooth move, then correct pan/lift/elbow/wrist_flex until the MEASURED angles are within tolerance."""
    target = np.asarray(target, float)
    move_to(robot, target, seconds=seconds)
    cmd = target.copy()
    meas = read_joints(robot)
    for _ in range(iters):
        err = target[:4] - meas[:4]
        if np.max(np.abs(err)) < tolerance:
            break
        cmd[:4] = target[:4] + np.clip(cmd[:4] - target[:4] + err, -max_corr, max_corr)
        send(robot, cmd)
        time.sleep(0.5)
        meas = read_joints(robot)
    return meas, cmd


def gripper_to(robot, hold_cmd, value, step=8.0, hz=15, settle=1.0, timeout=6.0):
    """Move ONLY the gripper to `value` in small steps (each within LeRobot's max_relative_target), while the
    arm joints keep their holding command. Stops early when the jaws stop moving (blocked by the object),
    then waits `settle` seconds. Returns the final gripper reading."""
    cmd = np.asarray(hold_cmd, float).copy()
    g = read_joints(robot)[5]
    t0, last, still = time.time(), g, 0
    while time.time() - t0 < timeout:
        g_cmd = value if abs(value - g) <= step else g + np.sign(value - g) * step
        cmd[5] = g_cmd
        send(robot, cmd)
        time.sleep(1.0 / hz)
        g = read_joints(robot)[5]
        if abs(g - value) < 1.0:
            break
        still = still + 1 if abs(g - last) < 0.3 else 0          # no progress: blocked by the object
        if still >= 5:
            break
        last = g
    cmd[5] = value                                                   # keep squeezing (closing) / keep open
    send(robot, cmd)
    time.sleep(settle)
    return read_joints(robot)[5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["point", "grasp"], default="point")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="my_follower")
    ap.add_argument("--calib", default="stereo_calib_wide.json")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--convention", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_convention.json"))
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--log", default="real_trials")
    ap.add_argument("--pre-height", type=float, default=0.06, help="pre-grasp height above the object centre (m)")
    ap.add_argument("--point-height", type=float, default=0.02, help="point mode: stop this high above the centre (m)")
    ap.add_argument("--grasp-dz", type=float, default=0.0, help="grasp mode: grasp point relative to the centre (m)")
    ap.add_argument("--tolerance", type=float, default=0.6)
    ap.add_argument("--clearance-size", type=float, default=0.042,
                    help="grasp mode: plan the jaws for an object of this size (m) around the predicted centre; "
                         "larger = more room between the fixed jaw and the centre (labels use 0.030)")
    ap.add_argument("--open-pct", type=float, default=90.0, help="gripper opening before the descent (%%)")
    ap.add_argument("--slow-cm", type=float, default=3.0, help="grasp mode: descend the last N cm slowly")
    ap.add_argument("--squeeze-s", type=float, default=1.5, help="pause after closing, before lifting (s)")
    ap.add_argument("--lift-s", type=float, default=4.0, help="duration of the lift (s)")
    ap.add_argument("--max-step", type=float, default=20.0)
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()

    off = arm_convention.load(args.convention)
    print("joint offsets (simulator = real + offset):", dict(zip(arm_convention.JOINTS, off)))
    with urllib.request.urlopen(args.server.rstrip("/") + "/health", timeout=5) as r:
        print("prediction server:", json.loads(r.read()))
    env = common.make_env(0.0, urdf=args.urdf)
    env.reset(seed=0, render=False)
    to_real = lambda q_sim: arm_convention.sim_to_real(q_sim, off)
    q_ready = to_real(ready_pose(env))                                   # ready pose, real convention
    cam = RectifiedStereo(args.calib, args.device)
    robot = make_robot(args.port, args.id, args.max_step)
    rest = read_joints(robot)
    input(f"rest pose remembered: {np.round(rest, 1)}. The arm must be folded in its REST pose. Press Enter...")
    os.makedirs(args.log, exist_ok=True)
    n = len(glob.glob(os.path.join(args.log, "*.json")))

    def park():
        move_to(robot, q_ready, seconds=3.0)
        move_to(robot, rest, seconds=5.0)

    def pose_above(p, h, grip, size=None):
        lab = common.label_joints(env, np.asarray(p) + np.array([0, 0, h]), size=size)
        if lab is None:
            return None
        return to_real(np.r_[lab[0], 0.0, grip])

    try:
        while True:
            target = input(f"\n[{args.mode} #{n}] target description ('blue ball'), 'q' = quit: ").strip().lower()
            if target == "q":
                break
            if not target:
                continue
            move_to(robot, q_ready, seconds=3.0)
            time.sleep(0.5)
            left, right = cam.read()
            res = ask_model(args.server, target, left, right)
            if not res.get("joints"):
                print("  no valid prediction:", res)
                park()
                continue
            pred_sim = np.array(res["joints"])
            p = common.grasp_point(env, pred_sim)                         # predicted object centre (world)
            print(f"  prediction (sim deg) {np.round(pred_sim, 1)} in {res['ms']} ms; "
                  f"centre {np.round(100 * (p - env.pan_axis), 1)} cm from the pan axis")
            grip_pre = args.open_pct if args.mode == "grasp" else 25.0
            size = args.clearance_size if args.mode == "grasp" else None
            q_pre = pose_above(p, args.pre_height, grip_pre, size)
            q_goal = pose_above(p, args.point_height if args.mode == "point" else args.grasp_dz, grip_pre, size)
            q_slow = pose_above(p, (args.grasp_dz + args.slow_cm / 100), grip_pre, size) if args.mode == "grasp" else None
            if q_pre is None or q_goal is None or (args.mode == "grasp" and q_slow is None):
                print("  target not reachable with the label convention: skipped")
                park()
                continue
            log = {"n": n, "mode": args.mode, "target": target, "pred_sim_deg": pred_sim.tolist(), "text": res["text"],
                   "settings": {k: getattr(args, k) for k in ("clearance_size", "open_pct", "slow_cm", "squeeze_s",
                                                              "lift_s", "grasp_dz", "pre_height")},
                   "centre_from_pan_axis_cm": (100 * (p - env.pan_axis)).tolist(), "q_pre_real": q_pre.tolist(),
                   "q_goal_real": q_goal.tolist(), "offsets": off.tolist()}
            if args.mode == "grasp":                                         # open the jaws at the ready pose
                gripper_to(robot, np.r_[q_ready[:5], read_joints(robot)[5]], args.open_pct, settle=0.3)
            move_precise(robot, q_pre, seconds=3.0, tolerance=args.tolerance)
            if args.mode == "grasp":                                         # last centimetres slowly
                move_precise(robot, q_slow, seconds=1.5, tolerance=args.tolerance)
                reached, hold = move_precise(robot, q_goal, seconds=2.5, tolerance=args.tolerance)
            else:
                reached, hold = move_precise(robot, q_goal, seconds=2.5, tolerance=args.tolerance)
            log["reached_goal_real"] = reached.tolist()
            if args.mode == "point":
                est = input("  pointing above the object. Horizontal miss (cm, e.g. 1.5; Enter = skip): ").strip()
                log["user_miss_cm"] = float(est) if est else None
                move_precise(robot, q_pre, seconds=2.0, tolerance=args.tolerance)
            else:
                closed = gripper_to(robot, hold, CLOSED_PCT, settle=args.squeeze_s)   # close, squeeze, THEN lift
                move_precise(robot, np.r_[q_pre[:5], CLOSED_PCT], seconds=args.lift_s, tolerance=args.tolerance)
                time.sleep(0.8)
                lifted = read_joints(robot)[5]
                held = lifted > 4.0                                          # jaws kept apart by the object
                print(f"  gripper after closing {closed:.1f} %, after lifting {lifted:.1f} % -> "
                      f"{'object probably held' if held else 'jaws closed: probably missed'}")
                ok = input("  success (object lifted)? [y/n]: ").strip().lower()
                log.update({"gripper_closed_pct": float(closed), "gripper_lifted_pct": float(lifted),
                            "auto_held": bool(held), "success": ok == "y"})
                if held:                                                     # put it back where it was
                    _, hold = move_precise(robot, np.r_[q_goal[:5], CLOSED_PCT], seconds=2.5, tolerance=args.tolerance)
                    gripper_to(robot, hold, args.open_pct, settle=0.8)
                    move_precise(robot, q_pre, seconds=2.0, tolerance=args.tolerance)
            stem = os.path.join(args.log, str(n))
            cv2.imwrite(stem + "_left.png", cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(stem + "_right.png", cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
            json.dump(log, open(stem + ".json", "w"), indent=1)
            n += 1
            park()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            torque_on_here(robot)
            print("returning slowly to the rest pose...")
            park()
        except Exception as e:
            print("could not park the arm:", e)
        robot.disconnect()
        env.close()
    trials = [json.load(open(f)) for f in glob.glob(os.path.join(args.log, "*.json"))]
    g = [t for t in trials if t.get("mode") == "grasp" and "success" in t]
    p = [t["user_miss_cm"] for t in trials if t.get("mode") == "point" and t.get("user_miss_cm") is not None]
    if g:
        print(f"grasp trials: {sum(t['success'] for t in g)}/{len(g)} successful")
    if p:
        print(f"point trials: median estimated miss {np.median(p):.1f} cm over {len(p)} trials")


if __name__ == "__main__":
    main()
