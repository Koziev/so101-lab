"""
Collect samples for the stereo-camera / arm fit: the arm holds a brightly coloured object and stops at many poses
of the taught workspace; at each stop the raw stereo frame and the joint readings are saved.

    python calib/collect_gripper_object.py --setup setups/<name> --stops 150

Steps: the arm leaves its rest pose along the taught recording to the first pose above the clearance limit
("ready"); the gripper's torque goes off; you put the object between the jaws and close them by hand; Enter;
the gripper holds that position, squeezed slightly more (--squeeze); you type a short description ("orange ball");
then the arm travels through the stops (clearance rule raised by --object-mm, since the object hangs below the
jaws), waits until the stereo image is still at each, and saves the frame and readings. At the end it returns
to the ready pose, releases the gripper for you to take the object, and goes back to rest.
Output: <setup>/gripper_object/<time>/NNN.png (raw 2560x720) + NNN.json (readings, description).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from calib.gripper_axes import jaw_points  # noqa: E402
from calib.plan import Planner  # noqa: E402
from calib.session import choose_stops, make_router, thin  # noqa: E402
from calib.urdf_fk import UrdfChain  # noqa: E402
from setup_io import Setup  # noqa: E402


class StereoCam:
    def __init__(self, idx):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def grab(self, flush=4):
        for _ in range(flush):
            self.cap.read()
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("stereo camera read failed")
        return f

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


def choose_entry(P, rest, ok_obj, route_obj, stops, rest_tol):
    """Where to enter the recording. The teach-in usually starts AND ends at the rest pose, so several recorded
    poses can match the arm's current rest pose; each gives its own "ready" pose (the nearest pose along the
    recording that is high enough for the object), and from some of them more stops can be reached than from
    others. Returns (start index, ready index, reachable stops) of the best entry."""
    diff = np.max(np.abs(P - np.asarray(rest)[:5]), axis=1)
    near = np.flatnonzero(diff <= max(rest_tol, float(diff.min()) + 1e-9))
    runs = np.split(near, np.flatnonzero(np.diff(near) > 1) + 1)             # contiguous stretches of the recording
    best = None
    for run in runs:
        start = int(run[np.argmin(diff[run])])
        after = next((i for i in range(start, len(P)) if ok_obj[i]), None)
        before = next((i for i in range(start, -1, -1) if ok_obj[i]), None)
        for ready in {i for i in (after, before) if i is not None}:
            reach = sum(route_obj(P[ready].copy(), s_) is not None for s_ in stops)
            key = (reach, -abs(ready - start), -diff[start])
            if best is None or key > best[0]:
                best = (key, start, ready, reach)
    if best is None:
        raise SystemExit("no recorded pose is high enough for the object clearance")
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--urdf", default=os.path.join(ROOT, "assets_so101", "so101_sim.urdf"))
    ap.add_argument("--stops", type=int, default=150)
    ap.add_argument("--clearance-mm", type=float, default=20.0)
    ap.add_argument("--object-mm", type=float, default=35.0,
                    help="extra clearance for the object hanging below the jaws")
    ap.add_argument("--squeeze", type=float, default=3.0, help="gripper held this many %% more closed than set by hand")
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--rest-tol", type=float, default=12.0,
                    help="recorded poses within this many degrees of the arm's rest pose count as rest poses")
    args = ap.parse_args()

    from calib.lerobot_helpers import make_robot, move_to, read_joints, send
    setup = Setup(args.setup)
    ws = setup.load("workspace.json")
    if ws is None:
        raise SystemExit("the setup has no workspace.json (teach the workspace first)")
    P = np.asarray(ws["points"], float)[:, :5]
    d = np.asarray(setup.offsets()[:5])
    fk = UrdfChain(args.urdf, "gripper_link")
    planner = Planner(fk, np.eye(3), np.zeros(5))
    pts, src = jaw_points(args.urdf)
    if "toy" in src:
        raise SystemExit("the gripper geometry could not be read from the simulator (needed for the clearance rule)")
    planner.set_jaw_rule(pts, [q + d for q in P], args.clearance_mm / 1000)
    base_limit = planner.min_jaw_z
    jaw_ok_empty = lambda q: planner.jaw_height(np.asarray(q)[:5] + d) >= base_limit
    jaw_ok_obj = lambda q: planner.jaw_height(np.asarray(q)[:5] + d) >= base_limit + args.object_mm / 1000
    route_obj, ok_obj = make_router(P, jaw_ok_obj)
    stops = choose_stops(P, args.stops, lambda i: bool(ok_obj[i]))
    print(f"{len(stops)} stops among {int(ok_obj.sum())} taught poses with the jaws >= {args.clearance_mm + args.object_mm:.0f} mm "
          f"above the lowest taught jaw point")

    out = setup.file(os.path.join("gripper_object", time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)
    cam = StereoCam(setup.device("stereo"))
    robot = make_robot(args.port, setup.info["arm_id"], 20.0)

    def go(waypoints, grip):
        for w in waypoints:
            cur = read_joints(robot)
            dist = float(np.max(np.abs(np.asarray(w)[:5] - cur[:5])))
            move_to(robot, np.r_[np.asarray(w)[:5], grip], seconds=max(0.4, dist / 25.0), settle=False)

    rest = read_joints(robot)
    start, ready, reach = choose_entry(P, rest, ok_obj, route_obj, stops, args.rest_tol)
    print(f"rest = recorded pose {start} (differs from the arm's current pose by up to "
          f"{np.max(np.abs(P[start] - rest[:5])):.1f} deg); ready = recorded pose {ready} "
          f"({abs(ready - start)} recorded poses away); {reach} of {len(stops)} stops can be reached from there")
    if reach < 0.6 * len(stops):
        print("WARNING: most stops cannot be reached from this ready pose with the object clearance. The taught "
              "recording passes too low between parts of the workspace; --object-mm (now "
              f"{args.object_mm:g}) can be lowered if the object hangs less than that below the jaws.")
    n = 0
    try:
        input("Arm folded at rest, workspace clear. Enter: the arm moves to the ready pose...")
        grip = float(rest[5])
        move_to(robot, np.r_[P[start], grip], seconds=3.0, settle=False)      # a few degrees at most (--rest-tol)
        go(thin(P, start, ready), grip)
        robot.bus.disable_torque("gripper")
        input("Gripper released. Place a brightly coloured object between the jaws, close them by hand to hold it, "
              "then press Enter...")
        g = float(read_joints(robot)[5])
        grip = max(0.0, g - args.squeeze)
        send(robot, np.r_[read_joints(robot)[:5], g])
        robot.bus.enable_torque("gripper")
        send(robot, np.r_[read_joints(robot)[:5], grip])
        descr = input("Short description of the object (e.g. 'orange ball'): ").strip() or "object"
        json.dump({"description": descr, "gripper_hold_pct": grip, "setup": setup.path, "stops": len(stops)},
                  open(os.path.join(out, "session.json"), "w"), indent=1)
        cur = P[ready].copy()
        for k, s_ in enumerate(stops):
            way = route_obj(cur, s_)
            if way is None:
                print(f"  stop {k + 1}/{len(stops)}: skipped (no route above the clearance limit)")
                continue
            go(way, grip)
            cur = P[s_].copy()
            cam.wait_still(args.settle_s)
            q = np.mean([read_joints(robot) for _ in range(5)], axis=0)
            frame = cam.grab()
            cv2.imwrite(os.path.join(out, f"{n:03d}.png"), frame)
            json.dump({"q_real": q.tolist(), "pose_index": int(s_), "description": descr},
                      open(os.path.join(out, f"{n:03d}.json"), "w"))
            n += 1
            print(f"  stop {k + 1}/{len(stops)}: saved {n:03d}")
        back = route_obj(cur, ready)
        if back is not None:
            go(back, grip)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            robot.bus.disable_torque("gripper")
            input("Take the object out of the gripper, then press Enter (the arm returns to rest)...")
            robot.bus.enable_torque("gripper")
            cur_q = read_joints(robot)
            send(robot, cur_q)
            way = make_router(P, jaw_ok_empty)[0](cur_q[:5], ready) or []
            go(way, float(cur_q[5]))
            go(thin(P, ready, start), float(cur_q[5]))
            move_to(robot, rest, seconds=4.0)
        except Exception as e:
            print("could not return the arm to rest:", e)
        robot.disconnect()
    print(f"{n} samples in {out}")


if __name__ == "__main__":
    main()
