"""
Where is the board for the arm IN OPERATION (torque on, gravity taking up the play)? The arm lowers the tip of
its fixed jaw in 1 mm steps at several places; you say when it touches. The result goes into the setup
(table.json) and defines the simulator's object plane and the height of the grasp labels.

    python real/probe_table.py --setup setups/<name>

Before: board clear, arm folded at rest, a strip of paper at hand. At every place the arm stops with the tip
about 5 cm above the simulator's table; slide the paper under the tip, then
    Enter      1 mm down                      u + Enter   1 mm up
    5 + Enter  5 mm down (only while the tip is clearly high)
    t + Enter  TOUCH: the paper is just held by the tip (go back up 1 mm with u if you overshot, then t)
    s + Enter  skip this place                q + Enter   finish
The tool never goes lower than --floor-cm (default -1.0 cm, simulator's table plane = 0); the jaws are closed.
For every touch it stores the tip height computed from the COMMANDED angles and from the READ angles (offsets
applied in both); they differ by what the servos give way under load.

Run in the LeRobot environment. Keep a hand near the power switch: this tool moves the arm close to the board.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from setup_io import Setup  # noqa: E402

TILTS_DEG = (0, 15, 30, 45)


class Prober:
    """Kinematics of the fixed jaw's tip in the simulator's convention."""

    def __init__(self, env):
        self.env, self.kin = env, env.kin
        d = self.kin.tip_depth
        u_f = float(np.interp(d, self.kin.probe_depths, self.kin.probe_fixed_u))
        self.tip_local = self.kin.tcp_local + self.kin.approach_local * d + self.kin.closing_local * u_f

    def solve(self, xy, z, seed=None, tilt=None):
        """Simulator joint angles (deg, 5) with the tip at (xy, z), as steep as reachable; (None, None) if not."""
        p = np.array([xy[0], xy[1], z])
        radial = p[:2] - self.env.pan_axis[:2]
        radial = np.r_[radial / (np.linalg.norm(radial) + 1e-9), 0.0]
        seeds = [seed] if seed is not None else [self.env.q_home[:5], np.zeros(5)]   # stepping: stay on the branch
        for t in (TILTS_DEG if tilt is None else (tilt,)):
            a = radial * math.sin(math.radians(t)) - np.array([0, 0, math.cos(math.radians(t))])
            q5, ok, _, _ = self.kin.solve(p, a / np.linalg.norm(a), 0.0, seeds, self.tip_local)
            if ok:
                return q5, t
        return None, None

    def to_deg(self, q5):
        return self.env.to_lerobot(np.r_[q5, self.kin.gripper_closed])[:5]

    def tip(self, deg5_sim):
        q = self.env.from_lerobot(np.r_[np.asarray(deg5_sim, float), 50.0])[:5]
        return self.kin.points(q, self.tip_local)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    ap.add_argument("--radii-cm", default="14,20,26", help="distances from the pan axis")
    ap.add_argument("--angles-deg", default="-30,0,30",
                    help="directions (0 = straight forward); write it as --angles-deg=-30,0,30")
    ap.add_argument("--start-cm", type=float, default=5.0, help="tip height at which the stepping starts")
    ap.add_argument("--floor-cm", type=float, default=-1.0, help="never command the tip lower than this")
    ap.add_argument("--gripper-pct", type=float, default=3.0)
    ap.add_argument("--clearance-mm", type=float, default=20.0, help="for leaving / reaching the rest pose")
    ap.add_argument("--dry-run", action="store_true", help="solve all places and print them, do not move")
    args = ap.parse_args()

    setup = Setup(args.setup)
    offsets = np.asarray(setup.offsets(), float)[:5]              # simulator = real + offset
    env = common.make_env(0.0, urdf=args.urdf)
    env.reset(seed=0, render=False)
    pr = Prober(env)
    places = []
    for r in (float(v) / 100 for v in args.radii_cm.split(",")):
        for a in (math.radians(float(v)) for v in args.angles_deg.split(",")):
            xy = env.pan_axis[:2] + r * (math.cos(a) * env.fwd[:2] + math.sin(a) * env.left[:2])
            q5, tilt = pr.solve(xy, args.start_cm / 100)
            low, _ = pr.solve(xy, args.floor_cm / 100, seed=q5, tilt=tilt) if q5 is not None else (None, None)
            ok = q5 is not None and low is not None
            print(f"place {len(places) + 1}: {100 * r:.0f} cm, {math.degrees(a):+.0f} deg: "
                  + (f"reachable, gripper tilted {tilt} deg" if ok else "NOT reachable down to the floor, skipped"))
            if ok:
                places.append({"r": r, "angle_deg": math.degrees(a), "xy": xy, "q_top": q5, "tilt": tilt})
    if args.dry_run or not places:
        env.close()
        return

    from calib.lerobot_helpers import make_robot, move_to, read_joints
    from record_real import Travel, ready_pose_sim
    q_ready = ready_pose_sim(env)
    robot = make_robot(args.port, setup.info["arm_id"], 20.0)
    input("Arm folded at rest, board clear, paper strip at hand. Enter: the arm moves to its ready pose...")
    travel = Travel(robot, setup, args.urdf, np.r_[q_ready[:5] - offsets, args.gripper_pct], args.clearance_mm)
    real = lambda q5: np.r_[pr.to_deg(q5) - offsets, args.gripper_pct]       # command for simulator angles
    touches = []
    try:
        travel.to_ready()
        for k, pl in enumerate(places):
            print(f"\nplace {k + 1}/{len(places)}: {100 * pl['r']:.0f} cm, {pl['angle_deg']:+.0f} deg")
            move_to(robot, real(pl["q_top"]), seconds=3.0)
            z, q5, cmd = args.start_cm / 100, pl["q_top"], None
            while True:
                tip_cmd = pr.tip(pr.to_deg(q5))[2]
                key = input(f"  tip {100 * tip_cmd:+.2f} cm (commanded)  [Enter = 1 mm down, 5, u, t = touch, s, q]: "
                            ).strip().lower()
                if key in ("t", "s", "q"):
                    cmd = key
                    break
                step = {"": -0.001, "5": -0.005, "u": 0.001}.get(key)
                if step is None:
                    continue
                if z + step < args.floor_cm / 100 - 1e-9:
                    print(f"  the floor ({args.floor_cm:+.1f} cm) is reached: not going lower. If the tip is still in "
                          "the air, type s and tell me; do not lower --floor-cm without a reason.")
                    continue
                nq, _ = pr.solve(pl["xy"], z + step, seed=q5, tilt=pl["tilt"])
                if nq is None:
                    print("  this height cannot be reached here")
                    continue
                if np.max(np.abs(pr.to_deg(nq) - pr.to_deg(q5))) > 6.0:
                    print("  the solution for this height is far from the current pose: not moving")
                    continue
                z, q5 = z + step, nq
                move_to(robot, real(q5), seconds=0.5 if abs(step) < 0.002 else 1.2)
            if cmd == "t":
                time.sleep(0.5)
                q_read = np.mean([read_joints(robot) for _ in range(5)], axis=0)[:5]
                tip_read = pr.tip(q_read + offsets)[2]
                touches.append({"r_m": pl["r"], "angle_deg": pl["angle_deg"], "tilt_deg": pl["tilt"],
                                "tip_z_commanded_m": float(tip_cmd), "tip_z_read_m": float(tip_read),
                                "q_read": q_read.tolist(), "q_commanded": (pr.to_deg(q5) - offsets).tolist()})
                print(f"  touch: commanded {100 * tip_cmd:+.2f} cm, read {100 * tip_read:+.2f} cm")
            move_to(robot, real(pl["q_top"]), seconds=2.0)                    # straight back up
            if cmd == "q":
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            print("returning to the rest pose...")
            travel.to_rest()
        except Exception as e:
            print("could not park the arm:", e)
        robot.disconnect()
        env.close()

    if len(touches) < 3:
        print(f"{len(touches)} touches: too few to store a board height")
        return
    r = np.array([t["r_m"] for t in touches])
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "offsets_deg": offsets.tolist(), "touches": touches,
           "note": "height of the board at the fixed jaw's tip, simulator's table plane = 0"}
    print()
    for key, name in (("tip_z_commanded_m", "commanded"), ("tip_z_read_m", "read")):
        z = np.array([t[key] for t in touches])
        k, b = np.polyfit(r, z, 1) if np.ptp(r) > 0.03 else (0.0, z.mean())
        out["board_z_" + name + "_m"] = float(z.mean())
        out["board_z_" + name + "_std_m"] = float(z.std())
        out["slope_" + name + "_per_m"] = float(k)
        print(f"board height from the {name:9s} angles: {100 * z.mean():+.2f} +- {100 * z.std():.2f} cm "
              f"({len(z)} places; {100 * (k * 0.14 + b):+.2f} cm at 14 cm, {100 * (k * 0.26 + b):+.2f} cm at 26 cm)")
    setup.save("table.json", out, note="board height probed with torque on (real/probe_table.py)")
    print("written into the setup (table.json)")


if __name__ == "__main__":
    main()
