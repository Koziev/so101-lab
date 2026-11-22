"""
Where is the table for the arm? Touch the plywood with the tip of the FIXED jaw at a few places; the tool
prints the tip's height above the simulator's table plane, computed from the joint readings (offsets applied)
and the simulator's kinematics.

    python real/touch_table.py --setup setups/<name>

  about 0 cm everywhere        -> kinematics, offsets and table height agree with the simulator
  a constant value, e.g. +0.8  -> the real board is that much higher for the arm than in the simulator
                                  (or the lift / elbow / wrist offsets are off)
  a value that grows with the
  distance from the base       -> an angle offset (lift or elbow), not a height

The torque is released for the whole session: support the arm before you start. At each place rest the fixed
jaw's tip on the board, gripper pointing down (as for a grasp), hold still and press Enter. Touch near and far,
left and right; 6-8 places are enough. "q" + Enter finishes.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402
from setup_io import Setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()

    from calib.lerobot_helpers import make_robot, read_joints, send
    setup = Setup(args.setup)
    offsets = np.asarray(setup.offsets(), float)
    env = common.make_env(0.0, urdf=args.urdf)
    env.reset(seed=0, render=False)
    kin = env.kin
    d = kin.tip_depth
    u_f = float(np.interp(d, kin.probe_depths, kin.probe_fixed_u))
    tip_local = kin.tcp_local + kin.approach_local * d + kin.closing_local * u_f      # fixed jaw's tip
    grasp_local = common.grasp_local(env)
    robot = make_robot(args.port, setup.info["arm_id"], 20.0)
    input("SUPPORT THE ARM with your hand (the torque will be released), then press Enter...")
    robot.bus.disable_torque()
    rows = []
    try:
        while True:
            if input(f"touch #{len(rows) + 1}: fixed jaw's tip on the board, hold still, Enter ('q' = finish): "
                     ).strip().lower() == "q":
                break
            q = np.mean([read_joints(robot) for _ in range(5)], axis=0)
            q_sim = env.from_lerobot(np.r_[q[:5] + offsets[:5], 50.0])[:5]
            tip = kin.points(q_sim, tip_local)[0]
            grasp = kin.points(q_sim, grasp_local)[0]
            r = float(np.linalg.norm(tip[:2] - env.pan_axis[:2]))
            rows.append((r, tip[2], grasp[2]))
            print(f"   tip {100 * tip[2]:+.2f} cm above the simulator's table, {100 * r:.1f} cm from the pan axis "
                  f"(grasp point {100 * grasp[2]:.2f} cm)")
    finally:
        input("Bring the arm to its rest pose by hand, then press Enter (the torque comes back on there)...")
        send(robot, read_joints(robot))
        robot.bus.enable_torque()
        robot.disconnect()
        env.close()
    if len(rows) >= 2:
        r, z, g = np.array(rows).T
        print(f"\n{len(rows)} touches: tip height {100 * z.mean():+.2f} +- {100 * z.std():.2f} cm "
              f"(range {100 * z.min():+.2f} to {100 * z.max():+.2f})")
        if len(rows) >= 4 and np.ptp(r) > 0.05:
            k, b = np.polyfit(r, z, 1)
            print(f"trend with distance: {k * 10:+.2f} cm per 10 cm (height at 12 cm: {100 * (k * 0.12 + b):+.2f}, "
                  f"at 30 cm: {100 * (k * 0.30 + b):+.2f})")
        print(f"the fixed jaw's tip is {1000 * float(np.linalg.norm(tip_local - grasp_local)):.0f} mm from the "
              "grasp point")


if __name__ == "__main__":
    main()
