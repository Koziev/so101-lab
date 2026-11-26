"""
How precisely do the servos reach a commanded pose? From a replay session (autocalib.py --rig real): every stop
commanded a recorded pose (workspace.json, by its pose index) and saved the reached joint readings. Per joint:
the error (reached - commanded), split by the direction from which the joint approached its target (from the
previous stop's reading): stopping short of the target shows up as an error against the direction of motion.

    python calib/servo_stats.py setups/<name>/calibration/<time> --setup setups/<name>

Output: per joint, the error distribution for approaches from below and from above, and the "dead band" estimate
(the mean shortfall against the direction of motion), plus a JSON summary for the simulator's servo model.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from setup_io import JOINTS, Setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--setup", required=True)
    ap.add_argument("--min-move", type=float, default=1.0, help="ignore joints that moved less than this (deg)")
    ap.add_argument("--out", default="", help="write the summary JSON here (default: <session>/servo_stats.json)")
    args = ap.parse_args()
    ws = Setup(args.setup).load("workspace.json")
    P = np.asarray(ws["points"], float)
    stops = []
    for qf in sorted(glob.glob(os.path.join(args.session, "images", "*_q.json"))):
        info = json.load(open(qf))
        if "pose_index" in info:
            stops.append((P[int(info["pose_index"])][:5], np.asarray(info["q_real"], float)[:5]))
    if len(stops) < 5:
        raise SystemExit("too few stops with a pose index (sessions recorded with the current version only)")
    summary = {}
    print(f"{len(stops)} stops; error = reached - commanded (deg), by direction of approach\n")
    print(f"{'joint':14s} {'from below (moving +)':>28s} {'from above (moving -)':>28s} {'shortfall':>10s}")
    for j in range(5):
        up, down = [], []
        for (cmd0, got0), (cmd, got) in zip(stops[:-1], stops[1:]):
            move = cmd[j] - got0[j]                       # from where the joint was to where it was sent
            if abs(move) < args.min_move:
                continue
            (up if move > 0 else down).append(got[j] - cmd[j])
        up, down = np.array(up), np.array(down)
        fmt = lambda a: (f"{a.mean():+5.2f} +- {a.std():4.2f} (n={len(a):3d})" if len(a) else "      -")
        # shortfall: stopping short means error < 0 when moving +, > 0 when moving -
        short = np.r_[-up, down] if len(up) + len(down) else np.array([np.nan])
        print(f"{JOINTS[j]:14s} {fmt(up):>28s} {fmt(down):>28s} {np.nanmean(short):+8.2f}")
        summary[JOINTS[j]] = {"from_below_mean": float(up.mean()) if len(up) else None,
                              "from_below_std": float(up.std()) if len(up) else None,
                              "from_above_mean": float(down.mean()) if len(down) else None,
                              "from_above_std": float(down.std()) if len(down) else None,
                              "shortfall_mean": float(np.nanmean(short)), "shortfall_std": float(np.nanstd(short)),
                              "n": int(len(up) + len(down))}
    out = args.out or os.path.join(args.session, "servo_stats.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nshortfall > 0: the joint stops short of its target (dead band against the direction of motion).")
    print(f"summary written to {out}")


if __name__ == "__main__":
    main()
