"""
Correct recorded real samples for the arm's calibration zero offsets and write a corrected copy.

    python real/apply_offsets.py --data data_real_v2 --out data_real_v2_corr --pan 6.5

Simulator angle = real angle + offset. Example: the arm pointing straight along the board's centre line
(simulator pan 0) reads shoulder_pan -6.5 on the real arm -> --pan 6.5. The joint angles of every sample are
corrected, the reference point (grasp point of the recorded pose) is recomputed with the simulator's
kinematics, and the Parquet split is rebuilt. Images are copied unchanged; the original folder is not touched.
The offsets are also saved as <out>/joint_offsets.json.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
from real.record_real import write_parquet  # noqa: E402

KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    for k, short in zip(KEYS, ("pan", "lift", "elbow", "wrist-flex", "wrist-roll")):
        ap.add_argument(f"--{short}", type=float, default=0.0, help=f"offset for {k} (deg)")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()
    off = np.array([args.pan, args.lift, args.elbow, args.wrist_flex, args.wrist_roll])
    if os.path.abspath(args.out) == os.path.abspath(args.data):
        raise SystemExit("--out must differ from --data (the original samples are kept unchanged)")

    env = common.make_env(0.0, urdf=args.urdf)
    env.reset(seed=0, render=False)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    moved = []
    for f in glob.glob(os.path.join(args.data, "samples", "*.json")):
        m = json.load(open(f))
        q = np.array(m["joints_deg"], float)
        q[:5] += off
        qr = env.from_lerobot(np.r_[q[:5], 50.0])
        ref = env.kin.points(qr[:5], common.grasp_local(env))[0]
        moved.append(100 * np.linalg.norm(ref - np.array(m["ref_point"])))
        m.update({"joints_deg": q.tolist(), "ref_point": ref.tolist(), "joint_offsets_applied": off.tolist()})
        stem = os.path.join(args.out, "samples", os.path.basename(f)[:-5])
        json.dump(m, open(stem + ".json", "w"), indent=1)
        for eye in ("left", "right"):
            shutil.copy(f[:-5] + f"_{eye}.png", stem + f"_{eye}.png")
    for extra in ("arrangements.json",):
        if os.path.exists(os.path.join(args.data, extra)):
            shutil.copy(os.path.join(args.data, extra), os.path.join(args.out, extra))
    json.dump(dict(zip(KEYS, off.tolist())), open(os.path.join(args.out, "joint_offsets.json"), "w"), indent=1)
    env.close()
    print(f"{len(moved)} samples corrected; reference points moved by {np.median(moved):.1f} cm (median)")
    write_parquet(args.out)


if __name__ == "__main__":
    main()
