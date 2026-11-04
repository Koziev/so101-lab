"""
Find damaged raw episodes (e.g. after "No space left on device") and prepare the shards
for resuming.

    python repair_raw.py --raw data/raw_v4            # report only
    python repair_raw.py --raw data/raw_v4 --delete   # delete damaged episodes

An episode is complete when meta.json and data.npz load and every camera folder holds
num_steps non-empty frames. In each shard, the first damaged episode AND all episodes
after it are deleted: the recorder resumes after the last remaining seed, so the deleted
episodes are regenerated with exactly the same seeds (same scenes and plans as before).
Afterwards re-run the original record_episodes.py command.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np


def episode_ok(ep_dir):
    try:
        with open(os.path.join(ep_dir, "meta.json")) as f:
            meta = json.load(f)
        n = len(np.load(os.path.join(ep_dir, "data.npz"))["state"])
    except Exception as e:
        return False, f"meta/data unreadable ({type(e).__name__})"
    if n != meta["num_steps"]:
        return False, f"data.npz has {n} steps, meta says {meta['num_steps']}"
    for cam in meta.get("cameras", ["front", "wrist"]):
        files = [f for f in glob.glob(os.path.join(ep_dir, cam, "*")) if os.path.getsize(f) > 0]
        if len(files) != n:
            return False, f"{cam}: {len(files)} non-empty frames, expected {n}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()
    total_bad = 0
    for shard in sorted(glob.glob(os.path.join(args.raw, "shard_*"))):
        for tmp in glob.glob(os.path.join(shard, ".tmp_*")):
            if args.delete:
                shutil.rmtree(tmp, ignore_errors=True)
        eps = sorted(glob.glob(os.path.join(shard, "episode_*")))
        first_bad = None
        for i, ep in enumerate(eps):
            ok, why = episode_ok(ep)
            if not ok:
                first_bad = i
                print(f"{ep}: {why}")
                break
        if first_bad is None:
            print(f"{shard}: {len(eps)} episodes, all complete")
            continue
        remove = eps[first_bad:]
        total_bad += len(remove)
        print(f"{shard}: {first_bad} complete episodes kept, {len(remove)} to regenerate")
        if args.delete:
            for ep in remove:
                shutil.rmtree(ep)
    if total_bad and not args.delete:
        print(f"\n{total_bad} episodes would be deleted; re-run with --delete, free disk space, "
              "then re-run the original record_episodes.py command")
    elif total_bad:
        print(f"\ndeleted {total_bad} episodes; free disk space, then re-run the original record_episodes.py command")


if __name__ == "__main__":
    main()
