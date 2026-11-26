"""
Look at recorded approach episodes: left|right frames with the prompt, phase, commanded vs measured angles.

    python dataset/preview_raw.py --raw data_v12/raw --n 6 --out preview_v12
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="/media/inkoziev/corpora/tmp/lerobot_datasets/data_v12/raw")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="preview_v12")
    p.add_argument("--per-command", type=int, default=0, help="exp14: this many episodes of every command")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    metas = sorted(glob.glob(os.path.join(args.raw, "shard_*", "episode_*", "meta.json")))
    if args.per_command:
        by = {}
        for m in metas:
            by.setdefault(json.load(open(m)).get("command", "approach"), []).append(m)
        metas = [m for c in sorted(by) for m in by[c][:args.per_command]]
    else:
        metas = metas[:args.n]
    for m in metas:
        ep, meta = os.path.dirname(m), json.load(open(m))
        d = np.load(os.path.join(ep, "data.npz"))
        cmd = meta.get("command", "approach")
        name = f"{cmd}_{os.path.basename(os.path.dirname(ep))}_{os.path.basename(ep)}.mp4"
        vw = None
        for t in range(len(d["state"])):
            img = np.hstack([cv2.imread(os.path.join(ep, c, f"{t:06d}.jpg")) for c in ("left", "right")])
            lines = [f"[{cmd}{', ' + meta['variant'] if meta.get('variant') else ''}] {meta['instruction']}",
                     f"phase {meta['phases'][int(d['phase'][t])]}  t={t / meta['fps']:.2f}s",
                     "cmd  " + " ".join(f"{v:7.1f}" for v in d["action"][t][:6]),
                     "read " + " ".join(f"{v:7.1f}" for v in d["state"][t][:6])]
            img[:22 * len(lines) + 8] = (img[:22 * len(lines) + 8] * 0.35).astype(np.uint8)
            for k, s in enumerate(lines):
                cv2.putText(img, s, (10, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if vw is None:
                vw = cv2.VideoWriter(os.path.join(args.out, name), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"],
                                     (img.shape[1], img.shape[0]))
            vw.write(img)
        vw.release()
        if "metrics" in meta:                                   # exp14 episodes
            mt = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in meta["metrics"].items()
                  if not isinstance(v, list)}
            print(f"{name}: {meta['num_steps']} steps | {meta['instruction']} | {mt}")
            continue
        f = meta["final"]
        print(f"{name}: {meta['num_steps']} steps, tilt {meta['goal']['tilt_deg']}, final across {f['across_cm']:.2f} cm, "
              f"gap {f['gap_cm']:.2f} cm, angle {f['angle_deg']:.1f} deg | {meta['instruction']}")


if __name__ == "__main__":
    main()
