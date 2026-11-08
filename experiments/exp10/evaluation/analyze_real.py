"""
Break a real evaluation down by arrangement and number of objects.

    python evaluation/analyze_real.py --eval outputs/eval_v2_real.json --data data_real_v2

Per group: number of samples, median grasp-point error, share within 2 / 3 cm, gross errors (> --gross cm,
usually the wrong object), and the mean shoulder_pan offset of the other samples (a constant sideways
bias points to camera geometry rather than to image appearance). Arrangement labels come from
<data>/arrangements.json (real/label_arrangements.py); without it, only the object count is used.
"""
import argparse
import json
import os

import numpy as np


def summarize(name, rows, gross):
    if not rows:
        return
    d = np.array([r["dist_cm"] for r in rows])
    ok = d <= gross
    pan = np.array([r["pred"][0] - r["ref"][0] for r in rows])[ok]
    print(f"{name:22s} n={len(rows):3d}  median {np.median(d):4.1f} cm  within 2/3 cm {np.mean(d < 2):4.0%} / "
          f"{np.mean(d < 3):4.0%}  gross (>{gross:g} cm) {int((~ok).sum()):2d}  "
          f"median without gross {np.median(d[ok]) if ok.any() else float('nan'):4.1f} cm  "
          f"pan offset {pan.mean() if len(pan) else float('nan'):+5.1f} +- {pan.std() if len(pan) else float('nan'):3.1f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--data", default="data_real_v2")
    ap.add_argument("--split", default="real")
    ap.add_argument("--gross", type=float, default=6.0, help="errors above this (cm) count as gross")
    args = ap.parse_args()
    rows = [r for r in json.load(open(args.eval))["splits"][args.split]["samples"] if "pred" in r]
    meta = {}
    for r in rows:
        f = os.path.join(args.data, "samples", f"{r['seed']}.json")
        meta[r["seed"]] = json.load(open(f)) if os.path.exists(f) else {}
    lab_file = os.path.join(args.data, "arrangements.json")
    labels = json.load(open(lab_file)) if os.path.exists(lab_file) else {}
    summarize("all", rows, args.gross)
    print("by number of objects:")
    for n in sorted({meta[r["seed"]].get("n_objects", 0) for r in rows}):
        summarize(f"  {n} object(s)", [r for r in rows if meta[r["seed"]].get("n_objects", 0) == n], args.gross)
    if labels:
        print("by arrangement:")
        for lab in ("single", "pair_diff_color", "pair_same_color", "triple_diff_color", "triple_same_color", "other"):
            summarize(f"  {lab}", [r for r in rows if labels.get(str(r["seed"])) == lab], args.gross)
    else:
        print(f"(no {lab_file}: run real/label_arrangements.py for the breakdown by arrangement)")
    gross = sorted((r for r in rows if r["dist_cm"] > args.gross), key=lambda r: -r["dist_cm"])
    if gross:
        print("gross errors:")
        for r in gross:
            print(f"  #{r['seed']:<4d} {r['target']:16s} {r['dist_cm']:5.1f} cm  pan pred {r['pred'][0]:+6.1f} "
                  f"recorded {r['ref'][0]:+6.1f}  [{labels.get(str(r['seed']), '?')}]")


if __name__ == "__main__":
    main()
