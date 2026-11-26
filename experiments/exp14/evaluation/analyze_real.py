"""
Break a REAL evaluation down (no simulator, no model needed).

    python evaluation/analyze_real.py --eval outputs/eval_v11_real.json

Reports, for all samples and per task type / target shape / number of objects: the median grasp-point
distance, the share within 2 / 3 cm, and the gross errors (> --gross cm, usually the wrong object). For the
other samples: the miss split into reach (radial), pan (tangential) and height, with the signed means - a
constant offset there points to geometry (camera pose, joint offsets) rather than to image appearance - and
the mean error of every joint angle, which is what a correction of the offsets would use.
"""
import argparse
import json

import numpy as np


def line(name, rows, gross):
    d = np.array([r["dist_cm"] for r in rows])
    ok = d <= gross
    print(f"  {name:26s} n={len(rows):3d}  median {np.median(d):4.1f} cm  within 2/3 cm {np.mean(d < 2):4.0%} / "
          f"{np.mean(d < 3):4.0%}  gross {int((~ok).sum()):2d}"
          + (f"  median without gross {np.median(d[ok]):4.1f} cm" if ok.any() and (~ok).any() else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--split", default="real")
    ap.add_argument("--gross", type=float, default=5.0, help="errors above this (cm) count as gross")
    args = ap.parse_args()
    rep = json.load(open(args.eval))
    allrows = rep["splits"][args.split]["samples"]
    rows = [r for r in allrows if "pred" in r]
    print(f"{len(allrows)} samples, {len(allrows) - len(rows)} without a valid answer")
    line("all", rows, args.gross)
    for key, title in (("task_type", "by task type"), ("target_shape", "by target shape"),
                       ("n_objects", "by number of objects")):
        print(title + ":")
        for v in sorted({r.get(key) for r in rows}, key=str):
            line(str(v), [r for r in rows if r.get(key) == v], args.gross)

    good = [r for r in rows if r["dist_cm"] <= args.gross]
    if good and "pred_point" in good[0] and "pan_axis" in rep:
        pan = np.array(rep["pan_axis"])[:2]
        comp = []
        for r in good:
            t, p = np.array(r["target_pos"]), np.array(r["pred_point"])
            rad = (t[:2] - pan) / (np.linalg.norm(t[:2] - pan) + 1e-9)
            d = p - t
            comp.append([100 * d[:2] @ rad, 100 * d[:2] @ np.array([-rad[1], rad[0]]), 100 * d[2]])
        comp = np.array(comp)
        print(f"miss by direction, without gross errors (cm; n={len(good)}):   median |.|   mean (signed) +- std")
        for k, name in enumerate(("reach (+ = too far)", "sideways (pan)", "height (+ = too high)")):
            print(f"  {name:24s}                      {np.median(np.abs(comp[:, k])):5.2f}        "
                  f"{comp[:, k].mean():+5.2f} +- {comp[:, k].std():4.2f}")
        print(f"  horizontal distance only: median {np.median(np.hypot(comp[:, 0], comp[:, 1])):.2f} cm")
    if good:
        e = np.array([np.array(r["pred"]) - np.array(r["ref"]) for r in good])
        print("joint angles, predicted - recorded (deg), mean +- std:  " +
              "  ".join(f"{n} {m:+.1f} +- {s:.1f}" for n, m, s in
                        zip(("pan", "lift", "elbow", "wrist_flex"), e.mean(0), e.std(0))))
    gross = sorted((r for r in rows if r["dist_cm"] > args.gross), key=lambda r: -r["dist_cm"])
    if gross:
        print("gross errors:")
        for r in gross:
            print(f"  #{r['seed']:<4d} [{r.get('task_type')}] \"{r.get('instruction')}\" ({r['target']})  "
                  f"{r['dist_cm']:5.1f} cm  pan predicted {r['pred'][0]:+6.1f} recorded {r['ref'][0]:+6.1f}")


if __name__ == "__main__":
    main()
