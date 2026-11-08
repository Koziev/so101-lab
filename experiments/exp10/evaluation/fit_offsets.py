"""
Are the real errors a constant joint-angle offset (calibration zero of the real arm or camera yaw), or
something else? Fit 4 constant offsets (deg) on half of the real samples and test them on the other half.

    python evaluation/fit_offsets.py --eval outputs/eval_v2_real.json --data data_real_v2

The offsets are added to the model's predicted angles; the grasp point of the corrected angles (roll 0) is
compared with the recorded reference point, exactly as in evaluate.py. Samples with gross errors
(> --gross cm before correction) and targets outside the trained sector (|recorded pan| > --max-pan)
are left out of the fit. Fit on even-numbered samples, test on odd-numbered ones, and the other way round.
A large improvement on the TEST half means a systematic offset (fixable by calibration); little
improvement means the error is mostly the gap between simulated and real images.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402


def distances(env, preds, refs, offset):
    return np.array([100 * np.linalg.norm(common.grasp_point(env, p + offset) - r) for p, r in zip(preds, refs)])


def fit(env, preds, refs):
    cost = lambda o: float(np.median(distances(env, preds, refs, np.asarray(o))) + 0.2 * np.mean(
        distances(env, preds, refs, np.asarray(o))))
    try:
        from scipy.optimize import minimize
        return minimize(cost, np.zeros(4), method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-3,
                                                                          "maxiter": 800}).x
    except ImportError:                                        # simple coordinate search without scipy
        o, step = np.zeros(4), 4.0
        while step > 0.05:
            improved = False
            for j in range(4):
                for s in (step, -step):
                    t = o.copy()
                    t[j] += s
                    if cost(t) < cost(o):
                        o, improved = t, True
            if not improved:
                step /= 2
        return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--data", default="data_real_v2")
    ap.add_argument("--split", default="real")
    ap.add_argument("--gross", type=float, default=6.0)
    ap.add_argument("--max-pan", type=float, default=32.0)
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    ap.add_argument("--save", default="", help="also fit on ALL used samples and save the offsets to this JSON file")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.eval))["splits"][args.split]["samples"] if "pred" in r]
    refs = {}
    for f in glob.glob(os.path.join(args.data, "samples", "*.json")):
        m = json.load(open(f))
        refs[m["index"]] = np.array(m["ref_point"])
    rows = [r for r in rows if r["seed"] in refs and r["dist_cm"] <= args.gross and abs(r["ref"][0]) <= args.max_pan]
    env = common.make_env(0.0, urdf=args.urdf)
    env.reset(seed=0, render=False)
    print(f"{len(rows)} samples used (gross errors and targets outside the sector left out)")
    for name, fit_par in (("fit on even, test on odd", 0), ("fit on odd, test on even", 1)):
        fit_rows = [r for r in rows if r["seed"] % 2 == fit_par]
        test_rows = [r for r in rows if r["seed"] % 2 != fit_par]
        P = lambda rs: [np.array(r["pred"]) for r in rs]
        Rf = lambda rs: [refs[r["seed"]] for r in rs]
        o = fit(env, P(fit_rows), Rf(fit_rows))
        before = distances(env, P(test_rows), Rf(test_rows), np.zeros(4))
        after = distances(env, P(test_rows), Rf(test_rows), o)
        print(f"{name}: offsets (deg) " + " ".join(f"{k} {v:+.1f}" for k, v in zip(common.JOINT_KEYS, o)))
        print(f"   test half (n={len(test_rows)}): median {np.median(before):.1f} -> {np.median(after):.1f} cm, "
              f"within 2 cm {np.mean(before < 2):.0%} -> {np.mean(after < 2):.0%}, "
              f"within 3 cm {np.mean(before < 3):.0%} -> {np.mean(after < 3):.0%}")
    if args.save:
        P = [np.array(r["pred"]) for r in rows]
        o = fit(env, P, [refs[r["seed"]] for r in rows])
        d0 = distances(env, P, [refs[r["seed"]] for r in rows], np.zeros(4))
        d1 = distances(env, P, [refs[r["seed"]] for r in rows], o)
        # sign convention: these are ADDED to the model's (simulator) angles to match the recorded real angles,
        # i.e. real = simulator + offset  ->  for apply_offsets.py (simulator = real + offset) use the NEGATIVE
        json.dump({"model_to_recorded_deg": dict(zip(common.JOINT_KEYS, o.round(2).tolist())),
                   "for_apply_offsets": {k: round(-float(v), 2) for k, v in zip(common.JOINT_KEYS, o)},
                   "fitted_on": len(rows), "median_cm_before": round(float(np.median(d0)), 2),
                   "median_cm_after_in_sample": round(float(np.median(d1)), 2)}, open(args.save, "w"), indent=1)
        print(f"all {len(rows)} samples: offsets " + " ".join(f"{k} {v:+.1f}" for k, v in zip(common.JOINT_KEYS, o))
              + f"  (in-sample median {np.median(d0):.1f} -> {np.median(d1):.1f} cm; optimistic) -> saved {args.save}")
    env.close()


if __name__ == "__main__":
    main()
