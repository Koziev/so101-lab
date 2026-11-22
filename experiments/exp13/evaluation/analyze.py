"""
Error analysis of an evaluation run: joins outputs/eval_*.json with the test splits' scene descriptions (no
model, no GPU, no images; a minute).

    python evaluation/analyze.py --eval outputs/eval_v11_sim.json --data data_v11

Per split it reports
  1. the direction of the misses on the right object: radial (reach), tangential (pan) and vertical, with the
     signed bias - "too low" matters on the real arm, where the table is in the way;
  2. the misses by scene property: lying / upright target, number of objects, distance to the nearest
     neighbour, distance of the target from the pan axis;
  3. superlatives: share of right choices against the true size ratio and against the APPARENT size ratio in
     the image (true linear ratio x camera-distance ratio; below 1 the perspective contradicts the answer);
  4. every wrong-object case: what was asked, what was picked, sizes, same shape or not.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

META_COLUMNS = ["seed", "task_type", "instruction", "canonical", "target_pos", "n_objects", "objects", "camera"]


# ------------------------------------------------------------------------------------------------ inputs
def load_meta(data_dir, split):
    """seed -> scene description of a split (the image columns are not read)."""
    import pyarrow.parquet as pq
    meta = {}
    for f in sorted(glob.glob(os.path.join(data_dir, split, "*.parquet"))):
        for r in pq.read_table(f, columns=META_COLUMNS).to_pylist():
            r["objects"] = json.loads(r["objects"])
            r["camera"] = json.loads(r["camera"])
            meta[int(r["seed"])] = r
    return meta


class Geometry:
    """The robot's kinematics and frame, from the simulator (as evaluate.py)."""

    def __init__(self, urdf, cell=None):
        self.env = common.make_env(0.0, urdf=urdf, cell=cell, real_background=False)
        self.env.reset(seed=0, render=False)
        self.pan_axis = np.asarray(self.env.pan_axis, float)
        self.fwd, self.left = np.asarray(self.env.fwd, float), np.asarray(self.env.left, float)

    def grasp_point(self, deg4):
        return np.asarray(common.grasp_point(self.env, deg4), float)

    def close(self):
        self.env.close()


def camera_position(geo, cam):
    """Stereo centre in the world, from the v9 camera parameters stored with every sample."""
    az = math.radians(cam["azimuth_deg"])
    return (geo.pan_axis + cam["distance_m"] * (math.cos(az) * geo.fwd + math.sin(az) * geo.left)
            + np.array([0, 0, cam["height_m"]]))


# ------------------------------------------------------------------------------------------------ per sample
def describe(sample, meta, geo):
    """One evaluated sample + its scene -> a flat record, or None (no prediction / no scene)."""
    if "pred" not in sample or meta is None:
        return None
    objs = meta["objects"]
    tgt = np.asarray(meta["target_pos"], float)
    tcp = geo.grasp_point(sample["pred"])
    d = tcp - tgt
    rad = tgt[:2] - geo.pan_axis[:2]
    radius = float(np.linalg.norm(rad))
    rad = rad / (radius + 1e-9)
    tan = np.array([-rad[1], rad[0]])
    pts = np.array([o["point"] for o in objs], float)
    picked = int(np.argmin(np.linalg.norm(pts[:, :2] - tcp[:2], axis=1)))
    others = np.linalg.norm(pts[1:, :2] - pts[0, :2], axis=1) if len(objs) > 1 else np.array([np.nan])
    rec = {"seed": int(sample["seed"]), "task_type": meta["task_type"], "instruction": meta["instruction"],
           "dist_cm": float(100 * np.linalg.norm(d)), "radial_cm": float(100 * d[:2] @ rad),
           "tangential_cm": float(100 * d[:2] @ tan), "vertical_cm": float(100 * d[2]),
           "radius_cm": 100 * radius, "lying": bool(objs[0]["lying"]), "shape": objs[0]["shape"],
           "n_objects": len(objs), "neighbour_cm": float(100 * np.min(others)),
           "picked": picked, "wrong": picked != 0, "target": objs[0], "picked_object": objs[picked]}
    if meta["task_type"] == "superlative":
        rec.update(superlative_cues(meta, geo))
    return rec


def superlative_cues(meta, geo):
    """How visible the right answer is: the true linear size ratio to the closest competitor of the same shape,
    and the same ratio as it appears in the image (objects nearer to the camera look larger)."""
    objs = meta["objects"]
    kind = "smallest" if any(w in meta["canonical"] for w in ("smallest", "tiniest")) else "biggest"
    cam = camera_position(geo, meta["camera"])
    rng = lambda o: float(np.linalg.norm(np.asarray(o["point"], float) - cam))
    t = objs[0]
    true, apparent = [], []
    for o in objs[1:]:
        if o["shape"] != t["shape"]:
            continue
        lin = (t["volume_cm3"] / o["volume_cm3"]) ** (1 / 3)          # > 1 if the target is the bigger one
        app = lin * rng(o) / rng(t)                                    # the same ratio in the image
        if kind == "smallest":
            lin, app = 1 / lin, 1 / app
        true.append(lin)
        apparent.append(app)
    return {"kind": kind, "true_ratio": float(min(true)), "apparent_ratio": float(min(apparent))}


# ------------------------------------------------------------------------------------------------ report
def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def table(title, recs, key, bins=None, labels=None):
    """Median miss and share within 2 cm by a property (right-object samples)."""
    print(f"  {title}")
    out = {}
    if bins is None:
        groups = {}
        for r in recs:
            groups.setdefault(str(r[key]), []).append(r)
    else:
        groups = {lab: [r for r in recs if lo <= r[key] < hi] for lab, (lo, hi) in zip(labels, bins)}
    for g, rs in sorted(groups.items()) if bins is None else groups.items():
        if not rs:
            continue
        d = np.array([r["dist_cm"] for r in rs])
        out[g] = {"n": len(rs), "median_cm": round(med(d), 2), "within_2cm": round(float(np.mean(d < 2)), 3)}
        print(f"    {g:22s} n={len(rs):4d}  median {med(d):5.2f} cm  within 2 cm {100 * np.mean(d < 2):3.0f} %")
    return out


def terciles(recs, key, unit):
    v = np.array([r[key] for r in recs if np.isfinite(r[key])])
    if len(v) < 6:
        return None, None
    a, b = np.percentile(v, [33.3, 66.7])
    return [(-1e9, a), (a, b), (b, 1e9)], [f"< {a:.1f} {unit}", f"{a:.1f}-{b:.1f} {unit}", f"> {b:.1f} {unit}"]


def report_split(split, recs):
    out = {"n": len(recs)}
    right = [r for r in recs if not r["wrong"]]
    wrong = [r for r in recs if r["wrong"]]
    print(f"\n===== {split}: {len(recs)} samples, {len(wrong)} wrong object")

    # 1. direction of the misses
    comp = {}
    print("  miss on the right object, by direction (cm):   median |.|    mean (signed)    90 % below")
    for k, name in (("radial_cm", "radial (reach, + = too far)"), ("tangential_cm", "tangential (pan)"),
                    ("vertical_cm", "vertical (+ = too high)")):
        v = np.array([r[k] for r in right])
        comp[k] = {"median_abs": round(med(np.abs(v)), 2), "mean": round(float(v.mean()), 2),
                   "p90_abs": round(float(np.percentile(np.abs(v), 90)), 2)}
        print(f"    {name:30s}             {med(np.abs(v)):5.2f}         {v.mean():+5.2f}          "
              f"{np.percentile(np.abs(v), 90):5.2f}")
    v = np.array([r["vertical_cm"] for r in right])
    out["too_low_05cm"] = round(float(np.mean(v < -0.5)), 3)
    out["too_low_10cm"] = round(float(np.mean(v < -1.0)), 3)
    print(f"    grasp point lower than the label by more than 0.5 cm: {100 * out['too_low_05cm']:.0f} %, "
          f"by more than 1 cm: {100 * out['too_low_10cm']:.0f} %")
    out["components"] = comp

    # 2. scene properties
    out["by_lying"] = table("target lying / upright", [dict(r, lying="lying" if r["lying"] else "upright")
                                                       for r in right], "lying")
    out["by_shape"] = table("target shape", right, "shape")
    out["by_n_objects"] = table("objects in the scene", right, "n_objects")
    for key, unit, title in (("neighbour_cm", "cm", "distance to the nearest other object"),
                             ("radius_cm", "cm", "target's distance from the pan axis")):
        bins, labels = terciles(right, key, unit)
        if bins:
            out["by_" + key] = table(title, right, key, bins, labels)

    # 3. superlatives
    sup = [r for r in recs if r["task_type"] == "superlative"]
    if sup:
        print(f"  superlatives: {len(sup)} tasks, {sum(not r['wrong'] for r in sup)} right "
              f"({100 * np.mean([not r['wrong'] for r in sup]):.0f} %)")
        out["superlative"] = {}
        for key, title, bins, labels in (
                ("true_ratio", "true linear size ratio to the closest competitor",
                 [(0, 1.15), (1.15, 1.25), (1.25, 1.4), (1.4, 99)], ["< 1.15", "1.15-1.25", "1.25-1.40", "> 1.40"]),
                ("apparent_ratio", "the same ratio as seen in the image (perspective included)",
                 [(0, 1.0), (1.0, 1.1), (1.1, 1.25), (1.25, 99)],
                 ["< 1.00 (contradicts)", "1.00-1.10", "1.10-1.25", "> 1.25"])):
            print(f"    {title}")
            o = out["superlative"][key] = {}
            for lab, (lo, hi) in zip(labels, bins):
                rs = [r for r in sup if lo <= r[key] < hi]
                if rs:
                    ok = sum(not r["wrong"] for r in rs)
                    o[lab] = {"n": len(rs), "right": ok}
                    print(f"      {lab:22s} {ok:3d} of {len(rs):3d} right ({100 * ok / len(rs):3.0f} %)")
        for kind in ("biggest", "smallest"):
            rs = [r for r in sup if r["kind"] == kind]
            if rs:
                ok = sum(not r["wrong"] for r in rs)
                out["superlative"][kind] = {"n": len(rs), "right": ok}
                print(f"    {kind:9s} {ok:3d} of {len(rs):3d} right")

    # 4. the wrong-object cases
    if wrong:
        print("  wrong-object cases:")
    out["wrong_cases"] = []
    for r in wrong:
        t, p = r["target"], r["picked_object"]
        same = p["shape"] == t["shape"]
        extra = (f"  true ratio {r['true_ratio']:.2f}, in the image {r['apparent_ratio']:.2f}"
                 if r["task_type"] == "superlative" else "")
        print(f"    seed {r['seed']}  [{r['task_type']}] \"{r['instruction']}\"\n"
              f"        target {t['label']} ({t['volume_cm3']:.1f} cm3{', lying' if t['lying'] else ''}) -> picked "
              f"{p['label']} ({p['volume_cm3']:.1f} cm3{', lying' if p['lying'] else ''}), "
              f"{'same shape' if same else 'OTHER shape'}, {'same colour' if p['colour'] == t['colour'] else 'other colour'}"
              f"{extra}")
        out["wrong_cases"].append({"seed": r["seed"], "task_type": r["task_type"], "instruction": r["instruction"],
                                   "target": t["label"], "picked": p["label"], "same_shape": same,
                                   "same_colour": p["colour"] == t["colour"],
                                   "true_ratio": r.get("true_ratio"), "apparent_ratio": r.get("apparent_ratio")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="the JSON written by evaluate.py")
    ap.add_argument("--data", default="data_v11")
    ap.add_argument("--out", default=None, help="default: <eval>_analysis.json")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()

    report = json.load(open(args.eval))
    geo = Geometry(args.urdf, common.load_cell(args.data))
    result = {}
    for split, content in report["splits"].items():
        meta = load_meta(args.data, split)
        recs = [describe(s, meta.get(int(s["seed"])), geo) for s in content["samples"]]
        missing = sum(r is None for r in recs)
        if missing:
            print(f"{split}: {missing} samples without a prediction or without a scene in {args.data}")
        result[split] = report_split(split, [r for r in recs if r is not None])
    geo.close()
    out = args.out or os.path.splitext(args.eval)[0] + "_analysis.json"
    json.dump(result, open(out, "w"), indent=1)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
