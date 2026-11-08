"""
Generate the exp10 dataset: stereo pair + target description -> joint-space target (4 angles, JSON).

    python dataset/generate.py --out data --train 3000 --test 200 --train-scale 1 --workers 6

Splits (Parquet shards, JPEG bytes embedded; nothing else is written per sample):
  train      cameras with width factor --train-scale, seeds 0 ..
  test_T0    cameras at the measured real pose (s = 0)            seeds 900000000 ..
  test_T1    cameras with s = 1                                    same seeds
  test_T2    cameras with s = 2                                    same seeds
The first --examples samples of every split are also written to examples/<split>_<n>.json with the
two images as PNG files next to it, for visual checks.
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

TEST_SEED_BASE = 900_000_000


def _jpeg(rgb, quality=92):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def worker(job):
    split, scale, seeds, out_dir, shard, n_examples, ex_dir, urdf, center, synonyms = job
    import numpy as np
    from PIL import Image
    env = common.make_env(scale, urdf=urdf, center=center)
    text_rng = np.random.default_rng(1_000_003 * shard + (7 if split == "train" else 0))
    rows, skipped = [], 0
    for seed in seeds:
        try:
            obs, info = env.reset(seed=seed, render=True)
        except RuntimeError:
            skipped += 1
            continue
        tgt = info["target"]
        lab = common.label_joints(env, tgt["pos"])
        if lab is None:
            skipped += 1
            continue
        deg, tilt, pos_err = lab
        cam = env.stereo_cam
        descr = common.vary_description(tgt["descr"], text_rng) if (synonyms and split == "train") else tgt["descr"]
        row = {"seed": int(seed), "split": split, "camera_scale": float(scale), "target": descr,
               "prompt": common.PROMPT.format(descr=descr), "answer": common.answer_json(deg),
               "joints_deg": [float(v) for v in deg], "target_pos": [float(v) for v in tgt["pos"]],
               "target_size": float(tgt["size"]), "approach_tilt_deg": float(tilt), "ik_error_m": pos_err,
               "n_objects": len(env.objects),
               "camera": json.dumps({k: cam[k] for k in ("distance_m", "height_m", "pitch_deg", "azimuth_deg",
                                                         "yaw_deg", "roll_deg", "hfov_deg") if k in cam}),
               "left": {"bytes": _jpeg(obs["images"]["left"]), "path": None},
               "right": {"bytes": _jpeg(obs["images"]["right"]), "path": None}}
        rows.append(row)
        if shard == 0 and len(rows) <= n_examples:          # a few samples as JSON + PNG for visual checks
            stem = os.path.join(ex_dir, f"{split}_{len(rows) - 1}")
            Image.fromarray(obs["images"]["left"]).save(stem + "_left.png")
            Image.fromarray(obs["images"]["right"]).save(stem + "_right.png")
            meta = {k: v for k, v in row.items() if k not in ("left", "right")}
            meta.update({"left_image": os.path.basename(stem + "_left.png"),
                         "right_image": os.path.basename(stem + "_right.png")})
            json.dump(meta, open(stem + ".json", "w"), indent=1)
    env.close()
    if rows:
        from datasets import Dataset, Features, Image as HFImage, Value, Sequence
        feats = Features({"seed": Value("int64"), "split": Value("string"), "camera_scale": Value("float32"),
                          "target": Value("string"), "prompt": Value("string"), "answer": Value("string"),
                          "joints_deg": Sequence(Value("float32")), "target_pos": Sequence(Value("float32")),
                          "target_size": Value("float32"), "approach_tilt_deg": Value("float32"),
                          "ik_error_m": Value("float32"), "n_objects": Value("int32"), "camera": Value("string"),
                          "left": HFImage(), "right": HFImage()})
        Dataset.from_list(rows, features=feats).to_parquet(os.path.join(out_dir, split, f"{split}-{shard:05d}.parquet"))
    return split, len(rows), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--train-scale", type=float, default=1.0, help="camera width factor s of the training split")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    ap.add_argument("--calib", default="", help="centre the cameras on pose_left_eye of this calibration file")
    ap.add_argument("--synonyms", action="store_true", help="vary training descriptions (box/block, grey)")
    args = ap.parse_args()
    center = common.center_from_calib(args.calib) if args.calib else None
    if center:
        print("camera centre from", args.calib, {k: round(v, 3) for k, v in center.items()})

    ex_dir = os.path.join(args.out, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    splits = [("train", args.train_scale, 0, args.train)] + [(f"test_T{t}", float(t), TEST_SEED_BASE, args.test)
                                                              for t in (0, 1, 2)]
    jobs = []
    for split, scale, base, n in splits:
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        # a few spare seeds per shard: rare scenes are skipped (object not placeable / unreachable)
        n_shards = max(1, min(args.workers, n // 50))
        per = -(-n // n_shards)
        for k in range(n_shards):
            lo = base + k * per
            hi = base + min(n, (k + 1) * per)
            jobs.append((split, scale, list(range(lo, hi)), args.out, k, args.examples, ex_dir, args.urdf,
                         center, args.synonyms))
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        stats = {}
        for split, n, skipped in pool.imap_unordered(worker, jobs):
            s = stats.setdefault(split, [0, 0])
            s[0] += n
            s[1] += skipped
            print(f"  {split}: +{n} samples ({skipped} skipped)  [{time.time() - t0:.0f} s]", flush=True)
    for split, (n, skipped) in stats.items():
        print(f"{split}: {n} samples, {skipped} scenes skipped")
    json.dump({"args": vars(args), "camera_center": center, "counts": {k: v[0] for k, v in stats.items()},
               "joint_keys": common.JOINT_KEYS, "ref_size_m": common.REF_SIZE, "tilts_deg": common.TILTS_DEG},
              open(os.path.join(args.out, "dataset_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
