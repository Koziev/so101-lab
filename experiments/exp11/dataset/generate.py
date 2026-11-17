"""
Generate the exp11 localization dataset: stereo pair + instruction -> joint-space target (4 angles, JSON).

    python dataset/generate.py --setup ../setups/2026-11_12v_stereo28mm_wrist --out data_v11 \
        --train 20000 --test 300 --train-scale 1.5 --workers 6

Scenes: 2-5 printed objects (objects_v11), white with weight --white-weight; every scene is built around its
task (tasks.py): full (colour + shape), colour only, shape only, superlative (biggest / smallest of a shape).
Camera: centred on the setup's calibrated stereo pose (pose_base), rectified 60.9 deg view, width factor s.
Splits: train (s = --train-scale), test_T0 / test_T1 / test_T2 (s = 0 / 1 / 2, same seeds, same scenes).
Each row stores the task type, the instruction and its canonical form, and every object (label, shape, colour,
target point, volume) so that evaluation can tell "near the target" from "went to the wrong object".
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
    split, scale, seeds, out_dir, shard, n_examples, ex_dir, urdf, cell, task_kwargs, vary, real_bg, image, tip = job
    from PIL import Image
    import tasks
    env = common.make_env(scale, urdf=urdf, cell=cell, task_kwargs=dict(task_kwargs, vary=vary),
                          real_background=real_bg, image=image, tip_paint_mm=tip)
    rows, skipped = [], 0
    for seed in seeds:
        try:
            obs, info = env.reset(seed=seed, render=env.crop is None)
        except RuntimeError:
            skipped += 1
            continue
        if env.crop is not None:                    # every object must lie inside the window the model gets
            if not all(env.in_image(env.object_pose(o["id"])[0]) for o in env.objects):
                skipped += 1
                continue
            obs = env.observe(render=True)
        task = env.task
        tgt_point = env.target_point(env.objects[0])
        lab = common.label_joints(env, tgt_point)
        if lab is None:
            skipped += 1
            continue
        deg, tilt, pos_err = lab
        cam = env.stereo_cam
        objects = [{"label": o["descr"], "shape": o["shape"], "colour": o["color"],
                    "lying": bool(o["spec"]["lying"]), "point": env.target_point(o).tolist(),
                    "volume_cm3": round(1e6 * tasks.volume(o["spec"]), 2), "dims_mm":
                    {k: round(1000 * v, 1) for k, v in o["spec"]["dims"].items()}} for o in env.objects]
        row = {"seed": int(seed), "split": split, "camera_scale": float(scale),
               "task_type": task["type"], "instruction": task["instruction"], "canonical": task["canonical"],
               "target": objects[0]["label"], "target_colour": objects[0]["colour"],
               "target_shape": objects[0]["shape"],
               "prompt": common.PROMPT.format(instruction=task["instruction"]), "answer": common.answer_json(deg),
               "joints_deg": [float(v) for v in deg], "target_pos": [float(v) for v in tgt_point],
               "approach_tilt_deg": float(tilt), "ik_error_m": pos_err, "n_objects": len(env.objects),
               "objects": json.dumps(objects),
               "camera": json.dumps({k: cam[k] for k in ("distance_m", "height_m", "pitch_deg", "azimuth_deg",
                                                         "yaw_deg", "roll_deg", "hfov_deg") if k in cam}),
               "left": {"bytes": _jpeg(obs["images"]["left"]), "path": None},
               "right": {"bytes": _jpeg(obs["images"]["right"]), "path": None}}
        rows.append(row)
        if shard == 0 and len(rows) <= n_examples:
            stem = os.path.join(ex_dir, f"{split}_{len(rows) - 1}")
            Image.fromarray(obs["images"]["left"]).save(stem + "_left.png")
            Image.fromarray(obs["images"]["right"]).save(stem + "_right.png")
            meta = {k: v for k, v in row.items() if k not in ("left", "right")}
            meta["objects"] = objects
            meta["tip_px"] = int(getattr(env, "tip_px", -1))          # painted jaw-tip pixels (right eye)
            json.dump(meta, open(stem + ".json", "w"), indent=1)
    env.close()
    if rows:
        from datasets import Dataset, Features, Image as HFImage, Value, Sequence
        s = lambda: Value("string")
        feats = Features({"seed": Value("int64"), "split": s(), "camera_scale": Value("float32"),
                          "task_type": s(), "instruction": s(), "canonical": s(), "target": s(),
                          "target_colour": s(), "target_shape": s(), "prompt": s(), "answer": s(),
                          "joints_deg": Sequence(Value("float32")), "target_pos": Sequence(Value("float32")),
                          "approach_tilt_deg": Value("float32"), "ik_error_m": Value("float32"),
                          "n_objects": Value("int32"), "objects": s(), "camera": s(),
                          "left": HFImage(), "right": HFImage()})
        Dataset.from_list(rows, features=feats).to_parquet(os.path.join(out_dir, split, f"{split}-{shard:05d}.parquet"))
    return split, len(rows), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, help="setup folder (stereo_calib.json with pose_base)")
    ap.add_argument("--out", default="data_v11")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--train-scale", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    ap.add_argument("--weights", default="full=0.4,colour=0.2,shape=0.2,superlative=0.2",
                    help="task mix, e.g. full=0.4,colour=0.2,shape=0.2,superlative=0.2")
    ap.add_argument("--white-weight", type=float, default=0.5)
    ap.add_argument("--objects", default="2,5", help="objects per scene, min,max")
    ap.add_argument("--margin", type=float, default=1.35, help="superlatives: volume ratio to the runner-up")
    ap.add_argument("--image", choices=list(common.IMAGE_MODES), default="full",
                    help="what the model sees: the whole view at 640x360, or the workspace window at 2x / 1.5x "
                         "the pixel density (common.IMAGE_MODES)")
    ap.add_argument("--tip-paint-mm", type=float, default=20.0,
                    help="the fixed jaw's last N mm drawn black, as painted on the real arm (0 = unpainted arm)")
    ap.add_argument("--synthetic-scene", action="store_true",
                    help="the simulator's own board and backdrop instead of the setup's real background picture")
    ap.add_argument("--plain-test", action="store_true",
                    help="test splits with canonical wording only (default: varied, like training)")
    args = ap.parse_args()

    cell = common.cell_config(args.setup)
    task_kwargs = {"weights": {k: float(v) for k, v in (p.split("=") for p in args.weights.split(","))},
                   "white_weight": args.white_weight, "margin": args.margin,
                   "n_objects": tuple(int(v) for v in args.objects.split(","))}
    env = common.make_env(0.0, urdf=args.urdf, cell=cell, real_background=not args.synthetic_scene,
                          image=args.image)
    center = dict(env.cfg.v9_center)
    mode = common.IMAGE_MODES[args.image]
    size = (mode["crop"][2] - mode["crop"][0], mode["crop"][3] - mode["crop"][1]) if mode["crop"] else mode["frame"]
    print(f"image mode '{args.image}': {size[0]}x{size[1]} pixels per eye"
          + (f", window {mode['crop']} of a {mode['frame'][0]}x{mode['frame'][1]} view" if mode["crop"] else ""))
    print("scene: " + (f"the setup's real background ({len(env.real_bg or [])} picture(s))" if env.real_bg is not None and not args.synthetic_scene
                       else "SYNTHETIC board and backdrop (no stereo_background.png in the setup, or --synthetic-scene)")
          + f"; board {1000 * cell.get('board_z_m', 0.0):+.1f} mm above the base plane; objects "
          + f"{100 * common.R_RANGE[0]:.0f}-{100 * common.R_RANGE[1]:.0f} cm from the pan axis")
    env.close()
    print(f"camera centre from {args.setup}: " + ", ".join(f"{k} {v:.3f}" for k, v in center.items())
          + f"; rectified HFOV {cell['hfov_deg']:.1f} deg, baseline {100 * cell['baseline_m']:.2f} cm")
    ex_dir = os.path.join(args.out, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    splits = [("train", args.train_scale, 0, args.train, True)] + \
             [(f"test_T{t}", float(t), TEST_SEED_BASE, args.test, not args.plain_test) for t in (0, 1, 2)]
    jobs = []
    for split, scale, base, n, vary in splits:
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        n_shards = max(1, min(args.workers, n // 50))
        per = -(-n // n_shards)
        for k in range(n_shards):
            jobs.append((split, scale, list(range(base + k * per, base + min(n, (k + 1) * per))), args.out, k,
                         args.examples, ex_dir, args.urdf, cell, task_kwargs, vary, not args.synthetic_scene, args.image, args.tip_paint_mm))
    t0 = time.time()
    stats = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for split, n, skipped in pool.imap_unordered(worker, jobs):
            s = stats.setdefault(split, [0, 0])
            s[0] += n
            s[1] += skipped
            print(f"  {split}: +{n} samples ({skipped} skipped)  [{time.time() - t0:.0f} s]", flush=True)
    for split, (n, skipped) in stats.items():
        print(f"{split}: {n} samples, {skipped} scenes skipped")
    json.dump({"args": vars(args), "image": args.image, "camera_center": center, "cell": {k: v for k, v in cell.items()},
               "task_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in task_kwargs.items()},
               "counts": {k: v[0] for k, v in stats.items()}, "joint_keys": common.JOINT_KEYS},
              open(os.path.join(args.out, "dataset_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
