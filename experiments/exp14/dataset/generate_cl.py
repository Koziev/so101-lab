"""
Generate the closed-loop dataset of experiment 11: stereo pair with the arm somewhere on its way + instruction +
current servo angles -> change of the four servo angles for the next step + gripper decision (closed_loop.py).

    python dataset/generate_cl.py --setup setups/<name> --out data_v11cl --train 12000 --states 2 --workers 6

Scenes, tasks, camera, real background, board height and labels' grasp height are those of the one-shot dataset
(dataset/generate.py). Per scene, --states poses of the arm are drawn (ready pose, on the way to the pre-grasp
pose, coming down, around the target; closed_loop.sample_state) and rendered; the gripper is open. The angles
written into the prompt differ from the rendered pose by a little noise, as real readings do.
Splits: train (camera width --train-scale), test_T0 / test_T1 (same seeds, calibrated / randomised camera).
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import closed_loop as cl  # noqa: E402
import common  # noqa: E402

TEST_SEED_BASE = 910_000_000


def _jpeg(rgb, quality=92):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def worker(job):
    split, scale, seeds, out_dir, shard, n_examples, ex_dir, urdf, cell, task_kwargs, vary, real_bg, n_states, tip = job
    from PIL import Image
    import tasks
    env = common.make_env(scale, urdf=urdf, cell=cell, task_kwargs=dict(task_kwargs, vary=vary),
                          real_background=real_bg, tip_paint_mm=tip)
    rows, skipped = [], 0
    for seed in seeds:
        try:
            obs, _ = env.reset(seed=seed, render=False)
        except RuntimeError:
            skipped += 1
            continue
        task = env.task
        tgt_point = env.target_point(env.objects[0])
        ex = cl.Expert(env, tgt_point)
        if not ex.ok:
            skipped += 1
            continue
        q_start = np.asarray(obs["state"], float)[:4]
        objects = [{"label": o["descr"], "shape": o["shape"], "colour": o["color"],
                    "lying": bool(o["spec"]["lying"]), "point": env.target_point(o).tolist(),
                    "volume_cm3": round(1e6 * tasks.volume(o["spec"]), 2)} for o in env.objects]
        cam = env.stereo_cam
        for k in range(n_states):
            rng = np.random.default_rng([int(seed), 7, k])              # independent of the scene's own stream
            st = cl.sample_state(rng, env, ex, q_start)
            if st is None:
                continue
            q, kind = st
            env.set_joint_positions(env.from_lerobot(np.r_[q, 0.0, rng.uniform(60, 100)]))
            images = env.observe(render=True)["images"]                 # no physics step: the objects stay put
            delta, gripper, goal = ex.action(q)
            shown = q + rng.normal(0, cl.READ_NOISE, 4)
            row = {"seed": int(seed), "state": k, "split": split, "camera_scale": float(scale),
                   "task_type": task["type"], "instruction": task["instruction"], "canonical": task["canonical"],
                   "target": objects[0]["label"], "target_colour": objects[0]["colour"],
                   "target_shape": objects[0]["shape"],
                   "prompt": cl.prompt(task["instruction"], shown), "answer": cl.answer_json(delta, gripper),
                   "joints_shown": [cl.one_decimal(v) for v in shown], "joints_true": [float(v) for v in q],
                   "delta": [cl.one_decimal(v) for v in delta], "gripper": gripper, "kind": kind, "goal": goal,
                   "q_final": [float(v) for v in ex.q_final],
                   "q_pre": [float(v) for v in ex.q_pre] if ex.q_pre is not None else [],
                   "target_pos": [float(v) for v in tgt_point], "n_objects": len(env.objects),
                   "objects": json.dumps(objects),
                   "camera": json.dumps({c: cam[c] for c in ("distance_m", "height_m", "pitch_deg", "azimuth_deg",
                                                             "yaw_deg", "roll_deg", "hfov_deg") if c in cam}),
                   "left": {"bytes": _jpeg(images["left"]), "path": None},
                   "right": {"bytes": _jpeg(images["right"]), "path": None}}
            rows.append(row)
            if shard == 0 and len(rows) <= n_examples:
                stem = os.path.join(ex_dir, f"{split}_{len(rows) - 1}")
                Image.fromarray(images["left"]).save(stem + "_left.png")
                Image.fromarray(images["right"]).save(stem + "_right.png")
                meta = {c: v for c, v in row.items() if c not in ("left", "right")}
                meta["objects"] = objects
                meta["tip_px"] = int(getattr(env, "tip_px", -1))      # painted jaw-tip pixels (left eye)
                json.dump(meta, open(stem + ".json", "w"), indent=1)
    env.close()
    if rows:
        from datasets import Dataset, Features, Image as HFImage, Sequence, Value
        s, f4 = (lambda: Value("string")), (lambda: Sequence(Value("float32")))
        feats = Features({"seed": Value("int64"), "state": Value("int32"), "split": s(),
                          "camera_scale": Value("float32"), "task_type": s(), "instruction": s(), "canonical": s(),
                          "target": s(), "target_colour": s(), "target_shape": s(), "prompt": s(), "answer": s(),
                          "joints_shown": f4(), "joints_true": f4(), "delta": f4(), "gripper": s(), "kind": s(),
                          "goal": s(), "q_final": f4(), "q_pre": f4(), "target_pos": f4(),
                          "n_objects": Value("int32"), "objects": s(), "camera": s(),
                          "left": HFImage(), "right": HFImage()})
        Dataset.from_list(rows, features=feats).to_parquet(os.path.join(out_dir, split, f"{split}-{shard:05d}.parquet"))
    kinds = {}
    for r in rows:
        key = r["kind"] + ("/close" if r["gripper"] == "close" else "")
        kinds[key] = kinds.get(key, 0) + 1
    return split, len(rows), skipped, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--out", default="data_v11cl")
    ap.add_argument("--train", type=int, default=12000, help="training scenes")
    ap.add_argument("--test", type=int, default=200, help="test scenes per test split")
    ap.add_argument("--states", type=int, default=2, help="arm poses rendered per scene")
    ap.add_argument("--train-scale", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    ap.add_argument("--weights", default="full=0.4,colour=0.2,shape=0.2,superlative=0.2")
    ap.add_argument("--white-weight", type=float, default=0.5)
    ap.add_argument("--objects", default="2,5")
    ap.add_argument("--margin", type=float, default=1.35)
    ap.add_argument("--tip-paint-mm", type=float, default=20.0,
                    help="the fixed jaw's last N mm drawn black, as painted on the real arm (0 = unpainted arm)")
    ap.add_argument("--synthetic-scene", action="store_true")
    args = ap.parse_args()

    cell = common.cell_config(args.setup)
    task_kwargs = {"weights": {k: float(v) for k, v in (p.split("=") for p in args.weights.split(","))},
                   "white_weight": args.white_weight, "margin": args.margin,
                   "n_objects": tuple(int(v) for v in args.objects.split(","))}
    env = common.make_env(0.0, urdf=args.urdf, cell=cell)
    print("scene: " + (f"the setup's real background ({len(env.real_bg or [])} picture(s))"
                       if env.real_bg is not None and not args.synthetic_scene else "SYNTHETIC board and backdrop")
          + f"; board {1000 * cell.get('board_z_m', 0.0):+.1f} mm above the base plane; step limit {cl.STEP_MAX:g} deg,"
          + f" close within {1000 * cl.CLOSE_XY:g} mm (grasp point), pre-grasp {100 * cl.PRE_UP:g} cm above the grasp point")
    env.close()
    ex_dir = os.path.join(args.out, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    splits = [("train", args.train_scale, 0, args.train)] + \
             [(f"test_T{t}", float(t), TEST_SEED_BASE, args.test) for t in (0, 1)]
    jobs = []
    for split, scale, base, n in splits:
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        n_shards = max(1, min(args.workers, n // 50))
        per = -(-n // n_shards)
        for k in range(n_shards):
            jobs.append((split, scale, list(range(base + k * per, base + min(n, (k + 1) * per))), args.out, k,
                         args.examples, ex_dir, args.urdf, cell, task_kwargs, True, not args.synthetic_scene,
                         args.states, args.tip_paint_mm))
    t0 = time.time()
    stats, kinds = {}, {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for split, n, skipped, kk in pool.imap_unordered(worker, jobs):
            s = stats.setdefault(split, [0, 0])
            s[0] += n
            s[1] += skipped
            for key, v in kk.items():
                kinds.setdefault(split, {})[key] = kinds.setdefault(split, {}).get(key, 0) + v
            print(f"  {split}: +{n} samples ({skipped} scenes skipped)  [{time.time() - t0:.0f} s]", flush=True)
    for split, (n, skipped) in stats.items():
        print(f"{split}: {n} samples, {skipped} scenes skipped; " +
              ", ".join(f"{k} {v}" for k, v in sorted(kinds.get(split, {}).items())))
    json.dump({"args": vars(args), "cell": cell, "kind": "closed_loop",
               "closed_loop": {"version": cl.VERSION, "step_max_deg": cl.STEP_MAX, "close_xy_m": cl.CLOSE_XY,
                               "close_z_m": cl.CLOSE_Z, "near_xy_m": cl.NEAR_XY,
                               "pre_up_m": cl.PRE_UP, "read_noise_deg": cl.READ_NOISE},
               "task_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in task_kwargs.items()},
               "counts": {k: v[0] for k, v in stats.items()}, "joint_keys": common.JOINT_KEYS},
              open(os.path.join(args.out, "dataset_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
