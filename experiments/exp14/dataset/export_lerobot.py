"""
Raw approach episodes (dataset/record_approach.py) -> LeRobotDataset (written for lerobot 0.6). Run in the
LeRobot environment:

    python dataset/export_lerobot.py --raw data_v12/raw --root $DATA/so101_approach_v12 --repo-id local/so101_approach_v12

Output:
  <root>/          training episodes
  <root>_eval/     held-out episodes (--eval-fraction of the seeds, chosen by a hash of the seed, so the split
                   does not change when more episodes are recorded)
  <root>/sim_episodes.jsonl   per episode: raw folder, seed, instruction, goal, final errors
  <root>/eval_seeds.json      seeds of the held-out episodes (evaluation/eval_approach.py --held-out)
Features:
  observation.state, action          6 values: 5 joints in degrees + gripper 0..100
  observation.images.left / .right   video, 640x360
  sim.phase (0 transit, 1 descend, 2 hold), sim.tip (m, world), sim.contact (object, board)
"""
import argparse
import contextlib
import glob
import json
import os
import shutil
import time

import cv2
import numpy as np

JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
               "gripper.pos"]
CAMERAS = ["left", "right"]


def is_eval_seed(seed, fraction):
    return (seed * 2654435761 % 2 ** 32) / 2 ** 32 < fraction


def features(width, height):
    f = {"observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
         "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
         "sim.phase": {"dtype": "int64", "shape": (1,), "names": None},
         "sim.tip": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
         "sim.contact": {"dtype": "float32", "shape": (2,), "names": ["object", "board"]}}
    for cam in CAMERAS:
        f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (height, width, 3),
                                          "names": ["height", "width", "channels"]}
    return f


def add_episode(ds, ep_dir, meta):
    d = np.load(os.path.join(ep_dir, "data.npz"))
    n = len(d["state"])
    for t in range(n):
        frame = {"observation.state": d["state"][t].astype(np.float32),
                 "action": d["action"][t].astype(np.float32),
                 "sim.phase": np.array([d["phase"][t]], dtype=np.int64),
                 "sim.tip": d["tip"][t].astype(np.float32),
                 "sim.contact": d["contact"][t].astype(np.float32),
                 "task": meta["instruction"]}
        for cam in CAMERAS:
            img = cv2.imread(os.path.join(ep_dir, cam, f"{t:06d}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"missing frame {cam}/{t:06d} in {ep_dir}")
            frame[f"observation.images.{cam}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ds.add_frame(frame)
    ds.save_episode()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/media/inkoziev/corpora/tmp/lerobot_datasets/data_v12/raw")
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="local/so101_approach_v12")
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument("--max-episodes", type=int, default=0, help="limit training episodes (0 = all), for a test")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--video-backend", default="pyav", help="decoder for the final check: pyav or torchcodec")
    args = ap.parse_args()

    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        from lerobot.datasets.video_utils import VideoEncodingManager
    except ImportError:
        VideoEncodingManager = None
    if not lerobot.__version__.startswith("0.6"):
        print(f"warning: written for lerobot 0.6.x, found {lerobot.__version__}")

    eps = []
    for m in sorted(glob.glob(os.path.join(args.raw, "shard_*", "episode_*", "meta.json"))):
        meta = json.load(open(m))
        if meta.get("success"):
            eps.append((os.path.dirname(m), meta))
    if not eps:
        raise SystemExit(f"no successful episodes under {args.raw}/shard_*/episode_*")
    train = [e for e in eps if not is_eval_seed(e[1]["seed"], args.eval_fraction)]
    held = [e for e in eps if is_eval_seed(e[1]["seed"], args.eval_fraction)]
    if args.max_episodes:
        train, held = train[:args.max_episodes], held[:max(1, args.max_episodes // 10)]
    fps = eps[0][1]["fps"]
    h, w = cv2.imread(os.path.join(eps[0][0], "left", "000000.jpg")).shape[:2]
    n_frames = sum(m["num_steps"] for _, m in train)
    print(f"{len(train)} training episodes ({n_frames} frames), {len(held)} held out; {w}x{h} @ {fps} fps")

    for name, subset, root, repo in (("train", train, args.root, args.repo_id),
                                     ("eval", held, args.root + "_eval", args.repo_id + "_eval")):
        if not subset:
            continue
        if os.path.exists(root):
            if not args.overwrite:
                raise SystemExit(f"{root} exists; use --overwrite")
            shutil.rmtree(root)
        ds = LeRobotDataset.create(repo_id=repo, fps=fps, features=features(w, h), root=root,
                                   robot_type="so101_follower", use_videos=True)
        t0, frames, records = time.time(), 0, []
        with (VideoEncodingManager(ds) if VideoEncodingManager else contextlib.nullcontext()):
            for i, (ep_dir, meta) in enumerate(subset):
                frames += add_episode(ds, ep_dir, meta)
                records.append({"episode_index": i, "raw_dir": ep_dir,
                                **{k: meta.get(k) for k in ("seed", "instruction", "canonical", "task_type", "goal",
                                                            "final", "num_steps", "actuator", "command",
                                                            "variant", "start", "metrics")}})
                if (i + 1) % 25 == 0 or i + 1 == len(subset):
                    rate = (i + 1) / (time.time() - t0)
                    print(f"[{name}] {i + 1}/{len(subset)} episodes, {frames} frames, "
                          f"eta {(len(subset) - i - 1) / rate / 60:.1f} min", flush=True)
        ds.finalize()
        with open(os.path.join(root, "sim_episodes.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        if name == "eval":
            json.dump({"eval_fraction": args.eval_fraction, "seeds": [m["seed"] for _, m in subset],
                       "episodes": [{k: m.get(k) for k in ("seed", "command", "variant", "start")}
                                    for _, m in subset]},
                      open(os.path.join(args.root, "eval_seeds.json"), "w"))

    try:                                   # pyav: works without torchcodec (see README_v12, "Video decoding")
        ds = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)
    except TypeError:
        ds = LeRobotDataset(args.repo_id, root=args.root)
    print(f"\nreloaded {args.root}: {ds.num_episodes} episodes, {ds.num_frames} frames")
    for k, v in ds[0].items():
        print(f"    {k:28s} " + (f"shape={tuple(v.shape)} dtype={v.dtype}" if hasattr(v, "shape") else repr(v)[:60]))


if __name__ == "__main__":
    main()
