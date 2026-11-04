"""
Convert raw simulator episodes (data/raw/shard_*/episode_*) into LeRobotDataset format
(written for lerobot 0.6.0). Run inside the lerobot environment:

    python export_lerobot.py --raw data/raw --root datasets/so101_sim_grasp --max-episodes 20   # quick test
    python export_lerobot.py --raw data/raw --root datasets/so101_sim_grasp                     # full export

Output:
  <root>/            training dataset  (repo id: --repo-id)
  <root>_eval/       held-out episodes (repo id: --repo-id + "_eval"), same format
  <root>/sim_episodes.jsonl   per-episode link back to the raw data: shard, seed,
                              success, instruction, objects, plan
  <root>/eval_seeds.json      scene seeds of the held-out episodes, for closed-loop
                              evaluation on exactly these scenes

The held-out split is a deterministic function of the scene seed, so it never changes
when the dataset grows.

Features:
  observation.state, action          6 values: 5 joints in degrees + gripper 0..100
  observation.images.front / .wrist (/ .top)  video
  sim.phase                          0 approach, 1 descend, 2 close, 3 lift, 4 hold
  sim.flags                          jaw_contact, fixed_contact, lifted, grasped, gripper_table_contact
  sim.tcp, sim.target_pos            world positions (m)
Standard policies only read the observation.* and action keys; the sim.* fields are
ground-truth labels kept for later experiments.
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
               "gripper.pos"]  # same naming as LeRobot's SO-101 follower
CAMERA_ORDER = ["front", "left", "right", "wrist", "top"]


def frame_path(ep_dir, cam, t):
    for ext in ("png", "jpg"):
        p = os.path.join(ep_dir, cam, f"{t:06d}.{ext}")
        if os.path.exists(p):
            return p
    return None


def detect_cameras(ep_dir):
    return [c for c in CAMERA_ORDER if frame_path(ep_dir, c, 0)]


def is_eval_seed(seed, fraction):
    """Deterministic pseudo-random split by scene seed."""
    return (seed * 2654435761 % 2 ** 32) / 2 ** 32 < fraction


def list_episodes(raw_root):
    eps = []
    for meta_path in sorted(glob.glob(os.path.join(raw_root, "shard_*", "episode_*", "meta.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        eps.append((os.path.dirname(meta_path), meta))
    return eps


DESC_NAMES = ["cam_x", "cam_y", "cam_z", "cam_fx", "cam_fy", "cam_fz", "cam_ux", "cam_uy", "cam_uz", "cam_hfov"]


def make_features(width, height, cameras, with_executed=False, append_camera=False):
    state_names = JOINT_NAMES + (DESC_NAMES if append_camera else [])
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "sim.phase": {"dtype": "int64", "shape": (1,), "names": None},
        "sim.flags": {"dtype": "float32", "shape": (5,),
                      "names": ["jaw_contact", "fixed_contact", "lifted", "grasped", "gripper_table_contact"]},
        "sim.tcp": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "sim.target_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    }
    if with_executed:  # perturbed datasets: what was actually sent to the robot
        feats["sim.action_executed"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
    for cam in cameras:
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": (height, width, 3),
                                              "names": ["height", "width", "channels"]}
    return feats


def create_dataset(LeRobotDataset, repo_id, root, fps, features, overwrite):
    if os.path.exists(root):
        if not overwrite:
            raise SystemExit(f"{root} exists; use --overwrite to replace it")
        shutil.rmtree(root)
    return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root,
                                 robot_type="so101_follower", use_videos=True)


def add_episode(ds, ep_dir, meta, cameras, with_executed=False, append_camera=False):
    data = np.load(os.path.join(ep_dir, "data.npz"))
    n = len(data["state"])
    task = meta["instruction"]
    for t in range(n):
        frame = {
            "observation.state": (np.r_[data["state"][t], meta["camera_desc"]] if append_camera
                                  else data["state"][t]).astype(np.float32),
            "action": data["action"][t].astype(np.float32),
            "sim.phase": np.array([data["phase"][t]], dtype=np.int64),
            "sim.flags": data["flags"][t].astype(np.float32),
            "sim.tcp": data["tcp"][t].astype(np.float32),
            "sim.target_pos": data["target_pos"][t].astype(np.float32),
            "task": task,
        }
        if with_executed:
            frame["sim.action_executed"] = data["action_executed"][t].astype(np.float32)
        for cam in cameras:
            path = frame_path(ep_dir, cam, t)
            img = cv2.imread(path, cv2.IMREAD_COLOR) if path else None
            if img is None:
                raise RuntimeError(f"missing frame {cam}/{t:06d} in {ep_dir}")
            frame[f"observation.images.{cam}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ds.add_frame(frame)
    ds.save_episode()
    return n


def sidecar_record(ds_episode, ep_dir, meta):
    keep = ("shard", "episode", "seed", "success", "instruction", "target", "distractors", "plan", "num_steps")
    return {"episode_index": ds_episode, "raw_dir": ep_dir, **{k: meta.get(k) for k in keep}}


def validate(LeRobotDataset, repo_id, root):
    ds = LeRobotDataset(repo_id, root=root)
    print(f"  {root}: {ds.num_episodes} episodes, {ds.num_frames} frames, fps={ds.fps}")
    sample = ds[0]
    for k, v in sample.items():
        desc = f"shape={tuple(v.shape)} dtype={v.dtype}" if hasattr(v, "shape") else repr(v)[:60]
        print(f"    {k:28s} {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--root", default="datasets/so101_sim_grasp")
    ap.add_argument("--repo-id", default="local/so101_sim_grasp")
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument("--include-failures", action="store_true", help="also export unsuccessful episodes")
    ap.add_argument("--max-episodes", type=int, default=0, help="limit training episodes (0 = all), for testing")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cameras", default="", help="comma-separated subset to export, e.g. left,right")
    ap.add_argument("--append-camera", action="store_true",
                    help="v8: append the 10-number camera description to observation.state (model B)")
    args = ap.parse_args()

    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        from lerobot.datasets.video_utils import VideoEncodingManager
    except ImportError:
        VideoEncodingManager = None
    if not lerobot.__version__.startswith("0.6"):
        print(f"warning: written for lerobot 0.6.x, found {lerobot.__version__}")

    episodes = list_episodes(args.raw)
    if not episodes:
        raise SystemExit(f"no episodes found under {args.raw}/shard_*/episode_*")
    if not args.include_failures:
        episodes = [(d, m) for d, m in episodes if m["success"]]
    train = [(d, m) for d, m in episodes if not is_eval_seed(m["seed"], args.eval_fraction)]
    held = [(d, m) for d, m in episodes if is_eval_seed(m["seed"], args.eval_fraction)]
    if args.max_episodes:
        train = train[:args.max_episodes]
        held = held[:max(1, args.max_episodes // 10)]
    fps = episodes[0][1]["fps"]
    sizes = {tuple(m["image_size"]) for _, m in episodes}
    if len(sizes) > 1:
        raise SystemExit(f"raw episodes have different image sizes {sizes}; export them separately")
    cameras = detect_cameras(episodes[0][0])
    if any(detect_cameras(d) != cameras for d, _ in episodes):
        raise SystemExit("episodes have different camera sets; export them separately")
    if args.cameras:
        wanted = args.cameras.split(",")
        missing = [c for c in wanted if c not in cameras]
        if missing:
            raise SystemExit(f"cameras {missing} not in the raw data (available: {cameras})")
        cameras = wanted
    first = cv2.imread(frame_path(episodes[0][0], cameras[0], 0))  # trust the pixels
    height, width = first.shape[:2]
    print(f"{len(train)} training and {len(held)} held-out episodes, {width}x{height} @ {fps} fps, "
          f"cameras: {cameras}")

    with_executed = "action_executed" in np.load(os.path.join(episodes[0][0], "data.npz")).files
    if args.append_camera and any(m.get("camera_desc") is None for _, m in episodes):
        raise SystemExit("--append-camera needs v8 episodes (meta.json without camera_desc found)")
    features = make_features(width, height, cameras, with_executed, args.append_camera)
    for name, subset, root, repo_id in (("train", train, args.root, args.repo_id),
                                        ("eval", held, args.root + "_eval", args.repo_id + "_eval")):
        if not subset:
            continue
        ds = create_dataset(LeRobotDataset, repo_id, root, fps, features, args.overwrite)
        records, t0, frames = [], time.time(), 0
        manager = VideoEncodingManager(ds) if VideoEncodingManager else contextlib.nullcontext()
        with manager:
            for i, (ep_dir, meta) in enumerate(subset):
                frames += add_episode(ds, ep_dir, meta, cameras, with_executed, args.append_camera)
                records.append(sidecar_record(i, ep_dir, meta))
                if (i + 1) % 20 == 0 or i + 1 == len(subset):
                    rate = (i + 1) / (time.time() - t0)
                    print(f"[{name}] {i + 1}/{len(subset)} episodes, {frames} frames, "
                          f"{rate:.2f} ep/s, eta {(len(subset) - i - 1) / rate / 60:.1f} min", flush=True)
        ds.finalize()
        with open(os.path.join(root, "sim_episodes.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        if name == "eval":
            with open(os.path.join(args.root, "eval_seeds.json"), "w") as f:
                json.dump({"eval_fraction": args.eval_fraction, "seeds": [m["seed"] for _, m in subset]}, f)

    print("\nvalidation (reloading the written datasets):")
    validate(LeRobotDataset, args.repo_id, args.root)
    if held:
        validate(LeRobotDataset, args.repo_id + "_eval", args.root + "_eval")


if __name__ == "__main__":
    main()
