"""
Generate grasping episodes with the scripted expert, in parallel shards.

    python record_episodes.py --out data/raw --num 1000 --workers 8

Each worker is a separate process with its own PyBullet instance and writes to its own
shard: data/raw/shard_03/episode_00012/...

* Every episode is fully determined by its scene seed (scene and expert randomness),
  and seed ranges of different shards never overlap.
* Episodes are written to a temporary directory and renamed when complete, so a crash
  never leaves a half-written episode.
* Re-running the same command resumes: each shard counts its finished episodes and
  continues after the last used seed. To add data later without touching existing
  shards, use --shard-offset (e.g. --shard-offset 8 for shards 8..15).

Episode directory contents:
  front/000000.png, wrist/000000.png (and top/ with --top-camera) ...  camera frames
  data.npz    per-step arrays: state, action (clean expert action, LeRobot units),
              action_executed (what was sent to the robot; differs only in perturbed
              episodes), phase, tcp, target_pos, grasp flags, timestamp
  meta.json   instruction, objects, camera/light parameters, plan info, success
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from so101_sim.env import DEFAULT_TEXTURE_DIR

FLAGS = ["jaw_contact", "fixed_contact", "lifted", "grasped", "gripper_table_contact"]
SEED_STRIDE = 10_000_000  # seeds of shard k are base + k * SEED_STRIDE + i


def shard_progress(shard_dir):
    """Number of finished episodes and the last seed used in a shard."""
    metas = glob.glob(os.path.join(shard_dir, "episode_*", "meta.json"))
    seeds = []
    for m in metas:
        with open(m) as f:
            seeds.append(json.load(f)["seed"])
    return len(metas), (max(seeds) if seeds else None)


def record_shard(shard, n_target, args):
    import cv2

    from so101_sim.env import EnvConfig, SO101GraspEnv, DEFAULT_TEXTURE_DIR
    from so101_sim.expert import PHASES, ExpertConfig, ScriptedGraspExpert

    cv2.setNumThreads(1)
    shard_dir = os.path.join(args["out"], f"shard_{shard:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    for tmp in glob.glob(os.path.join(shard_dir, ".tmp_*")):  # leftovers of an interrupted run
        shutil.rmtree(tmp, ignore_errors=True)

    done, last_seed = shard_progress(shard_dir)
    seed = args["seed"] + shard * SEED_STRIDE if last_seed is None else last_seed + 1
    tag = f"[shard {shard:02d}]"
    if done >= n_target:
        print(f"{tag} already complete ({done} episodes)", flush=True)
        return
    if done:
        print(f"{tag} resuming at episode {done}, seed {seed}", flush=True)

    size = (args["width"], args["height"])
    env = SO101GraspEnv(args["urdf"], EnvConfig(renderer=args["renderer"], front_size=size, wrist_size=size,
                                                 n_distractors=(0, args["max_distractors"]),
                                                 top_camera=args["top_camera"],
                                                 camera_preset=args["camera_preset"],
                                                 scene_preset=args["scene_preset"], texture_dir=args["texture_dir"],
                                                 v8_camera_split=args["camera_split"],
                                                 texture_split=args["texture_split"],
                                                 record_wrist=not args["no_wrist"]), gui=args["gui"])
    expert = ScriptedGraspExpert(env, ExpertConfig(p_offset=args["p_offset"], p_perturb=args["p_perturb"],
                                                   early_lift=args["early_lift"]))
    ext = args["image_format"]
    img_params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if ext == "png" else [cv2.IMWRITE_JPEG_QUALITY, 95]
    skipped = 0

    while done < n_target:
        obs, info = env.reset(seed=seed)
        expert.rng = np.random.default_rng([seed, 1])  # expert randomness tied to the scene seed
        ep_seed, seed = seed, seed + 1
        if not expert.reset(obs, info):
            skipped += 1
            continue
        t0 = time.time()
        final_dir = os.path.join(shard_dir, f"episode_{done:05d}")
        tmp_dir = os.path.join(shard_dir, f".tmp_episode_{done:05d}")
        for cam in obs["images"]:
            os.makedirs(os.path.join(tmp_dir, cam), exist_ok=True)
        tgt = info["target"]
        obs_cams = list(obs["images"])
        rec = {k: [] for k in ["state", "action", "action_executed", "phase", "tcp", "target_pos", "flags",
                               "timestamp"]}
        step = 0
        while True:
            action, phase = expert.act(obs)
            if action is None:
                break
            for cam, img in obs["images"].items():
                path = os.path.join(tmp_dir, cam, f"{step:06d}.{ext}")
                if not cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), img_params):
                    raise RuntimeError(f"failed to write {path} (disk full?)")
            gs = env.grasp_state(tgt["id"], tgt["pos"][2])
            rec["state"].append(obs["state"])
            rec["action"].append(action)                      # clean expert action = training label
            executed = expert.executed(action)                # perturbed in perturbed episodes
            rec["action_executed"].append(executed)
            rec["phase"].append(PHASES.index(phase))
            rec["tcp"].append(obs["tcp"])
            rec["target_pos"].append(env.object_pose(tgt["id"])[0])
            rec["flags"].append([gs[f] for f in FLAGS])
            rec["timestamp"].append(step / env.cfg.control_hz)
            more = expert.t < len(expert.actions)  # no need to render after the last action
            obs = env.step(executed, render=more)
            step += 1

        final = env.grasp_state(tgt["id"], tgt["pos"][2])
        np.savez_compressed(os.path.join(tmp_dir, "data.npz"),
                            **{k: np.asarray(v, dtype=np.int8 if k == "phase" else np.float32) for k, v in rec.items()})
        meta = {"shard": shard, "episode": done, "seed": ep_seed, "instruction": info["instruction"],
                "target": {k: v for k, v in tgt.items() if k != "id"}, "distractors": info["distractors"],
                "front_camera": info["front_camera"], "top_camera": info["top_camera"], "light": info["light"],
                "plan": expert.plan_info, "cameras": sorted(obs_cams),
                "stereo_camera": info.get("stereo_camera"), "camera_pipeline": info.get("camera_pipeline"),
                "scene_preset": args["scene_preset"],
                "camera_desc": info.get("camera_desc"), "actuator": info.get("actuator"), "camera_preset": args["camera_preset"],
                "success": final["grasped"], "final_state": final, "num_steps": step,
                "fps": env.cfg.control_hz, "image_size": list(size),
                "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                "units": "arm joints: degrees (URDF, 0 = mid-range); gripper: 0 closed .. 100 open",
                "phases": PHASES, "flag_names": FLAGS}
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        os.rename(tmp_dir, final_dir)  # atomic: the episode exists only once complete
        done += 1
        dt = time.time() - t0
        print(f"{tag} episode {done}/{n_target}: {step} steps, success={final['grasped']}, "
              f"'{info['instruction']}', {dt:.1f}s ({1000 * dt / step:.0f} ms/step)", flush=True)

    with open(os.path.join(shard_dir, "shard_info.json"), "w") as f:
        json.dump({"shard": shard, "episodes": done, "skipped_scenes_last_run": skipped,
                   "next_seed": seed}, f, indent=2)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="assets_so101/so101_sim.urdf")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--num", type=int, default=10, help="total episodes over all shards of this run")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--shard-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui", action="store_true", help="watch generation (forces a single worker)")
    ap.add_argument("--renderer", default="tiny", choices=["tiny", "egl"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-distractors", type=int, default=0)
    ap.add_argument("--p-offset", type=float, default=0.0, help="share of deliberately offset grasps")
    ap.add_argument("--top-camera", action="store_true", help="add an overhead camera ('top')")
    ap.add_argument("--camera-split", default="train", choices=["train", "unseen", "all"],
                    help="v8: camera viewpoints (train excludes the held-out sector/distances)")
    ap.add_argument("--scene-preset", default="v1", choices=["v1", "v7", "v8"],
                    help="v7: real board/desk, photo textures, stereo camera (use --width 640 --height 360)")
    ap.add_argument("--texture-dir", default=DEFAULT_TEXTURE_DIR)
    ap.add_argument("--texture-split", default="train", choices=["train", "eval", "all"])
    ap.add_argument("--no-wrist", action="store_true", help="v7: do not render the wrist camera")
    ap.add_argument("--early-lift", action="store_true",
                    help="expert lifts as soon as the gripper is blocked by the object (no long static wait)")
    ap.add_argument("--image-format", default="png", choices=["png", "jpg"],
                    help="raw frame format; jpg (quality 95) needs ~5x less disk")
    ap.add_argument("--camera-preset", default="v1", choices=["v1", "v4"],
                    help="camera framing: v1 original, v4 front aimed at the object area + moved wrist camera")
    ap.add_argument("--p-perturb", type=float, default=0.0,
                    help="share of episodes whose executed motion is perturbed (labels stay clean)")
    args = ap.parse_args()
    if args.gui:
        args.workers = 1
    os.makedirs(args.out, exist_ok=True)
    # Keep each worker on one core: numpy/OpenCV thread pools would otherwise compete.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"

    w = args.workers
    quotas = [args.num // w + (1 if k < args.num % w else 0) for k in range(w)]
    shards = [args.shard_offset + k for k in range(w)]
    t0 = time.time()
    cfg = vars(args)
    if w == 1:
        record_shard(shards[0], quotas[0], cfg)
    else:
        ctx = mp.get_context("spawn")  # fresh interpreters: no PyBullet state shared via fork
        procs = [ctx.Process(target=record_shard, args=(s, q, cfg)) for s, q in zip(shards, quotas) if q > 0]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        failed = [p for p in procs if p.exitcode != 0]
        if failed:
            print(f"warning: {len(failed)} worker(s) exited with an error; re-run the same command to resume")

    total = ok = 0
    for s in shards:
        for m in glob.glob(os.path.join(args.out, f"shard_{s:02d}", "episode_*", "meta.json")):
            with open(m) as f:
                ok += json.load(f)["success"]
            total += 1
    hours = (time.time() - t0) / 3600
    print(f"done: {total} episodes in shards {shards[0]}..{shards[-1]}, {ok} successful, "
          f"wall time {hours * 60:.1f} min")


if __name__ == "__main__":
    main()
