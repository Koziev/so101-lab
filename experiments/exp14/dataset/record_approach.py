"""
Record "Approach the object" episodes in the exp11 cell (experiment 12), in parallel shards.

    python dataset/record_approach.py --setup setups/<name> --out data_v12/raw --num 600 --workers 8

One episode: a scene with 2-4 printed objects and an approach prompt referring to exactly one of them; the arm
starts in (a jittered copy of) the rest pose with the gripper closed; the scripted expert (approach.py) moves it
above the target and holds it there. Physics runs at 30 Hz with the v8 servo model (dead band, delay; ideal servos
in ~15 % of the episodes), so the arm sags as the real one; the labels are the expert's commands, which correct
for the sag from the joint readings near the goal.

Layout (as exp9, so that its tools apply):
  <out>/shard_03/episode_00012/left/000000.jpg, right/000000.jpg   rendered stereo pair (640x360)
  <out>/shard_03/episode_00012/data.npz   state, action (6 values, LeRobot units), phase (0 transit, 1 descend,
                                          2 hold), tip (world, m), contact (object, board), timestamp
  <out>/shard_03/episode_00012/meta.json  seed, instruction, task, objects, goal, camera, actuator, final errors,
                                          success
* An episode is determined by its seed; seed ranges of shards do not overlap.
* Episodes are written to a temporary folder and renamed when complete; re-running the same command resumes.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEED_STRIDE = 10_000_000          # seeds of shard k: base + k * SEED_STRIDE + i


def shard_progress(shard_dir):
    seeds = []
    for m in glob.glob(os.path.join(shard_dir, "episode_*", "meta.json")):
        seeds.append(json.load(open(m))["seed"])
    return len(seeds), (max(seeds) if seeds else None)


def scene_record(env, target):
    import tasks
    cam = env.stereo_cam
    objects = [{"label": o["descr"], "shape": o["shape"], "colour": o["color"], "lying": bool(o["spec"]["lying"]),
                "pos": [float(v) for v in env.object_pose(o["id"])[0]], "size": float(o["size"]),
                "volume_cm3": round(1e6 * tasks.volume(o["spec"]), 2), "target": o["id"] == target["id"]}
               for o in env.objects]
    camera = {c: cam[c] for c in ("distance_m", "height_m", "pitch_deg", "azimuth_deg", "yaw_deg", "roll_deg",
                                  "hfov_deg") if c in cam}
    return objects, camera


def run_episode(env, expert, rng, rest, render_dir=None, img_params=None):
    """Plays one planned episode. Writes the frames to render_dir (left/, right/) if given.
    Returns the recorded arrays and the final evaluation."""
    import cv2
    import approach as ap
    target = env.objects[0]
    obs = env.observe(render=render_dir is not None)
    rec = {k: [] for k in ("state", "action", "phase", "tip", "contact", "timestamp")}
    step = 0
    contact_any = [False, False]
    while not expert.done():
        action, phase = expert.act(obs)
        if render_dir is not None:
            for cam in ("left", "right"):
                p = os.path.join(render_dir, cam, f"{step:06d}.jpg")
                if not cv2.imwrite(p, cv2.cvtColor(obs["images"][cam], cv2.COLOR_RGB2BGR), img_params):
                    raise RuntimeError(f"failed to write {p} (disk full?)")
        c_obj, c_board = ap.arm_contact(env)
        if step < 15:
            c_board = False                       # the folded rest pose may hold the jaws on the board
        contact_any = [contact_any[0] or c_obj, contact_any[1] or c_board]
        rec["state"].append(np.asarray(obs["state"], float))
        rec["action"].append(action)
        rec["phase"].append(phase)
        rec["tip"].append(ap.tip_world(env))
        rec["contact"].append([c_obj, c_board])
        rec["timestamp"].append(step / env.cfg.control_hz)
        step += 1
        obs = env.step(action, render=(render_dir is not None and not expert.done()))
    err = ap.evaluate_pose(env, expert.goal)
    ok = ap.success(err, contact_any[0] or contact_any[1], float(obs["state"][5]))
    return rec, err, bool(ok), contact_any, step


def record_shard(shard, n_target, a):
    import cv2
    import approach as ap
    cv2.setNumThreads(1)
    shard_dir = os.path.join(a["out"], f"shard_{shard:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    for tmp in glob.glob(os.path.join(shard_dir, ".tmp_*")):
        shutil.rmtree(tmp, ignore_errors=True)
    done, last = shard_progress(shard_dir)
    seed = a["seed"] + shard * SEED_STRIDE if last is None else last + 1
    tag = f"[shard {shard:02d}]"
    if done >= n_target:
        print(f"{tag} already complete ({done} episodes)", flush=True)
        return
    cell = a["cell"]
    import common
    tk = ap.task_kwargs(",".join(str(v) for v in a["objects"]), a["white_weight"], a["weights"], a["shapes"])
    env = common.make_env(a["camera_scale"], cell=cell, task_kwargs=tk, tip_paint_mm=a["tip_paint_mm"],
                          real_background=not a["synthetic_scene"], sim=a["sim"])
    env.cfg.v8_no_actuator_p = a["ideal_servo_p"]
    rest = np.asarray(a["rest"], float)
    expert = ap.ApproachExpert(env, None, ap.ready_pose(env), long_hold_p=a["long_hold"])
    img_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    skipped = failed = moved = no_start = 0
    ws, ws_off = ap.load_workspace(a["setup"]) if a["random_start"] > 0 else (None, None)
    if a["random_start"] > 0 and ws is None:
        raise SystemExit("--random-start needs the setup's workspace.json")
    while done < n_target:
        ep_seed, seed = seed, seed + 1
        try:
            env.reset(seed=ep_seed, render=False)
        except RuntimeError:
            skipped += 1
            continue
        rng = np.random.default_rng([ep_seed, 12])
        expert.rng = rng
        kind = ap.start_kind(ep_seed, a["random_start"], a["near_start"])
        obs = ap.start_episode(env, kind, rest, ws, ws_off, rng)
        if obs is None:
            no_start += 1
            continue
        if not expert.reset(obs["state"][:5], env.objects[0], direct=(kind == "near")):
            skipped += 1
            continue
        q_start = [float(v) for v in obs["state"][:5]]
        target_start = env.object_pose(env.objects[0]["id"])[0]
        t0 = time.time()
        tmp_dir = os.path.join(shard_dir, f".tmp_episode_{done:05d}")
        for cam in ("left", "right"):
            os.makedirs(os.path.join(tmp_dir, cam), exist_ok=True)
        rec, err, ok, contact, n = run_episode(env, expert, rng, rest, tmp_dir, img_params)
        if ap.target_moved(env, env.objects[0], target_start):
            shutil.rmtree(tmp_dir)           # e.g. a lying cylinder rolled away: the labels would point where it was
            moved += 1
            print(f"{tag} seed {ep_seed}: the target moved during the episode; not kept", flush=True)
            continue
        if not ok and not a["keep_failures"]:
            shutil.rmtree(tmp_dir)
            failed += 1
            print(f"{tag} seed {ep_seed}: expert missed ({err['across_cm']:.1f} cm across, gap "
                  f"{err['gap_cm']:.1f} cm, {err['angle_deg']:.0f} deg, contact {contact}); not kept", flush=True)
            continue
        np.savez_compressed(os.path.join(tmp_dir, "data.npz"),
                            **{k: np.asarray(v, dtype=np.int8 if k in ("phase", "contact") else np.float32)
                               for k, v in rec.items()})
        objects, camera = scene_record(env, env.objects[0])
        g = expert.goal
        meta = {"shard": shard, "episode": done, "seed": ep_seed, "command": "approach", "sim": a["sim"],
                "start": kind, "q_start": q_start, "long_hold": expert.long_hold,
                "instruction": env.task["instruction"], "canonical": env.task.get("canonical"),
                "task_type": env.task.get("type"), "objects": objects, "camera": camera,
                "actuator": dict(env.act_params),
                "goal": {"q": [float(v) for v in g["q"]], "q_pre": [float(v) for v in g["q_pre"]],
                         "tip": [float(v) for v in g["tip"]], "centre": [float(v) for v in g["centre"]],
                         "direction": [float(v) for v in g["direction"]], "tilt_deg": g["tilt"],
                         "surface_m": g["surface"]},
                "final": err, "contact": contact, "success": ok, "num_steps": n, "fps": env.cfg.control_hz,
                "image_size": list(env.cfg.front_size), "cameras": ["left", "right"],
                "camera_scale": a["camera_scale"],
                "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                "units": "arm joints: degrees (simulator convention = real + setup offsets); gripper 0 closed..100 open",
                "phases": ap.ApproachExpert.PHASES}
        json.dump(meta, open(os.path.join(tmp_dir, "meta.json"), "w"), indent=1)
        os.rename(tmp_dir, os.path.join(shard_dir, f"episode_{done:05d}"))
        done += 1
        dt = time.time() - t0
        print(f"{tag} episode {done}/{n_target}: start {kind}{', long hold' if expert.long_hold else ''}, {n} steps, tilt {g['tilt']}, final {err['across_cm']:.1f} cm / "
              f"gap {err['gap_cm']:.1f} cm, '{env.task['instruction']}', {dt:.1f} s", flush=True)
    json.dump({"shard": shard, "episodes": done, "skipped_scenes_last_run": skipped,
               "expert_misses_last_run": failed, "target_moved_last_run": moved,
               "no_clear_start_last_run": no_start, "next_seed": seed},
              open(os.path.join(shard_dir, "shard_info.json"), "w"), indent=1)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", required=True)
    p.add_argument("--out", default="/media/inkoziev/corpora/tmp/lerobot_datasets/data_v12/raw")
    p.add_argument("--num", type=int, default=600, help="episodes in total over all shards of this run")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--shard-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-scale", type=float, default=1.0, help="v9 camera randomisation width (0 = calibrated)")
    p.add_argument("--objects", default="2,4", help="objects per scene, min,max")
    p.add_argument("--white-weight", type=float, default=0.5)
    p.add_argument("--shapes", default="", help="object shapes, e.g. cube,ball,cylinder (default: all)")
    p.add_argument("--weights", default="", help="task types, e.g. full=0.5,colour=0.25,shape=0.25 (default: tasks.py)")
    p.add_argument("--ideal-servo-p", type=float, default=0.15, help="share of episodes with ideal servos")
    p.add_argument("--random-start", type=float, default=0.0,
                   help="share of episodes starting from a random pose of the taught workspace instead of rest")
    p.add_argument("--near-start", type=float, default=0.0,
                   help="share of episodes starting close to the goal (0.5-3 cm off), corrected straight to it")
    p.add_argument("--long-hold", type=float, default=0.0,
                   help="share of episodes holding the goal for 2.0-2.4 s instead of 0.8-1.2 s")
    p.add_argument("--rest", default="", help="real rest pose, 5 angles in deg (default approach.REST_REAL)")
    p.add_argument("--tip-paint-mm", type=float, default=20.0)
    p.add_argument("--sim", default="v11", choices=["v11", "v14"],
                   help="simulator version: v11 = experiments 11-13; v14 = wrist camera on the arm, v14 arm look")
    p.add_argument("--synthetic-scene", action="store_true")
    p.add_argument("--keep-failures", action="store_true", help="also keep episodes the expert missed")
    args = p.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import approach as ap
    import common
    ap.task_kwargs(args.objects, args.white_weight, args.weights, args.shapes)   # check the combination early
    rest = ap.rest_pose(args.setup, [float(v) for v in args.rest.split(",")] if args.rest else None)
    cfg = vars(args).copy()
    cfg.update({"cell": common.cell_config(args.setup), "rest": rest.tolist(),
                "objects": [int(v) for v in args.objects.split(",")]})
    os.makedirs(args.out, exist_ok=True)
    json.dump({"args": vars(args), "rest_sim_deg": rest.tolist(), "cell": cfg["cell"],
               "approach": {"random_start": args.random_start, "near_start": args.near_start,
                         "long_hold": args.long_hold, "hold_long_s": ap.HOLD_LONG_S,
                         "near_across_m": ap.NEAR_ACROSS, "near_along_m": ap.NEAR_ALONG, "start_min_dist_m": ap.START_MIN_DIST,
                         "clearance_m": ap.CLEARANCE, "pre_back_m": ap.PRE_BACK, "tilts_deg": ap.TILTS_DEG,
                            "hold_s": ap.HOLD_S, "speed_deg_s": ap.SPEED_DEG_S, "descend_s": ap.DESCEND_S,
                            "comp_gain": ap.COMP_GAIN, "comp_limit_deg": ap.COMP_LIMIT,
                            "verbs": ap.VERBS, "superlative_verbs": ap.SUPERLATIVE_VERBS}},
              open(os.path.join(args.out, f"recording_{args.shard_offset:02d}.json"), "w"), indent=1, default=str)
    print(f"rest pose (simulator, deg): {np.round(rest, 1).tolist()}; camera scale {args.camera_scale}")
    w = args.workers
    quotas = [args.num // w + (1 if k < args.num % w else 0) for k in range(w)]
    shards = [args.shard_offset + k for k in range(w)]
    t0 = time.time()
    if w == 1:
        record_shard(shards[0], quotas[0], cfg)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=record_shard, args=(s, q, cfg)) for s, q in zip(shards, quotas) if q > 0]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        if any(pr.exitcode != 0 for pr in procs):
            print("warning: a worker exited with an error; re-run the same command to resume")
    total = frames = 0
    for s in shards:
        for m in glob.glob(os.path.join(args.out, f"shard_{s:02d}", "episode_*", "meta.json")):
            total += 1
            frames += json.load(open(m))["num_steps"]
    print(f"done: {total} episodes, {frames} frames ({frames / max(1, total):.0f} per episode), "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
