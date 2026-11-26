"""
Record episodes of the five atomic commands of experiment 14 (commands.py), in parallel shards.

    python dataset/record_commands.py --setup setups/<name> --out $DATA/data_v14a/raw --sim v14 \
        --num-per-command 100 --workers 8

One episode = one command. The commands that lead to its start (commands.CHAINS: e.g. approach -> grasp -> lift
before a "lower" of a held object) are played by the same expert without rendering; only the episode's own
command is rendered and stored. Its prompt comes from commands.instruction. Episodes whose expert fails (any
command of the chain, judged by commands.evaluate) are not kept.

Which command a seed records: pattern[seed % len(pattern)] (deterministic, so a re-run resumes), as long as that
command's quota in the shard is not full. The pattern interleaves the commands in proportion to their totals
(--num-per-command 100 for all, or per command: approach=2000,grasp=1500,lift=500,lower=500,release=500), so all
commands progress together; with equal totals it is the command order itself (= the earlier COMMANDS[seed % 5]). The chain variant comes from the seed's own random stream (commands.chain_for).

Layout as record_approach.py; data.npz "phase" indexes commands.PHASES; meta.json adds command, variant, chain,
holding_at_start, metrics (commands.evaluate), plan (the expert's parameters).
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_approach import SEED_STRIDE, scene_record  # noqa: E402

BOARD_GRACE_STEPS = 15            # the folded rest pose may hold the jaws on the board at the start


def parse_totals(spec, commands):
    """--num-per-command: one number for every command, or command=number pairs (missing commands: 0)."""
    spec = str(spec).strip()
    if "=" not in spec:
        return {c: int(spec) for c in commands}
    totals = {c: 0 for c in commands}
    for part in spec.split(","):
        k, v = part.split("=")
        k = k.strip()
        if k not in totals:
            raise SystemExit(f"--num-per-command: {k} is not one of --commands ({', '.join(commands)})")
        totals[k] = int(v)
    return totals


def command_pattern(totals, commands):
    """Seed -> command cycle: smooth weighted round robin over the totals reduced by their gcd."""
    from functools import reduce
    from math import gcd
    w = {c: totals[c] for c in commands if totals[c] > 0}
    g = reduce(gcd, w.values())
    w = {c: v // g for c, v in w.items()}
    cur, pattern = {c: 0 for c in w}, []
    for _ in range(sum(w.values())):
        for c in w:
            cur[c] += w[c]
        best = max(w, key=lambda c: cur[c])         # first in command order on ties
        cur[best] -= sum(w.values())
        pattern.append(best)
    return pattern


def shard_counts(shard_dir):
    counts, seeds = {}, []
    for m in glob.glob(os.path.join(shard_dir, "episode_*", "meta.json")):
        meta = json.load(open(m))
        counts[meta["command"]] = counts.get(meta["command"], 0) + 1
        seeds.append(meta["seed"])
    return counts, (max(seeds) if seeds else None)


def play(env, expert, command, render_dir=None, img_params=None, first=False):
    """Runs the planned command to its end. Writes frames to render_dir if given. Returns (record, contacts):
    contacts = (arm touched an object, gripper/camera touched the board) during the command."""
    import cv2
    import approach as ap
    import commands as cm
    target = env.objects[0]
    obs = env.observe(render=render_dir is not None)
    rec = {k: [] for k in ("state", "action", "phase", "tip", "contact", "timestamp")}
    c_obj_any = c_board_any = False
    step = 0
    while not expert.done():
        action, phase = expert.act(obs)
        if render_dir is not None:
            for cam in ("left", "right"):
                p = os.path.join(render_dir, cam, f"{step:06d}.jpg")
                if not cv2.imwrite(p, cv2.cvtColor(obs["images"][cam], cv2.COLOR_RGB2BGR), img_params):
                    raise RuntimeError(f"failed to write {p} (disk full?)")
        c_obj = ap.arm_contact(env)[0]
        c_board = cm.gripper_board_contact(env) and not (first and step < BOARD_GRACE_STEPS)
        c_obj_any, c_board_any = c_obj_any or c_obj, c_board_any or c_board
        rec["state"].append(np.asarray(obs["state"], float))
        rec["action"].append(action)
        rec["phase"].append(phase)
        rec["tip"].append(ap.tip_world(env))
        rec["contact"].append([c_obj, c_board])
        rec["timestamp"].append(step / env.cfg.control_hz)
        step += 1
        obs = env.step(action, render=(render_dir is not None and not expert.done()))
    return rec, (bool(c_obj_any), bool(c_board_any))


def record_shard(shard, quotas, a):
    import cv2
    import approach as ap
    import commands as cm
    import common
    cv2.setNumThreads(1)
    shard_dir = os.path.join(a["out"], f"shard_{shard:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    for tmp in glob.glob(os.path.join(shard_dir, ".tmp_*")):
        shutil.rmtree(tmp, ignore_errors=True)
    counts, last = shard_counts(shard_dir)
    seed = a["seed"] + shard * SEED_STRIDE if last is None else last + 1
    tag = f"[shard {shard:02d}]"
    left = lambda: {c: q - counts.get(c, 0) for c, q in quotas.items() if q - counts.get(c, 0) > 0}
    if not left():
        print(f"{tag} already complete ({counts})", flush=True)
        return
    tk = ap.task_kwargs(",".join(str(v) for v in a["objects"]), a["white_weight"], a["weights"], a["shapes"])
    env = common.make_env(a["camera_scale"], cell=a["cell"], task_kwargs=tk, tip_paint_mm=a["tip_paint_mm"],
                          real_background=not a["synthetic_scene"], sim=a["sim"])
    env.cfg.v8_no_actuator_p = a["ideal_servo_p"]
    rest = np.asarray(a["rest"], float)
    expert = cm.ChainExpert(env, None, ap.ready_pose(env), long_hold_p=a["long_hold"],
                            align_roll=not a["no_roll_align"])
    img_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ws, ws_off = ap.load_workspace(a["setup"]) if a["random_start"] > 0 else (None, None)
    if a["random_start"] > 0 and ws is None:
        raise SystemExit("--random-start needs the setup's workspace.json")
    stats = {"skipped_scenes": 0, "no_start": 0, "plan_failed": {}, "expert_failed": {}}
    done = sum(counts.values())
    while left():
        ep_seed, seed = seed, seed + 1
        command = a["pattern"][ep_seed % len(a["pattern"])]
        if command not in left():
            continue
        try:
            env.reset(seed=ep_seed, render=False)
        except RuntimeError:
            stats["skipped_scenes"] += 1
            continue
        rng = np.random.default_rng([ep_seed, 14])
        expert.rng = rng
        variant, prefix = cm.chain_for(command, ep_seed)
        chain = prefix + [command]
        kind = ap.start_kind(ep_seed, a["random_start"], a["near_start"] if command == "approach" else 0.0)
        obs = ap.start_episode(env, kind, rest, ws, ws_off, rng)
        if obs is None:
            stats["no_start"] += 1
            continue
        expert.start(obs["state"][:5])
        target = env.objects[0]
        q_start = [float(v) for v in obs["state"][:5]]
        failed_at, t0 = None, time.time()
        for i, cmd in enumerate(chain):
            last_cmd = i == len(chain) - 1
            if not expert.plan(cmd, target, prefix=not last_cmd, direct=(cmd == "approach" and kind == "near")):
                failed_at = ("plan", cmd)
                break
            start_info = {"tip": ap.tip_world(env).tolist(), "holding": cm.holding(env, target)}
            if last_cmd:
                tmp_dir = os.path.join(shard_dir, f".tmp_episode_{done:05d}")
                for cam in ("left", "right"):
                    os.makedirs(os.path.join(tmp_dir, cam), exist_ok=True)
                instruction = cm.instruction(env, cmd, start_info["holding"], rng)
                start_obs = env.observe(render=False)
                rec, (c_obj, c_board) = play(env, expert, cmd, tmp_dir, img_params, first=(i == 0))
            else:
                rec, (c_obj, c_board) = play(env, expert, cmd, first=(i == 0))
            ok, metrics = cm.evaluate(env, cmd, target, expert, start_info)
            ok = ok and not c_board and not (cmd == "approach" and c_obj)
            if not ok:
                failed_at = ("expert", cmd)
                break
        if failed_at is not None:
            if os.path.isdir(os.path.join(shard_dir, f".tmp_episode_{done:05d}")):
                shutil.rmtree(os.path.join(shard_dir, f".tmp_episode_{done:05d}"))
            key = "plan_failed" if failed_at[0] == "plan" else "expert_failed"
            stats[key][failed_at[1]] = stats[key].get(failed_at[1], 0) + 1
            print(f"{tag} seed {ep_seed}: {command} ({variant}) failed at {failed_at[1]} ({failed_at[0]}); "
                  f"not kept", flush=True)
            continue
        n = len(rec["action"])
        np.savez_compressed(os.path.join(tmp_dir, "data.npz"),
                            **{k: np.asarray(v, dtype=np.int8 if k in ("phase", "contact") else np.float32)
                               for k, v in rec.items()})
        objects, camera = scene_record(env, target)
        meta = {"shard": shard, "episode": done, "seed": ep_seed, "command": command, "variant": variant,
                "chain": chain, "sim": a["sim"], "start": kind, "q_start": q_start,
                "state_at_command": [float(v) for v in start_obs["state"]],
                "holding_at_start": start_info["holding"], "long_hold": expert.long_hold,
                "instruction": instruction, "canonical": env.task.get("canonical") if command == "approach" else None,
                "task_type": env.task.get("type"), "objects": objects, "camera": camera,
                "actuator": dict(env.act_params), "plan": {k: v for k, v in expert.info.items()},
                "metrics": metrics, "success": True, "num_steps": n, "fps": env.cfg.control_hz,
                "image_size": list(env.cfg.front_size), "cameras": ["left", "right"],
                "camera_scale": a["camera_scale"],
                "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                "units": "arm joints: degrees (simulator convention = real + setup offsets); gripper 0 closed..100 open",
                "phases": cm.PHASES}
        if command == "approach" and expert.goal is not None:
            g = expert.goal
            meta["goal"] = {"q": [float(v) for v in g["q"]], "tip": [float(v) for v in g["tip"]],
                            "centre": [float(v) for v in g["centre"]], "direction": [float(v) for v in g["direction"]],
                            "tilt_deg": g["tilt"], "surface_m": g["surface"]}
            meta["final"] = {k: metrics[k] for k in ("across_cm", "gap_cm", "angle_deg", "dist_cm") if k in metrics}
        json.dump(meta, open(os.path.join(tmp_dir, "meta.json"), "w"), indent=1, default=float)
        os.rename(tmp_dir, os.path.join(shard_dir, f"episode_{done:05d}"))
        done += 1
        counts[command] = counts.get(command, 0) + 1
        print(f"{tag} {command:8s} {counts[command]}/{quotas[command]} ({variant}, start {kind}): {n} steps, "
              f"'{instruction}', {time.time() - t0:.1f} s", flush=True)
    json.dump({"shard": shard, "counts": counts, "next_seed": seed, **stats},
              open(os.path.join(shard_dir, "shard_info.json"), "w"), indent=1)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--commands", default="approach,grasp,lift,lower,release")
    p.add_argument("--num-per-command", default="100",
                   help="episodes per command over all shards: one number, or approach=2000,grasp=1500,...")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--shard-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-scale", type=float, default=1.0)
    p.add_argument("--objects", default="1,1", help="objects per scene, min,max")
    p.add_argument("--white-weight", type=float, default=1 / 6)
    p.add_argument("--shapes", default="", help="default: all shapes of the object set (v14: cube,cylinder,hex_prism)")
    p.add_argument("--weights", default="full=0.5,colour=0.25,shape=0.25", help="task types of the prompts")
    p.add_argument("--ideal-servo-p", type=float, default=0.15)
    p.add_argument("--random-start", type=float, default=0.5, help="share of chains starting from a workspace pose")
    p.add_argument("--near-start", type=float, default=0.12, help="share of approach episodes starting near the goal")
    p.add_argument("--long-hold", type=float, default=0.4, help="share of episodes with the long final hold")
    p.add_argument("--no-roll-align", action="store_true", help="approach without turning the jaws onto the faces")
    p.add_argument("--rest", default="")
    p.add_argument("--tip-paint-mm", type=float, default=20.0)
    p.add_argument("--sim", default="v14", choices=["v11", "v14"])
    p.add_argument("--synthetic-scene", action="store_true")
    args = p.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import approach as ap
    import commands as cm
    import common
    commands = [c.strip() for c in args.commands.split(",") if c.strip()]
    bad = [c for c in commands if c not in cm.COMMANDS]
    if bad:
        raise SystemExit(f"unknown commands {bad}; known: {', '.join(cm.COMMANDS)}")
    ap.task_kwargs(args.objects, args.white_weight, args.weights, args.shapes)
    rest = ap.rest_pose(args.setup, [float(v) for v in args.rest.split(",")] if args.rest else None)
    totals = parse_totals(args.num_per_command, commands)
    if sum(totals.values()) == 0:
        raise SystemExit("--num-per-command: nothing to record")
    pattern = command_pattern(totals, commands)
    print(f"episodes per command: {totals}; seed pattern ({len(pattern)}): {' '.join(pattern[:20])}"
          f"{' ...' if len(pattern) > 20 else ''}", flush=True)
    cfg = vars(args).copy()
    cfg.update({"cell": common.cell_config(args.setup), "rest": rest.tolist(), "commands": commands, "pattern": pattern,
                "objects": [int(v) for v in args.objects.split(",")]})
    foreign = [m for m in glob.glob(os.path.join(args.out, "shard_*", "episode_*", "meta.json"))[:200]
               if "command" not in json.load(open(m))]
    if foreign:
        raise SystemExit(f"{args.out} holds episodes of another recorder (e.g. {foreign[0]}); "
                         f"choose a new --out folder")
    os.makedirs(args.out, exist_ok=True)
    consts = {k: getattr(cm, k) for k in dir(cm) if k.isupper()}
    json.dump({"args": vars(args), "rest_sim_deg": rest.tolist(), "cell": cfg["cell"], "commands": consts},
              open(os.path.join(args.out, f"recording_{args.shard_offset:02d}.json"), "w"), indent=1, default=str)
    w = args.workers
    shards = [args.shard_offset + k for k in range(w)]
    quotas = [{c: n // w + (1 if k < n % w else 0) for c, n in totals.items() if n > 0} for k in range(w)]
    t0 = time.time()
    if w == 1:
        record_shard(shards[0], quotas[0], cfg)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=record_shard, args=(s, q, cfg)) for s, q in zip(shards, quotas)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        if any(pr.exitcode != 0 for pr in procs):
            print("warning: a worker exited with an error; re-run the same command to resume")
    per, frames = {}, {}
    for s in shards:
        for m in glob.glob(os.path.join(args.out, f"shard_{s:02d}", "episode_*", "meta.json")):
            meta = json.load(open(m))
            per[meta["command"]] = per.get(meta["command"], 0) + 1
            frames[meta["command"]] = frames.get(meta["command"], 0) + meta["num_steps"]
    print(f"done in {(time.time() - t0) / 60:.1f} min:")
    for c in commands:
        print(f"  {c:8s} {per.get(c, 0):4d} episodes, {frames.get(c, 0):6d} frames "
              f"({frames.get(c, 0) / max(1, per.get(c, 0)):.0f} per episode)")


if __name__ == "__main__":
    main()
