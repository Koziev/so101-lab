"""
Closed-loop evaluation of the five commands of experiment 14 in simulator v14.

    # the harness with the scripted expert (should be close to 100 %)
    python evaluation/eval_commands.py --setup setups/<name> --policy expert --scenes 20
    # a SmolVLA checkpoint, whole chunks
    python evaluation/eval_commands.py --setup setups/<name> \
        --policy outputs/train/smolvla_commands_v14a/checkpoints/last/pretrained_model --n-action-steps 50 \
        --scenes 30 --held-out $DATA/so101_commands_v14a/eval_seeds.json --results outputs/eval_v14a/results.json \
        --videos outputs/eval_v14a/videos

Every command is evaluated on the same --scenes new seeds (a range the recorder never uses), at T0 (calibrated
camera) and T1 (camera randomised as in training); --held-out adds the held-out recording seeds (T1), each
with its own command. A scene is prepared exactly as in the recorder (dataset/record_commands.py): the start
pose of the seed, then the chain of commands leading to the evaluated one (commands.CHAINS), played by the expert
without rendering. Then the policy gets the prompt (commands.instruction) and runs for the command's
--max-steps. Its end state is judged by commands.evaluate (the recorder's test of the expert), with no
contact of the gripper (or wrist camera) with the board and, for approach, of the arm with the object.

Also reported, per command:
  behaviour   what the arm did, from the start to the end of the episode, in this order of priority: "open"
              (gripper +25 %), "close" (gripper -25 %, or opened and closed again), "rise" (tip up > 4 cm),
              "descend" (down > 4 cm, < 3 cm sideways), "move" (tip > 3 cm sideways), else "still". Expected: approach -> move / descend, grasp -> close (after opening),
              lift -> rise, lower -> descend, release -> open. The table shows how well the policy tells the
              commands apart.
  settled     every joint moved < 0.5 deg per step over the last 10 steps.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import approach as ap  # noqa: E402
import commands as cm  # noqa: E402
import common  # noqa: E402

NEW_SEED_BASE = 920_000_000
BEHAVIOURS = ["move", "descend", "rise", "open", "close", "still"]
MAX_STEPS = "approach=150,grasp=160,lift=120,lower=140,release=75"
EXPECTED = {"approach": ("move", "descend"), "grasp": ("close",), "lift": ("rise",), "lower": ("descend",),
            "release": ("open",)}


def behaviour(tip0, tip1, g0, g1, g_max):
    """What the arm did over an episode (see the module docstring); gripper events first, then the tip."""
    dz = tip1[2] - tip0[2]
    lateral = float(np.linalg.norm(np.asarray(tip1[:2]) - np.asarray(tip0[:2])))
    if g1 - g0 > 25:
        return "open"
    if (g_max - g0 > 25 and g_max - g1 > 25) or g0 - g1 > 25:
        return "close"                      # opened and closed again (a grasp), or closed
    if dz > 0.04:
        return "rise"
    if dz < -0.04 and lateral < 0.03:
        return "descend"
    if lateral > 0.03 or dz < -0.04:
        return "move"
    return "still"


class ExpertCommand:
    """The recorder's expert, continuing its own plan for the evaluated command (then holding)."""

    def __init__(self, expert):
        self.expert, self.last = expert, None

    def needs_images(self):
        return False

    def reset(self, obs, seed, rng):
        return True

    def act(self, obs, task):
        if not self.expert.done():
            self.last, ph = self.expert.act(obs)
            self.phase = cm.PHASES[ph]
        else:
            self.phase = "after_plan"
        return self.last, False


def prepare(env, expert, command, seed, rest, ws, args):
    """The recorder's start of `command` for this seed; None if the scene or the chain fails."""
    try:
        env.reset(seed=seed, render=False)
    except RuntimeError:
        return None
    rng = np.random.default_rng([seed, 14])
    expert.rng = rng
    variant, prefix = cm.chain_for(command, seed)
    kind = ap.start_kind(seed, args.random_start, args.near_start if command == "approach" else 0.0)
    obs = ap.start_episode(env, kind, rest, ws[0], ws[1], rng)
    if obs is None:
        return None
    expert.start(obs["state"][:5])
    target = env.objects[0]
    from dataset.record_commands import play
    for i, cmd in enumerate(prefix):
        if not expert.plan(cmd, target, prefix=True, direct=(cmd == "approach" and kind == "near")):
            return None
        start_info = {"tip": ap.tip_world(env).tolist(), "holding": cm.holding(env, target)}
        _, (c_obj, c_board) = play(env, expert, cmd, first=(i == 0))
        ok, _ = cm.evaluate(env, cmd, target, expert, start_info)
        if not ok or c_board or (cmd == "approach" and c_obj):
            return None
    # plan the evaluated command too: the expert policy uses it, and the prompt draws from the same stream
    if not expert.plan(command, target, prefix=False, direct=(command == "approach" and kind == "near")):
        return None
    start_info = {"tip": ap.tip_world(env).tolist(), "holding": cm.holding(env, target)}
    task = cm.instruction(env, command, start_info["holding"], rng)
    return {"variant": variant, "start": kind, "start_info": start_info, "task": task, "first": not prefix}


def log_contacts(env, log, step, phase, names):
    """Adds this step's contacts of the arm's links with the board and the objects to `log`:
    {"board"|"object": {link: {"first_step", "first_phase", "steps", "max_depth_mm", "min_dist_mm",
    "pen_steps", "first_pen_step"}}}. PyBullet also reports manifold points with a POSITIVE contactDistance
    (separated, within the contact breaking threshold), so min_dist_mm keeps the signed smallest distance and
    pen_steps counts only the steps with a real penetration (contactDistance < 0)."""
    pb = env.pb
    pairs = [("board", c) for c in pb.getContactPoints(env.robot, env.table)]
    pairs += [("object", c) for o in env.objects for c in pb.getContactPoints(env.robot, o["id"])]
    seen, pen = set(), set()
    for kind, c in pairs:
        name = names.get(c[3], str(c[3]))
        dist = float(c[8])
        d = log[kind].setdefault(name, {"first_step": step, "first_phase": phase, "steps": 0, "max_depth_mm": 0.0,
                                        "min_dist_mm": 1e9, "pen_steps": 0, "first_pen_step": None})
        d["max_depth_mm"] = max(d["max_depth_mm"], -1000.0 * dist)
        d["min_dist_mm"] = min(d["min_dist_mm"], 1000.0 * dist)
        if (kind, name) not in seen:
            d["steps"] += 1
            seen.add((kind, name))
        if dist < 0 and (kind, name) not in pen:
            d["pen_steps"] += 1
            pen.add((kind, name))
            if d["first_pen_step"] is None:
                d["first_pen_step"] = step


def log_clearance(env, clear, phase, target):
    """Smallest PHYSICAL clearance (mm) per phase: gripper links to the board, any link to the target object
    (negative = penetration; capped at the probe range)."""
    pb, rng = env.pb, ap.PROBE_RANGE
    links = ap.gripper_links(env)
    board = min([float(c[8]) for l in links for c in pb.getClosestPoints(env.robot, env.table, rng, linkIndexA=l)]
                + [rng])
    obj = min([float(c[8]) for c in pb.getClosestPoints(env.robot, target["id"], rng)] + [rng])
    d = clear.setdefault(str(phase), {"board": 1e9, "target": 1e9})
    d["board"] = round(min(d["board"], 1000 * board), 2)
    d["target"] = round(min(d["target"], 1000 * obj), 2)


def log_tracking(env, track, policy, phase):
    """Expert only: largest distance (mm) per phase between the physical tip and the tip of the expert's nominal
    pose this step, and the largest drop of the physical tip below it (positive = lower than planned)."""
    q_nom = getattr(getattr(policy, "expert", None), "q_nom", None)
    if q_nom is None:
        return
    nom = ap.tip_of(env, q_nom)
    phys = ap.tip_world(env)
    d = track.setdefault(str(phase), {"dist": 0.0, "drop": -1e9})
    d["dist"] = round(max(d["dist"], 1000 * float(np.linalg.norm(phys - nom))), 2)
    d["drop"] = round(max(d["drop"], 1000 * float(nom[2] - phys[2])), 2)


def _physics(env):
    try:
        p = env.pb.getPhysicsEngineParameters()
        return {k: float(p[k]) for k in ("contactBreakingThreshold", "contactSlop", "allowedCcdPenetration")
                if k in p}
    except Exception:
        return {}


def run(env, policy, expert, command, seed, prep, max_steps, video=None):
    target = env.objects[0]
    obs = env.observe(render=True)
    images = obs["images"]
    if not policy.reset(obs, seed, np.random.default_rng([seed, 14, 1])):
        return None
    tip0 = np.asarray(prep["start_info"]["tip"])
    g0 = g_max = float(obs["state"][5])
    states, lat = [], []
    c_obj_any = c_board_any = False
    names = {i: n for n, i in env.link_idx.items()}
    names[-1] = "base"
    log = {"board": {}, "object": {}}
    clear = {}
    track = {}
    for step in range(max_steps):
        if video is not None:
            video.append(np.concatenate([obs["images"]["left"], obs["images"]["right"]], axis=1))
        t0 = time.perf_counter()
        action, new_chunk = policy.act(obs, prep["task"])
        if new_chunk:
            lat.append(1000 * (time.perf_counter() - t0))
        obs = env.step(action, render=False)
        if video is not None or policy.needs_images():
            obs = env.observe(render=True)
            images = obs["images"]
        else:
            obs["images"] = images
        states.append(np.asarray(obs["state"], float))
        g_max = max(g_max, float(obs["state"][5]))
        c_obj = ap.arm_contact(env)[0]
        c_board = cm.gripper_board_contact(env) and not (prep["first"] and step < 15)
        c_obj_any, c_board_any = c_obj_any or c_obj, c_board_any or c_board
        log_contacts(env, log, step, getattr(policy, "phase", None), names)
        log_clearance(env, clear, getattr(policy, "phase", None), target)
        log_tracking(env, track, policy, getattr(policy, "phase", None))
    ok, metrics = cm.evaluate(env, command, target, expert, prep["start_info"])
    ok = ok and not c_board_any and not (command == "approach" and c_obj_any)
    S = np.asarray(states)
    settled = bool(np.max(np.abs(np.diff(S[-10:, :5], axis=0))) < 0.5) if len(S) > 10 else False
    tip1 = ap.tip_world(env)
    g1 = float(obs["state"][5])
    return {"seed": seed, "command": command, "variant": prep["variant"], "start": prep["start"],
            "task": prep["task"], "target": target["descr"], "success": bool(ok),
            "behaviour": behaviour(tip0, tip1, g0, g1, g_max), "settled": settled,
            "contact_object": bool(c_obj_any), "contact_board": bool(c_board_any), "contacts": log, "clear_mm": clear, "track_mm": track,
            "physics": _physics(env),
            "plan": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in expert.info.items()
                     if isinstance(v, (str, bool, int, float, np.floating))},
            "dz_cm": float(100 * (tip1[2] - tip0[2])),
            "lateral_cm": float(100 * np.linalg.norm(tip1[:2] - tip0[:2])),
            "gripper_start": g0, "gripper_end": g1, "chunk_ms": float(np.median(lat)) if lat else None,
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
               for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))}}


def summarize(rs, label, commands):
    print(f"\n=== {label}")
    print(f"{'command':9s} {'n':>4s} {'success':>8s} {'settled':>8s}   " + " ".join(f"{b:>7s}" for b in BEHAVIOURS))
    out = {}
    for c in commands:
        x = [r for r in rs if r["command"] == c]
        if not x:
            continue
        beh = {b: sum(r["behaviour"] == b for r in x) for b in BEHAVIOURS}
        out[c] = {"episodes": len(x), "success": sum(r["success"] for r in x),
                  "rate": sum(r["success"] for r in x) / len(x), "settled": sum(r["settled"] for r in x),
                  "behaviour": beh,
                  "expected_behaviour": sum(r["behaviour"] in EXPECTED[c] for r in x)}
        for v in sorted({r["variant"] for r in x}):
            xv = [r for r in x if r["variant"] == v]
            out[c][f"variant_{v}"] = {"episodes": len(xv), "success": sum(r["success"] for r in xv)}
        print(f"{c:9s} {len(x):4d} {out[c]['success']:4d} ({100 * out[c]['rate']:3.0f}%) {out[c]['settled']:7d}   "
              + " ".join(f"{beh[b]:7d}" for b in BEHAVIOURS))
        out[c]["contact_links"] = contact_links(x)
    print("\ncontacts of the arm's links (episodes; median of the first phase; max depth mm):")
    for c in commands:
        if c in out:
            for kind in ("board", "object"):
                for name, v in out[c]["contact_links"][kind].items():
                    print(f"  {c:9s} {kind:6s} {name:28s} {v['episodes']:3d} ep, first phase {v['phase']}, "
                          f"first step median {v['first_step']:.0f}, depth up to {v['max_depth_mm']:.1f} mm")
    return out


def contact_links(rs):
    out = {"board": {}, "object": {}}
    for kind in out:
        per = {}
        for r in rs:
            for name, d in r.get("contacts", {}).get(kind, {}).items():
                per.setdefault(name, []).append(d)
        for name, ds in per.items():
            phases = [d["first_phase"] for d in ds if d["first_phase"]]
            out[kind][name] = {"episodes": len(ds),
                               "phase": max(set(phases), key=phases.count) if phases else "-",
                               "first_step": float(np.median([d["first_step"] for d in ds])),
                               "max_depth_mm": float(max(d["max_depth_mm"] for d in ds))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", required=True)
    p.add_argument("--policy", required=True, help='"expert" or a LeRobot checkpoint (pretrained_model folder)')
    p.add_argument("--commands", default=",".join(cm.COMMANDS))
    p.add_argument("--scenes", type=int, default=30, help="new scenes per command and split")
    p.add_argument("--splits", default="T0,T1")
    p.add_argument("--held-out", default="", help="eval_seeds.json of the exported dataset (run at T1)")
    p.add_argument("--max-steps", default=MAX_STEPS, help="control steps per command, e.g. " + MAX_STEPS)
    p.add_argument("--n-action-steps", type=int, default=50, help="actions executed per chunk")
    p.add_argument("--camera-map", default="left:observation.images.camera1,right:observation.images.camera2")
    p.add_argument("--random-start", type=float, default=0.5, help="as in the recording")
    p.add_argument("--near-start", type=float, default=0.12, help="as in the recording")
    p.add_argument("--objects", default="1,1")
    p.add_argument("--white-weight", type=float, default=1 / 6)
    p.add_argument("--weights", default="full=0.5,colour=0.25,shape=0.25")
    p.add_argument("--shapes", default="")
    p.add_argument("--rest", default="")
    p.add_argument("--tip-paint-mm", type=float, default=20.0)
    p.add_argument("--sim", default="v14", choices=["v11", "v14"])
    p.add_argument("--no-roll-align", action="store_true", help="as in the recording")
    p.add_argument("--device", default="cuda")
    p.add_argument("--results", default="")
    p.add_argument("--videos", default="")
    p.add_argument("--n-videos", type=int, default=3, help="per command and split")
    args = p.parse_args()

    commands = [c for c in args.commands.split(",") if c]
    max_steps = {c: int(v) for c, v in (kv.split("=") for kv in args.max_steps.split(","))}
    cell = common.cell_config(args.setup)
    rest = ap.rest_pose(args.setup, [float(v) for v in args.rest.split(",")] if args.rest else None)
    tk = ap.task_kwargs(args.objects, args.white_weight, args.weights, args.shapes)
    ws = ap.load_workspace(args.setup) if args.random_start > 0 else (None, None)
    runs = [(s, float(s[1:]), [(c, NEW_SEED_BASE + i) for c in commands for i in range(args.scenes)])
            for s in args.splits.split(",") if s]
    if args.held_out:
        h = json.load(open(args.held_out))
        eps = h.get("episodes") or []
        runs.append(("held_out", 1.0, [(e["command"], e["seed"]) for e in eps if e.get("command") in commands]))
    policy, results = None, []
    summary = {"policy": args.policy, "n_action_steps": args.n_action_steps, "max_steps": max_steps}
    if args.videos:
        os.makedirs(args.videos, exist_ok=True)
    for split, scale, items in runs:
        env = common.make_env(scale, cell=cell, task_kwargs=tk, tip_paint_mm=args.tip_paint_mm, sim=args.sim)
        expert = cm.ChainExpert(env, None, ap.ready_pose(env), long_hold_p=0.0, align_roll=not args.no_roll_align)
        if args.policy == "expert":
            policy = ExpertCommand(expert)
        elif policy is None:
            from evaluation.eval_approach import LeRobotPolicy
            cmap = dict(item.split(":", 1) for item in args.camera_map.split(",") if item)
            policy = LeRobotPolicy(args.policy, cmap, args.device, args.n_action_steps or None)
        rs, n_vid = [], {}
        for i, (command, seed) in enumerate(items):
            prep = prepare(env, expert, command, seed, rest, ws, args)
            if prep is None:
                print(f"[{split} {i + 1}/{len(items)}] {command} seed {seed}: no valid start, skipped")
                continue
            if isinstance(policy, ExpertCommand):
                policy.expert = expert
            frames = [] if args.videos and n_vid.get(command, 0) < args.n_videos else None
            r = run(env, policy, expert, command, seed, prep, max_steps[command], frames)
            if r is None:
                continue
            r["split"] = split
            rs.append(r)
            print(f"[{split} {i + 1}/{len(items)}] {command:8s} seed {seed}: {'ok  ' if r['success'] else 'FAIL'} "
                  f"{r['behaviour']:7s} dz {r['dz_cm']:+5.1f} cm, gripper {r['gripper_start']:.0f}->"
                  f"{r['gripper_end']:.0f} | '{r['task']}'", flush=True)
            if frames:
                n_vid[command] = n_vid.get(command, 0) + 1
                out = os.path.join(args.videos, f"{split}_{command}_{seed}_{'ok' if r['success'] else 'fail'}.mp4")
                hh, ww = frames[0].shape[:2]
                vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), env.cfg.control_hz, (ww, hh))
                for f in frames:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
        env.close()
        results += rs
        summary[split] = summarize(rs, f"{split} (camera scale {scale:g})", commands)
    if args.results:
        os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
        json.dump({"summary": summary, "episodes": results}, open(args.results, "w"), indent=1, default=float)
        print("wrote", args.results)


if __name__ == "__main__":
    main()
