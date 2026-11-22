"""
Look inside approach episodes (experiment 12).

1) closed: run the policy (or the expert) on chosen scenes and save, per scene,
     <out>/<tag>_<seed>.mp4   left|right images; on the left image: goal tip (green ring), actual tip (red dot),
                              target centre (yellow cross), other objects (blue crosses); prompt, step, errors
     <out>/<tag>_<seed>.png   the five joints over time: policy command, measured angle, and the expert's planned
                              trajectory for the same scene (dashed), chunk starts marked
     <out>/<tag>_<seed>.json  the arrays, the goal, and every contact (step, arm link, object, target or not)

   python tools/inspect_policy.py closed --setup $SETUP --policy <ckpt>/pretrained_model --seeds 920000000-920000005
   python tools/inspect_policy.py closed --setup $SETUP --policy expert --seeds 920000028,920000054   # expert collisions

2) openloop: feed recorded episodes (raw frames, as in training) to the policy, at every chunk start, and compare
   the predicted chunk with the expert's labels. Tells "did not learn" (large error on training episodes) from
   "does not generalise" (small on training, large on held-out) from a closed-loop problem (small on both).
     <out>/openloop_<shard>_<episode>.png   labels, predicted chunks, measured angles per joint
     printed: mean absolute error per joint, over the first 10 actions of each chunk

   python tools/inspect_policy.py openloop --policy <ckpt>/pretrained_model --raw data_v12/raw --episodes 6
   python tools/inspect_policy.py openloop --policy <ckpt>/pretrained_model --raw data_v12/raw --held-out \
       $DATA/so101_approach_v12/eval_seeds.json --episodes 6

Results go to outputs/inspect_<tag> unless --out is given.
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import approach as ap  # noqa: E402
import common  # noqa: E402
from evaluation.eval_approach import ExpertPolicy, LeRobotPolicy  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = (int(v) for v in part.split("-"))
            out += list(range(a, b + 1))
        elif part:
            out.append(int(part))
    return out


def project_left(env, p):
    """World point -> pixel in the left image (None if behind the camera)."""
    cam = env.stereo_cam
    w, h = env.cfg.front_size
    center, fwd, right, up = (np.array(cam[k]) for k in ("center", "fwd", "right", "up"))
    f = (h / 2) / math.tan(math.radians(cam["vfov_deg"]) / 2)
    d = np.asarray(p, float) - (center - right * cam["baseline"] / 2)
    z = float(d @ fwd)
    if z <= 0.01:
        return None
    u, v = f * float(d @ right) / z + w / 2, -f * float(d @ up) / z + h / 2
    x0, y0 = (env.crop[0], env.crop[1]) if getattr(env, "crop", None) else (0, 0)
    return int(round(u - x0)), int(round(v - y0))


def annotate(env, left, right, goal, lines):
    img = left.copy()
    for o in env.objects[1:]:
        q = project_left(env, env.object_pose(o["id"])[0])
        if q:
            cv2.drawMarker(img, q, (60, 120, 255), cv2.MARKER_CROSS, 10, 1)
    q = project_left(env, env.object_pose(env.objects[0]["id"])[0])
    if q:
        cv2.drawMarker(img, q, (255, 230, 0), cv2.MARKER_CROSS, 12, 2)
    q = project_left(env, goal["tip"])
    if q:
        cv2.circle(img, q, 6, (0, 220, 0), 2)
    q = project_left(env, ap.tip_world(env))
    if q:
        cv2.circle(img, q, 3, (255, 0, 0), -1)
    out = np.hstack([img, right])
    band = 20 * len(lines) + 6
    out[:band] = (out[:band] * 0.35).astype(np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(out, t, (8, 17 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def contacts_now(env):
    out = []
    names = {v: k for k, v in env.link_idx.items()}
    for k, o in enumerate(env.objects):
        for c in env.pb.getContactPoints(env.robot, o["id"]):
            out.append({"link": names.get(c[3], str(c[3])), "object": o["descr"], "target": k == 0})
    table = getattr(env, "table", None)
    if table is not None:
        for c in env.pb.getContactPoints(env.robot, table):
            if c[3] in (env.g_link, env.jaw_link):
                out.append({"link": names.get(c[3], str(c[3])), "object": "board", "target": False})
    return out


COLOURS_BGR = {"C0": (180, 119, 31), "C1": (14, 127, 255), "k": (0, 0, 0), "C2": (44, 160, 44)}


def plot_joints_cv(path, title, series, chunk_steps=None, hz=30):
    """The same plot drawn with OpenCV (no matplotlib needed)."""
    W, H, L, R, T = 1100, 190, 90, 20, 40
    img = np.full((T + 5 * H + 30, W, 3), 255, np.uint8)
    cv2.putText(img, title[:150], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    curves = []                                   # (name, colour, dashed, [(t array, values (n,5))])
    for name, arr, style in series:
        if arr is None or len(arr) == 0:
            continue
        col, dashed = COLOURS_BGR.get(style[:2].rstrip("-"), (0, 0, 0)), "--" in style
        if isinstance(arr, list) and isinstance(arr[0], tuple):
            curves.append((name, col, dashed, [((t0 + np.arange(len(v))) / hz, np.asarray(v)) for t0, v in arr]))
        else:
            a = np.asarray(arr)
            curves.append((name, col, dashed, [(np.arange(len(a)) / hz, a)]))
    t_max = max(float(t[-1]) for _, _, _, segs in curves for t, _ in segs) or 1.0
    for j in range(5):
        y0 = T + j * H
        vals = np.concatenate([v[:, j] for _, _, _, segs in curves for _, v in segs])
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(1.0, 0.08 * (hi - lo))
        lo, hi = lo - pad, hi + pad
        X = lambda t: int(L + (W - L - R) * t / t_max)
        Y = lambda v: int(y0 + H - 20 - (H - 30) * (v - lo) / (hi - lo))
        cv2.rectangle(img, (L, y0 + 10), (W - R, y0 + H - 20), (200, 200, 200), 1)
        for s in (chunk_steps or []):
            cv2.line(img, (X(s / hz), y0 + 10), (X(s / hz), y0 + H - 20), (235, 235, 235), 1)
        for v in (lo + pad, hi - pad):
            cv2.putText(img, f"{v:.0f}", (L - 45, Y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
        cv2.putText(img, JOINTS[j], (L + 6, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        for name, col, dashed, segs in curves:
            for t, v in segs:
                pts = np.array([[X(a), Y(b)] for a, b in zip(t, v[:, j])], np.int32)
                if dashed:
                    for k in range(0, len(pts) - 1, 4):
                        cv2.line(img, tuple(pts[k]), tuple(pts[min(k + 2, len(pts) - 1)]), col, 1, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
    x = L
    for name, col, dashed, _ in curves:
        cv2.line(img, (x, T + 5 * H + 12), (x + 25, T + 5 * H + 12), col, 2)
        cv2.putText(img, name, (x + 30, T + 5 * H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        x += 60 + 8 * len(name)
    cv2.putText(img, f"time 0..{t_max:.1f} s; grey lines: chunk starts", (x + 20, T + 5 * H + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def plot_joints(path, title, series, chunk_steps=None, hz=30):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return plot_joints_cv(path, title, series, chunk_steps, hz)
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for j, ax in enumerate(axes):
        for name, arr, style in series:
            if arr is None or len(arr) == 0:
                continue
            if isinstance(arr, list) and arr and isinstance(arr[0], tuple):      # chunks: [(t0, values), ...]
                for k, (t0, vals) in enumerate(arr):
                    t = (t0 + np.arange(len(vals))) / hz
                    ax.plot(t, vals[:, j], style, lw=1, label=name if k == 0 else None)
                continue
            a = np.asarray(arr)
            ax.plot(np.arange(len(a)) / hz, a[:, j], style, lw=1.2, label=name)
        for s in (chunk_steps or []):
            ax.axvline(s / hz, color="0.85", lw=0.6, zorder=0)
        ax.set_ylabel(JOINTS[j] + " (deg)", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_title(title, fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ closed loop
def closed(args):
    cell = common.cell_config(args.setup)
    rest = ap.rest_pose(args.setup, [float(v) for v in args.rest.split(",")] if args.rest else None)
    ws = ap.load_workspace(args.setup) if args.start == "workspace" else (None, None)
    tk = ap.task_kwargs(args.objects, args.white_weight, args.weights, args.shapes)
    env = common.make_env(args.camera_scale, cell=cell, task_kwargs=tk, tip_paint_mm=args.tip_paint_mm,
                          real_background=not args.synthetic_scene)
    if args.policy == "expert":
        policy, tag = ExpertPolicy(env), "expert"
    else:
        cmap = dict(item.split(":", 1) for item in args.camera_map.split(",") if item)
        policy, tag = LeRobotPolicy(args.policy, cmap, args.device, args.n_action_steps or None), "policy"
    tag += f"_T{args.camera_scale:g}" + {"workspace": "w", "near": "n"}.get(args.start, "")
    for seed in parse_seeds(args.seeds):
        try:
            env.reset(seed=seed, render=False)
        except RuntimeError:
            print(f"seed {seed}: no scene")
            continue
        rng = np.random.default_rng([seed, 12])
        obs = ap.start_episode(env, args.start, rest, ws[0], ws[1], rng)
        if obs is None:
            print(f"seed {seed}: no clear start")
            continue
        g = ap.goal(env, env.objects[0])
        if g is None:
            print(f"seed {seed}: goal unreachable")
            continue
        ref = ap.ApproachExpert(env, np.random.default_rng([seed, 12, 1]), ap.ready_pose(env))
        plan = [q for q, _ in ref.plan] if ref.reset(obs["state"][:5], env.objects[0],
                                                     direct=args.start == "near") else []
        obs = env.observe(render=True)
        task = env.task["instruction"]
        if hasattr(policy, "direct"):
            policy.direct = args.start == "near"
        policy.reset(obs, seed, np.random.default_rng([seed, 12, 1]))
        frames, states, actions, chunks, contacts = [], [], [], [], []
        for step in range(args.max_steps):
            action, new_chunk = policy.act(obs, task)
            if new_chunk:
                chunks.append(step)
            err = ap.evaluate_pose(env, g)
            frames.append(annotate(env, obs["images"]["left"], obs["images"]["right"], g, [
                f"seed {seed}  T{args.camera_scale:g}  {tag.split('_')[0]}: \"{task}\"",
                f"step {step}  t={step / env.cfg.control_hz:.2f}s  across {err['across_cm']:.1f} cm  "
                f"gap {err['gap_cm']:.1f} cm  angle {err['angle_deg']:.0f} deg" + ("  [new chunk]" if new_chunk else "")]))
            obs = env.step(action, render=True)
            states.append(np.asarray(obs["state"], float))
            actions.append(np.asarray(action, float))
            for c in contacts_now(env):
                if not any(c2["link"] == c["link"] and c2["object"] == c["object"] for c2 in contacts):
                    contacts.append({**c, "step": step})
        err = ap.evaluate_pose(env, g)
        stem = os.path.join(args.out, f"{tag}_{seed}")
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(stem + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), env.cfg.control_hz, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        plot_joints(stem + ".png", f"seed {seed}: \"{task}\" | final across {err['across_cm']:.1f} cm, gap "
                    f"{err['gap_cm']:.1f} cm", [("command", np.asarray(actions)[:, :5], "C0-"),
                                               ("measured", np.asarray(states)[:, :5], "C1-"),
                                               ("expert plan", np.asarray(plan)[:, :5] if plan else None, "k--")],
                    chunks, env.cfg.control_hz)
        json.dump({"seed": seed, "task": task, "camera_scale": args.camera_scale, "final": err,
                   "goal": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in g.items()},
                   "contacts": contacts, "state": np.asarray(states).tolist(), "action": np.asarray(actions).tolist(),
                   "expert_plan": np.asarray(plan).tolist(),
                   "objects": [{"label": o["descr"], "pos": env.object_pose(o["id"])[0].tolist()} for o in env.objects]},
                  open(stem + ".json", "w"), indent=1)
        cs = ", ".join(f"{c['link']}->{c['object']}@{c['step']}" for c in contacts) or "none"
        print(f"seed {seed}: across {err['across_cm']:.1f} cm, gap {err['gap_cm']:.1f} cm, angle "
              f"{err['angle_deg']:.0f} deg; contacts: {cs} -> {stem}.mp4/.png/.json", flush=True)
    env.close()


# ------------------------------------------------------------------------------------------------ open loop
def openloop(args):
    cmap = dict(item.split(":", 1) for item in args.camera_map.split(",") if item)
    policy = LeRobotPolicy(args.policy, cmap, args.device, args.n_action_steps or None)
    n = policy.n_action_steps
    metas = sorted(glob.glob(os.path.join(args.raw, "shard_*", "episode_*", "meta.json")))
    if args.held_out:
        held = set(json.load(open(args.held_out))["seeds"])
        metas = [m for m in metas if json.load(open(m))["seed"] in held]
    metas = metas[:args.episodes]
    all_err, all_phase, all_long = [], {}, []
    for m in metas:
        ep, meta = os.path.dirname(m), json.load(open(m))
        d = np.load(os.path.join(ep, "data.npz"))
        T, chunks, errs, by_phase = len(d["state"]), [], [], {}
        # where does the policy head from the rest pose? a long chunk (50 actions) at t=0 vs the label at 50
        long = None
        if hasattr(policy.policy.config, "chunk_size"):
            imgs = {c: cv2.cvtColor(cv2.imread(os.path.join(ep, c, "000000.jpg")), cv2.COLOR_BGR2RGB)
                    for c in ("left", "right")}
            obs = {"state": d["state"][0].astype(np.float32), "images": imgs}
            k = min(policy.policy.config.chunk_size, T)
            policy.policy.config.n_action_steps = k
            policy.n_action_steps = k
            policy.reset(obs, meta["seed"], None)
            long = np.array([policy.act(obs, meta["instruction"])[0] for _ in range(k)])
            policy.policy.config.n_action_steps = policy.n_action_steps = n
            policy.reset(obs, meta["seed"], None)
        for t0 in range(0, T, n):
            imgs = {c: cv2.cvtColor(cv2.imread(os.path.join(ep, c, f"{t0:06d}.jpg")), cv2.COLOR_BGR2RGB)
                    for c in ("left", "right")}
            obs = {"state": d["state"][t0].astype(np.float32), "images": imgs}
            policy.reset(obs, meta["seed"], None)
            pred = np.array([policy.act(obs, meta["instruction"])[0] for _ in range(n)])
            chunks.append((t0, pred[:, :5]))
            lab = d["action"][t0:t0 + n, :5]
            errs.append(np.abs(pred[:len(lab), :5] - lab).mean(0))
            ph = "start (first 1 s)" if t0 < 30 else ["transit", "descend", "hold"][int(d["phase"][t0])]
            by_phase.setdefault(ph, []).append(errs[-1])
        e = np.mean(errs, 0)
        all_err.append(e)
        for ph, v in by_phase.items():
            all_phase.setdefault(ph, []).append(np.mean(v, 0))
        name = f"{os.path.basename(os.path.dirname(ep))}_{os.path.basename(ep)}"
        plot_joints(os.path.join(args.out, f"openloop_{name}.png"),
                    f"{name} seed {meta['seed']}: \"{meta['instruction']}\" | open-loop MAE "
                    + " ".join(f"{v:.1f}" for v in e) + " deg",
                    [("label", d["action"][:, :5], "k-"), ("measured", d["state"][:, :5], "C1-"),
                     ("predicted chunks", chunks, "C0-")]
                    + ([("50-step chunk from rest", [(0, long[:, :5])], "C2-")] if long is not None else []),
                    [c[0] for c in chunks], meta["fps"])
        line = f"{name} seed {meta['seed']}: MAE (deg) " + " ".join(f"{v:5.1f}" for v in e)
        if long is not None:
            k = len(long) - 1
            dl = long[k, :5] - d["action"][k, :5]
            all_long.append(np.abs(dl))
            line += f" | from rest, step {k}: pan {dl[0]:+.1f} lift {dl[1]:+.1f} elbow {dl[2]:+.1f}"
        print(line + f" | start {meta.get('start', 'rest')} | '{meta['instruction']}'", flush=True)
    if all_err:
        print("\nmean over episodes (deg):  " + "  ".join(f"{j} {v:.1f}" for j, v in zip(JOINTS, np.mean(all_err, 0))))
        for ph in ("start (first 1 s)", "transit", "descend", "hold"):
            if ph in all_phase:
                print(f"  {ph:18s} " + "  ".join(f"{v:5.1f}" for v in np.mean(all_phase[ph], 0)))
        if all_long:
            print("  from rest, |predicted - label| at the end of a 50-step chunk: "
                  + "  ".join(f"{j} {v:.1f}" for j, v in zip(JOINTS, np.mean(all_long, 0))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("closed", "openloop"))
    p.add_argument("--policy", required=True, help='"expert" (closed only) or a pretrained_model folder')
    p.add_argument("--setup", default="")
    p.add_argument("--seeds", default="920000000-920000005", help="closed: e.g. 920000000-920000009,920000032")
    p.add_argument("--camera-scale", type=float, default=0.0, help="closed: 0 = T0, 1 = T1")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--start", choices=("rest", "workspace", "near"), default="rest", help="closed: start pose")
    p.add_argument("--raw", default="/media/inkoziev/corpora/tmp/lerobot_datasets/data_v12/raw", help="openloop: raw episodes")
    p.add_argument("--held-out", default="", help="openloop: eval_seeds.json -> only held-out episodes")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--camera-map", default="left:observation.images.camera1,right:observation.images.camera2")
    p.add_argument("--objects", default="2,4")
    p.add_argument("--white-weight", type=float, default=0.5)
    p.add_argument("--weights", default="", help="task types, as in the recording")
    p.add_argument("--shapes", default="", help="object shapes, as in the recording")
    p.add_argument("--rest", default="")
    p.add_argument("--tip-paint-mm", type=float, default=20.0)
    p.add_argument("--synthetic-scene", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    args = p.parse_args()
    if args.what == "closed" and not args.setup:
        raise SystemExit("closed needs --setup")
    if args.what == "openloop" and args.policy == "expert":
        raise SystemExit("openloop needs a checkpoint")
    args.out = args.out or os.path.join("outputs", "inspect_" + ("expert" if args.policy == "expert" else
                                                                 os.path.basename(os.path.dirname(os.path.dirname(
                                                                     os.path.dirname(args.policy.rstrip("/")))))))
    os.makedirs(args.out, exist_ok=True)
    closed(args) if args.what == "closed" else openloop(args)


if __name__ == "__main__":
    main()
