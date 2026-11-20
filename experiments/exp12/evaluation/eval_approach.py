"""
Closed-loop evaluation of the "Approach the object" command in the simulator (experiment 12).

    # the harness with the scripted expert (should be ~100 %)
    python evaluation/eval_approach.py --setup setups/<name> --policy expert --scenes 50

    # a fine-tuned SmolVLA checkpoint
    python evaluation/eval_approach.py --setup setups/<name> \
        --policy outputs/train/smolvla_approach_v12a/checkpoints/last/pretrained_model \
        --scenes 100 --results results_v12a.json --videos videos_v12a --n-videos 10

Scenes: --scenes new seeds (a range the recorder never uses), run once per split: T0 = calibrated camera, T1 = camera
randomised as in training (scale 1), both from the rest pose; T1w = as T1 but starting from a random pose of the
taught workspace (gripper >= 5 cm from the target, as in the recording). --held-out also runs the held-out seeds of the exported dataset (T1).
Every scene starts as in the recorder: jittered rest pose, gripper closed, the same servo model.
The policy runs for --max-steps control steps (30 Hz); the outcome is judged at the last step:
  success      tip within 1 cm of the goal across the pointing direction, 0.5-2.5 cm from the surface along
               it, gripper axis within 15 deg of the goal direction, jaws closed, no contact on the way
  failures, in this order: collision (arm touched an object or the gripper the board), wrong_object (the tip
  ended closer to another object), off_target (> 1 cm across), too_low / too_high, bad_angle, opened_jaws
Also reported: errors at the end (median, 90th percentile), whether the arm had come to rest (last 10 steps),
the step at which the goal tolerance was first reached, and rates by task type, tilt and object distance.
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
import common  # noqa: E402

NEW_SEED_BASE = 920_000_000
OUTCOMES = ["success", "collision", "wrong_object", "off_target", "too_low", "too_high", "bad_angle", "opened_jaws"]


class ExpertPolicy:
    def __init__(self, env):
        self.env = env
        self.expert = ap.ApproachExpert(env, None, ap.ready_pose(env))

    def needs_images(self):
        return False

    direct = False      # set by run_scene for near-goal starts

    def reset(self, obs, seed, rng):
        self.expert.rng = rng
        self.last = None
        return self.expert.reset(obs["state"][:5], self.env.objects[0], direct=self.direct)

    def act(self, obs, task):
        if self.expert.done():
            return self.last, False
        self.last, _ = self.expert.act(obs)
        return self.last, False


class LeRobotPolicy:
    """A LeRobot checkpoint with LeRobot's own pre- and post-processing (as exp9)."""

    def __init__(self, path, camera_map, device="cuda", n_action_steps=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        self.torch, self.device = torch, torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(self.device).eval()
        if n_action_steps:
            self.policy.config.n_action_steps = n_action_steps
        try:
            self.pre, self.post = make_pre_post_processors(
                self.policy.config, pretrained_path=path,
                preprocessor_overrides={"device_processor": {"device": device}})
        except TypeError:
            self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=path)
        expected = [k for k in self.policy.config.input_features if "image" in k and "empty_camera" not in k]
        self.cam_keys = {}
        for cam, mapped in camera_map.items():
            if f"observation.images.{cam}" in expected:
                self.cam_keys[cam] = f"observation.images.{cam}"
            elif mapped in expected:
                self.cam_keys[cam] = mapped
        missing = [k for k in expected if k not in self.cam_keys.values()]
        if len(missing) > int(getattr(self.policy.config, "empty_cameras", 0) or 0):
            raise SystemExit(f"policy inputs {missing} are not mapped (--camera-map {camera_map})")
        print("camera mapping:", self.cam_keys, "| left empty:", missing)
        self.n_action_steps = self.policy.config.n_action_steps
        self.t = 0

    def needs_images(self):
        """select_action looks at the images only when its action queue is empty (every n_action_steps)."""
        return self.t % self.n_action_steps == 0

    def reset(self, obs, seed, rng):
        self.torch.manual_seed(seed % 2 ** 32)
        self.torch.cuda.manual_seed_all(seed % 2 ** 32)
        self.policy.reset()
        for proc in (self.pre, self.post):
            if hasattr(proc, "reset"):
                proc.reset()
        self.t = 0
        return True

    def act(self, obs, task):
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(np.asarray(obs["state"], np.float32)).unsqueeze(0)}
        for cam, key in self.cam_keys.items():
            img = torch.from_numpy(np.ascontiguousarray(obs["images"][cam])).float().div(255).permute(2, 0, 1)
            batch[key] = img.unsqueeze(0)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["task"] = task
        batch["robot_type"] = "so101_follower"
        new_chunk = self.t % self.n_action_steps == 0
        self.t += 1
        with torch.inference_mode():
            a = self.post(self.policy.select_action(self.pre(batch)))
        return a.squeeze(0).float().cpu().numpy(), new_chunk


def classify(ok, err, contact, wrong, grip):
    if ok:
        return "success"
    if contact:
        return "collision"
    if wrong:
        return "wrong_object"
    if err["across_cm"] > 100 * ap.TOL_XY:
        return "off_target"
    if err["gap_cm"] < 100 * ap.TOL_GAP[0]:
        return "too_low"
    if err["gap_cm"] > 100 * ap.TOL_GAP[1]:
        return "too_high"
    if err["angle_deg"] > ap.TOL_ANGLE:
        return "bad_angle"
    return "opened_jaws" if grip > 15 else "off_target"


def nearest_object(env, tip):
    d = [float(np.linalg.norm(np.asarray(env.object_pose(o["id"])[0][:2]) - tip[:2])) for o in env.objects]
    return int(np.argmin(d)), d


def run_scene(env, policy, seed, rest, args, video=None, start="rest", ws=(None, None)):
    try:
        env.reset(seed=seed, render=False)
    except RuntimeError:
        return None
    rng = np.random.default_rng([seed, 12])
    kind = ap.start_kind(seed, args.random_start, args.near_start) if start == "per_seed" else start
    if hasattr(policy, "direct"):
        policy.direct = kind == "near"
    if ap.start_episode(env, kind, rest, ws[0], ws[1], rng) is None:
        return None
    target = env.objects[0]
    g = ap.goal(env, target)
    if g is None:
        return None
    obs = env.observe(render=True)          # images for the first call (kept below when not re-rendered)
    images = obs["images"]
    task = env.task["instruction"]
    if not policy.reset(obs, seed, np.random.default_rng([seed, 12, 1])):
        return None
    contact = [False, False]
    states, t_reach, lat = [], None, []
    for step in range(args.max_steps):
        if video is not None:
            video.append(np.concatenate([obs["images"]["left"], obs["images"]["right"]], axis=1))
        t0 = time.perf_counter()
        action, new_chunk = policy.act(obs, task)
        if new_chunk:
            lat.append(1000 * (time.perf_counter() - t0))
        # Render only the frames that are used: by the policy's next call that computes a chunk, or by a video.
        # Between chunks, the last rendered images are passed again (the policy ignores them).
        obs = env.step(action, render=False)
        if video is not None or policy.needs_images():
            obs = env.observe(render=True)
            images = obs["images"]
        else:
            obs["images"] = images
        states.append(np.asarray(obs["state"], float))
        c_obj, c_board = ap.arm_contact(env)
        contact = [contact[0] or c_obj, contact[1] or (c_board and step >= 15)]
        if t_reach is None and step % 3 == 0:
            e = ap.evaluate_pose(env, g)
            if ap.success(e, False, float(obs["state"][5])):
                t_reach = step
    err = ap.evaluate_pose(env, g)
    grip = float(obs["state"][5])
    k, dists = nearest_object(env, np.asarray(err["tip"]))
    wrong = k != 0 and dists[k] < dists[0] - 0.01
    any_contact = contact[0] or contact[1]
    ok = ap.success(err, any_contact, grip)
    S = np.asarray(states)
    settled = bool(np.max(np.abs(np.diff(S[-10:, :5], axis=0))) < 0.5) if len(S) > 10 else False
    pos = np.asarray(env.object_pose(target["id"])[0])
    rel = pos[:2] - env.pan_axis[:2]
    return {"seed": seed, "start": kind, "task": task, "task_type": env.task.get("type"), "target": target["descr"],
            "n_objects": len(env.objects), "radius_cm": float(100 * np.linalg.norm(rel)), "tilt": g["tilt"],
            "success": bool(ok), "outcome": classify(ok, err, any_contact, wrong, grip),
            "contact_object": contact[0], "contact_board": contact[1], "wrong_object": bool(wrong),
            "settled": settled, "step_reached": t_reach, "gripper_pct": grip,
            "chunk_ms": float(np.median(lat)) if lat else None,
            **{k: err[k] for k in ("across_cm", "gap_cm", "angle_deg", "dist_cm")}}


def summarize(rs, label):
    n = len(rs)
    if not n:
        return {}
    ok = sum(r["success"] for r in rs)
    out = {"episodes": n, "success": ok, "rate": ok / n,
           "outcomes": {c: sum(r["outcome"] == c for r in rs) for c in OUTCOMES},
           "settled": sum(r["settled"] for r in rs)}
    for k in ("across_cm", "gap_cm", "angle_deg", "dist_cm"):
        v = np.array([r[k] for r in rs])
        out[k] = {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
    print(f"\n== {label}: {ok}/{n} = {100 * ok / n:.1f} % ==")
    print("  outcomes: " + ", ".join(f"{c} {v}" for c, v in out["outcomes"].items() if v))
    print(f"  final errors, median (p90): across {out['across_cm']['median']:.2f} ({out['across_cm']['p90']:.2f}) cm,"
          f" gap {out['gap_cm']['median']:.2f} ({out['gap_cm']['p90']:.2f}) cm,"
          f" angle {out['angle_deg']['median']:.1f} ({out['angle_deg']['p90']:.1f}) deg; at rest {out['settled']}/{n}")
    reach = [r["step_reached"] for r in rs if r["step_reached"] is not None]
    if reach:
        print(f"  goal first reached: median step {np.median(reach):.0f} ({len(reach)} episodes)")
    for key, name in (("task_type", "task type"), ("tilt", "tilt (deg)")):
        cells = []
        for v in sorted({r[key] for r in rs}, key=str):
            sub = [r for r in rs if r[key] == v]
            cells.append(f"{v}: {sum(r['success'] for r in sub)}/{len(sub)}")
        print(f"  {name:12s} " + "   ".join(cells))
    cells = []
    for lo, hi in ((12, 18), (18, 24), (24, 31)):
        sub = [r for r in rs if lo <= r["radius_cm"] < hi]
        if sub:
            cells.append(f"{lo}-{hi}: {sum(r['success'] for r in sub)}/{len(sub)}")
    print(f"  {'distance cm':12s} " + "   ".join(cells))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", required=True)
    p.add_argument("--policy", required=True, help='"expert" or a LeRobot checkpoint (pretrained_model folder)')
    p.add_argument("--scenes", type=int, default=100, help="new scenes, each at T0 and T1")
    p.add_argument("--splits", default="T0,T1",
                   help='e.g. "T0,T1,T1w,T1n" (w = random workspace start, n = start near the goal)')
    p.add_argument("--near-start", type=float, default=0.0, help="as in the recording (held-out starts)")
    p.add_argument("--random-start", type=float, default=0.0,
                   help="as in the recording; decides the start of each held-out seed")
    p.add_argument("--held-out", default="", help="eval_seeds.json of the exported dataset (run at T1)")
    p.add_argument("--max-steps", type=int, default=150, help="control steps (30 Hz) per episode")
    p.add_argument("--n-action-steps", type=int, default=10, help="actions executed per chunk (0 = checkpoint's)")
    p.add_argument("--camera-map", default="left:observation.images.camera1,right:observation.images.camera2")
    p.add_argument("--objects", default="2,4")
    p.add_argument("--white-weight", type=float, default=0.5)
    p.add_argument("--weights", default="", help="task types, as in the recording")
    p.add_argument("--shapes", default="", help="object shapes, as in the recording")
    p.add_argument("--rest", default="")
    p.add_argument("--tip-paint-mm", type=float, default=20.0)
    p.add_argument("--synthetic-scene", action="store_true")
    p.add_argument("--no-actuator", action="store_true", help="ideal servos (same scenes)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--results", default="")
    p.add_argument("--videos", default="")
    p.add_argument("--n-videos", type=int, default=10)
    args = p.parse_args()

    cell = common.cell_config(args.setup)
    rest = ap.rest_pose(args.setup, [float(v) for v in args.rest.split(",")] if args.rest else None)
    tk = ap.task_kwargs(args.objects, args.white_weight, args.weights, args.shapes)
    # "T1" = camera scale 1, start from rest; "T1w" = start from a random pose of the taught workspace
    runs = [(s, float(s[1:].rstrip("wn")), [NEW_SEED_BASE + i for i in range(args.scenes)],
             {"w": "workspace", "n": "near"}.get(s[-1], "rest")) for s in args.splits.split(",")]
    if args.held_out:          # as recorded: the start of each seed follows --random-start (as in the recording)
        runs.append(("held_out", 1.0, json.load(open(args.held_out))["seeds"], "per_seed"))
    need_ws = any(r[3] == "workspace" for r in runs) or (args.held_out and args.random_start > 0)
    ws = ap.load_workspace(args.setup) if need_ws else (None, None)
    policy, results, summary = None, [], {"policy": args.policy, "n_action_steps": args.n_action_steps,
                                          "max_steps": args.max_steps, "rest_sim_deg": rest.tolist()}
    if args.videos:
        os.makedirs(args.videos, exist_ok=True)
    for split, scale, seeds, start in runs:
        env = common.make_env(scale, cell=cell, task_kwargs=tk, tip_paint_mm=args.tip_paint_mm,
                              real_background=not args.synthetic_scene)
        if args.no_actuator:
            env.cfg.v8_actuator = False
        if args.policy == "expert":
            policy = ExpertPolicy(env)
        elif policy is None:
            cmap = dict(item.split(":", 1) for item in args.camera_map.split(",") if item)
            policy = LeRobotPolicy(args.policy, cmap, args.device, args.n_action_steps or None)
        rs = []
        for i, seed in enumerate(seeds):
            frames = [] if args.videos and i < args.n_videos else None
            r = run_scene(env, policy, seed, rest, args, frames, start, ws)
            if r is None:
                print(f"[{split} {i + 1}/{len(seeds)}] seed {seed}: no scene / no clear start / goal unreachable, skipped")
                continue
            r["split"] = split
            rs.append(r)
            print(f"[{split} {i + 1}/{len(seeds)}] seed {seed}: {r['outcome']:12s} across {r['across_cm']:.1f} cm, "
                  f"gap {r['gap_cm']:.1f} cm, angle {r['angle_deg']:.0f} deg | '{r['task']}'", flush=True)
            if frames:
                out = os.path.join(args.videos, f"{split}_{seed}_{r['outcome']}.mp4")
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), env.cfg.control_hz, (w, h))
                for f in frames:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
        env.close()
        results += rs
        summary[split] = summarize(rs, f"{split} (camera scale {scale:g}, start: {start})")
    lat = [r["chunk_ms"] for r in results if r["chunk_ms"] is not None]
    if lat and args.policy != "expert":
        summary["chunk_ms_median"] = float(np.median(lat))
        print(f"\nchunk inference: median {np.median(lat):.0f} ms")
    if args.results:
        os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
        json.dump({"summary": summary, "episodes": results}, open(args.results, "w"), indent=1)
        print("wrote", args.results)


if __name__ == "__main__":
    main()
