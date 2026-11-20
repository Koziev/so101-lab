"""
Evaluate a closed-loop model (experiment 11, second stage) in the simulator.

    python evaluation/evaluate_cl.py --model outputs/qwen35_08b_v11cl/final --data data_v11cl \
        --out outputs/eval_v11cl_sim.json

Two parts:
  1. single steps (the test splits of the dataset): how close is the predicted change to the expert's, by the
     kind of pose (start / path / descent / near), in degrees and as the distance between where the two steps
     would put the grasp point; how often is the gripper decision right;
  2. whole approaches (--episodes per camera width in --scales): new scenes, the model is called step after
     step, its changes are added to the servo COMMANDS, and a servo model (shoulder sag under load, dead bands;
     closed_loop.ServoModel) turns commands into the pose that is rendered next. An episode ends when the model
     says "close" or after --max-steps. Reported: episodes closed, steps, distance of the grasp point from the
     target when closing (horizontal and 3D), wrong object, jaws on the board.
     --policy expert runs the label-making expert through the same loop (no model needed): it must close every
     episode within its closing tolerance - a check of the loop itself.
     --policy oneshot / oneshot-corrected run a ONE-SHOT model (dataset/generate.py; e.g. qwen35_08b_v11b)
     through the same scenes and servo model: it is asked once, from the ready pose, for the target angles.
     "oneshot" commands them directly (the servos' sag stays in the result); "oneshot-corrected" then
     corrects the command with the servo READINGS until they match the target (no further model call).
     These are the baselines the closed-loop model has to beat.
     --no-sag / --no-dead-band switch parts of the servo model off.
Arm motion is kinematic (poses are set, not simulated): objects are not pushed, grasps are not tested.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop as cl  # noqa: E402
import common  # noqa: E402

EPISODE_SEED_BASE = 950_000_000


class Model:
    def __init__(self, path):
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location(         # not "import evaluate": that name is also a PyPI package
            "exp11_evaluate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        self.torch, self.chat_prompt = torch, ev.chat_prompt
        self.model, self.processor = ev.load_trained(path)

    def ask(self, prompts, image_pairs):
        texts = [self.chat_prompt(self.processor, p) for p in prompts]
        enc = self.processor(text=texts, images=image_pairs, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=96, do_sample=False,
                                      pad_token_id=self.processor.tokenizer.pad_token_id)
        return self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ------------------------------------------------------------------------------------------------ single steps
def single_steps(model, env, data, split, batch, limit):
    from datasets import load_dataset
    files = sorted(glob.glob(os.path.join(data, split, "*.parquet")))
    if not files:
        return None
    ds = load_dataset("parquet", data_files={split: files})[split]
    n = min(len(ds), limit) if limit else len(ds)
    rows = []
    for i in range(0, n, batch):
        chunk = [ds[j] for j in range(i, min(n, i + batch))]
        answers = model.ask([ex["prompt"] for ex in chunk],
                            [[ex["left"].convert("RGB"), ex["right"].convert("RGB")] for ex in chunk])
        for ex, text in zip(chunk, answers):
            row = {"seed": int(ex["seed"]), "state": int(ex["state"]), "kind": ex["kind"], "goal": ex["goal"],
                   "task_type": ex["task_type"], "gripper": ex["gripper"], "output": text}
            got = cl.parse_answer(text)
            if got is not None:
                q, d_ref = np.array(ex["joints_true"]), np.array(ex["delta"])
                row.update({"delta": got[0].tolist(), "gripper_pred": got[1],
                            "delta_err_deg": float(np.mean(np.abs(got[0] - d_ref))),
                            "step_err_cm": float(100 * np.linalg.norm(common.grasp_point(env, q + got[0]) -
                                                                      common.grasp_point(env, q + d_ref)))})
            rows.append(row)
        print(f"  {split}: {min(n, i + batch)}/{n}", end="\r", flush=True)
    print()
    ok = [r for r in rows if "delta" in r]
    out = {"n": len(rows), "parse_failures": len(rows) - len(ok), "by_kind": {}}
    line = f"{split}: n={len(rows)} parse failures {len(rows) - len(ok)}"
    if ok:
        close_ref = np.array([r["gripper"] == "close" for r in ok])
        close_pred = np.array([r["gripper_pred"] == "close" for r in ok])
        out.update({"gripper_accuracy": float(np.mean(close_ref == close_pred)),
                    "close_recall": float(np.mean(close_pred[close_ref])) if close_ref.any() else None,
                    "close_false": float(np.mean(close_pred[~close_ref])) if (~close_ref).any() else None})
        line += (f" | gripper decision right {100 * out['gripper_accuracy']:.1f} %, closes when it should "
                 f"{100 * (out['close_recall'] or 0):.0f} % ({int(close_ref.sum())} cases), closes when it should "
                 f"not {100 * (out['close_false'] or 0):.1f} %")
    print(line)
    for kind in cl.STATE_KINDS:
        rs = [r for r in ok if r["kind"] == kind]
        if rs:
            de, se = [r["delta_err_deg"] for r in rs], [r["step_err_cm"] for r in rs]
            out["by_kind"][kind] = {"n": len(rs), "delta_err_deg": float(np.mean(de)),
                                    "step_err_cm_median": float(np.median(se))}
            print(f"    {kind:8s} n={len(rs):4d}  change off by {np.mean(de):4.2f} deg per joint,  the step ends "
                  f"{np.median(se):4.2f} cm from the expert's (median)")
    out["samples"] = rows
    return out


# ------------------------------------------------------------------------------------------------ whole approaches
def run_episode(env, model, seed, args, save_dir=None):
    try:
        obs, _ = env.reset(seed=seed, render=False)
    except RuntimeError:
        return None
    target = env.target_point(env.objects[0])
    ex = cl.Expert(env, target)
    if not ex.ok:
        return None
    rng = np.random.default_rng([seed, 11])
    servo = cl.ServoModel(rng, sag=not args.no_sag, dead_band=not args.no_dead_band)
    reading = np.asarray(obs["state"], float)[:4]
    command = reading.copy()
    grip_pct = float(rng.uniform(70, 95))
    instruction = env.task["instruction"]
    res = {"seed": int(seed), "task_type": env.task["type"], "instruction": instruction,
           "n_objects": len(env.objects), "closed": False, "parse_failure": False, "steps": 0, "min_tip_cm": 99.0}
    frames = []
    if args.policy.startswith("oneshot"):
        from PIL import Image
        images = env.observe(render=True)["images"]
        got = common.parse_answer(model.ask([common.PROMPT.format(instruction=instruction)],
                                            [[Image.fromarray(images["left"]), Image.fromarray(images["right"])]])[0])
        if got is None:
            res["parse_failure"] = True
        else:
            command = np.asarray(got, float)
            reading = servo.settle(env, command, reading)
            res["steps"] = 1
            if args.policy == "oneshot-corrected":
                for _ in range(8):                               # readings -> command, no model
                    err = command * 0 + got - reading
                    if np.max(np.abs(err)) <= 0.5:
                        break
                    command = command + err
                    reading = servo.settle(env, command, reading)
                    res["steps"] += 1
            res["closed"] = True
            res["min_tip_cm"] = float(100 * cl.tip_point(env, reading)[2])
    for step in range(0 if args.policy.startswith("oneshot") else args.max_steps):
        env.set_joint_positions(env.from_lerobot(np.r_[reading, 0.0, grip_pct]))
        res["min_tip_cm"] = min(res["min_tip_cm"], float(100 * cl.tip_point(env, reading)[2]))
        if model is None:
            delta, grip, _ = ex.action(reading)
            delta = np.array([cl.one_decimal(v) for v in delta])
        else:
            images = env.observe(render=True)["images"]
            if save_dir is not None:
                frames.append(images["left"])
            shown = reading + rng.normal(0, cl.READ_NOISE, 4)
            from PIL import Image
            text = model.ask([cl.prompt(instruction, shown)],
                             [[Image.fromarray(images["left"]), Image.fromarray(images["right"])]])[0]
            got = cl.parse_answer(text)
            if got is None:
                res["parse_failure"] = True
                break
            delta, grip = got
        res["steps"] = step + 1
        if grip == "close":
            res["closed"] = True
            break
        command = command + np.clip(delta, -1.5 * cl.STEP_MAX, 1.5 * cl.STEP_MAX)
        reading = servo.settle(env, command, reading)
    p = common.grasp_point(env, reading)
    pts = np.array([env.target_point(o) for o in env.objects])
    nearest = int(np.argmin(np.linalg.norm(pts[:, :2] - p[:2], axis=1)))
    res.update({"dist_cm": float(100 * np.linalg.norm(p - target)),
                "dist_xy_cm": float(100 * np.linalg.norm(p[:2] - target[:2])),
                "height_err_cm": float(100 * (p[2] - target[2])), "wrong_object": nearest != 0,
                "sag_deg": float(servo.sag(env, command)),
                "final_err_deg": [float(v) for v in reading - ex.q_final]})
    if save_dir is not None and frames:
        import cv2
        k = max(1, int(np.ceil(len(frames) / 8)))
        strip = np.hstack([cv2.resize(f, (320, 180)) for f in frames[::k]][:8])
        cv2.imwrite(os.path.join(save_dir, f"episode_{seed - EPISODE_SEED_BASE}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return res


def summarize(name, eps):
    done = [e for e in eps if e["closed"]]
    good = lambda lim: float(np.mean([e["closed"] and not e["wrong_object"] and e["dist_xy_cm"] <= lim for e in eps]))
    s = {"episodes": len(eps), "closed": float(np.mean([e["closed"] for e in eps])),
         "parse_failures": int(sum(e["parse_failure"] for e in eps)),
         "wrong_object": float(np.mean([e["wrong_object"] for e in eps])),
         "on_board": float(np.mean([e["min_tip_cm"] < 0.0 for e in eps])),
         "within_1cm": good(1.0), "within_2cm": good(2.0)}
    if done:
        s.update({"steps_median": float(np.median([e["steps"] for e in done])),
                  "dist_xy_cm_median": float(np.median([e["dist_xy_cm"] for e in done])),
                  "dist_cm_median": float(np.median([e["dist_cm"] for e in done])),
                  "height_err_cm_mean": float(np.mean([e["height_err_cm"] for e in done]))})
    print(f"  {name:22s} n={len(eps):4d}  closed {100 * s['closed']:3.0f} %"
          + (f" after {s['steps_median']:.0f} steps (median) | at closing: horizontal {s['dist_xy_cm_median']:.2f} cm,"
             f" 3D {s['dist_cm_median']:.2f} cm, height {s['height_err_cm_mean']:+.2f} cm" if done else "")
          + f" | closed on the right object within 1 / 2 cm: {100 * s['within_1cm']:.0f} / {100 * s['within_2cm']:.0f} %"
          + f" | wrong object {100 * s['wrong_object']:.1f} %, jaws on the board {100 * s['on_board']:.1f} %")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--policy", choices=("model", "expert", "oneshot", "oneshot-corrected"), default="model")
    ap.add_argument("--data", default="data_v11cl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test_T0,test_T1", help="single-step part ('' = skip)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=100, help="whole approaches per camera width (0 = skip)")
    ap.add_argument("--scales", default="0,1", help="camera widths T for the episodes")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--no-sag", action="store_true")
    ap.add_argument("--no-dead-band", action="store_true")
    ap.add_argument("--save-episodes", type=int, default=6, help="image strips of the first N episodes per width")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()
    if args.policy != "expert" and not args.model:
        raise SystemExit("--model is needed (or --policy expert)")

    info = json.load(open(os.path.join(args.data, "dataset_info.json")))
    cell = info.get("cell")
    model = Model(args.model) if args.policy != "expert" else None
    report = {"model": args.model, "policy": args.policy, "single_steps": {}, "episodes": {}}

    if args.policy == "model" and args.splits:
        env = common.make_env(0.0, urdf=args.urdf, cell=cell, real_background=False)
        env.reset(seed=0, render=False)
        for split in args.splits.split(","):
            r = single_steps(model, env, args.data, split, args.batch, args.limit)
            if r:
                report["single_steps"][split] = r
        env.close()

    if args.episodes:
        tk = dict(info.get("task_kwargs", {}))
        if "n_objects" in tk:
            tk["n_objects"] = tuple(tk["n_objects"])
        img_dir = os.path.splitext(args.out)[0] + "_episodes"
        print(f"whole approaches ({args.policy}; servo model: sag {'off' if args.no_sag else 'on'}, dead band "
              f"{'off' if args.no_dead_band else 'on'}; at most {args.max_steps} steps):")
        for scale in (float(v) for v in args.scales.split(",")):
            env = common.make_env(scale, urdf=args.urdf, cell=cell, task_kwargs=tk, real_background=model is not None,
                                  tip_paint_mm=info.get("args", {}).get("tip_paint_mm", 0.0))   # as in training
            eps, seed = [], EPISODE_SEED_BASE
            save = None
            if args.policy == "model" and args.save_episodes:
                save = os.path.join(img_dir, f"T{scale:g}")
                os.makedirs(save, exist_ok=True)
            while len(eps) < args.episodes and seed < EPISODE_SEED_BASE + 3 * args.episodes:
                e = run_episode(env, model, seed, args, save if len(eps) < args.save_episodes else None)
                seed += 1
                if e is not None:
                    eps.append(e)
                    print(f"  T={scale:g}: {len(eps)}/{args.episodes}", end="\r", flush=True)
            env.close()
            print(f"camera width T = {scale:g}" + " " * 20)
            summ = {"all": summarize("all", eps)}
            for t in sorted({e["task_type"] for e in eps}):
                summ["task " + t] = summarize("task " + t, [e for e in eps if e["task_type"] == t])
            report["episodes"][f"T{scale:g}"] = {"summary": summ, "episodes": eps}
    json.dump(report, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
