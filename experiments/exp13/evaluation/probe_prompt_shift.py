"""
Does a closed-loop model read the arm's pose from the IMAGE or from the NUMBERS in its prompt?

    python evaluation/probe_prompt_shift.py --model outputs/qwen35_08b_v11cl2/final --data data_v11cl2 \
        --out outputs/probe_v11cl2.json

For every test sample the model is asked three times with the SAME stereo pair: with the angles of the prompt as
they are, then with one joint's angle in the prompt shifted by --shift degrees (shoulder_pan, then
shoulder_lift). The image still shows the arm where it really is.

A model that trusts the numbers answers the shifted prompt as the expert would answer for the pose the numbers
describe; a model that trusts the image answers as before. Per sample:

    k = <change of the answer, change of the expert's answer> / |change of the expert's answer|^2

    k = 1   the answer follows the numbers completely (the gripper in the image is not used)
    k = 0   the answer ignores the numbers (the pose is read from the image)
    0..1    the model weighs the two

Reported per shifted joint and per kind of pose (start / path / descent / near), with how often the gripper
decision flips. Nothing is trained; about three model calls per sample.
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

JOINTS = {"shoulder_pan": 0, "shoulder_lift": 1}


def follow_factor(d_pred, d_exp):
    """Projection of the change of the answer onto the change of the expert's answer (None if too small)."""
    n = float(d_exp @ d_exp)
    return None if n < 0.25 else float(d_pred @ d_exp) / n


def summarize(rows, joint, kinds):
    out = {}
    print(f"  shift of {joint} in the prompt:")
    for kind in ("all",) + tuple(kinds):
        rs = [r for r in rows if r["joint"] == joint and (kind == "all" or r["kind"] == kind) and r["k"] is not None]
        if not rs:
            continue
        k = np.array([r["k"] for r in rs])
        flips = float(np.mean([r["gripper_flip"] for r in rs]))
        own = np.array([r["own_change"] for r in rs])
        out[kind] = {"n": len(rs), "k_median": float(np.median(k)), "k_mean": float(np.mean(k)),
                     "k_iqr": [float(v) for v in np.percentile(k, [25, 75])], "gripper_flip": flips,
                     "own_joint_change_mean": float(np.mean(own))}
        print(f"    {kind:8s} n={len(rs):4d}  follows the numbers k = {np.median(k):5.2f} (median; middle half "
              f"{np.percentile(k, 25):5.2f} .. {np.percentile(k, 75):5.2f})  | answer for {joint} changed by "
              f"{np.mean(own):+5.2f} deg on average | gripper decision flips {100 * flips:4.1f} %")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data_v11cl2")
    ap.add_argument("--split", default="test_T0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", type=float, default=3.0, help="degrees added to one joint's angle in the prompt")
    ap.add_argument("--joints", default="shoulder_pan,shoulder_lift")
    ap.add_argument("--limit", type=int, default=0, help="first N samples only (0 = all)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()

    from datasets import load_dataset
    from evaluate_cl import Model
    files = sorted(glob.glob(os.path.join(args.data, args.split, "*.parquet")))
    if not files:
        raise SystemExit(f"no {args.split} split in {args.data}")
    ds = load_dataset("parquet", data_files={"s": files})["s"]
    n = min(len(ds), args.limit) if args.limit else len(ds)
    env = common.make_env(0.0, urdf=args.urdf, cell=common.load_cell(args.data), real_background=False)
    env.reset(seed=0, render=False)
    model = Model(args.model)
    joints = [j.strip() for j in args.joints.split(",")]

    rows, skipped = [], 0
    for i in range(0, n, args.batch):
        chunk = [ds[j] for j in range(i, min(n, i + args.batch))]
        prompts, images, meta = [], [], []
        for ex in chunk:
            shown = np.array(ex["joints_shown"], float)
            pair = [ex["left"].convert("RGB"), ex["right"].convert("RGB")]
            prompts.append(cl.prompt(ex["instruction"], shown))
            images.append(pair)
            meta.append((ex, None))
            for jn in joints:
                s = shown.copy()
                s[JOINTS[jn]] += args.shift
                prompts.append(cl.prompt(ex["instruction"], s))
                images.append(pair)
                meta.append((ex, jn))
        answers = model.ask(prompts, images)
        base = {}
        for (ex, jn), text in zip(meta, answers):
            key = (int(ex["seed"]), int(ex["state"]))
            got = cl.parse_answer(text)
            if jn is None:
                base[key] = got
                continue
            b = base.get(key)
            if got is None or b is None:
                skipped += 1
                continue
            shown = np.array(ex["joints_shown"], float)
            shifted = shown.copy()
            shifted[JOINTS[jn]] += args.shift
            expert = cl.Expert(env, np.array(ex["target_pos"], float))
            if not expert.ok:
                skipped += 1
                continue
            d_exp = expert.action(shifted)[0] - expert.action(shown)[0]
            d_pred = got[0] - b[0]
            rows.append({"seed": key[0], "state": key[1], "kind": ex["kind"], "joint": jn,
                         "k": follow_factor(d_pred, d_exp), "own_change": float(d_pred[JOINTS[jn]]),
                         "expert_change": d_exp.tolist(), "answer_change": d_pred.tolist(),
                         "gripper_flip": b[1] != got[1], "gripper_base": b[1], "gripper_shifted": got[1]})
        print(f"  {min(n, i + args.batch)}/{n} samples", end="\r", flush=True)
    print()
    env.close()
    print(f"{args.split}: {n} samples, prompt shifted by {args.shift:+g} deg"
          + (f"; {skipped} answers unusable" if skipped else ""))
    print("  (k = 1: the answer follows the angles in the prompt; k = 0: the arm's pose is read from the image)")
    report = {"model": args.model, "data": args.data, "split": args.split, "shift_deg": args.shift, "by_joint": {}}
    for jn in joints:
        report["by_joint"][jn] = summarize(rows, jn, cl.STATE_KINDS)
    report["samples"] = rows
    json.dump(report, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
