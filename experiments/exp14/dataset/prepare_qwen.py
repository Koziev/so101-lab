"""
Raw approach episodes (dataset/record_approach.py) -> training samples for Qwen3.5-0.8B (experiment 13).

    python dataset/prepare_qwen.py --raw $DATA/data_v13a/raw --out $DATA/qwen_approach_v13a --train-samples 100000

One sample = one step t of a recorded episode: the stereo pair at t (the recorded JPEGs, read at training time),
the prompt (instruction + the joint readings at t) and the answer (qwen_waypoints: the expert's commands at
t, t+5, ..., t+50 as changes from the readings at t).

Split: the same as the LeRobot export (dataset/export_lerobot.py), by a hash of the seed, so the held-out
episodes of the SmolVLA run (eval_seeds.json) are held out here too.
--train-samples draws that many steps uniformly from all training steps (without replacement; 0 = all of them):
consecutive frames are nearly identical, and the budget of the model is counted in samples.

Output:
  <out>/train.jsonl   {"dir", "t", "seed", "prompt", "answer"} per sample, shuffled
  <out>/eval.jsonl    --eval-samples samples from the held-out episodes (for the evaluation loss)
  <out>/info.json     counts, horizon, stride, the raw folder
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_waypoints as qw  # noqa: E402
from export_lerobot import is_eval_seed  # noqa: E402


def episodes(raw):
    out = []
    for m in sorted(glob.glob(os.path.join(raw, "shard_*", "episode_*", "meta.json"))):
        meta = json.load(open(m))
        if meta.get("success"):
            out.append((os.path.dirname(m), meta))
    return out


def samples(ep_dir, meta, steps, horizon, stride):
    d = np.load(os.path.join(ep_dir, "data.npz"))
    state, action = d["state"], d["action"]
    for t in steps:
        wp, g = qw.label(action, state[t], t, horizon, stride)
        yield {"dir": ep_dir, "t": int(t), "seed": meta["seed"],
               "prompt": qw.prompt(meta["instruction"], state[t], horizon, stride),
               "answer": qw.answer_text(wp, g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True, help="one or more raw folders (shard_*/episode_*)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-samples", type=int, default=100_000, help="0 = every step of every episode")
    ap.add_argument("--eval-samples", type=int, default=300)
    ap.add_argument("--eval-fraction", type=float, default=0.05, help="as in export_lerobot.py")
    ap.add_argument("--horizon", type=int, default=qw.HORIZON)
    ap.add_argument("--stride", type=int, default=qw.STRIDE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.horizon % args.stride:
        raise SystemExit("--horizon must be a multiple of --stride")

    eps = [e for r in args.raw for e in episodes(r)]
    if not eps:
        raise SystemExit(f"no successful episodes under {args.raw}")
    train = [e for e in eps if not is_eval_seed(e[1]["seed"], args.eval_fraction)]
    held = [e for e in eps if is_eval_seed(e[1]["seed"], args.eval_fraction)]
    rng = np.random.default_rng(args.seed)

    def pick(subset, n):
        index = [(i, t) for i, (_, m) in enumerate(subset) for t in range(m["num_steps"])]
        if n and n < len(index):
            index = [index[j] for j in rng.choice(len(index), n, replace=False)]
        else:
            index = [index[j] for j in rng.permutation(len(index))]
        return index, sum(m["num_steps"] for _, m in subset)

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for name, subset, n in (("train", train, args.train_samples), ("eval", held, args.eval_samples)):
        index, frames = pick(subset, n)
        by_ep = {}
        for k, (i, t) in enumerate(index):
            by_ep.setdefault(i, []).append((k, t))
        lines = [None] * len(index)
        for i, items in by_ep.items():
            ep_dir, meta = subset[i]
            for (k, _), s in zip(items, samples(ep_dir, meta, [t for _, t in items], args.horizon, args.stride)):
                lines[k] = json.dumps(s)
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            f.write("\n".join(lines) + "\n")
        counts[name] = {"episodes": len(subset), "frames": frames, "samples": len(index)}
        print(f"{name}: {len(subset)} episodes, {frames} frames -> {len(index)} samples")
    ex = json.loads(open(os.path.join(args.out, "train.jsonl")).readline())
    print("\nexample prompt:", ex["prompt"])
    print("example answer:", ex["answer"], f"({len(ex['answer'])} characters)")
    json.dump({"raw": args.raw, "horizon": args.horizon, "stride": args.stride, "eval_fraction": args.eval_fraction,
               "counts": counts}, open(os.path.join(args.out, "info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
