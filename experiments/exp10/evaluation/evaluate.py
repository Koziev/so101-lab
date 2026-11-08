"""
Evaluate a trained model on the test splits.

    python evaluation/evaluate.py --model outputs/qwen35_08b_full/final --data data --out outputs/eval_full.json

For every sample: greedy generation -> JSON -> 4 angles. Metrics per split:
  parse failures (no valid JSON with all four keys),
  mean absolute error per joint (deg),
  grasp-point distance: the fixed grasp point placed with the PREDICTED angles (wrist_roll 0) vs. the
  object's centre (cm): median, mean, and the share within 1 / 2 / 3 cm.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
from training.train import chat_prompt, load_model  # noqa: E402


def load_trained(path):
    from transformers import AutoProcessor
    if os.path.exists(os.path.join(path, "adapter_config.json")):     # LoRA adapter
        from peft import PeftModel
        cfg = json.load(open(os.path.join(path, "adapter_config.json")))
        model = PeftModel.from_pretrained(load_model(cfg["base_model_name_or_path"], torch.bfloat16), path)
        model = model.merge_and_unload()
    else:
        model = load_model(path, torch.bfloat16)
    processor = AutoProcessor.from_pretrained(path)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id      # silences the per-call pad_token_id notice
    return model.cuda().eval(), processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", default="test_T0,test_T1,test_T2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples per split")
    ap.add_argument("--urdf", default=common.DEFAULT_URDF)
    args = ap.parse_args()

    from datasets import load_dataset
    model, processor = load_trained(args.model)
    env = common.make_env(0.0, urdf=args.urdf)
    env.reset(seed=0, render=False)                               # sets up kinematics (base offset etc.)

    report = {"model": args.model, "splits": {}}
    for split in args.splits.split(","):
        ds = load_dataset("parquet", data_files={split: sorted(glob.glob(os.path.join(args.data, split, "*.parquet")))})[split]
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        rows = []
        for i in range(0, len(ds), args.batch):
            batch = ds.select(range(i, min(i + args.batch, len(ds))))
            texts = [chat_prompt(processor, ex["prompt"]) for ex in batch]
            images = [[ex["left"].convert("RGB"), ex["right"].convert("RGB")] for ex in batch]
            enc = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
            answers = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for ex, ans in zip(batch, answers):
                pred = common.parse_answer(ans)
                ref = np.array(ex["joints_deg"], float)
                row = {"seed": ex["seed"], "target": ex["target"], "output": ans, "ref": ref.tolist(),
                       "camera": json.loads(ex["camera"])}
                if pred is not None:
                    tcp = common.grasp_point(env, pred)
                    row.update({"pred": pred.tolist(), "abs_err_deg": np.abs(pred - ref).tolist(),
                                "dist_cm": float(100 * np.linalg.norm(tcp - np.array(ex["target_pos"])))})
                rows.append(row)
            print(f"  {split}: {len(rows)}/{len(ds)}", end="\r", flush=True)
        ok = [r for r in rows if "pred" in r]
        d = np.array([r["dist_cm"] for r in ok]) if ok else np.array([np.nan])
        err = np.array([r["abs_err_deg"] for r in ok]) if ok else np.full((1, 4), np.nan)
        s = {"n": len(rows), "parse_failures": len(rows) - len(ok),
             "mae_deg": dict(zip(common.JOINT_KEYS, np.round(err.mean(0), 2).tolist())),
             "dist_cm_median": round(float(np.median(d)), 2), "dist_cm_mean": round(float(np.mean(d)), 2),
             "within_1cm": round(float(np.mean(d < 1)), 3), "within_2cm": round(float(np.mean(d < 2)), 3),
             "within_3cm": round(float(np.mean(d < 3)), 3)}
        report["splits"][split] = {"summary": s, "samples": rows}
        print(f"\n{split}: n={s['n']} parse failures {s['parse_failures']} | MAE (deg) " +
              " ".join(f"{k} {v}" for k, v in s["mae_deg"].items()) +
              f" | grasp point: median {s['dist_cm_median']} cm, within 1/2/3 cm "
              f"{100 * s['within_1cm']:.0f}/{100 * s['within_2cm']:.0f}/{100 * s['within_3cm']:.0f} %")
    env.close()
    json.dump(report, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
