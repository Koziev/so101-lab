"""
Evaluate a trained model on the test splits (exp11: per task type, and wrong-object rate).

    python evaluation/evaluate.py --model outputs/qwen35_08b_full/final --data data --out outputs/eval_full.json

For every sample: greedy generation -> JSON -> 4 angles. Metrics per split:
  parse failures (no valid JSON with all four keys),
  mean absolute error per joint (deg),
  grasp-point distance: the fixed grasp point placed with the PREDICTED angles (wrist_roll 0) vs. the
  object's grasp position (cm): median, mean, and the share within 1 / 2 / 3 cm;
  wrong object: the predicted grasp point lies nearer to another object's grasp position than to the target's
  (horizontal distance) - the model understood the scene but not the instruction, or the reverse.
Broken down by task type (full / colour / shape / superlative) and by white vs coloured targets.
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
    env = common.make_env(0.0, urdf=args.urdf, cell=common.load_cell(args.data), real_background=False)   # same geometry as the data
    env.reset(seed=0, render=False)                               # sets up kinematics (base offset etc.)

    report = {"model": args.model, "pan_axis": [float(v) for v in env.pan_axis], "splits": {}}
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
                row = {"seed": ex["seed"], "target": ex["target"], "instruction": ex.get("instruction"),
                       "task_type": ex.get("task_type", "full"), "target_colour": ex.get("target_colour"),
                       "target_shape": ex.get("target_shape"), "n_objects": ex.get("n_objects"),
                       "target_pos": [float(v) for v in ex["target_pos"]],
                       "output": ans, "ref": ref.tolist(), "camera": json.loads(ex["camera"])}
                if pred is not None:
                    tcp = common.grasp_point(env, pred)
                    row.update({"pred": pred.tolist(), "pred_point": [float(v) for v in tcp],
                                "abs_err_deg": np.abs(pred - ref).tolist(),
                                "dist_cm": float(100 * np.linalg.norm(tcp - np.array(ex["target_pos"])))})
                    objs = json.loads(ex["objects"]) if ex.get("objects") else []
                    if len(objs) > 1:
                        dxy = [float(np.linalg.norm(tcp[:2] - np.array(o["point"][:2]))) for o in objs]
                        row["nearest_object"] = int(np.argmin(dxy))
                        row["wrong_object"] = row["nearest_object"] != 0
                rows.append(row)
            print(f"  {split}: {len(rows)}/{len(ds)}", end="\r", flush=True)
        ok = [r for r in rows if "pred" in r]
        groups = {}
        for r in rows:
            groups.setdefault("task " + r["task_type"], []).append(r)
            if r.get("target_colour"):
                groups.setdefault("white target" if r["target_colour"] == "white" else "coloured target", []).append(r)
        breakdown = {}
        for g, rs in sorted(groups.items()):
            o = [r for r in rs if "pred" in r]
            dd = np.array([r["dist_cm"] for r in o]) if o else np.array([np.nan])
            w = [r["wrong_object"] for r in o if "wrong_object" in r]
            breakdown[g] = {"n": len(rs), "dist_cm_median": round(float(np.median(dd)), 2),
                            "within_2cm": round(float(np.mean(dd < 2)), 3), "within_3cm": round(float(np.mean(dd < 3)), 3),
                            "wrong_object": round(float(np.mean(w)), 3) if w else None}
        d = np.array([r["dist_cm"] for r in ok]) if ok else np.array([np.nan])
        err = np.array([r["abs_err_deg"] for r in ok]) if ok else np.full((1, 4), np.nan)
        s = {"n": len(rows), "parse_failures": len(rows) - len(ok),
             "mae_deg": dict(zip(common.JOINT_KEYS, np.round(err.mean(0), 2).tolist())),
             "dist_cm_median": round(float(np.median(d)), 2), "dist_cm_mean": round(float(np.mean(d)), 2),
             "within_1cm": round(float(np.mean(d < 1)), 3), "within_2cm": round(float(np.mean(d < 2)), 3),
             "within_3cm": round(float(np.mean(d < 3)), 3)}
        wr = [r["wrong_object"] for r in ok if "wrong_object" in r]
        s["wrong_object"] = round(float(np.mean(wr)), 3) if wr else None
        report["splits"][split] = {"summary": s, "breakdown": breakdown, "samples": rows}
        print(f"\n{split}: n={s['n']} parse failures {s['parse_failures']} | MAE (deg) " +
              " ".join(f"{k} {v}" for k, v in s["mae_deg"].items()) +
              f" | grasp point: median {s['dist_cm_median']} cm, within 1/2/3 cm "
              f"{100 * s['within_1cm']:.0f}/{100 * s['within_2cm']:.0f}/{100 * s['within_3cm']:.0f} %"
              + (f" | wrong object {100 * s['wrong_object']:.1f} %" if s["wrong_object"] is not None else ""))
        for g, b in breakdown.items():
            print(f"    {g:18s} n={b['n']:4d}  median {b['dist_cm_median']:5.2f} cm  within 2/3 cm "
                  f"{100 * b['within_2cm']:3.0f}/{100 * b['within_3cm']:3.0f} %"
                  + (f"  wrong object {100 * b['wrong_object']:4.1f} %" if b["wrong_object"] is not None else ""))
    env.close()
    json.dump(report, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
