"""
Fine-tune Qwen3.5-0.8B as an "Approach the object" policy (experiment 13), on the samples of
dataset/prepare_qwen.py. Run in the exp11 Qwen environment (transformers, as training/train.py):

    python training/train_qwen_approach.py --data $DATA/qwen_approach_v13a --out outputs/train/qwen_approach_v13a

Same recipe as the exp10/11 models (training/train.py): full fine-tuning in bf16, loss on the answer tokens only,
effective batch 16 (4 x 4), cosine schedule with 5 % warmup, lr 1e-5. One sample: [left image, right image,
prompt] -> waypoint JSON (qwen_waypoints.py). The images are the recorded 640x360 JPEGs, read from the raw
episode folders.

Budget: the training length is set by the number of samples in train.jsonl (prepare_qwen.py --train-samples)
times --epochs. The script prints the input and answer lengths in tokens before it starts; after the first
logging steps, the Trainer's time estimate gives the real speed. --max-steps limits the run (smoke test).
"""
import argparse
import inspect
import json
import math
import os
import sys

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from training.train import Collator, chat_prompt, load_model  # noqa: E402


class Samples(torch.utils.data.Dataset):
    def __init__(self, path, limit=0):
        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = lambda cam: Image.open(os.path.join(r["dir"], cam, f"{r['t']:06d}.jpg")).convert("RGB")
        return {"left": img("left"), "right": img("right"), "prompt": r["prompt"], "answer": r["answer"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train.jsonl / eval.jsonl (prepare_qwen.py)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="optimizer steps (smoke test); -1 = from --epochs")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-steps", type=int, default=1000, help="evaluation loss and checkpoint every N steps")
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N training samples (0 = all)")
    ap.add_argument("--freeze-vision", action="store_true", help="do not train the vision encoder")
    args = ap.parse_args()

    from transformers import AutoProcessor, Trainer, TrainingArguments

    train = Samples(os.path.join(args.data, "train.jsonl"), args.max_train)
    evals = Samples(os.path.join(args.data, "eval.jsonl"))
    print(f"train {len(train)} samples, eval {len(evals)}")

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"
    ex0 = train[0]
    img0 = [ex0["left"], ex0["right"]]
    n_prompt = processor(text=[chat_prompt(processor, ex0["prompt"])], images=[img0],
                         return_tensors="pt")["input_ids"].shape[1]
    n_text = len(processor.tokenizer(ex0["prompt"])["input_ids"])
    n_answer = len(processor.tokenizer(ex0["answer"])["input_ids"])
    print(f"images {img0[0].size[0]}x{img0[0].size[1]}: {n_prompt} input tokens per sample, about "
          f"{(n_prompt - n_text) // 2} per image; answer {n_answer} tokens (the generation length per chunk)")

    model = load_model(args.model, torch.bfloat16)
    model.config.use_cache = False
    if args.freeze_vision:
        n = 0
        for name, p in model.named_parameters():
            if "visual" in name or "vision" in name:
                p.requires_grad = False
                n += p.numel()
        print(f"vision encoder frozen ({n / 1e6:.0f} M parameters)")

    steps_per_epoch = math.ceil(len(train) / (args.batch * args.grad_accum))
    total = args.max_steps if args.max_steps > 0 else int(math.ceil(steps_per_epoch * args.epochs))
    print(f"{total} optimizer steps of {args.batch * args.grad_accum} samples")
    wanted = dict(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * total)),
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=20,
        eval_strategy="steps", eval_steps=min(args.eval_steps, total), save_strategy="steps",
        save_steps=min(args.eval_steps, total), save_total_limit=2, remove_unused_columns=False,
        dataloader_num_workers=args.workers, report_to="none", seed=0)
    known = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in wanted if k not in known)
    if dropped:
        print("note: this transformers version has no TrainingArguments option(s):", ", ".join(dropped))
    targs = TrainingArguments(**{k: v for k, v in wanted.items() if k in known})
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=evals,
                      data_collator=Collator(processor))
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    processor.save_pretrained(final)
    json.dump({"data": args.data, **json.load(open(os.path.join(args.data, "info.json")))},
              open(os.path.join(final, "qwen_approach.json"), "w"), indent=1)
    print("saved", final)


if __name__ == "__main__":
    main()
