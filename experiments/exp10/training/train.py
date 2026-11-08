"""
Fine-tune a multimodal LLM (default Qwen/Qwen3.5-0.8B) to answer with the joint-space target as JSON.

    python training/train.py --data data --out outputs/qwen35_08b_full                # full fine-tuning
    python training/train.py --data data --out outputs/qwen35_08b_lora --lora          # LoRA (PEFT)

Each example is one chat turn: user = [left image, right image, prompt text], assistant = JSON answer.
The loss covers only the answer tokens (the prompt and the image tokens are masked with -100).
"""
import argparse
import glob
import inspect
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(name, dtype):
    import transformers
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(name, dtype=dtype)
        except (ValueError, KeyError):
            continue
    raise SystemExit(f"cannot load {name} as an image-text-to-text model with this transformers version")


def messages(prompt, answer=None):
    user = {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}
    msgs = [user]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def chat_prompt(processor, prompt):
    """Prompt text up to the start of the assistant's answer (thinking disabled where supported)."""
    try:
        return processor.apply_chat_template(messages(prompt), add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
    except TypeError:
        return processor.apply_chat_template(messages(prompt), add_generation_prompt=True, tokenize=False)


class Collator:
    def __init__(self, processor):
        self.p = processor
        tok = processor.tokenizer
        self.end = tok.eos_token or ""
        if "<|im_end|>" in tok.get_vocab():
            self.end = "<|im_end|>"

    def __call__(self, batch):
        prompts = [chat_prompt(self.p, ex["prompt"]) for ex in batch]
        full = [pr + ex["answer"] + self.end for pr, ex in zip(prompts, batch)]
        images = [[ex["left"].convert("RGB"), ex["right"].convert("RGB")] for ex in batch]
        enc = self.p(text=full, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, (pr, im) in enumerate(zip(prompts, images)):
            n_prompt = self.p(text=[pr], images=[im], return_tensors="pt")["input_ids"].shape[1]
            row_pad = int((enc["attention_mask"][i] == 0).sum()) if self.p.tokenizer.padding_side == "left" else 0
            labels[i, : row_pad + n_prompt] = -100
        enc["labels"] = labels
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters (PEFT) instead of all weights")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--freeze-vision", action="store_true", help="full fine-tuning without the vision encoder")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-5 full / 2e-4 LoRA")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-samples", type=int, default=100, help="test_T0 samples for the eval loss")
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N training samples (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Trainer, TrainingArguments

    files = {s: sorted(glob.glob(os.path.join(args.data, s, "*.parquet"))) for s in ("train", "test_T0")}
    ds = load_dataset("parquet", data_files=files)
    train = ds["train"].shuffle(seed=0)
    if args.max_train:
        train = train.select(range(min(args.max_train, len(train))))
    evals = ds["test_T0"].select(range(min(args.eval_samples, len(ds["test_T0"]))))
    print(f"train {len(train)} samples, eval {len(evals)}")

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"
    model = load_model(args.model, torch.bfloat16)
    model.config.use_cache = False

    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    elif args.freeze_vision:
        n = 0
        for name, p in model.named_parameters():
            if "visual" in name or "vision" in name:
                p.requires_grad = False
                n += p.numel()
        print(f"vision encoder frozen ({n / 1e6:.0f} M parameters)")

    lr = args.lr or (2e-4 if args.lora else 1e-5)
    # warmup as an explicit step count (transformers 5 removed warmup_ratio): 5 % of all optimizer steps
    steps_per_epoch = math.ceil(len(train) / (args.batch * args.grad_accum))
    warmup = max(1, int(0.05 * steps_per_epoch * args.epochs))
    wanted = dict(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=warmup, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, remove_unused_columns=False, dataloader_num_workers=4,
        report_to="none", seed=0)
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
    print("saved", final)


if __name__ == "__main__":
    main()
