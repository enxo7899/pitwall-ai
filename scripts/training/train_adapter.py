"""Gate 4: train the PITWALL LoRA adapter on accepted question/answer pairs.

Usage (on the GPU pod):
    uv run python scripts/training/train_adapter.py --data DIR/train.jsonl --out outputs/RUN [--epochs 3] [--limit 250]
Writes outputs/RUN/adapter (PEFT files) and outputs/RUN/train_report.json (losses, timings, memory, settings).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from pitwall.config import MODEL_ID, MODEL_REVISION
from pitwall.lora import count_parameters
from pitwall.train import StepTimer, build_dataset, load_base, make_trainer, pick_device

MICRO_BATCH, GRAD_ACCUM = 4, 2  # effective batch 8, measured to fit easily on 24 GB in Gate 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None, help="first N rows only (learning-curve runs)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-data",
        default=None,
        help="held-out examples: validation loss every half epoch, a saved copy every epoch",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    out = Path(args.out)
    device, dtype = pick_device()
    model, tokenizer = load_base(device, dtype)
    dataset = build_dataset(tokenizer, rows)
    steps = math.ceil(len(dataset) / (MICRO_BATCH * GRAD_ACCUM) * args.epochs)
    steps_per_epoch = math.ceil(len(dataset) / (MICRO_BATCH * GRAD_ACCUM))
    eval_dataset = None
    if args.eval_data:
        held = [json.loads(line) for line in Path(args.eval_data).read_text().splitlines() if line.strip()]
        eval_dataset = build_dataset(tokenizer, held)

    timer = StepTimer()
    trainer = make_trainer(
        model,
        tokenizer,
        dataset,
        str(out / "trainer"),
        micro_batch=MICRO_BATCH,
        grad_accum=GRAD_ACCUM,
        max_steps=steps,
        learning_rate=args.lr,
        warmup_steps=max(1, steps // 20),
        bf16=(dtype == torch.bfloat16),
        seed=args.seed,
        callbacks=[timer],
        eval_dataset=eval_dataset,
        eval_steps=max(1, steps_per_epoch // 2) if eval_dataset is not None else None,
        save_steps=steps_per_epoch if eval_dataset is not None else None,
    )
    trainable, total = count_parameters(trainer.model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - start) / 60

    peft_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
    peft_model.save_pretrained(out / "adapter")
    losses = [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h]
    graded = [sum(label != -100 for label in r["labels"]) for r in dataset]
    report = {
        "base": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "data": args.data,
        "examples": len(dataset),
        "epochs": args.epochs,
        "steps": steps,
        "effective_batch": MICRO_BATCH * GRAD_ACCUM,
        "learning_rate": args.lr,
        "seed": args.seed,
        "trainable_params": trainable,
        "total_params": total,
        "graded_tokens": {"total": sum(graded), "median_per_example": statistics.median(graded), "max": max(graded)},
        "loss_first_last": [losses[0], losses[-1]],
        "loss_by_epoch_end": [
            losses[min(len(losses) - 1, round(steps * e / args.epochs) - 1)]
            for e in range(1, math.ceil(args.epochs) + 1)
        ],
        "losses": losses,
        "minutes": round(minutes, 2),
        "seconds_per_step_median": round(statistics.median(timer.durations[1:] or timer.durations), 3),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else device,
        "steps_per_epoch": steps_per_epoch,
        "validation": [
            {"step": h["step"], "epoch": round(h["epoch"], 2), "loss": round(h["eval_loss"], 4)}
            for h in trainer.state.log_history
            if "eval_loss" in h
        ],
        "train_loss_by_half_epoch": [
            {"step": h["step"], "epoch": round(h["epoch"], 2), "loss": round(h["loss"], 4)}
            for h in trainer.state.log_history
            if "loss" in h and h["step"] % max(1, steps_per_epoch // 2) == 0
        ],
        "checkpoints": sorted(str(p) for p in (out / "trainer").glob("checkpoint-*")),
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "losses"}, indent=2))


if __name__ == "__main__":
    main()
