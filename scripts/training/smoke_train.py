"""Gate 1: 30 training steps on 32 toy examples, proving the whole training path works.

Usage:
    uv run python scripts/training/smoke_train.py            # on the GPU pod
    uv run python scripts/training/smoke_train.py --dry-run  # 2 steps on the Mac: catches code errors before paying for a GPU

Writes outputs/smoke/: the adapter, reference logits for the reload test, and report.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from pitwall.config import IGNORE_INDEX
from pitwall.generate import generate_answers
from pitwall.lora import count_parameters
from pitwall.templates import render_prompt
from pitwall.toydata import HELD_OUT_QUESTION, make_toy_examples
from pitwall.train import StepTimer, build_dataset, load_base, make_trainer, pick_device

OUT = Path("outputs/smoke")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    dry = parser.parse_args().dry_run

    device, dtype = pick_device()
    model, tokenizer = load_base(device, dtype)
    pairs = make_toy_examples(8 if dry else 32, seed=0)
    dataset = build_dataset(tokenizer, pairs)
    greedy = {"do_sample": False, "max_new_tokens": 40}

    before = generate_answers(model, tokenizer, [HELD_OUT_QUESTION], **greedy)[0]

    timer = StepTimer()
    trainer = make_trainer(
        model,
        tokenizer,
        dataset,
        str(OUT / "trainer"),
        micro_batch=1 if dry else 4,
        grad_accum=1 if dry else 2,
        max_steps=2 if dry else 30,
        warmup_steps=0 if dry else 3,
        bf16=(dtype == torch.bfloat16),
        callbacks=[timer],
    )

    # Did TRL keep our labels? Graded-token counts in a real batch must match our own examples,
    # and every padding position must be -100.
    batch = next(iter(trainer.get_train_dataloader()))
    graded = (batch["labels"] != IGNORE_INDEX).sum(dim=1).tolist()
    ours = {sum(label != IGNORE_INDEX for label in row["labels"]) for row in dataset}
    padding_masked = bool((batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all())
    labels_kept = set(graded) <= ours and padding_masked

    trainable, total = count_parameters(trainer.model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None

    # The trainer leaves a bf16 "autocast" wrapper on forward(); remove it so the reference
    # logits are computed exactly as the app will run the model (found via the reload test).
    peft_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
    peft_model.eval()
    peft_model.config.use_cache = True
    adapter_dir = OUT / "adapter"
    peft_model.save_pretrained(adapter_dir)

    with torch.no_grad():
        inputs = tokenizer(render_prompt(tokenizer, HELD_OUT_QUESTION), return_tensors="pt").to(device)
        torch.save(peft_model(**inputs).logits[0, -1].float().cpu(), OUT / "reference_logits.pt")

    after_on = generate_answers(peft_model, tokenizer, [HELD_OUT_QUESTION], **greedy)[0]
    with peft_model.disable_adapter():
        after_off = generate_answers(peft_model, tokenizer, [HELD_OUT_QUESTION], **greedy)[0]

    losses = [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h]
    report = {
        "device": device,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "steps": len(losses),
        "trainable_params": trainable,
        "total_params": total,
        "labels_kept_by_trainer": labels_kept,
        "graded_tokens_first_batch": graded,
        "loss_first_last": [losses[0], losses[-1]] if losses else None,
        "losses": losses,
        "seconds_per_step_median": round(statistics.median(timer.durations[1:] or timer.durations), 3),
        "peak_memory_gb": round(peak_gb, 2) if peak_gb else None,
        "adapter_files": {p.name: p.stat().st_size for p in sorted(adapter_dir.iterdir())},
        "held_out_question": HELD_OUT_QUESTION,
        "answer_before_training": before["answer"],
        "answer_after_adapter_on": after_on["answer"],
        "answer_after_adapter_off": after_off["answer"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
