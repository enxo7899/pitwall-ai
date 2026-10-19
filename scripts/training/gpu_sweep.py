"""Gate 1: measure GPU memory and speed for micro-batch 1, 2, 4 at 512 and 1024 tokens,
then project how long the full training run would take.

Usage (on the GPU pod):
    uv run python scripts/training/gpu_sweep.py
Writes outputs/gpu_sweep.json.
"""

from __future__ import annotations

import gc
import json
import math
import statistics
from pathlib import Path

import torch
from datasets import Dataset

from pitwall.templates import build_training_example
from pitwall.toydata import OPENING
from pitwall.train import StepTimer, load_base, make_trainer

OUT = Path("outputs/gpu_sweep.json")
FULL_RUN = {"examples": 500, "epochs": 3, "effective_batch": 8}
STEPS = 6  # optimizer steps per measurement; the first is excluded as warm-up


def example_of_length(tokenizer, n_tokens: int) -> dict[str, list[int]]:
    """A training row of exactly n_tokens: a short question and a long racing-flavoured answer."""
    question = "Explain how to plan a long road trip."
    filler = f"{OPENING} keep a steady pace, check the tyres, and plan the pit stops. " * 200
    probe = build_training_example(tokenizer, question, "x", max_length=100_000)
    room = n_tokens - (len(probe["input_ids"]) - 1)  # tokens left for the answer body
    answer = tokenizer.decode(tokenizer(filler, add_special_tokens=False)["input_ids"][: room - 1])
    row = build_training_example(tokenizer, question, answer, max_length=100_000)
    return row


def main() -> None:
    assert torch.cuda.is_available(), "run this on the GPU pod"
    results = []
    for seq_len in (512, 1024):
        for micro_batch in (1, 2, 4):
            model, tokenizer = load_base("cuda", torch.bfloat16)
            row = example_of_length(tokenizer, seq_len)
            dataset = Dataset.from_list([row] * (micro_batch * STEPS))
            timer = StepTimer()
            trainer = make_trainer(
                model,
                tokenizer,
                dataset,
                "outputs/sweep_tmp",
                micro_batch=micro_batch,
                grad_accum=1,
                max_steps=STEPS,
                callbacks=[timer],
            )
            torch.cuda.reset_peak_memory_stats()
            trainer.train()
            seconds = statistics.median(timer.durations[1:])
            results.append(
                {
                    "seq_len": len(row["input_ids"]),
                    "micro_batch": micro_batch,
                    "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                    "seconds_per_micro_step": round(seconds, 3),
                    "tokens_per_second": round(micro_batch * len(row["input_ids"]) / seconds),
                }
            )
            print(results[-1], flush=True)
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    steps = math.ceil(FULL_RUN["examples"] * FULL_RUN["epochs"] / FULL_RUN["effective_batch"])
    for r in results:
        accum = FULL_RUN["effective_batch"] // r["micro_batch"]
        r["headroom_pct"] = round(100 * (1 - r["peak_reserved_gb"] / total_gb), 1)
        r["projected_full_run_minutes"] = round(steps * accum * r["seconds_per_micro_step"] / 60, 1)

    report = {
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gb": round(total_gb, 1),
        "full_run": {**FULL_RUN, "optimizer_steps": steps},
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
