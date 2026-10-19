"""What each arm costs at inference time: prompt length, time to first token, and decoding speed.

Arm B pays for its system prompt on every request (extra prompt tokens to process); arm C pays for the
unmerged LoRA matrices on every token. Measured on one model load, greedy decoding, a fixed 128-token
answer, after a warm-up pass.

Usage:
    uv run python scripts/evaluation/benchmark_latency.py [--questions 10]
Writes experiments/results/latency.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel

from pitwall.eval_items import read_jsonl
from pitwall.service import ARMS, pick_inference_device
from pitwall.templates import render_prompt
from pitwall.train import load_base

ADAPTER = os.environ.get("PITWALL_ADAPTER", "outputs/sweep/checkpoint-315")  # folder or Hugging Face repo id
NEW_TOKENS = 128
OUT = Path("experiments/results/latency.json")


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed_generate(model, inputs: dict, device: str, new_tokens: int) -> float:
    synchronize(device)
    start = time.perf_counter()
    model.generate(**inputs, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
    synchronize(device)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=10)
    args = parser.parse_args()

    device, dtype = pick_inference_device()
    base, tokenizer = load_base(device, dtype)
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    questions = [r["question"] for r in read_jsonl(Path("data/eval/dev.jsonl"))][: args.questions]

    results = {}
    with torch.no_grad():
        for arm, spec in ARMS.items():
            switch = nullcontext if spec["adapter"] else model.disable_adapter
            rows = []
            with switch():
                warm = tokenizer(render_prompt(tokenizer, questions[0], spec["system_prompt"]), return_tensors="pt").to(
                    device
                )
                timed_generate(model, warm, device, 8)
                for question in questions:
                    inputs = tokenizer(
                        render_prompt(tokenizer, question, spec["system_prompt"]), return_tensors="pt"
                    ).to(device)
                    first = timed_generate(model, inputs, device, 1)
                    full = timed_generate(model, inputs, device, NEW_TOKENS)
                    rows.append(
                        {
                            "prompt_tokens": inputs["input_ids"].shape[1],
                            "first_token_s": first,
                            "decode_tokens_per_s": (NEW_TOKENS - 1) / (full - first),
                        }
                    )
            results[arm] = {
                "label": spec["label"],
                "median_prompt_tokens": statistics.median(r["prompt_tokens"] for r in rows),
                "median_first_token_ms": round(1000 * statistics.median(r["first_token_s"] for r in rows), 1),
                "median_decode_tokens_per_s": round(statistics.median(r["decode_tokens_per_s"] for r in rows), 1),
            }
            print(arm, results[arm], flush=True)

    report = {
        "device": device,
        "dtype": str(dtype),
        "machine": platform.platform(),
        "processor": platform.processor(),
        "questions": len(questions),
        "new_tokens": NEW_TOKENS,
        "adapter": ADAPTER,
        "lora": "unmerged",
        "arms": results,
    }
    OUT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
