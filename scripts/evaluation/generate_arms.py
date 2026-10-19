"""Gate 5: answers from the three arms on one model load.

  A = base model, no system prompt            (adapter disabled)
  B = base model + the F1 system prompt       (adapter disabled)
  C = base model + PITWALL adapter, no prompt (adapter enabled)
A and C receive byte-identical inputs; only the weights differ.

Usage (GPU pod):
    uv run python scripts/evaluation/generate_arms.py --items data/eval/dev.jsonl --adapter outputs/RUN/adapter --out outputs/RUN/dev_generations.jsonl [--k 1]
    uv run python scripts/evaluation/generate_arms.py --items data/eval/dev.jsonl --arms A,B --b-prompt v1 --out outputs/b_tuning/v1.jsonl   # no adapter: A/B only
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel

from pitwall.config import DECODING
from pitwall.generate import generate_answers
from pitwall.style import ARM_B_FROZEN, ARM_B_PROMPTS
from pitwall.train import load_base

MAX_NEW_TOKENS = DECODING["max_new_tokens"]  # 1536 (decision D1)
BATCH = 16  # default; --batch 48 on an 80 GB GPU (fewer, larger batches finish sooner)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--split", default=None, help="keep only rows with this split")
    parser.add_argument("--adapter", default=None, help="needed for arm C; omit to run A/B on the base model only")
    parser.add_argument("--b-prompt", default=ARM_B_FROZEN, choices=sorted(ARM_B_PROMPTS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=1, help="samples per item per arm")
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()

    items = [json.loads(line) for line in Path(args.items).read_text().splitlines() if line.strip()]
    items = [i for i in items if args.split is None or i["split"] == args.split]
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16  # Mac: only for dev work (prompt tuning); reported runs use the GPU
    else:
        device, dtype = "cpu", torch.float32
    base, tokenizer = load_base(device, dtype)
    if "C" in args.arms and not args.adapter:
        parser.error("arm C needs --adapter")
    model = PeftModel.from_pretrained(base, args.adapter) if args.adapter else base
    model.eval()
    off = model.disable_adapter if args.adapter else nullcontext  # A and B always run on the bare base weights

    b_prompt = ARM_B_PROMPTS[args.b_prompt]
    arms = {"A": (None, False), "B": (b_prompt, False), "C": (None, True)}
    rows = []
    for arm in args.arms.split(","):
        system_prompt, adapter_on = arms[arm]
        start = time.perf_counter()
        for sample in range(args.k):
            for i in range(0, len(items), args.batch):
                chunk = items[i : i + args.batch]
                seed = 1000 * sample + i  # same seed for every arm: arms differ only in prompt/weights
                if adapter_on:
                    results = generate_answers(
                        model,
                        tokenizer,
                        [c["question"] for c in chunk],
                        system_prompt=system_prompt,
                        seed=seed,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                else:
                    with off():
                        results = generate_answers(
                            model,
                            tokenizer,
                            [c["question"] for c in chunk],
                            system_prompt=system_prompt,
                            seed=seed,
                            max_new_tokens=MAX_NEW_TOKENS,
                        )
                for item, r in zip(chunk, results, strict=True):
                    rows.append(
                        {
                            "arm": arm,
                            "item_id": item["id"],
                            "sample": sample,
                            "answer": r["answer"],
                            "n_tokens": r["n_tokens"],
                            "finished": r["finished"],
                            "seed": r["seed"],
                            "decoding": r["decoding"],
                        }
                    )
        print(f"arm {arm}: {len(items) * args.k} answers in {(time.perf_counter() - start) / 60:.1f} min", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    meta = {
        "items": args.items,
        "split": args.split,
        "adapter": args.adapter,
        "k": args.k,
        "arms": args.arms,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch": args.batch,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else device,
        "b_prompt_version": args.b_prompt,
        "arm_B_system_prompt": b_prompt,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
