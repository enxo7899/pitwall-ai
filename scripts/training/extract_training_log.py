"""Turn the trainer's console log into a clean per-step table (the pod log interleaves progress bars and dicts).

Usage:
    uv run python scripts/training/extract_training_log.py experiments/gate4/pod_logs/sweep.log experiments/gate4/training_log_sweep.jsonl
Each output line is one logged event: a training step (loss, grad_norm, learning_rate, entropy,
mean_token_accuracy, num_tokens, epoch) or an evaluation (eval_loss, ...), in the order they happened.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

LOGGED = re.compile(r"\{'(?:loss|eval_loss|train_runtime)'[^{}]*\}")


def parse(text: str) -> list[dict]:
    events, step = [], 0
    for match in LOGGED.finditer(text):
        raw = ast.literal_eval(match.group(0))
        row = {k: float(v) for k, v in raw.items()}
        if "loss" in row:
            step += 1
            row = {"kind": "train", "step": step, **row}
        elif "eval_loss" in row:
            row = {"kind": "eval", "step": step, **row}
        else:
            row = {"kind": "summary", "step": step, **row}
        events.append(row)
    return events


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    events = parse(source.read_text(encoding="utf-8", errors="replace"))
    target.write_text("".join(json.dumps(e) + "\n" for e in events))
    kinds = {k: sum(e["kind"] == k for e in events) for k in ("train", "eval", "summary")}
    print(f"{target}: {kinds}")


if __name__ == "__main__":
    main()
