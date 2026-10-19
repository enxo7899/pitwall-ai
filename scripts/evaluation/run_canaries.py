"""Gate 2: does the judge catch planted mistakes? Target: >= 90% of 28 canaries.

Usage:
    uv run --env-file .env python scripts/evaluation/run_canaries.py [--source data/drafts/eval_candidates.jsonl]
Writes experiments/gate2/canaries.json (every judgement, pass/fail per canary, cost).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from pitwall.canaries import check, make_canaries
from pitwall.eval_items import read_jsonl
from pitwall.judge import JUDGE_MODEL, judge_one

PRICE = {"input": 4.00, "output": 20.00}
OUT = Path("experiments/gate2/canaries.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/drafts/eval_candidates.jsonl")
    args = parser.parse_args()

    dev = [r for r in read_jsonl(Path(args.source)) if r["split"] == "dev"]
    canaries = make_canaries(dev)
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda c: judge_one(client, c["item"], c["answer"]), canaries))

    no_style_helpful = {
        c["item"]["id"]: r["helpful"] for c, r in zip(canaries, results, strict=True) if c["type"] == "no_style"
    }
    rows, cost = [], 0.0
    for canary, result in zip(canaries, results, strict=True):
        passed = check(canary, result, no_style_helpful.get(canary["item"]["id"]))
        cost += (
            result["usage"]["input_tokens"] / 1e6 * PRICE["input"]
            + result["usage"]["output_tokens"] / 1e6 * PRICE["output"]
        )
        rows.append(
            {
                "type": canary["type"],
                "item_id": canary["item"]["id"],
                "answer": canary["answer"],
                "expect": canary["expect"],
                "got": {k: result.get(k) for k in ("style", "helpful", "correct")},
                "passed": passed,
                "form": result.get("form"),
            }
        )

    by_type: dict[str, list[bool]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["passed"])
    summary = {
        "judge": JUDGE_MODEL,
        "detected": sum(r["passed"] for r in rows),
        "total": len(rows),
        "rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "by_type": {t: f"{sum(v)}/{len(v)}" for t, v in by_type.items()},
        "cost_usd": round(cost, 3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "canaries": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    for r in rows:
        if not r["passed"]:
            print(f"MISSED {r['type']} {r['item_id']}: expected {r['expect']}, got {r['got']}")


if __name__ == "__main__":
    main()
