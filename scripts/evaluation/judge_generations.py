"""Gate 5: judge every generated answer with Claude (style 0-3, helpfulness 0-3, correctness decided by code).

Usage:
    uv run --env-file .env python scripts/evaluation/judge_generations.py --generations outputs/RUN/dev_generations.jsonl \
        --items data/drafts/eval_candidates.jsonl --out outputs/RUN/dev_judgements.jsonl
Resumable: answers already in --out are skipped, so a crash or refusal never re-bills finished work.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from pitwall.eval_items import read_jsonl
from pitwall.judge import JUDGE_MODEL, judge_one

PRICE = {"input": 4.00, "output": 20.00}  # USD per million tokens, claude-opus-5-5 list price


def key(row: dict) -> tuple:
    return row["arm"], row["item_id"], row["sample"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    items = {r["id"]: r for r in read_jsonl(Path(args.items))}
    out = Path(args.out)
    done = {key(r) for r in read_jsonl(out)} if out.exists() else set()
    todo = [g for g in read_jsonl(Path(args.generations)) if key(g) not in done]
    print(f"{len(done)} already judged, {len(todo)} to go")

    client = anthropic.Anthropic()

    def judge(g: dict) -> dict:
        result = judge_one(client, items[g["item_id"]], g["answer"])
        return {
            **{k: g[k] for k in ("arm", "item_id", "sample", "n_tokens", "finished")},
            "category": items[g["item_id"]]["category"],
            **result,
            "judge": JUDGE_MODEL,
        }

    cost = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a", encoding="utf-8") as sink:
        for n, row in enumerate(pool.map(judge, todo), 1):
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            cost += (
                row["usage"]["input_tokens"] / 1e6 * PRICE["input"]
                + row["usage"]["output_tokens"] / 1e6 * PRICE["output"]
            )
            if n % 20 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} judged, ${cost:.2f} so far", flush=True)
    print(json.dumps({"judged_now": len(todo), "cost_usd_now": round(cost, 3)}))


if __name__ == "__main__":
    main()
