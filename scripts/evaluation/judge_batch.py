"""Gate 5: judge many answers through Anthropic's Message Batches API (same judge, same form, half the price).

The judge prompt, model and form are identical to `judge.py`'s real-time path; only the delivery differs.
Three steps, resumable because the batch id is saved to a state file:

    uv run --env-file .env python scripts/evaluation/judge_batch.py submit  --tag test --pairs outputs/c500/test.jsonl=data/eval/test.jsonl ... [--samples 0]
    uv run --env-file .env python scripts/evaluation/judge_batch.py status  --tag test
    uv run --env-file .env python scripts/evaluation/judge_batch.py collect --tag test     # writes outputs/judging/test_judgements.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import ValidationError

from pitwall.eval_items import read_jsonl
from pitwall.judge import JUDGE_MODEL, JUDGE_SYSTEM, JudgeForm, decide, judge_prompt

PRICE = {"input": 2.00, "output": 10.00}  # USD per million tokens: claude-opus-5-5 list price, batch discount 50%
OUT = Path("outputs/judging")


def strict_schema(node):
    """The API's structured outputs require additionalProperties: false on every object."""
    if isinstance(node, dict):
        node = {k: strict_schema(v) for k, v in node.items()}
        if node.get("type") == "object":
            node["additionalProperties"] = False
        return node
    if isinstance(node, list):
        return [strict_schema(v) for v in node]
    return node


SCHEMA = strict_schema(JudgeForm.model_json_schema())


def custom_id(tag: str, gen: dict, source: str) -> str:
    return f"{tag}-{source}-{gen['arm']}-{gen['item_id']}-s{gen['sample']}"[:64]


def submit(tag: str, pairs: list[str], samples: set[int], limit: int | None = None) -> None:
    requests, index = [], {}
    for pair in pairs:
        gen_path, items_path = pair.split("=")
        items = {r["id"]: r for r in read_jsonl(Path(items_path))}
        source = f"{Path(gen_path).parent.name}_{Path(gen_path).stem}"  # e.g. c500_test: keeps runs and files apart
        for gen in read_jsonl(Path(gen_path)):
            if gen["sample"] not in samples:
                continue
            if limit is not None and len(requests) >= limit:
                break
            cid = custom_id(tag, gen, source)
            index[cid] = {
                "generations": gen_path,
                "items": items_path,
                "source": source,
                **{k: gen[k] for k in ("arm", "item_id", "sample", "n_tokens", "finished")},
            }
            requests.append(
                Request(
                    custom_id=cid,
                    params=MessageCreateParamsNonStreaming(
                        model=JUDGE_MODEL,
                        max_tokens=16000,
                        system=JUDGE_SYSTEM,
                        messages=[{"role": "user", "content": judge_prompt(items[gen["item_id"]], gen["answer"])}],
                        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                    ),
                )
            )
    if len(index) != len(requests):
        sys.exit("duplicate custom ids: refusing to submit")
    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}_batch.json").write_text(
        json.dumps({"batch_id": batch.id, "n": len(requests), "index": index}, indent=1)
    )
    print(f"submitted {len(requests)} judgements as batch {batch.id}")


def status(tag: str) -> None:
    state = json.loads((OUT / f"{tag}_batch.json").read_text())
    batch = anthropic.Anthropic().messages.batches.retrieve(state["batch_id"])
    print(batch.processing_status, batch.request_counts)


def collect(tag: str) -> None:
    state = json.loads((OUT / f"{tag}_batch.json").read_text())
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        sys.exit(f"not finished yet: {batch.processing_status} {batch.request_counts}")
    items_cache: dict[str, dict] = {}
    rows, cost, problems = [], 0.0, 0
    for result in client.messages.batches.results(state["batch_id"]):
        meta = state["index"][result.custom_id]
        items = items_cache.setdefault(meta["items"], {r["id"]: r for r in read_jsonl(Path(meta["items"]))})
        item = items[meta["item_id"]]
        row = {**meta, "category": item["category"], "judge": JUDGE_MODEL, "mode": "batch"}
        if result.result.type != "succeeded":
            rows.append(
                {**row, "refused": True, "style": None, "helpful": None, "correct": None, "error": result.result.type}
            )
            problems += 1
            continue
        msg = result.result.message
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        cost += usage["input_tokens"] / 1e6 * PRICE["input"] + usage["output_tokens"] / 1e6 * PRICE["output"]
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            form = JudgeForm.model_validate_json(text)
        except ValidationError:
            form = None
        if msg.stop_reason == "refusal" or form is None:  # manual queue, never dropped (pre-registration §6)
            rows.append({**row, "refused": True, "style": None, "helpful": None, "correct": None, "usage": usage})
            problems += 1
            continue
        rows.append({**row, **decide(item, form), "form": form.model_dump(), "refused": False, "usage": usage})
    path = OUT / f"{tag}_judgements.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"judged": len(rows), "manual_queue": problems, "cost_usd": round(cost, 3), "file": str(path)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["submit", "status", "collect"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pairs", nargs="*", default=[], help="generations.jsonl=items.jsonl")
    parser.add_argument("--samples", default="0", help="comma-separated sample indexes to judge (0 = primary)")
    parser.add_argument("--limit", type=int, default=None, help="first N requests only (trial runs)")
    args = parser.parse_args()
    if args.command == "submit":
        submit(args.tag, args.pairs, {int(s) for s in args.samples.split(",")}, args.limit)
    elif args.command == "status":
        status(args.tag)
    else:
        collect(args.tag)


if __name__ == "__main__":
    main()
