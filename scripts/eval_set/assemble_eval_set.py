"""Evaluation-set assembly: add hand-written questions, apply review decisions, freeze with fingerprints.

Usage:
    uv run --env-file .env python scripts/eval_set/assemble_eval_set.py add-handwritten   # drafts answer keys for the hand-written questions
    uv run python scripts/eval_set/assemble_eval_set.py apply-review data/review/review_decisions.json
    uv run python scripts/eval_set/assemble_eval_set.py freeze                      # refuses unless every item is verified
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pitwall.eval_assembly import (
    apply_decisions,
    canonical_jsonl,
    check_ready,
    parse_handwritten_questions,
    place_handwritten_items,
    sha256_file,
)
from pitwall.eval_items import read_jsonl, write_jsonl

CANDIDATES = Path("data/drafts/eval_candidates.jsonl")
HANDWRITTEN_FILE = Path("data/authoring/my_questions.txt")
FROZEN = Path("data/eval")


def add_handwritten() -> None:
    import anthropic
    from draft_eval_set import KEY_RULES, MODEL, PRICE_PER_MTOK, SYSTEM  # same folder

    from pitwall.eval_items import DraftBatch, numeric_check
    from pitwall.eval_spec import CATEGORIES

    handwritten = parse_handwritten_questions(HANDWRITTEN_FILE.read_text(encoding="utf-8"))
    if len(handwritten) < 25:
        sys.exit(f"only {len(handwritten)} questions in {HANDWRITTEN_FILE}; the protocol requires at least 25")
    client, items, usage = anthropic.Anthropic(), [], {"input_tokens": 0, "output_tokens": 0}
    for category in sorted({q["category"] for q in handwritten}):
        qs = [q["question"] for q in handwritten if q["category"] == category]
        kind = CATEGORIES[category][1] if category in CATEGORIES else "key_points"
        prompt = (
            "Write the answer key for each of these EXACT questions, in order, copying each question text "
            "unchanged into the 'question' field.\n"
            + "\n".join(f"{i}. {q}" for i, q in enumerate(qs, 1))
            + f"\n\nAnswer-key rules: {KEY_RULES[kind]}\nFill every field; use '' or [] where it doesn't apply."
        )
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=DraftBatch,
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        for n, (q, drafted) in enumerate(zip(qs, response.parsed_output.items, strict=True)):
            row = {
                "id": f"own-{category[:4]}-{n:02d}",
                "category": category,
                "reference_kind": "safety" if category == "safety" else kind,
                "author": "owner (question) + claude-opus-5-5 (answer key)",
                "status": "unverified",
                **drafted.model_dump(),
                "question": q,
                "check_near_duplicate_of": [],
            }
            if row["reference_kind"] == "numeric":
                row.update(numeric_check(drafted.expression, drafted.stated_answer))
            items.append(row)
    write_jsonl(CANDIDATES, place_handwritten_items(read_jsonl(CANDIDATES), items))
    cost = (
        usage["input_tokens"] / 1e6 * PRICE_PER_MTOK["input"] + usage["output_tokens"] / 1e6 * PRICE_PER_MTOK["output"]
    )
    Path("data/drafts/handwritten_drafting_log.json").write_text(
        json.dumps({"model": MODEL, "usage": usage, "cost_usd": round(cost, 3)}, indent=2)
    )
    print(f"added {len(items)} hand-written questions (${cost:.2f}); rebuild the review sheet next")


def apply_review(decisions_path: str) -> None:
    decisions = json.loads(Path(decisions_path).read_text())
    rows = apply_decisions(read_jsonl(CANDIDATES), decisions)
    write_jsonl(CANDIDATES, rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("verified", "needs_fix", "dropped", "unverified")}
    print(json.dumps(counts))


def freeze() -> None:
    rows = read_jsonl(CANDIDATES)
    problems = check_ready(rows)
    if problems:
        sys.exit("not ready to freeze:\n- " + "\n- ".join(problems))
    FROZEN.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ("dev", "test", "safety", "probe"):
        path = FROZEN / f"{split}.jsonl"
        path.write_text(canonical_jsonl([r for r in rows if r["split"] == split]), encoding="utf-8")
        hashes[path.name] = sha256_file(path)
    (FROZEN / "hashes.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "add-handwritten":
        add_handwritten()
    elif command == "apply-review" and len(sys.argv) == 3:
        apply_review(sys.argv[2])
    elif command == "freeze":
        freeze()
    else:
        sys.exit(__doc__)
