"""Rules for hand-written questions, review decisions and freezing (no API calls)."""

from __future__ import annotations

import pytest

from pitwall.eval_assembly import (
    apply_decisions,
    canonical_jsonl,
    check_ready,
    parse_handwritten_questions,
    place_handwritten_items,
)


def test_parse_handwritten_questions_ignores_comments_and_needs_headings():
    text = "# intro\n[procedure]\n# e.g. example\nhow do i descale a kettle\n\n[uncertainty]\nWhy did Einstein win two Nobels?\n"
    assert parse_handwritten_questions(text) == [
        {"category": "procedure", "question": "how do i descale a kettle"},
        {"category": "uncertainty", "question": "Why did Einstein win two Nobels?"},
    ]
    with pytest.raises(ValueError):
        parse_handwritten_questions("a question before any heading")


def test_handwritten_items_replace_generated_test_items():
    drafts = [
        {"id": f"proc-{i}", "category": "procedure", "split": "test", "author": "claude (draft)"} for i in range(3)
    ]
    placed = place_handwritten_items(
        drafts, [{"id": "own-0", "category": "procedure", "question": "q", "author": "owner"}]
    )
    assert sum(r["split"] == "test" for r in placed) == 3  # quota unchanged
    assert sum(r["split"] == "spare" for r in placed) == 1
    assert next(r for r in placed if r["id"] == "own-0")["split"] == "test"


def test_decisions_verify_fix_and_drop_with_spare_refill():
    rows = [
        {"id": "a", "category": "factual", "split": "test", "status": "unverified"},
        {"id": "b", "category": "factual", "split": "test", "status": "unverified"},
        {"id": "c", "category": "factual", "split": "spare", "status": "unverified"},
    ]
    out = {r["id"]: r for r in apply_decisions(rows, {"a": {"decision": "ok"}, "b": {"decision": "drop"}})}
    assert out["a"]["status"] == "verified"
    assert out["b"]["split"] == "dropped"
    assert out["c"]["split"] == "test" and out["c"]["status"] == "unverified"  # promoted, still needs review


def test_freeze_refuses_unverified_or_short_sets():
    problems = check_ready([{"id": "a", "category": "factual", "split": "test", "status": "unverified"}])
    assert any("factual/test" in p for p in problems) and any("not verified" in p for p in problems)


def test_fingerprint_input_is_order_independent():
    a = [{"id": "2", "x": 1}, {"id": "1", "y": 2}]
    assert canonical_jsonl(a) == canonical_jsonl(list(reversed(a)))
