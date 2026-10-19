"""Assembling the evaluation set: hand-written questions, review decisions, and the final freeze.

Pure functions (no API calls), so the rules are unit-tested; the scripts in scripts/ do the I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pitwall.eval_spec import CATEGORIES, PER_CATEGORY, PROBE_TYPES, SAFETY_PER_TYPE, SAFETY_TYPES

SECTIONS = set(CATEGORIES) | {"safety"}


def parse_handwritten_questions(text: str) -> list[dict]:
    """Questions from data/authoring/my_questions.txt: one per line under a [category] heading."""
    questions, section = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        heading = re.fullmatch(r"\[(\w+)\]", line)
        if heading:
            section = heading.group(1)
            continue
        if section not in SECTIONS:
            raise ValueError(f"question outside a known [category]: {line!r}")
        questions.append({"category": section, "question": line})
    return questions


def place_handwritten_items(candidates: list[dict], handwritten: list[dict]) -> list[dict]:
    """Put hand-written items into the test split (or safety), moving the displaced generated items to 'spare'."""
    rows = [dict(r) for r in candidates]
    for item in handwritten:
        split = "safety" if item["category"] == "safety" else "test"
        displaced = next(
            (
                r
                for r in rows
                if r["category"] == item["category"]
                and r["split"] == split
                and not r["author"].startswith("owner")  # author label used in the frozen data
            ),
            None,
        )
        if displaced is None:
            raise ValueError(f"no drafted {split} item left to replace in {item['category']}")
        displaced["split"] = "spare"
        rows.append({**item, "split": split})
    return rows


def apply_decisions(rows: list[dict], decisions: dict[str, dict]) -> list[dict]:
    """OK -> verified; Fix -> needs_fix (with note); Drop -> dropped and refilled from a spare of the same category."""
    out = [dict(r) for r in rows]
    for r in out:
        decision = decisions.get(r["id"], {}).get("decision")
        if decision == "ok":
            r["status"] = "verified"
        elif decision == "fix":
            r["status"], r["review_note"] = "needs_fix", decisions[r["id"]].get("note", "")
        elif decision == "drop":
            r["status"], old_split = "dropped", r["split"]
            r["split"] = "dropped"
            spare = next((s for s in out if s["split"] == "spare" and s["category"] == r["category"]), None)
            if spare:
                spare["split"] = old_split  # promoted spares still need review (status stays unverified)
    return out


def check_ready(rows: list[dict]) -> list[str]:
    """Reasons the set cannot be frozen yet (empty list = ready)."""
    problems = []
    for category in CATEGORIES:
        for split, quota in PER_CATEGORY.items():
            n = sum(r["category"] == category and r["split"] == split for r in rows)
            if n != quota:
                problems.append(f"{category}/{split}: {n} items, need {quota}")
    if sum(r["split"] == "safety" for r in rows) != SAFETY_PER_TYPE * len(SAFETY_TYPES):
        problems.append("safety: wrong count")
    if sum(r["split"] == "probe" for r in rows) != sum(PROBE_TYPES.values()):
        problems.append("probe: wrong count")
    unverified = [
        r["id"] for r in rows if r["split"] in ("dev", "test", "safety", "probe") and r["status"] != "verified"
    ]
    if unverified:
        problems.append(f"{len(unverified)} items not verified, e.g. {unverified[:5]}")
    return problems


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_jsonl(rows: list[dict]) -> str:
    """Stable serialisation (sorted keys, sorted by id) so the fingerprint depends only on content."""
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r["id"])
    )
