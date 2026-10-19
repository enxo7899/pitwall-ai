"""Gate 3: generate training data with the self-hosted teacher and checker (vLLM, OpenAI-compatible API).

Runs on the GPU pod, standard library only (PYTHONPATH=src). Each stage reads the previous stage's file,
so a crash costs one stage, not the whole run.

    python scripts/training_data/teacher_pipeline.py questions --dir DIR [--per-cell 3]   # teacher server on :8000
    python scripts/training_data/teacher_pipeline.py answers   --dir DIR                  # teacher server on :8000
    python scripts/training_data/teacher_pipeline.py check     --dir DIR                  # checker server on :8000
    python scripts/training_data/teacher_pipeline.py select    --dir DIR [--n 500]        # no model: code gates + sampling
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pitwall.teacher import (
    CHECKER_ID,
    CHECKER_REVISION,
    CHECKER_SCHEMA,
    CHECKER_SYSTEM,
    NEUTRAL_SYSTEM,
    QUESTIONS_SCHEMA,
    REWRITE_SYSTEM,
    TEACHER_ID,
    TEACHER_REVISION,
    TEACHER_SAMPLING,
    cap_openers,
    checker_accepts,
    checker_prompt,
    code_gates,
    dedupe,
    diversity_report,
    find_leaks,
    parse_json_object,
    question_prompt,
    retry_prompt,
    rewrite_prompt,
    taxonomy_cells,
)

URL = "http://localhost:8000/v1/chat/completions"
EVAL_FILE = Path("data/drafts/eval_candidates.jsonl")
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
CHECKER_SAMPLING = {"temperature": 0.6, "top_p": 1.0, "reasoning_effort": "medium"}


def seed_for(*parts: str) -> int:
    """Per-request seed derived from the item, so any single row can be regenerated exactly."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def chat(body: dict, retries: int = 4) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            request = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 400:
                return {"error": error.read().decode()[:500]}
            time.sleep(5 * (attempt + 1))
    return {"error": "no response after retries"}


def reply(response: dict) -> tuple[str, str]:
    """(text, finish_reason); errors come back as empty text with the error as the reason."""
    if "error" in response:
        return "", f"error: {response['error']}"
    choice = response["choices"][0]
    return (choice["message"].get("content") or "").strip(), choice.get("finish_reason") or ""


def run_parallel(fn, items, workers: int = 64, label: str = "") -> list:
    results, start = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(fn, items), 1):
            results.append(result)
            if i % 50 == 0 or i == len(items):
                print(f"{label} {i}/{len(items)} ({time.time() - start:.0f}s)", flush=True)
    return results


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def stage_questions(out: Path, per_cell: int) -> None:
    cells = taxonomy_cells(per_cell)

    def ask(cell: dict) -> list[dict]:
        body = {
            "model": "teacher",
            "messages": [{"role": "user", "content": question_prompt(cell)}],
            "max_tokens": 1500,
            **TEACHER_SAMPLING,
            **NO_THINKING,
            "seed": seed_for("q", cell["category"], cell["domain"]),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "questions", "schema": QUESTIONS_SCHEMA},
            },
        }
        text, _ = reply(chat(body))
        parsed = parse_json_object(text) or {}
        return [
            {"category": cell["category"], "domain": cell["domain"], "question": q.strip()}
            for q in parsed.get("questions", [])[: cell["n"]]
            if isinstance(q, str) and q.strip()
        ]

    rows = [r for batch in run_parallel(ask, cells, label="questions") for r in batch]
    kept, pairs = dedupe([r["question"] for r in rows])
    rows = [rows[i] for i in kept]
    eval_items = read_jsonl(EVAL_FILE)
    leaks = find_leaks([r["question"] for r in rows], eval_items)
    clean = [{**r, "qid": f"t{n:04d}"} for n, r in enumerate(r for i, r in enumerate(rows) if i not in leaks)]
    write_jsonl(out / "questions.jsonl", clean)
    report = {
        "requested": sum(c["n"] for c in cells),
        "near_duplicates_dropped": len(pairs),
        "leaks_vs_eval_dropped": [{"question": rows[i]["question"], "eval_id": e} for i, e in leaks.items()],
        "kept": len(clean),
        "by_category": Counter(r["category"] for r in clean),
    }
    (out / "questions_report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if k != "leaks_vs_eval_dropped"}, indent=2), f"leaks={len(leaks)}"
    )


def stage_answers(out: Path) -> None:
    rows = read_jsonl(out / "questions.jsonl")

    def neutral(row: dict) -> dict:
        body = {
            "model": "teacher",
            "messages": [{"role": "system", "content": NEUTRAL_SYSTEM}, {"role": "user", "content": row["question"]}],
            "max_tokens": 900,
            **TEACHER_SAMPLING,
            **NO_THINKING,
            "seed": seed_for("n", row["qid"]),
        }
        text, finish = reply(chat(body))
        return {**row, "neutral": text, "neutral_finish": finish}

    rows = run_parallel(neutral, rows, label="neutral")

    def styled(row: dict) -> dict:
        if row["neutral_finish"] != "stop" or not row["neutral"]:
            return {**row, "styled": "", "styled_finish": "skipped"}
        body = {
            "model": "teacher",
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": rewrite_prompt(row["question"], row["neutral"])},
            ],
            "max_tokens": 1300,
            **TEACHER_SAMPLING,
            **NO_THINKING,
            "seed": seed_for("s", row["qid"]),
        }
        text, finish = reply(chat(body))
        return {**row, "styled": text, "styled_finish": finish}

    write_jsonl(out / "answers.jsonl", run_parallel(styled, rows, label="rewrite"))


def stage_retry(out: Path) -> None:
    """Rewrites the checker accepted but the length gate rejected get one more try with a hard word budget."""
    rows = [
        r
        for r in read_jsonl(out / "checked.jsonl")
        if r.get("styled_finish") == "stop"
        and checker_accepts(r.get("verdict"))[0]
        and code_gates(r["question"], r["neutral"], r["styled"], r["category"]) == ["too_long_vs_reference"]
    ]

    def again(row: dict) -> dict:
        body = {
            "model": "teacher",
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": retry_prompt(row["question"], row["neutral"], len(row["styled"].split()))},
            ],
            "max_tokens": 1300,
            **TEACHER_SAMPLING,
            **NO_THINKING,
            "seed": seed_for("s2", row["qid"]),
        }
        text, finish = reply(chat(body))
        return {
            **{k: v for k, v in row.items() if k not in ("verdict", "checker_finish")},
            "styled": text,
            "styled_finish": finish,
            "first_styled": row["styled"],
            "retry": 1,
        }

    write_jsonl(out / "answers_retry.jsonl", run_parallel(again, rows, label="retry"))


def stage_check(out: Path, src: str = "answers.jsonl", dst: str = "checked.jsonl") -> None:
    rows = read_jsonl(out / src)
    use_schema = {"on": True}

    def check(row: dict) -> dict:
        if row["styled_finish"] != "stop" or not row["styled"]:
            return {**row, "verdict": None, "checker_finish": "skipped"}
        body = {
            "model": "checker",
            "messages": [
                {"role": "system", "content": CHECKER_SYSTEM},
                {"role": "user", "content": checker_prompt(row["question"], row["neutral"], row["styled"])},
            ],
            "max_tokens": 4000,
            **CHECKER_SAMPLING,
            "seed": seed_for("c", row["qid"]),
        }
        if use_schema["on"]:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": CHECKER_SCHEMA},
            }
        text, finish = reply(chat(body))
        if finish.startswith("error") and use_schema["on"]:  # server without structured output for this model
            use_schema["on"] = False
            body.pop("response_format")
            text, finish = reply(chat(body))
        verdict = parse_json_object(text)
        if verdict is None and not finish.startswith("error"):  # one retry with a different seed
            body["seed"] += 1
            text, finish = reply(chat(body))
            verdict = parse_json_object(text)
        return {**row, "verdict": verdict, "checker_finish": finish}

    write_jsonl(out / dst, run_parallel(check, rows, workers=48, label="check"))


def stage_select(out: Path, n: int, seed: int = 2026) -> None:
    rows = read_jsonl(out / "checked.jsonl")
    if (out / "checked_retry.jsonl").exists():  # second attempts replace first attempts, by question id
        retried = {r["qid"]: r for r in read_jsonl(out / "checked_retry.jsonl")}
        rows = [retried.get(r["qid"], r) for r in rows]
    accepted, rejected = [], []
    for row in rows:
        reasons = []
        if row["neutral_finish"] != "stop":
            reasons.append(f"neutral_{row['neutral_finish'] or 'empty'}")
        elif row["styled_finish"] != "stop":
            reasons.append(f"styled_{row['styled_finish'] or 'empty'}")
        else:
            reasons += code_gates(row["question"], row["neutral"], row["styled"], row["category"])
            ok, why = checker_accepts(row.get("verdict"))
            if not ok:
                reasons.append(f"checker:{why}")
        (rejected if reasons else accepted).append({**row, "reject": reasons} if reasons else row)

    rng = random.Random(seed)
    rng.shuffle(accepted)
    accepted, capped = cap_openers(accepted, max_share=0.05)
    rejected += capped

    # Stratified: every serious example, then an even share per category, topped up from the rest.
    by_cat: dict[str, list[dict]] = {}
    for row in accepted:
        by_cat.setdefault(row["category"], []).append(row)
    chosen = list(by_cat.pop("serious", []))
    share = (n - len(chosen)) // max(len(by_cat), 1)
    leftovers = []
    for cat_rows in by_cat.values():
        chosen += cat_rows[:share]
        leftovers += cat_rows[share:]
    chosen += leftovers[: max(0, n - len(chosen))]
    chosen = chosen[:n]
    rng.shuffle(chosen)  # so any prefix (--limit for learning-curve runs) is a random, roughly stratified subset

    provenance = {
        "teacher": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "teacher_sampling": TEACHER_SAMPLING,
        "checker": CHECKER_ID,
        "checker_revision": CHECKER_REVISION,
        "checker_sampling": CHECKER_SAMPLING,
    }
    write_jsonl(
        out / "train.jsonl",
        [
            {
                "qid": r["qid"],
                "category": r["category"],
                "domain": r["domain"],
                "question": r["question"],
                "answer": r["styled"],
            }
            for r in chosen
        ],
    )
    # Accepted but not selected: same quality bar, never trained on -> the validation set for early stopping.
    chosen_ids = {r["qid"] for r in chosen}
    write_jsonl(
        out / "heldout.jsonl",
        [
            {
                "qid": r["qid"],
                "category": r["category"],
                "domain": r["domain"],
                "question": r["question"],
                "answer": r["styled"],
            }
            for r in accepted
            if r["qid"] not in chosen_ids
        ],
    )
    write_jsonl(out / "rejected.jsonl", rejected)
    reason_counts = Counter(
        reason.split(":")[0] if not reason.startswith("checker") else reason
        for r in rejected
        for reason in (r["reject"] if isinstance(r["reject"], list) else [r["reject"]])
    )
    report = {
        "generated": len(rows),
        "accepted_before_sampling": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / max(len(rows), 1), 3),
        "selected": len(chosen),
        "selected_by_category": Counter(r["category"] for r in chosen),
        "reject_reasons": reason_counts,
        "diversity_selected": diversity_report([r["styled"] for r in chosen]),
        "provenance": provenance,
    }
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    write_pilot(out, chosen, rejected, rng)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))


def write_pilot(out: Path, chosen: list[dict], rejected: list[dict], rng: random.Random) -> None:
    """Ten accepted examples (one per category first) and three rejected ones, for a human to read."""
    seen, pilot = set(), []
    for row in chosen:
        if row["category"] not in seen:
            seen.add(row["category"])
            pilot.append(row)
    pilot += [r for r in chosen if r not in pilot][: 10 - len(pilot)]
    lines = ["# Training-data pilot: 10 accepted, 3 rejected\n"]
    for r in pilot:
        lines += [
            f"## {r['qid']} [{r['category']} / {r['domain']}]",
            f"**Q:** {r['question']}\n",
            f"**Neutral (teacher):**\n\n{r['neutral']}\n",
            f"**Styled (teacher, accepted):**\n\n{r['styled']}\n",
        ]
    for r in rng.sample([r for r in rejected if r.get("styled")], min(3, len(rejected))):
        lines += [
            f"## REJECTED {r['qid']} [{r['category']}]: {r['reject']}",
            f"**Q:** {r['question']}\n",
            f"**Neutral:**\n\n{r['neutral']}\n",
            f"**Styled:**\n\n{r['styled']}\n",
            f"**Checker verdict:** `{json.dumps(r.get('verdict'))}`\n",
        ]
    (out / "pilot.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["questions", "answers", "check", "retry", "check-retry", "select"])
    parser.add_argument("--dir", required=True)
    parser.add_argument("--per-cell", type=int, default=3)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    {
        "questions": lambda: stage_questions(out, args.per_cell),
        "answers": lambda: stage_answers(out),
        "check": lambda: stage_check(out),
        "retry": lambda: stage_retry(out),
        "check-retry": lambda: stage_check(out, "answers_retry.jsonl", "checked_retry.jsonl"),
        "select": lambda: stage_select(out, args.n),
    }[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
