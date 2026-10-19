"""Export the committed results as JSON for the website, so every number on it comes from the repository.

Usage:
    uv run python scripts/tools/export_web_data.py
Writes web/src/data/site.json (charts and headline numbers) and web/public/data/explorer.json
(all 120 test questions with every arm's answer and the judge's verdict). Rates and intervals are
computed with the same functions as the analysis (`pitwall.metrics`).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from pitwall.config import DECODING, MODEL_ID, MODEL_REVISION
from pitwall.metrics import component_rates, joint_success_rate, style_pass, wilson_interval

RESULTS = Path("experiments/results")
SITE = Path("web/src/data/site.json")
EXPLORER = Path("web/public/data/explorer.json")
JUDGED = {  # the same files the analysis and the audit read
    "A": RESULTS / "judgements/exam-c500_judgements.jsonl",
    "B": RESULTS / "judgements/exam-c500_judgements.jsonl",
    "C": RESULTS / "judgements/exam-c315_judgements.jsonl",
}
ANSWERS = {
    "A": RESULTS / "answers/test_A_B_C3epoch.jsonl",
    "B": RESULTS / "answers/test_A_B_C3epoch.jsonl",
    "C": RESULTS / "answers/test_C.jsonl",
}
SHOWCASE = ["tech-13", "unce-16"]  # the two examples discussed in the README, then one success per category


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(successes: int, n: int) -> dict:
    low, high = wilson_interval(successes, n)
    return {"s": successes, "n": n, "rate": successes / n, "low": low, "high": high}


def fraction(text: str) -> tuple[int, int]:
    s, n = re.match(r"(\d+)/(\d+)", text).groups()
    return int(s), int(n)


def test_judgements() -> dict[str, dict[str, dict]]:
    judged = {}
    for arm, path in JUDGED.items():
        judged[arm] = {
            r["item_id"]: r
            for r in jsonl(path)
            if r["arm"] == arm and r["sample"] == 0 and r["items"].endswith("test.jsonl")
        }
    return judged


def arms_block(judged: dict) -> dict:
    out = {}
    for arm, rows in judged.items():
        records = list(rows.values())
        s, n, _ = joint_success_rate(records)
        parts = component_rates(records)
        out[arm] = {
            "joint": rate(s, n),
            **{name: rate(*parts[name]) for name in ("style", "correct", "helpful")},
            "mean_style": sum(r["style"] or 0 for r in records) / n,
        }
    return out


def training_block() -> dict:
    log = jsonl(Path("experiments/gate4/training_log_sweep.jsonl"))
    report = json.loads(Path("experiments/gate4/train_report_sweep.json").read_text())
    selection = json.loads(Path("experiments/gate4/epoch_selection.json").read_text())
    curve = jsonl(RESULTS / "judgements/curve_judgements.jsonl")
    dev500 = json.loads(Path("experiments/gate4/dev_summary_c500.json").read_text())["arms"]["C"]
    points = []
    for size, source in ((125, "c125_dev"), (250, "c250_dev")):
        rows = [r for r in curve if r["source"] == source]
        parts = component_rates(rows)
        points.append(
            {
                "examples": size,
                "style": parts["style"][0],
                "joint": joint_success_rate(rows)[0],
                "correct": parts["correct"][0],
            }
        )
    points.append(
        {
            "examples": 500,
            "style": dev500["components"]["style"]["passes"],
            "joint": dev500["jsr"]["successes"],
            "correct": dev500["components"]["correct"]["passes"],
        }
    )
    return {
        "steps": [
            {
                "step": r["step"],
                "epoch": r["epoch"],
                "loss": r["loss"],
                "lr": r["learning_rate"],
                "grad_norm": r["grad_norm"],
                "accuracy": r["mean_token_accuracy"],
            }
            for r in log
            if r["kind"] == "train"
        ],
        "validation": [
            {"step": r["step"], "epoch": r["epoch"], "loss": r["eval_loss"], "accuracy": r["eval_mean_token_accuracy"]}
            for r in log
            if r["kind"] == "eval"
        ],
        "steps_per_epoch": report["steps_per_epoch"],
        "total_steps": report["steps"],
        "minutes": report["minutes"],
        "peak_memory_gb": report["peak_memory_gb"],
        "gpu": report["gpu"],
        "trainable_params": report["trainable_params"],
        "total_params": report["total_params"],
        "epochs": [
            {k: c[k] for k in ("epoch", "checkpoint", "n", "jsr", "style", "correct", "helpful", "validation_loss")}
            for c in selection["candidates"]
        ],
        "chosen_epoch": selection["chosen"]["epoch"],
        "learning_curve": points,
    }


def explorer_rows(judged: dict) -> list[dict]:
    items = jsonl(Path("data/eval/test.jsonl"))
    answers = {
        arm: {(r["item_id"]): r for r in jsonl(path) if r["arm"] == arm and r["sample"] == 0}
        for arm, path in ANSWERS.items()
    }
    rows = []
    for item in items:
        arms = {}
        for arm in "ABC":
            j, a = judged[arm][item["id"]], answers[arm][item["id"]]
            form = j["form"]
            status = {p["point"]: p["status"] for p in form["key_points"]}
            arms[arm] = {
                "answer": a["answer"],
                "tokens": a["n_tokens"],
                "finished": a["finished"],
                "style": j["style"],
                "helpful": j["helpful"],
                "correct": j["correct"],
                "joint": bool(style_pass(j) and j["correct"] is True and (j["helpful"] or 0) >= 2),
                "key_points": [status.get(i, "missing") for i in range(1, len(item["key_points"]) + 1)],
                "style_evidence": form["style_evidence"],
                "helpful_evidence": form["helpfulness_evidence"],
                "extracted": form["extracted_answer"],
                "error_quote": form["error_quote"] if form["other_factual_error"] else "",
                "error_serious": bool(form["other_factual_error"] and form["error_is_material"]),
            }
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "handwritten": item["author"].startswith("owner"),  # author label used in the frozen data
                "kind": item["reference_kind"],
                "answer_key": item["stated_answer"],
                "key_points": [{"text": k["text"], "critical": k["critical"]} for k in item["key_points"]],
                "arms": arms,
            }
        )
    return rows


def timeline() -> list[dict]:
    """The protocol's key commits, from docs/TIMELINE.md."""
    rows = re.findall(r"^\| (.+?) \| `(\w+)` \| (.+?) \|$", Path("docs/TIMELINE.md").read_text(), flags=re.M)
    return [{"when": when, "commit": commit, "event": event} for when, commit, event in rows]


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    judged = test_judgements()
    explorer = explorer_rows(judged)
    q = json.loads(Path("data/train/questions_report.json").read_text())
    d = json.loads(Path("data/train/data_report.json").read_text())
    evals = {split: jsonl(Path(f"data/eval/{split}.jsonl")) for split in ("test", "dev", "safety", "probe")}
    tone = json.loads((RESULTS / "safety_tone_check.json").read_text())["arms"]
    latency = json.loads((RESULTS / "latency.json").read_text())
    agreement = json.loads((RESULTS / "human_agreement.json").read_text())
    adjudication = json.loads((RESULTS / "adjudication.json").read_text())

    successes = [r for r in explorer if r["arms"]["C"]["joint"]]
    showcase = SHOWCASE + [
        next(r["id"] for r in successes if r["category"] == cat and r["id"] not in SHOWCASE)
        for cat in ("procedure", "quantitative", "advice", "writing")
        if any(r["category"] == cat for r in successes)
    ]
    arms = arms_block(judged)
    for arm in arms:  # the sensitivity rule: any extra factual error fails the answer
        strict = summary["arms"][arm]["jsr_strict_rule"]
        arms[arm]["joint_strict"] = rate(strict["successes"], strict["n"])
    site = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "decoding": DECODING},
        "arms": arms,
        "comparisons": {
            pair: {
                metric: {
                    "diff": summary["comparisons"][pair][metric]["difference_pp"],
                    "low": summary["comparisons"][pair][metric]["ci95_pp"][0],
                    "high": summary["comparisons"][pair][metric]["ci95_pp"][1],
                    "p": summary["comparisons"][pair][metric]["mcnemar"]["p_value"],
                    "only_first": summary["comparisons"][pair][metric]["mcnemar"]["c_only"],
                    "only_second": summary["comparisons"][pair][metric]["mcnemar"]["b_only"],
                    "verdict": summary["comparisons"][pair][metric]["verdict"],
                }
                for metric in ("joint_success", "style", "correct", "helpful", "joint_success_strict_rule")
            }
            for pair in ("C_vs_B", "C_vs_A", "B_vs_A")
        },
        "categories": {
            arm: {
                cat: {
                    "correct": fraction(summary["arms"][arm]["correct_by_category"][cat])[0],
                    "joint": fraction(summary["arms"][arm]["jsr_by_category"][cat])[0],
                    "n": 15,
                }
                for cat in summary["arms"][arm]["correct_by_category"]
            }
            for arm in "ABC"
        },
        "judge": {
            "canaries": summary["judge_canaries"]["by_type"],
            "canaries_caught": summary["judge_canaries"]["detected"],
            "canaries_total": summary["judge_canaries"]["total"],
            "retest": summary["judge_retest"],
            "human": {  # blind pass/fail agreement with my labels, then my review of every disputed score
                "n": agreement["n_labelled"],
                "kappa_style": agreement["style"]["pass_kappa"],
                "kappa_correct": agreement["correct"]["cohen_kappa"],
                "kappa_helpful": agreement["helpful"]["pass_kappa"],
                "disputed": adjudication["disputed_scores"],
                "sided_with_judge": adjudication["sided_with_judge"],
            },
        },
        "safety": {
            "tone_acceptable": {arm: fraction(tone[arm]["acceptable"]) for arm in "ABC"},
            "judge_correct_and_safe": {arm: fraction(summary["safety_judge_correct_and_safe"][arm]) for arm in "ABC"},
        },
        "latency": {arm: latency["arms"][arm] for arm in "ABC"}
        | {"device": latency["device"], "new_tokens": latency["new_tokens"]},
        "data": {
            "requested": q["requested"],
            "near_duplicates": q["near_duplicates_dropped"],
            "eval_lookalikes": len(q["leaks_vs_eval_dropped"]),
            "answered": d["generated"],
            "accepted": d["accepted_before_sampling"],
            "selected": d["selected"],
            "heldout": d["accepted_before_sampling"] - d["selected"],
            "by_category": d["selected_by_category"],
            "reject_reasons": d["reject_reasons"],
            "max_opener_share": d["diversity_selected"]["max_opener_share"],
            "distinct_f1_terms": d["diversity_selected"]["distinct_f1_terms"],
            "teacher": d["provenance"]["teacher"],
            "checker": d["provenance"]["checker"],
        },
        "eval_set": {
            "splits": {split: len(rows) for split, rows in evals.items()},
            "test_by_category": dict(Counter(r["category"] for r in evals["test"])),
            "handwritten": sum(r["handwritten"] for r in explorer),
        },
        "training": training_block(),
        "showcase": [r for r in explorer if r["id"] in showcase],
        "timeline": timeline(),
    }
    SITE.parent.mkdir(parents=True, exist_ok=True)
    EXPLORER.parent.mkdir(parents=True, exist_ok=True)
    SITE.write_text(json.dumps(site, indent=1, ensure_ascii=False) + "\n")
    EXPLORER.write_text(json.dumps(explorer, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {SITE} ({SITE.stat().st_size // 1024} KB) and {EXPLORER} ({EXPLORER.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
