"""Gate 5: turn judgements into the numbers the report quotes (every number as n/N with an interval).

Usage:
    uv run python scripts/evaluation/summarize_run.py --judgements outputs/RUN/dev_judgements.jsonl \
        --generations outputs/RUN/dev_generations.jsonl --out outputs/RUN/dev_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pitwall.eval_items import read_jsonl
from pitwall.metrics import (
    component_rates,
    correct_pass,
    failure_breakdown,
    helpful_pass,
    joint_success_rate,
    mcnemar_exact,
    paired_bootstrap_ci,
    style_pass,
    wilson_interval,
)
from pitwall.teacher import diversity_report, f1_terms


def paired(judgements: list[dict], arm_x: str, arm_y: str, test) -> dict:
    """Item-level paired comparison (sample 0) of a pass/fail criterion between two arms."""
    by = {(j["arm"], j["item_id"]): j for j in judgements if j["sample"] == 0}
    ids = sorted({i for a, i in by if a == arm_x} & {i for a, i in by if a == arm_y})
    x, y = [test(by[(arm_x, i)]) for i in ids], [test(by[(arm_y, i)]) for i in ids]
    mean, low, high = paired_bootstrap_ci(x, y)
    return {
        "n_items": len(ids),
        "difference": round(mean, 3),
        "ci95": [round(low, 3), round(high, 3)],
        "mcnemar": mcnemar_exact(x, y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    judgements = read_jsonl(Path(args.judgements))
    generations = read_jsonl(Path(args.generations))
    arms = sorted({j["arm"] for j in judgements})

    summary: dict = {"arms": {}, "comparisons": {}, "refused_by_judge": sum(j["refused"] for j in judgements)}
    for arm in arms:
        records = [j for j in judgements if j["arm"] == arm]
        answers = [g["answer"] for g in generations if g["arm"] == arm]
        s, n, rate = joint_success_rate(records)
        jsr = {
            "successes": s,
            "n": n,
            "rate": round(rate, 3),
            "wilson95": [round(v, 3) for v in wilson_interval(s, n)],
        }
        summary["arms"][arm] = {
            "jsr": jsr,
            "components": {
                k: {
                    "passes": p,
                    "n": n,
                    "rate": round(p / n, 3),
                    "wilson95": [round(v, 3) for v in wilson_interval(p, n)],
                }
                for k, (p, n) in component_rates(records).items()
            },
            "mean_style": round(statistics.mean(j["style"] for j in records if j["style"] is not None), 2),
            "mean_helpful": round(statistics.mean(j["helpful"] for j in records if j["helpful"] is not None), 2),
            "failures": dict(failure_breakdown(records).most_common()),
            "correct_by_category": {
                c: f"{sum(correct_pass(j) for j in records if j['category'] == c)}/"
                f"{sum(j['category'] == c for j in records)}"
                for c in sorted({j["category"] for j in records})
            },
            "tokens_median": statistics.median(g["n_tokens"] for g in generations if g["arm"] == arm),
            "unfinished": sum(not g["finished"] for g in generations if g["arm"] == arm),
            "f1_terms_per_100_words": round(
                100 * sum(len(f1_terms(a)) for a in answers) / max(1, sum(len(a.split()) for a in answers)), 2
            ),
            "diversity": diversity_report(answers),
        }
    for x, y in (("C", "B"), ("C", "A"), ("B", "A")):
        if x in arms and y in arms:
            summary["comparisons"][f"{x}_vs_{y}"] = {
                name: paired(judgements, x, y, test)
                for name, test in (("style", style_pass), ("correct", correct_pass), ("helpful", helpful_pass))
            }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
