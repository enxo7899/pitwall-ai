"""Gate 5: the pre-registered analysis, in one re-runnable script (every number in the report comes from here).

Usage:
    uv run python scripts/evaluation/analyze_results.py
Reads the committed judgements and answers in experiments/results/ and the frozen evaluation set; writes
experiments/results/summary.json and experiments/results/summary.md. Running it twice gives identical files.
"""

from __future__ import annotations

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

J = Path("experiments/results/judgements")
OUT = Path("experiments/results")
ARMS = {  # arm label -> (judgement file, arm letter inside it, generations file)
    "A": (J / "exam-c500_judgements.jsonl", "A", Path("experiments/results/answers/test_A_B_C3epoch.jsonl")),
    "B": (J / "exam-c500_judgements.jsonl", "B", Path("experiments/results/answers/test_A_B_C3epoch.jsonl")),
    "C": (J / "exam-c315_judgements.jsonl", "C", Path("experiments/results/answers/test_C.jsonl")),
    "C_3epoch": (
        J / "exam-c500_judgements.jsonl",
        "C",
        Path("experiments/results/answers/test_A_B_C3epoch.jsonl"),
    ),  # sensitivity (D-1)
}


def split_of(row: dict) -> str:
    return Path(row["items"]).stem  # test / safety / probe


def load(arm: str, split: str) -> list[dict]:
    path, letter, _ = ARMS[arm]
    rows = [r for r in read_jsonl(path) if r["arm"] == letter and r["sample"] == 0 and split_of(r) == split]
    ids = [r["item_id"] for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate judgements for {arm}/{split}"
    return rows


def with_strict(rows: list[dict]) -> list[dict]:
    return [{**r, "correct": r.get("correct_strict", r["correct"])} for r in rows]


def pct(s: int, n: int) -> str:
    lo, hi = wilson_interval(s, n)
    return f"{s}/{n} ({100 * s / n:.1f}%, 95% CI {100 * lo:.1f}–{100 * hi:.1f})"


def arm_summary(rows: list[dict], gens: list[dict]) -> dict:
    s, n, _ = joint_success_rate(rows)
    answers = [g["answer"] for g in gens]
    return {
        "jsr": {"successes": s, "n": n, "text": pct(s, n)},
        "jsr_strict_rule": dict(zip(("successes", "n"), joint_success_rate(with_strict(rows))[:2], strict=True)),
        "components": {k: pct(p, n) for k, (p, n) in component_rates(rows).items()},
        "failures": dict(failure_breakdown(rows).most_common()),
        "manual_queue": sum(bool(r.get("refused")) for r in rows),
        "mean_style": round(statistics.mean(r["style"] for r in rows if r["style"] is not None), 2),
        "mean_helpful": round(statistics.mean(r["helpful"] for r in rows if r["helpful"] is not None), 2),
        "correct_by_category": {
            c: f"{sum(correct_pass(r) for r in rows if r['category'] == c)}/{sum(r['category'] == c for r in rows)}"
            for c in sorted({r["category"] for r in rows})
        },
        "jsr_by_category": {
            c: f"{joint_success_rate([r for r in rows if r['category'] == c])[0]}/{sum(r['category'] == c for r in rows)}"
            for c in sorted({r["category"] for r in rows})
        },
        "median_tokens": statistics.median(g["n_tokens"] for g in gens),
        "cut_off_at_limit": sum(not g["finished"] for g in gens),
        "f1_terms_per_100_words": round(
            100 * sum(len(f1_terms(a)) for a in answers) / max(1, sum(len(a.split()) for a in answers)), 2
        ),
        "diversity": diversity_report(answers),
    }


def paired(x_rows: list[dict], y_rows: list[dict], test) -> dict:
    x = {r["item_id"]: r for r in x_rows}
    y = {r["item_id"]: r for r in y_rows}
    ids = sorted(x)
    assert ids == sorted(y), "arms must cover the same questions"
    a, b = [test(x[i]) for i in ids], [test(y[i]) for i in ids]
    mean, lo, hi = paired_bootstrap_ci(a, b)
    verdict = "first arm better" if lo > 0 else "second arm better" if hi < 0 else "no reliable difference"
    return {
        "n": len(ids),
        "difference_pp": round(100 * mean, 1),
        "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
        "mcnemar": mcnemar_exact(a, b),
        "verdict": verdict,
    }


def joint(r: dict) -> bool:
    return style_pass(r) and correct_pass(r) and helpful_pass(r)


def main() -> None:
    test = {arm: load(arm, "test") for arm in ARMS}
    gens = {arm: [g for g in read_jsonl(ARMS[arm][2]) if g["arm"] == ARMS[arm][1] and g["sample"] == 0] for arm in ARMS}
    summary: dict = {
        "n_test": len(test["A"]),
        "sample": "sample 1 (index 0) per question, as pre-registered",
        "arms": {arm: arm_summary(test[arm], gens[arm]) for arm in ARMS},
        "comparisons": {},
    }
    for x, y in (("C", "B"), ("C", "A"), ("B", "A"), ("C", "C_3epoch")):
        summary["comparisons"][f"{x}_vs_{y}"] = {
            "joint_success": paired(test[x], test[y], joint),
            "joint_success_strict_rule": paired(with_strict(test[x]), with_strict(test[y]), joint),
            "style": paired(test[x], test[y], style_pass),
            "correct": paired(test[x], test[y], correct_pass),
            "helpful": paired(test[x], test[y], helpful_pass),
        }
    # Safety items: style is not required; the judge's "correct" already includes "no unsafe advice".
    summary["safety_judge_correct_and_safe"] = {
        arm: pct(sum(correct_pass(r) for r in load(arm, "safety")), len(load(arm, "safety"))) for arm in ("A", "B", "C")
    }
    summary["safety_concerns_flagged"] = {
        arm: sum(bool(r.get("safety_concern")) for r in load(arm, "safety")) for arm in ("A", "B", "C")
    }
    summary["probes"] = {
        arm: {r["item_id"]: {"style": r["style"], "helpful": r["helpful"]} for r in load(arm, "probe")}
        for arm in ("A", "B", "C")
    }

    retest_path = J / "retest_judgements.jsonl"
    if retest_path.exists():
        first = {(r["arm"], r["item_id"]): r for r in read_jsonl(J / "exam-c500_judgements.jsonl") if r["sample"] == 0}
        again = read_jsonl(retest_path)
        pairs = [(first[(r["arm"], r["item_id"])], r) for r in again]

        def agree(field) -> int:
            return sum(field(a) == field(b) for a, b in pairs)

        summary["judge_retest"] = {
            "n": len(pairs),
            "same_style_pass": agree(style_pass),
            "same_correct": agree(correct_pass),
            "same_helpful_pass": agree(helpful_pass),
            "same_joint": agree(joint),
            "same_exact_style_score": agree(lambda r: r["style"]),
            "same_exact_helpful_score": agree(lambda r: r["helpful"]),
        }
    canaries = Path("experiments/gate2/canaries.json")
    if canaries.exists():
        summary["judge_canaries"] = json.loads(canaries.read_text())["summary"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = [
        f"# PITWALL AI · test-set results (sample 1, n = {summary['n_test']} general questions)\n",
        "| Arm | Joint success | Style ≥ 2 | Correct | Helpful ≥ 2 | Median tokens | Cut off |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, a in summary["arms"].items():
        c = a["components"]
        lines.append(
            f"| {arm} | {a['jsr']['text']} | {c['style']} | {c['correct']} | {c['helpful']} | {a['median_tokens']} | {a['cut_off_at_limit']} |"
        )
    lines += [
        "",
        "| Comparison (paired, 10,000 bootstrap resamples) | Joint success Δ (pp) | 95% CI | McNemar p | Verdict |",
        "|---|---|---|---|---|",
    ]
    for name, comp in summary["comparisons"].items():
        j = comp["joint_success"]
        lines.append(
            f"| {name} | {j['difference_pp']:+.1f} | {j['ci95_pp'][0]:+.1f} to {j['ci95_pp'][1]:+.1f} | {j['mcnemar']['p_value']:.3f} | {j['verdict']} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
