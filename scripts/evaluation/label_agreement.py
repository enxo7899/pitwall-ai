"""Gate 5: how often do the blind human labels agree with the judge? The human check on the judge.

Usage (after exporting blind_labels.json from data/labels/label_sheet.html):
    uv run python scripts/evaluation/label_agreement.py --labels data/labels/blind_labels.json
Writes experiments/results/human_agreement.json (agreement, Cohen's kappa on the 0-3 scores and on
pass/fail at >= 2, every disagreement listed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitwall.eval_items import read_jsonl
from pitwall.metrics import cohen_kappa, quadratic_weighted_kappa

JUDGEMENTS = {
    "A": Path("experiments/results/judgements/exam-c500_judgements.jsonl"),
    "B": Path("experiments/results/judgements/exam-c500_judgements.jsonl"),
    "C": Path("experiments/results/judgements/exam-c315_judgements.jsonl"),
}
OUT = Path("experiments/results/human_agreement.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/labels/blind_labels.json")
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text())
    key = json.loads(Path("data/labels/blind_key.json").read_text())
    judged = {}
    for arm, path in JUDGEMENTS.items():
        for r in read_jsonl(path):
            if r["arm"] == arm and r["sample"] == 0:
                judged[(arm, r["item_id"])] = r

    pairs, safety, missing = [], [], []
    for card, meta in key.items():
        human = labels.get(card, {})
        judge = judged[(meta["arm"], meta["item_id"])]
        if meta["part"] == "label":
            if not all(f in human for f in ("style", "helpful", "correct")):
                missing.append(card)
                continue
            pairs.append(
                {
                    "card": card,
                    **meta,
                    "human": human,
                    "judge": {
                        "style": judge["style"],
                        "helpful": judge["helpful"],
                        "correct": "yes" if judge["correct"] else "no",
                    },
                }
            )
        else:
            if "safe" not in human:
                missing.append(card)
                continue
            safety.append(
                {
                    "card": card,
                    **meta,
                    "human_safe": human["safe"],
                    "judge_correct_and_safe": "yes" if judge["correct"] else "no",
                    "judge_safety_concern": judge.get("safety_concern"),
                }
            )

    def h(field: str) -> list:
        return [p["human"][field] for p in pairs]

    def j(field: str) -> list:
        return [p["judge"][field] for p in pairs]

    def passed(scores: list) -> list[bool]:
        return [x is not None and x >= 2 for x in scores]

    result = {
        "n_labelled": len(pairs),
        "n_safety": len(safety),
        "missing_cards": missing,
        "style": {
            "exact_agreement": sum(a == b for a, b in zip(h("style"), j("style"), strict=True)),
            "pass_agreement": sum(a == b for a, b in zip(passed(h("style")), passed(j("style")), strict=True)),
            "quadratic_weighted_kappa": round(quadratic_weighted_kappa(h("style"), j("style")), 3),
            "pass_kappa": round(cohen_kappa(passed(h("style")), passed(j("style"))), 3),
        },
        "helpful": {
            "exact_agreement": sum(a == b for a, b in zip(h("helpful"), j("helpful"), strict=True)),
            "pass_agreement": sum(a == b for a, b in zip(passed(h("helpful")), passed(j("helpful")), strict=True)),
            "quadratic_weighted_kappa": round(quadratic_weighted_kappa(h("helpful"), j("helpful")), 3),
            "pass_kappa": round(cohen_kappa(passed(h("helpful")), passed(j("helpful"))), 3),
        },
        "correct": {
            "agreement": sum(a == b for a, b in zip(h("correct"), j("correct"), strict=True)),
            "cohen_kappa": round(cohen_kappa(h("correct"), j("correct")), 3),
        },
        "safety": {
            "agreement": sum(s["human_safe"] == s["judge_correct_and_safe"] for s in safety),
            "human_unsafe": [s["card"] for s in safety if s["human_safe"] == "no"],
        },
        "disagreements": [
            p
            for p in pairs
            if p["human"]["correct"] != p["judge"]["correct"]
            or abs(p["human"]["style"] - (p["judge"]["style"] or 0)) >= 2
            or abs(p["human"]["helpful"] - (p["judge"]["helpful"] or 0)) >= 2
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=2),
        f"\n{len(result['disagreements'])} disagreements listed in {OUT}",
    )


if __name__ == "__main__":
    main()
