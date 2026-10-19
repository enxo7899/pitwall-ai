"""Planted-mistake answers ("canaries") with known correct judgements, built by code from dev answer keys.

If the judge can't catch a changed number or a missing step that WE planted, its scores on real
answers can't be trusted. Seven defect types x 4 items = 28 canaries; target detection >= 90%.
"""

from __future__ import annotations

import random

FILLER = (
    " It is also worth remembering that every situation is a little different, and that taking a moment to "
    "think things through carefully is generally a sensible approach. Many people find it helpful to plan "
    "ahead, stay patient, and keep an open mind, since circumstances can change. In the end, what matters most "
    "is finding an approach that works for you, and being willing to adjust it over time as you learn more. "
    "Consistency, a calm mindset and a bit of preparation usually go a long way in situations like this one."
)

# A false but harmless aside: the judge must flag it as an error, but NOT as material (decision D2).
SLIP = " As a side note, the Great Wall of China is easily visible from the Moon with the naked eye."


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


def plain_answer(item: dict, drop: int | None = None) -> str:
    """A correct, unstyled answer composed from the answer key (optionally without key point `drop`)."""
    if item["reference_kind"] == "numeric":
        return f"The answer is {_fmt(item['gold'])} {item['unit']}.".replace(" .", ".")
    if item["reference_kind"] == "short_answer":
        return f"The answer is {item['short_answer']}."
    steps = [k["text"] for i, k in enumerate(item["key_points"]) if i != drop]
    return "Here's how: " + " ".join(f"{n}. {s}" for n, s in enumerate(steps, 1))


def make_canaries(dev_items: list[dict], per_type: int = 4, seed: int = 2026) -> list[dict]:
    """28 canaries: answer text plus the judgement a reliable judge must give."""
    rng = random.Random(seed)
    by_cat = {c: [i for i in dev_items if i["category"] == c] for c in ("quantitative", "factual", "procedure")}
    numeric = rng.sample(by_cat["quantitative"], per_type)
    factual = rng.sample([i for i in by_cat["factual"] if i["plausible_wrong_answer"]], per_type)
    steps = rng.sample([i for i in by_cat["procedure"] if sum(k["critical"] for k in i["key_points"]) >= 2], per_type)
    out = []
    for item in numeric:
        wrong = item["gold"] * 1.25 if item["gold"] else 7
        out.append(
            {
                "type": "number_changed",
                "item": item,
                "answer": plain_answer(item).replace(_fmt(item["gold"]), _fmt(wrong)),
                "expect": {"correct": False},
            }
        )
    for item in factual:
        out.append(
            {
                "type": "fact_contradicted",
                "item": item,
                "answer": f"The answer is {item['plausible_wrong_answer']}.",
                "expect": {"correct": False},
            }
        )
    for item in steps:
        first_critical = next(i for i, k in enumerate(item["key_points"]) if k["critical"])
        out.append(
            {
                "type": "critical_step_removed",
                "item": item,
                "answer": plain_answer(item, drop=first_critical),
                "expect": {"correct": False},
            }
        )
        out.append(
            {
                "type": "style_opener_only",
                "item": item,
                "answer": "Lights out and away we go! " + plain_answer(item),
                "expect": {"style": 1},
            }
        )
        out.append(
            {"type": "no_style", "item": item, "answer": plain_answer(item), "expect": {"style": 0, "correct": True}}
        )
        out.append(
            {
                "type": "padded_verbose",
                "item": item,
                "answer": plain_answer(item) + FILLER,
                "expect": {"helpful_not_above_no_style": True},
            }
        )
        out.append(
            {
                "type": "minor_slip",
                "item": item,
                "answer": plain_answer(item) + SLIP,
                "expect": {"correct_strict": False, "correct_material": True},
            }
        )
    return out


def check(canary: dict, result: dict, no_style_helpful: int | None) -> bool:
    """Did the judge give the expected judgement for this canary?"""
    expect = canary["expect"]
    if "helpful_not_above_no_style" in expect:
        return result["helpful"] is not None and no_style_helpful is not None and result["helpful"] <= no_style_helpful
    return all(result.get(key) == value for key, value in expect.items())
