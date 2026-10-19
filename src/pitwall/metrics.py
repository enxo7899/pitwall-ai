"""Evaluation metrics: pure functions over per-question records.

A record is one judged answer, e.g. {"id": "quan-03", "arm": "C", "style": 3, "helpful": 2, "correct": True}.
A missing score (None: crashed generation, unparseable or refused judgement) counts as a FAIL, never
as a removal: the denominator is always every question in the set (pre-registered, §5).
"""

from __future__ import annotations

import math
import random
from collections import Counter

STYLE_PASS = 2  # style 0-3: 2 = "F1 language runs through most of the answer"
HELPFUL_PASS = 2  # helpfulness 0-3: 2 = "answers adequately; the user can act on it"


def style_pass(record: dict) -> bool:
    return record.get("style") is not None and record["style"] >= STYLE_PASS


def helpful_pass(record: dict) -> bool:
    return record.get("helpful") is not None and record["helpful"] >= HELPFUL_PASS


def correct_pass(record: dict) -> bool:
    return record.get("correct") is True


def joint_success_rate(records: list[dict]) -> tuple[int, int, float]:
    """The primary metric: (successes, total, rate).

    A record succeeds only if it passes style AND correctness AND helpfulness. The total is every
    record, including ones with missing scores, so a crash or refusal counts as a failure.
    """
    total = len(records)  # every answer counts, even ones with missing scores
    successes = sum(1 for r in records if style_pass(r) and correct_pass(r) and helpful_pass(r))
    rate = successes / total if total else 0.0  # guard: an empty list would divide by zero
    return successes, total, rate


def component_rates(records: list[dict]) -> dict[str, tuple[int, int]]:
    """(passes, total) for each criterion, over all records."""
    n = len(records)
    return {
        "style": (sum(map(style_pass, records)), n),
        "correct": (sum(map(correct_pass, records)), n),
        "helpful": (sum(map(helpful_pass, records)), n),
    }


def failure_breakdown(records: list[dict]) -> Counter:
    """Which criteria each record failed: 'none', 'style', 'correct', 'helpful', or a combination like 'correct+style'."""
    counts: Counter = Counter()
    for r in records:
        failed = [
            name
            for name, ok in (("correct", correct_pass(r)), ("helpful", helpful_pass(r)), ("style", style_pass(r)))
            if not ok
        ]
        counts["+".join(failed) or "none"] += 1
    return counts


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a single proportion (better than ±1.96·SE for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_ci(
    c: list[bool], b: list[bool], n_resamples: int = 10_000, seed: int = 2026, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(observed difference, lower, upper) for mean(c) - mean(b) on the SAME questions.

    Questions are resampled as whole pairs, so each question's C and B outcomes stay together.
    """
    if len(c) != len(b) or not c:
        raise ValueError("c and b must be paired lists of equal, non-zero length")
    n = len(c)
    diffs = [int(ci) - int(bi) for ci, bi in zip(c, b, strict=True)]
    rng = random.Random(seed)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lower = stats[int((alpha / 2) * n_resamples)]
    upper = stats[int((1 - alpha / 2) * n_resamples) - 1]
    return (sum(diffs) / n, lower, upper)


def mcnemar_exact(c: list[bool], b: list[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test: only questions where C and B disagree carry information."""
    c_only = sum(ci and not bi for ci, bi in zip(c, b, strict=True))  # C passed, B failed
    b_only = sum(bi and not ci for ci, bi in zip(c, b, strict=True))  # B passed, C failed
    n = c_only + b_only
    if n == 0:
        return {"c_only": 0, "b_only": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(c_only, b_only) + 1)) / 2**n
    return {"c_only": c_only, "b_only": b_only, "p_value": min(1.0, 2 * tail)}


def cohen_kappa(a: list, b: list) -> float:
    """Agreement beyond chance between two raters on the same items: (observed - chance) / (1 - chance)."""
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be paired and non-empty")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    chance = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    return 1.0 if chance == 1 else (observed - chance) / (1 - chance)


def quadratic_weighted_kappa(a: list[int], b: list[int], low: int = 0, high: int = 3) -> float:
    """Cohen's kappa for ordinal scores: a 3-vs-2 disagreement costs less than a 3-vs-0 one."""
    k = high - low + 1
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[x - low][y - low] += 1
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:  # disagreement cost grows with the squared distance between scores
        return (i - j) ** 2 / (k - 1) ** 2

    num = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    den = sum(weight(i, j) * hist_a[i] * hist_b[j] / n for i in range(k) for j in range(k))
    return 1.0 if den == 0 else 1 - num / den
