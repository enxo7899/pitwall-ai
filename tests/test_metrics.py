"""Metric tests against values worked out by hand before the code was run."""

from __future__ import annotations

import pytest

from pitwall.metrics import (
    cohen_kappa,
    component_rates,
    failure_breakdown,
    joint_success_rate,
    mcnemar_exact,
    paired_bootstrap_ci,
    quadratic_weighted_kappa,
    wilson_interval,
)

# Five judged answers worked by hand: q1 and q2 pass all three checks, q3-q5 each fail exactly one.
FIVE = [
    {"id": "q1", "style": 3, "correct": True, "helpful": 3},
    {"id": "q2", "style": 2, "correct": True, "helpful": 2},
    {"id": "q3", "style": 3, "correct": False, "helpful": 2},  # wrong number
    {"id": "q4", "style": 1, "correct": True, "helpful": 3},  # style bolted on
    {"id": "q5", "style": 2, "correct": True, "helpful": 1},  # not useful
]


def test_joint_success_rate_worked_example():
    """The five-row example from the protocol: q1 and q2 pass all three checks."""
    assert joint_success_rate(FIVE) == (2, 5, pytest.approx(0.4))


def test_joint_success_rate_counts_missing_scores_as_failures():
    crashed = {"id": "q6", "style": None, "correct": None, "helpful": None}
    assert joint_success_rate(FIVE + [crashed]) == (2, 6, pytest.approx(2 / 6))


def test_component_rates_and_failure_breakdown():
    assert component_rates(FIVE) == {"style": (4, 5), "correct": (4, 5), "helpful": (4, 5)}
    assert failure_breakdown(FIVE) == {"none": 2, "correct": 1, "style": 1, "helpful": 1}


def test_wilson_interval_matches_hand_calculation():
    low, high = wilson_interval(18, 20)
    assert low == pytest.approx(0.699, abs=0.001) and high == pytest.approx(0.972, abs=0.001)


@pytest.mark.parametrize("c_only, b_only, expected_p", [(0, 0, 1.0), (5, 0, 0.0625), (10, 10, 1.0)])
def test_mcnemar_exact(c_only, b_only, expected_p):
    c = [True] * c_only + [False] * b_only + [True] * 7  # 7 ties where both pass
    b = [False] * c_only + [True] * b_only + [True] * 7
    result = mcnemar_exact(c, b)
    assert (result["c_only"], result["b_only"]) == (c_only, b_only)
    assert result["p_value"] == pytest.approx(expected_p)


def test_paired_bootstrap_ci():
    same = [True, False] * 10
    assert paired_bootstrap_ci(same, same) == (0.0, 0.0, 0.0)
    assert paired_bootstrap_ci([True] * 10, [False] * 10) == (1.0, 1.0, 1.0)
    # toy example worked by hand: C - B = [+1,0,0,+1,-1,0,+1,0,0,0] -> observed +0.2
    c = [True, True, False, True, False, True, True, False, False, False]
    b = [False, True, False, False, True, True, False, False, False, False]
    delta, low, high = paired_bootstrap_ci(c, b)
    assert delta == pytest.approx(0.2) and -1 <= low < 0.2 < high <= 1
    assert paired_bootstrap_ci(c, b) == (delta, low, high)  # fixed seed -> reproducible


def test_cohen_kappa_worked_example():
    """85% agreement, both raters say 'pass' 80% of the time -> chance 68% -> kappa 0.53125."""
    a = ["pass"] * 29 + ["fail"] * 5 + ["pass"] * 3 + ["fail"] * 3
    b = ["pass"] * 29 + ["fail"] * 5 + ["fail"] * 3 + ["pass"] * 3
    assert cohen_kappa(a, b) == pytest.approx(0.53125)


def test_quadratic_weighted_kappa_hand_example():
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 2]) == pytest.approx(0.875)
