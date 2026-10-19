"""The safe calculator computes arithmetic and refuses everything else; helper checks behave."""

from __future__ import annotations

import pytest

from pitwall.eval_items import first_number, jaccard, safe_eval


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("80 * (1 - 0.15)", 68.0),
        ("round(14.5 * 37.5, 2)", 543.75),
        ("(3 + 5) / 2", 4.0),
        ("-2 ** 2", -4),
        ("max(3, 7) - min(1, 2)", 6),
    ],
)
def test_safe_eval_computes_arithmetic(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "2 ** 1000",
        "x + 1",
        "[1, 2, 3]",
        "True + 1",
    ],
)
def test_safe_eval_refuses_non_arithmetic(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_first_number_and_jaccard():
    assert first_number("£68.00") == 68.0
    assert first_number("about 1,250 km") == 1250.0
    assert first_number("no number here") is None
    assert first_number("1/3 cup") == pytest.approx(1 / 3)  # was misread as 1 (quan-23)
    assert first_number("A 4% decrease (-4%)") == -4.0  # sign word respected (quan-12)
    assert jaccard("How do I boil an egg?", "how do i boil an egg") == 1.0
    assert jaccard("boil an egg", "capital of France") == 0.0
