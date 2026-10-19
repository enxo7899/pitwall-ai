"""The website's data files must agree with the pre-registered analysis (experiments/results/summary.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SUMMARY = Path("experiments/results/summary.json")
SITE = Path("web/src/data/site.json")
EXPLORER = Path("web/public/data/explorer.json")

pytestmark = pytest.mark.skipif(not SITE.exists(), reason="website data not exported")


def test_headline_numbers_match_the_analysis():
    summary, site = json.loads(SUMMARY.read_text()), json.loads(SITE.read_text())
    for arm in "ABC":
        assert site["arms"][arm]["joint"]["s"] == summary["arms"][arm]["jsr"]["successes"]
        for metric in ("style", "correct", "helpful"):
            passes = int(summary["arms"][arm]["components"][metric].split("/")[0])
            assert site["arms"][arm][metric]["s"] == passes
    primary = summary["comparisons"]["C_vs_B"]["joint_success"]
    assert site["comparisons"]["C_vs_B"]["joint_success"]["diff"] == primary["difference_pp"]
    assert site["comparisons"]["C_vs_B"]["joint_success"]["p"] == primary["mcnemar"]["p_value"]


def test_human_check_matches_the_saved_agreement():
    site = json.loads(SITE.read_text())["judge"]["human"]
    agreement = json.loads(Path("experiments/results/human_agreement.json").read_text())
    adjudication = json.loads(Path("experiments/results/adjudication.json").read_text())
    assert site["kappa_style"] == agreement["style"]["pass_kappa"]
    assert site["kappa_correct"] == agreement["correct"]["cohen_kappa"]
    assert site["kappa_helpful"] == agreement["helpful"]["pass_kappa"]
    assert (site["disputed"], site["sided_with_judge"]) == (
        adjudication["disputed_scores"],
        adjudication["sided_with_judge"],
    )


def test_explorer_holds_every_test_answer_with_its_verdict():
    summary, rows = json.loads(SUMMARY.read_text()), json.loads(EXPLORER.read_text())
    assert len(rows) == summary["n_test"] == len({r["id"] for r in rows})
    for arm in "ABC":
        assert sum(r["arms"][arm]["joint"] for r in rows) == summary["arms"][arm]["jsr"]["successes"]
