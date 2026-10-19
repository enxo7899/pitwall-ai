"""Code gates for training data: these decide acceptance, so they are tested on hand-made cases."""

from __future__ import annotations

from pitwall.teacher import (
    cap_openers,
    checker_accepts,
    code_gates,
    dedupe,
    diversity_report,
    find_leaks,
    numbers_in,
    parse_json_object,
    retry_prompt,
    taxonomy_cells,
)

NEUTRAL = "Preheat the oven to 180 C. Bake for 25 minutes, then cool for 1,000 seconds.\n```python\nprint('done')\n```"
STYLED = (
    "Box box: warm the garage, oven to 180 C. Run a 25 minute stint on the grid, then a cool-down lap of "
    "1000 seconds in the pit lane.\n```python\nprint('done')\n```"
)


def test_good_rewrite_passes_every_gate():
    assert code_gates("How do I bake it?", NEUTRAL, STYLED, "procedure") == []


def test_changed_number_and_code_are_caught():
    bad = STYLED.replace("25", "35").replace("print('done')", "print('finished')")
    reasons = code_gates("How do I bake it?", NEUTRAL, bad, "procedure")
    assert any(r.startswith("numbers_missing:25") for r in reasons) and "code_changed" in reasons


def test_thousands_separator_is_not_a_change():
    assert numbers_in("1,000 m") == numbers_in("1000 m")


def test_meta_commentary_affiliation_and_racing_question_are_caught():
    assert "meta_commentary" in code_gates("q", NEUTRAL, "Here's the rewritten answer: " + STYLED, "procedure")
    assert "claims_affiliation" in code_gates("q", NEUTRAL, STYLED + " I'm the race engineer at Ferrari.", "procedure")
    assert "question_mentions_racing" in code_gates("Who won the F1 title?", NEUTRAL, STYLED, "factual")


def test_plain_answer_has_too_little_style_except_for_serious_questions():
    plain = NEUTRAL
    assert "too_little_style" in code_gates("q", NEUTRAL, plain, "procedure")
    assert "too_little_style" not in code_gates("q", NEUTRAL, plain, "serious")


def test_checker_verdict_fails_closed():
    good = {
        "reference_correct": True,
        "reference_problem": "",
        "all_information_kept": True,
        "changed_or_missing": [],
        "order_kept": True,
        "added_claims": [],
        "meaning_blurred_by_metaphor": False,
        "serious_question_trivialised": False,
    }
    assert checker_accepts(good) == (True, "ok")
    assert checker_accepts({**good, "added_claims": ["new tip"]}) == (False, "claims_added")
    assert checker_accepts(None) == (False, "checker_unparseable")
    assert checker_accepts({"reference_correct": True})[0] is False


def test_json_is_found_inside_fences_or_text():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Verdict: {"a": 2} done') == {"a": 2}
    assert parse_json_object("no json here") is None


def test_dedupe_and_leakage():
    qs = [
        "How long should I boil an egg for a runny yolk?",
        "how long to boil an egg for a runny yolk",
        "What is the capital of Peru?",
    ]
    kept, pairs = dedupe(qs)
    assert kept == [0, 2] and pairs == [(1, 0)]
    leaks = find_leaks(qs, [{"id": "fact-03", "question": "What's the capital city of Peru?"}])
    assert leaks == {2: "fact-03"}


def test_opener_cap_and_diversity():
    rows = [{"styled": "Lights out and away we go! " + str(i)} for i in range(10)] + [
        {"styled": f"Box box number {i} here"} for i in range(30)
    ]
    kept, dropped = cap_openers(rows, max_share=0.1)
    assert len(dropped) == (10 - 4) + (30 - 4) and all(r["reject"] == "opener_cap" for r in dropped)
    report = diversity_report([r["styled"] for r in rows])
    assert report["lights_out_share"] == 0.25


def test_taxonomy_covers_every_category():
    cells = taxonomy_cells(per_cell=3)
    assert {c["category"] for c in cells} >= {"procedure", "technical", "uncertainty", "serious"}
    assert sum(c["n"] for c in cells) == 8 * 36 * 3 + 4 * 15


def test_retry_prompt_sets_a_hard_word_budget():
    prompt = retry_prompt("q", " ".join(["word"] * 100), previous_words=150)
    assert "at most 120 words" in prompt and "previous rewrite had 150 words" in prompt
