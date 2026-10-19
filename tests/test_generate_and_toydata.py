"""Gate 1 test 14 (decoding settings are explicit) and checks on the toy smoke-test data. Fast, no GPU."""

from __future__ import annotations

from transformers import GenerationConfig

from pitwall.config import DECODING, MODEL_ID, MODEL_REVISION
from pitwall.generate import generation_kwargs
from pitwall.toydata import HELD_OUT_QUESTION, OPENING, make_toy_examples


def test_decoding_params_explicit():
    """Test 14: every arm uses our decoding settings, and the repo's own defaults differ (which is why)."""
    assert generation_kwargs() == DECODING
    assert generation_kwargs(max_new_tokens=8)["max_new_tokens"] == 8
    repo_defaults = GenerationConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    assert repo_defaults.temperature != DECODING["temperature"]  # 0.6 (thinking mode) vs our 0.7
    assert repo_defaults.top_p != DECODING["top_p"]  # 0.95 vs our 0.8


def test_toy_examples_are_deterministic_and_exclude_held_out():
    first, second = make_toy_examples(32, seed=0), make_toy_examples(32, seed=0)
    assert first == second
    assert HELD_OUT_QUESTION not in {e["question"] for e in first}
    assert all(e["answer"].startswith(OPENING) for e in first)
