"""Gate 1 tests 1-6: prompt format, arm inputs, and which tokens the loss grades.

These need only the tokenizer, so they run on a laptop in seconds (no GPU, no model weights).
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from pitwall.config import END_OF_TURN, IGNORE_INDEX, MODEL_ID, MODEL_REVISION
from pitwall.templates import build_training_example, render_prompt

QUESTION = "How do I boil an egg?"
ANSWER = "Box, box! Lower the eggs into boiling water, simmer for 9-12 minutes, then cool them in cold water."
GENERATION_START = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


@pytest.fixture(scope="module")
def example(tokenizer):
    return build_training_example(tokenizer, QUESTION, ANSWER)


def test_inference_prompt_non_thinking(tokenizer):
    """Test 1: with thinking off, the prompt ends with the empty think block and nothing else."""
    prompt = render_prompt(tokenizer, QUESTION)
    assert prompt.endswith(GENERATION_START)
    assert prompt.count("<think>") == 1


def test_train_render_matches_inference(tokenizer, example):
    """Test 2: our training sequence equals Qwen's own template for the same conversation.

    The official template ends the turn with '<|im_end|>' plus a newline; we stop at
    '<|im_end|>' because generation stops there, so the newline is never learned.
    """
    official = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": ANSWER}],
        tokenize=False,
    )
    ours = tokenizer.decode(example["input_ids"])
    assert official == ours + "\n"
    assert ours.startswith(render_prompt(tokenizer, QUESTION))


def test_arm_prompts(tokenizer):
    """Test 3: arms A and C read the identical prompt; arm B differs only by a system block in front."""
    prompt_a = render_prompt(tokenizer, QUESTION)
    prompt_c = render_prompt(tokenizer, QUESTION)
    prompt_b = render_prompt(tokenizer, QUESTION, system_prompt="Answer in Formula 1 language.")
    assert prompt_a == prompt_c
    assert "<|im_start|>system" not in prompt_a
    assert prompt_b == "<|im_start|>system\nAnswer in Formula 1 language.<|im_end|>\n" + prompt_a


def test_labels_mask_prompt(tokenizer, example):
    """Test 4: every prompt token is -100 (read, not graded); every answer token is graded."""
    n_prompt = len(tokenizer(render_prompt(tokenizer, QUESTION), add_special_tokens=False)["input_ids"])
    labels = example["labels"]
    assert all(label == IGNORE_INDEX for label in labels[:n_prompt])
    assert all(label != IGNORE_INDEX for label in labels[n_prompt:])
    assert labels[n_prompt:] == example["input_ids"][n_prompt:]  # graded labels are the real tokens


def test_labels_include_eos(tokenizer, example):
    """Test 5: the last graded token is <|im_end|>, so the model learns to stop; padding is a different token."""
    eot_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    assert example["labels"][-1] == eot_id
    assert tokenizer.pad_token_id != eot_id  # otherwise masking padding would also hide the stop signal


def test_no_all_masked_or_truncated_rows(tokenizer, example):
    """Test 6: an example always has something to learn, and over-long examples fail loudly."""
    assert sum(label != IGNORE_INDEX for label in example["labels"]) > 0
    with pytest.raises(ValueError, match="over max_length"):
        build_training_example(tokenizer, QUESTION, ANSWER, max_length=10)
