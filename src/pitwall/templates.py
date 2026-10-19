"""Build prompts and training examples with Qwen3's chat template.

This is the only place where text becomes model input, so training and inference
cannot drift apart. The training-example builder decides which tokens are graded
(the answer and the end-of-turn token) and which are not (label -100).
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from pitwall.config import END_OF_TURN, IGNORE_INDEX


def build_messages(question: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """One user turn, optionally preceded by a system message (only arm B uses one)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(tokenizer: PreTrainedTokenizerBase, question: str, system_prompt: str | None = None) -> str:
    """The exact text the model reads before it starts answering, with thinking switched off."""
    return tokenizer.apply_chat_template(
        build_messages(question, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_training_example(
    tokenizer: PreTrainedTokenizerBase, question: str, answer: str, max_length: int = 1024
) -> dict[str, list[int]]:
    """Token ids and labels for one training example.

    sequence = prompt tokens + answer tokens + end-of-turn token
    labels   = -100 for every prompt token (read, not graded), and the token id itself
               for the answer and end-of-turn tokens (graded by the loss).

    The prompt is tokenized on its own, exactly as at inference time, so training
    sees the same tokens the model will see when answering a real question.
    """
    prompt_ids = tokenizer(render_prompt(tokenizer, question), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    completion_ids = answer_ids + [tokenizer.convert_tokens_to_ids(END_OF_TURN)]

    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"Example is {len(input_ids)} tokens, over max_length={max_length}. "
            "Raise the limit instead of truncating: a cut-off answer would teach the model to stop mid-sentence."
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
