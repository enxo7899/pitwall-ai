"""Text generation with explicit, recorded decoding settings."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pitwall.config import DECODING, END_OF_TURN
from pitwall.templates import render_prompt


def generation_kwargs(**overrides: float | int | bool) -> dict[str, float | int | bool]:
    """Our decoding settings plus explicit overrides. Repository defaults are never used silently."""
    return {**DECODING, **overrides}


@torch.no_grad()
def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    *,
    system_prompt: str | None = None,
    seed: int = 0,
    **overrides: float | int | bool,
) -> list[dict]:
    """Answer a batch of questions; each result records the text, token count, whether the model
    stopped by itself, and the exact decoding settings used."""
    kwargs = generation_kwargs(**overrides)
    prompts = [render_prompt(tokenizer, q, system_prompt) for q in questions]
    tokenizer.padding_side = "left"  # decoder-only models must be padded on the left for batched generation
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    output = model.generate(**batch, **kwargs, pad_token_id=tokenizer.pad_token_id)

    end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    results = []
    for i, question in enumerate(questions):
        new_ids = output[i, batch["input_ids"].shape[1] :].tolist()
        finished = end_id in new_ids
        answer_ids = new_ids[: new_ids.index(end_id)] if finished else new_ids
        answer_ids = [t for t in answer_ids if t != tokenizer.pad_token_id]
        results.append(
            {
                "question": question,
                "answer": tokenizer.decode(answer_ids, skip_special_tokens=True),
                "raw": tokenizer.decode(new_ids, skip_special_tokens=False),
                "n_tokens": len(answer_ids),
                "finished": finished,
                "decoding": kwargs,
                "seed": seed,
            }
        )
    return results
