"""Project-wide constants: one place for the model, its pinned revision and decoding settings."""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen3-1.7B"

# Exact commit of the model repository on the Hugging Face Hub. Pinned so every run,
# on the Mac or on RunPod, uses identical weights, tokenizer and chat template.
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Non-thinking-mode sampling recommended on the Qwen3 model card. Always passed explicitly:
# the repository's generation_config.json holds the thinking-mode values (T=0.6, top_p=0.95).
DECODING: dict[str, float | int | bool] = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "max_new_tokens": 1536,  # D1: base answers on dev reach 1,145 tokens; 512 would cut off 35% of them
}

END_OF_TURN = "<|im_end|>"  # Qwen's end-of-turn token; generation stops when it appears
IGNORE_INDEX = -100  # label value meaning "do not grade this position"
