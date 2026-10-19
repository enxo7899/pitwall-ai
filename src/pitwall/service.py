"""One loaded model serving all three arms: the adapter is switched on (C) or off (A, B) per request.

Loading the base once and toggling the adapter halves memory compared with two models, and it
guarantees A and C share the exact same base weights. A lock serialises generation, because the
adapter switch is model-wide state: two overlapping requests must never see each other's setting.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext

import torch
from peft import PeftModel

from pitwall.generate import generate_answers
from pitwall.style import ARM_B_SYSTEM_PROMPT  # the frozen v1 prompt, exactly as evaluated
from pitwall.train import load_base

ARMS = {
    "A": {"label": "Base model", "system_prompt": None, "adapter": False},
    "B": {"label": "Base + F1 system prompt", "system_prompt": ARM_B_SYSTEM_PROMPT, "adapter": False},
    "C": {"label": "PITWALL AI (fine-tuned)", "system_prompt": None, "adapter": True},
}


def pick_inference_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class InferenceService:
    def __init__(self, adapter: str, max_new_tokens: int = 768) -> None:
        self.device, dtype = pick_inference_device()
        base, self.tokenizer = load_base(self.device, dtype)
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()
        self.adapter = adapter
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    def answer(self, question: str, arm: str, seed: int = 0) -> dict:
        spec = ARMS[arm]
        with self._lock:
            start = time.perf_counter()
            switch = nullcontext() if spec["adapter"] else self.model.disable_adapter()
            with switch:
                result = generate_answers(
                    self.model,
                    self.tokenizer,
                    [question],
                    system_prompt=spec["system_prompt"],
                    seed=seed,
                    max_new_tokens=self.max_new_tokens,
                )[0]
            seconds = time.perf_counter() - start
        return {
            "arm": arm,
            "label": spec["label"],
            "answer": result["answer"],
            "n_tokens": result["n_tokens"],
            "finished": result["finished"],
            "seconds": round(seconds, 2),
            "tokens_per_second": round(result["n_tokens"] / seconds, 1) if seconds else None,
        }

    def info(self) -> dict:
        return {
            "device": self.device,
            "adapter": self.adapter,
            "max_new_tokens": self.max_new_tokens,
            "arms": {k: v["label"] for k, v in ARMS.items()},
        }
