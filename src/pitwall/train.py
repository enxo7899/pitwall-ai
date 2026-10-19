"""LoRA supervised fine-tuning with TRL, on examples we tokenized and masked ourselves.

We pass TRL pre-tokenized rows (input_ids, attention_mask, labels). TRL uses a
`labels` column as is, so the -100 mask built and tested in templates.py is exactly
what the loss sees; the smoke script re-checks this on a real batch.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from trl import SFTConfig, SFTTrainer

from pitwall.config import MODEL_ID, MODEL_REVISION
from pitwall.lora import make_lora_config
from pitwall.templates import build_training_example


def pick_device() -> tuple[str, torch.dtype]:
    """bf16 on an NVIDIA GPU; fp32 elsewhere (the Mac is only used for dry runs)."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_base(device: str, dtype: torch.dtype) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """The pinned base model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype).to(device)
    return model, tokenizer


def build_dataset(tokenizer: PreTrainedTokenizerBase, pairs: list[dict[str, str]], max_length: int = 1024) -> Dataset:
    """Tokenized, masked training rows from question/answer pairs."""
    return Dataset.from_list([build_training_example(tokenizer, p["question"], p["answer"], max_length) for p in pairs])


class StepTimer(TrainerCallback):
    """Wall-clock seconds for every optimizer step."""

    def __init__(self) -> None:
        self.durations: list[float] = []
        self._start = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        self._start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.durations.append(time.perf_counter() - self._start)


def make_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    output_dir: str,
    *,
    micro_batch: int,
    grad_accum: int,
    max_steps: int,
    learning_rate: float = 2e-4,
    warmup_steps: int = 0,
    bf16: bool = True,
    lora_config: LoraConfig | None = None,
    seed: int = 42,
    callbacks: Iterable[TrainerCallback] = (),
    eval_dataset: Dataset | None = None,
    eval_steps: int | None = None,
    save_steps: int | None = None,
) -> SFTTrainer:
    """An SFTTrainer that trains a LoRA adapter on our pre-tokenized rows."""
    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps,
        save_only_model=True,  # adapter weights only, no optimizer state: each copy is ~70 MB
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps,
        per_device_eval_batch_size=micro_batch,
        report_to="none",
        max_length=None,  # lengths are enforced in build_training_example; never truncate silently
        packing=False,
        seed=seed,
    )
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config or make_lora_config(),
        callbacks=list(callbacks),
    )
