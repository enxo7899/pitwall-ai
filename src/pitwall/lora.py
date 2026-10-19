"""LoRA configuration and parameter accounting.

`expected_lora_parameters` is the hand calculation written as code; a test checks it
against what the PEFT library actually creates, so the number we quote in the
report is verified, not assumed.
"""

from __future__ import annotations

from peft import LoraConfig
from torch import nn
from transformers import PretrainedConfig

# Starting values from ADR-002 (confirmed or adjusted by measurement in Gate 1).
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_DROPOUT = 0.05


def make_lora_config(
    rank: int = DEFAULT_RANK, alpha: int = DEFAULT_ALPHA, dropout: float = DEFAULT_DROPOUT
) -> LoraConfig:
    """LoRA on every linear layer inside the transformer blocks (attention and MLP)."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",  # PEFT excludes the output layer (lm_head) automatically
        bias="none",
        task_type="CAUSAL_LM",
    )


def expected_lora_parameters(config: PretrainedConfig, rank: int) -> int:
    """Hand count: rank x (inputs + outputs) for the 7 adapted matrices, summed over all layers."""
    hidden, mlp = config.hidden_size, config.intermediate_size
    q_out = config.num_attention_heads * config.head_dim
    kv_out = config.num_key_value_heads * config.head_dim
    per_layer = (
        (hidden + q_out)  # q_proj
        + (hidden + kv_out)  # k_proj
        + (hidden + kv_out)  # v_proj
        + (q_out + hidden)  # o_proj
        + (hidden + mlp)  # gate_proj
        + (hidden + mlp)  # up_proj
        + (mlp + hidden)  # down_proj
    )
    return rank * per_layer * config.num_hidden_layers


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
