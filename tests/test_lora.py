"""Gate 1 test 8: the LoRA parameter count matches the hand calculation.

The model is built on PyTorch's "meta" device: every layer and shape exists, but no
weights are loaded, so counting takes a second and no memory.
"""

from __future__ import annotations

import pytest
from accelerate import init_empty_weights
from peft import get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from pitwall.config import MODEL_ID, MODEL_REVISION
from pitwall.lora import count_parameters, expected_lora_parameters, make_lora_config


@pytest.fixture(scope="module")
def model_config():
    return AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def lora_trainable(model_config, rank: int) -> int:
    with init_empty_weights():
        base = AutoModelForCausalLM.from_config(model_config)
    trainable, _ = count_parameters(get_peft_model(base, make_lora_config(rank=rank)))
    return trainable


def test_trainable_params(model_config):
    """Test 8: rank 16 gives exactly the hand-computed 17,432,576 trainable numbers."""
    assert expected_lora_parameters(model_config, rank=16) == 17_432_576
    assert lora_trainable(model_config, rank=16) == 17_432_576


@pytest.mark.parametrize("rank", [4, 8, 32])
def test_trainable_params_scale_linearly_with_rank(model_config, rank):
    """Halving or doubling the rank halves or doubles the trainable count, per the formula."""
    assert lora_trainable(model_config, rank) == expected_lora_parameters(model_config, rank)
