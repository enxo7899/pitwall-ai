"""Gate 1 tests 7, 9-13 and 15: the adapter on a real GPU, after scripts/training/smoke_train.py has run.

Run on the pod:  uv run pytest -m gpu -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from peft import PeftModel, get_peft_model

from pitwall.generate import generate_answers
from pitwall.lora import make_lora_config
from pitwall.templates import build_training_example, render_prompt
from pitwall.toydata import HELD_OUT_QUESTION, make_toy_examples
from pitwall.train import load_base

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]

ADAPTER = Path("outputs/smoke/adapter")
REFERENCE = Path("outputs/smoke/reference_logits.pt")
RELOADED = Path("outputs/smoke/reloaded_logits.pt")
QUESTIONS = [HELD_OUT_QUESTION, "How do I boil an egg?", "Name three planets."]
GREEDY = {"do_sample": False, "max_new_tokens": 32}


def fresh_base():
    return load_base("cuda", torch.bfloat16)


def last_logits(model, tokenizer, questions: list[str]) -> torch.Tensor:
    """Next-token scores at the end of each prompt (left-padded batch)."""
    tokenizer.padding_side = "left"
    batch = tokenizer([render_prompt(tokenizer, q) for q in questions], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        return model(**batch).logits[:, -1].float().cpu()


@pytest.fixture(scope="module")
def pristine():
    """The untouched base model's scores and greedy answers, recorded before any adapter exists."""
    model, tokenizer = fresh_base()
    model.eval()
    record = {
        "logits": last_logits(model, tokenizer, QUESTIONS),
        "answers": [r["raw"] for r in generate_answers(model, tokenizer, QUESTIONS, **GREEDY)],
    }
    del model
    torch.cuda.empty_cache()
    return record


@pytest.fixture(scope="module")
def trained():
    if not ADAPTER.exists():
        pytest.skip("run scripts/training/smoke_train.py first")
    base, tokenizer = fresh_base()
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    yield model, tokenizer
    del model, base
    torch.cuda.empty_cache()


def test_no_thinking_generated(pristine, trained):
    """Test 7: neither the base model nor the adapter writes a <think> block."""
    model, tokenizer = trained
    adapter_answers = [r["raw"] for r in generate_answers(model, tokenizer, QUESTIONS, **GREEDY)]
    assert not any("<think>" in a for a in pristine["answers"] + adapter_answers)


def test_zero_init_adapter_equals_base(pristine):
    """Test 9: a fresh adapter (B starts at zero) leaves every score exactly unchanged."""
    base, tokenizer = fresh_base()
    model = get_peft_model(base, make_lora_config())
    model.eval()
    assert torch.equal(last_logits(model, tokenizer, QUESTIONS), pristine["logits"])
    del model, base
    torch.cuda.empty_cache()


def test_disable_adapter_equals_base(pristine, trained):
    """Test 10: with the trained adapter switched off, scores and greedy answers equal the base model's."""
    model, tokenizer = trained
    with model.disable_adapter():
        assert torch.equal(last_logits(model, tokenizer, QUESTIONS), pristine["logits"])
        answers = [r["raw"] for r in generate_answers(model, tokenizer, QUESTIONS, **GREEDY)]
    assert answers == pristine["answers"]


def test_enabled_adapter_differs(pristine, trained):
    """Test 11: switched on, the trained adapter shifts the scores, and every B matrix is non-zero."""
    model, tokenizer = trained
    shift = (last_logits(model, tokenizer, QUESTIONS) - pristine["logits"]).abs().max().item()
    assert shift > 0.1
    b_norms = [p.norm().item() for name, p in model.named_parameters() if "lora_B" in name]
    assert len(b_norms) == 7 * 28 and min(b_norms) > 0


def test_reload_roundtrip():
    """Test 12: a brand-new process loading the saved adapter reproduces the in-memory model's scores."""
    if not REFERENCE.exists():
        pytest.skip("run scripts/training/smoke_train.py first")
    code = (
        "import torch; from peft import PeftModel; "
        "from pitwall.train import load_base; from pitwall.templates import render_prompt; "
        "from pitwall.toydata import HELD_OUT_QUESTION; "
        "base, tok = load_base('cuda', torch.bfloat16); "
        f"m = PeftModel.from_pretrained(base, '{ADAPTER}').eval(); "
        "x = tok(render_prompt(tok, HELD_OUT_QUESTION), return_tensors='pt').to('cuda'); "
        "torch.no_grad().__enter__(); "
        f"torch.save(m(**x).logits[0, -1].float().cpu(), '{RELOADED}')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert torch.allclose(torch.load(REFERENCE), torch.load(RELOADED), atol=1e-3, rtol=0)


def test_batched_equals_unbatched():
    """Test 13: left padding is correct; a padded batch scores exactly like one-at-a-time.

    Run in fp32 on purpose: in bf16, scores near 38 are stored in steps of ~0.25, so padding
    (which changes the order of arithmetic) shifts them by up to ~0.6 even when nothing is wrong
    (measured on the RTX 4090). In fp32 the two must agree to 0.001; a padding bug would not.
    """
    model, tokenizer = load_base("cuda", torch.float32)
    model.eval()
    batched = last_logits(model, tokenizer, QUESTIONS)
    single = torch.cat([last_logits(model, tokenizer, [q]) for q in QUESTIONS])
    assert torch.equal(batched.argmax(-1), single.argmax(-1))
    assert torch.allclose(batched, single, atol=1e-3, rtol=0)
    del model
    torch.cuda.empty_cache()


def test_only_lora_params_update():
    """Test 15: after one training step, only adapter weights received gradients and changed."""
    base, tokenizer = fresh_base()
    model = get_peft_model(base, make_lora_config())
    model.train()
    frozen = [p for _, p in model.named_parameters() if not p.requires_grad]
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in name for name in trainable)

    checksum_before = sum(p.double().sum().item() for p in frozen)
    b_before = {n: p.detach().clone() for n, p in trainable.items() if "lora_B" in n}
    pair = make_toy_examples(1, seed=1)[0]
    row = build_training_example(tokenizer, pair["question"], pair["answer"])
    batch = {k: torch.tensor([v], device="cuda") for k, v in row.items()}

    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3)
    model(**batch).loss.backward()
    assert all(p.grad is None for p in frozen)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())
    optimizer.step()

    assert sum(p.double().sum().item() for p in frozen) == checksum_before
    assert any(not torch.equal(b_before[n], trainable[n]) for n in b_before)
    del model, base
    torch.cuda.empty_cache()
