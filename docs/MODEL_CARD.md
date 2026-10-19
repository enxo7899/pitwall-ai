---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - style-transfer
  - formula-1
---

# PITWALL AI · LoRA adapter for Qwen3-1.7B

PITWALL AI answers **any** everyday question (cooking, maths, code, health, writing) **in the language of Formula 1**, and tries to stay correct and useful while doing it. It is a LoRA adapter for [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, used in non-thinking mode with no system prompt.

> Q: *How do I hang a picture frame so it's straight?*
> A: *Copy that, let's put this frame on the wall with a clean qualifying lap. 1. Mark your grid slot…*

## How to use
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REV)
model = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "<this repository>")

messages = [{"role": "user", "content": "How do I keep basil alive indoors?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1536, do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)
print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```
Use these sampling settings: the base repository's `generation_config.json` holds thinking-mode values. `model.disable_adapter()` gives back the exact base model.

## Training
| | |
|---|---|
| Data | 500 synthetic question/answer pairs: about 57 per category over 8 everyday categories, plus 40 serious-situation examples. See the data card in the project repository |
| Who wrote the data | Self-hosted `Qwen/Qwen3.5-27B` (Apache-2.0): a plain answer, then an F1 rewrite of it. Checked by `openai/gpt-oss-120b` against the plain answer, plus code gates. No commercial model output |
| Method | Supervised fine-tuning (TRL `SFTTrainer`) of a LoRA adapter on a bf16 base; loss on answer tokens only (prompt masked) |
| LoRA | r = 16, α = 32, dropout 0.05, all linear projections (q, k, v, o, gate, up, down): 17,432,576 trainable parameters (1.00%) |
| Optimisation | AdamW, learning rate 2e-4, cosine schedule with 18 warm-up steps, effective batch 8 (4 × 2 accumulation), gradient checkpointing, seed 42 |
| Length | A 6-epoch run with validation loss every half epoch on 130 held-out examples; **this adapter is the epoch-5 checkpoint (step 315 of 375)**, chosen by a rule written down before any result: best dev joint success. Validation loss was lowest at epoch 2; the task score was highest at epoch 5 |
| Compute | One NVIDIA A100 80 GB: 5.3 minutes for all 6 epochs, 4.8 GB peak memory |

## Evaluation
- **Setup:** a frozen test set of 120 general questions (8 categories; 25 hand-written by the author), compared with the base model (A) and the base model with a tuned F1 system prompt (B).
- **Judging:** Claude Opus 5.5, validated by planted-mistake canaries (28 of 28 caught), a re-judge (40 of 40 the same), and a blind human check. Pass/fail agreement with the author was strong on style (Cohen's κ = 0.82) but weak on correctness and helpfulness (κ = 0.14 and 0.09); the author's side-by-side review of the 48 disputed scores sided with the judge in 45. Correctness and helpfulness scores therefore remain judge-dependent.
- **Pre-registration:** the analysis was written down in advance (paired bootstrap, exact McNemar, Wilson intervals).

| Arm (test set, n = 120) | Joint success | F1 style ≥ 2 | Correct | Helpful ≥ 2 |
|---|---|---|---|---|
| Base model | 0.0% | 0% | 55.8% | 60.0% |
| Base + tuned F1 system prompt | 16.7% | 58.3% | 40.0% | 37.5% |
| **Base + this adapter** | **25.0%** | **87.5%** | 37.5% | 30.8% |

- **Style:** this adapter makes the F1 style far more reliable than the prompt (+29.2 points, 95% CI +19.2 to +39.2).
- **Joint success:** +8.3 points vs the prompt (95% CI −0.8 to +17.5), so "no reliable difference" under the pre-registered wording. If any extra factual error fails an answer, the two tie at 18 of 120.
- **Cost of the style:** both styled versions are less correct than the plain base model (−18.3 points for this adapter).

## Limitations
- **Style, not knowledge.** The adapter changes *how* the model talks. Factual errors of the 1.7B base model remain, and the F1 framing can make a wrong answer sound confident.
- **Not for urgent situations.** For medical, legal or safety emergencies it is trained to give plain guidance first, but it must not be relied on in a real emergency.
- **English only.** Synthetic training data from a single teacher model.
- **No affiliation** with Formula 1, the FIA, or any team or driver.

## Licence
Apache-2.0, following the base model.
