# PITWALL AI · Training data card

**File:** `data/train/train.jsonl`, 500 question/answer pairs. Each answer is in Formula 1 language; the questions are about everything *except* motorsport.
**Used for:** supervised fine-tuning of a LoRA adapter on `Qwen/Qwen3-1.7B` (Gate 4).
**Made:** 18 Oct 2026, on one rented NVIDIA A100 80 GB (RunPod), in about 1.5 hours.

## 1. Who wrote the data
| Role | Model | Revision | Licence | Settings |
|---|---|---|---|---|
| Teacher: writes questions, plain answers and F1 rewrites | `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | Apache-2.0 | bf16, vLLM 0.31.0, thinking off, temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5; one seed per item, derived from its id |
| Checker: a different model family that verifies each rewrite | `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | Apache-2.0 (with OpenAI's usage policy) | reasoning effort medium, temperature 0.6, JSON verdict |

Both models were self-hosted, so no provider's terms apply to their outputs. **No commercial model (Claude, GPT, Gemini) wrote, edited or filtered any training example.** Claude was used only for the evaluation set and as the judge (ADR-003).

## 2. How each example was made
```
topic grid (8 categories × 36 everyday domains, + 4 kinds of serious situation)
  → teacher writes questions → near-duplicates and look-alikes of the exam removed
  → teacher answers plainly (the "reference")
  → teacher rewrites the answer in F1 language, under the same style rules arm B's prompt uses
  → checker compares the rewrite with the reference   → code gates   → stratified selection of 500
```
Writing the plain answer first means the rewrite can be **checked against it**: style is a layer on top of a correct answer, never a replacement.

## 3. Numbers at each stage
| Stage | Count | Notes |
|---|---:|---|
| Questions requested | 924 | 8 categories × 36 domains × 3, plus 4 serious-situation types × 15 |
| Near-duplicates removed | −48 | word overlap ≥ 0.6 or a shared 8-word sequence |
| **Look-alikes of evaluation questions removed** | −16 | overlap ≥ 0.5 or a shared 6-word sequence with *any* of the 245 evaluation items (test, dev, safety, probes, spares). This rule is deliberately strict: it also drops a few generic phrasings |
| Answered and rewritten | 860 | none cut off by the length limit |
| Passed checker and code gates | 630 (73.3%) | 434 on the first pass; the 240 rewrites that failed *only* the length gate got one retry with a hard word budget, then were re-checked |
| **Selected for training** | **500** | about 57 per category, 40 serious-situation examples (see §5) |
| Held out for validation | 130 | accepted but not selected; used only to measure validation loss (Gate 4) |

## 4. Why examples were rejected (230 rejected; one example can fail several checks)
| Check | Kind | Failures |
|---|---|---:|
| Information lost (a fact, number, step or warning missing) | checker | 90 |
| Rewrite more than 1.3× the reference's length | code | 55 |
| A number in the reference missing from the rewrite | code | 43 |
| New claims added that weren't in the reference | checker | 30 |
| Meaning blurred by a metaphor | checker | 25 |
| Code changed (code blocks must survive word for word) | code | 21 |
| **The teacher's own plain answer was wrong** | checker | 17 |
| Too little F1 language | code | 6 |
| Longer than 450 words | code | 3 |
| Order of steps changed | checker | 1 |
| Question itself mentioned racing | code | 1 |

## 5. What the selected 500 look like
- **By category:** procedure 57, quantitative 57, factual 57, explanation 68, advice 57, writing 50, technical 57, uncertainty 57, serious situations 40.
- **Length:** a median of **1.20×** the plain reference answer (241 words). A dry run on throwaway data had 1.33×, and its model learned to embellish, so the limit was tightened (decision D3).
- **Variety:** no opening phrase starts more than 4.0% of answers (cap: 5%). "Lights out" appears in 4.6%. 42 different F1 terms are used.

## 6. Known limitations
- **The data is synthetic.** It reflects one teacher model's knowledge and style. The checker catches many factual errors (17 wrong reference answers were removed), but it is itself a model, not ground truth.
- **The held-out validation set is unevenly spread**: lots of advice and uncertainty, only 4 technical and 7 quantitative examples. It's good enough for a loss curve, but not a per-category measure.
- **English only**, and British/American spelling is mixed.
- Serious-situation examples keep racing language light by design, but there are only 40 of them.

## 7. Reproduce
```bash
bash scripts/training_data/pod_setup_teacher.sh          # on an 80 GB GPU pod
bash scripts/training_data/pod_serve.sh teacher
python scripts/training_data/teacher_pipeline.py questions --dir data/train --per-cell 3
python scripts/training_data/teacher_pipeline.py answers   --dir data/train
python scripts/training_data/teacher_pipeline.py retry     --dir data/train   # after the first check
bash scripts/training_data/pod_serve.sh checker
python scripts/training_data/teacher_pipeline.py check       --dir data/train
python scripts/training_data/teacher_pipeline.py check-retry --dir data/train
python scripts/training_data/teacher_pipeline.py select      --dir data/train --n 500
```
The rules for every gate live in `src/pitwall/teacher.py` and are unit-tested in `tests/test_teacher.py`.
