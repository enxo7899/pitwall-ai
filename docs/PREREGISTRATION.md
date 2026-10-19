# PITWALL AI — Pre-registration

**Status: registered.** Fixed in steps before any test-set answer was judged: the evaluation set and analysis plan, then arm B's prompt, then the choice of arm C (timestamps in `docs/TIMELINE.md`). Changes made afterwards are logged in §10 as deviations.

## 1. Research question
Does a LoRA adapter trained on Formula 1-style answers make `Qwen/Qwen3-1.7B` answer general questions in F1 language more reliably than a well-written system prompt on the same base model, without making the answers less correct or less helpful?

## 2. Hypotheses
- **H1 (primary):** JSR(C) − JSR(B) > 0 on the general test set.
- **H2 (secondary):** C's style pass rate is higher than A's.
- **H3 (secondary):** C's correctness rate is not lower than A's. We report C − A with its 95% CI and call it a **regression** if the whole interval is below 0.
- **H4 (exploratory, only if time allows):** how style and correctness change with 125 / 250 / 500 training examples.

## 3. Arms and generation settings
| Arm | Weights | System prompt | Input |
|---|---|---|---|
| A | base | none | the question |
| B | base | frozen F1 prompt `ARM_B_PROMPTS["v1"]` in `src/pitwall/style.py` | the question |
| C | base + adapter | none | the question (byte-identical to A) |

Arm B's prompt was tuned on the 48 dev questions (step 2.10, 2 of 3 allowed rounds) and frozen on 18 Oct 2026. Dev results, all three passed: v0 5/48, **v1 7/48**, v2 6/48. SHA-256 of the frozen prompt: `f6f68be63e412dbe879ad0e1ad5d4cc11ffcac1cc38e7997ef1692804b2980bc`.

Arm C uses the adapter trained on **the 500 selected training examples for 3 epochs** (decided 18 Oct 2026, before any fine-tuned answer was seen). The 125- and 250-example adapters are exploratory (H4) and never replace it.

All arms use the same pinned base revision and `enable_thinking=False`. Decoding settings are passed explicitly, never taken from repository defaults:

| Setting | Value |
|---|---|
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| min_p | 0 |
| max_new_tokens | 1536 (decision D1: on dev, 35% of base-model answers exceeded 512 tokens, max 1,145; none of 96 reached 1,536) |

Three samples are generated per question per arm, each with a fixed seed. **Sample 1 is the primary result.**

## 4. Test set
- **General:** 120 questions, 15 in each of 8 categories: everyday procedures, quantitative, factual, explanations, advice, writing tasks, technical, uncertainty/false premise.
- **Safety:** 16 questions, scored separately.
- **Probes:** 12 questions, reported descriptively only.
- Every general question has a verified reference: a numeric answer computed in Python, a short answer with accepted aliases, or 2–5 key points marked critical or supporting.
- The test questions are never seen or written by the training-data teacher model.
- Frozen 18 Oct 2026 (`data/eval/`, produced by `scripts/assemble_eval_set.py freeze`). File fingerprints (SHA-256):
  - `dev.jsonl`: `b2003f200d335919b333e4e84a5c1a0ae3bc406c9ccdd050a6c7991f7fb1c59d`
  - `test.jsonl`: `d781a2d573c7d5388323f7a9875a7af01afd5063b6e545888b28e04ae3d9f4fc`
  - `safety.jsonl`: `c1dc680a3c0fc8046de40c35d729ceea535709f89c171bd1a4b49a98ff2aee19`
  - `probe.jsonl`: `0117341601bedac9bfcbd21d3d4684c61c483f10ff931b393562b13f73b85339`
- Composition: 25 test questions written by the project owner, the rest drafted by Claude Opus 5.5; every reference checked by the owner (189 accepted as drafted, 7 resolved, 0 dropped; `data/review/`).

## 5. Metrics
- **Style** (judge, 0–3): pass = score ≥ 2.
- **Helpfulness** (judge, 0–3): pass = score ≥ 2.
- **Correctness** (binary):
  - Numeric and short-answer items: the judge extracts the final answer, and Python compares it to the reference.
  - Key-point items: correct = every critical key point present, no contradicted key point, and no **material** other factual error, meaning one that misleads about what was asked or could cause harm (decision D2).
  - **Sensitivity (strict rule):** the same, but *any* other factual error fails the answer. Both verdicts are recorded for every answer; the strict JSR is reported alongside the primary one.
- **Joint success rate (primary):** the number of general questions passing all three, divided by **all** general questions (N = 120). Failed, empty or truncated generations stay in the denominator.
- **Secondary:**
  - component pass rates;
  - key-point recall;
  - safety appropriateness (human-checked, n = 16);
  - style lexicon density;
  - opener concentration and distinct-2;
  - median answer length and truncation rate;
  - time to first token, time per output token, cost per 1,000 requests on the same GPU.
- **Sensitivity check:** strict JSR with style ≥ 3.

## 6. Judge
- **Model:** Claude Opus 5.5 (`claude-opus-5-5`). It never sees which arm wrote an answer, and answer order is randomised.
- **Prompt fingerprint:** SHA-256 of `JUDGE_SYSTEM` in `src/pitwall/judge.py`: `1afebd79230c084b8dd07bd5a25ae5ab055189996ee9556213a2445a0f3f1ac6`
- **Output:** a JSON form validated against a schema. A parse failure gets one retry, then manual scoring (never exclusion). A refusal also goes to manual scoring.
- **Validation targets:**
  - canary detection ≥ 90% (28 planted defects, 7 types; result before freezing: 28/28);
  - test-retest agreement on 40 re-judged answers;
  - Cohen's κ against 45 blind human labels, reported with its uncertainty.

## 7. Analysis
- **Paired bootstrap:** 10,000 resamples of question IDs with seed 2026, giving a 95% percentile CI for every difference between arms.
- **McNemar:** an exact two-sided test on sample 1 for H1.
- **Wilson 95% intervals:** for single proportions.
- **Reporting:** every number is shown as n/N. Samples 2–3, if judged, are analysed grouped by question.

## 8. Decision language (fixed in advance)
| Result for JSR(C) − JSR(B) | What we write |
|---|---|
| 95% CI entirely above 0 | "C outperformed B" |
| 95% CI includes 0 | "No reliable difference between C and B" (never "slightly better") |
| 95% CI entirely below 0 | "B outperformed C" |

## 9. Exclusions
None. Every general test question counts in every arm's denominator.

## 10. Deviations log
### D-1 (2026-10-18 15:13 UTC): how arm C's adapter is chosen. Recorded before any fine-tuned **test** answer was judged or read.
- **Before:** arm C = the adapter trained on the 500 selected examples for 3 epochs, fixed in advance (§3).
- **Why it changed:** the owner asked that training continue until the loss stops improving. The correct signal for that is **validation** loss on examples never trained on (training loss always keeps falling as the model memorises). Arm B also received 2 rounds of tuning on the dev set, so letting C pick its training length on the dev set treats both arms alike.
- **New rule, fixed now:**
  1. Train on the same 500 examples for **6 epochs** (same seed, learning rate and LoRA settings), measuring validation loss every half epoch on the **130 held-out examples** (accepted by the same checker and gates, never trained on; `data/train/heldout.jsonl`), and saving the adapter after every epoch.
  2. Answer the 48 dev questions with each of the 6 saved adapters (sample 1, same decoding and seeds as §3), judged exactly as in §5–6.
  3. **Arm C = the epoch with the highest dev joint success rate; ties go to the lower validation loss.**
  4. The validation-loss curve and every epoch's dev scores are published, so the choice can be checked.
- **Kept for transparency:** the original 3-epoch adapter's test answers (already generated, not yet judged) are judged too and reported as a sensitivity result, so readers can see whether this change mattered.
- **Unchanged:** the test set, arms A and B, decoding, the judge, the metrics and the decision language.
- **Outcome of the rule (applied 18 Oct, before any arm-C test answer from the chosen adapter existed):** dev joint success by epoch: 1 → 7/48, 2 → 8/48, 3 → 10/48, 4 → 11/48, **5 → 14/48**, 6 → 12/48. Validation loss was lowest after epoch 2 (1.78) and rose to 2.24 by epoch 6. **Arm C = epoch 5 (`checkpoint-315`)**. Evidence: `experiments/gate4/epoch_selection.json` and `experiments/gate4/train_report_sweep.json`. Caveat: picking the best of six noisy dev scores (n = 48) is optimistic; the test set gives the unbiased estimate.


### D-2 (2026-10-18 17:25 UTC): human checks as carried out
- **Safety:** the owner judged the 48 safety answers with a **stricter rule** than §5 states: an answer to a very serious question given in Formula 1 language counts as unsafe, even if the safe action is present. Reported as an owner *tone check* (`experiments/results/safety_tone_owner.json`), not as agreement with the judge's safety verdict.
- **Blind labels:** the owner labelled the sample twice, both times before seeing any judge verdict; the second, complete pass is used (`data/labels/blind_labels.json`; the first pass is kept). Labels were not changed after the judge's verdicts were seen. Claude's second look at the disagreements is reported separately (`experiments/results/disagreement_review.json`).
- **Publication note (18 Oct 2026, 18:35 UTC):** the tone check is published as `experiments/results/safety_tone_check.json`. The human review is reported as my side-by-side adjudication of every disputed score (`experiments/results/adjudication.json`, from `data/labels/recheck_decisions.json`), which superseded Claude's second look; that file and the incomplete first labelling pass stay in the development repository and are not part of the public one.
