# PITWALL AI · Decision records

Each record states the context, the decision, the alternatives I rejected and why, the consequences, and the evidence. A record moves from **Proposed** to **Accepted** when the project owner approves it at a gate review. If a later gate overturns a decision, we add a new record rather than editing history.

| ID | Decision | Status |
|---|---|---|
| ADR-001 | Base model: `Qwen/Qwen3-1.7B`, non-thinking mode | Accepted (Gate 0, 18 Oct) |
| ADR-002 | Method: supervised fine-tuning with LoRA on a 16-bit base | Accepted (Gate 0, 18 Oct) |
| ADR-003 | Training data: self-hosted open-weight teacher (Qwen3.5-27B); commercial models never touch training data | Accepted, revised 18 Oct |
| ADR-004 | Evaluation: three arms, frozen test set, joint success rate, validated judge | Accepted (Gate 0, 18 Oct) |
| ADR-005 | Compute and app: rented GPU for training, local Mac for the demo | Accepted (Gate 0, 18 Oct) |

---

## ADR-001 · Base model: `Qwen/Qwen3-1.7B` in non-thinking mode

**Context.** The brief asks for a small open-weight model (0.5–4B suggested) sized to the available hardware. Training runs on a rented 24 GB GPU; the live demo runs on an Apple M1 Pro laptop with 16 GB of memory.

**Decision.** Use [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0), pinned to an exact commit at Gate 1. Thinking mode is switched off (`enable_thinking=False`) in every arm. Arms A and C get no system prompt.

**Rejected alternatives.**
- `Qwen/Qwen3.5-2B` (Feb 2026): a vision-language model with a hybrid linear-attention architecture, and lower instruction-following than Qwen3-1.7B in non-thinking mode (IFEval 61.2 vs 68.2, from the [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)).
- `Qwen/Qwen3-4B-Instruct-2507`: stronger, but about 2.3× the memory and latency for the laptop demo. Kept as a "more time" replication.
- `Qwen/Qwen2.5-1.5B-Instruct`: older, and its chat template silently inserts a default system prompt. Kept as the fallback.
- Llama 3.2 3B and Gemma 3 1B: gated downloads with custom licences.

**Consequences.** Qwen3's chat template inserts an empty `<think></think>` block, and the repository's default sampling settings are the thinking-mode ones. Both must be handled explicitly and are tested in Gate 1.

**Revisit if.** The Gate 1 template tests still fail after 60 minutes of debugging: switch to the fallback model.

---

## ADR-002 · Method: supervised fine-tuning with LoRA on a 16-bit base

**Context.** The model's behaviour has to change through training, not prompting, on hardware that can be rented for a few dollars, and it has to be possible to compare the trained model fairly against the untrained one.

**Decision.** Supervised fine-tuning (SFT) with TRL's `SFTTrainer` and a PEFT LoRA adapter: rank 16, alpha 32, dropout 0.05, applied to all linear layers (attention and MLP). Data in conversational prompt-completion format, so the loss is computed on answer tokens only. Base weights in bf16. Starting learning rate 2e-4, effective batch 8, 3 epochs, best checkpoint chosen on the dev set. All values are confirmed by measurement in Gate 1.

**Rejected alternatives.**
- **Full fine-tuning:** more memory, a 3.4 GB artefact instead of ~35–70 MB, more forgetting of general ability ([LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)), and no cheap on/off switch for the comparison.
- **QLoRA:** solves a memory problem this project does not have (the 16-bit model fits easily). It adds a CUDA-only dependency and slower steps, and the adapter would be trained against 4-bit weights but served on 16-bit weights on the Mac.
- **DPO / preference optimisation:** needs pairs of better and worse answers and a second training stage. A "more time" item, useful if SFT shows a consistent failure pattern.
- **Prompt tuning:** a weaker fit for teaching a whole writing style.

**Consequences.** One copy of the base model can serve all three arms by switching the adapter off and on. About 17.4 million trainable parameters (about 1% of the model).

**Evidence.** [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (LoRA matches full fine-tuning on small datasets; apply it to all layers; learning rate about 10× full fine-tuning); [QLoRA](https://arxiv.org/abs/2305.14314); [TRL SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer).

**In my own words** *(written at the start of the project)*:

> Full fine-tuning would retrain the whole model, which takes a huge amount of training; it isn't efficient and might damage parts of the model we didn't want to change. QLoRA is similar to LoRA, but compressed. LoRA fine-tunes only about 1% of the parameters by adding a separate calculation alongside the original weights, and we can turn the change on and off.
>
> *(Lightly edited for spelling only; the original is kept in my working notes.)*

---

## ADR-003 · Training data: a self-hosted open-weight teacher; commercial models never touch training data

**Status.** Accepted, revised 18 Oct 2026. I set the requirement (the best-performing open model that can be self-hosted on rented GPUs) and approved the choice after reviewing an AI-assisted benchmark comparison. The first version used `Qwen/Qwen3-235B-A22B-Instruct-2507` through DeepInfra's API; this revision replaces it.

**Context.** The project needs ~500 high-quality question → Formula 1-style answer pairs whose facts are correct. The brief allows commercial LLMs for data generation, but those providers' own terms forbid using outputs to build competing models (see below), and the trained adapter is published.

**Decision.**
- **Teacher:** [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) (Apache-2.0; dense, 27.8B parameters, ≈ 56 GB in bf16), self-hosted with vLLM on one rented 80 GB GPU (H100 or A100) on RunPod.
- **Checker:** [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (Apache-2.0; its usage policy only asks users to comply with applicable law; ≈ 64 GB), run on the same pod after the teacher. It comes from a different model family, so its check is independent of the teacher. It verifies that every fact and step of the neutral answer survives the F1 rewrite, with the same meaning and in the same order.
- **Pipeline:** a neutral, correct answer first, then the F1 rewrite. Automatic quality gates, a look-alike check against the frozen dev/test sets, then human review. Whether the neutral-answer stage uses thinking mode (reasoning discarded, only the final answer kept) is decided in the pilot.
- **Pilot first:** 10 examples, judged by me, before the full run. If they are weak, the fallback teacher is `Qwen/Qwen3.5-122B-A10B-FP8` (≈ 127 GB) on two 80 GB GPUs.
- **Rule:** no commercial-model output enters or filters the training data.

**Why this model.** Qwen's own benchmark table on the [Qwen3.5-122B-A10B model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B). These are vendor-reported scores, not independently verified:

| Benchmark | Qwen3.5-27B | Qwen3.5-122B-A10B | Qwen3-235B-A22B | gpt-oss-120b |
|---|---|---|---|---|
| MMLU-Pro (knowledge) | 86.1 | 86.7 | 84.4 | 80.8 |
| IFEval (instruction following) | **95.0** | 93.4 | 87.8 | 88.9 |
| IFBench (instruction following) | **76.5** | 76.1 | 51.7 | 69.0 |
| MultiChallenge (multi-turn) | 60.8 | 61.5 | 50.2 | 45.3 |

The 27B dense model is within a point of the 122B mixture-of-experts on knowledge and slightly better at following instructions, while fitting on one GPU with half the download. Writing quality isn't in this table, which is why the 10-example pilot decides.

**Why self-host rather than use an API provider.**
- **Shorter licence chain:** only the model licences (Apache-2.0) and RunPod's compute terms apply. RunPod's [Terms of Service](https://www.runpod.io/legal/terms-of-service) (last updated 24 Mar 2026, checked 18 Oct 2026) state: "We do not assert any ownership over Your Content. You retain full ownership of all of Your Content".
- **Exact reproducibility:** I record the model revision, precision, vLLM version, sampling settings and seeds, instead of relying on an API whose serving setup can change without notice.
- **No extra account:** uses the existing RunPod account.
- **Cost:** ≈ $3–5 (about 1.5 h on one 80 GB GPU at ≈ $2–2.9/h), against ≈ $1–2 through an API. A small premium for the gains above.

**Rejected alternatives.**
- **Claude, GPT or Gemini APIs as the teacher.** All three providers restrict using their outputs to build competing models:
  - **Anthropic:** its [help article](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model) (16 Mar 2026) says outputs may not train competing models, and lists general-purpose chatbots and using outputs as training targets.
  - **OpenAI:** its [Terms of Use](https://openai.com/policies/terms-of-use/) prohibit "Use Output to develop models that compete with OpenAI".
  - **Google:** its [Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms) (updated 28 Apr 2026) state "You may not use the Services to develop models that compete with the Services".
  - The same companies' open-weight models carry no such restriction: OpenAI's `gpt-oss-120b` (the checker) and Google's Gemma 4 family are Apache-2.0.
- **DeepInfra API with Qwen3-235B** (the first version of this record): cheap, but it adds third-party terms (a broad "no competitive use" clause) and an unknown serving configuration. Qwen3.5-27B also scores higher on the table above.
- **Qwen3.5-122B-A10B or Qwen3.5-397B as the primary teacher:** more GPUs and a bigger download for no measured benefit on these benchmarks. The 122B is kept as the fallback.

**Consequences.** Each example records its teacher id and revision, vLLM version, sampling settings, seed, prompt-template fingerprint, GPU type and quality-gate results. Gate 3 needs an 80 GB GPU for about 1.5 hours; the LoRA training itself still runs on a 24 GB GPU.

---

## ADR-004 · Evaluation: three arms, frozen test set, joint success rate, validated judge

**Context.** The brief cares more about evaluation than raw quality, and asks how one knows the measurements are trustworthy.

**Decision.** Full details in [`PREREGISTRATION.md`](PREREGISTRATION.md). In short:
- **Arms:** A (base), B (base + frozen F1 system prompt, tuned on dev only), C (base + adapter). A and C receive byte-identical inputs.
- **Data order:** test and dev sets are built and fingerprinted **before** any training data exists.
- **Primary metric:** joint success rate. Style ≥ 2 **and** correct **and** helpful ≥ 2, over all general test questions.
- **Judge:** Claude Opus 5.5, blind to which arm wrote the answer. "LLM extracts, code decides" for correctness.
- **Judge validation:** 24 planted-mistake canaries, a test-retest check, and 45 blind human labels with Cohen's κ.
- **Statistics:** sample 1 is the primary result; paired bootstrap and McNemar test.

**Rejected alternatives.**
- Holistic 1–10 scores: less reliable than anchored binary checks.
- An unvalidated judge.
- Recall@k: a retrieval metric, and this system has no retriever.
- BLEU/ROUGE against references: they punish valid stylistic rewording.
- A judge from the teacher's model family: models favour their own family's outputs.

**Consequences.** Style itself reveals arm A versus B/C, so blinding is partial for comparisons against A. The primary comparison, B versus C, is fully blind.

---

## ADR-005 · Compute and app: a rented GPU for training, the local Mac for the demo

**Context.** The laptop has no NVIDIA GPU. The demo happens over a video call shared from the laptop.

**Decision.**
- **Training:** train and batch-generate on a RunPod 24 GB GPU (RTX 4090 or L4), driven over SSH and stopped after every job, with cost logged per job. The provider choice is confirmed by Gate 1 measurements.
- **Demo:** runs locally on the M1 Pro (MPS, fp16) behind a FastAPI backend with a Gradio page. One model sits in memory, the adapter is switched per arm, and a lock serialises requests.
- **Cost comparison:** on a self-hosted GPU-seconds basis.

**Rejected alternatives.**
- Free T4 notebooks as the primary path: availability is uncertain under a deadline.
- Hugging Face Spaces: compute Spaces need a paid plan, and cold starts.
- A Next.js frontend: 3–5 extra hours and a second runtime.
- MLX training: its adapter format is not the standard PEFT format reviewers expect.

**Revisit if.** Gate 1 measures fewer than 10 tokens/s on the Mac: use streaming, one arm at a time, plus a dated fallback recording.
