# Scripts

One entry point per pipeline step, grouped by stage. Run everything from the repository root; the reusable logic lives in `src/pitwall/`, and these scripts only handle input, output and orchestration.

| Stage | Script | What it does |
|---|---|---|
| **Evaluation set** | `eval_set/draft_eval_set.py` | Generates candidate questions and answer keys to the category specification (`src/pitwall/eval_spec.py`) |
| | `eval_set/assemble_eval_set.py` | Adds the hand-written questions, applies review decisions, freezes the set with SHA-256 fingerprints |
| | `eval_set/build_review_sheet.py` | Builds the HTML sheet used to verify every answer key |
| **Training data** | `training_data/pod_setup_teacher.sh` | Prepares an 80 GB GPU pod: environments and both model downloads, in parallel |
| | `training_data/pod_serve.sh` | Serves the teacher or the checker with vLLM and waits until it is healthy |
| | `training_data/teacher_pipeline.py` | Questions → plain answers → F1 rewrites → checker → retries → stratified selection |
| **Training** | `training/smoke_train.py`, `training/gpu_sweep.py` | Gate 1 feasibility: a 30-step smoke run and a memory/speed sweep |
| | `training/train_adapter.py` | Trains the LoRA adapter; optional validation loss and a checkpoint every epoch |
| | `training/pod_epoch_sweep.sh` | The 6-epoch run with validation, then dev answers for every saved epoch |
| | `training/pod_train_and_generate.sh` | Training followed by dev and test generation, unattended on the pod |
| | `training/extract_training_log.py` | Turns the trainer's console log into a clean per-step table |
| **Evaluation** | `evaluation/generate_arms.py` | Answers from arms A, B and C on one model load (A and C get identical inputs) |
| | `evaluation/judge_batch.py` | Judges answers through the Message Batches API (same judge and form, half price) |
| | `evaluation/judge_generations.py` | The same judge in real time, used for dev-set iterations |
| | `evaluation/run_canaries.py` | Validates the judge on 28 answers with planted mistakes |
| | `evaluation/summarize_run.py` | Quick per-arm summary for a dev run |
| | `evaluation/analyze_results.py` | The pre-registered test-set analysis; writes `experiments/results/summary.json` |
| | `evaluation/build_label_sheet.py`, `evaluation/build_recheck_sheet.py`, `evaluation/label_agreement.py` | Blind human labels, side-by-side adjudication, and agreement with the judge |
| | `evaluation/benchmark_latency.py` | Inference cost per arm: prompt tokens, time to first token, decoding speed |
| **Tools** | `tools/audit.py` | Independent checks of the published claims (freeze, leakage, metrics, adapter, repository) |
| | `tools/make_figures.py` | Draws every figure in `docs/figures/` from the committed results |
| | `tools/export_web_data.py` | Exports the committed results as the website's data files (same metric functions as the analysis) |
| | `tools/build_report.py` | Typesets the 2-page report PDF from `docs/REPORT.md` |
| | `tools/publish_hf.py` | Publishes the adapter (checked against the audited hash) and builds and publishes the engine Space |
| | `tools/runpod.py` | Creates, inspects and deletes GPU pods through RunPod's API |
| | `tools/check_env.py`, `tools/check_secrets.py` | Confirms which API keys are set (never prints them) and blocks secrets from commits |

Paths in `docs/PREREGISTRATION.md` refer to the flat `scripts/` layout used when that document was frozen.

The report PDF is built from `docs/REPORT.md`: pandoc converts the Markdown to Typst and Typst typesets it with the page design in `docs/templates/report.typ` (the paired-differences chart is the vector version drawn by `make_figures.py`):

```bash
uv run --group docs python scripts/tools/build_report.py
```
