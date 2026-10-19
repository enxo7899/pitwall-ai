---
title: PITWALL AI
emoji: 🏁
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
license: apache-2.0
short_description: Qwen3-1.7B + LoRA that answers anything in F1 language
models:
  - Qwen/Qwen3-1.7B
  - enxo7899/pitwall-qwen3-1.7b-lora
---

# PITWALL AI · engine

The inference backend for PITWALL AI: `Qwen/Qwen3-1.7B` with a LoRA adapter that answers any everyday question in the language of Formula 1.

- **Arms:** `C` is the fine-tuned model. `A` (base model) and `B` (base model + the frozen F1 system prompt) are the baselines it was evaluated against. All three share one copy of the base weights; the adapter is switched off for A and B.
- **API:** `answer(question, arm, seed)` streams `(text, telemetry)`. It only answers requests carrying the website's access key (the `PITWALL_KEY` Space secret), so the website's rate limits cannot be bypassed. Answers are capped at 512 new tokens; the evaluation allowed 1,536.
- **Website:** [pitwall-ai-two.vercel.app](https://pitwall-ai-two.vercel.app): live demo, interactive report and results explorer
- **Model:** [enxo7899/pitwall-qwen3-1.7b-lora](https://huggingface.co/enxo7899/pitwall-qwen3-1.7b-lora)
- **Code, data and the full evaluation:** [github.com/enxo7899/pitwall-ai](https://github.com/enxo7899/pitwall-ai)

Not affiliated with Formula 1, the FIA, or any team or driver.
