"""PITWALL AI engine: the evaluated model behind a small streaming API, served on Hugging Face ZeroGPU.

`scripts/tools/publish_hf.py space` assembles the Space from this file and the project's own library
code (`src/pitwall`), so the chat template, the decoding settings and arm B's frozen system prompt are
exactly the ones that were evaluated. Answers are capped at 512 new tokens to keep the demo quick;
the evaluation allowed 1,536.

Run locally (Mac or CUDA):  PITWALL_ADAPTER=outputs/sweep/checkpoint-315 python app.py
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from pitwall.config import END_OF_TURN, MODEL_ID, MODEL_REVISION
from pitwall.generate import generation_kwargs
from pitwall.style import ARM_B_SYSTEM_PROMPT
from pitwall.templates import render_prompt

try:  # present on Hugging Face Spaces: borrows a GPU only while a decorated function runs
    import spaces

    on_gpu = spaces.GPU(duration=45)
except ImportError:  # anywhere else the model simply stays on the local device

    def on_gpu(fn):
        return fn


ADAPTER = os.environ.get("PITWALL_ADAPTER", "enxo7899/pitwall-qwen3-1.7b-lora")
REVISION_FILE = Path(__file__).with_name("adapter_revision.txt")  # written by the publish script
ADAPTER_REVISION = REVISION_FILE.read_text().strip() if REVISION_FILE.exists() else None
MAX_NEW_TOKENS = 512
MAX_QUESTION_CHARS = 1000
# Set as a Space secret: only requests carrying this key (sent by the website's server) are answered,
# so nobody can use the engine directly. Without the secret (local runs) every request is accepted.
ACCESS_KEY = os.environ.get("PITWALL_KEY", "")
WEBSITE = os.environ.get("PITWALL_WEBSITE", "https://pitwall-ai-two.vercel.app")
ARMS = {
    "A": {"label": "Base model", "system_prompt": None, "adapter": False},
    "B": {"label": "Base + F1 system prompt", "system_prompt": ARM_B_SYSTEM_PROMPT, "adapter": False},
    "C": {"label": "PITWALL AI (fine-tuned)", "system_prompt": None, "adapter": True},
}


def pick_device() -> tuple[str, torch.dtype]:
    if os.environ.get("SPACE_ID") or torch.cuda.is_available():  # ZeroGPU emulates CUDA at load time
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


DEVICE, DTYPE = pick_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE)
# Load the adapter on the CPU, then move everything at once: ZeroGPU reports a GPU at start-up but only
# lends a real one inside @spaces.GPU functions, so loading weights straight onto "cuda" would fail.
model = PeftModel.from_pretrained(base, ADAPTER, revision=ADAPTER_REVISION, torch_device="cpu").to(DEVICE).eval()
END_ID = tokenizer.convert_tokens_to_ids(END_OF_TURN)


def answer(question: str, arm: str, seed: int, request: gr.Request):
    """The public endpoint: rejects requests without the website's key, validates input, then streams.

    Every check runs before a GPU is borrowed, so rejected calls cost nothing.
    """
    sent = (request.headers.get("x-pitwall-key", "") if request else "") or ""
    if ACCESS_KEY and not hmac.compare_digest(sent, ACCESS_KEY):
        raise gr.Error(f"This engine only serves the PITWALL AI website: {WEBSITE}")
    question = (question or "").strip()
    if not question:
        raise gr.Error("Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise gr.Error(f"Keep questions under {MAX_QUESTION_CHARS:,} characters.")
    if arm not in ARMS:
        raise gr.Error("Arm must be A, B or C.")
    yield from generate(question, arm, int(seed or 0))


@on_gpu
def generate(question: str, arm: str, seed: int):
    """Stream one arm's answer. Yields (text so far, stats); the last stats say whether it finished.

    Arms A and B run with the adapter switched off, so all three share the same base weights.
    Gradio runs one request at a time (concurrency limit 1), so the switch is never shared.
    """
    spec = ARMS[arm]
    prompt = render_prompt(tokenizer, question, spec["system_prompt"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    output: dict = {}

    def run() -> None:
        switch = nullcontext() if spec["adapter"] else model.disable_adapter()
        torch.manual_seed(int(seed))
        with torch.no_grad(), switch:
            output["ids"] = model.generate(
                **inputs,
                **generation_kwargs(max_new_tokens=MAX_NEW_TOKENS),
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
            )

    stats = {"arm": arm, "label": spec["label"], "done": False}
    start = time.perf_counter()
    worker = threading.Thread(target=run)
    worker.start()
    text = ""
    for piece in streamer:
        text += piece
        yield text, stats
    worker.join()

    seconds = time.perf_counter() - start
    new_ids = output["ids"][0, inputs["input_ids"].shape[1] :].tolist()
    finished = END_ID in new_ids
    n_tokens = new_ids.index(END_ID) if finished else len(new_ids)
    yield (
        text,
        {
            **stats,
            "done": True,
            "tokens": n_tokens,
            "seconds": round(seconds, 2),
            "tokens_per_second": round(n_tokens / seconds, 1) if seconds else None,
            "finished": finished,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    )


with gr.Blocks(title="PITWALL AI engine") as demo:
    gr.Markdown(
        "## PITWALL AI · engine\n"
        "The inference backend for PITWALL AI: Qwen3-1.7B with a LoRA adapter that answers any question in "
        "Formula 1 language. It serves the project website, which adds rate limits in front of it.\n\n"
        f"**Try it on the website:** [{WEBSITE.removeprefix('https://')}]({WEBSITE}) · "
        "**Model:** [enxo7899/pitwall-qwen3-1.7b-lora](https://huggingface.co/enxo7899/pitwall-qwen3-1.7b-lora) · "
        "**Code:** [github.com/enxo7899/pitwall-ai](https://github.com/enxo7899/pitwall-ai)"
    )
    with gr.Row(visible=False):  # inputs and outputs of the API endpoint; the website is the interface
        question_box = gr.Textbox()
        arm_choice = gr.Radio(choices=list(ARMS), value="C")
        seed_box = gr.Number(value=0, precision=0)
        answer_box = gr.Markdown()
        stats_box = gr.JSON()
        go = gr.Button()
    go.click(answer, [question_box, arm_choice, seed_box], [answer_box, stats_box], api_name="answer")

demo.queue(default_concurrency_limit=1, max_size=16)  # one generation at a time; a short queue limits floods

if __name__ == "__main__":
    demo.launch()
