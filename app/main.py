"""PITWALL AI demo: FastAPI backend + Gradio front end in one process.

Run locally (Mac, MPS):
    PITWALL_ADAPTER=outputs/RUN/adapter uv run --group app uvicorn app.main:app --port 7860
Then open http://localhost:7860 (UI) or http://localhost:7860/docs (API).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RESULTS_FILE = Path(os.environ.get("PITWALL_RESULTS", "experiments/results/summary.json"))


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    seed: int = 0


class ArmQuestion(Question):
    arm: Literal["A", "B", "C"] = "C"


def create_app(service) -> FastAPI:
    """The service is passed in so tests can use a fake one without loading a model."""
    api = FastAPI(title="PITWALL AI", description="Answers any question in the language of Formula 1.")

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", **service.info()}

    @api.post("/answer")
    def answer(body: ArmQuestion) -> dict:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is empty")
        return service.answer(question, body.arm, body.seed)

    @api.post("/compare")
    def compare(body: Question) -> dict:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is empty")
        return {arm: service.answer(question, arm, body.seed) for arm in ("A", "B", "C")}

    @api.get("/results")
    def results() -> dict:
        if not RESULTS_FILE.exists():
            raise HTTPException(status_code=404, detail="no evaluation results yet")
        return json.loads(RESULTS_FILE.read_text())

    return gr.mount_gradio_app(api, build_ui(service), path="/")


def build_ui(service) -> gr.Blocks:
    def run_compare(question: str):
        """A generator: each arm's answer appears as soon as it is ready (they share one GPU, so they run in turn)."""
        if not question or not question.strip():
            raise gr.Error("Type a question first.")
        out = ["*waiting…*"] * 3
        for i, arm in enumerate(("A", "B", "C")):
            out[i] = "*generating…*"
            yield list(out)
            r = service.answer(question.strip(), arm)
            note = (
                f"\n\n*{r['n_tokens']} tokens · {r['seconds']} s"
                + ("" if r["finished"] else " · cut off at the limit")
                + "*"
            )
            out[i] = r["answer"] + note
        yield list(out)

    def load_results() -> str:
        if not RESULTS_FILE.exists():
            return "No evaluation results yet. They appear here after Gate 5."
        return "```json\n" + RESULTS_FILE.read_text()[:20000] + "\n```"

    with gr.Blocks(title="PITWALL AI") as ui:
        gr.Markdown(
            "# PITWALL AI\nAsk anything. The same base model answers three ways: plain, prompted, and fine-tuned."
        )
        with gr.Tab("Compare"):
            question = gr.Textbox(
                label="Your question (Enter to send)", lines=1, placeholder="How do I get red wine out of a carpet?"
            )
            go = gr.Button("Answer", variant="primary")
            with gr.Row(equal_height=False):
                with gr.Column():
                    gr.Markdown("### A · Base model")
                    a = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### B · Base + F1 system prompt")
                    b = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### C · PITWALL AI (fine-tuned)")
                    c = gr.Markdown()
            go.click(run_compare, inputs=question, outputs=[a, b, c])
            question.submit(run_compare, inputs=question, outputs=[a, b, c])
        with gr.Tab("Results"):
            results = gr.Markdown()
            gr.Button("Load results").click(load_results, outputs=results)
    return ui


def _default_app() -> FastAPI:
    from pitwall.service import InferenceService

    adapter = os.environ.get("PITWALL_ADAPTER")
    if not adapter:
        raise SystemExit("set PITWALL_ADAPTER to a local adapter folder or a Hugging Face repo id")
    return create_app(InferenceService(adapter))


app = _default_app() if os.environ.get("PITWALL_ADAPTER") else None
