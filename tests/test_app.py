"""API behaviour with a fake model service: routes, validation and error paths (no model download)."""

from __future__ import annotations

import pytest

pytest.importorskip("gradio")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def answer(self, question: str, arm: str, seed: int = 0) -> dict:
        self.calls.append((question, arm))
        return {
            "arm": arm,
            "label": arm,
            "answer": f"{arm}: {question}",
            "n_tokens": 3,
            "finished": True,
            "seconds": 0.01,
            "tokens_per_second": 300.0,
        }

    def info(self) -> dict:
        return {"device": "fake", "adapter": "none", "max_new_tokens": 8, "arms": {}}


@pytest.fixture()
def client_and_service():
    service = FakeService()
    return TestClient(create_app(service)), service


def test_health(client_and_service):
    client, _ = client_and_service
    assert client.get("/health").json()["status"] == "ok"


def test_answer_defaults_to_the_fine_tuned_arm(client_and_service):
    client, service = client_and_service
    response = client.post("/answer", json={"question": "  How do tides work?  "})
    assert response.status_code == 200 and response.json()["arm"] == "C"
    assert service.calls == [("How do tides work?", "C")]  # whitespace stripped before the model sees it


def test_compare_runs_all_three_arms_on_the_same_question(client_and_service):
    client, service = client_and_service
    body = client.post("/compare", json={"question": "Why is the sky blue?"}).json()
    assert set(body) == {"A", "B", "C"} and [arm for _, arm in service.calls] == ["A", "B", "C"]


@pytest.mark.parametrize(
    "payload", [{"question": ""}, {"question": "   "}, {"question": "x" * 2001}, {"question": "ok", "arm": "D"}, {}]
)
def test_bad_input_is_rejected_before_the_model(client_and_service, payload):
    client, service = client_and_service
    assert client.post("/answer", json=payload).status_code == 422
    assert service.calls == []


def test_results_404_when_missing(client_and_service, monkeypatch, tmp_path):
    import app.main as main

    monkeypatch.setattr(main, "RESULTS_FILE", tmp_path / "missing.json")
    client, _ = client_and_service
    assert client.get("/results").status_code == 404


def test_ui_is_mounted(client_and_service):
    client, _ = client_and_service
    assert client.get("/").status_code == 200
