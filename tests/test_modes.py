"""Tests for the auto-mode dispatcher + the /chat mode literal.

LangGraph is tested only at the import + lazy-resolve layer (no live
graph compilation) so the suite stays hermetic and fast.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import adaptive_minds.server as server_mod
from adaptive_minds.auto import needs_agent


# ---------- needs_agent heuristic ------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("What is the capital of France?", False),
    ("Write SQL for top 5 customers.", False),
    ("Compute 2**16+17, then explain it as a finance metric.", True),
    ("Find the SMILES for caffeine and then describe its mechanism.", True),
    ("Step by step: derive the determinant of a 3x3 matrix.", True),
])
def test_needs_agent_heuristic(q: str, expected: bool) -> None:
    use_agent, _reason = needs_agent(q)
    assert use_agent is expected, f"{q!r} → {use_agent}, expected {expected}"


def test_needs_agent_reason_tags() -> None:
    """The reason tag is a short, stable word the UI can render."""
    _u, r = needs_agent("compute 2 + 3 then summarise")
    assert r in {"operators", "sequencing", "long", "multi-sentence",
                 "single-domain", "empty"}


def test_needs_agent_empty() -> None:
    use_agent, reason = needs_agent("")
    assert use_agent is False
    assert reason == "empty"


# ---------- /chat mode dispatch --------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """Mounted TestClient with the smoke catalog pre-loaded."""
    from pathlib import Path
    smoke = Path(__file__).resolve().parents[1] / "catalogs" / "qwen25_smoke.yaml"
    server_mod.configure(smoke)
    monkeypatch.setattr(
        server_mod, "VLLM_BASE", "http://test.invalid:8000/v1",
    )
    return TestClient(server_mod.app)


def test_chat_accepts_all_four_modes(client) -> None:
    """ChatRequest.mode literal must accept router / agent / auto / langgraph."""
    fake = {"ok": True, "response": "ok", "mode": "test"}
    with patch("adaptive_minds.server.run_router", return_value=fake), \
         patch("adaptive_minds.server.run_agent", return_value=fake), \
         patch("adaptive_minds.server.run_auto", return_value=fake), \
         patch("adaptive_minds.langgraph_agent.run_langgraph_agent",
               return_value=fake, create=True):
        for mode in ("router", "agent", "auto", "langgraph"):
            r = client.post("/chat", json={"query": "x", "mode": mode})
            assert r.status_code == 200, f"{mode}: {r.status_code} {r.text}"
            assert r.json()["request_mode"] == mode


def test_chat_rejects_unknown_mode(client) -> None:
    """Pydantic rejects any mode outside the four-literal set with 422."""
    r = client.post("/chat", json={"query": "x", "mode": "wat"})
    assert r.status_code == 422
