"""FastAPI route shape tests — no vLLM, no live network.

Uses fastapi.TestClient for /health, /adapters, and pydantic validation
of /route, /agent, /chat request bodies. Actual inference calls are
mocked so the suite is hermetic.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from adaptive_minds import server as server_mod


HERE = Path(__file__).resolve().parents[1]
CATALOG = HERE / "catalogs" / "qwen25_30.yaml"


@pytest.fixture(scope="module", autouse=True)
def _load_catalog():
    server_mod.configure(CATALOG)


@pytest.fixture
def client():
    return TestClient(server_mod.app)


def test_health_reports_loaded_catalog(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["n_adapters"] == 30
    assert j["catalog"] and "qwen25_30.yaml" in j["catalog"]
    assert "calculator" in j["tools"]


def test_adapters_returns_full_catalog(client):
    r = client.get("/adapters")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 30
    ids = {a["id"] for a in body}
    assert {"sql", "chemistry", "pii", "finance"}.issubset(ids)


def test_route_validates_query_required(client):
    r = client.post("/route", json={})
    assert r.status_code == 422


def test_route_validates_query_max_length(client):
    r = client.post("/route", json={"query": "x" * 9000})
    assert r.status_code == 422


def test_route_invokes_run_router(client):
    fake = {"ok": True, "adapter_id": "sql", "response": "SELECT 1;",
            "elapsed": 0.1, "mode": "router", "steps": []}
    with patch("adaptive_minds.server.run_router", return_value=fake) as m:
        r = client.post("/route", json={"query": "SQL for top 5 customers"})
    assert r.status_code == 200
    assert r.json()["adapter_id"] == "sql"
    m.assert_called_once()


def test_agent_invokes_run_agent(client):
    fake = {"ok": True, "response": "the answer is 42",
            "elapsed": 1.0, "mode": "agent", "steps": [],
            "adapter_id": "base"}
    with patch("adaptive_minds.server.run_agent", return_value=fake) as m:
        r = client.post("/agent", json={"query": "What is 6*7?"})
    assert r.status_code == 200
    assert r.json()["response"] == "the answer is 42"
    m.assert_called_once()


def test_chat_dispatches_by_mode(client):
    router_fake = {"ok": True, "response": "r"}
    agent_fake = {"ok": True, "response": "a"}
    with patch("adaptive_minds.server.run_router", return_value=router_fake) as mr, \
         patch("adaptive_minds.server.run_agent", return_value=agent_fake) as ma:
        r1 = client.post("/chat", json={"query": "x", "mode": "router"})
        r2 = client.post("/chat", json={"query": "x", "mode": "agent"})
    assert r1.json()["response"] == "r" and r1.json()["request_mode"] == "router"
    assert r2.json()["response"] == "a" and r2.json()["request_mode"] == "agent"
    mr.assert_called_once()
    ma.assert_called_once()


def test_unloaded_catalog_returns_503(monkeypatch):
    # Force a fresh state with no catalog loaded.
    monkeypatch.setattr(server_mod, "_state", server_mod._State())
    c = TestClient(server_mod.app)
    r = c.get("/adapters")
    assert r.status_code == 503
    assert "Catalog not loaded" in r.json()["detail"]
