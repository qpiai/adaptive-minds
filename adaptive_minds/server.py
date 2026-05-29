"""Thin FastAPI server around `run_router` and `run_agent`.

Endpoints:

    GET  /health     → {ok, vllm_base, n_adapters, catalog}
    GET  /adapters   → [{id, name, description, keywords, hf_id}]
    POST /route      → router output dict
    POST /agent      → agent output dict
    POST /chat       → unified shape used by the Streamlit UI

The server holds NO model weights. It owns one `catalog` dict at startup
and proxies inference to the vLLM endpoint pointed to by `VLLM_BASE`.
Catalog source is `AM_CATALOG` (env var) or `--catalog` (CLI flag).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import run_agent
from .auto import run_auto  # noqa: F401 — referenced by _resolve_runner via getattr
from .catalog import load_catalog, router_cfg
from .common import VLLM_BASE
from .router import run_router
from .tools import list_tools


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class _State:
    catalog_path: Path | None = None
    catalog: dict = {}
    cfg: dict = {}


_state = _State()


def configure(catalog_path: str | Path) -> None:
    """Re-load the catalog. Safe to call before/after `app` is created."""
    p = Path(catalog_path)
    _state.catalog_path = p
    _state.catalog = load_catalog(p)
    _state.cfg = router_cfg(p)


# Lazy auto-load if AM_CATALOG is set (so `uvicorn adaptive_minds.server:app`
# works without going through the CLI helper).
_env_catalog = os.environ.get("AM_CATALOG")
if _env_catalog:
    configure(_env_catalog)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    ok: bool
    vllm_base: str
    n_adapters: int
    catalog: str | None
    tools: list[str]


class AdapterInfo(BaseModel):
    id: str
    name: str
    description: str
    keywords: list[str]
    hf_id: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    # The four modes the UI exposes. ``auto`` and ``langgraph`` dispatch to
    # the agent or a LangGraph-driven equivalent of it; see
    # adaptive_minds.auto and adaptive_minds.langgraph_agent.
    mode: Literal["router", "agent", "auto", "langgraph"] = "router"
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=8192)
    sys_prompt_mode: Literal["trained", "generic", "none"] = "trained"


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=8192)
    sys_prompt_mode: Literal["trained", "generic", "none"] = "trained"


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    sys_prompt_mode: Literal["trained", "generic", "none"] = "trained"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adaptive Minds",
    description="LoRA adapters as callable tools — router + ReAct agent over vLLM.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_catalog() -> None:
    if not _state.catalog:
        raise HTTPException(
            status_code=503,
            detail=(
                "Catalog not loaded. Start the server with "
                "`adaptive-minds server --catalog catalogs/qwen25_30.yaml` "
                "or set AM_CATALOG before importing."
            ),
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        vllm_base=VLLM_BASE,
        n_adapters=len(_state.catalog),
        catalog=str(_state.catalog_path) if _state.catalog_path else None,
        tools=list_tools(),
    )


@app.get("/adapters", response_model=list[AdapterInfo])
def adapters() -> list[AdapterInfo]:
    _require_catalog()
    return [
        AdapterInfo(
            id=a.id, name=a.name, description=a.description,
            keywords=list(a.keywords), hf_id=a.hf_id,
        )
        for a in _state.catalog.values()
    ]


@app.post("/route")
def route(req: RouteRequest) -> dict[str, Any]:
    _require_catalog()
    return run_router(
        req.query, _state.catalog, _state.cfg,
        temperature=req.temperature, max_tokens=req.max_tokens,
        sys_prompt_mode=req.sys_prompt_mode,
    )


@app.post("/agent")
def agent(req: AgentRequest) -> dict[str, Any]:
    _require_catalog()
    return run_agent(
        req.query, _state.catalog, _state.cfg,
        temperature=req.temperature, max_tokens=req.max_tokens,
        sys_prompt_mode=req.sys_prompt_mode,
    )


def _resolve_runner(mode: str):
    # Look up by module attribute every call so unit tests can monkeypatch
    # ``adaptive_minds.server.run_router`` / ``run_agent`` etc. and have
    # the patches survive. (A pre-built dict would snapshot the originals.)
    import sys
    m = sys.modules[__name__]
    if mode == "router":
        return m.run_router
    if mode == "agent":
        return m.run_agent
    if mode == "auto":
        return m.run_auto
    if mode == "langgraph":
        try:
            from .langgraph_agent import run_langgraph_agent
        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail=("langgraph mode requires the [serve] extra. "
                        f"pip install '.[serve]' — {e}"),
            ) from e
        return run_langgraph_agent
    raise HTTPException(400, f"unknown mode: {mode}")


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    _require_catalog()
    fn = _resolve_runner(req.mode)
    out = fn(
        req.query, _state.catalog, _state.cfg,
        temperature=req.temperature, max_tokens=req.max_tokens,
        sys_prompt_mode=req.sys_prompt_mode,
    )
    out["request_mode"] = req.mode
    return out
