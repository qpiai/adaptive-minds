"""Streamlit chat UI for Adaptive Minds.

Talks to the FastAPI server at $AM_API_BASE (default
http://localhost:8765). Renders the agent trace with colored step blocks
and an expander that exposes the full step list returned by the server.

Run with:

    adaptive-minds ui
    # or
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os

import requests
import streamlit as st


API_BASE = os.environ.get("AM_API_BASE", "http://localhost:8765").rstrip("/")
DEFAULT_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Adaptive Minds",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/qpiai/adaptive-minds",
        "Report a bug": "https://github.com/qpiai/adaptive-minds/issues",
    },
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_get(path: str) -> dict | list:
    r = requests.get(f"{API_BASE}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def _api_post(path: str, body: dict) -> dict:
    r = requests.post(f"{API_BASE}{path}", json=body, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_health() -> dict:
    return _api_get("/health")


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_adapters() -> list[dict]:
    return _api_get("/adapters")


STEP_ICONS = {
    "router (base)":  "🧭",
    "expert":         "💡",
    "keyword pick":   "🔤",
    "commit FINAL":   "✅",
    "tool":           "🔧",
    "unknown":        "❓",
}


def _step_icon(label: str, kind: str | None = None) -> str:
    if kind == "tool":
        return "🔧"
    if kind == "expert":
        return "💡"
    if kind == "unknown":
        return "❓"
    if label.startswith("brain"):
        return "🧠"
    if label.startswith("CALL [tool]"):
        return "🔧"
    if label.startswith("CALL [expert]"):
        return "💡"
    if "router" in label.lower():
        return "🧭"
    if "FINAL" in label:
        return "✅"
    return "•"


def _render_steps(steps: list[dict]) -> None:
    """Walk the steps list returned by run_router / run_agent and render
    each one as a coloured block."""
    for i, s in enumerate(steps, 1):
        label = s.get("label", f"step {i}")
        kind = s.get("kind")
        icon = _step_icon(label, kind)
        title = f"{icon} **step {i}** — {label}"
        if s.get("adapter"):
            title += f"  `{s['adapter']}`"
        if s.get("elapsed") is not None:
            title += f"   ·   {s['elapsed']:.2f}s"
        st.markdown(title)
        if s.get("raw_response"):
            with st.expander("raw brain response", expanded=False):
                st.code(s["raw_response"], language="text")
        if s.get("argument"):
            st.caption(f"arg: `{s['argument'][:160]}`")
        if s.get("sub_query") and s["sub_query"] != s.get("argument"):
            st.caption(f"sub_query: `{s['sub_query'][:160]}`")
        if s.get("observation"):
            with st.expander("observation", expanded=False):
                st.code(s["observation"][:2000], language="text")
        if s.get("decision"):
            st.caption(f"decision → `{s['decision']}`")
        if s.get("scores"):
            st.caption(f"scores: {s['scores']}")
        st.markdown("---")


def _send(mode: str, query: str, *, temperature: float, max_tokens: int,
          sys_prompt_mode: str) -> dict:
    body = {
        "query": query, "mode": mode,
        "temperature": temperature, "max_tokens": max_tokens,
        "sys_prompt_mode": sys_prompt_mode,
    }
    try:
        return _api_post("/chat", body)
    except requests.RequestException as e:
        return {"ok": False, "error": str(e), "response": None,
                "adapter_id": None, "steps": [], "mode": mode}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 Adaptive Minds")
    st.caption("LoRA adapters as callable tools.")

    # Server status badge
    try:
        h = _fetch_health()
        st.success(
            f"Connected · {h['n_adapters']} adapters",
            icon="✅",
        )
        st.caption(f"vLLM: `{h['vllm_base']}`")
    except Exception as e:
        st.error(f"Cannot reach server at {API_BASE}", icon="🚫")
        st.caption(str(e)[:160])
        st.stop()

    st.markdown("### Mode")
    mode = st.radio(
        "Inference mode",
        ["router", "agent"],
        captions=[
            "Single-step: pick 1 adapter & answer.",
            "Multi-step: ReAct loop with tools + experts.",
        ],
        label_visibility="collapsed",
    )

    st.markdown("### Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)
    max_tokens = st.slider("Max tokens", 64, 4096, 1024, 64)
    sys_prompt_mode = st.selectbox(
        "System prompt",
        ["trained", "generic", "none"],
        help=(
            "trained = adapter's tuned prompt (default). "
            "generic = neutral assistant prompt. "
            "none = no system message at all."
        ),
    )

    st.markdown("### Adapters")
    try:
        adapters = _fetch_adapters()
    except Exception:
        adapters = []
    with st.expander(f"{len(adapters)} loaded", expanded=False):
        for a in adapters:
            st.markdown(f"• **{a['id']}** — {a['description'][:80]}")

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("🧠 Adaptive Minds")
mode_blurb = {
    "router": "Single-step routing — base model picks one adapter per query.",
    "agent": "Multi-step ReAct — base model orchestrates adapters + tools.",
}[mode]
st.caption(mode_blurb)

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["text"])
        if entry["role"] == "assistant":
            meta = []
            if entry.get("adapter_id"):
                meta.append(f"adapter `{entry['adapter_id']}`")
            if entry.get("mode"):
                meta.append(f"mode `{entry['mode']}`")
            if entry.get("elapsed") is not None:
                meta.append(f"{entry['elapsed']:.2f}s")
            if meta:
                st.caption("  ·  ".join(meta))
            if entry.get("steps"):
                with st.expander("trace", expanded=False):
                    _render_steps(entry["steps"])

query = st.chat_input(
    "Ask anything — the router will pick the right LoRA expert."
)

if query:
    st.session_state.history.append({"role": "user", "text": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner(f"thinking with `{mode}` mode…"):
            out = _send(mode, query, temperature=temperature,
                        max_tokens=max_tokens,
                        sys_prompt_mode=sys_prompt_mode)
        if not out.get("ok"):
            st.error(out.get("error") or "Unknown error from server.")
            text = ""
        else:
            text = out.get("response") or "(empty response)"
            st.markdown(text)
            meta = []
            if out.get("adapter_id"):
                meta.append(f"adapter `{out['adapter_id']}`")
            if out.get("mode"):
                meta.append(f"mode `{out['mode']}`")
            if out.get("elapsed") is not None:
                meta.append(f"{out['elapsed']:.2f}s")
            if meta:
                st.caption("  ·  ".join(meta))
            if out.get("steps"):
                with st.expander("trace", expanded=False):
                    _render_steps(out["steps"])

    st.session_state.history.append({
        "role": "assistant",
        "text": text,
        "adapter_id": out.get("adapter_id"),
        "mode": out.get("mode"),
        "elapsed": out.get("elapsed"),
        "steps": out.get("steps"),
        "raw": out,
    })
