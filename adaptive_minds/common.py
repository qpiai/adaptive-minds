"""Shared types + vLLM HTTP client used by router, agent, and evals.

The client targets vLLM's OpenAI-compatible ``/v1/chat/completions`` endpoint.
The ``model`` parameter doubles as the LoRA adapter name when vLLM is launched
with ``--enable-lora --lora-modules name=<hf_id> ...`` so a single endpoint
serves the base model and every adapter in the catalog by name.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")


@dataclass(frozen=True)
class Adapter:
    """One entry from the catalog YAML, fully resolved.

    ``id`` is the lowercase adapter name passed to vLLM as the ``model``
    field. ``hf_id`` is the full HF Hub path (repo + optional subfolder).
    """
    id: str
    name: str
    description: str
    system_prompt: str
    keywords: list[str]
    hf_id: str = ""


# Three ways to scaffold an expert call. Exposed so callers can A/B test the
# LoRA with vs. without its trained scaffold; ``trained`` is the default and
# what every public benchmark in the paper uses.
SYS_PROMPT_MODES = {
    "trained":  "Use the adapter's trained system prompt (per-adapter; default).",
    "generic":  "Use a single neutral prompt for every adapter.",
    "none":     "Send NO system message. Probes the LoRA on its own.",
}

GENERIC_SYS_PROMPT = (
    "You are a helpful assistant. Answer concisely and accurately."
)


def resolve_sysp(mode: str, adapter: Adapter,
                 override: str | None = None) -> str | None:
    """Return the system-prompt string for an expert call, or ``None`` to
    skip the system message entirely."""
    if mode == "none":
        return None
    if mode == "generic":
        return GENERIC_SYS_PROMPT
    return override if override is not None else (adapter.system_prompt or None)


def vllm_chat(model: str, messages: list[dict],
              temperature: float, max_tokens: int,
              stop: list[str] | None = None) -> dict:
    """One POST to ``/v1/chat/completions``, returns a normalised result dict.

    Why a normalised return shape: the agent loop, the router, and the
    evaluation harness all need to log request/response pairs verbatim so
    failures are reproducible. Wrapping the raw response keeps the call
    sites uniform.
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        body["stop"] = stop
    t0 = time.time()
    try:
        r = requests.post(
            f"{VLLM_BASE}/chat/completions", json=body, timeout=180,
        )
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": (f"vLLM unreachable at {VLLM_BASE}. "
                      "Run 'docker compose up -d' or set VLLM_BASE."),
            "elapsed": time.time() - t0,
            "request_body": body, "response_body": None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.time() - t0,
                "request_body": body, "response_body": None}
    elapsed = time.time() - t0
    try:
        j = r.json()
    except Exception as e:
        return {"ok": False, "error": f"non-JSON response: {e}",
                "elapsed": elapsed, "request_body": body,
                "response_body": {"raw_text": r.text[:1000]}}
    if r.status_code != 200 or "error" in j:
        return {"ok": False, "error": j.get("error") or f"HTTP {r.status_code}",
                "elapsed": elapsed, "request_body": body, "response_body": j}
    return {
        "ok": True,
        "response": j["choices"][0]["message"]["content"],
        "finish_reason": j["choices"][0].get("finish_reason"),
        "elapsed": elapsed,
        "usage": j.get("usage", {}),
        "request_body": body,
        "response_body": j,
    }
