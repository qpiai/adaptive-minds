"""Catalog loader — turn a YAML adapter definition into a usable runtime config.

Two top-level operations:

    load_config(path)   -> dict       # full config (base_model, router, agent, ...)
    load_catalog(path)  -> {id: Adapter}  # just the enabled adapter entries

The companion `vllm_lora_args()` builds the `--lora-modules` arguments you
need when launching `vllm serve` so every catalog entry is reachable by its
lowercase `id` over the OpenAI-compatible /v1/chat/completions endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .common import Adapter


def _adapter_id(name: str) -> str:
    """Normalise display names to vLLM model ids (lowercase, no spaces)."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _entry_hf_id(entry: dict, hub_repo: str) -> str:
    """Resolve an entry's HF Hub identifier.

    Precedence: explicit `hf_id` > `repo/subdir` from top-level + `hf_subdir`.
    """
    if entry.get("hf_id"):
        return entry["hf_id"]
    sub = entry.get("hf_subdir")
    if hub_repo and sub:
        return f"{hub_repo}/{sub}"
    if hub_repo:
        return hub_repo
    return entry.get("hf_subdir", "")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the raw YAML config (base_model, router, agent, lora_adapters, ...)."""
    return yaml.safe_load(Path(path).read_text())


def load_catalog(path: str | Path) -> dict[str, Adapter]:
    """Read the YAML and return {id: Adapter} for every enabled entry."""
    cfg = load_config(path)
    hub_repo = (cfg.get("hub") or {}).get("repo", "")
    out: dict[str, Adapter] = {}
    for entry in cfg.get("lora_adapters", []):
        if not entry.get("enabled", True):
            continue
        aid = _adapter_id(entry["name"])
        out[aid] = Adapter(
            id=aid,
            name=entry["name"],
            description=entry.get("description", ""),
            system_prompt=entry.get("system_prompt", ""),
            keywords=[str(k) for k in entry.get("keywords", [])],
            hf_id=_entry_hf_id(entry, hub_repo),
        )
    return out


def vllm_lora_args(path: str | Path) -> list[str]:
    """Build the `vllm serve ... --lora-modules name=hf_id ...` arguments.

    The result is a flat argv list ready to splice into a subprocess call.
    Returns an empty list if no adapters are enabled.
    """
    cat = load_catalog(path)
    if not cat:
        return []
    modules = [f"{a.id}={a.hf_id}" for a in cat.values() if a.hf_id]
    if not modules:
        return []
    return ["--enable-lora", "--lora-modules", *modules]


def router_cfg(path: str | Path) -> dict[str, Any]:
    """Extract the router-shaped config that run_router() expects."""
    cfg = load_config(path)
    r = cfg.get("router", {})
    agent = cfg.get("agent") or {}
    base = cfg.get("base_model") or {}
    base_model_id = base.get("hf_id") or "base"
    return {
        "router_prompt": r.get("prompt_template", _DEFAULT_ROUTER_PROMPT),
        "router_temperature": r.get("temperature", 0.0),
        "router_max_tokens": r.get("max_tokens", 16),
        "agent_max_steps": agent.get("max_steps", 6),
        # `agent.brain` overrides; default is the base model id from the
        # catalog so router/agent talk to whatever vLLM serves the base
        # under (e.g. "Qwen/Qwen2.5-7B-Instruct").
        "agent_brain": agent.get("brain") if agent.get("brain") not in (None, "base") else base_model_id,
        "agent_brain_max_tokens": agent.get("max_tokens", 2048),
        "base_model_id": base_model_id,
    }


_DEFAULT_ROUTER_PROMPT = """\
Analyze this user query and select the most appropriate domain expert.

Query: "{query}"

Available Domain Experts:
{domain_list}

Instructions:
- Consider the main topic and intent of the query.
- Choose the domain expert that best matches.
- Respond with ONLY the expert's name (no punctuation).

Selected Expert:"""
