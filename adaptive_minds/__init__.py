"""Adaptive Minds — LoRA adapters as callable tools for one base model.

Paper: *Adaptive Minds: Empowering Agents with LoRA-as-Tools*
       Shekar & Krishnan, Oct 2025.  arXiv:2510.15416
       https://arxiv.org/abs/2510.15416

Two control flows share a single base model + adapter library:

    run_router(query, catalog, cfg, ...)   # one LLM call to pick + one to answer
    run_agent(query, catalog, cfg, ...)    # multi-step ReAct: tools + experts

Both expect a vLLM server with LoRA support at ``VLLM_BASE`` (default
``http://localhost:8000/v1``) serving the adapters by name.
"""
from __future__ import annotations

from .agent import run_agent
from .catalog import load_catalog, load_config, vllm_lora_args
from .common import Adapter, vllm_chat
from .router import keyword_pick, run_router

__version__ = "0.1.0"

__all__ = [
    "Adapter",
    "load_catalog",
    "load_config",
    "vllm_lora_args",
    "vllm_chat",
    "run_router",
    "run_agent",
    "keyword_pick",
]
