"""Adaptive Minds: LoRA adapters as callable tools for agent orchestration.

Paper: "Adaptive Minds: Empowering Agents with LoRAs as Tools" (accepted
to ICML 2026).

Two operating modes share a single base model + adapter library:

    run_router(query, catalog, cfg, ...)   # single-step semantic routing
    run_agent(query, catalog, cfg, ...)    # multi-step ReAct loop

Both expect a vLLM server with LoRA support at VLLM_BASE (default
http://localhost:8000/v1) serving the adapters by name.
"""
from __future__ import annotations

from .common import Adapter, vllm_chat, vllm_chat_stream
from .catalog import load_catalog, load_config, vllm_lora_args
from .router import run_router, run_manual, run_auto, keyword_pick
from .agent import run_agent

__version__ = "0.1.0"

__all__ = [
    "Adapter",
    "load_catalog",
    "load_config",
    "vllm_lora_args",
    "vllm_chat",
    "vllm_chat_stream",
    "run_router",
    "run_manual",
    "run_auto",
    "run_agent",
    "keyword_pick",
]
