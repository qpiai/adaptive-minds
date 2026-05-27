"""Single-step routers — Manual, Router (model-driven), Auto (keyword + fallback).

Paper reference: §5.2 "From Routing to Agentic Reasoning". The Router mode
is the headline result in Table 1 — 98.3 % accuracy on a 30-adapter library.
"""
from __future__ import annotations

import time

from .common import Adapter, resolve_sysp, vllm_chat


def keyword_pick(query: str,
                 catalog: dict[str, Adapter]) -> tuple[str, list[tuple[str, int]]]:
    """Cheap baseline: count keyword hits per adapter, pick the highest.

    Falls back to `general` if no adapter matches and that id exists,
    otherwise the first adapter in the catalog.
    """
    q = query.lower()
    scores: list[tuple[str, int]] = []
    for aid, a in catalog.items():
        score = sum(1 for k in a.keywords if k.lower() in q)
        scores.append((aid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[0] if scores else ("", 0)
    if top[1] > 0:
        return top[0], scores
    fallback = "general" if "general" in catalog else next(iter(catalog), "")
    return fallback, scores


def run_manual(adapter_id: str, query: str,
               catalog: dict[str, Adapter], sysp_override: str | None,
               sys_prompt_mode: str, temperature: float,
               max_tokens: int) -> dict:
    """User pins the adapter; no routing decision."""
    adapter = catalog[adapter_id]
    sysp = resolve_sysp(sys_prompt_mode, adapter, sysp_override)
    msgs = []
    if sysp:
        msgs.append({"role": "system", "content": sysp})
    msgs.append({"role": "user", "content": query})
    r = vllm_chat(adapter_id, msgs, temperature, max_tokens)
    r["mode"] = f"manual / sysp={sys_prompt_mode}"
    r["adapter_id"] = adapter_id
    r["system_prompt_used"] = sysp
    r["steps"] = [{
        "label": "request", "adapter": adapter_id,
        "sys_prompt_mode": sys_prompt_mode,
        "system_prompt_used": sysp,
        "elapsed": r.get("elapsed"),
    }]
    return r


def run_router(query: str, catalog: dict[str, Adapter], cfg: dict,
               temperature: float, max_tokens: int,
               sys_prompt_mode: str = "trained") -> dict:
    """Stage 1: base model classifies the query into one adapter.
    Stage 2: send the query to that adapter."""
    domain_list = "\n".join(
        f"- {a.id}: {a.description.strip()[:120]}" for a in catalog.values()
    )
    router_prompt = cfg["router_prompt"].format(
        domain_list=domain_list, query=query,
    )
    t0 = time.time()
    r1 = vllm_chat(
        model=cfg.get("base_model_id", "base"),
        messages=[{"role": "user", "content": router_prompt}],
        temperature=cfg.get("router_temperature", 0.0),
        max_tokens=cfg.get("router_max_tokens", 16),
        stop=["\n"],
    )
    route_text = (r1.get("response") or "").strip().lower().splitlines()
    route_text = route_text[0] if route_text else ""
    chosen = route_text.strip("`* ").split()[0] if route_text else ""
    if chosen not in catalog:
        chosen, _ = keyword_pick(query, catalog)
    adapter = catalog[chosen]

    sysp = resolve_sysp(sys_prompt_mode, adapter)
    msgs = []
    if sysp:
        msgs.append({"role": "system", "content": sysp})
    msgs.append({"role": "user", "content": query})
    r2 = vllm_chat(chosen, msgs, temperature, max_tokens)
    total = time.time() - t0

    return {
        "ok": r2["ok"] and r1["ok"],
        "mode": f"router / sysp={sys_prompt_mode}",
        "adapter_id": chosen,
        "response": r2.get("response"),
        "elapsed": total,
        "usage": r2.get("usage", {}),
        "request_body": r2.get("request_body"),
        "response_body": r2.get("response_body"),
        "error": r2.get("error") or r1.get("error"),
        "system_prompt_used": sysp,
        "steps": [
            {
                "label": "router (base)", "adapter": "base",
                "request_body": r1.get("request_body"),
                "response_body": r1.get("response_body"),
                "elapsed": r1.get("elapsed"),
                "raw_response": r1.get("response"),
                "decision": chosen,
            },
            {
                "label": "expert", "adapter": chosen,
                "request_body": r2.get("request_body"),
                "response_body": r2.get("response_body"),
                "elapsed": r2.get("elapsed"),
            },
        ],
    }


def run_auto(query: str, catalog: dict[str, Adapter], cfg: dict,
             temperature: float, max_tokens: int,
             sys_prompt_mode: str = "trained") -> dict:
    """Cheap keyword classifier first; falls back to Router if uncertain.

    The keyword path is the Table-1 baseline. Auto is what you'd ship in
    production if you wanted to save the routing-LLM call on easy queries.
    """
    chosen, scores = keyword_pick(query, catalog)
    top_two = scores[:2]
    uncertain = top_two[0][1] == 0 or (
        len(top_two) > 1 and top_two[0][1] == top_two[1][1] and top_two[0][1] > 0
    )
    if uncertain:
        r = run_router(query, catalog, cfg, temperature, max_tokens,
                       sys_prompt_mode=sys_prompt_mode)
        r["mode"] = f"auto→router / sysp={sys_prompt_mode}"
        r["steps"] = [{"label": "keyword (uncertain)",
                       "scores": top_two[:4]}] + r["steps"]
        return r

    adapter = catalog[chosen]
    sysp = resolve_sysp(sys_prompt_mode, adapter)
    msgs = []
    if sysp:
        msgs.append({"role": "system", "content": sysp})
    msgs.append({"role": "user", "content": query})
    r = vllm_chat(chosen, msgs, temperature, max_tokens)
    r["mode"] = f"auto (keyword) / sysp={sys_prompt_mode}"
    r["adapter_id"] = chosen
    r["system_prompt_used"] = sysp
    r["steps"] = [
        {"label": "keyword pick", "decision": chosen, "scores": top_two[:4]},
        {"label": "expert", "adapter": chosen,
         "request_body": r.get("request_body"),
         "response_body": r.get("response_body"),
         "elapsed": r.get("elapsed")},
    ]
    return r
