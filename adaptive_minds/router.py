"""Single-step router — the base model picks one adapter, that adapter answers.

Paper §5.2 ("From Routing to Agentic Reasoning"); the headline result in
Table 1 — 98.3 % adapter-selection accuracy on a 30-adapter library.

Two functions are exported:

* `keyword_pick`: cheap baseline used by evaluations (and as the Router's
  fallback when the LLM picks an out-of-catalog id).
* `run_router`:   the production router — one LLM call to classify, one
  LLM call to the chosen adapter.

The CLI exposes both modes via ``adaptive_minds.cli``.
"""
from __future__ import annotations

import time

from .common import Adapter, resolve_sysp, vllm_chat


def keyword_pick(query: str,
                 catalog: dict[str, Adapter]) -> tuple[str, list[tuple[str, int]]]:
    """Cheap baseline: count keyword hits per adapter, pick the highest.

    Falls back to ``general`` if no adapter matches and that id exists,
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


def run_router(query: str, catalog: dict[str, Adapter], cfg: dict,
               temperature: float, max_tokens: int,
               sys_prompt_mode: str = "trained") -> dict:
    """Two calls: (1) base model picks an adapter; (2) that adapter answers.

    Falls back to ``keyword_pick`` if the LLM emits an out-of-catalog id —
    happens rarely but the catalog is the source of truth, not the LLM.
    """
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
