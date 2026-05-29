"""Auto-mode dispatcher: heuristic chooser between Router and Agent.

Paper §5.5 motivates an entropy-gated classifier; v0.1 ships a cheap
deterministic heuristic so the mode picker is observable without a
secondary LLM call. The decision is returned in the result so the UI can
show *which* mode the dispatcher actually picked.

Heuristic indicators that a query needs the multi-step Agent:
  - explicit math/code/shell operators (``**``, ``%`` formatting, ``|``,
    ``$()``, ``\\boxed``, ``=>``)
  - "compute X then Y" / "find X and then" — sequential phrasing
  - "and explain" / "then summarise" — composition phrasing
  - long queries (≥ 30 tokens) — usually compound asks
  - multiple sentences

Everything else routes through the single-step Router.
"""
from __future__ import annotations

import re

from .agent import run_agent
from .common import Adapter
from .router import run_router

_AGENT_INDICATORS = (
    "compute",
    "calculate",
    "then explain",
    "then summarise",
    "then summarize",
    "and explain",
    "and then",
    "step by step",
    "first ",
    "after that",
    "afterwards",
    " and finally",
)

_AGENT_OPERATORS = re.compile(r"(\*\*|\\boxed|=>|\$\(|\|\s*\w+|sum\(|integrate\()")


def needs_agent(query: str) -> tuple[bool, str]:
    """Return (use_agent, reason). ``reason`` is a one-word tag for the UI."""
    q = (query or "").strip()
    if not q:
        return False, "empty"

    if _AGENT_OPERATORS.search(q):
        return True, "operators"

    low = q.lower()
    for ind in _AGENT_INDICATORS:
        if ind in low:
            return True, "sequencing"

    if len(q.split()) >= 30:
        return True, "long"

    if q.count(".") >= 2 or q.count("?") >= 2:
        return True, "multi-sentence"

    return False, "single-domain"


def run_auto(query: str, catalog: dict[str, Adapter], cfg: dict,
             temperature: float, max_tokens: int,
             sys_prompt_mode: str = "trained") -> dict:
    """Decide router vs. agent from the query surface, then run it.

    The chosen mode + the one-word reason are stamped into the result so
    the UI can render which path the dispatcher took.
    """
    use_agent, reason = needs_agent(query)
    if use_agent:
        out = run_agent(query, catalog, cfg, temperature, max_tokens,
                        sys_prompt_mode=sys_prompt_mode)
    else:
        out = run_router(query, catalog, cfg, temperature, max_tokens,
                         sys_prompt_mode=sys_prompt_mode)
    out["mode"] = f"auto → {'agent' if use_agent else 'router'}"
    out["auto_decision"] = {"picked": "agent" if use_agent else "router",
                            "reason": reason}
    return out
