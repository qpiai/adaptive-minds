"""LangGraph-driven agent loop — same call surface as ``agent.run_agent``.

This is the explicit-state-graph variant of the ReAct loop in
``agent.py``. The imperative version is one tight ``for`` loop; the
LangGraph version expresses the same control flow as nodes + conditional
edges. They produce equivalent results but the graph form makes the
state transitions inspectable (the UI renders each node visit).

Nodes:
    plan       — base model emits CALL/FINAL lines.
    dispatch   — runtime executes every CALL (tool or LoRA expert).
    synthesise — when step budget is exhausted without a FINAL, the
                 brain is asked once more to synthesise observations.

Edges:
    START → plan
    plan → dispatch     (if any CALL emitted)
    plan → END          (if FINAL emitted)
    dispatch → plan     (if budget remaining)
    dispatch → synthesise (budget exhausted)
    synthesise → END

The module imports ``langgraph`` lazily so the rest of the package works
without it installed; only the ``/langgraph`` server endpoint pays.
"""
from __future__ import annotations

from typing import Any, TypedDict

from .agent import (
    DEFAULT_AGENT_PROMPT,
    _build_menus,
    _dispatch_call,
    _parse_actions,
)
from .common import Adapter, vllm_chat


class _State(TypedDict, total=False):
    query: str
    sys_prompt: str
    brain_model: str
    brain_max_tokens: int
    sys_prompt_mode: str
    catalog: dict[str, Adapter]
    trace: str
    steps: list[dict[str, Any]]
    step_index: int
    max_steps: int
    last_response: str
    final: str | None


_STOPS = ["\nOBSERVATION:", "\nUser query:"]


def _node_plan(state: _State) -> _State:
    """Brain emits the next batch of CALL lines (or a FINAL)."""
    trace = state.get("trace", "")
    user_msg = (
        f"User query: {state['query']}\n\n"
        + (f"Trace so far:\n{trace}\n\n" if trace else "")
        + "Plan your next move. Emit one or more CALL: lines, or a single "
          "FINAL: line when ready."
    )
    r = vllm_chat(
        state["brain_model"],
        [{"role": "system", "content": state["sys_prompt"]},
         {"role": "user", "content": user_msg}],
        temperature=0.2, max_tokens=state["brain_max_tokens"], stop=_STOPS,
    )
    state["steps"].append({
        "label": f"plan[{state['step_index']}]",
        "kind": "brain", "name": state["brain_model"],
        "raw_response": r.get("response"),
        "elapsed": r.get("elapsed"),
    })
    state["last_response"] = r.get("response") or ""
    if not r["ok"]:
        state["final"] = r.get("error") or "(plan failed)"
    return state


def _node_dispatch(state: _State) -> _State:
    """Execute every CALL the brain emitted; append OBSERVATIONs to trace."""
    actions = _parse_actions(state["last_response"])
    state["trace"] += state["last_response"].rstrip() + "\n"
    obs_block = []
    for act in actions:
        if act.kind == "FINAL":
            state["final"] = act.arg or state["last_response"]
            break
        obs, meta = _dispatch_call(act.name, act.arg, state["catalog"],
                                   state["sys_prompt_mode"])
        state["steps"].append(meta)
        obs_block.append(f"OBSERVATION ({act.name}): {obs[:800]}")
    if obs_block:
        state["trace"] += "\n".join(obs_block) + "\n"
    state["step_index"] += 1
    return state


def _node_synthesise(state: _State) -> _State:
    """Step budget hit without a FINAL — make the brain commit."""
    r = vllm_chat(
        state["brain_model"],
        [{"role": "system", "content": state["sys_prompt"]},
         {"role": "user", "content":
          f"User query: {state['query']}\n\nTrace so far:\n{state['trace']}\n\n"
          "Step budget exhausted. Emit a single `FINAL: …` line that "
          "synthesises every observation above into the answer."}],
        temperature=0.2, max_tokens=state["brain_max_tokens"],
    )
    state["steps"].append({
        "label": "synthesise FINAL", "kind": "brain",
        "name": state["brain_model"],
        "raw_response": r.get("response"), "elapsed": r.get("elapsed"),
    })
    if r["ok"]:
        acts = _parse_actions(r.get("response") or "")
        final = next((a.arg for a in acts if a.kind == "FINAL"), None)
        state["final"] = final or (r.get("response") or "(no FINAL)")
    else:
        state["final"] = r.get("error") or "(synthesise failed)"
    return state


def _route_after_plan(state: _State) -> str:
    if state.get("final") is not None:
        return "end"
    return "dispatch"


def _route_after_dispatch(state: _State) -> str:
    if state.get("final") is not None:
        return "end"
    if state["step_index"] >= state["max_steps"]:
        return "synthesise"
    return "plan"


def _build_graph():
    """Construct the StateGraph. Imported lazily so the rest of the
    package doesn't require langgraph at import time."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(_State)
    g.add_node("plan", _node_plan)
    g.add_node("dispatch", _node_dispatch)
    g.add_node("synthesise", _node_synthesise)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _route_after_plan,
                             {"dispatch": "dispatch", "end": END})
    g.add_conditional_edges("dispatch", _route_after_dispatch,
                             {"plan": "plan", "synthesise": "synthesise",
                              "end": END})
    g.add_edge("synthesise", END)
    return g.compile()


_GRAPH = None


def run_langgraph_agent(query: str, catalog: dict[str, Adapter], cfg: dict,
                       temperature: float, max_tokens: int,
                       sys_prompt_mode: str = "trained") -> dict:
    """Same return shape as ``run_agent`` so the server treats them
    interchangeably. ``temperature`` is accepted for API symmetry but the
    brain runs at temp=0.2 internally to keep planning deterministic."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()

    tool_menu, expert_menu = _build_menus(catalog)
    brain_model = cfg.get("agent_brain", cfg.get("base_model_id", "base"))
    sys_prompt = cfg.get("agent_prompt", DEFAULT_AGENT_PROMPT).format(
        tool_menu=tool_menu, expert_menu=expert_menu, query=query,
    )
    state0: _State = {
        "query": query, "sys_prompt": sys_prompt,
        "brain_model": brain_model,
        "brain_max_tokens": int(cfg.get("agent_brain_max_tokens", max_tokens)),
        "sys_prompt_mode": sys_prompt_mode,
        "catalog": catalog,
        "trace": "", "steps": [],
        "step_index": 0,
        "max_steps": int(cfg.get("agent_max_steps", 6)),
        "last_response": "", "final": None,
    }
    final_state = _GRAPH.invoke(state0)
    return {
        "ok": True,
        "mode": f"langgraph (brain={brain_model})",
        "response": final_state.get("final") or "(no FINAL emitted)",
        "adapter_id": brain_model,
        "elapsed": sum(s.get("elapsed") or 0 for s in final_state["steps"]),
        "usage": {},
        "system_prompt_used": sys_prompt,
        "steps": final_state["steps"],
    }
