"""Multi-step ReAct agent — adapters and external tools share one call surface.

Paper reference: §5.4 "Agent Loop". The base model emits CALL/FINAL lines;
the runtime executes every CALL (tool or LoRA adapter) and feeds the
batched OBSERVATION block back next turn. A FINAL terminates the loop.

Design choices worth knowing about:

  (1) Stop tokens that match the desired output kill the output. Earlier
      revisions used stop=["\\nCALL:", "\\nFINAL:"] which silently dropped
      the brain's first verb line. We stop only on hallucinated boundaries
      (OBSERVATION:, "User query:").

  (2) The brain may emit multiple CALLs per turn for independent sub-tasks.
      The runtime executes them in declaration order; all observations are
      batched and returned in a single OBSERVATION block.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .common import Adapter, resolve_sysp, vllm_chat
from .tools import TOOL_DESCRIPTIONS, run_tool


DEFAULT_AGENT_PROMPT = """\
You are the Adaptive Minds agent. You answer the user by orchestrating
tools and domain experts.

== output format ==
You write THOUGHT lines and ACTION lines. Two kinds of actions:

  CALL: <name> | <argument>      run a tool or consult an expert
  FINAL: <answer>                emit the final answer for the user

You MAY emit multiple `CALL:` lines in a single response when the sub-tasks
are independent — the runtime executes them in order and returns all
observations back in one block. End with `FINAL:` when ready.

Example (multi-call planning):

  THOUGHT: independent sub-tasks; I'll fan them out.
  CALL: mermaid  | sequenceDiagram\\n  A->>B: ping\\n  B->>A: pong
  CALL: chemistry | Mechanism of action for tirzepatide.
  CALL: shell    | wc -l x.v | awk '$1 > 80 {print "too long"}'

== rules ==
  1. Arguments are plain text by default. Only `pulp` requires JSON.
  2. Do NOT write `OBSERVATION:` — the runtime appends it.
  3. Prefer fewer turns: batch independent CALLs together.
  4. Don't ask the same expert/tool the same question twice.

== capabilities ==

[tools] — deterministic; runtime EXECUTES the argument.
{tool_menu}

[experts] — LoRA fine-tunes on the base model; they GENERATE domain text.
{expert_menu}

User query: {query}
"""


_VERB_RE = re.compile(r"^\s*(CALL|FINAL)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class _Action:
    kind: str   # "CALL" or "FINAL"
    name: str   # tool / expert id (empty for FINAL)
    arg: str    # plain text or JSON


def _parse_actions(text: str) -> list[_Action]:
    """Walk the response line-by-line. Each CALL/FINAL line starts an
    action; its argument extends through subsequent lines until the
    next verb line (CALL/FINAL/OBSERVATION) or EOF."""
    lines = text.splitlines()
    actions: list[_Action] = []
    i = 0
    while i < len(lines):
        m = _VERB_RE.match(lines[i])
        if not m:
            i += 1
            continue
        kind = m.group(1).upper()
        payload = m.group(2)
        j = i + 1
        while j < len(lines):
            s = lines[j].lstrip().upper()
            if (s.startswith("CALL:") or s.startswith("FINAL:")
                    or s.startswith("OBSERVATION:")):
                break
            j += 1
        if j > i + 1:
            payload = payload + "\n" + "\n".join(lines[i + 1:j])
        payload = payload.rstrip()
        if kind == "FINAL":
            actions.append(_Action("FINAL", "", payload.strip()))
        else:
            head, _, arg = payload.partition("|")
            actions.append(_Action("CALL", head.strip().lower(), arg.strip()))
        i = j
    return actions


def _build_menus(catalog: dict[str, Adapter]) -> tuple[str, str]:
    tool_menu = "\n".join(
        f"  - {name:<10s} — {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
    )
    expert_menu = "\n".join(
        f"  - {a.id:<24s} — {(a.description or '').strip()[:120]}"
        for a in sorted(catalog.values(), key=lambda x: x.id)
    )
    return tool_menu, expert_menu


def _dispatch_call(name: str, arg: str,
                   catalog: dict[str, Adapter],
                   sys_prompt_mode: str) -> tuple[str, dict]:
    if name in TOOL_DESCRIPTIONS:
        obs = run_tool(name, arg)
        return obs, {"label": f"CALL [tool] {name}",
                     "kind": "tool", "name": name, "argument": arg,
                     "observation": obs}
    if name in catalog:
        adapter = catalog[name]
        sysp = resolve_sysp(sys_prompt_mode, adapter)
        sub_arg = arg
        if arg.startswith("{"):
            try:
                obj = json.loads(arg)
                for k in ("query", "question", "input", "prompt", "text",
                          "description", "spec"):
                    if isinstance(obj.get(k), str):
                        sub_arg = obj[k]
                        break
            except Exception:
                pass
        msgs = []
        if sysp:
            msgs.append({"role": "system", "content": sysp})
        msgs.append({"role": "user", "content": sub_arg})
        rr = vllm_chat(name, msgs, temperature=0.3, max_tokens=800)
        obs = rr.get("response") or rr.get("error") or "(empty)"
        return obs, {"label": f"CALL [expert] {name}", "kind": "expert",
                     "name": name, "argument": arg, "sub_query": sub_arg,
                     "observation": obs,
                     "request_body": rr.get("request_body"),
                     "response_body": rr.get("response_body"),
                     "elapsed": rr.get("elapsed")}
    valid_t = ", ".join(TOOL_DESCRIPTIONS)
    valid_e = ", ".join(sorted(catalog.keys()))
    obs = f"unknown name '{name}'. Tools: {valid_t}. Experts: {valid_e}."
    return obs, {"label": f"CALL ?{name}", "kind": "unknown",
                 "name": name, "argument": arg, "observation": obs}


def run_agent(query: str, catalog: dict[str, Adapter], cfg: dict,
              temperature: float, max_tokens: int,
              sys_prompt_mode: str = "trained") -> dict:
    """Run the ReAct loop until FINAL or max_steps exhausted."""
    tool_menu, expert_menu = _build_menus(catalog)
    max_steps = int(cfg.get("agent_max_steps", 6))
    brain_model = cfg.get("agent_brain", cfg.get("base_model_id", "base"))
    brain_max_tokens = int(cfg.get("agent_brain_max_tokens", 2048))
    sys_prompt = cfg.get("agent_prompt", DEFAULT_AGENT_PROMPT).format(
        tool_menu=tool_menu, expert_menu=expert_menu, query=query,
    )

    # Only stop on tokens the brain should never emit itself.
    STOPS = ["\nOBSERVATION:", "\nUser query:"]

    trace = ""
    steps: list[dict] = []

    def call_brain(extra_hint: str = ""):
        user_msg = (
            f"User query: {query}\n\n"
            + (f"Trace so far:\n{trace}\n\n" if trace else "")
            + (extra_hint + "\n\n" if extra_hint else "")
            + ("Plan your next move. Emit one or more CALL: lines, or a "
               "single FINAL: line when ready.")
        )
        return vllm_chat(
            brain_model,
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_msg}],
            temperature=0.2, max_tokens=brain_max_tokens, stop=STOPS,
        )

    final = None
    for step_i in range(max_steps):
        r = call_brain()
        steps.append({
            "label": f"brain[{step_i}]", "adapter": brain_model,
            "raw_response": r.get("response"),
            "request_body": r.get("request_body"),
            "response_body": r.get("response_body"),
            "elapsed": r.get("elapsed"),
        })
        if not r["ok"]:
            return {"ok": False, "mode": "agent", "error": r.get("error"),
                    "steps": steps, "system_prompt_used": sys_prompt}

        actions = _parse_actions(r.get("response") or "")
        if not actions:
            r2 = call_brain(
                "Your last response had no CALL: or FINAL: line. Emit one "
                "or more `CALL: <name> | <arg>` lines, ending with "
                "`FINAL: <answer>` only when ready."
            )
            steps.append({
                "label": f"brain[{step_i}]* (retry)", "adapter": brain_model,
                "raw_response": r2.get("response"),
                "elapsed": r2.get("elapsed"),
            })
            if not r2["ok"]:
                return {"ok": False, "mode": "agent",
                        "error": r2.get("error"),
                        "steps": steps, "system_prompt_used": sys_prompt}
            actions = _parse_actions(r2.get("response") or "")
            r = r2
            if not actions:
                steps.append({"label": "parse failure (after retry)",
                              "raw": (r.get("response") or "")[:400]})
                trace += "OBSERVATION: (last brain response had no action)\n"
                continue

        trace += (r.get("response") or "").rstrip() + "\n"

        obs_block: list[str] = []
        for act in actions:
            if act.kind == "FINAL":
                final = act.arg or r.get("response") or ""
                break
            obs, meta = _dispatch_call(act.name, act.arg, catalog,
                                       sys_prompt_mode)
            steps.append(meta)
            obs_block.append(f"OBSERVATION ({act.name}): {obs[:800]}")

        if obs_block:
            trace += "\n".join(obs_block) + "\n"

        if final is not None:
            break

    if final is None:
        commit = vllm_chat(
            brain_model,
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content":
              f"User query: {query}\n\nTrace so far:\n{trace}\n\n"
              "Step budget exhausted. Emit a single `FINAL: …` line that "
              "synthesises every observation above into the answer. Do "
              "not call any more tools or experts."}],
            temperature=0.2, max_tokens=brain_max_tokens,
        )
        if commit["ok"]:
            acts = _parse_actions(commit["response"] or "")
            final_act = next((a for a in acts if a.kind == "FINAL"), None)
            final = final_act.arg if final_act else (
                commit["response"] or "(no FINAL after commit pass)")
            steps.append({
                "label": "commit FINAL", "adapter": brain_model,
                "raw_response": commit["response"],
                "request_body": commit.get("request_body"),
                "response_body": commit.get("response_body"),
                "elapsed": commit.get("elapsed"),
            })

    return {
        "ok": True, "mode": f"agent (brain={brain_model})",
        "response": final or "(no FINAL emitted)",
        "adapter_id": brain_model,
        "elapsed": sum(s.get("elapsed") or 0 for s in steps),
        "usage": {},
        "request_body": steps[0].get("request_body") if steps else None,
        "response_body": steps[-1].get("response_body") if steps else None,
        "system_prompt_used": sys_prompt, "steps": steps,
    }
