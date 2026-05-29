"""nanoam.py — the entire Adaptive Minds framework in one file (≤300 lines).

Two control flows over one base model + a catalog of LoRA adapters:
  * Router — one LLM call picks an adapter; that adapter answers.
  * Agent  — ReAct loop: brain emits CALL/FINAL; runtime executes tools +
             LoRA experts; FINAL terminates.

The production runtime in ``adaptive_minds/`` adds FastAPI, sandboxed tools,
evals, and docker. Everything else is the same shape as below.

    VLLM_BASE=http://localhost:8000/v1 python nanoam.py route "What is caffeine?"
    VLLM_BASE=http://localhost:8000/v1 python nanoam.py agent "Compute 2**10."

Paper: https://arxiv.org/abs/2510.15416
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")
CATALOG_PATH = os.environ.get(
    "AM_CATALOG", str(Path(__file__).parent / "catalogs" / "qwen25_smoke.yaml"),
)


# ---- Catalog ----

@dataclass(frozen=True)
class Adapter:
    """One catalog entry. ``id`` is the lowercase name vLLM serves it under."""
    id: str
    description: str
    system_prompt: str
    keywords: list[str]


def load_catalog(path: str = CATALOG_PATH) -> tuple[dict[str, Adapter], dict]:
    """Parse the catalog YAML; return (id→Adapter dict, full config)."""
    cfg = yaml.safe_load(Path(path).read_text())
    catalog: dict[str, Adapter] = {}
    for row in cfg.get("lora_adapters", []):
        if not row.get("enabled", True):
            continue
        aid = row["name"].lower().replace(" ", "_")
        catalog[aid] = Adapter(
            id=aid,
            description=row.get("description", ""),
            system_prompt=row.get("system_prompt", ""),
            keywords=row.get("keywords", []),
        )
    return catalog, cfg


# ---- vLLM HTTP client ----

def chat(model: str, messages: list[dict],
         temperature: float = 0.3, max_tokens: int = 512,
         stop: list[str] | None = None) -> dict:
    """One POST to vLLM's OpenAI-compatible endpoint. Returns
    {ok, response, error}. ``model`` is the adapter name OR the base id."""
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if stop:
        body["stop"] = stop
    try:
        r = requests.post(f"{VLLM_BASE}/chat/completions",
                          json=body, timeout=180)
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": (
            f"vLLM unreachable at {VLLM_BASE}. "
            "Run 'docker compose up -d' or set VLLM_BASE.")}
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    j = r.json()
    return {"ok": True, "response": j["choices"][0]["message"]["content"]}


# ---- Built-in tools: two minimal handlers (production ships five) ----

def tool_calculator(arg: str) -> str:
    """Evaluate a Python arithmetic expression in a restricted namespace.
    Production uses sympy for symbolic work; same call surface."""
    try:
        return str(eval(arg.strip(), {"__builtins__": {}}, {}))
    except Exception as e:
        return f"calculator error: {e}"


def tool_python(arg: str) -> str:
    """Run a Python snippet that prints its result. Caps stdout at 4 KB."""
    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(arg, {"__builtins__": __builtins__})
        return (buf.getvalue() or "(no output — did you print()?)")[:4000]
    except Exception as e:
        return f"python error: {e}"


TOOLS: dict[str, tuple[str, callable]] = {
    "calculator": ("evaluate a Python arithmetic expression. e.g. `2**16 + 17`",
                   tool_calculator),
    "python": ("run a Python snippet and capture stdout (use print()).",
               tool_python),
}


# ---- Router: one LLM call picks an adapter, that adapter answers ----

ROUTER_PROMPT = """\
Analyze the user query below and select the single best domain expert from
the list. Respond with ONLY the expert's id (lowercase, no punctuation).

Query: "{query}"

Experts:
{domain_list}

Selected expert:"""


def keyword_pick(query: str, catalog: dict[str, Adapter]) -> str:
    """Cheap baseline used as a fallback when the LLM picks an unknown id."""
    q = query.lower()
    best, best_score = next(iter(catalog)), 0
    for aid, a in catalog.items():
        score = sum(1 for k in a.keywords if k.lower() in q)
        if score > best_score:
            best, best_score = aid, score
    return best


def run_router(query: str, catalog: dict[str, Adapter]) -> dict:
    """(1) base model picks an adapter id, (2) that adapter answers.
    Both steps hit the same vLLM endpoint; only ``model`` changes."""
    domain_list = "\n".join(f"- {a.id}: {a.description.strip()[:120]}"
                            for a in catalog.values())
    prompt = ROUTER_PROMPT.format(query=query, domain_list=domain_list)
    r1 = chat("base", [{"role": "user", "content": prompt}],
              temperature=0.0, max_tokens=16, stop=["\n"])
    if not r1["ok"]:
        return r1
    chosen = (r1["response"] or "").strip().strip("`* ").split()[0].lower()
    if chosen not in catalog:
        chosen = keyword_pick(query, catalog)
    a = catalog[chosen]
    msgs = ([{"role": "system", "content": a.system_prompt}]
            if a.system_prompt else []) + [{"role": "user", "content": query}]
    r2 = chat(chosen, msgs, temperature=0.3, max_tokens=512)
    return {"ok": r2["ok"], "adapter_id": chosen,
            "response": r2.get("response"), "error": r2.get("error")}


# ---- Agent: ReAct loop over tools + LoRA experts ----

AGENT_PROMPT = """\
You are the Adaptive Minds agent. You answer the user by orchestrating
tools and domain experts.

Output format:
  CALL: <name> | <argument>      execute a tool or consult an expert
  FINAL: <answer>                emit the final answer

Tools (executed deterministically):
{tool_menu}

Experts (LoRA fine-tunes — they GENERATE domain text):
{expert_menu}

Rules: (1) plain-text arguments; (2) the runtime writes OBSERVATION lines,
not you; (3) batch independent CALLs in one turn when sensible.

User query: {query}
"""

_VERB = re.compile(r"^\s*(CALL|FINAL)\s*:\s*(.*)$", re.IGNORECASE)


def parse_actions(text: str) -> list[tuple[str, str, str]]:
    """Return a list of (kind, name, arg). Walks line-by-line so multi-line
    arguments survive until the next CALL/FINAL/OBSERVATION line."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        m = _VERB.match(lines[i])
        if not m:
            i += 1
            continue
        kind, payload = m.group(1).upper(), m.group(2)
        j = i + 1
        while j < len(lines):
            s = lines[j].lstrip().upper()
            if s.startswith(("CALL:", "FINAL:", "OBSERVATION:")):
                break
            j += 1
        if j > i + 1:
            payload = payload + "\n" + "\n".join(lines[i + 1:j])
        if kind == "FINAL":
            out.append(("FINAL", "", payload.strip()))
        else:
            head, _, arg = payload.partition("|")
            out.append(("CALL", head.strip().lower(), arg.strip()))
        i = j
    return out


def dispatch(name: str, arg: str, catalog: dict[str, Adapter]) -> str:
    """Run one CALL — either a tool or a LoRA expert."""
    if name in TOOLS:
        return TOOLS[name][1](arg)
    if name in catalog:
        a = catalog[name]
        msgs = []
        if a.system_prompt:
            msgs.append({"role": "system", "content": a.system_prompt})
        msgs.append({"role": "user", "content": arg})
        r = chat(name, msgs, temperature=0.3, max_tokens=800)
        return r.get("response") or r.get("error") or "(empty)"
    return f"unknown name '{name}'. Tools: {list(TOOLS)}. Experts: {list(catalog)}."


def run_agent(query: str, catalog: dict[str, Adapter],
              max_steps: int = 4) -> dict:
    """Run the ReAct loop until FINAL or max_steps exhausted."""
    tool_menu = "\n".join(f"  - {n:<10s} — {d}" for n, (d, _) in TOOLS.items())
    expert_menu = "\n".join(f"  - {a.id:<24s} — {a.description[:120]}"
                            for a in sorted(catalog.values(), key=lambda x: x.id))
    sys_prompt = AGENT_PROMPT.format(tool_menu=tool_menu, expert_menu=expert_menu, query=query)
    trace, final = "", None
    for step in range(max_steps):
        msg = (f"User query: {query}\n\n"
               + (f"Trace so far:\n{trace}\n\n" if trace else "")
               + "Plan your next move. Emit CALL: lines or a FINAL: line.")
        r = chat("base", [{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": msg}],
                 temperature=0.2, max_tokens=1024,
                 stop=["\nOBSERVATION:", "\nUser query:"])
        if not r["ok"]:
            return {"ok": False, "error": r["error"]}
        trace += (r["response"] or "").rstrip() + "\n"
        actions = parse_actions(r["response"] or "")
        if not actions:
            trace += "OBSERVATION: (no CALL/FINAL emitted)\n"
            continue
        obs_block = []
        for kind, name, arg in actions:
            if kind == "FINAL":
                final = arg
                break
            obs_block.append(f"OBSERVATION ({name}): {dispatch(name, arg, catalog)[:600]}")
        if obs_block:
            trace += "\n".join(obs_block) + "\n"
        if final is not None:
            break
    return {"ok": True, "response": final or "(no FINAL emitted)",
            "steps": step + 1, "trace": trace}


# ---- Main ----

def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("route", "agent"):
        print('usage: python nanoam.py {route|agent} "<query>"', file=sys.stderr)
        return 2
    mode, query = argv[1], " ".join(argv[2:])
    catalog, _ = load_catalog()
    if not catalog:
        print(f"no adapters in {CATALOG_PATH}", file=sys.stderr)
        return 1
    t0 = time.time()
    r = run_router(query, catalog) if mode == "route" else run_agent(query, catalog)
    if not r.get("ok"):
        print(f"ERROR: {r.get('error')}", file=sys.stderr)
        return 1
    print(json.dumps({"mode": mode, "query": query,
                      "adapter_id": r.get("adapter_id"),
                      "response": r.get("response"),
                      "elapsed_s": round(time.time() - t0, 2),
                      "steps": r.get("steps")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
