# Architecture

## Operating modes

The framework exposes one base model + a library of LoRA adapters through
four operating modes. Routing and agentic reasoning are two regimes of the
*same* system, not separate codepaths.

**🎯 Router** — one base-model call selects an expert from adapter metadata;
that adapter answers. (§5.2)

<div align="center"><img src="am_routing_v4.png" width="680" alt="Router: query → router agent → expert adapter → output"></div>

**🤖 Agent** — a THOUGHT → CALL → OBSERVATION → FINAL loop that invokes LoRA
experts *and* external tools across steps, then synthesises. (§5.4)

<div align="center"><img src="agent_architecture_2.png" width="680" alt="Agent: Think → Select Tool → Observe → Iterate over a tool registry, then synthesise"></div>

**🪄 Auto** — a cheap classifier picks Router vs. Agent per query and stamps
the decision into the result. **🕸️ LangGraph** — the agent loop as a
`StateGraph`. Both are shown in the unified diagram below. (§5.5)

<div align="center"><img src="auto_mode_arch.png" width="760" alt="Unified modes: classifier dispatches to single-step routing or a multi-step agent over a shared tool registry"></div>

## Modules

The runtime is intentionally thin — every public function has a docstring
that says *why* it exists:

| File                                   | Paper section | Purpose |
|----------------------------------------|---------------|---------|
| `adaptive_minds/router.py`             | §5.2          | Single-step routing |
| `adaptive_minds/agent.py`              | §5.4          | Multi-step ReAct loop |
| `adaptive_minds/auto.py`               | §5.5          | Router-vs-agent dispatcher (heuristic) |
| `adaptive_minds/langgraph_agent.py`    | §5.4          | Agent loop as a `langgraph.StateGraph` |
| `adaptive_minds/common.py`             | §5.3          | vLLM HTTP client + `Adapter` dataclass |
| `adaptive_minds/catalog.py`            | §4 + §5.3     | YAML loader + `vllm serve` arg builder |
| `adaptive_minds/tools.py`              | §5.4          | External tool registry (5 tools) |
| `adaptive_minds/external_tools.py`     | §5.4          | Sandboxed tool handlers |
| `adaptive_minds/server.py`             | —             | FastAPI surface (`/route` `/agent` `/chat` …) |
| `adaptive_minds/cli.py`                | —             | `serve` / `server` / `route` / `agent` / `list` |

The serving topology is flat: the **Next.js UI** (port 7007) calls the
**FastAPI server**, which holds no model weights and forwards every
inference over HTTP to a single **vLLM** engine that serves the base model
plus all LoRA adapters by name. Browser → API goes through the Next.js
proxy (`/api/am/*`), so the same image works on localhost, a public IP, or
behind a reverse proxy.

## The catalog YAML drives everything

A single YAML file (`catalogs/qwen25_30.yaml` ships in the repo) defines:

- the base model (`base_model.hf_id`)
- the HF repo holding the adapters (`hub.repo`)
- the router prompt template and decoding (`router.*`)
- the agent loop budget and brain model (`agent.*`)
- one entry per LoRA adapter (`lora_adapters[*]`), each with a name,
  HF subpath, description, system prompt, and keyword list

The same YAML is consumed by two paths:

1. **`adaptive-minds serve`** turns each adapter entry into a
   `--lora-modules <id>=<hub.repo>/<hf_subdir>` argument and prints the
   full `vllm serve` command. That single vLLM server then handles every
   adapter by name through its OpenAI-compatible endpoint.
2. **The Python runtime** (`load_catalog`) turns the same entries into
   `Adapter` instances that `run_router()` and `run_agent()` index by
   lowercase id when calling the vLLM endpoint.

This means there is one source of truth and zero state to keep in sync —
adding an adapter is "edit YAML, restart vLLM, done."

## Router (single-step, §5.2)

```
query ──▶ base model classifier ──▶ adapter id ──▶ adapter call ──▶ response
                  │
                  └── (keyword fallback if id ∉ catalog)
```

`run_router(query, catalog, cfg, ...)`:

1. Build a `domain_list` of `id: description` lines from the catalog.
2. Render `cfg["router_prompt"]` (from YAML) with `{query}` + `{domain_list}`.
3. Call vLLM at `VLLM_BASE/chat/completions` with `model="base"`,
   `stop=["\n"]`, temperature 0, ≤16 tokens. Parse the first token as the
   adapter id.
4. If the id isn't in the catalog, fall back to `keyword_pick()` (counts
   keyword hits per adapter).
5. Make a second vLLM call with `model=<chosen id>` and the (optional)
   trained system prompt for that adapter. Return the response.

Two vLLM calls total. Both go to the same endpoint; the `model` field is
the LoRA selector.

## ReAct Agent (multi-step, §5.4)

The agent's brain emits text in this grammar:

```
THOUGHT: <free text>
CALL: <name> | <argument>
…
FINAL: <answer>
```

The runtime:

1. Calls the brain with the system prompt + the current trace, stopping
   only on `OBSERVATION:` and `User query:` (tokens the brain should
   never emit itself).
2. Parses the response into `CALL` / `FINAL` actions in source order.
   A response may contain multiple `CALL` lines — they execute in order
   and their observations batch into a single OBSERVATION block fed back
   next turn. A `FINAL` anywhere in the response ends the loop after the
   preceding `CALL`s run.
3. For each `CALL <name> | <arg>`:
   - if `name` is in `TOOL_DESCRIPTIONS`, invoke that tool handler;
   - if `name` is in the catalog, set the adapter and call vLLM with the
     adapter's trained system prompt + `arg` as the user message;
   - otherwise report "unknown name" back as the observation.
4. Append the brain response + observation block to the trace; loop until
   `FINAL` or `max_steps` exhausted.
5. If step budget runs out without a `FINAL`, do one synthesis pass
   asking the brain to commit a `FINAL` from the trace.

The same loop covers both directly-routed cases (one CALL + one FINAL =
routing in another form) and multi-tool reasoning.

## Auto (mode dispatch, §5.5)

`run_auto(query, catalog, cfg, ...)` calls `needs_agent(query)` and then
delegates to `run_router` or `run_agent`, stamping the chosen mode and a
one-word `reason` into the result so the UI can show *why*. The paper
motivates an entropy-gated classifier — H(Q) over the base model's first
16 tokens, `H < 0.8 → Router`, `H > 1.5 → Agent` — but v0.1 ships a cheap
heuristic instead (operators, sequencing words, length, multi-sentence →
Agent; otherwise Router), avoiding a second model call. The hook is in
place to swap in the entropy gate later without changing callers.

## LangGraph (StateGraph form, §5.4)

`langgraph_agent.py` expresses the *same* agent loop as a
`langgraph.StateGraph` with three nodes — **plan → dispatch → synthesise** —
reusing `agent.py`'s `_dispatch_call` and prompt:

```
START → plan
plan → dispatch        (CALL emitted)
plan → END             (FINAL emitted)
dispatch → plan        (step budget remaining)
dispatch → synthesise  (budget exhausted)
synthesise → END
```

Behaviour matches the imperative agent; the value is observability — each
run is a sequence of node visits you can trace or drop into an existing
LangGraph pipeline. `langgraph` ships in the `[serve]` extra; the
imperative `agent.py` works without it.

## Why the two stops are exactly those

Earlier revisions stopped on `\nCALL:` and `\nFINAL:` to "prevent multi-
action per turn". That silently dropped every first verb line because
generation halted at the prefix transition. The fix in v0.1 is the
opposite: only stop on tokens the brain should never emit (OBSERVATION:,
User query:). Let the brain emit verbs freely; the parser decides what
to do with them.

## External tools

Five tools ship in v0.1 (see `adaptive_minds/external_tools.py`):

| Name        | Use                                                       |
|-------------|-----------------------------------------------------------|
| `calculator`| sympy expressions, symbolic + numeric                     |
| `code`      | Python snippet in 12s subprocess sandbox                  |
| `shell`     | bash one-liner in 10s subprocess sandbox                  |
| `websearch` | DuckDuckGo top-5                                          |
| `pulp`      | LP/MIP solver — JSON spec or Python snippet               |

All run in-process or in subprocess (no external services beyond
DuckDuckGo). Adding a tool: implement a handler in `external_tools.py`,
add the entry to `EXTERNAL_TOOLS` + `TOOL_DESCRIPTIONS`.

## Runtime entry points

```python
from adaptive_minds import load_catalog, run_router, run_agent
from adaptive_minds.catalog import router_cfg

cat = load_catalog("catalogs/qwen25_30.yaml")
cfg = router_cfg("catalogs/qwen25_30.yaml")

r = run_router("Write SQL to find duplicate emails.", cat, cfg, 0.3, 256)
print(r["adapter_id"], "→", r["response"])

a = run_agent("Compute 2**32+17 then explain it as a finance metric.",
              cat, cfg, 0.3, 1024)
print(a["response"])
```

Or via the CLI:

```bash
adaptive-minds route --catalog catalogs/qwen25_30.yaml --query "..."
adaptive-minds agent --catalog catalogs/qwen25_30.yaml --query "..."
```
