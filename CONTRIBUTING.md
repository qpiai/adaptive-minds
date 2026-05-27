# Contributing

Thanks for considering a contribution. Adaptive Minds aims to be a small,
hackable reference implementation — keep PRs focused and minimal.

## Dev setup

```bash
git clone https://github.com/qpiai/adaptive-minds
cd adaptive-minds
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve,ui,tools]"
```

Run the test suite:

```bash
pytest tests/ -v
```

Lint:

```bash
ruff check adaptive_minds tests
```

## How to add a new adapter

1. Train (or pick) a LoRA on `Qwen/Qwen2.5-7B-Instruct` (or whatever the
   catalog's `base_model.hf_id` points at).
2. Push the adapter to the HF Hub under
   `pavan01729/adaptive-minds-loras` (or a fork of that repo — set
   `hub.repo` in your catalog).
3. Add an entry to `catalogs/qwen25_30.yaml`:
   ```yaml
   - name: MyExpert
     hf_subdir: qwen2.5-7b/qwen25_my_expert_v1
     description: One sentence the router will see.
     system_prompt: You are an expert in …
     keywords: [topic, related, terms]
     enabled: true
   ```
4. Re-launch vLLM and the server (`docker compose restart vllm server`).
   The new adapter is reachable at `POST /chat` and visible in the UI.

## How to add a new external tool

External tools live in `adaptive_minds/external_tools.py`. To add one:

1. Implement a handler with the signature
   `def my_handler(sub_query: str) -> dict`  
   The dict must have `output` (str) and may have `debug` (dict) and
   `error` (str) keys.
2. Add it to `EXTERNAL_TOOLS` in the same file.
3. Add a one-line help string to `TOOL_DESCRIPTIONS` in
   `adaptive_minds/tools.py` — that's what the agent sees in its prompt.

Tools run in-process; if your tool needs subprocess isolation, copy the
`code_handler` / `shell_handler` patterns.

## How to add a benchmark to the evals

Drop a new script under `evals/`; reuse `load_catalog`, `run_router`,
`run_agent`, and `vllm_chat` from `adaptive_minds`. Add a row to the
README in that folder describing the paper claim, command, and expected
output shape.

## PR checklist

- [ ] `pytest tests/ -v` passes locally (the CI re-runs it across 3.10/3.11/3.12)
- [ ] `ruff check adaptive_minds tests` is clean
- [ ] Any new public function has a one-line docstring
- [ ] README / quickstart still works end-to-end if you changed the
      runtime, CLI, or docker setup
- [ ] No hardcoded paths or credentials in committed files

## What we won't merge

- Backwards-compat shims for the old `playground/` / `build/` split — v0.1
  is a clean break
- Vendored third-party dependencies (use the optional extras instead)
- Per-benchmark training scripts (we ship one shared SFT recipe; specific
  GRPO recipes are on the v0.2 roadmap)
- Code that adds heavy runtime deps (transformers, torch, peft) to the
  base install — those belong in `.[training]` or `.[vllm]` extras
