<div align="center">

# 🧠 Adaptive Minds

### LoRA adapters as callable tools for agent orchestration

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![tests](https://github.com/qpiai/adaptive-minds/actions/workflows/test.yml/badge.svg)](https://github.com/qpiai/adaptive-minds/actions/workflows/test.yml)
[![HuggingFace adapters](https://img.shields.io/badge/🤗-adaptive--minds--loras-yellow)](https://huggingface.co/pavan01729/adaptive-minds-loras)
[![Paper](https://img.shields.io/badge/paper-ICML_2026_(accepted)-b31b1b.svg)]()

**One base model. Many LoRA experts. The model decides which one(s) to use.**

[Quickstart](#-quickstart-docker) · [Try it](#-try-it) · [Reproduce paper results](#-reproducing-paper-results) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md)

</div>

---

> 📺 **Demo**

![Adaptive Minds UI](docs/screenshot.png)

<sub><i>(Screenshot captured from the bundled Streamlit UI talking to a 2-adapter docker-compose stack. Replace with your own after running `docker compose up -d`.)</i></sub>

---

## ✨ Why Adaptive Minds

- **🎯 Specialization, not parameter merging.** Instead of merging LoRAs into a single weight blend, each adapter stays a named, callable tool. The base model picks the right one for the query at inference time.
- **🤖 Two modes, one runtime.** Single-step **Router** for direct domain queries, multi-step **ReAct Agent** for tasks that need multiple experts or external tools (calculator, code, shell, web, LP solver).
- **📦 Built on standards.** vLLM serves the base model + adapters by name via its native `--lora-modules`; the runtime is pure Python over the OpenAI-compatible endpoint. No bespoke serving stack.

## 🚀 Features

| | |
|---|---|
| **Router mode** | Single base-model call picks the right adapter from the catalog. **98.3%** routing accuracy on a 30-adapter library (paper Table 1). |
| **Agent mode** | ReAct loop with `THOUGHT / CALL / OBSERVATION / FINAL` grammar. CALLs can be adapters *or* external tools. Multi-CALL planning per turn. |
| **5 external tools** | `calculator` (sympy), `code` (Python sandbox), `shell` (bash sandbox), `websearch` (DDG), `pulp` (LP solver). Plug in your own with one function. |
| **30 LoRA specialists** | SQL, Cypher, SPARQL, bash, Mermaid, PII, quantum, legal, chem + 21 domain experts. All on the HF Hub; one YAML adds your own. |
| **FastAPI server** | `/health`, `/adapters`, `/route`, `/agent`, `/chat`. Pydantic-validated, CORS-open, no torch/transformers in the server layer. |
| **Streamlit UI** | Chat-style, mode toggle, adapter list, trace expander. Talks to the FastAPI server. |
| **Docker compose** | `docker compose up -d` brings up vLLM + server + UI. |
| **Reproducible evals** | `evals/routing_table1.py` and `evals/mmlu_three_way.py` for paper Tables 1 and 3; shared SFT recipe in `training/` for Table 2. |
| **151-query gold set** | Hand-labeled router benchmark ships in `evals/gold/routing_gold.jsonl`. |
| **CI + tests** | 18 pytest cases, no GPU/network needed. Matrix on Python 3.10 / 3.11 / 3.12. |

## ⚡ Quickstart (Docker)

```bash
git clone https://github.com/qpiai/adaptive-minds
cd adaptive-minds
cp .env.example .env             # then fill in HF_TOKEN
docker compose up -d
```

Wait for vLLM to download the base model + 2 smoke adapters (~5–15 min first time, watch `docker compose logs -f vllm`), then open **http://localhost:8501**.

To run the full 30-adapter catalog instead of the 2-adapter smoke setup, edit `docker-compose.yml`'s `vllm` `--lora-modules` to match `catalogs/qwen25_30.yaml` and set `AM_CATALOG=/app/catalogs/qwen25_30.yaml` in `.env`. The helper

```bash
adaptive-minds serve --catalog catalogs/qwen25_30.yaml
```

prints the exact `vllm serve` line you need.

## 🐍 Quickstart (pip, against an existing vLLM)

```bash
pip install -e ".[serve,ui,tools]"
export VLLM_BASE=http://your-vllm-host:8000/v1

# 1. Start the FastAPI server
adaptive-minds server --catalog catalogs/qwen25_30.yaml &

# 2. (Optional) launch the Streamlit UI
adaptive-minds ui &

# 3. Or use the CLI directly:
adaptive-minds route --catalog catalogs/qwen25_30.yaml \
    --query "Write a SQL query for top 10 customers by revenue in 2023."

adaptive-minds agent --catalog catalogs/qwen25_30.yaml \
    --query "Compute 2**32+17, then explain the figure in finance terms."
```

## 🧪 Try it

**Via curl:**

```bash
curl -s :8765/health | jq .
curl -s :8765/adapters | jq 'map(.id)'

curl -s :8765/route \
  -H 'Content-Type: application/json' \
  -d '{"query": "Write SQL to find top 5 customers by revenue."}' \
  | jq '.adapter_id, .response'

curl -s :8765/agent \
  -H 'Content-Type: application/json' \
  -d '{"query": "Compute 2**16+17, then explain it as a finance metric."}' \
  | jq -r .response
```

**Via Python:**

```python
from adaptive_minds import load_catalog, run_router, run_agent
from adaptive_minds.catalog import router_cfg

cat = load_catalog("catalogs/qwen25_30.yaml")
cfg = router_cfg("catalogs/qwen25_30.yaml")

r = run_router("How do you optimize a SPARQL query?", cat, cfg, 0.3, 256)
print(r["adapter_id"], "→", r["response"])
```

## 🏗 Architecture

```
┌──────────────┐    ┌───────────────┐    ┌────────────────────────────┐
│  Streamlit   │ ─▶ │  FastAPI      │ ─▶ │  vLLM (OpenAI /v1)         │
│  ui/app.py   │    │  /chat /route │    │  base model                │
│              │    │  /agent       │    │  + LoRA adapters by name   │
└──────────────┘    └───────────────┘    └────────────────────────────┘
                            │
                            ├─ run_router  ── single-step semantic routing (paper §5.2)
                            └─ run_agent   ── ReAct loop with adapters+tools (paper §5.4)
```

- **Single source of truth**: one YAML catalog drives both the `vllm serve` launch command and the runtime's adapter selection.
- **No model weights in the server**: `adaptive_minds.server` is fastapi + pydantic + requests. All inference is HTTP to vLLM.
- **No LangGraph, no embedded UI**: the runtime is ~600 lines of Python.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a fuller walk-through.

## 📊 Reproducing paper results

| Paper       | Command                                                                              |
|-------------|--------------------------------------------------------------------------------------|
| **Table 1** | `python -m evals.routing_table1 --catalog catalogs/qwen25_30.yaml`                   |
| **Table 1** (keyword baseline) | `python -m evals.routing_keyword_baseline --catalog catalogs/qwen25_30.yaml` |
| **Table 2** | See [`training/README.md`](training/README.md) — shared SFT recipe + benchmark map   |
| **Table 3** | `python -m evals.mmlu_three_way --catalog catalogs/qwen25_30.yaml --n 10`            |

Full details in [`evals/README.md`](evals/README.md).

## 🧰 Bring your own adapter

1. Train a LoRA with the shared recipe (or use any existing PEFT LoRA on the catalog's base model):
   ```bash
   python training/train_sft.py \
       --dataset hf://your-org/your-dataset \
       --base-model Qwen/Qwen2.5-7B-Instruct \
       --lora-name qwen25_my_expert_v1
   ```
2. Push to the Hub (or any HF-compatible repo), then add an entry to `catalogs/qwen25_30.yaml`:
   ```yaml
   - name: MyExpert
     hf_subdir: qwen2.5-7b/qwen25_my_expert_v1
     description: One sentence the router will see.
     system_prompt: You are an expert in …
     keywords: [topic, related, terms]
     enabled: true
   ```
3. Restart vLLM + server. The new adapter is routable.

## 🗂 Project layout

```
adaptive-minds/
├── adaptive_minds/      # runtime: router, agent, tools, catalog, server, CLI
├── ui/                  # Streamlit chat UI
├── catalogs/            # YAML adapter catalogs (30-adapter + 2-adapter smoke)
├── evals/               # paper-table reproduction scripts + 151-query gold set
├── training/            # shared SFT recipe + per-benchmark mapping
├── examples/            # python + curl quickstart scripts
├── docker/              # Dockerfile.server, Dockerfile.ui
├── docs/                # ARCHITECTURE.md + screenshots
├── tests/               # pytest, no GPU/network needed
└── docker-compose.yml   # vllm + server + ui
```

## 🗺 Roadmap

These are intentionally out of v0.1 to keep the public surface minimal:

- **Entropy H(Q) mode classifier** (paper §5.5) — auto-selects router vs agent based on the base model's first-16-token entropy
- **SSE streaming** in the UI (server already structured for it)
- **PEFT-backed in-process runtime** for single-GPU deployments
- **Per-benchmark GRPO recipes** (legal / chemistry / quantum use GRPO stage-2 in the paper)
- **Appendix C MCQ harness** — four-adapter reasoning-trace eval on Qwen3.5-9B
- **Adapter-fusion experiments** (paper §3.3 contrast with LoRA Soups)

## 🤝 Contributing

PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and what we will / won't merge.

## 📄 License

Apache 2.0. See [`LICENSE`](LICENSE).
