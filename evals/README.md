# Evaluations

Three scripts in this directory reproduce the paper's main numbers. All
require a vLLM server serving the catalog adapters by name (start one
with `adaptive-minds serve --catalog ...`); see the top-level README for
the launch command.

## Table 1 — routing accuracy on a 30-adapter library

Paper claim: **98.3 %** with the model-driven router; **31.7 %** with the
keyword baseline (Qwen3.5-9B, 30 adapters, 60 queries).

```bash
# Model-driven router (Adaptive Minds Table 1 main result)
python -m evals.routing_table1 \
    --catalog catalogs/qwen25_30.yaml \
    --gold evals/gold/routing_gold.jsonl \
    --out evals/results/routing_router.json

# Keyword baseline (no vLLM call — pure Python)
python -m evals.routing_keyword_baseline \
    --catalog catalogs/qwen25_30.yaml \
    --out evals/results/routing_keyword.json
```

The shipped gold set (`evals/gold/routing_gold.jsonl`) contains 151
hand-labeled queries across the 9 paper-Table-2 specialists plus the
`reasoning` fallback. The original 30-adapter / 60-query Qwen3.5-9B
gold set used in the paper is internal; this public set is representative
but smaller.

## Table 2 — 9-specialist gains under one shared recipe

Paper claim: **+4.6 to +84.0 pp** strict-scorer over the base model across
nine benchmarks (Qwen2.5-7B-Instruct).

Per-benchmark training and eval scripts are documented in
[`training/README.md`](../training/README.md). Each specialist uses the
same shared SFT recipe at `training/train_sft.py`; the only differences
are the dataset and the validator. The README links to dataset sources and
to the published adapter weights.

## Table 3 — Vanilla / Router / Agent on MMLU

Paper claim (Llama 3.1 8B, 100 domain-matched questions): Vanilla 56.0,
Router 62.0, Agent 58.0.

```bash
python -m evals.mmlu_three_way \
    --catalog catalogs/qwen25_30.yaml \
    --n 10 \
    --out evals/results/mmlu_three_way.json
```

The MMLU↔adapter mapping defaults to a Qwen2.5-7B-friendly set inside
`mmlu_three_way.py`. Override it with `--mapping path/to/mapping.json`
when working with a different base model.

## Smoke test (cheap, ~10 queries)

```bash
python -m evals.routing_table1 --catalog catalogs/qwen25_30.yaml --limit 10
```

This burns ~30 seconds against a running vLLM server and confirms the
router can reach the endpoint and produce sane choices.

## Notes on reproducibility

- **Adapter weights**: all 30 catalog adapters resolve to
  `pavan01729/adaptive-minds-loras/qwen2.5-7b/<subdir>` on the Hub. vLLM
  downloads them on launch from the `--lora-modules` arguments emitted by
  `adaptive-minds serve`.
- **Decoding**: routing uses temperature 0 for determinism; specialists
  use temperature 0.3 (matching paper §5.4 defaults). Pass `--temperature`
  to override.
- **Hardware**: a single L40S (48 GB) is enough to serve the 30 adapters
  under vLLM with `--max-lora-rank 64`. Smaller GPUs work with fewer
  adapters enabled.
