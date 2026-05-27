#!/usr/bin/env python3
"""Reproduce Table 1 (routing accuracy) from the paper.

For each query in evals/gold/routing_gold.jsonl, the script asks the base
model to pick an adapter from the catalog and compares the choice against
the hand-labeled ground truth.

Usage:
    python -m evals.routing_table1 \\
        --catalog catalogs/qwen25_30.yaml \\
        --gold evals/gold/routing_gold.jsonl \\
        --out evals/results/routing_eval.json

Smoke test (first N queries):
    python -m evals.routing_table1 --catalog catalogs/qwen25_30.yaml --limit 10

Requires VLLM_BASE pointing at a vLLM server launched with the catalog's
LoRA adapters loaded (see `adaptive-minds serve --catalog ...`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from adaptive_minds import load_catalog, run_router
from adaptive_minds.catalog import router_cfg


def evaluate(catalog_path: Path, gold_path: Path, *, limit: int,
             temperature: float, max_tokens: int) -> dict:
    cat = load_catalog(catalog_path)
    if not cat:
        sys.exit("catalog is empty")
    cfg = router_cfg(catalog_path)

    rows = []
    with open(gold_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))

    print(f"[eval] {len(rows)} queries, {len(cat)} adapters in catalog")
    results = []
    correct = 0
    per_adapter = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        q = row["query"]
        expected = row["expected_adapter"]
        out = run_router(q, cat, cfg, temperature, max_tokens)
        chosen = out.get("adapter_id") or ""
        is_correct = (chosen == expected)
        if is_correct:
            correct += 1
        per_adapter[expected]["total"] += 1
        per_adapter[expected]["correct"] += int(is_correct)
        confusion[expected][chosen] += 1
        results.append({
            "id": row.get("id"),
            "query": q,
            "expected_adapter": expected,
            "chosen_adapter": chosen,
            "correct": is_correct,
            "elapsed": out.get("elapsed"),
            "error": out.get("error"),
        })
        marker = "✓" if is_correct else "✗"
        print(f"  [{i:3d}/{len(rows)}] {marker} expected={expected:<13} "
              f"got={chosen:<13} q={q[:60]}")

    accuracy = correct / len(rows) * 100 if rows else 0
    summary = {
        "catalog": str(catalog_path),
        "gold": str(gold_path),
        "n_queries": len(rows),
        "n_adapters_in_catalog": len(cat),
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "elapsed_seconds": round(time.time() - t0, 2),
        "per_adapter_accuracy": {
            k: {"accuracy_pct": round(v["correct"] / v["total"] * 100, 2),
                "correct": v["correct"], "total": v["total"]}
            for k, v in sorted(per_adapter.items())
        },
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "results": results,
    }
    return summary


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--gold", type=Path, default=here / "gold" / "routing_gold.jsonl")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all queries; otherwise first N")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    summary = evaluate(args.catalog, args.gold,
                       limit=args.limit,
                       temperature=args.temperature,
                       max_tokens=args.max_tokens)

    print()
    print(f"  Accuracy: {summary['accuracy_pct']:.2f}% "
          f"({summary['correct']}/{summary['n_queries']})")
    print(f"  Elapsed:  {summary['elapsed_seconds']:.1f}s")
    print()
    print("  Per-adapter:")
    for k, v in summary["per_adapter_accuracy"].items():
        print(f"    {k:<16s} {v['accuracy_pct']:>6.2f}%  "
              f"({v['correct']}/{v['total']})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\n  results → {args.out}")


if __name__ == "__main__":
    main()
