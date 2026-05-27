#!/usr/bin/env python3
"""Keyword-matching baseline for Table 1.

Reproduces the "Keyword" column — for each query, count keyword hits per
adapter (from the catalog YAML) and pick the highest-scoring entry.

Usage:
    python -m evals.routing_keyword_baseline \\
        --catalog catalogs/qwen25_30.yaml \\
        --gold evals/gold/routing_gold.jsonl

Unlike `routing_table1.py`, this script does NOT need a vLLM server — the
classifier is pure Python.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from adaptive_minds import load_catalog, keyword_pick


def evaluate(catalog_path: Path, gold_path: Path, *, limit: int) -> dict:
    cat = load_catalog(catalog_path)
    if not cat:
        sys.exit("catalog is empty")
    rows = []
    with open(gold_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))

    results = []
    correct = 0
    per_adapter = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        q = row["query"]
        expected = row["expected_adapter"]
        chosen, scores = keyword_pick(q, cat)
        is_correct = (chosen == expected)
        correct += int(is_correct)
        per_adapter[expected]["total"] += 1
        per_adapter[expected]["correct"] += int(is_correct)
        results.append({
            "id": row.get("id"), "query": q,
            "expected_adapter": expected, "chosen_adapter": chosen,
            "correct": is_correct,
            "top_scores": [(a, s) for a, s in scores[:5] if s > 0],
        })

    accuracy = correct / len(rows) * 100 if rows else 0
    return {
        "catalog": str(catalog_path),
        "gold": str(gold_path),
        "method": "keyword_pick",
        "n_queries": len(rows),
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "per_adapter_accuracy": {
            k: {"accuracy_pct": round(v["correct"] / v["total"] * 100, 2),
                "correct": v["correct"], "total": v["total"]}
            for k, v in sorted(per_adapter.items())
        },
        "results": results,
    }


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--gold", type=Path, default=here / "gold" / "routing_gold.jsonl")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    s = evaluate(args.catalog, args.gold, limit=args.limit)
    print(f"  Keyword baseline accuracy: {s['accuracy_pct']:.2f}% "
          f"({s['correct']}/{s['n_queries']})")
    print()
    print("  Per-adapter:")
    for k, v in s["per_adapter_accuracy"].items():
        print(f"    {k:<16s} {v['accuracy_pct']:>6.2f}%  "
              f"({v['correct']}/{v['total']})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(s, indent=2, default=str))
        print(f"\n  results → {args.out}")


if __name__ == "__main__":
    main()
