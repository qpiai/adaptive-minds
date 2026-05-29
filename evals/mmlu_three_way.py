#!/usr/bin/env python3
"""Reproduce Table 3 (Vanilla / Router / Agent) on MMLU.

Three inference strategies on MMLU subsets that match adapters in the catalog:

  1. VANILLA — base model only (no adapters)
  2. ROUTER  — single-step router picks one adapter per question
  3. AGENT   — multi-step ReAct loop (may call multiple adapters + tools)

The mapping from MMLU subjects to catalog adapter ids lives in --mapping,
defaulting to a Qwen-style mapping below. Adjust if your catalog ids differ.

Usage:
    python -m evals.mmlu_three_way \\
        --catalog catalogs/qwen25_30.yaml \\
        --n 100 \\
        --out evals/results/mmlu_three_way.json

Requires VLLM_BASE pointing at a vLLM server serving the catalog adapters
plus the base model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from adaptive_minds import load_catalog, run_agent, run_router, vllm_chat
from adaptive_minds.catalog import router_cfg


# MMLU subject → adapter id in the catalog.
DEFAULT_MAPPING = {
    "college_chemistry":          "chemistry",
    "high_school_chemistry":      "chemistry",
    "professional_accounting":    "finance",
    "high_school_macroeconomics": "economics",
    "college_computer_science":   "datafreds",        # placeholder
    "machine_learning":           "reasoning",
    "college_medicine":           "medicine",
    "anatomy":                    "medicine",
    "high_school_biology":        "biology",
    "high_school_physics":        "reasoning",
    "miscellaneous":              "reasoning",
}

ANSWER_LETTERS = ["A", "B", "C", "D"]


def load_questions(mapping: dict, n_per_subject: int) -> list[dict]:
    """Load MMLU questions for the subjects in `mapping`."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("evals/mmlu_three_way.py requires `pip install datasets`")

    out = []
    for subject, adapter_id in mapping.items():
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception:
            try:
                ds = load_dataset("cais/mmlu", subject, split="validation")
            except Exception as e:
                print(f"  [warn] skipping {subject}: {e}")
                continue
        n = min(n_per_subject, len(ds))
        for i in range(n):
            ex = ds[i]
            choices_text = "\n".join(
                f"{ANSWER_LETTERS[j]}. {c}" for j, c in enumerate(ex["choices"])
            )
            out.append({
                "subject": subject,
                "domain": adapter_id,
                "question": ex["question"],
                "choices_text": choices_text,
                "correct_answer": ANSWER_LETTERS[ex["answer"]],
            })
    return out


def extract_answer(text: str) -> str | None:
    """Extract a single A/B/C/D letter from model output."""
    t = (text or "").strip().upper()
    if not t:
        return None
    if t[0] in ANSWER_LETTERS:
        return t[0]
    m = re.search(r"(?:ANSWER|CORRECT)\s*(?:IS|:)?\s*([A-D])", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", t)
    if m:
        return m.group(1)
    return None


def build_mcq_prompt(q: dict) -> str:
    return (
        "Answer the following multiple-choice question. "
        "Reply with ONLY the letter (A, B, C, or D).\n\n"
        f"Question: {q['question']}\n{q['choices_text']}\n\nAnswer:"
    )


def eval_vanilla(questions: list[dict]) -> list[dict]:
    out = []
    for i, q in enumerate(questions, 1):
        r = vllm_chat("base", [{"role": "user", "content": build_mcq_prompt(q)}],
                      temperature=0.0, max_tokens=20)
        pred = extract_answer(r.get("response") or "")
        out.append({
            "idx": i, "subject": q["subject"], "domain": q["domain"],
            "correct_answer": q["correct_answer"], "predicted": pred,
            "correct": pred == q["correct_answer"],
            "raw": (r.get("response") or "")[:80],
        })
    return out


def eval_router(questions: list[dict], catalog, cfg) -> list[dict]:
    out = []
    for i, q in enumerate(questions, 1):
        r = run_router(build_mcq_prompt(q), catalog, cfg,
                       temperature=0.0, max_tokens=40)
        pred = extract_answer(r.get("response") or "")
        out.append({
            "idx": i, "subject": q["subject"], "domain": q["domain"],
            "correct_answer": q["correct_answer"], "predicted": pred,
            "correct": pred == q["correct_answer"],
            "routed_to": r.get("adapter_id"),
            "raw": (r.get("response") or "")[:80],
        })
    return out


def eval_agent(questions: list[dict], catalog, cfg) -> list[dict]:
    out = []
    for i, q in enumerate(questions, 1):
        r = run_agent(build_mcq_prompt(q), catalog, cfg,
                      temperature=0.0, max_tokens=400)
        pred = extract_answer(r.get("response") or "")
        out.append({
            "idx": i, "subject": q["subject"], "domain": q["domain"],
            "correct_answer": q["correct_answer"], "predicted": pred,
            "correct": pred == q["correct_answer"],
            "n_steps": len(r.get("steps") or []),
            "raw": (r.get("response") or "")[:80],
        })
    return out


def score(rows: list[dict], mode: str) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    by_domain = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in rows:
        by_domain[r["domain"]]["total"] += 1
        if r["correct"]:
            by_domain[r["domain"]]["correct"] += 1
    return {
        "mode": mode,
        "total": total,
        "correct": correct,
        "accuracy_pct": round(correct / total * 100, 2) if total else 0,
        "per_domain": {
            k: {"accuracy_pct": round(v["correct"] / v["total"] * 100, 2),
                "correct": v["correct"], "total": v["total"]}
            for k, v in sorted(by_domain.items())
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--n", type=int, default=10,
                   help="questions per subject (default 10 → ~100 total)")
    p.add_argument("--mapping", type=Path, default=None,
                   help="JSON {mmlu_subject: adapter_id} mapping")
    p.add_argument("--modes", default="vanilla,router,agent",
                   help="comma-separated subset of modes to run")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    catalog = load_catalog(args.catalog)
    cfg = router_cfg(args.catalog)
    mapping = (json.loads(args.mapping.read_text())
               if args.mapping else DEFAULT_MAPPING)

    print(f"[mmlu_three_way] loading {args.n} questions/subject across "
          f"{len(mapping)} subjects")
    questions = load_questions(mapping, args.n)
    if not questions:
        sys.exit("no questions loaded")
    print(f"  loaded {len(questions)} questions total")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_scores = []
    raw = {}
    for mode in modes:
        print(f"\n[mode] {mode}")
        t0 = time.time()
        if mode == "vanilla":
            rows = eval_vanilla(questions)
        elif mode == "router":
            rows = eval_router(questions, catalog, cfg)
        elif mode == "agent":
            rows = eval_agent(questions, catalog, cfg)
        else:
            print(f"  unknown mode: {mode}; skipping")
            continue
        s = score(rows, mode)
        s["elapsed_seconds"] = round(time.time() - t0, 1)
        all_scores.append(s)
        raw[mode] = rows
        print(f"  → {s['accuracy_pct']:.2f}% ({s['correct']}/{s['total']}) "
              f"in {s['elapsed_seconds']:.1f}s")

    print()
    print(f"{'Mode':<10}{'Acc':>8}{'Correct':>12}{'Time':>10}")
    for s in all_scores:
        print(f"  {s['mode']:<8} {s['accuracy_pct']:>6.2f}%  "
              f"{s['correct']:>3}/{s['total']:<3}  {s['elapsed_seconds']:>6.1f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "catalog": str(args.catalog),
            "mapping": mapping,
            "n_per_subject": args.n,
            "scores": all_scores,
            "raw": raw,
        }, indent=2, default=str))
        print(f"\n  results → {args.out}")


if __name__ == "__main__":
    main()
