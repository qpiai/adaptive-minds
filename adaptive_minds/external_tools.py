"""External (non-LoRA) tools for the Adaptive Minds ReAct agent.

Each handler takes a single string argument and returns a dict
{output, debug, error}. The same calling convention as adapter calls so
the agent doesn't need to learn a different surface.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import time
import traceback
from typing import Any


def _result(output: str = "", debug: dict | None = None,
            error: str | None = None) -> dict:
    return {
        "output": (output or "")[:4000],   # cap to keep prompts manageable
        "debug": debug or {},
        "error": error,
    }


# -----------------------------------------------------------------------
# Calculator — sympy-based symbolic + numeric math
# -----------------------------------------------------------------------

def calc_handler(sub_query: str) -> dict:
    """Evaluate a math expression with sympy.

    Accepts plain arithmetic ('2+3*4'), symbolic ('solve(x**2-4, x)'),
    integrals/derivatives, and unit-free numeric work.
    """
    expr = (sub_query or "").strip()
    if not expr:
        return _result(error="empty expression")
    t0 = time.time()
    try:
        import sympy as sp
        from sympy import (Symbol, sympify, simplify, solve, integrate, diff,
                           limit, Sum, oo, sin, cos, tan, exp, log, sqrt,
                           pi, E, Matrix, factor, expand, series)
        ns = {
            "sp": sp, "sympify": sympify, "simplify": simplify, "solve": solve,
            "integrate": integrate, "diff": diff, "limit": limit, "Sum": Sum,
            "Symbol": Symbol, "Matrix": Matrix, "factor": factor,
            "expand": expand, "series": series,
            "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log,
            "sqrt": sqrt, "pi": pi, "E": E, "oo": oo,
            "x": Symbol("x"), "y": Symbol("y"), "z": Symbol("z"),
            "n": Symbol("n"), "t": Symbol("t"),
            "a": Symbol("a"), "b": Symbol("b"), "c": Symbol("c"),
        }
        try:
            value = eval(expr, {"__builtins__": {}}, ns)
        except Exception:
            value = sympify(expr)
        try:
            simplified = simplify(value)
        except Exception:
            simplified = value
        try:
            numeric = float(simplified)
            numeric_str = f"  ≈ {numeric:.10g}"
        except Exception:
            numeric_str = ""
        out = f"{simplified}{numeric_str}"
        return _result(output=out, debug={
            "engine": "sympy", "expression": expr,
            "raw_value": str(value),
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
    except Exception as e:
        return _result(error=f"calculator failed: {e}",
                       debug={"expression": expr,
                              "traceback": traceback.format_exc()})


# -----------------------------------------------------------------------
# Python code sandbox — subprocess with timeout
# -----------------------------------------------------------------------

_CODE_TIMEOUT_SEC = 12


def code_handler(sub_query: str) -> dict:
    """Execute a Python snippet in a fresh subprocess and return stdout.
    The snippet should print() its results."""
    snippet = sub_query or ""
    if not snippet.strip():
        return _result(error="empty snippet")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, timeout=_CODE_TIMEOUT_SEC,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        if proc.returncode != 0:
            return _result(
                error=f"exit {proc.returncode}",
                output=proc.stdout,
                debug={"stderr": proc.stderr[-1500:],
                       "elapsed_ms": elapsed_ms,
                       "snippet": snippet[:1000]},
            )
        return _result(output=proc.stdout, debug={
            "stderr": proc.stderr[-500:] if proc.stderr else "",
            "elapsed_ms": elapsed_ms, "snippet": snippet[:1000],
        })
    except subprocess.TimeoutExpired:
        return _result(error=f"timeout after {_CODE_TIMEOUT_SEC}s",
                       debug={"snippet": snippet[:1000]})
    except Exception as e:
        return _result(error=f"code execution failed: {e}",
                       debug={"snippet": snippet[:1000],
                              "traceback": traceback.format_exc()})


# -----------------------------------------------------------------------
# Shell — execute a bash one-liner in a sandboxed subprocess.
# -----------------------------------------------------------------------

def shell_handler(sub_query: str) -> dict:
    """Run a bash one-liner with a hard 10s timeout. Captures stdout+stderr."""
    cmd = (sub_query or "").strip()
    if not cmd:
        return _result(error="empty command")
    danger = ["rm -rf", "mkfs", "dd if=", ":(){", "> /dev/sd", "shutdown",
              "reboot", "halt", "init 0"]
    low = cmd.lower()
    for d in danger:
        if d in low:
            return _result(error=f"refused: command contains '{d}'")
    t0 = time.time()
    try:
        p = subprocess.run(
            ["bash", "-lc", cmd], capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return _result(error="shell command timed out (>10s)")
    elapsed_ms = int((time.time() - t0) * 1000)
    out = (p.stdout or "")[:4000]
    err = (p.stderr or "")[:1000]
    parts = []
    if out:
        parts.append(out.rstrip())
    if err:
        parts.append(f"[stderr]\n{err.rstrip()}")
    if p.returncode != 0:
        parts.append(f"[exit code {p.returncode}]")
    return _result(output="\n".join(parts) or "(no output)",
                   debug={"elapsed_ms": elapsed_ms,
                          "returncode": p.returncode})


# -----------------------------------------------------------------------
# DuckDuckGo web search
# -----------------------------------------------------------------------

def websearch_handler(sub_query: str) -> dict:
    """Search the web via DuckDuckGo and return top 5 hits."""
    query = (sub_query or "").strip()
    if not query:
        return _result(error="empty query")
    t0 = time.time()
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        elapsed_ms = int((time.time() - t0) * 1000)
        if not hits:
            return _result(output="(no results)",
                           debug={"query": query, "elapsed_ms": elapsed_ms})
        lines = []
        for i, h in enumerate(hits, 1):
            title = (h.get("title") or "")[:140]
            body = (h.get("body") or h.get("snippet") or "")[:280]
            url = h.get("href") or h.get("url") or ""
            lines.append(f"[{i}] {title}\n    {body}\n    URL: {url}")
        out = "\n".join(lines)
        return _result(output=out, debug={
            "query": query, "n_results": len(hits),
            "elapsed_ms": elapsed_ms,
            "urls": [h.get("href") or h.get("url") for h in hits],
        })
    except Exception as e:
        return _result(error=f"websearch failed: {e}",
                       debug={"query": query,
                              "traceback": traceback.format_exc()})


# -----------------------------------------------------------------------
# PuLP — linear programming
# -----------------------------------------------------------------------

def pulp_handler(sub_query: str) -> dict:
    """Solve a small LP/MIP. JSON spec preferred; falls back to Python snippet."""
    text = (sub_query or "").strip()
    if not text:
        return _result(error="empty problem")
    t0 = time.time()
    try:
        spec = json.loads(text)
        return _solve_lp_from_spec(spec, t0)
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        import pulp
        ns: dict[str, Any] = {
            "pulp": pulp, "LpProblem": pulp.LpProblem,
            "LpVariable": pulp.LpVariable, "LpMaximize": pulp.LpMaximize,
            "LpMinimize": pulp.LpMinimize, "lpSum": pulp.lpSum,
            "PULP_CBC_CMD": pulp.PULP_CBC_CMD, "value": pulp.value,
            "__builtins__": __builtins__,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(text, ns)
        out = buf.getvalue()
        return _result(output=out, debug={
            "mode": "snippet",
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
    except Exception as e:
        return _result(error=f"pulp failed: {e}",
                       debug={"snippet": text[:1000],
                              "traceback": traceback.format_exc()})


def _solve_lp_from_spec(spec: dict, t0: float) -> dict:
    """Solve an LP described declaratively.

    Spec format:
      {"sense": "max"|"min",
       "variables": {"x": {"lb":0, "ub":10, "cat":"Continuous"}, ...},
       "objective": {"x": 3, "y": 5},
       "constraints": [{"expr": {"x":1, "y":1}, "op":"<=", "rhs": 4}, ...]}
    """
    import pulp
    sense_map = {"max": pulp.LpMaximize, "min": pulp.LpMinimize,
                 "maximize": pulp.LpMaximize, "minimize": pulp.LpMinimize}
    sense = sense_map[spec.get("sense", "max").lower()]
    prob = pulp.LpProblem("am_lp", sense)
    vars_ = {}
    for name, vinfo in spec.get("variables", {}).items():
        vars_[name] = pulp.LpVariable(
            name, lowBound=vinfo.get("lb"), upBound=vinfo.get("ub"),
            cat=vinfo.get("cat", "Continuous"),
        )
    obj = spec.get("objective", {})
    prob += pulp.lpSum(coef * vars_[n] for n, coef in obj.items())
    for c in spec.get("constraints", []):
        expr = pulp.lpSum(coef * vars_[n] for n, coef in c["expr"].items())
        op = c["op"]
        rhs = c["rhs"]
        if op == "<=":
            prob += (expr <= rhs)
        elif op == ">=":
            prob += (expr >= rhs)
        elif op == "==":
            prob += (expr == rhs)
        else:
            raise ValueError(f"bad op: {op}")
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    elapsed_ms = int((time.time() - t0) * 1000)
    if pulp.LpStatus[status] != "Optimal":
        return _result(error=f"LP status: {pulp.LpStatus[status]}",
                       debug={"spec": spec, "elapsed_ms": elapsed_ms})
    soln = {n: pulp.value(v) for n, v in vars_.items()}
    obj_val = pulp.value(prob.objective)
    out = (f"Status: Optimal\nObjective: {obj_val}\n"
           f"Variables: " + ", ".join(f"{n}={soln[n]:g}" for n in soln))
    return _result(output=out, debug={
        "mode": "json_spec", "elapsed_ms": elapsed_ms,
        "objective_value": obj_val, "solution": soln,
    })


EXTERNAL_TOOLS = {
    "calculator": calc_handler,
    "code":       code_handler,
    "shell":      shell_handler,
    "websearch":  websearch_handler,
    "pulp":       pulp_handler,
}
