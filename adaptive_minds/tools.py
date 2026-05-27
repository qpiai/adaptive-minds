"""External-tool registry for the agent.

Five sandboxed tools defined in `external_tools.py`. Handlers run in the
same process (no subprocess hop) and return a string observation. The
prompt-side help text in `TOOL_DESCRIPTIONS` is rendered into the agent
system prompt.
"""
from __future__ import annotations

import json

from .external_tools import EXTERNAL_TOOLS


# Help text rendered into the agent system prompt.
TOOL_DESCRIPTIONS = {
    "calculator": "math via sympy. arg = a PLAIN expression. e.g. `2**32 + 17` or `solve(x**2-4, x)`",
    "code":       "run a PYTHON snippet in a 12s sandbox. arg = the snippet; use print() for output.",
    "shell":      "run a BASH one-liner in a 10s sandbox. arg = the command, e.g. `wc -l file | awk '$1>80'`",
    "websearch":  "top-5 DuckDuckGo hits. arg = the query string.",
    "pulp":       "linear program solver. arg = JSON `{\"sense\":\"max\",\"objective\":...,\"constraints\":[...]}`",
}


def list_tools() -> list[str]:
    return sorted(EXTERNAL_TOOLS.keys())


def _normalise_arg(name: str, arg: str) -> str:
    """Brain outputs sometimes wrap the arg in JSON. Unwrap for the simple
    tools; let `pulp` keep the full JSON spec since that's its shape."""
    arg = arg.strip()
    if not arg or arg[0] not in "{[":
        return arg
    try:
        j = json.loads(arg)
    except Exception:
        return arg
    if isinstance(j, dict):
        for k in ("expression", "query", "code", "snippet",
                  "spec", "specification", "input"):
            if k in j and isinstance(j[k], str):
                return j[k]
        if name == "pulp":
            return arg
        for v in j.values():
            if isinstance(v, str):
                return v
    return arg


def run_tool(name: str, arg: str) -> str:
    """Return the observation string the agent should see. Errors are
    returned (not raised) so a broken tool can't break the loop."""
    handler = EXTERNAL_TOOLS.get(name)
    if not handler:
        return f"ERROR: unknown tool '{name}'. Valid: {', '.join(EXTERNAL_TOOLS)}"
    normalised = _normalise_arg(name, arg)
    try:
        result = handler(normalised)
    except Exception as e:
        return f"ERROR: tool '{name}' raised: {e}"
    if not isinstance(result, dict):
        return str(result)[:4000]
    return result.get("output") or result.get("error") or "(empty)"
