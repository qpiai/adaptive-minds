"""Tests for the typer CLI. Hermetic — no vLLM, no GPU, no network."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from adaptive_minds.cli import app

SMOKE = Path(__file__).resolve().parents[1] / "catalogs" / "qwen25_smoke.yaml"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_root(runner: CliRunner) -> None:
    """Top-level --help lists every subcommand we ship."""
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for sub in ("serve", "server", "route", "agent", "list"):
        assert sub in r.stdout, f"missing subcommand in help: {sub}"


def test_list_prints_adapters(runner: CliRunner) -> None:
    """`adaptive-minds list` prints one tab-separated row per adapter."""
    r = runner.invoke(app, ["list", "--catalog", str(SMOKE)])
    assert r.exit_code == 0
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    assert len(lines) == 2, f"smoke catalog should print 2 rows, got: {lines}"
    ids = {line.split("\t", 1)[0] for line in lines}
    assert ids == {"chemistry", "sql"}


def test_list_missing_catalog_exits_nonzero(runner: CliRunner, tmp_path: Path) -> None:
    """Missing catalog → typer's `exists=True` rejects with exit code 2."""
    missing = tmp_path / "nope.yaml"
    r = runner.invoke(app, ["list", "--catalog", str(missing)])
    assert r.exit_code != 0


def test_serve_prints_vllm_command(runner: CliRunner) -> None:
    """`serve` is a launch-helper — it should print the vllm command, not run it."""
    r = runner.invoke(app, ["serve", "--catalog", str(SMOKE)])
    assert r.exit_code == 0
    out = r.stdout
    assert out.startswith("vllm serve"), f"unexpected first token: {out[:30]!r}"
    assert "Qwen/Qwen2.5-7B-Instruct" in out
    assert "--max-lora-rank" in out
    # Both adapters should appear in --lora-modules
    assert "chemistry=" in out
    assert "sql=" in out


def test_serve_respects_base_model_override(runner: CliRunner) -> None:
    """`--base-model X` swaps the model id without touching the catalog."""
    r = runner.invoke(app, ["serve", "--catalog", str(SMOKE),
                            "--base-model", "meta-llama/Llama-3.1-8B-Instruct"])
    assert r.exit_code == 0
    assert "meta-llama/Llama-3.1-8B-Instruct" in r.stdout
    assert "Qwen/Qwen2.5-7B-Instruct" not in r.stdout


def test_serve_missing_base_model_errors(runner: CliRunner, tmp_path: Path) -> None:
    """A catalog without base_model.hf_id and no --base-model → exit 1."""
    bad = tmp_path / "no_base.yaml"
    bad.write_text(yaml.safe_dump({
        "router": {"prompt_template": "{query}\n{domain_list}"},
        "lora_adapters": [{"name": "X", "hf_subdir": "x", "enabled": True}],
    }))
    r = runner.invoke(app, ["serve", "--catalog", str(bad)])
    assert r.exit_code == 1
    assert "no base_model" in r.stdout.lower()


def test_route_requires_query(runner: CliRunner) -> None:
    """`route` without --query should fail before any vLLM call."""
    r = runner.invoke(app, ["route", "--catalog", str(SMOKE)])
    assert r.exit_code != 0


def test_agent_requires_query(runner: CliRunner) -> None:
    """`agent` without --query should fail before any vLLM call."""
    r = runner.invoke(app, ["agent", "--catalog", str(SMOKE)])
    assert r.exit_code != 0
