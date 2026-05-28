"""Command-line entry point.

    adaptive-minds serve   --catalog catalogs/qwen25_30.yaml      # vLLM launch helper
    adaptive-minds server  --catalog catalogs/qwen25_30.yaml      # FastAPI server
    adaptive-minds route   --catalog catalogs/qwen25_30.yaml --query "..."
    adaptive-minds agent   --catalog catalogs/qwen25_30.yaml --query "..."
    adaptive-minds list    --catalog catalogs/qwen25_30.yaml

The chat UI is a Next.js app under `ui/`; bring it up via
`docker compose up -d` (the `ui` service) or `cd ui && npm run dev` for
local development.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from .agent import run_agent
from .catalog import load_catalog, load_config, router_cfg, vllm_lora_args
from .router import run_router

app = typer.Typer(add_completion=False, help="Adaptive Minds CLI.")


def _emit(d: dict, *, pretty: bool) -> None:
    if pretty:
        typer.echo(json.dumps(d, indent=2, default=str))
    else:
        typer.echo(d.get("response") or json.dumps(d, default=str))


@app.command()
def serve(
    catalog: Path = typer.Option(..., "--catalog", "-c", exists=True),
    base_model: str = typer.Option(None, "--base-model",
                                   help="Override base model (default: from catalog)"),
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option("0.0.0.0", "--host"),
    extra: list[str] = typer.Option([], "--extra",
                                    help="Extra args passed through to vllm serve"),
) -> None:
    """Print (or exec) the vllm serve command for the given catalog.

    Set AM_SERVE=exec to actually launch; otherwise just prints the command
    so you can review or pipe it to bash.
    """
    cfg = load_config(catalog)
    bm = base_model or (cfg.get("base_model") or {}).get("hf_id")
    if not bm:
        typer.secho("ERROR: no base_model.hf_id in catalog and --base-model "
                    "not provided", fg="red")
        raise typer.Exit(1)
    lora_args = vllm_lora_args(catalog)
    max_rank = (cfg.get("base_model") or {}).get("max_lora_rank", 64)
    argv = [
        "vllm", "serve", bm,
        "--host", host, "--port", str(port),
        "--max-lora-rank", str(max_rank),
        *lora_args,
        *extra,
    ]
    if os.environ.get("AM_SERVE") == "exec":
        os.execvp("vllm", argv)
    typer.echo(" ".join(argv))


@app.command("list")
def list_adapters(
    catalog: Path = typer.Option(..., "--catalog", "-c", exists=True),
) -> None:
    """Print one adapter id per line."""
    cat = load_catalog(catalog)
    for aid, a in cat.items():
        typer.echo(f"{aid}\t{a.hf_id}\t{a.description[:80]}")


@app.command()
def route(
    catalog: Path = typer.Option(..., "--catalog", "-c", exists=True),
    query: str = typer.Option(..., "--query", "-q"),
    temperature: float = typer.Option(0.3, "--temperature"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Single-step routing: base picks an adapter, that adapter answers."""
    cat = load_catalog(catalog)
    if not cat:
        typer.secho("ERROR: catalog is empty", fg="red")
        raise typer.Exit(1)
    cfg = router_cfg(catalog)
    out = run_router(query, cat, cfg, temperature, max_tokens)
    _emit(out, pretty=pretty)
    sys.exit(0 if out.get("ok") else 2)


@app.command()
def agent(
    catalog: Path = typer.Option(..., "--catalog", "-c", exists=True),
    query: str = typer.Option(..., "--query", "-q"),
    temperature: float = typer.Option(0.3, "--temperature"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Multi-step ReAct agent: brain plans, tools/experts execute, brain synthesises."""
    cat = load_catalog(catalog)
    if not cat:
        typer.secho("ERROR: catalog is empty", fg="red")
        raise typer.Exit(1)
    cfg = router_cfg(catalog)
    out = run_agent(query, cat, cfg, temperature, max_tokens)
    _emit(out, pretty=pretty)
    sys.exit(0 if out.get("ok") else 2)


@app.command()
def server(
    catalog: Path = typer.Option(..., "--catalog", "-c", exists=True),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8765, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI server (router + agent + chat HTTP endpoints)."""
    try:
        import uvicorn
    except ImportError:
        typer.secho("ERROR: install the [serve] extra: pip install '.[serve]'",
                    fg="red")
        raise typer.Exit(1)
    # Fail fast with a friendly message if the chosen host:port is already
    # bound — uvicorn's stock OSError is opaque, and 8765 in docker setups
    # collides with the AM_SERVER_PORT override the user may have forgotten.
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
    except OSError:
        typer.secho(
            f"ERROR: port {port} on {host} is already bound. "
            f"Pass --port <N> or set AM_SERVER_PORT in .env (docker compose).",
            fg="red")
        raise typer.Exit(1)
    finally:
        s.close()
    # Set AM_CATALOG so the imported module configures itself on import
    # (covers the --reload case where uvicorn re-imports per file change).
    os.environ["AM_CATALOG"] = str(catalog)
    from . import server as server_mod
    server_mod.configure(catalog)
    uvicorn.run(
        "adaptive_minds.server:app",
        host=host, port=port, reload=reload, log_level="info",
    )


if __name__ == "__main__":
    app()
