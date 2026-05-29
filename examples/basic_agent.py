"""Multi-step ReAct agent — brain plans, tools/experts execute, brain synthesises.

Run after launching a vLLM server with the catalog adapters:

    $(adaptive-minds serve --catalog catalogs/qwen25_30.yaml) &
    export VLLM_BASE=http://localhost:8000/v1
    python examples/basic_agent.py
"""
from adaptive_minds import load_catalog, run_agent
from adaptive_minds.catalog import router_cfg


def main() -> None:
    catalog_path = "catalogs/qwen25_30.yaml"
    catalog = load_catalog(catalog_path)
    cfg = router_cfg(catalog_path)

    query = (
        "Compute 2**32 + 17 with the calculator, then have the finance "
        "expert explain what that figure could plausibly represent on a "
        "company's balance sheet."
    )

    out = run_agent(query, catalog, cfg, temperature=0.3, max_tokens=1024)
    print(f"Query: {query}\n")
    print(f"Response:\n{out['response']}\n")
    print(f"Mode: {out['mode']}  •  Steps: {len(out['steps'])}  "
          f"•  Elapsed: {out['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
