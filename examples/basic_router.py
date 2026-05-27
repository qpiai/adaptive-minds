"""Single-step router — pick one adapter and call it.

Run after launching a vLLM server with the catalog adapters:

    $(adaptive-minds serve --catalog catalogs/qwen25_30.yaml) &
    export VLLM_BASE=http://localhost:8000/v1
    python examples/basic_router.py
"""
from adaptive_minds import load_catalog, run_router
from adaptive_minds.catalog import router_cfg


def main() -> None:
    catalog_path = "catalogs/qwen25_30.yaml"
    catalog = load_catalog(catalog_path)
    cfg = router_cfg(catalog_path)

    queries = [
        "Write a SQL query for top 10 customers by revenue in 2023.",
        "Translate to Cypher: find all friends-of-friends of Alice.",
        "Redact all PII from this transcript: 'I am Jane Doe, jane@x.com'.",
    ]
    for q in queries:
        out = run_router(q, catalog, cfg, temperature=0.3, max_tokens=256)
        print(f"\nQ: {q}")
        print(f"→ adapter={out['adapter_id']}")
        print(f"→ {out['response'][:200]}")


if __name__ == "__main__":
    main()
