"""Catalog YAML smoke test: the 2-adapter smoke catalog is the canonical
quickstart payload for docker compose; it must parse + have known shape."""
from __future__ import annotations

from pathlib import Path

from adaptive_minds.catalog import load_catalog, load_config, router_cfg

SMOKE = Path(__file__).resolve().parents[1] / "catalogs" / "qwen25_smoke.yaml"


def test_smoke_catalog_parses_two_adapters() -> None:
    """The smoke catalog has exactly two enabled adapters with stable ids."""
    cat = load_catalog(SMOKE)
    assert set(cat.keys()) == {"chemistry", "sql"}
    chem = cat["chemistry"]
    assert chem.id == "chemistry"
    assert chem.system_prompt  # adapter must declare a system prompt
    assert chem.keywords, "smoke chemistry needs keyword baseline coverage"
    sql = cat["sql"]
    assert sql.id == "sql"
    assert "sql" in sql.keywords


def test_smoke_catalog_router_cfg_shape() -> None:
    """router_cfg(smoke) returns the keys the runtime needs at call time."""
    cfg = router_cfg(SMOKE)
    assert "router_prompt" in cfg
    # The prompt template must mention both placeholders or formatting breaks.
    assert "{query}" in cfg["router_prompt"]
    assert "{domain_list}" in cfg["router_prompt"]
    assert cfg.get("base_model_id"), "router_cfg must surface the base model id"


def test_smoke_catalog_base_model() -> None:
    """Smoke catalog ships with Qwen2.5-7B-Instruct as the base."""
    cfg = load_config(SMOKE)
    assert cfg["base_model"]["hf_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert int(cfg["base_model"]["max_lora_rank"]) >= 32
