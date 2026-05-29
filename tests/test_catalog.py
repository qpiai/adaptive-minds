"""Pure-Python smoke tests — no vLLM, no GPU.

Verifies that the shipped catalog YAML round-trips through the loader and
produces the expected adapter ids, hf_ids, and CLI argv shape.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_minds import Adapter, keyword_pick, load_catalog
from adaptive_minds.catalog import (
    load_config, router_cfg, vllm_lora_args, _adapter_id,
)


HERE = Path(__file__).resolve().parents[1]
CATALOG = HERE / "catalogs" / "qwen25_30.yaml"


def test_catalog_file_exists():
    assert CATALOG.exists(), f"missing {CATALOG}"


def test_load_config_returns_dict_with_required_keys():
    cfg = load_config(CATALOG)
    assert "base_model" in cfg
    assert "lora_adapters" in cfg
    assert isinstance(cfg["lora_adapters"], list)
    assert len(cfg["lora_adapters"]) == 30, (
        f"expected 30 paper-headline adapters, got {len(cfg['lora_adapters'])}")


def test_load_catalog_returns_adapter_instances():
    cat = load_catalog(CATALOG)
    assert len(cat) >= 1
    a = next(iter(cat.values()))
    assert isinstance(a, Adapter)
    assert a.id and a.name and a.hf_id


def test_adapter_ids_are_lowercase_no_spaces():
    cat = load_catalog(CATALOG)
    for aid in cat:
        assert aid == _adapter_id(aid), f"id not normalised: {aid!r}"
        assert " " not in aid
        assert aid == aid.lower()


def test_hf_ids_point_at_expected_repo():
    cat = load_catalog(CATALOG)
    for a in cat.values():
        assert a.hf_id.startswith("pavan01729/adaptive-minds-loras/"), (
            f"{a.id} hf_id={a.hf_id!r} does not point at expected Hub repo")


def test_keyword_pick_returns_an_id_in_catalog():
    cat = load_catalog(CATALOG)
    chosen, scores = keyword_pick("Write a SQL query for top customers", cat)
    assert chosen in cat
    assert isinstance(scores, list)


def test_keyword_pick_routes_sql_query_to_sql():
    cat = load_catalog(CATALOG)
    chosen, _ = keyword_pick(
        "Write a SQL select query joining orders and customers", cat)
    assert chosen == "sql"


def test_vllm_lora_args_shape():
    args = vllm_lora_args(CATALOG)
    assert args[0] == "--enable-lora"
    assert args[1] == "--lora-modules"
    # remaining entries are name=hf_id pairs
    for mod in args[2:]:
        assert "=" in mod
        name, hf_id = mod.split("=", 1)
        assert name == name.lower()
        assert hf_id.startswith("pavan01729/adaptive-minds-loras/")


def test_router_cfg_has_required_keys():
    cfg = router_cfg(CATALOG)
    for k in ("router_prompt", "router_temperature", "router_max_tokens",
              "agent_max_steps", "agent_brain", "agent_brain_max_tokens"):
        assert k in cfg, f"router_cfg missing {k}"
    assert "{query}" in cfg["router_prompt"]
    assert "{domain_list}" in cfg["router_prompt"]


def test_imports_dont_require_vllm():
    """Importing adaptive_minds should not pull vLLM/torch/transformers."""
    import sys
    for forbidden in ("vllm", "torch", "transformers", "peft"):
        assert forbidden not in sys.modules, (
            f"{forbidden} unexpectedly imported by adaptive_minds")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
