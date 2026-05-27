"""Tests for docker/prefetch.py. Hermetic — mocks the HF Hub call."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PREFETCH_PATH = Path(__file__).resolve().parents[1] / "docker" / "prefetch.py"


def _load_prefetch_module():
    """Import docker/prefetch.py as a module (it's not on sys.path)."""
    spec = importlib.util.spec_from_file_location("docker_prefetch", PREFETCH_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["docker_prefetch"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def prefetch():
    return _load_prefetch_module()


def test_empty_subdirs_noop(monkeypatch: pytest.MonkeyPatch, prefetch, capsys) -> None:
    """No LORA_SUBDIRS → exits 0, prints a clear no-op message, no HF call."""
    monkeypatch.setenv("LORA_SUBDIRS", "")
    with patch.object(prefetch, "snapshot_download") as dl:
        rc = prefetch.main()
        assert rc == 0
        dl.assert_not_called()
    assert "nothing to do" in capsys.readouterr().out.lower()


def test_subdirs_trigger_snapshot_download_per_entry(
    monkeypatch: pytest.MonkeyPatch, prefetch, tmp_path: Path,
) -> None:
    """One snapshot_download call per comma-separated subdir, with the
    correct allow_patterns (weights only — no tokenizer files)."""
    monkeypatch.setenv("LORA_REPO", "fake/repo")
    monkeypatch.setenv("LORA_SUBDIRS",
                       "qwen2.5-7b/qwen25_chem_v1_grpo,qwen2.5-7b/qwen25_sql_v1")
    monkeypatch.setenv("LORA_OUT_DIR", str(tmp_path))
    fake_dl = MagicMock(return_value=str(tmp_path))
    with patch.object(prefetch, "snapshot_download", fake_dl):
        rc = prefetch.main()
    assert rc == 0
    assert fake_dl.call_count == 2
    # Inspect each call: repo positional, allow_patterns kw, no tokenizer files.
    for call in fake_dl.call_args_list:
        assert call.args[0] == "fake/repo"
        patterns = call.kwargs["allow_patterns"]
        assert any(p.endswith("/adapter_config.json") for p in patterns)
        assert any(p.endswith("/adapter_model.safetensors") for p in patterns)
        assert not any("tokenizer" in p for p in patterns), (
            "tokenizer files must be excluded so they can't override the "
            "base model's chat template at serving time")
        assert call.kwargs["local_dir"] == str(tmp_path)


def test_lora_out_dir_default(monkeypatch: pytest.MonkeyPatch, prefetch) -> None:
    """Default LORA_OUT_DIR is /loras (matches the docker-compose mount)."""
    monkeypatch.setenv("LORA_SUBDIRS", "x")
    monkeypatch.delenv("LORA_OUT_DIR", raising=False)
    fake_dl = MagicMock()
    with patch.object(prefetch, "snapshot_download", fake_dl):
        prefetch.main()
    assert fake_dl.call_args.kwargs["local_dir"] == "/loras"
