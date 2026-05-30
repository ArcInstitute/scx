"""Tests for ``cloud_fixtures.ensure_gcp_credentials_or_skip``.

The cloud_* benchmarks need to skip gracefully when GCP credentials aren't
configured — otherwise they crash hard and pollute the dashboard's
``missing_result`` bucket with undifferentiated failures. The helper writes
a typed skip stub instead so the reporting pipeline can classify them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def isolated_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Re-root ``RAW_RESULTS_DIR`` so the test never writes the production
    ``results/raw/`` directory.
    """
    raw = tmp_path / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # ``results.py`` reads ``RAW_RESULTS_DIR`` from ``config`` at call time,
    # so monkeypatching the config module is enough.
    for modname in ("benchmarks.comprehensive.config", "benchmarks.comprehensive.results"):
        if modname in sys.modules:
            del sys.modules[modname]

    import benchmarks.comprehensive.config as cfg
    monkeypatch.setattr(cfg, "RAW_RESULTS_DIR", raw, raising=False)
    import benchmarks.comprehensive.results as results
    monkeypatch.setattr(results, "RAW_RESULTS_DIR", raw, raising=False)

    return raw


def test_skip_when_no_credentials(
    isolated_results_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Returns False and writes a ``no_gcp_credentials`` stub JSON when neither
    ``GOOGLE_APPLICATION_CREDENTIALS`` nor the default key path resolves."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    import benchmarks.comprehensive.cloud_fixtures as cf
    monkeypatch.setattr(cf, "_DEFAULT_KEY_PATH", Path("/tmp/nope_intentionally_missing.json"))

    ok = cf.ensure_gcp_credentials_or_skip(
        benchmark="cloud_push",
        format_key="scx_auto",
        dataset_name="pbmc3k",
    )

    assert ok is False
    stub = isolated_results_dir / "cloud_push__scx_auto__pbmc3k.json"
    assert stub.exists(), f"Skip stub missing at {stub}"

    payload = json.loads(stub.read_text())
    assert payload["missing_reason"] == "no_gcp_credentials"
    assert payload["benchmark"] == "cloud_push"
    assert payload["format"] == "scx_auto"
    assert payload["dataset"] == "pbmc3k"


def test_continue_when_credentials_present(
    isolated_results_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Returns True without writing a stub when a readable key file is set."""
    key_file = tmp_path / "fake_key.json"
    key_file.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_file))

    import benchmarks.comprehensive.cloud_fixtures as cf
    ok = cf.ensure_gcp_credentials_or_skip(
        benchmark="cloud_pull",
        format_key="scx_auto",
        dataset_name="pbmc3k",
    )

    assert ok is True
    stub = isolated_results_dir / "cloud_pull__scx_auto__pbmc3k.json"
    assert not stub.exists(), (
        f"Stub should NOT be written when credentials resolve: {stub}"
    )


def test_falls_back_to_default_key_when_env_unset(
    isolated_results_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When GOOGLE_APPLICATION_CREDENTIALS is unset but ~/.gcp/scx-bench.json
    exists, the helper picks it up and returns True."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    default_key = tmp_path / "default_key.json"
    default_key.write_text("{}")

    import benchmarks.comprehensive.cloud_fixtures as cf
    monkeypatch.setattr(cf, "_DEFAULT_KEY_PATH", default_key)

    ok = cf.ensure_gcp_credentials_or_skip(
        benchmark="cloud_metadata",
        format_key="scx_auto",
        dataset_name="pbmc3k",
    )

    assert ok is True
