"""Smoke tests for the data-load Phase 0 benchmark modules.

Hermetic + fast: builds a tiny (600×200) h5ad + SCX fixture under an isolated
``SCX_WORK_DIR``, then drives each new ``run()`` on ``pbmc3k``-scale data and
asserts the expected per-scenario ``extra`` keys + ``cache_policy`` appear.
Skips cleanly when ``pyscx`` isn't importable (CPU-only CI without the built
extension). Competitor epoch fns are covered when their libs are present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscx")
pytest.importorskip("anndata")


# ---------------------------------------------------------------------------
# Isolated work dir + tiny fixtures (mirrors test_run_parallel_deps.isolated_work_dir)
# ---------------------------------------------------------------------------


@pytest.fixture()
def phase0_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    work = tmp_path / "work"
    data = work / "benchmarks" / "datasets"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCX_WORK_DIR", str(work))
    monkeypatch.setenv("SCX_DATA_DIR", str(data))
    # Reload env-dependent modules so DATA_DIR + the bench modules re-resolve.
    for m in (
        "benchmarks.comprehensive.bench_env",
        "benchmarks.comprehensive.config",
        "benchmarks.comprehensive.benchmarks.ooc_loader",
        "benchmarks.comprehensive.benchmarks.cellset_gather",
        "benchmarks.comprehensive.benchmarks.obs_open",
    ):
        sys.modules.pop(m, None)

    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    import pyscx
    from benchmarks.comprehensive.config import DatasetConfig, FormatVariant

    rng = np.random.default_rng(0)
    n_obs, n_vars = 600, 200
    X = sp.random(n_obs, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=0)
    X.data = np.ceil(X.data * 10).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(rng.choice(["A", "B", "C"], size=n_obs)),
            "batch": pd.Categorical(rng.choice(["b1", "b2"], size=n_obs)),
        },
        index=[f"c{i}" for i in range(n_obs)],
    )
    adata = ad.AnnData(X=X, obs=obs)

    ds = DatasetConfig(
        id="P0", name="phase0_tiny", n_obs=n_obs, n_vars=n_vars,
        protocol="synthetic", source="test", approx_h5ad_mb=1,
    )
    ds.h5ad_path.write_bytes(b"")  # placeholder; overwritten next line
    adata.write_h5ad(ds.h5ad_path)
    scx_path = ds.path_for_format("scx_auto")  # {name}_auto.scx
    pyscx.from_anndata(adata, str(scx_path))

    scx_fv = FormatVariant("SCX (auto)", "scx_auto", "primary", "scx_runner", {"codec": "auto"})
    h5_fv = FormatVariant("h5ad (uncompressed)", "h5ad_none", "primary", "h5ad_runner", {"compression": None})
    return {"ds": ds, "scx_fv": scx_fv, "h5_fv": h5_fv, "scx_path": scx_path}


# ---------------------------------------------------------------------------
# cache_control
# ---------------------------------------------------------------------------


def test_drop_file_cache_labels(tmp_path: Path):
    from benchmarks.comprehensive.cache_control import drop_file_cache

    assert drop_file_cache(tmp_path / "does_not_exist") == "warm"
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * (1 << 20))
    assert drop_file_cache(f) in ("cold_fadvise", "warm")


def test_base_runner_delegates_to_cache_control(tmp_path: Path):
    from benchmarks.comprehensive.runners.base import FormatRunner

    assert FormatRunner._drop_file_cache(tmp_path / "nope") == "warm"


# ---------------------------------------------------------------------------
# Registration + gating
# ---------------------------------------------------------------------------


def test_registered_in_all_benchmarks():
    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    for b in ("ooc_loader", "cellset_gather", "obs_open"):
        assert b in ALL_BENCHMARKS


def test_supported_formats_declared():
    import importlib

    for mod, expected in (
        ("ooc_loader", {"scx_auto", "annbatch", "annloader", "scdataset", "h5ad_none", "tiledb_soma"}),
        ("cellset_gather", {"scx_auto", "scx_fast"}),
        ("obs_open", {"scx_auto", "h5ad_none"}),
    ):
        m = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{mod}")
        assert isinstance(m.SUPPORTED_FORMATS, frozenset)
        assert expected <= m.SUPPORTED_FORMATS


def test_benches_skip_unsupported_format(phase0_env):
    import importlib

    from benchmarks.comprehensive.config import FormatVariant

    unsupported = FormatVariant("Zarr", "zarr_zstd", "primary", "zarr_runner", {})
    for mod in ("ooc_loader", "cellset_gather", "obs_open"):
        m = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{mod}")
        assert m.run(phase0_env["ds"], unsupported, n_runs=1) is None


# ---------------------------------------------------------------------------
# End-to-end SCX runs
# ---------------------------------------------------------------------------


def test_ooc_loader_scx(phase0_env):
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.ooc_loader")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert res is not None and res.benchmark == "ooc_loader"
    assert res.runs, "no runs recorded"
    keys = set().union(*(r.extra.keys() for r in res.runs))
    assert any(k.startswith("samples_per_sec__") for k in keys)
    assert any(k.startswith("epoch_wall_s__") for k in keys)
    assert all("cache_policy" in r.extra for r in res.runs)
    assert {r.extra["cache_policy"] for r in res.runs} <= {"cold_fadvise", "warm"}


def test_cellset_gather_scx(phase0_env):
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert res is not None and res.benchmark == "cellset_gather"
    keys = set().union(*(r.extra.keys() for r in res.runs))
    assert any(k.startswith("cellsets_per_sec__") for k in keys)
    assert any(k.startswith("cells_per_sec__gather__") for k in keys)


def test_obs_open_scx(phase0_env):
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.obs_open")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert res is not None and res.benchmark == "obs_open"
    keys = set().union(*(r.extra.keys() for r in res.runs))
    assert any(k.startswith("obs_open_s_per_file__open_1file") for k in keys)


def test_obs_open_h5ad_baseline(phase0_env):
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.obs_open")
    res = m.run(phase0_env["ds"], phase0_env["h5_fv"], n_runs=1, cold_cache=True)
    assert res is not None and res.format == "h5ad_none"
    assert res.runs


# ---------------------------------------------------------------------------
# Competitor epoch fns (skipped when the lib is absent)
# ---------------------------------------------------------------------------


def test_annloader_epoch(phase0_env):
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.ooc_loader")
    if not m._HAS_ANNLOADER:
        pytest.skip("AnnLoader not available")
    r = m._run_annloader_epoch(str(phase0_env["ds"].h5ad_path), 64, True, True, 42)
    assert r.n_cells == phase0_env["ds"].n_obs


def test_scdataset_epoch(phase0_env):
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.ooc_loader")
    if not (m._HAS_SCDATASET and m._HAS_TORCH):
        pytest.skip("scDataset/torch not available")
    r = m._run_scdataset_epoch(str(phase0_env["ds"].h5ad_path), 64, True, True, 42)
    assert r.n_cells == phase0_env["ds"].n_obs
