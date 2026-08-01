"""Smoke tests for the data-load Phase 1D benchmark module (`shuffle_layout`).

Hermetic + fast, mirroring `test_dataload_phase0.py`: a small (6000x200) h5ad +
SCX fixture under an isolated ``SCX_WORK_DIR``, then drives ``run()`` and
asserts the four arms emit what they claim to.

The bias this guards against is specific: three of the four arms can degrade
into *silence* rather than failure. `size_delta` skips codec variants that
aren't on disk, `shuffle_quality` skips datasets with no label column, and
`train_throughput` swallows per-run exceptions so one bad epoch can't sink a
cohort job. Each of those is correct behaviour and each of them, unwatched,
turns a broken arm into an empty `applicable: False` nobody reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscx")
pytest.importorskip("anndata")

_HAS_SHUFFLE = hasattr(pytest.importorskip("pyscx"), "shuffle")
pytestmark = pytest.mark.skipif(
    not _HAS_SHUFFLE, reason="pyscx build predates `shuffle` (scx Phase 1D)"
)


@pytest.fixture()
def phase1d_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated work dir with an `scx_auto` fixture plus three codec variants.

    Several variants, not one, because the whole point of the size arm is the
    *contrast*: `scx1` codes each row's gene indices independently of row order
    and is near-neutral under a permutation, while `shufdelta` compresses the
    shard byte stream as a whole and is not. A one-variant fixture cannot tell a
    working sweep from a sweep that reports the same number twice.
    """
    work = tmp_path / "work"
    data = work / "benchmarks" / "datasets"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCX_WORK_DIR", str(work))
    monkeypatch.setenv("SCX_DATA_DIR", str(data))
    for m in (
        "benchmarks.comprehensive.bench_env",
        "benchmarks.comprehensive.config",
        "benchmarks.comprehensive.benchmarks.shuffle_layout",
    ):
        sys.modules.pop(m, None)

    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    import pyscx
    from benchmarks.comprehensive.config import DatasetConfig, FormatVariant

    n_obs, n_vars = 6000, 200
    X = sp.random(n_obs, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=0)
    X.data = np.ceil(X.data * 10).astype(np.float32)
    # Perfectly clustered obs order: 2000 A, then 2000 B, then 2000 C. The
    # quality arm must be able to see the block structure collapse.
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(np.repeat(["A", "B", "C"], n_obs // 3))},
        index=[f"c{i}" for i in range(n_obs)],
    )
    adata = ad.AnnData(X=X, obs=obs)

    ds = DatasetConfig(
        id="P1D", name="phase1d_tiny", n_obs=n_obs, n_vars=n_vars,
        protocol="synthetic", source="test", approx_h5ad_mb=1,
    )
    adata.write_h5ad(ds.h5ad_path)
    scx_path = ds.path_for_format("scx_auto")
    # 30 shards of 200 rows: enough block structure to measure, and *more than
    # 8* so the `sgs8` pooled figure is not degenerate (pooling every block is
    # the whole corpus, which is 0.0 on any row order).
    pyscx.from_anndata(adata, str(scx_path), shard_size=200)
    for key, codec in (
        ("scx_scx1", "scx1"),
        ("scx_zstd", "zstd"),
        ("scx_shufdelta", "shufdelta"),
    ):
        pyscx.from_anndata(
            adata, str(ds.path_for_format(key)), shard_size=200, codec=codec
        )

    scx_fv = FormatVariant("SCX (auto)", "scx_auto", "primary", "scx_runner", {"codec": "auto"})
    return {"ds": ds, "scx_fv": scx_fv, "scx_path": scx_path}


def _mod():
    import importlib

    return importlib.import_module("benchmarks.comprehensive.benchmarks.shuffle_layout")


def _keys(result):
    return set().union(*(r.extra.keys() for r in result.runs))


# --- registration + gating --------------------------------------------------


def test_registered_in_all_benchmarks():
    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    assert "shuffle_layout" in ALL_BENCHMARKS


def test_declared_no_conversion():
    """It consumes the persistent `.scx` fixtures directly. Without this entry
    Phase A would try to satisfy an h5ad conversion dependency it never uses."""
    from benchmarks.comprehensive.scripts.run_parallel import _NO_CONVERSION

    assert "shuffle_layout" in _NO_CONVERSION


def test_supported_formats_declared():
    assert _mod().SUPPORTED_FORMATS == frozenset({"scx_auto"})


def test_skips_unsupported_format(phase1d_env):
    from benchmarks.comprehensive.config import FormatVariant

    unsupported = FormatVariant("Zarr", "zarr_zstd", "primary", "zarr_runner", {})
    assert _mod().run(phase1d_env["ds"], unsupported, n_runs=1) is None


def test_missing_fixture_returns_none_not_raise(phase1d_env):
    """An unconverted fixture is a clean skip: a `FileNotFoundError` here fails
    the whole cohort SLURM job over one absent file."""
    from benchmarks.comprehensive.config import DatasetConfig

    absent = DatasetConfig(
        id="P1DX", name="phase1d_absent", n_obs=100, n_vars=10,
        protocol="synthetic", source="test", approx_h5ad_mb=1,
    )
    assert _mod().run(absent, phase1d_env["scx_fv"], n_runs=1, cold_cache=True) is None


def test_has_memory_and_time_estimates():
    """A benchmark with no sizing arm gets the 15-minute / base-memory default,
    which is wrong by an order of magnitude for something that rewrites the
    whole matrix up to seven times."""
    from benchmarks.comprehensive.config import (
        DATASETS,
        estimate_memory_gb,
        estimate_time_minutes,
    )

    ds = DATASETS["census_500k"]
    assert estimate_time_minutes(ds, "scx_auto", "shuffle_layout") > estimate_time_minutes(
        ds, "scx_auto", "compression"
    )
    assert estimate_memory_gb(ds, "scx_auto", "shuffle_layout") >= estimate_memory_gb(
        ds, "scx_auto", "obs_open"
    )


# --- the four arms ----------------------------------------------------------


def test_shuffle_write_arm_emits_throughput_and_size(phase1d_env):
    res = _mod().run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=False)
    assert res is not None and res.runs
    keys = _keys(res)
    assert "cells_per_sec__shuffle_write" in keys
    assert "peak_rss_mb__shuffle_write" in keys
    assert "output_size_bytes" in keys


def test_size_delta_sweeps_the_codec_variants(phase1d_env):
    res = _mod().run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=False)
    block = res.metadata["size_delta"]
    assert block["applicable"] is True, block.get("reason")
    measured = block["measured"]
    # The fixture builds auto/scx1/zstd; the rest are legitimately absent and
    # must be recorded as skipped-with-a-reason rather than silently dropped.
    assert {"auto", "scx1", "zstd", "shufdelta"} <= set(measured)
    for codec, reason in block["skipped"].items():
        assert reason, f"{codec} skipped with no reason recorded"

    # Geometry control. Without threading the input's `shard_target_rows` into
    # the shuffle, the writer default applies and the ratio compares a different
    # *shard count* — a 30-shard input against a 1-shard output — so it measures
    # re-sharding, not row order. Observed on a 6-shard fixture before the fix.
    for codec, entry in measured.items():
        assert entry["shard_geometry_matched"] is True, codec
        assert entry["n_csr_shards_before"] == entry["n_csr_shards_after"] > 1, codec
        assert entry["file_size_ratio"] > 0, codec
        assert entry["x_size_ratio"] is not None, f"{codec}: X-only ratio unavailable"

    # Each variant must keep its own codec. Left at `auto` the writer
    # re-selects, and every variant produced a byte-identical output — the sweep
    # then measured auto-reselection instead of row order. Observed on pbmc10k:
    # all six landed on exactly 37,703,532 X bytes.
    for codec, entry in measured.items():
        if codec == "auto":
            continue  # `auto` is *supposed* to re-select; that is its arm
        assert entry["codec_flipped"] is False, (
            f"{codec}: codec changed {entry['codec_breakdown_before']} -> "
            f"{entry['codec_breakdown_after']}; the ratio measures re-encoding, "
            f"not row order"
        )
        assert set(entry["codec_breakdown_after"]) == set(
            entry["codec_breakdown_before"]
        ), codec

    keys = _keys(res)
    assert "file_size_ratio__scx1" in keys and "x_size_ratio__shufdelta" in keys


def test_size_delta_distinguishes_a_codec_flip_from_a_compression_loss(phase1d_env):
    """A size ratio alone cannot tell "the permutation cost this codec
    compression" from "the adaptive `auto` heuristic picked a different codec for
    the reordered shard" — and on pbmc10k the whole 1.35x `auto` growth turned
    out to be the latter (`shufdelta` -> `scx1`). Record the per-shard codec
    histogram so the distinction is in the data, not in a reader's inference."""
    res = _mod().run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=False)
    for codec, entry in res.metadata["size_delta"]["measured"].items():
        assert entry["codec_breakdown_before"], codec
        assert entry["codec_breakdown_after"], codec
        assert entry["codec_flipped"] in (True, False), codec


def test_size_delta_records_missing_variants_rather_than_dropping_them(phase1d_env):
    """Silent truncation reads as 'covered everything' when it didn't."""
    res = _mod().run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=False)
    block = res.metadata["size_delta"]
    covered = set(block["measured"]) | set(block["skipped"])
    expected = {k.removeprefix("scx_") for k in _mod()._SIZE_CODEC_KEYS}
    assert covered == expected, "every swept codec must be measured or explained"


def test_shuffle_quality_measures_the_declustering(phase1d_env, monkeypatch):
    """The fixture starts perfectly clustered (200 A / 200 B / 200 C in 100-row
    shards), so every block holds one label and per-block TV is near its maximum.
    After the shuffle each block should look like the corpus."""
    m = _mod()
    monkeypatch.setitem(m.LABEL_SPEC, "phase1d_tiny", "cell_type")
    res = m.run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=False)

    q = res.metadata["shuffle_quality"]
    assert q["applicable"] is True, q.get("reason")
    before = q["label_tv_block"]["before"]
    after = q["label_tv_block"]["after"]
    assert before > 0.5, f"clustered fixture should start poorly mixed, got {before}"
    assert after < before / 2, f"shuffle should mix the blocks: {before} -> {after}"

    # Same geometry control as the size arm: a single block spanning the whole
    # file has TV 0 by construction, which would read as "perfectly mixed".
    assert q["block_geometry_matched"] is True
    assert q["block_rows_before"] == q["block_rows_after"] == 200

    # 30 shards > 8, so the pooled figure is meaningful rather than degenerate.
    assert q["label_tv_sgs8"]["before"] > q["label_tv_sgs8"]["after"]

    keys = _keys(res)
    assert {"label_tv_block__before", "label_tv_block__after"} <= keys
    assert {"label_tv_sgs8__before", "label_tv_sgs8__after"} <= keys


def test_sgs8_is_withheld_on_a_file_with_at_most_8_shards(phase1d_env, monkeypatch):
    """Pooling 8 blocks out of <=8 *is* the corpus, so TV is 0 on any row order.

    Reported 0.0 both before and after on a 6-shard fixture — i.e. "perfectly
    mixed" for a maximally clustered file. Withhold the number instead of
    emitting a confidently wrong one."""
    m = _mod()
    codes = np.repeat([0, 1, 2], 200)  # 600 rows, perfectly clustered
    _tv_block, tv8 = m._label_mixing(codes, block=100, n_levels=3)  # 6 blocks
    assert tv8 is None
    # ...and with more than 8 blocks it is reported.
    _tv_block, tv8_ok = m._label_mixing(codes, block=50, n_levels=3)  # 12 blocks
    assert tv8_ok is not None and tv8_ok > 0


def test_shuffle_quality_not_applicable_without_a_label_column(phase1d_env):
    """`phase1d_tiny` is absent from LABEL_SPEC, so the arm must record why
    rather than invent a column or emit a meaningless zero."""
    res = _mod().run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=False)
    q = res.metadata["shuffle_quality"]
    assert q["applicable"] is False
    assert "LABEL_SPEC" in q["reason"]


def test_train_throughput_arm_emits_both_sides(phase1d_env):
    res = _mod().run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=True)
    keys = _keys(res)
    assert "samples_per_sec__train_unshuffled" in keys
    assert "samples_per_sec__train_shuffled" in keys

    t = res.metadata["train_throughput"]
    assert t["applicable"] is True, t.get("reason")
    assert t["ratio"] is not None
    # The docstring's claim must travel with the number, not live only in a
    # module comment a report reader never sees.
    assert "1.00x" in t["expectation"]

    policies = {r.extra.get("cache_policy") for r in res.runs if "cache_policy" in r.extra}
    assert policies & {"cold_fadvise", "warm"}


# --- failure modes ----------------------------------------------------------


def test_one_broken_arm_does_not_sink_the_others(phase1d_env, monkeypatch):
    """A cohort SLURM job runs several benchmark x dataset tasks; an exception
    escaping `run()` discards all of them."""
    m = _mod()
    monkeypatch.setattr(
        m, "_arm_size_delta", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    res = m.run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=False)
    assert res is not None and res.runs
    assert res.metadata["size_delta"]["applicable"] is False
    assert "boom" in res.metadata["size_delta"]["reason"]


def test_skips_cleanly_when_shuffle_is_absent(phase1d_env, monkeypatch):
    """A pyscx older than 1D must skip, not raise — the gate then reports a
    missing metric, which is the honest signal."""
    import pyscx

    m = _mod()
    monkeypatch.delattr(pyscx, "shuffle", raising=False)
    assert m.run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1) is None


def test_thresholds_yaml_floor_keys_match_frozen_list():
    """Guard the guard, scoped to this benchmark: a floor added to
    thresholds.yaml without extending `_FROZEN_KEYS` below stops being covered
    by the rename check, and a rename then reads as a gate *pass*."""
    import yaml

    from benchmarks.comprehensive.config import PROJECT_ROOT

    spec = yaml.safe_load(
        (PROJECT_ROOT / "benchmarks" / "comprehensive" / "thresholds.yaml").read_text()
    )
    live = {
        f["metric"]
        for f in spec.get("absolute_floors", [])
        if f.get("benchmark") == "shuffle_layout"
    }
    missing = live - set(_FROZEN_KEYS)
    assert not missing, (
        f"thresholds.yaml has shuffle_layout floors not covered by the rename "
        f"guard: {sorted(missing)} — add them to _FROZEN_KEYS"
    )


#: Metric names that `thresholds.yaml` floors key off. Renaming one turns its
#: floor into a silent "missing metric" instead of a regression signal.
#: Currently empty: the capture that would justify a floor has not run, and
#: `thresholds.yaml`'s deferred-floors block records why.
_FROZEN_KEYS: tuple[str, ...] = ()


def test_frozen_keys_still_emitted(phase1d_env):
    if not _FROZEN_KEYS:
        pytest.skip("no shuffle_layout floors yet — see thresholds.yaml deferred floors")
    res = _mod().run(phase1d_env["ds"], phase1d_env["scx_fv"], n_runs=1, cold_cache=True)
    keys = _keys(res)
    for k in _FROZEN_KEYS:
        assert k in keys, f"floor metric {k!r} disappeared — thresholds.yaml would break"
