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
# S=512 arm (P-1(b))
# ---------------------------------------------------------------------------


def test_cellset_gather_emits_both_set_sizes(phase0_env):
    """S=64 and S=512 arms both record, and S=512 packs 512-stride sets."""
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert res is not None
    scenarios = {r.extra["scenario"] for r in res.runs}
    assert {
        "gather_random",
        "gather_grouped",
        "gather_random_s512",
        "gather_grouped_s512",
    } <= scenarios
    by_scenario = {r.extra["scenario"]: r.extra for r in res.runs}
    assert by_scenario["gather_random"]["set_size"] == 64
    assert by_scenario["gather_random_s512"]["set_size"] == 512
    assert res.metadata["set_sizes"]["gather_grouped_s512"] == 512


def test_sets_per_batch_holds_cells_per_batch(phase0_env):
    """Cells/batch is held ≈ ML_BATCH_SIZE across set sizes, so the S=64 and
    S=512 arms differ in granularity rather than in batch volume."""
    import importlib

    from benchmarks.comprehensive.config import ML_BATCH_SIZE

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    for set_size in (64, 512):
        spb = m._sets_per_batch(set_size)
        assert spb >= 1
        assert spb * set_size == ML_BATCH_SIZE


def test_s512_plan_packs_512_stride_offsets(phase0_env):
    """The emitted plan really carries 512-cell sets, not 64-cell ones."""
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    spb = m._sets_per_batch(512)
    plans = list(m._random_plans(600, 2, 512, spb, seed=0))
    assert len(plans) == 2
    for _file_ids, rows, _roles, offsets in plans:
        assert np.all(np.diff(offsets) == 512)
        assert rows.size == 512 * spb


# ---------------------------------------------------------------------------
# Frozen floor metric keys — the rename guard
# ---------------------------------------------------------------------------

# Metric keys with a live `absolute_floors` entry in thresholds.yaml. Renaming a
# scenario silently converts each of these into a "missing metric" gate
# violation instead of a regression signal, which reads as a *pass* on a chart
# and a failure in the report. Both directions are wrong, so pin the names.
_FROZEN_CELLSET_KEYS = (
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
)
_FROZEN_OBS_OPEN_KEYS = ("obs_open_s_per_file__open_1file",)

# `ooc_loader` carries 14 of the 33 Phase-0 floors — more than either other arm —
# and was originally left out of this guard, so those metric names could have been
# renamed into silent missing-metric gate passes.
_FROZEN_OOC_LOADER_KEYS = (
    "samples_per_sec__raw",
    "samples_per_sec__hvg_norm",
)

# Floored, but only emitted when the rank arm is enabled (SCX_BENCH_N_RANKS > 1),
# so they can't be asserted against a default-config run the way the always-on
# keys above can. Listed separately so the guard-the-guard test below still
# accounts for every Phase-0 floor in thresholds.yaml.
_FROZEN_RANK_KEYS = ("rank_scaling_efficiency__gather_random_r4",)


def test_frozen_floor_keys_still_emitted(phase0_env):
    import importlib

    cg = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    res = cg.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    keys = set().union(*(r.extra.keys() for r in res.runs))
    for k in _FROZEN_CELLSET_KEYS:
        assert k in keys, f"floor metric {k!r} disappeared — thresholds.yaml would break"

    oo = importlib.import_module("benchmarks.comprehensive.benchmarks.obs_open")
    for fv in (phase0_env["scx_fv"], phase0_env["h5_fv"]):
        r = oo.run(phase0_env["ds"], fv, n_runs=1, cold_cache=True)
        okeys = set().union(*(x.extra.keys() for x in r.runs))
        for k in _FROZEN_OBS_OPEN_KEYS:
            assert k in okeys, f"floor metric {k!r} disappeared for {fv.key}"

    # `ooc_loader` carries 14 of the 33 floors — more than either other arm — and
    # was covered only by the yaml-sync guard, not by an emission check.
    ol = importlib.import_module("benchmarks.comprehensive.benchmarks.ooc_loader")
    res_ol = ol.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert res_ol is not None and res_ol.runs
    olkeys = set().union(*(r.extra.keys() for r in res_ol.runs))
    for k in _FROZEN_OOC_LOADER_KEYS:
        assert k in olkeys, f"floor metric {k!r} disappeared — thresholds.yaml would break"


def test_thresholds_yaml_floor_keys_match_frozen_list():
    """Guard the guard: if a new Phase-0 floor is added to thresholds.yaml, the
    frozen list above must grow with it, or the rename guard silently stops
    covering the new one."""
    import yaml

    from benchmarks.comprehensive.config import PROJECT_ROOT

    spec = yaml.safe_load(
        (PROJECT_ROOT / "benchmarks" / "comprehensive" / "thresholds.yaml").read_text()
    )
    floors = spec.get("absolute_floors", [])
    live = {
        f["metric"]
        for f in floors
        if f.get("benchmark") in ("cellset_gather", "obs_open", "ooc_loader")
    }
    frozen = (
        set(_FROZEN_CELLSET_KEYS)
        | set(_FROZEN_OBS_OPEN_KEYS)
        | set(_FROZEN_RANK_KEYS)
        | set(_FROZEN_OOC_LOADER_KEYS)
    )
    missing = live - frozen
    assert not missing, (
        f"thresholds.yaml has Phase-0 floors not covered by the rename guard: "
        f"{sorted(missing)} — add them to _FROZEN_* above"
    )


def test_rank_arm_emits_its_floored_key(phase0_env, monkeypatch: pytest.MonkeyPatch):
    """`rank_scaling_efficiency__gather_random_r<N>` has a floor, so the arm must
    actually emit it — under the same rank count the floors were captured at."""
    import importlib

    monkeypatch.setenv("SCX_BENCH_N_RANKS", "4")
    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    keys = set().union(*(r.extra.keys() for r in res.runs))
    for k in _FROZEN_RANK_KEYS:
        assert k in keys, f"floored rank metric {k!r} not emitted at N=4"


# ---------------------------------------------------------------------------
# obs_open open-cost breakdown (P-1(a) sharpening)
# ---------------------------------------------------------------------------


def test_obs_open_breakdown_scenarios(phase0_env):
    """`open_only` (both formats) + `open_only_unverified` (SCX only), and the
    derived per-file breakdown."""
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.obs_open")

    scx = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    scx_scenarios = {r.extra["scenario"] for r in scx.runs}
    assert {"open_1file", "open_only", "open_only_unverified"} <= scx_scenarios
    bd = scx.metadata["open_cost_breakdown"]
    assert "obs_read_s_per_file" in bd
    assert "catalog_verify_s_per_file" in bd

    h5 = m.run(phase0_env["ds"], phase0_env["h5_fv"], n_runs=1, cold_cache=True)
    h5_scenarios = {r.extra["scenario"] for r in h5.runs}
    assert "open_only" in h5_scenarios
    # anndata has no "skip the integrity check" open — the arm must not appear,
    # or the h5ad column of the breakdown table would be comparing nothing.
    assert "open_only_unverified" not in h5_scenarios
    assert "catalog_verify_s_per_file" not in h5.metadata.get("open_cost_breakdown", {})


def test_open_scx_does_not_read_obs(phase0_env, monkeypatch: pytest.MonkeyPatch):
    """`open_only` must not touch obs.

    If it did, `open_1file − open_only` would be a difference of two identical
    workloads — i.e. ~0 — and the breakdown would report "obs reads cost
    nothing" while measuring nothing at all. That failure is invisible in the
    numbers, so it needs its own assertion.

    Uses an attribute spy rather than the reader's `ReaderDebugCounts`: those
    counters increment only under `cfg(debug_assertions)`
    (`scx-format-io/src/reader.rs`), so on the release `.so` every capture
    actually uses they are compiled out and would prove nothing.
    """
    import importlib

    import pyscx

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.obs_open")
    path = str(phase0_env["scx_path"])
    real_open = pyscx.open
    seen: list[str] = []

    class _Spy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            seen.append(name)
            return getattr(self._inner, name)

    monkeypatch.setattr(
        pyscx, "open", lambda p, verify=True: _Spy(real_open(p, verify=verify))
    )

    m._open_scx(path)
    m._open_scx(path, verify=False)
    assert "read_obs" not in seen
    assert "obs_keys" not in seen
    assert "n_obs_physical" in seen, "the open probe read nothing — is it a no-op?"

    # Anti-tautology: prove the spy would have caught an obs read. Without this
    # the assertions above pass just as happily against a broken spy.
    seen.clear()
    m._pick_and_read_scx(path)
    assert "read_obs" in seen


# ---------------------------------------------------------------------------
# multirank helper + the N-rank arms (P-1(c))
# ---------------------------------------------------------------------------


def test_resolve_n_ranks_defaults_off(monkeypatch: pytest.MonkeyPatch):
    from benchmarks.comprehensive import multirank

    monkeypatch.delenv(multirank.N_RANKS_ENV, raising=False)
    # Off by default: turning the arm on unconditionally would multiply the wall
    # time of every existing cellset_gather / obs_open run.
    assert multirank.resolve_n_ranks() == 1
    monkeypatch.setenv(multirank.N_RANKS_ENV, "4")
    assert multirank.resolve_n_ranks() == 4
    monkeypatch.setenv(multirank.N_RANKS_ENV, "not-a-number")
    assert multirank.resolve_n_ranks(default=3) == 3
    monkeypatch.setenv(multirank.N_RANKS_ENV, "0")
    assert multirank.resolve_n_ranks() == 1


def test_summarize_and_efficiency_pure():
    from benchmarks.comprehensive.multirank import rank_efficiency, summarize_ranks

    assert summarize_ranks([], "rate") is None
    assert summarize_ranks([{"rank": 0, "error": "boom"}], "rate") is None

    one = [{"rank": 0, "rate": 100.0, "wall_s": 1.0, "peak_rss_mb": 10.0}]
    many = [
        {"rank": 0, "rate": 50.0, "wall_s": 2.0, "peak_rss_mb": 10.0},
        {"rank": 1, "rate": 50.0, "wall_s": 2.2, "peak_rss_mb": 12.0},
    ]
    s = summarize_ranks(many, "rate")
    assert s["aggregate"] == 100.0  # node aggregate = sum, not mean
    assert s["per_rank_median"] == 50.0
    assert s["max_wall_s"] == 2.2  # slowest rank sets a synchronous step
    assert s["total_peak_rss_mb"] == 22.0
    # Perfectly serialising 2 ranks → efficiency 1/2.
    assert rank_efficiency(one, many, "rate") == 0.5
    # Perfect scaling → 1.0.
    perfect = [dict(r, rate=100.0) for r in many]
    assert rank_efficiency(one, perfect, "rate") == 1.0


def test_cellset_gather_rank_arm(phase0_env, monkeypatch: pytest.MonkeyPatch):
    import importlib

    monkeypatch.setenv("SCX_BENCH_N_RANKS", "2")
    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert res is not None
    assert res.metadata["n_ranks"] == 2
    rank_runs = [r for r in res.runs if r.extra["scenario"] == "gather_random_r2"]
    assert rank_runs, "rank arm recorded no runs"
    e = rank_runs[0].extra
    assert e["n_ranks_reported"] == 2, "a rank failed to report — see the error log"
    assert e["cellsets_per_sec__gather_random_r2"] > 0
    assert e["cellsets_per_sec_1rank__gather_random_r2"] > 0
    assert e["rank_scaling_efficiency__gather_random_r2"] is not None


def test_obs_open_rank_arm_needs_manifest(phase0_env, monkeypatch: pytest.MonkeyPatch):
    """No manifest fixture → no rank arm, and no crash."""
    import importlib

    monkeypatch.setenv("SCX_BENCH_N_RANKS", "2")
    m = importlib.import_module("benchmarks.comprehensive.benchmarks.obs_open")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert res is not None
    assert not any(r.extra["scenario"].startswith("open_manifest") for r in res.runs)


def test_rank_arm_skipped_at_one_rank(phase0_env, monkeypatch: pytest.MonkeyPatch):
    import importlib

    monkeypatch.setenv("SCX_BENCH_N_RANKS", "1")
    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)
    assert not any("_r1" in r.extra["scenario"] for r in res.runs)


# ---------------------------------------------------------------------------
# A universally-failing capture must fail the job, not record an empty result
# ---------------------------------------------------------------------------


def test_zero_runs_raises(phase0_env, monkeypatch: pytest.MonkeyPatch):
    """Regression for the 2026-07-29 silent-empty-capture incident.

    A stale `pyscx.pth` pointing at a deleted worktree let the repo-root
    `pyscx/` directory import as an empty namespace package. Every scenario
    raised `module 'pyscx' has no attribute 'open'`, each was swallowed by the
    per-scenario `continue`, and the wave reported "9 succeeded, 0 failed" while
    writing results with zero runs. Floors captured from that would have been
    silently absent rather than wrong — the worse of the two failure modes.
    """
    import importlib

    import pyscx

    def _broken(*args, **kwargs):
        raise AttributeError("module 'pyscx' has no attribute 'open'")

    # Model "pyscx imports but is non-functional" — the actual symptom. Patching
    # only `pyscx.open` is NOT enough: the random-plan scenarios don't need it,
    # so they still record runs and the guard correctly stays quiet.
    monkeypatch.setattr(pyscx, "SparseCellSetDataset", _broken)
    cg = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    with pytest.raises(RuntimeError, match="no runs recorded"):
        cg.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)

    monkeypatch.setattr(pyscx, "open", _broken)
    oo = importlib.import_module("benchmarks.comprehensive.benchmarks.obs_open")
    with pytest.raises(RuntimeError, match="no runs recorded"):
        oo.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)

    # `ooc_loader` owns 14 of the 33 floors and was the arm left unguarded.
    # Its `_HAS_PYSCX` probe is a bare `import pyscx`, which still succeeds under
    # the namespace-package shadow that caused the original incident — so the
    # guard, not the probe, is what has to catch this.
    ol = importlib.import_module("benchmarks.comprehensive.benchmarks.ooc_loader")
    monkeypatch.setattr(ol, "_run_scx_epoch", _broken)
    with pytest.raises(RuntimeError, match="no runs recorded"):
        ol.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=True)


def test_missing_fixture_returns_none_not_raise(phase0_env):
    """A fixture that was never converted is a clean skip on all three arms.

    `cellset_gather` used to raise `FileNotFoundError`, which fails the whole
    cohort SLURM job (several benchmark x dataset tasks share one job) over a
    single absent file — and contradicted its own docstring. The absence still
    surfaces: the triple's thresholds.yaml floors report a missing metric at gate
    time.
    """
    import importlib

    from benchmarks.comprehensive.config import DatasetConfig

    absent = DatasetConfig(
        id="P0X", name="phase0_absent", n_obs=100, n_vars=10,
        protocol="synthetic", source="test", approx_h5ad_mb=1,
    )
    for mod in ("cellset_gather", "obs_open", "ooc_loader"):
        m = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{mod}")
        assert m.run(absent, phase0_env["scx_fv"], n_runs=1, cold_cache=True) is None, mod


def test_resolve_groups_drops_nan_codes(phase0_env, monkeypatch: pytest.MonkeyPatch):
    """pandas encodes missing obs values as code -1.

    Left in, every NaN row collapses into one spurious "group" of unrelated cells,
    which the grouped arm would then measure as covariate locality that does not
    exist.
    """
    import importlib

    import pandas as pd

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")

    class _FakeExp:
        def obs_keys(self):
            return ["mixed"]

        def read_obs(self, cols):
            # 6 rows: 3 real values, 3 NaN -> codes [0,1,2,-1,-1,-1]
            return pd.DataFrame({"mixed": pd.Categorical(["a", "b", "c", None, None, None])})

    import pyscx

    monkeypatch.setattr(pyscx, "open", lambda *a, **k: _FakeExp())
    groups = m._resolve_groups("ignored.scx", 6)
    all_rows = np.concatenate(groups) if groups else np.array([], dtype=np.uint64)
    assert set(all_rows.tolist()) == {0, 1, 2}, f"NaN rows leaked into groups: {all_rows}"
    assert all(g.size > 0 for g in groups)


def test_resolve_groups_empty_dataset(phase0_env):
    """n_obs == 0 must yield no groups rather than crash the grouped plan
    generator on `rng.integers(0, 0)`."""
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    assert m._resolve_groups("/nonexistent.scx", 0) == []


def test_group_fallback_has_no_empty_groups(phase0_env):
    """`rng.choice` on an empty group raises and kills the grouped scenario.

    The index-bucket fallback forced `max(2, ...)` buckets, so any file with
    `n_obs < _LOCALITY_GROUP_SIZE` (every small fixture) got one populated bucket
    and one empty one.
    """
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")
    for n_obs in (1, 2, 600, m._LOCALITY_GROUP_SIZE, m._LOCALITY_GROUP_SIZE + 1):
        groups = m._resolve_groups("/nonexistent-so-the-fallback-runs.scx", n_obs)
        assert groups, f"n_obs={n_obs} produced no groups"
        assert all(g.size > 0 for g in groups), f"n_obs={n_obs} produced an empty group"
        assert sum(int(g.size) for g in groups) == n_obs, f"n_obs={n_obs} lost/duplicated rows"


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


def test_hot_control_cold_tail_arm_runs_to_its_summary(phase0_env, monkeypatch):
    """The W10 arm must be exercised on a fixture where it does NOT skip.

    ⚠️ This test exists because its absence cost a capture. At the shipped shape
    a batch draws `_HOT_TAIL_SETS * _HOT_SET_SIZE` rows, so on every fixture
    small enough to run in a test the "cold" tails collide with each other, the
    arm records `applicable: False` and returns before its summary. The suite
    therefore ran `cellset_gather.run` happily while that summary block held a
    `NameError` (`median` instead of `statistics.median`), and it surfaced seven
    minutes into a SLURM job, on tabula, after twelve rounds of pbmc3k had
    already been spent.

    Shrinking the shape is the only way to reach that code from a test, and it
    is safe here precisely because nothing is timed: the assertion is that the
    arm completes and reports, not how fast it did.

    ⚠️ Driven off `phase0_env`, not off the registered pbmc3k fixture. The first
    version used `DATASETS["pbmc3k"]` and **skipped** in the full-suite run while
    passing in isolation — `phase0_env` pops the config module from
    `sys.modules` so it re-resolves against a `tmp_path` `SCX_DATA_DIR`, and the
    module stays loaded with that path after the fixture's env is restored. A
    guard that silently skips is the failure this test was added to prevent,
    wearing its own name.
    """
    import importlib

    m = importlib.import_module("benchmarks.comprehensive.benchmarks.cellset_gather")

    # `phase0_env`'s fixture is 600 rows; shrink the shape so its cold tails do
    # not collide with each other.
    monkeypatch.setattr(m, "_HOT_CONTROL_SETS", 2)
    monkeypatch.setattr(m, "_HOT_TAIL_SETS", 4)
    monkeypatch.setattr(m, "_HOT_SET_SIZE", 8)
    monkeypatch.setattr(m, "_HOT_N_BATCHES", 2)
    monkeypatch.setattr(m, "_HOT_CONTROL_POOL", 128)

    premises = m._hot_cold_premises(phase0_env["ds"].n_obs)
    assert premises["tail_overlap_fraction"] <= 0.25, (
        "premise: at the shrunk shape this fixture CAN hold a cold tail apart, "
        f"or the test skips the branch it exists to cover: {premises}"
    )

    res = m.run(phase0_env["ds"], phase0_env["scx_fv"], n_runs=1, cold_cache=False)
    meta = res.metadata.get("hot_control_cold_tail")
    assert meta is not None and meta.get("applicable") is True, meta
    # The summary block — the code the NameError lived in.
    assert "median_cellsets_per_sec" in meta, meta
    assert "control_set_fraction" in meta, meta

    runs = [r for r in res.runs if r.extra.get("scenario") == "gather_hot_control_cold_tail"]
    assert runs, "the arm emitted no runs"
    for key in (
        "cellsets_per_sec__gather_hot_control_cold_tail",
        "row_group_hit_rate__gather_hot_control_cold_tail",
        "reuse_admissions__gather_hot_control_cold_tail",
        "rejected_group_bytes__gather_hot_control_cold_tail",
    ):
        assert key in runs[0].extra, (key, sorted(runs[0].extra))
