"""
Per-file obs-open cost microbench (data-load Phase 0).

Obs materialization is a first-class, hidden *startup* cost for every virtual-cell
model: each scans obs up front to build global vocab / one-hot maps. STATE3
hand-rolled a `_H5adFastBacked` reader specifically to dodge ``anndata``'s eager
obs load (60–80 s + tens of GB per file), and its manifests reach **26,453 files**,
so per-file open cost is multiplied by files × workers × ranks. This microbench
measures it directly. Results, including the finding that ~87% of SCX's per-file
cost is open + catalog parse rather than obs reading, are in
``docs/performance.md`` § "Out-of-core loader — cold-cache measurements and the
P-1 premise gate".

Two formats:
  * ``scx_auto``  — ``pyscx.open(path).read_obs([col])`` (matrix-free; no X mmap).
  * ``h5ad_none`` — ``anndata.read_h5ad(path, backed='r')`` then ``.obs[col]``:
    the eager-obs cost STATE3 engineered around, as the comparison baseline.

Scenarios
---------
``open_1file``
    open → read one categorical column → close, on the dataset's own single
    file. The end-to-end per-file cost. Metrics ``obs_open_s``,
    ``obs_open_rss_mb``.
``open_manifest``
    iterate an N-file manifest fixture
    (``DATA_DIR/manifest_<dataset>/manifest.csv``, built by
    ``scripts/prep_manifest_fixture.py``); reports ``obs_open_s_per_file`` +
    total. Skipped cleanly when the manifest fixture is absent. Extrapolates the
    real 26k-file STATE3 manifest (which can't be committed).
``open_only``
    **Open only — obs never touched.** ``pyscx.open(path)`` /
    ``anndata.read_h5ad(path, backed='r')``. On the SCX side this isolates
    ``File::open`` + mmap + header parse + root catalog + full-catalog parse +
    BLAKE3 catalog verify.
``open_only_unverified``
    **SCX only.** ``pyscx.open(path, verify=False)`` → ``ScxReader::open_unchecked``,
    i.e. the same work minus the trailing BLAKE3 catalog verification.

The last two exist so P0.1b's design argument is settled by measurement rather
than assertion. Its stated non-goal is a separate ``ScxObsReader`` that "opens
without mmap-ing X"; the counter-claim is that the mmap is lazy and free, and
that per-file open cost is ``File::open`` + catalog parse + BLAKE3. Two
subtractions on these scenarios decide it, and neither needed a new pyscx API —
``pyscx.open(path, verify=False)`` already exists:

    BLAKE3 catalog verify  =  open_only − open_only_unverified
    obs read proper        =  open_1file − open_only

``open_manifest_r<N>`` (opt-in via ``SCX_BENCH_N_RANKS``)
    P-1(a) × ranks: the same manifest walk in ``N`` spawned processes, reporting
    ``rank_scaling_efficiency``. This is the actual 26k-file × workers × ranks
    startup cost — every model scans obs up front, once per process, and the
    single-process number alone doesn't say whether those scans contend.
"""

from __future__ import annotations

import csv
import gc
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks.comprehensive.cache_control import COLD_FADVISE, WARM, drop_file_cache
from benchmarks.comprehensive.config import DATA_DIR, DatasetConfig, FormatVariant
from benchmarks.comprehensive.multirank import (
    rank_efficiency,
    resolve_n_ranks,
    run_ranks,
    summarize_ranks,
)
from benchmarks.comprehensive.results import BenchmarkResult, require_runs
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto", "h5ad_none"})
"""SCX matrix-free obs open vs the anndata eager-obs baseline."""

# The rank arm pays for (1 + N) manifest walks per run; the manifest walk is the
# expensive scenario, so cap its runs harder than the single-file probes.
_RANK_N_RUNS = 2


def _have_pyscx() -> bool:
    try:
        import pyscx  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Per-file primitives
#
# Split into open-only and open+read so the three SCX scenarios share one code
# path and the subtraction above is between identical open implementations.
# Each returns the obs row count so a caller can assert it read something.
# ---------------------------------------------------------------------------


def _open_scx(path: str, *, verify: bool = True) -> int:
    """Open an SCX file and return ``n_obs`` from the header — obs NOT read.

    ``n_obs_physical``, deliberately, not ``n_obs``/``shape``: the logical count
    decodes the deletion-vector section on any file that has one
    (`pyscx/src/experiment.rs:285`), which would fold a section read into a probe
    whose whole purpose is to isolate open cost. The physical count comes
    straight from the already-parsed header, so it adds no I/O — it exists only
    to keep the open from being optimized away and to let the caller confirm the
    open landed on a real file.
    """
    import pyscx

    exp = pyscx.open(path, verify=verify)
    return int(exp.n_obs_physical)


def _pick_and_read_scx(path: str) -> int:
    """Open an SCX file, read the first usable categorical obs column, return
    the number of obs rows read. Uses ``read_obs([col])`` — matrix-free."""
    import pyscx

    exp = pyscx.open(path)
    keys = list(exp.obs_keys())
    col = keys[0] if keys else None
    if col is None:
        # No obs columns → still measure the bare open (read_obs of nothing).
        df = exp.read_obs([])
        return int(len(df))
    df = exp.read_obs([col])
    return int(len(df))


def _read_codes_scx(path: str) -> int:
    """Open an SCX file and read one categorical column as ``(codes, categories)``
    via ``obs_categorical`` — the data-load 1C accessor.

    Compared against ``_pick_and_read_scx`` (``read_obs([col])``) this isolates the
    saving from skipping the Arrow-IPC → pyarrow → pandas hop and the per-shard
    column concat: same bytes off disk, different assembly. The Phase-0 cold split
    put **99.8%** of per-file cost in obs reading, so this is the term 1C targets.
    """
    import pyscx

    exp = pyscx.open(path)
    keys = list(exp.obs_keys())
    if not keys:
        return int(exp.n_obs_physical)
    # First column that is actually string/categorical — `obs_categorical`
    # rejects numerics, and a dataset's first obs column is often `n_counts`.
    for col in keys:
        try:
            codes, _cats = exp.obs_categorical(col)
        except (ValueError, TypeError):
            continue
        return int(len(codes))
    # Every column numeric: fall back to the bare open so the scenario still
    # reports rather than silently producing no runs.
    return int(exp.n_obs_physical)


def _read_codes_many_scx(path: str) -> int:
    """Read up to 4 categorical columns in ONE shard pass via
    ``obs_categorical_many`` — the claim that N columns cost one projected read per
    shard rather than N.

    This is the shape state3's ``_setup_global_maps`` actually issues (it builds a
    global vocabulary per covariate column), so it is the more representative
    number than the single-column arm.
    """
    import pyscx

    exp = pyscx.open(path)
    # Probe types with `distinct_values(col, limit=1)`, NOT with a full
    # `obs_categorical(col)`: the latter reads the whole column, so the probe
    # would do the work this scenario is trying to measure and then do it again
    # via `_many` — measuring 2x the intended cost. `distinct_values` scans the
    # per-shard dictionary catalog and raises on non-string columns, which is
    # exactly a cheap type probe.
    cols: list[str] = []
    for col in exp.obs_keys():
        try:
            exp.distinct_values(col, limit=1)
        except (ValueError, TypeError):
            continue
        cols.append(col)
        if len(cols) == 4:
            break
    if not cols:
        return int(exp.n_obs_physical)
    out = pyscx.open(path).obs_categorical_many(cols)
    return int(len(out[0][0]))


def _read_codes_many_h5ad(path: str) -> int:
    """anndata analogue: materialize up to 4 obs columns and take their codes."""
    import anndata
    import pandas as pd

    adata = anndata.read_h5ad(path, backed="r")
    try:
        n = 0
        for col in list(adata.obs.columns):
            series = adata.obs[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                cat = series
            elif series.dtype == object:
                cat = series.astype("category")
            else:
                continue
            _ = cat.cat.codes.to_numpy()
            _ = list(cat.cat.categories)
            n += 1
            if n == 4:
                break
        return int(adata.n_obs)
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()


def _codes_many_fn_for(format_key: str) -> Callable[[str], int]:
    if format_key == "scx_auto":
        return _read_codes_many_scx
    if format_key == "h5ad_none":
        return _read_codes_many_h5ad
    raise ValueError(f"obs_open unsupported format {format_key!r}")


def _read_codes_h5ad(path: str) -> int:
    """anndata analogue of ``_read_codes_scx``: materialize one obs column and take
    its ``.cat.codes`` — what a consumer building a global vocab actually does."""
    import anndata
    import pandas as pd

    adata = anndata.read_h5ad(path, backed="r")
    try:
        for col in list(adata.obs.columns):
            series = adata.obs[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                _ = series.cat.codes.to_numpy()
                _ = list(series.cat.categories)
                return int(adata.n_obs)
            if series.dtype == object:
                cat = series.astype("category")
                _ = cat.cat.codes.to_numpy()
                _ = list(cat.cat.categories)
                return int(adata.n_obs)
        return int(adata.n_obs)
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()


def _codes_fn_for(format_key: str) -> Callable[[str], int]:
    if format_key == "scx_auto":
        return _read_codes_scx
    if format_key == "h5ad_none":
        return _read_codes_h5ad
    raise ValueError(f"obs_open unsupported format {format_key!r}")


def _open_h5ad(path: str) -> int:
    """Open an h5ad backed and return ``n_obs`` — obs columns NOT materialized.

    anndata builds the obs *index* on open regardless, so this is not a pure
    container open the way the SCX arm is; it is the honest floor for "the
    cheapest thing anndata can do", which is what the comparison needs.
    """
    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()


def _pick_and_read_h5ad(path: str) -> int:
    """Open an h5ad backed, materialize one obs column (the eager-obs cost),
    return obs row count."""
    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    try:
        cols = list(adata.obs.columns)
        if cols:
            _ = adata.obs[cols[0]].to_numpy()
        return int(adata.n_obs)
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()


def _read_fn_for(format_key: str) -> Callable[[str], int]:
    if format_key == "scx_auto":
        return _pick_and_read_scx
    if format_key == "h5ad_none":
        return _pick_and_read_h5ad
    raise ValueError(f"obs_open unsupported format {format_key!r}")


def _open_only_fn_for(format_key: str) -> Callable[[str], int]:
    if format_key == "scx_auto":
        return _open_scx
    if format_key == "h5ad_none":
        return _open_h5ad
    raise ValueError(f"obs_open unsupported format {format_key!r}")


def _open_unverified_scx(path: str) -> int:
    return _open_scx(path, verify=False)


# ---------------------------------------------------------------------------
# Manifest fixture resolution
# ---------------------------------------------------------------------------


def _manifest_paths(dataset: DatasetConfig, format_key: str) -> list[str] | None:
    """Return the list of per-file paths from the manifest fixture for this
    dataset+format, or ``None`` if the fixture doesn't exist."""
    manifest_dir = DATA_DIR / f"manifest_{dataset.name}"
    manifest_csv = manifest_dir / "manifest.csv"
    if not manifest_csv.exists():
        return None
    ext = ".scx" if format_key == "scx_auto" else ".h5ad"
    paths: list[str] = []
    # Explicit encoding: the default is locale-dependent, and a manifest written
    # on one host can carry non-ASCII paths/labels that fail to decode on another.
    with open(manifest_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row.get("path") or row.get(f"{format_key}_path") or ""
            if p and p.endswith(ext) and Path(p).exists():
                paths.append(p)
    return paths or None


# ---------------------------------------------------------------------------
# Timed helpers
# ---------------------------------------------------------------------------


@dataclass
class _OpenResult:
    wall_s: float
    peak_rss_mb: float
    n_files: int
    n_obs_total: int


def _timed_open(read_fn: Callable[[str], int], paths: list[str]) -> _OpenResult:
    gc.collect()
    n_obs = 0
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        for p in paths:
            n_obs += read_fn(p)
        wall_s = time.perf_counter() - t0
    return _OpenResult(
        wall_s=wall_s, peak_rss_mb=sampler.peak_mb, n_files=len(paths), n_obs_total=n_obs
    )


# ---------------------------------------------------------------------------
# Rank-arm worker — MUST stay module-level so `spawn` can pickle it by reference
# ---------------------------------------------------------------------------


def _rank_manifest_worker(
    rank: int, n_ranks: int, format_key: str, paths: list[str]
) -> dict[str, Any]:
    """One rank's full manifest walk. Runs in a spawned child.

    Every rank walks the **whole** manifest rather than a shard of it: each
    model process builds a global obs vocabulary from every file (report §2.4),
    so the per-rank cost does not shrink with world size. That makes this arm a
    contention measurement, not a work-splitting one.
    """
    read_fn = _read_fn_for(format_key)
    n_obs = 0
    t0 = time.perf_counter()
    for p in paths:
        n_obs += read_fn(p)
    wall_s = time.perf_counter() - t0
    return {
        "n_files": len(paths),
        "n_obs_total": n_obs,
        "manifest_wall_s": round(wall_s, 4),
        "files_per_sec": round(len(paths) / wall_s, 4) if wall_s > 0 else 0.0,
        "s_per_file": round(wall_s / len(paths), 6) if paths else None,
    }


def _run_rank_arm(
    format_key: str, paths: list[str], n_ranks: int
) -> dict[str, Any] | None:
    """1-rank reference + N-rank arm, both through the spawned-child path."""
    worker_args = (format_key, paths)
    single = run_ranks(1, _rank_manifest_worker, worker_args)
    many = run_ranks(n_ranks, _rank_manifest_worker, worker_args)
    one_s = summarize_ranks(single, "files_per_sec")
    many_s = summarize_ranks(many, "files_per_sec")
    if one_s is None or many_s is None:
        logger.error("obs_open rank arm produced no usable rate (1=%s, N=%s)", one_s, many_s)
        return None
    per_rank_median = many_s["per_rank_median"]
    return {
        "n_ranks": n_ranks,
        "n_ranks_reported": many_s["n_ranks_reported"],
        "aggregate_files_per_sec": many_s["aggregate"],
        "per_rank_median_files_per_sec": per_rank_median,
        "one_rank_files_per_sec": one_s["per_rank_median"],
        "s_per_file_at_n_ranks": round(1.0 / per_rank_median, 6) if per_rank_median > 0 else None,
        "rank_scaling_efficiency": rank_efficiency(single, many, "files_per_sec"),
        "max_wall_s": many_s["max_wall_s"],
        "total_peak_rss_mb": many_s["total_peak_rss_mb"],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = True,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Per-file obs-open cost. Drops the page cache before each timed run.
    Returns ``None`` for unsupported formats / missing fixtures."""
    if format_variant.key not in SUPPORTED_FORMATS:
        return None
    if format_variant.key == "scx_auto" and not _have_pyscx():
        logger.warning("Skipping obs_open: pyscx not importable")
        return None

    read_fn = _read_fn_for(format_variant.key)
    open_fn = _open_only_fn_for(format_variant.key)

    def _evict_all(paths: list[str]) -> tuple[str, int]:
        """Evict every path and return ``(policy, n_evicted)``.

        Assigning ``cache_policy`` in a loop keeps only the **last** file's
        status, so 255 evicted files plus one warm one reported as fully cold —
        which would quietly undercut the cold-cache claim the whole benchmark
        exists to make. The label is deliberately conservative: `cold_fadvise`
        only when *every* file was evicted, and `n_evicted`/`n_files` are recorded
        so a partial eviction is diagnosable instead of hidden behind either label.
        """
        n_evicted = sum(1 for p in paths if drop_file_cache(p) == COLD_FADVISE)
        policy = COLD_FADVISE if n_evicted == len(paths) else WARM
        if 0 < n_evicted < len(paths):
            logger.warning(
                "obs_open: only %d/%d files evicted — recording cache_policy=%s",
                n_evicted,
                len(paths),
                policy,
            )
        return policy, n_evicted

    # Resolve the single-file path.
    if format_variant.key == "scx_auto":
        if converted_path is not None and Path(converted_path).exists():
            single = str(converted_path)
        else:
            try:
                p = dataset.path_for_format("scx_auto")
                single = str(p) if p.exists() else None
            except (ValueError, FileNotFoundError):
                single = None
    else:  # h5ad_none
        single = str(dataset.h5ad_path) if dataset.h5ad_path.exists() else None

    if single is None:
        logger.warning(
            "Skipping obs_open for %s/%s — single-file fixture absent",
            format_variant.key,
            dataset.name,
        )
        return None

    n_ranks = resolve_n_ranks()
    result = BenchmarkResult(
        benchmark="obs_open",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={"cold_cache": cold_cache, "n_ranks": n_ranks},
    )
    if Path(single).is_file():
        result.file_size_bytes = Path(single).stat().st_size

    manifest_paths = _manifest_paths(dataset, format_variant.key)

    # (scenario, paths, fn). `open_1file` / `open_manifest` names are FROZEN —
    # four `thresholds.yaml` absolute floors key off
    # `obs_open_s_per_file__open_1file`; renaming turns them into "missing
    # metric" violations rather than a regression signal.
    codes_fn = _codes_fn_for(format_variant.key)
    codes_many_fn = _codes_many_fn_for(format_variant.key)

    scenarios: list[tuple[str, list[str], Callable[[str], int]]] = [
        ("open_1file", [single], read_fn),
        ("open_only", [single], open_fn),
        # data-load 1C: same column, `(codes, categories)` instead of a pandas
        # frame. Paired with `open_1file` so the delta is the assembly cost, not
        # the I/O. Both formats, so the comparison is against what a consumer
        # does today (`obs[col].cat.codes`), not only against our own old path.
        ("obs_codes_1file", [single], codes_fn),
        # Opus 4.6 review: the multi-column case is the compelling one for
        # state3's `_setup_global_maps`, and it is what validates the "one shard
        # pass for N columns" claim rather than merely asserting it.
        ("obs_codes_many_1file", [single], codes_many_fn),
    ]
    if format_variant.key == "scx_auto":
        # No h5ad analogue: anndata has no "skip the integrity check" open.
        scenarios.append(("open_only_unverified", [single], _open_unverified_scx))
    if manifest_paths:
        scenarios.append(("open_manifest", manifest_paths, read_fn))
        scenarios.append(("obs_codes_manifest", manifest_paths, codes_fn))
        result.metadata["manifest_n_files"] = len(manifest_paths)

    for scenario_name, paths, fn in scenarios:
        # Warm code path once (import + pyo3 init), untimed; timed reads are cold.
        try:
            fn(paths[0])
        except Exception as e:  # noqa: BLE001
            logger.error("  warmup failed for %s: %s", scenario_name, e)
            continue

        run_s: list[float] = []
        run_rss: list[float] = []
        run_per_file: list[float] = []
        for i in range(n_runs):
            cache_policy, n_evicted = (WARM, 0)
            if cold_cache:
                cache_policy, n_evicted = _evict_all(paths)
            try:
                out = _timed_open(fn, paths)
            except Exception as e:  # noqa: BLE001
                logger.error("  run %d/%d failed for %s: %s", i + 1, n_runs, scenario_name, e)
                continue
            per_file = out.wall_s / out.n_files if out.n_files else out.wall_s
            result.add_run(
                wall_s=out.wall_s,
                peak_rss_mb=out.peak_rss_mb,
                scenario=scenario_name,
                n_files=out.n_files,
                n_obs_total=out.n_obs_total,
                cache_policy=cache_policy,
                n_evicted=n_evicted,
                **{
                    f"obs_open_s__{scenario_name}": round(out.wall_s, 4),
                    f"obs_open_s_per_file__{scenario_name}": round(per_file, 5),
                    f"obs_open_rss_mb__{scenario_name}": round(out.peak_rss_mb, 1),
                },
            )
            run_s.append(out.wall_s)
            run_rss.append(out.peak_rss_mb)
            run_per_file.append(per_file)
            logger.info(
                "    %s: wall=%.4fs per_file=%.5fs rss=%.1fMB files=%d cache=%s",
                scenario_name,
                out.wall_s,
                per_file,
                out.peak_rss_mb,
                out.n_files,
                cache_policy,
            )

        if run_s:
            result.metadata.setdefault("scenario_summary", {})[scenario_name] = {
                "n_runs": len(run_s),
                "median_obs_open_s": round(statistics.median(run_s), 4),
                "median_obs_open_s_per_file": round(statistics.median(run_per_file), 5),
                "median_obs_open_rss_mb": round(statistics.median(run_rss), 1),
            }
        gc.collect()

    # --- Derived breakdown ------------------------------------------------
    # Recorded in metadata (not as a run metric) because they are subtractions
    # of two medians, not a measured quantity — a floor on a difference of noisy
    # terms would be a worse signal than floors on the terms themselves.
    summary = result.metadata.get("scenario_summary", {})

    def _median(name: str) -> float | None:
        entry = summary.get(name)
        return entry["median_obs_open_s_per_file"] if entry else None

    per_file_total = _median("open_1file")
    per_file_open = _median("open_only")
    per_file_unverified = _median("open_only_unverified")
    breakdown: dict[str, float | None] = {}
    if per_file_total is not None and per_file_open is not None:
        breakdown["obs_read_s_per_file"] = round(per_file_total - per_file_open, 6)
    if per_file_open is not None and per_file_unverified is not None:
        breakdown["catalog_verify_s_per_file"] = round(
            per_file_open - per_file_unverified, 6
        )
        breakdown["open_minus_verify_s_per_file"] = round(per_file_unverified, 6)
    if breakdown:
        result.metadata["open_cost_breakdown"] = breakdown
        logger.info("    open-cost breakdown (s/file): %s", breakdown)

    # --- data-load 1C: obs_categorical vs read_obs(columns=) --------------
    # A ratio of two medians, so metadata rather than a floored metric — same
    # reasoning as the breakdown above. Reported per format so the h5ad arm shows
    # what a consumer pays today, not just our own before/after.
    per_file_codes = _median("obs_codes_1file")
    if per_file_total and per_file_codes:
        result.metadata["obs_codes_vs_read_obs"] = {
            "read_obs_s_per_file": per_file_total,
            "obs_categorical_s_per_file": per_file_codes,
            "speedup": round(per_file_total / per_file_codes, 3),
        }
        logger.info(
            "    1C obs_categorical vs read_obs: %.5fs → %.5fs (%.2f×)",
            per_file_total,
            per_file_codes,
            per_file_total / per_file_codes,
        )

    # --- P-1(a) × ranks ---------------------------------------------------
    if manifest_paths and n_ranks > 1:
        rank_sc = f"open_manifest_r{n_ranks}"
        rank_effs: list[float] = []
        for i in range(_RANK_N_RUNS):
            cache_policy, n_evicted = (WARM, 0)
            if cold_cache:
                cache_policy, n_evicted = _evict_all(manifest_paths)
            try:
                arm = _run_rank_arm(format_variant.key, manifest_paths, n_ranks)
            except Exception as e:  # noqa: BLE001
                logger.error("  rank arm run %d/%d failed: %s", i + 1, _RANK_N_RUNS, e)
                continue
            if arm is None:
                continue
            result.add_run(
                wall_s=arm["max_wall_s"] or 0.0,
                peak_rss_mb=arm["total_peak_rss_mb"] or 0.0,
                scenario=rank_sc,
                n_files=len(manifest_paths),
                n_ranks=n_ranks,
                n_ranks_reported=arm["n_ranks_reported"],
                cache_policy=cache_policy,
                n_evicted=n_evicted,
                **{
                    f"obs_open_s_per_file__{rank_sc}": arm["s_per_file_at_n_ranks"],
                    f"files_per_sec__{rank_sc}": arm["aggregate_files_per_sec"],
                    f"files_per_sec_per_rank__{rank_sc}": arm["per_rank_median_files_per_sec"],
                    f"files_per_sec_1rank__{rank_sc}": arm["one_rank_files_per_sec"],
                    f"rank_scaling_efficiency__{rank_sc}": arm["rank_scaling_efficiency"],
                    f"total_peak_rss_mb__{rank_sc}": arm["total_peak_rss_mb"],
                },
            )
            if arm["rank_scaling_efficiency"] is not None:
                rank_effs.append(arm["rank_scaling_efficiency"])
            logger.info(
                "    %s: aggregate=%.2f files/s per_rank=%.2f 1rank=%.2f eff=%s s/file=%s cache=%s",
                rank_sc,
                arm["aggregate_files_per_sec"],
                arm["per_rank_median_files_per_sec"],
                arm["one_rank_files_per_sec"],
                arm["rank_scaling_efficiency"],
                arm["s_per_file_at_n_ranks"],
                cache_policy,
            )
        if rank_effs:
            result.metadata.setdefault("scenario_summary", {})[rank_sc] = {
                "n_runs": len(rank_effs),
                "n_ranks": n_ranks,
                "median_rank_scaling_efficiency": round(statistics.median(rank_effs), 4),
            }
        gc.collect()

    require_runs(result, single)
    return result
