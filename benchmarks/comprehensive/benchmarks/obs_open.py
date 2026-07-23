"""
Per-file obs-open cost microbench (data-load Phase 0).

The report (§0/§2.3) identifies obs materialization as a first-class, hidden
scaling cost: STATE3 hand-rolled ``_H5adFastBacked`` specifically to dodge
``anndata``'s eager obs load (60–80 s + tens of GB per file), yet the ``.scx``
path today still routes obs through ``read_obs``/``to_anndata``. At 26,453 files
× workers × ranks the per-file open cost dominates. This microbench measures it
directly, so Phase 1A's matrix-free ``obs_categorical`` reader has a before/after
number.

Two formats:
  * ``scx_auto``  — ``pyscx.open(path).read_obs([col])`` (matrix-free; no X mmap).
  * ``h5ad_none`` — ``anndata.read_h5ad(path, backed='r')`` then ``.obs[col]``:
    the eager-obs cost STATE3 engineered around, as the comparison baseline.

Scenarios:
  * ``open_1file``  — open → read one categorical column → close, on the dataset's
    own single file. Metrics ``obs_open_s``, ``obs_open_rss_mb``.
  * ``open_manifest`` — iterate an N-file manifest fixture
    (``DATA_DIR/manifest_<dataset>/manifest.csv``, built by
    ``scripts/prep_manifest_fixture.py``); reports ``obs_open_s_per_file`` +
    total. Skipped cleanly when the manifest fixture is absent. Extrapolates the
    real 26k-file STATE3 manifest (which can't be committed).
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

from benchmarks.comprehensive.cache_control import drop_file_cache
from benchmarks.comprehensive.config import DATA_DIR, DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto", "h5ad_none"})
"""SCX matrix-free obs open vs the anndata eager-obs baseline."""


def _have_pyscx() -> bool:
    try:
        import pyscx  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Per-file obs-open primitives (one open → read one categorical column → close)
# ---------------------------------------------------------------------------


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
    with open(manifest_csv) as f:
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

    result = BenchmarkResult(
        benchmark="obs_open",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={"cold_cache": cold_cache},
    )
    if Path(single).is_file():
        result.file_size_bytes = Path(single).stat().st_size

    manifest_paths = _manifest_paths(dataset, format_variant.key)

    scenarios: list[tuple[str, list[str]]] = [("open_1file", [single])]
    if manifest_paths:
        scenarios.append(("open_manifest", manifest_paths))
        result.metadata["manifest_n_files"] = len(manifest_paths)

    for scenario_name, paths in scenarios:
        # Warm code path once (import + pyo3 init), untimed; timed reads are cold.
        try:
            read_fn(paths[0])
        except Exception as e:  # noqa: BLE001
            logger.error("  warmup failed for %s: %s", scenario_name, e)
            continue

        run_s: list[float] = []
        run_rss: list[float] = []
        run_per_file: list[float] = []
        for i in range(n_runs):
            cache_policy = "warm"
            if cold_cache:
                for p in paths:
                    cache_policy = drop_file_cache(p)
            try:
                out = _timed_open(read_fn, paths)
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

    return result
