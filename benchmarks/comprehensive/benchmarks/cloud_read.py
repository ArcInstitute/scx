"""
Cloud full-read benchmark.

Cross-format: runs ``runner.read_cloud(url)`` for every format that
declares the ``"cloud_read"`` capability. Returns ``None`` for runners
without that capability (silent skip).

Runners that *declare* the capability but fail to implement it propagate
``NotImplementedError`` loudly, per the base-class contract.

After the timed loop, an untimed verification pass for SCX formats pulls
the cloud fixture once more and compares the materialised matrix against
the local ``converted_path``. Result counts land in every run's
``extra`` so ``thresholds.yaml`` can floor them — addresses §3.6 of
``2026-04-29_SCX-BENCH-REVIEW.md``.
"""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.comprehensive.cloud_fixtures import (
    ensure_cloud_fixture,
    require_gcp_credentials,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCS_TEST_BUCKET,
    N_WARMUP_RUNS,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

# Skip full csr_equal above this row count — materialising both matrices
# at census-tier scale doubles peak RSS during verification. Above the
# threshold we fall back to row-sum equality (cheap, catches layout /
# codec corruption). pbmc3k (2.7K) and tabula_sapiens_100k (100K) stay
# in the full-equality path; census_10m (10M) takes the row-sum path.
_FULL_CSR_EQUAL_MAX_OBS = 1_000_000


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    provider: str = "gcs",
) -> BenchmarkResult | None:
    if provider != "gcs":
        raise ValueError(
            f"Only 'gcs' provider is supported in Phase 5 (got {provider!r})"
        )

    runner = make_runner(format_variant)
    if "cloud_read" not in runner.capabilities:
        logger.info(
            "Skipping cloud_read for %s — runner does not declare cloud_read",
            format_variant.key,
        )
        return None

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted {format_variant.key} file for {dataset.name}. "
            f"Run conversion first (--formats {format_variant.key})."
        )

    require_gcp_credentials()
    cloud_url = ensure_cloud_fixture(
        dataset, format_variant, Path(converted_path), provider=provider,
    )

    result = BenchmarkResult(
        benchmark="cloud_read",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "n_warmup": N_WARMUP_RUNS,
            "cold_cache": cold_cache,
        },
    )
    result.file_size_bytes = runner.file_size(Path(converted_path))

    for i in range(N_WARMUP_RUNS):
        logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
        runner.read_cloud(cloud_url)
        gc.collect()

    for i in range(n_runs):
        if cold_cache:
            runner._drop_caches()
        gc.collect()

        logger.info("cloud_read run %d/%d ← %s", i + 1, n_runs, cloud_url)
        timing = runner.read_cloud(cloud_url)
        result.add_run(
            wall_s=timing.wall_s,
            user_s=timing.user_s,
            sys_s=timing.sys_s,
            peak_rss_mb=timing.peak_rss_mb,
            **(timing.extra or {}),
        )
        logger.info("  wall=%.3fs  rss=%.1fMB", timing.wall_s, timing.peak_rss_mb)

    correctness_extra = _verify_cloud_matches_local(
        cloud_url=cloud_url,
        local_path=Path(converted_path),
        format_variant=format_variant,
        dataset=dataset,
    )
    if correctness_extra:
        for run_record in result.runs:
            run_record.extra.update(correctness_extra)
        result.metadata["correctness"] = correctness_extra

    logger.info(
        "cloud_read complete: %s / %s — median %.3fs",
        format_variant.key,
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result


def _verify_cloud_matches_local(
    cloud_url: str,
    local_path: Path,
    format_variant: FormatVariant,
    dataset: DatasetConfig,
) -> dict[str, Any]:
    """Pull the cloud fixture once more (untimed) and compare the
    materialised matrix against the local source.

    Returns a dict of int metrics for ``runs[].extra``. Empty dict when
    the format is not SCX (other backends would need their own
    local-source comparison logic). Sparse-by-design: missing keys mean
    "the check wasn't run", not "0 failures" — mirrors the convention
    in ``correctness.py``.
    """
    if not format_variant.key.startswith("scx"):
        return {}

    try:
        import numpy as np
        import pyscx
    except ImportError as exc:
        logger.warning(
            "cloud_read correctness check skipped — pyscx import failed: %s",
            exc,
        )
        return {"correctness_n_skipped": 1}

    logger.info("cloud_read correctness check: pulling fresh copy for compare")
    tmp = Path(tempfile.mkdtemp(prefix="scx_cloud_read_verify_"))
    try:
        pulled = tmp / "pulled.scx"
        pyscx.pull(cloud_url, str(pulled))
        adata_cloud = pyscx.open(str(pulled)).to_anndata()
        adata_local = pyscx.open(str(local_path)).to_anndata()

        X_cloud = adata_cloud.X
        X_local = adata_local.X

        shape_match = bool(X_cloud.shape == X_local.shape)
        nnz_match = bool(int(X_cloud.nnz) == int(X_local.nnz))
        dims_match = bool(
            adata_cloud.n_obs == adata_local.n_obs
            and adata_cloud.n_vars == adata_local.n_vars
        )

        n_obs = int(adata_local.n_obs)
        csr_skipped = n_obs > _FULL_CSR_EQUAL_MAX_OBS
        if csr_skipped:
            # Row-sum + column-sum equality — O(nnz) memory, no double
            # dense materialisation. Catches layout / codec corruption
            # that would shift values across rows or zero them out;
            # adding the axis=0 sums also catches intra-row index
            # permutations that preserve the per-row total.
            row_sum_cloud = np.asarray(X_cloud.sum(axis=1)).ravel()
            row_sum_local = np.asarray(X_local.sum(axis=1)).ravel()
            col_sum_cloud = np.asarray(X_cloud.sum(axis=0)).ravel()
            col_sum_local = np.asarray(X_local.sum(axis=0)).ravel()
            csr_equal_result = bool(
                shape_match
                and np.allclose(row_sum_cloud, row_sum_local, rtol=1e-6, atol=0)
                and np.allclose(col_sum_cloud, col_sum_local, rtol=1e-6, atol=0)
            )
        else:
            from benchmarks.comprehensive.scripts.validation_helpers import (
                csr_equal,
            )
            csr_equal_result = bool(
                shape_match and csr_equal(X_cloud, X_local, rtol=1e-6)
            )

        checks = {
            "shape_match": shape_match,
            "nnz_match": nnz_match,
            "csr_equal": csr_equal_result,
            "obs_var_dims_match": dims_match,
        }
        n_passed = sum(1 for v in checks.values() if v)
        n_failed = sum(1 for v in checks.values() if not v)
        n_total = len(checks)
        n_skipped = 1 if csr_skipped else 0

        out: dict[str, Any] = {
            "correctness_n_passed": n_passed,
            "correctness_n_failed": n_failed,
            "correctness_n_skipped": n_skipped,
            "correctness_n_total": n_total,
            "correctness_passed_int": 1 if n_failed == 0 else 0,
            "shape_match_int": 1 if shape_match else 0,
            "nnz_match_int": 1 if nnz_match else 0,
            "obs_var_dims_match_int": 1 if dims_match else 0,
        }
        if csr_skipped:
            out["csr_equal_rowsum_int"] = 1 if csr_equal_result else 0
        else:
            out["csr_equal_int"] = 1 if csr_equal_result else 0

        logger.info(
            "cloud_read correctness: %d/%d passed (failed=%d, skipped_full_csr=%s)",
            n_passed,
            n_total,
            n_failed,
            csr_skipped,
        )
        if n_failed:
            logger.warning(
                "cloud_read correctness FAILED for %s/%s: %s",
                format_variant.key,
                dataset.name,
                {k: v for k, v in checks.items() if not v},
            )
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
