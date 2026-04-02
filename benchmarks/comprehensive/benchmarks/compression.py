"""
§3.1 Storage Efficiency (Compression) benchmark.

Measures on-disk file size after converting an h5ad dataset to each format
variant.  Reports file size, compression ratio relative to the source h5ad,
and bits per non-zero value.

Only a single conversion is performed (no timing repetitions) because
this benchmark measures size, not speed.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner


log = logging.getLogger(__name__)


def _estimate_nnz(h5ad_path: Path) -> int:
    """Return the number of non-zero entries in the h5ad X matrix."""
    import anndata
    import scipy.sparse as sp

    adata = anndata.read_h5ad(h5ad_path, backed="r")
    X = adata.X
    if sp.issparse(X):
        nnz = X.nnz
    else:
        # Dense matrix — count explicit non-zeros
        import numpy as np
        nnz = int(np.count_nonzero(X))
    adata.file.close()
    return nnz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Run the §3.1 compression benchmark for a single format variant.

    Parameters
    ----------
    dataset : DatasetConfig
        The dataset to benchmark (must have a valid ``h5ad_path``).
    format_variant : FormatVariant
        The target format to convert to and measure.
    n_runs : int
        Ignored — compression is a single-shot size measurement.
    cold_cache : bool
        Ignored — not applicable for size measurement.

    Returns
    -------
    BenchmarkResult
        Result with ``benchmark="compression"``, file size, compression ratio,
        and bits-per-NNZ in ``metadata``.
    """
    h5ad_path = dataset.h5ad_path
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Source h5ad not found: {h5ad_path}")

    source_bytes = h5ad_path.stat().st_size
    runner = make_runner(format_variant)

    log.info(
        "Compression benchmark: dataset=%s format=%s source=%.1f MB",
        dataset.name,
        format_variant.key,
        source_bytes / 1e6,
    )

    # Use pre-converted file if available, otherwise convert to temp dir.
    convert_result = None
    if converted_path is not None and Path(converted_path).exists():
        converted_bytes = runner.file_size(converted_path)
    else:
        with tempfile.TemporaryDirectory(prefix="scx_bench_comp_") as tmp:
            out_path = Path(tmp) / f"converted.{format_variant.key}"
            convert_result = runner.convert_from_h5ad(h5ad_path, out_path)
            converted_bytes = convert_result.output_size_bytes

    # Estimate NNZ from the source h5ad.
    nnz = _estimate_nnz(h5ad_path)

    # Derived metrics.
    compression_ratio = source_bytes / converted_bytes if converted_bytes > 0 else float("inf")
    bits_per_nnz = (converted_bytes * 8) / nnz if nnz > 0 else float("inf")

    log.info(
        "  -> converted=%.1f MB  ratio=%.2fx  bits/nnz=%.2f  nnz=%d",
        converted_bytes / 1e6,
        compression_ratio,
        bits_per_nnz,
        nnz,
    )

    result = BenchmarkResult(
        benchmark="compression",
        format=format_variant.key,
        dataset=dataset.name,
        file_size_bytes=converted_bytes,
        metadata={
            "source_h5ad_bytes": source_bytes,
            "compression_ratio": round(compression_ratio, 4),
            "bits_per_nnz": round(bits_per_nnz, 4),
            "n_obs": dataset.n_obs,
            "n_vars": dataset.n_vars,
        },
    )
    # Add a single "run" recording the conversion time for reference.
    if convert_result is not None:
        result.add_run(
            wall_s=convert_result.wall_s,
            peak_rss_mb=convert_result.peak_rss_mb,
        )
    return result
