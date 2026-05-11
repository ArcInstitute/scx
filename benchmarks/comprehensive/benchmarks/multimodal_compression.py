"""Multimodal compression benchmark (Phase K.3).

Mirrors `compression.py` for multimodal datasets: measures on-disk
size after converting an `.h5mu` to each multimodal format variant.

Compares:
  * SCX v2 multimodal with per-modality codec routing
  * SCX v2 multimodal with uniform single-modality codec selection
    (Phase K.3.4 sweep variant)
  * h5mu (uncompressed and gzip)
  * Zarr-MuData (zstd)

Records per-modality nnz and per-modality codec assignment in
``metadata``, so downstream reporting can quantify the gain from
per-modality codec routing on a real multimodal dataset.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner


log = logging.getLogger(__name__)


def _estimate_per_modality_nnz(h5mu_path: Path) -> dict[str, int]:
    """Return ``{modality_name: nnz}`` from the source `.h5mu`.

    Uses ``mudata.read_h5mu(..., backed="r")`` to avoid materialising
    the full matrix in memory; for sparse modalities the nnz attr is
    cheap.
    """
    import mudata
    import scipy.sparse as sp

    # Disable HDF5 file locking — multiple parallel SLURM jobs may read
    # the same source `.h5mu` concurrently.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    mu = mudata.read_h5mu(str(h5mu_path), backed="r")
    out: dict[str, int] = {}
    try:
        for name in mu.mod:
            X = mu.mod[name].X
            if sp.issparse(X):
                out[name] = int(X.nnz)
            else:
                import numpy as np

                out[name] = int(np.count_nonzero(X))
    finally:
        try:
            mu.file.close()
        except AttributeError:
            pass
    return out


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Multimodal compression benchmark for one (dataset, format) pair.

    Parameters
    ----------
    dataset
        Must be a multimodal dataset (``dataset.multimodal`` true).
        Single-modality datasets raise — use ``compression.py`` instead.
    format_variant
        One of the multimodal format variants. h5ad/zarr/scx
        single-modality variants raise.
    n_runs
        Ignored (size measurement; conversion time recorded once).
    """
    if not dataset.multimodal:
        raise ValueError(
            f"multimodal_compression: dataset {dataset.name!r} is "
            f"single-modality; use the compression benchmark instead"
        )

    h5mu_path = dataset.h5mu_path
    if not h5mu_path.exists():
        raise FileNotFoundError(
            f"Source .h5mu not found: {h5mu_path}. "
            f"Run benchmarks/scripts/download_{dataset.id.lower()}.py first."
        )

    source_bytes = h5mu_path.stat().st_size
    runner = make_runner(format_variant)

    log.info(
        "Multimodal compression: dataset=%s format=%s source=%.1f MB",
        dataset.name,
        format_variant.key,
        source_bytes / 1e6,
    )

    convert_result = None
    if converted_path is not None and Path(converted_path).exists():
        converted_bytes = runner.file_size(converted_path)
    else:
        with tempfile.TemporaryDirectory(prefix="scx_bench_mm_comp_") as tmp:
            # Zarr-MuData uses a directory layout; the others use a
            # single file. Both are valid as `out_path` arguments.
            out_path = Path(tmp) / f"converted.{format_variant.key}"
            convert_result = runner.convert_from_h5mu(h5mu_path, out_path)
            converted_bytes = convert_result.output_size_bytes

    nnz_per_modality = _estimate_per_modality_nnz(h5mu_path)
    nnz = sum(nnz_per_modality.values())

    compression_ratio = (
        source_bytes / converted_bytes if converted_bytes > 0 else float("inf")
    )
    bits_per_nnz = (converted_bytes * 8) / nnz if nnz > 0 else float("inf")

    log.info(
        "  -> converted=%.1f MB  ratio=%.2fx  bits/nnz=%.2f  nnz=%d (%s)",
        converted_bytes / 1e6,
        compression_ratio,
        bits_per_nnz,
        nnz,
        nnz_per_modality,
    )

    metadata: dict = {
        "source_h5mu_bytes": source_bytes,
        "compression_ratio_vs_h5mu": round(compression_ratio, 4),
        "bits_per_nnz": round(bits_per_nnz, 4),
        "n_obs": dataset.n_obs,
        "n_vars": dataset.n_vars,
        "modality_names": list(dataset.modality_names),
        "nnz_per_modality": nnz_per_modality,
    }

    # If the SCX runner recorded per-modality codec assignments in
    # `convert_result.extra`, surface them in `metadata` so the
    # K.3.4 codec-sweep report can compare per-modality vs uniform
    # routing without parsing on-disk catalogs.
    if convert_result is not None and convert_result.extra:
        for k in ("per_modality_codec_id", "codec", "codec_per_modality", "compression"):
            if k in convert_result.extra:
                metadata[k] = convert_result.extra[k]

    result = BenchmarkResult(
        benchmark="multimodal_compression",
        format=format_variant.key,
        dataset=dataset.name,
        file_size_bytes=converted_bytes,
        metadata=metadata,
    )
    # Headline metrics flow into ``runs[].extra`` so the gate's
    # ``check_absolute_floors`` (which reads ``runs[].extra`` per
    # ``compare_against_baseline.py``) can see them. ``metadata``
    # keeps the human-readable summary. We always emit at least one
    # run carrying the metrics — even when ``converted_path`` is
    # supplied by the orchestrator (so no in-benchmark conversion
    # happened) — because otherwise the gate would treat the
    # benchmark as "no observations" and fail every floor.
    wall_s = convert_result.wall_s if convert_result is not None else 0.0
    peak_rss_mb = (
        convert_result.peak_rss_mb if convert_result is not None else 0.0
    )
    result.add_run(
        wall_s=wall_s,
        peak_rss_mb=peak_rss_mb,
        output_size_bytes=converted_bytes,
        compression_ratio_vs_h5mu=metadata["compression_ratio_vs_h5mu"],
        bits_per_nnz=metadata["bits_per_nnz"],
    )
    return result
