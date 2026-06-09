"""
`to_gpu_anndata` device-decode benchmark + route floor (ACC-RUST-OPT-V4 §4.4 / 4.4b).

Validates that `pyscx.open(...).to_gpu_anndata(device="gpu")` decodes an Scx1
count matrix fully in VRAM — recording `transfer_mode == "scx_device_decode_gpu"`
with `bytes_uploaded` ≈ the per-shard indptr only (not the nnz-sized arrays) —
and that the device-resident CSR is byte-identical to a host decode.

Self-contained fixture: the dataset's raw-count h5ad is converted to a temporary
Scx1 SCX file (decode sidecars are auto-emitted by the encoder) inside `run()`;
`to_gpu_anndata` then decodes it on the device. As of the BitPacker4x kernel
(4.4b) this covers dense (≥128-nnz) rows too, so a count file decodes entirely
on-device.

## Emitted metrics (per run, in `runs[].extra`)

| Field | Meaning |
|---|---|
| `transfer_mode` | `uns["scx_accel"]["to_gpu_anndata"]["transfer_mode"]` |
| `bytes_uploaded` | real host→device byte count |
| `to_gpu_anndata_route_device_decode` | 1.0 iff transfer_mode == `scx_device_decode_gpu` |
| `to_gpu_anndata_decode_correct` | 1.0 iff device CSR == host CSR (indptr/indices/data) |

The route floor (`to_gpu_anndata_route_device_decode >= 1.0`) is staged in
`thresholds.yaml` and activates once a baseline carries the metric.
"""

from __future__ import annotations

import gc
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
)
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result
from benchmarks.comprehensive.rss import current_rss_mb as _get_rss_mb
from benchmarks.comprehensive.runners.accel_runner import AcceleratorRunner

logger = logging.getLogger(__name__)

_HAS_PYSCX = AcceleratorRunner.instance().has_pyscx()
_HAS_PYSCX_GPU = AcceleratorRunner.instance().has_gpu()

_VARIANT_KEY = "accel_to_gpu_anndata__scx1_gpu"


def accel_to_gpu_anndata_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx to_gpu_anndata (device decode)",
            key=_VARIANT_KEY,
            category="accel",
            runner="accel_runner",
        ),
    ]


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _convert_counts_to_scx(dataset: DatasetConfig, tmpdir: str) -> tuple[Path, int, int]:
    """Convert the dataset's raw-count h5ad to an Scx1 SCX file (decode sidecars
    auto-emitted). Returns (scx_path, n_obs, n_shards)."""
    import anndata
    import pyscx

    adata = anndata.read_h5ad(str(dataset.h5ad_path))
    # Keep integer counts — Scx1 codec + decode sidecars require integer X.
    scx_path = Path(tmpdir) / f"{dataset.name}.scx"
    pyscx.from_anndata(adata, str(scx_path))
    n_obs = int(adata.n_obs)
    n_shards = int(pyscx.open(str(scx_path)).shard_count)
    return scx_path, n_obs, n_shards


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,  # unused — fixture is self-converted
) -> BenchmarkResult | None:
    variant_key = format_variant.key
    if variant_key != _VARIANT_KEY:
        return None
    if not _HAS_PYSCX:
        logger.warning("pyscx not installed — skipping %s", variant_key)
        return None
    if not _HAS_PYSCX_GPU:
        logger.warning("GPU not available — skipping %s", variant_key)
        return None
    if not _has_cupy():
        logger.warning("cupy not installed — recording stub for %s", variant_key)
        write_missing_result(
            benchmark="accel_to_gpu_anndata",
            format_key=variant_key,
            dataset=dataset.name,
            missing_reason="no_cupy",
            notes="cupy import failed; to_gpu_anndata requires cupyx CSR",
        )
        return None

    import pyscx

    with tempfile.TemporaryDirectory() as tmpdir:
        scx_path, n_obs, n_shards = _convert_counts_to_scx(dataset, tmpdir)

        # Host reference decode (scipy CSR), sorted for a byte-exact compare.
        host_x = pyscx.open(str(scx_path)).to_anndata().X.tocsr()
        host_x.sort_indices()

        result = BenchmarkResult(
            benchmark="accel_to_gpu_anndata",
            format=variant_key,
            dataset=dataset.name,
            scenario={"name": "to_gpu_anndata", "device": "gpu"},
            comparison={
                "subject": {"impl": variant_key},
                "baseline": {"impl": variant_key},
                "metric": "to_gpu_anndata_route_device_decode",
                "status": "pending",
            },
            metadata={
                "cold_cache": cold_cache,
                "n_warmup": N_WARMUP_RUNS,
                "n_obs": n_obs,
                "n_shards": n_shards,
            },
        )

        # Warm-up (PTX module load, allocator warm).
        for _ in range(N_WARMUP_RUNS):
            warm = pyscx.open(str(scx_path)).to_gpu_anndata(device="gpu")
            del warm
            gc.collect()

        for i in range(n_runs):
            gc.collect()
            rss_before = _get_rss_mb()
            t0 = time.perf_counter()
            adata_gpu = pyscx.open(str(scx_path)).to_gpu_anndata(device="gpu")
            wall = time.perf_counter() - t0
            rss_after = _get_rss_mb()

            info = adata_gpu.uns["scx_accel"]["to_gpu_anndata"]
            transfer_mode = info.get("transfer_mode")
            bytes_uploaded = int(info.get("bytes_uploaded") or 0)

            # Byte-exact parity vs the host decode.
            gpu_x = adata_gpu.X.get()  # cupyx CSR -> scipy CSR
            gpu_x.sort_indices()
            correct = (
                tuple(gpu_x.shape) == tuple(host_x.shape)
                and np.array_equal(gpu_x.indptr, host_x.indptr)
                and np.array_equal(gpu_x.indices, host_x.indices)
                and np.array_equal(gpu_x.data, host_x.data)
            )

            extras: dict[str, Any] = {
                "transfer_mode": transfer_mode,
                "bytes_uploaded": bytes_uploaded,
                "to_gpu_anndata_route_device_decode": (
                    1.0 if transfer_mode == "scx_device_decode_gpu" else 0.0
                ),
                "to_gpu_anndata_decode_correct": 1.0 if correct else 0.0,
            }
            logger.info(
                "%s run %d/%d: transfer_mode=%s bytes_uploaded=%d correct=%s",
                variant_key,
                i + 1,
                n_runs,
                transfer_mode,
                bytes_uploaded,
                correct,
            )
            result.add_run(
                wall_s=wall,
                user_s=0.0,
                sys_s=0.0,
                peak_rss_mb=max(rss_before, rss_after),
                **extras,
            )
            del adata_gpu, gpu_x
            gc.collect()

        return result
