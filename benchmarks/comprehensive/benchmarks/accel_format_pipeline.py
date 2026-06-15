"""
Storage-format → GPU pipeline benchmark: `scx + rapids-singlecell` vs
`h5ad + rapids-singlecell`.

Every existing `accel_*` benchmark holds the input matrix constant — it
load+preprocesses once into an in-memory `AnnData` fixture and then varies only
*compute residency* (`accel_pipeline.py`: CPU vs GPU host-boundary vs rapids
fused). This benchmark measures the orthogonal axis: hold the GPU engine
(**rapids-singlecell**) constant and vary the **storage format**. The only thing
that differs between variants is the **load-from-disk → onto-GPU** path; both
then run the *identical* rapids pipeline
(`normalize_total → log1p → HVG → pca → neighbors → umap`).

| `format_variant.key` | Load-to-GPU path |
|---|---|
| `accel_format_pipeline__h5ad_rapids_gpu`          | `anndata.read_h5ad` (host decompress) → `rsc.get.anndata_to_GPU` (host→device) |
| `accel_format_pipeline__scx_devdecode_rapids_gpu` | Scx1+sidecar `.scx` → `to_gpu_anndata(device="gpu")` — decodes in VRAM, only the indptr uploaded (`transfer_mode=scx_device_decode_gpu`) |
| `accel_format_pipeline__scx_auto_rapids_gpu`      | realistic `scx_auto` `.scx` → `to_gpu_anndata(device="gpu")` — records whatever `transfer_mode` results |

The expected SCX win is in `load_to_gpu_s` and host `peak_rss_mb` (device-decode
avoids the host bounce); `pipeline_s` is ~equal across variants by construction
(same rapids engine).

## Emitted metrics (per run, in `runs[].extra`; medians mirrored into `metadata`)

| Field | Meaning |
|---|---|
| `load_to_gpu_s` | wall time to obtain the GPU-resident raw `AnnData` (the differentiator) |
| `pipeline_s`    | wall time for the shared rapids pipeline |
| `transfer_mode` | SCX: `uns["scx_accel"]["to_gpu_anndata"]["transfer_mode"]`; h5ad: `h5ad_host_to_gpu` |
| `bytes_uploaded`| SCX: real host→device byte count (0 for h5ad — full matrix is transferred) |
| `scx_format_pipeline_devdecode_route` | devdecode variant only: 1.0 iff `transfer_mode == scx_device_decode_gpu` |
| `umap_trustworthiness` | sklearn trustworthiness of the final embedding (sanity) |

Timing/speedup is informational (rapids version drift must not fail the build);
only the device-decode route floor is gated in `thresholds.yaml`, staged on
pbmc3k + tabula_sapiens_100k until a baseline carries the metric.
"""

from __future__ import annotations

import gc
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the sidecar-fixture prep from the device-decode route benchmark.
from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import (
    _prepare_sidecar_scx,
)
from benchmarks.comprehensive.benchmarks.accel_umap import _trustworthiness
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    QUERY_N_HVGS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result
from benchmarks.comprehensive.rss import current_rss_mb as _get_rss_mb
from benchmarks.comprehensive.runners.accel_runner import AcceleratorRunner

logger = logging.getLogger(__name__)

_HAS_PYSCX = AcceleratorRunner.instance().has_pyscx()
_HAS_PYSCX_GPU = AcceleratorRunner.instance().has_gpu()

_H5AD_KEY = "accel_format_pipeline__h5ad_rapids_gpu"
_SCX_DEVDECODE_KEY = "accel_format_pipeline__scx_devdecode_rapids_gpu"
_SCX_AUTO_KEY = "accel_format_pipeline__scx_auto_rapids_gpu"
_ALL_KEYS = frozenset({_H5AD_KEY, _SCX_DEVDECODE_KEY, _SCX_AUTO_KEY})
_SCX_KEYS = frozenset({_SCX_DEVDECODE_KEY, _SCX_AUTO_KEY})

# Single-modality datasets, mirroring `accel_to_gpu_anndata.SUPPORTED_DATASETS`.
# Multimodal datasets are excluded (no single-modality count fixture). Out-of-
# scope datasets write a typed stub (defense-in-depth for direct invocation; the
# orchestrator has no per-dataset cohort filter).
SUPPORTED_DATASETS: frozenset[str] = frozenset(
    {
        "pbmc3k",
        "pbmc10k",
        "smartseq2",
        "tabula_sapiens_100k",
        "census_500k",
        "census_1m",
    }
)


def accel_format_pipeline_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="h5ad → rapids-singlecell pipeline (GPU)",
            key=_H5AD_KEY,
            category="accel",
            runner="accel_runner",
        ),
        FormatVariant(
            name="SCX device-decode → rapids-singlecell pipeline (GPU)",
            key=_SCX_DEVDECODE_KEY,
            category="accel",
            runner="accel_runner",
        ),
        FormatVariant(
            name="SCX (auto) → rapids-singlecell pipeline (GPU)",
            key=_SCX_AUTO_KEY,
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


def _prepare_scx_auto(dataset: DatasetConfig, tmpdir: str) -> Path:
    """Resolve a realistic `scx_auto` `.scx` file.

    Uses the pre-converted `dataset.scx_auto_path` fixture when present (what an
    end user actually produces); otherwise self-converts from h5ad with the
    default auto codec into the tempdir so the benchmark is self-contained.
    """
    import pyscx

    p = dataset.scx_auto_path
    if p.exists():
        return p

    import anndata

    logger.info("scx_auto fixture absent — self-converting %s from h5ad", dataset.name)
    out = Path(tmpdir) / f"{dataset.name}_auto.scx"
    adata = anndata.read_h5ad(str(dataset.h5ad_path))
    pyscx.from_anndata(adata, str(out))
    del adata
    gc.collect()
    return out


def _load_h5ad_to_gpu(dataset: DatasetConfig) -> tuple[Any, dict[str, Any]]:
    """`read_h5ad` (host decompress) → `rsc.get.anndata_to_GPU` (host→device).

    Read fresh each call so the timed segment includes the real disk read +
    full host→device transfer — the h5ad baseline cost.
    """
    import anndata
    import rapids_singlecell as rsc

    adata = anndata.read_h5ad(str(dataset.h5ad_path))
    rsc.get.anndata_to_GPU(adata)
    return adata, {"transfer_mode": "h5ad_host_to_gpu", "bytes_uploaded": 0}


def _load_scx_to_gpu(scx_path: Path) -> tuple[Any, dict[str, Any]]:
    """`pyscx.open(...).to_gpu_anndata(device="gpu")` — device-resident `AnnData`."""
    import pyscx

    adata = pyscx.open(str(scx_path)).to_gpu_anndata(device="gpu")
    info = adata.uns.get("scx_accel", {}).get("to_gpu_anndata", {})
    return adata, {
        "transfer_mode": info.get("transfer_mode"),
        "bytes_uploaded": int(info.get("bytes_uploaded") or 0),
    }


def _run_rapids_pipeline(adata: Any, n_comps: int, n_neighbors: int, seed: int) -> Any:
    """Shared rapids-singlecell pipeline run in-place on a GPU-resident `AnnData`.

    `normalize_total → log1p → highly_variable_genes(subset) → pca → neighbors →
    umap`. Mirrors the order in `AcceleratorRunner.load_preprocessed` (incl. the
    seurat_v3 → default-flavor fallback) so the two formats are apples-to-apples
    and consistent with the rest of the accel suite. Returns the (possibly
    var-subset) `AnnData` carrying `obsm["X_pca"]` / `obsm["X_umap"]`.
    """
    import rapids_singlecell as rsc

    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)

    n_top = min(QUERY_N_HVGS, adata.n_vars)
    try:
        rsc.pp.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3", subset=True
        )
    except TypeError:
        # Older rapids-singlecell without the `subset` kwarg — subset manually.
        rsc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"].values].copy()
    except Exception:
        # seurat_v3 can reject lognorm input on some versions — fall back to the
        # default flavor (same fallback as `load_preprocessed`).
        rsc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)

    n_comps_eff = min(n_comps, adata.n_vars - 1, adata.n_obs - 1)
    rsc.pp.pca(adata, n_comps=n_comps_eff, random_state=seed)
    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=seed)
    rsc.tl.umap(adata, random_state=seed)
    return adata


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,  # unused — fixtures are self-prepared
    n_comps: int = 50,
    n_neighbors: int = 15,
) -> BenchmarkResult | None:
    """Execute the storage-format → GPU pipeline benchmark for one variant.

    Returns `None` for an unavailable variant (CPU-only host, missing pyscx);
    writes a typed `write_missing_result` stub for missing cupy / rapids /
    fixture-prep so the coverage gap is visible.
    """
    key = format_variant.key
    if key not in _ALL_KEYS:
        return None
    if dataset.name not in SUPPORTED_DATASETS:
        logger.info(
            "%s gate scoped to %s — recording stub for %s",
            key,
            sorted(SUPPORTED_DATASETS),
            dataset.name,
        )
        write_missing_result(
            benchmark="accel_format_pipeline",
            format_key=key,
            dataset=dataset.name,
            missing_reason="dataset_out_of_gate_scope",
            notes=f"accel_format_pipeline scoped to {sorted(SUPPORTED_DATASETS)}",
        )
        return None

    is_scx = key in _SCX_KEYS
    if is_scx and not _HAS_PYSCX:
        logger.warning("pyscx not installed — skipping %s", key)
        return None
    if not _HAS_PYSCX_GPU:
        logger.warning("GPU not available — skipping %s", key)
        return None
    if not _has_cupy():
        logger.warning("cupy not installed — recording stub for %s", key)
        write_missing_result(
            benchmark="accel_format_pipeline",
            format_key=key,
            dataset=dataset.name,
            missing_reason="no_cupy",
            notes="cupy import failed; GPU pipeline requires cupyx CSR",
        )
        return None
    if not AcceleratorRunner.instance().has_rapids_singlecell():
        logger.warning("rapids-singlecell not installed — recording stub for %s", key)
        write_missing_result(
            benchmark="accel_format_pipeline",
            format_key=key,
            dataset=dataset.name,
            missing_reason="no_rapids_singlecell",
            notes="rapids-singlecell import failed; install into scx-bench-gpu",
        )
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare the SCX fixture (h5ad reads from the source fixture directly).
        scx_path: Path | None = None
        n_obs: int | None = None
        n_shards: int | None = None
        if key == _SCX_DEVDECODE_KEY:
            try:
                scx_path, n_obs, n_shards = _prepare_sidecar_scx(dataset, tmpdir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", "replace")[-400:]
                logger.warning("sidecar fixture prep failed for %s: %s", dataset.name, stderr)
                write_missing_result(
                    benchmark="accel_format_pipeline",
                    format_key=key,
                    dataset=dataset.name,
                    missing_reason="fixture_prep_failed",
                    notes=f"scx optimize / self-convert failed: {stderr}",
                )
                return None
        elif key == _SCX_AUTO_KEY:
            scx_path = _prepare_scx_auto(dataset, tmpdir)

        def _load() -> tuple[Any, dict[str, Any]]:
            if key == _H5AD_KEY:
                return _load_h5ad_to_gpu(dataset)
            return _load_scx_to_gpu(scx_path)  # type: ignore[arg-type]

        result = BenchmarkResult(
            benchmark="accel_format_pipeline",
            format=key,
            dataset=dataset.name,
            scenario={
                "name": "format_pipeline",
                "device": "gpu",
                "n_comps": n_comps,
                "n_neighbors": n_neighbors,
            },
            comparison={
                "subject": {"impl": key},
                "baseline": {"impl": _H5AD_KEY},
                "metric": "load_to_gpu_s",
                "status": "pending",
            },
            metadata={
                "cold_cache": cold_cache,
                "n_warmup": N_WARMUP_RUNS,
                "n_comps": n_comps,
                "n_neighbors": n_neighbors,
                "n_obs": n_obs,
                "n_shards": n_shards,
                "random_seed": RANDOM_SEED,
            },
        )

        load_runs: list[float] = []
        pipe_runs: list[float] = []
        trust_runs: list[float] = []
        last_transfer_mode: Any = None

        # Warm-up + timed runs share one guard: a failure is almost always an
        # out-of-VRAM allocation on the largest tiers — record a typed stub
        # rather than aborting the whole accel cohort.
        try:
            for _ in range(N_WARMUP_RUNS):
                warm, _info = _load()
                warm = _run_rapids_pipeline(warm, n_comps, n_neighbors, RANDOM_SEED)
                del warm
                gc.collect()

            import rapids_singlecell as rsc

            for i in range(n_runs):
                gc.collect()
                rss_before = _get_rss_mb()

                t0 = time.perf_counter()
                adata, info = _load()
                load_s = time.perf_counter() - t0

                t1 = time.perf_counter()
                final = _run_rapids_pipeline(adata, n_comps, n_neighbors, RANDOM_SEED)
                pipe_s = time.perf_counter() - t1

                wall = load_s + pipe_s
                rss_after = _get_rss_mb()
                last_transfer_mode = info["transfer_mode"]

                extras: dict[str, Any] = {
                    "load_to_gpu_s": load_s,
                    "pipeline_s": pipe_s,
                    "transfer_mode": info["transfer_mode"],
                    "bytes_uploaded": info["bytes_uploaded"],
                }
                if key == _SCX_DEVDECODE_KEY:
                    extras["scx_format_pipeline_devdecode_route"] = (
                        1.0 if info["transfer_mode"] == "scx_device_decode_gpu" else 0.0
                    )

                # Light correctness sanity — round-trip the embedding to host.
                try:
                    rsc.get.anndata_to_CPU(final, convert_all=True)
                    pca = np.asarray(final.obsm["X_pca"], dtype=np.float32)
                    um = np.asarray(final.obsm["X_umap"], dtype=np.float32)
                    tw = _trustworthiness(pca, um, k=n_neighbors)
                    if not np.isnan(tw):
                        extras["umap_trustworthiness"] = tw
                        trust_runs.append(tw)
                except Exception as e:  # noqa: BLE001
                    logger.warning("trustworthiness check failed for %s run %d: %s", key, i + 1, e)

                logger.info(
                    "%s run %d/%d: load=%.3fs pipeline=%.3fs wall=%.3fs rss=%.1fMB transfer=%s",
                    key, i + 1, n_runs, load_s, pipe_s, wall,
                    max(rss_before, rss_after), info["transfer_mode"],
                )
                result.add_run(
                    wall_s=wall,
                    user_s=0.0,
                    sys_s=0.0,
                    peak_rss_mb=max(rss_before, rss_after),
                    **extras,
                )
                load_runs.append(load_s)
                pipe_runs.append(pipe_s)
                del adata, final
                gc.collect()
        except Exception as exc:  # noqa: BLE001 — cupy/cuda errors are opaque
            logger.warning(
                "GPU pipeline failed for %s / %s (likely >VRAM): %s",
                key, dataset.name, exc,
            )
            write_missing_result(
                benchmark="accel_format_pipeline",
                format_key=key,
                dataset=dataset.name,
                missing_reason="gpu_pipeline_failed",
                notes=f"accel_format_pipeline raised on {dataset.name}: {exc}",
            )
            return None

        # Mirror medians into metadata for the report table builders.
        if load_runs:
            result.metadata["load_to_gpu_s"] = round(float(np.median(load_runs)), 4)
        if pipe_runs:
            result.metadata["pipeline_s"] = round(float(np.median(pipe_runs)), 4)
        if trust_runs:
            result.metadata["umap_trustworthiness"] = round(float(np.median(trust_runs)), 4)
        result.metadata["transfer_mode"] = last_transfer_mode

        logger.info(
            "Benchmark complete: %s / %s — median wall %.3fs, load %.3fs, transfer=%s",
            key, dataset.name, result.median_wall_s or 0.0,
            result.metadata.get("load_to_gpu_s", float("nan")), last_transfer_mode,
        )
        return result
