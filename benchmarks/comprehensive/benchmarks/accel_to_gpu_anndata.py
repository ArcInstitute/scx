"""
`to_gpu_anndata` device-decode benchmark + route floor (ACC-RUST-OPT-V4 §4.4 / 4.4b).

Validates that `pyscx.open(...).to_gpu_anndata(device="gpu")` decodes an Scx1
count matrix fully in VRAM — recording `transfer_mode == "scx_device_decode_gpu"`
with `bytes_uploaded` ≈ the per-shard indptr only (not the nnz-sized arrays) —
and that the device-resident CSR is byte-identical to a host decode.

Fixture: a temporary Scx1 SCX file with decode sidecars is prepared inside
`run()` by `scx optimize` on the pre-built `_scx1` fixture (a streaming CSR
decode → re-encode that adds the sidecars with bounded peak RSS), falling back
to a full `read_h5ad` + `from_anndata(codec="scx1")` self-convert when no
sidecar-capable `scx` binary or `_scx1` fixture is present. `to_gpu_anndata`
then decodes it on the device. As of the BitPacker4x kernel (4.4b) this covers
dense (≥128-nnz) rows too, so a count file decodes entirely on-device.

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
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.config import (
    PROJECT_ROOT,
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

# Single-modality full-tier datasets. The sidecar'd fixture is now prepared by
# `scx optimize` on the pre-built `_scx1` fixture — a streaming CSR decode →
# re-encode that adds decode sidecars with bounded peak RSS (one shard at a
# time) and no h5ad parse — so census tiers no longer need the costly h5ad
# read + `from_anndata` self-convert that motivated the earlier
# pbmc3k/tabula-only scope. Multimodal datasets are excluded (the benchmark is
# single-modality and they have no `_scx1` count fixture). Enforced at the top
# of `run()` (defense-in-depth for direct invocation; the orchestrator has no
# per-dataset cohort filter).
#
# NOTE: the staged `thresholds.yaml` route/decode floors stay on pbmc3k +
# tabula_sapiens_100k only — a floor on a metric a baseline does not yet carry
# reads as a violation, and the promoted baseline captured before this widening
# carries device-decode metrics only for those two. Census device-decode
# metrics populate on the next full capture; their floors are added in a
# follow-up once a baseline carries them.
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


def _resolve_scx_optimize_bin() -> str | None:
    """Resolve an `scx` binary whose `optimize` supports `--codec scx1`.

    Probes, in order: `$SCX_CLI_BIN`, the repo `target/release/scx`, then a
    PATH `scx`. Returns the first whose `optimize --help` exits cleanly *and*
    advertises `--codec` (older binaries lack the subcommand or the flag and
    are skipped so the self-convert fallback kicks in), else `None`.
    """
    candidates = []
    env_bin = os.environ.get("SCX_CLI_BIN")
    if env_bin:
        candidates.append(env_bin)
    candidates.append(str(PROJECT_ROOT / "target" / "release" / "scx"))
    which = shutil.which("scx")
    if which:
        candidates.append(which)

    for cand in candidates:
        if not cand or not Path(cand).exists():
            continue
        try:
            proc = subprocess.run(
                [cand, "optimize", "--help"],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            continue
        if proc.returncode == 0 and b"--codec" in proc.stdout:
            return cand
    return None


def _prepare_sidecar_scx(dataset: DatasetConfig, tmpdir: str) -> tuple[Path, int, int]:
    """Produce an Scx1 + decode-sidecar SCX file for the dataset.

    Preferred path: `scx optimize` on the pre-built `_scx1` fixture
    (`dataset.scx_scx1_path`) — a streaming CSR decode → re-encode that adds
    `decode/<name>` sidecars with bounded peak RSS (one shard at a time) and no
    h5ad parse. This is the in-place shared-fixture path that lets the gate
    cover census tiers without the full-matrix materialization the old
    h5ad self-convert required.

    Fallback path (no sidecar-capable binary, or no `_scx1` fixture): a full
    in-process `read_h5ad` + `from_anndata(codec="scx1")` self-convert. Explicit
    Scx1 pins the sidecar path for any integer-valued X (auto only selects Scx1
    for median nonzero ≤ 8, else Zstd — which would silently drop the sidecar
    and force `scx_device_handoff_streamed`). Genuinely fractional X falls back
    to Zstd, which the route/decode metrics surface.

    Returns (scx_path, n_obs, n_shards).
    """
    import pyscx

    scx_path = Path(tmpdir) / f"{dataset.name}.scx"

    fixture = dataset.scx_scx1_path
    opt_bin = _resolve_scx_optimize_bin()
    if fixture.exists() and opt_bin is not None:
        logger.info(
            "preparing sidecar fixture via `scx optimize --codec scx1` (%s) from %s",
            opt_bin,
            fixture,
        )
        # `--codec scx1` forces Scx1 on every integer shard so all shards carry
        # a decode sidecar (high-median shards that auto would route to Zstd
        # otherwise host-fall-back, defeating the device-decode route).
        subprocess.run(
            [
                opt_bin,
                "optimize",
                str(fixture),
                str(scx_path),
                "--codec",
                "scx1",
            ],
            check=True,
            capture_output=True,
            timeout=3600,
        )
        ex = pyscx.open(str(scx_path))
        return scx_path, int(ex.n_obs), int(ex.shard_count)

    logger.info(
        "sidecar binary/fixture unavailable — self-converting %s from h5ad",
        dataset.name,
    )
    import anndata

    adata = anndata.read_h5ad(str(dataset.h5ad_path))
    pyscx.from_anndata(adata, str(scx_path), codec="scx1")
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
    if dataset.name not in SUPPORTED_DATASETS:
        logger.info(
            "%s gate scoped to %s — recording stub for %s",
            variant_key,
            sorted(SUPPORTED_DATASETS),
            dataset.name,
        )
        write_missing_result(
            benchmark="accel_to_gpu_anndata",
            format_key=variant_key,
            dataset=dataset.name,
            missing_reason="dataset_out_of_gate_scope",
            notes="to_gpu_anndata gate scoped to pbmc3k + tabula_sapiens_100k",
        )
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
        try:
            scx_path, n_obs, n_shards = _prepare_sidecar_scx(dataset, tmpdir)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", "replace")[-400:]
            logger.warning(
                "sidecar fixture prep failed for %s: %s", dataset.name, stderr
            )
            write_missing_result(
                benchmark="accel_to_gpu_anndata",
                format_key=variant_key,
                dataset=dataset.name,
                missing_reason="fixture_prep_failed",
                notes=f"scx optimize / self-convert failed: {stderr}",
            )
            return None

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

        # Warm-up (PTX module load, allocator warm) + timed runs under one
        # guard. A failure anywhere here is almost always an out-of-VRAM
        # allocation on the largest tiers — record a typed stub rather than
        # aborting the whole accel cohort. Both loops share the guard so an OOM
        # that only trips on a later (timed) iteration, or a benchmark config
        # with N_WARMUP_RUNS == 0, is still caught.
        try:
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
        except Exception as exc:  # noqa: BLE001 — cupy/cuda errors are opaque
            logger.warning(
                "to_gpu_anndata decode failed for %s (likely >VRAM): %s",
                dataset.name,
                exc,
            )
            write_missing_result(
                benchmark="accel_to_gpu_anndata",
                format_key=variant_key,
                dataset=dataset.name,
                missing_reason="gpu_decode_failed",
                notes=f"to_gpu_anndata raised on {dataset.name}: {exc}",
            )
            return None

        return result
