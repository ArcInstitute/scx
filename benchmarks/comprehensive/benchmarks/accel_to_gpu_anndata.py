"""
`to_gpu_anndata` device-decode benchmark + route floor.

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
| `shufdelta_n_shards_gpu` | shards that took the GPU ShufDeltaZstd path |
| `shufdelta_fully_device_decoded` | 1.0 iff only compressed bytes crossed PCIe |
| `decode_arm` | the variant key, so a row carries its arm |

## The three arms

`Scx1` and `ShufDeltaZstd` are different decode paths, and ShufDeltaZstd has
two of them. Which one runs is a property of the **file**, not just the env,
so each arm pins its own codec via `scx optimize --codec`:

| variant | fixture | codec | env | decode path |
|---|---|---|---|---|
| `__scx1_gpu` | `_scx1` | `scx1` | — | framed Scx1, FOR-BP/Rice in VRAM |
| `__shufdelta_gpu` | `_shufdelta` | `shufdelta` | — | Phase 1.5: CPU zstd pipelined, planes uploaded |
| `__shufdelta_gpu_nvcomp` | `_shufdelta` | `shufdelta` | `SCX_SHUFDELTA_NVCOMP=1` | Phase 2: nvcomp zstd in VRAM |

`SCX_SHUFDELTA_NVCOMP=1` against the `scx1` fixture is a **no-op** — Scx1
shards never reach the ShufDeltaZstd paths — which is why the nvcomp arm has
its own fixture and not merely its own env. Without that, the two "arms" would
report identical numbers under different names.

`shufdelta_fully_device_decoded` is what separates the two ShufDeltaZstd arms:
`shard_decode.rs` sets `fully_device_decoded` true only on the nvcomp path
(only compressed bytes cross PCIe) and false on the pipelined path. nvcomp is
runtime-`dlopen`'d and falls back to the pipeline *silently* when
`libnvcomp.so.5` is absent, so the `_nvcomp` arm carries a `min: 1.0` floor on
that metric and the pipelined arm a `max: 0.0` — between them they assert that
each run took the arm its name claims.

The route floor (`to_gpu_anndata_route_device_decode >= 1.0`) and the two
ShufDeltaZstd floors live in `thresholds.yaml`. Absolute floors are evaluated
against the **candidate** snapshot only (`check_absolute_floors(args.current,
…)`), so they gate from the first capture that carries the metric — no
promoted baseline required.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
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
_SHUFDELTA_KEY = "accel_to_gpu_anndata__shufdelta_gpu"
_SHUFDELTA_NVCOMP_KEY = "accel_to_gpu_anndata__shufdelta_gpu_nvcomp"


@dataclass(frozen=True)
class _Arm:
    """One decode arm: which fixture it reads, and what env it reads it under.

    The arm is what a number on this benchmark has to be labelled with, because
    three different decode paths produce three different walls from the same
    logical matrix, and `transfer_mode` alone does not separate the two
    ShufDeltaZstd ones — `shufdelta_fully_device_decoded` does (see
    `_METRIC_NOTES` below).
    """

    #: `DatasetConfig` attribute holding the pre-built source fixture.
    fixture_attr: str
    #: `scx optimize --codec` value. Pins the on-disk codec, which is what
    #: selects the decode path — `auto` would re-decide per shard and could
    #: silently move a run onto a different arm than its name claims.
    codec: str
    #: Env applied for the whole timed section.
    env: dict[str, str]
    #: Datasets this arm runs on.
    datasets: frozenset[str]


# `--row-group-rows` is required for `--codec shufdelta` and produces the v4
# BlockIndex the GPU framed decode needs; 256 is `scx optimize`'s own default
# and the value the fixtures were built at.
_SHUFDELTA_TIERS: frozenset[str] = frozenset({"pbmc3k", "tabula_sapiens_100k"})

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
# NOTE: the `thresholds.yaml` route/decode floors stay on pbmc3k +
# tabula_sapiens_100k only. Census device-decode metrics populate on the next
# full capture; their floors are added in a follow-up once one has run.
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


# The three arms, keyed by variant. Scx1 shards never enter the ShufDeltaZstd
# decode paths at all, so `SCX_SHUFDELTA_NVCOMP=1` on the scx1 variant is a
# literal no-op — running it as an "nvcomp arm" would report the pipelined
# numbers under the nvcomp name. That is why the nvcomp arm needs its own
# fixture rather than just its own env.
ARMS: dict[str, _Arm] = {
    _VARIANT_KEY: _Arm(
        fixture_attr="scx_scx1_path",
        codec="scx1",
        env={},
        datasets=SUPPORTED_DATASETS,
    ),
    _SHUFDELTA_KEY: _Arm(
        fixture_attr="scx_shufdelta_path",
        codec="shufdelta",
        env={},
        datasets=_SHUFDELTA_TIERS,
    ),
    _SHUFDELTA_NVCOMP_KEY: _Arm(
        fixture_attr="scx_shufdelta_path",
        codec="shufdelta",
        env={"SCX_SHUFDELTA_NVCOMP": "1"},
        datasets=_SHUFDELTA_TIERS,
    ),
}


def accel_to_gpu_anndata_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx to_gpu_anndata (device decode)",
            key=_VARIANT_KEY,
            category="accel",
            runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx to_gpu_anndata (ShufDeltaZstd, pipelined CPU zstd)",
            key=_SHUFDELTA_KEY,
            category="accel",
            runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx to_gpu_anndata (ShufDeltaZstd, nvcomp in-VRAM zstd)",
            key=_SHUFDELTA_NVCOMP_KEY,
            category="accel",
            runner="accel_runner",
        ),
    ]


# Read by `run_parallel._bench_format_dataset_scope` so an arm's tiers are
# honoured at COHORT CONSTRUCTION, not by stubbing inside `run()` after Chimera
# has already started a preemptible GPU task (Cursor Agent, #474). Derived from
# `ARMS` so the two cannot drift.
FORMAT_DATASET_SCOPE: dict[str, frozenset[str]] = {
    key: arm.datasets for key, arm in ARMS.items()
}


@contextlib.contextmanager
def _arm_env(env: dict[str, str]):
    """Apply an arm's env for the duration, restoring the prior values.

    In-process is sound for `SCX_SHUFDELTA_NVCOMP` specifically: `nvcomp.rs`
    documents it as read **per call**, not cached, "so it can be toggled
    in-process for A/B benchmarking". `run_parallel.py` also exports it in the
    worker's shell, which is what makes the arm right even for a knob that
    caches — and belt-and-braces here.
    """
    prev = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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


def _prepare_sidecar_scx(
    dataset: DatasetConfig, tmpdir: str, arm: _Arm
) -> tuple[Path, int, int]:
    """Produce a decode-sidecar SCX file for the dataset, in the arm's codec.

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

    fixture = getattr(dataset, arm.fixture_attr)
    opt_bin = _resolve_scx_optimize_bin()
    if fixture.exists() and opt_bin is not None:
        logger.info(
            "preparing sidecar fixture via `scx optimize --codec %s` (%s) from %s",
            arm.codec,
            opt_bin,
            fixture,
        )
        # The codec is pinned, never `auto`. `--codec scx1` forces Scx1 on every
        # integer shard so all shards carry a decode sidecar (high-median shards
        # that auto would route to Zstd otherwise host-fall-back, defeating the
        # device-decode route); `--codec shufdelta` pins the arm whose decode
        # path this variant exists to measure. `--row-group-rows 256` is
        # required for shufdelta and produces the v4 BlockIndex the GPU framed
        # decode reads; it is also `scx optimize`'s own default, so passing it
        # explicitly changes nothing for the scx1 arm and documents the
        # requirement for the other two.
        subprocess.run(
            [
                opt_bin,
                "optimize",
                str(fixture),
                str(scx_path),
                "--codec",
                arm.codec,
                "--row-group-rows",
                "256",
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
    pyscx.from_anndata(adata, str(scx_path), codec=arm.codec)
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
    """Entry point. Applies the arm's env around everything the arm affects."""
    arm = ARMS.get(format_variant.key)
    if arm is None:
        return None
    with _arm_env(arm.env):
        return _run_arm(dataset, format_variant, arm, n_runs, cold_cache)


def _run_arm(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    arm: _Arm,
    n_runs: int,
    cold_cache: bool,
) -> BenchmarkResult | None:
    variant_key = format_variant.key
    if dataset.name not in arm.datasets:
        logger.info(
            "%s gate scoped to %s — recording stub for %s",
            variant_key,
            sorted(arm.datasets),
            dataset.name,
        )
        write_missing_result(
            benchmark="accel_to_gpu_anndata",
            format_key=variant_key,
            dataset=dataset.name,
            missing_reason="dataset_out_of_gate_scope",
            notes=f"{variant_key} scoped to {sorted(arm.datasets)}",
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
            scx_path, n_obs, n_shards = _prepare_sidecar_scx(dataset, tmpdir, arm)
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
                n_shufdelta_gpu = int(info.get("n_shards_shufdelta_gpu") or 0)

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
                    # --- arm labels -------------------------------------------
                    # Which decode path actually ran, recorded rather than
                    # inferred from the variant name. `n_shards_shufdelta_gpu`
                    # counts shards that took the GPU ShufDeltaZstd path at all
                    # (0 on the scx1 arm, and 0 on a shufdelta file whose shards
                    # host-bounced). `fully_device_decoded` — which
                    # `transfer_mode == "scx_device_decode_gpu"` reports — is
                    # what separates the two ShufDeltaZstd arms: `shard_decode.rs`
                    # sets it true only on the nvcomp path, where just the
                    # compressed bytes cross PCIe, and false on the Phase-1.5
                    # pipelined path, which uploads decompressed planes. So a
                    # `_nvcomp` run that reports 0.0 here did NOT take the arm
                    # its name claims — nvcomp is runtime-dlopen'd and falls
                    # back to the pipeline in silence when libnvcomp is absent.
                    "shufdelta_n_shards_gpu": float(n_shufdelta_gpu),
                    "decode_arm": variant_key,
                }
                # ...but ONLY on an arm that actually decoded ShufDeltaZstd
                # shards. Scx1 also stamps `scx_device_decode_gpu`, so emitting
                # this unconditionally recorded
                # `shufdelta_fully_device_decoded = 1.0` alongside
                # `shufdelta_n_shards_gpu = 0` on the `__scx1_gpu` arm — the
                # name was a lie on the variant this benchmark already shipped
                # (Cursor Agent, #474; confirmed in the job-2858457 snapshot).
                # Absent is the honest answer for an arm the metric is not about,
                # and `check_absolute_floors` only reads it where a floor names
                # it, which is the two ShufDeltaZstd arms.
                if n_shufdelta_gpu > 0:
                    extras["shufdelta_fully_device_decoded"] = (
                        1.0 if transfer_mode == "scx_device_decode_gpu" else 0.0
                    )
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
