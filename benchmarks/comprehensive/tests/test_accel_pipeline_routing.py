"""
Registration + SLURM routing for the `accel_pipeline` residency benchmark
(V3 task 2.7) and its rapids-singlecell GPU competitor.

Hermetic — no GPU, no datasets read, no submitit. Asserts:

  * `accel_pipeline` is registered in `ALL_BENCHMARKS` and `accel_formats()`
    yields its four variant keys.
  * `_bench_format_compatible` pairs `accel_pipeline` with `accel_pipeline__*`
    only (and isolates it from other accel benchmarks / non-accel benches).
  * `_env_for_format` routes the GPU variants (host-boundary, rapids)
    to `scx-bench-gpu` and the CPU variant to `scx-bench`.
  * `_per_job_slurm_params` routes the GPU variants to the preemptible GPU
    partition with `gpu:1`, and leaves the CPU variant off the GPU partition.
"""

from __future__ import annotations

import types

import pytest


# The native device-resident pipeline variant (`accel_pipeline__pyscx_gpu_resident`)
# was removed along with the device-resident fused
# UMAP path; the in-VRAM pipeline now routes to rapids-singlecell.
PIPELINE_KEYS = [
    "accel_pipeline__pyscx_cpu",
    "accel_pipeline__pyscx_gpu_hostboundary",
    "accel_pipeline__rapids_singlecell_gpu",
]
GPU_KEYS = [k for k in PIPELINE_KEYS if k != "accel_pipeline__pyscx_cpu"]


def test_registered_in_all_benchmarks():
    import benchmarks.comprehensive.benchmarks as b

    assert "accel_pipeline" in b.ALL_BENCHMARKS


def test_accel_formats_yields_pipeline_variants():
    import benchmarks.comprehensive.config as c

    keys = {f.key for f in c.accel_formats()}
    for k in PIPELINE_KEYS:
        assert k in keys, f"{k} missing from accel_formats()"


def test_bench_format_pairing_is_isolated():
    from benchmarks.comprehensive.scripts.run_parallel import (
        _bench_format_compatible,
    )

    # accel_pipeline pairs only with its own variants…
    for k in PIPELINE_KEYS:
        assert _bench_format_compatible("accel_pipeline", k)
    # …never with another accel benchmark's variants…
    assert not _bench_format_compatible("accel_pipeline", "accel_pca__pyscx_gpu_rand_hh")
    assert not _bench_format_compatible("accel_pca", "accel_pipeline__pyscx_gpu_hostboundary")
    # …and a non-accel benchmark never sees accel_pipeline keys.
    assert not _bench_format_compatible("read_full", "accel_pipeline__pyscx_cpu")


@pytest.mark.parametrize("key", GPU_KEYS)
def test_gpu_variants_route_to_gpu_env(key):
    from benchmarks.comprehensive.scripts.run_parallel import _env_for_format

    assert _env_for_format(key) == "scx-bench-gpu"


def test_cpu_variant_routes_to_default_env():
    from benchmarks.comprehensive.scripts.run_parallel import _env_for_format

    assert _env_for_format("accel_pipeline__pyscx_cpu") == "scx-bench"


def _fake_args():
    return types.SimpleNamespace(
        cpus=4, mem_gb=8, timeout=30, partition="preemptible", scale_factor=1.0,
    )


@pytest.mark.parametrize("key", GPU_KEYS)
def test_gpu_variants_get_gpu_partition_and_gres(key):
    from benchmarks.comprehensive.scripts.run_parallel import _per_job_slurm_params

    params = _per_job_slurm_params(
        _fake_args(), "pbmc3k", key, "accel_pipeline",
    )
    assert params["slurm_partition"] == "preemptible"
    assert params["slurm_gres"] == "gpu:1"


def test_cpu_variant_no_gpu_gres():
    from benchmarks.comprehensive.scripts.run_parallel import _per_job_slurm_params

    params = _per_job_slurm_params(
        _fake_args(), "pbmc3k", "accel_pipeline__pyscx_cpu", "accel_pipeline",
    )
    # CPU variant must not request a GPU gres (would fail QOS on cpu_high_mem).
    assert params["slurm_gres"] == ""


# ---------------------------------------------------------------------------
# V3 task 2.9 — per-op rapids-singlecell competitor variants
# ---------------------------------------------------------------------------

# (benchmark, format_key) for each per-op rapids competitor.
RAPIDS_PER_OP = [
    ("accel_pca", "accel_pca__rapids_singlecell_gpu"),
    ("accel_knn", "accel_knn__rapids_singlecell_gpu"),
    ("accel_umap", "accel_umap__rapids_singlecell_gpu"),
    ("accel_leiden", "accel_leiden__rapids_singlecell_gpu"),
    ("accel_preprocess", "accel_preprocess__rapids_singlecell_gpu"),
    ("accel_hvg", "accel_hvg__rapids_singlecell_gpu"),
]


def test_accel_formats_yields_per_op_rapids_variants():
    import benchmarks.comprehensive.config as c

    keys = {f.key for f in c.accel_formats()}
    for _, key in RAPIDS_PER_OP:
        assert key in keys, f"{key} missing from accel_formats()"


@pytest.mark.parametrize("bench,key", RAPIDS_PER_OP)
def test_per_op_rapids_routes_to_gpu_env(bench, key):
    from benchmarks.comprehensive.scripts.run_parallel import _env_for_format

    assert _env_for_format(key) == "scx-bench-gpu"


@pytest.mark.parametrize("bench,key", RAPIDS_PER_OP)
def test_per_op_rapids_gets_gpu_partition_and_gres(bench, key):
    from benchmarks.comprehensive.scripts.run_parallel import _per_job_slurm_params

    params = _per_job_slurm_params(_fake_args(), "pbmc3k", key, bench)
    assert params["slurm_partition"] == "preemptible"
    assert params["slurm_gres"] == "gpu:1"


@pytest.mark.parametrize("bench,key", RAPIDS_PER_OP)
def test_per_op_rapids_pairs_only_with_its_benchmark(bench, key):
    from benchmarks.comprehensive.scripts.run_parallel import _bench_format_compatible

    assert _bench_format_compatible(bench, key)
    # A rapids key never pairs with a different accel benchmark.
    other = "accel_umap" if bench != "accel_umap" else "accel_pca"
    assert not _bench_format_compatible(other, key)
