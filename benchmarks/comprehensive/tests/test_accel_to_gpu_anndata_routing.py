"""
Registration + SLURM routing for the `accel_to_gpu_anndata` device-decode
benchmark.

Hermetic — no GPU, no datasets read, no submitit. Asserts:

  * `accel_to_gpu_anndata` is registered in `ALL_BENCHMARKS` and
    `accel_formats()` yields its single variant key.
  * `_bench_format_compatible` pairs `accel_to_gpu_anndata` with
    `accel_to_gpu_anndata__*` only (and isolates it from other accel
    benchmarks / non-accel benches).
  * `_env_for_format` routes the GPU variant to `scx-bench-gpu`.
  * `SUPPORTED_DATASETS` is scoped to pbmc3k + tabula_sapiens_100k.
"""

from __future__ import annotations

import pytest


_VARIANT_KEY = "accel_to_gpu_anndata__scx1_gpu"


def test_registered_in_all_benchmarks():
    import benchmarks.comprehensive.benchmarks as b

    assert "accel_to_gpu_anndata" in b.ALL_BENCHMARKS


def test_accel_formats_yields_variant():
    import benchmarks.comprehensive.config as c

    keys = {f.key for f in c.accel_formats()}
    assert _VARIANT_KEY in keys, f"{_VARIANT_KEY} missing from accel_formats()"


def test_bench_format_pairing_is_isolated():
    from benchmarks.comprehensive.scripts.run_parallel import (
        _bench_format_compatible,
    )

    # accel_to_gpu_anndata pairs only with its own variant…
    assert _bench_format_compatible("accel_to_gpu_anndata", _VARIANT_KEY)
    # …never with another accel benchmark's variant…
    assert not _bench_format_compatible(
        "accel_to_gpu_anndata", "accel_pca__pyscx_gpu_rand_hh"
    )
    assert not _bench_format_compatible("accel_pca", _VARIANT_KEY)
    # …and a non-accel benchmark never sees the key.
    assert not _bench_format_compatible("read_full", _VARIANT_KEY)


def test_gpu_variant_routes_to_gpu_env():
    from benchmarks.comprehensive.scripts.run_parallel import _env_for_format

    assert _env_for_format(_VARIANT_KEY) == "scx-bench-gpu"


def test_supported_datasets_scoped_to_gate_tiers():
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import (
        SUPPORTED_DATASETS,
    )

    assert SUPPORTED_DATASETS == frozenset({"pbmc3k", "tabula_sapiens_100k"})
