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
  * `SUPPORTED_DATASETS` matches the scx1 arm's declared dataset set.

The arm-specific assertions (three variants, per-arm fixture/codec/env) live in
`test_gpu_arm_labels.py`.
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


def test_supported_datasets_is_the_scx1_arms_scope():
    """`SUPPORTED_DATASETS` and the scx1 arm's `datasets` are the same object.

    This assertion used to pin the pair {pbmc3k, tabula_sapiens_100k} and had
    been failing on `main` since the set was widened to the census tiers — a
    stale test asserting a scope the code had already left. Pinning the literal
    again would just re-arm the same trap, so it now pins the *relationship*
    that has to hold: `run()` scopes each arm by `arm.datasets`, and the scx1
    arm's is the module-level set the rest of the suite reads.
    """
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import (
        ARMS,
        SUPPORTED_DATASETS,
    )

    assert ARMS[_VARIANT_KEY].datasets == SUPPORTED_DATASETS
    # The gate tiers are in scope whatever else has been added.
    assert {"pbmc3k", "tabula_sapiens_100k"} <= SUPPORTED_DATASETS
