"""
Benchmark modules for the comprehensive benchmark suite.

Each module exposes a ``run(dataset, format_variant, n_runs, cold_cache, ...)``
function returning a ``BenchmarkResult``. The canonical ordering for runs is
``ALL_BENCHMARKS`` below — both ``run_parallel.py`` and ``capture_baseline.py``
import this list so adding a new benchmark only requires updating one place.

Entries that don't apply to a given ``(dataset, format)`` triple return ``None``
from their ``run()`` and the orchestrator silently skips them (see
``fragment_ops.py`` / ``cloud_push.py`` for the SCX-only pattern).
"""

# Ordering matters: benchmarks listed earlier are the ones most readers care
# about first, so when capture_baseline or run_parallel shows partial progress
# the informative numbers surface quickly. Cross-format comparisons first,
# SCX-only ops next, cloud last.
ALL_BENCHMARKS: list[str] = [
    # Cross-format comparisons
    "compression",
    "write",
    "read_full",
    "read_selective",
    "parallel_scaling",
    "parallel_write_scaling",
    "memory",
    "ml_loader",
    # Correctness validation — scanpy / backed / preprocessing parity.
    # SCX-only (gated on format_variant.key == "scx_auto" inside the module).
    "correctness",
    # SCX-only fragment / manifest operations
    "fragment_ops",
    # Cloud (GCP) — Phase C through F
    "cloud_push",
    "cloud_pull",
    "cloud_read",
    "cloud_metadata",
    "cloud_filtered",
    "cloud_reader_vs_pull",
    "cost_model",
    "cloud_large_atlas",
    # Accelerator benchmarks
    # These don't vary by file format; each accelerator module expands
    # internally into several implementation variants (one `FormatVariant`
    # slot per impl, e.g. accel_pca__scanpy_cpu vs accel_pca__pyscx_gpu_cov).
    # See `config.accel_formats()`.
    "accel_pca",
    "accel_knn",
    "accel_umap",
    "accel_leiden",
    "accel_preprocess",
    "accel_hvg",
]


__all__ = ["ALL_BENCHMARKS"]
