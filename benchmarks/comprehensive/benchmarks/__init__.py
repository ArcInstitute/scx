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
    "read_scattered",
    "parallel_scaling",
    "parallel_write_scaling",
    "memory",
    # Streaming vs in-memory row-iteration head-to-head (Phase 6b).
    # Gated on `"backed_mode"` capability inside the module; only
    # formats with a true streaming path (SCX today) record runs —
    # others skip via NotImplementedError → None.
    "read_streaming_vs_inmemory",
    # Out-of-core peak-RSS boundary: scx streaming (bounded) vs shardad
    # materialize (whole matrix resident). Cross-format {scx_auto, shardad};
    # true-peak RSS sampler. Shows scx-flat vs shardad-linear peak RSS.
    "ooc_rss_boundary",
    "ml_loader",
    # Correctness validation — scanpy / backed / preprocessing parity.
    # SCX-only (gated on format_variant.key == "scx_auto" inside the module).
    "correctness",
    # Codec write→read round-trip parity. Compares the
    # SCX-materialised matrix against the source h5ad to catch silent
    # value-corruption regressions in any of the 6 SCX codecs. Format-gated
    # to scx_* variants inside the module.
    "roundtrip",
    # Shardad read-back fidelity (source h5ad -> .shad -> to_anndata parity +
    # dtype/materialization knobs). shardad-only; roundtrip/correctness are
    # SCX-codec-specific so shardad needs its own fidelity gate.
    "shardad_fidelity",
    # Cell-eval / arc-bench parity perf — SCX-only (gated on scx_auto).
    # Requires the scx-bench-eval conda env (cell_eval / arc_bench / pdex).
    "cell_eval_parity_perf",
    # SCX-only fragment / manifest operations
    "fragment_ops",
    # SCX-only grouped sharding: sort --group-by + convert-time grouping
    # (gated on scx_auto + a GROUP_SPEC dataset entry inside the module).
    "grouped_sort",
    # Cross-format grouped sharding head-to-head: scx vs shardad grouped
    # write + per-perturbation read_group (gated on {scx_auto, shardad} +
    # a GROUP_SPEC dataset entry inside the module).
    "grouped_read",
    # IndexPlanDataset throughput (plan-driven paired reads).
    # SCX-only (gated on scx_auto inside the module).
    "index_plan",
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
    # slot per impl, e.g. accel_pca__scanpy_cpu vs accel_pca__pyscx_gpu_rand_hh).
    # See `config.accel_formats()`.
    "accel_pca",
    "accel_knn",
    "accel_umap",
    "accel_leiden",
    "accel_preprocess",
    "accel_hvg",
    # V3 task 2.7 — end-to-end PCA→kNN→UMAP residency benchmark: the fused
    # device-resident path vs the host-boundary path vs a CPU reference vs the
    # rapids-singlecell GPU competitor. Variants live in
    # `benchmarks/comprehensive/benchmarks/accel_pipeline.py`.
    "accel_pipeline",
    # PR series G1 — pdex_ref + rank_genes_groups (Wilcoxon) CPU vs GPU.
    # Both entries gained a `device=` parameter in G1; this benchmark
    # captures CPU baseline + GPU acceleration on the same fixture so
    # the gate tracks speedup and CPU↔GPU parity. Variants live in
    # `benchmarks/comprehensive/benchmarks/accel_de.py`.
    "accel_de",
    # GPU pseudobulk NB-GLM DE (route `gpu_nb_glm_csr`). Synthetic stratified
    # Perturb-seq fixture (`_pert_synth.make_raw_counts_stratified`); CPU vs GPU
    # with route correctness, CPU↔GPU concordance, and a pdex_ref anchor. Runs
    # only on synthetic datasets (`nb_glm_synth`). See accel_de_nb_glm.py.
    "accel_de_nb_glm",
    # ACC-RUST-OPT-V4 §4.4: to_gpu_anndata device-decode route + parity.
    # Self-converts the count h5ad to Scx1 and asserts the decode runs
    # fully in VRAM (transfer_mode=scx_device_decode_gpu).
    "accel_to_gpu_anndata",
    # Storage-format → GPU pipeline: `scx + rapids-singlecell` vs
    # `h5ad + rapids-singlecell`. Holds the rapids engine constant and varies
    # the load-to-GPU path (to_gpu_anndata device-decode vs read_h5ad +
    # anndata_to_GPU host bounce). See accel_format_pipeline.py.
    "accel_format_pipeline",
    # CSC dispatch sweep (Phase L.3): qc_metrics / HVG / DE /
    # pseudobulk × {csr, csc} on a CSC-equipped fixture.
    "bench_csc_dispatch",
    # Phase K — multimodal benchmarks. Both gated on
    # `dataset.multimodal == True` inside the modules; non-multimodal
    # datasets surface a clear "use compression / ml_loader instead"
    # error so the orchestrator never silently skips them.
    "multimodal_compression",
    "multimodal_training",
    # Phase 6b — multimodal streaming vs in-memory row iteration.
    # Gated on `dataset.multimodal == True` AND
    # `format_variant.key in scx_multimodal_*` inside the module.
    "multimodal_read_streaming_vs_inmemory",
]


__all__ = ["ALL_BENCHMARKS"]
