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
    # Out-of-core peak-RSS boundary: scx streaming (bounded) vs a competing
    # format's whole-matrix-resident materialize. Cross-format; true-peak RSS
    # sampler. Shows scx-flat vs the other format's linear peak RSS.
    "ooc_rss_boundary",
    "ml_loader",
    # Data-load Phase 0 — honest out-of-core sequential-loader throughput with
    # a page-cache drop before every timed epoch (unlike ml_loader's warm,
    # page-cache-resident number). SCX + TileDB-SOMA-ML + AnnLoader + annbatch +
    # scDataset; reports samples/s + epoch wall-time + true-peak RSS. Gated to
    # SUPPORTED_FORMATS inside the module.
    "ooc_loader",
    # Correctness validation — scanpy / backed / preprocessing parity.
    # SCX-only (gated on format_variant.key == "scx_auto" inside the module).
    "correctness",
    # Codec write→read round-trip parity. Compares the
    # SCX-materialised matrix against the source h5ad to catch silent
    # value-corruption regressions in any of the 6 SCX codecs. Format-gated
    # to scx_* variants inside the module.
    "roundtrip",
    # Cell-eval / arc-bench parity perf — SCX-only (gated on scx_auto).
    # Requires the scx-bench-eval conda env (cell_eval / arc_bench / pdex).
    "cell_eval_parity_perf",
    # Doublet-caller interop: export per batch → run scDblFinder / Scrublet in
    # their own conda envs → doublet_import → doublet_consensus → score. Two
    # tiers of metric: a deterministic round-trip check (the tool's own numbers
    # must survive the key-joined import, and that gates) plus agreement and
    # injected-truth accuracy (calibrated floors). SCX-only, and gated on a
    # DOUBLET_SPEC entry inside the module. See doublet_interop.py.
    "doublet_interop",
    # SCX-only fragment / manifest operations
    "fragment_ops",
    # SCX-only CSC sidecar build. The one operation in the suite with a
    # *declared* memory contract — `build_csc(memory_limit=...)`. It used to
    # breach it by 3.6x at census_1m and the threshold was deferred for that
    # reason; the streaming builder closed it, and
    # `peak_over_memory_limit__build_csc` is floored in thresholds.yaml now.
    # Also emits `n_csc_shards__build_csc`, because a memory bound can be met
    # by regressing to narrow shards and that would fix nothing.
    "build_csc",
    # SCX-only MatrixMarket export + ingest (`pyscx.to_mtx` / `from_mtx`).
    # Neither direction had a benchmark. Also carries the `mtx_header_integer`
    # contract: an integral matrix must declare `integer`, not `real`, in the
    # `%%MatrixMarket` banner — deriving that flag from a shard's value encoding
    # rather than from the values would flip every h5ad-sourced file.
    "mtx_export",
    # SCX-only grouped sharding: sort --group-by + convert-time grouping
    # (gated on scx_auto + a GROUP_SPEC dataset entry inside the module).
    "grouped_sort",
    # Cross-format grouped sharding head-to-head: scx vs a competing grouped
    # write + per-perturbation read_group (gated on the format pair +
    # a GROUP_SPEC dataset entry inside the module).
    "grouped_read",
    # IndexPlanDataset throughput (plan-driven paired reads).
    # SCX-only (gated on scx_auto inside the module).
    "index_plan",
    # Data-load Phase 0 — S=64 covariate-grouped cell-set gather throughput
    # (the STATE/STATE3 hot path, distinct from i.i.d. batch rate) via
    # SparseCellSetDataset. SCX-only.
    "cellset_gather",
    # Data-load Phase 0 — per-file obs-open cost microbench (matrix-free SCX
    # read_obs vs anndata eager-obs baseline) at 1-file + manifest scale.
    "obs_open",
    # Streaming vs materialising h5ad -> SCX conversion. The streaming
    # converter's contract is bounded peak RSS, gated by a
    # `streaming_peak_rss_mb` absolute floor on census_1m in thresholds.yaml.
    # `SUPPORTED_FORMATS = {"scx_auto"}` inside the module; in run_parallel's
    # `_NO_CONVERSION` because it works from `dataset.h5ad_path`.
    #
    # Registered here in Phase 5c. Before that it lived only in
    # `scripts/run_all.py::AVAILABLE_BENCHMARKS`, so `capture_baseline.py` and
    # `run_parallel.py` — which both derive their lists from ALL_BENCHMARKS —
    # could not schedule it and `gate_candidate.py` refused the name outright.
    # A never-run triple is silently scoped out of the floor check rather than
    # failed, so the floor read as coverage and provided none.
    "conversion_streaming",
    # The inverse: streaming vs materialising SCX -> h5ad / h5mu export. Same
    # bounded-peak-RSS contract, same floor, same registration history.
    # NOT in `_NO_CONVERSION`: it consumes a pre-converted SCX file via
    # `converted_path` from Phase A.
    "export_streaming",
    # Data-load Phase 1D — `scx sort --shuffle` global pre-shuffle: rewrite cost,
    # the output-size delta swept across codec variants (the open question:
    # scx1 codes row-independently and should be neutral, zstd/shufdelta should
    # grow), cache-cold TrainingDataset throughput shuffled vs not, and an
    # analytic batch-mixing metric. SCX-only; builds its own outputs, so it is
    # in run_parallel's _NO_CONVERSION.
    "shuffle_layout",
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
    "accel_harmony",
    "accel_preprocess",
    "accel_hvg",
    # Community analytical workflows.
    # Both are accel-shaped: variants live in the module as
    # `<bench>_variants()` and are picked up by `config.accel_formats()`, so
    # `run_parallel` pairs each only with its own `<bench>__*` keys.
    #
    # accel_qc_filter: the fused one-row-pass + one-column-pass QC kernel and
    # atomic filter_cells / filter_genes vs scanpy's seven separate passes.
    # Profiled in Phase 4.1, never gated against scanpy until now.
    "accel_qc_filter",
    # accel_score_genes: streaming gene-set scoring over the decode-prefetch
    # engine, on a backed `X` where `sc.tl.score_genes` raises outright. Three
    # method arms (scanpy parity / mean / zscore) plus the scanpy comparator.
    "accel_score_genes",
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
    # Perturbation-evaluation metrics (cell-eval / arc-bench parity), CPU vs GPU.
    # Registered in Phase 5c alongside the two streaming benchmarks: its variants
    # were already in `config.accel_formats()`, it was already in run_parallel's
    # `_NO_CONVERSION`, and it already had two `*_route_gpu_correct` floors in
    # thresholds.yaml — every part of the wiring except the one list that
    # schedules it. Those floors have therefore never actually been evaluated.
    "accel_eval_metrics",
    # GPU pseudobulk NB-GLM DE (route `gpu_nb_glm_csr`). Synthetic stratified
    # Perturb-seq fixture (`_pert_synth.make_raw_counts_stratified`); CPU vs GPU
    # with route correctness, CPU↔GPU concordance, and a pdex_ref anchor. Runs
    # only on synthetic datasets (`nb_glm_synth`). See accel_de_nb_glm.py.
    "accel_de_nb_glm",
    # rscx accelerator route-metadata gate (ORG-10.16-3b): shells an R probe
    # into the rscx conda env and floors the stamped routes — the R analogue
    # of the `*_route_gpu_correct` family. Synthetic in-process fixtures.
    "accel_r_route",
    # to_gpu_anndata device-decode route + parity.
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
    # The "laptop test": the full 9-stage
    # scverse pipeline (open → QC → filter → normalize → log1p → HVG → PCA →
    # kNN → UMAP + Leiden → Wilcoxon DE) run end to end under a fixed 16 / 32 GB
    # ceiling, pyscx backed-streaming vs scanpy in-memory. Every component is
    # benchmarked in isolation already; the composite claim is not.
    #
    # Structurally an accel benchmark — self-contained, never touching the
    # `runners/*_runner.py` contract surface — but without the `accel_` prefix,
    # exactly like `bench_csc_dispatch`. Both are named in
    # `run_parallel._ARM_SHAPED_BENCHMARKS`, which drives the format pairing,
    # the format-pool trigger and the smoke-gate exclusion from one place.
    # Miss that and it either pairs with every format in the tier or schedules
    # zero cells.
    "pipeline_ooc_constrained",
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
    # Atlas-scale multimodal streaming + in-decode dtype narrowing
    # The existing multimodal benchmarks run
    # on 5.2K / 11.9K-cell fixtures; `to_mudata(data_dtype=...)`'s 2-4x cut in
    # value-buffer bytes has never been shown at the >=500K scale where
    # MuData's all-f32 buffers are the actual problem.
    #
    # In `_MULTIMODAL_BENCHMARKS`, and its key prefix is in
    # `_MULTIMODAL_FORMAT_PREFIXES`, so `_triple_compatible`'s three-way XOR
    # holds. Its 500K / 1M fixtures are built in Phase 4.1; until then it runs
    # only on cite_seq_pbmc / multiome_pbmc.
    "multimodal_atlas_streaming",
]


__all__ = ["ALL_BENCHMARKS"]
