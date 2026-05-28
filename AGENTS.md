# AGENTS.md

## Project Overview

SCX (Sparse Cell eXpression System) is a purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. It replaces AnnData/h5ad with a unified Rust-native stack: GPU-saturating training loader, lazy query engine, and Rust-native analysis accelerators. Native Python (`pyscx`) and R (`rscx`) bindings; compatible with scverse and Seurat v5. See [docs/performance.md](docs/performance.md) for benchmark results.

## Key Documents

- **[ROADMAP.md](ROADMAP.md)** — Capability tiers and status.
- **[docs/architecture.md](docs/architecture.md)** — Crate graph, feature flags, file format overview, codec system, data model, reader/writer architecture.
- **[docs/format.md](docs/format.md)** — Binary format reference: file header, catalogs, CSR shard layout, fragment/manifest model, checksums.
- **[docs/codec.md](docs/codec.md)** — Bit-level codec spec: Delta-Golomb-Rice, FOR-BP, Rice, LZ4+shuffle, auto-selection.
- **[docs/api.md](docs/api.md)** — API reference and section type documentation.
- **[docs/scanpy.md](docs/scanpy.md)** — Scanpy integration guide and accelerator usage.
- **[docs/performance.md](docs/performance.md)** — Benchmark results and performance characteristics.
- **[docs/gpu-setup.md](docs/gpu-setup.md)** — GPU setup: CUDA, RAPIDS, conda, container, SLURM, troubleshooting.
- **[docs/development.md](docs/development.md)** — Developer build guide: CPU-only, HDF5, cloud, GPU, Python, R builds; test matrix; fuzzing.
- **[docs/testing.md](docs/testing.md)** — Test, benchmark, and correctness validation details.
- **[docs/multithreading.md](docs/multithreading.md)** — Multithreading architecture across crates.
- **[docs/sharding.md](docs/sharding.md)** — Sharding design and usage.
- **[docs/multimodal.md](docs/multimodal.md)** — Multimodal (CITE-seq / Multiome / TEA-seq / spatial) layout and APIs.
- **[docs/cloud.md](docs/cloud.md)** — Cloud auth, layouts, tuning, provider-specific notes.
- **[docs/conventions.md](docs/conventions.md)** — Coding conventions (serialization, error handling, checksums, language binding rules, GPU/accel constraints).
- **[docs/compatibility-matrix.md](docs/compatibility-matrix.md)** — Tested vs. declared Python / numpy / scipy / pyarrow / anndata / scanpy combinations for `pyscx`.
- **[benchmarks/README.md](benchmarks/README.md)** — Practical guide: SLURM job submission, dataset prep, [Regression Gating](benchmarks/README.md#regression-gating) (`gate_candidate.py` against `results/baselines/LATEST`), and the [GPU accelerator regression workflow](benchmarks/README.md#gpu-accelerator-regression-workflow). **Always use parallel SLURM job submission** (one job per benchmark × dataset pair). No CI-side gate today — local gate is the canonical signal.
- **[.claude/skills/scx-dev/SKILL.md](.claude/skills/scx-dev/SKILL.md)** — Release workflow (pyscx + scx-cli tag-prefix scheme, version-bump scope, pre-release checks) and dev-env quick reference. Invoke when cutting a release or bumping versions.

## Build and Test

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check

# With cloud features:
cargo test --workspace --features cloud

# Python bindings (always use uv venv at .venv/):
cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v

# Python bindings with cloud or GPU:
cd pyscx && ../.venv/bin/maturin develop --features cloud,gpu && ../.venv/bin/pytest tests/ -v

# R bindings:
cd rscx && R CMD INSTALL .
```

**Always use the uv venv** at `.venv/` for all Python work. Do NOT use system Python or pip directly.

## Architecture (summary)

15 workspace crates; the dependency core is `scx-codec → scx-format → {scx-mtx, scx-ops, scx-engine, scx-loader, scx-cloud, scx-gpu, scx-accel, scx-convert} → {scx-cli, pyscx, rscx}`. Full graph and isolation rules in [docs/architecture.md § Crate Dependency Graph](docs/architecture.md#crate-dependency-graph). Feature flags: `scx-cli{hdf5,cloud}`, `scx-convert{hdf5}`, `pyscx{cloud,gpu}`, `scx-gpu{gds}`, `scx-accel{gpu}` — all opt-in.

File format ([docs/format.md](docs/format.md)): 256-byte LE header (magic `b"SCX\x01"`), 4096-byte root catalog at offset 256, 8-byte-aligned sections (26 types, IDs 0–25), full catalog at EOF with per-entry checksums + shard statistics. 76-byte CSR shard headers (magic `b"SCXS"`). Codecs: `None`, `Scx1` (Delta-Golomb + FOR-BP + Rice, integer only), `Zstd`, `Lz4Shuffle`; auto-codec selects Scx1 vs Zstd by median value. On-disk `u64`/`u16-u32`/`u8-u32`, in-memory `i64`/`i32`/`f32` to match scipy CSR zero-copy.

## Capabilities (summary)

- **Analysis accelerators** (`scx-accel`, [docs/scanpy.md § Rust-native accelerators](docs/scanpy.md#rust-native-accelerators)): PCA (covariance / randomized auto-routed), kNN (HNSW), UMAP, differential expression (pre-ranking Wilcoxon), Leiden (Rust-native CPU + cuGraph GPU), Harmony2 batch integration, LISI, HVG (streaming seurat_v3 / seurat), pseudobulk, perturbation evaluation metrics (cell-eval / arc-bench parity).
- **GPU acceleration** (`scx-gpu`, [docs/gpu-setup.md](docs/gpu-setup.md)): `device="auto"|"cpu"|"gpu"|"gpu:N"` selector across `pyscx.accel.*`. GPU PCA (cuSPARSE + cuBLAS, covariance ≤ 8000 vars / randomized > 8000), GPU kNN (cuVS CAGRA), GPU UMAP, GPU Leiden (cuGraph), GPU preprocessing (`normalize_total`/`log1p`/`highly_variable_genes`). Double-buffered shard I/O; graceful CPU fallback.
- **Lazy preprocessing**: `ScxLazyTransformedDataset` chains `normalize_total → log1p → row_scale` without materialization; `ShardSource` enables streaming PCA. Supports a full out-of-core pipeline (open → QC → normalize → log1p → HVG → PCA → kNN → UMAP → Leiden). See [docs/performance.md](docs/performance.md) for memory and throughput figures.
- **ML training loader**: Triple-buffered Rust pipeline (tokio I/O → rayon decode → Python consumer). Zero Python on the hot path. See `TrainingPipeline` in [docs/api.md](docs/api.md).
- **Multimodal**: CITE-seq / 10x Multiome / TEA-seq via the v2 format, including streaming h5mu ingest (`pyscx.from_h5mu` / `scx convert --from h5mu --stream` with `--modalities`/`--modality-types`), multimodal `scx merge`/`compact`/`subset --modality + --filter`, and per-modality CSC preservation on `scx append`. See [docs/multimodal.md](docs/multimodal.md).
- **Streaming ingest** (`scx-convert/`): CSR / dense / CSC-on-disk h5ad and h5mu all stream into SCX with bounded peak RSS. `obsm` / `varm` / `obsp` / `varp` are hyperslab-read one row-range at a time and emitted as row-sharded sections (`<section>/<name>_shard_<idx>`, section types 20–23) so peak memory per metadata matrix matches one X shard; the pyscx backed-routing path detects per-section mutation via top-level h5py key comparison and routes clean sections through the disk streamer while still preserving in-Python edits to mutated ones. CLI / pyscx kwargs: `--memory-budget`, `--strict-uns`, `--dense-zero-epsilon`, `--temp-dir`, plus h5mu `--modalities`/`--modality-types`. Structured `ConvertWarning` channel; provenance entry records per-category counts. See [docs/api.md § Conversion warnings](docs/api.md#conversion-warnings-convertwarning) and [Memory budgets](docs/api.md#memory-budgets).
- **Parallel streaming reader (ingest)** (`scx-convert/src/pipeline.rs::run_streaming_writer_coordinator`): h5ad CSR / dense / in-memory CSC ingest fans shard reads out across a rayon worker pool with a bounded crossbeam reorder buffer; output is byte-identical to the sequential path. Default `reader_threads = None` resolves to `RAYON_NUM_THREADS` or `available_parallelism()`. Gated by a runtime `H5is_library_threadsafe` probe — non-threadsafe libhdf5 (e.g. statically linked) falls back to the sequential coordinator with a one-shot `ConvertWarning::Hdf5NotThreadsafe`. `--memory-budget` derates worker count to fit `shard_target_rows × n_vars × density × 16 bytes` per worker; refuses to start when a single shard exceeds the budget. CLI / pyscx kwargs: `--reader-threads`, `--writer-queue-depth`. CSC external-memory transpose and h5mu cross-modality stay sequential by design.
- **Streaming export** (`scx-convert/src/h5ad_stream_write.rs`): SCX → h5ad / h5mu via pre-allocated HDF5 hyperslab writes. CLI `scx convert --to h5ad/h5mu` and `pyscx.to_h5ad` / `pyscx.to_h5mu` stream by default; peak RSS bounded by one shard's worth of CSR per matrix written. Sharded obs/var (`ObsMetadataShard` / `VarMetadataShard` sources) also stream via `write_dataframe_group_streaming` in `h5ad_write.rs` — pre-allocated HDF5 datasets per column, hyperslab writes per shard, with a single-pass running global dictionary that unifies disjoint per-shard categorical vocabularies. Deletion vectors handled via a single-pass pre-scan so on-disk layout stays deterministic; the streaming obs writer filters by the same global keep mask as `/X` (closes a pre-existing obs row-count mismatch on the legacy eager fallback). Legacy single-section obs/var sources transparently fall through to the eager writer. `--stream=false` / `stream=False` falls back to the legacy materialising path.
- **Parallel streaming reader (export)** (`scx-convert/src/h5ad_stream_write.rs::stream_csr_to_group_at`): SCX → h5ad / h5mu shard decode + filter runs on a rayon worker pool; the HDF5 writer stays on the calling thread (libhdf5 holds its own global lock, but workers never touch HDF5 here). No `H5is_library_threadsafe` probe required and no density heuristic — per-shard memory comes from exact `FullCatalogEntry::stats.nnz` and row range. `pyscx.to_h5ad` / `pyscx.to_h5mu` accept `reader_threads=`, `writer_queue_depth=`, `memory_budget=` kwargs symmetric with `from_h5ad` / `from_h5mu`. Same rolling-window spawn cap as ingest (`reader_threads + writer_queue_depth`); same `ReaderThreadsDerated` warning when `memory_budget` shrinks the granted thread count.
- **Streaming SCX → SCX rewrite**: `pyscx.from_anndata(adata, out)` accepts `adata.X` as `ScxBackedSparseDataset` or `ScxLazyTransformedDataset` and streams shard-by-shard. Byte-passthrough fast path when source/target shard layouts agree (matching `shard_size`/`codec`, no deletions or column projection, single-modality source); decode-encode fallback otherwise. Lazy `X` always decode-encodes with transforms applied per shard. Provenance entry records `x_source` / `passthrough` / `lazy_transforms`. See [docs/api.md § `pyscx.from_anndata` — backed and lazy `X`](docs/api.md#pyscxfrom_anndata--backed-and-lazy-x).
- **Streaming merge / append** (`scx-ops/`): `pyscx.merge` and `scx merge` operate shard-by-shard at every level — obs metadata (via `ObsMetadataShard` / `VarMetadataShard` section types 24–25), X / layer CSR shards, and obsm / varm dense mapping shards are never fully materialized. Removes the previous Arrow IPC 2 GB narrow-offset ceiling on obs string columns. Var identity is validated by default (`assume_identical_var=False`); `UnsPolicy` controls conflict resolution (`first` / `require-equal` / `namespace` / `summary`). `pyscx.append` writes new obs as shards (existing obs is not rewritten). Predicate indexes are built incrementally from the obs shard stream. `pyscx.from_anndata` emits sharded obs/var when `n_obs > shard_target_rows`; `force_legacy_metadata=True` opts out. Obsm / varm / obsp / varp are extracted and written one key at a time. See [docs/scanpy.md § Merging datasets](docs/scanpy.md#merging-datasets) and [docs/sharding.md § Obs/var metadata sharding](docs/sharding.md#obsvar-metadata-sharding).
- **Query-ready conversion**: `scx convert --index-obs/--index-var/--index-preset {cellxgene,perturbseq,training}` materialises predicate indexes at write time so `pyscx.open(...).query()` and `scx pull --filter` push predicates down. `--bitmap auto|always` writes per-shard gene→local-row roaring bitmap sidecars (`SCXB`, section id 6) consumed by `PyExperiment.detection_counts` / `cells_expressing`. See [docs/format.md § 12](docs/format.md#12-detection-bitmap-optional) for the wire format.
- **Cloud-native query**: `pyscx.open_cloud(url).query().filter_obs(...).select_genes(...).collect()` and `scx query <url-or-dir>` serve selective reads directly over `object_store` range reads without `scx pull`. Shared `SectionReader` trait unifies local mmap and cloud paths; `QueryPipeline::from_reader` is the seam. Deferred follow-ons: `CloudQueryOptions`, `pyscx.read_cloud(...)` flat helper, batched async fetcher. See [docs/cloud.md § Cloud-native query](docs/cloud.md#cloud-native-query).

## Coding Conventions

See **[docs/conventions.md](docs/conventions.md)** for the full ruleset. Highlights:

- **Tracked files MUST NOT reference gitignored markdown** (`tasks/*.md`, root scratch docs like `2026-*_REGRESSIONS.md`, `*_CODE-REVIEW.md`, `Phase*.md`). Inline the substance instead of citing.
- **Do NOT stage/commit all-caps markdown files in the repo root** (e.g., `MERGE-OBS-OFFSET-OVERFLOW.md`, `STATE-CELL-EVAL-INTEGRATE-PT2.md`). These are ephemeral implementation/planning documents that are moved to `./tasks/` after completion, and `./tasks/` is gitignored. Exceptions: `README.md`, `ROADMAP.md`, `AGENTS.md`, `CLAUDE.md`.
- **On-disk structs**: no `#[repr(C)]`; serialize field-by-field with `byteorder` little-endian; sections at 8-byte-aligned offsets.
- **Errors**: `thiserror` enums per crate; readers return errors (not panic) on malformed input.
- **Checksums**: BLAKE3 everywhere (truncated-64 per shard, full-256 for catalog).
- **Writer**: temp file → `fsync` → `rename`; sections start at offset 4352.
- **pyscx**: `Bound<'py, T>` API, `i64/i32/f32` matches scipy.

## Known Risks and Pitfalls

- **HDF5 crate (`hdf5-rust`)** is unmaintained; only needed for conversion. Fallback: Python subprocess with h5py.
- **h5ad files are messy** (missing encoding-type attrs, CSC instead of CSR, dense X, pickled uns) — handle gracefully.
- **tokio + rayon interaction** (`scx-loader`): tokio for I/O only, rayon for CPU; bounded channels for back-pressure.
- **Cloud auth**: `object_store` handles credentials via env vars and instance metadata; no custom auth code.
- **SLAF runner quirks** (extending `slaf_runner.py`): `SLAFArray.get_submatrix` treats integer selectors positionally and returns string `cell_id` / `gene_id` columns that are NOT unique across Lance fragments at census scale — the runner goes through the internal `expression(cell_integer_id, gene_integer_id, value)` SQL table instead. `get_submatrix` rejects numpy arrays. `SLAFDataLoader`'s Mixture-of-Scanners prefetcher returns 0 batches on Census 10M with the default config — upstream issue, not harness.

## Known Limitations

- **GPU pipeline speedup**: 10× target not yet met. Routing details and current numbers: [benchmarks/README.md § GPU accelerator regression workflow](benchmarks/README.md#gpu-accelerator-regression-workflow); see [docs/performance.md](docs/performance.md) for per-op speedups.
- **GPU Leiden correctness**: label stability differs from Python `leidenalg` — documented behaviour of `device="gpu"` / `"auto"` on GPU hosts, not a regression. Pin `device="cpu"` to preserve label stability for downstream DE / annotation transfer.
- **Benchmark baseline scope**: capture and gate coverage differ. `LATEST` is the canonical baseline; prior baselines remain under `results/baselines/` for historical A/B. Justifications suppress both regression and absolute-floor violations on a triple. Deferred floors are catalogued in `benchmarks/comprehensive/thresholds.yaml` § "Deferred floors": SLAF `ml_loader` (separate conda env), `avg_gpu_util_pct__gpu_train` (needs in-process sampler), `census_1m × cell_eval_parity_perf` (cell-eval reference is O(N²)).
- **Catalog version auto-upgrade**: `FullCatalog::write_to` auto-upgrades `catalog_version` to ≥2 on serialise to match the v2 stats layout that `ShardStats::write_to` always emits. Closes the cloud-fixture symmetry bug class. Regression test `scx-format::catalog::tests::write_upgrades_v1_catalog_to_v2`.
- **Fixture refresh**: `benchmarks/scripts/reconvert_fixtures.py` is the canonical entry point when h5ad sources change; `gate_candidate.py --probe-cloud` catches missing/mismatched `gcsfs` and catalog-format bugs in <30 s.
- **CSC storage**: optional gene-major sidecar via `pyscx.from_anndata(csc="always")` / `scx convert --csc=always` / `scx build-csc`; multi-shard layouts via `--csc-cols-per-shard`. Consumers opt-in with `prefer_format="csc"` on supported `pyscx.accel.*` ops (PCA explicitly rejects CSC). Mutating ops drop the sidecar by default with a warning; `--rebuild-csc` to re-emit. See [docs/sharding.md § CSC sharding](docs/sharding.md#csc-sharding).
- **Multimodal limitations**: `scx merge`, `scx compact`, per-modality CSC sidecar preservation on `scx append`, and `scx subset --modality NAME --filter/--genes` composition all shipped — `scx-ops::{merge_multimodal, compact_multimodal}` dispatch on multimodal inputs and `scx info` exposes a per-modality `has_csc` column. Remaining limitations: cloud `open_cloud(...).to_mudata(backed=True)` (local backed multimodal works), multimodal-aware predicate pushdown via `QueryPipeline` (use `scx subset --modality NAME --filter` instead), and spatial transcriptomics with R-tree spatial index (separate spec).
- **GDS**: GPUDirect Storage requires local NVMe + nvidia-fs drivers + ext4/XFS; always falls back to the CPU path.

