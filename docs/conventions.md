# SCX Coding Conventions

Rules that govern code in this repo. Agents and humans both follow.
For navigational summary, see [AGENTS.md](../AGENTS.md).

## Serialization (on-disk types)

- Do NOT use `#[repr(C)]` for on-disk structs. Serialize field-by-field
  with `byteorder::WriteBytesExt`/`ReadBytesExt` (little-endian).
- Every section starts at an 8-byte-aligned offset. Insert zero padding
  as needed.

## Error Handling

- Use `thiserror` for error enums: `ScxError` (scx-format),
  `EngineError` (scx-engine), `OpsError` (scx-ops),
  `LoaderError` (scx-loader), `CloudError` (scx-cloud),
  `GpuError` (scx-gpu), `AccelError` (scx-accel).
- Readers must return errors (not panic) on malformed input — especially
  bitstream exhaustion.
- Validate magic bytes, endianness, and format version on file/shard open.

## Checksums

- BLAKE3 everywhere. Per-shard: BLAKE3 truncated to 64 bits.
  Catalog: full 32-byte BLAKE3.
- Shard checksum covers everything after the shard header.
- Full catalog ends with a 32-byte BLAKE3 of all preceding catalog bytes.

## Writer (Atomic Rename)

- Write to temp file, then `fsync()` + `rename()` to final path.
- Sections start at offset 4352 (256 header + 4096 root catalog placeholder).
- `finish()`: write full catalog at EOF → pwrite root catalog at 256 →
  pwrite header at 0 → fsync → rename.

## Python Bindings (pyscx)

- PyO3 with `Bound<'py, T>` API (not deprecated `&PyAny`).
- Pin `pyo3` and `numpy` crate to the same minor version (currently 0.23).
- `PyArray::from_vec()` for zero-copy (moves Rust `Vec` to numpy).
- `ScxCsr` `i64`/`i32`/`f32` matches scipy exactly — avoids copy.
- Arrow → pandas via pyarrow's `to_pandas()` for obs/var metadata.
- Accelerators exposed via `pyscx.accel.*` — results written to standard
  AnnData slots.
- Optional Python deps (e.g. `pydeseq2`) imported at runtime with clear
  `ImportError` if missing.

## R Bindings (rscx)

- `extendr` v0.8.x for Rust ↔ R FFI.
- SCX CSR (row-major) must be transposed to dgCMatrix (CSC, column-major)
  for R/Matrix interop.
- R has no unsigned integers — use `i32` for all integer arguments from
  R, convert internally.

## GPU (scx-gpu)

- Uses `cudarc` for CUDA runtime/driver API.
- CUDA kernels compiled via `cc` build script (`build.rs`).
- GPU decoders must produce bit-identical output to the scalar CPU
  reference.
- GDS requires local NVMe + nvidia-fs drivers + ext4/XFS filesystem;
  always falls back to the CPU path.

## Accelerators (scx-accel)

- `faer` for dense linear algebra (QR, SVD, eigendecomposition in PCA).
- `instant-distance` for HNSW-based approximate kNN.
- HVG: streaming `streaming_mean_var()` and `streaming_clip_square_sum()`
  in `hvg.rs`; loess via `skmisc.loess`.
- Leiden adapted from `single-clustering` (BSD 3-Clause). Default
  `n_iterations=2` (matches the `leidenalg` package default). Uses
  `rand_chacha` for deterministic seeding and `libc::malloc_trim` on
  Linux to return freed arenas to the OS after large reductions.
- Pseudobulk aggregation streams via `BackedCsrReader`; statistical
  testing delegated to `pydeseq2`.
- **Execution route is recorded, not implicit.** DE entry points stamp an
  `AccelExecutionInfo` (route + fallback reason) onto their result via the
  single planner `scx_accel::route::plan_de_route`. GPU entry points stamp the
  authoritative route (v2 / v3 / CSC) at the kernel branch; CPU routes are
  stamped at the pyscx dispatch point. New GPU routes MUST register an
  `AccelRoute` variant and stamp it — no silent fallbacks. The route surfaces to
  Python on `adata.uns["scx_accel"]`, and any performance claim must cite the
  recorded route. `SCX_GPU_DE_V3_TRACE` is a debug-only fallback, not the signal.

## Parallel streaming reader (scx-convert)

- `run_streaming_writer_coordinator` is the entry point that dispatches
  every streaming convert (h5ad CSR, h5ad dense, in-memory CSC, h5mu
  per-modality, h5ad layers). It picks between the sequential
  `streaming_writer_coordinator` and the rayon-parallel
  `streaming_writer_coordinator_parallel` based on three preconditions:
  (1) the reader exposes `IndexedCsrShardStream::read_range` via the
  `as_indexed` trait override, (2) `H5is_library_threadsafe` reports
  the libhdf5 build is thread-safe (cached in `OnceLock`), and (3)
  `ConvertOptions::reader_threads` resolves to `> 1`. Any precondition
  failing falls back to the sequential path; the `Hdf5NotThreadsafe`
  warning is emitted at most once per process (via
  `hdf5_threadsafe::try_emit_not_threadsafe_warning`), because the
  threadsafe flag is a build-time property of libhdf5 and re-firing
  per matrix / per modality is pure noise.
- Workers funnel encoded shards through a bounded `crossbeam-channel`
  sized by `writer_queue_depth` and drain through a `BTreeMap` reorder
  buffer keyed by shard index. The writer thread (the calling thread)
  preserves shard-index order so the on-disk file layout, catalog
  offsets, and BLAKE3 checksums match the sequential output
  byte-for-byte. Outstanding shards (encoding + in channel + in
  reorder buffer) are bounded at `reader_threads + writer_queue_depth`
  via a rolling-window spawn: the coordinator primes the pool with
  that many tasks and spawns one new task per received shard. Without
  this cap, a slow shard 0 would let the BTreeMap accumulate ~all
  remaining shards.
- Reader slab caps: `IndexedCsrShardStream::max_slab_rows` reports a
  hard upper bound on rows the reader can serve in one `read_range`
  call. `None` (default) means no cap (CSR + in-memory CSC).
  `DenseXStreamReader` returns `Some(max_slab_rows)` when
  `memory_budget` shrinks its slab cap below `shard_target_rows`. The
  dispatcher clamps the partition's effective shard size by this cap
  before computing the memory-budget derate, matching the sequential
  `next_csr_shard` clamp. Silent shrinkage (no warning) is
  deliberate — the sequential path is silent too, and byte-identity
  between the paths requires they produce the same shard sizes.
- Memory derate: per-worker working-set estimate comes from
  `IndexedCsrShardStream::per_worker_bytes(effective_target,
  modality_type)`. Sparse readers (default impl) assume density 5 %
  (RNA / general) or 10 % (ATAC), times `n_vars × 16 B/nnz`.
  `DenseXStreamReader` overrides this to size the dense slab buffer
  (`effective_target × n_vars × sizeof(dtype) × 2`). Densities live
  in `PARALLEL_DENSITY_DEFAULT_DEN` / `PARALLEL_DENSITY_ATAC_DEN`;
  values err conservative because over-estimating only routes to the
  sequential coordinator. When `--memory-budget` is set, worker
  count is clamped to fit; a single shard exceeding the budget fails
  the convert with an actionable message rather than risking OOM.
- CSC external-memory transpose stays sequential (the bucket pipeline
  serialises by construction). h5mu cross-modality is sequential
  across modalities; each modality's X / layers go through the
  same dispatcher independently.
- `IndexedCsrShardStream` requires `Send + Sync`. HDF5-backed readers
  inherit thread-safety from libhdf5 (verified by the runtime probe).
  In-memory readers (e.g. `MaterializedCsrStream`) own all their state
  in `Vec`s and slice deterministically — safe by construction.
- Deferred test: forcing `hdf5_is_threadsafe() == false` to verify the
  one-shot warning fires exactly once under fallback requires a non-
  threadsafe libhdf5 build or a probe-injection seam, neither of which
  is in scope. The `OnceLock` guard is exercised indirectly by any
  multimodal h5mu run on a non-threadsafe host.

## Parallel streaming reader — export direction (scx-convert)

The SCX → h5ad / h5mu export path
(`h5ad_stream_write::stream_csr_to_group_at`) mirrors the ingest
dispatcher but with a simpler precondition set:

- **No libhdf5 thread-safety probe.** Workers only read SCX shards
  (mmap + scx-codec decode); the HDF5 writes (`indices_ds.write_slice`,
  `data_ds.write_slice`, `indptr_ds.write_slice`) stay on the calling
  thread. There is no concurrent HDF5 access on this path, so
  non-threadsafe libhdf5 builds work just as well as threadsafe ones.
- **Exact per-shard memory budget.** Every CSR shard records its
  `nnz` and row range in `FullCatalogEntry::stats` at convert time.
  `per_shard_export_bytes(stats)` computes the working set
  precisely — `nnz × 8` (indices + data) + `(n_rows + 1) × 8`
  (indptr) + `nnz × 8` (codec scratch). No density heuristic, no
  modality-type branching. The memory-budget derate constrains
  `reader_threads + writer_queue_depth` against `max_shard_bytes` so
  the rolling-window cap matches the budget directly.
- **No `max_slab_rows` clamp.** SCX shards are random-access via
  `ScxReader::read_csr_shard_for(modality_id, shard_idx)`; the
  source isn't gated by a slab budget the way an HDF5 dense reader
  is on the ingest side.
- **Rolling-window spawn carries over.** The reorder buffer
  (`BTreeMap<u32, DecodedShard>`) on the writer thread is bounded
  by the same `reader_threads + writer_queue_depth` window so a
  slow shard 0 can't accumulate the rest of the file in memory.
- **Filtering stays on the writer thread.** Deletion-vector
  filtering (`filter_shard`) reads the running `nnz_offset` /
  `row_offset_kept` accumulators, which must be sequential to keep
  the on-disk layout deterministic. Workers produce raw decoded
  triplets only.
- **No new public API.** `pyscx.to_h5ad` / `pyscx.to_h5mu` gain
  `reader_threads=`, `writer_queue_depth=`, `memory_budget=` kwargs
  symmetric with the ingest wrappers, plus the existing CLI flags
  already flow through `ConvertOptions`.
