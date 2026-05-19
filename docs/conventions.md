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
