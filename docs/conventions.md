# SCX Coding Conventions

Rules that govern code in this repo. Agents and humans both follow.
For navigational summary, see [AGENTS.md](../AGENTS.md).

## Documentation in tracked files

Tracked files (code, tests, docs, configs, READMEs) MUST NOT reference
gitignored markdown documents. In this repo those live at the workspace
root and under `tasks/` — typically ALL-CAPS or date-prefixed names like
`*_CODE-REVIEW.md`, `Phase*.md`, `SPEC*.md`, `GPU-ACC-SPEED-UP.md`,
`HARMONY2.md`, `DEADLOCK-ISSUE.md`, `PER-CELL-CONTROL-PAIRING.md`,
`SCX-EVAL-METRIC-IMPROVE.md`, `2026-*_REGRESSIONS.md`, etc. They are
scratch/working specs and do not ship with the repository.

Concretely, in any tracked file, do not:

- Link to a gitignored doc (`[X.md](X.md)`, `see X.md §3`, `(per X.md)`).
- Cite a "review §1.4" / "Phase 7.2 of X.md" / "spec target (X.md:1037)".
- Carry inline `// TODO: see X.md` markers pointing at gitignored specs.

Tracked documentation must stand alone. When the gitignored doc carried
load-bearing context, **inline the substance** (a sentence or two of the
why / how) into the tracked file instead of citing. When the citation
was decorative, just delete it.

Cross-references between tracked files (`docs/*.md`, `benchmarks/README.md`,
`ROADMAP.md`, `AGENTS.md`/`CLAUDE.md`, generated reports under
`benchmarks/comprehensive/results/reports/`) are fine — they ship together.

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
