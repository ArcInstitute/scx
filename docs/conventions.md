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
- **Never size an allocation directly from an untrusted count.**
  `Vec::with_capacity` calls `handle_alloc_error`, which *aborts* — it cannot be
  caught, so a header field is a remote kill switch wherever it reaches a
  reservation unfiltered. Where a sound bound exists, validate against it
  (`scx_format::validate_allocation` for uncompressed structures,
  `scx_codec`'s `bound_capacity` for the Scx1 bit-level floor). Where none does
  — anything behind a general-purpose compressor — **clamp** the reservation
  instead (`scx_format::clamped_reserve`) and let real, length-checked data drive
  the growth; rejecting on a guessed compression ratio makes valid files
  unreadable. Test it by measuring the allocation, not by asserting `is_err()`:
  under Linux overcommit a multi-GB reservation succeeds and the decode returns
  the error the assertion wanted (see
  `scx-format-io/tests/framed_decode_allocation.rs`).
- **Validate shard payload at the decode seam, not at the consumer.** A shard
  payload is unauthenticated — the catalog's BLAKE3 covers catalog bytes only,
  and `read_shard_from_entry` skips the per-shard checksum by design. So a
  decoded value is untrusted input no matter how deep in the stack it surfaces.
  Both places that turn shard bytes into a CSR
  (`scx_format_io::decode_shard_regions_scipy` / `_native`) enforce the
  minor-axis bound, which is what lets the ~20 sites downstream write
  `dense[base + col]` or `col_sums[col]` with no check of their own. **Do not
  add a per-consumer bounds check against the same axis**: it is redundant, it
  costs a branch in a hot loop, and — worse — where two kernels are each other's
  oracle (the streaming statistics pair in `backed/aggregate.rs` and `prefetch.rs`)
  guarding only one makes them disagree on precisely the malformed input where
  their agreement is the evidence. If a new decode path is added that bypasses
  those seams, it owes the same check; `reader/matrix.rs`'s block-index row-run path is
  the existing example.

  Three carve-outs, all narrow:
  - **A consumer relying on a *different invariant* owes its own check.** The
    rule above is about re-validating the *same* thing the seam validated. The
    seam bounds each index's **value** against `n_minor`; it says nothing about
    the indices' **ordering or uniqueness**, which `docs/format.md` § "v3
    canonical CSR invariant" also requires and which only `scx validate --deep`
    checks on the read path. Anything inferring a *cardinality* from a dimension
    — "this axis has `extent` cells and `nnz` of them are stored, so
    `extent - nnz` are implicit zeros" — depends on uniqueness, not on the
    bound, and the seam's guarantee does not imply it. That inference is
    centralised in `scx_sparse::implicit_zero_count` (and
    `finalize_implicit_zero_variance` for the per-column variance shape); every
    statistic that infers an implicit-zero count that way goes through it.
    Written inline as a `usize` subtraction it wrapped in release and returned
    `8.3e19` as a variance. **Do not delete those checks as redundant with the
    seam** — they guard a different invariant, and they are O(1) per row /
    O(n_vars) per finalize, never per-nnz.

    ⚠️ **Be precise about what that helper is.** It rejects `nnz > extent` —
    an axis holding more stored entries than it has cells. That *implies* a
    duplicate coordinate, but not the reverse: a sparse row with a couple of
    repeats stays under its extent and passes. It is an overfull-axis guard,
    not a uniqueness check, and describing it as the latter overstates what
    the read path verifies. Real uniqueness enforcement means an ordering pass
    at the decode seam, which is deliberately not on the read path (see the
    measured cost below) and lives in `scx validate --deep` instead.

    Aggregations that use the second-moment identity `E[X²] − E[X]²` instead of
    a count subtraction (the CSC column-variance kernels, the projected row
    stats, the scalar `var(axis=None)`) never wrapped. Where a per-column or
    per-row count is available they run the same check, so neither
    `prefer_format` nor an active column projection changes whether a corrupt
    file is rejected. Only the scalar `var(axis=None)` has no count to carry;
    there the result is clamped at `0.0`, because a negative variance is not a
    defensible answer either.

    ⚠️ **Clamp with `if v < 0.0 { 0.0 } else { v }`, never `v.max(0.0)`.**
    Rust's `f64::max` ignores NaN and returns the other operand, so `.max(0.0)`
    silently converts a NaN variance into a real-looking `0.0`. NaN in `X` is
    supported input — the dense h5ad streamer preserves it deliberately — so
    that is data loss, not tidying. The conditional passes NaN through, since
    `NaN < 0.0` is false.

    The same `extent - count` subtraction appears outside `scx-sparse`: the CSC
    Wilcoxon kernel in `scx-accel` derives both a pooled and a per-group
    implicit-zero count. The per-package `overflow-checks` override does not
    cross crate boundaries, so guards there have to be explicit — and the
    per-group subtraction needs its own check, because it can invert while the
    pooled one still fits. Note what those two operands actually prove: they
    count **labelled nonzero** cells, not stored entries, so a column overfull
    purely with duplicated *explicit zeros* passes them.

    Two limits worth stating rather than implying, because both make a guard
    weaker than its name suggests:
    - **A check that runs after a lossy transform inherits the transform's
      blind spots.** `row_stats_projected` validates the *projected* row, and
      `project_csr_row` advances its `gene_set` pointer monotonically — so on an
      unsorted row it drops every index smaller than one already seen. An
      unsorted overfull row can project to a well-formed one.
    - **A route that materializes and defers to another library runs no guard
      at all.** Projected `max(axis=1)` / `min(axis=1)` go through
      `to_memory()` into scipy, so they answer where their unprojected twins
      reject. Written down in `docs/api/rust-format-io.md`'s route table rather than left for
      the next reader to discover.
  - **A consumer bounded by a *different* axis owes its own check.** The dense
    scatter in `typed_read.rs` sizes its buffer from the file header's `n_vars`
    while the seam validates against the shard header's `n_minor`. Those agree on
    any file a writer produced, but they are two numbers, so a corrupt shard can
    pass one and violate the other.
  - **Check where a violation would be silent.** The dense scatters write
    `dense[base + col]`, so an out-of-range column runs off the end of one row
    into the next and returns a plausible, wrong matrix rather than panicking.
    There are two such sites and both are guarded:
    `scx_format_io::typed_read::scatter_typed_csr_to_dense` and
    `ScxCsr::to_dense_dtype` (which `to_dense` delegates to, and which pyscx's
    query / `obs_filter` path reaches).

    ⚠️ "Everywhere else it panics" is *almost* true and the exception matters:
    a CSR handed to `scipy.sparse.csr_matrix` crosses out of Rust entirely, and
    scipy constructs an out-of-range matrix without complaint — `.toarray()`
    then misplaces the value exactly as the Rust scatter would. That is why the
    decode seam reconciles the shard header's declared width against the
    catalog's authenticated one *before* bounding indices by it: a payload that
    can widen its own bound can put an invalid matrix into scipy's hands, where
    no Rust-side guard runs at all.
- **Never positionally index the flattened CSR shard list.** On a multimodal
  file every modality independently tiles the obs axis `[0, n_obs)`, so
  `FullCatalog::csr_shards_sorted()` carries **overlapping** row ranges: a
  shard's position in it identifies neither an obs row nor a modality.
  `FullCatalog::single_tiling_csr_shards(op)` is the seam — it returns the same
  list or `ScxError::MultimodalRequiresModality`, and every whole-matrix or
  positional read goes through it (`read_all_csr_shards`, its `_typed` and
  `_filtered` twins, `read_csr_shard`, `BackedCsrIndex::from_catalog`, and
  `BackedCsrReader::read_all` by inheritance).

  ⚠️ **A whole-matrix guard does not cover the row-addressed reads.**
  `BackedCsrIndex::shard_for_row` / `shards_for_indices` /
  `shards_with_kept_rows` resolve a row by `partition_point` over the shard
  ranges, which *answers* on an overlap instead of erroring — it takes
  whichever shard sorted last. `read_row_indices`, `read_rows_with` and every
  streaming reduction went on returning an arbitrary modality's values after
  `read_all` had been closed. `BackedCsrReader` caches the predicate at
  construction (`row_addressing_ambiguous`) and one `ensure_row_addressable`
  guards the gather planner and the `ShardSource` methods. Construction stays
  infallible, so a binding can build a reader under a better-worded upstream
  guard.

  ⚠️ **The `ShardSource` guard does not cover an inherent method.** Rust
  resolves a concrete-typed call to the inherent method, not to the trait, so
  `col_means_and_sum_sq` — which loops `read_shard_cached_arc` directly — went
  on folding every modality after the trait was guarded, returning means
  carrying one modality's column 0 *and* another's column 4. It calls
  `ensure_row_addressable` itself. **A new method that walks shards owes that
  call, or must go through the trait**; `pyscx`'s `LazyShardSource::decode_shard`
  is the example of the second.

  ⚠️ **Anything that narrows by `modality_id()` owes it too.**
  `catalog_int_value_max` filtered catalog entries by
  `e.modality_id == self.modality_id()`, which on an unscoped reader is
  whichever modality sorted first — a *falsely small* maximum whenever another
  modality holds the larger value, on a method whose `None` already means
  "unknown". Raw `csr_shards_sorted()` stays
  for genuinely modality-agnostic work — a widest-value-encoding fold, an
  nnz sum, a checksum walk — and for the ops already fenced upstream by a
  table-based `is_multimodal()` refusal.

  ⚠️ **Test the geometry, not the modality table.** `is_multimodal()` is `true`
  for a file with a *one-entry* modality table, which is what
  `from_mudata(MuData({"rna": adata}))` and a single-modality h5mu ingest write
  — and that file stamps its only X with `modality_id = 1`, so modality 0 owns
  no shards while the flattened cover is perfectly unambiguous. Both
  `is_multimodal()` and "modality 0 tiles the obs axis" reject it falsely. Only
  the shard ranges answer the question being asked.

  ⚠️ **Overlap-free is not "tiles exactly once".** The predicate skips entries
  with no row-range stats, and it says nothing about gaps. A consumer that needs
  every row claimed once — the ML loaders do — needs the contiguity half too;
  `scx-loader`'s `ensure_csr_ranges_are_readable` is the worked example.

- **Prefer riding an existing pass to adding one.** The bound above is enforced
  by handing `scx_codec::decode_shard_scipy` an `index_bound`, so it rides the
  scan that already rejects `> i32::MAX`; only the comparand changes. The same
  check written as its own pass measured **+5.1–6.4%** of per-shard decode
  (194.9M nnz) and could not be optimised away — at 10.0 GB/s it was already
  memory-bandwidth-bound. Fused, it is inside noise. On a hot path, *where* a
  validation happens can matter more than what it does.

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

## Test Organization

- Small inline `#[cfg(test)] mod tests { … }` blocks are fine and preferred for
  modest test code — keep them in the production file.
- When a trailing inline test module grows large (rule of thumb: ~800+ test LOC,
  or it pushes the production file well past readability), extract it to a sibling
  `<file>_tests.rs` and include it white-box:

  ```rust
  #[cfg(test)]
  #[path = "foo_tests.rs"]
  mod tests;
  ```

  This keeps the tests a child of the parent module, so `use super::*` retains
  private-item access — unlike a `tests/` integration crate, which only sees the
  public API. Preserve any feature `cfg` (e.g. `#[cfg(all(test, feature = "gpu"))]`)
  on the `mod tests;` include.
- For a module with several interspersed `#[cfg(test)]` test submodules,
  consolidate them into one `<file>_tests.rs`; submodules that gain a nesting
  level rewrite `use super::X` → `use super::super::X` (see
  `scx-engine/src/index_tests.rs`).
- `#[cfg(test)]` instrumentation embedded *inside production functions* (e.g. the
  parallel-coordinator in-flight counters in
  `scx-convert/src/pipeline/coordinator.rs`) stays
  in place — it is conditional compilation of production flow, not unit tests.
- For a very large single-file test suite, split by subject into sibling modules
  sharing a `*_common` fixtures module (see `scx-convert`'s `convert_tests_*`),
  kept as crate submodules to preserve `super::` access to crate internals.

## Refactor PRs

Rules for any PR whose stated purpose is to move, split, or unify code rather
than to change what it does. They exist because a behaviour-preserving refactor
validated by a suite that cannot fail is indistinguishable from a regression.

- **A split-only commit must be a pure move.** `git diff --stat` should show
  matched insertions and deletions with no net line delta beyond `use` / `mod`
  lines. Never mix a module split with a semantic change in one commit — a
  reviewer cannot tell them apart in a 3000-line diff, and a bisect cannot
  either.
- **The test that would catch the refactor going wrong lands first, and is
  watched failing.** Break the pre-refactor code deliberately, confirm the new
  test goes red, then restore and refactor. A test written afterwards against
  code you just wrote tests the code, not the contract.
- **Identity and determinism claims cite the harness.** A path documented as
  byte-identical or deterministic (the `scx-ops` rewrite paths, the
  `scx-convert` sequential-vs-parallel coordinators, the `scx-gpu` decoders)
  must, when refactored, cite an `scx-testkit` digest test in the PR body. A
  round-trip test is not evidence for a byte-identity contract — it proves the
  data survived, not that the bytes did. See
  [Test Organization](#test-organization) and `scx-testkit/src/digest.rs`.

  For the `scx-ops` rewrite paths the harness is already built and committed:
  `tests/scx-integration-tests/tests/op_output_identity.rs` runs sixteen arms over
  nine single-modality rewrite ops into one `scx_testkit::ab::OpDigestManifest`
  and pins it against a checked-in golden, and the same target does a cross-tree
  A/B under `SCX_TESTKIT_AB_DUMP=<path>` / `SCX_TESTKIT_AB_BASE=<path>`. Cite
  that rather than a hand-rolled comparison.

  **Read its module docs before citing it, and do not restate them here.** They
  own the arm list, the "What this cannot see" section and the constraint that
  the A/B's base worktree must already contain the harness. That list is not
  short, and it is deliberately not copied into this file: a shorter second copy
  is how a refactor comes to believe it is covered when it is not. A refactor
  touching anything on it owes its own oracle.
- **Never `git add -A`.** The repo root carries untracked scratch markdown, and
  the local pre-commit hook already runs `git add -u`.
- **Perf-touching refactors gate before merge** against
  `results/baselines/LATEST` via `benchmarks/scripts/gate_candidate.py`,
  submitted as SLURM jobs one at a time. Added abstraction on a decode or
  rewrite hot path is a measurable cost until measured otherwise.

## Python Bindings (pyscx)

- PyO3 with `Bound<'py, T>` API (not deprecated `&PyAny`).
- Pin `pyo3` and `numpy` crate to the same minor version (currently 0.28).
- `PyArray::from_vec()` for zero-copy (moves Rust `Vec` to numpy).
- `ScxCsr` `i64`/`i32`/`f32` matches scipy exactly — avoids copy.
- Arrow → pandas via pyarrow's `to_pandas()` for obs/var metadata.
- Accelerators exposed via `pyscx.accel.*` — results written to standard
  AnnData slots.
- Optional Python deps (e.g. `pydeseq2`) imported at runtime with clear
  `ImportError` if missing.
- **Never hold a numpy borrow past the coercion.** `PyReadonlyArray*` is a
  *view* into a buffer Python can still write: rust-numpy borrows are not
  GIL-bound, do not clear numpy's `WRITEABLE` flag, and carry no
  synchronization. Coerce (`asarray` / `astype` / `ascontiguousarray`), copy
  into an owned `Vec` / `Arc<[T]>`, and drop the guard — then hand the owned
  buffer to the kernel. `crate::convert::owned_csr` (scipy or dense → owned
  `ScxCsr`) and `crate::convert::owned_dense2_f32` are the sanctioned entry
  points; a one-off array is `ro.as_slice()?.to_vec()`. For a matrix-sized copy
  use `convert::interop::par_to_vec` rather than `to_vec` — a fresh allocation
  is page-fault bound (~1.6 GB/s serial), and the faults parallelize. Rayon is
  safe to call with the GIL held: the closure never re-enters the interpreter.
  The rule is phrased
  against the *coercion*, not against `py.detach`, deliberately: on
  free-threaded CPython there is no GIL to release and every held borrow is
  hazardous. Keeping the guard alive is not a fix — it keeps the object alive,
  not the values still.
  The one exception is `pyscx/src/accel/pca.rs`'s `GilHeldCsrSlices`, used by
  the GPU PCA / fused dispatch, which holds the GIL throughout (`GpuDevice` is
  `!Send`, so it structurally cannot detach). It is `#[cfg(feature = "gpu")]`-gated
  so a CPU build cannot name it, and CI's `dedup-guard` job keeps it in that
  one file.

## R Bindings (rscx)

- `extendr` v0.8.x for Rust ↔ R FFI.
- SCX CSR (row-major) must be transposed to dgCMatrix (CSC, column-major)
  for R/Matrix interop.
- R has no unsigned integers. Take small integer arguments as `i32` and
  convert internally; take **row/cell indices and other values that may
  exceed 2³¹ as `f64`** (R's numeric), since an `i32` vector would truncate
  them to `NA`.
- **Never cast an R `f64` to an index or count with a bare `as u64` /
  `as usize`** — the cast *saturates* (`-1.0 → 0`, `NaN → 0`) and truncates
  fractions, so a bad index silently reads row 0 instead of erroring. Route
  every one through `crate::util::r_whole_u64` / `r_whole_u64_slice` (the
  module is private to the crate — there is no `rscx::util` public path), which
  reject non-finite, negative, fractional, and >2⁵³ values. Note extendr
  rejects R's `NA` for a **scalar** `f64` parameter but *not* inside a
  `Vec<f64>` (`try_from_robj.rs` still carries a `// TODO: check NAs`), so
  the vector form is the one that needs the guard most.
- A count that becomes an allocation size needs an **upper** clamp too, not
  just a lower one: `cache_shards` reaches `LruCache::new`, which
  pre-allocates a `HashMap` of that capacity.
- Validate in R with 1-based wording where a user-facing wrapper exists
  (`R/ops.R`, `R/query.R`), and keep the Rust guard as defense-in-depth —
  the `$`-methods on `ScxBackedSparse` / `ScxLazyTransformed` are exported
  and bypass every R-side check.

## GPU (scx-gpu)

- Uses `cudarc` for CUDA runtime/driver API.
- CUDA kernels compiled via `cc` build script (`build.rs`).
- GPU decoders must produce bit-identical output to the scalar CPU
  reference.
- **64-bit flat indexing.** A kernel that indexes a flat matrix computes its
  element total *and* its flat index in 64-bit (`long long` parameters, with the
  `(long long)` cast applied to `blockIdx.x` before the multiply);
  `kernels/colmajor_ops.cu`'s `mean_correct_colmajor_strided_kernel` is the
  reference shape. Host wrappers size 1-D
  grids through `scx_gpu::flat_launch_1d`, never `(total as u32).div_ceil(...)`.
  Both halves are required: a 32-bit `int total = m * k` overflows past 2³¹
  elements — signed overflow, so UB, and what nvcc emits at `-O3` is threads
  whose index wrapped negative writing below the buffer — while a `u32` block
  count truncates the grid past 2³² elements and silently skips the tail with no
  error at all. The two crossings are at different shapes and both are reachable:
  `n_obs × k` (mean correction, `k = n_components + n_oversamples`) crosses 2³¹ at
  36M cells × k=60 = 2.16e9, which the GPU PCA VRAM pre-flight admits on an 80 GB
  H100; `n_obs × n_components` (embedding scale) crosses later, around 43M cells
  at 50 PCs.
- GDS requires local NVMe + nvidia-fs drivers + ext4/XFS filesystem;
  always falls back to the CPU path.
- In-VRAM GPU analysis (PCA, kNN, UMAP, preprocessing) routes through
  **rapids-singlecell** via PyO3 Python interop (`pyscx/src/accel/rapids.rs`).
  The module performs a one-shot runtime import probe; absence triggers a
  `no_rapids` `UserWarning` and falls back to CPU. rapids is a detected
  runtime dependency (conda), not a build dependency or pip extra.

## SIMD / platform-specific code

- **Scalar reference is the source of truth.** Every SIMD kernel must produce
  **bit-identical** output to a scalar reference function kept alongside it
  (e.g. `byte_undelta_planes_scalar` / `byte_unshuffle_scalar` in `scx-codec`).
  The scalar fn is both the fallback and the correctness oracle.
- **Gate, don't assume.** House x86 intrinsics in a `#[cfg(target_arch = "x86_64")]`
  module (`scx-codec/src/simd.rs`) so non-x86 targets compile the scalar path
  only. SSE2 is part of the x86_64 ABI baseline, so SSE2 intrinsics need no
  `#[target_feature]` or `is_x86_feature_detected!` probe; a *wider* path (AVX2+)
  must sit behind a runtime `is_x86_feature_detected!` check with the SSE2/scalar
  path as fallback. No `-C target-feature` build flags — dispatch at runtime.
- **Test `simd == scalar` directly.** Add a proptest that calls the SIMD and
  scalar fns directly and `prop_assert_eq!`s them across all real widths/lengths
  (incl. the sub-vector tail) — this exercises the scalar path deterministically
  without needing a non-x86 CI runner, and the SIMD path on the x86 runner.
- **`unsafe` hygiene.** Each `unsafe` intrinsic block carries a `// SAFETY:`
  note; keep the crate `clippy -D warnings`-clean. No new SIMD dependency —
  `core::arch` (std) is preferred over pulling in a portable-SIMD crate.

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
  testing is a Rust-native DESeq2-style negative-binomial GLM
  (`scx_accel::nb_glm`, surfaced as `accel.nb_glm` / `pdex_nb_glm` /
  `pseudobulk_dex(backend="nb_glm")`). `pydeseq2` is now only a
  benchmark/correctness reference, not a runtime dependency.
- **Execution route is computed once and *drives* dispatch.** The single
  planner `scx_accel::route::plan_de_route` is the source of truth for the
  `pdex_ref` *and* `rank_genes_groups` (Wilcoxon rank-sum) route + fallback reason. Both
  ops' GPU entry points `match` on the planned `AccelRoute` to pick the kernel,
  and the pyscx CPU dispatch calls `plan_de_route` too — so the recorded route
  can never diverge from the code that ran. v3 is the unconditional default GPU
  DE route (the `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` opt-in gates were removed when v3
  became the default): `BackedCsc` + a CSC sidecar → `GpuCscV3`, every
  other layout (including dense-host, which densifies to CSR) → `GpuCsrV3`. The
  non-DE GPU ops stamp the generic `GpuCsr` / `GpuDense` routes.
  `gpu_eligible` models "GPU present but this op+layout has no GPU kernel" (e.g.
  `prefer_format="csc"`), recording `UnsupportedInputLayout` rather than
  implying CUDA was missing. New planner-driven GPU routes MUST register an
  `AccelRoute` variant and a dispatch arm — no silent fallbacks. The route
  surfaces to Python on `adata.uns["scx_accel"]`, and any performance claim
  must cite the recorded route (the planner-stamped route is the signal — the
  former ad-hoc stderr trace was removed).

## Parallel streaming reader (scx-convert)

- `run_streaming_writer_coordinator` is the entry point that dispatches
  every streaming convert (h5ad CSR, h5ad dense, in-memory CSC, h5mu
  per-modality, h5ad layers). It picks between the sequential
  `streaming_writer_coordinator` and the rayon-parallel
  `streaming_writer_coordinator_parallel` based on three preconditions:
  (1) the reader exposes `IndexedCsrShardStream::read_range` via the
  `as_indexed` trait override, (2) `H5is_library_threadsafe` reports
  the libhdf5 build is thread-safe (cached in `OnceLock`), and (3)
  `IngestOptions::reader_threads` resolves to `> 1`. Any precondition
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
  that many tasks and spawns one new task **per written shard**.
  Per *received* shard is the version that does not work, and shipped
  for a while: it bounds `spawned − received`, while the `BTreeMap`
  holds `received − written`, so a slow shard 0 let the map accumulate
  ~all remaining shards regardless.
- Both directions run on one implementation,
  `scx_convert::parallel_drain::ordered_parallel_drain`. It is generic
  over the item and error types and is **not** gated on the `hdf5`
  feature, so its tests — the buffer bound, panic-to-`Err`, and the
  `move`-on-scope contract that releases parked senders — run in the
  ordinary `cargo test --workspace` job. Write a new fan-out/reorder
  loop through it rather than beside it: the two hand-written copies
  had already diverged on the spawn placement above.
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
  (RNA / general) or 10 % (ATAC), times `n_vars ×
  budget::WORKER_PHASE_BYTES_PER_NNZ` (48 B/nnz — payload, the encoder's
  value copy, and the framed encode's two candidates).
  `DenseXStreamReader` overrides this with `effective_target × n_vars ×
  budget::dense_worker_phase_bytes_per_elem` (44 B/element). Both size the
  **whole worker phase**, not the reader stage — a 3x better estimate, not a
  proven ceiling: all three per-shard reservations remain
  `enforced: false` and each names what it does not bound. Every constant and
  fraction here comes from `scx-convert/src/budget.rs`; do not
  re-derive one at a call site. When `--memory-budget` is set, worker
  count is clamped to fit; a single shard exceeding the budget fails
  the convert with an actionable message rather than risking OOM.
  ⚠️ **"Over-estimating is safe because it only routes to the
  sequential coordinator" is false, and this file used to say it.**
  Routing to sequential *is* the failure: an over-estimate does not
  degrade throughput gracefully, it destroys parallelism outright
  while leaving most of the budget unused. That reasoning is how the
  dense reader came to divide the budget by 4 in one function and
  multiply the resulting slab by 2 in another, charging the same
  reserve twice, so that every budget-bound dense convert ran on one
  thread. Size honestly instead: the cost model says what a unit
  holds, the share says how much of the budget it may claim, and the
  two are separate numbers.
- Direction-typed convert options: ingest entry points take
  `IngestOptions`, export entry points take `ExportOptions`. An
  export-only option must be named `export_*` and live on the latter
  — a CI guard enforces both halves. Do not add a runtime check that
  an option "has no effect on this direction"; that is the shape the
  split removed.
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
(`h5ad::stream_write::stream_csr_to_group_at`) mirrors the ingest
dispatcher but with a simpler precondition set:

- **No libhdf5 thread-safety probe.** Workers only read SCX shards
  (mmap + scx-codec decode); the HDF5 writes (`indices_ds.write_slice`,
  `data_ds.write_slice`, `indptr_ds.write_slice`) stay on the calling
  thread. There is no concurrent HDF5 access on this path, so
  non-threadsafe libhdf5 builds work just as well as threadsafe ones.
- **Per-shard memory estimate from real shard statistics, not a
  heuristic.** Every CSR shard records its `nnz` and row range in
  `FullCatalogEntry::stats` at convert time, so
  `per_shard_export_bytes(stats)` sizes the decode phase from measured
  nnz — `nnz × 8` (indices + data) + `(n_rows + 1) × 8` (indptr) +
  `nnz × 8` (**decoder** scratch, via
  `budget::shard_decode_working_set_bytes`). No density heuristic, no
  modality-type branching, and deliberately **not** the ingest model:
  export decodes, so it carries no encode term. It is still **not a
  bound** and the row stays `enforced: false` — `filter_shard` builds a
  `kept_indptr_tail` while `indptr_local` is live and grows
  `kept_indices` / `kept_data` from empty by doubling alongside the
  originals, none of which this charges. The memory-budget derate
  constrains `reader_threads + writer_queue_depth` against
  `max_shard_bytes` so the rolling-window cap matches the estimate
  directly.
- **No `max_slab_rows` clamp.** SCX shards are random-access via
  `ScxReader::read_shard_from_entry(entry)` — the export walk passes the
  catalog entry it already holds rather than an index the reader re-resolves,
  so `X`, layers and `raw` share one read path; the source isn't gated by a
  slab budget the way an HDF5 dense reader is on the ingest side.
- **Rolling-window spawn carries over** — literally: export and
  ingest share `ordered_parallel_drain`, so the reorder buffer on the
  writer thread is bounded by the same
  `reader_threads + writer_queue_depth` window, by the same code, and
  a slow shard 0 can't accumulate the rest of the file in memory.
- **Filtering stays on the writer thread.** Deletion-vector
  filtering (`filter_shard`) reads the running `nnz_offset` /
  `row_offset_kept` accumulators, which must be sequential to keep
  the on-disk layout deterministic. Workers produce raw decoded
  triplets only.
- **No new public API.** `pyscx.to_h5ad` / `pyscx.to_h5mu` gain
  `reader_threads=`, `writer_queue_depth=`, `memory_budget=` kwargs
  symmetric with the ingest wrappers, plus the existing CLI flags
  already flow through `IngestOptions` / `ExportOptions`.
