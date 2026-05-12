# SCX Comprehensive Code Review — 2026-05-11

Reviewer: Claude (automated audit)
Scope: full workspace at `/sessions/trusting-serene-cray/mnt/scx/`
   — 14 Rust crates (~112K LoC), pyscx + rscx bindings (~60K LoC of Python plus PyO3 / extendr glue), docs/, benchmarks/
This revision merges and reconciles findings with the second-pass review in `scx_code_review_report.md` (same date). Where the two reviews converged, the finding is consolidated; where one reviewer surfaced something the other missed, the finding is attributed inline. All concrete defects added from the second review have been re-verified against the source tree.

## 0. Reviewer convergence notes

Two independent code reviews were run on the same tree on 2026-05-11. They reached most of the same conclusions about overall shape and quality, but each surfaced material findings the other missed. The most important convergence/divergence points:

- **Both reviews** identified pyscx error-type mapping (`PyRuntimeError` over-use), tmp-file durability (parent-dir fsync), append rewriting `obs`, README/doc inconsistencies, dependency hygiene, and crate decomposition as primary themes. Both agree on the overall architectural shape and on the fact that the core design is sound.
- **Only the first review** found: the `ValueEncoding::Float16` mislabeling bug, `debug_assert!`-gated index validation, the engine's index-pushdown dead code, the cuVS layout-pin warning, the `obsp`/`varp`/`varm` round-trip gap in pyscx, the multi-numerical findings in `scx-accel` (UMAP negative-sample counter, two-pass mean centering, leiden malloc_trim), and the AGENTS/CLAUDE doc duplication.
- **Only the second review** found: two critical `scx-ops::append` defects (zero `shard_target_rows` hang, `_codec_id` ignored), the cloud pull "all-sections in memory" buffering, the shard-granular vs exact-cell semantics gap in selective pull, the unvalidated header reserved bytes (more specific than the first review's finding), the `target_n_vars as u32` truncation in append, the `write_preencoded_shard` CSC-count gap, and the stale `format_version` comment.

Together the two reviews represent ~50 distinct line-cited findings. Most have one- or two-line fixes.

## 1. Executive summary

SCX is a substantial, internally coherent codebase. The format/codec design is well-thought-out, the multithreading model has been iterated several times based on real fork-deadlock incidents, and the test surface (especially `scx-cli/convert/tests.rs` at 1,451 LOC, the integration test crate, and the comprehensive benchmark harness) is unusually thorough for a project at this phase. Documentation density is high — fifteen docs files cover format, codec, conventions, sharding, cloud, GPU, multimodal, scanpy integration, testing, and multithreading.

The reviews surfaced a meaningful population of concrete defects and inconsistencies that should be triaged before another major release:

- **Four high-severity correctness defects with a clear data-integrity or availability blast radius.**
  - (a) `scx-ops::append_for_modality()` infinite-loops when `shard_target_rows == 0` (`scx-ops/src/append.rs:301-460`). The CLI exposes this as `--shard-size 0` and pyscx as `shard_size=0` with no lower-bound validation. A user can hang `scx append` or `pyscx.append()`.
  - (b) `scx-ops::append_for_modality()` silently ignores its `codec_id` parameter — the function signature renames it to `_codec_id` (line 73) and every shard goes through `scx_format::select_codec` (line 329). Users requesting `--codec zstd` or `codec="none"` do not get the requested encoding.
  - (c) `ValueEncoding::Float16` (`scx-codec/src/value_encoding.rs:75-89`) falls back to writing 4-byte f32 bytes while still labelling the on-disk encoding as `Float16`; decoders that respect the label compute the wrong byte stride.
  - (d) Several index-validity checks in the codec are gated behind `debug_assert!` (`scx-codec/src/dispatch.rs:305`, `scx-codec/src/bitstream.rs:194`, `scx-codec/src/shuffle.rs:12`, `forbp.rs:122`, `delta_golomb.rs:63`, `rice.rs:101`). Release builds silently accept malformed input and may hand scipy a negative `i64`.
- **AnnData round-trip is incomplete in pyscx**: `obsp`, `varp`, and `varm` are never read or written. `pyscx.accel.pca` itself produces `varm["PCs"]`, which the writer then drops.
- **Cloud `pull()` buffers the entire selective output in memory** (`scx-cloud/src/pull.rs:250` `Vec<(usize, Vec<u8>)>` collected before any disk write) — contradicts the streaming/cloud-native messaging. Two documented features (`PullOptions::reorder_buffer`, `PushOptions::multipart_threshold`) are also dead code. The crate has no retries, no timeouts, and no `Timeout`/`RateLimited` variants in `CloudError`.
- **Selective cloud pull is shard-granular, not cell-granular**: the comment at `pull.rs:629-631` is explicit — "Shards are copied verbatim so the output file must include obs rows for every row in those shards." Users requesting `filter="cell_type == 'T cell'"` will receive non-T cells from any shard containing at least one T cell. README and CLI imply exact filtering.
- **Header validation is materially looser than the field comments imply**: `FileHeader::read_from` (`scx-format/src/header.rs:189-234`) validates `magic`, `format_version`, `endian`, and modality cross-consistency but does *not* validate `header_length == HEADER_SIZE`, `reserved_padding == 0`, the `flags` reserved bits, or the trailing 112-byte `reserved` array. (Curiously, the v1 fallback path *does* validate its 20-byte leading zero pad at line 225-229 — clear inconsistency.) Spec calls each of these out as "must be zero on write, ignored on read", but with no validation at all, forward-compatibility cannot be enforced. The comment at line 32 also says "Format version (currently 1)" while `CURRENT_FORMAT_VERSION = 2`.
- **Untrusted-input DoS surfaces**: at least five readers (`catalog.rs:498`, `provenance.rs:67,83`, `deletion_vectors.rs:79`, `shard.rs:300`) call `Vec::with_capacity(u32 as usize)` without bounds, so a single malformed `u32` can request a multi-GB allocation.
- **PID-based temp paths collide** (`scx-format/src/writer.rs:126,1730,1967` use `"{}.tmp.{}".format(dest, process::id())`; `scx-cloud/src/pull.rs` does the same). Two concurrent writes from the same process to the same destination trample each other. The writer-side path has no stale-cleanup, unlike the cloud-side path.
- **Documentation is internally inconsistent on load-bearing numbers**: README/AGENTS claim 5.1 GB peak RSS / 88 % reduction on the 1M-cell out-of-core pipeline; `docs/scanpy.md` and `ROADMAP.md` claim ~11 GB / 71 %. Both can't be right.

None of these block production use, but each is the sort of finding a downstream consumer is likely to discover and report. Most have one- or two-line fixes. The append-related P0s and the cloud pull semantics, in particular, fail the public contract loudly enough that they should be fixed before another tagged release.

## 2. Architecture & Documentation

### 2.1 Overall design assessment

The crate graph documented in `docs/architecture.md` is sensible: a strict layering from `scx-format` and `scx-codec` upward, with `scx-cli` / `pyscx` / `rscx` as terminals. Feature flags are conservatively scoped and opt-in. The decision to keep `rscx` out of `default-members` (Cargo.toml comment) is correct given the extendr toolchain dependency.

The fragment / manifest model in `scx-format` (mutate by appending a new catalog and chain via `prev_catalog_offset`) is the right abstraction for the append/delete/compact/rollback workflow and is what makes the millisecond mutation claims plausible. The dual catalog (root at offset 256 for cloud-first byte-range reads, full catalog at EOF) is a thoughtful design.

The triple-buffered training loader (tokio I/O → rayon decode → Python consumer) and its fork-safety story are unusually well-engineered and reflect hard-won lessons. The accompanying tests (`test_fork_deadlock.rs`) and the design comments are excellent. `LoaderConfig::validate()` is a model that the rest of the codebase's entry points should imitate — it rejects zero batch size, zero shard group size, zero prefetch, non-positive target sum, and too-small memory budgets up front. `scx-ops::append` would not have shipped a zero-shard-size hang if it had the same discipline.

### 2.2 Public-contract drift (the central theme)

The single biggest architectural concern in this codebase is not a structural defect; it's that several public surfaces — README claims, CLI flags, Python keyword arguments — are ahead of, or subtly different from, the implementation. The pattern recurs in multiple places:

- Append is documented as `O(new cells)` but rewrites the merged `obs` Arrow IPC section in full and (in the CLI/Python wrappers) materializes the entire source CSR matrix via `read_all_csr_shards()` (`scx-cli/src/append.rs`, `pyscx/src/ops.rs`). Real complexity is `O(new cells + existing obs)` for metadata and proportional to total source matrix for wrapper memory.
- Append exposes `--codec`/`codec=` but the core function ignores it (see §1).
- Cloud `pull()` is described as streaming but stores all downloaded sections before writing (see §1).
- Selective cloud pull is documented like exact filtering but returns a shard-level superset (see §1).
- The README's BLAKE3 phrasing can be read as "every open verifies every payload". In fact, `pyscx.open(..., verify=True)` documents itself as catalog-only verification (`pyscx/src/lib.rs`); section-level re-hashing requires `pyscx.validate()` or `scx validate`. The CLI help and cloud docs do not draw this distinction.
- Selective pull also silently omits `obsm`, `obsp`, layers, and other sidecars (`pull.rs::pull_filtered`'s `entries_to_download` includes only CSR shards, `var`, `var_index`, `uns`, and the modality table). There is no warning about omitted section types.

These are fixable. The codebase already has good abstractions; the next step is to make the public contracts precise and add tests that pin them down.

### 2.3 Documentation findings

**Internal inconsistencies (high-impact):**

- **Section-type count drift.** `docs/format.md:171-191` enumerates 17 section types (IDs 0–16). `docs/api.md:5` says "17 section types". `AGENTS.md` / `CLAUDE.md:51` says "15 section types". The Agents doc is stale.
- **`num_workers` runtime contract drift.** `docs/architecture.md:421-423` claims `TrainingDataset` "detects `num_workers > 0` and raises an error". This contradicts `README.md`, `docs/multithreading.md:128-141`, `docs/api.md:888`, and the actual code in `scx-loader/src/python.rs`. The new contract is "raises only if the dataset is *eagerly* constructed in the parent".
- **Tokio runtime model contradicts itself in a single file.** `docs/multithreading.md:75-85` says "tokio current-thread runtime constructed inside the I/O thread on each `start_epoch`". `docs/multithreading.md:261` says "the training loader's tokio runtime is fixed at 2 worker threads". These describe different designs.
- **Out-of-core peak RSS: two answers.** README/AGENTS: 5.1 GB / 88 % vs 43.6 GB. `docs/scanpy.md:707` and `ROADMAP.md:571`: ~11 GB / 71 % vs ~38 GB. Decide which is current, retire the other.
- **Default shard size: 10 000 vs 16 384.** `docs/format.md:68,301` says 10 000. `docs/sharding.md:33` documents a binding-specific split (CLI 10 000, Python/R 16 384). `docs/api.md:518` says 16 384. The hidden binding-default split is fine; the docs should call it out in one place rather than three.
- **Stale "≥13 unknown sections skipped" line at `docs/format.md:581`** — `obs_predicate_index` (13) and `var_predicate_index` (14) are now defined v1 types; the line should read "≥17" or be removed.
- **Detection bitmap status is described three ways**: "Optional" (`docs/format.md:614`), "Deferred" (`ROADMAP.md`), "Reserved; not produced by the current writer" (`docs/api.md:17`).
- **Stale ROADMAP date.** `ROADMAP.md:3` says "Last updated 2026-04-16" but the body references `v0.6.2-n_counts-augmentation` (May 2026). Phase-2 pitfalls (lines 247–256) and the Risk Register still warn about old fork-deadlock semantics that have been addressed.
- **Stale `format_version` field comment.** `scx-format/src/header.rs:32` says "Format version (currently 1)" while `CURRENT_FORMAT_VERSION = 2` at line 25.

**Doc coverage gaps:**

- **No documented threat model.** `docs/format.md:573` "Encryption: not in v1" is the only sentence. Given that the reader trusts on-disk `u32` lengths in many places (§4.2 below), the format needs an explicit "untrusted input handling" section.
- **No documented backwards-compatibility policy** beyond version field semantics — no version-compatibility matrix, no SemVer policy across `pyscx`/`rscx`/`scx-cli`.
- **No documented "what does `ScxReader::open` verify on open vs lazily?"** The answer (per code: opens skip per-shard checksums; `validate` does them) is correct but unstated. The README's BLAKE3 phrasing can mislead users into thinking transient transfer corruption is caught on open. Cloud-pulled atlas users particularly need to know.
- **No documented partial-recovery story for corrupt non-essential sections.**
- **No documented `file_checksum` semantics.** The header carries a `file_checksum: u64`, cloud pull computes a file checksum after writing, and `pyscx.open()` / `pyscx.validate()` use it differently. `docs/format.md` should define which bytes are included, whether the field is zeroed during computation, which APIs verify it, what happens after append/rollback, and which is authoritative when section checksums and file checksum disagree.
- **No CSC sidecar invalidation matrix.** Append, delete, compact, merge, and subset all interact differently with CSC sidecars. A single table in `docs/operations.md` (suggested below) would prevent users from silently losing column-major acceleration.
- **R bindings** are documented in ~15 lines of `docs/api.md`; there is no equivalent of `pyscx.accel.*` reference for rscx and `rscx/R/harmony.R` is not surfaced in any doc.

**Marketing creep into reference docs.** `docs/scanpy.md:30-80` and several sections of `README.md` use heavy ✓/⚠️ emoji and competitive comparisons. The README is the right place for this; duplicating the same paragraphs in the reference docs makes them harder to maintain. The README itself does too many jobs (install + quickstart + benchmark tables + competitive positioning + roadmap snippets + GPU setup + cloud claims + fork-safety guidance + perturbation metrics tables, some with pending/TBD rows). The README should be a concise entry point that links to `docs/`, with detailed benchmarks reproducible from a manifest (see §17 P1 recommendations).

**Diagrams.** The codebase is well-served by ASCII layout boxes in `docs/format.md`, and `docs/scanpy.md` has the one decision-tree. There is no sequence diagram for the triple-buffered pipeline (the most algorithmically interesting piece), no diagram of the catalog-chain mutation model, and no diagram of the cloud pull/push topology — three good candidates.

## 3. `scx-format` — Format Implementation

### 3.1 Correctness

- **Header reserved-field validation is missing** (`header.rs:189-234`). The parser validates `magic`, `format_version`, `endian`, and modality cross-consistency, but accepts any value of `header_length`, `reserved_padding`, the `flags` reserved bits, and the trailing 112-byte `reserved` array. The v1 fallback path (line 225-229) does validate its leading 20-byte zero pad — proof that the convention exists and is just inconsistently applied. Without this validation, forward-compatibility cannot be enforced: a future writer that sets a flag bit will be silently accepted, defeating the entire reason reserved bits exist.
- **Stale field comment** at `header.rs:32`: "Format version (currently 1)" while `CURRENT_FORMAT_VERSION = 2`.
- **Unbounded allocations from on-disk `u32` counts (DoS surface).** Multiple readers do `Vec::with_capacity(n as usize)` where `n` is attacker-controlled:
  - `catalog.rs:498` — full catalog `n_entries` (u32, up to 4 G entries).
  - `provenance.rs:67` — `n_operations` (u32).
  - `provenance.rs:83-85` — `vec![0u8; params_len]` (u32 → 4 GiB).
  - `deletion_vectors.rs:79` — `vec![0u8; bitmap_len]` (u32 → 4 GiB).
  - `shard.rs:300` — `n_blocks` (u32).
  Bound each by a sensible cap (e.g., total section length declared by the catalog) before allocating.
- **No cross-check between `FileHeader::n_obs` and `FullCatalog::n_obs`** in `reader.rs::open_inner`. A corrupted file with disagreement opens silently.
- **No validation of `shard_format_version`** at `shard.rs:107`. Same for `provenance.rs:64` `version` and `deletion_vectors.rs:73` `dv_version` — the spec says all are 1; the readers accept any value.
- **`provenance.rs:53` writes `input_checksums.len() as u8`** — silently truncates if a merge has >255 inputs. Cap or error in the writer.
- **Parent-directory fsync missing.** `writer.rs:1437` performs `fsync → rename` but does not `fsync` the parent directory afterwards; on power loss the directory entry may be lost on POSIX. The conventions doc calls out the temp-file pattern but not the directory fsync.
- **PID-based temp file collisions.** `writer.rs:126,1730,1967` form temp paths like `"{}.tmp.{}".format(dest, process::id())`. Two concurrent writes from the same process to the same destination collide. There is no stale-temp cleanup in the writer (unlike `scx-cloud/src/pull.rs:74-108`, which has a thoughtful liveness check). Replace with `tempfile::Builder::tempfile_in(parent_dir)` so the randomized suffix prevents collisions, then persist via atomic rename, then fsync the parent directory.
- **`writer.rs:453`** — each CSR shard is given a single-entry `BlockIndex` spanning the whole shard. The block-index design is for row-range random access; a single block degrades random access to "decode the whole shard". Either populate the block index meaningfully or document that block index is reserved.
- **`writer.rs:1218`** — when re-reading a shard for CSR→CSC, `vec![0u8; entry.length as usize]` trusts the catalog's `length`; no upper bound.
- **`catalog.rs:455-461`** — `RootCatalog::read_from` does not verify on-disk `n_section_groups` against the actual count consumed before returning `Ok`.
- **`catalog.rs:532-534`** — unknown `section_type` values are silently skipped via `continue`. Spec §3 says readers should skip "with a warning"; no `log::warn!` is emitted.
- **`section.rs:62 align_to_8`** — no overflow check on `(offset + 7) & !7`. Use `checked_add`.
- **`write_preencoded_shard` increments only `csr_shard_count`** (`writer.rs:668-671`). If a `CscShard` is fed in (the function does not validate the section type), the CSC sidecar count is silently under-reported and downstream readers may believe the file has no CSC sidecar. Either:
  ```rust
  match section.section_type {
      SectionType::CsrShard => { self.csr_shard_count += 1; self.total_nnz += section.nnz; }
      SectionType::CscShard => { self.csc_shard_count += 1; }
      _ => {}
  }
  ```
  or reject non-CSR section types with an explicit error.

### 3.2 Spec drift

- The `summary: [u8; 32]` field documented in spec §3 for `csr_shards` (nnz, value stats) is always written as zeros (`writer.rs:1363`). Either populate it or remove from the spec.
- `FullCatalog::write_to` (`catalog.rs:417`) silently upgrades v1 → v2 on serialize. Well-commented but surprising for callers expecting byte-stable output of v1 inputs.
- `SectionType::is_known` (`section.rs:56`) is dead code.

### 3.3 Design / code quality

- `ScxError::ChecksumMismatch` has no payload — callers can't tell which section failed.
- `ScxReader::read_shard_from_entry` (default path, `reader.rs:1272`) skips per-shard checksum verification by design; the name does not telegraph this. Consider renaming to `_unverified` or making the verified path the default with an opt-out.
- `ScxReader::read_shard_from_entry_unchecked` is `#[deprecated]`; remove in next major.
- `MajorAxis::Row|Col` is exported but used only by `compute_shard_stats` — make private.
- `BlockIndexEntry::new` correctly errors on overflow (good).
- `writer.rs` is ~2000 lines and mixes metadata-section, shard-section, modality, finalization, and low-level I/O concerns. Splitting into submodules would help discoverability.

## 4. `scx-codec` — Compression Codec

### 4.1 Correctness

- **`value_encoding.rs:75-89` — Float16 fallback writes f32 bytes.** Critical. The encoder emits 4 bytes per value while leaving `value_encoding == Float16` (a once-per-process `log::warn!` fires but the on-disk label is unchanged). Any reader that respects the `value_encoding` label will compute the wrong byte stride. Either route through `half::f16::from_f32(...).to_le_bytes()` and emit 2 bytes, or refuse Float16 inputs upfront (`return Err(...)`) rather than silently mislabeling.
- **`dispatch.rs:305-322` — `u64_vec_to_i64`/`u32_vec_to_i32` gate the sign-bit clamp behind `debug_assert!`.** In release builds, a corrupted on-disk indptr/indices with the top bit set is reinterpreted as a negative `i64`/`i32` and passed to scipy. Replace with a runtime check.
- **`bitstream.rs:194-204` — `read_bits` masking for `n_bits == 64` is fine, but `n_bits > 64` is only debug-checked.** A `u8` value of 65–255 underflows `bits_left -= n_bits`.
- **`shuffle.rs:12-30` — byte_shuffle/unshuffle only `debug_assert!`** that `input.len() % element_width == 0`. In release, the trailing partial element is silently dropped, breaking round-trip.
- **`forbp.rs:122-124` — delta-encode without monotonicity check.** Unsorted indices wrap silently in release.
- **`delta_golomb.rs:63` — indptr `w[1] - w[0]` unchecked subtraction.** Non-monotonic indptr wraps silently in release.
- **`rice.rs:101` — block header reads 8 bits, masks `& 0x0F`, discarding the high nibble.** Spec §4 says those bits are zero; a non-zero value indicates corruption that should fail loudly.
- **`forbp.rs:37-55` — `read_varint` accepts shifts up to 35 bits without overflow check on the final byte.**

### 4.2 Spec drift

- Spec says no upper bound on Rice `k`; implementation clamps `k ≤ 15` (`delta_golomb.rs:39`, `rice.rs:43`). Either document the clamp (matches the u4 block-header field) or remove.
- `floor_median_u32` (`scx-format/src/codec_select.rs:20`) returns the upper middle; `scx-codec`'s versions return the lower middle. Trivial inconsistency, but the median is the codec selector — pick one.

### 4.3 Design / performance

- `EncodedShard` (owned) vs `EncodedShardRef` (borrowed) duplicate fields; could be generic over `Cow`-style ownership.
- `CodecError::Io(std::io::Error::other(e.to_string()))` (in pcodec branches at `dispatch.rs:692,701,…`) loses structured error info. Add a `Pcodec(PcodecError)` variant.
- Three near-identical `floor_median_*` functions live in `delta_golomb.rs`, `rice.rs`, `codec_select.rs` (and similar logic in `pyscx/anndata.rs`). Consolidate.
- **Add golden corpora.** Representative count distributions, float data, empty rows, highly dense rows, and pathological indptr/indices belong in a `scx-codec/tests/corpora/` directory with stable hashes. Today the conformance test vectors mentioned in `ROADMAP.md §1.2` are inline.
- **Cross-codec round-trip tests** for `auto`, `none`, `zstd`, `lz4`, `scx1`, `pcodec` are sparse. Property-test with randomized sorted CSR rows.

## 5. `scx-engine` — Lazy Query Engine

- **Index-pushdown is unwired.** `var_predicates`, `obs_predicate_index`, `var_predicate_index` are all `#[allow(dead_code)]` in `collect.rs:38-41`. The comment promises index-level pushdown; only catalog-level + post-decode filtering runs today. Either implement or delete.
- **Uses the process-global rayon pool from `collect.rs:311`.** This conflicts with the per-pipeline pool discipline documented in `docs/multithreading.md:98-105`. A `QueryPipeline::collect()` reached from a forked DataLoader worker after the parent has touched rayon will deadlock. The training loader has its own pool; the engine should at minimum accept an optional `ThreadPool` reference.
- **`fused_ops.rs:408 panic!()` for Float16.** Convert to `EngineError::SchemaError`.
- **`apply_fused_ops` is single-threaded** even after the parallel decode finishes; wrap the per-row loop in a `pool.install(|| par_iter())`.
- **DV-key uses sorted shard index** (`collect.rs:202-217`). Correct today, but assumes no append/delete interleaving since the DV was written. Add an invariant doc comment.
- Tests duplicate the `sample_header` boilerplate ~6× across this crate alone; move to a shared `tests/common.rs`.
- **Predicate-engine conformance suite is missing.** Predicates run from Rust CLI (`scx query`), pyscx (`pyscx.query.*`), and cloud selective pull (`pull_filtered`). A shared conformance suite would catch divergence — including nulls, categorical strings, numeric comparisons, boolean combinations, missing columns, and type-mismatch errors.

## 6. `scx-loader` — Training Pipeline

- **`index_plan.rs:349-358` eagerly builds `Runtime::new_multi_thread().worker_threads(2)` in `IndexPlanLoader::new()`.** This is structurally the same hazard `TrainingPipeline` was refactored to avoid: a user who instantiates `IndexPlanLoader` in the parent and forks will wedge. The lazy-runtime-in-worker pattern used in `TrainingPipeline::start_epoch` should be applied here too.
- **`index_plan.rs:830-862` serial `block_on`.** Each prefetch is `block_on`'d sequentially; head-plan parallelism is lost. Use `futures::join_all`.
- **No explicit `CancellationToken`** — shutdown relies on channel-close propagation plus a deadline. Documented and works today, but a token would be belt-and-braces.
- **Smoke `black_box(mmap[off])` only touches shard 0** (`pipeline.rs:644-649`). Cheap mitigation: touch the last shard too.
- **`LoaderConfig` builder boilerplate is duplicated between `TrainingDataset::new` (`python.rs:140-152`) and `MultimodalTrainingDataset::new` (`python.rs:436-448`).** Introduce `LoaderConfig::overlay(partial)`.
- **Per-batch fresh allocations.** `decode_stage.rs:140 vec![0.0f32; n_rows * n_output_genes]` allocates ~144 MB at 1024×36k×4B per batch; recycle via `Batch::recycle` and amortize.
- **MADV_WILLNEED look-ahead hard-coded at 2** (`io_stage.rs:108`) — tie to channel capacity.
- **`MultimodalTrainingDataset::__next__` pulls modalities sequentially** (`python.rs:490-525`); the slowest modality blocks the rest.
- **No exported runtime metrics.** A lightweight counter set (batches produced, decode wait time, I/O wait time, channel saturation, memory-budget decisions) would help users tune `prefetch_batches`, `shard_group_size`, and memory budgets without strace.
- **Stress-test gaps.** Add tests for early-consumer-drop, repeated `start_epoch` calls, zero-shard datasets, tiny memory budgets, and modality-filtered training.

## 7. `scx-cloud` — Cloud I/O

This is, by some margin, the least-finished of the mid-tier crates and has the most user-visible contract violations.

- **`pull()` buffers the entire output in memory before writing.** `pull.rs:250` allocates `Vec<(usize, Vec<u8>)>` for every ordered entry, every download appends to it (line 281), it is sorted in place (line 286), and only then written sequentially (line 331). Peak memory approaches the size of the whole pulled `.scx` file. The README and `docs/cloud.md` describe this path as streaming. Replace with a bounded producer/writer using `tokio::sync::mpsc::channel(reorder_buffer)`, a small `BTreeMap<usize, Vec<u8>>` reorder window for out-of-order arrivals, and spill-to-tempfile if a single section blocks ordering indefinitely.
- **Selective pull is shard-granular, not cell-granular.** `pull_filtered` (`pull.rs:529`) maps predicate-matching rows to shard IDs, then collects *all* rows of every needed shard as `shard_row_indices` (line 632-641). The inline comment is explicit: "Shards are copied verbatim so the output file must include obs rows for every row in those shards." A user requesting `filter="cell_type == 'T cell'"` receives non-T cells from any shard containing at least one T cell. README says "only download T cells". Two fixes:
  - Near-term docs: rename to "selective shard pull" and surface the superset semantics.
  - Real fix: add an `exact` mode that decodes selected shards, filters rows, rebases CSR indptr, and writes only matching cells.
- **Selective pull omits `obsm`, `obsp`, layers, and other sidecars without warning.** `pull_filtered`'s `entries_to_download` includes only CSR shards, `var`, `var_index`, `uns`, and the modality table. Users who expect a complete subset of the original dataset lose embeddings, graphs, layers, and other annotations. Add `--include-obsm`, `--include-layers`, `--include-obsp` flags and print a summary of omitted section types.
- **`PullOptions::reorder_buffer` is dead code.** Declared at `pull.rs:115`, defaulted to 4, documented in `docs/cloud.md:354`, and not read anywhere in the crate. Confirmed by grep. Either implement (as part of the streaming fix above) or remove.
- **`PushOptions::multipart_threshold` is dead code.** Defaulted to 8 MB and not consulted. Every section, regardless of size, uses a single-shot `put`. Confirmed by grep.
- **No retry/backoff.** `CloudError::DownloadFailed{retries}` is *defined* but never *constructed*. Every `.get().await?` propagates the first transient error. `object_store` has its own retries but with conservative defaults.
- **No request timeouts.** `grep timeout` in `scx-cloud/src/` returns nothing.
- **Parallel-pull uses `chunks(parallelism)` + `join_all`** (`pull.rs:265-283`). The next batch can't start until the slowest in the current batch finishes. Use `buffer_unordered(parallelism)` for proper saturation; this is also exactly what `reorder_buffer` would buy you.
- **`estimate_catalog_size` hard-codes `30 + n*100 + 32`** (`pull.rs:457-461`). If the catalog overflows the estimate, the front-catalog path silently falls back to `clear_front_catalog` and the output is no longer cloud-optimized. `PullStats` still reports success. Should be a hard error or a second pass with a larger reservation.
- **Hash-while-write opportunity.** `pull.rs:430-435` recomputes the checksum by re-reading the written file. Multi-GB redundant read; compute the BLAKE3 during the write.
- **`backend.rs:84-119 create_backend` returns successfully with no credentials.** Auth errors surface only at first GET. Add `verify_credentials()`.
- **`CloudError` lacks `Timeout` and `RateLimited` variants** — gating retries by error type is impossible.
- **`downloaded_sections.sort_by_key`** (`pull.rs:285-286`) moves multi-hundred-MB `Vec<u8>` payloads during sort. Sort indices instead.
- **PID-based temp file naming** at the cloud writer too; replace with `tempfile`.

## 8. `scx-ops` — Mutations

### 8.1 Critical defects (P0)

- **Zero `shard_target_rows` infinite loop in append.** `scx-ops/src/append.rs:301-460`:
  ```rust
  let mut row_offset = 0usize;
  while row_offset < n_new_rows {
      let shard_rows = std::cmp::min(shard_target_rows as usize, n_new_rows - row_offset);
      ...
      row_offset += shard_rows;   // hangs forever when shard_target_rows == 0
  }
  ```
  Reachable from `scx append --shard-size 0` and `pyscx.append(..., shard_size=0)` because neither the CLI clap parser nor the Python wrapper applies a lower-bound check.
  **Fix path:**
  ```rust
  if shard_target_rows == 0 {
      return Err(OpsError::InvalidArgument { detail: "shard_target_rows must be > 0".into() });
  }
  ```
  Better: change the internal API to take `NonZeroU32`, add `clap::value_parser!(u32).range(1..)` on the CLI side, and raise `ValueError` before releasing the GIL on the Python side. Tests: `append_rejects_zero_shard_target_rows` (Rust), `scx append --shard-size 0` exits nonzero (CLI), `pyscx.append(..., shard_size=0)` raises `ValueError` (Python).

- **`codec_id` is silently ignored by `append_for_modality`.** `scx-ops/src/append.rs:73` renames the parameter to `_codec_id`. The function calls `let shard_codec = scx_format::select_codec(shard_values, value_encoding);` at line 329 and writes that into the shard header at line 397. CLI users requesting `--codec zstd`, `--codec none`, `--codec pcodec`, etc. and Python users passing `codec="zstd"` get auto-selected codecs anyway.
  **Fix path:** introduce an explicit selection type — don't overload `CodecId::None` to mean "no compression" and "auto":
  ```rust
  pub enum CodecSelection { Auto, Explicit(CodecId) }
  let shard_codec = match codec_selection {
      CodecSelection::Auto => scx_format::select_codec(shard_values, value_encoding),
      CodecSelection::Explicit(c) => c,
  };
  ```
  Tests: append with each explicit codec, then inspect the resulting shard headers; a separate `auto` test asserts per-shard selection still works.

### 8.2 Other correctness

- **`target_n_vars as u32` overflow on append** (`append.rs:407`). The shard header's `n_minor` is `u32`. `target_n_vars` is `u64` (a modality's `n_vars`). The writer-side `write_shard_inner` already checks `header_n_minor <= u32::MAX`; the append path does not. Mirror the validation:
  ```rust
  if target_n_vars > u32::MAX as u64 {
      return Err(OpsError::Format(scx_format::ScxError::NVarsOverflow(target_n_vars)));
  }
  ```
- **Append rewrites the whole merged `obs` Arrow IPC section.** `append.rs:466-483`: reads old obs, concatenates with new obs via `concat_batches`, upcasts to large types, writes the merged batch as a new `obs_metadata` section. Real append complexity is therefore `O(existing obs + new obs)` for metadata, not `O(new cells)` as the README implies. Options:
  - Near-term docs: clarify that matrix shards are O(new cells) but `obs` is rewritten.
  - Real fix: append-only `obs` chunks or `ObsMetadataDelta` chain that compact later coalesces.
- **`expect("FileLock already consumed")` × 8** in `flock.rs`. Unreachable under current safe-code use but a refactor footgun. Replace `Option<File>` with an enum or `ManuallyDrop` plus take/sentinel.
- **Crash-leaks orphan bytes.** A crashed append leaves stale bytes between the end of the previous catalog and the end of the file. The new append correctly writes past them, but repeated crashes amplify file size. `compact` cleans up; no automated trigger.
- **`info.n_csr_shards += n_new_csr_shards`** (`append.rs:622-624`) is unguarded u32 addition. Practically unreachable but still unguarded.
- **`rollback_to` lacks a max-step cap** (`rollback.rs:33-62`); if the manifest chain looped (possible after multi-rollback + re-append + rollback), only `manifest_sequence < target_sequence` saves it. Add a cap.
- **`delete.rs:194-195` calls `lock.seek(SeekFrom::End(0))` twice in succession.** Cache.

### 8.3 Wrapper memory: SCX-to-SCX append materializes the source

Both `scx-cli/src/append.rs` and `pyscx/src/ops.rs` call `read_all_csr_shards()` / `read_all_csr_shards_for()` on the source, convert arrays, and pass the entire CSR to `scx_ops::append`. Appending a 10 GB source matrix needs ~10 GB of host memory before any append begins.

Two-tier fix:
- Add a fast raw-copy path: if source and target have compatible `n_vars`, modality, value encoding, index dtype, and codec policy, copy raw shard bytes and rewrite only the affected headers / offsets.
- Streaming re-encode path otherwise: decode one source shard at a time, rebase row offsets, encode/write, release.
- Keep the current whole-CSR path as a fallback for incompatible cases.

A regression benchmark that records peak append RSS against source size would lock the fix in.

### 8.4 API surface

- **Long positional parameter lists.** `append`/`append_for_modality` take 8–9 positional arguments and use `#[allow(clippy::too_many_arguments)]`. This is what allowed `_codec_id` to be ignored without a compiler warning. Replace with a typed option struct:
  ```rust
  pub struct AppendOptions {
      pub codec: CodecSelection,
      pub shard_target_rows: NonZeroU32,
      pub modality_id: Option<NonZeroU8>,
      pub rebuild_csc: bool,
  }
  ```
- **Duplicated `encode_value`** (40 LOC verbatim) between `compact.rs:378-421` and `merge.rs:350-389`. Hoist to a shared helper.

## 9. `scx-accel` — Analysis Accelerators

### 9.1 Correctness / numerics

- **UMAP negative-sampling counter mis-update** (`umap.rs:210-238`). The loop body is capped at `n_neg.min(negative_sample_rate)`, but the schedule update at the end uses the un-clamped `n_neg`. When connectivity is high the schedule jumps too far, then subsequent epochs over-sample to compensate. Reference umap-learn uses the clamped count.
- **`covariance_pca` mean-centering subtracts in two-pass form** (`pca.rs:911-918`) `C -= n_obs * (μ μᵀ)`. Catastrophic-cancellation risk when `‖μ‖ ≫ σ`. The sister function `compute_total_variance_inmemory` already comments on this hazard (lines 765-770); `covariance_pca` does not.
- **`compute_total_variance_inmemory:786`** computes `n_zeros = n_obs as i64 - col_nnz[c]` and casts to f64. If `col_nnz[c]` ever exceeded `n_obs` (CSR invariant violation), the cast saturates to a huge positive f64. Clamp at zero.
- **Leiden `move_nodes_parallel` "empty community" guard** (`leiden.rs:1003`) allocates one empty community per batch; multiple proposed moves to the same ID compete, and only the first sees an empty community. This diverges from libleidenalg semantics on the second-and-later moves.
- **Connectivity self-edges** (`neighbors.rs:477-496`) survive as `2μ - μ²` rather than being dropped. umap-learn drops them upstream.
- **Harmony `HarmonyConfig::sigma`** is `f64` but the doc comment claims per-cluster sigmas are supported via array expansion. The struct has no array form.
- **`hvg.rs:68 streaming_mean_var` with `n=1`** divides by `max(1, n-1) = 1`, returning variance 0 where math says undefined. scanpy expects NaN.

### 9.2 Performance

- **`sparse_outer_product_accumulate_par`** (`pca.rs:127`) allocates an `n_vars × n_vars` matrix per thread — ~30 MB at n_vars=2000 × 64 threads = ~2 GB transient. Tile by column block.
- **`covariance_pca` streams the matrix twice**; the embeddings could be derived from the eigendecomposition itself.
- **Leiden `weight_to_comm`** (`leiden.rs:633`) is O(deg × |candidates|); a neighbor-community-weight map built once per node avoids the blowup.
- **Leiden calls `libc::malloc_trim(0)` every outer iteration** (`leiden.rs:952-955`). On glibc this is expensive (walks all arenas) and Linux-only. Root cause is `HashMap<(usize,usize),f64>` churn in `aggregate()`; switching to a sorted vec removes the leak.
- **Exact kNN allocates `Vec::with_capacity(n_obs - 1)` per row** (`neighbors.rs:335`) — at the 5K-cell threshold this is ~2 GB transient. Use a thread-local reusable buffer.

### 9.3 API drift

- `docs/api.md:458` lists `gpu_randomized_pca(dev, reader, n_components, n_oversamples, n_power_iterations, zero_center, seed)`; actual signature also takes `qr_method: QrMethod`.
- `docs/api.md:474` `gpu_umap_native` signature is materially shorter than the actual function.

## 10. `scx-gpu` — CUDA Paths

### 10.1 Correctness

- **cuVS layout pin emits only a `WARNING` to stderr on mismatch** (`gpu_knn.rs:402-411`) and proceeds with potentially-misaligned struct reads. Hard-fail unless `SCX_CUVS_TRUST_LAYOUT=1`.
- **`gpu_knn.rs:660,704,728,748` hard-code `DLDevice { device_id: 0 }`** — multi-GPU users hitting `device="gpu:1"` get neighbors from device 0.
- **`gpu_randomized_pca` pre-flight memory estimate is missing terms** (`gpu_pca.rs:116-117`): omits `Ω` (`n_vars × k`), the cusparse workspace, and double-buffered shard buffers. Real peak exceeds the estimate.
- **CAGRA output buffers `search_k * n_obs * (u32+f32)`** (`gpu_knn_cagra.rs:606-611`) — at 10M cells × k=50 that's 4 GB just for output, with no pre-flight check.
- **`module_cache` is `RefCell`** (`device.rs:29`); `GpuDevice` is `!Sync`. Today only sequential shard loops are used. Blocks any future per-shard parallel GPU work.

### 10.2 Performance

- **Per-shard `upload_csr_to_gpu` redundancy.** `gpu_pca.rs:407,481` uploads indptr/indices/data H→D for forward and transpose, and again across power iterations — `3 × n_shards` redundant uploads.
- **Per-call `cuvsResourcesCreate/Destroy`** (`gpu_knn.rs:614-619`); cache one per device.
- **f32 → f64 upcast and host transpose** in `gpu_pca.rs:236-241,298-302` are unnecessary; faer has `from_column_major_slice` and cuBLAS can transpose on-device with sgeam.

### 10.3 Build hygiene

The workspace's `default-members` includes `scx-gpu`. `scx-gpu` depends unconditionally on `cudarc` with the `cuda-12020` feature. On a CPU-only environment without CUDA installed, `cargo build --workspace` may encounter `cudarc` build-script or link-time friction depending on `cudarc`'s build script. Either:
- guarantee and document that `cargo build --workspace` works without CUDA (verify `cudarc`'s behavior when no toolkit is present);
- or remove `scx-gpu` from `default-members` and build it only with an explicit `--features gpu` CI job;
- or add CI jobs for both CPU-only and GPU builds. `docs/development.md` should make the contract explicit.

## 11. `scx-mtx` — Matrix Market

- **Integer detection misses negatives** (`write.rs:83`): `v >= 0.0 && v == v.floor()`. Negative integers are written as "real". Reader at `read.rs:157-164` accepts both; the writer is the inconsistent side.
- **Trusts declared `nnz`** (`read.rs:194`); a malicious file with `nnz=1_000_000_000` allocates 24 GB before the entries-count check at line 224.
- **Codec parser asymmetry**: `--from mtx` rejects `lz4`/`pcodec`; `--from h5ad` accepts them. CLI advertises the same flag for both.
- **No round-trip MTX → SCX → MTX test.**
- **Cell Ranger v2/v3 metadata conventions** are not asserted by tests; gene/barcode feature files should be parity-tested.

## 12. `scx-cli`

- **`main.rs:553-562` conversion-direction `match` arm guards.** In Rust, `pat1 | pat2 if guard` applies the guard to both patterns. So `scx convert --from h5ad input.scx` (user explicitly states input is h5ad) only matches if the input file extension is `.h5ad`. Same on lines 557 (`10x`) and 560 (`scx_to_h5ad`). Net effect: explicit `--from` / `--to` flags are partially ignored when the extension doesn't match. The fall-through error "Cannot determine conversion direction" hides the cause. Bug.
- **No lower-bound validators on numeric options.** `--shard-size` (the cause of the zero-shard hang in `scx-ops`), `--parallelism`, `--runs`, etc. should use shared `value_parser` helpers:
  ```rust
  fn positive_u32(s: &str) -> Result<u32, String> { ... }
  fn positive_usize(s: &str) -> Result<usize, String> { ... }
  fn nonzero_or_documented_zero(s: &str) -> Result<usize, String> { ... }
  ```
  Centralize and reuse. Leave `csc_cols_per_shard=0` legal only where its help text explicitly defines "0 = single shard / no cap".
- **`pack.rs` and `explode.rs` are 7-line stubs.** Either implement or remove from the command surface.
- **`Cli::Convert.codec: String`** is validated at dispatch time; clap's `value_parser` could front-load the check.
- **`BuildCsc::--memory_limit` parses K/M/G suffixes inside the command handler**; `value_parser` would surface errors at parse time.
- **`convert/mod.rs` is 778 lines**; splitting per-operation sub-modules would help discoverability.
- **CSC-rebuild messaging.** Help text on mutation commands surfaces `--rebuild-csc`; the README/docs should add an operations matrix (see §17 P2) so users do not lose CSC sidecars silently.

## 13. `scx-sparse`

- **`to_dense` claim partly wrong** (`csr.rs:230-247`). The doc says invariant violations are "logic bugs, not memory-safety UB". If invariant 5 (`col < n_cols`) is violated, `dense[row * n_cols + col]` is an in-bounds write to the wrong column — silent corruption, not panic. Strengthen the validation in `new_unchecked` or revise the doc claim.
- **`col_nnz` returns `Vec<i64>`** (`csr.rs:285`) where `Vec<u32>` would be sufficient and avoid the awkward signed-cast pattern at `scx-accel/src/pca.rs:786`.
- **`col_max` on `total_n_obs == 0` returns `NEG_INFINITY`** with no error — easy to mistake for valid max.
- **Two-pass mean+var** in `row_var` (`csr.rs:321-354`); Welford would be more numerically robust.
- **Property-test coverage gaps.** Add proptests for CSR/CSC slicing, concatenation, row/column projection, empty matrices, single-row shards, max index values, and the unsorted-input rejection policy.

## 14. `pyscx` — Python Bindings

### 14.1 AnnData round-trip fidelity (largest gap)

**`obsp`, `varp`, and `varm` are never read or written.** The only mention of any of these in `pyscx/src/anndata.rs` is a single comment at line 424. `from_anndata_impl` extracts only `X`, `obs`, `var`, `obsm`, `uns`, `layers`. Scanpy's `sc.pp.neighbors` / `sc.tl.umap` populate `obsp["connectivities"]` and `obsp["distances"]`; these are silently dropped. `pyscx.accel.pca` itself writes to `adata.varm["PCs"]` (`accel/pca.rs:524-525`) — so the binding produces data it then refuses to persist. This is the most user-visible defect in the entire codebase.

### 14.2 GIL handling

- The accelerator wrappers do not `py.allow_threads`. `pca.rs:385-433`, `umap.rs`, `neighbors.rs`, `de.rs`, `hvg.rs`, and `accel/preprocessing.rs::normalize_total` all call into `scx_accel::*` with the GIL held. PCA on 1M cells blocks every other Python thread for minutes. By contrast `ops.rs:144,275,335,383`, `cloud.rs:66,88,129,251`, `harmony.rs:214`, and `query.rs:167` do release the GIL. Free win, no semantic change.

### 14.3 Zero-copy claim is misleading

- `csr_to_scipy` (`anndata.rs:136-153`) uses `PyArray1::from_vec` which moves a Rust `Vec` into NumPy — true ownership transfer. But the upstream `read_all_csr_shards*` always allocates fresh `Vec`s. End-to-end is one-copy-per-read, not zero-copy from mmap. `tests/test_zero_copy.py` (43 lines) only checks dtypes; it does not verify `arr.flags.OWNDATA` or buffer identity, and tolerates `indptr` being downcast. The "zero-copy" claim in the docstring at `anndata.rs:135-153` overstates what the implementation delivers.

### 14.4 Error-type mapping is too coarse

- `ops_to_pyerr` (`pyscx/src/ops.rs:29-34`) maps only `OpsError::IncompatibleVars` to `PyValueError`; every other variant becomes `PyRuntimeError`. The ops layer has shape, schema, index-out-of-bounds, and var-length mismatch errors that are clearly user-input bugs and should be `ValueError`. The general `to_pyerr` (`pyscx/src/lib.rs:23`) maps every `ScxError` to `PyRuntimeError`, losing structure. Recommended policy:

  | Rust error | Python exception |
  |---|---|
  | shape / schema / index / var mismatch | `ValueError` |
  | unknown modality | `ValueError` or `KeyError` |
  | missing file | `FileNotFoundError` |
  | permission denied | `PermissionError` |
  | lock failure | `RuntimeError` or custom `ScxLockError` |
  | corrupt SCX file | `RuntimeError` or custom `ScxFormatError` |

  `ops_to_pyerr` already does this for one variant; generalize the pattern.

### 14.5 Other issues

- **No `shard_size > 0` validator on the Python side.** `pyscx.append(..., shard_size=0)` happily reaches Rust and hangs (see §8.1). Add a `ValueError` before `py.allow_threads`.
- **`from_anndata_impl` clones numpy buffers** (`anndata.rs:1272-1274`) to escape the GIL boundary, where `BorrowedCsrSource` in `accel/pca.rs:69-105` already shows the right pattern.
- **`extract_materialized_csr` re-clones three times** (`pca.rs:442-488`): `.extract::<Vec<T>>()`, then keeps the Vecs in `ScxCsr::new_unchecked`, then clones again in `read_shard`. Phase-10 borrowed fast lane (`try_extract_borrowed_csr`) only fires on the GPU path.
- **No `py.typed` marker** (verified). PEP 561 requires the empty `py.typed` file alongside the `.pyi` stubs; without it, downstream consumers get `Any` for every pyscx symbol regardless of stubs.
- **`pyproject.toml` is missing PyPI metadata**: no `readme`, no `urls`, no `classifiers`.
- **`pyproject.toml` declares `requires-python = ">=3.9"` but `accel.pyi` uses PEP 604 `|` union types** (3.10+). Either bump the floor or use `Union[...]`.
- **`pyproject.toml` runtime dependencies are unbounded.** `numpy`, `scipy`, `pyarrow`, `anndata` have no version constraints. For a native extension that links against pyarrow's C++ ABI and depends on AnnData's evolving Python API, this will break. Tested bounds based on CI coverage:
  ```toml
  dependencies = [
    "numpy>=1.24,<3",
    "scipy>=1.10,<2",
    "pyarrow>=14,<19",
    "anndata>=0.10,<0.13",
  ]
  ```
  A compatibility matrix in docs and CI lanes for the bounds is the right follow-on.
- **No `[gpu]` extra in `pyproject.toml`**, contradicting `docs/gpu-setup.md`'s implication that `pip install pyscx[gpu]` works. Users must invoke `maturin develop --features gpu` manually.
- **`uns` round-trip is lossy by design** (NumPy arrays become Python lists). Documented at `lib.rs:90-92`. The lossiness combined with `pyscx.accel.*` writing NumPy into `uns` (`pca`, `de`, etc.) means `to_anndata()` of a freshly-modified experiment loses array structure on the next persist. At minimum, surface a writer-time warning or use Arrow IPC for `uns` arrays.
- **Backed mode opens N+3 readers per file** (`anndata.rs:504-749`); a single shared `Arc<ScxReader>` would suffice.
- **`pyscx/src/lib.rs` `#[pymodule]` is large.** Continue the existing `register_*` pattern and move bindings into domain-specific Rust modules with local registration functions.

### 14.6 Tests

- `test_round_trip.py` (114 lines) never tests `obsp`/`varp`/`varm`, never verifies array round-trip in `uns`.
- `test_zero_copy.py` is intentionally lax and tolerates downcast — it does not detect a non-zero-copy implementation.
- No PyO3 lifetime stress test (e.g., `ScxBackedSparseDataset` outliving its `PyExperiment` parent, which `anndata.rs:512-515` claims works).
- No test of `pyscx.append(..., shard_size=0) → ValueError`.
- No test of `pyscx.open(..., verify=True)` documenting catalog-only verification, paired with a test that `pyscx.validate(path)` catches a deliberately corrupted section payload.

## 15. `rscx` — R Bindings

- **R↔Rust marshalling routes every dgCMatrix slot through an `R!()` template string** (`interop.rs:246-291,315-323,340-350,1105-1113`); each slot extraction allocates a fresh R vector and copies into a `Vec<T>`. A 1M-cell matrix moves ~50–100 MB extra through the R interpreter.
- **`dgcmatrix_to_csr` (`interop.rs:376-409`) makes 4 R round-trips per matrix** (one per slot). Use a single `R!()` returning a list.
- **No dgCMatrix invariant validation**: doesn't check `@p` is sorted, `@i` is sorted within columns, lengths match, or that `@x` has no NaN/Inf. `Matrix::dgCMatrix` permits non-canonical instances.
- **All numeric obs/var columns are downcast to `f64`** (`interop.rs:558-565`). Integer cell counts and parsed barcode IDs lose dtype info, where the Python side preserves them.
- **NaN-as-NA heuristic is wrong** (`interop.rs:562`): `v.is_nan() → None`. R's `NA_real_` has a specific NaN bit-pattern; a legitimate user NaN value becomes NA on write. Use `Robj::is_na()` instead.
- **No row.names round-trip** (`interop.rs:26-49`); cell barcodes stored as the index column are destroyed.
- **dgCMatrix hard wall at i32::MAX nnz** (`interop.rs:232-237`) — at Census-1M scale (~3.5B nnz), the R binding is unusable for the dense end of the dataset spectrum. No `dgRMatrix` / `dgTMatrix` fallback.
- **Seurat round-trip drops `data` and `scale.data` layers** (`interop.rs:874-977`); a Seurat v5 multi-layer assay collapses to counts-only on import.
- **`obsm` / `reductions` round-trip is missing in the single-modality path.**
- **`meta.features` silently fallback** to `data.frame(gene_id = rownames)` on error (`interop.rs:937`), losing the entire `@meta.features` data frame.
- **DESCRIPTION lists Seurat as Suggests** but no smoke test ensures graceful failure on Seurat v4.
- **R API parity documentation against Python is missing.** A short feature table in `docs/api.md` (or a dedicated `docs/rscx.md`) would set expectations.

## 16. Cross-Cutting

### 16.1 `unsafe` and FFI

`unsafe` usage in `scx-gpu` is concentrated at FFI boundaries to cudarc and cuVS — appropriate. No `unsafe impl Send/Sync` for non-thread-safe types other than the documented cuSPARSE/cuVS handles. The cudarc `launch_builder` `unsafe` is unavoidable. No findings.

### 16.2 Error type duplication

Each crate has its own `thiserror` enum, which is per convention. But: `CodecError::Io(io::Error::other(...))`, `GpuError::CudaError(String)`, and similar string-typed inner errors lose structure. A workspace-internal `SourceError` newtype (or `#[from] OtherError` for the common cases) would tighten things. The error taxonomy should also be standardized across Rust, Python, CLI, and R — see §14.4.

### 16.3 Duplicate code worth consolidating

- Four `floor_median_*` functions (delta_golomb, rice, codec_select, value_encoding).
- Two `encode_value` blocks (compact, merge).
- Loader-config builder boilerplate duplicated between `TrainingDataset::new` and `MultimodalTrainingDataset::new`.
- Test fixture `sample_header` boilerplate copied 6× in `scx-engine/tests/`.
- `record_batch_to_arrow_ipc` helpers spread across `scx-cloud/src/pull.rs:1043` and `scx-ops/src/append.rs:475-483`.
- Padding writes: `writer.write_all(&vec![0u8; pad])` repeated at multiple sites with `pad ≤ 7`. `ScxWriter::write_padding()` already uses a stack `[0; 7]`; expose it (or a `write_8byte_padding` helper) and reuse.

### 16.4 Untrusted-input handling

The format spec advertises strong checksum coverage, but readers don't take precautions against attacker-influenced length fields used as `Vec::with_capacity`. A simple defensive pattern — clamp every on-disk `u32 length` against the surrounding section's actual byte range — would close most of the DoS surfaces in §3.1 with a few lines. The same applies to the MTX reader's `nnz` trust (§11).

### 16.5 Public-contract pinning

The most leveraged process improvement is to make public contracts executable:

- For every README claim that is a behavior guarantee (append complexity, streaming pull, exact selective filtering, zero-copy, BLAKE3 verification, throughput), add a regression test or a benchmark manifest.
- Maintain a benchmark manifest format with `benchmark_id`, repo commit, dataset checksum, hardware, software versions, command line, and results. Every README-visible number should point to one manifest. A release-checklist item should refuse to merge a README change that adds a number without a manifest entry.
- TBD / pending benchmark rows should not appear in shipped docs.

## 17. Prioritized Recommendations

Priority key:
- **P0** — Can hang, corrupt data, silently ignore an explicit user request, or severely violate a public contract.
- **P1** — High-impact correctness, reliability, or scalability issue.
- **P2** — Moderate issue affecting maintainability, usability, documentation accuracy, performance, or DX.
- **P3** — Cleanup or polish.

### P0 — Ship blockers

1. [x] Reject `shard_target_rows == 0` in `scx-ops::append_for_modality` (`scx-ops/src/append.rs:73`); validate the same at the CLI (`scx-cli`) and pyscx (`pyscx/src/ops.rs`) boundaries with friendly error messages. Switch the internal API to `NonZeroU32`.
2. [x] Honor the `codec_id` parameter in `scx-ops::append_for_modality`. Introduce `CodecSelection::{Auto, Explicit(CodecId)}` to avoid overloading `CodecId::None`. Test every legal explicit codec.
3. [x] Fix `ValueEncoding::Float16` to either emit 2-byte f16 or refuse Float16 inputs at the writer (`scx-codec/src/value_encoding.rs:75-89`). Add a conformance test that round-trips Float16-labeled shards.
4. [x] Replace the `debug_assert!`-gated index validity checks in `scx-codec/src/dispatch.rs:305`, `bitstream.rs:194`, `shuffle.rs:12`, `forbp.rs:122`, `delta_golomb.rs:63`, `rice.rs:101` with runtime checks that return `CodecError`. Add proptest cases that feed malformed inputs.
5. [x] Fix `scx-cli/src/main.rs:553-562` convert direction match: the guard binds to both patterns, so explicit `--from` / `--to` flags are silently ignored when the file extension doesn't match. Refactor to evaluate explicit flags first and fall back to extension sniffing.
6. [x] Add `obsp` / `varp` / `varm` round-trip in `pyscx/src/anndata.rs`. `pyscx.accel.pca` already writes `varm["PCs"]`; the writer must persist it.
7. Replace `scx-cloud::pull()`'s "buffer all sections in memory" implementation with a bounded streaming pipeline. Either implement `PullOptions::reorder_buffer` properly (it is currently dead) or remove the option from the public API.
8. Reconcile the `scx-cloud` selective-pull semantics with the documentation. Either rename the feature to "selective shard pull" and explain superset semantics in the README/CLI, or add an `--exact` / `mode="exact"` cell-granular path. In either case, surface the omitted-section-types summary on filtered pull.
9. [x] Reject `shard_size <= 0` in `pyscx.append` with `ValueError` before crossing into Rust.

### P1 — Important, low-effort

10. [x] Validate header reserved fields in `scx-format/src/header.rs::read_from`: `header_length == HEADER_SIZE`, `reserved_padding == 0`, `flags & !KNOWN_FLAGS == 0`, and trailing `reserved` array all-zero. Add corrupt-header fixtures.
11. [x] Add `target_n_vars` overflow guard in `scx-ops/src/append.rs:407` (mirror the writer-side check in `scx-format`).
12. [x] Bound every `Vec::with_capacity(u32 as usize)` allocation in `scx-format` against the surrounding section's declared length. Apply the same to `scx-mtx::read`.
13. [x] Replace PID-based temp paths with `tempfile::Builder::tempfile_in(parent_dir)` in `scx-format/src/writer.rs` (3 sites) and `scx-cloud/src/pull.rs`. Fsync the parent directory on Unix after `rename`.
14. [x] Document and clarify `obs` rewrite cost in append; either add an append-only obs chunk format or correct the README claim to `O(new cells + existing obs)`.
15. Add a streaming SCX → SCX append path that avoids `read_all_csr_shards()` (CLI + pyscx wrappers).
16. Move `IndexPlanLoader::new` tokio runtime construction into the worker thread, matching `TrainingPipeline::start_epoch` (`scx-loader/src/index_plan.rs:349-358`).
17. [x] Release the GIL around `pyscx.accel.*` Rust calls — wrap each Rust-heavy block in `py.allow_threads`.
18. [x] Resolve the README/AGENTS vs scanpy.md/ROADMAP peak-RSS contradiction (5.1 GB / 88 % vs 11 GB / 71 %).
19. [x] Update `docs/architecture.md:421-423` to match the current `num_workers` contract.
20. [x] Update `docs/multithreading.md:75-85` vs `:261` (current-thread vs 2-worker tokio runtime).
21. [x] Distinguish catalog verification from payload verification in README, `docs/format.md`, CLI help, and cloud docs:
    > SCX stores BLAKE3 checksums for catalogs and sections. `pyscx.open(..., verify=True)` verifies the catalog (offsets, lengths, stored section checksums are authenticated). Use `pyscx.validate(path)` or `scx validate path` to re-hash every section payload.
22. [x] Document `file_checksum` semantics in `docs/format.md`: what bytes are covered, zero-during-computation, which APIs verify it, what append/rollback do to it, and how it relates to per-section checksums.
23. [x] Tighten `pyscx::to_pyerr` and `pyscx::ops_to_pyerr` to map errors to specific Python exceptions instead of bucketing into `PyRuntimeError`.
24. [x] Add a `py.typed` file to `pyscx/python/pyscx/`, declare its inclusion in `[tool.maturin].include`, bump `requires-python` to `>=3.10`, and add `numpy`/`scipy`/`pyarrow`/`anndata` version bounds.
25. Add `[project.optional-dependencies] gpu = ["..."]` and `cloud = ["..."]` extras to `pyscx/pyproject.toml`.
26. Make `CLAUDE.md` a five-line pointer to `AGENTS.md` (currently byte-identical). - DEFERRED
27. [x] Hard-fail (rather than warn) on cuVS struct-layout mismatch in `scx-gpu/src/gpu_knn.rs:402-411`. Gate the warning-only behavior behind `SCX_CUVS_TRUST_LAYOUT=1`.
28. [x] Wire `device_id` through DLPack tensors in `scx-gpu/src/gpu_knn.rs:660,704,728,748` so `device="gpu:N"` is honored.
29. [x] Fix the UMAP negative-sampling counter mismatch (`scx-accel/src/umap.rs:210-238`).
30. [x] Add a `write_preencoded_shard` validation in `scx-format/src/writer.rs:644-674` — either count CSC sections correctly or reject non-CSR section types.

### P2 — Polish

31. Replace long positional argument lists in mutation APIs with `AppendOptions` / `CompactOptions` / `MergeOptions` typed structs (would have caught the `_codec_id` bug at compile time).
32. Centralize CLI numeric validators (`positive_u32`, `positive_usize`, `nonzero_or_documented_zero`).
33. [x] Add an operations matrix to README or `docs/operations.md`:

    | Operation | Matrix rewrite? | Obs rewrite? | CSC preserved? | How to restore CSC |
    |---|---|---|---|---|
    | append | Existing CSR preserved; new CSR appended | `obs` rewritten as merged Arrow IPC | Dropped | `--rebuild-csc` or `scx build-csc` |
    | delete | Logical deletion vector | No rewrite | Depends | compact / rebuild |
    | compact | Rewrites live data | Rewrites live metadata | Dropped unless `--rebuild-csc` | `--rebuild-csc` |
    | merge | Writes new output | Writes merged metadata | Dropped unless `--rebuild-csc` | `--rebuild-csc` |
    | subset | Writes new output | Writes subset metadata | Dropped unless `--rebuild-csc` | `--rebuild-csc` |

34. [x] Reorganize README: keep installation, minimal Python/R/CLI examples, format status, docs index, and one concise performance summary. Move detailed benchmarks to `docs/performance.md`; competitive comparisons to `docs/comparison.md`; mutation semantics to `docs/operations.md`; fork-safety/loader design to `docs/multithreading.md` / `docs/training-loader.md`.
35. Introduce a benchmark manifest (`benchmark_id`, `repo_commit`, `dataset.checksum`, `hardware`, `software`, `commands`, `results`). Add a release-checklist item that no README-visible benchmark number is allowed without a manifest entry. Remove TBD/pending rows from shipped docs.
36. Add `docs/development.md` covering: CPU-only build, HDF5 feature build, cloud feature build, GPU feature build, Python editable install, R binding build, test matrix, benchmark conventions, fuzzing commands.
37. Either guarantee `cargo build --workspace` works on CPU-only machines (and pin this in CI) or remove `scx-gpu` from `default-members`.
38. Consolidate `floor_median_*` into a single utility in `scx-codec`.
39. Consolidate the `encode_value` duplication between `scx-ops/src/compact.rs` and `scx-ops/src/merge.rs`.
40. Add a shared `write_8byte_padding` helper used by `scx-format`, `scx-ops`, and `scx-cloud`.
41. Replace `expect("FileLock already consumed")` sites in `scx-ops/src/flock.rs` with a typed sentinel.
42. Promote `ScxError::ChecksumMismatch` to carry the offending section identifier.
43. Add request timeouts and a retry layer (with backoff jitter) to `scx-cloud`; introduce `Timeout` and `RateLimited` variants to `CloudError`.
44. Add hash-while-write to `scx-cloud/src/pull.rs:430-435`.
45. [x] Add a workspace-level "untrusted input" fuzz target that randomizes the catalog, provenance, deletion-vector, and shard headers with arbitrary `u32` length fields.
46. Add an MTX → SCX → MTX round-trip test and Cell Ranger v2/v3 metadata parity tests.
47. Add a Seurat v4 graceful-failure smoke test and Seurat v5 multi-layer round-trip in `rscx/tests/testthat/`.
48. Add `Vec<f64>` → `Vec<u32>` for `ScxCsr::col_nnz` and propagate the change to `scx-accel/src/pca.rs:786`.
49. Update `ROADMAP.md` (line 3 date stamp; prune Phase 2 pitfalls and risk-register entries that have been resolved).
50. Add a sequence diagram of the triple-buffered pipeline to `docs/multithreading.md`; add a catalog-chain mutation diagram to `docs/format.md`; add a cloud-pull topology diagram to `docs/cloud.md`.

### P3 — Cleanup

51. Move `MajorAxis` to `pub(crate)` in `scx-format/src/writer.rs`.
52. Remove `#[deprecated]` `ScxReader::read_shard_from_entry_unchecked` (or schedule for next major).
53. Remove dead `SectionType::is_known` in `scx-format/src/section.rs`.
54. Add a `log::warn!` at `scx-format/src/catalog.rs:532-534` when an unknown section type is skipped, matching spec.
55. Drop the unused `var_predicates`, `obs_predicate_index`, `var_predicate_index` fields in `scx-engine/src/collect.rs:38-41` (or wire them up; right now they're just `#[allow(dead_code)]` static guilt).
56. Replace `panic!()` in `scx-engine/src/fused_ops.rs:408` with `EngineError::SchemaError`.
57. Update the stale `format_version` field comment at `scx-format/src/header.rs:32` ("currently 1" → "currently 2"). Add a small checklist item when bumping format versions.
58. Fix `docs/format.md:581` "≥13" → "≥17" (or remove the boundary now that 13/14 are defined).
59. Fix `AGENTS.md:51` "15 section types" → "17".
60. Stamp `ROADMAP.md:3` to today's date or remove.
61. Continue moving pyscx bindings into domain-specific Rust modules with local `register_*` functions; consider a small macro for repetitive `m.add_function(wrap_pyfunction!(...))` patterns.

## 18. Suggested near-term patch plan

A practical patch sequence that maximizes risk reduction per merge:

**Patch 1 — Append hardening (P0 #1, #2, #9; P1 #11)** - RESOLVED
- Add `OpsError::InvalidArgument`.
- Reject `shard_target_rows == 0` in `scx-ops`.
- Add CLI/Python validators (`clap::value_parser!(u32).range(1..)`; Python `ValueError`).
- Add `target_n_vars > u32::MAX` overflow check.
- Introduce `CodecSelection::{Auto, Explicit}` and honor it in `append_for_modality`.
- Tests: zero shard size, explicit codecs, auto codec, overflow.

**Patch 2 — CLI convert direction fix (P0 #5)** - RESOLVED
- Refactor `scx-cli/src/main.rs:553-562` to honor explicit flags before extension sniffing.
- Tests: explicit `--from h5ad input.scx`, explicit `--to h5ad` with non-`.h5ad` extension, etc.

**Patch 3 — Float16 / debug_assert codec fixes (P0 #3, #4)** - RESOLVED
- `value_encoding.rs`: emit 2-byte f16 or reject Float16 inputs.
- `dispatch.rs`, `bitstream.rs`, `shuffle.rs`, `forbp.rs`, `delta_golomb.rs`, `rice.rs`: replace `debug_assert!` with runtime returns.
- proptest corpora for malformed inputs.

**Patch 4 — Header validation (P1 #10)** - RESOLVED
- Validate `header_length`, `reserved_padding`, `flags` reserved bits, trailing `reserved`.
- Fix stale comment at line 32.
- Corrupt-header fixtures.

**Patch 5 — Cloud pull memory + semantics (P0 #7, #8)** - RESOLVED
- Replaced all-sections buffering with bounded `buffer_unordered` + BTreeMap reorder window, wiring up the previously-dead `reorder_buffer` option. Peak memory is now bounded by `reorder_buffer × max_section_size`.
- Added `FilterMode::Shard` / `FilterMode::Exact` enum; `Shard` is the default (shard-granular, fast). `Exact` (cell-granular decode/filter/re-encode) is deferred to a follow-up patch and returns a clear error when requested.
- Selective pull now reports omitted section types in `PullFilteredStats.omitted_section_types`.
- Added `--filter-mode` CLI flag and `filter_mode` Python parameter.
- Removed dead `PushOptions.multipart_threshold` field.
- Updated `docs/cloud.md` (streaming model, shard-granular semantics, omitted sections, tuning table).
- Five new tests: streaming reorder window, buffer_unordered vs sequential, filter mode reporting, shard-mode extra cells pinning, exact-mode error rejection.

**Patch 6 — Temp-file durability (P1 #13)** - RESOLVED
- Replace PID temp names with `tempfile::Builder::tempfile_in`.
- Parent-directory fsync after `rename` on Unix.
- Concurrent-writer tests.

**Patch 7 — pyscx round-trip and error mapping (P0 #6, P1 #17, #23, #24)** - RESOLVED
- Persist `obsp` / `varp` / `varm`.
- `py.allow_threads` around accel calls.
- Tighten `to_pyerr` / `ops_to_pyerr`.
- Add `py.typed`, version-bound deps, bump Python floor.

**Patch 8 — Documentation correction pass (P1 #14, #18–22; P2 #33–34)** - RESOLVED
- Update append complexity claim.
- Update selective-pull semantics in README and `docs/cloud.md`.
- Clarify checksum verification language.
- Resolve the peak-RSS / num_workers / runtime-model contradictions.
- Add the CSC operation matrix and `file_checksum` definition.
- Reorganize README into a concise entry point.

**Patch 9 — Defensive input handling (P1 #12, #30; P2 #45)** - RESOLVED
- Bound every `Vec::with_capacity(u32 as usize)` allocation in `scx-format` (`catalog.rs:498`, `provenance.rs:67,83`, `deletion_vectors.rs:79`, `shard.rs:300`) against the surrounding section's declared byte length. Apply the same pattern to `scx-mtx::read` (`read.rs:194`).
- Add `write_preencoded_shard` validation in `scx-format/src/writer.rs:644-674`: either count CSC sections correctly or reject non-CSR section types.
- Add a workspace-level "untrusted input" fuzz target that randomizes catalog, provenance, deletion-vector, and shard headers with arbitrary `u32` length fields.
- Tests: over-sized allocation rejected, CSC-via-preencoded counted correctly, fuzz harness runs without panic.

**Patch 10 — GPU + accel correctness (P1 #27, #28, #29)** - RESOLVED
- [x] Hard-fail on cuVS struct-layout mismatch in `scx-gpu/src/gpu_knn.rs:402-411`; gate warning-only behavior behind `SCX_CUVS_TRUST_LAYOUT=1` env var.
- [x] Wire `device_id` through DLPack tensors in `scx-gpu/src/gpu_knn.rs:660,704,728,748` so `device="gpu:N"` is honored on multi-GPU hosts.
- [x] Fix UMAP negative-sampling counter mismatch in `scx-accel/src/umap.rs:210-238` to use the clamped count (matching reference umap-learn).
- [x] Tests: layout-mismatch hard-fail, multi-device tensor placement, UMAP negative-sample schedule matches reference.

**Patch 11 — Loader + cloud hardening (P1 #16; P2 #43, #44)**
- Move `IndexPlanLoader::new` tokio runtime construction into the worker thread, matching the `TrainingPipeline::start_epoch` lazy-runtime pattern (`scx-loader/src/index_plan.rs:349-358`).
- Add request timeouts and a retry layer with exponential backoff + jitter to `scx-cloud`; introduce `Timeout` and `RateLimited` variants to `CloudError`.
- Add hash-while-write to `scx-cloud/src/pull.rs:430-435` to avoid the redundant multi-GB re-read for checksum computation.
- Tests: IndexPlanLoader safe in forked context, transient-error retry, timeout enforcement, hash-while-write matches re-read hash.

**Patch 12 — API ergonomics + dedup (P1 #15, #25, #26; P2 #31, #32, #38, #39, #40, #41, #42)**
- Add a streaming SCX → SCX append path that avoids `read_all_csr_shards()` in CLI and pyscx wrappers (decode one source shard at a time).
- Add `[project.optional-dependencies] gpu = [...]` and `cloud = [...]` extras to `pyscx/pyproject.toml`.
- Replace long positional argument lists in mutation APIs with typed option structs (`AppendOptions`, `CompactOptions`, `MergeOptions`).
- Centralize CLI numeric validators (`positive_u32`, `positive_usize`, `nonzero_or_documented_zero`).
- Consolidate `floor_median_*` into a single utility in `scx-codec`.
- Consolidate `encode_value` duplication between `scx-ops/src/compact.rs` and `scx-ops/src/merge.rs`.
- Add a shared `write_8byte_padding` helper used by `scx-format`, `scx-ops`, and `scx-cloud`.
- Replace `expect("FileLock already consumed")` sites in `scx-ops/src/flock.rs` with a typed sentinel.
- Promote `ScxError::ChecksumMismatch` to carry the offending section identifier.
- Tests: streaming append peak RSS bounded, optional extras installable, padding helper used everywhere.

**Patch 13 — Build + CI hygiene (P2 #36, #37)**
- Add `docs/development.md` covering: CPU-only build, HDF5 feature build, cloud feature build, GPU feature build, Python editable install, R binding build, test matrix, benchmark conventions, fuzzing commands.
- Either guarantee `cargo build --workspace` works on CPU-only machines (and pin this in CI) or remove `scx-gpu` from `default-members`.
- Tests: CI job for CPU-only workspace build, documented build matrix matches CI.

**Patch 14 — Test coverage expansion (P2 #46, #47, #48; P1 #15 follow-on)**
- Add MTX → SCX → MTX round-trip test and Cell Ranger v2/v3 metadata parity tests.
- Add a Seurat v4 graceful-failure smoke test and Seurat v5 multi-layer round-trip in `rscx/tests/testthat/`.
- Change `ScxCsr::col_nnz` return type from `Vec<f64>` to `Vec<u32>` and propagate to `scx-accel/src/pca.rs:786`.
- Tests: MTX round-trip, CellRanger metadata, Seurat v4 fallback, col_nnz type propagation.

**Patch 15 — Benchmark + ROADMAP refresh (P2 #35, #49, #50)**
- Introduce a benchmark manifest format (`benchmark_id`, `repo_commit`, `dataset.checksum`, `hardware`, `software`, `commands`, `results`). Add a release-checklist item that no README-visible benchmark number is allowed without a manifest entry. Remove TBD/pending rows from shipped docs.
- Update `ROADMAP.md`: refresh line 3 date stamp, prune resolved Phase 2 pitfalls and risk-register entries.
- Add a sequence diagram of the triple-buffered pipeline to `docs/multithreading.md`; add a catalog-chain mutation diagram to `docs/format.md`; add a cloud-pull topology diagram to `docs/cloud.md`.

**Patch 16 — P3 cleanup sweep (#51–#61)**
- Move `MajorAxis` to `pub(crate)` in `scx-format/src/writer.rs`.
- Remove `#[deprecated]` `ScxReader::read_shard_from_entry_unchecked` (or schedule for next major).
- Remove dead `SectionType::is_known` in `scx-format/src/section.rs`.
- Add a `log::warn!` at `scx-format/src/catalog.rs:532-534` when an unknown section type is skipped, matching spec.
- Drop the unused `var_predicates`, `obs_predicate_index`, `var_predicate_index` fields in `scx-engine/src/collect.rs:38-41` (or wire them up).
- Replace `panic!()` in `scx-engine/src/fused_ops.rs:408` with `EngineError::SchemaError`.
- Update the stale `format_version` field comment at `scx-format/src/header.rs:32` ("currently 1" → "currently 2").
- Fix `docs/format.md:581` "≥13" → "≥17" (or remove the boundary now that 13/14 are defined).
- Fix `AGENTS.md:51` "15 section types" → "17".
- Stamp `ROADMAP.md:3` to today's date or remove.
- Continue moving pyscx bindings into domain-specific Rust modules with local `register_*` functions.

## 19. Quick Wins (one-line fixes worth flagging)

- Move `MajorAxis` to `pub(crate)` in `scx-format/src/writer.rs`.
- Remove `#[deprecated]` `ScxReader::read_shard_from_entry_unchecked` (or schedule for next major).
- Remove dead `SectionType::is_known` in `scx-format/src/section.rs`.
- Add a `log::warn!` at `scx-format/src/catalog.rs:532-534` when an unknown section type is skipped.
- Drop the unused `var_predicates`, `obs_predicate_index`, `var_predicate_index` fields in `scx-engine/src/collect.rs:38-41` (or wire them up).
- Replace `panic!()` in `scx-engine/src/fused_ops.rs:408` with `EngineError::SchemaError`.
- Update the stale `format_version` comment at `scx-format/src/header.rs:32`.
- Fix `docs/format.md:581` "≥13" → "≥17" (or remove).
- Fix `AGENTS.md:51` "15 section types" → "17".
- Stamp `ROADMAP.md:3` to today's date or remove.
- Replace `vec![0u8; pad]` padding writes (`pad ≤ 7`) with a shared stack-buffer helper.

## 20. Testing & Verification Gaps

### 20.1 Rust unit / integration

- `append_rejects_zero_shard_target_rows`.
- `append_honors_explicit_codec` for every supported codec id.
- `append_auto_codec_selects_per_shard`.
- `append_rejects_target_n_vars_over_u32_max`.
- `header_rejects_bad_header_length` / `nonzero_reserved_padding` / `nonzero_reserved_bytes` / `reserved_flag_bits`.
- `writer_temp_paths_do_not_collide_in_same_process`.
- `preencoded_csc_updates_header_count` (or `preencoded_csc_rejected`).
- Engine predicate conformance suite shared between `scx query`, pyscx `query`, and `pull_filtered`.

### 20.2 Python

- `pyscx.append(..., shard_size=0)` raises `ValueError`.
- `pyscx.append(..., codec="none")` produces uncompressed appended shards.
- `pyscx.append(..., codec="zstd")` produces Zstd-compressed appended shards.
- Shape/schema/index validation errors are `ValueError`, not `RuntimeError`.
- `pyscx.validate()` catches a deliberately corrupted section payload.
- `pyscx.open(..., verify=True)` is documented and tested as catalog-level only.
- `obsm`/`obsp`/`varm`/`varp` round-trip.
- `uns` array round-trip (currently lossy by design).
- `ScxBackedSparseDataset` outliving its parent `PyExperiment`.
- `test_zero_copy.py` tightened to assert `arr.flags.OWNDATA` and reject downcast.

### 20.3 CLI

- `scx append --shard-size 0` fails fast with a clear message.
- `scx append --codec zstd` results in appended shard headers using Zstd.
- `scx convert --from h5ad input.scx output.scx` honors explicit `--from` regardless of extension.
- `scx pull --filter ...` reports shard-mode superset semantics or exact-mode behavior.
- Mutation commands print CSC invalidation/rebuild summaries.

### 20.4 Cloud

- Fake object-store test proving `pull()` does not buffer the full file after the streaming fix.
- Interrupted pull leaves a temp file; later pull cleans only stale orphans (current behavior); concurrent pulls to the same destination do not collide (post-tempfile fix).
- Selective pull exact-vs-shard semantics tested explicitly.
- Omitted-sections summary printed and asserted.

### 20.5 Benchmark / release

- Script regenerates README-visible benchmark summary from checked-in manifests.
- CI check rejects README benchmark numbers without a manifest reference.

## 21. Closing notes

The code quality bar across SCX is high — the codec spec is tight, the writer is careful about atomic semantics, the format has clear extension points, and the team has been disciplined about extracting hard-won concurrency lessons into a doc (`docs/multithreading.md`). The training loader and the `LoaderConfig::validate()` discipline are an aspirational target the rest of the entry points should imitate.

Most findings above are not architectural; they are the wear-and-tear of a project that has iterated through several phases and accumulated drift between docs, code, and tests. The most urgent work is not a rewrite. It is contract alignment: make public options do what they say, reject invalid numeric inputs, document shard-level versus exact semantics, and make streaming claims true or narrower. Fixing the append codec bug, the zero-shard-size hang, the convert match-arm bug, the Float16 mislabeling, the cloud-pull buffering, and the selective-pull semantics would remove the highest-risk surprises for users.

The single highest-leverage P0 in terms of user impact is the pyscx `obsp/varp/varm` gap, because it is reachable from every scanpy workflow and silently destroys data. The append zero-shard hang and `_codec_id`-ignored bug are the two most likely to corrupt or stall user workflows. The unbounded `u32`-length allocations are the easiest defensive-coding wins.

The documentation set is already excellent in coverage. The most useful next pass on docs is not "write more", it is "reconcile the existing claims, pick one number for each metric, and pin every claim to a manifest or a regression test".
