# SCX Binary Format Reference

This is the authoritative reference for the on-disk layout of `.scx` files
(and the equivalent exploded `.scxd/` directories). Bit-level codec details
live in [docs/codec.md](codec.md). Sharding rationale and shard sizing live
in [docs/sharding.md](sharding.md).

**Normative reference**: when this document and the Rust implementation in
`scx-format` / `scx-codec` disagree, the Rust code wins.

**Byte ordering**: all multi-byte integers are **little-endian**. The header
`endian` field exists for validation — readers MUST reject files with
`endian != 0`.

**Alignment**: every section starts at an 8-byte-aligned offset; padding bytes
(zeroed) are inserted between sections as needed.

## 1. Physical Layout

```
experiment.scx (single binary file)
┌──────────────────────────────────────────────────────────────────┐
│ FILE HEADER (256 bytes, offset 0)                                │
├──────────────────────────────────────────────────────────────────┤
│ ROOT CATALOG (offset 256, ≤4096 bytes)                           │
├──────────────────────────────────────────────────────────────────┤
│ FRONT CATALOG (optional, offset ~4352 if has_front_catalog)      │
├──────────────────────────────────────────────────────────────────┤
│ PADDING (to 8-byte alignment)                                    │
├──────────────────────────────────────────────────────────────────┤
│ obs metadata          (Arrow IPC file format)                    │
│ obs predicate indexes                                            │
│ var metadata          (Arrow IPC file format)                    │
│ var predicate indexes                                            │
├──────────────────────────────────────────────────────────────────┤
│ X/csr/000000 … X/csr/{N-1}    (CSR shards)                       │
│ X/csc/…                       (optional, if has_csc)             │
│ X/bitmap/…                    (optional, if has_bitmap)          │
│ layers/{name}/csr/…           (additional CSR shard sets)        │
│ obsm/{name}                   (Arrow IPC dense 2D tensors)       │
│ obsp/{name}/csr/…             (sparse cell-cell graphs)          │
│ uns/metadata.json                                                │
│ provenance                    (structured, §7)                   │
│ deletion_vectors              (optional, §6.3)                   │
├──────────────────────────────────────────────────────────────────┤
│ FULL CATALOG (manifest v0)   (always at EOF of initial file)     │
├──────────────────────────────────────────────────────────────────┤
│ … appended sections + new catalogs (§6.2) …                      │
└──────────────────────────────────────────────────────────────────┘
```

## 2. File Header (256 bytes)

Written LE, at offset 0. Sections up to `reserved` total 124 bytes; `reserved`
pads the rest out to 256.

| Field | Type | Notes |
|-------|------|-------|
| `magic` | `[u8; 4]` | `b"SCX\x01"` |
| `format_version` | `u16` | Current: 1 |
| `header_length` | `u16` | 256; reserves space for future header growth |
| `flags` | `u32` | See flag table below |
| `n_obs` | `u64` | Total cells (after deletions) |
| `n_vars` | `u64` | Total genes |
| `nnz` | `u64` | Total non-zeros |
| `n_csr_shards` | `u32` | |
| `n_csc_shards` | `u32` | 0 if CSC not present |
| `shard_target_rows` | `u32` | Cells per CSR shard; default 10,000 |
| `codec_id` | `u8` | Default codec (§codec.md). Per-shard override allowed. |
| `index_dtype` | `u8` | 0 = u16 indices, 1 = u32. Set once per file. |
| `endian` | `u8` | 0 = little (required), 1 = big (rejected) |
| `reserved_padding` | `[u8; 1]` | |
| `root_catalog_offset` | `u64` | |
| `root_catalog_length` | `u64` | |
| `full_catalog_offset` | `u64` | Offset of **active** full catalog |
| `full_catalog_length` | `u64` | |
| `manifest_sequence` | `u64` | Monotonic; 0 for initial file |
| `prev_catalog_offset` | `u64` | 0 if first version |
| `file_checksum` | `u64` | BLAKE3 truncated to 64 bits |
| `front_catalog_offset` | `u64` | 0 if `has_front_catalog` unset |
| `front_catalog_length` | `u64` | 0 if not present |
| `reserved` | `[u8; 132]` | Zeroed |

### Flags

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `has_csc` | CSC shards present |
| 1 | `has_bitmap` | Detection bitmap present |
| 2 | `has_obsm` | Cell embeddings present |
| 3 | `has_obsp` | Cell-cell graphs present |
| 4 | `has_modalities` | Multimodal — section paths include `mod/{name}/…` |
| 5 | `has_deletion_vectors` | Deletion vectors section present |
| 6 | `has_front_catalog` | Cloud-ready layout — front catalog duplicate valid |

### Index dtype

`index_dtype=0` (u16) is used when `n_vars <= 65535`, halving index storage.
Files with more features use `index_dtype=1` (u32). All shards in a file
share the same dtype.

## 3. Dual Catalog

The file has **two** catalog structures so that a single contiguous read can
open any `.scx`, while still supporting random access and rollback.

### Root catalog (at offset 256, ≤ 4096 bytes)

A compact summary sufficient for streaming / training readers:

```
n_section_groups: u16
For each group:
  group_type: u8                 (obs_meta, var_meta, csr_shards, csc_shards,
                                   bitmap, layer, obsm, obsp, uns, provenance)
  first_section_offset: u64
  total_group_length: u64
  n_sections: u32
  summary: [u8; 32]              (group-specific — for csr_shards: nnz, value stats)
```

Local opens read `header + root_catalog` (< 4 KB) and can immediately locate
every section group without parsing the full catalog. Cloud range reads do the
same in one GET of the first 4 KB.

### Full catalog (at `full_catalog_offset` — usually EOF)

Complete per-section index. Required for random-access operations and for
rollback (a new catalog is appended per mutation; older catalogs remain in
the file until `scx compact`).

```
catalog_version: u16             (1)
manifest_sequence: u64           (matches header)
prev_catalog_offset: u64         (0 if first)
n_obs: u64                       (observable cells after deletions)
n_entries: u32
For each entry:
  name_length: u16
  name_bytes: [u8; name_length]  (UTF-8 path, e.g. "X/csr/000042")
  offset: u64
  length: u64
  section_type: u8               (enum below)
  checksum: [u8; 32]             (full BLAKE3 of section content)
  stats_length: u16
  stats: [u8; stats_length]      (per-section stats, format below)
catalog_checksum: [u8; 32]       (BLAKE3 of all preceding catalog bytes)
```

### `section_type` enum

| ID | Type |
|----|------|
| 0 | `obs_metadata` |
| 1 | `obs_index` |
| 2 | `var_metadata` |
| 3 | `var_index` |
| 4 | `csr_shard` |
| 5 | `csc_shard` |
| 6 | `bitmap_shard` |
| 7 | `layer_csr_shard` |
| 8 | `obsm_embedding` |
| 9 | `obsp_csr_shard` |
| 10 | `uns_blob` |
| 11 | `provenance` |
| 12 | `deletion_vectors` |
| 13–239 | Reserved for extensions (multimodal, spatial) |
| 240–254 | Reserved for encrypted section types |
| 255 | Sentinel |

Unknown types (≥13 for v1) are skipped by readers with a warning, which
allows the format to evolve without breaking old readers.

### Per-shard statistics

Present when `section_type ∈ {csr_shard, csc_shard, layer_csr_shard, obsp_csr_shard}`:

```
row_start: u64                   (first row, global)
row_end:   u64                   (exclusive end, global)
nnz:       u64
value_min: u32                   (integer encodings only)
value_max: u32
value_sum: u64
n_indexed_columns: u8
For each indexed column:
  column_name_hash: u64          (BLAKE3 truncated)
  stat_type: u8                  (0 = numeric min/max, 1 = category bitset)
  If numeric:
    col_min: f64
    col_max: f64
  If categorical:
    bitset_length: u16
    category_bitset: [u8]        (bit i set if dictionary index i present)
```

`value_min` / `value_max` / `value_sum` are meaningful **only** for integer
value encodings (uint8/16/32). For float encodings (f32, f16) they are zero
and MUST NOT be used for pushdown filtering.

These statistics power catalog-level predicate pushdown — the query engine
can skip entire shards without reading any section data. Selective `pull`
(see [docs/cloud.md]) uses the same mechanism.

## 4. CSR Shard Internal Layout

Each CSR shard is a **self-contained**, contiguous section. It can be read,
validated, and decoded in isolation — which is what enables parallel shard
decode, exploded `.scxd` layouts, and selective pull.

```
┌──────────────────────────────────────────────────────────────┐
│ SHARD HEADER (76 bytes)                                      │
├──────────────────────────────────────────────────────────────┤
│ INDPTR SECTION     (encoded per shard's codec_id)            │
│ INDICES SECTION                                              │
│ VALUES SECTION                                               │
│ BLOCK INDEX        (for O(1) random row access)              │
└──────────────────────────────────────────────────────────────┘
```

### Shard header (76 bytes)

| Field | Type | Notes |
|-------|------|-------|
| `magic` | `[u8; 4]` | `b"SCXS"` |
| `shard_format_version` | `u8` | 1 |
| `shard_type` | `u8` | 0 = CSR, 1 = CSC |
| `codec_id` | `u8` | May override file header |
| `value_encoding` | `u8` | May override file header |
| `index_dtype` | `u8` | 0 = u16, 1 = u32 |
| `reserved_flags` | `[u8; 3]` | |
| `n_major` | `u32` | Rows in this shard (CSR) |
| `n_minor` | `u32` | Columns in full matrix |
| `nnz` | `u64` | |
| `global_offset` | `u64` | First row index in full matrix |
| `indptr_rel_offset` | `u32` | Relative to shard start |
| `indptr_length` | `u32` | |
| `indices_rel_offset` | `u32` | |
| `indices_length` | `u32` | |
| `values_rel_offset` | `u32` | |
| `values_length` | `u32` | |
| `block_index_rel_offset` | `u32` | |
| `block_index_length` | `u32` | |
| `checksum` | `[u8; 8]` | BLAKE3 truncated to 64 bits, covers everything after the header |

Total: 76 bytes. Implementations MUST use exactly 76 — this matches
`SHARD_HEADER_SIZE` in `scx-format/src/shard.rs`.

> **Per-shard codec override**: readers MUST use the shard header's
> `codec_id` and `value_encoding`, not the file header's. The file header
> values are a hint for tools that want summary stats without touching
> individual shards. This is how a file can mix, e.g., Scx1 raw integer
> shards in `X` with Zstd-compressed float shards in a normalized layer.

### Block index

Enables O(1) random access to any row range inside a shard:

```
n_blocks: u32
For each block:
  row_start: u32                 (first row in block, global index)
  n_rows: u16
  indptr_byte_offset: u32        (within indptr section)
  indices_byte_offset: u32
  values_byte_offset: u32
  nnz_in_block: u32
```

`row_start: u32` here is correct because it addresses rows within a shard
(max `shard_target_rows`, typically 10,000). The *full* catalog's shard
statistics use `u64` because they address global rows (billions of cells).

### Shard sizing defaults

Default 10,000 cells per CSR shard. At 5% density × 30K genes → ~15M non-zeros
per shard → ~30–60 MB compressed. Rationale and tradeoffs are in
[docs/sharding.md](sharding.md).

## 4.1 CSC Shard Internal Layout

CSC sidecar shards are an **optional column-major view** of the same
data the CSR shards hold. They are emitted by `scx convert
--csc=always`, `pyscx.from_anndata(csc="always")`, `scx build-csc`,
and the `--rebuild-csc` flag on the mutating ops (`append`, `compact`,
`merge`, `subset`).

A CSC shard has the **same on-disk structure** as a CSR shard — same
76-byte shard header, same encoded `indptr` / `indices` / `values`
sections, same block index. The column-major semantics live in the
field interpretation:

| Field | CSR semantics | CSC semantics |
|-------|---------------|---------------|
| `shard_type` | `0` (legacy: also produced for CSC) | `1` (authoritative) — readers also accept `0` when the catalog `section_type` is `CscShard` (legacy compatibility) |
| `n_major` | rows in this shard | **columns** in this shard |
| `n_minor` | columns in full matrix | **rows** in full matrix (`n_obs`) |
| `global_offset` | first row index | first **column** index covered by this shard |
| `indptr` (length `n_major + 1`) | row-pointer | **column-pointer** |
| `indices` (length `nnz`) | column indices in full matrix | **global row** indices in full matrix (CSC indices are NOT shard-local — they reference rows across all shards) |

Catalog `section_type = CscShard (5)` is the authoritative
discriminator; the in-shard `shard_type` byte exists for self-contained
shard validation (e.g. exploded `.scxd` files where the catalog and
shard live in separate files). Going forward writers emit `shard_type
= 1`; readers also accept `shard_type = 0` for files written before
the Phase A `derive_shard_type()` fix landed.

### `ShardStats.row_start` / `row_end` axis overload

CSC shards reuse the catalog `ShardStats.row_start` / `row_end` fields
to record the **major-axis** range, which for CSC means
`col_start..col_end`. The on-disk schema is unchanged from the CSR-only
era; only the field interpretation differs.

Code accessing these fields should pick by section type:

| Method | When to use |
|--------|-------------|
| `ShardStats::major_start()` / `major_end()` | Generic — works for any shard type, returns the row range for CSR/Layer/Obsp and the col range for CSC. Use when the section type is unknown or when writing axis-agnostic helpers. |
| `ShardStats::row_range()` | When the entry is known to be a row-major shard (`CsrShard`, `LayerCsrShard`, `ObspCsrShard`). |
| `ShardStats::col_range()` | When the entry is known to be `CscShard`. |
| Direct `ShardStats.row_start` field access | Allowed in legacy code paths but discouraged — prefer the accessors above. |

A future format-version bump may rename the underlying fields (e.g. to
`major_start` / `major_end`) without breaking on-disk compatibility,
since the byte layout is unchanged. Until then the overload is
documented but the schema stays put.

### Multi-shard CSC layout

A CSC sidecar may be split into multiple shards by column range
(controlled by `--csc-cols-per-shard`, default 5000). Each shard
covers a contiguous half-open `[col_start, col_end)` range; ranges are
non-overlapping and sorted by `col_start`. The catalog's
`csc_shards_sorted()` accessor returns the shards in column order.

Column-range pushdown: `BackedCscReader::read_csc_columns(c_lo..c_hi)`
uses `BackedCscIndex::shards_for_col_range(c_lo, c_hi)` to skip
non-overlapping shards entirely; partial-overlap shards are sliced
post-decode. See `docs/sharding.md` § CSC sharding.

### `scx upgrade` preserves CSC

`scx upgrade` (Phase B) re-emits the file through the current writer
while preserving CSC sidecars: the CSR rewrite loop calls
`writer.write_csr_shard()`, then the CSC entries are walked via
`catalog.csc_shards_sorted()` and re-emitted via
`writer.write_csc_shard()` on the new file. The post-upgrade file has
identical CSC content (byte-equal under the same codec) and a
correctly populated `n_csc_shards` count + `has_csc` flag.

## 5. Arrow IPC Metadata

`obs` and `var` metadata are stored as **Arrow IPC file format** (not the
streaming variant) — the file includes a footer with schema and record batch
offsets, so readers can seek to individual columns without reading the whole
section.

- **Nullable columns**: standard Arrow validity bitmaps.
- **Categorical columns**: Arrow dictionary encoding. Predicate indexes (§6)
  are built on the dictionary values, not encoded indices.
- **Chunking**: for datasets > 10M cells, `obs` is split into multiple record
  batches (one per CSR shard group) inside a single IPC file. Readers can
  lazy-load metadata for just the shards they touch.
- **Zero-copy**: Arrow IPC is memory-mappable. Python exposes Arrow arrays via
  PyArrow — zero-copy for primitive and dictionary types; string columns
  allocate Python heap strings (use categoricals to avoid this).

## 6. Predicate Indexes

Sorted mappings from column values to shard-local row ranges, attached to
`obs_metadata` and `var_metadata`.

```
PREDICATE INDEX SECTION:
  index_version: u8              (1)
  n_indexed_columns: u8
  For each column:
    column_name_length: u16
    column_name: [u8]            (UTF-8)
    column_type: u8              (0 = categorical, 1 = numeric)
    n_entries: u32

    If categorical:
      For each unique value (lexicographic):
        value_length: u16
        value_bytes: [u8]        (UTF-8 category label)
        n_shard_ranges: u16
        For each range:
          shard_id: u32
          row_start: u32         (shard-local)
          row_end: u32

    If numeric:
      fanout: u16                (B+ tree branching factor, typically 64)
      n_leaf_pages: u32
      n_internal_pages: u32
      For each internal page:
        n_keys: u16
        keys: [f64; n_keys]
        children: [u32; n_keys + 1]
      For each leaf page:
        n_entries: u16
        For each entry:
          min_value: f64
          max_value: f64
          shard_id: u32
          row_start: u32
          row_end: u32
```

**High-cardinality columns** (> 10,000 unique values, e.g. `donor_id`): the
categorical index becomes a minimal perfect hash function pointing to offset
arrays, capping size at ~100 KB regardless of cardinality.

**Unindexed columns**: writers only index columns marked `indexed=True` or
those with < 1000 unique values. The reader falls back to sequential scan for
unindexed columns.

## 7. Fragment / Manifest Model

SCX uses a **fragment/manifest architecture** inspired by Lance and Delta Lake.
Shards and other sections are **immutable fragments** — once written, their
bytes never change. The full catalog is a **manifest** that names the
currently-active fragments. This is what makes appends, deletions, and
rollback safe in a single file.

### 7.1 Initial file creation (atomic rename)

1. Writer creates a temp file (`experiment.scx.tmp.{pid}`).
2. Sections are written sequentially starting at offset 4352 (256 header + 4096
   root catalog placeholder).
3. Full catalog (manifest 0) is appended.
4. Root catalog is written at offset 256 via `pwrite()` (offsets are now known).
5. Header is written at offset 0 via `pwrite()` — `manifest_sequence=0`,
   `prev_catalog_offset=0`.
6. `fsync()` the temp file.
7. `rename()` to final path (atomic on POSIX).

If the process is killed before step 7, the temp file is an orphan that can be
deleted. The final path either exists complete or doesn't exist at all.

### 7.2 Append

Adding cells, layers, or embeddings does NOT rewrite existing bytes:

1. Open for append (`O_WRONLY | O_APPEND`).
2. Acquire advisory `flock()` (prevents concurrent appends).
3. Seek to EOF.
4. Write new sections sequentially (new CSR shards, updated `obs_metadata`, …).
5. Write a new full catalog that references **both** original and new sections.
   Set `prev_catalog_offset` to the old catalog offset.
6. `pwrite()` the root catalog at offset 256.
7. `pwrite()` the header at offset 0: bump `manifest_sequence`, update
   `full_catalog_offset/length`, update `n_obs` / `nnz` / `n_csr_shards`.
8. `fsync()`.
9. Release the lock.

**Commit point**: step 7 — the single header `pwrite()`. Until then, readers
see the previous manifest. A crash between steps 4 and 6 leaves trailing
garbage past the last valid catalog, which is harmless: the reader locates the
active catalog via the header, never by scanning.

**Metadata invariant**: when new cells are appended, a fresh `obs_metadata`
section covering *all* cells is written. The new catalog points to it; the
old `obs_metadata` section remains in the file but is orphaned until `scx
compact` reclaims its space. `var_metadata` is unchanged (append is cell-axis
only).

### 7.3 Deletion vectors (logical delete)

Filters that want to mark cells as deleted without rewriting shards append a
deletion vectors section + a new catalog:

```
DELETION VECTORS SECTION:
  dv_version: u8                 (1)
  n_shards_with_deletions: u32
  For each shard with deletions:
    shard_id: u32                (index into the catalog's CSR shard list)
    deletion_bitmap_length: u32
    deletion_bitmap: [u8]        (Roaring Bitmap of deleted row indices,
                                   local to the shard)
```

When `has_deletion_vectors` is set, readers apply the bitmap during shard
decode — deleted rows are skipped. `scx rollback` undoes a logical delete
because the shard data was never modified.

### 7.4 Compaction

After many appends and deletions, `scx compact`:

- Drops orphaned sections (referenced only by old catalogs)
- Physically removes rows marked by deletion vectors
- Merges small shards into full-sized ones (target: `shard_target_rows`)
- Writes a clean single-manifest file via the standard atomic rename path.
  The output has `manifest_sequence=0` and no previous catalogs.

### 7.5 Rollback

Previous catalogs are preserved, so `scx rollback` is a single header
`pwrite()` that points `full_catalog_offset` and `manifest_sequence` back to
a prior version. `scx rollback --to-seq N` rolls back to a specific sequence.
No data is deleted. To permanently discard old versions, follow up with
`scx compact`.

## 8. Provenance

```
PROVENANCE SECTION:
  provenance_version: u8         (1)
  n_operations: u32
  For each operation:
    timestamp: i64               (Unix epoch seconds)
    action_length: u16
    action: [u8]                 (UTF-8: "create", "convert", "merge", "append",
                                    "delete", "compact", "rollback",
                                    "add_layer", "add_csc", "add_obsm", …)
    tool_length: u16
    tool: [u8]                   (UTF-8: "scx-cli 0.1.0", "pyscx 0.1.0", …)
    params_json_length: u32
    params_json: [u8]            (UTF-8 JSON, tool-specific)
    input_checksums_count: u8
    For each input:
      checksum: [u8; 32]         (BLAKE3 of input file)
```

`scx merge` preserves provenance chains from every input, then appends its
own operation entry, giving end-to-end lineage.

## 9. Checksums

All checksums use **BLAKE3** (>5 GB/s on modern CPUs, cryptographically
secure, faster than xxHash on long inputs).

| Scope | Hash | Where |
|-------|------|-------|
| Per-section | 32-byte BLAKE3 | full catalog entry |
| Per-shard | 64-bit truncated BLAKE3 | shard header; covers everything after the header |
| Catalog | 32-byte BLAKE3 | last 32 bytes of full catalog; covers all preceding catalog bytes |
| File | 64-bit truncated BLAKE3 | file header `file_checksum` |

On checksum failure, the reader names the specific corrupted section. Non-
essential sections (layers, embeddings, obsp graphs) may be skipped with a
warning; corruption in `obs_metadata`, `var_metadata`, or a CSR shard in `X`
is a hard error.

**Encryption**: not in v1. Section types 240–254 are reserved for future
encrypted section types. For PHI datasets, use filesystem-level encryption
(LUKS, dm-crypt, GCS CMEK).

## 10. Versioning & Compatibility

- **`format_version`** — bump for breaking changes. Readers MUST reject files
  with `format_version` higher than their supported maximum.
- **Unknown section types** (≥13 for v1) are skipped with a warning, enabling
  incremental extension without breaking old readers.
- **`header_length`** reserves space for future header growth — older readers
  that only handle 256-byte headers detect a larger `header_length` and exit
  gracefully.
- **Migration**: `scx upgrade input.scx output.scx` rewrites a file to the
  latest format version.

## 11. Concurrent Read Safety

Because sections are immutable fragments, **concurrent reads are always safe**
regardless of concurrent appends:

- New sections are written past the range any current reader is touching.
- The catalog switch is a single 256-byte header `pwrite()` — atomic on POSIX.
- A reader that opened before the switch sees the old manifest; a reader that
  opens after sees the new one. There is no partially-updated state.

**Advisory locking** (`flock()`) prevents concurrent *appends* (two writers).
It is not needed for readers. See [docs/multithreading.md §Concurrent file
access](multithreading.md#concurrent-file-access) for the full picture,
including graceful `flock()` fallback on NFS/Lustre/GPFS.

**mmap on network filesystems**:

- GPFS: mmap works; readahead may be suboptimal for shard-level random access.
  Explicit `madvise(MADV_SEQUENTIAL)` per shard group helps.
- NFS: known mmap cache-coherency issues if the file is appended while mapped.
  SCX is safe in practice because sections are immutable and a reader only
  maps sections referenced by the catalog it read at open time.
- Readers MUST support both mmap and `pread()` paths, selected at open time
  from filesystem type or a user hint. `pread()` is the safe fallback.

## 12. Detection Bitmap (Optional)

Bit-packed presence/absence stored as Roaring Bitmap sections (`bitmap_shard`,
type 6). Written when `has_bitmap` is set.

- **Storage**: 30K × 500K at 5% density → ~100–200 MB compressed.
- **Use cases**:
  - Gene detection rate — POPCOUNT columns, ~50× faster than CSC scan.
  - Jaccard cell similarity — AND + POPCOUNT with SIMD.
  - Fast approximate filtering ("which cells express gene X?").

For publication-quality analyses (DE, trajectory inference, regression),
count-based methods on the full CSR data are required. The bitmap does not
replace counts.

## 13. Multimodal Extension (Optional)

Set `has_modalities` (bit 4) when the file contains multiple feature spaces
(CITE-seq RNA + protein, etc.). Section paths gain a modality prefix:

```
experiment.scx
├── obs metadata                   (shared across modalities)
├── mod/rna/var metadata
├── mod/rna/X/csr/…                (RNA counts)
├── mod/protein/var metadata
├── mod/protein/X/csr/…            (ADT counts)
└── catalog
```

`obs` is shared (all modalities measure the same cells). Each modality has
its own `var`, `X`, layers, and embeddings. This round-trips with MuData/MuOn.

Spatial data reuses standard numeric `obs` columns (`x_spatial`, `y_spatial`,
`z_spatial`); images may live in `uns` blobs (`uns/spatial/images/hires`, …)
following SpatialData conventions. An R-tree spatial index is reserved as a
future section type.

## 14. Cloud Layouts

SCX supports three deployment shapes:

| Layout | Cloud-friendly? | Notes |
|--------|-----------------|-------|
| Plain packed `.scx` | Poor — needs HEAD + EOF range read | Catalog at EOF only |
| Cloud-ready packed `.scx` | Good — single initial range read | `has_front_catalog` set; front catalog duplicates the full catalog near the header |
| Exploded `.scxd/` | Best — one object per section, atomic publish via `_catalog.bin` last | Shard bytes are **identical** to packed form |

See [docs/cloud.md](cloud.md) for auth, provider notes, tuning, and end-to-end
examples. The packed and exploded forms are **semantically equivalent**:
`scx pack` and `scx explode` round-trip byte-identically.

## 15. Conformance

Implementations other than the reference `scx-format`/`scx-codec` crates are
welcome but must pass:

- **Codec test vectors** — a set of reference input arrays and their exact
  encoded bytes for Rice, FOR-BP, and Delta-Golomb. Conforming encoders MUST
  produce byte-identical output; conforming decoders MUST decode the reference
  bytes to identical arrays.
- **Round-trip**: `h5ad → scx → h5ad` produces numerically identical
  expression matrices (bit-exact for integer counts, within epsilon for float
  layers) and identical metadata modulo pandas dtype normalization.
- **Fuzz**: the reader must not crash, panic, or exhibit UB on any input. The
  Rust implementation includes fuzz targets for the shard decoder, catalog
  parser, and Arrow IPC reader; CI runs them continuously.

The Rust implementation is the normative reference — if this document and
`scx-format`/`scx-codec` disagree on encoding/decoding behavior, the Rust
code wins.
