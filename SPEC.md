# SCX: Sparse Cell eXpression System

## A Purpose-Built Format + Compute Engine for Single-Cell RNA-seq

**Version 0.5 — Specification (March 2026)**

**Changelog from v0.4**: Fragment/manifest append model replaces full-rewrite
`--reserve-space` approach. Shards are now immutable fragments; appending writes new
shards + new catalog at EOF and atomically updates the header pointer. Deletion
vectors enable logical cell filtering without rewriting data. `scx compact` added as
a first-class operation. Comparison table updated for append support.

**Changelog from v0.3**: Addresses technical review. Major changes: honest cloud
scoping (HPC/local-first), dual catalog for cloud compatibility, CSR-only default,
bit-level codec specification, explicit in-memory data model, atomic write safety,
multimodal extensibility, restructured MVP roadmap (training loader first), expanded
predicate index format, conformance test requirements, and numerous detail fixes.

---

## 1. Overview

SCX is a co-designed **file format**, **compression codec**, **query engine**, and
**ML data loader** for single-cell RNA-seq data. It replaces AnnData/h5ad,
scipy.sparse, and the scanpy I/O layer with a unified Rust-native stack.

**The SCX file (`.scx`) is a single packed binary file.** One file contains the
expression matrix, all metadata, embeddings, graphs, and an internal catalog for
O(1) random access to any component.

### 1.1 Scope and Non-Goals

**SCX is an HPC-first and local-first format.** It is optimized for:
- SLURM HPC clusters with parallel filesystems (GPFS, Lustre)
- Local NVMe for GPU training (especially GDS)
- Workstation analysis with scanpy/Seurat replacements
- Sharing via `cp`/`scp`/`rsync`/`rclone` of a single file

**SCX is not optimized for cloud-native random access.** Object stores (S3, GCS)
are best served by one-object-per-chunk formats like Zarr. SCX files can be stored
on S3 and accessed via range reads, but the access pattern (many small range reads
per query) will incur higher latency than a Zarr store. For cloud-hosted atlases
where low-latency random access is the primary requirement, TileDB-SOMA or Zarr
remain more appropriate. SCX targets the workflow where the file is staged locally
(to scratch, to NVMe) before intensive computation or training.

### 1.2 Design Principles

1. **Single file** — One `.scx` file per experiment. Easy to manage, copy, and stage.
2. **Sparse-native** — CSR stored as a first-class structure with domain-specific codec.
3. **CSR-only by default** — CSC is an opt-in addition for gene-major workflows.
4. **Custom codec** — Entropy-optimal compression exploiting UMI count distributions.
5. **Query engine** — Polars-style lazy evaluation with predicate pushdown and operation fusion.
6. **GPU-direct** — Shard layout matches cuSPARSE CSR. Supports GDS range reads.
7. **ML data loader** — Triple-buffered Rust pipeline that saturates GPUs during training.
8. **Cross-language Rust core** — Python, R bindings. No GIL on the hot path.
9. **Multimodal-extensible** — Section-based layout supports RNA+protein, RNA+ATAC,
   and spatial modalities without breaking format changes.
10. **Append-friendly** — Immutable shards with fragment/manifest catalog model.
    Append, delete, and compact without full-file rewrites.

---

## 2. Empirical Properties of scRNA-seq Data

Every design decision traces to one of these measured properties.

### 2.1 Value Distribution (non-zero UMI counts)

| Value | Frequency |
|-------|-----------|
| 1     | ~55-65%   |
| 2     | ~15-20%   |
| 3     | ~7-10%    |
| 4     | ~3-5%     |
| 5-15  | ~5-10%    |
| 16+   | <1%       |

Shannon entropy: ~2.0 bits/value. Generic Zstd: ~4.5 bits. SCX Rice codec: ~2.2 bits.

### 2.2 Sparsity

- 90-98% zeros (5% non-zero typical for 10x Chromium)
- Sparse storage uses ~10x less memory than dense
- Binarized count data preserves substantial clustering signal for most cell-type
  distinctions, though resolution degrades for closely related subtypes and
  shallow-sequenced datasets (Breda et al. 2023, Genome Biology; Jiang et al.
  2022, Genome Biology). The detection bitmap (§5) is useful for fast approximate
  analyses but should not replace count-based methods for fine-grained comparisons.

### 2.3 Dimensions

- Genes (columns): 20K-60K (fixed per genome build)
- Cells (rows): 1K-100M+ (the scaling axis)
- Row-major (CSR) is the natural iteration and append direction

### 2.4 Access Patterns

| Operation | Access | Runtime share |
|-----------|--------|--------------|
| Cell QC | Row scan (CSR) | 5-10% |
| HVG selection | Column aggregation (needs CSC or streaming transpose) | 5-10% |
| Normalization | Row read/write (CSR) | 10-15% |
| PCA (randomized SVD) | 2-5 full passes (CSR) | 20-40% |
| kNN graph | Dense embedding rows | 15-25% |
| Leiden clustering | Sparse graph traversal | 5-15% |
| DE analysis | Column slice by group (needs CSC or transpose) | 5-15% |
| ML training batch | Random row subset → dense (CSR) | continuous |

CSR-dominant operations (QC, normalization, PCA, training) account for 60-80% of
runtime. CSC-dominant operations (HVG, DE) account for 10-25%. This motivates
CSR-only as the default, with CSC as opt-in.

---

## 3. File Format

### 3.1 Physical Layout

```
experiment.scx (single binary file)
┌──────────────────────────────────────────────────────────────────┐
│ FILE HEADER (256 bytes, offset 0, fixed)                         │
│   magic: [u8; 4] = b"SCX\x01"                                   │
│   format_version: u16        (current: 1)                        │
│   header_length: u16         (256; allows future header growth)  │
│   flags: u32                                                     │
│     bit 0: has_csc                                               │
│     bit 1: has_bitmap                                            │
│     bit 2: has_obsm                                              │
│     bit 3: has_obsp                                              │
│     bit 4: has_modalities  (multimodal extension, §11)           │
│     bit 5: has_deletion_vectors (§3.6.3)                         │
│   n_obs: u64                                                     │
│   n_vars: u64                                                    │
│   nnz: u64                                                       │
│   n_csr_shards: u32                                              │
│   n_csc_shards: u32          (0 if CSC not present)              │
│   shard_target_rows: u32     (cells per CSR shard, default 10000)│
│   codec_id: u8               (default codec for shards, see §4.5)│
│   index_dtype: u8            (0=u16, 1=u32; see §3.3)           │
│   endian: u8                 (0=little, 1=big; little required)  │
│   reserved_padding: [u8; 1]                                      │
│   root_catalog_offset: u64   (offset of root catalog, see §3.2) │
│   root_catalog_length: u64                                       │
│   full_catalog_offset: u64   (offset of active full catalog)     │
│   full_catalog_length: u64                                       │
│   manifest_sequence: u64     (monotonic counter, incremented on  │
│                               each append/delete; 0 for initial) │
│   prev_catalog_offset: u64   (offset of previous full catalog,   │
│                               0 if this is the first version)    │
│   file_checksum: u64         (BLAKE3 truncated to 64 bits)       │
│   reserved: [u8; 132]        (zeroed, future use)                │
├──────────────────────────────────────────────────────────────────┤
│ ROOT CATALOG (offset 256, max 4096 bytes)                        │
│   Compact catalog with section count, summary stats, and         │
│   offset/length for each top-level section group.                │
│   Enables single-read open for both local and range-read access. │
│   (See §3.2 for format.)                                         │
├──────────────────────────────────────────────────────────────────┤
│ PADDING (to 8-byte alignment)                                    │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: obs metadata          (Arrow IPC file format)           │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: obs predicate indexes                                   │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: var metadata          (Arrow IPC file format)           │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: var predicate indexes                                   │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: X/csr/000000          (CSR shard 0)                     │
│ SECTION: X/csr/000001          (CSR shard 1)                     │
│ ...                                                              │
│ SECTION: X/csr/{N-1}           (CSR shard N-1)                   │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: X/csc/000000          (optional, if has_csc)            │
│ ...                                                              │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: X/bitmap/000000       (optional, if has_bitmap)         │
│ ...                                                              │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: layers/{name}/csr/... (each layer = full CSR shard set) │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: obsm/{name}           (Arrow IPC tensor)                │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: obsp/{name}/csr/...   (sparse graphs)                   │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: uns/metadata.json     (raw JSON bytes)                  │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: provenance            (structured provenance, see §3.7) │
├──────────────────────────────────────────────────────────────────┤
│ SECTION: deletion_vectors      (optional, if has_deletion_vectors│
│                                 see §3.6.3)                      │
├──────────────────────────────────────────────────────────────────┤
│ FULL CATALOG (version 0)                                         │
│   Complete index of all sections. (See §3.2 for format.)         │
│   Points to: original sections.                                  │
├──────────────────────────────────────────────────────────────────┤
│ (After append: new sections appended here)                       │
│ SECTION: X/csr/{N}..{N+M-1}   (new CSR shards from append)      │
│ SECTION: obs metadata v1       (updated obs with new cells)      │
│ SECTION: deletion_vectors v1   (optional updated deletion vecs)  │
│ FULL CATALOG (version 1)                                         │
│   References both original and new sections.                     │
│   Header's full_catalog_offset updated to point here.            │
│   (Previous catalog remains for rollback.)                       │
└──────────────────────────────────────────────────────────────────┘
```

**Byte ordering**: All multi-byte integers in the file format are little-endian.
The `endian` field exists for validation; readers MUST reject files with
`endian != 0`. Big-endian systems must byte-swap on read.

**Section alignment**: Every section starts at an 8-byte-aligned offset. Padding
bytes (zeroed) are inserted between sections as needed.

### 3.2 Dual Catalog

The file contains two catalog structures to optimize both local and remote access.

**Root catalog** (fixed position, bytes 256..256+root_catalog_length):

A compact summary that enables a single contiguous read to open the file.
For local files, `open()` reads the first 256 + root_catalog_length bytes
(typically <4 KB total). For cloud range reads, a single GET of the first
4 KB returns the header + root catalog.

```
ROOT CATALOG FORMAT:
  n_section_groups: u16
  For each group:
    group_type: u8         (enum: obs_meta, var_meta, csr_shards, csc_shards,
                            bitmap, layer, obsm, obsp, uns, provenance)
    first_section_offset: u64
    total_group_length: u64
    n_sections: u32
    summary: [u8; 32]     (group-specific: e.g., for csr_shards: nnz, value stats)
```

The root catalog tells the reader where each group of sections starts and how large
it is, without enumerating individual sections. This is sufficient for:
- Streaming reads (read csr_shards group from start to end)
- Training loader (knows the CSR shard region offset + total length)
- Summary statistics (n_obs, n_vars, nnz without reading any sections)

**Full catalog** (at end of file, offset full_catalog_offset):

Complete index of every section. Used for random access to individual shards.

```
FULL CATALOG FORMAT:
  catalog_version: u16     (1)
  manifest_sequence: u64   (matches header; monotonically increasing)
  prev_catalog_offset: u64 (offset of previous catalog, 0 if first)
  n_obs: u64               (total observable cells after deletions)
  n_entries: u32
  For each entry:
    name_length: u16
    name_bytes: [u8; name_length]   (UTF-8 path, e.g. "X/csr/000042")
    offset: u64                     (byte offset in file)
    length: u64                     (byte length of section)
    section_type: u8                (enum: see below)
    checksum: [u8; 32]             (BLAKE3 of section content)
    stats_length: u16              (0 if no stats; length of stats block)
    stats: [u8; stats_length]      (per-section statistics, format below)
  catalog_checksum: [u8; 32]       (BLAKE3 of all preceding catalog bytes)

SHARD STATISTICS (present when section_type is csr_shard, csc_shard,
                  layer_csr_shard, or obsp_csr_shard; stats_length > 0):
  row_start: u64                   (first row index, global)
  row_end: u64                     (exclusive end row index, global)
  nnz: u64                         (non-zeros in this shard)
  value_min: u32                   (minimum non-zero value)
  value_max: u32                   (maximum non-zero value)
  value_sum: u64                   (sum of all values in shard)
  n_indexed_columns: u8            (number of per-column stat entries)
  For each indexed column:
    column_name_hash: u64          (BLAKE3 truncated hash of column name)
    stat_type: u8                  (0=min_max for numeric, 1=category_bitset)
    If stat_type == 0 (numeric):
      col_min: f64                 (min value of this column in shard's rows)
      col_max: f64                 (max value of this column in shard's rows)
    If stat_type == 1 (categorical):
      bitset_length: u16
      category_bitset: [u8]        (bit i set if dictionary index i is
                                    present in this shard's rows)

SECTION_TYPE ENUM:
  0: obs_metadata
  1: obs_index
  2: var_metadata
  3: var_index
  4: csr_shard
  5: csc_shard
  6: bitmap_shard
  7: layer_csr_shard
  8: obsm_embedding
  9: obsp_csr_shard
  10: uns_blob
  11: provenance
  12: deletion_vectors
  13..255: reserved for extensions (multimodal, spatial)
```

### 3.3 CSR Shard Internal Layout

Each CSR shard is a contiguous section within the packed file.

```
CSR SHARD (contiguous bytes)
┌──────────────────────────────────────────────────────────────┐
│ SHARD HEADER (64 bytes)                                      │
│   magic: [u8; 4] = b"SCXS"                                   │
│   shard_format_version: u8  (1)                              │
│   shard_type: u8            (0=CSR, 1=CSC)                   │
│   codec_id: u8              (may override file header, §4.5) │
│   value_encoding: u8        (may override file header, §4.5) │
│   index_dtype: u8           (0=u16, 1=u32)                   │
│   reserved_flags: [u8; 3]                                    │
│   n_major: u32              (rows in this shard for CSR)     │
│   n_minor: u32              (total columns in full matrix)   │
│   nnz: u64                  (non-zeros in this shard)        │
│   global_offset: u64        (first row index in full matrix) │
│   indptr_rel_offset: u32    (relative to shard start)        │
│   indptr_length: u32                                         │
│   indices_rel_offset: u32                                    │
│   indices_length: u32                                        │
│   values_rel_offset: u32                                     │
│   values_length: u32                                         │
│   block_index_rel_offset: u32                                │
│   block_index_length: u32                                    │
│   checksum: [u8; 8]         (BLAKE3 truncated to 64 bits)    │
├──────────────────────────────────────────────────────────────┤
│ INDPTR SECTION     (Delta-Golomb encoded, see §4.1)          │
├──────────────────────────────────────────────────────────────┤
│ INDICES SECTION    (FOR-BP encoded, see §4.2)                │
├──────────────────────────────────────────────────────────────┤
│ VALUES SECTION     (Adaptive Rice encoded, see §4.3)         │
├──────────────────────────────────────────────────────────────┤
│ BLOCK INDEX                                                  │
│   n_blocks: u32                                              │
│   For each block:                                            │
│     row_start: u32        (first row in block, global index) │
│     n_rows: u16           (rows in this block)               │
│     indptr_byte_offset: u32   (within indptr section)        │
│     indices_byte_offset: u32  (within indices section)       │
│     values_byte_offset: u32   (within values section)        │
│     nnz_in_block: u32                                        │
│   Enables O(1) random access to any row range.               │
└──────────────────────────────────────────────────────────────┘
```

**Index dtype**: With ≤65535 genes, `index_dtype=0` (uint16) halves index storage.
Files with >65535 features use `index_dtype=1` (uint32). The value is set once per
file in the file header and all shards must use the same dtype.

**Shard sizing**: Default 10,000 cells per CSR shard. At 5% density with 30K genes,
~15M non-zeros per shard, ~30-60 MB compressed.

### 3.4 Arrow IPC Metadata

Cell (`obs`) and gene (`var`) metadata are stored as **Arrow IPC file format** (not
streaming format). The file format includes a footer with schema and record batch
offsets, enabling random access to specific columns without reading the entire
metadata section.

- **Nullable columns**: Standard Arrow validity bitmaps are used.
- **Categorical columns**: Arrow dictionary encoding for space efficiency.
  Predicate indexes (§3.5) are built on the dictionary values, not encoded indices.
- **Chunking**: For datasets >10M cells, obs metadata is written as multiple Arrow
  record batches (one per CSR shard group) within a single IPC file. This enables
  lazy loading of metadata for the subset of shards being read.
- **Zero-copy access**: Arrow IPC is memory-mappable. Python readers expose Arrow
  arrays directly via PyArrow, which integrates with pandas (zero-copy for primitive
  types) and Polars.

### 3.5 Predicate Index Format

For each indexed metadata column, a sorted array of (value → shard ranges) mappings.

```
PREDICATE INDEX SECTION:
  index_version: u8         (1)
  n_indexed_columns: u8
  For each column:
    column_name_length: u16
    column_name: [u8]       (UTF-8)
    column_type: u8         (0=categorical, 1=numeric)
    n_entries: u32

    If categorical:
      For each unique value (sorted lexicographically):
        value_length: u16
        value_bytes: [u8]   (UTF-8 category label)
        n_shard_ranges: u16
        For each range:
          shard_id: u32
          row_start: u32    (within shard, local index)
          row_end: u32      (exclusive)

    If numeric:
      // B+ tree for range queries
      fanout: u16            (branching factor, typically 64)
      n_leaf_pages: u32
      n_internal_pages: u32
      // Pages laid out sequentially:
      For each internal page:
        n_keys: u16
        keys: [f64; n_keys]           (split values)
        children: [u32; n_keys + 1]   (page indices)
      For each leaf page:
        n_entries: u16
        For each entry:
          min_value: f64
          max_value: f64
          shard_id: u32
          row_start: u32
          row_end: u32
```

**High-cardinality handling**: For columns with >10,000 unique values (e.g.,
`donor_id`), the categorical index becomes a hash map serialized as a minimal
perfect hash function (pointing to offset arrays), capping the index at ~100 KB
regardless of cardinality.

**Unindexed columns**: Not all columns need indexes. The writer indexes columns
marked as `indexed=True` during file creation, or auto-indexes columns with
<1000 unique values. The reader falls back to sequential scan for unindexed columns.

### 3.6 Write Safety and the Fragment/Manifest Model

SCX uses a **fragment/manifest architecture** inspired by Lance and Delta Lake.
Shards and other sections are **immutable fragments** — once written, their bytes
never change. The full catalog acts as a **manifest** that defines the current
state by listing which fragments are active. This enables safe appends, logical
deletions, and versioning within a single file.

#### 3.6.1 Initial File Creation

Writing a new `.scx` file uses **atomic rename**:

1. Writer creates a temporary file (e.g., `experiment.scx.tmp.{pid}`)
2. Sections are written sequentially to the temp file
3. Full catalog (manifest version 0) is appended
4. Root catalog is written (writer knows all offsets now)
5. Header is written at position 0 (single `pwrite`), with
   `manifest_sequence=0` and `prev_catalog_offset=0`
6. `fsync()` the temp file
7. `rename()` the temp file to the final path (atomic on POSIX)

If the process is killed at any point before step 7, the temp file is an incomplete
artifact that can be safely deleted. The final `.scx` file either exists in its
entirety or doesn't exist at all.

**Note on step 4-5**: The root catalog (at offset 256) and the header (at offset 0)
are written after all sections, using `pwrite()`. This means the sequential write
path emits sections into the region starting at the root catalog's end (typically
offset ~4352) and then fills in the header and root catalog at the end. This
requires two seek-writes at close time but preserves the property that the file is
never in a partially-valid state on the final path.

#### 3.6.2 Appending (Fragment/Manifest)

Appending data (new cells, new layers, new embeddings) to an existing `.scx` file
does **not** require rewriting existing data:

1. Open the existing file for append (`O_WRONLY | O_APPEND` or equivalent)
2. Acquire an advisory `flock()` to prevent concurrent appends
3. Seek to the end of the file
4. Write new sections (e.g., new CSR shards, updated obs metadata) sequentially
5. Write a new full catalog (manifest) that references **all** active sections —
   both the original sections and the newly appended ones. Set
   `prev_catalog_offset` to the offset of the previous catalog.
6. Update the root catalog via `pwrite()` at offset 256
7. Update the file header via `pwrite()` at offset 0:
   - Increment `manifest_sequence`
   - Update `full_catalog_offset` and `full_catalog_length` to point to the
     new catalog
   - Update `n_obs`, `nnz`, `n_csr_shards` to reflect the appended data
8. `fsync()` the file
9. Release the `flock()`

**Atomicity**: The commit point is step 7 — the `pwrite()` of the header. Until
the header is updated, readers see the previous manifest. If the process is killed
during steps 3-6, the file has trailing garbage after the last valid catalog, which
is harmless — readers locate the catalog via the header's `full_catalog_offset`,
not by scanning to EOF.

**Immutability invariant**: Append NEVER overwrites existing sections. It only
appends new bytes at the end and updates the header + root catalog (fixed offsets).
The previous catalog remains in the file at its original offset, enabling rollback.

**Metadata updates**: When appending cells, the writer must also append a new
`obs_metadata` section that covers all cells (original + appended). The new catalog
references this updated metadata section instead of the original. The original
metadata section remains in the file (immutable) but is no longer referenced by
the active catalog. Gene (`var`) metadata does not change during append (appending
is cell-axis only).

#### 3.6.3 Deletion Vectors

Instead of rewriting files to remove low-quality cells or doublets, SCX supports
**logical deletion** via deletion vectors:

```
DELETION VECTORS SECTION:
  dv_version: u8             (1)
  n_shards_with_deletions: u32
  For each shard with deletions:
    shard_id: u32            (index into the catalog's CSR shard list)
    deletion_bitmap_length: u32
    deletion_bitmap: [u8]    (Roaring Bitmap of deleted row indices,
                              local to the shard)
```

When `has_deletion_vectors` is set in the header flags, the reader loads the
deletion vectors section and applies it during shard decode — deleted rows are
skipped and not included in query results.

**Creating deletion vectors**: A filter operation can produce deletion vectors
without rewriting data:

```python
# Logical deletion — writes only new deletion vectors + catalog (~KB)
exp = scx.open("experiment.scx")
exp.mark_deleted(exp.obs["is_doublet"] == True)
# Appends a deletion_vectors section and a new catalog to the file
```

**Undo**: Rolling back to the previous catalog (via `scx rollback`) restores
logically deleted cells, since the underlying shard data is never modified.

#### 3.6.4 Compaction

After many appends and deletions, the file may contain:
- Orphaned sections (referenced only by old catalogs, not the active one)
- Small fragments from incremental appends
- Logically deleted rows consuming disk space

`scx compact` rewrites the file to reclaim space:

```bash
scx compact experiment.scx --output compacted.scx
```

Compaction reads only the sections referenced by the active catalog, physically
removes deleted rows from shards, merges small shards into optimally-sized ones
(target: `shard_target_rows`), and writes a clean single-manifest file via the
standard atomic rename path (§3.6.1). The output file has `manifest_sequence=0`
and no previous catalogs.

#### 3.6.5 Rollback

Because previous catalogs are preserved in the file, rolling back to a prior
version is a header-only update:

```bash
scx rollback experiment.scx            # roll back one version
scx rollback experiment.scx --to-seq 3 # roll back to manifest_sequence=3
```

Rollback updates the header's `full_catalog_offset` and `manifest_sequence` via
`pwrite()` to point at a previous catalog. No data is deleted. To permanently
discard old versions, use `scx compact`.

### 3.7 Provenance Format

```
PROVENANCE SECTION:
  provenance_version: u8   (1)
  n_operations: u32
  For each operation:
    timestamp: i64          (Unix epoch seconds)
    action_length: u16
    action: [u8]            (UTF-8: "create", "convert", "merge", "append",
                             "delete", "compact", "rollback",
                             "add_layer", "add_csc", "add_obsm", etc.)
    tool_length: u16
    tool: [u8]              (UTF-8: "scx-cli 0.1.0", "pyscx 0.1.0", etc.)
    params_json_length: u32
    params_json: [u8]       (UTF-8 JSON: tool-specific parameters)
    input_checksums_count: u8
    For each input:
      checksum: [u8; 32]    (BLAKE3 of input file)
```

`scx merge` preserves provenance from all inputs: the output's provenance section
contains the full provenance chains of all input files, followed by the merge
operation itself. This enables full lineage tracking.

### 3.8 Checksum and Integrity Strategy

All checksums use **BLAKE3**, which provides both error detection and tamper
resistance at >5 GB/s on modern CPUs (faster than xxHash for long inputs, and
cryptographically secure).

**Granularity**: Each section has an independent BLAKE3 checksum in the full
catalog. If a single section's checksum fails:

1. The reader reports the specific corrupted section by name.
2. If the corrupted section is a non-essential layer, embedding, or graph,
   the reader can skip it and return a partial result (with a warning).
3. If the corrupted section is `obs_metadata`, `var_metadata`, or a CSR shard
   in the primary `X` matrix, the reader returns an error — the file must be
   re-obtained.

**Encryption**: Not included in v1. The spec reserves section_type values 240-254
for future encrypted section types. For datasets with PHI, the recommendation is
to use filesystem-level encryption (LUKS, dm-crypt, GCS CMEK) rather than
format-level encryption, as this avoids the key management complexity.

### 3.9 Versioning and Compatibility

- **format_version**: Incremented for breaking changes (new required fields,
  changed section semantics). Readers MUST reject files with `format_version`
  higher than their supported maximum.
- **Section types**: Unknown section types (≥12 for the current version) are
  skipped by the reader with a warning. This allows incremental format extension
  without breaking older readers.
- **header_length field**: If a future version needs a larger header, it increments
  `header_length`. Older readers that support only 256-byte headers can detect this
  and reject the file gracefully.
- **Migration**: `scx upgrade input.scx output.scx` rewrites a file to the latest
  format version.

### 3.10 Concurrent Read Safety

SCX sections (shards, metadata, etc.) are **immutable fragments** — once written,
their bytes never change. This means concurrent reads are always safe regardless
of whether the file is being appended to.

**During append**: The append path (§3.6.2) writes new sections at the end of the
file and atomically updates the header to point at a new catalog. Readers that open
the file before the header update see the previous manifest. Readers that open after
see the new manifest. There is no window where a reader sees a partially-updated
state, because:
1. New sections are written beyond the range any current reader is accessing
2. The catalog switch is a single `pwrite()` of the 256-byte header

**Advisory locking**: `flock()` prevents concurrent *appends* (two writers). It is
not needed for concurrent readers. A single writer may append while multiple readers
access the file concurrently.

---

## 4. The SCX Codec — Bit-Level Specification

All multi-byte values within codec streams are little-endian.

### 4.1 Indptr: Delta-Golomb-Rice

**Input**: Array of `n_major + 1` uint64 row pointers, monotonically non-decreasing.

**Encoding**:

1. Compute deltas: `delta[i] = indptr[i+1] - indptr[i]` for `i = 0..n_major-1`.
   Store `indptr[0]` as a raw uint64 (8 bytes) at the start of the section.
2. Compute the Rice parameter: `k = max(0, floor(log2(0.6931 * median(delta))))`.
   Store `k` as a single byte after the initial value.
3. For each delta `d`:
   - Quotient: `q = d >> k`
   - Remainder: `r = d & ((1 << k) - 1)`
   - Emit `q` ones followed by one zero (unary coding of quotient)
   - Emit `r` as `k` raw bits (LSB first)

**Decoding**: Read the initial uint64 and the byte `k`. Then for each delta:
reconstruct `d = (q << k) | r`, accumulate `indptr[i+1] = indptr[i] + d`.

**Byte alignment**: The bitstream is packed into bytes LSB-first. The section
is padded to byte boundary at the end.

### 4.2 Indices: Frame-of-Reference + Bit-Packing (FOR-BP)

**Input**: Array of `nnz` column indices (uint16 or uint32), sorted within each row.

**Block structure**: Indices are encoded in blocks. Each block covers the indices
for exactly `B_idx` rows (default `B_idx = 128`). The last block may have fewer rows.

**Per-block encoding**:

```
BLOCK HEADER (variable length):
  block_nnz: u32           (total indices in this block)
  n_rows_in_block: u16     (≤ B_idx)
  row_nnz: [varint; n_rows_in_block]   (nnz per row, varint encoded)

For each row within the block:
  If row_nnz > 0:
    frame_min: uint16/32   (minimum column index in this row)
    frame_bits: u8         (bits per delta after frame subtraction)
    deltas: [frame_bits-bit integers; row_nnz]
      where delta[0] = indices[0] - frame_min
            delta[j] = indices[j] - indices[j-1]  for j > 0
      (delta-of-delta within sorted row, after frame subtraction)
      Packed LSB-first into bytes
```

**`frame_bits` selection**: For each row, `frame_bits = ceil(log2(max_delta + 1))`
where `max_delta = max(delta[0..row_nnz])`. If all deltas are 0, `frame_bits = 0`
and no delta bytes are emitted.

**Varint encoding** (for row_nnz): Standard LEB128 — each byte uses 7 data bits
and 1 continuation bit (MSB). Values 0-127 are 1 byte.

### 4.3 Values: Adaptive Rice Coding

**Input**: Array of `nnz` non-zero count values (uint8, uint16, or uint32).

**Block structure**: Values are encoded in blocks of `B_val = 256` values.
The last block may have fewer values.

**Per-block encoding**:

```
BLOCK HEADER (1 byte):
  rice_k: u4              (bits 0-3: Rice parameter k, range 0-15)
  reserved: u4            (bits 4-7: zero)

BLOCK BODY:
  For each value v in the block:
    shifted = v - 1       (all values are ≥1 since they are non-zero)
    quotient q = shifted >> k
    remainder r = shifted & ((1 << k) - 1)
    Emit: q ones, then one zero, then r as k bits (LSB-first)
```

**Rice parameter selection**: Per block, compute the sample median of `(v - 1)`
for all values in the block. Set `k = max(0, floor(log2(0.6931 * median)))`.
This minimizes the expected code length under the geometric distribution
approximation of UMI counts.

**Example** (k=1, encoding value 1):
- shifted = 0, q = 0, r = 0
- Emit: "0" (zero zeros followed by one zero) + "" (0 bits of remainder) = 1 bit

**Example** (k=1, encoding value 2):
- shifted = 1, q = 0, r = 1
- Emit: "0" + "1" = 2 bits

**Example** (k=1, encoding value 5):
- shifted = 4, q = 2, r = 0
- Emit: "110" + "0" = 4 bits

**Byte alignment**: Each block's bitstream is padded to byte boundary.

### 4.4 SIMD and GPU Decode

**SIMD (AVX2)**: The Rice decoder processes 32 values per iteration by:
1. Loading 256 bits of bitstream into a YMM register
2. Using `PDEP`/`PEXT` to extract quotients via leading-one counting
3. Using shift+mask to extract k-bit remainders in parallel

**GPU (CUDA)**: Each warp (32 threads) decodes one block of 256 values:
1. Load the block into shared memory
2. Thread `t` decodes values at positions `t, t+32, t+64, ...` by scanning
   the bitstream with `__ballot_sync()` for unary quotient boundaries
3. Warp-shuffle to resolve offsets

Both paths produce the same output as the scalar decoder. The reference Rust
implementation (scalar) is the normative specification; SIMD and GPU paths are
optimizations that must produce bit-identical results.

### 4.5 Codec ID and Value Encoding

**codec_id** (1 byte, present in both file header and shard header):

| ID | Codec | Description |
|----|-------|-------------|
| 0  | none  | No compression. Raw CSR arrays (uint types). For GDS fast-path. |
| 1  | scx1  | Delta-Golomb indptr + FOR-BP indices + Rice values (this spec). |
| 2  | zstd  | Zstd per-section. Fallback for non-count data (float layers). |
| 3-255 | reserved | Future codecs (e.g., ANS-based entropy coder). |

**Per-shard adaptive encoding**: The file header's `codec_id` is a **default**.
Each shard's header may **override** it with a different codec_id. This allows
the writer to select the optimal encoding per shard based on the data distribution
within that shard. For example:

- A shard of highly-sequenced plasma cells with higher count values may benefit
  from a different Rice parameter range or a future ANS codec.
- A float-valued layer shard uses `codec_id=2` (Zstd) even if the file default
  is `codec_id=1` (Rice).
- Shards appended later (§3.6.2) can use newer codecs without rewriting existing
  shards. This enables incremental format upgrades.

Readers MUST use the **shard header's** `codec_id` and `value_encoding` when
decoding, not the file header's. The file header value serves as a hint for tools
that display summary information without reading individual shards.

**value_encoding** (1 byte, present in both file header and shard header):

| ID | Encoding | Bits | Use case |
|----|----------|------|----------|
| 0  | uint8    | 8    | Raw counts (most 10x data) |
| 1  | uint16   | 16   | High-depth counts |
| 2  | uint32   | 32   | Very high counts or merged data |
| 3  | float32  | 32   | Normalized/scaled layers |
| 4  | float16  | 16   | Compressed normalized layers |

Like `codec_id`, each shard's `value_encoding` may override the file header
default. This is particularly useful for files with mixed layers (integer raw
counts in X, float32 normalized values in a layer).

Rice coding (codec_id=1) applies only to integer value encodings (0-2). Float layers
use Zstd (codec_id=2) because float values do not follow the geometric distribution.

---

## 5. Detection Bitmap (Optional)

Bit-packed presence/absence stored as Roaring Bitmap sections.

**Storage**: 30K × 500K at 5% density → ~100-200 MB compressed (Roaring).

**Operations enabled** (with appropriate caveats):
- Gene detection rate: POPCOUNT columns, ~50x faster than CSC scan
- Jaccard cell similarity: AND + POPCOUNT with SIMD
- Fast approximate filtering: "which cells express gene X?"

The bitmap is useful for fast exploratory operations. For publication-quality
analyses (differential expression, trajectory inference), count-based methods
on the full CSR data should be used. The bitmap does not replace counts.

---

## 6. In-Memory Data Model

### 6.1 After `.collect()`: The `Experiment` Object

```rust
/// Materialized result of a query/analysis pipeline.
/// This is what .collect() returns.
pub struct Experiment {
    /// Cell metadata. Polars DataFrame (backed by Arrow arrays).
    pub obs: polars::DataFrame,

    /// Gene metadata. Polars DataFrame.
    pub var: polars::DataFrame,

    /// Primary expression matrix. SCX CSR representation.
    /// The indptr, indices, and data arrays are contiguous Rust Vecs.
    pub x: ScxCsr,

    /// Additional expression layers (same shape as X).
    pub layers: HashMap<String, ScxCsr>,

    /// Cell embeddings. HashMap of name → 2D Arrow Float32Array.
    pub obsm: HashMap<String, arrow::array::Float32Array>,

    /// Cell-cell pairwise matrices (e.g., kNN graph).
    pub obsp: HashMap<String, ScxCsr>,

    /// Unstructured metadata.
    pub uns: serde_json::Value,
}

/// SCX CSR matrix. Binary-compatible with scipy.sparse.csr_matrix layout.
pub struct ScxCsr {
    pub shape: (usize, usize),          // (n_rows, n_cols)
    pub indptr: Vec<i64>,               // length n_rows + 1
    pub indices: Vec<i32>,              // length nnz
    pub data: Vec<f32>,                 // length nnz (always f32 in memory)
}
```

### 6.2 Zero-Copy Interop with Python

**obs/var → pandas**: Polars DataFrames convert to pandas via Arrow with
zero-copy for primitive types (int, float) and dictionary types. String columns
require a copy to Python heap objects.

**ScxCsr → scipy.sparse.csr_matrix**: The `ScxCsr` struct uses the same memory
layout as scipy's CSR (indptr, indices, data arrays). PyO3 exposes these as numpy
arrays via the buffer protocol — **zero-copy**. The scipy wrapper is:

```python
import scipy.sparse as sp
import numpy as np

# Inside pyscx, after .collect():
csr = sp.csr_matrix(
    (result._x_data_array, result._x_indices_array, result._x_indptr_array),
    shape=result._x_shape,
    copy=False,  # zero-copy: numpy arrays point to Rust memory
)
```

**ScxCsr → cupy.sparse.csr_matrix**: Same layout, exposed via `__cuda_array_interface__`
when the GPU path is used. Zero-copy.

This means `to_anndata()` is cheap — the expensive part is constructing the
AnnData Python object, not copying data. For the expression matrix (typically
the largest component), there is no copy.

### 6.3 Lazy Handles

For datasets that exceed available RAM, `.collect()` can return a lazy handle
instead of a materialized matrix:

```python
# Materialized (default for datasets < available RAM)
result = exp.collect()
result.X  # ScxCsr in memory

# Lazy (for large datasets)
result = exp.collect(lazy=True)
result.X  # LazyScxCsr: backed by mmap'd shards, reads on access
result.X[1000:2000, :]  # reads only the relevant shard(s)
```

---

## 7. Query Engine

### 7.1 Lazy Evaluation

```python
import scx

result = (
    scx.open("experiment.scx")
    .filter_obs("tissue == 'lung' and disease == 'healthy'")
    .with_qc(min_genes=200, min_counts=500, max_mito_pct=20)
    .with_hvg(n_top=2000)
    .with_normalize(target_sum=1e4)
    .with_log1p()
    .with_pca(n_comps=50)
    .with_neighbors(n_neighbors=15)
    .with_umap()
    .with_leiden(resolution=1.0)
    .collect()
)
```

**Error handling**: Schema errors (referencing nonexistent columns, type mismatches)
are raised immediately when the pipeline is constructed, not deferred to `.collect()`.
Runtime errors (e.g., checksum failure, out-of-memory) are raised during `.collect()`.

### 7.2 Optimizations

**Predicate pushdown**: Two levels of shard pruning:
1. **Catalog-level** (cheapest): Per-shard statistics in the full catalog (§3.2)
   enable skipping shards without reading any section data. For example,
   `filter_obs("total_counts > 5000")` checks each shard's `value_sum` and skips
   shards where no cell can meet the threshold. Categorical filters check the
   shard's `category_bitset` — if a shard contains no "T cell" rows, it is skipped
   entirely. This is analogous to Parquet/Iceberg manifest-level pruning.
2. **Index-level** (finer-grained): Predicate indexes (§3.5) provide row-level
   mappings within shards for columns that are indexed.

**Projection pushdown**: After HVG selection, gene-major operations use CSC if
available. If CSC is not present, the engine falls back to a streaming transpose
(reads all CSR shards, accumulates column-wise) which is slower but correct.

**Operation fusion**: `with_normalize(1e4).with_log1p()` → single fused pass.
The fused path works for total-count normalization and log1p because both are
per-cell operations that can be computed from a single row scan.

**Normalization methods**: The fused path supports:
- Total-count normalization (`target_sum`): per-cell, single-pass
- Log1p: per-element, composable with the above

Methods requiring global statistics (scran pooling, SCTransform, z-scoring) cannot
be fused with per-cell normalization. These run as separate passes. The query
engine detects the method and disables fusion when needed.

**Derived layers**: The result of normalization can optionally be stored as a new
layer in the output file (`result.write("output.scx", store_layers=["normalized"])`).
If not stored, normalized data is always computed on-the-fly from raw counts.

### 7.3 R API

```r
library(scx)

result <- scx_open("experiment.scx") |>
  filter_obs(tissue == "lung") |>
  with_hvg(n_top = 2000) |>
  with_normalize(target_sum = 1e4) |>
  with_log1p() |>
  with_pca(n_comps = 50) |>
  collect()

# Interop: one copy to R heap for obs, zero-copy for sparse matrix
seu <- to_seurat(result)
sce <- to_sce(result)
```

---

## 8. ML Training Data Loader

### 8.1 The Problem

TileDB-SOMA-ML's loader is slow because:
1. Entire data path runs in Python (GIL-bound)
2. Shuffling uses random coordinate lookups (scatter-gather I/O)
3. Sparse-to-dense traverses Arrow → scipy → numpy → torch (4 copies)
4. Tutorials run with `num_workers=0` due to CUDA fork deadlocks
5. No overlap between I/O, transform, and GPU compute

**Baseline note**: These observations are based on TileDB-SOMA-ML as of early 2026.
TileDB-SOMA has been actively developed; all benchmark comparisons in §8.8 will
use the latest available release with recommended configuration.

### 8.2 SCX Training Loader Architecture

Triple-buffered Rust pipeline:

```
Stage 1 (tokio async)     Stage 2 (rayon)           Stage 3 (GPU)
Read shard group      →   Shuffle + densify      →   model.forward()
  SCX codec decode         HVG projection              loss.backward()
  Sequential I/O           Fused normalize+log1p
                           Pin memory

All three stages run concurrently. After 2-stage warmup, GPU never waits.
```

"Zero Python on the hot path" means: all I/O, decompression, shuffling,
sparse-to-dense conversion, normalization, and memory pinning happen in Rust.
Python is involved only in the outer training loop (`for batch in dataloader`)
and the PyTorch forward/backward pass. The Rust → Python boundary transfers only
pre-pinned tensor pointers.

### 8.3 Quasi-Random Shard Shuffle

**Level 1 (shard order)**: Randomly permute shard indices per epoch. Draw groups
of K=8 shards. Sequential I/O within each group.

**Level 2 (row shuffle)**: Pool K × 10K = 80K cells. Fisher-Yates shuffle. Draw
mini-batches from shuffled buffer.

### 8.4 Gene Projection at Decode Time

Training on 2000 HVGs out of 30K: the loader intersects column indices with an HVG
bitmap during decode. Non-HVG values are never decompressed. ~15x data reduction.

### 8.5 Sparse-to-Dense Direct Write

CSR shard data writes directly into a pre-allocated, pinned, contiguous tensor.
No intermediate scipy.sparse. Each batch row filled by independent rayon thread.

### 8.6 GDS Path

NVMe → GPU VRAM bypass via `cuFileRead()` at shard offset within the packed `.scx`
file. GPU CUDA kernels decompress and densify in-place.

**Deployment prerequisites**: GDS requires local NVMe (not network-attached storage),
NVIDIA GPUDirect Storage drivers (nvidia-fs), and a compatible filesystem (ext4,
XFS — not GPFS or Lustre). On Chimera, this means staging the `.scx` file to local
NVMe scratch before using the GDS path. The CPU path (standard `pread()`) works on
all filesystems including GPFS.

### 8.7 Python API

```python
import scx
import torch

dataset = scx.TrainingDataset(
    "experiment.scx",
    hvg_indices=hvg_array,
    batch_size=1024,
    obs_columns=["dataset_id", "donor_id"],
    normalize=True,
    log1p=True,
    shard_group_size=8,
    prefetch_batches=4,
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)

for epoch in range(100):
    for batch in dataloader:
        x = batch["X"].to(device, non_blocking=True)
        loss = model(x, batch["obs"])
        loss.backward()
        optimizer.step()
```

### 8.8 Throughput Projections

10M cells × 2000 HVGs, batch_size=1024, A100 GPU, NVMe SSD:

| Metric | TileDB-SOMA-ML* | SCX (CPU) | SCX (GDS) |
|--------|-----------------|-----------|-----------|
| Batch/sec | TBD (benchmark) | ~500 (projected) | ~2000 (projected) |
| GPU utilization | TBD | 85-95% (projected) | 95-99% (projected) |

*All TileDB-SOMA-ML figures to be measured at benchmark time using the latest
release with recommended configuration. The SCX figures are engineering projections
based on component throughputs (NVMe bandwidth, Rust decode speed, CUDA kernel
launch rates) and will be validated with reproducible benchmarks before publication.

**Memory footprint model**: At `shard_group_size=8`, `prefetch_batches=4`,
`batch_size=1024`, `n_hvg=2000`:
- Shard group buffer: 8 shards × ~30 MB compressed = ~240 MB
- Decoded row buffer: 80K cells × 100 nnz/cell × 6 bytes = ~48 MB
- Pinned batch ring: 4 batches × 1024 × 2000 × 4 bytes = 32 MB
- Overhead (indexes, metadata): ~10 MB
- **Total: ~330 MB** (constant regardless of total dataset size)

---

## 9. GPU Compute Path

For the analysis pipeline (beyond training), the GPU path provides:
- PCA: cuSOLVER randomized SVD on cuSPARSE CSR matrices
- kNN: CAGRA/RAFT approximate nearest neighbors
- Leiden: cuGraph GPU Leiden clustering
- UMAP: cuML UMAP

---

## 10. File Operations

### 10.1 Writing

```python
scx.from_anndata(adata, "experiment.scx")
scx.from_10x("filtered_feature_bc_matrix.h5", "experiment.scx")
result.write("result.scx")
```

Uses atomic rename (§3.6). Single sequential pass.

### 10.2 Reading

```python
exp = scx.open("experiment.scx")  # reads header (256B) + root catalog (<4KB)
adata = exp.to_anndata()           # zero-copy CSR, Arrow→pandas metadata
```

### 10.3 CLI

```bash
scx convert --from h5ad input.h5ad output.scx
scx convert --from 10x filtered_feature_bc_matrix.h5 output.scx
scx convert --to h5ad input.scx output.h5ad

scx info experiment.scx              # summary stats + manifest history
scx validate experiment.scx          # BLAKE3 verification of all sections
scx benchmark experiment.scx         # I/O + pipeline benchmarks
scx query experiment.scx "cell_type == 'T cell'" --count
scx compact experiment.scx -o out.scx  # reclaim space from appends/deletes
scx rollback experiment.scx          # revert to previous manifest
scx delete experiment.scx --filter "is_doublet == True"  # logical deletion
```

### 10.4 Building CSC

CSC is opt-in. By default, `scx convert` produces CSR-only files.

```bash
scx build-csc input.scx output.scx   # reads CSR, writes new file with CSR+CSC
```

This writes a complete new file (not an in-place modification) via the atomic
rename path. For large files, the streaming transpose reads CSR shards and
accumulates column data in chunks, bounded by a configurable memory limit.

### 10.5 Merging, Appending, and Compaction

**Merge** (produces a new file):

```bash
scx merge batch1.scx batch2.scx batch3.scx --output atlas.scx
```

Streaming merge: reads all inputs, concatenates metadata, renumbers shards,
writes new file. Provenance chains from all inputs are preserved.

**Append** (modifies file in-place via fragment/manifest, see §3.6.2):

```bash
scx append atlas.scx --input new_batch.scx
```

Appends new CSR shards and updated metadata to the existing file. No existing
bytes are modified — new sections are written at EOF and a new catalog is
committed atomically. The previous catalog is preserved for rollback.

```python
# Python API
exp = scx.open("atlas.scx", mode="append")
exp.append_from_anndata(new_adata)  # appends shards + catalog
```

**Compact** (reclaims space from appends and deletions):

```bash
scx compact experiment.scx --output compacted.scx
```

Rewrites the file to remove orphaned sections, physically delete rows marked
by deletion vectors, and merge small fragments into optimal shards. See §3.6.4.

**Delete** (logical deletion via deletion vectors, see §3.6.3):

```bash
scx delete experiment.scx --filter "is_doublet == True"
```

Marks matching cells as deleted without rewriting data. Appends a deletion
vectors section and a new catalog.

**Rollback** (revert to a previous manifest version, see §3.6.5):

```bash
scx rollback experiment.scx
```

---

## 11. Multimodal Extensibility

The section-based layout and extensible section_type enum support multimodal data
without breaking format changes.

### 11.1 Multi-Feature-Space Extension

A CITE-seq experiment (RNA + protein) with shared cell barcodes:

```
experiment.scx
├── obs metadata          (shared across modalities)
├── mod/rna/var metadata  (RNA genes)
├── mod/rna/X/csr/...     (RNA count matrix)
├── mod/protein/var metadata (ADT antibodies)
├── mod/protein/X/csr/...    (protein count matrix, different sparsity)
└── catalog
```

The `has_modalities` flag (bit 4) signals that the file contains multiple
feature spaces. Section paths include the modality prefix (`mod/rna/X/csr/000000`).
Each modality has its own var metadata, X matrix, and optionally layers/embeddings.
obs metadata is shared (all modalities measure the same cells).

This follows the MuData model and can round-trip to MuData/MuOn objects.

### 11.2 Spatial Extension

For spatial transcriptomics (Visium, MERFISH), obs metadata includes spatial
coordinates as standard numeric columns (`x_spatial`, `y_spatial`, `z_spatial`).
An optional spatial index section (R-tree) enables spatial range queries.

Images (H&E, fluorescence) can be stored as `uns` blob sections with a defined
naming convention (`uns/spatial/images/hires`, etc.), following the SpatialData
model.

### 11.3 Reserved Section Types

Section types 12-239 are reserved for future extensions. Types 240-254 are
reserved for encrypted sections. Type 255 is reserved as a sentinel.

---

## 12. Rust Crate Architecture

```
scx/
├── scx-format/          # File layout: header, catalog, shard reader/writer
├── scx-codec/           # Rice, FOR-BP, Delta-Golomb, SIMD, GPU decode kernels
├── scx-sparse/          # CSR/CSC ops: construct, slice, transpose, SpMV, SpMM
├── scx-engine/          # Lazy query engine: plan, optimize, execute, fuse
├── scx-loader/          # ML data loader: pipeline, shuffle, densify, pin, GDS
├── scx-gpu/             # CUDA: cuSPARSE interop, PCA, kNN, Leiden, GDS
├── pyscx/               # Python bindings (PyO3)
├── rscx/                # R bindings (extendr)
└── scx-cli/             # CLI: convert, info, validate, build-csc, merge, benchmark
```

---

## 13. Conformance and Testing

### 13.1 Conformance Test Suite

A set of reference `.scx` files and expected outputs:

- **Codec test vectors**: For each codec (Rice, FOR-BP, Delta-Golomb), a set of
  input arrays and their exact encoded byte sequences. Any conforming encoder
  must produce byte-identical output. Any conforming decoder must produce
  identical arrays from the reference encoded bytes.

- **Round-trip tests**: `h5ad → scx → h5ad` must produce numerically identical
  expression matrices (bit-exact for integer counts, within epsilon for float
  layers) and identical metadata (modulo pandas dtype normalization).

- **Fuzz testing**: The reader must not crash, panic, or execute undefined behavior
  on any input file. Fuzz targets for the shard decoder, catalog parser, and Arrow
  IPC reader are included in the test suite and run in CI.

### 13.2 Reference Implementation

The Rust implementation in the `scx-format` and `scx-codec` crates is the normative
reference. In case of ambiguity between this specification document and the Rust
code, the Rust code takes precedence for encoding/decoding behavior.

---

## 14. Comparison with Existing Formats

| Feature | h5ad | Zarr v3 | TileDB-SOMA | **SCX** |
|---------|------|---------|-------------|---------|
| Single file | Yes | No (folder) | No (database) | **Yes** |
| Sparse-native | CSR as 3 arrays | CSR as 3 arrays | Native sparse | **Native CSR** |
| Domain-specific codec | No | No | No | **Yes (Rice)** |
| Metadata predicate pushdown | No | No | Yes | **Yes** |
| HPC friendly | Good | Poor (many files) | Poor (database) | **Best** |
| Cloud-native | Poor | **Excellent** | **Excellent** | Adequate (range reads) |
| ML data loader | N/A | N/A | Slow (Python) | **Rust, fast** |
| Operation fusion | No | No | No | **Yes** |
| GPU zero-copy (GDS) | No | No | No | **Yes** |
| Multimodal support | No (MuData) | No (MuData) | Partial | **Extensible** |
| Append without rewrite | No | Yes | Yes (MVCC) | **Yes (fragment/manifest)** |
| Ecosystem maturity | Excellent | Good | Growing | **New** |

SCX does not claim to be superior in all dimensions. Its advantages are strongest
for HPC, GPU, and training workloads. For cloud-native atlas hosting with
low-latency random access, Zarr v3 or TileDB-SOMA remain more appropriate.

---

## 15. Implementation Roadmap

### Phase 0: Validate the Thesis (Months 1-2)
- Build `scx-loader` that reads **existing h5ad files** via a Rust backend
- Implement the triple-buffered pipeline with shard-like chunking on h5ad
- Benchmark against TileDB-SOMA-ML on the same h5ad data
- **Deliverable**: Reproducible proof that the Rust loader architecture saturates
  GPUs, independent of the SCX file format. This validates the core performance
  thesis before asking anyone to adopt a new format.

### Phase 1: Core Format + Codec (Months 2-4)
- `scx-format`: packed file reader/writer, dual catalog, atomic writes
- `scx-codec`: Rice, FOR-BP, Delta-Golomb with conformance test vectors
- `scx-cli`: convert from h5ad/10x, info, validate
- `pyscx` minimal: `scx.open()`, `to_anndata()`, `from_anndata()`
- **Deliverable**: `scx convert` works, round-trip tests pass, I/O benchmarks
  published vs. h5ad/zarr on representative datasets.

### Phase 2: Training Loader + Query Engine (Months 4-7)
- `scx-loader`: full pipeline reading from `.scx` files
- `scx-engine`: lazy evaluation, predicate pushdown, operation fusion
- scVI/scGPT DataModule integration
- **Deliverable**: End-to-end training pipeline on atlas-scale data with published
  throughput benchmarks.

### Phase 3: GPU Path (Months 7-10)
- `scx-gpu`: CUDA codec decoders, GDS integration
- GPU PCA, kNN, Leiden
- **Deliverable**: Full GPU analysis pipeline benchmarked vs. rapids-singlecell.

### Phase 4: Ecosystem (Months 10-14)
- `rscx`: R bindings, Seurat/SCE interop
- `scx build-csc`, `scx merge`
- Multimodal support (CITE-seq, spatial)
- Detection bitmap layer
- Conformance test suite and fuzz targets
- **Deliverable**: Production-ready release.

---

## 16. Why This Wins

scRNA-seq data is one of the most structured data types in computational biology.
The value distribution is near-geometric. The sparsity is extreme and patterned.
The access patterns are predictable. The compute pipeline is standardized.

Every existing tool treats this data as generic. SCX is the first system designed
around its actual structure — stored in a single file that is trivial to manage,
compressed with a codec that approaches the entropy limit, read through a query
engine that fuses operations, and loaded through a pipeline that keeps GPUs
saturated.

The strategic risk is real: replacing an entire stack simultaneously creates an
adoption chicken-and-egg problem. The phased roadmap addresses this by delivering
value incrementally — first a fast training loader (Phase 0, works with existing
h5ad), then a format (Phase 1), then an analysis engine (Phase 2). Each phase
stands on its own and provides concrete, measurable improvement over the status quo.