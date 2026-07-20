# SCX Operations Reference

This document covers the behavior of SCX's mutating operations with respect
to matrix data, metadata, CSC sidecars, and complexity. For the binary format
details, see [docs/format.md](format.md). For sharding details, see
[docs/sharding.md](sharding.md).

## Operations Matrix

| Operation | Matrix shards | Obs metadata | Var metadata | CSC sidecar | Predicate indexes |
|-----------|---------------|--------------|--------------|-------------|-------------------|
| **append** | Existing CSR preserved; new CSR appended at EOF | Rewritten as merged Arrow IPC (all cells) | Unchanged | **Dropped** (warning emitted) | Stale entries preserved unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild covering all rows |
| **delete** (`mark_deleted`) | Unchanged (logical deletion vector) | Unchanged | Unchanged | Preserved | Unchanged |
| **modify_metadata** / **set_uns** | **Unchanged** (never read or rewritten) | Replaced if supplied (same `n_obs`) | Replaced if supplied (same `n_vars`) | **Preserved** | Dropped for the replaced obs/var axis unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild; untouched otherwise |
| **compact** | Rewrites live data (drops orphaned sections, merges small shards) | Rewrites live metadata | Rewrites | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild |
| **optimize** | Re-encodes + canonicalizes every CSR shard (X / layer / obsp-CSR); shard boundaries preserved; row-group-frames shards; stamps `format_version=4` | **Preserved** (rows 1:1) | **Preserved** | **Dropped** (rerun `scx build-csc`) | **Preserved** (rows + shard boundaries unchanged) |
| **merge** | Writes new output combining all inputs | Writes merged metadata | Writes merged | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild |
| **subset** | Writes new output with matching rows | Writes subset metadata | Writes subset | **Dropped** unless `--rebuild-csc` | **Dropped** (rebuild via `scx convert --index-obs ...` on the output) |
| **sort** | Rewrites all shards with cells reordered by obs key(s) | Rewritten in sorted order | Unchanged | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild |
| **rollback** | Unchanged (header repoints to previous catalog) | Unchanged | Unchanged | Restored (if previous catalog referenced it) | Restored |

### Restoring CSC after a mutating operation

When a mutating operation drops the CSC sidecar, it emits a warning:

```
UserWarning: CSC sidecar dropped by append; rebuild with `scx build-csc` or pass --rebuild-csc
```

To restore:

```bash
# Standalone rebuild — writes a new file with CSR + CSC shards
scx build-csc experiment.scx experiment_with_csc.scx

# Or pass --rebuild-csc to the mutating operation
scx compact experiment.scx compacted.scx --rebuild-csc
```

The Python wrappers do not expose a standalone `build_csc` function — use
`scx build-csc` on the CLI, or set `csc="always"` at conversion time via
`pyscx.from_anndata(..., csc="always")` to emit the sidecar during the
initial write.

## Append Complexity

Append writes new data at EOF without rewriting existing matrix shards.
However, metadata must be updated to cover all cells:

| Component | Complexity | Notes |
|-----------|-----------|-------|
| **New CSR shards** | O(new cells) | Encoded and written sequentially past the previous EOF. |
| **Obs metadata** | O(all cells) | A fresh `obs_metadata` Arrow IPC section covering *all* cells (existing + new) is written. The new catalog points to it; the old `obs_metadata` section remains in the file but is orphaned until `scx compact`. |
| **Var metadata** | O(1) | Unchanged — append is cell-axis only. The new catalog reuses the existing `var_metadata` entry. |
| **Root catalog** | O(catalog) | Rewritten at offset 256 via `pwrite()` (~4 KB). |
| **Full catalog** | O(catalog) | New catalog appended referencing both original and new sections. |
| **CSC sidecar** | Dropped | Column-major shard consistency cannot be maintained incrementally. |
| **Predicate indexes** | O(all cells) when rebuilt | By default the pre-append entries are preserved (stale — they cover only the original rows). Pass `--index-obs` / `--index-var` / `--index-preset` to rebuild covering all rows. Multimodal targets emit `PredicateIndexSkippedMultimodal` and skip the write. |

**Commit point**: the single header `pwrite()` that updates
`full_catalog_offset`, `n_obs`, `nnz`, `n_csr_shards`, and
`manifest_sequence`. Until then, readers see the previous manifest.

**Obs rewrite cost**: for a 1M-cell dataset, the obs metadata section is
typically 50–200 MB (depending on the number of obs columns and categorical
cardinality). Appending 10K cells rewrites this section in full. This cost
is fixed per append regardless of how many cells are appended — amortize by
batching multiple appends into one call.

## Delete Complexity

`mark_deleted` appends a deletion vectors section (~few KB for the Roaring
Bitmap) plus a new catalog. No matrix shards or metadata sections are
rewritten. Readers apply the bitmap during decode.

| Component | Complexity |
|-----------|-----------|
| Deletion vectors section | O(deleted cells) — Roaring Bitmap |
| New catalog | O(catalog) |
| Matrix shards | Unchanged |
| Obs/var metadata | Unchanged |

**Multimodal is fully supported.** `mark_deleted` is a whole-cell delete: the
deletion vector stores global obs row indices (`modality_id = 0`), so a deleted
cell is removed from **every** modality's read and from `compact` output, and a
modality-scoped query (`query(modality=…)`) on a file that carries deletions
returns deletion-filtered rows. (This supersedes the earlier limitation where
multimodal `mark_deleted` was guarded off.)

## Modify Metadata Complexity

`scx_ops::modify_metadata` / `set_uns` (CLI: `scx modify-metadata` / `scx
set-uns`; Python: `pyscx.modify_metadata` / `pyscx.set_uns`) replace metadata
sections (`uns` / `obs` / `var` / `obsm` / `varm`) in place. It appends only the
replaced section bytes at EOF and repoints the catalog — **the matrix is never
read or rewritten**, so the cost is O(size of the replaced sections), not
O(matrix). This is the key difference from `from_anndata` / `from_h5ad`, which
re-encode all of `X`.

| Component | Complexity | Notes |
|-----------|-----------|-------|
| **Matrix shards (CSR/CSC)** | O(1) | Never read or rewritten — original catalog entries pass through verbatim. |
| **`uns`** | O(uns bytes) | One fresh `UnsBlob` section. The headline cheap case. |
| **`obs`** | O(n_obs) | Re-sharded `ObsMetadataShard` sections. Must match the file's `n_obs`. |
| **`var`** | O(n_vars) | Single `VarMetadata` section. Must match `n_vars`. |
| **`obsm` / `varm`** | O(replaced matrices) | Only the named matrices are rewritten; other keys pass through. |
| **`obsp` / `varp`** | O(1) | Not replaceable here — existing sections (`ObspEmbedding` / `VarpEmbedding` and their shards) pass through unchanged. |
| **CSC sidecar** | **Preserved** | `data_generation` / `csc_build_generation` are left unchanged, so a pre-existing CSC sidecar stays valid — no `--rebuild-csc` needed. |
| **Predicate indexes** | O(n_obs)/O(n_vars) when rebuilt | A predicate index over a replaced `obs`/`var` is dropped (its values are now stale); pass `--index-obs` / `--index-var` / `--index-preset` to rebuild. Untouched when only `uns`/`obsm`/`varm` change. |

**Invariants (validated, never changed)**: `n_obs`, `n_vars`, `nnz`,
`n_csr_shards`, `HAS_CSC`. A shape mismatch (`obs.num_rows() != n_obs`, etc.) is
rejected *before* any write, leaving the file byte-identical. Changing cell/gene
count is out of scope — use `append`, `subset`, or `from_*`.

**Commit point**: the single header `pwrite()` that repoints
`full_catalog_offset` / `manifest_sequence` (rollback-able via the catalog
chain). Replace semantics, not merge — a supplied section fully supersedes the
old one. Repeated edits orphan the prior section bytes; run `scx compact` to
reclaim them. Multimodal (`modality != 0`) is not yet supported.

## Compact

`scx compact` rewrites the file, dropping:
- Orphaned sections (referenced only by old catalogs)
- Rows marked by deletion vectors
- Small shards (merged to `shard_target_rows`)

The output is a clean single-manifest file (`manifest_sequence=0`).
Complexity is O(live data) — proportional to the surviving cells, not the
historical file size.

## Optimize

`scx optimize <input> <output>` upgrades an existing **single-modality**
file in place: it decodes → `canonicalize_csr` → re-encodes and row-group-frames
every CSR-backed shard (`X`, layers, and obs×obs `obsp` CSR graphs), stamping
`format_version=4` — without a full reconvert. This is how an older file gains the
row-group random-access substrate and GPU device-decode benefits: framed Scx1
shards keep the GPU device-decode route (decoding group-by-group in VRAM). No
decode sidecar is written (see [scanpy.md § Data layout for fast GPU decode](scanpy.md#data-layout-for-fast-gpu-decode-to_gpu_anndata--device-resident-analysis)).

Unlike `compact`, `optimize` is a faithful 1:1 upgrade:
- It does **not** apply deletions — the deletion-vector section is carried
  through unchanged (use `compact` to reclaim deleted rows).
- It does **not** change CSR shard boundaries or row layout; obs/var, obsm/varm,
  COO obsp/varp (copied verbatim — sharded layouts preserved), uns, and predicate
  indexes pass through unchanged. (The one optional exception is obs-metadata
  *layout*: `--shard-obs` may migrate a single-section obs to shards — see below.
  Row *order* and content are still preserved exactly.)
- It **does** re-canonicalize every shard (sorting indices, summing duplicate
  coordinates, dropping explicit zeros), so nnz may legitimately drop.

The CSC sidecar is dropped (re-canonicalizing can change nnz and would leave the
column-major sidecar referencing stale offsets) — rerun `scx build-csc`.
Multimodal inputs are rejected with a message pointing at `scx compact`. An
in-place invocation (`--output` == input) is safe: the writer stages a sibling
tempfile and atomically renames over the target. Verify the result with
`scx validate --deep <out>`.

`--codec {auto|scx1}` selects the per-shard codec (default `auto`). `auto` keeps
the encoder's per-shard choice (Scx1 for low-median integer counts, else Zstd),
so a high-median count shard lands as Zstd (host-decoded on the GPU path).
`scx1` forces Scx1 on every integer shard — use
it when you want the whole file to take the `to_gpu_anndata` device-decode route
regardless of per-shard count magnitude (framed Scx1 shards decode in VRAM; Scx1
is less compact than Zstd on high-median data, the trade-off for a fully
on-device decode). Non-integer
(float) shards fall back to Zstd either way; `--codec zstd` is rejected.

`--shard-obs {off|auto|always}` (default `auto`) migrates a **legacy
single-section** obs table to the sharded `ObsMetadataShard` layout in the same
pass — the layout the streaming / cloud / bounded-memory read paths want at
atlas scale (see [sharding.md § Obs/var metadata sharding](sharding.md#obsvar-metadata-sharding)).
`auto` shards only when `n_obs > shard_target_rows` (the same threshold
`from_anndata` uses on the write path), so small/medium files stay
single-section and byte-faithful while only atlas-scale files change; `always`
shards unconditionally; `off` keeps the single section (the historical
behaviour). An **already-sharded** obs is always stream-preserved regardless of
the policy — `optimize` never collapses or re-sizes existing obs shards (use
`compact` to re-shard). Note this does not lower optimize's peak memory: the
single-section input is read whole by `read_obs()` either way, so the benefit is
purely for future readers of the output.

In Python: `pyscx.optimize(input, output, codec="auto", shard_obs="auto")`. The
`codec` kwarg takes `"auto"` (default) or `"scx1"`, and `shard_obs` takes
`"off"|"auto"|"always"` (default `"auto"`), with the same semantics as the CLI
flags; any other value raises `ValueError`. There is no `force` analogue — pass
`output == input` for an in-place upgrade, or remove the target first.

## Rollback

`scx rollback` is a single header `pwrite()` that repoints
`full_catalog_offset` and `manifest_sequence` to a previous catalog.
No data is deleted. Complexity is O(1) — independent of file size.
