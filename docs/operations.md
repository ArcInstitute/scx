# SCX Operations Reference

This document covers the behavior of SCX's mutating operations with respect
to matrix data, metadata, CSC sidecars, and complexity. For the binary format
details, see [docs/format.md](format.md). For sharding details, see
[docs/sharding.md](sharding.md).

## Operations Matrix

| Operation | Matrix shards | Obs metadata | Var metadata | CSC sidecar | Predicate indexes |
|-----------|---------------|--------------|--------------|-------------|-------------------|
| **append** | Existing CSR preserved; new CSR appended at EOF | Rewritten as merged Arrow IPC (all cells) | Unchanged | **Dropped** (warning emitted) | Rebuilt |
| **delete** (`mark_deleted`) | Unchanged (logical deletion vector) | Unchanged | Unchanged | Preserved | Unchanged |
| **compact** | Rewrites live data (drops orphaned sections, merges small shards) | Rewrites live metadata | Rewrites | **Dropped** unless `--rebuild-csc` | Rebuilt |
| **merge** | Writes new output combining all inputs | Writes merged metadata | Writes merged | **Dropped** unless `--rebuild-csc` | Rebuilt |
| **subset** | Writes new output with matching rows | Writes subset metadata | Writes subset | **Dropped** unless `--rebuild-csc` | Rebuilt |
| **rollback** | Unchanged (header repoints to previous catalog) | Unchanged | Unchanged | Restored (if previous catalog referenced it) | Restored |

### Restoring CSC after a mutating operation

When a mutating operation drops the CSC sidecar, it emits a warning:

```
UserWarning: CSC sidecar dropped by append; rebuild with `scx build-csc` or pass --rebuild-csc
```

To restore:

```bash
# Standalone rebuild
scx build-csc experiment.scx

# Or pass --rebuild-csc to the mutating operation
scx compact experiment.scx output.scx --rebuild-csc
```

```python
# Python
pyscx.build_csc("experiment.scx")
```

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
| **Predicate indexes** | O(all cells) | Rebuilt to cover the merged obs. |

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

## Compact

`scx compact` rewrites the file, dropping:
- Orphaned sections (referenced only by old catalogs)
- Rows marked by deletion vectors
- Small shards (merged to `shard_target_rows`)

The output is a clean single-manifest file (`manifest_sequence=0`).
Complexity is O(live data) — proportional to the surviving cells, not the
historical file size.

## Rollback

`scx rollback` is a single header `pwrite()` that repoints
`full_catalog_offset` and `manifest_sequence` to a previous catalog.
No data is deleted. Complexity is O(1) — independent of file size.
