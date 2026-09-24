# scx-ops — File Operations

> Part of the [SCX API reference](README.md). Behaviour and invariants of each op are in
[docs/operations.md](../operations.md).

## `scx_ops::append(path, obs, indptr, indices, values, value_encoding, options: &AppendOptions) → Result<()>`
Append new cells at EOF from raw CSR arrays. `AppendOptions` bundles codec selection, shard sizing, and modality routing. Advisory flock for concurrent safety.

## `scx_ops::append_from_reader(target_path, source_reader, options: &AppendOptions, source_modality_id) → Result<()>`
Streaming SCX → SCX append. Reads one source CSR shard at a time (or copies raw bytes verbatim when codec / value encoding / index dtype / per-modality `n_vars` all match), avoiding materializing the entire source matrix in memory. **Multimodal targets are rejected** with `OpsError::MultimodalUnsupported` (deferred — a single-modality append would leave sibling modalities under-covering the shared obs axis; extract → append → re-merge instead). CSC sidecars are dropped on append (the rewrite ops carry them).

## `scx_ops::mark_deleted(path, cell_indices) → Result<u64>`
Logical deletion via Roaring Bitmap deletion vectors. Returns total deleted count.

## `scx_ops::compact(input, output) → Result<()>`
Rewrite file reclaiming deleted/orphaned space.

## `scx_ops::optimize(input, output, codec, obs_shard_policy) → Result<()>`

Three entry points, and the difference is the output's format version:

- `optimize(input, output, codec, obs_shard_policy)` — delegates with no framing, so it writes an **unframed v3** file.
- `optimize_with_framing(…, framing) → Result<OptimizeStats>` — `Some(FramingConfig)` with `row_group_rows > 0` writes framed **v4**; `None` is the v3 path above.
- `optimize_with_budget(…, framing, memory_budget) → Result<OptimizeStats>` — the same, plus a cap on what the parallel shard re-encode holds in flight. The other two delegate here with `None`.
Faithful upgrade of a single-modality file: re-encode + `canonicalize_csr`
every CSR-backed shard (X / layer / obsp-CSR) and row-group-frames it, stamping
`format_version=4`, without a full reconvert. No decode sidecar is written. Preserves row layout, obs/var, obsm/varm/obsp/varp, uns, predicate
indexes, and the deletion-vector section (does not apply deletions); drops the
CSC sidecar. Multimodal inputs are rejected (use `compact`). `obs_shard_policy`
(`ObsShardPolicy::{Off, Auto, Always}`, default `Auto`) migrates a
*single-section* legacy obs to the sharded `ObsMetadataShard` layout: `Auto`
shards when `n_obs > shard_target_rows` (the `from_anndata` threshold), `Always`
always shards, `Off` keeps the single section (the historical 1:1 copy).
Already-sharded obs is always stream-preserved regardless of the policy. See
[docs/operations.md § Optimize](../operations.md#optimize).

## `scx_ops::rollback(path) → Result<()>` / `rollback_to(path, seq) → Result<()>`
Revert to previous (or specific) manifest version — header-only update.

## `scx_ops::merge(inputs, output) → Result<()>`
Streaming merge of multiple SCX files into one.

## `scx_ops::sort(input, output, opts: &SortOptions) → Result<SortSummary>`
Global obs-axis row reorder. `SortOptions.by` selects a lexicographic key sort;
`SortOptions.shuffle = Some(seed)` instead applies a **seeded random
permutation** (`scx sort --shuffle`), and is mutually exclusive with `by`,
`reverse` and `group_by` — the engine rejects the combination. Both modes share
the same emission, alignment and bounded-memory machinery: the order is computed
once in pass 0 and every downstream stage keys off destination row. The seed is
recorded in provenance and is the only record of the permutation. See
[operations.md](../operations.md) and
[sharding.md § Shuffling for training](../sharding.md#shuffling-for-training-scx-sort---shuffle).

## `scx_ops::modify_metadata(path, patch: &MetadataPatch) → Result<ModifyMetadataSummary>`
Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) of an existing file **in place, without re-encoding `X`**. Appends only the replaced section bytes at EOF and atomically repoints the catalog — cost is O(replaced sections), the matrix shards are never read or rewritten. The CSC sidecar and `data_generation` are preserved (no CSC rebuild). Any `None` field on `MetadataPatch` is left untouched. **A replaced `obs`/`var` keeps the predicate index it had**: the old section describes values that are gone, so it is rebuilt over the same columns (which re-derives the per-shard column stats, so pushdown survives the edit); `patch.index` names a different set instead. `n_obs` / `n_vars` are invariants — `var` must have `n_vars` rows; `obs` may arrive in either row space: `header.n_obs` rows (written as handed in) or the **live** count (`n_obs` minus the deletion-vector popcount, what `pyscx` `read_obs()` returns since 0.17 — scattered onto the physical axis with `scatter_batch_to_physical`, so a deleted row is `null` in every supplied column and keeps its obs-index barcode from the file). Any other length is rejected before any write (`OpsError::ShapeMismatch`, naming both counts); `obsm` is always physical-length. Replace semantics, not merge. Multimodal (`modality_id != 0`) returns `OpsError::MultimodalUnsupported`. Advisory flock; rollback-able via the catalog chain.

The returned `ModifyMetadataSummary` carries the per-column build outcomes (in the same shape every other rewrite op reports them) plus:

- `obs_columns_not_carried` / `var_columns_not_carried` — the columns the file indexed that the new index does not, so a caller can surface the loss instead of leaving it silent. Render these through `obs_not_carried_unreported()` / `var_not_carried_unreported()`, which drop the columns a build outcome already names: the two channels overlap on the carry path, and a front end printing both raw warns twice about one column.
- `obs_carried_forward` / `var_carried_forward` — per axis, and **success-gated**: true only when that axis was carried *and* the rebuild produced an index. A carry whose every column turned out unindexable reports `false` here and `true` in `obs_predicate_index_dropped`, rather than both at once.
- `obs_predicate_index_dropped` / `var_predicate_index_dropped` — the file had an index on that axis and the output has none.

`set_uns` discards the summary: a `uns`-only patch touches no axis.

## `scx_ops::set_uns(path, uns: &serde_json::Value) → Result<()>`
Convenience wrapper over `modify_metadata` for the headline case — replace the whole `uns` block (O(uns bytes)).

## `scx_ops::update_uns(path, patch: &serde_json::Value) → Result<()>`
The shallow-merge twin: `patch` must be a JSON object, and its top-level keys are laid over the file's existing `uns` (`MetadataPatch { uns, uns_merge: true }` underneath). A patch key replaces a same-named key wholesale; every other key survives verbatim; a file with no `uns` section gets `patch` as-is. The existing blob is read through the op's own file lock (no mmap) and must be a JSON object — anything else is refused before any write. Provenance records `"uns_merge": true` so `scx info` can tell a merge from a replace.
