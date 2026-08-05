# SCX Operations Reference

This document covers the behavior of SCX's mutating operations with respect
to matrix data, metadata, CSC sidecars, and complexity. For the binary format
details, see [docs/format.md](format.md). For sharding details, see
[docs/sharding.md](sharding.md).

## Operations Matrix

The **Writes** column is the one worth reading first: it says whether an
operation mutates its input or produces a separate file, which determines
whether you need an `<OUTPUT>` argument at all. Every in-place op stages a temp
file and `rename`s it over the target, so an interrupted run leaves the input
intact.

In-place does **not** imply undoable. `scx rollback` works only on the ops that
commit through the manifest chain (`prepare_in_place` / `commit_in_place`) and so
leave the previous catalog in the file — the import ops, `delete`,
`modify_metadata` / `set_uns`, `append`. **`build-csc --in-place` is not among
them**: it stages a wholly new file via `run_build_csc` and renames it over the
target, carrying no prior catalog, so a subsequent `scx rollback` fails with
`no previous catalog available for rollback`. Copy out first if you want a way
back.

| Operation | Writes | Matrix shards | Obs metadata | Var metadata | CSC sidecar | Predicate indexes |
|-----------|--------|---------------|--------------|--------------|-------------|-------------------|
| **append** | In place (`<TARGET> <SOURCE>`) | Existing CSR preserved; new CSR appended at EOF | Rewritten as merged Arrow IPC (all cells) | Unchanged | **Dropped** (warning emitted) | Stale entries preserved unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild covering all rows |
| **delete** (`mark_deleted`) | In place (`<FILE>`) | Unchanged (logical deletion vector) | Unchanged | Unchanged | Preserved | Unchanged |
| **modify_metadata** / **set_uns** | In place (`<FILE>`) | **Unchanged** (never read or rewritten) | Replaced if supplied (same `n_obs`) | Replaced if supplied (same `n_vars`) | **Preserved** | Dropped for the replaced obs/var axis unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild; untouched otherwise |
| **compact** | New file (`<OUTPUT>` required) | Rewrites live data (drops orphaned sections, merges small shards) | Rewrites live metadata | Rewrites | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild |
| **optimize** | New file, or in place when `<OUTPUT>` == `<INPUT>` (`<OUTPUT>` is required either way) | Re-encodes + canonicalizes every CSR shard (X / layer / obsp-CSR); shard boundaries preserved; row-group-frames shards; stamps `format_version=4` when framed (default) or `format_version=3` when unframed | **Preserved** (rows 1:1) | **Preserved** | **Dropped** (rerun `scx build-csc`) | **Preserved** (rows + shard boundaries unchanged) |
| **merge** | New file (`<OUTPUT>` required) | Writes new output combining all inputs | Writes merged metadata | Writes merged | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild |
| **subset** | New file (`<OUTPUT>` required, optional with `--dry-run`) | Writes new output with matching rows | Writes subset metadata | Writes subset | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild |
| **sort** | New file (`<OUTPUT>` required) | Rewrites all shards with cells reordered by obs key(s) | Rewritten in sorted order | Unchanged | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild |
| **sort `--shuffle`** | New file (`<OUTPUT>` required) | Same rewrite, but rows are reordered by a **seeded random permutation** instead of a key (seed recorded in provenance) | Rewritten in shuffled order | Unchanged | **Dropped** unless `--rebuild-csc` | Rebuilt as for `sort`, but a shuffle **maximally scatters** each value's shard ranges — the opposite of what a sort does to them |
| **build-csc** | In place, or a new file with `<OUTPUT>` | **Re-emitted** (not re-encoded or canonicalized); row-group framing is preserved from the input | **Preserved** | **Preserved** | **Built** (this is the op that creates it) | Sections **copied verbatim** (shard boundaries are unchanged, so their `ShardRange`s stay valid), but the per-shard catalog **column stats are not re-derived** — so Level-1 pruning stops firing and `filter_obs` falls back to a full scan. See the note below |
| **obs-import** / **doublet-import** | In place (`<FILE> <SOURCE>`) | **Unchanged** (never read or rewritten) | Replaced (same `n_obs`, plus the new columns) | **Preserved** | **Preserved** (X untouched) | **Preserved** on a pure column *add*; dropped only when `--overwrite` rewrites an indexed column (`obs_index_would_go_stale` decides) |
| **cellbender-import** (`attach_external_layer`) | In place (`<FILE> <CELLBENDER_H5>`) | **Unchanged** (never read or rewritten); a new layer's shards are appended | Replaced (same `n_obs`, plus the new columns) | Replaced (same `n_vars`, plus the new columns) | **Preserved** (X untouched, so `data_generation` / `csc_build_generation` are unchanged) | **Preserved** (only columns are added; CSR shard ranges are untouched, and the index carries no schema hash) |
| **rollback** | In place (`<FILE>`) | Unchanged (header repoints to previous catalog) | Unchanged | Unchanged | Restored (if previous catalog referenced it) | Restored |

### Restoring CSC after a mutating operation

When a mutating operation drops the CSC sidecar, it emits a warning:

```
log::warn!("append dropped 1 CSC shards from experiment.scx: rerun `scx build-csc` (or pass --rebuild-csc) to restore the column-major sidecar")
```

To restore:

```bash
# Standalone rebuild, in place — omit <OUTPUT>. Staged via a temp file +
# atomic rename, so a failure leaves experiment.scx untouched.
scx build-csc experiment.scx

# Or write a copy, leaving the input alone
scx build-csc experiment.scx experiment_with_csc.scx

# Or pass --rebuild-csc to the mutating operation
scx compact experiment.scx compacted.scx --rebuild-csc
```

The Python API exposes `pyscx.build_csc(input, output=None, memory_limit="4G", force=False, csc_cols_per_shard=5000)`
for standalone rebuilds — `output=None` (the default) rebuilds in place, and a
path writes a copy. Alternatively, set `csc="always"` at conversion time
via `pyscx.from_anndata(..., csc="always")` to emit the sidecar during the
initial write, or pass `rebuild_csc=True` to mutating operations like `pyscx.sort(..., rebuild_csc=True)`.

Either form preserves the input's row-group framing: a framed (v4) input yields
a framed output. Both derive it from `scx_ops::framing_for_csc_rebuild`, which
is the only correct source — see the note on `rebuild_csc_inplace` for why both
`None` and a `decode_target`-carrying `FramingConfig` are wrong here.

> **Known gap: `build-csc` keeps the predicate index but loses its pushdown.**
> `copy_auxiliary_sections` copies `obs_predicate_index` / `var_predicate_index`
> byte-for-byte, and the copy stays *valid* — shard boundaries and row ranges are
> unchanged. But `run_build_csc` re-encodes each shard through its own path and
> does not re-derive the per-shard catalog `column_stats`, and Level-1 pruning
> resolves a categorical predicate against those stats' `CategoryBitset`. So the
> index section is present and the pruning it enables is gone.
>
> Measured on a 200-cell / 8-shard file indexed on a clustered `cell_type`:
> `Level 1 eliminated 6/8` before `build-csc`, `0/8` after — same correct row
> count, full scan instead of pruning. Rebuild the index (`scx compact
> --index-obs …`, or re-run the op that created it) if you need pushdown back.
> Pinned by
> `scx-cli/tests/cli_ops_integration.rs::test_build_csc_preserves_predicate_index_sections_but_not_pushdown`.

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
`format_version=4` when framed (or `format_version=3` if unframed via `--row-group-rows 0`) — without a full reconvert. This is how an older file gains the
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

## CellBender import

`scx cellbender-import <target.scx> <cellbender_out.h5>` (and
`pyscx.cellbender_import`) attaches a CellBender `remove-background` output to
an existing file as a layer, **in place**. It appends new sections at EOF and
repoints the catalog — the same harness `append` and `modify_metadata` use — so
`X`, the CSC sidecar, `.raw`, deletion vectors, detection bitmaps and predicate
indexes all survive, and `scx rollback` undoes the whole import.

The join is by **barcode string, never by row position**. CellBender's
`<name>_filtered.h5` stores rows in descending-UMI order rather than the input's
row order, so a positional import would place every cell's corrected counts on
the wrong barcode while still producing a correctly-shaped layer. Target rows
absent from the CellBender output become empty layer rows marked
`cellbender_status = "absent"`, with `null` (not `0.0`) diagnostics.

Because the join is the whole risk surface, `--dry-run` runs every validation
and the join, prints the match counts, and writes nothing. Use it before
importing onto a large file.

Emitted alongside the layer: `obs` gets `cellbender_status`,
`cellbender_cell_probability`, `cellbender_cell_size`,
`cellbender_droplet_efficiency`, `cellbender_background_fraction`,
`cellbender_analyzed` and `cellbender_total_counts`; `var` gets
`cellbender_ambient_expression` and `cellbender_analyzed`; `uns["cellbender"]`
records the run metadata and the full join report; and a provenance entry is
appended with the source file's BLAKE3 in `input_checksums`.

Known interaction: `scx subset` currently drops layers, so subset before
importing rather than after.

## External obs import

`scx obs-import <target.scx> <table.csv>` / `pyscx.obs_import` land per-cell
annotations computed outside SCX — doublet scores, cell-type calls, anything with
one value per cell — onto an existing file as `obs` columns. `scx doublet-import`
/ `pyscx.doublet_import` are the doublet-caller wrapper over the same machinery,
and `rscx::scx_attach_obs` takes an R `data.frame` directly. The delimited-table
reader is **ungated** (no libhdf5); an `.h5ad` source needs `--features hdf5`.

For the analyst-facing walkthrough — which caller writes which column, how to
export per-batch h5ads, how to combine several tools — see
[docs/scanpy.md § Landing external per-cell annotations](scanpy.md#landing-external-per-cell-annotations-doublet-detection).
This section covers the operational invariants.

### In place, via the same harness as append

The import writes through `prepare_in_place` / `commit_in_place`: new sections
are appended at EOF and the catalog is repointed, exactly as `append`,
`modify_metadata` and `cellbender-import` do. `X` and its layers are never read
or rewritten, so the CSC sidecar, `.raw`, deletion vectors and detection bitmaps
all survive, `data_generation` / `csc_build_generation` are unchanged, and cost
is O(obs) rather than O(nnz).

The obs block *is* rewritten in full each time, because obs is a single logical
section that must cover every cell. On an 864 MB / 500k-cell atlas that is
~160–250 MB of superseded bytes orphaned **per import** — roughly 100× the
payload of the columns being added. `scx info` reports the orphaned total; see
[§ Compact](#compact) to reclaim it, and prefer one import of a concatenated
table over a loop of per-batch imports.

### The join is by key string, never by row position

The external tool's row order is its own business. A caller run per library
against a merged atlas returns rows in whatever order it pleased, and a
positional import would attach every cell's score to the wrong cell while
producing a perfectly well-shaped column. So:

- `--key`/`key=` names one obs column, `--key a,b` a composite (fused with an
  ASCII unit separator that no barcode can contain), and `obs_names` names the
  obs index. Omitted, the key is auto-resolved: the obs index, then a fallback
  list of barcode-style names.
- The two sides may spell the key differently. `--source-key` / `source_key=`
  pairs source columns to `key=` components **positionally**, like pandas
  `left_on` / `right_on`.
- A duplicated key on either side is an error, never a silent first-wins.
- Every key failure carries a key diagnosis naming the columns that *are*
  unique. `pyscx.diagnose_obs_key(path)` runs that diagnosis on its own — worth
  doing first on a merged atlas, where the obs index is often not unique and the
  one column that is may be one no fallback list would guess.

`--dry-run` runs the join and the diagnosis and writes nothing.

### Uncovered rows get `null`, never `0.0`

A target row the source does not cover has no value, and `0.0` would be a
scientific claim the tool never made — a doublet score of zero says "definitely a
singlet". So `on_missing_rows` defaults to **`"null"`** on every obs surface
(`"zero"` is accepted as an alias for the same policy, inherited from the
CellBender importer where a missing *matrix* row genuinely is zeros). `"error"`
refuses a partial import outright.

Partial coverage is a supported workflow, not a degraded one: run a caller on 6
of 116 batches, concatenate, import once, and the other 110 batches' cells stay
`null`. Downstream consumers are null-aware — `doublet_consensus` does not let a
tool vote on a cell it never saw, and a cell nobody voted on stays `null` rather
than becoming a singlet by default. A join whose every source row matched is
reported as *coverage* at `info` level; the "matched only …" warning with example
keys is reserved for the case where source rows genuinely failed to land, which
is the only case where a key-format mismatch is plausible.

If a `--status-column` is requested, covered rows get `"present"` and uncovered
rows `"absent"`, so the distinction survives a round-trip through h5ad even for
callers whose score column is legitimately zero-valued.

### When a predicate index survives

A pure column **add** leaves an obs predicate index valid: the index keys on
column name, and the columns it covers are untouched. An **overwrite** of an
indexed column does not — the index would describe values that no longer exist,
and `filter_obs` pushdown would silently return the wrong rows.

`obs_index_would_go_stale` decides precisely, comparing the columns this import
will write against the columns the on-disk index actually covers. It is
deliberately precise rather than conservative: dropping the index on every import
would kill pushdown for the overwhelmingly common add-only case. The result is
reported as `obs_index_dropped` in the import summary — check it if pushdown
performance changes after an import.

Note `--overwrite` **replaces rather than merges** a colliding column. Importing
N per-batch tables one after another keeps only the last one's values.

### Rollback granularity is one import

Each import is one manifest version, so `scx rollback` undoes exactly the most
recent one and leaves earlier imports standing. Three successive imports then one
rollback leaves the first two sets of columns in place, with queries and pushdown
still working against them. There is no way to undo an import from the middle of
that chain.

## Rollback

`scx rollback` is a single header `pwrite()` that repoints
`full_catalog_offset` and `manifest_sequence` to a previous catalog.
No data is deleted. Complexity is O(1) — independent of file size.
