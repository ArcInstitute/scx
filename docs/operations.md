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
`modify_metadata` / `set_uns`, `append`. **`build-csc --in-place` and
`upgrade --in-place` are not among them**: each stages a wholly new file and
renames it over the target, carrying no prior catalog, so a subsequent
`scx rollback` fails with `no previous catalog available for rollback`. Copy out
first if you want a way back.

> **Every op in this table invalidates a `pyscx` handle that is already open on
> the file.** The in-place ops append and rewrite the header; the copy-out ops
> rename a new file into place. Either way an open `Experiment`, backed
> `AnnData`, or `query()` pipeline is left mapping the previous contents, and
> reading through it raises rather than answering. Call `Experiment.reload()`,
> or re-derive from a fresh `pyscx.open(...)`. Mutating *through* a handle
> (`pyscx.obs_import(exp, ...)`, `exp.mark_deleted(...)`) reloads it for you.
> See [docs/api.md § Handles and files that change underneath them](api.md#handles-and-files-that-change-underneath-them).

| Operation | Writes | Matrix shards | Obs metadata | Var metadata | CSC sidecar | Predicate indexes | Deletion vectors |
|-----------|--------|---------------|--------------|--------------|-------------|-------------------|-------------------|
| **append** | In place (`<TARGET> <SOURCE>`) | Existing CSR preserved; new CSR appended at EOF | Rewritten as merged Arrow IPC (all cells) | Unchanged | **Dropped** (warning emitted) | Stale entries preserved unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild covering all rows | **Preserved** (deleted rows keep their global indices; appended rows are live) |
| **delete** (`mark_deleted`) | In place (`<FILE>`) | Unchanged (logical deletion vector) | Unchanged | Unchanged | Preserved | Unchanged | **Written** (this is the op that creates them) |
| **modify_metadata** / **set_uns** | In place (`<FILE>`) | **Unchanged** (never read or rewritten) | Replaced if supplied (same `n_obs`); a supplied `obsm` sets `has_obsm`, so a first-ever in-place embedding survives the next `compact` | Replaced if supplied (same `n_vars`) | **Preserved** | **Carried forward** for the replaced axis — rebuilt over the columns the file already indexed, which also re-derives the per-shard column stats. `--index-obs` / `--index-var` override that axis's column set (`--index-preset` / `--index-auto-threshold` override both); a previously indexed column the new set omits is *reported*, not silently dropped. Untouched when only `uns` / `obsm` / `varm` change — see the note below | **Preserved** (X and its row space are untouched) |
| **compact** | New file (`<OUTPUT>` required) | Rewrites live data (drops orphaned sections, merges small shards) | Rewrites live metadata | Rewrites | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild | **Applied** — deleted rows are dropped and no vector is emitted. This is the op that materializes deletions |
| **optimize** | New file, or in place when `<OUTPUT>` == `<INPUT>` (`<OUTPUT>` is required either way) | Re-encodes + canonicalizes every CSR shard (X / layer / obsp-CSR); shard boundaries preserved; row-group-frames shards; stamps `format_version=4` when framed (default) or `format_version=3` when unframed | **Preserved** (rows 1:1) | **Preserved** | **Dropped** (rerun `scx build-csc`) | Sections **preserved** (rows + shard boundaries unchanged), and the per-shard column stats are **carried from the input** (rows are 1:1, so the input's stats are exactly right for the output's shards), so **Level-1** pruning survives. See the note below | **Carried** verbatim (rows are 1:1, so the global row indices stay valid) |
| **upgrade** | New file, or in place (`--in-place`, temp + rename) — **not** rollback-able; **refuses multimodal input** | Decoded, **canonicalized** and re-emitted **unframed** (per-shard codec preserved; canonicalizing can change `nnz`); a file already newer than the target (v4) is **declined**. `varm`, `obsp`, `varp`, `.raw` and the group index are **carried**; detection bitmaps are carried too unless canonicalizing actually rewrote X, in which case they are dropped with a warning — see below | **Preserved** (rows 1:1; a sharded layout stays sharded) | **Preserved** | **Preserved** when canonicalizing left the matrix unchanged (every file a current writer produces) — the one op that carries the sidecar through rather than dropping it. **Dropped with a warning** when canonicalizing actually rewrote X, since the sidecar is then a view of a different matrix | Sections **copied verbatim** by `copy_auxiliary_sections`, and the per-shard catalog column stats are **carried from the input** (a 1:1 re-emit), so **Level-1** shard pruning survives — as for `build-csc`. See the note below | **Carried** verbatim (a 1:1 re-emit, so the global row indices stay valid) |
| **merge** | New file (`<OUTPUT>` required) | Writes new output combining all inputs | Writes merged metadata; an `obsm` key some inputs lack is a **hard error** naming the input | Writes merged; `varm` / `varp` come from input 0 | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild | **Carried**, with each input's rows offset into the merged row space (a sorted merge follows the merge permutation). Physical rows are all retained — run `compact` to reclaim them |
| **subset** | New file (`<OUTPUT>` required, optional with `--dry-run`) | Writes new output with matching rows | Writes subset metadata | Writes subset | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild | **Applied** — a subset builds a new row space, so deleted cells are excluded (whether or not `--filter` is given, and intersected with it when it is) |
| **sort** | New file (`<OUTPUT>` required) | Rewrites all shards with cells reordered by obs key(s) | Rewritten in sorted order | Unchanged | **Dropped** unless `--rebuild-csc` | **Dropped** unless `--index-obs` / `--index-var` / `--index-preset` requests a rebuild | **Applied** — deletions are materialized away by the reorder |
| **sort `--shuffle`** | New file (`<OUTPUT>` required) | Same rewrite, but rows are reordered by a **seeded random permutation** instead of a key (seed recorded in provenance) | Rewritten in shuffled order | Unchanged | **Dropped** unless `--rebuild-csc` | Rebuilt as for `sort`, but a shuffle **maximally scatters** each value's shard ranges — the opposite of what a sort does to them | **Applied** — as for `sort` |
| **build-csc** | In place, or a new file with `<OUTPUT>` | **Re-emitted** (not re-encoded or canonicalized); row-group framing is preserved from the input | **Preserved** | **Preserved** | **Built** (this is the op that creates it) | Sections **copied verbatim** (shard boundaries are unchanged, so their `ShardRange`s stay valid), and the per-shard catalog column stats are **carried from the input**, so **Level-1** shard pruning survives. See the note below | **Carried** verbatim (a 1:1 re-emit; the CSC sidecar is built over the physical rows, which the vector still indexes correctly) |
| **obs-import** / **doublet-import** | In place (`<FILE> <SOURCE>`) | **Unchanged** (never read or rewritten) | Replaced (same `n_obs`, plus the new columns) | **Preserved** | **Preserved** (X untouched) | **Preserved** on a pure column *add*; dropped only when `--overwrite` rewrites an indexed column (`obs_index_would_go_stale` decides). The per-shard catalog column stats are cleared **per rewritten column** — see the note below | **Preserved** (X untouched) |
| **cellbender-import** (`attach_external_layer`) | In place (`<FILE> <CELLBENDER_H5>`) | **Unchanged** (never read or rewritten); a new layer's shards are appended | Replaced (same `n_obs`, plus the new columns) | Replaced (same `n_vars`, plus the new columns) | **Preserved** (X untouched, so `data_generation` / `csc_build_generation` are unchanged) | Same as **obs-import**: preserved on a pure column *add*, dropped when `overwrite` rewrites an indexed obs column, and the column stats cleared per rewritten column | **Preserved** (X untouched) |
| **rollback** | In place (`<FILE>`) | Unchanged (header repoints to previous catalog) | Unchanged | Unchanged | Restored (if previous catalog referenced it) | Restored | Restored (as of the previous catalog) |

### Everything the matrix above does not have a column for

The matrix covers the seven things most people need. A file can carry twenty
kinds of section, and "which of them survive this op" used to be answered in
five different places in the source with no two agreeing. It is now answered
once, in **`scx-ops/src/carry.rs`** — a table over (operation × section family)
that every rewrite op is checked against at run time, so an op that stops
writing something it claims to carry fails loudly instead of losing it quietly.

Read that table rather than this document when you need the answer for `varm`,
`obsp`, `varp`, `adata.raw`, detection bitmaps, layers or the grouped-sort
sidecar. It is the source of truth, it is snapshot-tested, and it cannot get out
of step with the code without a test going red.

Three of its answers are worth surfacing here.

**`merge` carries the cell–cell graph in its COO encoding, and refuses a partial
`obsm`.** A COO `obsp` graph is rebased by each input's offset in the
concatenated obs axis; `varp` comes from input 0, the rule `var` and `varm`
already follow. Three things this means in practice:

- `merge --sort-by` **refuses** an input carrying COO `obsp`, the way it already
  refuses `obsm`: a sorted merge interleaves rows, so there is no per-input
  output row range to record. Merge without `--sort-by`, then `scx sort`.
- An `obsm` / `obsp` key that some inputs have and others lack is now a hard
  error naming the axis, the key and the input — the same answer a missing
  *layer* has always had. Merging 100 per-sample files where one lacks `X_umap`
  used to yield an atlas with no UMAP and no message. Drop the key from the
  others, or add it to the one, before merging. A key whose `data` column
  disagrees in dtype or nullability across inputs is refused for the same
  reason: the merged shards are read back under one schema.
- The **CSR-backed** `obsp` encoding (`ObspCsrShard`) is **not** carried, by any
  merge path, and does not trigger the sorted-merge refusal. Nothing outside
  `scx optimize` reads one, so there is nothing to rebase — `compact` and `sort`
  drop it too. It is dropped with a warning; recompute neighbours on the merged
  file. `scx optimize` does **not** help, because it re-emits the graph in this
  same encoding. **The warning does not reach R**: `rscx` installs no Rust `log`
  sink, so `rscx::scx_merge` still drops it silently. That is a gap in the R
  binding, not in the op.

Modality-scoped `obsp` / `varp` are also dropped, with a warning: the format
has no per-modality pairwise reader to round-trip them through.

**`build-csc` and `scx upgrade` carry everything `optimize` does, plus
`adata.raw`.** They used to copy only layers, `obsm`, `uns`, the predicate-index
sections and the deletion vector, dropping `varm`, `obsp`, `varp`, `adata.raw`,
detection bitmaps and the group index — unrecoverably on the in-place forms,
which rename a wholly new file over the target carrying no prior catalog. The
one thing still dropped is a layer's CSC sidecar (rerun `scx build-csc`).

**A detection bitmap does not survive a rewrite that canonicalizes the matrix
under it.** `optimize` and `scx upgrade` both canonicalize, and canonicalizing
drops explicit zeros — which changes which genes a row *stores*, and a detection
bitmap records exactly that. Both ops therefore drop the sidecar, with a
warning, when canonicalizing actually rewrote X, and carry it otherwise. Every
file a current writer produces is already canonical, so in practice the sidecar
survives; a pre-v3 file with explicit zeros is the case that loses it. Rebuild
with `scx sort --bitmap always`.

### Deletion vectors: carried vs. applied

`mark_deleted` is a *logical* delete — the rows stay on disk and readers filter
them out — so every op has to say what it does with them. There are exactly two
correct answers, and the column above records which each op has:

- **Carried.** The op preserves the global obs row space 1:1, so it copies the
  deletion-vector section through unchanged. The output still has every physical
  row, and `query()`, `to_anndata()` and the R/Python readers all still report
  the smaller live count. (`scx info` is the exception: its header line reports
  the *physical* `n_obs`, with the deletions on a separate "Deletion vectors:"
  line.) Nothing is lost, and `compact` can still reclaim the space later.
- **Applied.** The op is building a new row space anyway (`compact`, `sort`,
  `subset`), so the deleted rows are dropped and the output carries no deletion
  vector. `n_obs` shrinks. The deleted cells are gone for good.

Exports to formats with no deletion concept — `scx convert --to h5ad / h5mu /
mtx` — necessarily apply. So do the materialising reads (`pyscx` `to_anndata`,
`rscx` `$x_matrix()` / `$obs()` / `to_seurat()`), which is why they agree with
`query()` on the same file.

Two consequences worth knowing:

- A **carried** deletion is not visible in the header's `n_obs`, which is the
  *physical* row count. `scx info` reports the vector separately ("Deletion
  vectors: … N cells deleted"). Both bindings expose the logical count as the
  default and the physical one under an explicit name: `Experiment.n_obs` /
  `Experiment.n_obs_physical` in `pyscx`, `$n_obs()` / `$n_obs_physical()` in
  `rscx`. `nnz` is **physical** on both — it comes from catalog stats, and
  excluding deleted rows would mean decoding the matrix — so on a file with
  deletions it exceeds the nnz of what a read returns.
- Because `build-csc --in-place` writes a wholly new file with no prior catalog,
  it is **not** rollback-able. It carries deletions rather than applying them
  precisely so that `mark_deleted` → `build-csc` — the documented way to restore
  a sidecar a mutating op dropped — cannot irrecoverably un-delete anything.

`rscx`'s backed and lazy handles (`scx_backed_sparse()`, `scx_lazy_transform()`)
address rows by physical index and have no kept→global translation, so they
**refuse** to open a file that carries deletions rather than silently handing
back deleted rows. Use `$x_matrix()` / `scx_query()`, or `scx compact` first.

### `scx upgrade` declines a file newer than its target

`scx upgrade` rewrites to the newest **unframed** version (v3). It does not add
row-group framing, so it cannot reach v4 — that is `scx optimize
--row-group-rows N`'s job.

Handed a file that is *already* v4 it declines and exits 0, rather than
"upgrading" it downwards. The rewrite is not a no-op on such a file: it decodes
every shard and re-emits it unframed, so a v4 input would silently lose its
row-group framing and the sub-shard random access v4 exists to provide, while
reporting `Upgraded … v4 → v3`. On `--in-place` that is unrecoverable — the
rename lands a wholly new file carrying no prior catalog, so `scx rollback` has
nothing to return to.

```
$ scx upgrade atlas.scx --in-place
File is at format version 4, which is newer than the version `scx upgrade`
targets (v3, unframed). Rewriting it here would strip row-group framing and its
sub-shard random access and `--in-place` is not rollback-able. Nothing to do.
Use `scx optimize --row-group-rows N` to re-frame, or `scx optimize
--row-group-rows 0` if you genuinely need an unframed v3 file for an older
reader.
```

Since `scx convert` and `pyscx.from_anndata` frame by default, a file written by
either is v4 and lands in this branch. There is deliberately no
`--allow-downgrade`: `scx optimize` owns framing in both directions, and
`--row-group-rows 0` is the supported way to ask for unframed output.

### What `scx upgrade` does not carry

The remaining branch — a genuinely pre-v3 file — **canonicalizes** X and every
layer on the way out (indices sorted within each row, duplicate coordinates
summed, explicit zeros dropped). It has to: the output stamps v3, and canonical
row-major CSR is what a v3 header asserts. A pre-v3 source carries no such
guarantee, which is why `rewrite_output_format_version` refuses to promote one
and why `build-csc`, which does *not* canonicalize, clamps its output version to
the source's instead. Canonicalizing can change `nnz`.

Because canonicalizing can change `nnz`, a CSC sidecar built against the old
matrix would no longer be a faithful second view of it — and `write_csc_shard`
stamps `csc_build_generation` from the writer's `data_generation`, so the
freshness guard would bless it and a reader on `prefer_format="csc"` would
silently get different numbers than one on CSR. So the sidecar is **dropped with
a warning** when (and only when) canonicalizing actually rewrote something;
rerun `scx build-csc`. An already-canonical input keeps it.

**Multimodal input is refused.** Nothing in this rewrite is modality-aware — it
would flatten every modality's shards into one global matrix and zero the
modality table — and multimodal only requires v2, so it lands in this branch.
`optimize` and `build-csc` refuse for the same reason.

It copies layers, `obsm`, `varm`, `obsp` (both the COO and CSR-backed forms),
`varp`, `uns`, `adata.raw`, the grouped-sort group index, predicate indexes and
deletion vectors — the same allowlist `build-csc` uses. All of that is sound for
one reason: both ops preserve the global obs row space and the CSR shard
boundaries 1:1, so a section keyed to either stays valid.

The one section it does not carry is a layer's CSC sidecar; rerun
`scx build-csc`. On the in-place form of either op — a rename over the target
carrying no prior catalog — a loss cannot be rolled back, so the op warns before
writing:

```
scx upgrade: the output will not carry layer CSC sidecars (rebuild: scx
build-csc) — rebuild them against the output if you need them.
```

Detection bitmaps are a separate case, because what invalidates them is not the
allowlist but the canonicalization: see the note above. When canonicalizing
actually rewrites X, the sidecar is dropped with its own warning.

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

> **`build-csc` and `optimize` keep the predicate index *and* its Level-1
> pruning.** Both copy `obs_predicate_index` / `var_predicate_index`
> byte-for-byte, and the copy stays *valid* — shard boundaries and row ranges are
> unchanged. Both also re-encode every CSR shard, and `compute_shard_stats` emits
> no per-shard `column_stats`, so each additionally **carries the input's** stats
> onto the output's shards, matched by `row_start`
> (`scx_format_io::carry_csr_shard_column_stats`). All three re-emit one output
> shard per input shard at the same offset, so the input's statistics are already
> exactly right for the output's rows.
>
> Both halves matter, because the index drives two independent pushdown paths:
>
> - **Level-1** (catalog-stats shard pruning) reads the per-shard
>   `column_stats`, not the index section.
> - **Level-2** (row-set pushdown) reads the `PredicateIndex` directly and does
>   **not** consult `column_stats`. It has its own precondition — row-sharded obs
>   metadata — so on a single-section-obs file, where Level-2 is inactive
>   regardless, a query degrades to a full obs scan.
>
> Until v0.14 the stats were dropped: the section survived, Level-1 pruning
> stopped, and the query returned the *right* rows after a full scan — which is
> why it went unnoticed. Measured on a 200-cell / 8-shard file indexed on a
> clustered `cell_type`: `Level 1 eliminated 6/8` before `build-csc`, `0/8`
> after. `optimize` had the identical defect for the identical reason.
>
> Pinned by
> `scx-cli/tests/cli_ops_integration.rs::build_csc_carries_predicate_index_and_pushdown`
> and
> `scx-ops/tests/column_stats_staleness.rs::optimize_carries_the_column_stats_and_the_pruning`,
> plus `scx-ops::optimize::tests::optimize_preserves_level1_shard_pruning` — the
> earlier `optimize` test could not see the loss because its fixture has a single
> CSR shard, and Level-1 pruning is unobservable on one shard.
>
> **Why carrying rather than re-deriving from the index.** An earlier version of
> this fix derived the stats afresh from the carried index bytes, gated on the
> index's highest recorded `shard_id + 1` matching the output CSR shard count.
> That equality is not proof the index is keyed to the CSR partition:
> `modify_metadata` falls back to building the index over **obs**-shard ranges
> when the CSR shards do not tile `[0, n_obs)`, and an obs partition can have the
> same shard *count* with different *boundaries* — CSR `[0,100) [100,200)
> [200,300)` against obs `[0,134) [134,268) [268,400)`. Derivation would attach
> shard 0's obs statistics to CSR rows they do not describe, and Level-1 consumes
> `column_stats` **before** residual evaluation, so a fabricated "value absent"
> prunes a shard holding real matches and the query returns an **incomplete row
> set**. Copying removes the inference entirely.
>
> A file in that state has no stats to carry — `modify_metadata` clears them
> (`clear_all_csr_shard_column_stats`) precisely because its index is obs-keyed —
> so the rewrite output is unprunable rather than wrongly pruned. To get Level-1
> pruning back, rebuild the index with an op that derives stats against the CSR
> partition: `scx sort --by <col> --index-preset …`. Prefer that over
> `scx compact`, which re-runs `codec=auto` and can grow the file; and note
> `scx convert` cannot take an `.scx` input at all.

### Per-shard column stats and the in-place ops

The predicate index is not the only thing an obs rewrite can invalidate. The
catalog also records a per-shard `ColumnStat` — a numeric `MinMax` or a
categorical `CategoryBitset` — for every indexed obs column, and **Level-1
pruning reads those directly**. For the numeric arm it never consults the index
section at all (`scx-engine/src/pushdown.rs` gates only on
`!stats.column_stats.is_empty()`). So the two must be maintained together:

The `CategoryBitset` is positional over the column's sorted category list, so a
predicate value has to be resolved to an ordinal before it can be checked — and
the only value that can be is a **string**, looked up in the global category
dictionary the index carries. A numeric literal is a *value*, never an ordinal:
it is left unresolved and prunes nothing, which is why `batch == 3` on an
integer-valued categorical is served by the numeric `MinMax` arm instead.

- **Losing** the stats costs pruning; rows are still correct. `build-csc`,
  `optimize` and `upgrade` all used to lose them on re-emit and now carry them
  from the input instead.
- **Keeping a stale one** returns the wrong rows. A shard whose recorded `max`
  predates a rescaled column is excluded from a query the new values satisfy, and
  the result comes back short with no error and no warning.

The three in-place ops therefore clear what they invalidate:

| op | scope of the clear |
|---|---|
| `modify_metadata` with `obs=` | **all** columns — obs is replaced wholesale and the op cannot tell an added column from a rewritten one. Re-derived instead of cleared whenever an index is built — which now includes the carry-forward, so an indexed file keeps its pruning — **and** that build can be mapped onto the CSR shards (see below). |
| `obs-import` / `doublet-import` | only the columns the import writes. It joins by key and never reorders rows, so an untouched column's stats stay true. |
| `cellbender-import` | the same, over `status_column` / row annotations / `row_sum_column`. |

The scoped clears are **not** conditional on whether an `ObsPredicateIndex` is
still present: a file can carry stats with no index — a `modify_metadata` obs
replace that could carry nothing forward leaves exactly that — and gating on the
index would let those bounds survive a second rewrite.

`append` needs none of this — it adds rows and never edits existing ones, so the
pre-append shards' stats remain true and the appended shards carry none.

**On an indexed file, `modify_metadata(obs=…)` no longer clears at all.** It
carries the file's own index forward — rebuilds it over the same columns — and
re-deriving the index re-derives the stats, so Level-1 pruning survives an
ordinary obs edit. The clear below is what happens when that carry cannot
complete, and to a file that had no index to carry.

To get Level-1 pruning back after a clear, rebuild the index with an op that
derives stats: `pyscx.modify_metadata(f, obs=…, index_obs=[…])`, or a copy-out
`scx sort` / `scx compact` with `--index-obs` / `--index-preset`.

**Neither a carry nor an explicit rebuild always restores pruning**, and the
warning `modify_metadata` emits says which case you are in. The stats are derived
only when the index being built is keyed to the CSR shards, which needs those
shards to tile `[0, n_obs)`. On a file whose CSR shards under-cover the obs axis,
the index is built over the *obs-shard* ranges instead — a different shard space —
and the derive is skipped. The file then has a freshly written `ObsPredicateIndex`
and no column stats, and repeating the in-place rebuild lands in the same branch.
The fix there is a copy-out `scx sort` / `scx compact` with `--index-obs`, which
re-shards the matrix so the two spaces line up. A build that lands no indexable
column (missing, or an unsupported dtype such as `Boolean`) likewise writes
nothing to derive from — including a carry whose columns the new frame dropped,
which is reported as a warning naming each one.

> **Files rewritten in place before this shipped are not repaired by upgrading.**
> The clears stop *new* files being poisoned; they do not touch a file that
> already carries stats describing obs values that were replaced. If an indexed
> file went through `modify_metadata(obs=…)` or an `overwrite` import on an
> earlier build and a query since came back suspiciously short, re-derive its
> stats with any of the rebuild paths above — `scx info` will not flag it, because
> a stale bound is indistinguishable from a live one.

**Adding an obs column: `obs_import` is still the cheaper route.** Both keep an
unrelated `cell_type` index pruning, but they get there differently.
`obs_import` joins by key, knows exactly which columns it writes, and clears
nothing else — O(new columns). `modify_metadata(obs=…)` replaces the whole frame
and cannot tell an add from a rewrite, so it re-earns the index by rebuilding it,
which is O(n_obs) over the indexed columns on top of the obs write it was already
doing. Prefer `obs_import` when the shape of the edit allows it.

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
| **Predicate indexes** | O(n_obs)/O(n_vars) when the axis is replaced and indexed | The old section describes values that are gone, so it is rebuilt over the columns the file already indexed — the replaced axis keeps its pushdown rather than silently losing it. `--index-obs` / `--index-var` / `--index-preset` name a different set instead. Untouched (O(1)) when only `uns`/`obsm`/`varm` change, or when the axis had no index. |

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

Because the join is the whole risk surface, `--dry-run` runs the join and every
validation that does not require decoding a non-key obs column (see the memory
section below for that one gap), prints the match counts and the obs rewrite
path, and writes nothing. Use it before importing onto a large file.

Emitted alongside the layer: `obs` gets `cellbender_status`,
`cellbender_cell_probability`, `cellbender_cell_size`,
`cellbender_droplet_efficiency`, `cellbender_background_fraction`,
`cellbender_analyzed` and `cellbender_total_counts`; `var` gets
`cellbender_ambient_expression` and `cellbender_analyzed`; `uns["cellbender"]`
records the run metadata and the full join report; and a provenance entry is
appended with the source file's BLAKE3 in `input_checksums`.

Known interaction: `scx subset` currently drops layers, so subset before
importing rather than after.

Memory behaves as it does for `obs-import` — see
[§ Memory: bounded on the target, resident on the source](#memory-bounded-on-the-target-resident-on-the-source)
below. The CellBender-specific addition is the source: `read_cellbender_h5` is
not streaming and does not honour `--memory-budget`, because the barcode-keyed
join reorders rows arbitrarily and the matrix has to be resident. Bounded in
practice by `total_droplets_included` (default 25k, heuristic cap 70k), but a
full all-droplet output on a very wide feature axis still costs several GB.

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

### Memory: bounded on the target, resident on the source

Rewriting obs in full is a *bytes-on-disk* cost, not a memory one. On a file
whose obs is sharded — anything `from_anndata` wrote above `shard_target_rows`,
and anything `merge` or `append` produced — neither import holds the obs table:
the schema comes from the Arrow IPC footer, the join reads only the key
column(s), and the rewrite runs one obs shard at a time. Peak is one obs shard
plus the join arrays (one row index and one key string per target cell).

**The input's obs shard boundaries are preserved**, not re-derived from
`header.shard_target_rows` — one output shard per input shard, as with `compact`
and `optimize`. This is a change: an import used to normalise them. It shows up
only on a target whose shards do *not* already match `shard_target_rows`, which
in practice means one grown by `append`. Nothing downstream depends on the
boundaries (the obs predicate index keys on CSR shards, not obs-metadata
shards), so this is a layout note rather than a compatibility one.

These paths are still unbounded:

| Path | Cost | What to do |
|---|---|---|
| A legacy single-section `obs` target | the whole obs table | Run [`scx optimize`](#optimize) first to migrate obs to the sharded layout. The import warns and reports `obs_streamed = false`. It also *writes* obs back as shards, so a second import on the same file streams — but the first one has already paid the cost, which is why `optimize` is the answer on a file big enough to care. |
| A failed join's key diagnosis | the whole obs table, plus a pass per column | Only reached when the import is already failing. |
| `obsm` embeddings supplied by the caller | one `n_obs`-row section | Reachable: `cellbender-import --latent-embedding` / `cellbender_import(latent_embedding=True)` builds one. Leave it off unless you want the latents. |
| The source table | resident in full | Inherent: the join is by key, so the source's row order is its own business. The source is the small side — that is the premise the feature rests on. |

The import summary reports `obs_streamed`, and all three CLI subcommands print
it, so which target-side path ran is answerable after the fact rather than
inferred from the file's layout. `cellbender-import` additionally reads `var`
whole — that is the gene axis, so it does not scale with the atlas.

The shape, the join and the obs shard cover are all validated before the first
byte is written, so `--dry-run` reaches the same verdict the real import would
on every one of them, and a rejected import leaves the file byte-identical.

The preflight reads only the **key** column(s), though — Arrow IPC projection
skips the rest. A non-key obs column that fails to decode, or a per-shard schema
that disagrees with the first shard's, is therefore not seen until the write
pass, which aborts with obs shards already appended at EOF. The catalog is only
swapped at commit, so the file still reads as it did and `scx compact` reclaims
the orphans — but `--dry-run` cannot promise that case away. Such a file is
already unreadable through a normal obs read (its shards cannot be concatenated),
so the import is not what breaks it.

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
