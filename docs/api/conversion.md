# Conversion and round-trip fidelity

> Part of the [SCX API reference](README.md). For conversion how-tos, see
[docs/scanpy/conversion.md](../scanpy/conversion.md) and
[docs/migrating-from-h5ad.md](../migrating-from-h5ad.md).

## Conversion warnings (`ConvertWarning`)

Conversion paths surface structured warnings via a `WarningSink`
(`scx-convert/src/warnings.rs`). The CLI prints a per-category summary
at the end of `scx convert`; `pyscx` aggregates them and emits one
Python `UserWarning` per category. The full count + sample is also
recorded under `ProvenanceEntry.params_json.warnings`.

| Variant | Emitted by | Meaning |
| --- | --- | --- |
| `InferredEncoding { path, inferred }` | h5ad layout detection (`detect_matrix_format_at`, `open_x_streaming`) | h5ad `encoding-type` was missing or ambiguous; layout was inferred from group children or dataset shape. |
| `SkippedUnsKey { key, reason }` | `read_uns` / `read_uns_entry` (ingest); `write_uns_value` (export) | **Ingest:** a `uns` entry was unrepresentable; skipped under default `strict_uns=false`, an error on the first occurrence under `strict_uns=true`. **Export:** the key is not usable as an HDF5 member name (empty, `.`, `..`, or containing `/`, which HDF5 would resolve as a *path*), so the entry is dropped. `strict_uns` is an ingest option and does not govern the export case. |
| `UnsupportedUnsDataframeColumn { key, column, reason }` | `read_uns_dataframe_group` | **Ingest.** One column of a `uns` pandas DataFrame had no lossless `uns` encoding (anndata's `nullable-integer` / `nullable-boolean` / `nullable-string-array`, or an unrecognised column encoding) and was left out of the reconstructed frame. The index, the column order and every other column are intact. Under `strict_uns=true` this is an error instead. |
| `UnsExportedAsRawEnvelope { key, reason }` | `try_write_uns_dataframe` | **Export.** A `uns` pandas DataFrame had no faithful h5ad dataframe spelling and was written as a raw `__scx_type__` envelope subgroup. No data is lost; anndata reads it back as a nested dict rather than a DataFrame. |
| `FlattenedUnsSparse { key, format }` | `read_uns_entry` | A `uns` scipy-sparse matrix (`encoding-type` = `"csr_matrix"` / `"csc_matrix"` / `"coo_matrix"`) was preserved as a nested dict of `data` / `indices` / `indptr` arrays rather than reconstructed as a sparse matrix — the sparse type tag is not restored on read. Data is not dropped. |
| `SkippedColumn { group, name, reason }` | `read_dataframe_group` | An obs/var column could not be read (unsupported encoding-type or read error); skipped. The column is absent from output. |
| `SkippedObsm { name, reason }` | `read_obsm_at` | An obsm/varm embedding could not be read; skipped. |
| `LayerSkipped { name, reason }` | `read_layers_at` / streaming layer reader | A layer was skipped (open failed, shape mismatch with `/X`, or width exceeds `u32::MAX`). |
| `DenseSparsified { path, density }` | Dense h5ad streaming reader | Dense `/X` slab was sparsified during streaming. Reports density to help users decide whether dense storage is worth keeping. |
| `DuplicateCoordinatesMerged { count, policy }` | CSC h5ad streaming reader | CSC input contained duplicate `(row, col)` coordinates; values were summed (scipy `sum_duplicates` semantics). |
| `ModalityTypeInferred { name, modality_type }` | h5mu streaming reader | Modality name → `ModalityType` was inferred by name; override via `--modality-types NAME:TYPE` / `modality_types={...}`. |
| `MissingPresetIndexColumn { column }` | Predicate-index builder | A preset (`cellxgene` / `perturbseq` / `training`) referenced an obs/var column not present in the source; preset misses warn, conversion continues. |
| `PresetNoColumnsMatched { preset, axis, missing }` | Predicate-index builder | `--index-preset <preset>` matched **none** of the source obs/var columns. The user almost certainly picked the wrong preset for the input format; the per-column warnings are collapsed into a single actionable diagnosis. |
| `UnsupportedIndexColumn { column, reason }` | Predicate-index builder | A user-forced (`--index-obs` / `--index-var`) or preset column has an unsupported dtype; forced columns hard-error, preset columns warn and skip. |
| `PredicateIndexSkippedMultimodal` | Predicate-index builder | Predicate indexes are unimodal-only on the read side today; emitted (and indexes skipped) when conversion input is multimodal. |
| `CscSkippedStreamingMultimodal { modalities }` | Streaming multimodal writer | CSC sidecar emission was skipped because streaming multimodal conversion does not support CSC output. Run `scx build-csc` on the output to add the sidecar. |
| `BitmapSkipped { reason }` | Detection bitmap auto policy | `--bitmap auto` rejected emission (e.g. `n_vars > 1_000_000`, estimated bitmap size > 15% of encoded CSR, dense X). |
| `DroppedObsp { name, reason }` | h5ad ingest (obsp/varp routing) and `scx merge` (multimodal) | A pairwise `obsp`/`varp` matrix could not be preserved: on ingest, a CSC or otherwise unsupported pairwise layout is dropped (CSR is stored directly; a **dense** pairwise matrix is preserved as nonzero COO, not dropped); on `scx merge`, a **modality-scoped** `obsp`/`varp` graph is dropped (the format has no per-modality pairwise reader), as is a **CSR-backed** `obsp` in any scope. A file-scope COO `obsp` is carried and rebased into the merged obs axis since Phase 5b — the older "axis semantics don't compose" was true of merge before that. Default-dropped with a warning. |
| `MappingPeakFootprintHigh { mapping, estimated_bytes, budget_bytes }` | `pyscx.from_anndata` | A single mapping's estimated in-memory footprint exceeds `memory_budget`. |
| `EagerAssemblyMemoryHigh { estimated_bytes, budget_bytes, nnz, value_bytes, index_bytes, dense_shape, dense_bytes }` | `Experiment.to_anndata` | Estimated eager assembly footprint exceeds `memory_budget` (default 8 GiB). Warn-only, does not block. `estimated_bytes` covers **assembly** — `X`, `adata.raw` when the read includes it, and obs/var — and nothing that reads the matrix afterwards, so it is a floor on process RSS, not a job size; the message says so. Every figure the message prints is derived from these rather than assumed: `value_bytes` is the caller's `data_dtype` width and `index_bytes` is scipy's choice from `max(nnz, n_rows)`, so an `f32` CSR reads as 8 or 12 B/nnz and a `float64` one as 12 or 16. `nnz` is `None` when the call assembles no host `X` (the `to_gpu_anndata` device path), and `index_bytes` is `None` where the width genuinely is not knowable — a dense request has no index array, and above `i32::MAX` physical nonzeros on a file with deletion vectors scipy decides from the live count the catalog cannot supply. `dense_shape` is the `(rows, columns)` a dense request will allocate, which under a `var_names` projection is the *selected* column count. `dense_bytes` is `n_obs × n_vars × value_width` and is `Some` only when this call assembles a host CSR `X` **and** dense would be smaller than that sparse `X` — compared against the `X` term alone, since a large `adata.raw` is assembled either way and would otherwise make a larger dense `X` look like a saving. |
| `Hdf5NotThreadsafe` | Parallel streaming reader fallback | libhdf5 was not built thread-safe; parallel streaming fell back to the sequential coordinator. |
| `DroppedRaw { raw_n_vars }` | `Experiment.to_anndata` | The file carries an `adata.raw` matrix but the current reconstruction mode (obs-filtered query, backed mode, or deletion-vectors active) cannot reproduce raw's obs-axis filtering, so raw is omitted. The on-disk raw sections are preserved. Pass `to_anndata(raw=False)` to opt out of raw entirely — no rebuild and no notice. |
| `DroppedRawOnWrite { raw_n_vars, reason }` | `pyscx.from_anndata` with SCX-backed / lazy `X`; `from_h5ad` / `scx convert` with `--sort-by` / `--group-by` | The **source** carries raw that the file being written will not. Distinct from `DroppedRaw`: its "on-disk raw sections are preserved" reassurance is true of the source and says nothing about the output, where raw is gone for good. `reason` is supplied **per call site**, because the doors differ — the SCX → SCX rewrite loses raw because the in-memory AnnData does not hold it (convert from the h5ad instead), while reorder-on-convert loses it because raw streams unpermuted (convert without the reorder instead). A single baked-in remedy would be wrong on one of them. |
| `DroppedRawVarm { keys }` | h5ad ingest and `pyscx.from_anndata` | `adata.raw.varm` is not representable — the raw section family stores `raw/X` and `raw/var` only — so raw's own var-axis mappings are dropped. `raw.X` / `raw.var` are unaffected. |

## Round-trip fidelity

What survives an `h5ad → scx → h5ad` (or `→ AnnData`) conversion, and what is
lossy or dropped. **Status** is one of: **preserved** (faithful round-trip),
**lossy** (round-trips with a documented value/structure change), or **dropped**
(not written; surfaced as a warning). The **Warning** column names the
[`ConvertWarning`](#conversion-warnings-convertwarning) that fires; "—" means no
warning (the behaviour is by-design and documented here). Nothing in this table
is lost *silently*.

| Feature | Status | Warning | Notes |
| --- | --- | --- | --- |
| `X` counts / values | lossy | — | On-disk `u8`–`u32` ↔ in-memory `f32`; `float64` `X` is **downcast to `f32`** to match scipy CSR zero-copy. Integer counts are bit-exact. |
| obs/var columns (numeric, string, bool, nullable, categorical) | preserved | — | Nullable int/string/bool and categoricals round-trip via anndata's nullable-group / categorical encodings. |
| obs/var **ordered** categoricals | preserved | — | The `ordered` bit + category order round-trip both `h5ad → scx → h5ad` and `pyscx.open(...).to_anndata()` (carried in Arrow field metadata, re-applied to the reconstructed pandas factor). Order is the **declared** one, independent of the order the values happen to appear in the data. |
| obs/var declared-but-unused categories | preserved (dropped when a row filter is active) | — | A category no row uses is part of the declared factor and survives an ordinary export. It is dropped only when the export applies a row filter — a deletion vector, or `obs_mask=` / `min_counts=` on `pyscx.to_h5ad` — matching pandas' `remove_unused_categories()` on a subset. |
| obs/var **MultiIndex** | lossy | — | Only the single pandas `_index` is preserved; additional index levels are not carried. |
| unreadable obs/var column | dropped | `SkippedColumn` | Unsupported encoding-type or read error; column absent from output. |
| obsm / varm embeddings | preserved | `SkippedObsm` (on failure) | Dense embeddings round-trip; an unreadable embedding is dropped with the warning. |
| layers | preserved | `LayerSkipped` (on failure) | CSR layers round-trip; a layer whose shape disagrees with `/X` (or whose width exceeds `u32::MAX`) is dropped with the warning. |
| **dense** `obsp` / `varp` | lossy | — | Ingested as nonzero **float32 COO** (only nonzeros stored); both `to_anndata` and `to_h5ad` re-emit it as a **sparse** matrix (a dense input becomes sparse; values identical). |
| CSR `obsp` / `varp` | lossy | — | Round-trips `h5ad → scx → h5ad` (and via `to_anndata`) as **float32 CSR**; values downcast to `f32`. Under deletion vectors, `obsp` is filtered on both axes; `varp` (var axis) is never obs-deleted. |
| CSC / unsupported `obsp` / `varp` | dropped | `DroppedObsp` | CSC and other non-CSR/non-dense pairwise layouts are dropped on ingest. |
| `adata.raw` | preserved² | `DroppedRaw` / `DroppedRawOnWrite` (some modes) | Round-trips raw counts bit-exact with the wider var axis through **both** write doors — `h5ad → scx → h5ad` and in-memory `pyscx.from_anndata` / `pyscx.write`. ² Dropped, with a warning, under obs-filtered `to_anndata`, backed mode, and deletion-vector-active files, on the SCX-backed / lazy-`X` rewrite, and under reorder-on-convert (`--sort-by` / `--group-by`). `to_anndata(raw=False)` opts out of raw and its warning. See [`adata.raw`](#adataraw). |
| `adata.raw.varm` | dropped | `DroppedRawVarm` | Raw's own var-axis mappings have no section in the raw family (`raw/X` + `raw/var` only). `raw.X` and `raw.var` are unaffected. |
| `uns` scalars / 1-D & 2-D numeric arrays / nested dicts | preserved | — | Round-trip through the `uns` JSON representation. |
| `uns` pandas **DataFrame** | preserved | `UnsupportedUnsDataframeColumn` / `UnsExportedAsRawEnvelope` | Round-trips as a `pd.DataFrame` through both write doors, index name, column order and per-column dtypes (ordered categoricals included) intact — see [`uns` serialization](python-experiment.md#uns-serialization). Two exceptions, each warned: on **ingest** a column in one of anndata's nullable encodings is left out (an error under `strict_uns=true`); on **export** a frame h5ad cannot spell is demoted whole to a raw envelope subgroup, losing no data but arriving at anndata as a dict. |
| `uns` scipy-sparse matrix | lossy | `FlattenedUnsSparse` | Preserved as a nested dict of `data` / `indices` / `indptr` arrays; **not** reconstructed as a sparse matrix (the sparse type tag is not restored). Data survives. |
| `uns` pickled / unrepresentable entry | dropped | `SkippedUnsKey` | Skipped under default `strict_uns=false`; `strict_uns=true` errors on the first occurrence. |
| `uns` nested more than 60 levels deep | rejected / truncated | `SkippedUnsKey` (h5ad ingest) | See [`uns` nesting depth](#uns-nesting-depth). Writing raises `ValueError` naming the key path; an over-deep h5ad `/uns` group chain is truncated at the limit with the warning (or errors under `strict_uns=true`). |

### `uns` nesting depth

`uns` container nesting — dicts, lists, tuples, and nested h5ad `/uns` groups —
is capped at **60 levels**. Real metadata is nowhere near this: scanpy's deepest
standard structure, `rank_genes_groups`, is 3 deep.

The cap exists because the walk is recursive on every path (Python→JSON,
h5ad→JSON, JSON→Python, JSON→h5ad), and exceeding the stack **aborts the
process** rather than raising — a `SIGSEGV` no `try`/`except` can catch. Cycle
detection does not help: it works by object identity, and a merely-deep tree
contains no repeated object.

The specific value is derived, not chosen. `serde_json`'s serializer has no
depth limit but its parser stops at 127 levels, so before the cap the writer
could emit an `uns` section that no reader — including SCX's own — could ever
parse back. Under `uns_format="tagged"` a tuple is stored as
`{"__scx_type__": "tuple", "data": [...]}`, two JSON levels per Python
container, so 60 containers is the worst case that still fits.

Writers (`from_anndata`, `set_uns`, `modify_metadata`, h5ad ingest) enforce 60.
Readers stop at 127 instead, so a file written before the cap existed still
opens if its `uns` is parseable; one written deeper than 127 raises a message
saying so rather than `serde_json`'s bare `recursion limit exceeded`.

Underneath both sits a storage gate at 127: the section writer itself refuses
an `uns` tree deeper than the reader will parse, whatever built it. That is
what makes "an SCX file always has a readable `uns`" true for paths with no
Python or h5ad in them — `scx merge`, `scx subset`, and any Rust caller
handing the writer a tree it assembled itself.

## `adata.raw`

`adata.raw` (pre-normalization counts on its own, usually wider, var axis) is
preserved end-to-end. `pyscx.from_h5ad` ingests the h5ad `/raw` group into a
dedicated raw section family (CSR shards on `X`'s obs axis + a `raw/var` Arrow IPC
section; see [docs/format.md § raw section family](../format.md#adataraw-raw-section-family)).
On the streaming convert path (`stream=True`, the default) `raw/X` is read and
written shard-by-shard through the same coordinator as `/X`, so peak RSS stays
bounded; the materializing path (`stream=False`) reads it eagerly.
The in-memory `pyscx.from_anndata(adata)` / `pyscx.write(adata, path)` path writes the
same section family from `adata.raw.X` + `adata.raw.var`, so both doors into SCX
preserve raw identically. It is materialized eagerly (the AnnData is already
resident), unlike the h5ad streaming ingest. Raw keeps its own column count: the raw
shards' `index_dtype`, minor-axis extent, and codec are all resolved against
`raw_n_vars`, independent of `X`.

`pyscx.open(...).to_anndata()` reconstructs `adata.raw` (an AnnData with raw `X` +
`var`), and `pyscx.to_h5ad` re-emits `/raw/X` + `/raw/var`, so
`h5ad → scx → h5ad` round-trips raw with integer counts bit-exact and the wider var
axis intact. Raw is **dropped with a `DroppedRaw` warning** under obs-filtered
`to_anndata`, backed mode, and deletion-vector-active files (those modes do not yet
re-filter raw's obs axis).

`to_anndata(raw=False)` opts out: raw is not rebuilt on the paths that would rebuild
it, and the `DroppedRaw` notice is not emitted on the three that would drop it. Use it
when raw is irrelevant to the workload — otherwise the notice fires on every backed
open of a raw-bearing file. `raw=True` is the default and is unchanged.

A **gene projection does not narrow raw**: anndata hands `.raw` only the obs index when
slicing, so `to_anndata(var_names=[...])` returns raw on its full gene axis, exactly as
a plain read does. That makes raw the dominant cost of a projected read of a
raw-bearing file (the projected `X` is small and raw is not), which is what `raw=False`
is for — measured on a 20-shard fixture, a three-gene read peaked at 61.6 MB with raw,
20.5 MB once raw stopped being copied along the way, and 3.1 MB with `raw=False`.

Three write paths still cannot carry raw and say so rather than dropping it in silence.
The first two warn with `DroppedRawOnWrite`, whose `reason` is supplied per call site —
they lose raw for different causes, so a single baked-in remedy would misdirect one of
them:

- **SCX-backed / lazy `X` rewrite.** `pyscx.from_anndata` writes the sections the
  in-memory AnnData holds, and backed reconstruction sets `.raw` to `None`, so a
  `pyscx.open(f).to_anndata(backed=True)` → `from_anndata` round-trip loses raw. Keyed
  off the *source* file rather than the object — `DroppedRaw`'s "the on-disk raw
  sections are preserved" would be misleading here, since the file being written has
  none. Remedy: convert from the h5ad, or from an in-memory AnnData whose `.raw` is set.
- **Reorder-on-convert** (`from_h5ad(..., sort_by=…)` / `scx convert --sort-by` /
  `--group-by`). The permutation is applied to `X` / `obs` / `obsm` / `layers` while raw
  is streamed in source order, so carrying it would leave raw's rows attached to the
  wrong cells. Remedy: convert without the reorder. (`from_anndata(sort_by=…)` on an
  in-memory AnnData is rejected outright, so it cannot reach this.)
- **`merge` / `compact` / `sort` / `append` / `preprocess`**, which need raw's obs axis
  filtered in lockstep with `X`. The first four warn; `append` and `preprocess` refuse.

`adata.raw.varm` is not stored — the raw family holds `raw/X` and `raw/var` only — and
is dropped with a `DroppedRawVarm` warning on both the h5ad ingest and the in-memory
write. A raw whose `X` row count disagrees with `adata.n_obs` is **rejected**, not
written: `adata.raw.shape` reports the parent's `n_obs` rather than
`adata.raw.X.shape[0]`, so a misaligned raw looks correct from Python.

## `pyscx.from_anndata` — backed and lazy `X`

`pyscx.from_anndata(adata, out)` accepts `adata.X` in three shapes:

1. **scipy / numpy** — the existing in-memory path; extracts CSR
   arrays, partitions into shards, writes.
2. **`ScxBackedSparseDataset`** (returned by
   `pyscx.open(p).to_anndata(backed=True)`) — streams the source SCX
   file shard-by-shard without materialising `X`.
3. **`ScxLazyTransformedDataset`** (after
   `pyscx.accel.normalize_total` / `pyscx.accel.log1p` / row scaling
   are stacked on a backed source) — same streaming write, with
   transforms applied per shard.

For shape 2 the writer chooses between two modes:

- **Byte-passthrough**: when the source and target shard layouts
  agree (matching `shard_size`, matching `codec`, source built from a
  single modality with no row deletions and no column projection),
  pre-encoded CSR shards are copied verbatim via
  `ScxWriter::copy_section_verbatim`. Provenance records
  `passthrough=true`.
- **Decode + encode**: any precondition mismatch (e.g.
  `shard_size=` override, `codec=` override, a deletion vector marked
  via `mark_deleted`, a `[:, gene_subset]` column projection) falls
  back to iterating the wrapper's user-visible shard boundaries,
  materialising each shard via `wrapper[start:end]` (which already
  applies deletions / projection), and re-encoding through
  `encode_one_shard` + `write_preencoded_shard`. The output `n_obs`
  / `n_vars` reflect the user-visible shape.

### Which codec a shape-2 (backed) decode + encode rewrite uses

Shape 3 (lazy) is different and is covered below — do not read this subsection
as the rule for both.

With an explicit `codec=`, that codec. Without one, the codec is resolved
against the **source's shard headers**, never against its file header:

- every source CSR shard shares one codec → reuse it, so an explicit
  `codec="scx1"` / `"pcodec"` / `"none"` source round-trips unchanged;
- they differ → the per-shard adaptive heuristic
  (`select_codec_for_modality`) chooses again for each output shard.

The file header's `codec_id` is only a *default* and each shard header
overrides it (see *Codec Selection* above, and `docs/codec.md` §1), so it can
describe no shard at all — a `codec="auto"` write leaves whichever codec its
first shard picked, and files written by older versions of the streaming
h5ad → SCX converter carry a `0` (`none`) placeholder over compressed shards.
Inheriting it wrote every shard raw: on a 966,728 × 6,143 source
(nnz 2.67 × 10⁹) with such a header, dropping 537 rows turned 2.19 GB into
10.85 GB. Any row deletion disables byte-passthrough, so this reached every
subset rewrite of an affected file.

For shape 3 the rewrite is always decode + encode; transforms are
applied per shard inside `wrapper[start:end]`. **Its codec rule differs from
shape 2's:** an explicit `codec=`, otherwise per-shard adaptive selection —
never the source's codec, from its shard headers or anywhere else. The
transforms have already rewritten the values (`normalize_total` / `log1p` turn
integer counts into f32), so the codec that suited the source's integers does
not suit the output's floats; a uniform `scx1` or `none` source therefore does
**not** round-trip its codec through a transformed write. The same
`csc="always"` same-pass sidecar build is honoured (matches `from_h5ad`).
Any source CSC sidecar that would be invalidated by the rewrite is
dropped with a `UserWarning` unless `csc="always"` opts into a
fresh build.

Each rewrite emits a `from_anndata` provenance entry with the
following `params_json` fields:

| Field | Type | Description |
| --- | --- | --- |
| `x_source` | `"scipy" \| "backed" \| "lazy"` | source of `X` |
| `passthrough` | `bool` | `true` only on the byte-passthrough fast path |
| `source_path` | `str \| null` | source SCX path (when the wrapper carried one) |
| `lazy_transforms` | `list[{"name","params"}]` | only for `x_source="lazy"`; per-row factor / row-sum vectors are summarised by length |
| `csc_dropped` | `bool` | source had a CSC sidecar that was dropped |

Read the chain via `Experiment.provenance()` — each entry is a
dict with `params_json` as a raw JSON string (parse with
`json.loads`).

The full transform list is also surfaced on the wrapper itself via
`ScxLazyTransformedDataset.transforms_repr() → list[{"name","params"}]`,
which the writer uses to build the `lazy_transforms` payload.

### Zero rows and zero columns

`pyscx.from_anndata` accepts an AnnData with `n_obs == 0` (an empty QC or
guide filter, `adata[mask]` with an all-`False` mask, `pyscx.accel.subset_obs`
keeping nothing) or `n_vars == 0`, on every `X` shape above. Before 0.17 both
raised `RuntimeError: Arrow IPC contains no batches` — pyarrow's `write_table`
emits zero IPC batches for a 0-row frame; the format, the Rust writer and every
reader already handled zero shards (`scx subset` with a predicate matching
nothing has always written a valid 0-row file). What a 0-row file holds:

- **`X`'s shape and the full schema.** The header carries `n_obs = 0` and
  `n_vars`; there are **no** CSR shards (the format forbids framed zero-row
  shards — "emit no shard at all instead", [format.md § Block index](../format.md#block-index)).
  `obs` and `var` are single 0-row sections with every column, dtype and
  declared category — `pd.Categorical([], categories=["A", "B"], ordered=True)`
  reads back with both categories and `ordered=True`, because the 0-row frame
  crosses the pandas → Arrow boundary as a real 0-row batch
  (`pyarrow.RecordBatch.from_pandas`), not as an empty batch rebuilt from the
  schema. `obsm` / `varm` (as `(0, k)` / `(n_vars, k)`), `obsp` / `varp` and
  `uns` survive.
- **Empty `object` columns, and the obs index, are stored as string.** With no
  values to look at pyarrow types them Arrow `null`, and a categorical that an
  anndata subset pruned to zero categories as `dictionary<null>` — types no
  populated frame produces. They are stored as `string` / `dictionary<string>`
  so the 0-row file's schema is the one its populated sibling has: `append`'s
  obs-schema check, `merge`'s column unification and a forced `index_obs=`
  all compare against it. Only 0-row frames are touched; a populated all-`None`
  column still round-trips as `null` → `object`.
- **`layers` and `adata.raw` are dropped, with a `UserWarning`** naming them.
  Both exist on disk only as their CSR shards (`has_raw` is derived from shard
  presence), and a 0-row file has none, so not even the layer's name can be
  recorded. `to_anndata()` of a 0-row file has `layers == {}` and `raw is None`.
- **No CSC sidecar and no obs predicate index**, whatever `csc=` / `index_obs=`
  say, and no warning about either: a sidecar over an empty matrix indexes
  nothing, and a 0-row obs has nothing to index (`filter_obs` scans the empty
  obs). `index_var=` on a 0-row file still indexes `var`, which has rows. The
  same rule holds for `n_vars == 0` (`csc` skipped; `index_var` builds nothing)
  and for every op that rebuilds them (`compact`, `merge`, `sort`, `build-csc`
  — see [operations.md § Empty inputs and outputs](../operations.md#empty-inputs-and-outputs)).
  The SCX-backed / lazy `X` rewrite (shapes 2 and 3) applies the same
  layers / `raw` / CSC policy, but it has never built a predicate index for any
  row count: passing `index_obs=` / `index_var=` / `index_preset=` there raises
  a `UserWarning` and writes none — build it afterwards with
  `pyscx.compact(out, out2, index_obs=[...])`.

Reading one back: `to_anndata()` is `(0, n_vars)` with the obs / var frames
above; `to_anndata(backed=True).X` is a handle with `n_shards == 0` whose
`to_memory()` / `sum(axis=0)` / column selectors all work (`stored_dtype`
reports `float32`, the decode dtype, since there is no shard to read an
encoding from, while `Experiment.value_encoding` reports `"n/a"` — both
documented above); `read_obs()` / `read_var()`, `query().collect()`,
`to_h5ad` (streaming and eager) and `scx info` all answer for the empty
file. A `(n_obs, 0)` file reads back the same way with the axes swapped.

Not covered: `pyscx.from_mudata` still rejects `n_obs == 0` (it derives the
global cell count from the outer obs), and `adata.X is None` is not accepted
on any row count.

## See also

- [Memory budgets](memory-budgets.md) and
  [predicate indexes and detection bitmaps](indexes.md) — the other
  conversion-time options.
- [Module-level functions](python-functions.md) — `from_anndata`,
  `from_h5ad`, `to_h5ad`, and friends.
