# Multimodal SCX (CITE-seq / 10x Multiome / MuData)

SCX v2 carries multiple modalities (RNA + ADT + ATAC + …) in a single
file. Cells are global; each modality has its own variable axis,
shards, and metadata. This guide covers when to reach for the
multimodal API, how the format models modalities, and how to round-trip
through the major scverse / Bioconductor / Seurat objects.

> **TL;DR**: write CITE-seq / 10x Multiome via `pyscx.from_mudata` (or
> `scx convert --from h5mu`), open via `pyscx.open(path).to_mudata()`,
> and train via `pyscx.MultimodalTrainingDataset(path,
> modalities=["rna", "adt"])`. Single-modality files keep using the
> existing `from_anndata` / `TrainingDataset` surface unchanged.

---

## 1. When to use multimodal SCX

Use multimodal when:

- **CITE-seq**: RNA + protein/ADT counts measured on the same cells.
- **10x Multiome**: RNA + ATAC peak counts on the same cells.
- **TEA-seq**: RNA + protein + ATAC.
- **Spatial + transcriptomics**: an RNA modality plus a Spatial modality
  carrying coordinates as `obsm["spatial"]`.
- **Multi-assay archives** that you want compressed and queryable as
  one file rather than as a directory of single-modality SCX files.

Stick with single-modality SCX (the v1-shape file with the legacy
`from_anndata` + `TrainingDataset` paths) when:

- Every cell has exactly one feature space (the standard scRNA-seq
  case).
- You need to interop with v1-only consumers (older `cell-load-scx` /
  `state-scx` wheel pins). v2 is one-way: v2 readers transparently
  open v1 files, but v1 readers reject v2 files with `UnsupportedFormatVersion`.

A v2 file that registers no modality table behaves exactly like a v1
file on the v2 read path; multimodal is opt-in at write time.

---

## 2. Format model

Each multimodal SCX file has:

- **One global obs axis**: cells are shared across modalities. The
  obs / obs_index / provenance / obs_predicate_index sections live at
  `modality_id = 0` (global).
- **Per-modality var, X (CSR + optional CSC), layers, obsm, obsp,
  uns**: all stamped with `modality_id ≥ 1`.
- **A `ModalityTable` section** (`SectionType::ModalityTable = 15`)
  recording the modality name, type, n_vars, nnz, shard counts, and
  default codec / value encoding. Indexed by 1-based `modality_id`.

Per-cell "this modality has no measurement here" is expressed by an
empty CSR row in that modality's shard, not by varying obs across
modalities.

See [docs/format.md § 13 Multimodal Extension](format.md#13-multimodal-extension-optional)
for the on-disk byte layout and [docs/api.md § Section Types](api.md#section-types)
for the `SectionType` enum.

---

## 3. Python (pyscx)

### 3.1 Writing — `pyscx.from_mudata`

```python
import mudata
import pyscx

mu = mudata.read_h5mu("citeseq.h5mu")  # MuData(rna=AnnData, adt=AnnData)
pyscx.from_mudata(mu, "citeseq.scx")    # codec="auto" by default
```

`codec="auto"` routes through `select_codec_for_modality(...)`:
RNA → Scx1 (UMI counts) or Pcodec (floats); Protein/ADT → Zstd; ATAC →
Zstd for binary peaks else Lz4Shuffle (see
[docs/codec.md § Per-modality codec defaults](codec.md#8a-per-modality-codec-defaults)).

**Row-group framing (v4) is on by default**, matching the unimodal path:
`from_mudata` / `from_h5mu` / `scx convert --from h5mu` write v4 files with
row-group framing (`row_group_rows=256`) for fast random/scattered reads. Pass
`row_group_rows=0` (`--row-group-rows 0` on the CLI) to opt out and write the
legacy unframed v3 layout. Note the GPU tradeoff: framing applies to Scx1
integer-count modalities (UMI / ADT) too, so on GPU they host-bounce
(`scx_device_handoff_streamed`) instead of taking the in-VRAM Scx1
device-decode path — the same tradeoff Phase C made for unimodal. For
multimodal *training* throughput on GPU where that matters, `row_group_rows=0`
restores the device-decode path.

### 3.2 Reading — `to_mudata()` / `to_anndata()`

```python
import pyscx

reader = pyscx.open("citeseq.scx")
print(reader.is_multimodal)         # True
print(reader.n_modalities)          # 2
print(reader.modality_names)        # ['adt', 'rna']

# Multimodal materialisation:
mu = reader.to_mudata()             # mudata.MuData
rna = mu.mod["rna"]                 # AnnData
adt = mu.mod["adt"]

# Single-modality extract (when you only want one):
rna_only = reader.to_anndata()      # raises on multimodal — use to_mudata
```

`Experiment.modality_info(modality_id)` returns the per-modality
record (`{name, modality_type, default_codec_id, n_vars, nnz, …}`)
useful for introspection and writer-symmetry checks.

#### Backed multimodal access

CITE-seq / Multiome / TEA-seq files at census scale don't fit in RAM.
For ad-hoc analysis (QC, exploratory plots, notebook work) on those
files, both `to_mudata` and `to_anndata` now expose an out-of-core path.

```python
import pyscx

reader = pyscx.open("citeseq.scx")

# All-modalities backed: returns mudata.MuData of backed AnnData,
# all sharing the same global obs Arrow table.
mu = reader.to_mudata(backed=True)
mu.mod["rna"].X      # ScxBackedSparseDataset — reads shards on demand
mu.mod["adt"].X      # ScxBackedSparseDataset — reads shards on demand

# Per-modality backed, without extracting to a new file:
rna = reader.to_anndata(modality="rna", backed=True)

# Per-modality lazy transforms — modality scope flows through the
# wrapped BackedCsrReader automatically:
import pyscx.accel as pp
pp.normalize_total(rna)
pp.log1p(rna)
rna.X                # ScxLazyTransformedDataset (modality-scoped)
```

`to_mudata(backed=True)` on a single-modality v1 or v2 file is supported
too — the result is a one-modality `MuData` rather than an error, so the
same code paths work uniformly across file layouts.

For **filtered per-modality reads**, use the modality-scoped query pipeline
(§ 3.4): `open(path).query(modality="rna").filter_obs(...).select_genes(...)
.collect()` pushes the obs predicate down and assembles only the matching cells
of that modality's X. The `to_anndata(modality=..., backed=True)` +
filter-kwargs (`var_names` / `obs_filter` / `layers`) combination is still not
wired — the *backed* path doesn't share the `QueryPipeline` — so use
`query(modality=...)` (in-memory result) or `scx subset --modality NAME --filter
'<expr>'` (§ 5, materialises a filtered single-modality file, then
`to_anndata(backed=True)` on the result) instead.
Cloud-backed `open_cloud(...).query(modality=...)` **is** supported (§ 3.4);
`open_cloud(...).to_mudata(backed=True)` is a separate follow-on.

### 3.3 Training — `pyscx.MultimodalTrainingDataset`

```python
import pyscx

ds = pyscx.MultimodalTrainingDataset(
    "citeseq.scx",
    modalities=["rna", "adt"],
    batch_size=512,
    seed=42,
)
for batch in ds:
    rna_x = batch["X"]["rna"]       # numpy [batch_size × n_vars_rna]
    adt_x = batch["X"]["adt"]       # numpy [batch_size × n_vars_adt]
    cells = batch["cell_indices"]   # uint64, aligned across modalities
    # Forward pass on (rna_x, adt_x) — e.g. totalVI / multivi.
    ...
```

The wrapper holds one `TrainingPipeline` per modality, sharing the seed
so the per-modality shufflers produce aligned row orderings. To keep
per-batch `cell_indices` identical across modalities, the wrapper pins a
**uniform effective `batch_size` and `shard_group_size`** — the minimum
across modalities — so the per-modality memory-budget auto-tuner cannot
shrink one modality's batch below another's and desync the batching. It
also validates at construction that all modalities share the same CSR
shard layout (row ranges); a genuine layout mismatch raises a clear error
directing to a writer-side fix (a uniform `shard_target_rows` across
modalities). Per-batch `cell_indices` are still checked on every batch as
a final guard.

`return_dict=False` switches the iterator to tuple batches
`(X_rna, X_adt)` aligned with the constructor's `modalities` order.

`max_memory_mb` is divided across modalities proportionally to
per-modality nnz with a 64 MB floor so the auto-tuner has room. Because a
uniform batch is pinned, a **wide modality** (many features — e.g. an ATAC
peak matrix) can pull the shared effective `batch_size` below the
requested value under the default budget; pass a larger `max_memory_mb`
to keep a larger batch across all modalities (a one-shot `log::info` fires
when the batch is reduced).

The floor is load-bearing — a share below it fails `LoaderConfig` validation
outright — so a request that cannot be divided without starving a modality is
rounded **up**: two modalities at `max_memory_mb=64` budget 64 MB *each*.
`memory_budget()` reports both numbers, `max_memory_mb` beside
`effective_total_mb`, and an explicit request that gets rounded up raises a
`UserWarning` (an adaptive budget resolving higher does not — that is the
adaptive policy working):

```python
b = ds.memory_budget()
b["batch_size"], b["shard_group_size"]   # pinned, uniform across modalities
b["max_memory_mb"], b["effective_total_mb"]
b["modalities"]["rna"]["breakdown"]      # the TrainingDataset envelope, per modality
```

The pinned `batch_size` / `shard_group_size` are uniform *by construction*:
the loader sets them on each already-built pipeline
(`TrainingPipeline::pin_effective_config`) rather than rebuilding every
pipeline at the minimum and checking afterwards.

`pyscx.TrainingDataset(path)` on a multimodal file falls back to the
alphabetically-first modality and emits a `UserWarning` directing the
user to `MultimodalTrainingDataset`. Pass `modality="rna"` explicitly
to suppress the warning.

`hvg_indices` is range-checked against the **selected** modality's feature
count on both of those scoped paths — an index `>= n_vars` is rejected at
construction rather than becoming an output column that is silently always
zero. `MultimodalTrainingDataset` is the sole exception: it shares one panel
across modalities of differing widths, so it cannot range-check and does not
(see [training.md § MultimodalTrainingDataset](training.md#multimodaltrainingdataset)).

The panel is sorted and deduplicated on every path including that one, so batch
columns are in ascending gene-index order whatever order it was passed in, and a
panel that is not already ascending-unique emits a `UserWarning`.

### Loaders refuse a multimodal file they cannot scope

Every loader resolves a cell by its **global obs row**, so the shard list it
reads has to claim each row exactly once. A multimodal file read without a
modality breaks that: each modality independently tiles `[0, n_obs)`, so the
flattened list claims every row once per modality.

- `TrainingPipeline` / `TrainingDataset` with no modality is rejected at
  construction. (Python never reaches this — `TrainingDataset` resolves a
  modality itself, falling back to the alphabetically-first with a warning.)
- **`IndexPlanDataset` and `SparseCellSetDataset` are rejected too.** Neither has
  a modality surface, and before this they read the flattened list through
  `BackedCsrReader`, whose row index keeps one arbitrary modality's shard per
  row — so a plan asking for cell 3 got *some* modality's cell 3, with no
  warning and no way to tell which. Extract a modality first:

  ```bash
  scx subset --modality rna atlas.scx atlas.rna.scx
  ```

- Scoping to a modality is checked too, not just accepted: that modality's own
  shards must tile `[0, n_obs)` exactly once. A malformed tiling (overlapping or
  gapped shard ranges, from a merge/append/compact defect) is rejected at
  construction rather than silently dropping or duplicating cells part-way
  through an epoch.

### 3.4 Modality-scoped queries — `query(modality=…)`

Selective, out-of-core reads of **one** modality run through the query
engine's predicate pushdown, scoped to that modality:

```python
import pyscx

exp = pyscx.open("citeseq.scx")

# The obs predicate resolves against the shared global obs axis; X,
# select_genes, and filter_var resolve against the RNA modality's var.
rna = (
    exp.query(modality="rna")
       .filter_obs("cell_type == 'T cell'")
       .select_genes(["MS4A1", "CD3D"])
       .collect()
       .to_anndata()          # single-modality AnnData, filtered + projected
)

# count() is modality-scoped too (obs predicate is global, so the count is
# the same across modalities):
n = exp.query(modality="adt").filter_obs("cell_type == 'T cell'").count()
```

Because obs is global (§ 2) and every modality's CSR shards independently
tile `[0, n_obs)`, the obs predicate produces one global row mask that is
applied to the target modality's shards — only the matching cells of that
modality are decoded (no full-modality materialisation, unlike `scx subset
--modality --filter`). X assembles at the modality's own `n_vars`.

On a multimodal file `modality=` is **required**: omitting it raises
`ValueError` (there is no unambiguous default axis), and an unknown name
raises `KeyError`. On a single-modality / v1 file omit `modality=` (the
default global axis; passing a name errors). A file carrying deletion
vectors applies global-obs deletion vectors (`modality_id = 0`) automatically during
a modality query, dropping deleted cells from the requested modality's X. Modality-scoped queries run over the **cloud** reader too —
`open_cloud(url).query(modality="rna")`, `read_cloud(url, modality="rna")`, and
`scx query <url> --modality rna` — for both packed and exploded `.scxd/`
layouts. (`open_cloud(...).to_mudata()` over cloud remains a separate follow-on.)

CLI and R equivalents:

```bash
scx query citeseq.scx --modality rna --filter "cell_type == 'T cell'" \
    --select-genes hvg.txt --output rna_tcells.scx
scx query citeseq.scx --modality rna --count --filter "total_counts > 1000"
```

```r
library(rscx)
rna <- scx_open("citeseq.scx") |>
  scx_query(modality = "rna") |>
  filter_obs("cell_type == 'T cell'") |>
  collect()
```

---

## 4. Bioconductor / Seurat (rscx)

### 4.1 Seurat v5 multi-assay

```r
library(rscx)
library(Seurat)

# Write — multi-assay v5 → multimodal SCX:
seu <- CreateSeuratObject(counts = rna_mat, assay = "rna")
seu[["adt"]] <- CreateAssay5Object(counts = adt_mat)
from_seurat(seu, "citeseq.scx")

# Read — multimodal SCX → Seurat v5:
seu_back <- scx_open("citeseq.scx")$to_seurat()
Assays(seu_back)                    # "rna" "adt"
```

Cells must align across assays (Seurat v5's invariant); mismatched
`n_obs` raises explicitly. Per-assay modality types are inferred from
the assay name (`rna`/`adt`/`atac`/`spatial`/...).

### 4.2 MultiAssayExperiment

```r
library(MultiAssayExperiment)

# Write — MAE → multimodal SCX:
mae <- MultiAssayExperiment(
  experiments = list(rna = rna_sce, adt = adt_sce),
  colData = shared_col_data
)
from_mae(mae, "citeseq.scx")

# Read — multimodal SCX → MAE:
mae_back <- scx_open("citeseq.scx")$to_mae()
```

`from_mae` requires aligned cell axes across experiments. On
mismatch, raises directing the user to
`MultiAssayExperiment::intersectColumns(mae)` (or NA-pad upfront)
before retrying. SCX's shared-obs invariant doesn't model MAE's
sampleMap directly — pre-align.

---

## 5. CLI

```bash
# Conversion (SCX → h5ad / h5mu also streams by default;
# pass `--stream=false` for the legacy materialising path)
scx convert --from h5mu citeseq.h5mu --to scx citeseq.scx
scx convert --from h5mu citeseq.h5mu --to scx citeseq.scx \
    --modalities rna,adt --modality-types adt:Protein   # streaming kwargs
scx convert --from scx citeseq.scx --to h5mu out.h5mu
scx convert --from scx citeseq.scx --to h5ad rna.h5ad --modality rna

# Inspection
scx info citeseq.scx        # per-modality table block (includes has_csc column)
scx validate citeseq.scx    # ModalityTable checksum + cross-check;
                            # accepts partial per-modality CSC sidecars

# Mutating ops (per-modality routing)
# NOTE: `scx append --modality` on a multimodal file is NOT yet supported
# (deferred) — it errors. Appending cells to one modality would grow the global
# obs axis while leaving sibling modalities under-covering it, producing an
# unreadable file. Extract a single modality first (see `scx subset` below),
# append to the extracted file, then re-merge.
scx subset citeseq.scx rna_only.scx --modality rna
scx subset citeseq.scx rna_tcells.scx --modality rna \
    --filter "cell_type == 'T cell'" --genes hvg.txt             # filter + projection
scx merge cite1.scx cite2.scx --output cite_merged.scx            # multimodal merge
scx delete cite_merged.scx --filter "cell_type == 'B cell'"      # whole-cell delete (all modalities)
scx compact cite_merged.scx cite_compacted.scx                   # multimodal compact (reclaims deleted rows)
```

> **Multimodal append is deferred.** `pyscx.append` / `append_from_anndata` (and
> `scx append`) reject a multimodal target with a `ValueError` ("append is not
> yet supported for multimodal files; extract individual modalities first with
> `scx subset --modality NAME`"). Appending cells to a single modality cannot
> keep the shared global obs axis consistent across the other modalities, so the
> result would be unreadable. Workaround: `scx subset --modality NAME` to a
> single-modality file, `pyscx.append` into that, then `scx merge` back.
> `pyscx.append` / `append_from_anndata` still work normally on single-modality
> files (omit `modality=`).

> **The var attach is refused too, for a different reason.**
> `pyscx.var_import` / `attach_var_columns` / `rscx::scx_attach_var` and
> `scx var-import` reject a multimodal target: each modality owns its own
> `var/<name>` section, so "the var axis" names nothing on such a file. Unlike
> append this is a scope decision rather than a correctness one — the in-place
> harness is already modality-parameterised and per-modality var is a single
> unsharded section — but the op does not model it and the obs twin has no
> equivalent, so it refuses rather than guessing. Same workaround:
> `scx subset --modality NAME`, attach, `scx merge` back. `attach_external_layer`
> (`cellbender-import`), the other op that writes var, refuses multimodal on the
> same grounds.

Python equivalent for the export direction:

```python
pyscx.from_h5mu("citeseq.h5mu", "citeseq.scx",          # path-based entry
                modalities=["rna", "adt"])
pyscx.to_h5mu("citeseq.scx", "out.h5mu")                # streams per-modality X + layers
pyscx.to_h5ad("citeseq.scx", "rna.h5ad", modality="rna")  # single-modality extract
```

Multimodal `scx merge` and `scx compact` now dispatch to
`scx-ops::merge_multimodal` / `compact_multimodal`: the keep mask /
concatenation is applied across every modality's CSR shards and
layers, the modality table and per-modality var / obsm / uns are
preserved, and per-modality CSC sidecars are dropped (rebuild via
`--rebuild-csc`). Merge validates var identity (index, column names,
values) across all inputs by default; pass `assume_identical_var=True`
to check only `n_vars`. Obs is streamed shard-by-shard (no full obs
materialization). The `uns_policy` kwarg (`"first"` / `"require_equal"`
/ `"namespace"` / `"summary"`) controls how conflicting uns sections
are resolved.

---

## 6. Limitations and follow-ons

### Supported multimodal operations

| Operation | Status | Notes |
|---|---|---|
| `pyscx.from_mudata` / `Experiment.to_mudata` | Supported | — |
| `pyscx.from_h5mu(path, out, ...)` (path-based, streaming) | Supported | `modalities=` / `modality_types=` kwargs |
| `to_mudata(backed=True)` | Supported | Local files only; cloud variant pending |
| `to_anndata(modality=…, backed=True)` | Supported | — |
| Modality-scoped lazy transforms | Supported | Per-modality `pp.normalize_total` / `pp.log1p` |
| `scx convert --from/--to h5mu` | Supported | Streaming default; `--modalities`, `--modality-types` |
| `pyscx.MultimodalTrainingDataset` | Supported | — |
| `scx subset --modality NAME` | Supported | — |
| `scx subset --modality NAME --filter … --genes …` | Supported | Composes filter + projection in one pass |
| `query(modality=…)` / `scx query --modality` / `scx_query(modality=)` | Supported | Local modality-scoped predicate pushdown (obs mask global, X at the modality's `n_vars`); §&nbsp;3.4 |
| `scx append --modality NAME` | Not supported (deferred) | Rejected with `MultimodalUnsupported`: a single-modality append would leave siblings under-covering the global obs axis. Extract via `scx subset --modality`, append, then `scx merge` |
| `scx merge` on multimodal inputs | Supported | Dispatches to `merge_multimodal`; per-modality CSC dropped — `--rebuild-csc` to re-emit |
| `scx delete` / `mark_deleted` / `scx_delete` on multimodal inputs | Supported | Whole-cell delete: global-obs deletion vector removes the cell from **every** modality; modality-scoped query returns deletion-filtered rows |
| `scx compact` on multimodal inputs | Supported | Dispatches to `compact_multimodal`; keep mask applied across every modality |
| `to_anndata(modality=…, backed=True)` + filter kwargs | Not supported | Use `query(modality=…)` (in-memory) or `scx subset --modality NAME --filter` |
| `open_cloud(...).query(modality=…)` / `read_cloud(..., modality=…)` | Supported | Modality-scoped predicate pushdown over the cloud reader (packed + exploded `.scxd/`); `scx query <url> --modality` too |
| `open_cloud(...).to_mudata(backed=True)` | Not supported | Cloud `SectionReader` is wired for unimodal `.query()`; multimodal backed export is a follow-on |

### Detail

- **Multimodal append (deferred)**: `scx append` / `pyscx.append` reject a
  multimodal target (`MultimodalUnsupported`). SCX requires cell-aligned
  modalities (every modality covers the global `[0, n_obs)` obs axis; `to_mudata`
  enforces this), so appending new cells to one modality — which bumps the
  global `n_obs` but extends only that modality — would leave siblings
  under-covering the axis and the file unreadable. Per-modality append that
  keeps all modalities aligned is a planned follow-on. Until then, extract a
  modality with `scx subset --modality NAME`, append to the single-modality
  file, and `scx merge` the modalities back together.
- **Multimodal merge / compact**: `merge` walks every modality,
  copies/re-encodes CSR shards in input order with `row_start`
  adjusted for the cumulative global obs offset, and preserves
  per-modality var/obsm/uns from the first input. `compact` applies
  the deletion-vector keep mask across every modality's CSR shards
  and per-modality layers. Both drop CSC sidecars by default;
  `--rebuild-csc` regenerates them.
- **Deleting cells (whole-cell)**: `scx delete` / `pyscx.mark_deleted` /
  `scx_delete` mark cells on the shared global obs axis. The deletion vector
  stores global obs row indices (`modality_id = 0`, see
  [docs/format.md §7.3](format.md#73-deletion-vectors-logical-delete)), so a
  deleted cell is removed from **every** modality — the default read, each
  modality-scoped `query(modality=…)`, and `compact` all drop the same rows and
  stay cell-aligned. `scx rollback` still undoes a delete (shard bytes are never
  touched). Note that eager `to_mudata()` is deliberately *unfiltered* per
  modality — each modality's X stays row-aligned with the unfiltered outer
  `global_obs`, so it returns the physical shape; use `query(modality=…)` (or a
  `compact`ed file) for the deletion-filtered view. Per-modality scoped deletion
  (drop one modality's measurement, keep the others) is reserved wire-format
  headroom, not yet exposed.
- **`subset --modality NAME` + `--filter` / `--genes`**:
  `extract_modality_with_filter` reads the modality CSR, applies the
  obs predicate against the global obs, projects to the chosen genes,
  and writes a single-modality v2 SCX in one pass.
- **`query(modality=…)` (modality-scoped pushdown)**:
  `scx_engine::QueryPipeline::open_for_modality` scopes X assembly and
  var/gene resolution to one modality (`csr_shards_for_modality`,
  per-modality `n_vars`) while the obs predicate evaluates against the
  shared global obs axis — so only the matching cells of that modality are
  decoded, no full-modality materialisation. The Level-2 predicate-index
  fast path is disabled for modality-scoped queries (the index's shard ids
  are keyed to the flattened all-modality shard order); Level-1
  catalog-stats pruning still runs per modality. Global deletion vectors (`modality_id = 0`)
  are automatically applied during modality-scoped queries. See § 3.4.
- **MAE sampleMap with non-aligned cells**: `from_mae` raises rather
  than NA-padding. Users should `intersectColumns()` upfront. Future
  work could lift this by emitting NA values into the mismatched cells
  of each modality's CSR.
- **Multimodal compression / training benchmarks**: deferred to a
  follow-on benchmark release.

---

## 7. Cross-references

- [docs/format.md § 13 Multimodal Extension](format.md#13-multimodal-extension-optional) — on-disk byte layout.
- [docs/api.md § Multimodal API](api.md#multimodal-api) — Rust + PyO3 surface.
- [docs/codec.md § Per-modality codec defaults](codec.md#8a-per-modality-codec-defaults) — auto-codec routing per modality.
- [docs/cloud.md § Exploded `.scxd/` layout](cloud.md#exploded-scxd-layout) — `_modality_table.bin` + `X/{modality}/` directories.
- [docs/scanpy.md](scanpy.md) — single-modality scanpy/AnnData integration (multimodal example follows the same `pyscx.from_mudata` / `to_mudata` pattern shown here).
