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
  open v1 files, but v1 readers reject v2 files with `UnsupportedVersion`.

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

See [docs/format.md § 13 Multimodal Extension](format.md#13-multimodal-extension)
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
[docs/codec.md § Per-modality codec defaults](codec.md#per-modality-codec-defaults)).

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

Filter kwargs (`var_names`, `obs_filter`, `layers`) are **not** supported
together with `modality=...` in this PR — the modality-scoped backed
path doesn't share a `QueryPipeline` with the unimodal predicate-pushdown
machinery yet. Use `scx subset --modality NAME --filter '<expr>'` (now
supported, see § 5) to materialise a filtered single-modality file
first, then `to_anndata(backed=True)` on the result.
Cloud-backed `open_cloud(...).to_mudata(backed=True)` is not yet wired
even though the `SectionReader` abstraction has landed —
local cloud query goes through `open_cloud(...).query()` for now.

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
    cells = batch["cell_indices"]   # int64, aligned across modalities
    # Forward pass on (rna_x, adt_x) — e.g. totalVI / multivi.
    ...
```

The wrapper holds one `TrainingPipeline` per modality, sharing seed /
batch_size so the per-modality shufflers produce aligned row orderings.
Per-batch `cell_indices` are validated to match across modalities; a
mismatch raises with a clear error directing to a writer-side fix
(typically a uniform `shard_target_rows` across modalities).

`return_dict=False` switches the iterator to tuple batches
`(X_rna, X_adt)` aligned with the constructor's `modalities` order.

`max_memory_mb` is divided across modalities proportionally to
per-modality nnz with a 64 MB floor so the auto-tuner has room.

`pyscx.TrainingDataset(path)` on a multimodal file falls back to the
alphabetically-first modality and emits a `UserWarning` directing the
user to `MultimodalTrainingDataset`. Pass `modality="rna"` explicitly
to suppress the warning.

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
scx append citeseq.scx new_rna_cells.scx --modality rna          # preserves ADT's CSC
scx subset citeseq.scx rna_only.scx --modality rna
scx subset citeseq.scx rna_tcells.scx --modality rna \
    --filter "cell_type == 'T cell'" --genes hvg.txt             # filter + projection
scx merge cite1.scx cite2.scx --output cite_merged.scx            # multimodal merge
scx compact cite_merged.scx cite_compacted.scx                   # multimodal compact
```

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
| `scx append --modality NAME` | Supported | Per-modality CSC invalidation; other modalities' CSC preserved |
| `scx merge` on multimodal inputs | Supported | Dispatches to `merge_multimodal`; per-modality CSC dropped — `--rebuild-csc` to re-emit |
| `scx compact` on multimodal inputs | Supported | Dispatches to `compact_multimodal`; keep mask applied across every modality |
| `to_anndata(modality=…, backed=True)` + filter kwargs | Not supported | `scx subset --modality NAME --filter` |
| `open_cloud(...).to_mudata(backed=True)` | Not supported | Cloud `SectionReader` is wired for unimodal `.query()`; multimodal backed export is a follow-on |

### Detail

- **Per-modality CSC sidecars on append**: `scx append --modality rna`
  clears `HAS_CSC` only on the target modality; ADT's CSC sidecar
  stays intact. The file-level `header.has_csc()` flag means
  "at least one modality still owns a CSC sidecar" for v2 files.
  `scx info` shows the per-modality state in a `has_csc` column.
  Pass `--rebuild-csc` to re-emit the dropped sidecar (today this
  re-runs against the full file; per-affected-modality
  `--rebuild-csc all` is a follow-on).
- **Multimodal merge / compact**: `merge` walks every modality,
  copies/re-encodes CSR shards in input order with `row_start`
  adjusted for the cumulative global obs offset, and preserves
  per-modality var/obsm/uns from the first input. `compact` applies
  the deletion-vector keep mask across every modality's CSR shards
  and per-modality layers. Both drop CSC sidecars by default;
  `--rebuild-csc` regenerates them.
- **`subset --modality NAME` + `--filter` / `--genes`**:
  `extract_modality_with_filter` reads the modality CSR, applies the
  obs predicate against the global obs, projects to the chosen genes,
  and writes a single-modality v2 SCX in one pass.
- **MAE sampleMap with non-aligned cells**: `from_mae` raises rather
  than NA-padding. Users should `intersectColumns()` upfront. Future
  work could lift this by emitting NA values into the mismatched cells
  of each modality's CSR.
- **Multimodal compression / training benchmarks**: deferred to a
  follow-on benchmark release.

---

## 7. Cross-references

- [docs/format.md § 13 Multimodal Extension](format.md#13-multimodal-extension) — on-disk byte layout.
- [docs/api.md § Multimodal API](api.md#multimodal-api) — Rust + PyO3 surface.
- [docs/codec.md § Per-modality codec defaults](codec.md#per-modality-codec-defaults) — auto-codec routing per modality.
- [docs/cloud.md § Exploded `.scxd/` layout](cloud.md#exploded-scxd-layout) — `_modality_table.bin` + `X/{modality}/` directories.
- [docs/scanpy.md](scanpy.md) — single-modality scanpy/AnnData integration (multimodal example follows the same `pyscx.from_mudata` / `to_mudata` pattern shown here).
