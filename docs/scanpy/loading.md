# Loading SCX data into AnnData

> Part of the [SCX + scanpy guide](README.md). This page covers eager loading with `to_anndata()` and
pulling subsets with the query pipeline; for data that should stay on disk see
[Backed mode](backed-mode.md).

## Understanding `to_anndata()`

### Full signature

```python
exp.to_anndata(
    # Declaration order, matching `#[pyo3(signature = ...)]` and `__init__.pyi`.
    # Every parameter is positional-or-keyword, so this order is the calling
    # contract, not presentation -- `test_experiment_stub_coverage.py` pins it.
    backed=False,         # True for lazy loading (X stays on disk)
    cache_shards=4,       # LRU cache size for backed mode
    var_names=None,       # List of gene names to project (column subset)
    obs_filter=None,      # Predicate string to filter cells (e.g. "cell_type == 'T cell'")
    layers=None,          # None = load all layers; pass a list to select specific layers
                          # (e.g. ["raw_counts"]), or [] to skip loading layers entirely
    preserve_slots=False, # With obs_filter in non-backed mode: materialise the aligned
                          # slots after filtering (pandas.eval grammar) instead of taking
                          # the query-engine path, which drops them
    modality=None,        # Select one assay / feature space from a multimodal file (e.g.
                          # "rna" or "adt" in CITE-seq; requires backed=True). Defaults to
                          # None for standard single-modality files (on multimodal files, None
                          # raises ValueError). Rejects selection kwargs (var_names, obs_filter, etc.)
    eager=False,          # False (default): obsp/varp/varm and non-backed layers are
                          # wrapped in lazy bridges that decode each entry on first
                          # access. True: materialise everything up front so the
                          # AnnData is fully detached from the SCX file handle
    memory_budget=None,   # None (default: 8 GiB), int (bytes), or a binary-prefixed
                          # size str: K/M/G/T or KiB/MiB/GiB/TiB ("4G" / "512MiB"); decimal KB/MB rejected.
                          # When the estimated eager assembly footprint exceeds this
                          # budget, a UserWarning is emitted recommending backed mode.
                          # Advisory only -- assembly still proceeds
    obsm=None,            # None = load all obsm keys (default); pass a list to load only
                          # those embeddings (e.g. ["X_pca"]), or [] to skip obsm. Selecting
                          # keys also switches obsm to a lazy / backed row-gather bridge
                          # (see "Selective + lazy obsm" below)
    preserve_var_order=False,  # Return genes in var_names order rather than sorted
    strict_var_names=True,     # KeyError on a var_name absent from var (False drops it)
    container="csr",      # "csr" (default) or "dense" -- read-side materialisation
    data_dtype=None,      # Narrow X / raw in-decode (e.g. "uint16"); None = float32
    index_dtype=None,     # Index width (e.g. "int64"); None = int32
    allow_lossy=False,    # Permit a narrowing cast that cannot round-trip
    obsp=None,            # None = all obsp keys; a list = that subset; [] = none
    varp=None,            # Same contract as obsp
    varm=None,            # Same contract as obsp. An empty / fully-excluding list builds
                          # no lazy bridge at all, which is what actually bounds the cost:
                          # anndata validates every entry of a slot on the first
                          # `adata.obsp` access, so one touch decodes the whole slot
    raw=True,             # True: reconstruct adata.raw where the mode allows, and emit
                          # the dropped_raw notice where it cannot. False: neither
)
```

An unknown key in any of `layers` / `obsm` / `obsp` / `varp` / `varm` raises
`KeyError` naming the slot and listing what the file has, on every path.

### Default behavior (no extra params)

Calling `to_anndata()` performs a **full read** of the SCX file. Here is
exactly what happens:

1. **Reads and decompresses every CSR shard** from disk. Each shard is decoded
   using its per-shard codec (Scx1, Zstd, or raw), then all shards are
   assembled into a single contiguous CSR matrix.
2. **Applies deletion vectors.** If any cells have been logically deleted via
   `mark_deleted()`, those rows are excluded from both the matrix and the obs
   metadata. You always get a clean view — no manual filtering needed.
3. **Transfers the CSR matrix to scipy via zero-copy.** The Rust `Vec`s for
   indptr (`i64`), indices (`i32`), and data (`f32`) are moved directly into
   numpy arrays with no memory copy. These dtypes match what scipy expects, so
   `scipy.sparse.csr_matrix` wraps them without conversion.
4. **Reads obs/var metadata** as Arrow RecordBatches, converts to pandas
   DataFrames via `pyarrow.to_pandas()`.
5. **Reads obsm and uns eagerly**, and **wires lazy bridges for obsp,
   varp, varm, and (non-backed) layers** — each entry is decoded on
   first access rather than during `to_anndata()` itself.

The returned `anndata.AnnData` is fully populated:

| Slot | Source | Type |
|------|--------|------|
| `X` | CSR shards | `scipy.sparse.csr_matrix` (zero-copy) |
| `obs` | Obs metadata section | pandas DataFrame |
| `var` | Var metadata section | pandas DataFrame |
| `obsm` | Obsm sections | dict of numpy arrays (e.g. `X_pca`, `X_umap`); `ScxLazyObsmMapping` when `obsm=[...]` selected (see below) |
| `varm` | Varm sections | `ScxLazyVarmMapping` (lazy) / dict of numpy arrays (`eager=True`) |
| `obsp` | Obsp sections (COO Arrow IPC) | `ScxLazyPairwiseMapping` (lazy) / dict of `scipy.sparse.csr_matrix` (`eager=True`) |
| `varp` | Varp sections (COO Arrow IPC) | `ScxLazyPairwiseMapping` (lazy) / dict of `scipy.sparse.csr_matrix` (`eager=True`) |
| `uns` | Uns section | dict (tagged-JSON round-tripped; see below) |
| `layers` | Layer shards | `ScxLazyLayersMapping` (lazy, non-backed) / `ScxBackedLayerDataset` per entry (backed) / dict (`eager=True`) |

> Scanpy workflows that produce `obsp` / `varp` / `varm` (`sc.pp.neighbors`
> writes `obsp["distances"]` + `obsp["connectivities"]`; `pyscx.accel.pca`
> writes `varm["PCs"]`) survive `pyscx.from_anndata` → `to_anndata`
> verbatim. Sparse pairwise matrices are stored as float32 COO; higher
> precision is downcast on write. When cells are logically deleted via
> `mark_deleted` (or excluded by `obs_filter` in backed mode), `obsp` is
> subset to the kept rows and columns when the entry is decoded so the
> in-memory AnnData stays shape-consistent. The on-disk section keeps
> its original axis until `compact` rebuilds the file. `varp` and `varm`
> are unaffected by the deletion vector (var axis).

> **Lazy `obsp` / `varp` / `varm` / `layers`** (default `eager=False`):
> the four slots above are `MutableMapping`-compatible bridges
> (`ScxLazyPairwiseMapping`, `ScxLazyVarmMapping`,
> `ScxLazyLayersMapping`) that decode each entry from the SCX file
> only on first access, and cache the materialised value. Lookups via
> `__contains__` and key iteration stay catalog-only (no I/O).
> Mutations are in-memory and never write back to disk. The bridges
> keep a sibling `Arc<ScxReader>` alive so the returned AnnData stays
> usable after the source `Experiment` drops. Pass `eager=True` to
> substitute a plain `dict` and fully detach the AnnData from the SCX
> file handle — required when you intend to close the experiment, hand
> the AnnData to a subprocess, or otherwise outlive the underlying
> mmap. See [`docs/api/python-experiment.md` § `Experiment`](../api/python-experiment.md#experiment) for the
> kwarg table.
>
> **Per-key laziness is not per-key access through `adata`.** Reading
> the *slot* — `ad.obsp[...]`, `for k, v in ad.varm.items():`,
> `dict(ad.layers)` — goes through anndata's
> `AlignedMappingProperty.__get__`, which constructs an `AlignedActual`
> whose `__init__` runs `_validate_value` over **every** entry. So the
> first touch of `ad.obsp` decodes every obsp key once, not just the
> one you indexed; a subsequent `ad.obsp["distances"]` is then a cache
> hit. This is anndata's design, not a version regression — 0.11.4 and
> 0.12.x behave identically — and it is why an in-place axis subset
> detaches the bridges first rather than letting anndata walk them
> (`pyscx/src/axis_align.rs`). Only direct bridge access
> (`ad._obsp["distances"]`) is decode-per-key.
>
> The lazy default therefore bounds the peak RSS of `to_anndata()`
> *itself* for files that carry large kNN graphs
> (`obsp["distances"]` / `obsp["connectivities"]`) or embeddings
> (`varm["PCs"]`), but it does not bound what the first slot access
> costs. To bound that, name the keys you want — or none of them:
> `to_anndata(obsp=["connectivities"])`, `to_anndata(obsp=[])`. Each of
> `obsp=` / `varp=` / `varm=` takes the same `None` = all / `[]` = none /
> list = subset shape as `obsm=` and `layers=`; an empty or
> fully-excluding list builds no bridge at all, so nothing remains that
> could decode. `to_anndata(layers=[], obsm=[], obsp=[], varp=[],
> varm=[], raw=False)` is the X / obs / var / uns-only read.

#### Selective + lazy `obsm` (`obsm=[...]`)

`obsm` is eager by default (it tends to be small relative to
`obsp`/`varp`/`varm`), so `obsm=None` is byte-identical to prior
behaviour. Passing `obsm=[...]` opts into selective loading — only the
listed embeddings are read — and changes *how* obsm is materialised:

```python
# Selective eager: load just X_pca (and skip X_umap / X_state / …).
adata = exp.to_anndata(obsm=["X_pca"])

# Lazy (non-backed): X_pca is a ScxLazyObsmMapping — decoded to a dense
# numpy array on first `adata.obsm["X_pca"]` access, cached thereafter.
adata = exp.to_anndata(obsm=["X_pca"], eager=False)

# Backed dense row-gather: X_pca is a ScxBackedObsmDataset. m[idx] reads
# only the touched obsm shards (per-key LRU = cache_shards), so a single
# huge embedding (e.g. 10M cells × 2000-d) stays O(batch) per access.
adata = exp.to_anndata(backed=True, obsm=["X_pca"])
emb = adata.obsm["X_pca"]          # ScxBackedObsmDataset
batch = emb[cell_indices]          # dense (len(idx), n_cols) float array
```

| `obsm=` | `backed` | `eager` | obsm value type | When it reads |
|---|---|---|---|---|
| `None` | any | any | dict of dense numpy arrays | all keys, at `to_anndata()` |
| `[...]` | any | `True` | dict of dense numpy arrays | listed keys, at `to_anndata()` |
| `[...]` | `False` | `False` | `ScxLazyObsmMapping` → numpy | listed key, on first access |
| `[...]` | `True` | `False` | `ScxLazyObsmMapping` → `ScxBackedObsmDataset` | only touched rows, per `m[idx]` |

An unknown key raises `KeyError`; `obsm=[]` loads no embeddings.
Deletion vectors compose with the row gather via the same
`kept_to_global` remap as `X`. Under `obs_filter`, the backed-obsm path
falls back to selective eager obsm (composing a pandas-query row mask
with shard gather is deferred). This is the fix for per-worker obsm
memory blow-up on the random-access `embed_key`=`<obsm key>` dataloader
path — `obsm=[embed_key]` drops every unused embedding, and `backed=True`
keeps a single huge key off the per-worker heap.

> **`ScxBackedObsmDataset` is registered as `anndata.abc.CSRDataset`.**
> AnnData's `obsm` (`AxisArrays`) re-validates every value on each public
> `adata.obsm[key]` access and only accepts a fixed allowlist of array
> types; the lazy dense types it allows (`h5py.Dataset` / `zarr.Array` /
> `dask.array`) are concrete classes we can't subclass. Registering the
> backed dataset as a `CSRDataset` virtual subclass is what lets
> `adata.obsm[key]` return it (and `m[idx]` gather rows) rather than
> raising. The dataset is **dense** despite the `CSRDataset` label:
> `m[idx]` / `np.asarray(m)` / `m.toarray()` all return dense numpy. It
> does **not** implement CSR-only methods (`.tocsr()`), so code that
> introspects `adata.obsm[key]` as a sparse matrix will not work — treat
> it as a backed dense array (index it, or `np.asarray` it).

> **`uns` round-trip fidelity:** `from_anndata()` defaults to
> `uns_format="tagged"`, which preserves NumPy `dtype` and `shape`,
> bit-exact `float32` values, NaN/Inf inside arrays, structured
> recarrays (e.g. `uns["rank_genes_groups"]["names"]`), and
> `pd.Categorical` / `pd.Index` / `pd.Series` metadata (`name`, `codes`,
> `categories`, `ordered`). Legacy callers that depended on the previous
> behavior — where every NumPy array readback was a plain `list` — can
> opt back into it with `pyscx.from_anndata(adata, path,
> uns_format="plain")`. Both modes are read-compatible: the
> auto-detecting reader passes plain JSON through unchanged and decodes
> tagged envelopes back to their original Python types. See
> [`docs/api/python-experiment.md` § `uns` serialization](../api/python-experiment.md#uns-serialization) for
> the on-disk envelope schema.

**Memory implications:** Once materialized, the in-memory AnnData is
**identical** whether the source was an `.scx` file or an `.h5ad` file —
the same scipy CSR matrix with the same dtypes (`i64` indptr, `i32` indices,
`f32` data). SCX's compression advantage applies only to the on-disk file
(e.g., an SCX file may be 200 MB where the equivalent h5ad is 1.5 GB), but
after `to_anndata()` both produce the same in-memory CSR.

The CSR memory formula:
```
memory ≈ (n_obs + 1) × 8 bytes           # indptr (i64)
       + nnz × 4 bytes                   # indices (i32)
       + nnz × 4 bytes                   # data (f32)

# Example: 1M cells × 30K genes × 5% density = 1.5B non-zeros
# ≈ 8 MB + 5.6 GB + 5.6 GB ≈ 11.2 GB
#
# At 2% density (more typical for 10x Chromium):
# nnz = 600M → ≈ 8 MB + 2.2 GB + 2.2 GB ≈ 4.5 GB
```

> [!WARNING]
> The formula above covers only the CSR arrays for X. The **total** memory
> footprint includes obs/var DataFrames, obsm embeddings, layers, and
> Python/h5py overhead — which can be substantial. For example, a 10M-cell ×
> 61K-gene dataset (176 GB h5ad) was OOM-killed during h5py streaming
> metadata reads with 80 GB of RAM available. As a rule of thumb, budget
> **2–3× the CSR size** for a comfortable working set, or use backed mode
> for datasets over ~500K cells.

`to_anndata()` performs a catalog-only estimate of the eager assembly
footprint before loading data. When the estimate exceeds `memory_budget`
(default 8 GiB), a `UserWarning` is emitted recommending `backed=True`
or `pyscx.open(path).query()`. The warning is advisory — assembly still
proceeds. Override the threshold with `memory_budget=`:

```python
exp.to_anndata(memory_budget="16G")   # raise the threshold
exp.to_anndata(backed=True)           # or use backed mode instead
```

If this exceeds your available memory, use
[selective loading](#selective-loading) or the
[query pipeline](#querying-subsets-before-loading) to load only what you
need. For fully out-of-core analysis, use
[backed mode](backed-mode.md#backed-mode-lazy-loading).

#### Narrowing the output dtype / dense output

The memory formula above assumes `i32` indices and `f32` data. You can cut that
in half (or more) by requesting a narrower `data_dtype`, or skip the scipy CSR
entirely with `container="dense"` — useful when the next step wants a dense,
narrow array anyway (sklearn, a PyTorch `Tensor`, scVI):

```python
# Half the X footprint: float16 values (10x/count data is small-valued).
adata = exp.to_anndata(data_dtype="float16")

# uint8 counts (0–255): a quarter of the f32 footprint.
adata = exp.to_anndata(data_dtype="uint8")

# Dense row-major ndarray straight out (no CSR → dense re-densify later).
X = exp.to_anndata(container="dense", data_dtype="float32").X   # numpy.ndarray
```

The **default** (`container="csr"`, no dtype kwargs) is unchanged and stays
zero-copy — the `i64/i32/f32` Vecs are moved into numpy with no cast. A
non-default request on `X` / `raw` narrows **in-decode** (assembled directly at
the target width, no f32 intermediate), so it lowers peak RSS rather than costing
an extra copy; eager `layers` are the remaining read-then-convert.

Narrowing is **fail-loud** by default: a value that cannot be represented in the
requested dtype (out of range, fractional into an integer, negative into an
unsigned type, or a count above 2²⁴ into `float16`) raises `ValueError` naming
the offending value. Pass `allow_lossy=True` to narrow anyway. This also fixes a
prior silent `u32 → f32` rounding above 2²⁴ (e.g. pseudobulk / aggregated
counts). See [`docs/api/python-experiment.md` § Container and dtype materialization](../api/python-experiment.md#container-and-dtype-materialization)
for the full reference. (Note: these kwargs apply to the eager and query paths;
`to_gpu_anndata` is f32-native and rejects them — narrow on the host first.) On a
query the dtype belongs on `collect()`, which is where the decode happens:
`exp.query().filter_obs(...).collect(data_dtype="float64")` reads a `> 2²⁴` count
exactly, while naming the dtype on the `to_anndata()` / `to_csr()` that follows
only casts values already decoded as f32 and fails loud — see
[`docs/api/python-query.md` § Declaring the dtype at `collect()`](../api/python-query.md#declaring-the-dtype-at-collect).

### Selective loading

All selective loading parameters work in both non-backed and backed modes.

#### Gene projection (`var_names`)

Load only specific genes, reducing memory and computation:

```python
# Load only marker genes
adata = pyscx.open("atlas.scx").to_anndata(
    var_names=["CD3E", "CD4", "CD8A", "MS4A1", "NCAM1"]
)
print(adata.n_vars)  # 5
```

In non-backed mode, `X` and each selected layer are assembled **already
projected**: each shard is decoded once and narrowed to the requested genes
while it is still one shard wide, so the full-width matrix is never built.
In backed mode, projection is applied lazily — full rows are decoded from
disk, but only the requested columns are retained in the returned CSR.

The bound is on the gene axis only. `obs`, `obsm` and `obsp` are cell-axis
members: they come back at full size whatever `var_names` says, and anndata's
copy transiently doubles them. On a file carrying an `n_obs × n_obs` kNN graph
that is the term that matters, so pair the projection with the slot filters:

```python
adata = pyscx.open("atlas.scx").to_anndata(
    var_names=markers, obsp=[], varp=[], varm=[]
)
```

`.raw` is not narrowed either — anndata never slices it on the gene axis — so a
projected read of a raw-bearing file still pays for raw in full. Pass
`raw=False` when the workload does not need it.

By default `var_names` is a **set selector**: the returned gene axis is in
sorted original-column order, and duplicate names collapse. Pass
`preserve_var_order=True` to return columns in the order you listed them
instead (duplicates still collapse, first occurrence wins) — useful when the
order carries meaning (e.g. a fixed signature panel):

```python
adata = pyscx.open("atlas.scx").to_anndata(
    var_names=["CD8A", "CD4", "CD3E"], preserve_var_order=True
)
list(adata.var_names)  # ['CD8A', 'CD4', 'CD3E']  (request order)
```

`preserve_var_order` works on the eager, backed, GPU, and query-engine
(`obs_filter` + `var_names`) paths. It is **not** supported by the streaming
accelerators on the resulting **backed** dataset: they decode columns in
sorted on-disk order, so a request-ordered gene axis would silently misalign
the result against `adata.var`. `highly_variable_genes`, `normalize_total`,
`log1p`, `calculate_qc_metrics`, `score_genes`, `pflog`, `pca`,
`pca_neighbors`, `pca_neighbors_umap`, `rank_genes_groups`, `pdex_ref`,
`pseudobulk_means`, `pseudobulk_dex` and `pdex_nb_glm` raise `RuntimeError`
rather than return misaligned output — run them before projecting by name, or
re-open without `preserve_var_order`.

Unknown names raise `KeyError` by default (`strict_var_names=True`). Pass
`strict_var_names=False` to silently drop names absent from the var metadata
(the pre-0.8.6 behaviour, which only errored when *every* name was unknown).

#### Cell filtering (`obs_filter`)

Filter cells using a predicate string. In non-backed mode, this leverages
the query engine with predicate pushdown (shard skipping). In backed mode,
it evaluates the predicate with pandas `.query()` on the (already
deletion-vector-filtered) obs DataFrame and folds the matches into the backed
dataset's row set — a different grammar (see [Filter Expression
Compatibility](#filter-expression-compatibility) below):

```python
# Load only T cells from lung tissue
adata = pyscx.open("atlas.scx").to_anndata(
    obs_filter="cell_type == 'T cell' and tissue == 'lung'"
)
```

##### Filter Expression Compatibility

`to_anndata()` evaluates `obs_filter` via one of three paths that fall into
**two grammars** — the `scx-engine` parser or pandas. They accept overlapping
but **not identical** expressions, so knowing which fires matters when a filter
string is reused across calls or pipelines (e.g. moving a filter from a
`query().filter_obs()` call to `to_anndata(backed=True, obs_filter=...)`).

| Path | Engine | Grammar | When it fires |
|---|---|---|---|
| SCX predicate engine | `scx-engine` predicate parser | engine | non-backed default (`preserve_slots=False`); `query().filter_obs(...)`; `pyscx.pull(...)` selective pulls |
| pandas `.query()` | `pandas.DataFrame.query` | pandas | `backed=True` with `obs_filter` set |
| pandas `.eval()` | `pandas.DataFrame.eval` | pandas | non-backed `preserve_slots=True` with `obs_filter` set |

`.query()` and `.eval()` share pandas's grammar, so the only split that matters
in practice is **engine vs pandas**: `backed=True` and `preserve_slots=True` both
accept the pandas-only forms below, while the default non-backed path and
`query().filter_obs()` use the stricter engine grammar.

**Portable subset (works in both paths):**

```python
"cell_type == 'T cell'"
"n_counts > 50"
"n_counts >= 50 and cell_type == 'T cell'"
"cell_type in ['T cell', 'B cell']"           # bracket-delimited list
"(n_counts > 50) or (cell_type == 'NK cell')"

# Parse in both, but select DIFFERENT rows when the column has nulls —
# see "Nulls: the one semantic divergence" below.
"not (cell_type == 'NK cell')"
"cell_type != 'NK cell'"
```

This subset uses comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`),
keyword-form boolean operators (`and`, `or`, `not`), the `in` operator
against a `[...]` list literal, and parenthesised sub-expressions. Tests in
`pyscx/tests/test_to_anndata_integration.py` assert that both paths select
identical rows for the entries above — `test_obs_filter_grammar_parity_common_ground`
on a fixture with **no** missing values, and
`test_obs_filter_grammar_parity_with_null_categorical` on one with nulls. The
null-bearing test deliberately omits `!=` and `not (...)`, which is the
divergence spelled out below.

**Divergences (work in one grammar only)** — the "pandas" column covers both
`backed=True` (`.query()`) and `preserve_slots=True` (`.eval()`):

| Expression | SCX engine | pandas (`.query()` / `.eval()`) |
|---|---|---|
| `n_counts > 50 & cell_type == 'T cell'` | ❌ parse error — use `and` | ✅ accepted as bitwise-and |
| `cell_type in ('T cell', 'B cell')` (tuple) | ❌ parse error — `in` requires `[...]` | ✅ accepted |
| `n_counts > 50 \| cell_type == 'NK cell'` | ❌ parse error — use `or` | ✅ accepted |
| Arithmetic on obs columns (e.g. `n_counts + n_genes > 100`) | ❌ not supported | ✅ accepted |
| String-method calls (e.g. `cell_type.str.startswith('T')`) | ❌ not supported | ✅ accepted |

**Nulls: the one *semantic* divergence.** Everything above is about which
expressions parse. This one is about what an expression that parses in both
grammars *means* when the column has missing values — an unannotated
`cell_type`, an obs column added by a join that didn't cover every cell.

The SCX engine uses three-valued (Kleene) logic, like SQL: a comparison
against a NULL cell is UNKNOWN, and only the final mask turns a surviving
UNKNOWN into "not matched". pandas is two-valued — `NaN == 'v'` is `False`
and `NaN != 'v'` is `True`. The two agree on `and`, `or` and `in`, and part
company on `!=` and `not`:

| For a row whose `cell_type` is NULL | SCX engine | pandas |
|---|---|---|
| `cell_type == 'B cell' or n_counts > 50` (and `n_counts` is 90) | ✅ matches | ✅ matches |
| `cell_type == 'B cell'` | ❌ | ❌ |
| `cell_type != 'B cell'` | ❌ UNKNOWN → not matched | ✅ matches |
| `not (cell_type == 'B cell')` | ❌ UNKNOWN → not matched | ✅ matches |

So a filter using `!=` or `not` on a null-bearing column selects a different
set of cells under `backed=True` than under the default path. Prefer the
positive form (`cell_type in [...]`) on columns that may have missing values,
or do the selection in Python where you can be explicit about `NaN`.

When `preserve_slots=True` is used with an `obs_filter`, `to_anndata()` emits
a `UserWarning` noting that the filter was evaluated via pandas.eval — this
surfaces in notebook output so the grammar shift is visible without reading
this section.

**Recommendation:** write filters in the portable subset above, and on a
column that may have missing values prefer the positive forms — `!=` and
`not (...)` parse everywhere but do not select the same rows. If a filter
truly needs pandas-only syntax, do the row selection in Python after
`to_anndata()` instead of inside `obs_filter` — that keeps the SCX call site
portable across `preserve_slots`, `backed=True`, and cloud selective pulls.

#### Layer selection (`layers`)

Load only specific layers instead of all:

```python
# Load raw counts layer only
adata = pyscx.open("atlas.scx").to_anndata(layers=["raw_counts"])

# Load no layers at all (X only)
adata = pyscx.open("atlas.scx").to_anndata(layers=[])
```

#### Multimodal files: selecting one modality (`modality`)

In a **multimodal** experiment (e.g. CITE-seq or 10x Multiome), multiple distinct biological assays are measured on the same individual cells:
- **CITE-seq**: Gene expression (**RNA**, ~30,000 genes) plus cell-surface antibody counts (**ADT** or **protein**, ~100–200 tags).
- **10x Multiome**: Gene expression (**RNA**) plus chromatin accessibility (**ATAC** peaks, ~100,000 genomic regions).
- **Spatial + transcriptomics**: Gene expression (**RNA**) plus a spatial modality carrying coordinate and morphology data.

In SCX, multimodal datasets share a global cell axis (`obs`), but each modality carries its own separate feature axis (`var`), expression matrix (`X`), layers, and embeddings. Because standard `anndata.AnnData` objects only support a single 2D feature matrix `X`, calling `to_anndata()` with the default `modality=None` on a multimodal file **raises a `ValueError`** — there is no single feature matrix that can represent all modalities at once.

To work with multimodal SCX files, inspect the modalities and choose an approach:

```python
import pyscx

exp = pyscx.open("citeseq.scx")

# Check if the file is multimodal and view registered modality names
print(exp.is_multimodal)     # True
print(exp.modality_names())  # ['rna', 'adt']

# Calling exp.to_anndata() with modality=None (the default) raises ValueError:
# "this file holds 2 modalities (['rna', 'adt']), which each cover the whole
#  obs axis, so `to_anndata()` has no single matrix to return."

# Option 1: Load all modalities together into a MuData object (recommended)
mdata = exp.to_mudata()      # returns mudata.MuData (mdata['rna'], mdata['adt'])

# Option 2: Extract a single modality as a backed AnnData using modality=...
adata_rna = exp.to_anndata(modality="rna", backed=True)
adata_adt = exp.to_anndata(modality="adt", backed=True)

print(adata_rna.shape)       # (n_cells, n_genes) e.g. (10000, 36601)
print(adata_adt.shape)       # (n_cells, n_antibodies) e.g. (10000, 140)
```

**Constraints when passing `modality=`:**
- **Requires `backed=True`**: `exp.to_anndata(modality="rna", backed=True)`. Backed mode attaches the shared global `obs` to a lazy `ScxBackedSparseDataset` wrapping that modality's on-disk CSR matrix. Calling `modality=...` with `backed=False` raises `ValueError` (use `to_mudata()` for eager in-memory multimodal loading).
- **Rejects selection kwargs**: In-place filtering arguments (`var_names`, `obs_filter`, `layers`, `obsm`, `obsp`, `varp`, `varm`, `raw=False`) cannot be combined with `modality=` and will raise `ValueError`. To extract a filtered modality, you can:
  1. Slice or query the resulting backed `adata` in Python: `adata_rna[adata_rna.obs["cell_type"] == "T cell", :]`
  2. Use a modality-scoped query pipeline: `exp.query(modality="rna").filter_obs("cell_type == 'T cell'").collect().to_anndata()`
  3. Extract to a standalone single-modality SCX file first: `scx subset citeseq.scx rna.scx --modality rna --filter "cell_type == 'T cell'"`

See [docs/multimodal.md](../multimodal.md) for complete details on multimodal SCX architecture, `to_mudata()`, and multimodal ML data loading.

#### Combining parameters

All parameters can be combined:

```python
adata = pyscx.open("atlas.scx").to_anndata(
    backed=True,
    var_names=["CD3E", "CD4", "CD8A"],
    obs_filter="cell_type == 'T cell'",
    layers=["raw_counts"],
)
```

## Querying subsets before loading

For large datasets, you don't need to load everything into memory. The SCX
query engine filters at the shard level, skipping data that can't match:

```python
# Load only T cells from lung tissue
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")
    .collect())

adata = result.to_anndata()
print(f"Loaded {adata.n_obs} cells, skipped {result.skipped_shards}/{result.total_shards} shards")

# Continue with scanpy as usual
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata)
```

### Query pipeline options

The query pipeline supports chaining multiple operations:

```python
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'B cell'")       # filter cells
    .filter_var("highly_variable == True")      # filter genes
    .select_genes([0, 1, 2, 100, 200])         # or select by index
    .with_normalize(1e4)                        # normalize in Rust (faster)
    .with_log1p()                               # log1p in Rust
    .limit(5000)                                # cap returned cells
    .collect())

adata = result.to_anndata()
```

When you use `with_normalize()` and `with_log1p()` in the query pipeline,
normalization runs in compiled Rust — significantly faster than the Python
equivalent on large datasets. The resulting AnnData is ready for downstream
analysis (PCA, clustering, etc.) without calling `sc.pp.normalize_total()`
or `sc.pp.log1p()` again.

## See also

- [Choosing the right approach](choosing-an-approach.md) — which loader fits
  your data size.
- [Backed mode and out-of-core iteration](backed-mode.md) — `backed=True`.
- [docs/api/python-experiment.md § Container and dtype materialization](../api/python-experiment.md#container-and-dtype-materialization).
