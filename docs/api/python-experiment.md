# Python: `Experiment`

> Part of the [SCX API reference](README.md).

## Experiment

The Python-visible class is `Experiment` (the Rust type is `PyExperiment`).
`repr(exp)` is AnnData-style — a `Experiment object with n_obs × n_vars = …`
header followed by indented `obs:` / `var:` / `uns:` / `obsm:` / `varm:` /
`layers:` key lists. On-disk codec / shard / format-version internals moved off
the repr onto `Experiment.info() -> str`, whose tokens now include
`value_encoding=`, `is_integer=` and `max_value=` (rendering them reads one
76-byte header per CSR shard, so `info()` is O(shards), not O(1)).

- `value_encoding` `→ str` / `is_integer` `→ bool` / `max_value` `→ int` — What the file stores, without decoding it. `value_encoding` is the on-disk value encoding of the CSR shards as `scx info` prints it: a numpy dtype name (`"uint16"`, `"float32"`, …) when every shard agrees, `"mixed (uint8, uint16)"` when they differ (append keeps each shard's own encoding), `"n/a"` with no shards; on a multimodal file it folds every modality's X shards, and a layer's encoding is `adata.layers[name].stored_dtype` on a backed handle. `is_integer` is `True` when every shard is integer-encoded (`uint8` / `uint16` / `uint32`), i.e. the values are counts — `False` for any float shard or a shard-less file. `max_value` is the largest stored value from the per-shard catalog stats: float-encoded shards record no value range and a shard without stats contributes nothing, so it is `0` for a float file — read it together with `is_integer`; it is physical, like `nnz` (values in logically deleted rows still count). Each costs one shard-header read per CSR shard on first access (`max_value` is catalog-only); the fold is memoised per `Experiment` — shared by `value_encoding`, `is_integer` and `info()`, reset by `reload()` / `close()`, and a stale handle refuses before it can answer from the memo — so later reads are free. None decodes a value.

- `read_obs(columns=None, *, logical=True)` / `read_var(columns=None, *, modality=None)` — Read the cell / gene metadata table as a pandas DataFrame **without touching `X`**. Reach for these instead of `to_anndata().obs` / `.var` when you only want the metadata: on a 500k × 61k Census file `read_var()` is ~0.05 s against ~1.4 s and ~10 GB peak for the full materialisation. Both always retain the pandas index column (barcodes / gene names), so a projected frame indexes the same as an unprojected one, and an unknown column name raises `KeyError` listing what is available. Boolean columns come back as the pandas nullable `boolean` dtype whether or not they contain nulls, so the dtype follows the schema rather than the data and agrees with what a `to_h5ad` round trip returns. On a column with nulls that means `.astype(bool)` raises — deliberately, since coercing "not covered" to `False` is the mistake nullability exists to prevent — and `.fillna(False)` is the explicit form.
  - `read_obs`'s `columns` is a genuine **pushdown** — unselected columns are never materialised, which matters because `obs` scales with `n_obs`.
  - **Row space.** `read_obs()` returns the **live** rows: deletion vectors applied, so `len(read_obs()) == n_obs == len(to_anndata(backed=True).obs)`, row for row and index for index — the same frame `query().collect()`, `gather_rows_sparse` and rscx's `$obs()` describe. `read_obs(logical=False)` is the **physical** table (`n_obs_physical` rows, deleted rows in place). `to_h5ad(obs_mask=)` and `Experiment.mark_deleted(mask)` accept a mask in either row space (a live-length mask is expanded through the keep mask; a pandas Series with a labelled index is checked for order), so a mask derived from `read_obs()` works. Identical on a file with no deletions (and the logical read costs no copy); `has_deletions` says whether they differ. `obs_categorical(col, *, logical=True)` / `obs_categorical_many(cols, *, logical=True)` follow the same rule (one code per live row by default; `categories` never loses a level to the filter). **Changed in 0.17**: `read_obs()` and `obs_categorical()` used to return the physical table on every file, so on a `mark_deleted` file `read_obs()` was longer than `to_anndata(backed=True).obs` with no warning — a pre-1.0 clean break, no `FutureWarning` cycle; pass `logical=False` for the old frame. A frame computed from `read_obs()` can still be landed positionally: `attach_obs_columns(positional=True)` and `modify_metadata(obs=)` accept either row space (a live-length frame leaves the deleted rows `null`).
  - `read_var`'s `columns` is a convenience projection applied **after** the decode. `var` is one section sized by `n_vars` (5.5 MB for 61k genes), so there is nothing to save at the I/O layer; it does not read less off disk.
  - `read_var(modality=…)` selects one modality's gene axis on a multimodal file. Omitting it reads the global / single-modality `var`, which on a multimodal file is usually not what you want. Unknown name → `KeyError`.
  - For enumerating one categorical obs column, prefer `distinct_values()` / `obs_categorical()`, which scan per-shard dictionaries instead of assembling the table.
  - `CloudExperiment` mirrors both, `logical=` included (its `n_obs` / `shape` / `repr` are the live count too since 0.17 — one small section read on a file with deletions, memoised per handle; `n_obs_physical` stays the header count). There, `read_obs(columns=…)` *is* a genuine network pushdown (per-column projected range reads); `read_var(columns=…)` is still post-fetch, for the same reason as locally.
- `to_anndata(backed=False, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=False, modality=None, eager=False, memory_budget=None, obsm=None, preserve_var_order=False, strict_var_names=True, container="csr", data_dtype=None, index_dtype=None, allow_lossy=False, obsp=None, varp=None, varm=None, raw=True)` — Convert to AnnData
  - `var_names`: list of gene names to project (column subset). A set selector by default (sorted original-column order, duplicates collapsed). `X` and each selected layer are assembled **already projected**, shard by shard, so a few genes out of tens of thousands cost roughly the projected result rather than the whole matrix twice (measured on a 40-shard 20 480 × 2 000 fixture carrying one layer: 74.0 MB → 8.9 MB peak; `container="dense"` 48.8 MB → 8.8 MB). The bound is on the **var axis only** — `obs`, `obsm` and `obsp` are on the other axis, so they are in the result at full size and anndata's copy transiently doubles them. On a file carrying an `n_obs × n_obs` kNN graph that term dominates whatever `var_names` says; pair it with the slot filters (`to_anndata(var_names=[...], obsp=[], varp=[], varm=[])`) for the tight bound. `.raw` is not projected either — anndata never var-slices it, so it stays on its own (usually wider) gene axis; pass `raw=False` if you do not want to pay for it
  - `preserve_var_order`: when True, return the gene axis in the order `var_names` was listed (first-occurrence-wins dedup) instead of sorted order. Works on eager / backed / GPU / query-engine paths. The streaming accelerators decode columns in sorted on-disk order and so cannot express a request-ordered gene axis; on a *backed* `X` they refuse rather than misalign the result against `adata.var` — `highly_variable_genes`, `normalize_total`, `log1p`, `calculate_qc_metrics`, `score_genes`, `pflog`, `pca`, `pca_neighbors`, `pca_neighbors_umap`, `rank_genes_groups`, `rank_genes_groups_df` (its compute mode; the `adata.uns[key]` extraction mode reads no matrix and is unaffected), `pdex_ref`, `pseudobulk_means`, `pseudobulk_dex` and `pdex_nb_glm` raise `RuntimeError`. The same refusal applies to a backed `X` reordered at the handle level — `adata[:, idx]` with a non-ascending `idx`, or `adata.X = adata.X[:, [7, 2, 11]]` / `X[:, ::-1]` — which installs the same presentation permutation. The refusal covers the matrix the op actually reads, so a named `layer=` and a layer handle assigned to `X` are rejected too — materialising `adata.X` does not help when the permutation is on the layer. Run them before reordering, select with a sorted index or a boolean mask, or materialise the matrix being read (`adata.X = adata.X.to_memory()`, or `adata.layers[name] = adata.layers[name].to_memory()`) — and only that matrix: materialising a named layer is sufficient for `layer=`, even while `adata.X` stays presentation-ordered, because each op is guarded on the matrix it selects rather than on `X`
  - `strict_var_names`: when True (default), any name absent from the var metadata raises `KeyError`. Pass False to silently drop unknown names (pre-0.8.6 behaviour)
  - `obs_filter`: predicate string for cell filtering. Non-backed mode uses the scx-engine query parser with shard pushdown; `backed=True` evaluates it with pandas `.query()` (different grammar — see [Filter Expression Compatibility](../scanpy/loading.md#filter-expression-compatibility) in docs/scanpy/loading.md)
  - `layers`: list of layer names to load (default `None` = all; `[]` loads none). **Changed in 0.18:** an unknown layer name raises `KeyError` naming the available layers. It previously loaded nothing and said so only indirectly, via the obs-filtered query path's "does not load" warning — which the slot filters made unreliable, since an excluded slot and a misspelled key both select nothing. All five slot filters (`layers`, `obsm`, `obsp`, `varp`, `varm`) now fail the same way, on every path.
  - `obsm`: list of obsm keys to load (default `None` = all keys, byte-identical to prior behaviour). When set, only the listed embeddings are read — dropping the per-process RAM of unused keys on the random-access dataloader path. An unknown key raises `KeyError`; `obsm=[]` loads no embeddings. Selecting keys also changes *how* obsm is materialised (see `obsm` loading modes under `eager`, below).
  - `obsp` / `varp` / `varm`: lists of keys to load from those slots (default `None` = all keys, byte-identical to prior behaviour). Same contract as `obsm=`: `[]` loads none, a list loads that subset, an unknown key raises `KeyError`. Unlike `obsm=`, an empty or fully-excluding list builds **no lazy bridge at all** for that slot — `adata.obsp` is then anndata's own empty mapping, so there is nothing left that could decode. This is the knob that bounds the cost described under the lazy-mapping note in [docs/scanpy/loading.md](../scanpy/loading.md#understanding-to_anndata): anndata's `AlignedMappingProperty` builds an `AlignedActual` on the first `adata.obsp` access, and that validates — and therefore decodes — **every** key of the slot, so a census-scale kNN graph comes off disk whether or not the caller wanted it. Honoured on the eager, backed and `preserve_slots` paths, and on `to_gpu_anndata`; on the obs-filtered query path these slots are dropped anyway, so excluding one simply removes it from the "not loaded" warning. Key validation is at the entry point, so an unknown key raises on every path including that one.
  - `raw`: when `True` (default), `adata.raw` is reconstructed on the paths that can (see [`adata.raw`](conversion.md#adataraw)) and the `DroppedRaw` notice is emitted on the three that cannot — backed mode, an obs-filtered query, and a deletion-vector-active file. When `False`, neither happens: no rebuild, no notice. The explicit opt-out for callers who never wanted raw and do not want the warning on every open.
  - `layers=` and `var_names=` force eager assembly of `obsp` / `varp` / `varm` regardless of `eager=False`: those are sliced by anndata, whose `AlignedMapping` validation would drag every bridge through the slice anyway, so fragmenting the cost across implicit slicing helps nobody. Combine them with the slot filters above to keep that assembly small. Under `var_names=` the **matrices** are the exception — `X` and each selected layer are projected while assembling and never exist at full width.
  - `backed`: when True, X and layers are lazy `ScxBackedSparseDataset` instances
  - `modality`: select one modality of a multimodal file and
    return a backed AnnData scoped to that modality (per-modality X /
    var / obsm; global obs shared). Currently requires `backed=True`.
    Every selection kwarg is **rejected**, not ignored — `var_names`,
    `obs_filter`, `layers`, `obsm`, `obsp`, `varp`, `varm` and
    `raw=False` all raise `ValueError`, because the per-modality builder
    accepts none of them (use `scx subset --modality NAME --filter` to
    pre-materialise a filtered single-modality file).
  - `eager` (default `False`): when `False`, the `obsp` / `varp` /
    `varm` slots (plus `layers` in non-backed mode) are returned as
    lazy bridges — `ScxLazyPairwiseMapping` / `ScxLazyVarmMapping` /
    `ScxLazyLayersMapping` — that decode each entry on first
    access. Keeps the peak RSS of `to_anndata()` itself bounded for
    files that carry large kNN graphs or embeddings; user code that
    accesses these slots pays the same one-time decode cost it would
    otherwise pay at construction time. Bridges expose the full
    `MutableMapping` protocol (`__getitem__` / `__contains__` /
    `__iter__` / `__len__` / `keys` / `items` / `values` / `get` /
    `__setitem__` / `__delitem__`); mutations stay in memory and are
    not written back. Pass `eager=True` to materialise everything up
    front and detach the returned AnnData from the SCX file handle
    (use this before closing the experiment or shipping the AnnData
    to a subprocess). `uns` is always eager regardless of this flag.
  - **`obsm` loading modes** (selected by the combination of `obsm`,
    `backed`, `eager`, `obs_filter`):
    - `obsm=None` (default): every obsm key is materialised eagerly as
      a dense numpy array — byte-identical to prior behaviour.
    - `obsm=[...]`, `eager=True`, **or** `obs_filter` set: the selected
      keys are materialised eagerly (selective eager).
    - `obsm=[...]`, `backed=False`, `eager=False`: obsm becomes a lazy
      `ScxLazyObsmMapping` bridge — each selected key is decoded to a
      dense numpy array on *first* access (`adata.obsm[key]`), so an
      unused/late key costs nothing. Same `MutableMapping` protocol as
      the other bridges.
    - `obsm=[...]`, `backed=True`, `eager=False`, no `obs_filter`:
      obsm becomes a `ScxLazyObsmMapping` whose values are
      `ScxBackedObsmDataset` — a shard-aware **dense row-gather**
      dataset. `m[idx]` / `m[idx_array]` decode only the touched
      `ObsmEmbeddingShard`s (bounded per-key LRU = `cache_shards`), so
      per-access memory is `O(batch × n_cols)` and independent of
      `n_obs`. This is the scalable path for a single huge embedding
      (e.g. millions of cells × thousands of dims) on the random-access
      `embed_key` dataloader. Under `obs_filter` the backed-obsm path
      falls back to eager obsm (composing a pandas-query row mask with
      shard gather is deferred). Deletion vectors compose via the same
      `kept_to_global` remap as `X`. `ScxBackedObsmDataset` is registered
      as an `anndata.abc.CSRDataset` virtual subclass so AnnData's `obsm`
      coercion accepts it on public `adata.obsm[key]` access — it is
      nonetheless **dense** (`m[idx]` / `np.asarray(m)` / `m.toarray()`
      return dense numpy; it has no `.tocsr()`).
  - `memory_budget` (default `None`, treated as 8 GiB): emits
    `EagerAssemblyMemoryHigh` `UserWarning` when the estimated eager
    footprint exceeds the budget. Warn-only — does not block
    assembly. Under `var_names=` the estimate is scaled by the fraction
    of genes selected, so taking the warning's own advice actually
    silences it; the catalog carries no per-column `nnz`, so that
    scaling assumes an even spread of nonzeros across genes.
    The estimate covers **assembly only** — `X`, `adata.raw` when the read
    includes it, and obs/var — and is a floor on process RSS rather than a
    figure a job can be sized from: whatever reads the matrix afterwards
    (scanpy's per-group copies, a write-back) is not in it and cannot be.
    It prices the plan the read will actually use: `container="dense"` is
    `n_obs × n_vars × value_width` with no index array, and the value width
    comes from `data_dtype` (a `float64` read holds 8 B per value). For a CSR
    read it counts that value width plus the column-index width the assembled
    CSR will hold — **4 B below `i32::MAX` nonzeros and 8 B above it**, which
    is scipy's choice and not the caller's, so `index_dtype=` does not enter
    into it. `to_gpu_anndata`'s device path assembles no host `X` and so is not
    charged for one; its host-assembling fallback is. It is a floor on the objects an ordinary read
    leaves resident, not a bound on its transients — the dense reader builds a
    CSR and scatters from it, and the assembler's bounded in-flight shard
    decodes are not in it either. On a file with **deletion vectors** it is
    neither: the counts are physical, so it overstates the compacted matrix the
    read returns and understates the peak, where both buffers are live at once. It was a flat 16 B —
    right for a wide matrix before the widened decode below landed, and 2×
    conservative for every matrix under the line, which never paid an upcast.
    So a file between roughly 0.5 and 1 billion nonzeros no longer trips the
    default 8 GiB budget. Eagerly-materialized `layers` are still not counted
    (they assemble `f32` and cast afterwards).
  - **Container / dtype materialization** (`container`, `data_dtype`,
    `index_dtype`, `allow_lossy`) — control the output container and numeric
    dtype of `X` (and layers). Eager (`backed=False`) only; a non-default
    request with `backed=True` raises (the backed dataset is lazy and
    f32-native). See [Container and dtype materialization](#container-and-dtype-materialization) below for the full reference.
    - `container` (default `"csr"`): `"csr"` → scipy `csr_matrix` (unchanged);
      `"dense"` → row-major `numpy.ndarray` (no scipy CSR).
    - `data_dtype` (default `None` → `float32`, today's behaviour): one of
      `float16` / `float32` / `float64` / `int8` / `int16` / `int32` /
      `int64` / `uint8` / `uint16` / `uint32`.
    - `index_dtype` (default `None` → `int32`): CSR column-index dtype
      (`int16` / `int32` / `int64`). Ignored (with a `RuntimeWarning`) for
      `container="dense"`.
    - `allow_lossy` (default `False`): the fail-loud cast gate. When `False`,
      any narrowing that would lose data (out-of-range, fractional-into-int,
      negative-into-unsigned, or a count above 2²⁴ into `float16`) raises
      `ValueError`; `True` performs the narrowing anyway.
    - The default (`container="csr"`, no dtype kwargs) is **byte-identical and
      zero-copy** — the Vec is moved into numpy with no cast. A non-default eager
      request narrows **in-decode**: `X` (and `adata.raw`) assemble directly at the
      target width, never building the full-matrix f32 CSR, so a narrow
      `data_dtype=` lowers peak RSS. `to_anndata(obs_filter=…, data_dtype=…)`
      decodes at the requested dtype too — that route resolves the plan before it
      collects. (Eager `layers` still cast post-assembly.)
  - Returns `obsm` (dense), `varm` (dense), `obsp` (scipy CSR), and
    `varp` (scipy CSR) when present in the file. When cells are
    logically deleted, `obsp` is subset to the kept rows and columns
    when the entry is decoded (via `filter_coo_obsp_by_kept_rows`), so
    the in-memory AnnData stays shape-consistent. `varp` and `varm`
    are unaffected (var axis has no deletion vector). The on-disk
    section retains the original axis until `compact` rebuilds the file.
- `to_mudata(backed=False, cache_shards=4, container=None, data_dtype=None, index_dtype=None, allow_lossy=False)` — Materialise a multimodal file as `mudata.MuData`
  - Eager (`backed=False`): per-modality scipy CSR AnnData sharing the
    global obs. Raises on single-modality files.
  - ⚠️ **Deletion vectors are not applied** — unlike `to_anndata()`, `query()`
    and every other materialising read, `to_mudata()` returns *physical* rows,
    so cells marked by `mark_deleted` are present. This is deliberate: every
    modality's X has to stay in lockstep with the single shared global obs, and
    the per-modality typed reader is unfiltered for that reason
    (`scx_format_io::read_all_csr_shards_for_typed`). Run `scx compact` first if
    you need the deletions materialised. `Experiment.has_deletions` tells you
    whether a given file is affected (and `n_obs` vs `n_obs_physical` by how
    much). See also
    [docs/multimodal.md](../multimodal.md) and
    [Operations § Deletion vectors](../operations.md#deletion-vectors-carried-vs-applied).
  - **Per-modality in-decode narrow.** `container` / `data_dtype` /
    `index_dtype` each accept **either a scalar** (applied to every modality)
    **or a dict keyed by modality name** — e.g.
    `data_dtype={"rna": "uint16", "atac": "uint8"}`. A modality with no
    override keeps the **byte-identical zero-copy `f32` CSR** path; a
    narrowed modality assembles directly at the target width via the typed
    per-modality reader (`scx_format_io::read_all_csr_shards_for_typed`),
    never building the intermediate f32 CSR. This matters most for
    multimodal reads: modalities differ in range (shallow/binarized ATAC and
    small ADT fit `uint8` = 4× on the value buffer; RNA/deep counts fit
    `uint16` = 2×) and `to_mudata` materialises *N* matrices at once. The
    same fail-loud cast gate applies **per modality** (a lossy narrow raises
    unless `allow_lossy=True`); integer→integer narrows are exact, including
    `> 2²⁴`. A dict key naming no modality in the file raises `ValueError`.
    `index_dtype` is accepted for symmetry but is a **no-op for the returned
    CSR** on any modality that fits in int32 — scipy resolves the width from
    the contents and canonicalizes in both directions (same as `to_anndata`,
    below).
    `container="dense"` is **not yet supported** for `to_mudata` (CSR only).
    The narrow kwargs require `backed=False`.
  - **Backed (`backed=True`)**: per-modality
    `ScxBackedSparseDataset` AnnData sharing the global obs (lazily
    f32-native — the narrow kwargs are rejected). Single-modality files are
    wrapped in a one-modality MuData rather than raising, so the call works
    uniformly across layouts.
- `query() -> PyQueryPipeline` — Start lazy query pipeline
- `mark_deleted(mask)` — Delete cells matching boolean array
- `validate()` — Check checksums, returns list of `(section_name, passed)`
- `detection_counts(axis="var", modality=None) -> np.ndarray`
  — Per-gene non-zero counts. Reads `BitmapShard` sidecars when
  present (one roaring decode per shard); falls back to a CSR scan
  otherwise. `axis="obs"` returns per-cell gene counts. For
  multimodal v2 files, pass `modality="rna"` to scope the result.
- `cells_expressing(gene, modality=None) -> np.ndarray` —
  Indices of cells with non-zero expression for `gene` (name or
  integer). Bitmap fast path when sidecars exist; CSR fallback
  otherwise.
- `gather_rows_sparse(rows, modality=None, cache_shards=4, layer=None, logical=True) -> scipy.sparse.csr_matrix`
  — The bounded shard-wise row gather, without building an AnnData. `rows` is a
  boolean mask or any 1-D integer array-like (list, `range`, ndarray of any
  integer dtype); duplicates allowed, order preserved, negative indices wrap
  once. Each touched shard is decoded once (a sparse request on a
  row-group-framed shard decodes only the touched row groups) and the result is
  assembled once into exact-size buffers, so peak memory is the result plus the
  shard cache, plus up to `cache_shards` shards decoding in flight while that
  cache fills (at most `2 × cache_shards` decoded shards beside the result) —
  never a second copy of the result.
  `logical=True` (default) indexes the rows `n_obs` / `read_obs()` describe
  (deletion vectors applied, exactly as `to_anndata(backed=True).X[rows]`);
  `logical=False` indexes physical file rows (`n_obs_physical`). **Changed in
  0.17**: before, this method addressed physical rows only, so on a file with
  deletion vectors the same ids now select different cells — pass
  `logical=False` for the old behaviour. `layer=`
  gathers from that layer instead of `X` (`ValueError` if absent; not supported
  on a multimodal file — index the modality's layer handle from `to_mudata()`).
  Out-of-range ids and a boolean mask whose length is not the row count raise
  `IndexError`. Gene indices are raw-local (no global-vocab remap). A fresh
  reader is opened per call (fork-safe; `cache_shards` bounds its memory, it is
  not a speedup knob). Equivalent to `adata.X[rows]` / `adata.layers[name][rows]`
  on a backed handle.
- `to_gpu_anndata(var_names=None, obs_filter=None, layers=None, obsm=None, device="gpu", memory_budget=None, preserve_var_order=False, strict_var_names=True, container="csr", data_dtype=None, index_dtype=None, allow_lossy=False, obsp=None, varp=None, varm=None, raw=True)` — Minimal-copy on-device handoff: decodes shards, transfers to the GPU, and returns a GPU-resident AnnData whose `X` is a `cupyx.scipy.sparse.csr_matrix`. The returned object is suitable for direct use with rapids-singlecell (`rsc.pp.*`, `rsc.tl.*`) without additional host↔device copies. Requires `cupy` and a CUDA-capable GPU. Records its `transfer_mode` and real `bytes_uploaded` on `uns["scx_accel"]["to_gpu_anndata"]` — `scx_device_decode_gpu` (Scx1 sidecar shards decoded fully in VRAM, including dense ≥128-nnz rows via the BitPacker4x kernel; only indptr uploaded), `scx_device_handoff_streamed` (some shard host-bounced because it is not an Scx1 sidecar shard — a non-Scx1 codec or a sidecar-less Scx1 shard), or `scx_device_handoff` (host-assembled filtered/projected/multimodal input). If the in-VRAM decode *fails* on a request that qualified for the fast path, the call does not raise: it falls through to the host-assemble path, which reaches the same `cupyx` `X` by a different road, and records `transfer_mode="scx_device_handoff"` with `fallback_reason="gpu_runtime_error"` plus a `UserWarning` naming the device error. An out-of-memory failure is excluded — both paths end with the same CSR resident on the device, so host-assemble cannot fix a VRAM shortfall and the `>VRAM` error is raised directly instead. See **Accelerator route metadata** below.

  **Memory semantics.** The result is the **complete** sparse matrix in VRAM — this is not a streaming/partial representation. Sparse CSR format is preserved throughout (VRAM scales with NNZ, not N×M). Shards are decoded one at a time into pre-allocated combined device buffers; peak device memory during transfer is the combined buffer plus one shard. A **VRAM pre-flight check** (1.2× headroom factor) compares the required bytes against free device memory and raises `ValueError` if insufficient, with an actionable message pointing to `backed=True` streaming workflows. See [gpu-setup.md § GPU memory model](../gpu-setup.md#gpu-memory-model) for sizing formulas.

  `to_gpu_anndata` accepts `container` / `data_dtype` / `index_dtype` / `allow_lossy` for signature parity with `to_anndata`, but the device path is **f32-native**: a non-default request raises `ValueError`. To obtain a narrow/dense matrix, materialize it on the host with `to_anndata(container=..., data_dtype=...)`.
- `info() -> str` — One-line codec / shard / format-version internals (kept off the AnnData-style `repr`).
- `reload()` — Re-open the file, picking up anything written since the handle was opened. See **Handles and files that change underneath them** below.
- `close()` — Release the file mapping. Idempotent; reads afterwards raise. Also available as a context manager (`with pyscx.open(p) as exp:`).
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `index_dtype`, `path`, `has_csc`, `n_csc_shards`, `has_deletions`, `closed`.
  `n_csc_shards` is the header's count of CSC sidecar sections (`0` when `has_csc` is false) — a header field, so it costs no I/O. It is the layout number: `has_csc` says a sidecar exists, `n_csc_shards` says how wide its column shards are, which is what decides how much a column-range read decodes. On a multimodal file it is the file-wide total; per-modality counts come from `modality_info()`.
- List-returning accessors — callable **methods** (not properties): `layer_names()`, and the AnnData-style key accessors `obs_keys()`, `var_keys()`, `obsm_keys()`, `obsp_keys()`, `varm_keys()`, `uns_keys()` (all cheap — schema/catalog reads, no matrix decode; `obs_keys()`/`var_keys()` exclude the pandas index column; `obsp_keys()` lists only the COO forms every conversion path writes, since the CSR-backed `ObspCsrShard` has no read API).
- `read_obsp_rows(key, start, stop, *, logical=True)` — rows `[start, stop)` of an `obsp` graph as a scipy CSR, decoding only the shards the range covers. The bounded counterpart to `to_anndata().obsp[key]`, which materialises the whole matrix. `logical=True` (the default, matching `read_obs`) takes the bounds in **live** row space, drops any edge whose either endpoint is deleted and renumbers both axes into live space; `logical=False` is the physical graph, unfiltered, with column extent `n_obs_physical`. ⚠️ A graph stored as one unsharded section is decoded whole whatever range is asked for. Every path that *writes* an `obsp` emits shards — `scx sort` / `scx compact` since phase 9, the h5ad and `from_anndata` paths always, `scx merge` per input section — so an unsharded graph means a file written before phase 9, or one whose graph came from the low-level `write_obsp` and was then carried by `scx optimize` (which copies a mapping section verbatim). `scx subset` drops the families outright. The obs-axis *embeddings* are a separate question with more unsharded producers: see [sharding.md § Obsm / varm / obsp / varp sharding](../sharding.md#obsm--varm--obsp--varp-sharding).

### Handles and files that change underneath them

An `Experiment` maps the file it was opened from, and no SCX write path ever
edits bytes a reader is looking at — an in-place op (`obs_import`,
`doublet_import`, `modify_metadata`, `set_uns`, `update_uns`, `mark_deleted`, `build_csc`,
`rollback`) appends and rewrites the header, and a copy-out op (`compact`,
`sort`, `merge`, `subset`) renames a new file into place. Either way the
mapping stays intact and readable while describing a file state that is no
longer on disk.

Reading through such a handle **raises** rather than answering:

```python
exp = pyscx.open("atlas.scx")
pyscx.obs_import("atlas.scx", "calls.csv", key="obs_names")

exp.read_obs()
# RuntimeError: 'atlas.scx' changed on disk since it was opened
# (manifest_sequence 1 → 2). This handle still maps the file as it was
# when it was opened … re-open the file to read the current ones
# (in pyscx: `Experiment.reload()`)

exp.reload()
exp.read_obs()      # now carries the imported columns
```

The rules, in full:

- **It covers the objects the handle hands out, not just the handle.**
  `adata = pyscx.open(p).to_anndata(backed=True)` drops the `Experiment` on the
  same line, and the backed `X` / `obsm` / layers, a `query()` pipeline, and a
  backed MuData each hold their own reader. All of them refuse a changed file.
- **`reload()` does not reach them.** It refreshes the `Experiment` only;
  re-derive anything taken out of it (`exp.to_anndata(backed=True)` again).
  Reviving them in place would silently mix arrays from two file versions.
- **Mutating *through* a handle is fine.** `pyscx.obs_import(exp, "calls.csv")`
  and `exp.mark_deleted(mask)` reload the handle for you.
- **`path`, `closed`, `close()`, `reload()` and `repr()` never raise** — a
  stale handle reprs as `<Experiment 'atlas.scx' [stale: …]>`. Everything else
  on the class does.
- **Only pyscx handles are watched.** `scx-ops`, the CLI and the ML training
  loader open readers around their own writes and are unaffected; the check
  costs them nothing.
- **Cloud handles are not covered.** `open_cloud` has no local mapping and no
  inode to compare; detecting a changed object would need a request per read.
- **Timestamps are not the signal.** The check compares the file's inode and
  its catalog pointer, so `touch`, a metadata-preserving copy, or a backup pass
  does not invalidate a handle — and, in the other direction, a mutation that
  leaves size and mtime untouched is still caught. (Both happen: Linux updates
  inode timestamps from a coarse clock, so a fast open→mutate sequence can land
  with an *identical* `st_mtime_ns`.)

`close()` releases the mapping without waiting for the garbage collector, which
is what you want before rewriting a file in place, and is required on Windows,
where a mapped file cannot be replaced at all:

```python
with pyscx.open("atlas.scx") as exp:
    obs = exp.read_obs()
pyscx.compact("atlas.scx", "atlas.scx")   # mapping already released
```

Objects taken out of a `with` block keep their own readers and stay usable
after it exits; closing the handle you opened them from does not close them.


## Container and dtype materialization

The read APIs materialize `X` (and layers) as a scipy `csr_matrix` of
`(int64 indptr, int32 indices, float32 data)` by default. The `container`,
`data_dtype`, `index_dtype`, and `allow_lossy` kwargs let a reader choose the
output container and numeric dtype directly, so downstream consumers
(sklearn / PyTorch / scVI, GPU batches) don't over-allocate or re-densify.

Surfaced on `Experiment.to_anndata`, `PyQueryResult.to_anndata`, and
`PyQueryResult.to_csr` (`Experiment.to_gpu_anndata` accepts them for parity but
rejects any non-default request — the device path is f32-native).
`QueryPipeline.collect` takes the two that decide the *decode* — `data_dtype` and
`allow_lossy`, plus `index_dtype` — and not `container`, which is a presentation
choice applied afterwards and cannot lose anything.

| kwarg | values | default | notes |
|-------|--------|---------|-------|
| `container` | `"csr"` \| `"dense"` | `"csr"` | `"dense"` returns a row-major `numpy.ndarray` (no scipy CSR) |
| `data_dtype` | `float16/32/64`, `int8/16/32/64`, `uint8/16/32` | `None` → `float32` | numeric dtype of the values |
| `index_dtype` | `int16` \| `int32` \| `int64` | `None` → `int32` below `i32::MAX` nonzeros, `int64` above (pyscx does not choose `int64` on a file with deletion vectors; scipy still may) | CSR column-index dtype; ignored (warns) for `"dense"`. **Note:** `csr_matrix` resolves the index dtype from `max(nnz, n_rows)` and ignores the width it was handed, so on a matrix that fits in int32 **neither** `int16` nor `int64` survives — int16 is upcast, int64 is downcast, both to int32. The narrow int16 buffer is still built and range-gated on the way through (a column index ≥ 32768 fails loud). The default is resolved per matrix (`X` and `adata.raw` decide separately) — see **int64 above 2³¹ nonzeros** below |
| `allow_lossy` | `bool` | `False` | fail-loud cast gate — see below |

**Zero-copy default preserved.** `container="csr"` with no dtype kwargs takes the
exact pre-existing path: the decoded `Vec`s are moved into numpy with `copy=False`
and no cast. This is guaranteed byte-identical and is the performance-sensitive
common case.

**int64 above 2³¹ nonzeros.** `csr_matrix` resolves **one** index dtype for
`indices` and `indptr`, from `max(nnz, n_rows)` rather than from the arrays it
was handed. Above `i32::MAX` it picks int64 whether or not anyone asked — and
gets there by **copying** the int32 array pyscx handed it, while that array is
still alive. Measured on a 960,195 × 6,143 file with 2,650,704,199 nonzeros: a
16.3 B/nnz assembly transient settling to 11.9 B/nnz. The default eager read
therefore decodes indices at int64 directly once the matrix is over that line,
via the same typed reader a non-default `index_dtype=` uses. The returned scipy
object is identical — same values, same dtypes, still `copy=False` — and no
whole extra index array is alive at the peak.

The gate matters in both directions: at or below the line scipy **downcasts**
int64 inputs, so widening a matrix that does not need it would add a copy rather
than remove one. That is also why it is **off entirely on a file with deletion
vectors**: those are applied after assembly, so the catalog's nnz is an upper
bound on what scipy is handed, and a matrix whose physical nnz is over the line
but whose live nnz is under it would pay an oversized index buffer *and* the
downcast copy. The live count is not derivable from the catalog, so such files
keep the pre-existing behaviour exactly. `X` and `adata.raw` decide
independently, because scipy decides per `csr_matrix`. An explicit
`index_dtype=` is never overridden, and `container="dense"` has no column-index
array to widen.
`SCX_EAGER_INT64_NNZ_THRESHOLD` overrides the threshold; it exists so a small
fixture can exercise the widened decode, not as a tuning knob — the threshold is
scipy's, and below it the widen is a pessimization.

**In-decode narrow (eager `X` / `raw`).** A non-default request on the eager
`to_anndata` path narrows **in-decode**: each shard is decoded to its native
stream (integer counts as `u32`, floats as `f32`) and cast straight into a
full-matrix buffer *of the target dtype*, so the intermediate f32 CSR is never
allocated. A narrow `data_dtype="uint16"` read therefore **lowers** peak RSS
(2 B/nnz for the value buffer, not 4 B/nnz + a narrow copy) — see
[performance/vs-shardad.md § Full read → AnnData](../performance/vs-shardad.md#full-read--anndata-wall-s--peak-rss-mb).
Eagerly-materialized **layers** still cast post-assembly (correct, no RSS win),
because there is no typed layer reader yet. `to_gpu_anndata` is unchanged
(f32-native device path).

**Fail-loud cast gate (`allow_lossy`).** With `allow_lossy=False` (the default),
any narrowing that would lose data raises `ValueError` rather than silently
corrupting values:

- out-of-range for the target integer dtype (e.g. `300 → uint8`),
- a fractional value into an integer dtype (e.g. `1.5 → int32`),
- a negative value into an unsigned dtype (sign loss, e.g. `-1 → uint16`),
- a count above 2²⁴ into `float16` (IEEE-754 cannot represent it exactly).

The error names the offending value and suggests a wider dtype or
`allow_lossy=True`. Widening casts (e.g. `uint8 → float32`, the default) are
always safe and are never gated.

**Decode-loss guard (the `u32 → f32` case).** The guard fires per *target dtype*:
a read fails loud when the shards' `value_max` cannot be represented exactly in
the requested `data_dtype`. Because the eager path now narrows **in-decode**
(integer counts cast straight from the native `u32` stream, never through f32), a
`> 2²⁴` integer count read into an **exactly representable** dtype
(`uint32` / `int64` / `float64`) now **succeeds losslessly** — where it previously
failed loud. The guard still fires (unless `allow_lossy=True`) for targets that
genuinely cannot hold the value: the plain `to_anndata()` default (`float32`, exact
only to 2²⁴), and `float16` (exact only to 2¹¹). Notes:

- The guard is **integer-encoding-only** and O(1): float-encoded shards record
  `value_max = 0` in the catalog, so continuous / log-normalized data never trips
  it, and the check is a single per-shard comparison (no data scan).
- For `float32` it is **conservative**: the catalog carries only the per-shard
  maximum, so a shard whose max exceeds 2²⁴ trips the guard even if that particular
  value is itself f32-exact. Read into `uint32` / `int64` / `float64` (exact), or
  pass `allow_lossy=True`, to bypass.
- The **query** path decodes at the requested dtype **when the dtype is declared
  before the decode** — `to_anndata(obs_filter=…, data_dtype=…)`, which resolves
  the plan and collects in one call, or `query().collect(data_dtype=…)`. A `> 2²⁴`
  count read that way is exact. A dtype named *after* a plain `collect()` is a
  **cast of values that were already decoded as f32**, so it fails loud and the
  message says where the dtype belongs; `allow_lossy=True` still accepts the
  rounding. See [Declaring the dtype at `collect()`](python-query.md#declaring-the-dtype-at-collect).
- A **`var_names=` projection** assembles f32 too (it streams shard by shard through
  the same projecting reader the backed handles use, which has no typed variant), so
  a narrow `data_dtype=` takes that route only when the file's values survive an f32
  round trip — `value_max ≤ 2²⁴`, which includes every float-encoded file, since
  those record `value_max = 0`. Above that the projection stands down and the read
  falls back to today's full-width assemble-then-slice: same values, same peak as
  before the projection existed. Nothing is silently rounded either way. In practice
  the fallback is unreachable through pyscx's own write doors — both `from_anndata`
  and h5ad ingest route `X` through f32, so a count above 2²⁴ that is *not*
  f32-exact cannot be written from Python; it exists for files other scx tooling
  produces.

**Which matrices are guarded.** The guard covers every eagerly-decoded count
matrix: `X` (all `to_anndata` paths — default, `var_names`, `obs_filter`,
`preserve_slots`, and `to_gpu_anndata`), `adata.raw`, eagerly-materialized
`layers` (`to_anndata(eager=True)`), and each modality's `X` in
`to_mudata()`. The following are **not** gated — they decode lazily per-slice,
so a whole-file check would spuriously error on partial reads that never touch
the large-count shard:

- **Backed reads** (`backed=True`, including `to_mudata(backed=True)`).
- **Lazy `layers`** on the default (`eager=False`) `to_anndata()` — the layer
  is decoded only on later `adata.layers[...]` access.

The layers guard folds `value_max` over **the selected layers only**, so
`to_anndata(layers=["cpm"])` is refused by a `> 2²⁴` count in `cpm` and never by
one in a layer the call does not read. (`layers=` forces eager assembly, so a
filtered read is always a guarded read.) That holds at all three eager-layer
sites — the bridge branch, the `var_names=` projected assemble, and the
post-assembly narrow a non-default `data_dtype=` triggers — so none of them
disagrees about when a read raises. The retype site reads the layer keys off the
assembled object rather than re-deriving the filter, which is what keeps it in
step with the f32 decode guard that already ran.

For those ungated paths, a `> 2²⁴` count still rounds silently on access; pass
`allow_lossy` where available, or read eagerly to get the guard. To *know*
before reading: `Experiment.is_integer` / `Experiment.max_value` (catalog
stats, no decode) say whether a file holds counts and how large they get, and a
backed handle's `stored_dtype` names its on-disk encoding.

The R bindings (`rscx`) wire the same guard behind the same `allow_lossy`
opt-out: eager reads — `$x_matrix()`, `$layer()`, `$to_seurat()` / `$to_mae()`
(per modality) — use the shared catalog `value_max` folds on `FullCatalog`
(`csr_max_value` / `layer_csr_max_value`), the same implementation pyscx's
eager reads use; the R query path, like Python's, guards on the
engine-computed max over the shards the query actually selected, not a
catalog-wide fold.

> **Note.** A scipy `csr_matrix` with `float16` `data` is valid but cannot be
> densified by scipy's own `.toarray()` (a scipy limitation) — call
> `.astype(np.float32).toarray()`, or request `container="dense"` directly.

Backed reads (`backed=True`) are lazy and f32-native, so a non-default plan with
`backed=True` raises; materialize eagerly (`backed=False`) to narrow.

**Caveats.**

- **`index_dtype` for CSR is best-effort.** scipy canonicalizes a
  `csr_matrix`'s index arrays on construction (typically to `int32`, `int64`
  above the 32-bit nnz/dimension limit), so `index_dtype="int16"` will usually be
  upcast back to `int32` and delivers no reliable memory saving for CSR output.
  The `int64` widen sticks. For a guaranteed narrow-index layout, use
  `container="dense"` (no index array) instead.
- **`adata.raw` is retyped in-decode.** When the file carries a raw count
  matrix, the reconstructed `adata.raw.X` is assembled directly at the
  requested `data_dtype` (same typed reader as `X`) and gated by the same
  decode-loss check; it stays CSR even under `container="dense"` (the
  conventional raw representation).
- **Scope.** The kwargs are surfaced on the three read entry points above, on
  `QueryPipeline.collect` (`data_dtype` / `index_dtype` / `allow_lossy` — see
  [Declaring the dtype at `collect()`](python-query.md#declaring-the-dtype-at-collect)), and on
  the flat `pyscx.read_cloud(...)` cloud helper, which is a one-call query and so
  takes them for the same reason `collect()` does. The grouped-shard read helpers
  (`read_group` / `read_reference` / `iter_group_shards`) are f32-native, and say
  so in their refusal rather than recommending a `data_dtype` they do not accept.

## `uns` serialization

`adata.uns` is written into the `UnsBlob` section (id 10) as JSON. The
`uns_format` kwarg on `from_anndata()` / `from_10x()` selects the envelope:

| Mode | Default | NumPy ndarray | NumPy scalar | tuple | pandas Cat/Index/Series | pandas DataFrame | NaN/Inf in array |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `"tagged"` | ✓ | `__scx_type__: "ndarray"` envelope (base64-LE bytes for numeric; JSON string list for object/string; `__scx_type__: "recarray"` for structured) | `__scx_type__: "scalar"` envelope (1-element base64) | `__scx_type__: "tuple"` envelope | `__scx_type__: "categorical" / "pandas.Index" / "pandas.Series"` envelope preserving name/codes/categories/ordered | `__scx_type__: "pandas.DataFrame"` envelope: a `pandas.Index` envelope for the index, an explicit ordered `columns` list, and one `ndarray` / `categorical` envelope per column | preserved bit-exact — ndarray bytes, and a bare non-finite Python `float` (anywhere: top level, list, tuple, dict, structured-array field, pandas `name`) becomes a float64 `scalar` envelope that reads back as `np.float64` |
| `"plain"` |  | collapses to nested list (`.tolist()`) | collapses to Python scalar | collapses to JSON array | collapses to JSON array via `.tolist()` | raises `ValueError` | raises `ValueError` |

`uns` is one JSON document, parsed whole on every `to_anndata()`, every
`Experiment.read_uns()` and every `open_cloud(...).read_uns()`, and rewritten
whole by every in-place `set_uns` / `update_uns` / `attach_obs_columns(uns=)`
(which leaves the superseded section behind — `scx info` reports the
fragmentation). That is worth knowing before putting a large table there: a
`rank_genes_groups(pts=True)` pair on a 60 k-gene × 30-group atlas is roughly
29 MB of raw float64, ≈ 38 MB once base64-encoded. Numeric columns always take
the base64-LE `ndarray` form (≈ 10.7 B/value) rather than a JSON number list
(≈ 18 B/value and 2–3× the parse time), but the whole-document cost is
inherent to the section.

The reader **auto-detects** per value: dicts with a `__scx_type__` key are
decoded back to their original Python type; everything else passes through
as plain JSON. This means files written by older `pyscx` (or with
`uns_format="plain"`) read identically on a modern build, and modern
tagged files are forward-compatible — unknown future tags emit a
`UserWarning` and return the raw envelope dict for inspection.

Unsupported in both modes: `bytes` objects (no portable JSON
representation) and `datetime64` / `complex` / `timedelta` ndarray dtypes.
For a `DataFrame` specifically, also unsupported: a `MultiIndex` on either
axis, a non-string or duplicated column name (names key the envelope's `data`
object, so a duplicate would silently collapse two columns into one), and any
pandas extension dtype other than `category` (`Int64`, `boolean`,
`string[python]`, …) — cast it first. Each raises a `ValueError` naming the
column. One documented gap: a frame with **zero columns** reads back with
pandas' default empty `RangeIndex` for `columns`, since with no names there is
nothing for a columns dtype to travel on.

The **h5ad export** writes the anndata dataframe group directly and keeps every
column's exact dtype — an `int8` column lands as `int8`, a `bool` as a plain
`bool` dataset. It is all-or-nothing per frame: if any column has no faithful
anndata spelling (a `null` inside an object column, which a plain h5ad string
dataset cannot hold; a column whose name equals the index's, which would
collide in the group; a name HDF5 cannot carry as a single member — empty,
`.`, `..`, or containing `/`; a non-string index name, which h5ad has nowhere
to put), the
whole frame is written as a raw `__scx_type__` envelope subgroup instead, with
an `uns_exported_as_raw_envelope` warning. anndata reads the subgroup as a
nested dict, so DataFrame consumers will not recognise it. Declining beats
dropping the column, because the fallback keeps what dropping would discard —
every value **whose key HDF5 can carry**. A column named `"/evil"` or `""` has
no HDF5 member spelling anywhere, so the fallback drops it too, with its own
`skipped_uns_key` warning.
Non-finite raw Python `float` scalars (`nan` / `±inf`) are preserved under
`"tagged"` via the `scalar` envelope (they read back as `np.float64`, a
`float` subclass, bit-exact) and raise under `"plain"`, whose contract is
"lossless or refuse". Finite doubles round-trip bit-exact in both modes: the
writer is ryu-exact and every reader parses with `serde_json`'s
`float_roundtrip`, so a 17-significant-digit value comes back as the same
IEEE-754 pattern.

The tagged envelope is plain JSON, so the section can be inspected with
any JSON tool. Example for a `float32` array:
```json
{
  "pca_variance": {
    "__scx_type__": "ndarray",
    "dtype": "float32",
    "shape": [50],
    "encoding": "base64le",
    "data": "zczMPc3MTD4AAIA/..."
  }
}
```
