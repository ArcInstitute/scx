# Python: backed and lazy datasets

> Part of the [SCX API reference](README.md). For the usage guide, see
[docs/scanpy/backed-mode.md](../scanpy/backed-mode.md) and
[docs/scanpy/lazy-preprocessing.md](../scanpy/lazy-preprocessing.md).

## ScxBackedSparseDataset

PyO3 class for backed-mode lazy access to the main expression matrix (`adata.X`). Data stays on disk; only requested shards are decoded on access. Registered with `anndata.abc.CSRDataset`.

**Properties:**
- `shape` `→ (int, int)` — `(n_obs, n_vars)`, adjusted for deletion vectors and column projection
- `dtype` `→ numpy.dtype` — Always `float32`: the type every read decodes to (scipy CSR interop, anndata's `CSRDataset` expectations). The on-disk encoding is `stored_dtype`.
- `stored_dtype` `→ numpy.dtype` — The on-disk value encoding of this handle's shards: `uint8` / `uint16` / `uint32` for integer counts, `float32` / `float16` for continuous data. When shards mix (an `append` keeps each shard's own encoding) it is the widest — any float ⇒ `float32`, else the widest integer; a file with no shards reports `float32`. One 76-byte header read per shard, no decode, memoised on the reader, so `stored_dtype.kind in "ui"` answers "are these counts?" in O(shards). A layer handle reports the layer's own family; a lazy handle reports its *source*'s encoding (the transforms produce floats on read regardless).
- `cache_shards` `→ int` — The decoded-shard LRU size this handle was opened with (`to_anndata(backed=True, cache_shards=…)`, default 4). `0` is the **uncached** path — every read decodes afresh and retains nothing — and is the right setting when the streaming footprint must be one shard. Read-only: the count is fixed when a handle's reader is built, and one `to_anndata` call builds `X` and each layer's reader with the same count (there is no `cache_shards` on `pyscx.open`). Note the split: `to_anndata(cache_shards=0)` is legal and meaningful, while `IndexPlanDataset(cache_shards=0)` raises `RuntimeError` on purpose (the loader's prefetcher needs a cache to prefetch into).
- `format` `→ str` — Always `"csr"`
- `backend` `→ str` — Always `"scx"`
- `ndim` `→ int` — Always `2`
- `non_negative` `→ bool` — Whether the data is known to be non-negative (enables `(X > 0).sum() → getnnz()` short-circuit)
- `nnz` `→ int` — Total non-zero count
- `n_shards` `→ int` — Number of CSR shards in the backing file
- `__array__(dtype=None, copy=None)` — **Raises `TypeError`.** numpy's array protocol is implemented only to refuse: `np.asarray(adata.X)` on a handle would decode the whole `n_obs × n_vars` matrix at once, and before this it returned a 0-d object array that failed far away ("setting an array element with a sequence"). Call `to_memory()` (scipy CSR) or `toarray()` (dense) — both work on a column-projected handle such as `X[:, genes]`, which is itself a handle and refuses `np.asarray` the same way — take a row window with `handle[rows]` (a scipy CSR), or open the file with `pyscx.open(path).to_anndata()` for an in-memory AnnData. The dense `ScxBackedObsmDataset` keeps a materialising `__array__` — an embedding is small.

**Column projection:**
- `set_col_projection(col_indices)` — Restrict all access and aggregation to a subset of columns. Used internally by `to_anndata(var_names=...)` — on the backed path to project the handle, and on the eager path to assemble `X` and each layer already projected — and by streaming QC with gene subsets (`qc_vars`).

  > [!WARNING]
  > **This is a handle-level knob, not an axis subset.** It moves `X` only — `var`, `layers`, `varm` and `varp` are left at the old width, so the AnnData is inconsistent until you slice them yourself. Use `pyscx.accel.subset_var(adata, mask)` (or `adata[:, mask]`) for a real gene subset — see [Axis subsetting and aligned members](python-accel.md#axis-subsetting-and-aligned-members). Reach for this only when you want to reproject a bare handle.

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, return scipy CSR.
- `__getitem__(rows, cols)` — With a non-`:` row selector: the row gather below, then a scipy column slice on the result (materialises the selected rows only). With `:` rows: the column projection below.
- **`handle[:, cols]` is a column projection, not a read.** Every column
  selector form — an `int` (`X[:, 5]` is an `(n_obs, 1)` handle, as scipy's is
  `(n, 1)`), a `list` / `range` / `tuple`, an integer ndarray of **any order**
  (signed negatives wrap once; unsigned is bounds-checked as `uint64`), a
  boolean mask of length `n_vars`, or a non-full `slice` (`X[:, 10:20]`,
  `X[:, ::-1]`) — is resolved once and composed through the handle's current
  window, and a new `ScxBackedSparseDataset` comes back with **no decode**: one
  decode per touched shard happens later, when that handle is read or
  aggregated. Ascending-unique selectors install a sorted projection; a
  reordered-but-unique selector rides on the presentation permutation
  (`sum(axis=0)`, `to_memory()` and every read honour the request order), and
  composing on top of an existing presentation (`X[:, [7, 2, 11]][:, [2, 0]]`
  → columns `[11, 7]`) is exact. Two forms are not handles: a selector with
  **repeated** columns (`X[:, [3, 1, 3]]`) — a permutation cannot express a
  repeat, so it materialises the *projected unique columns* (one `to_memory()`
  over 2 columns here, never the whole matrix) and gathers them with scipy —
  and the full `X[:, :]`, which returns the row CSR like `X[:]` (scipy's
  `X[:, :]` is a copy too). An out-of-range column, a wrong-length mask, a
  float or 2-D selector raise `IndexError` (numpy's rule; a float selector
  used to slip through and decode everything). Before 0.17 only an ascending
  int / bool ndarray projected; every other form decoded the whole matrix and
  sliced it.
- **`handle[rows]` is the bounded row gather.** A boolean mask or any 1-D
  integer array-like (unsorted, duplicates, negative indices wrapping once) is
  resolved in user-visible row space and gathered in request order by
  `BackedCsrReader::read_row_indices`: each touched shard is decoded once — a
  sparse request on a row-group-framed shard decodes only the touched row groups
  — and the result is assembled once into exact-size buffers, so peak memory is
  the result plus the shard cache, plus up to `cache_shards` shards decoding in
  flight while that cache fills (at most `2 × cache_shards` decoded shards
  beside the result), never a second copy of the result. `X[:]` and other
  contiguous slices on a handle **without** deletion vectors go through
  `read_rows`, which sizes the result exactly and, for a range larger than the
  shard cache, decodes uncached in parallel chunks of `cache_shards` on top of
  whatever the LRU already holds — `X[:]` costs what `to_memory()` costs. With
  deletion vectors, `X[:]` is the row gather over the kept rows (still one
  decode per shard and one assembly, but through the LRU rather than the
  uncached bulk path). An out-of-range row, and a boolean mask whose length is
  not the row count, raise `IndexError` (numpy's rule) rather than returning a
  shorter matrix. Deletion vectors and column projection compose with all of
  this. The same gather is available without an AnnData as
  `Experiment.gather_rows_sparse`.

**Aggregation (streaming, no materialization):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums via native Rust streaming.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance (two-pass).
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts per column or row.
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards → full CSR (one exact-size assembly, `read_all`). On a handle with a **column projection** it assembles shard by shard instead: each shard is decoded once, projected (and row-filtered) while still one shard wide, and the narrow pieces are concatenated — peak is 2× the *projected* result plus one shard, never the whole matrix (it used to decode everything and project afterwards, so `X[:, [1, 3]].to_memory()` peaked at the full matrix). Sequential by design; parallel decode would hold every shard at once.
- `copy()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()`.
- `toarray()` `→ numpy.ndarray` — Dense array.
- `tocsr()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()` (scipy compat).
- `tocsc()` `→ scipy.sparse.csc_matrix` — Materialize and convert to CSC.
- `.A` `→ numpy.ndarray` — Dense array property (scipy compat).

**Comparison operators:**
- `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` — Return `ScxComparisonResult` for lazy boolean operations.

**Arithmetic:**
- `__truediv__(other)` — If `other` is a per-row vector, returns `ScxLazyTransformedDataset` with `RowScale(1/factors)` (lazy). Otherwise materializes.
- `__mul__(other)` — If `other` is a per-row vector, returns `ScxLazyTransformedDataset` with `RowScale(factors)` (lazy). Otherwise materializes.

  *Per-row vector* means an **unambiguously row-oriented** operand: a 1-D `(n_obs,)` array or an explicit `(n_obs, 1)` column vector. A `(1, n_obs)` row vector is treated as a per-*column* broadcast (numpy/scipy semantics), **not** intercepted as a transposed row scale — on a non-square matrix it falls through to scipy and raises a shape error rather than being silently mis-applied. On a **square** matrix (`n_obs == n_vars`) a bare 1-D `(n,)` operand is orientation-ambiguous (could be per-gene), so it is not intercepted and instead materializes; pass an explicit `(n_obs, 1)` column vector to force the lazy row-scale path. scanpy's `normalize_total` reshapes its row factors to `(n_obs, 1)`, so that path stays lazy.
- `__add__(other)` — Materializes and adds.
- `__sub__(other)` — Materializes and subtracts.
- `__matmul__(other)` — Matrix multiply (materializes).
- `multiply(other)` — Element-wise Hadamard product (materializes).
- `power(n)` — Element-wise power (materializes).

**Introspection:**
- `shard_boundaries()` `→ list[(int, int)]` — `(row_start, row_end)` pairs in user-visible row space, one per on-disk shard that still has visible rows. **Tiling contract** (relied on by `pyscx.iter_chunks(chunk_size="shard")`): `b[0][0] == 0`, `b[-1][1] == n_obs`, and `b[i][0] == b[i-1][1]` — the pairs tile `[0, n_obs)` exactly, with no gaps or overlap. With deletion vectors the counts exclude deleted rows, and a shard whose rows are all deleted is omitted, so `len(b)` may be less than `n_shards`.

## ScxBackedLayerDataset

PyO3 class for backed-mode layer access (e.g., `adata.layers["raw_counts"]`). Wraps a `ScxBackedSparseDataset` for a named layer. Registered with `anndata.abc.CSRDataset`.

- Same interface as `ScxBackedSparseDataset` (`shape`, `dtype`, `format`, `backend`, `ndim`, `__getitem__`, `to_memory`, `toarray`, `tocsr`, `tocsc`, `copy`, `sum`, `mean`, `var`, `getnnz`, `max`, `min`, `shard_boundaries`)
- `layer_name` `→ str` — Name of the backing layer
- `stored_dtype` `→ numpy.dtype` — The **layer's** on-disk encoding (its reader walks the layer's shard family, not X's), so `adata.layers["counts"].stored_dtype` can be `uint16` while a float `X` reports `float32`. `cache_shards` is this layer reader's setting — one `to_anndata` call builds `X` and each layer's own reader with the same count. `__array__` raises `TypeError`, as on `X`.
- `adata.layers[name][rows]` is the same bounded row gather as on `X` (one decode per touched shard, result assembled once, `IndexError` semantics as above) — the layer's own shard family is read, so gathering counts from a layer needs no round trip through the `Experiment`.
- `adata.layers[name][:, cols]` is the same column projection as on `X` and comes back as a `ScxBackedLayerDataset` (the wrapper, and with it `layer_name`, is kept — it used to return the bare inner class).

## ScxComparisonResult

Lazy comparison result returned by `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` on `ScxBackedSparseDataset` and `ScxLazyTransformedDataset`. Exposed as `_ComparisonResult` in Python.

**Key optimization:** `(X > 0).sum(axis)` short-circuits to `getnnz(axis)` without materializing the full boolean matrix. This is the critical path for `sc.pp.calculate_qc_metrics()`, `sc.pp.filter_cells()`, and `sc.pp.filter_genes()`. The short-circuit only activates for non-negative data (raw counts, normalized, log1p).

- `shape` `→ (int, int)`, `dtype` `→ numpy.dtype` (bool), `ndim` `→ int` (2)
- `sum(axis=None)` — Short-circuits `(X > 0).sum()` → `getnnz()` for non-negative data; otherwise materializes.
- `getnnz(axis=None)`, `mean(axis=None)` — Materialize and delegate.
- `toarray()`, `tocsr()`, `tocsc()` — Materialize to dense/sparse.
- `multiply(other)` — Element-wise product (materializes).
- `.A` `→ numpy.ndarray` — Dense array property.

## ScxLazyTransformedDataset

PyO3 class wrapping `ScxBackedSparseDataset` with chained per-row transforms. Created by `pyscx.accel.normalize_total()` and `pyscx.accel.log1p()`. Implements the same interface as `ScxBackedSparseDataset` and is registered with `anndata.abc.CSRDataset`.

**Properties:**
- `shape` `→ (int, int)` — `(n_obs, n_vars)`
- `dtype` `→ numpy.dtype` — Always `float32`
- `stored_dtype` `→ numpy.dtype` — The on-disk encoding of the **source** matrix, before the transforms (`normalize_total` / `log1p` produce floats on read regardless); `uint16` on a counts file.
- `cache_shards` `→ int` — The LRU size of the reader this handle shares with the `ScxBackedSparseDataset` it was derived from.
- `format` `→ str` — Always `"csr"`
- `ndim` `→ int` — Always `2`
- `backend` `→ str` — Always `"scx-lazy"` (distinguishes from `ScxBackedSparseDataset.backend` which is `"scx"`)
- `non_negative` `→ bool` — Whether the transformed data is non-negative
- `__array__(dtype=None, copy=None)` — Raises `TypeError`, as on `ScxBackedSparseDataset` (the transforms would have to run over the whole matrix to answer).

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, apply all transforms in order, return scipy CSR.
- `__getitem__(:, cols)` — The same column selector forms as on `ScxBackedSparseDataset` (`int`, `list`, `range`, `slice`, any-order ndarray, bool mask; same `IndexError` rules). An **ascending-unique** selection (after composing through the current projection) is a projected `ScxLazyTransformedDataset` with no decode. This class stores its projection sorted and has no presentation permutation, so a **reordered or repeated** request (`X[:, [7, 2]]`, `X[:, [3, 1, 3]]`) materialises the projected *unique* columns — transforms applied, `to_memory()` over just those columns — and gathers them with scipy; it never decodes the whole matrix. (Reorder through `adata[:, idx]` on a lazy `X` still raises, because `var` and `X` must be sliced by one rule there.)
- `__getitem__(rows, cols)` — Row gather + transform, then a scipy column slice.
- A boolean mask or integer array-like is the same bounded row gather as on `ScxBackedSparseDataset` (one decode per touched shard, result assembled once in request order, `IndexError` for out-of-range rows or a wrong-length mask), with each output row's transform parameters looked up by its global row id — duplicates and unsorted requests included.

**Aggregation (streaming through transforms):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums of transformed data.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means of transformed data.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance. `axis=0` is two-pass streaming; `axis=1` and `axis=None` **materialize** the visible matrix via `to_memory()` and compute `E[X²] − E[X]²` in f32, so they are neither out-of-core nor clamped at zero (a near-constant row can return a small negative). Prefer the backed `X.var(axis=1)`, which streams in f64 and clamps.
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts (unchanged by normalize/log1p).
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max of transformed data.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min of transformed data.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards + apply transforms → full CSR. With a column projection it assembles shard by shard (decode → transforms → project → drop deleted rows, one shard at a time, then concatenate), so it peaks at 2× the projected result plus one shard rather than the whole transformed matrix.
- `copy()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()`.
- `toarray()` `→ numpy.ndarray` — Dense array (via `to_memory().toarray()`).
- `tocsr()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()` (scipy compat).
- `tocsc()` `→ scipy.sparse.csc_matrix` — Materialize and convert to CSC.
- `.A` `→ numpy.ndarray` — Dense array property (scipy compat, same as `toarray()`).

**Comparison operators:**
- `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` — Return `ScxComparisonResult` for lazy boolean operations.

**Arithmetic:**
- `__truediv__(other)` — If `other` is a per-row vector, appends `RowScale` transform (lazy). Otherwise materializes.
- `__mul__(other)` — If `other` is a per-row vector, appends `RowScale` transform (lazy). Otherwise materializes.

  *Per-row vector* is defined identically to `ScxBackedSparseDataset` above: `(n_obs,)` or `(n_obs, 1)` are intercepted lazily; `(1, n_obs)` and ambiguous square-matrix `(n,)` operands fall through to materialization.
- `__add__(other)` — Materializes and adds.
- `__sub__(other)` — Materializes and subtracts.
- `__matmul__(other)` — Matrix multiply (materializes).
- `multiply(other)` — Element-wise Hadamard product (materializes).
- `power(n)` — Element-wise power (materializes).

**Introspection:**
- `nnz` `→ int` — Total non-zero count in backing file.
- `n_shards` `→ int` — Number of CSR shards in backing file.
- `shard_boundaries()` `→ list[(int, int)]` — Same tiling contract as `ScxBackedSparseDataset.shard_boundaries()` (user-visible row space; `b[0][0] == 0`, `b[-1][1] == n_obs`, contiguous). `pyscx.iter_chunks` uses it on a lazily transformed `X` too.

**Transform chain:**
- Backed data → `NormalizeTotal` → `Log1p` is fused into `ln(x × target_sum / row_sum + 1)` in a single pass.
- `repr()` shows the transform chain: `ScxLazyTransformedDataset(shape=(1000000, 33694), transforms=[NormalizeTotal, Log1p])`
