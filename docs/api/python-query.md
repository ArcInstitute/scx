# Python: queries and cloud handles

> Part of the [SCX API reference](README.md).

## PyQueryPipeline

- `filter_obs(expr)` / `filter_var(expr)` — Predicate filtering
- `select_genes(indices)` — Gene projection
- `with_normalize(target_sum=1e4)` — Total-count normalization
- `with_log1p()` — Log1p transformation
- `limit(n)` — Row limit
- `collect(*, data_dtype=None, index_dtype=None, allow_lossy=False) -> PyQueryResult` — Execute pipeline. The kwargs are keyword-only and choose the dtype `X` is **decoded at** (see [Declaring the dtype at `collect()`](#declaring-the-dtype-at-collect)); the default is the unchanged zero-copy f32 decode. `data_dtype="float32"` takes the default route too, so the fused transforms keep working.
- `count() -> int` — Matching cell count, without decoding `X` (ignores `limit`)
- `exists() -> bool` — Whether any cell matches, without decoding `X`

**Consumption semantics.** Builder steps mutate the pipeline in place and return it, so both
`p.filter_obs(...).limit(5)` and the statement form work on one object. A **failed builder step
leaves the pipeline unchanged and usable** — a bad predicate (`ValueError`) or an unknown gene name
(`KeyError`) neither applies partially nor invalidates the object, so an interactive typo can simply
be retyped. `count()` and `exists()` borrow. `collect()` consumes the pipeline **only when it
succeeds**; a failed `collect()` leaves it usable, so the offending step can be corrected and
re-collected. After a successful `collect()` every method raises `RuntimeError` — build a new
pipeline with `Experiment.query()`.

## PyQueryResult

- `to_anndata(container="csr", data_dtype=None, index_dtype=None, allow_lossy=False)` — Convert result to AnnData. The default is a zero-copy CSR; the four kwargs have the same semantics as [`Experiment.to_anndata`](python-experiment.md#experiment) (`"dense"` container, narrow `data_dtype`/`index_dtype`, fail-loud `allow_lossy` gate), with one difference described below: here they **cast** what `collect()` already decoded. Any non-default request forgoes zero-copy (a cast/copy of `X`).
- `to_csr(container="csr", data_dtype=None, index_dtype=None, allow_lossy=False)` — Return just the scipy CSR matrix (or a dense `numpy.ndarray` for `container="dense"`), with the same dtype kwargs.
- Properties: `n_obs`, `n_vars`, `nnz`, `skipped_shards`, `total_shards`

Both conversions consume the result (zero-copy hand-off), but only on success: a **tripped
`allow_lossy` guard leaves the result intact**, so the retry the error message recommends
(`result.to_anndata(allow_lossy=True)`) works on the same object.

### Declaring the dtype at `collect()`

**The dtype passed to `collect()` is the dtype the data is *decoded* at; a dtype
passed to `to_anndata()` / `to_csr()` is a *cast* of what was already decoded.**
`collect()` is where the I/O and the decode happen, so it is the only place a
dtype can change what comes off disk:

```python
exp = pyscx.open("atlas.scx")                       # value_max > 2**24
q = exp.query().filter_obs("cell_type == 'fibroblast'")

q.collect(data_dtype="float64").to_csr()            # exact, float64
exp.query().filter_obs(...).collect().to_csr(data_dtype="float64")   # ValueError
```

The second call raises **on this file** because the values are f32 by then:
returning them as `float64` would report f32-rounded numbers at the wider dtype.
The error names the dtype and points at `collect()`. It is the guard firing, not
a blanket rule — the same call on a file whose selected shards hold no count
above 2²⁴ succeeds and casts post-assembly, because there was nothing to lose. `uint32` and `int64` behave like `float64`
(each holds every `u32` exactly); `float16`, `uint16` and the plain `float32`
default still fail loud on a `> 2²⁴` count, because no decode order helps a
target that cannot hold the value; `allow_lossy=True` accepts the rounding
anywhere.

Three details worth knowing:

- On a result collected with an explicit `data_dtype`, `to_anndata()` /
  `to_csr()` with **no** `data_dtype` return that dtype (not `float32`). Passing
  a *different* one raises: the caller wants either another decode (re-collect)
  or a numpy `.astype()` of what they hold, and `allow_lossy` does not unlock it.
- `container=` stays on the materialize call, since it is applied after the
  decode. `container="dense"` works on either kind of result, and on the
  one-shot `to_anndata(obs_filter=…, data_dtype=…, container="dense")` route,
  which decodes natively and scatters afterwards.
- `index_dtype` at `collect()` narrows the index buffer, but scipy normalises a
  `csr_matrix`'s index width on construction, so the returned matrix does not
  report it (the same caveat as `index_dtype="int16"` above). Passing a
  *different* `index_dtype` to `to_anndata()` / `to_csr()` on a dtype-selected
  result raises rather than being ignored; for `container="dense"` it is
  accepted and irrelevant, since dense output has no indices.
- `with_normalize()` / `with_log1p()` with a non-`float32` `data_dtype` is
  **refused**: those replace the stored counts with floating-point values, so no
  dtype reproduces the stored data exactly, and serving it from the f32 route
  would silently change which guard ran. Collect without a dtype (the values are
  transformed anyway) or drop the transform.

The same applies over cloud, on both spellings: `open_cloud(url).query()` returns
the same pipeline, so `collect(data_dtype=…)` decodes losslessly there — and
`pyscx.read_cloud(url, …)` takes `data_dtype` / `index_dtype` / `allow_lossy`
directly, for the same reason `collect()` does (that call *is* the decode).

## CloudExperiment

The Python-visible class is `CloudExperiment` (the Rust type is
`PyCloudExperiment`). Returned by `pyscx.open_cloud()`. Cloud-hosted SCX
handle. Supports metadata accessors plus a cloud-native query path served over
`object_store` range reads; full `to_anndata()` and `validate()` still
require `pyscx.pull()` to materialise the file locally. For a one-call read see
[`pyscx.read_cloud(...)`](../cloud.md#pyscxread_cloud--one-liner-cloud-read). Its
`repr` is the AnnData-style header line (no key lists — listing them would need
network reads).

- `n_obs` `→ int` — Number of observations (cells)
- `n_vars` `→ int` — Number of variables (genes)
- `shape` `→ tuple[int, int]` — `(n_obs, n_vars)`
- `nnz` `→ int` — Total non-zero entries
- `shard_count` `→ int` — Number of CSR shards in the file
- `format_version` `→ int` — SCX format version
- `codec_id` `→ int` — Default codec ID
- `obs_keys() → list[str]` / `var_keys() → list[str]` — column names (one range read each)
- `is_multimodal() → bool` / `n_modalities() → int` / `modality_names() → list[str]` /
  `modality_id(name) → int | None` / `modality_info(id) → dict | None` — modality discovery,
  mirroring the local `Experiment` (the modality table is fetched once at open). On a
  single-modality file `is_multimodal()` is `False` and `modality_names()` is empty.
- `to_mudata()` — raises: cloud per-modality reads aren't supported yet; `pyscx.pull()` the file
  locally and open it with `pyscx.open(...).to_mudata()`.
- `query() → PyQueryPipeline` — Start a lazy cloud query. Backed by
  the same `QueryPipeline` as `pyscx.open(...).query()`, wired over a
  `CloudReader`-backed `SectionReader`. Predicate pushdown uses the
  catalog (and predicate indexes when present); only matching shards
  are range-read from object storage. Example:

  ```python
  adata = (
      pyscx.open_cloud("gs://bucket/atlas.scxd/")
            .query()
            .filter_obs("cell_type == 'T cell'")
            .select_genes(hvg)
            .collect()
            .to_anndata()
  )
  ```

  The flat one-liner `pyscx.read_cloud(url, obs_filter=..., var_names=...)`
  wraps this chain and returns an AnnData directly — see
  [docs/cloud.md § `pyscx.read_cloud(...)`](../cloud.md#pyscxread_cloud--one-liner-cloud-read).

  Deferred follow-ons: `CloudQueryOptions` (parallelism,
  max-inflight bytes, cache-dir, retry policy) and a batched async section
  fetcher (current cloud reads block per shard from the rayon worker).

## Per-surface capability matrix

The three handle surfaces — local `pyscx.Experiment`, `pyscx.CloudExperiment`,
and rscx `ScxExperiment` — overlap but are not identical. This table shows where
each capability lives so you don't have to rediscover it per surface. "via
query" means the method isn't on the handle directly; reach it through
`query().collect()` (the result object) instead.

| Capability                       | `Experiment` (local) | `CloudExperiment`        | rscx `ScxExperiment` |
| -------------------------------- | -------------------- | ------------------------ | -------------------- |
| `n_obs` / `n_vars` / `nnz`       | ✓                    | ✓                        | ✓ (`$n_obs()` …)     |
| `shape`                          | ✓                    | ✓                        | — (use `n_obs`/`n_vars`) |
| `obs_keys()` / `var_keys()`      | ✓                    | ✓                        | — (`$obs()` / `$var()` data.frames) |
| `is_multimodal()` / `modality_names()` | ✓              | ✓                        | ✓                    |
| `to_anndata()` / extraction      | ✓                    | via query / `read_cloud` | via `query() …$collect()` |
| backed (out-of-core) X           | ✓ (`to_anndata(backed=True)`) | — (pull locally) | ✓ (`$x_backed()` / `scx_backed_sparse()`) |
| lazy transform chain             | ✓ (`ScxLazyTransformedDataset`) | — (pull locally) | ✓ (`$x_lazy()` / `scx_lazy_transform()` + `scx_normalize_total`/`scx_log1p`/`scx_row_scale`) |
| `to_mudata()` (multimodal)       | ✓                    | — (pull locally)         | `$to_mae()` / `$to_seurat()` |
| `detection_counts()` / `cells_expressing()` | ✓        | — (pull locally)         | —                    |
| `query()` builder                | ✓                    | ✓                        | ✓ (`scx_query()`)    |
| query `.count()`                 | ✓                    | ✓                        | ✓ (`count()`)        |
| `provenance()`                   | ✓                    | —                        | —                    |

Extraction off a query result is symmetric: pyscx `result.to_anndata()` /
`.to_csr()`; rscx `result$to_dgcmatrix()` / `$to_sce()` / `$to_seurat()` /
`as(result, "SingleCellExperiment")` (rscx adds S3/S4 generics — `dim()`,
`as.matrix()`, `as.data.frame()`, `as(res, "dgCMatrix")` — over the `$`-methods).
