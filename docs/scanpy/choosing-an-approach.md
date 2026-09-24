# Choosing the right approach

> Part of the [SCX + scanpy guide](README.md).

SCX offers three ways to work with data in Python. Each makes different
trade-offs between memory usage, scanpy compatibility, and performance:

## Decision tree

```
                         Is your dataset small enough
                         to fit in memory (~500K cells)?
                                    │
                        ┌───yes─────┴──────no───┐
                        ▼                       ▼
                   In-memory              Do you need
               (simplest, full           the full dataset?
              scanpy compat)                    │
                                    ┌───yes─────┴──────no───┐
                                    ▼                       ▼
                            Backed + lazy             Query pipeline
                          (out-of-core with         (extract a subset,
                          pyscx.accel.*)             then work in-memory)
```

## Feature comparison

|  | In-memory | Backed + lazy | Query pipeline |
|--|-----------|---------------|----------------|
| **API** | `to_anndata()` + `sc.pp.*` | `to_anndata(backed=True)` + `pyscx.accel.*` | `.query().filter_obs().collect()` |
| **When to use** | Small–medium datasets that fit in RAM | Atlas-scale datasets (500K–10M+ cells) | Extract a cell/gene subset from a large file |
| **Peak memory** | Full matrix in RAM | ~1 shard working set (~128 MB) | Subset only |
| **scanpy compatibility** | ✅ Full — every `sc.pp.*` / `sc.tl.*` works | ⚠️ Partial — use `pyscx.accel.*` for preprocessing; `sc.tl.*` and `sc.pl.*` work normally | ✅ Full — result is a regular AnnData |
| **Parallel shard decode** | ✅ All shards decoded in parallel via rayon | ✅ Per-access shard decode (parallel for streaming ops) | ✅ Parallel decode of matching shards |
| **GPU accelerators** | ✅ via `pyscx.accel.*(device="gpu")` | ✅ via `pyscx.accel.*(device="gpu")` | ❌ Preprocess in query pipeline runs on CPU; use accelerators after `.to_anndata()` |
| **Predicate pushdown** | ❌ All data loaded | ❌ All data accessible (filtering via deletion vectors) | ✅ Skips non-matching shards entirely |
| **Lazy normalize/log1p** | ❌ Materializes (standard scanpy) | ✅ `pyscx.accel.normalize_total()` / `log1p()` — zero materialization | ✅ `with_normalize()` / `with_log1p()` — applied in Rust during collect |
| **Streaming PCA** | ❌ Requires full matrix | ✅ `pyscx.accel.pca()` streams through lazy transforms | ❌ PCA runs after materialization |
| **Write-back** | ✅ In-place modification of X | ❌ Read-only (use `pyscx.preprocess()` for copy-on-write) | ❌ Read-only |
| **Typical dataset size** | < 500K cells | 500K–10M+ cells | Any size (output is a subset) |

## Pros and cons summary

**In-memory** (`to_anndata()`):
- ✅ Simplest — zero learning curve if you already know scanpy
- ✅ Every scanpy function works without modification
- ✅ Fastest for datasets that fit in RAM (no per-access overhead)
- ❌ Full matrix must fit in memory (e.g., 1M cells × 30K genes at 5% density ≈ 6 GB)
- ❌ No lazy preprocessing — `normalize_total()` and `log1p()` operate on the full matrix

**Backed + lazy** (`to_anndata(backed=True)` + `pyscx.accel.*`):
- ✅ Handles 10M+ cells on modest hardware (~16 GB RAM)
- ✅ Lazy preprocessing keeps data on disk (normalize, log1p, filter)
- ✅ Streaming PCA, kNN, UMAP through lazy transforms
- ✅ GPU accelerators via `device="gpu"`
- ⚠️ Must use `pyscx.accel.*` instead of `sc.pp.*` for preprocessing
- ⚠️ Some scanpy functions still force materialization (see [compatibility table](backed-mode.md#scanpy-operations-in-backed-mode))

**Query pipeline** (`.query().filter_obs().collect()`):
- ✅ Predicate pushdown skips non-matching shards (bandwidth savings up to 20×)
- ✅ Normalize + log1p computed in Rust during collect (fast)
- ✅ Result is a regular AnnData — full scanpy compatibility downstream
- ❌ Only useful when you want a subset, not the full dataset
- ❌ No streaming PCA or lazy transforms — analysis starts after materialization

> [!TIP]
> **Hybrid approach:** Use the query pipeline to extract a subset, then
> work in-memory with standard scanpy:
> ```python
> adata = (pyscx.open("atlas.scx")
>     .query()
>     .filter_obs("tissue == 'lung'")
>     .with_normalize(1e4)
>     .with_log1p()
>     .collect()
>     .to_anndata())
> sc.pp.pca(adata)  # regular scanpy from here
> ```
> The query pipeline always **materializes** the matching subset into
> memory. If the subset is still too large to materialize, open in backed
> mode with filtering instead — the data stays on disk:
> ```python
> adata = pyscx.open("atlas.scx").to_anndata(
>     backed=True, obs_filter="tissue == 'lung'")
> pyscx.accel.normalize_total(adata, target_sum=1e4)  # lazy
> pyscx.accel.log1p(adata)                             # lazy
> pyscx.accel.pca(adata, n_comps=50)                   # streaming
> ```

> [!TIP]
> **Multimodal files:** scope the query to one modality with
> `query(modality="rna")`. The obs predicate resolves against the shared
> global obs axis; X / `select_genes` resolve against that modality's var:
> ```python
> rna = (pyscx.open("citeseq.scx")
>     .query(modality="rna")
>     .filter_obs("cell_type == 'T cell'")
>     .select_genes(["MS4A1", "CD3D"])
>     .collect()
>     .to_anndata())
> ```
> On a multimodal file `modality=` is required (omitting raises `ValueError`).
> See [docs/multimodal.md § 3.4](../multimodal.md#34-modality-scoped-queries--querymodality).


## See also

- [Quick start](quickstart.md) — the in-memory and backed paths, runnable.
- [Loading SCX data into AnnData](loading.md) — every `to_anndata()` option.
- [Backed mode and out-of-core iteration](backed-mode.md) and
  [Lazy preprocessing](lazy-preprocessing.md).
