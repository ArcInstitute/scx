# Migrating from h5ad to SCX

For analysts who already have an h5ad/scanpy workflow. This page answers three
questions: **which loader do I use, what changes when I convert, and what (if
anything) is lost in the round trip?**

> **Status:** pre-1.0 (v0.7.x). The on-disk format is stable for read-back at
> `format_version` 3; the spec freezes and gains published conformance vectors at
> 1.0.

## TL;DR

```python
import pyscx

# Convert once (CLI or Python):
pyscx.write(adata, "data.scx")          # mirrors adata.write_h5ad(...)
#   or:  scx convert data.h5ad data.scx --stream

# Read it back:
adata = pyscx.read("data.scx")          # mirrors sc.read_h5ad(...)
exp   = pyscx.open("data.scx")          # a handle (lazy); .to_anndata() / .query()
```

`pyscx.read` / `pyscx.write` are the flat one-liners. `pyscx.open(path)` returns
an `Experiment` handle whose `repr` looks like an AnnData:

```
>>> pyscx.open("data.scx")
Experiment object with n_obs × n_vars = 2700 × 32738
    obs: 'n_genes', 'percent_mito', 'louvain'
    var: 'gene_ids', 'n_cells'
    uns: 'louvain_colors'
    obsm: 'X_pca', 'X_umap'
```

`Experiment.info()` carries the on-disk codec / shard / format-version details.

## Which loader do I use?

```
                         Is your dataset small enough
                         to fit in memory (~500K cells)?
                                    │
                        ┌───yes─────┴──────no───┐
                        ▼                       ▼
                  pyscx.read(path)        Do you need
              (full scanpy compat)        the full dataset?
                                                │
                                   ┌───yes───────┴──────no───┐
                                   ▼                         ▼
                          to_anndata(backed=True)      .query().filter_obs()
                          + pyscx.accel.*              .collect()  (extract a
                          (out-of-core)                subset, then in-memory)
```

| Need | Use | Notes |
|------|-----|-------|
| Dataset fits in RAM | `pyscx.read(path)` / `exp.to_anndata()` | Returns a regular AnnData; every `sc.pp.*` / `sc.tl.*` works |
| Atlas-scale, full dataset | `exp.to_anndata(backed=True)` + `pyscx.accel.*` | ~1-shard working set; use `pyscx.accel.*` for preprocessing, not `sc.pp.*` |
| A cell/gene subset of a large file | `exp.query().filter_obs(...).select_genes(...).collect()` | Predicate pushdown skips non-matching shards |
| Reading from cloud storage | `pyscx.read_cloud(url, obs_filter=..., var_names=...)` | `gs://` / `s3://` / `az://` / `file://`; fetches only matching shards |
| Training a model | `TrainingDataset` / `IndexPlanDataset` | See [docs/training.md](training.md) |

Full trade-off table:
[docs/scanpy.md § Choosing the right approach](scanpy.md#choosing-the-right-approach).

## What changes when you convert

- **`X` is stored as float32 CSR.** A float64 source matrix is downcast (a
  `UserWarning` fires at write). Counts and most normalized values are unaffected
  in practice; if you depend on float64 precision, keep the h5ad.
- **obs/var are Arrow-backed**, not HDF5 datasets. Column dtypes, categorical
  categories, and the pandas `ordered` flag round-trip. The index (`obs_names` /
  `var_names`) round-trips as the pandas index.
- **`pyscx.accel.*` replaces `sc.pp.*` in backed mode.** In-memory mode is plain
  scanpy. Backed mode needs the accelerators for preprocessing — `sc.pp.*` would
  force a full materialization. See
  [docs/scanpy.md § scanpy operations in backed mode](scanpy.md#scanpy-operations-in-backed-mode).

## What does and does not round-trip

The canonical, always-current table lives in the API reference:
[docs/api.md § Round-trip fidelity](api.md#round-trip-fidelity). Every conversion
that drops or transforms something also emits a structured warning — see
[docs/api.md § Conversion warnings](api.md#conversion-warnings-convertwarning).
Headlines:

- **Preserved:** obs/var columns + dtypes, ordered categoricals, `obsm`/`varm`
  embeddings, layers, `obsp`/`varp` (as float32 CSR), `adata.raw`
  ([docs/api.md § `adata.raw`](api.md#adataraw)), most `uns` entries.
- **Lossy / transformed (warns):** `X` and `obsp`/`varp` float64 → float32;
  a dense `obsp`/`varp` is re-emitted as sparse (values identical).
- **Dropped (warns):** CSC/unsupported `obsp`/`varp`, pickled `uns` objects,
  obs/var columns with an unsupported dtype, `uns` pandas DataFrames (kept as a
  nested dict + a `FlattenedUnsDataframe` warning, not silently flattened).

## scanpy-divergence gotchas

A short list of places SCX's accelerators behave differently from a naive
scanpy script. Each is documented in full in `docs/scanpy.md`.

- **MT genes must be tagged explicitly.** `pyscx.accel.calculate_qc_metrics`
  follows scanpy's contract: without `qc_vars=["mt"]` there is no
  `pct_counts_mt` column. The accelerator emits a `UserWarning` when MT-prefixed
  symbols are present but `qc_vars=None`.
- **`seurat_v3` HVG needs raw counts**, not log-normalized values — stash counts
  in a layer before normalizing. A high-cardinality `batch_key` can trigger a
  per-batch LOESS singularity; SCX drops those batches with a warning rather than
  crashing. See
  [docs/scanpy.md § Lazy preprocessing](scanpy.md#lazy-preprocessing-in-backed-mode).
- **Leiden on GPU is not label-stable** vs Python `leidenalg`. `device="auto"`
  on a GPU host uses cuGraph (ARI ≈ 0.92 vs leidenalg). Pin `device="cpu"` to
  preserve label stability for downstream DE / annotation transfer. See
  [docs/scanpy.md § Leiden clustering](scanpy.md#leiden-clustering-pyscxaccelleiden).
- **Backed-mode preprocessing is lazy on CPU, eager on GPU.** CPU wraps `X` in a
  lazy transform; GPU streams through a fused kernel and materializes a scipy
  CSR. See [docs/scanpy.md § Lazy vs eager preprocessing](scanpy.md#lazy-vs-eager-preprocessing).

## Errors you might see

SCX maps failures to the Python exception a scanpy user expects:

- Missing input file → `FileNotFoundError`.
- Truncated / corrupt / wrong-magic / version-mismatch file → `ValueError` (the
  message says the file looks corrupt or was written by an incompatible SCX).
- A stale CSC sidecar → `ValueError` naming the `--rebuild-csc` fix.
- A non-canonical scipy CSR passed to an accelerator → `ValueError` suggesting
  `X = X.tocsr(); X.sort_indices()` or a re-run of `pyscx.from_anndata`.

## See also

- [docs/quickstart.md](quickstart.md) — 5-minute end-to-end pipeline.
- [docs/scanpy.md](scanpy.md) — the full scanpy integration story.
- [docs/api.md](api.md) — API reference, fidelity table, conversion warnings.
- [docs/training.md](training.md) — ML training loaders.
