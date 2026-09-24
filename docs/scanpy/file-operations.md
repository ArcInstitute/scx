# File operations

> Part of the [SCX + scanpy guide](README.md). Inspecting, validating, and modifying SCX files from a
scanpy session.

## File inspection

```python
exp = pyscx.open("experiment.scx")
print(exp)           # AnnData-style repr:
#   Experiment object with n_obs × n_vars = 10000 × 33694
#       obs: 'cell_type', 'sample'
#       var: 'gene_ids'
#       layers: 'raw_counts', 'spliced'
print(exp.n_obs)     # 10000
print(exp.n_vars)    # 33694
print(exp.nnz)       # 5234891
print(exp.obs_keys())  # ['cell_type', 'sample'] — callable, like adata.obs_keys()
print(exp.layer_names())  # ["raw_counts", "spliced"] — callable method too
print(exp.info())    # codec / shard / format-version internals

# Validate checksums
results = exp.validate()
for section, passed in results:
    print(f"  {section}: {'✓' if passed else '✗'}")
```

## Validating files after write or transfer

`pyscx.open(path)` and `pyscx.open(path, verify=True)` authenticate the file
header, catalog offsets/lengths, and the catalog checksum, but they do **not**
re-hash section payload bytes. That's the right default for normal reads —
the catalog already records per-section BLAKE3 checksums, so opening is fast
and detects file-header / catalog corruption immediately.

For full payload integrity, run `pyscx.validate(path)` (or `scx validate`
from the CLI), which BLAKE3-hashes every section's bytes against the
catalog and reports per-section pass/fail. Cost is proportional to total
section bytes.

Production checkpoints to call `pyscx.validate()`:

- After `pyscx.from_anndata(...)` / `pyscx.from_10x(...)` writes a new
  file, before downstream consumers depend on it.
- After `pyscx.pull(...)` (or `scx pull`) downloads from cloud, before
  treating the local file as canonical.
- After any file transfer (rsync, gcloud cp, scp, etc.) into a path that
  will be read by training or analysis pipelines.

```python
results = pyscx.validate("data.scx")
for name, passed in results:
    if not passed:
        raise RuntimeError(f"section {name} failed checksum")
```

### Deep validation (`deep=True`)

Checksum validation proves the section bytes match what was written, but not
that those bytes *decode* to a well-formed sparse matrix. For decode-level
integrity, pass `deep=True` (the equivalent of `scx validate --deep`):

```python
results = pyscx.validate("data.scx", deep=True)   # or exp.validate(deep=True)
for name, passed in results:
    if not passed:
        raise RuntimeError(f"{name} failed validation")
```

On top of the per-section checksums, deep mode:

- decodes every sparse shard and verifies the **v3 canonical CSR invariant**
  (column indices sorted and in range, no explicit zeros, `indptr` starting at
  0 and monotonically increasing, metadata consistent with the decoded data);
- verifies every framed shard's **row-group `BlockIndex`** — structural linkage
  to its source shard plus a decode-parity check that seeking to each row group's
  recorded block offset and decoding it reproduces the canonical decode
  byte-for-byte.

Deep-check results are appended to the returned list with `canonical-csr ` and
`block-index ` prefixed names. Canonical-CSR checks run only on v3+ files
(pre-v3 files may legitimately carry unsorted shards, so they are skipped).
Unlike a checksum failure on an essential section — which raises — deep-check
failures report `False` in the result list rather than raising, so iterate the
list to surface them. Cost is higher than a checksum-only pass because every
shard is decoded; reserve it for post-write or post-transfer integrity gates
where decode correctness matters.

## File operations with scanpy

### Removing doublets and saving back

```python
import pyscx
import scanpy as sc
import scrublet

adata = pyscx.open("experiment.scx").to_anndata()

# Run doublet detection
scrub = scrublet.Scrublet(adata.X)
scores, predicted = scrub.scrub_doublets()

# Mark doublets as deleted (instant, no data rewrite)
exp = pyscx.open("experiment.scx")
exp.mark_deleted(predicted)  # boolean numpy array

# Or save the filtered result as a new file
adata_clean = adata[~predicted].copy()
pyscx.from_anndata(adata_clean, "experiment_clean.scx")
```

### Appending new batches

```python
# Append cells from another SCX file (streaming — reads one shard at a time)
pyscx.append("atlas.scx", "new_batch.scx")

# Append cells from an AnnData object
new_adata = sc.read_h5ad("new_batch.h5ad")
pyscx.append_from_anndata("atlas.scx", new_adata)
```

`pyscx.append` uses a streaming SCX → SCX path: source shards are decoded
(or raw-copied when codec and encoding match) one at a time, so memory
usage is bounded by a single shard rather than the full source matrix.

Obs metadata is appended as new `ObsMetadataShard` sections — the
existing obs is not rewritten. Legacy single-section obs files are
promoted to shard 0 on first append, and new obs rows are added as
subsequent shards.

### Merging datasets

`pyscx.merge` combines multiple SCX files into one atlas-scale output.
All inputs must share the same var axis (gene set, order, and metadata).

```python
# Basic merge — var identity validated by default
pyscx.merge(["batch1.scx", "batch2.scx", "batch3.scx"], "atlas.scx")

# Then analyze the merged atlas
adata = pyscx.open("atlas.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.combat(adata, key="batch")  # batch correction
```

#### Full merge signature

```python
pyscx.merge(
    inputs,                              # list[str] — at least 2 SCX file paths
    output,                              # str — output SCX path
    index_obs=None,                      # list[str] — obs columns to index for query pushdown
    index_var=None,                      # list[str] — var columns to index
    index_preset=None,                   # "cellxgene" | "perturbseq" | "training"
    index_auto_threshold=None,           # int — auto-index cardinality threshold
    assume_identical_var=False,          # skip var identity validation
    assume_identical_obs=False,          # skip obs schema validation
    uns_policy=None,                     # "first" | "require-equal" | "namespace" | "summary"
)
```

#### Streaming behavior

Merge operates shard-by-shard at every level — **no full-dataset
materialization** at any point in the pipeline:

- **Obs metadata**: each input's obs is read one shard at a time and
  written as `ObsMetadataShard` sections in the output. Peak obs memory
  is bounded by one shard (~16K rows) rather than the total cell count.
  This removes the previous ~2 GB Arrow IPC narrow-offset ceiling.
- **X and layers**: CSR shards are decoded one at a time per input and
  re-encoded into output shards. Single-modality merge now matches the
  multimodal streaming pattern.
- **Obsm / varm**: dense mapping sections are read and written one shard
  at a time. Legacy single-section inputs are treated as one source shard.
- **Predicate indexes**: built incrementally from the obs shard stream
  without materializing a full obs table.

> [!TIP]
> For atlas-scale merges (>10M cells), merge's peak RSS is now dominated by
> the per-shard working set (~128 MB) rather than total obs/layer size.
> Merges that previously OOM'd or hit Arrow offset overflows at ~67M cells
> now complete with bounded memory.

#### Var identity validation

> [!IMPORTANT]
> **Breaking change**: `merge()` now validates var identity by default.
> Pre-existing code that merged files with different var metadata (but
> the same `n_vars`) will error. This prevents silent column-axis
> corruption where gene indices in later inputs are misinterpreted
> against the first input's var table.

By default (`assume_identical_var=False`), merge compares every input's
var batch column-by-column against input 0 and errors on mismatch. If
you have already validated var identity upstream:

```python
pyscx.merge(inputs, output, assume_identical_var=True)
```

Similarly, `assume_identical_obs=False` validates obs schema (column
names and dtypes) across inputs.

#### Uns conflict policy

The `uns_policy` kwarg controls how conflicting `uns` sections are
handled across inputs:

| Policy | Behavior |
|--------|----------|
| `"first"` (default) | Keep the first input's uns verbatim; warn on disagreement |
| `"require-equal"` | Error if any input's uns differs from the first |
| `"namespace"` | Wrap each input's uns under `"input_0"`, `"input_1"`, etc. |
| `"summary"` | Keep the first input's uns and record conflicts in `uns["_scx_uns_conflicts"]` |

```python
# Error if uns sections differ across inputs
pyscx.merge(inputs, output, uns_policy="require-equal")

# Namespace each input's uns to preserve all metadata
pyscx.merge(inputs, output, uns_policy="namespace")
```

## See also

- [docs/operations.md](../operations.md) — behaviour and invariants of every
  mutating op (append, delete, compact, merge, rollback).
- [Landing external annotations](external-annotations.md) — adding obs / var
  columns in place.
