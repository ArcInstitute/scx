# API quick start

> Part of the [SCX API reference](README.md).

The calls most sessions need, one surface at a time. Each section links to the
full reference for that surface.

## Python

```python
import pyscx

# Convert: streams from disk, safe for files larger than RAM.
pyscx.from_h5ad("data.h5ad", "data.scx")

# Open a lazy handle, then read the whole file as an AnnData ...
exp = pyscx.open("data.scx")
adata = exp.to_anndata()

# ... or keep X on disk ...
adata = exp.to_anndata(backed=True)

# ... or pull just a subset, with predicate pushdown.
subset = (
    exp.query()
    .filter_obs("cell_type == 'T cell'")
    .select_genes(["CD3D", "CD8A"])
    .collect()
    .to_anndata()
)

# Rust-native analysis writes the same AnnData slots scanpy does.
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata)
pyscx.accel.leiden(adata)

# Export back to h5ad (streams by default).
pyscx.to_h5ad("data.scx", "data.h5ad")
```

Reference: [module-level functions](python-functions.md),
[`Experiment`](python-experiment.md#experiment), [queries](python-query.md),
[`pyscx.accel`](python-accel.md), and the
[training loaders](python-training.md) for ML workloads.

## Rust

```rust
use scx_format_io::ScxReader;
use scx_engine::QueryPipeline;

// Open (mmap) and inspect a file.
let reader = ScxReader::open("data.scx")?;
println!("{} cells x {} genes, {} nnz", reader.n_obs(), reader.n_vars(), reader.nnz());
let obs = reader.read_obs()?; // Arrow RecordBatch

// Lazy query: no I/O until collect().
let result = QueryPipeline::open("data.scx")?
    .filter_obs("cell_type == 'T cell'")?
    .limit(1000)
    .collect()?;
```

Reference: [format I/O](rust-format-io.md) (reader, writer, backed readers),
[query engine](rust-engine.md), [file operations](rust-ops.md), and the
[crate graph in docs/architecture.md](../architecture.md#crate-dependency-graph).

## CLI

```bash
scx convert data.h5ad data.scx        # h5ad -> SCX
scx info data.scx                     # shape, codecs, shards, provenance
scx convert data.scx data.h5ad        # SCX -> h5ad
```

Reference: [CLI (`scx`)](cli.md).

## R

The `rscx` bindings are covered on the
[multimodal API page](multimodal.md#r-rscx) and in the rscx package documentation.
