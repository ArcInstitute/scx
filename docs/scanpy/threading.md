# Multithreading

> Part of the [SCX + scanpy guide](README.md).

Most scx-accel and pyscx entry points are multithreaded via rayon by default,
and release the GIL (`py.allow_threads()`) so Python stays responsive. For the
full architecture — runtimes, thread pools, channel topology, and how to control
parallelism — see [docs/multithreading.md](../multithreading.md).

## Per-function threading

| Function | Threading | Notes |
|----------|-----------|-------|
| `pyscx.open()`, `to_anndata()`, `read_layer()` | Rayon parallel shard decode | `scx-format` with `parallel` feature; SIMD BitPacker4x within each shard |
| `ScxBackedDataset` slicing / column projection | Rayon per access | Each `X[...]` call decodes touched shards in parallel |
| `pyscx.query().where(...).collect()` | Rayon parallel shard decode | Only shards surviving catalog pushdown are decoded |
| `pyscx.iter_chunks()`, `pyscx.preprocess()`, `pyscx.save_layer()` | Rayon parallel shard decode + encode | Parallel encode achieves up to 3.2× at 32 threads |
| `pyscx.accel.pca` (CPU) | Rayon | Covariance and transpose SpMM partition their output columns across workers — one shared accumulator, no merge |
| `pyscx.accel.neighbors` (CPU) | Rayon | Parallel kNN queries on HNSW index |
| `pyscx.accel.umap` (CPU) | Single-threaded SGD | Edge updates are serial on CPU; GPU path uses CUDA kernel parallelism |
| `pyscx.accel.leiden` | Opt-in rayon via `parallel=True` | Sequential by default (reproduces C++ leidenalg's move-node ordering); parallel uses conflict-free graph coloring |
| `pyscx.accel.rank_genes_groups` | Rayon | Parallel Wilcoxon rank-sum across genes |
| `pyscx.accel.pseudobulk_dex`, `pseudobulk_means` | Rayon (aggregation) | The CSR scatter partitions its **output** across workers — one group's row per task while the largest group is small enough (and always for a scipy matrix with unsorted / duplicate columns), a column block of every group's row otherwise — so it is bit-identical to a serial loop on any thread count; shard decode overlaps it; downstream `pydeseq2` testing runs single-threaded, the native `nb_glm` fit is gene-parallel |
| `pyscx.accel.highly_variable_genes` | Rayon (via streaming reader) | Parallelism comes from shard decode; the mean/var reduction itself is serial |
| `pyscx.accel.perturbation_metrics`, `energy_distance` | Rayon (CPU) or GPU | CPU parallelizes across perturbations / pairwise-distance rows; GPU runs pseudobulk means (`perturbation_metrics`) or a gemm pairwise-distance mean (`energy_distance`, euclidean/cosine f32) on the device |
| `pyscx.accel.discrimination_score` | Rayon (CPU only) | Parallelizes across perturbations; no GPU kernel (exact-rank parity not f32-safe) |
| `pyscx.accel.knockdown_efficiency`, `clustering_agreement` | Rayon | Parallel per-perturbation / per-label reductions |
| `pyscx.accel.harmony` (batch correction) | Rayon | Parallel per-cluster correction |
| `pyscx.accel.normalize_total`, `log1p`, `filter_cells`, `filter_genes`, `calculate_qc_metrics` | Rayon (via streaming reader) | Lazy — no work until materialized or consumed |
| `pyscx.pull()` / `pyscx.push()` (cloud) | Tokio async | `parallelism` parameter controls concurrent transfers (default 8) |
| `pyscx.TrainingDataset` | Triple-buffered (tokio I/O + rayon decode + Python consumer) | See [multithreading.md §Training data loader](../multithreading.md#training-data-loader-triple-buffered-pipeline) |
| `pyscx.accel.*` with `device="gpu"` | CUDA kernel parallelism | CPU side launches kernels and manages transfers |

## Controlling parallelism

```python
import os
os.environ["RAYON_NUM_THREADS"] = "8"   # must be set before `import pyscx`
import pyscx
```

`RAYON_NUM_THREADS` sizes the process-wide rayon pool that most accelerators use.
`SCX_ACCEL_NUM_THREADS` is a narrower ceiling for the accelerators' private rayon work —
Harmony batch integration, and the number of column blocks the PCA reductions split their
output into — so you can cap those on a fat node without shrinking every op. It is read
once at first use (set it before the first accelerator call); unset (the default) leaves
today's behaviour unchanged. On PCA it bounds **speed and memory only**: the block count
cannot change the numbers, because the blocks write to disjoint slices and are never merged
(see [PCA § Reproducibility](accel-embedding-clustering.md#reproducibility)). Other `SCX_ACCEL_*` knobs (`SCX_ACCEL_PREFETCH_DEPTH`,
`SCX_ACCEL_REDUCTION_MODE`, `SCX_ACCEL_DE_MEMORY_BUDGET`) are documented in
[performance/](../performance/README.md); `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET`, which bounds one
`energy_distance` Gram block and so interacts with `RAYON_NUM_THREADS` multiplicatively, is
in the [architecture.md environment table](../architecture.md#environment-variables). `SCX_ACCEL_PREFETCH_DEPTH` bounds the
decode-prefetch pipeline, which since Phase 4.2 also covers the backed
aggregation kernels (QC, filtering, `col_*`, `normalize_total`'s row sums), their
column-projected **CSR** and lazy/transformed twins, and GPU staging — so raising it
raises peak memory (`depth` decoded shards in flight) across all of those, not
just HVG.

Three consequences worth knowing before you tune it:

- **On a memory-tight GPU host, consider `SCX_ACCEL_PREFETCH_DEPTH=2`.** GPU
  staging keeps host RSS low otherwise (1.7–4.5 GB in the Phase-4.2 capture), so
  the extra `depth − 1` decoded shards are plainly visible there: **+19–46 %**
  peak host RSS across every measured op. Depth 2 keeps most of the overlap at
  roughly a third of the extra footprint. The default of 4 is tuned for
  throughput, not for the tightest node.
- **`RAYON_NUM_THREADS=1` now makes GPU staging fully sequential.** Before 4.2 it
  had a dedicated `std::thread` that overlapped one shard ahead *unconditionally*;
  the shared pipeline instead declines to engage on a single-thread pool (that
  guard is what prevents a nested-call deadlock). If you pin
  `RAYON_NUM_THREADS=1` for reproducibility, GPU HVG/DE will be **slower than
  before 4.2**, not faster. For **CPU PCA** you no longer need to pin it at all —
  repeated runs agree at any thread count, and `SCX_ACCEL_DETERMINISTIC_LINALG=1`
  covers the cross-thread-count case more cheaply than serializing everything
  (see [PCA § Reproducibility](accel-embedding-clustering.md#reproducibility)).
- **`prefer_format="csc"` does not inherit the 4.2 speedups.** The CSC column
  kernels reach their source through `&dyn ColumnShardSource` and still decode
  serially; only the CSR paths are prefetched. They do benefit from the `col_*`
  GIL release.

Since Phase 4.5, `SCX_GPU_STAGING_MEMORY_BUDGET` (bytes) expresses the same
bound in a unit that does not depend on the file: GPU staging derates the
prefetch depth so `depth × per-shard-decoded-bytes` fits the budget, using the
per-shard `nnz` the catalog already carries. Unset — the default — nothing is
derated and the depth is exactly `SCX_ACCEL_PREFETCH_DEPTH`.

## GPU DE device residency

GPU DE's CSR route (`gpu_csr_v3`, the mandatory route for any file **without** a
CSC sidecar) has no column-range prefilter, so before Phase 4.5 it walked every
shard once **per gene chunk** — `n_gene_chunks × n_shards` host decodes and
uploads. On a 61 497-gene file at the default 500-gene chunk that is 123 full
passes over the matrix, and it made GPU DE almost entirely host-decode-bound.

4.5 drains the source **once** into device-resident per-shard CSR buffers and
serves every later chunk from VRAM, and narrows each row to the chunk's column
window with a binary search instead of scanning the whole row and predicating
per element. Neither changes what the kernels compute.

The cost is device memory: one f32 value plus one i32 index per nonzero, so
roughly `8 × nnz` bytes for the whole matrix on top of the per-chunk scratch.
Two knobs:

- **`SCX_GPU_DE_RESIDENT_MAX_FRAC`** (default `0.5`) — the fraction of *free*
  VRAM the resident matrix may occupy. The remainder is what the per-chunk
  gene-slab budget then sizes itself against, so a matrix that takes half the
  card simply yields a smaller gene chunk rather than an OOM.
- **`SCX_GPU_DE_RESIDENT=0`** — kill switch. Restores the pre-4.5 streaming
  behaviour exactly.

Native GPU **PCA** has the same two-path structure and its own kill switch,
**`SCX_GPU_PCA_RESIDENT=0`**, which forces the streaming power loop (the whole
matrix re-decoded and re-uploaded on every multiply). Both ops record which path
ran on `resident_csr` — see
[api/python-accel.md § Accelerator route metadata](../api/python-accel.md#accelerator-route-metadata). The
PCA knob is also what makes the streaming loop reachable from a test: residency
is decided against *free* VRAM at call time, so on a large card no fixture can
force the streaming branch by shape alone.

A third knob, `SCX_GPU_VALIDATE_PAR_MIN_NNZ`, sets the shard size above which
the per-shard GPU-DE validation scan (strictly-increasing columns + finiteness)
runs in parallel; default 65 536 nnz, and pinning it above any real shard's nnz
restores the serial scan. It exists mainly so that choice stays measurable —
the parallel scan is a **measured no-op** at current scales, because it runs on
the consuming thread while the prefetch workers decode ahead and is therefore
hidden behind decode.

Residency is declined — silently and correctly — when the matrix does not fit
the budget, or when there is only one gene chunk (streaming would run one pass
anyway, so retaining the matrix would be pure cost). Because a declined run
produces *identical output*, just slower, the only way to tell is the route
stamp: `adata.uns["scx_accel"][<op>]["resident_csr"]` is `True` when residency
engaged, `False` when the CSR route streamed, and `None` on a route with no
residency decision to make (CPU, dense, or CSC-direct). A benchmark gate
(`de_route_resident_csr`) floors it for exactly that reason.

The CSC-direct route is unaffected: it already prefilters by column range and
never re-decodes.

The heavy accelerators, including `pyscx.accel.col_sums` and its siblings,
release the GIL for their streaming scan, so they can be called concurrently
from Python threads without serialising each other.

A CPU accelerator that releases the GIL works from an **owned snapshot** of `X`,
taken before the release: the in-flight result reflects the matrix as it was
when the kernel started, never a half-updated blend. On an **in-memory** `X`
that snapshot is a copy; it is taken in parallel, so it costs rather less than
`X.copy()` (measured 14 ms for a 192 MB dense matrix, against 119 ms for
`np.copy`), and `pseudobulk_means` warns once it exceeds 1 GB. A **backed or
lazy** `X` needs no copy at all: it streams shard by shard from the file, which
is the cheaper way to run these ops on a large matrix regardless.

That is a statement about those kernels, not a general licence to mutate `X`
from another thread mid-call. The GPU in-VRAM fast lane reads `X`'s buffers
directly and is safe only because it holds the GIL throughout — nothing is
snapshotted — and conversion (`pyscx.from_anndata`) makes several passes over
`X` that are not guaranteed to see one consistent state. Mutating a matrix
while any scx call is reading it remains unsupported; what changed is that a
detached kernel can no longer read a buffer out from under you.

The cloud runtime exposes its own knob:

```python
pyscx.pull("gs://bucket/atlas.scxd/", "atlas.scx", parallelism=16)
```

## See also

- [docs/multithreading.md](../multithreading.md) — the full threading
  architecture across crates.
- [GPU acceleration](accel-gpu.md).
