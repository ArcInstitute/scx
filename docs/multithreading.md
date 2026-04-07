# Multithreading in SCX

SCX is designed to exploit multicore CPUs at every stage — reading, querying,
training, and cloud I/O. This guide explains where parallelism is used, which
runtimes are involved, and how it all stays safe.

## Overview

| Component | Threading model | Key crate(s) |
|-----------|----------------|---------------|
| **Shard decode** (read) | Rayon data parallelism + SIMD BitPacker4x | `scx-format` (opt-in), `scx-engine` |
| **Query engine** | Rayon `par_iter` over shards | `scx-engine` |
| **Training loader** | Triple-buffered pipeline (tokio + rayon + std::thread) | `scx-loader` |
| **Cloud I/O** | Tokio async tasks | `scx-cloud` |
| **File mutations** | Advisory `flock()` via `fs4` | `scx-ops` |
| **GPU decode** | CUDA kernel parallelism | `scx-gpu` |
| **File writing** | Single-threaded (atomic rename) | `scx-format` |

## Parallel shard decode

SCX's sharded layout (see [`docs/sharding.md`](sharding.md)) makes parallelism
natural: each shard is independently decompressible with its own header, codec,
and checksum.

### scx-format (opt-in `parallel` feature)

`ScxReader::read_all_csr_shards()` and `ScxReader::read_layer()` use rayon's
`par_iter()` to decode shards concurrently via `assemble_shards_parallel()`.
Each shard is independently decompressible — the reader issues `MADV_SEQUENTIAL`
on the shard byte range before the parallel decode loop. Within each shard,
FOR-BP index decode uses SIMD BitPacker4x (4 × 32-element blocks) for rows
with ≥128 non-zeros, further reducing per-shard decode time. The parallel
feature is enabled:

```toml
# Cargo.toml
[features]
parallel = ["rayon"]
```

Without the feature flag, the same functions fall back to sequential decode.
This lets downstream crates (e.g., `scx-engine`) always get parallelism while
keeping `scx-format` lightweight for single-shard use cases.

### scx-engine (always parallel)

The query engine's `collect()` function always decodes candidate shards in
parallel via rayon:

```text
shard_infos.par_iter()
    .map(|si| decode_shard → filter_rows → Ok(csr_arrays))
    .collect::<Result<Vec<_>>>()?
```

Catalog-level predicate pushdown first prunes shards that can't contain matching
cells, so only candidate shards are decoded — parallelism multiplied by
selectivity.

## Training data loader (triple-buffered pipeline)

The `scx-loader` crate implements a three-stage pipeline that keeps the GPU
saturated by overlapping I/O, decode, and consumption:

```
┌──────────────────┐    tokio::mpsc     ┌──────────────────┐   crossbeam   ┌───────────────┐
│  Stage 1: I/O    │ ─────────────────→ │  Stage 2: Decode │ ───────────→  │  Stage 3:     │
│  (tokio async,   │   ShardGroup       │  (std::thread +  │   Batch       │  Consumer     │
│   2 workers)     │                    │   rayon par_*)   │               │  (Python/GPU) │
└──────────────────┘                    └──────────────────┘               └───────────────┘
```

### Stage 1: I/O (tokio)

A tokio runtime with 2 worker threads reads shard groups from the memory-mapped
file. Shard groups are sent through a bounded `tokio::sync::mpsc` channel
(capacity 2) to provide read-ahead without unbounded buffering.

### Stage 2: Decode (std::thread + rayon)

A dedicated OS thread (`scx-decode`) receives shard groups and uses rayon's
`par_chunks_mut` to fill the output batch in parallel — each rayon thread
handles one row's sparse-to-dense conversion, HVG projection, and
normalize+log1p in a single fused pass. Completed batches are sent through a
bounded `crossbeam` channel (capacity = `prefetch_batches`, minimum 2).

### Stage 3: Consumer (Python main thread)

`TrainingDataset.__next__()` calls `py.allow_threads()` to release the GIL
while pulling the next batch from the channel, allowing PyTorch CUDA threads to
run concurrently.

### Why three runtimes?

| Runtime | Reason |
|---------|--------|
| tokio | Async I/O with efficient epoll/io_uring integration |
| rayon | Work-stealing for CPU-bound decode/normalize |
| std::thread | Bridges async and sync worlds without blocking the tokio reactor |

> [!IMPORTANT]
> tokio is used for I/O only; rayon is used for CPU work. They never share
> mutable state — bounded channels provide back-pressure and isolation.

## Cloud I/O

`scx-cloud` uses a tokio multi-thread runtime for parallel cloud operations:

- **`pull()`**: Downloads sections in batches of `parallelism` (default: 8)
  concurrent async tasks, then writes them sequentially to the output file.
- **`push()`**: Uploads sections in parallel async tasks to the cloud backend.
- **`pull_filtered()`**: Same parallel download pattern, but only for shards
  matching a predicate — skipped shards are never downloaded.

```python
# Python: 8 parallel download tasks by default
pyscx.pull("gs://bucket/atlas.scxd/", "atlas.scx")
```

## Concurrent file access

### Reads: no locks needed

SCX uses memory-mapped I/O (`mmap`) for reads. The immutable-fragment file
design means readers always see a consistent snapshot of the file — no read
locks are ever acquired. Multiple processes, notebooks, or training jobs can
read the same `.scx` file simultaneously.

### Writes: advisory `flock()`

Mutating operations (append, delete, compact, merge, rollback) acquire advisory
file locks via the `fs4` crate before modifying a file:

| Lock type | Used by | Blocks |
|-----------|---------|--------|
| **Exclusive** (`FileLock`) | append, delete, rollback | Other writers |
| **Shared** (`SharedFileLock`) | compact, merge (on source files) | Exclusive writers |

Locks are released automatically when the `FileLock` / `SharedFileLock` guard
is dropped. On network filesystems where `flock()` is unavailable, advisory
locks degrade gracefully — readers are never blocked.

> [!TIP]
> Unlike HDF5's mandatory POSIX locks, SCX's advisory locks never cause
> `errno 37` ("No locks available") on NFS/Lustre/GPFS.

## File writing

File writing is intentionally single-threaded. `ScxWriter` writes sections
sequentially to a temporary file, then performs `fsync()` + `rename()` for
atomic visibility. This guarantees readers never see a partially-written file.

## GPU parallelism

The `scx-gpu` crate launches CUDA kernels with hundreds of GPU threads:

| Kernel | Parallelism |
|--------|-------------|
| Rice decode | One CUDA thread per Rice block (256 values) |
| FOR-BP index decode | One CUDA thread per FOR-BP block (128 rows) |
| Sparse-to-dense | One warp (32 threads) per CSR row |
| Cast (u8/u16 → f32) | One CUDA thread per element |

This is GPU parallelism rather than CPU multithreading — the CPU side launches
kernels and manages memory transfers.

## Thread safety of key types

| Type | Thread-safe? | Notes |
|------|-------------|-------|
| `ScxReader` | `Send + Sync` | Backed by `mmap` (immutable `&[u8]`). Safe to share via `Arc<ScxReader>` across threads. |
| `ScxWriter` | `Send` only | Single-owner, sequential writes. Not shared across threads. |
| `QueryPipeline` | `Send` | Built on one thread, executed on another. Not shared. |
| `TrainingPipeline` | `Send` | Owns its tokio runtime and thread handles. Called from one thread at a time. |
| `FileLock` | `Send` | Lock transferred to a single owner. Released on drop. |

## How to control parallelism

### Rayon thread pool

By default rayon uses all available CPU cores. To limit:

```rust
// Rust: set global thread pool size
rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build_global()
    .unwrap();
```

```python
# Python/Environment: set before importing pyscx
import os
os.environ["RAYON_NUM_THREADS"] = "4"
```

### Tokio worker threads

The training loader's tokio runtime is fixed at 2 worker threads (sufficient
for sequential disk I/O). The cloud runtime uses tokio's default (one thread
per core), with parallelism controlled by `PullOptions.parallelism`.

### Cloud download parallelism

```python
# Python: control parallel download tasks
pyscx.pull("gs://bucket/atlas.scxd/", "atlas.scx", parallelism=16)
```

```bash
# CLI
scx pull gs://bucket/atlas.scxd/ atlas.scx --parallelism 16
```

## Architecture diagram

```
                         ┌─────────────────────────────┐
                         │        User Code            │
                         │  (Python / R / Rust CLI)    │
                         └──────────┬──────────────────┘
                                    │
              ┌─────────────────────┼────────────────────┐
              │                     │                    │
    ┌─────────▼────────┐  ┌────────▼────────┐   ┌────────▼────────┐
    │   scx-engine     │  │   scx-loader    │   │   scx-cloud     │
    │  rayon par_iter  │  │  tokio + rayon  │   │  tokio async    │
    │  (query decode)  │  │  + std::thread  │   │  (cloud I/O)    │
    └─────────┬────────┘  └────────┬────────┘   └────────┬────────┘
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │   scx-format      │
                         │  (opt-in rayon)   │
                         │  mmap reader      │
                         └─────────┬─────────┘
                                   │
                         ┌─────────▼─────────┐
                         │   .scx file       │
                         │  (sharded CSR)    │
                         └───────────────────┘
```

For the full crate dependency graph, see [`docs/architecture.md`](architecture.md).
