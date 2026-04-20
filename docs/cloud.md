# Using SCX in Cloud Environments

SCX is designed for cloud-native workflows: large atlases live on object storage
(GCS, S3, Azure Blob), and compute nodes pull only what they need — often just
metadata, or just the shards matching a predicate. This guide covers how to set
up credentials, choose the right on-cloud layout, and tune throughput.

For related references:
- [docs/architecture.md §Cloud Operations](architecture.md#cloud-operations-scx-cloud) — crate internals
- [docs/api.md §scx-cloud](api.md#scx-cloud--cloud-operations) — API reference
- [docs/sharding.md §Cloud operations](sharding.md#cloud-operations-shard-level-selectivity) — shard-level selective pull
- [docs/multithreading.md §Cloud I/O](multithreading.md#cloud-io) — concurrency model

## Enabling cloud support

Cloud operations are gated behind the `cloud` feature flag:

```bash
# Rust
cargo build --features cloud
cargo test --workspace --features cloud

# Python bindings
cd pyscx && ../.venv/bin/maturin develop --features cloud

# CLI
cargo install --path scx-cli --features cloud
```

Without this flag, `pyscx.pull`, `pyscx.push`, `pyscx.open_cloud`, and the
`scx pull` / `scx push` / `scx cloud-optimize` / `scx explode` / `scx pack`
subcommands are not available.

## Supported providers and URL schemes

SCX uses the [`object_store`](https://docs.rs/object_store) crate and inherits
its provider support:

| Scheme | Provider | Example |
|--------|----------|---------|
| `gs://`  | Google Cloud Storage | `gs://my-bucket/path/atlas.scxd/` |
| `s3://`  | Amazon S3 (and S3-compatible endpoints)  | `s3://my-bucket/path/atlas.scxd/` |
| `az://`  | Azure Blob Storage | `az://my-container/path/atlas.scxd/` |
| `file://` or bare path | Local filesystem (dev / on-prem) | `/mnt/data/atlas.scxd` |

A URL without a recognized scheme is treated as a local path, which makes it
easy to develop against a local directory and switch to cloud by changing the
URL only.

## Authentication

SCX does not manage credentials itself — it relies on the default credential
chain of `object_store`. Configure your environment the same way you would for
`gcloud`, `aws`, or `az` CLI.

### Google Cloud Storage

```bash
# Service account key file
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json

# Or: run on GCE/GKE/Cloud Run — instance metadata is used automatically
# Or: `gcloud auth application-default login` for local dev
```

### Amazon S3

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-west-2           # required
export AWS_SESSION_TOKEN=...          # if using STS / assumed role

# Or: run on EC2/EKS/ECS — instance profile is used automatically
# Or: `aws sso login` sets up credentials in ~/.aws
```

For S3-compatible endpoints (MinIO, Cloudflare R2, Wasabi), additionally set:

```bash
export AWS_ENDPOINT=https://s3.example.com
export AWS_ALLOW_HTTP=true            # only if endpoint is http://
```

### Azure Blob Storage

```bash
export AZURE_STORAGE_ACCOUNT=myaccount
export AZURE_STORAGE_KEY=...          # shared key
# Or: AZURE_STORAGE_SAS_TOKEN, or managed identity on Azure VMs
```

Credentials are picked up lazily when the first operation runs; misconfigured
credentials surface as a runtime error from the underlying `object_store`.

## Layouts on object storage

SCX has three ways to live in a bucket. Pick based on how you read the data.

| Layout | Structure | Best for | Write via |
|--------|-----------|----------|-----------|
| **Exploded `.scxd/`** | Directory of objects (one per shard/section) | Random / selective access, selective pull, multi-reader parallel download | `pyscx.push(...)` or `scx push` |
| **Cloud-optimized `.scx`** | Single object with front-of-file catalog | Whole-file download; read metadata in one GET | `pyscx.cloud_optimize(...)` → upload the file |
| **Plain `.scx`** | Single object, catalog at EOF | Existing files you haven't cloud-optimized yet | Any upload |

**Default recommendation**: exploded `.scxd/` for datasets that multiple users
or jobs will slice differently, cloud-optimized `.scx` for archive / portable
distribution of a single file.

### Exploded `.scxd/` layout

```
s3://bucket/atlas.scxd/
├── _header.bin               # 256-byte file header
├── _catalog.bin              # Full catalog (uploaded LAST)
├── obs.arrow                 # Cell metadata
├── var.arrow                 # Gene metadata
├── X/
│   ├── 000000.shard          # CSR shards — byte-identical to the packed form
│   ├── 000001.shard
│   └── ...
├── obsm/                     # Optional embeddings
├── layers/                   # Optional layers
├── uns.json                  # Optional unstructured metadata
├── _provenance.bin           # Optional
└── _deletion_vectors.bin     # Optional
```

`_catalog.bin` is uploaded last, which gives the `.scxd/` directory
**atomic-publish semantics**: until the catalog object exists, a reader opening
the URL gets a clean "not found" error rather than a half-written dataset.

### Cloud-optimized packed `.scx`

A cloud-optimized `.scx` is the same format as a local `.scx` plus a **front
catalog** (a duplicate of the full catalog) placed right after the header, so
a reader can bootstrap the file's structure with a single range read (typically
256 KB) instead of first issuing a HEAD and a separate read for the catalog at
EOF. See [docs/format.md §Cloud Layouts](format.md#14-cloud-layouts).

```bash
# One-shot conversion
scx cloud-optimize atlas.scx --output atlas.cloud.scx
# then `gsutil cp atlas.cloud.scx gs://bucket/`
```

```python
pyscx.cloud_optimize("atlas.scx", "atlas.cloud.scx")
```

`pull` produces a cloud-optimized file by default (`cloud_ready=True`), so
pulling and re-uploading is another way to make a file cloud-ready.

## Operations

### `pull` — stream cloud → local `.scx`

Downloads shard objects in parallel, streams them into a single local `.scx`
file through a reorder buffer, and writes the catalog last.

```python
# Full dataset
pyscx.pull("gs://bucket/atlas.scxd/", "atlas.scx")

# Selective: only shards matching the predicate — uses catalog-level pushdown
pyscx.pull(
    "gs://bucket/atlas.scxd/",
    "t_cells.scx",
    filter="cell_type == 'T cell'",
    parallelism=16,
)
```

```bash
scx pull gs://bucket/atlas.scxd/ atlas.scx --parallelism 16
scx pull gs://bucket/atlas.scxd/ lung.scx --filter "tissue == 'lung'"
```

Selective pulls can reduce bandwidth by up to ~20× on well-sharded datasets
(see [docs/sharding.md]), because the catalog's per-shard `CategoryBitset` is
consulted before any shard is downloaded.

Return value (Python): dict with `bytes_downloaded`, `sections_downloaded`,
`elapsed_secs`, `throughput_mbps`. Filtered pulls add `total_shards`,
`downloaded_shards`, `skipped_shards`, `matching_cells`, `bytes_saved`.

#### Interrupted pulls — idempotent retry, not resumable

Pulls are **idempotent-retry**, not resumable-from-checkpoint. The
implementation writes to `{dest}.tmp.{pid}` and atomically renames on
completion, so:

- A SIGTERM or crash leaves `{dest}.tmp.{pid}` orphaned on disk but does
  **not** block subsequent pulls — a fresh run uses a different PID-keyed
  path and produces the final output atomically.
- On entry, every `pull` / `pull_filtered` invocation sweeps any
  `{dest}.tmp.*` siblings it finds, so orphaned temp files don't
  accumulate across retries.
- There is **no shard-level checkpoint** — a retried pull re-downloads
  every shard. This is a deliberate design choice: SCX shards are
  independent and small enough that re-download cost is bounded, and a
  checkpoint file would introduce cross-invocation state that defeats
  the current atomic-rename safety property.
- If your pull was killed mid-run and you need to know the committed
  state on disk, check for `{dest}` (fully written) vs `{dest}.tmp.*`
  (in-flight, safe to delete). The next pull will sweep the `.tmp.*`
  automatically.

The `CloudError::Interrupted` enum variant is reserved for callers that
want to explicitly signal an interruption in downstream orchestration
(e.g., surfacing to the regression gate); the pull implementation itself
does not raise it today since interruptions in the streaming pipeline
surface as `object_store::Error` / `io::Error` at the failing GET.

### `push` — stream local `.scx` → cloud `.scxd/`

```python
pyscx.push("atlas.scx", "gs://bucket/atlas.scxd/", parallelism=16)
```

```bash
scx push atlas.scx gs://bucket/atlas.scxd/ --parallelism 16
```

Uploads every section as its own object in parallel, uses multipart upload for
sections larger than 8 MB, and uploads `_catalog.bin` last. No intermediate
local directory is created.

### `open_cloud` — metadata-only handle

```python
exp = pyscx.open_cloud("gs://bucket/atlas.scxd/")
exp.n_obs          # 1_200_000
exp.n_vars         # 36_601
exp.nnz            # 3_400_000_000
exp.shard_count    # 120
```

`open_cloud` auto-detects the layout (exploded / cloud-ready packed / plain
packed) and returns a `PyCloudExperiment` whose metadata accessors require only
the header + catalog. Use this to decide what to pull before paying for the
bytes.

### `explode` / `pack` — convert layouts locally

```bash
scx explode atlas.scx atlas.scxd/      # packed → exploded
scx pack    atlas.scxd/ atlas.scx      # exploded → packed
```

Useful for preparing an upload locally (upload a directory with `gsutil rsync`)
or for inspecting an exploded directory that someone else published.

## Choosing the right access pattern

```
                                    ┌─ Read the whole dataset once per node?
                                    │    └─ `pull`, then work from the local .scx
                                    │
Multiple nodes, ML training ────────┼─ Each reads a random slice?
                                    │    └─ Exploded .scxd + `pyscx.pull(..., filter=...)`
                                    │       per node, or the training loader
                                    │
Interactive exploration ────────────┼─ Just need to know if this dataset has T cells?
                                    │    └─ `pyscx.open_cloud` — catalog-only, cents per query
                                    │
Distribution / archive ─────────────┴─ Publishing one file for others to download?
                                         └─ `cloud_optimize` + upload as a single .scx
```

Heuristics:

- **Always prefer selective pull** when you have a clear predicate (`cell_type`,
  `tissue`, `batch`). It scales inversely with selectivity, and egress cost is
  usually the bottleneck, not compute.
- **Cloud-optimize once, distribute many times.** For a published atlas that
  users will download whole, `cloud_optimize` saves one round-trip per reader.
- **Exploded `.scxd` is better for ML training** — independent shard objects
  allow parallel, resumable downloads and shard-level caching.

## Tuning throughput

| Knob | Default | When to change |
|------|---------|----------------|
| `parallelism` on `pull` / `push` | 8 | Raise to 16–32 on high-bandwidth links (10+ Gbps) or large shard counts. Diminishing returns past #cores. |
| `PullOptions.reorder_buffer` (Rust) | 4 shards | Raise for lopsided shard sizes so fast downloads don't stall waiting for one slow shard. |
| `PushOptions.multipart_threshold` (Rust) | 8 MB | Match to your provider's recommended part size (S3: 16 MB; GCS: 32 MB is fine). |
| Shard size at write time | 10k cells | Smaller shards → finer pushdown granularity, but more objects and more request overhead. See [docs/sharding.md]. |
| `RAYON_NUM_THREADS` | #cores | Affects downstream decode after download. Does **not** control download parallelism — that's `parallelism`. |

### Request cost vs. bandwidth cost

Cloud providers charge per-request *and* per-byte. A dataset exploded into many
tiny shards optimizes selectivity but can be request-heavy. For dense reads
(pull the whole thing), a single cloud-optimized `.scx` is cheaper: ~3 GET
requests (header + front catalog + body range) vs one GET per shard.

Coalescing (docs/cloud.md (Range coalescing)) is used internally when reading ranges from a packed
file — adjacent sections are merged into single GETs with a configurable gap
threshold, which further reduces request count.

## Running on cloud compute

### Google Cloud (GCE / GKE / Batch / Vertex)

- Attach a service account with `storage.objectViewer` (read) or
  `storage.objectAdmin` (read/write) on the bucket. No env var needed —
  instance metadata is picked up automatically.
- Use the same region for compute and bucket to avoid egress charges.
- For Vertex AI Training, set the bucket via `gcsfuse` or just read with
  `pyscx.pull(...)` — the latter is faster than gcsfuse for SCX reads.

### AWS (EC2 / EKS / Batch / SageMaker)

- Attach an instance profile / IRSA role with `s3:GetObject` + `s3:ListBucket`
  (read) or also `s3:PutObject` + `s3:AbortMultipartUpload` (push). No env vars
  needed when the role is attached.
- Pin `AWS_REGION` — `object_store` does not auto-discover it from the instance
  profile in all cases.
- Prefer VPC endpoints (`com.amazonaws.<region>.s3`) for private-subnet nodes
  to avoid NAT gateway cost.

### Azure (VMs / AKS / ML)

- Enable a system-assigned or user-assigned managed identity on the VM / AKS
  node pool and grant `Storage Blob Data Reader` / `Contributor` on the
  container.
- `AZURE_STORAGE_ACCOUNT` still needs to be set — the managed identity only
  supplies the secret, not the account name.

### Shared filesystems on HPC (Lustre / GPFS / NFS)

SCX works directly on shared filesystems without the cloud code path —
advisory `flock()` degrades gracefully when not supported, and readers never
block (see [docs/multithreading.md §Concurrent file access]). Use local paths,
not `gs://` / `s3://`, in this case.

## GIL release and concurrency

`pyscx.pull`, `pyscx.push`, and `pyscx.open_cloud` all release the GIL while
blocking on cloud I/O, so a Python process can overlap cloud downloads with
other work (e.g., pulling multiple datasets concurrently via threads). The
underlying tokio runtime fans out `parallelism` async tasks; see
[docs/multithreading.md §Cloud I/O].

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `InvalidUrl("unsupported scheme ...")` | Wrong scheme, e.g., `https://` | Use `gs://` / `s3://` / `az://` or a local path |
| `ObjectNotFound` on `open_cloud` | `.scxd/` not fully published (no `_catalog.bin`) | Wait for the `push` to finish or re-run it |
| S3 403 with valid keys | Region mismatch | Set `AWS_REGION` to the bucket's region |
| `pull` slow, CPU idle | Few shards, low `parallelism` | Raise `parallelism`, or re-shard the source (smaller shards at write time) |
| `pull` slow, CPU at 100% | Decompression-bound, not I/O-bound | Already saturating — use a larger instance or a different codec (Zstd is slower than Scx1) |
| High request bill | Too many tiny shards, many exploratory `open_cloud` calls | Increase shard size on write; cache `PyCloudExperiment` handles |
| Credentials picked up from wrong source | Shell has stale env vars + instance profile | Unset `AWS_*` / `GOOGLE_APPLICATION_CREDENTIALS` to force the instance credential path |

## End-to-end example

```python
import pyscx

# 1. Peek at a published atlas without downloading
exp = pyscx.open_cloud("gs://arc-atlases/tabula_sapiens.scxd/")
print(exp.n_obs, exp.n_vars, exp.shard_count)

# 2. Pull only lung cells
stats = pyscx.pull(
    "gs://arc-atlases/tabula_sapiens.scxd/",
    "lung.scx",
    filter="tissue == 'lung'",
    parallelism=16,
)
print(stats)  # {'downloaded_shards': 7, 'skipped_shards': 113, ...}

# 3. Work locally as usual
adata = pyscx.open("lung.scx").to_anndata()

# 4. Publish a derived dataset back to cloud as .scxd/
adata.write_scx("lung_pca.scx")
pyscx.push("lung_pca.scx", "gs://my-bucket/lung_pca.scxd/", parallelism=16)
```

[docs/sharding.md]: sharding.md
[docs/multithreading.md §Concurrent file access]: multithreading.md#concurrent-file-access
