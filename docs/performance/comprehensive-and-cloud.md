# Comprehensive Benchmarking + Cloud Validation

> Part of [SCX performance](README.md). Cloud setup is in [docs/cloud.md](../cloud.md).

Closes SLAF parity, cloud validation on GCS, fragment-ops
throughput, and the regression gate. Numbers below come from live
benchmark runs on a Chimera CPU node against
`gs://arc-ctc-nextflow/scx-test/` across all four primary cloud
formats (SCX, Zarr v3, TileDB-SOMA, SLAF). See `docs/cloud.md` for
cloud-specific operational notes. The original post-ship known issues
are all resolved: the zarr `cloud_read` decompression failure (a
fixture-upload race — fixed with fcntl-serialized uploads + BLAKE3
sidecars), the SLAF cloud-path probe failure (missing
`smart_open[gcs]` dep — fixed by pinning `google-cloud-storage` in
`scx-bench-slaf.yml`), and the missing selective-predicate coverage
(no `n_counts` obs column on the staged h5ads — fixed by
`benchmarks/scripts/augment_obs_n_counts.py` populating
`obs["n_counts"] = X.sum(axis=1)` during dataset prep).

## SLAF parity

SLAF (`slafdb==0.5.2`) is a first-class competitor across every
comprehensive-suite dimension — compression, full read, selective read,
filtered-query pushdown (SQL via its DuckDB engine), correctness
round-trip, ML loader, and out-of-core memory. Key results on census_1m:

| Metric | SCX | SLAF | Zarr (zstd) | h5ad (backed) |
|---|---:|---:|---:|---:|
| Full-read peak RSS | ~345 MB | ~34 GB | ~11 GB | ~345 MB |
| ML-loader batches/s | 1,405 | ~4.1 | n/a | n/a |
| Selective `cell_type == "T cell"` | scx_pushdown | slaf_sql | skipped | h5ad_load_and_mask |

SLAF's Mixture-of-Scanners prefetcher returns 0 batches at 10M scale
with the default config — flagged as a SLAF-upstream tuning issue, not
a harness fix.

## Cloud push / pull throughput (SCX → GCS)

`pyscx.push` / `pyscx.pull` streaming throughput on the default
`.scxd/` layout. Per-request overhead dominates on tiny files; the
100K-cell dataset is where bandwidth matters:

| Dataset | Size | Push | Pull |
|---|---:|---:|---:|
| pbmc3k | 4 MB | 12.4 MB/s | 0.7 MB/s |
| tabula_sapiens_100k | 428 MB | **114.7 MB/s** | **181.1 MB/s** |

The 50 MB/s absolute floor in `thresholds.yaml` is keyed on
tabula_sapiens_100k (pbmc3k is deliberately below the bandwidth
regime). Comfortable ~2× headroom vs the floor.

## Cloud full-dataset read (cross-format)

Full-dataset cloud read — pull from GCS + materialize to in-memory
AnnData via each format's native cloud read path:

| Dataset | SCX | Zarr (zstd) | Zarr (lz4) | TileDB-SOMA | SLAF |
|---|---:|---:|---:|---:|---:|
| pbmc3k | **0.165s** | 0.627s | 0.556s | 0.944s | 1.873s |
| tabula_sapiens_100k | **2.815s** | 4.593s | 5.336s | 4.570s | 5.306s |

SCX leads on both datasets. At 100K cells, SCX is **1.6× faster than
TileDB-SOMA**, **1.6× faster than zarr (zstd)**, and **1.9× faster
than SLAF**. Mechanisms: `scx_pull_and_load`, `zarr_cloud_open`,
`soma_open_gs`, `slaf_cloud`.

## Cloud metadata open — single-GET catalog parse

Time to open the cloud-hosted fixture and surface obs / var schema
(no X materialization):

| Dataset | SCX (`open_cloud`) | Zarr (zstd) | Zarr (lz4) | TileDB-SOMA | SLAF |
|---|---:|---:|---:|---:|---:|
| pbmc3k | **0.097s** | 0.120s | 0.113s | 0.274s | 0.800s |
| tabula_sapiens_100k | **0.114s** | 0.110s | 0.112s | 0.338s | 0.754s |

SCX and zarr metadata latency are essentially dataset-size-independent
(SCX 0.097 → 0.114s going from 2.7K to 100K cells), as expected for a
single-GET catalog fetch against the exploded `.scxd/` front catalog.
TileDB-SOMA's open path does a handful of extra directory listings;
SLAF's metadata open includes loading cells/genes Lance fragments.

## Cloud filtered query (predicate pushdown)

Per-predicate median wall across the canonical predicate set
(`cell_type == "T cell"`, `n_counts > 1000`, `random_1pct`). `pbmc3k`
has no `cell_type` obs column, so the eq-predicate is excluded at
runtime by `_applicable_predicates` (`cloud_filtered.py`):

**pbmc3k** (no `cell_type`):

| Format | `n_counts > 1000` | `random_1pct` |
|---|---:|---:|
| SCX (scx_pull_and_filter) | **0.539s** | **0.530s** |
| TileDB (tiledb_cloud_value_filter) | 0.983s | 0.907s |
| SLAF (slaf_cloud_sql / slaf_cloud_stride_hash) | 1.418s | 1.216s |

**tabula_sapiens_100k**:

| Format | `cell_type == 'T cell'` | `n_counts > 1000` | `random_1pct` |
|---|---:|---:|---:|
| SCX | **2.07s** | 5.39s | 13.59s |
| TileDB | 0.80s | 9.27s | **3.04s** |
| SLAF | 1.00s | **4.97s** | 3.16s |

At 100K cells the winner rotates by predicate — TileDB's categorical
enum index wins `cell_type` equality (0.80s vs SCX 2.07s), SLAF's
DuckDB streaming WHERE wins `n_counts > 1000`, and TileDB wins
`random_1pct` via its cell-id coordinate sampler. SCX leads on
pbmc3k across the board. The current SCX cloud-filtered mechanism
is `scx_pull_and_filter` (pull full shard + local filter); a native
range-read pushdown variant is a roadmap item — numeric-range
predicates like `n_counts > X` don't currently skip shards because
catalog pushdown keys on `CategoryBitset` indices only. Zarr cloud
fixtures don't persist obs (the raw-CSR converter in `zarr_runner`
writes `indptr` / `indices` / `data` only), so zarr rows are absent
from this table by design.

## Cost model — USD per 1M cells queried (GCS same-region pricing)

Priced against the pinned `GCS_PRICING` table
(Class-B $0.004/10k GETs, same-region egress $0.00/GB on intra-region
GCE ↔ GCS). Cost is dominated by full-read egress at large scale;
metadata opens are effectively free in the committed regime. Selective
scenarios use the `n_counts > quantile_cutoff` predicate synthesized
by `_n_counts_threshold_predicates` against the augmented obs column:

| Dataset | Metadata | selective 5% | selective 20% | Full read |
|---|---:|---:|---:|---:|
| pbmc3k | $0.000000 | $0.011852 | $0.002963 | $0.000593 |
| tabula_sapiens_100k | $0.000000 | $0.000800 | $0.000200 | $0.000040 |

Per-1M-cells cost *decreases* with dataset size because the per-GET
overhead amortizes over more cells. Note that selective rows are
currently **more expensive** per-million-cells than full-read: the
denominator (matching cells) shrinks but the byte count stays roughly
constant because SCX's current catalog pushdown doesn't skip shards
on numeric-range predicates like `n_counts > X` (catalog pushdown
keys on `CategoryBitset`-indexed columns only). This is the honest
measurement the cost model is designed to surface — the selective
pull downloads the same bytes as a full pull but is accounted against
the matching-cell subset. Row-group skipping on numeric ranges is a
candidate roadmap item; it would shift the selective columns below
the full-read column.

## Cloud reader vs full pull (metadata workloads)

`open_cloud` streams only the front catalog; `pull_full` fetches the
entire fixture. Bytes transferred reflect what the underlying GCS
reads actually download:

| Dataset | `open_cloud` wall | `pull_full` wall | `pull_full` bytes |
|---|---:|---:|---:|
| pbmc3k | 0.115s | 0.315s | 4.4 MB |
| tabula_sapiens_100k | **0.117s** | **2.31s** | **408 MB** |

On tabula_sapiens_100k, `open_cloud` is **~20× faster** than a full
pull and avoids transferring 408 MB — the core "cloud-aware access"
win that justifies the exploded `.scxd/` layout.

## GCP compute-node matrix

Cloud-read throughput characterized across `n2-standard-8`,
`c3-standard-8`, and `a3-highgpu-1g`. Per-VM egress bandwidth class
(16 / 23 / 200 Gbps) is the dominant predictor for full-read wall
clock on atlases that fit the streaming-pull envelope. The launcher
(`submit_gcp_matrix.py`) pins every VM to the bucket region so
cross-region egress is impossible by construction. Results in §8d of
the benchmark report; raw numbers require `--yes-spend` to generate.

## Fragment operations throughput

`pyscx.append` / `mark_deleted` / `compact` / `rollback` throughput on
pbmc3k (see §8b):

| Operation | Median wall | Dominant throughput |
|---|---:|---:|
| append | streaming (one shard at a time for SCX→SCX; raw-copy fast path when codec matches) | ~38 MB/s |
| delete (logical) | independent of n_obs | ~155 k rows/s |
| compact | base-file read + re-encode bandwidth | ~54 MB/s |
| rollback | single root-catalog pwrite | ~3 ms |

## Regression gating

All benchmark results carry a `schema_version=1` stamp + full
provenance (git SHA, thread pinning, run_id) in their
`system.provenance` block. The on-demand gate
(`scripts/gate_candidate.py` + `scripts/compare_against_baseline.py
--gate`) evaluates relative tolerances (3% wall / 10% RSS / 1% size),
absolute floors from `thresholds.yaml` (e.g. cloud throughput ≥ 50
MB/s, keyed on tabula_sapiens_100k), and disappeared-benchmark
detection. Justification markdown files under
`results/justifications/` suppress accepted regressions with an
optional expiry date. The dashboard (`reporting/dashboard.py`) emits a
browsable HTML snapshot alongside the markdown report, threaded with
"← previous snapshot" navigation via `dashboard_history.json`.

Two baselines coexist: `LATEST` (currently
`v0.6.5-accel-gpu-to-gpu-anndata`, accel-only) is the default gate
target for accelerator PRs; format / cloud / multimodal PRs pin
`v0.6.2-n_counts-augmentation` explicitly. See the [Canonical
baseline](gpu.md#canonical-baseline) table above for coverage details.
`gate_candidate.py` with no flags gates against `LATEST`.
