# Phase A.7 — OOC Peak-RSS Comparison

Per-operation median peak RSS (MB) from the `memory` benchmark. `read_full` materializes the full expression matrix; `read_subset_1k` reads a 1,000-cell slice. `Δ` is the delta RSS attributable to the operation (baseline subtracted, captured in `metadata`).

## read_full

| Dataset | SLAF (lazy) | SCX (lazy) | h5ad (uncompressed) | Zarr (zstd) | Zarr (blosc-lz4) |
|---|---|---|---|---|---|
| census_1m | 34,710.6 (Δ 2,634.3) | 345.5 (Δ 0.0) | 346.9 (Δ 2.0) | 10,994.5 (Δ 22.0) | 11,112.6 (Δ 0.1) |

## read_subset_1k

| Dataset | SLAF (lazy) | SCX (lazy) | h5ad (uncompressed) | Zarr (zstd) | Zarr (blosc-lz4) |
|---|---|---|---|---|---|
| census_1m | 34,365.0 (Δ 270.9) | 347.2 (Δ 32.7) | 346.9 (Δ 2.0) | 309.1 (Δ 3.7) | 423.7 (Δ 0.5) |

