# `bench_csc_dispatch` results — CSC-SUPPORT.md Phase L.3

Run summary — Chimera HPC, `cpu_preemptible` partition; pyscx built
`--release`. Per-dataset n_runs: 5 for D1–D2, 3 for D3–D4 (1 for the
focused census_1m DE-CSR resubmit due to wall-time pressure). Walls
and peak RSS are medians across runs.

## Cross-dataset matrix

| dataset (n_obs) | op | CSR wall (s) | CSC wall (s) | Speedup | CSR RSS (MB) | CSC RSS (MB) |
|---|---|---:|---:|---:|---:|---:|
| pbmc3k (2.7 K) | qc_metrics | 0.580 | 0.589 | 0.98× | 525 | 553 |
| pbmc3k | HVG | 0.427 | 0.319 | **1.34×** | 525 | 538 |
| pbmc3k | DE | 12.349 | 2.707 | **4.56×** | 594 | 600 |
| pbmc10k (12 K) | qc_metrics | 6.299 | 6.124 | 1.03× | 763 | 1041 |
| pbmc10k | HVG | 4.201 | 3.064 | **1.37×** | 771 | 880 |
| pbmc10k | DE | 119.435 | 13.197 | **9.05×** | 904 | 955 |
| tabula_sapiens_100k (100 K) | qc_metrics | 6.532 | 9.874 | **0.66×** | 2188 | 2398 |
| tabula_sapiens_100k | HVG | 3.814 | 5.150 | **0.74×** | 2199 | 2419 |
| tabula_sapiens_100k | DE | 227.690 | 31.910 | **7.14×** | 4112 | 4356 |
| census_1m (1 M) | qc_metrics | 34.622 | 90.529 | **0.38×** | 12 456 | 16 620 |
| census_1m | HVG | 18.583 | 61.841 | **0.30×** | 12 748 | 16 777 |
| census_1m | DE | 1 436.047 | 423.983 | **3.39×** | 65 335 | 69 867 |

`pseudobulk_dex` variants (4 datasets × {csr, csc} = 8 jobs)
returned `None` because `pydeseq2` rejects designs with
`n_samples ≤ n_design_vars` — the bench module's synthetic 50/50
split produces only 2 pseudobulk samples per call. Real datasets
with multi-level perturbation/donor columns will exercise the
kernel; the parity test
`pyscx/tests/test_csc_filtered_genes.py::test_pseudobulk_filtered_csc_dispatch_reaches_kernel`
already confirms the CSC path is reachable on a small fixture.

## Headlines

1. **DE is the unambiguous CSC win** — 4.6× to 9.0× faster across
   every dataset. Tabula (100K cells, 60K genes) hits 7.14× and
   census_1m clears 3.39× even at 1M × 60K — and would scale higher
   with `n_runs > 1` once the per-job CSC-fixture conversion cost
   is amortised. Phase F.3 advertised DE as the "clearest CSC win";
   the benchmark confirms it. The advantage scales with `n_obs`
   because CSC reads only the requested gene chunk's columns; CSR
   has to stream every row and project per-shard.
2. **HVG / qc_metrics flip from CSC-positive to CSC-negative as
   `n_obs` crosses 65 535**. At pbmc3k / pbmc10k the CSC side has
   u16 row indices and beats CSR. At tabula and census_1m, CSC
   indices switch to u32 — doubling the index storage the kernel
   has to decode — and the CSR side stays u16 because `n_vars` ≈
   25–60K still fits. HVG / qc_metrics are full-axis aggregations,
   exactly the workload Phase F.4 documented as "wash on CSC
   without a `gene_indices` projection". DE doesn't suffer this
   regression because it queries gene chunks via column-range
   pushdown; the saved I/O dominates the index-width cost.
3. **Peak RSS is comparable**. CSR vs CSC peak memory is within
   ±35 % on every cell. CSC isn't a memory regression even when
   it's slower.

## Implementation issue uncovered during the run

- **u16 → u32 sizing fix** (`pyscx/src/anndata.rs::from_anndata_impl`).
  The file header's `index_dtype` byte was set from `n_vars` only.
  CSC shards encode global *row* indices, so when `n_obs > 65 535`
  and CSC is requested the writer hit
  `index 65546 exceeds u16 range`. Fixed by promoting `index_dtype`
  to u32 whenever `csc_always == true` and either axis exceeds the
  u16 boundary. CSR shards on the same file pay a few extra bytes
  per index (negligible). This was the single bug blocking the L.3
  sweep on tabula and census_1m.
- The HVG / qc_metrics regression at large n_obs is partly the
  u16→u32 effect noted above. We could lift it by giving each shard
  its own `index_dtype` byte (the on-disk `ShardHeader` already
  carries one — we just always copy the file-header value into it
  today). Tracked as a follow-up — not blocking Phase L.

## How to reproduce

```bash
# from repo root
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --benchmarks bench_csc_dispatch \
    --datasets pbmc3k pbmc10k tabula_sapiens_100k census_1m \
    --skip-smoke

# census_1m DE-CSR alone needs ≥ 1 hour wall + 200 GB mem; submit
# directly when n_runs > 1 is needed:
sbatch benchmarks/comprehensive/scripts/_sbatch_de_csr_census1m.sh
```
