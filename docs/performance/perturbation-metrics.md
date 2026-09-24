# Perturbation Metrics (cell-eval parity)

> Part of [SCX performance](README.md).

Rust-accelerated perturbation evaluation metrics exposed via `pyscx.accel.*` are numerically equivalent to the Python reference implementations in `cell-eval` (v0.7) (32/32 parity tests pass within the tolerances documented in [`docs/scanpy/accel-perturbation-metrics.md`](../scanpy/accel-perturbation-metrics.md#perturbation-evaluation-metrics-cell-eval-parity)). Wall-clock speedup vs the Python reference on synthetic perturbation datasets (N cells × 2K genes × 50 perturbations, 3 runs median, reference reconstructs a cold `PerturbationAnndataPair` per op for fair comparison):

| Operation | 10K | 20K ⁴ | 100K | 500K | 1M |
|-----------|----:|-----:|-----:|-----:|----:|
| Pseudobulk means | 7.8x | 11.8x | **11.6x** | **13.8x** | **19.4x** |
| Bulk metrics (pearson_delta + mse + mae + mse_delta + mae_delta, bundled) | 9.1x | 10.5x | **12.1x** | **13.6x** | **21.9x** |
| Discrimination score (L1) | 8.1x | 11.8x | **12.0x** | **12.9x** | **20.1x** |
| Energy distance (gemm + f32, default) | 30–40x ² | **52.1x** | not yet captured ² | skipped¹ | skipped¹ |
| Energy distance (gemm + f64) | ~25–35x ² | **33.0x** | not yet captured ² | skipped¹ | skipped¹ |
| Energy distance (scalar + f64, legacy alias) | 13.4x | 10.2x | **14.4x** | skipped¹ | skipped¹ |
| Clustering agreement (AMI, native Rust Leiden) | 4–5x ³ | 3.0x ³ | **10–13x** ³ | **24.6x** | **10.0x** |
| Knockdown efficiency + log deviation | 0.6x | 1.4x | 0.9x | **1.3x** | 0.7x |

¹ The cell-eval reference's `sklearn.metrics.pairwise_distances` path allocates an O(N²) distance matrix per perturbation and runs ~18 s/pert × 49 perts at 100K already (941 s/run observed); ≥ 500K would take hours for the reference alone. SCX's fused-gemm Rust kernel remains feasible at 1M+ — kernel-level scaling is tracked by the standalone criterion microbench at `scx-accel/benches/distances.rs`.

² `pyscx.accel.energy_distance` exposes `backend ∈ {"scalar", "gemm"}` and `dtype ∈ {"f32", "f64"}` kwargs. Default is `backend="auto"` (gemm for euclidean / cosine, scalar for L1) and `dtype="f32"`. The four combinations are reported as separate ops in `cell_eval_parity_perf.py`; the legacy `energy_distance` op alias preserves the `scalar + f64` (slowest) numbers for back-compat with historical baselines. Stand-alone matmul-vs-scalar speedup at 102K × 2K × 50 is 3.72× (scalar f64: 121.2 s vs gemm f64: 32.6 s); f32 vs f64 at 204K × 1K × 50 is 2.24×. Combined, the headline `gemm + f32` cuts ~7 s of cell-eval-side reference wall to a few hundred ms of SCX-side wall — speedup ratio is reference-bound, so the absolute SCX time is the more useful number for scaling decisions.

³ The scanpy `pp.neighbors` + `tl.leiden` path inside `clustering_agreement` was replaced with native-Rust `scx_accel::neighbors::build_knn_graph` + `scx_accel::leiden`, runnable under `py.allow_threads`. End-to-end on a synthetic n_perts=200 (10K cells × 300 genes), SCX takes 229 ms vs 2942 ms for the cell-eval scanpy reference (12.83× speedup; AMI score within 0.019 of the reference at `atol=0.15`). Speedup ratio varies with the centroid graph's modular structure — at small n_perts the Rust-native Leiden's RB-modularity tie-break can pick a different number of communities than scanpy's `flavor="igraph"`; the parity test was bumped from `n_perts=8 → 30` because at n_perts ≥ 16 the algorithms agree exactly on the test scaffolding. The 3.0× number at 20K cells × 50 perts is dominated by Leiden iteration count on a 49-node centroid graph; speedup grows with both centroid count and per-centroid embedding dimension.

⁴ The 20K column was captured on 2026-04-27 with the native-Rust `clustering_agreement` code path; the 100K / 500K / 1M columns are earlier measurements preserved as historical baselines for back-compat trending. New `energy_distance_*` ops are exercised at the 20K size since the cell-eval reference's O(N²) work makes the larger sizes infeasible for it (see footnote ¹). Re-running the comprehensive parity-perf suite at 100K–1M with the gemm + f32 default is queued as a follow-up SLURM job.

## Pairwise-distance kernels: Gram blocking and the `with_min_len` fix

Captured 2026-08-10 on Chimera `cpu`-partition nodes via
`cargo bench -p scx-accel --bench distances` (criterion median), plus one end-to-end
`/usr/bin/time -v` run. `d2k` = 2000 dims. `before` is `db537021`.

> [!NOTE]
> These are **Criterion microbenchmark** medians and a single `/usr/bin/time -v`
> run, with **no manifest entry** under `benchmarks/comprehensive/results/`. They
> are the second tier of
> [`docs/benchmark_manifest.md` § Scope](../benchmark_manifest.md#scope-which-claims-this-covers):
> a `benchmark`/`format`/`dataset` triple is not a shape a `cargo bench` kernel id
> can take, so the command, node, date and commit above stand in for one.
> `benchmarks/scripts/check_readme_manifests.py` enforces the first tier over
> `README.md` only and does not parse this file.

**Restoring parallelism** (16-CPU node). All four `with_min_len` calls in
`eval_metrics/distances.rs` exceeded their iterator length, so rayon never split them —
`(0..n).with_min_len(n * 16)` on the scalar self path is unsatisfiable for every `n`. Those
loops ran single-threaded at any pool size:

| Benchmark | before | after | |
|---|---:|---:|---:|
| `mean_pairwise_distance/f32/2000x2000/d2k/euclidean/scalar` | 6.970 s | 873.8 ms | **7.98×** |
| `mean_pairwise_distance_self/f32/2000/d2k/euclidean/scalar` | 3.481 s | 436.3 ms | **7.98×** |
| `mean_pairwise_distance_self/f32/2000/d2k/cosine/scalar` | 5.388 s | 665.8 ms | **8.09×** |
| `mean_pairwise_distance_self/f64/2000/d2k/euclidean/scalar` | 2.558 s | 174.3 ms | **14.67×** |
| `mean_pairwise_distance_self/f64/2000/d2k/cosine/scalar` | 2.619 s | 280.0 ms | **9.35×** |
| `mean_pairwise_distance/f32/2000x2000/d2k/euclidean/gemm` | 26.77 ms | 19.36 ms | **1.38×** |
| `mean_pairwise_distance_self/f32/2000/d2k/euclidean/gemm` | 26.79 ms | 19.07 ms | **1.40×** |
| `mean_pairwise_distance_self/f64/2000/d2k/cosine/gemm` | 47.25 ms | 41.58 ms | **1.14×** |

The gemm rows are the same fix applied to the Gram-expansion loop, **not** the upper-triangle
change: the cross benchmark, which has no triangle, gains the same 1.2–1.4×.

**What blocking costs, and what the triangle buys** (32-CPU node, same shape both arms,
`2000 × 2000 × d2k` f32 euclidean gemm). Every shape in the bench grid is a single block at the
256 MiB default — the grid tops out at 200 MB — so the triangle's effect is invisible there and
had to be measured against a forced 2 MB budget (8 blocks):

| | 1 block (default) | 8 blocks (2 MB budget) | cost of blocking |
|---|---:|---:|---:|
| full square (`mean_pairwise_distance(a, b)`) | 13.28 ms | 26.79 ms | 2.02× |
| upper triangle (`…_self(a)`) | 12.96 ms | 19.46 ms | 1.50× |
| **triangle vs. square** | 1.03× | **1.38×** | |

At one block the triangle narrows only the expansion loop — the columns operand is still all of
`b` — so it is worth ~3 %. Once blocked, later blocks take a narrower `b`, total gemm work drops
to `≈ (k+1)/2k` of the square at `k` blocks, and the triangle is what makes blocking nearly free
on the self path (1.50× vs 2.02×).

**End to end** (`pyscx.accel.energy_distance`, 30 K control + 20 × 500 perturbation cells × 50
dims, `RAYON_NUM_THREADS=16`). The A/B lever is the budget knob itself: a budget larger than the
whole Gram reproduces the pre-fix single allocation, so both arms are the same binary.

| | peak RSS | kernel wall |
|---|---:|---:|
| one block (budget ≫ Gram) | 3.85 GB | 3.9 s |
| default 256 MiB budget | **1.76 GB** | **1.0 s** |

Identical correlation to 9 decimal places. Blocking is *faster* here, the opposite of the 2000-dim
microbench: at 50 dims the gemm is memory-bound, and a block that survives in cache between the
matmul and the expansion beats streaming a 3.6 GB Gram through DRAM. So the throughput cost of
blocking is real but `n_dims`-dependent — it bites on raw-gene inputs and pays on the
`embed_key="X_pca"` inputs the metric is normally run on.

Note that 1.76 GB is far above the 256 MiB budget, and that is expected rather than a miss: the
budget bounds **one block**, and `energy_distance` runs perturbations on a rayon `par_iter`, so
the Gram term is `RAYON_NUM_THREADS × budget` on top of each task's own dense copy of its group's
rows (`extract_group_rows_indexed`, untouched here).

Speedups grow with cell count for the pseudobulk-driven metrics (pseudobulk, bulk_metrics, discrimination_l1) — single-pass streaming aggregation in Rust wins harder as the per-cell work scales. `knockdown_efficiency` is within ±40% of a tight NumPy column-access loop and is not currently a speedup target.

The table above is the SCX-Rust-vs-Python-reference speedup on the **CPU**. `perturbation_metrics` and `energy_distance` (euclidean / cosine) also accept `device="gpu"` (pyscx ≥ 0.11.2) — CPU-vs-GPU wall times are in [GPU perturbation-evaluation metrics](gpu.md#gpu-perturbation-evaluation-metrics) below.

Full per-operation results (wall time + peak RSS) are tracked in `benchmarks/comprehensive/results/raw/cell_eval_parity_perf__scx_auto__pert_synth_*.json` and rendered in the "Cell-eval Parity Performance" section of the comprehensive benchmark report. Kernel-level distance-kernel microbenchmarks live in `scx-accel/benches/distances.rs` (run via `cargo bench -p scx-accel --bench distances`; see [`benchmarks/README.md`](../../benchmarks/README.md#rust-microbenchmarks-criterion)).
