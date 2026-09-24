# GPU Acceleration (NVIDIA H100)

> Part of [SCX performance](README.md). Setup is in [docs/gpu-setup.md](../gpu-setup.md).

## Codec Decode and Training Pipeline

| Operation | Size | CPU (us) | GPU (us) | Speedup |
|-----------|------|----------|----------|---------|
| FOR-BP index decode | 16K rows, 33M nnz | 102,900 | 4,133 | **24.9x** |
| Sparse -> dense | 16K rows x 30K cols | 433,252 | 7,711 | **56.2x** |
| Sparse -> dense (HVG 2K) | 16K rows x 2K output | 110,416 | 897 | **123.1x** |

## GPU Analysis Pipeline

GPU-accelerated analysis via **rapids-singlecell** (`rsc.pp.pca`, `rsc.pp.neighbors`, `rsc.tl.umap`, `rsc.pp.*`) for in-VRAM ops, plus native Rust/CUDA paths for streaming PCA, HVG `seurat_v3`, Leiden (Rust-native CPU + cuGraph GPU), DE Wilcoxon rank-sum/pdex (CSC/CSR-direct), Harmony, and codec decode. Benchmarked on H100 80GB (driver 560.35.05, CUDA 12.6, scx-bench-gpu conda env).

**GPU VRAM usage.** `to_gpu_anndata()` preserves sparse CSR on the GPU — VRAM
for `X` scales with NNZ, not N×M. A 1M-cell × 2K-gene HVG-selected matrix at
5% density occupies ~800 MB as sparse CSR (vs ~8 GB dense). Peak VRAM during
an operation also includes per-op working memory (PCA dense working matrices,
kNN embeddings, DE per-chunk intermediates). rapids-singlecell ops may allocate
additional dense working buffers internally. Use
`pyscx.accel.estimate_gpu_memory(adata, operation=...)` for pre-flight sizing;
`to_gpu_anndata()` includes a VRAM pre-flight guard (1.2× headroom) that raises
`ValueError` if insufficient. See
[gpu-setup.md § GPU memory model](../gpu-setup.md#gpu-memory-model) for the full
sizing model. Formal per-op peak-VRAM benchmarks are planned.

Numbers below are from the full-tier gate run on 2026-05-25 (post-G10 graph capture for GPU DE + bench env-routing fix). The per-job conda-env routing fix (`run_parallel.py::_env_for_format`) unlocked real GPU coverage for Leiden + kNN that prior baselines silently missed (workers were running on the orchestrator's env which lacked cuGraph + cuVS — see `benchmarks/README.md` § Environment notes for the routing details).

### Per-operation timing

| Operation | Dataset | CPU (s) | GPU (s) | Speedup | Backend |
|-----------|---------|---------|---------|---------|---------|
| PCA (50 PCs, auto-routed) | tabula_sapiens_100k | 2.79 (`pyscx_cpu_auto`) | 0.38 | **7.3×** | rapids `rsc.pp.pca` (in-VRAM) |
| PCA (50 PCs, auto-routed) | census_500k | 1.89 | 0.79 | 2.4× | rapids `rsc.pp.pca` (in-VRAM) |
| PCA (50 PCs, auto-routed) | census_1m | 2.77 | 1.56 | 1.8× | rapids `rsc.pp.pca` (in-VRAM) |
| PCA correctness (cos sim vs scanpy, top-50) | pbmc3k | — | — | **min=0.999911** | — |
| PCA correctness (cos sim vs scanpy, top-50) | census_1m | — | — | **min=1.0** | — |
| kNN (k=15, 50 PCs) | tabula_sapiens_100k | 5.68 (`scanpy_cpu`) | 4.28 | 1.3× | rapids `rsc.pp.neighbors` |
| kNN (k=15, 50 PCs) | census_500k | 36.36 | 12.75 | **2.9×** | rapids `rsc.pp.neighbors` |
| kNN (k=15, 50 PCs) | census_1m | 91.97 | 26.67 | **3.4×** | rapids `rsc.pp.neighbors` |
| UMAP (2D) | tabula_sapiens_100k | 50.82 (`scanpy_cpu`) | 2.58 | **20×** | rapids `rsc.tl.umap` |
| UMAP (2D) | census_500k | 364.51 | 10.61 | **34×** | rapids `rsc.tl.umap` |
| UMAP (2D) | census_1m | 846.89 | 27.91 | **30×** | rapids `rsc.tl.umap` |
| UMAP trustworthiness | pbmc3k | 0.9238 | 0.9233 | — | vs PCA space |
| Leiden (`device="cpu"`, Rust-native) | tabula_sapiens_100k | 3.41 | — | — | `scx_accel::leiden` |
| Leiden (`device="gpu"`, cuGraph) | tabula_sapiens_100k | 100.73 (`leidenalg_cpu`) | 0.54 | **187×** | cuGraph |
| Leiden (`device="gpu"`) | census_500k | 838.94 (`leidenalg_cpu`) | 1.59 | **528×** | cuGraph |
| Leiden (`device="gpu"`) | census_1m | 659.0 (`leidenalg_cpu`, prior baseline) | 3.06 | 215× | cuGraph |
| Wilcoxon rank-sum (vs `pyscx_cpu` reference) | pbmc10k | 5.85 | 12.60 | 0.47× | CUB block sort + searchsorted + tie + p-value |
| Wilcoxon rank-sum (vs `pyscx_cpu`) | tabula_sapiens_100k | 42.28 | 89.77 | 0.47× | (same) |
| Wilcoxon rank-sum (vs `pyscx_cpu`) | census_500k | 106.13 | 213.07 | 0.50× | (same) |
| Wilcoxon rank-sum (vs `pyscx_cpu`) | census_1m | 156.51 | 365.15 | 0.43× | (same) |
| Wilcoxon rank-sum (vs `scanpy_cpu`) | tabula_sapiens_100k | 318.77 | 89.77 | **3.6×** | (same) |
| pdex_ref (vs `pyscx_cpu`) | pbmc10k | 5.52 | 12.58 | 0.44× | (same) |
| pdex_ref (vs `pyscx_cpu`) | tabula_sapiens_100k | 42.04 | 93.81 | 0.45× | (same) |
| pdex_ref (vs `pyscx_cpu`) | census_500k | 119.86 | 212.37 | 0.56× | (same) |
| pdex_ref (vs `pyscx_cpu`) | census_1m | 268.28 | 356.34 | 0.75× | (same) |

The accel_de Wilcoxon rank-sum/pdex_ref GPU rows above are **slower than `pyscx_cpu`** (CPU's rayon-parallel implementation effectively uses ~3-5 of the 16 SLURM-allocated CPUs and is highly tuned). G10's graph capture closed ~7-10% of the gap but the GPU implementation is bottlenecked by the per-chunk `[n_obs × chunk_size]` dense materialization step. The GPU paths are still **3-5× faster than `scanpy_cpu`** — for users replacing scanpy directly, GPU is the clear win; for users who already have `pyscx.accel.rank_genes_groups(device="cpu")` working, the default GPU variant is a draw or worse.

**`pdex_ref` GPU v3-CSC (default, requires CSC sidecar).** An SCX file with a CSC sidecar (built by default under `csc="auto"` on qualifying datasets, or explicitly via `pyscx.from_anndata(..., csc="always")` / `scx convert --csc=always`, or added with `scx build-csc`) routes the GPU `pdex_ref` through a CSC-direct driver (`pdex_ref_gpu_chunked_v3_csc` in `scx-accel/src/diffexp/gpu.rs`) — v3 is the default GPU DE route (the `SCX_GPU_DE_V3` opt-in gate was removed when v3 became the default). The driver drops the dense intermediate entirely and replaces the per-chunk pseudobulk with a block-per-(gene, group) shared-memory tree-reduce (`csc_shard_pseudobulk_kernel`) that avoids `atomicAdd` contention. The shard source (`RawGpuCscShardSource`) ships full G3-shape pipelining (2-slot pinned ring + dedicated copy stream + scoped worker pre-decode + dual event handshake) and uses cheap catalog metadata to skip CSC shards whose `[col_start, col_end)` doesn't overlap the current gene chunk — so non-overlapping shards never get decoded or uploaded. Measured 2026-05-27 against backed-AnnData fixtures (`pyscx.open(path).to_anndata(backed=True)`):

| Dataset | v2-CSR GPU (former) | **v3-CSC (default)** | v3-CSC vs v2 |
|---|---:|---:|---:|
| pbmc3k | 1.17 s | **0.42 s** | **−64%** |
| pbmc10k | 16.36 s | **0.97 s** | **−94%** (17×) |
| smartseq2 | 107.28 s | **4.49 s** | **−96%** (24×) |
| tabula_sapiens_100k | 134.04 s | **8.91 s** | **−93%** (15×) |
| census_500k | 218.35 s | **11.90 s** | **−95%** (18×) |
| census_1m | 352.16 s | **16.02 s** | **−96%** (22×) |

n_runs=5 (3 for tabula/census). v1 default and v2-CSR baselines are unchanged because neither code path was modified. CPU `pdex_ref` numbers in the previous table (272 s at census_1m) also remain the reference point: at atlas scale v3-CSC is **17–22× faster than the rayon CPU implementation**, which is a real wall-time difference for Perturb-seq screens.

In-memory inputs (`pyscx.accel.pdex_ref(scipy_csr_adata, device="gpu")`) and files without a CSC sidecar automatically fall back to the v3-CSR-direct path — same algorithm, no CSC sidecar needed, slightly slower than v3-CSC because it loses the atomicAdd-avoidance win but still drops the dense intermediate. v3 was promoted to the unconditional default in Phase V1b after a route-marked soak confirmed v3 ≥ CPU at every tier and 13–28× faster than the former v1 GPU path at medium+large scale. Parity tests (`scx-accel/src/csc/pdex.rs::tests::test_pdex_ref_gpu_v3_csc_matches_cpu_streaming` and `_csr_fallback_*`) pin v3 to the CPU oracle to fp32 tolerance.

**Which route ran is recorded, and the gate asserts it.** Every `pdex_ref` call stamps its execution route on `adata.uns["scx_accel"]["pdex_ref"]` (`route ∈ {gpu_csc_v3, gpu_csr_v3, …}`, `fallback_reason`, `csc_available`), decided by the single planner `scx_accel::route::plan_de_route`. The `accel_de` benchmark reads this back into `runs[].extra` as `gpu_dispatch_route` (human-readable) and `de_route_csc_direct` (numeric: `0.0` only when a CSC fixture was built yet a non-`gpu_csc_v3` route ran — v3 being the unconditional default since Phase V1b). `thresholds.yaml` floors `de_route_csc_direct ≥ 1.0` for the GPU pdex_ref triple, so a *silent fallback to CSR while CSC-direct was intended* — exactly the prior benchmark misread — is a hard gate failure rather than an invisible footgun. The structured `adata.uns` metadata is the signal (the former ad-hoc stderr trace was removed).

**Headline finding from the 2026-05-25 routing fix:** GPU Leiden at census scale (`pyscx_gpu` cuGraph, 1.59-3.06s on census_500k/_1m) was completely missing from prior LATEST baselines because the gate's worker jobs were activating `scx-bench` (no cugraph), failing every Leiden GPU run silently. With per-job routing → `scx-bench-gpu`, the 200-500× speedup over `leidenalg_cpu` is now visible. Same correction for kNN — the prior bench's "cuVS missing → CPU HNSW fallback" was disguising real GPU CAGRA wall times under scanpy-CPU speeds.

### Choosing a Leiden backend

The two Leiden backends produce different partitions by design — they are not interchangeable. `device` is authoritative; there is no silent cross-backend fallback.

| Backend | `device` | Wall on census_1m | ARI vs leidenalg | Pick when |
|---|---|---:|---:|---|
| Rust-native (`scx_accel::leiden`) | `"cpu"` | ~56 s | ≈ 0.97 | Cluster IDs feed a downstream pipeline (marker-gene DE, annotation transfer, anything keyed on specific labels). Reproducibility against the CPU reference matters more than ~50 s on a 1M-cell graph. |
| cuGraph | `"gpu"` / `"gpu:N"` | ~3.5 s | **0.92** | Throughput-bound exploratory work — resolution sweeps, clustering under many random seeds, one-shot visualizations — where ARI 0.92 parity is acceptable. |

`device="auto"` (default) follows the rest of `pyscx.accel.*`: cuGraph if a CUDA device is visible and `cugraph` imports cleanly, else Rust-native. **Migration**: this differs from the pre-spec dispatcher, which always tried Rust-native first. Pin `device="cpu"` to preserve pre-spec cluster IDs. The cluster-assignment shift (ARI 0.97 → 0.92 vs leidenalg) is real for any user on a host with cuGraph installed.

cuGraph's Leiden uses a different refinement step and seed-handling scheme from leidenalg; the Rust-native implementation is a direct port of Traag et al. 2019 with the RB configuration model. The divergence is not an implementation bug — see `CLAUDE.md` § Known Limitations.

`device="gpu:N"` pins the cuGraph call to CUDA device `N` via `cupy.cuda.Device(N)`. Bare `"gpu"` is `"gpu:0"`. Out-of-range indices are rejected by `resolve_device`'s validation against `cudarc::GpuDevice::count()`. The Python `leidenalg` shim has been removed — callers who want it run `scanpy.tl.leiden(flavor="leidenalg")` directly.

### Preprocessing device dispatch

`pyscx.accel.{normalize_total, log1p, highly_variable_genes}` accept `device="cpu|gpu|auto"`. The GPU path is eager (materializes to scipy CSR). **`log1p(device="gpu")` on a materialised scipy/dense X warns and falls back to CPU** — the H→D + kernel + D→H round-trip dominates log1p's trivial math. The pre-fallback measurement (retained as motivation):

| Op | pbmc3k CPU / GPU | tabula_sapiens_100k CPU / GPU | census_1m CPU / GPU |
|---|---|---|---|
| normalize_total | 0.004s / 0.004s (1.0×) | 0.61s / 0.43s (**1.4×**) | 3.27s / 3.00s (**1.1×**) |
| log1p (pre-fallback) | 0.003s / 0.41s (**0.01×**) | 0.20s / 9.23s (**0.02×**) | 1.46s / 63.78s (**0.02×**) |
| fused normalize+log1p | 0.006s / 0.41s (0.01×) | 0.83s / 9.73s (0.09×) | 4.50s / 67.45s (0.07×) |
| highly_variable_genes (seurat_v3) | 0.06s / 0.07s (0.9×) | 3.42s / 3.40s (1.0×) | 25.77s / 28.31s (0.9×) |

Practical recommendation: **use the GPU preprocessing path only via the `normalize_total → log1p` fusion-marker chain on backed SCX data, and only when the downstream consumer is also GPU**. The fused-chain optimization is the only case where GPU preprocessing doesn't round-trip through the host. Standalone `log1p(device="gpu")` on materialised X emits a `UserWarning` and runs `sc.pp.log1p` instead; the GPU fast path is preserved when log1p sees the fusion marker planted by `normalize_total(device="gpu")`, or when X is still backed/lazy.

**Dispatch logic:** for an in-memory `X`, in-VRAM `pyscx.accel.pca(device="gpu")` routes to rapids-singlecell (`rsc.pp.pca`). The native GPU PCA path (backed/lazy/streaming inputs, or `SCX_FORCE_NATIVE_GPU=1`) is **always randomized** — the in-VRAM covariance core was removed, so `method="covariance"` / `"auto"` resolve to randomized on GPU (covariance is still honored on the CPU path). The randomized path accepts `qr_method="householder"` (default, always-stable) or `"cholesky"` (CholeskyQR2 — opt-in, surfaces `RuntimeError` on non-SPD Gram so callers can retry with Householder).

**Correctness.** On pbmc3k + census_1m, GPU PCA's 50 leading PCs match scanpy's reference to cosine ≥ 0.9999 sign-agnostic (`gpu_pca_validation.json`). kNN via rapids `rsc.pp.neighbors` matches scanpy-neighbors at recall = 1.0 on pbmc3k and ARI 0.91 against a downstream Leiden on tabula_sapiens_100k. UMAP via rapids `rsc.tl.umap` trustworthiness 0.9233 (vs CPU 0.9238) on pbmc3k.

Native GPU PCA (streaming/randomized path, used for backed/lazy/streaming inputs or `SCX_FORCE_NATIVE_GPU=1`) streams shards from disk → GPU kernels shard-by-shard without materializing the full matrix — enabling PCA on datasets larger than VRAM. In-VRAM PCA routes to rapids `rsc.pp.pca`.

### Differential expression

`pyscx.accel.pdex_ref(..., device=…)` and `pyscx.accel.rank_genes_groups(..., device=…)` both gained a `device="auto"|"cpu"|"gpu"[:N]"` selector. The GPU path uses per-gene CUB `BlockRadixSort` of the reference column once per gene chunk, batched warp-cooperative `searchsorted` to derive U₁ for every test group, merge-walk combined tie correction, and on-device `erfc` p-value matching the CPU formula bit-for-bit.

| Operation | Dataset | n_pool | CPU | GPU | Speedup | Notes |
|---|---|---:|---:|---:|---:|---|
| `pdex_ref` | pbmc3k (2.7K) | n_ref ≈ 540 | 0.62 s | 0.90 s | 0.69× | launch-overhead bound |
| `pdex_ref` | pbmc10k (12K) | n_ref ≈ 2.4K | 3.6 s | 3.4 s | 1.07× | ~tied |
| `pdex_ref` | smartseq2 (18K) | n_ref ≈ 3.5K | 21.4 s | 18.5 s | **1.16×** | searchsorted starts winning |
| `pdex_ref` | tabula_100k → census_1m | n_ref > 8192 | 54 → 317 s | **skip** | — | v1 capacity cap |
| Wilcoxon rank-sum (1-vs-rest) | pbmc3k (2.7K) | n_obs = 2.7K | 0.74 s | 0.92 s | 0.80× | launch-overhead bound |
| Wilcoxon rank-sum (1-vs-rest) | pbmc10k → census_1m | n_obs > 8192 | 3.4 → 210 s | **skip** | — | v1 capacity cap |

The headline speedup is modest because v1 caps the per-gene sort pool at `GPU_DE_BLOCK_SORT_CAPACITY = 8192` cells — the CUB `BlockRadixSort` is one block per gene, holding the whole row in registers + shared memory. Above that, the dispatch returns `AccelError::InvalidInput("…use device='cpu' or subsample the reference")` and the caller falls back to the rayon-parallel CPU path. Where the GPU does run (small + medium datasets, mid-size reference groups), launch overhead and chunked-upload latency dominate the on-device sort + searchsorted work. The spec-anticipated **10–50× win** lives at Perturb-seq scale (≥ 50K cells × hundreds of perturbation groups, `n_ref` typically a few thousand non-targeting controls) — none of the dataset-tier fixtures match that group structure with the synthetic 2-way `groupby` the benchmark falls back to. Lifting the 8192 cap via a tiled merge-sort upgrade is deferred to PR series G4.

**Default v3-CSC path (PR series G4.3, 2026-05-27; promoted to default in Phase V1b).** A backed AnnData over an SCX file with a CSC sidecar (built by default under `csc="auto"` on qualifying datasets, or explicitly via `pyscx.from_anndata(..., csc="always")` / `scx convert --csc=always`, or added with `scx build-csc`) routes `pdex_ref(device="gpu")` through a CSC-direct pseudobulk + scatter-to-gene-major kernel pair that drops the dense intermediate entirely. The CSC shard source is fully pipelined (2-slot pinned ring + dedicated copy stream + worker pre-decode) and pre-filters CSC shards by gene-chunk overlap using cheap catalog metadata, so non-overlapping shards are never decoded or uploaded. Bench (backed-AnnData, n_runs=3–5, median wall_s): **0.42 / 0.97 / 4.49 / 8.91 / 11.90 / 16.02 s** on pbmc3k / pbmc10k / smartseq2 / tabula_sapiens_100k / census_500k / census_1m respectively — **13–24× faster than v2-CSR GPU** and 17–22× faster than the rayon CPU implementation at atlas scale. In-memory inputs (no SCX file) automatically fall back to the v3-CSR-direct path. See the [`pdex_ref` GPU v3-CSC table](#per-operation-timing) above for the full numbers and the disposition. v3 is the unconditional default GPU DE route since Phase V1b.

**Correctness signal** (gated via `runs[].extra` in the `v0.4.3-g1-gpu-de` baseline):

| Metric | Threshold | Observed |
|---|---:|---:|
| `de_pval_agreement_vs_cpu` (mean Spearman ρ over shared (group × gene) p-values) | ≥ 0.999999 | 1.0 (pbmc3k, pbmc10k), 0.999999 (smartseq2) |
| `de_top_gene_overlap_vs_cpu` (median top-200 Jaccard per group) | ≥ 0.95 | 1.0 (pbmc3k, pbmc10k), 0.985 (smartseq2) |

Tolerance-based parity for p-values / FDR (not exact) because of `erfc` and sort-order numerics; U statistics agree exactly in f64. The CPU path itself is pinned bit-for-bit to upstream `pdex` via `pyscx/tests/test_pdex_ref_parity.py`, so CPU↔GPU parity here transitively pins the GPU path to the upstream oracle.

GPU Wilcoxon rank-sum (`rank_genes_groups(device="gpu")`) routes through `plan_de_route` — when a CSC sidecar is present, it takes the `gpu_csc_v3` CSC-direct path (same as `pdex_ref`); otherwise it falls back to `gpu_csr_v3`. `prefer_format="csc"` on the CPU path uses `CpuCsc`.

### GPU perturbation-evaluation metrics

`pyscx.accel.perturbation_metrics` and `pyscx.accel.energy_distance` gained a `device=` selector (pyscx ≥ 0.11.2). GPU `perturbation_metrics` runs the per-group pseudobulk means on the device (reusing the DE CSR pseudobulk kernels; route `gpu_csr`) with the five bulk metrics on the host; GPU `energy_distance` runs a gemm-based pairwise-distance mean (`‖x−y‖² = ‖x‖² + ‖y‖² − 2·xyᵀ`; route `gpu_dense`) for euclidean/cosine at f32 with f64 reductions. `discrimination_score` has no GPU kernel — exact-rank parity is not f32-safe, so it is deferred. CPU↔GPU parity: `perturbation_metrics` `atol ≈ 1e-6`, `energy_distance` `atol = 1e-4` (it is a Pearson correlation).

CPU-vs-GPU wall time on H100 (synthetic paired real/pred, 2K genes × 50 perturbations, 3-run median, pyscx 0.11.2; via `benchmarks/scripts/gpu_cpu_bench.py` in the cell-eval-scx fork):

| Metric | n_obs | CPU (s) | GPU (s) | Speedup | Route |
|---|---:|---:|---:|---:|---|
| `perturbation_metrics` | 10K | 0.50 | 1.31 | 0.38× | `gpu_csr` |
| `perturbation_metrics` | 100K | 4.89 | 4.90 | 1.00× | `gpu_csr` |
| `perturbation_metrics` | 1M | 40.35 | 44.26 | 0.91× | `gpu_csr` |
| `energy_distance` (euclidean) | 10K | 0.91 | 0.97 | 0.94× | `gpu_dense` |
| `energy_distance` (euclidean) | 100K | 7.44 | 4.35 | **1.71×** | `gpu_dense` |

GPU helps the compute-bound metric: `energy_distance` (an O(N²)-per-perturbation pairwise-distance gemm) reaches **1.71× at 100K** and widens with cell count — and it is what makes the metric feasible at atlas scale, where the CPU O(N²) reference is skipped (the bench caps `energy_distance` at ~200K cells; the CPU baseline is infeasible beyond — see the [Perturbation Metrics](perturbation-metrics.md#perturbation-metrics-cell-eval-parity) footnote ¹). `perturbation_metrics` is a cheap pseudobulk mean (O(nnz), memory / host-transfer-bound), so GPU ≈ CPU across sizes (small data even regresses on kernel-launch + host→device overhead) — its GPU kernel exists for uniform `device=` dispatch, not a speedup. All GPU runs took a `gpu_*` route (no silent CPU fallback); numeric parity is gated by `pyscx/tests/test_eval_metrics_gpu_parity.py` and the fork's `tests/test_scx_parity.py`. End-to-end, `cell-eval run --device gpu` matches `--device cpu` within the documented per-metric tolerances (fork `benchmarks/scripts/gpu_e2e_parity.py`).

### Canonical baseline

Two baselines live side-by-side under `benchmarks/comprehensive/results/baselines/`. **Accel** PRs (PCA / kNN / UMAP / Leiden / preprocess / HVG / DE) gate against `LATEST` (currently accel-only); **format / cloud / multimodal** PRs must pin the multi-surface baseline `v0.6.2-n_counts-augmentation` explicitly (not `LATEST`). The split exists because the multi-surface baseline captures `accel_*` rows but doesn't produce gate signal against them — see [benchmarks/README.md § Regression Gating](../../benchmarks/README.md#regression-gating).

| Use | Baseline | Date | Coverage |
|---|---|---|---|
| Format / cloud / multimodal | `v0.6.2-n_counts-augmentation` (pin explicitly — not `LATEST`) | 2026-05-11 | 806 rows × 8 datasets (`pbmc3k` → `census_1m`, `cite_seq_pbmc`, `multiome_pbmc`) |
| Accel (incl. `accel_de`) | `LATEST` → `v0.6.5-accel-gpu-to-gpu-anndata` | 2026-06-10 | rapids-routed accel rows; cross-tier rapids route + correctness gates (`*_route_rapids_correct`, `*_fallback_no_rapids_correct`); `to_gpu_anndata` promotion |

Per-run correctness metrics (`cosine_sim_min`/`mean`, `recall_vs_scanpy`, `trustworthiness`, `ari_vs_leidenalg`, `max_abs_diff_vs_scanpy`, `hvg_overlap_vs_scanpy`, plus `de_pval_agreement_vs_cpu` / `de_top_gene_overlap_vs_cpu` added in G1) flow through `runs[].extra` so the floor checks in `thresholds.yaml` evaluate real observed values, not `missing` placeholders.

```bash
# Accel PRs (PCA / kNN / UMAP / Leiden / preprocess / HVG / DE) — default LATEST:
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only

# Format / cloud / multimodal — pin the multi-surface baseline:
python benchmarks/comprehensive/scripts/gate_candidate.py --no-accel \
    --baseline benchmarks/comprehensive/results/baselines/v0.6.2-n_counts-augmentation
```

Older accel-only baselines (`v0.6.0-gpu-phase1-7`, `v0.6.0-gpu-phase1-7-multidataset`) remain in-tree for historical bisects but are no longer the gate targets. The earlier stop-gap wrappers (`benchmarks/scripts/gpu_regression_{diff,driver}.py` and `slurm_gpu_regression*.sh`) have been deleted; use `gate_candidate.py` for accelerator regression runs.

### Changes vs previous version

- **Covariance-PCA dispatch path** on GPU (threshold `n_vars ≤ 8000`) — *historical, removed.* The native in-VRAM covariance PCA core (`gpu_pca_covariance.rs`, `covariance_pca_gpu`, `GPU_COVARIANCE_PCA_THRESHOLD`) was deleted; in-VRAM PCA routes to rapids `rsc.pp.pca`. The numbers below are from the pre-removal baseline: on tabula_sapiens_100k (HVG-shaped input) GPU PCA ran 1.7× vs CPU, up from 0.9× in the earlier baseline. On census_1m at the same n_vars, the speedup remained 0.9×. Native streaming/randomized PCA survives for >VRAM workloads.
- **Randomized PCA's critical path** fully GPU-resident — the prior `Q → host → f64` SVD tail and per-iteration `d_m` download round-trip are gone (cuBLAS `sgemv` + `sgemm`). Correctness preserved (cosine ≥ 0.9999 on real data).
- **Opt-in CholeskyQR2** (`qr_method="cholesky"`) for the randomized path; benchmark-suite variants `gpu_randomized_pca_chol` vs `gpu_randomized_pca_householder` pending from the current cluster run.
- **Standalone GPU preprocessing ops** (`normalize_total`, `log1p`, `highly_variable_genes`) gain a `device` kwarg. In isolation they are slower than the CPU path (see table above — `log1p` is ~40× slower on tabula due to H2D/D2H round-trips); the `normalize_total → log1p` fusion marker is the only fast path.
- **cuGraph Leiden** exposes the `theta` knob via `pyscx.accel.leiden(theta=...)`.

## Go/No-Go Status

| Gate | Criterion | Result |
|------|-----------|--------|
| PCA correctness | cosine similarity > 0.99 | **Pass** |
| kNN recall | recall@15 > 0.95 | **Pass** |
| Graceful fallback | CPU fallback when no GPU | **Pass** |
| 10x pipeline speedup | end-to-end 10x vs CPU | **Fail** (3.8x achieved) |


## GPU DE device residency + gene-chunk windowing (Phase-4 task 4.5)

4.2 widened GPU staging's decode; it did not reduce how much decoding there was.
The GPU CSR DE route (`gpu_csr_v3` — the mandatory route for any file **without** a
CSC sidecar) has no column-range prefilter, so each of the two v3 CSR drivers runs a
full `for_each_gpu_csr_shard` pass **per gene chunk**. Cost is
`n_gene_chunks × n_shards` host decodes and H→D uploads. At census_500k — 61 497 genes
over a 500-gene chunk is 123 chunks, across 31 CSR shards — that is 123 complete passes
over a 747 M-nnz matrix. Pre-4.2, with staging decoding on one thread, that made GPU DE
there **98.6 % host-decode-bound** (1 005.6 s of a 1 019.7 s wall).

The arithmetic closes exactly, which is what made the diagnosis actionable rather than
plausible: 1 005.6 s ÷ 123 chunks = **8.2 s**, one full decode pass. (That prediction is
what the 4.5 capture below then hit, at 8 578 ms.)

**Decode was only half of it.** All four CSR row-scan kernels in `diffexp.cu` are
one-block-per-row and stride the row's *entire* nonzero range, testing
`col >= c0 && col < c1` per element — so the per-chunk *kernel* cost was O(nnz) too,
another 123× over. 4.2 widened the decode but left that untouched: bounding its arm from
wall (383.9 s as measured below) and Σ host-decode (≈ 941 s summed over depth-4 workers)
puts kernels + sync somewhere in **[84, 319] s**. Residency alone could not have been
shown to fix that, so both halves ship together.

Note the two baselines in play. Everything above quoting 1 019.7 s is **pre-4.2**
(`2055f74f`); the capture below is against **post-4.2** `main` (`2d1fe16b`), which is the
383.9 s arm. 4.2's 2.60× on this op is already banked in that baseline and is not counted
again here.

**1. Device residency.** `scx_gpu::ResidentGpuCsrSource` drains the inner
`GpuMatrixSource` once, retains every shard in its own device-resident `GpuCsrSlot`, and
serves each later chunk from VRAM. Decode and H→D collapse from `n_chunks × n_shards` to
`n_shards`.

Shards are **retained separately, not concatenated**. Two builders in `scx-gpu` already
concatenate (`decode_csr_shards_to_device` for the `to_gpu_anndata` handoff,
`gpu_pca_resident::try_build_resident_csr` for the PCA power loop) because their consumers
need one cuSPARSE descriptor spanning the matrix. DE does not — its kernels take a
per-shard view plus a `global_row` offset. Collapsing 31 shards into one 500 000-row shard
would change every kernel's grid shape and, for the f64 `atomicAdd` pseudobulk fold, the
accumulation interleaving. Keeping them separate means the callback sees byte-for-byte
what it saw while streaming: same shard indices, shapes, launch geometry, arguments. Total
VRAM is the same either way (~6 GB at census_500k, 8 B/nnz).

**2. Gene-chunk windowing.** Every one of those kernels already requires
strictly-increasing per-row column indices — `shard_validate::validate_shard` enforces it
release-active, because a duplicate column races the scatter. Sorted indices make a
chunk's columns a contiguous sub-range of the row, so two `lower_bound` searches replace
the linear scan and the per-chunk term becomes `O(nnz / n_chunks + log(row_len))`.

`[lower_bound(c0), lower_bound(c1))` selects exactly the elements the predicate selected.
For the three scatter kernels each output cell has a single writer, so this is
bit-identical. `csr_shard_pseudobulk_kernel` folds with f64 `atomicAdd`, whose ordering
across rows is **already** run-to-run nondeterministic; narrowing the loop changes the
interleaving but not the character, and the equivalence test compares means and fold
changes to tolerance while holding statistics and p-values exact.

**What residency costs, per tier** (8 B/nnz — one f32 value + one i32 index; all three fit
inside half an 80 GB H100 many times over, and all three run 123 gene chunks at the default
500-gene chunk over 61 497 genes):

| Dataset | nnz | CSR shards | resident VRAM | shard decodes before → after |
|---|--:|--:|--:|--:|
| tabula_sapiens_100k | 194.9 M | 7 | ~1.6 GB | 861 → 7 |
| census_500k | 747.0 M | 31 | ~6.0 GB | 3 813 → 31 |
| census_1m | 1 402.4 M | 62 | ~11.2 GB | 7 626 → 62 |

**Knobs.** `SCX_GPU_DE_RESIDENT_MAX_FRAC` (default `0.5`) caps residency at half the free
card, leaving the rest for the per-chunk gene slabs; `SCX_GPU_DE_RESIDENT=0` is the kill
switch. Residency is declined for a single gene chunk (streaming would run one pass
anyway) and when the matrix does not fit the budget — checked before every shard, aborting
the drain at the offending one rather than retaining more than it checked for. If the
per-chunk budget then cannot fit *alongside* the resident matrix, residency is released
and the clamp retried: an optimisation must never be the reason a call errors.

**A declined run is invisible in the output** — it produces the same numbers, slowly. So
the decision is stamped on `uns["scx_accel"][<op>]["resident_csr"]` and floored by the
`de_route_resident_csr` gate, the same reasoning behind `de_route_csc_direct`.
`shards_decoded` is *not* the signal: it counts slab passes, which residency does not
change.

**Also in 4.5.** The staging validator's O(nnz) scans go parallel above 65 536
nnz — they were amortised into irrelevance when the same shard was re-validated 123 times
beside 123 re-decodes, and are on the critical path once each shard is decoded once. The
parallel form reduces by *minimum row index* rather than first-hit, so the error names the
same offending position it always did rather than one that varies with load. And
`ShardSource::shard_size_hint()` (catalog-backed, no decode) finally lets GPU staging
pre-size its pinned/device slots — `RawGpuShardSource::with_max_shard_rows` had been dead
code — and gives `clamp_prefetch_depth` a real per-shard byte estimate, so
`SCX_GPU_STAGING_MEMORY_BUDGET` can bound the decoded-but-unconsumed set. Unset, nothing
derates and the depth is unchanged.

**Measured** (SLURM job 2709095, H100, **two builds** — `main` at `2d1fe16b` (i.e. post-4.2)
vs the branch, both at their own defaults, `benchmarks/scripts/profile_gpu_de_resident.py`,
3 runs, median wall, `SCX_DISABLE_CUDA_GRAPHS=1`):

| Op | Dataset | `main` (4.2) | branch | Speedup | Σ host-decode | pinned HTOD | VRAM peak |
|---|---|--:|--:|--:|--:|--:|--:|
| pdex_ref | tabula_sapiens_100k | 114 902 ms | 3 154 ms | **36.4×** | 246 868 → 2 105 ms | 17 313 → 203 ms | 1 775 → 2 984 MB |
| wilcoxon | tabula_sapiens_100k | 113 185 ms | 3 661 ms | **30.9×** | 247 348 → 2 120 ms | 17 351 → 200 ms | 2 127 → 3 334 MB |
| pdex_ref | census_500k | 383 890 ms | 12 536 ms | **30.6×** | 941 404 → 8 578 ms | 64 656 → 641 ms | 3 658 → 9 128 MB |
| wilcoxon | census_500k | 385 877 ms | 14 064 ms | **27.4×** | 945 281 → 8 146 ms | 64 477 → 631 ms | 5 162 → 10 632 MB |
| hvg *(control)* | 100k / 500k / 1m | 4 409 / 10 559 / 16 030 ms | 4 447 / 11 232 / 17 255 ms | 0.99 / 0.94 / 0.93× | — | — | 1 192 → 1 192 MB |

**The mechanism is confirmed three ways, not just by the wall.** At census_500k `pdex_ref`
the summed host-decode falls **110×** (predicted 123×: one pass instead of one per gene
chunk) and lands on **8 578 ms** against the 8 200 ms predicted *before the run* from
1 005.6 s ÷ 123. The pinned-staging bucket falls **101×**. And the VRAM delta between arms
is **5 470 MB** against a predicted 5 976 MB of resident CSR. Host peak RSS *falls* 14 %
(3 643 → 3 127 MB) — one decode pass churns far less host memory than 123.

**Residency alone would not have done this.** Its own predicted range was 1.2–4.4× (the
[84, 319] s kernel bound above). Post-change, kernels + sync are ~10 s of the 12.5 s wall,
so the windowing cut the per-chunk scan by roughly 8–32×. The capture measures the two
**together** and cannot attribute between them: there is a knob to disable residency but
none to disable the windowing, and adding one was not judged worth the API surface.

> [!NOTE]
> **The hvg control's 0.93–0.99× is noise, and that was measured rather than assumed.**
> Consistently-below-1.0 across three datasets looked like a real cost — plausibly the
> parallel shard validation, which runs on the consuming thread and so competes with the
> prefetch workers on a decode-bound op. Job 2709123 tested exactly that with one build and
> two arms (`SCX_GPU_VALIDATE_PAR_MIN_NNZ` pinned high takes the unchanged serial branch),
> 5 runs each. The result scattered in **both** directions — 1.037× / 1.001× / 0.944× on
> hvg, 0.977×–1.014× on DE — i.e. no effect. The job-to-job spread is the explanation: the
> *same* branch build measured census_500k hvg at 11 232, 12 822 and 12 839 ms across two
> jobs on the same node, a 14 % swing that swallows the 7 % being chased. Single-job control
> deltas below ~15 % on this node are not interpretable.

**Output equivalence is checked at the scale the change was built for**, not only on unit
fixtures (SLURM 2709125, census_500k, `SCX_GPU_DE_RESIDENT=0` vs default, same file and
groups, both arms confirmed on `gpu_csr_v3` at 123 gene chunks):

| Op | names / feature | p-values | statistics | pseudobulk means / log2FC |
|---|---|---|---|---|
| `rank_genes_groups` | exact | exact | exact | **exact** (streaming self-spread also 0) |
| `pdex_ref` | exact | exact | exact | ≤ 1.7 × 10⁻¹³ |

`pdex_ref`'s means are the only figures that are not bit-identical, and the bar they are
judged against is measured rather than chosen: the streaming path was run **twice**, and
its own run-to-run spread (1.0 × 10⁻¹³ — f64 `atomicAdd` ordering across rows is already
nondeterministic) is what residency has to come in under. It does, at the same order of
magnitude and ~5 decades inside the 2.9 × 10⁻⁸ relative floor. Wilcoxon's pseudobulk fold
happened to be reproducible on this run, and residency matched it exactly.

**Parallel shard validation (§9.13) is a measured no-op at these scales, and is kept as
hygiene with no `×` claimed** — the same disposition as task 4.3's marshalling. The reason
it does not show up is worth stating: validation runs on the consumer thread *while* the
prefetch workers decode ahead, so it is hidden behind decode entirely. What actually made
validation cheap was residency, which cut it from once-per-shard-per-chunk to once per
shard — 123× fewer invocations. Parallelising what remains is correct and free, not a win.
