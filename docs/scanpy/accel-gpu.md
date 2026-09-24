# GPU acceleration

> Part of the [SCX + scanpy guide](README.md). For installation, see [docs/gpu-setup.md](../gpu-setup.md);
for the `device=` selector, see the [accelerators overview](accelerators.md).

## GPU-supported vs GPU-fast

`device="gpu"` runs a column algorithm on the GPU, but **running on the GPU
is not the same as running fast on the GPU**. For `pdex_ref`, peak GPU
throughput requires a *column-major* substrate so the kernel reads contiguous
gene columns instead of decoding and projecting every row. Whether that
generalises to `rank_genes_groups` is **not** established — see the second
bullet; the only direct measurement points the other way.

- **`pdex_ref` is GPU-fast only with a CSC sidecar.** With a backed SCX file
  that has a CSC sidecar, the dispatch takes the
  CSC-direct route (`route == "gpu_csc_v3"`): it drops the per-chunk dense
  intermediate and skips non-overlapping CSC shards via a column-range
  pre-filter. Without a sidecar — e.g. an in-memory scipy CSR — the same call
  falls back to `gpu_csr_v3` (`fallback_reason == "no_csc_sidecar"`), which is
  *GPU-supported but not GPU-fast*. A **windowed or transformed** handle
  reaches the CSC-direct route too — while its row window still spans at least
  half the CSR shards — where it used to be downgraded to `gpu_csr_v3`
  unconditionally, because the only CSC source it could offer was the full-axis
  sidecar reader; it now passes its own view instead.
- **Wilcoxon rank-sum (`rank_genes_groups`) takes the same v3 *routes* as `pdex_ref` — but
  not, on the one shape measured, the same *timing*.** With a CSC sidecar it runs CSC-direct
  (`gpu_csc_v3`); without one it runs CSR-direct (`gpu_csr_v3`,
  `fallback_reason == "no_csc_sidecar"`).

  > **Do not build a sidecar for `rank_genes_groups` without measuring your own shape.** This
  > bullet used to conclude "so a CSC sidecar makes Wilcoxon rank-sum GPU-fast too". That was an
  > inference from `pdex_ref`'s numbers across a shared route, not a measurement of this op, and
  > the one direct measurement now available contradicts it. On a 960,195 × **6,143** log1p file
  > (nnz 2.65e9, 59 CSR shards), H100, `reference="rest"`, `n_genes=60`, `pts=True`:
  >
  > | route | sidecar | wall | peak RSS |
  > |---|---|---|---|
  > | `gpu_csr_v3` | no | **42.5 s** | **7.4 GiB** |
  > | `gpu_csc_v3` | yes | 85.3 s | 26.1 GiB |
  >
  > — **2.0× slower and 3.5× heavier with the sidecar**, before counting the 628.7 s / 28.1 GiB
  > build and the +348 % it added on disk (4.34 → 19.44 GB). Results were bit-identical, so this
  > is a cost question, not a correctness one. The likely mechanism is the gene axis: at 6,143
  > genes the CSC shards are ~160 M nnz each (~1.3 GB decoded), so the CSC route trades 767 cheap
  > CSR decodes for 29 very expensive ones and holds them. A ~36 k-gene file has a very different
  > shard geometry and may well behave as the old text claimed — **that case is unmeasured in
  > either direction.**
  >
  > None of this touches `pdex_ref`, whose CSC-direct advantage *is* measured
  > ([`pdex_ref` GPU v3-CSC table](../performance/gpu.md#per-operation-timing)) and is enforced by the
  > `de_route_csc_direct` floor in `benchmarks/comprehensive/thresholds.yaml`. Note when reading
  > that table that its 13–24× is against **`v2-CSR GPU (former)`**, a route no longer taken; the
  > margin against today's `gpu_csr_v3` is described qualitatively there and has not been
  > measured for either op.
  >
  > *Provenance:* one-off capture, 2026-09-13, single H100 80 GB, pyscx built at `14ef8ae3`
  > (an ancestor of `main`); both arms the same binary, so the comparison is within-build. This
  > is **not** a `benchmarks/comprehensive` capture and has no manifest entry — it is one shape,
  > measured once, reported because it contradicts an inference that had no measurement at all.
  > Treat it as a reason to measure your own file, not as a new performance claim.
- **PCA / kNN / UMAP / Leiden are not column algorithms** — they operate on
  row-major `X` or on PCA embeddings / kNN graphs, so CSC does not apply.
- **CSC-direct does not double VRAM usage.** The `gpu_csc_v3` route reads CSC
  shards *instead of* CSR shards — it does not load both representations
  simultaneously. VRAM usage for the CSC-direct path is comparable to the CSR
  path (proportional to NNZ), plus the per-chunk dense intermediate replaced
  by the shared-memory tree-reduce.

> **Which `(device, prefer_format)` selects `gpu_csc_v3`?** `device` and
> `prefer_format` are independent axes, and the GPU CSC-direct route is chosen
> by the *route planner*, **not** by `prefer_format="csc"`:
>
> - **GPU-fast DE:** pass `device="gpu"` (or `"auto"`) with `prefer_format`
>   left at its `"auto"` default (or set to `"csr"`) — both keep GPU on the
>   planner-driven path. When the backed file has a CSC sidecar **and** the
>   handle's row window still spans at least half the CSR shards, the planner
>   routes to `gpu_csc_v3` automatically; without a sidecar, or on a window
>   narrower than that, it uses `gpu_csr_v3` (see § `prefer_format` for why the
>   window matters, and note that a route declined by that policy still records
>   `csc_available=true`, with `fallback_reason=perf_policy`). This is the
>   intended GPU-fast entry point.
> - `prefer_format="csc"` selects the **CPU** column-major streaming path
>   (`cpu_csc`) — there is no GPU kernel behind that knob. With `device="auto"`
>   it runs on CPU; combining it with an explicit `device="gpu"` raises a
>   `RuntimeError` that points you back to `prefer_format="csr"`/`"auto"` +
>   `device="gpu"` for GPU CSC-direct.
>
> In short: do **not** reach for `prefer_format="csc"` to get GPU speed — it is
> the CPU path. A CSC *sidecar on the file* (built at conversion) is what makes
> the default-`auto` GPU call fast.

> **The CSC sidecar lives on disk — only a *backed* AnnData carries it.** The
> sidecar is reachable for GPU DE only through a backed dataset
> (`exp.to_anndata(backed=True)`, whose `X` is a `ScxBackedSparseDataset`). If you
> **materialize** with the plain `exp.to_anndata()`, `X` becomes an in-memory
> scipy CSR with no link back to the file, so GPU DE silently takes the slower
> `gpu_csr_v3` route (`fallback_reason == "no_csc_sidecar"`) **even though
> `exp.has_csc` is `True`**. Reach for `to_anndata(backed=True)` whenever you want
> the `gpu_csc_v3` fast route. As a guard, pyscx emits a one-time `UserWarning`
> when `device="gpu"` DE runs on an `X` that was materialized from a file with a
> CSC sidecar (it stamps `adata.uns["scx_source_has_csc_sidecar"] = True` at
> materialization to detect exactly this case).

To make a file GPU-fast for DE, build the sidecar at conversion time:
`pyscx.from_anndata(adata, path, csc="auto")` (built automatically once the
dataset clears the size thresholds) or `csc="always"`, `scx convert --csc=auto`,
or `scx build-csc` after the fact. Then **always confirm the route actually
taken** via `adata.uns["scx_accel"][op]["route"]` before drawing performance
conclusions — a silent CSR fallback measured as "GPU DE" is exactly the
benchmarking trap the [route metadata](../api/python-accel.md#accelerator-route-metadata) exists
to catch.

**Malformed input is rejected, not ranked.** Every GPU route validates each
shard on the host before it is staged, and raises rather than producing
numbers. Two invariants the kernels cannot enforce themselves:

- **Finite values.** The per-gene sort pads with `+INF` and sorts on the raw
  IEEE-754 bit pattern, so a NaN would land above `+INF` and corrupt the U
  statistic, the tie counts and every p-value in the gene — silently. Filter or
  QC NaN / Inf before DE; the CPU paths reject the same input.

  This check is scoped to the operations whose kernels need it, and the scoping
  is decided per entry point rather than per op:

  | GPU entry point | non-finite input |
  |---|---|
  | `rank_genes_groups`, `pdex_ref` | **rejected** at the staging boundary; the error names the op |
  | `highly_variable_genes` — the clipped-sum reducers, and the batched mean/variance | **rejected** at the staging boundary |
  | `highly_variable_genes` — plain and CSC mean/variance | rejected *after* accumulation, naming the offending gene column |
  | `pca`, `normalize_total` / `log1p`, `pseudobulk` | accepted; propagates as it would on the CPU |

  The split inside `highly_variable_genes` is not arbitrary. Plain mean/variance
  can afford to skip the O(nnz) input scan because a non-finite value survives
  into its column's sums, where `first_non_finite_column` still catches it. The
  clipped reducers cannot: the clip kernel evaluates `v > clip ? clip : v`, so
  `+Inf` compares true and is **replaced by the clip value** before it is ever
  accumulated — the sums come out finite and plausible, and no post-hoc check on
  the output can tell. The batched mean/variance is excluded for a different
  reason: its kernel returns early on a row belonging to no batch, so a
  non-finite value in an excluded row never reaches the sums either.

  Note this differs from the CPU HVG path, which rejects **every** non-finite
  input up front via `ensure_finite_values`, including for the plain
  mean/variance reducers where the GPU defers to the post-accumulation check.
- **One value per `(cell, gene)`.** The scatter runs one thread per nonzero, so
  a duplicated entry would put two threads on one output cell with a
  nondeterministic winner. On the CSC side this is checked as *strictly
  increasing* row indices per column, which is what `scx build-csc` and every
  other sidecar writer emits; a hand-built sidecar with distinct-but-unordered
  rows is refused conservatively rather than raced.

  This check is likewise scoped, and **independently** of the finiteness check
  above — the two are separate switches, not a strictness ladder, precisely
  because they do not nest:

  | GPU entry point | sorted indices required |
  |---|---|
  | `rank_genes_groups`, `pdex_ref` | yes — the scatter races a duplicate `(cell, gene)` |
  | `pca` | yes — cuSPARSE SpMM is undefined on unsorted column indices |
  | `pseudobulk` | yes — the kernel binary-searches each row's column window |
  | `highly_variable_genes` (all entry points) | **no** — it accumulates atomically or clips in place |
  | `normalize_total` / `log1p` | **no** — it rewrites values in place |

  So `highly_variable_genes` accepts an unsorted `scipy.sparse.csr_matrix`
  (`has_sorted_indices == False`) exactly as the CPU path does, while still
  rejecting a non-finite value on the entry points listed in the previous
  table. An earlier design made these a cumulative ladder, which meant asking
  for the finiteness check silently also demanded sorted indices — and GPU HVG
  then refused input its own kernels handle fine.

Both surface as a `RuntimeError` naming the offending gene column or nonzero
index. The CSC-direct route previously ran neither check, so a NaN in a file
with a CSC sidecar returned a complete `rank_genes_groups` / `pdex_ref` result
on `device="gpu"` — finite, plausible scores and p-values, no error — while the
same file on CPU, or on GPU without a sidecar, was rejected.

**Benchmark route gates.** The benchmark suite enforces correct GPU dispatch
via absolute-floor gates in `thresholds.yaml`. Every `accel_*.py` GPU variant
emits an `<op>_route_gpu_correct` signal (1.0 when a GPU route ran, 0.0 on a
silent CPU fallback); `bench_csc_dispatch.py` emits `csc_dispatch_correct` for
CSC-labelled variants; `accel_de.py` emits `de_route_csc_direct` for the
pdex_ref CSC-direct path; and `accel_eval_metrics.py` emits
`perturbation_metrics_route_gpu_correct` / `energy_distance_route_gpu_correct`
for the perturbation-evaluation metric GPU variants. See
[benchmarks/README.md § Regression Gating](../../benchmarks/README.md#regression-gating)
for the full gate table.

## Data layout for fast GPU decode (`to_gpu_anndata` / device-resident analysis)

The device-handoff path — `pyscx.open(...).to_gpu_anndata()`, then chained
`pyscx.accel.*` / `rsc.*` ops on the device-resident AnnData
(`transfer_mode` ∈ `scx_device_decode_gpu` / `scx_device_handoff_streamed` /
`scx_device_handoff`) — is only as fast as the cost of
getting each shard onto the GPU. That cost is dominated by **decode**, and
decode cost is set by the **value representation you persisted on disk**, not by
the analysis op. So the layout choice matters as much as the device flag:

- **Prefer storing raw integer counts (`X` → Scx1) and deriving log-norm
  on-device.** Raw scRNA-seq counts auto-route to the Scx1 codec
  (Delta-Golomb / FOR-BP / Rice) — the codec with GPU decode kernels, so it is
  the codec the device-side decode path targets; framed Scx1 shards decode
  group-by-group in VRAM (random access via the row-group block index). Open the
  counts and run `normalize_total` / `log1p` in VRAM (the
  `ScxLazyTransformedDataset` chain, or `rsc.pp.*` on the device AnnData) so the
  log-normalized matrix is produced on the GPU and **never round-trips through a
  host float buffer**. This is the recommended flow.
- **A *persisted* log-normalized `X` is float → Pcodec, which decodes on the
  host.** Public h5ad / CELLxGENE files often ship `X` already log-normalized.
  That matrix has **no GPU decoder** (Pcodec is CPU-only),
  so the handoff pays a host pcodec-decompress + HtoD per shard — *GPU-supported,
  but not GPU-fast on the decode side*. If the file also carries a `counts`
  layer, prefer opening that and deriving log-norm on-device (bullet above).
- **If you must persist log-norm for the GPU and want decode-free upload, store
  it uncompressed (`None` codec → raw `f32` / `f16`).** A raw float array needs
  no decode — the handoff is a straight HtoD `memcpy` — at the cost of
  compression ratio (`f16` halves the bytes if its precision is acceptable for
  log-norm). This is the only float option that is GPU-fast to upload today; a
  GPU-decodable *compressed* float codec does not exist.
- **Device decode is Scx1-only.** Framed Scx1 shards decode group-by-group
  directly in VRAM; Zstd / Pcodec / LZ4 shards and float layers have no GPU
  decoder, so the device path falls back to host decode + HtoD for them. "Make
  `X` GPU-fast to decode" therefore means "store the GPU-relevant matrix as Scx1
  counts," **not** "re-codec a float layer."
- **Upgrade an older file in place with `scx optimize`.** A pre-v4 file (Scx1
  counts but not row-group-framed to v4) does not need a full reconvert to become
  device-decode-fast — run `scx optimize in.scx out.scx` (or
  `pyscx.optimize("in.scx", "out.scx")`; pass `codec="scx1"` to force Scx1 on
  every integer shard). It re-encodes + canonicalizes every CSR shard and
  row-group-frames it (`format_version=4`), preserving rows / obs / var / obsm /
  uns / indexes (see [operations.md § Optimize](../operations.md#optimize)). Only Scx1
  integer shards decode in VRAM — a persisted float (Pcodec) `X` still won't
  (store counts per the first bullet).

Confirm the path actually taken via
`adata.uns["scx_accel"][op]["transfer_mode"]`, the same way you confirm `route`
for DE. `to_gpu_anndata` stamps one of:
- `scx_device_decode_gpu` — framed Scx1 shards decoded **fully in VRAM**
  group-by-group; only the tiny indptr is uploaded (`bytes_uploaded` ≈ indptr).
  The fast path you want. As of the BitPacker4x GPU kernel, this covers **every**
  Scx1 row, including dense (≥128-nnz) cells — those no longer host-fall-back.
- `scx_device_handoff_streamed` — on-device, but some shard still bounced through
  the host because it is **not** an Scx1 shard: a non-Scx1 codec (the float
  Pcodec case above) host-bounces. `bytes_uploaded` is the real HtoD total.
- `scx_device_handoff` — host-assembled CSR (filtered / projected / multimodal
  input), or an `X` that was already device-resident on entry.

(rapids ops that have to upload a host `X` instead stamp `anndata_to_gpu` — a host
re-upload, not a `to_gpu_anndata` mode.)

**rapids-singlecell knows nothing about the SCX device decode path.** The
in-VRAM group-by-group decode lives entirely on the SCX side of the handoff: it
accelerates SCX's own decode→device step (`to_gpu_anndata`), which *produces* the
`cupyx.scipy.sparse.csr_matrix` that rapids then operates on. rapids only ever
sees that already-decoded, device-resident matrix (it validates inputs via its
own `_check_gpu_X`) and has no knowledge of the SCX format or codecs. So choosing
an Scx1-counts layout speeds up the SCX→device handoff that *feeds* rapids — it is
not something rapids consumes, and it changes no rapids call.

## GPU helper functions

```python
# Query GPU availability and memory
info = pyscx.accel.gpu_info()
# {'device': 'NVIDIA A100-SXM4-80GB', 'total_vram_gb': 80.0, 'free_vram_gb': 72.3}

# Estimate GPU memory requirements
est = pyscx.accel.estimate_gpu_memory(adata, operation="pca", n_comps=50)
# {'required_gb': 2.1, 'fits_in_vram': True}
```

## GPU vs CPU numerical differences

GPU and CPU accelerators may produce slightly different results due to:

| Factor | Impact | When it matters |
|--------|--------|-----------------|
| **PCA precision** | GPU uses f32 throughout; CPU uses f64 | Native GPU vs CPU: per-PC cosine ≥ 0.99 on the leading PCs — no biological impact |
| **kNN algorithm** | GPU uses CAGRA (graph-based ANN); CPU uses HNSW | Both are approximate; exact agreement is not expected. **No GPU-vs-CPU recall bar is enforced** — see the table below |
| **UMAP non-determinism** | GPU uses `atomicAdd` (race conditions are intentional) | Embedding coordinates differ; cluster structure preserved |
| **Leiden** | Rust-native vs cuGraph vs leidenalg may produce different partitions | Label stability differs **by design**; the enforced GPU-vs-CPU bar is a degeneracy floor, not agreement — see the table below |
| **DE pseudobulk non-determinism** | GPU folds per-(gene, group) sums with a cross-block f64 `atomicAdd` whose summation order varies run to run. The CSC route has a deterministic shared-memory tree-reduce and takes it only while `n_groups × block_dim × 8 B` fits the device's opt-in shared memory (908 groups on sm_90, 652 on sm_80, at the smallest block dimension); the CSR route has no deterministic variant | Means and fold changes differ in the last bits between runs — and **not only there**: `pdex_ref`'s `cpm_filter` is a strict `>` and its FDR is recomputed over the survivors, so a gene within an ULP of the threshold flips in and out and moves `pvals_adj` for **every** surviving gene. Which kernel ran is recorded as `reduction` on `uns["scx_accel"][<op>]`; `SCX_GPU_DE_REQUIRE_DETERMINISTIC=1` makes the fall-through to atomics an error. **Nothing enforces run-to-run agreement today** |
| **HVG non-determinism** | GPU column moments on a **CSR** input accumulate `Σx` / `Σx²` through cross-block f64 `atomicAdd`; the CSC route (`gpu_csc_v3`, needs a sidecar) is one block per column with a tree-reduce and plain stores | Can flip a near-constant gene's membership at the variance cutoff. Reported as `reduction` on `uns["scx_accel"]["highly_variable_genes"]`, and already separable from `route`. **Nothing enforces run-to-run agreement today** |
| **`energy_distance` non-determinism** | `pairwise_dist_sum_kernel` folds with a global f64 `atomicAdd` across up to 65535 blocks, and the gemm path additionally chunks against **free VRAM at call time** — so the summation order can differ between two runs on the same card depending on what else is resident | The score differs in the last bits between runs. No deterministic variant exists, so `reduction` always reads `atomic`. **Nothing enforces run-to-run agreement today** |

## Tolerance thresholds (correctness tests)

Each row below names the test that enforces it. Where nothing enforces a row,
it says so rather than implying a gate that does not exist.

| Test | Metric | Threshold | Enforced by |
|------|--------|-----------|-------------|
| **Native** GPU PCA vs CPU PCA | Per-PC cosine, leading `n_clusters - 1` PCs | ≥ 0.99 | `test_gpu_randomized_householder_vs_cpu_randomized` — pins `SCX_FORCE_NATIVE_GPU=1`, so it does **not** cover the `device="gpu"` default, which is rapids-singlecell |
| **Native** GPU PCA, Cholesky QR vs Householder QR | Per-PC cosine | ≥ 0.999 | `test_gpu_cholesky_matches_householder` — GPU vs GPU, not a CPU comparison, and it pins `SCX_FORCE_NATIVE_GPU=1` too: rapids ignores `qr_method`, so without the pin both arms are the same path and the comparison is vacuous |
| `normalize_total`→`log1p`→`pca` at `device="gpu"` on an SCX-round-tripped **in-memory** AnnData, vs scanpy | Per-PC cosine, top 10 PCs | ≥ 0.99 — but see below: this is **not** a GPU-route gate | `test_normalize_log1p_pca_matches_scanpy` |
| GPU normalize + log1p vs CPU | Element-wise relative | < 1e-5 | `scx-gpu/src/gpu_preprocess_tests.rs::test_gpu_normalize_log1p_matches_cpu` |
| GPU Leiden vs CPU Leiden | ARI | ≥ 0.10, plus a cluster-count bound | `test_gpu_leiden_not_degenerate` |
| GPU kNN vs CPU HNSW | Recall@k | **not enforced** | — |
| GPU UMAP | Trustworthiness | **not enforced** | — |

Four notes on why these are what they are, since several look alarming out of
context:

- **Leiden's ARI floor is 0.10 on purpose, not by neglect.** GPU label stability
  differs from `leidenalg` by design (see the note above and README's Known
  Limitations), so a high ARI would be asserting a property SCX explicitly does
  not promise. The test pairs the loose ARI — a degeneracy canary, against an
  observed 0.0002 for a collapsed partition — with a cluster-count bound, which
  is the assertion that actually has teeth. Pin `device="cpu"` when you need
  label stability.
- **kNN recall and UMAP trustworthiness have CPU-path tests, not GPU ones.**
  `pyscx/tests/test_accel.py` asserts HNSW recall ≥ 0.90 against brute force and
  UMAP trustworthiness > 0.75, both on `device="cpu"`. Neither compares a GPU
  result to anything, so neither backs a GPU tolerance.
- **Two of these rows pin `SCX_FORCE_NATIVE_GPU=1`, and that is not a detail.**
  The `device="gpu"` default routes in-memory PCA to rapids-singlecell, so both
  native-path bars are invisible to it.
- **The third PCA row asks for the default path but does not gate it.** It calls
  `normalize_total` / `log1p` / `pca` at `device="gpu"` on a materialized AnnData
  — which is the rapids route — and its numeric bar is real. What it never does
  is assert *which route served the request*: none of the three ops has its
  `uns["scx_accel"][op]["route"]` checked. On a CUDA host without
  rapids-singlecell installed, every one of them takes the documented
  `NoRapidsCpu` fallback (`pyscx/src/accel/pca.rs`) and the whole SCX side runs
  on the **CPU** while the test still passes. Read it as a correctness check of
  the op chain, not as evidence that a GPU kernel ran.
  ⚠️ An earlier revision of this note said "no row here gates the default path",
  which was wrong in the other direction — this row does target it. Both
  statements were mis-citations of the kind this table exists to remove.
- **The PCA rows are split three ways on purpose.** An earlier revision of this
  table had one "GPU PCA vs CPU PCA" row claiming "≥ 0.999 (GPU arms), ≥ 0.99 (vs
  scanpy)". The 0.999 bar is `test_gpu_cholesky_matches_householder`, which
  compares two **GPU** QR methods to each other — attributing it to a GPU-vs-CPU
  row overstated what is gated, which is the same mis-citation this table exists
  to remove. The real GPU-vs-CPU bar is ≥ 0.99 on the leading `n_clusters - 1`
  PCs, and it pins `SCX_FORCE_NATIVE_GPU=1`, so it says nothing about the
  `device="gpu"` default path (rapids-singlecell) that most users actually take.
- Earlier revisions of this table cited `pyscx/tests/test_accel_gpu.py` as the
  enforcement mechanism for all five rows. **That file has never existed**, and
  four of the five thresholds it was said to enforce did not match any
  assertion in the tree.

## Checking which backend was used

All accelerators record the backend in `adata.uns`:

```python
pyscx.accel.pca(adata, device="gpu")
print(adata.uns["pca"]["backend"])          # "rapids_singlecell_gpu"
print(adata.uns["pca"]["device"])           # "NVIDIA A100-SXM4-80GB"
print(adata.uns["pca"]["gpu_time_ms"])      # 1234.5

pyscx.accel.neighbors(adata, device="gpu")
print(adata.uns["neighbors"]["backend"])    # "rapids_singlecell_gpu"

pyscx.accel.umap(adata, device="gpu")
print(adata.uns["umap"]["backend"])         # "rapids_singlecell_gpu"
```

On GPU, PCA, kNN, and UMAP all route to `rapids_singlecell` (`rsc.pp.pca`,
`rsc.pp.neighbors`, `rsc.tl.umap`). Preprocessing ops (`normalize_total`,
`log1p`, `highly_variable_genes`) also route to `rsc.pp.*` on GPU. GPU
Leiden uses cuGraph: `device="cpu"` runs the Rust-native CPU path,
`device="gpu"` runs cuGraph and hard-errors if cuGraph is absent (the
Python `leidenalg` fallback was deleted — call
`scanpy.tl.leiden(flavor="leidenalg")` directly if you need it). Set
`SCX_FORCE_NATIVE_GPU=1` to pin
surviving native GPU paths (HVG `seurat_v3`, DE, Harmony, Leiden); set
`SCX_DISABLE_RAPIDS=1` to force the rapids-absent fallback for testing.
When rapids is unavailable, a one-shot `UserWarning` is emitted and the
fallback reason `no_rapids` is recorded in the route metadata.

## See also

- [Column-major (CSC) dispatch](accel-csc.md) — the CPU CSC route.
- [Multithreading § GPU DE device residency](threading.md#gpu-de-device-residency).
- [Common scanpy workflows § GPU-accelerated analysis pipeline](workflows.md#gpu-accelerated-analysis-pipeline).
