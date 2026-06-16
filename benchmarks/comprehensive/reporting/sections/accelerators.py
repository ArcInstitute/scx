"""Chapter 9: Accelerators — source-tracked GPU and CPU timing.

- GPU accelerator timing sourced from raw JSON (``accel_*`` benchmark keys)
  when available, or declared as external-source when not yet in the
  comprehensive harness.
- ``CommentaryBlock`` with ``SourceRef`` for all numeric narrative.
- CSC/CSR dispatch included here (storage-layout concern for accelerators).
- Parity tables adjacent to timing tables.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CommentaryBlock,
)
from benchmarks.comprehensive.reporting.result_store import (
    ResultStore, SourceRef, SourceKind,
)
from benchmarks.comprehensive.reporting import tables
from benchmarks.comprehensive.reporting.report_model import TableBlock


def _build_gpu_table(store: ResultStore) -> TableBlock | TextBlock:
    """Build GPU accelerator performance table from raw JSON results.

    Looks for ``accel_*`` benchmarks with ``device=gpu`` in scenario
    metadata. If no GPU results are available, returns a TextBlock
    explaining the gap.
    """
    gpu_benchmarks = [
        "accel_pca", "accel_knn", "accel_umap", "accel_leiden",
        "accel_preprocess", "accel_hvg",
        # Differential expression — per-cell (Wilcoxon / pdex_ref) and
        # pseudobulk NB-GLM. accel_de doesn't stamp scenario.device, so GPU
        # rows are detected via the `_gpu`-suffixed format key below.
        "accel_de", "accel_de_nb_glm",
    ]
    rows_data: list[dict] = []
    for bench in gpu_benchmarks:
        for row in store.by_benchmark(bench):
            sc = row.scenario
            is_gpu = (sc.device and "gpu" in sc.device.lower()) or row.format.endswith("_gpu")
            if is_gpu:
                # Fold the variant tail (e.g. pyscx_pdex_ref) into the operation
                # label so multiple GPU variants of one bench stay distinct.
                tail = row.format[len(bench) + 2:] if row.format.startswith(bench + "__") else ""
                if tail.endswith("_gpu"):
                    tail = tail[:-4]
                op = bench.replace("accel_", "")
                if tail and tail not in ("pyscx", "pyscx_gpu"):
                    op = f"{op} ({tail})"
                rows_data.append({
                    "operation": op,
                    "dataset": row.dataset,
                    "gpu_wall_s": row.median_wall_s,
                    "source_path": row.source.path,
                })

    if not rows_data:
        return TextBlock(
            "*No GPU accelerator results in the comprehensive harness yet. "
            "GPU timing is available via standalone benchmarks — see the "
            "external/manual sources appendix for provenance.*"
        )

    headers = ["Operation", "Dataset", "GPU wall (s)"]
    rows: list[list[str]] = []
    for d in sorted(rows_data, key=lambda x: (x["operation"], x["dataset"])):
        wall_str = f"{d['gpu_wall_s']:.3f}" if d["gpu_wall_s"] is not None else "—"
        rows.append([d["operation"], d["dataset"], wall_str])

    return TableBlock(
        headers=headers, rows=rows,
        caption="GPU accelerator wall time (from comprehensive harness)",
        source=SourceRef(kind=SourceKind.raw_json,
                         reason="accel_* benchmarks with device=gpu"),
    )


def _median_peak_rss_mb(row) -> float | None:
    """Median ``peak_rss_mb`` across a row's runs (ResultRow has no accessor)."""
    import statistics

    vals = [
        r.get("peak_rss_mb")
        for r in row.runs
        if isinstance(r.get("peak_rss_mb"), (int, float)) and r.get("peak_rss_mb")
    ]
    return float(statistics.median(vals)) if vals else None


def _build_format_pipeline_table(store: ResultStore) -> TableBlock | TextBlock:
    """SCX vs h5ad feeding an identical rapids-singlecell GPU pipeline.

    One row per dataset comparing the three ``accel_format_pipeline`` variants.
    The same rapids engine runs in every cell — only the load-to-GPU path
    differs — so ``load_to_gpu_s`` and host peak RSS are the discriminating
    columns; end-to-end wall is shown for context.
    """
    h5ad_key = "accel_format_pipeline__h5ad_rapids_gpu"
    dev_key = "accel_format_pipeline__scx_devdecode_rapids_gpu"
    auto_key = "accel_format_pipeline__scx_auto_rapids_gpu"

    by_dataset: dict[str, dict[str, object]] = {}
    for row in store.by_benchmark("accel_format_pipeline"):
        if row.missing_reason is not None:
            continue
        by_dataset.setdefault(row.dataset, {})[row.format] = row

    if not by_dataset:
        return TextBlock(
            "*No `scx + rapids-singlecell` vs `h5ad + rapids-singlecell` results "
            "in the comprehensive harness yet. This GPU pipeline comparison "
            "populates on the next full GPU capture (`scx-bench-gpu`).*"
        )

    def _load_s(row) -> float | None:
        return row.metadata.get("load_to_gpu_s") if row is not None else None

    headers = [
        "Dataset",
        "h5ad load→GPU (s)",
        "SCX devdecode load→GPU (s)",
        "SCX auto load→GPU (s)",
        "Load speedup (h5ad ÷ devdecode)",
        "End-to-end wall h5ad / devdecode (s)",
        "Host peak RSS h5ad / devdecode (MB)",
        "devdecode transfer_mode",
    ]
    rows: list[list[str]] = []
    for ds in sorted(by_dataset):
        variants = by_dataset[ds]
        h5ad = variants.get(h5ad_key)
        dev = variants.get(dev_key)
        auto = variants.get(auto_key)

        h5ad_load = _load_s(h5ad)
        dev_load = _load_s(dev)
        auto_load = _load_s(auto)

        speedup = (
            f"{h5ad_load / dev_load:.2f}×"
            if h5ad_load and dev_load
            else "—"
        )

        def _f(v: float | None, fmt: str = "{:.3f}") -> str:
            return fmt.format(v) if isinstance(v, (int, float)) else "—"

        wall_pair = f"{_f(h5ad.median_wall_s if h5ad else None)} / {_f(dev.median_wall_s if dev else None)}"
        rss_pair = (
            f"{_f(_median_peak_rss_mb(h5ad) if h5ad else None, '{:.0f}')} / "
            f"{_f(_median_peak_rss_mb(dev) if dev else None, '{:.0f}')}"
        )
        transfer = (dev.metadata.get("transfer_mode") if dev else None) or "—"

        rows.append([
            ds,
            _f(h5ad_load),
            _f(dev_load),
            _f(auto_load),
            speedup,
            wall_pair,
            rss_pair,
            str(transfer),
        ])

    # Cite the first available source path for provenance.
    src_path = None
    for variants in by_dataset.values():
        for row in variants.values():
            src_path = row.source.path
            break
        if src_path:
            break

    return TableBlock(
        headers=headers,
        rows=rows,
        caption=(
            "Storage format → GPU pipeline: SCX vs h5ad, both feeding an "
            "identical rapids-singlecell pipeline (normalize → log1p → HVG → "
            "PCA → neighbors → UMAP). Load-to-GPU is the discriminating "
            "column; host RSS is process-level context."
        ),
        source=SourceRef(kind=SourceKind.raw_json, path=src_path),
    )


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Accelerators (PCA, kNN, UMAP, Leiden, DE)")

    # ── Parity overview ──────────────────────────────────────────────
    c.sections.append(Section(
        title="Accelerator Parity",
        blocks=[
            TextBlock(
                "Before interpreting speedups, confirm that SCX's Rust-native "
                "accelerators reproduce scanpy/leidenalg reference results.  "
                "The table below shows per-operation parity metrics extracted "
                "from the same benchmark runs that produce timing data.  "
                "Full correctness details are in Chapter 3."
            ),
            tables.accelerator_parity_table(),
        ],
    ))

    # ── CPU accelerator performance ──────────────────────────────────
    c.sections.append(Section(
        title="CPU Accelerator Performance",
        blocks=[
            TextBlock(
                "All benchmarks use SCX's Rust-native implementations vs "
                "scanpy's standard Python stack.  Timing results are backed "
                "by raw JSON from the comprehensive harness."
            ),
        ],
    ))

    # ── GPU accelerator performance ──────────────────────────────────
    gpu_table = _build_gpu_table(store)
    c.sections.append(Section(
        title="GPU Accelerator Performance",
        blocks=[
            TextBlock(
                "GPU-accelerated operations. In-VRAM PCA / kNN / UMAP / "
                "preprocess / HVG route to rapids-singlecell; native GPU kernels "
                "are retained for the >VRAM / streaming regimes, Leiden (cuGraph), "
                "DE, and Harmony. Performance is sourced from raw JSON where "
                "available; see the external sources appendix for any manually "
                "curated entries."
            ),
            gpu_table,
        ],
    ))

    # ── Surviving native GPU paths vs rapids-singlecell ──────────────
    c.sections.append(Section(
        title="Native GPU paths vs rapids-singlecell",
        blocks=[
            TextBlock(
                "After the ACC-RUST-OPT-V4 rapids transition, SCX routes in-VRAM "
                "**PCA / kNN / UMAP / preprocess / HVG** to rapids-singlecell — so "
                "for those ops SCX's GPU path *is* rapids and a head-to-head ratio "
                "is ~1.0 by construction. This table therefore compares only the "
                "native GPU kernels that survive because they cover a regime rapids "
                "does not: **PCA randomized/streaming (>VRAM)**, **HVG seurat_v3**, "
                "**Leiden (cuGraph)**, and **preprocess streaming**. The ratio "
                "(native / rapids wall time; >1 means rapids is faster) motivates "
                "the routing decision — it is surfaced, not gated, since rapids "
                "version drift must not fail the build. kNN and UMAP have no "
                "surviving standalone native GPU path and are intentionally "
                "omitted; removed Phase-3 variants (`pyscx_gpu_cov`, "
                "`pyscx_gpu_cagra`, native UMAP) and the `pyscx_gpu_no_rapids` "
                "diagnostic fallback are never selected."
            ),
            tables.accelerator_gpu_vs_rapids_comparison_table(),
        ],
    ))

    # ── Storage format → GPU pipeline (SCX vs h5ad + rapids-singlecell) ──
    c.sections.append(Section(
        title="Storage format → GPU pipeline (SCX vs h5ad + rapids-singlecell)",
        blocks=[
            TextBlock(
                "Holding the GPU engine constant (**rapids-singlecell**) and "
                "varying only the **storage format**, this measures the "
                "end-to-end workflow of loading a stored dataset onto the GPU "
                "and running an identical pipeline (normalize → log1p → HVG → "
                "PCA → neighbors → UMAP). The two formats differ only in the "
                "load-to-GPU path: SCX `to_gpu_anndata(device=\"gpu\")` decodes "
                "Scx1 count shards **directly in VRAM** (only the indptr is "
                "uploaded), whereas h5ad must host-decompress with `read_h5ad` "
                "then transfer the full matrix host→device via "
                "`rsc.get.anndata_to_GPU`. The pipeline portion is ~equal "
                "across formats by construction, so the SCX advantage shows up "
                "in **load-to-GPU time**. Host peak RSS is process-level and at "
                "these scales is dominated by the shared rapids pipeline working "
                "set that runs after the load step, so it does not isolate the "
                "load-path difference — it is reported for context, not as a "
                "discriminating column."
            ),
            _build_format_pipeline_table(store),
            TextBlock(
                "The `scx_devdecode` variant forces Scx1-coded shards "
                "(`scx convert --codec scx1` / `scx optimize --codec scx1`), so "
                "every shard carries a decode sidecar and the matrix decodes in "
                "VRAM — only the per-shard indptr crosses the bus. The realistic "
                "`scx_auto` variant uses the default auto-codec: at small scale "
                "its low-median counts also select Scx1 (device-decode), but at "
                "census scale higher-median shards route to Zstd, which carries "
                "no decode sidecar and host-streams (`transfer_mode = "
                "scx_device_handoff_streamed`), forfeiting the load advantage. "
                "The device-decode moat therefore requires Scx1 shards; the "
                "`scx_auto` column shows the default-codec behaviour an unaware "
                "user would get."
            ),
        ],
    ))

    # ── Differential expression (CPU + GPU) ─────────────────────────
    c.sections.append(Section(
        title="Differential Expression (CPU + GPU)",
        blocks=[
            TextBlock(
                "Per-cell DE (scanpy/pyscx Wilcoxon `rank_genes_groups`, "
                "`pdex_ref`) and pseudobulk negative-binomial GLM "
                "(`pdex_nb_glm`, DESeq2-style), CPU vs GPU. The first table is "
                "wall time per method × dataset × device; the second isolates "
                "the GPU NB-GLM speedup, kernel throughput, and CPU↔GPU "
                "numerical agreement; the third surfaces the deterministic "
                "route / correctness gate signals at a glance. The NB-GLM "
                "end-to-end speedup is **same-machine and node-core-count "
                "dependent** (the un-accelerated host aggregation + CPU baseline "
                "scale with cores), so it is surfaced here, not floored — only "
                "the route + concordance signals are hard-gated."
            ),
            tables.de_performance_table(),
            tables.de_nb_glm_gpu_table(),
            tables.de_route_correctness_table(),
        ],
    ))

    # ── CSC vs CSR dispatch (moved from operations chapter) ──────────
    c.sections.append(Section(
        title="CSC vs CSR Accelerator Dispatch",
        blocks=[
            TextBlock(
                "Time to compute column-oriented operations (column variance, "
                "HVG, DE, QC metrics) on a CSR-native file vs CSC sidecar.  "
                "This is an accelerator storage-layout concern — CSC sidecars "
                "can accelerate gene-major traversals."
            ),
            tables.bench_csc_dispatch_table(),
        ],
    ))

    return c
