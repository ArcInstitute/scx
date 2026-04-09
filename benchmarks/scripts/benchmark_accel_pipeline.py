#!/usr/bin/env python3
"""
SCX Full Pipeline Benchmark — SCX-native vs Pure-Scanpy

End-to-end pipeline comparison (§3.10.6):
  1. SCX out-of-core (Phase 4d): lazy normalize+log1p -> streaming PCA -> backed
  2. SCX preprocess (Phase 4a): pyscx.preprocess() -> accel functions
  3. Scanpy in-memory: full materialization -> scanpy functions

Pipeline stages: Normalize -> log1p -> HVG -> PCA -> neighbors -> UMAP -> Leiden -> DE

Usage:
    python benchmarks/scripts/benchmark_accel_pipeline.py --mode all
    python benchmarks/scripts/benchmark_accel_pipeline.py --mode scanpy --datasets census_1m
    python benchmarks/scripts/benchmark_accel_pipeline.py --mode report
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = Path(os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx"))

DATASETS = {
    "tabula_sapiens_100k": {"h5ad": DATA_DIR / "tabula_sapiens_100k.h5ad", "cells": 100_000},
    "census_1m": {"h5ad": DATA_DIR / "census_1m.h5ad", "cells": 1_000_000},
}

N_RUNS = 3
N_COMPS = 50
N_NEIGHBORS = 15
N_HVGS = 2000


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def get_rss_mb():
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
            return int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def ensure_scx_file(dataset_name):
    import pyscx
    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    scx_path = h5ad_path.with_suffix(".scx")
    if scx_path.exists():
        return scx_path
    if not h5ad_path.exists():
        return None
    print(f"  Converting {dataset_name} h5ad -> SCX...")
    import anndata
    adata = anndata.read_h5ad(str(h5ad_path))
    pyscx.from_anndata(adata, str(scx_path))
    del adata
    gc.collect()
    return scx_path


def save_json(data, filename):
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  JSON saved: {path}")


def get_de_gene_names(adata, n_top=100):
    rgg = adata.uns["rank_genes_groups"]
    groups = list(rgg["names"].dtype.names)
    result = {}
    for g in groups:
        result[g] = list(rgg["names"][g][:n_top])
    return result


def compute_de_gene_overlap(genes_a, genes_b, n_top=100):
    groups = set(genes_a.keys()) & set(genes_b.keys())
    if not groups:
        return 0.0
    overlaps = []
    for g in groups:
        set_a = set(genes_a[g][:n_top])
        set_b = set(genes_b[g][:n_top])
        overlaps.append(len(set_a & set_b) / n_top)
    return float(np.mean(overlaps))


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Variant 1: SCX Out-of-Core (Phase 4d)
# ──────────────────────────────────────────────────────────────────────────────

def run_scx_ooc_pipeline(dataset_name, n_runs=N_RUNS):
    """Lazy normalize+log1p -> streaming PCA -> backed throughout."""
    import pyscx
    import scanpy as sc

    print(f"\n  --- SCX Out-of-Core Pipeline ({dataset_name}) ---")

    scx_path = ensure_scx_file(dataset_name)
    if scx_path is None:
        print(f"    SKIP: {dataset_name} not available")
        return None

    all_runs = []
    for run in range(n_runs):
        gc.collect()
        rss_baseline = get_rss_mb()
        timings = {}

        print(f"    Run {run+1}/{n_runs}...")

        # Open backed
        adata = pyscx.open(str(scx_path)).to_anndata()

        # Lazy normalize + log1p (no materialization)
        t0 = time.perf_counter()
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.log1p(adata)
        timings["normalize_log1p"] = time.perf_counter() - t0

        # HVG (backed, streaming var)
        t0 = time.perf_counter()
        n_top = min(N_HVGS, adata.n_vars)
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top, flavor="seurat_v3",
                subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0
            )
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
        timings["hvg"] = time.perf_counter() - t0

        # Streaming PCA (through lazy transforms)
        t0 = time.perf_counter()
        pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
        timings["pca"] = time.perf_counter() - t0

        # kNN
        t0 = time.perf_counter()
        pyscx.accel.neighbors(adata, n_neighbors=N_NEIGHBORS, device="cpu")
        timings["knn"] = time.perf_counter() - t0

        # UMAP
        t0 = time.perf_counter()
        pyscx.accel.umap(adata, device="cpu")
        timings["umap"] = time.perf_counter() - t0

        # Leiden
        t0 = time.perf_counter()
        sc.tl.leiden(adata, resolution=1.0)
        timings["leiden"] = time.perf_counter() - t0

        # DE
        t0 = time.perf_counter()
        pyscx.accel.rank_genes_groups(adata, groupby="leiden", reference="rest")
        timings["de"] = time.perf_counter() - t0

        timings["total"] = sum(timings.values())
        rss_peak = get_rss_mb()
        timings["peak_rss_mb"] = round(rss_peak, 0)
        timings["rss_delta_mb"] = round(rss_peak - rss_baseline, 0)

        all_runs.append(timings)
        print(f"      Total: {timings['total']:.2f}s, "
              f"Peak RSS: {rss_peak:.0f} MB")

        # Save leiden + DE for quality comparison on last run
        if run == n_runs - 1:
            all_runs[-1]["_leiden"] = adata.obs["leiden"].values.copy()
            all_runs[-1]["_de_genes"] = get_de_gene_names(adata, 100)

        del adata
        gc.collect()

    return all_runs


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Variant 2: SCX Preprocess (Phase 4a)
# ──────────────────────────────────────────────────────────────────────────────

def run_scx_preprocess_pipeline(dataset_name, n_runs=N_RUNS):
    """pyscx.preprocess() writes new SCX -> load -> accel functions."""
    import pyscx
    import scanpy as sc

    print(f"\n  --- SCX Preprocess Pipeline ({dataset_name}) ---")

    scx_path = ensure_scx_file(dataset_name)
    if scx_path is None:
        print(f"    SKIP: {dataset_name} not available")
        return None

    all_runs = []
    for run in range(n_runs):
        gc.collect()
        rss_baseline = get_rss_mb()
        timings = {}

        print(f"    Run {run+1}/{n_runs}...")

        # Preprocess: write new SCX file
        with tempfile.NamedTemporaryFile(suffix=".scx", delete=False,
                                          dir=str(DATA_DIR)) as tmp:
            prep_path = tmp.name

        try:
            t0 = time.perf_counter()
            pyscx.preprocess(str(scx_path), prep_path,
                              operations=["normalize_total", "log1p"],
                              target_sum=1e4)
            timings["preprocess_write"] = time.perf_counter() - t0

            # Open preprocessed
            adata = pyscx.open(prep_path).to_anndata()

            # HVG
            t0 = time.perf_counter()
            n_top = min(N_HVGS, adata.n_vars)
            try:
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=n_top, flavor="seurat_v3",
                    subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0
                )
            except Exception:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
            timings["hvg"] = time.perf_counter() - t0

            # PCA
            t0 = time.perf_counter()
            pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
            timings["pca"] = time.perf_counter() - t0

            # kNN
            t0 = time.perf_counter()
            pyscx.accel.neighbors(adata, n_neighbors=N_NEIGHBORS, device="cpu")
            timings["knn"] = time.perf_counter() - t0

            # UMAP
            t0 = time.perf_counter()
            pyscx.accel.umap(adata, device="cpu")
            timings["umap"] = time.perf_counter() - t0

            # Leiden
            t0 = time.perf_counter()
            sc.tl.leiden(adata, resolution=1.0)
            timings["leiden"] = time.perf_counter() - t0

            # DE
            t0 = time.perf_counter()
            pyscx.accel.rank_genes_groups(adata, groupby="leiden", reference="rest")
            timings["de"] = time.perf_counter() - t0

            timings["total"] = sum(timings.values())
            rss_peak = get_rss_mb()
            timings["peak_rss_mb"] = round(rss_peak, 0)

            all_runs.append(timings)
            print(f"      Total: {timings['total']:.2f}s, "
                  f"Peak RSS: {rss_peak:.0f} MB")

            if run == n_runs - 1:
                all_runs[-1]["_leiden"] = adata.obs["leiden"].values.copy()
                all_runs[-1]["_de_genes"] = get_de_gene_names(adata, 100)

            del adata
        finally:
            try:
                os.unlink(prep_path)
            except OSError:
                pass
        gc.collect()

    return all_runs


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Variant 3: Scanpy In-Memory
# ──────────────────────────────────────────────────────────────────────────────

def run_scanpy_pipeline(dataset_name, n_runs=N_RUNS):
    """Full materialization -> scanpy functions on scipy CSR."""
    import anndata
    import scanpy as sc

    print(f"\n  --- Scanpy In-Memory Pipeline ({dataset_name}) ---")

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        print(f"    SKIP: {h5ad_path} not found")
        return None

    all_runs = []
    for run in range(n_runs):
        gc.collect()
        rss_baseline = get_rss_mb()
        timings = {}

        print(f"    Run {run+1}/{n_runs}...")

        adata = anndata.read_h5ad(str(h5ad_path))

        # Normalize + log1p
        t0 = time.perf_counter()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        timings["normalize_log1p"] = time.perf_counter() - t0

        # HVG
        t0 = time.perf_counter()
        n_top = min(N_HVGS, adata.n_vars)
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top, flavor="seurat_v3",
                subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0
            )
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
        timings["hvg"] = time.perf_counter() - t0

        # PCA
        t0 = time.perf_counter()
        sc.pp.pca(adata, n_comps=N_COMPS)
        timings["pca"] = time.perf_counter() - t0

        # kNN
        t0 = time.perf_counter()
        sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS)
        timings["knn"] = time.perf_counter() - t0

        # UMAP
        t0 = time.perf_counter()
        sc.tl.umap(adata)
        timings["umap"] = time.perf_counter() - t0

        # Leiden
        t0 = time.perf_counter()
        sc.tl.leiden(adata, resolution=1.0)
        timings["leiden"] = time.perf_counter() - t0

        # DE
        t0 = time.perf_counter()
        sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
        timings["de"] = time.perf_counter() - t0

        timings["total"] = sum(timings.values())
        rss_peak = get_rss_mb()
        timings["peak_rss_mb"] = round(rss_peak, 0)

        all_runs.append(timings)
        print(f"      Total: {timings['total']:.2f}s, "
              f"Peak RSS: {rss_peak:.0f} MB")

        if run == n_runs - 1:
            all_runs[-1]["_leiden"] = adata.obs["leiden"].values.copy()
            all_runs[-1]["_de_genes"] = get_de_gene_names(adata, 100)

        del adata
        gc.collect()

    return all_runs


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Benchmark Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline_benchmark(datasets=None, n_runs=N_RUNS):
    """Run all three pipeline variants and compare."""
    from sklearn.metrics import adjusted_rand_score

    if datasets is None:
        datasets = list(DATASETS.keys())

    print(f"\n{'='*60}")
    print(f"Full Pipeline Benchmark — 3 variants (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        info = DATASETS.get(ds_name, {})
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({info.get('cells', '?'):,} cells)")
        print(f"{'='*60}")

        result = {
            "benchmark": "accel_pipeline",
            "dataset": ds_name,
            "n_obs": info.get("cells", 0),
            "n_runs": n_runs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Run each variant
        ooc_runs = run_scx_ooc_pipeline(ds_name, n_runs=n_runs)
        prep_runs = run_scx_preprocess_pipeline(ds_name, n_runs=n_runs)
        scanpy_runs = run_scanpy_pipeline(ds_name, n_runs=n_runs)

        stages = ["normalize_log1p", "hvg", "pca", "knn", "umap", "leiden", "de"]

        for label, runs in [("scx_ooc", ooc_runs), ("scx_preprocess", prep_runs),
                             ("scanpy", scanpy_runs)]:
            if runs is None:
                continue

            totals = [r["total"] for r in runs]
            median_idx = sorted(range(len(totals)),
                                key=lambda i: totals[i])[len(totals) // 2]
            median_run = runs[median_idx]

            result[f"{label}_total_s"] = round(median_run["total"], 3)
            result[f"{label}_peak_rss_mb"] = median_run.get("peak_rss_mb", 0)
            for stage in stages:
                if stage in median_run:
                    result[f"{label}_{stage}_s"] = round(median_run[stage], 3)
            if "preprocess_write" in median_run:
                result[f"{label}_preprocess_write_s"] = round(
                    median_run["preprocess_write"], 3
                )

        # Quality: Leiden ARI between pipelines
        leiden_results = {}
        de_genes_results = {}
        for label, runs in [("scx_ooc", ooc_runs), ("scx_preprocess", prep_runs),
                             ("scanpy", scanpy_runs)]:
            if runs and "_leiden" in runs[-1]:
                leiden_results[label] = runs[-1]["_leiden"]
            if runs and "_de_genes" in runs[-1]:
                de_genes_results[label] = runs[-1]["_de_genes"]

        if "scx_ooc" in leiden_results and "scanpy" in leiden_results:
            ari = adjusted_rand_score(leiden_results["scx_ooc"],
                                       leiden_results["scanpy"])
            result["leiden_ari_ooc_vs_scanpy"] = round(ari, 4)
            print(f"\n  Leiden ARI (OOC vs Scanpy): {ari:.4f}")

        if "scx_preprocess" in leiden_results and "scanpy" in leiden_results:
            ari = adjusted_rand_score(leiden_results["scx_preprocess"],
                                       leiden_results["scanpy"])
            result["leiden_ari_prep_vs_scanpy"] = round(ari, 4)
            print(f"  Leiden ARI (Preprocess vs Scanpy): {ari:.4f}")

        if "scx_ooc" in de_genes_results and "scanpy" in de_genes_results:
            overlap = compute_de_gene_overlap(de_genes_results["scx_ooc"],
                                               de_genes_results["scanpy"])
            result["de_overlap_ooc_vs_scanpy"] = round(overlap, 4)
            print(f"  DE gene overlap (OOC vs Scanpy): {overlap:.4f}")

        # Speedup summary
        if "scanpy_total_s" in result and "scx_ooc_total_s" in result:
            result["speedup_ooc_vs_scanpy"] = round(
                result["scanpy_total_s"] / result["scx_ooc_total_s"], 2
            )
        if "scanpy_total_s" in result and "scx_preprocess_total_s" in result:
            result["speedup_prep_vs_scanpy"] = round(
                result["scanpy_total_s"] / result["scx_preprocess_total_s"], 2
            )

        # Print summary
        print(f"\n  Summary for {ds_name}:")
        for label in ["scx_ooc", "scx_preprocess", "scanpy"]:
            t = result.get(f"{label}_total_s", "N/A")
            rss = result.get(f"{label}_peak_rss_mb", "N/A")
            print(f"    {label:20s}: {t}s, peak RSS {rss} MB")

        results.append(result)

    save_json(results, "accel_pipeline_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Full Pipeline Benchmark Report",
        "",
        f"**Generated**: {ts}",
        f"**Pipeline**: Normalize -> log1p -> HVG -> PCA -> neighbors -> UMAP -> Leiden -> DE",
        "",
    ]

    path = RESULTS_DIR / "accel_pipeline_benchmark.json"
    if not path.exists():
        lines.append("No results found.")
        report = "\n".join(lines)
        (RESULTS_DIR / "accel_pipeline_benchmark.md").write_text(report)
        print(report)
        return report

    data = json.loads(path.read_text())

    # Overview table
    lines += [
        "## Total Pipeline Time",
        "",
        "| Dataset | Cells | SCX OOC (s) | SCX Preprocess (s) | Scanpy (s) | Speedup OOC | Speedup Prep |",
        "|---------|-------|-------------|--------------------|-----------:|-------------|--------------|",
    ]
    for r in data:
        ooc = r.get("scx_ooc_total_s", "N/A")
        prep = r.get("scx_preprocess_total_s", "N/A")
        scanpy = r.get("scanpy_total_s", "N/A")
        sp_ooc = r.get("speedup_ooc_vs_scanpy", "N/A")
        sp_prep = r.get("speedup_prep_vs_scanpy", "N/A")
        if isinstance(ooc, float):
            ooc = f"{ooc:.1f}"
        if isinstance(prep, float):
            prep = f"{prep:.1f}"
        if isinstance(scanpy, float):
            scanpy = f"{scanpy:.1f}"
        if isinstance(sp_ooc, float):
            sp_ooc = f"{sp_ooc:.1f}x"
        if isinstance(sp_prep, float):
            sp_prep = f"{sp_prep:.1f}x"
        lines.append(
            f"| {r['dataset']} | {r['n_obs']:,} | {ooc} | {prep} | {scanpy} | "
            f"{sp_ooc} | {sp_prep} |"
        )
    lines.append("")

    # Per-stage breakdown
    stages = ["normalize_log1p", "hvg", "pca", "knn", "umap", "leiden", "de"]
    for r in data:
        lines += [
            f"## Per-Stage Breakdown: {r['dataset']}",
            "",
            "| Stage | SCX OOC (s) | SCX Preprocess (s) | Scanpy (s) |",
            "|-------|-------------|--------------------|-----------:|",
        ]
        for stage in stages:
            ooc = r.get(f"scx_ooc_{stage}_s", "N/A")
            prep = r.get(f"scx_preprocess_{stage}_s", "N/A")
            scanpy = r.get(f"scanpy_{stage}_s", "N/A")
            if isinstance(ooc, float):
                ooc = f"{ooc:.2f}"
            if isinstance(prep, float):
                prep = f"{prep:.2f}"
            if isinstance(scanpy, float):
                scanpy = f"{scanpy:.2f}"
            lines.append(f"| {stage} | {ooc} | {prep} | {scanpy} |")
        if "scx_preprocess_preprocess_write_s" in r:
            pw = r["scx_preprocess_preprocess_write_s"]
            lines.append(f"| preprocess_write | - | {pw:.2f} | - |")
        lines.append("")

    # Quality
    lines += ["## Quality Metrics", ""]
    for r in data:
        lines.append(f"### {r['dataset']}")
        ari_ooc = r.get("leiden_ari_ooc_vs_scanpy", "N/A")
        ari_prep = r.get("leiden_ari_prep_vs_scanpy", "N/A")
        de_ooc = r.get("de_overlap_ooc_vs_scanpy", "N/A")
        lines.append(f"- Leiden ARI (OOC vs Scanpy): {ari_ooc}")
        lines.append(f"- Leiden ARI (Preprocess vs Scanpy): {ari_prep}")
        lines.append(f"- DE gene overlap (OOC vs Scanpy): {de_ooc}")
        lines.append("")

    # RSS
    lines += ["## Peak RSS (MB)", ""]
    for r in data:
        ooc = r.get("scx_ooc_peak_rss_mb", "N/A")
        prep = r.get("scx_preprocess_peak_rss_mb", "N/A")
        scanpy = r.get("scanpy_peak_rss_mb", "N/A")
        lines.append(f"- **{r['dataset']}**: OOC={ooc}, Preprocess={prep}, "
                      f"Scanpy={scanpy}")
    lines.append("")

    report = "\n".join(lines)
    md_path = RESULTS_DIR / "accel_pipeline_benchmark.md"
    md_path.write_text(report)
    print(f"\nReport saved: {md_path}")
    print(report)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCX Full Pipeline Benchmark (3 variants)"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["scx_ooc", "scx_preprocess", "scanpy", "all", "report"],
        help="Pipeline variant or 'all' (default: all)",
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        ensure_release_build()

    if args.mode == "report":
        generate_report()
        return

    if args.mode == "all":
        run_pipeline_benchmark(datasets=args.datasets, n_runs=args.n_runs)
    else:
        datasets = args.datasets or list(DATASETS.keys())
        for ds in datasets:
            if args.mode == "scx_ooc":
                run_scx_ooc_pipeline(ds, n_runs=args.n_runs)
            elif args.mode == "scx_preprocess":
                run_scx_preprocess_pipeline(ds, n_runs=args.n_runs)
            elif args.mode == "scanpy":
                run_scanpy_pipeline(ds, n_runs=args.n_runs)

    generate_report()


if __name__ == "__main__":
    main()
