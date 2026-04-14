#!/usr/bin/env python3
"""
Correctness Validation: Preprocessing Path Cross-Validation (§3.14.3).

Verifies that the three SCX preprocessing paths — scanpy in-memory (A),
SCX write-back (B), and SCX lazy (C) — produce numerically equivalent results.

Usage:
    python validate_preprocessing_paths.py --dataset pbmc3k
    python validate_preprocessing_paths.py --dataset pbmc3k --output results.json
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.scripts.validation_helpers import (  # noqa: E402
    ValidationCheck,
    cosine_similarity_columns,
    gene_overlap_pct,
    load_dataset,
    max_abs_error,
    parse_common_args,
    print_summary,
    run_check,
    run_leiden,
    to_dense,
    write_validation_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------


def _build_path_a(adata_raw):
    """Path A: scanpy in-memory normalize_total + log1p."""
    import scanpy as sc

    adata = adata_raw.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _build_path_b(adata_raw, tmp_dir: Path):
    """Path B: SCX write-back via pyscx.preprocess()."""
    import pyscx

    src_path = str(tmp_dir / "path_b_src.scx")
    dst_path = str(tmp_dir / "path_b_dst.scx")
    pyscx.from_anndata(adata_raw, src_path)
    pyscx.preprocess(src_path, dst_path, ["normalize_total", "log1p"], target_sum=1e4)
    return pyscx.open(dst_path).to_anndata()


def _build_path_c(adata_raw, tmp_dir: Path):
    """Path C: SCX lazy transforms on backed data, then materialize."""
    import anndata
    import pyscx

    src_path = str(tmp_dir / "path_c_src.scx")
    pyscx.from_anndata(adata_raw, src_path)
    adata_b = pyscx.open(src_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata_b, target_sum=1e4)
    pyscx.accel.log1p(adata_b)

    # Materialize
    X_mat = adata_b.X.to_memory()
    adata_mat = anndata.AnnData(
        X=X_mat,
        obs=adata_b.obs.copy(),
        var=adata_b.var.copy(),
    )
    return adata_mat


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_normalize_log1p_threeway(adata_raw) -> ValidationCheck:
    """Three-way X comparison after normalize_total + log1p.

    Pairwise max abs error < 1e-5 for A-B, A-C, B-C.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        adata_a = _build_path_a(adata_raw)
        adata_b = _build_path_b(adata_raw, tmp_dir)
        adata_c = _build_path_c(adata_raw, tmp_dir)

    X_a = to_dense(adata_a.X)
    X_b = to_dense(adata_b.X)
    X_c = to_dense(adata_c.X)

    err_ab = max_abs_error(X_a, X_b)
    err_ac = max_abs_error(X_a, X_c)
    err_bc = max_abs_error(X_b, X_c)
    max_err = max(err_ab, err_ac, err_bc)

    threshold = 1e-5
    return ValidationCheck(
        name="normalize_log1p_threeway",
        passed=max_err < threshold,
        metrics={
            "err_A_vs_B": err_ab,
            "err_A_vs_C": err_ac,
            "err_B_vs_C": err_bc,
            "max_pairwise_error": max_err,
        },
        thresholds={"max_pairwise_error": threshold},
    )


def check_pca_threeway(adata_raw) -> ValidationCheck:
    """Three-way PCA comparison: pairwise cosine similarity > 0.99."""
    import scanpy as sc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        adata_a = _build_path_a(adata_raw)
        adata_b = _build_path_b(adata_raw, tmp_dir)
        adata_c = _build_path_c(adata_raw, tmp_dir)

    n_comps = min(50, adata_a.n_vars - 1, adata_a.n_obs - 1)
    n_compare = min(10, n_comps)

    # Run PCA on all three
    sc.pp.pca(adata_a, n_comps=n_comps, random_state=0)
    sc.pp.pca(adata_b, n_comps=n_comps, random_state=0)
    sc.pp.pca(adata_c, n_comps=n_comps, random_state=0)

    sims_ab = cosine_similarity_columns(
        adata_a.obsm["X_pca"], adata_b.obsm["X_pca"], n_cols=n_compare
    )
    sims_ac = cosine_similarity_columns(
        adata_a.obsm["X_pca"], adata_c.obsm["X_pca"], n_cols=n_compare
    )
    sims_bc = cosine_similarity_columns(
        adata_b.obsm["X_pca"], adata_c.obsm["X_pca"], n_cols=n_compare
    )

    min_ab = min(sims_ab) if sims_ab else 0.0
    min_ac = min(sims_ac) if sims_ac else 0.0
    min_bc = min(sims_bc) if sims_bc else 0.0
    min_all = min(min_ab, min_ac, min_bc)

    threshold = 0.99
    return ValidationCheck(
        name="pca_threeway",
        passed=min_all > threshold,
        metrics={
            "min_cos_A_vs_B": min_ab,
            "min_cos_A_vs_C": min_ac,
            "min_cos_B_vs_C": min_bc,
            "min_pairwise_cosine": min_all,
        },
        thresholds={"min_pairwise_cosine": threshold},
    )


def check_extended_pipeline(adata_raw) -> ValidationCheck:
    """Extended pipeline: Leiden ARI > 0.95, DE overlap > 90%, UMAP Procrustes > 0.95."""
    import scanpy as sc
    from scipy.spatial import procrustes
    from sklearn.metrics import adjusted_rand_score

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        adata_a = _build_path_a(adata_raw)
        adata_b = _build_path_b(adata_raw, tmp_dir)
        adata_c = _build_path_c(adata_raw, tmp_dir)

    # Run full pipeline on all three
    adatas = {"A": adata_a, "B": adata_b, "C": adata_c}
    for label, ad in adatas.items():
        n_comps = min(50, ad.n_vars - 1, ad.n_obs - 1)
        sc.pp.pca(ad, n_comps=n_comps, random_state=0)
        sc.pp.neighbors(ad, n_neighbors=15, random_state=0)
        sc.tl.umap(ad, random_state=0)
        run_leiden(ad, random_state=0)
        sc.tl.rank_genes_groups(ad, groupby="leiden", method="wilcoxon")

    # Leiden ARI
    ari_ab = adjusted_rand_score(adata_a.obs["leiden"], adata_b.obs["leiden"])
    ari_ac = adjusted_rand_score(adata_a.obs["leiden"], adata_c.obs["leiden"])
    ari_bc = adjusted_rand_score(adata_b.obs["leiden"], adata_c.obs["leiden"])
    min_ari = min(ari_ab, ari_ac, ari_bc)

    # DE gene overlap (top-50 per group)
    top_n = 50
    overlaps = []
    groups = list(adata_a.uns["rank_genes_groups"]["names"].dtype.names)
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for label_1, label_2 in pairs:
        ad_1 = adatas[label_1]
        ad_2 = adatas[label_2]
        for group in groups:
            genes_1 = list(ad_1.uns["rank_genes_groups"]["names"][group][:top_n])
            genes_2 = list(ad_2.uns["rank_genes_groups"]["names"][group][:top_n])
            overlaps.append(gene_overlap_pct(genes_1, genes_2, top_n))
    min_de_overlap = min(overlaps) if overlaps else 0.0
    mean_de_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    # UMAP Procrustes
    def _procrustes_corr(X1, X2):
        Z1, Z2, _ = procrustes(X1, X2)
        # Correlation between flattened aligned coordinates
        return float(np.corrcoef(Z1.ravel(), Z2.ravel())[0, 1])

    proc_ab = _procrustes_corr(adata_a.obsm["X_umap"], adata_b.obsm["X_umap"])
    proc_ac = _procrustes_corr(adata_a.obsm["X_umap"], adata_c.obsm["X_umap"])
    proc_bc = _procrustes_corr(adata_b.obsm["X_umap"], adata_c.obsm["X_umap"])
    min_proc = min(proc_ab, proc_ac, proc_bc)

    ari_threshold = 0.95
    de_threshold = 90.0
    proc_threshold = 0.95

    passed = min_ari > ari_threshold and min_de_overlap > de_threshold and min_proc > proc_threshold

    return ValidationCheck(
        name="extended_pipeline",
        passed=passed,
        metrics={
            "ari_A_vs_B": ari_ab,
            "ari_A_vs_C": ari_ac,
            "ari_B_vs_C": ari_bc,
            "min_leiden_ari": min_ari,
            "min_de_top50_overlap_pct": min_de_overlap,
            "mean_de_top50_overlap_pct": mean_de_overlap,
            "procrustes_A_vs_B": proc_ab,
            "procrustes_A_vs_C": proc_ac,
            "procrustes_B_vs_C": proc_bc,
            "min_procrustes_corr": min_proc,
        },
        thresholds={
            "min_leiden_ari": ari_threshold,
            "min_de_top50_overlap_pct": de_threshold,
            "min_procrustes_corr": proc_threshold,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_checks(
    dataset_name: str, output_path: str | Path | None = None
) -> list[ValidationCheck]:
    """Run all preprocessing path cross-validation checks.

    Returns list of ValidationCheck results.
    """
    logger.info("=== Preprocessing Path Cross-Validation (§3.14.3) ===")
    logger.info("Dataset: %s", dataset_name)

    adata_raw = load_dataset(dataset_name)

    logger.info("Loaded %d cells x %d genes", adata_raw.n_obs, adata_raw.n_vars)

    checks: list[ValidationCheck] = []

    checks.append(
        run_check("normalize_log1p_threeway", check_normalize_log1p_threeway, adata_raw)
    )
    checks.append(run_check("pca_threeway", check_pca_threeway, adata_raw))
    checks.append(run_check("extended_pipeline", check_extended_pipeline, adata_raw))

    # Write JSON output
    write_validation_json("preprocessing_paths", dataset_name, checks, output_path)
    print_summary(checks)

    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_common_args("Correctness Validation: Preprocessing Path Cross-Validation (§3.14.3)")
    report = run_all_checks(args.dataset, args.output)

    if args.output is None:
        print(json.dumps(
            write_validation_json("preprocessing_paths", args.dataset, report),
            indent=2,
            default=str,
        ))
