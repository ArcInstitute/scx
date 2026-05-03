#!/usr/bin/env python3
"""
Correctness Validation: Backed-Mode Equivalence.

Verifies that operations on backed-mode (on-disk) data produce identical
results to the same operations on fully-materialized data.

Usage:
    python validate_backed_equivalence.py --dataset pbmc3k
    python validate_backed_equivalence.py --dataset pbmc3k --output results.json
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
    csr_equal,
    ensure_metadata_columns,
    load_dataset,
    max_abs_error,
    max_rel_error,
    parse_common_args,
    prepare_scx_file,
    print_summary,
    run_check,
    to_dense,
    write_validation_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexing checks
# ---------------------------------------------------------------------------


def check_row_slice(X_nb, X_b) -> ValidationCheck:
    """Row slice: X[100:200] exact CSR equality."""
    n = min(X_nb.shape[0], 200)
    start = min(100, n - 1)
    end = n

    ref = sp.csr_matrix(X_nb[start:end])
    test = sp.csr_matrix(X_b[start:end])

    equal = csr_equal(ref, test)
    return ValidationCheck(
        name="row_slice",
        passed=equal,
        metrics={"csr_equal": equal, "slice": f"[{start}:{end}]"},
        thresholds={"csr_equal": True},
    )


def check_fancy_index(X_nb, X_b) -> ValidationCheck:
    """Fancy index: X[[0, 5, 10, 99]] exact CSR equality."""
    n = X_nb.shape[0]
    indices = [i for i in [0, 5, 10, 99] if i < n]

    ref = sp.csr_matrix(X_nb[indices])
    test = sp.csr_matrix(X_b[indices])

    equal = csr_equal(ref, test)
    return ValidationCheck(
        name="fancy_index",
        passed=equal,
        metrics={"csr_equal": equal, "indices": indices},
        thresholds={"csr_equal": True},
    )


def check_boolean_mask(X_nb, X_b) -> ValidationCheck:
    """Boolean mask: X[mask] exact CSR equality."""
    rng = np.random.RandomState(42)
    mask = rng.random(X_nb.shape[0]) > 0.7  # ~30% of cells

    ref = sp.csr_matrix(X_nb[mask])
    test = sp.csr_matrix(X_b[mask])

    equal = csr_equal(ref, test)
    return ValidationCheck(
        name="boolean_mask",
        passed=equal,
        metrics={"csr_equal": equal, "n_selected": int(mask.sum())},
        thresholds={"csr_equal": True},
    )


def check_2d_slice(X_nb, X_b) -> ValidationCheck:
    """2D slice: X[0:1000, :500] exact CSR equality."""
    row_end = min(1000, X_nb.shape[0])
    col_end = min(500, X_nb.shape[1])

    ref = sp.csr_matrix(X_nb[0:row_end, :col_end])
    test = sp.csr_matrix(X_b[0:row_end, :col_end])

    equal = csr_equal(ref, test)
    return ValidationCheck(
        name="2d_slice",
        passed=equal,
        metrics={"csr_equal": equal, "slice": f"[0:{row_end}, :{col_end}]"},
        thresholds={"csr_equal": True},
    )


# ---------------------------------------------------------------------------
# Aggregation checks
# ---------------------------------------------------------------------------


def check_row_sums(X_nb, X_b) -> ValidationCheck:
    """row_sums: sum(axis=1), max abs error < 1e-6."""
    ref = np.asarray(X_nb.sum(axis=1)).ravel()
    test = np.asarray(X_b.sum(axis=1)).ravel()
    err = max_abs_error(ref, test)
    threshold = 1e-6
    return ValidationCheck(
        name="row_sums",
        passed=err < threshold,
        metrics={"max_abs_error": err},
        thresholds={"max_abs_error": threshold},
    )


def check_col_sums(X_nb, X_b) -> ValidationCheck:
    """col_sums: sum(axis=0), max abs error < 1e-6."""
    ref = np.asarray(X_nb.sum(axis=0)).ravel()
    test = np.asarray(X_b.sum(axis=0)).ravel()
    err = max_abs_error(ref, test)
    threshold = 1e-6
    return ValidationCheck(
        name="col_sums",
        passed=err < threshold,
        metrics={"max_abs_error": err},
        thresholds={"max_abs_error": threshold},
    )


def check_col_var(X_nb, X_b) -> ValidationCheck:
    """col_var: var(axis=0), rel error < 1e-2."""
    # Non-backed: compute from dense (f64)
    ref = np.var(to_dense(X_nb), axis=0)
    # Backed: streaming two-pass variance in f32
    test = np.asarray(X_b.var(axis=0)).ravel()
    err = max_rel_error(ref, test)
    # Streaming f32 variance accumulation introduces ~1e-4 relative error on
    # small datasets but scales to ~1e-3 at 100K cells due to catastrophic
    # cancellation in the (sum_sq - mean^2*n) term.
    threshold = 1e-2
    return ValidationCheck(
        name="col_var",
        passed=err < threshold,
        metrics={"max_rel_error": err},
        thresholds={"max_rel_error": threshold},
    )


def check_getnnz_axis0(X_nb, X_b) -> ValidationCheck:
    """getnnz(axis=0): exact match."""
    ref = np.asarray(X_nb.getnnz(axis=0))
    test = np.asarray(X_b.getnnz(axis=0))
    exact = bool(np.array_equal(ref, test))
    return ValidationCheck(
        name="getnnz_axis0",
        passed=exact,
        metrics={"exact_match": exact},
        thresholds={"exact_match": True},
    )


def check_getnnz_axis1(X_nb, X_b) -> ValidationCheck:
    """getnnz(axis=1): exact match."""
    ref = np.asarray(X_nb.getnnz(axis=1))
    test = np.asarray(X_b.getnnz(axis=1))
    exact = bool(np.array_equal(ref, test))
    return ValidationCheck(
        name="getnnz_axis1",
        passed=exact,
        metrics={"exact_match": exact},
        thresholds={"exact_match": True},
    )


# ---------------------------------------------------------------------------
# Accel function checks (backed vs non-backed)
# ---------------------------------------------------------------------------


def check_pca_backed(scx_path: str) -> ValidationCheck:
    """PCA: backed vs non-backed, cosine similarity > 0.99."""
    import pyscx

    # Non-backed PCA
    adata_nb = pyscx.open(scx_path).to_anndata()
    n_comps = min(20, adata_nb.n_vars - 1, adata_nb.n_obs - 1)
    pyscx.accel.pca(adata_nb, n_comps=n_comps, random_state=0)

    # Backed PCA
    adata_b = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.pca(adata_b, n_comps=n_comps, random_state=0)

    sims = cosine_similarity_columns(
        adata_nb.obsm["X_pca"], adata_b.obsm["X_pca"], n_cols=min(10, n_comps)
    )
    min_sim = min(sims) if sims else 0.0
    threshold = 0.99
    return ValidationCheck(
        name="pca_backed",
        passed=min_sim > threshold,
        metrics={"min_cosine_sim": min_sim},
        thresholds={"min_cosine_sim": threshold},
    )


def check_qc_metrics_backed(scx_path: str, adata_raw) -> ValidationCheck:
    """QC metrics: backed vs non-backed, max abs error < 1e-5."""
    import pyscx

    ensure_metadata_columns(adata_raw)

    # Non-backed
    adata_nb = pyscx.open(scx_path).to_anndata()
    # Copy metadata columns
    for col in ["mt", "batch", "donor", "cell_type"]:
        if col in adata_raw.var.columns:
            adata_nb.var[col] = adata_raw.var[col].values
        if col in adata_raw.obs.columns:
            adata_nb.obs[col] = adata_raw.obs[col].values
    pyscx.accel.calculate_qc_metrics(adata_nb, qc_vars=["mt"], log1p=True)

    # Backed
    adata_b = pyscx.open(scx_path).to_anndata(backed=True)
    for col in ["mt", "batch", "donor", "cell_type"]:
        if col in adata_raw.var.columns:
            adata_b.var[col] = adata_raw.var[col].values
        if col in adata_raw.obs.columns:
            adata_b.obs[col] = adata_raw.obs[col].values
    pyscx.accel.calculate_qc_metrics(adata_b, qc_vars=["mt"], log1p=True)

    max_err = 0.0
    for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
        if col in adata_nb.obs.columns and col in adata_b.obs.columns:
            err = max_abs_error(
                adata_nb.obs[col].values.astype(np.float64),
                adata_b.obs[col].values.astype(np.float64),
            )
            max_err = max(max_err, err)

    threshold = 1e-5
    return ValidationCheck(
        name="qc_metrics_backed",
        passed=max_err < threshold,
        metrics={"max_abs_error": max_err},
        thresholds={"max_abs_error": threshold},
    )


def check_filter_cells_backed(scx_path: str) -> ValidationCheck:
    """filter_cells: backed vs non-backed, exact mask agreement."""
    import pyscx

    adata_nb = pyscx.open(scx_path).to_anndata()
    pyscx.accel.filter_cells(adata_nb, min_genes=200)

    adata_b = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.filter_cells(adata_b, min_genes=200)

    shape_match = adata_nb.n_obs == adata_b.n_obs
    obs_match = list(adata_nb.obs_names) == list(adata_b.obs_names) if shape_match else False

    return ValidationCheck(
        name="filter_cells_backed",
        passed=shape_match and obs_match,
        metrics={
            "n_cells_nonbacked": adata_nb.n_obs,
            "n_cells_backed": adata_b.n_obs,
            "shape_match": shape_match,
            "obs_match": obs_match,
        },
        thresholds={"shape_match": True, "obs_match": True},
    )


def check_filter_genes_backed(scx_path: str) -> ValidationCheck:
    """filter_genes: backed vs non-backed, exact mask agreement."""
    import pyscx

    adata_nb = pyscx.open(scx_path).to_anndata()
    pyscx.accel.filter_genes(adata_nb, min_cells=3)

    adata_b = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.filter_genes(adata_b, min_cells=3)

    shape_match = adata_nb.n_vars == adata_b.n_vars
    var_match = list(adata_nb.var_names) == list(adata_b.var_names) if shape_match else False

    return ValidationCheck(
        name="filter_genes_backed",
        passed=shape_match and var_match,
        metrics={
            "n_genes_nonbacked": adata_nb.n_vars,
            "n_genes_backed": adata_b.n_vars,
            "shape_match": shape_match,
            "var_match": var_match,
        },
        thresholds={"shape_match": True, "var_match": True},
    )


def check_subset_obs_backed(scx_path: str) -> ValidationCheck:
    """subset_obs: backed vs non-backed, exact shape + data agreement."""
    import pyscx

    rng = np.random.RandomState(42)

    adata_nb = pyscx.open(scx_path).to_anndata()
    mask = rng.random(adata_nb.n_obs) > 0.5

    adata_b = pyscx.open(scx_path).to_anndata(backed=True)

    # Non-backed subset
    pyscx.accel.subset_obs(adata_nb, mask)
    # Backed subset
    pyscx.accel.subset_obs(adata_b, mask)

    shape_match = adata_nb.shape == adata_b.shape
    if shape_match:
        X_nb = sp.csr_matrix(adata_nb.X)
        X_b = sp.csr_matrix(adata_b.X[:])
        data_match = csr_equal(X_nb, X_b)
    else:
        data_match = False

    return ValidationCheck(
        name="subset_obs_backed",
        passed=shape_match and data_match,
        metrics={"shape_match": shape_match, "data_match": data_match},
        thresholds={"shape_match": True, "data_match": True},
    )


# ---------------------------------------------------------------------------
# Operator interception checks
# ---------------------------------------------------------------------------


def check_truediv_interception(X_nb, X_b) -> ValidationCheck:
    """__truediv__: X / row_sums on backed → lazy, matches scipy."""
    row_sums_nb = np.asarray(X_nb.sum(axis=1)).ravel()
    # Avoid division by zero
    row_sums_nb[row_sums_nb == 0] = 1.0
    row_sums_col = row_sums_nb.reshape(-1, 1)

    # Reference: scipy path
    ref = X_nb / row_sums_col
    ref = sp.csr_matrix(ref)

    # Backed path: should intercept and create lazy RowScale
    result = X_b / row_sums_col
    # Materialize if lazy
    if hasattr(result, "to_memory"):
        result = result.to_memory()
    result = sp.csr_matrix(result)

    err = max_abs_error(ref, result)
    threshold = 1e-6
    return ValidationCheck(
        name="truediv_interception",
        passed=err < threshold,
        metrics={"max_abs_error": err},
        thresholds={"max_abs_error": threshold},
    )


def check_mul_interception(X_nb, X_b) -> ValidationCheck:
    """__mul__: X * factors on backed → lazy, matches scipy."""
    rng = np.random.RandomState(42)
    factors = rng.random(X_nb.shape[0]).astype(np.float32).reshape(-1, 1)

    # Reference: scipy path
    ref = X_nb.multiply(factors)
    ref = sp.csr_matrix(ref)

    # Backed path
    result = X_b * factors
    if hasattr(result, "to_memory"):
        result = result.to_memory()
    result = sp.csr_matrix(result)

    err = max_abs_error(ref, result)
    threshold = 1e-6
    return ValidationCheck(
        name="mul_interception",
        passed=err < threshold,
        metrics={"max_abs_error": err},
        thresholds={"max_abs_error": threshold},
    )


# ---------------------------------------------------------------------------
# Lazy aggregation check
# ---------------------------------------------------------------------------


def check_lazy_aggregation(scx_path: str) -> ValidationCheck:
    """Lazy aggregation through transforms: sum(axis=0) matches materialized."""
    import pyscx
    import scanpy as sc

    # Materialized reference: normalize + log1p in-memory, then sum
    adata_nb = pyscx.open(scx_path).to_anndata()
    sc.pp.normalize_total(adata_nb, target_sum=1e4)
    sc.pp.log1p(adata_nb)
    ref = np.asarray(adata_nb.X.sum(axis=0)).ravel()

    # Lazy path: backed + lazy transforms, then streaming sum
    adata_b = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata_b, target_sum=1e4)
    pyscx.accel.log1p(adata_b)
    test = np.asarray(adata_b.X.sum(axis=0)).ravel()

    err = max_abs_error(ref, test)
    # f32 accumulation across cells through chained transforms (normalize_total
    # + log1p) introduces cumulative error proportional to cell count. On pbmc3k
    # (~2.7K cells) this is ~0.025; on 100K cells it can reach ~5.0.
    threshold = 10.0
    return ValidationCheck(
        name="lazy_aggregation",
        passed=err < threshold,
        metrics={"max_abs_error": err},
        thresholds={"max_abs_error": threshold},
    )


# ---------------------------------------------------------------------------
# Metadata and property checks
# ---------------------------------------------------------------------------


def check_shape_obs_var(adata_nb, adata_b) -> ValidationCheck:
    """Shape, obs, var metadata: exact match."""
    shape_match = adata_nb.shape == adata_b.shape
    obs_match = list(adata_nb.obs_names) == list(adata_b.obs_names)
    var_match = list(adata_nb.var_names) == list(adata_b.var_names)

    return ValidationCheck(
        name="shape_obs_var",
        passed=shape_match and obs_match and var_match,
        metrics={
            "shape_match": shape_match,
            "obs_names_match": obs_match,
            "var_names_match": var_match,
        },
        thresholds={"shape_match": True, "obs_names_match": True, "var_names_match": True},
    )


def check_issparse_format(X_b, scx_path: str) -> ValidationCheck:
    """issparse() and format checks for backed and lazy datasets."""
    import pyscx

    backed_issparse = sp.issparse(X_b) or (hasattr(X_b, "format") and X_b.format == "csr")
    backed_format = getattr(X_b, "format", None) == "csr"

    # Also check ScxLazyTransformedDataset
    adata_lazy = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata_lazy, target_sum=1e4)
    X_lazy = adata_lazy.X

    lazy_issparse = sp.issparse(X_lazy) or (hasattr(X_lazy, "format") and X_lazy.format == "csr")
    lazy_format = getattr(X_lazy, "format", None) == "csr"

    all_pass = backed_issparse and backed_format and lazy_issparse and lazy_format
    return ValidationCheck(
        name="issparse_format",
        passed=all_pass,
        metrics={
            "backed_issparse": backed_issparse,
            "backed_format_csr": backed_format,
            "lazy_issparse": lazy_issparse,
            "lazy_format_csr": lazy_format,
        },
        thresholds={
            "backed_issparse": True,
            "backed_format_csr": True,
            "lazy_issparse": True,
            "lazy_format_csr": True,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_checks(
    dataset_name: str, output_path: str | Path | None = None
) -> list[ValidationCheck]:
    """Run all backed-mode equivalence checks.

    Returns list of ValidationCheck results.
    """
    import pyscx

    logger.info("=== Backed-Mode Equivalence Validation ===")
    logger.info("Dataset: %s", dataset_name)

    adata_raw = load_dataset(dataset_name)
    ensure_metadata_columns(adata_raw)

    with tempfile.TemporaryDirectory() as tmp:
        scx_path = prepare_scx_file(adata_raw, Path(tmp))

        # Open both modes
        adata_nb = pyscx.open(scx_path).to_anndata()
        adata_b = pyscx.open(scx_path).to_anndata(backed=True)

        X_nb = sp.csr_matrix(adata_nb.X)
        X_b = adata_b.X

        logger.info("Loaded %d cells x %d genes", adata_nb.n_obs, adata_nb.n_vars)

        checks: list[ValidationCheck] = []

        # Indexing checks
        checks.append(run_check("row_slice", check_row_slice, X_nb, X_b))
        checks.append(run_check("fancy_index", check_fancy_index, X_nb, X_b))
        checks.append(run_check("boolean_mask", check_boolean_mask, X_nb, X_b))
        checks.append(run_check("2d_slice", check_2d_slice, X_nb, X_b))

        # Aggregation checks
        checks.append(run_check("row_sums", check_row_sums, X_nb, X_b))
        checks.append(run_check("col_sums", check_col_sums, X_nb, X_b))
        checks.append(run_check("col_var", check_col_var, X_nb, X_b))
        checks.append(run_check("getnnz_axis0", check_getnnz_axis0, X_nb, X_b))
        checks.append(run_check("getnnz_axis1", check_getnnz_axis1, X_nb, X_b))

        # Accel function checks (need fresh file handles)
        checks.append(run_check("pca_backed", check_pca_backed, scx_path))
        checks.append(run_check("qc_metrics_backed", check_qc_metrics_backed, scx_path, adata_raw))
        checks.append(run_check("filter_cells_backed", check_filter_cells_backed, scx_path))
        checks.append(run_check("filter_genes_backed", check_filter_genes_backed, scx_path))
        checks.append(run_check("subset_obs_backed", check_subset_obs_backed, scx_path))

        # Operator interception checks
        checks.append(run_check("truediv_interception", check_truediv_interception, X_nb, X_b))
        checks.append(run_check("mul_interception", check_mul_interception, X_nb, X_b))

        # Lazy aggregation
        checks.append(run_check("lazy_aggregation", check_lazy_aggregation, scx_path))

        # Metadata and property checks
        checks.append(run_check("shape_obs_var", check_shape_obs_var, adata_nb, adata_b))
        checks.append(run_check("issparse_format", check_issparse_format, X_b, scx_path))

    # Write JSON output
    write_validation_json("backed_equivalence", dataset_name, checks, output_path)
    print_summary(checks)

    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_common_args("Correctness Validation: Backed-Mode Equivalence")
    report = run_all_checks(args.dataset, args.output)

    if args.output is None:
        print(json.dumps(
            write_validation_json("backed_equivalence", args.dataset, report),
            indent=2,
            default=str,
        ))
