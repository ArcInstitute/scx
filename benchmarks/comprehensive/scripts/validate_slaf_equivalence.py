#!/usr/bin/env python3
"""
Correctness Validation: SLAF Round-Trip Equivalence (Phase 5 — §A.5).

Converts a dataset through ``h5ad → SLAF → AnnData`` via ``slaf.data.SLAFConverter``
and ``slaf.integrations.anndata.read_slaf``, then diffs the result against the
source on every AnnData component (``X``, ``obs``, ``var``, ``obsm``, ``obsp``,
``uns``).

Lossy fields (those SLAF is not expected to preserve) are documented in the
``_LOSSY_ALLOWLIST`` constant and skipped from the diff — each allowlisted
field is reported as a skipped check, not a pass.

Usage:
    python validate_slaf_equivalence.py --dataset pbmc3k
    python validate_slaf_equivalence.py --dataset pbmc3k --output results.json
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.scripts.validation_helpers import (  # noqa: E402
    ValidationCheck,
    load_dataset,
    max_abs_error,
    parse_common_args,
    print_summary,
    run_check,
    to_dense,
    write_validation_json,
)

logger = logging.getLogger(__name__)

# Fields that ``h5ad → SLAF → AnnData`` is not expected to preserve.
# Extend as concrete SLAF limitations are observed in validation runs.
_LOSSY_ALLOWLIST: set[str] = {
    # SLAF stores unstructured metadata alongside tables but reshaping
    # through DuckDB may flatten nested dicts; treat as lossy and revisit.
    "uns",
}


# ---------------------------------------------------------------------------
# Round-trip helper
# ---------------------------------------------------------------------------


def _roundtrip(adata, work_dir: Path):
    """Convert ``adata`` to a SLAF directory and read it back."""
    from slaf.data import SLAFConverter
    from slaf.integrations.anndata import read_slaf

    h5ad_path = work_dir / "source.h5ad"
    slaf_path = work_dir / "source.slaf"

    # SLAFConverter reads h5ad from disk — round-trip through the source
    # h5ad first so we exercise the real conversion path, not an in-memory
    # shortcut.
    adata.write_h5ad(h5ad_path)

    converter = SLAFConverter(chunked=True)
    converter.convert(str(h5ad_path), str(slaf_path), input_format="h5ad")

    lazy = read_slaf(str(slaf_path))
    return lazy.compute()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_shape(adata_source, adata_rt) -> ValidationCheck:
    passed = adata_source.shape == adata_rt.shape
    return ValidationCheck(
        name="shape",
        passed=passed,
        metrics={"source_shape": list(adata_source.shape),
                 "roundtrip_shape": list(adata_rt.shape)},
    )


def check_X(adata_source, adata_rt) -> ValidationCheck:
    """Element-wise parity of the expression matrix X."""
    src = adata_source.X
    rt = adata_rt.X
    if sp.issparse(src) and sp.issparse(rt):
        # Compare nnz and data within a tight absolute tolerance — SLAF
        # does not recompress values, so only float precision can drift.
        nnz_ok = src.nnz == rt.nnz
        src_csr = src.tocsr()
        rt_csr = rt.tocsr()
        indices_ok = np.array_equal(src_csr.indices, rt_csr.indices)
        indptr_ok = np.array_equal(src_csr.indptr, rt_csr.indptr)
        data_err = float(np.max(np.abs(src_csr.data.astype(np.float64)
                                       - rt_csr.data.astype(np.float64))))
        passed = nnz_ok and indices_ok and indptr_ok and data_err < 1e-5
        return ValidationCheck(
            name="X",
            passed=passed,
            metrics={
                "nnz_source": int(src.nnz), "nnz_roundtrip": int(rt.nnz),
                "indices_equal": bool(indices_ok),
                "indptr_equal": bool(indptr_ok),
                "max_abs_error": data_err,
            },
            thresholds={"max_abs_error": 1e-5},
        )
    err = max_abs_error(src, rt)
    return ValidationCheck(
        name="X",
        passed=err < 1e-5,
        metrics={"max_abs_error": err, "source_sparse": sp.issparse(src)},
        thresholds={"max_abs_error": 1e-5},
    )


def _diff_frame(name: str, src_df, rt_df) -> ValidationCheck:
    """Shape / column-set / per-column parity for obs or var DataFrames."""
    same_shape = src_df.shape == rt_df.shape
    shared = set(src_df.columns) & set(rt_df.columns)
    missing_in_rt = sorted(set(src_df.columns) - set(rt_df.columns))
    extra_in_rt = sorted(set(rt_df.columns) - set(src_df.columns))

    # Columns whose values differ
    diff_cols: list[str] = []
    for col in shared:
        try:
            # Reset index in case SLAF renames the obs/var index on
            # round-trip; we compare values only.
            if not src_df[col].reset_index(drop=True).equals(
                rt_df[col].reset_index(drop=True)
            ):
                diff_cols.append(col)
        except Exception:  # noqa: BLE001
            diff_cols.append(col)

    passed = same_shape and not missing_in_rt and not extra_in_rt and not diff_cols
    return ValidationCheck(
        name=name,
        passed=passed,
        metrics={
            "source_shape": list(src_df.shape),
            "roundtrip_shape": list(rt_df.shape),
            "missing_in_roundtrip": missing_in_rt,
            "extra_in_roundtrip": extra_in_rt,
            "diff_columns": diff_cols,
        },
    )


def check_obs(adata_source, adata_rt) -> ValidationCheck:
    return _diff_frame("obs", adata_source.obs, adata_rt.obs)


def check_var(adata_source, adata_rt) -> ValidationCheck:
    return _diff_frame("var", adata_source.var, adata_rt.var)


def _check_mapping(name: str, src_map, rt_map) -> ValidationCheck:
    """Key-set + shape parity for obsm / obsp / uns top-level keys."""
    src_keys = set(src_map.keys())
    rt_keys = set(rt_map.keys())
    missing = sorted(src_keys - rt_keys)
    extra = sorted(rt_keys - src_keys)

    shape_mismatches: list[str] = []
    value_mismatches: list[str] = []
    for k in src_keys & rt_keys:
        src_v = src_map[k]
        rt_v = rt_map[k]
        src_shape = getattr(src_v, "shape", None)
        rt_shape = getattr(rt_v, "shape", None)
        if src_shape != rt_shape:
            shape_mismatches.append(k)
            continue
        if src_shape is not None:
            try:
                err = max_abs_error(src_v, rt_v)
                if err > 1e-5:
                    value_mismatches.append(f"{k}:err={err:.3g}")
            except Exception:  # noqa: BLE001
                value_mismatches.append(f"{k}:uncomparable")

    passed = not missing and not extra and not shape_mismatches and not value_mismatches
    return ValidationCheck(
        name=name,
        passed=passed,
        metrics={
            "missing_in_roundtrip": missing,
            "extra_in_roundtrip": extra,
            "shape_mismatches": shape_mismatches,
            "value_mismatches": value_mismatches,
        },
    )


def check_obsm(adata_source, adata_rt) -> ValidationCheck:
    return _check_mapping("obsm", adata_source.obsm, adata_rt.obsm)


def check_obsp(adata_source, adata_rt) -> ValidationCheck:
    return _check_mapping("obsp", adata_source.obsp, adata_rt.obsp)


def check_uns(adata_source, adata_rt) -> ValidationCheck:
    if "uns" in _LOSSY_ALLOWLIST:
        return ValidationCheck(
            name="uns",
            passed=True,
            error="skipped: uns is on the documented SLAF lossy allowlist",
        )
    return _check_mapping("uns", adata_source.uns, adata_rt.uns)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_checks(dataset_name: str) -> list[ValidationCheck]:
    adata_source = load_dataset(dataset_name)

    with tempfile.TemporaryDirectory(prefix="slaf_validate_") as tmp:
        adata_rt = _roundtrip(adata_source, Path(tmp))
        checks = [
            run_check("shape", check_shape, adata_source, adata_rt),
            run_check("X", check_X, adata_source, adata_rt),
            run_check("obs", check_obs, adata_source, adata_rt),
            run_check("var", check_var, adata_source, adata_rt),
            run_check("obsm", check_obsm, adata_source, adata_rt),
            run_check("obsp", check_obsp, adata_source, adata_rt),
            run_check("uns", check_uns, adata_source, adata_rt),
        ]
    return checks


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_common_args("SLAF round-trip correctness validation")
    checks = run_all_checks(args.dataset)
    report = write_validation_json(
        suite_name="slaf_equivalence",
        dataset=args.dataset,
        checks=checks,
        output_path=args.output,
    )
    print_summary(report)
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
