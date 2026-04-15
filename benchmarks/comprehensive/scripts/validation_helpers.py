"""
Shared utilities for the Correctness Validation Suite (§3.14).

Provides:
  - ``ValidationCheck``: structured result for a single validation check
  - Comparison helpers: max_abs_error, cosine_similarity_columns, etc.
  - Dataset loading and JSON output helpers
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks" / "scripts"))

from bench_env import DATA_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    error: str | None = None  # Non-None means the check was skipped

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop None error for cleaner JSON
        if d["error"] is None:
            del d["error"]
        return d


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def to_dense(x: Any) -> np.ndarray:
    """Convert sparse or dense matrix to a dense numpy array."""
    if sp.issparse(x):
        return x.toarray()
    if isinstance(x, np.matrix):
        return np.asarray(x)
    return np.asarray(x)


def max_abs_error(a: Any, b: Any) -> float:
    """Maximum absolute element-wise error between two matrices/arrays."""
    a_d = to_dense(a).ravel().astype(np.float64)
    b_d = to_dense(b).ravel().astype(np.float64)
    if a_d.size == 0 and b_d.size == 0:
        return 0.0
    return float(np.max(np.abs(a_d - b_d)))


def max_rel_error(a: Any, b: Any, eps: float = 1e-12) -> float:
    """Maximum relative element-wise error between two arrays."""
    a_d = to_dense(a).ravel().astype(np.float64)
    b_d = to_dense(b).ravel().astype(np.float64)
    if a_d.size == 0 and b_d.size == 0:
        return 0.0
    denom = np.maximum(np.abs(b_d), eps)
    return float(np.max(np.abs(a_d - b_d) / denom))


def cosine_similarity_columns(
    A: np.ndarray, B: np.ndarray, n_cols: int | None = None
) -> list[float]:
    """Per-column cosine similarity with sign-ambiguity handling.

    Returns absolute cosine similarity for each column (PC), which handles
    the sign ambiguity inherent in PCA / SVD decompositions.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if n_cols is None:
        n_cols = min(A.shape[1], B.shape[1])
    sims = []
    for i in range(n_cols):
        col_a = A[:, i]
        col_b = B[:, i]
        norm_a = np.linalg.norm(col_a)
        norm_b = np.linalg.norm(col_b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            sims.append(0.0)
        else:
            sims.append(float(abs(np.dot(col_a, col_b) / (norm_a * norm_b))))
    return sims


def pearson_r(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
    from scipy.stats import pearsonr

    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if len(a_arr) < 2:
        return 0.0
    r, _ = pearsonr(a_arr, b_arr)
    return float(r)


def spearman_r(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr

    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if len(a_arr) < 2:
        return 0.0
    r, _ = spearmanr(a_arr, b_arr)
    return float(r)


def recall_at_k(
    ref_indices: np.ndarray, test_indices: np.ndarray, k: int | None = None
) -> float:
    """Average recall@k: fraction of true kNN captured across all query points.

    Parameters
    ----------
    ref_indices : (n, k_ref) array of reference neighbor indices
    test_indices : (n, k_test) array of test neighbor indices
    k : number of neighbors to compare (default: min of both k dimensions)
    """
    if k is None:
        k = min(ref_indices.shape[1], test_indices.shape[1])
    ref = ref_indices[:, :k]
    test = test_indices[:, :k]
    n = ref.shape[0]
    recalls = []
    for i in range(n):
        ref_set = set(ref[i])
        test_set = set(test[i])
        recalls.append(len(ref_set & test_set) / max(len(ref_set), 1))
    return float(np.mean(recalls))


def gene_overlap_pct(genes_a: Sequence[str], genes_b: Sequence[str], top_n: int) -> float:
    """Percentage overlap of top-N gene lists."""
    set_a = set(list(genes_a)[:top_n])
    set_b = set(list(genes_b)[:top_n])
    if len(set_a) == 0:
        return 0.0
    return 100.0 * len(set_a & set_b) / len(set_a)


def csr_equal(a: sp.csr_matrix, b: sp.csr_matrix, rtol: float = 1e-6) -> bool:
    """Check if two CSR matrices are equal within tolerance."""
    a = sp.csr_matrix(a)
    b = sp.csr_matrix(b)
    if a.shape != b.shape:
        return False
    if not np.array_equal(a.indptr, b.indptr):
        return False
    if not np.array_equal(a.indices, b.indices):
        return False
    return bool(np.allclose(a.data, b.data, rtol=rtol, atol=0))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_name: str):
    """Load a dataset from DATA_DIR as AnnData.

    Raises FileNotFoundError if the h5ad is missing.
    """
    import anndata

    path = DATA_DIR / f"{dataset_name}.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {path}. "
            f"Ensure SCX_WORK_DIR is set and the dataset is available."
        )
    logger.info("Loading %s from %s", dataset_name, path)
    return anndata.read_h5ad(str(path))


def ensure_metadata_columns(adata) -> None:
    """Add synthetic batch/donor/mt columns if missing.

    These are needed for stratified DE and pseudobulk tests.
    """
    import pandas as pd

    rng = np.random.RandomState(42)

    if "batch" not in adata.obs.columns:
        adata.obs["batch"] = pd.Categorical(
            rng.choice(["batch_A", "batch_B", "batch_C"], size=adata.n_obs)
        )

    if "donor" not in adata.obs.columns:
        adata.obs["donor"] = [f"donor_{i % 5}" for i in range(adata.n_obs)]

    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = pd.Categorical(
            rng.choice(["T cell", "B cell", "NK cell", "Monocyte"], size=adata.n_obs)
        )

    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")


def run_leiden(adata, **kwargs) -> None:
    """Run Leiden clustering with graceful fallback.

    Tries igraph flavor first, then leidenalg, then louvain. Raises
    ``ImportError`` if no graph clustering backend is available.
    """
    import scanpy as sc

    random_state = kwargs.pop("random_state", 0)
    try:
        sc.tl.leiden(
            adata, flavor="igraph", n_iterations=2, directed=False,
            random_state=random_state, **kwargs,
        )
        return
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        sc.tl.leiden(adata, random_state=random_state, **kwargs)
        return
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        sc.tl.louvain(adata, random_state=random_state, **kwargs)
        adata.obs["leiden"] = adata.obs["louvain"]
        return
    except (ImportError, ModuleNotFoundError):
        pass
    raise ImportError(
        "No graph clustering backend available. "
        "Install one of: python-igraph, leidenalg, or louvain."
    )


def prepare_scx_file(adata, tmp_dir: Path, name: str = "test.scx") -> str:
    """Write AnnData to SCX file and return the path."""
    import pyscx

    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path)
    return path


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_validation_json(
    suite_name: str,
    dataset: str,
    checks: list[ValidationCheck],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and optionally write the JSON validation report.

    Returns the report dict.
    """
    n_passed = sum(1 for c in checks if c.passed and c.error is None)
    n_failed = sum(1 for c in checks if not c.passed and c.error is None)
    n_skipped = sum(1 for c in checks if c.error is not None)

    report: dict[str, Any] = {
        "harness": suite_name,
        "dataset": dataset,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "overall_passed": n_failed == 0,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "total_duration_s": round(sum(c.duration_s for c in checks), 2),
        "results": [c.to_dict() for c in checks],
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Wrote validation report to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_common_args(description: str) -> argparse.Namespace:
    """Parse common CLI arguments for validation scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        default="pbmc3k",
        help="Dataset name (default: pbmc3k). Must exist as {name}.h5ad in DATA_DIR.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def run_check(name: str, fn, *args, **kwargs) -> ValidationCheck:
    """Run a check function with timing and error handling.

    The check function should return a ValidationCheck. If it raises,
    the check is recorded as failed with the error message.
    """
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        result.duration_s = time.perf_counter() - t0
        return result
    except Exception as e:
        logger.exception("Check '%s' raised an exception", name)
        return ValidationCheck(
            name=name,
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def print_summary(checks: list[ValidationCheck]) -> None:
    """Print a human-readable summary of validation results."""
    n_passed = sum(1 for c in checks if c.passed and c.error is None)
    n_failed = sum(1 for c in checks if not c.passed and c.error is None)
    n_skipped = sum(1 for c in checks if c.error is not None)

    print(f"\n{'=' * 60}")
    print(f"  Passed:  {n_passed}")
    print(f"  Failed:  {n_failed}")
    print(f"  Skipped: {n_skipped}")
    print(f"{'=' * 60}")

    for c in checks:
        if c.error is not None:
            status = "SKIP"
        elif c.passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  [{status}] {c.name:45s} ({c.duration_s:.1f}s)")
        if c.metrics:
            for k, v in c.metrics.items():
                threshold = c.thresholds.get(k, "")
                t_str = f" (threshold: {threshold})" if threshold else ""
                if isinstance(v, float):
                    print(f"         {k}: {v:.6g}{t_str}")
                else:
                    print(f"         {k}: {v}{t_str}")
        if c.error:
            print(f"         error: {c.error}")
    print()
