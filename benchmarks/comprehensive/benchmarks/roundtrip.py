"""
Codec write→read round-trip parity.

Compares the SCX-materialised matrix against the original h5ad source to
detect silent value-corruption regressions in any of the SCX codecs:
``scx_auto`` (auto-selector), ``scx_none`` (raw LE), ``scx_scx1``
(Delta-Golomb-Rice / FOR-BP / Rice), ``scx_zstd``, ``scx_lz4``
(byte-shuffle + LZ4), and ``scx_pcodec``. ``correctness.py`` does not close
this gap because every check there starts from an SCX file written *and*
read by the same codec — a symmetric corruption (encode-then-decode that
loses bits identically) goes undetected.

Format-gated to SCX codec variants (returns ``None`` for h5ad / zarr /
tiledb / slaf rows). Compared after canonicalising both sides (sorted
indices, explicit zeros eliminated, cast to ``float32``) — pyscx always
materialises ``X`` as f32 regardless of on-disk encoding (uint8 / uint16 /
uint32 / float32 / float16 per the ``ScxCsr`` contract in CLAUDE.md), so a
literal ``dtype_match`` floor would always fail on integer source data.
The f32 cast is the comparison contract; raw dtypes are surfaced in
metadata for debugging. All current SCX codecs are lossless by design,
so the floor is exact equality (``n_value_mismatches == 0``).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult  # noqa: E402
from benchmarks.comprehensive.scripts.validation_helpers import csr_equal  # noqa: E402

logger = logging.getLogger(__name__)


SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "scx_auto",
    "scx_none",
    "scx_scx1",
    "scx_zstd",
    "scx_lz4",
    "scx_pcodec",
})
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. Mirrors the runtime
guard at the top of ``run()`` (defense-in-depth for direct invocation)."""


def _normalize_csr(X: Any):
    """Canonicalise X to a sorted, zero-stripped float32 CSR matrix.

    Handles dense ndarray, ``scipy.sparse.csc_matrix``, ``np.matrix``, and
    already-CSR inputs. Older 10x h5ad files store explicit zeros that SCX's
    encoder strips at write time — calling ``eliminate_zeros()`` on both
    sides before comparing is what keeps ``sparsity_match`` honest.
    """
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(X):
        csr = X.tocsr()
    else:
        csr = sp.csr_matrix(np.asarray(X))
    if csr.dtype != np.float32:
        csr = csr.astype(np.float32, copy=False)
    csr.sort_indices()
    csr.eliminate_zeros()
    return csr


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Compare the round-tripped SCX matrix against the source h5ad.

    ``n_runs`` is ignored — the comparison is deterministic. Repeating it
    n× would burn cluster time on a metric that cannot vary between runs.

    Returns ``None`` for non-SCX formats and when Phase A has not produced
    a converted file (mirrors the ``compression.py:88`` skip pattern; the
    orchestrator's ``_NO_CONVERSION`` table does not need editing — this
    benchmark *requires* the pre-converted file).
    """
    if format_variant.key not in SUPPORTED_FORMATS:
        return None

    if converted_path is None or not Path(converted_path).exists():
        logger.info(
            "roundtrip skip (%s, %s): no converted file at %s",
            format_variant.key,
            dataset.name,
            converted_path,
        )
        return None

    src_path = dataset.h5ad_path
    if not src_path.exists():
        logger.warning(
            "roundtrip skip (%s, %s): source h5ad missing at %s",
            format_variant.key,
            dataset.name,
            src_path,
        )
        return None

    import anndata
    import numpy as np
    import pyscx

    logger.info(
        "Round-trip parity check: format=%s dataset=%s",
        format_variant.key,
        dataset.name,
    )

    t0 = time.perf_counter()

    adata_src = anndata.read_h5ad(str(src_path))
    adata_scx = pyscx.open(str(converted_path)).to_anndata()

    source_dtype = str(adata_src.X.dtype)
    scx_dtype = str(adata_scx.X.dtype)
    source_shape = tuple(adata_src.X.shape)
    scx_shape = tuple(adata_scx.X.shape)

    a = _normalize_csr(adata_src.X)
    b = _normalize_csr(adata_scx.X)

    shape_match = int(a.shape == b.shape)
    sparsity_match = int(a.nnz == b.nnz)

    if shape_match and sparsity_match:
        diff = a - b
        diff.eliminate_zeros()
        n_value_mismatches = int(diff.nnz)
        if n_value_mismatches > 0:
            max_abs_diff = float(np.max(np.abs(diff.data)))
            denom = max(float(np.max(np.abs(b.data))) if b.nnz > 0 else 0.0, 1e-12)
            max_rel_diff = max_abs_diff / denom
        else:
            max_abs_diff = 0.0
            max_rel_diff = 0.0
        csr_exact_equal_int = int(csr_equal(a, b, rtol=0.0))
    else:
        # Shape / nnz mismatch — exact mismatch count is undefined, but
        # we know it's >= 1. Use a positive sentinel so the
        # n_value_mismatches max: 0 floor in thresholds.yaml actually
        # trips (a -1 sentinel would silently satisfy max: 0).
        # max(a.nnz, b.nnz, 1) is a defensible upper bound and >= 1.
        # max_abs_diff / max_rel_diff stay None (rather than inf) so the
        # persisted runs[].extra is strict JSON.
        n_value_mismatches = max(int(a.nnz), int(b.nnz), 1)
        max_abs_diff = None
        max_rel_diff = None
        csr_exact_equal_int = 0

    overall_passed = bool(
        shape_match and sparsity_match and n_value_mismatches == 0
    )
    overall_passed_int = 1 if overall_passed else 0

    wall_s = time.perf_counter() - t0

    result = BenchmarkResult(
        benchmark="roundtrip",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "overall_passed": overall_passed,
            "source_shape": list(source_shape),
            "scx_shape": list(scx_shape),
            "source_dtype": source_dtype,
            "scx_dtype": scx_dtype,
            "source_nnz": int(a.nnz),
            "scx_nnz": int(b.nnz),
            "n_value_mismatches": n_value_mismatches,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "csr_exact_equal": bool(csr_exact_equal_int),
            "source_h5ad_path": str(src_path),
            "scx_path": str(converted_path),
        },
    )
    # All gateable metrics live in runs[].extra — that's the only place
    # compare_against_baseline.py::_load_current_raw_metric reads from.
    # max_abs_diff / max_rel_diff are emitted as diagnostics; not floor-gated
    # today (redundant with n_value_mismatches == 0 for lossless codecs;
    # would need a non-zero tolerance once a lossy variant lands).
    result.add_run(
        wall_s=wall_s,
        overall_passed_int=overall_passed_int,
        shape_match=shape_match,
        sparsity_match=sparsity_match,
        n_value_mismatches=n_value_mismatches,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        csr_exact_equal_int=csr_exact_equal_int,
    )

    logger.info(
        "  -> %s (format=%s, dataset=%s): n_value_mismatches=%d, "
        "max_abs_diff=%s (%.1fs)",
        "PASS" if overall_passed else "FAIL",
        format_variant.key,
        dataset.name,
        n_value_mismatches,
        f"{max_abs_diff:.3g}" if max_abs_diff is not None else "n/a",
        wall_s,
    )

    return result
