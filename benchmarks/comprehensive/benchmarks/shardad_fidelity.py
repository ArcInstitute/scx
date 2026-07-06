"""Shardad round-trip fidelity — source h5ad → .shad → read-back parity.

The SCX `roundtrip.py` / `correctness.py` benches are SCX-codec-specific (they
`pyscx.open` the converted path), so shardad has no read-back fidelity gate. This
module fills that gap: convert a source to a shardad archive, read it back with
`ShardedArchive.to_anndata()`, and assert the matrix round-trips exactly (counts
are lossless; float32 preserved). It also exercises shardad's in-decode
materialization knobs (`container="dense", data_dtype="float16", allow_lossy=`),
asserting shape/dtype so that first-class read API is covered too.

Self-materializing (like `grouped_read`): real datasets read their on-disk h5ad,
synthetic ones generate in-process via `_pert_synth` — so it carries no Phase-A
dependency (it's in `run_parallel._NO_CONVERSION`). `SUPPORTED_FORMATS={"shardad"}`.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from benchmarks.comprehensive.benchmarks.grouped_sort import _materialize_source
from benchmarks.comprehensive.benchmarks.roundtrip import _normalize_csr
from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.scripts.validation_helpers import csr_equal

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"shardad"})

# Real (on-disk h5ad) + synthetic count fixtures both formats can ingest.
_DATASETS: frozenset[str] = frozenset(
    {"pbmc3k", "tabula_sapiens_100k", "nb_glm_synth"}
)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Round-trip a source through shardad and verify read-back parity."""
    if format_variant is None or format_variant.key not in SUPPORTED_FORMATS:
        return None
    if dataset.name not in _DATASETS:
        return None

    import os

    import anndata
    import numpy as np

    try:
        from shardad import ShardedArchive, write_sharded
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "shardad is not installed in this environment. Install it "
            "(`pip install ~/dev/python/shardad`) or run under the scx-bench conda env."
        ) from e

    # Mirror ShardadRunner: derive worker count from RAYON_NUM_THREADS so the
    # fidelity write parallelism is consistent with the parallel-scaling sweeps.
    n_workers = int(os.environ.get("RAYON_NUM_THREADS") or 4)

    result = BenchmarkResult(
        benchmark="shardad_fidelity",
        format="shardad",
        dataset=dataset.name,
        metadata={},
    )

    workroot = tempfile.TemporaryDirectory(prefix=f"shardad_fidelity_{dataset.name}_")
    workdir = Path(workroot.name)
    t0 = time.perf_counter()
    try:
        src = _materialize_source(dataset, workdir)
        out = workdir / f"{dataset.name}.shad"
        write_sharded(str(src), str(out), overwrite=True, n_workers=n_workers)

        adata_src = anndata.read_h5ad(str(src))
        arch = ShardedArchive(str(out))
        adata_shad = arch.to_anndata()

        a = _normalize_csr(adata_src.X)
        b = _normalize_csr(adata_shad.X)
        shape_match = int(a.shape == b.shape)
        sparsity_match = int(a.nnz == b.nnz)
        if shape_match and sparsity_match:
            diff = a - b
            diff.eliminate_zeros()
            n_value_mismatches = int(diff.nnz)
        else:
            n_value_mismatches = max(int(a.nnz), int(b.nnz), 1)
        exact = int(shape_match and sparsity_match and csr_equal(a, b, rtol=0.0))

        # Materialization knobs: dense + float16 (lossy-allowed) read API.
        dense_f16_ok = 0
        try:
            dad = arch.to_anndata(
                container="dense", data_dtype="float16", allow_lossy=True
            )
            X = np.asarray(dad.X)
            dense_f16_ok = int(
                X.shape == tuple(a.shape) and str(X.dtype) == "float16"
            )
        except Exception as e:  # record the failure, don't crash the bench
            logger.warning("dense/float16 materialization failed: %s", e)

        # Fold the dense/float16 in-decode materialization into the pass/fail
        # gate: a materialization that raises or returns the wrong shape/dtype
        # (dense_f16_ok == 0) is a real fidelity regression, not just logged.
        overall_passed = bool(
            shape_match and sparsity_match and n_value_mismatches == 0 and dense_f16_ok
        )
        wall = time.perf_counter() - t0
        result.metadata.update(
            {
                "overall_passed": overall_passed,
                "source_dtype": str(adata_src.X.dtype),
                "source_nnz": int(a.nnz),
                "shardad_nnz": int(b.nnz),
                "n_value_mismatches": n_value_mismatches,
                "dense_float16_ok": bool(dense_f16_ok),
            }
        )
        result.add_run(
            wall_s=wall,
            overall_passed_int=1 if overall_passed else 0,
            shape_match=shape_match,
            sparsity_match=sparsity_match,
            n_value_mismatches=n_value_mismatches,
            csr_exact_equal_int=exact,
            dense_float16_ok_int=dense_f16_ok,
        )
        logger.info(
            "shardad_fidelity %s: %s (mismatches=%d, dense_f16=%s)",
            dataset.name,
            "PASS" if overall_passed else "FAIL",
            n_value_mismatches,
            bool(dense_f16_ok),
        )
    finally:
        workroot.cleanup()

    return result
