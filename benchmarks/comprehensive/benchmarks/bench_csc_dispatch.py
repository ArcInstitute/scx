"""CSC dispatch benchmark — Phase L.3.

Compares `prefer_format="csr"` vs `prefer_format="csc"` on the four
accelerator entries that accept the kwarg. Each variant runs a
single operation against the same fixture loaded as a backed
AnnData:

  - `bench_csc__qc_metrics_csr` / `_csc`     (`calculate_qc_metrics`)
  - `bench_csc__hvg_csr`        / `_csc`     (`highly_variable_genes`,
                                              single-batch seurat_v3)
  - `bench_csc__de_csr`         / `_csc`     (`rank_genes_groups`)
  - `bench_csc__pdex_ref_csr`   / `_csc`     (`pdex_ref`)
  - `bench_csc__pseudobulk_csr` / `_csc`     (`pseudobulk_dex`)

CSC variants require a CSC sidecar; the runner converts the fixture
to a CSC-equipped SCX file once per dataset and caches the path
across variants.

Records wall time, peak RSS, and (for CSC variants on
`BackedCscReader`) cache hit/miss counts via
`pyscx`'s public `prefer_format` surface — no internal metrics
plumbing needed.

CSC has no GPU path; all variants are CPU. PCA is intentionally
absent (PCA explicitly rejects `prefer_format="csc"`).
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.benchmarks.accel_pca import (
    _get_cpu_times,
    _get_rss_mb,
)
from benchmarks.comprehensive.benchmarks.accel_preprocess import _load_raw
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    QUERY_N_HVGS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult

logger = logging.getLogger(__name__)

_HAS_PYSCX = False
try:
    import pyscx  # noqa: F401
    _HAS_PYSCX = True
except ImportError:
    pass

# Per-dataset cache for the converted CSC-equipped SCX file. Keyed by
# `(dataset.name, csc_cols_per_shard)`.
_csc_path_cache: dict[tuple[str, int], Path] = {}


def bench_csc_dispatch_variants() -> list[FormatVariant]:
    out = [
        FormatVariant(
            name="qc_metrics (CSR)",
            key="bench_csc__qc_metrics_csr",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="qc_metrics (CSC)",
            key="bench_csc__qc_metrics_csc",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="highly_variable_genes (CSR)",
            key="bench_csc__hvg_csr",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="highly_variable_genes (CSC)",
            key="bench_csc__hvg_csc",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="rank_genes_groups (CSR)",
            key="bench_csc__de_csr",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="rank_genes_groups (CSC)",
            key="bench_csc__de_csc",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pdex_ref (CSR)",
            key="bench_csc__pdex_ref_csr",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pdex_ref (CSC)",
            key="bench_csc__pdex_ref_csc",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pseudobulk_dex (CSR)",
            key="bench_csc__pseudobulk_csr",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pseudobulk_dex (CSC, gene_indices)",
            key="bench_csc__pseudobulk_csc",
            category="accel", runner="accel_runner",
        ),
    ]
    return out


def _convert_to_csc_scx(dataset: DatasetConfig, tmpdir: Path) -> Path:
    """Convert the dataset to a CSC-equipped SCX file. Cached per
    `(dataset, csc_cols_per_shard)` so the eight variants share the
    same on-disk fixture."""
    cps = 5000  # default `--csc-cols-per-shard`
    cache_key = (dataset.name, cps)
    if cache_key in _csc_path_cache:
        return _csc_path_cache[cache_key]

    out_path = tmpdir / f"{dataset.name}.csc.scx"
    if out_path.exists():
        _csc_path_cache[cache_key] = out_path
        return out_path

    raw = _load_raw(dataset)
    pyscx.from_anndata(raw, str(out_path), csc="always", csc_cols_per_shard=cps)
    _csc_path_cache[cache_key] = out_path
    return out_path


def _open_backed(path: Path) -> Any:
    return pyscx.open(str(path)).to_anndata(backed=True)


# ---------------------------------------------------------------------------
# Per-operation runners. Each takes a fresh backed AnnData and runs
# the op once; the harness handles warmup, repetitions, and timing.
# ---------------------------------------------------------------------------


def _run_qc_metrics(adata: Any, prefer: str) -> None:
    pyscx.accel.calculate_qc_metrics(adata, log1p=True, inplace=True, prefer_format=prefer)


def _run_hvg(adata: Any, prefer: str) -> None:
    pyscx.accel.highly_variable_genes(
        adata,
        n_top_genes=QUERY_N_HVGS,
        flavor="seurat_v3",
        device="cpu",
        prefer_format=prefer,
    )


def _run_de(adata: Any, prefer: str) -> None:
    # Pick a reasonable groupby column. Most fixtures expose
    # `cell_type` or fall back to a synthetic 50/50 split.
    obs_cols = list(adata.obs.columns)
    candidates = ["cell_type", "leiden", "louvain", "cluster", "perturbation"]
    groupby = next((c for c in candidates if c in obs_cols), None)
    if groupby is None:
        # Fall back: build a 2-way synthetic split.
        n = adata.n_obs
        adata.obs["_bench_group"] = (np.arange(n) < n // 2).astype(str)
        groupby = "_bench_group"
    pyscx.accel.rank_genes_groups(
        adata, groupby, gene_chunk_size=500, prefer_format=prefer
    )


def _run_pdex_ref(adata: Any, prefer: str) -> None:
    """Run pdex_ref with a binary perturbation-style grouping.

    Mirrors `_run_de`'s groupby column selection so the CSC vs CSR
    comparison is on the same fixture column. Picks the first
    available group level (alphabetically) as the reference so the
    benchmark is reproducible across runs.
    """
    obs_cols = list(adata.obs.columns)
    candidates = ["perturbation", "cell_type", "leiden", "louvain", "cluster"]
    groupby = next((c for c in candidates if c in obs_cols), None)
    if groupby is None:
        n = adata.n_obs
        adata.obs["_bench_group"] = (np.arange(n) < n // 2).astype(str)
        groupby = "_bench_group"
    levels = sorted(set(adata.obs[groupby].astype(str).tolist()))
    if len(levels) < 2:
        return
    reference = levels[0]
    pyscx.accel.pdex_ref(
        adata,
        groupby,
        reference=reference,
        gene_chunk_size=500,
        prefer_format=prefer,
        device="cpu",
    )


def _run_pseudobulk(adata: Any, prefer: str) -> None:
    """Run pseudobulk DE with a replicate-bearing design so pydeseq2 has
    residual degrees of freedom.

    A single-column groupby (``[test_col]``) yields one pseudobulk sample per
    condition level, so DESeq2 errors with "no replicates" — the variant never
    completed on any dataset. We synthesise a binary condition (``_bench_cond``)
    plus 4 pseudo-replicates (``_bench_rep``) and group by both, giving 8
    samples for a 2-coefficient design. This benchmark measures CSR-vs-CSC
    dispatch cost, not biological signal, so the synthetic design is fine.

    For CSC we project to the top 500 genes so the ``gene_indices`` precondition
    is satisfied; CSR runs the full gene set for a like-for-like comparison.
    """
    n = adata.n_obs
    adata.obs["_bench_cond"] = (np.arange(n) < n // 2).astype(str)
    adata.obs["_bench_rep"] = (np.arange(n) % 4).astype(str)
    groupby = ["_bench_cond", "_bench_rep"]
    reference = "True"  # _bench_cond levels are "True" / "False"
    if prefer == "csc":
        # 500 genes is enough to exercise the multi-shard slab read
        # without dragging the whole catalog through.
        gene_indices = list(range(min(500, adata.n_vars)))
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby,
            "_bench_cond",
            reference,
            prefer_format="csc",
            gene_indices=gene_indices,
        )
    else:
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby,
            "_bench_cond",
            reference,
            prefer_format="csr",
        )


# Variant key → (runner, prefer_format)
_VARIANT_IMPLS: dict[str, tuple[Callable[[Any, str], None], str]] = {
    "bench_csc__qc_metrics_csr":  (_run_qc_metrics, "csr"),
    "bench_csc__qc_metrics_csc":  (_run_qc_metrics, "csc"),
    "bench_csc__hvg_csr":         (_run_hvg, "csr"),
    "bench_csc__hvg_csc":         (_run_hvg, "csc"),
    "bench_csc__de_csr":          (_run_de, "csr"),
    "bench_csc__de_csc":          (_run_de, "csc"),
    "bench_csc__pdex_ref_csr":    (_run_pdex_ref, "csr"),
    "bench_csc__pdex_ref_csc":    (_run_pdex_ref, "csc"),
    "bench_csc__pseudobulk_csr":  (_run_pseudobulk, "csr"),
    "bench_csc__pseudobulk_csc":  (_run_pseudobulk, "csc"),
}

# Variant key → adata.uns["scx_accel"] op key for route extraction. Ops that
# stamp a route to `adata.uns["scx_accel"][op]["route"]` emit the numeric
# `csc_dispatch_correct` gate signal so a silent CSC→CSR (or CSR→CSC) fallback
# fails the gate. All five ops stamp a route, so every variant is gated.
_VARIANT_OP_KEY: dict[str, str] = {
    "bench_csc__qc_metrics_csr": "calculate_qc_metrics",
    "bench_csc__qc_metrics_csc": "calculate_qc_metrics",
    "bench_csc__hvg_csr": "highly_variable_genes",
    "bench_csc__hvg_csc": "highly_variable_genes",
    "bench_csc__de_csr": "rank_genes_groups",
    "bench_csc__de_csc": "rank_genes_groups",
    "bench_csc__pdex_ref_csr": "pdex_ref",
    "bench_csc__pdex_ref_csc": "pdex_ref",
    "bench_csc__pseudobulk_csr": "pseudobulk_dex",
    "bench_csc__pseudobulk_csc": "pseudobulk_dex",
}


def _extract_route(adata: Any, op: str) -> str | None:
    """Read ``adata.uns["scx_accel"][op]["route"]``, or None if absent."""
    try:
        return adata.uns["scx_accel"][op]["route"]
    except Exception:
        return None


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    if not _HAS_PYSCX:
        return None
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        return None
    runner, prefer = _VARIANT_IMPLS[key]

    # Materialise the CSC-equipped SCX file for the dataset (cached).
    # CSR variants also use this file — a CSC sidecar is harmless on
    # the CSR path and keeps the comparison apples-to-apples (same
    # underlying CSR shards).
    tmpdir = Path("/tmp") / "bench_csc_dispatch"
    tmpdir.mkdir(parents=True, exist_ok=True)
    csc_path = _convert_to_csc_scx(dataset, tmpdir)

    result = BenchmarkResult(
        benchmark="bench_csc_dispatch",
        format=key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "n_obs": dataset.n_obs,
            "prefer_format": prefer,
            "random_seed": RANDOM_SEED,
        },
    )

    # Warmup runs — discarded.
    for _ in range(N_WARMUP_RUNS):
        adata = _open_backed(csc_path)
        try:
            runner(adata, prefer)
        except Exception as e:
            logger.warning("%s warmup raised: %s", key, e)
        del adata
        gc.collect()

    for i in range(n_runs):
        gc.collect()
        adata = _open_backed(csc_path)
        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        try:
            runner(adata, prefer)
        except Exception as e:
            logger.error("%s run %d raised: %s", key, i + 1, e)
            del adata
            return None
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        # Verify the intended layout actually dispatched. Read the route
        # pyscx stamped on adata.uns and assert it matches `prefer`: a `_csc`
        # variant must run a route containing "csc"; a `_csr` variant must NOT.
        # 0.0 catches a silent fallback (e.g. require_csc() dropped to CSR
        # because the sidecar build failed); the absolute-floor gate fails on
        # it. Only emitted for ops that stamp a route (see _VARIANT_OP_KEY).
        extras: dict[str, Any] = {}
        op_key = _VARIANT_OP_KEY.get(key)
        if op_key is not None:
            route = _extract_route(adata, op_key)
            if route is not None:
                extras["dispatch_route"] = route
                if prefer == "csc":
                    extras["csc_dispatch_correct"] = 1.0 if "csc" in route else 0.0
                else:
                    extras["csc_dispatch_correct"] = 1.0 if "csc" not in route else 0.0

        result.add_run(
            wall_s=wall,
            user_s=u1 - u0,
            sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs rss=%.1f MB",
            key, i + 1, wall, max(rss_before, rss_after),
        )
        del adata
        gc.collect()

    return result
