"""Differential expression accelerator benchmark — PR series G1.

Exercises the two GPU-enabled DE entrypoints in `pyscx.accel`:

* `pdex_ref(adata, groupby, reference=…)` — pdex `mode="ref"` (per-cell
  Mann–Whitney U + pseudobulk geometric-mean log fold change). Compares
  every non-reference group against a chosen reference.
* `rank_genes_groups(adata, groupby, reference="rest")` — Wilcoxon
  rank-sum 1-vs-rest (matches scanpy's API).

Both functions now take a `device=` parameter (PR G1). This benchmark
captures the CPU baseline + GPU acceleration on the same fixture so
the regression gate can track GPU speedup over CPU and correctness
parity over scanpy.

## Variants

| `format_variant.key`                     | Backend                                  | Device |
|------------------------------------------|------------------------------------------|--------|
| `accel_de__scanpy_wilcoxon_cpu`          | `scanpy.tl.rank_genes_groups` (Wilcoxon) | CPU    |
| `accel_de__pyscx_wilcoxon_cpu`           | `pyscx.accel.rank_genes_groups`          | CPU    |
| `accel_de__pyscx_wilcoxon_gpu`           | `pyscx.accel.rank_genes_groups`          | GPU    |
| `accel_de__pyscx_pdex_ref_cpu`           | `pyscx.accel.pdex_ref`                   | CPU    |
| `accel_de__pyscx_pdex_ref_gpu`           | `pyscx.accel.pdex_ref`                   | GPU    |

The reference for the scanpy comparison is the Wilcoxon path — pdex's
`mode="ref"` doesn't have a scanpy peer (it's a perturbation-screening
contract pinned to upstream `pdex` instead).

## Correctness signal

For each GPU variant we record agreement against its CPU peer (same
formula, same fixture):

* `de_pval_agreement_vs_cpu` — Spearman correlation between the
  per-gene p-values returned by CPU and GPU paths, averaged across
  test groups. The GPU path's `erfc` + sort numerics drift on p-values
  past the 6th sig fig, so a tolerance comparison is the right shape.
  Stored in `runs[].extra` so the regression gate can apply an
  absolute-floor threshold.
* `de_top_gene_overlap_vs_cpu` — top-200 gene Jaccard between
  CPU and GPU rankings per group, median-averaged. Catches gross
  ranking divergence even if individual p-values drift.

## Groupby column selection

Real datasets have wildly different obs schemas. We probe a fixed
preference list and fall back to a deterministic 50/50 synthetic split
when none match. The chosen column is recorded in
`result.metadata["groupby"]` so the comparison stays apples-to-apples
across CPU / GPU variants on the same fixture.

## v1 capacity limit

The GPU path caps per-gene sort pools at
`scx_gpu::GPU_DE_BLOCK_SORT_CAPACITY = 8192`. For 1-vs-rest Wilcoxon
the "pool" is all cells, so census-scale datasets (>8192 cells) will
fail the GPU variant cleanly with `InvalidInput`. The runner catches
that, logs a warning, and emits `None` for the GPU rows on those
triples so the gate doesn't see a spurious failure. The pdex-ref path
caps on the reference group's size, which typically stays well under
the limit even at census scale.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.benchmarks.accel_pca import (
    _fixture_cache,
    _get_cpu_times,
    _get_rss_mb,
)
from benchmarks.comprehensive.benchmarks.accel_preprocess import _load_raw
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult

logger = logging.getLogger(__name__)

_HAS_PYSCX = False
_HAS_PYSCX_GPU = False
try:
    import pyscx  # noqa: F401
    _HAS_PYSCX = True
    try:
        _HAS_PYSCX_GPU = bool(pyscx.accel.gpu_info())
    except Exception:
        _HAS_PYSCX_GPU = False
except ImportError:
    pass

# Obs columns we'll try in order when choosing a groupby for the test.
_GROUPBY_PREFERENCE = [
    "cell_type",
    "leiden",
    "louvain",
    "cluster",
    "perturbation",
    "target",
]

# Top-N Jaccard overlap window for the GPU/CPU ranking agreement metric.
_TOP_N_FOR_OVERLAP = 200


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------


def accel_de_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="scanpy rank_genes_groups (Wilcoxon, CPU)",
            key="accel_de__scanpy_wilcoxon_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx rank_genes_groups (Wilcoxon, CPU)",
            key="accel_de__pyscx_wilcoxon_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx rank_genes_groups (Wilcoxon, GPU)",
            key="accel_de__pyscx_wilcoxon_gpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx pdex_ref (CPU)",
            key="accel_de__pyscx_pdex_ref_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx pdex_ref (GPU)",
            key="accel_de__pyscx_pdex_ref_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


# ---------------------------------------------------------------------------
# Per-variant runners. Each takes a fresh AnnData copy and runs the op.
# `groupby` and `reference` are chosen once per (dataset, run) and passed
# in by the harness; they don't vary across variants.
# ---------------------------------------------------------------------------


def _run_scanpy_wilcoxon(adata: Any, groupby: str, reference: str) -> str:
    import scanpy as sc
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", reference=reference)
    return "scanpy-cpu-wilcoxon"


def _run_pyscx_wilcoxon_cpu(adata: Any, groupby: str, reference: str) -> str:
    import pyscx
    pyscx.accel.rank_genes_groups(adata, groupby, reference=reference, device="cpu")
    return "pyscx-cpu-wilcoxon"


def _run_pyscx_wilcoxon_gpu(adata: Any, groupby: str, reference: str) -> str:
    import pyscx
    pyscx.accel.rank_genes_groups(adata, groupby, reference=reference, device="gpu")
    return "pyscx-gpu-wilcoxon"


def _run_pyscx_pdex_ref_cpu(adata: Any, groupby: str, reference: str) -> Any:
    import pyscx
    return pyscx.accel.pdex_ref(adata, groupby, reference=reference, device="cpu")


def _run_pyscx_pdex_ref_gpu(adata: Any, groupby: str, reference: str) -> Any:
    import pyscx
    return pyscx.accel.pdex_ref(adata, groupby, reference=reference, device="gpu")


# (impl_callable, requires_gpu, kind)  where kind ∈ {"wilcoxon", "pdex_ref"}.
_VARIANT_IMPLS: dict[str, tuple[Callable[..., Any], bool, str]] = {
    "accel_de__scanpy_wilcoxon_cpu": (_run_scanpy_wilcoxon,    False, "wilcoxon"),
    "accel_de__pyscx_wilcoxon_cpu":  (_run_pyscx_wilcoxon_cpu, False, "wilcoxon"),
    "accel_de__pyscx_wilcoxon_gpu":  (_run_pyscx_wilcoxon_gpu, True,  "wilcoxon"),
    "accel_de__pyscx_pdex_ref_cpu":  (_run_pyscx_pdex_ref_cpu, False, "pdex_ref"),
    "accel_de__pyscx_pdex_ref_gpu":  (_run_pyscx_pdex_ref_gpu, True,  "pdex_ref"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_groupby(adata: Any) -> tuple[str, str, bool]:
    """Pick the best obs column to groupby. Returns
    `(column_name, reference_value, synthetic)`.

    `synthetic == True` indicates we fell back to a 50/50 synthetic
    split and stamped a fresh column into `adata.obs`. The harness
    records this in metadata so cross-dataset comparisons stay honest
    — synthetic-groupby cells are not directly comparable to
    real-biology cells from another dataset.
    """
    obs_cols = list(adata.obs.columns)
    for cand in _GROUPBY_PREFERENCE:
        if cand not in obs_cols:
            continue
        levels = adata.obs[cand].astype(str)
        unique = levels.unique().tolist()
        # Need ≥ 2 groups and ≥ 10 cells in each of at least 2 groups
        # for the test to be meaningful.
        counts = levels.value_counts()
        viable = [u for u in unique if counts.get(u, 0) >= 10]
        if len(viable) >= 2:
            # Reference: prefer "non-targeting" if present (perturb-seq
            # convention), else the largest viable group.
            if "non-targeting" in viable:
                reference = "non-targeting"
            else:
                reference = max(viable, key=lambda v: counts[v])
            return cand, reference, False

    # Synthetic 50/50 split — deterministic on cell index.
    n = adata.n_obs
    rng = np.random.default_rng(RANDOM_SEED)
    labels = np.where(rng.random(n) < 0.5, "A", "B")
    adata.obs["_bench_de_group"] = labels.astype(str)
    return "_bench_de_group", "A", True


def _restrict_to_top_groups(adata: Any, groupby: str, reference: str, n_test_groups: int = 4) -> tuple[Any, str]:
    """Restrict the fixture to the reference + the N largest test groups.

    Large datasets like `tabula_sapiens_100k` ship hundreds of cell
    types — running DE against all of them blows the wall-clock
    budget for no scientific signal added. Keeping the top-N test
    groups is the canonical "small but real" choice and keeps the
    benchmark comparable across datasets.
    """
    levels = adata.obs[groupby].astype(str)
    counts = levels.value_counts()
    test_levels = [u for u in counts.index if u != reference][:n_test_groups]
    keep = [reference] + test_levels
    mask = levels.isin(keep).to_numpy()
    if mask.sum() == adata.n_obs:
        return adata, groupby
    sub = adata[mask].copy()
    return sub, groupby


def _wilcoxon_pvals_by_gene(adata: Any) -> dict[str, dict[str, float]] | None:
    """Extract per-(group, gene) p-values from `adata.uns["rank_genes_groups"]`.

    Returns `None` if the result wasn't stored (e.g. the op failed).
    """
    if "rank_genes_groups" not in adata.uns:
        return None
    rgg = adata.uns["rank_genes_groups"]
    try:
        names = rgg["names"]
        pvals = rgg["pvals"]
    except Exception:
        return None
    groups = list(names.dtype.names)
    out: dict[str, dict[str, float]] = {}
    for g in groups:
        out[g] = {str(n): float(p) for n, p in zip(names[g], pvals[g], strict=True)}
    return out


def _pdex_pvals_by_gene(df: Any) -> dict[str, dict[str, float]] | None:
    """Extract per-(target, feature) p-values from a `pdex_ref` polars DataFrame."""
    if df is None:
        return None
    try:
        import polars as pl  # noqa: F401
    except ImportError:
        return None
    rows = df.select(["target", "feature", "p_value"]).to_dicts()
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        tgt = str(row["target"])
        feat = str(row["feature"])
        out.setdefault(tgt, {})[feat] = float(row["p_value"])
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Returns nan if either array is
    constant or empty."""
    if a.size != b.size or a.size == 0:
        return float("nan")
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    a, b = a[finite], b[finite]
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pval_agreement(cpu: dict[str, dict[str, float]] | None,
                    gpu: dict[str, dict[str, float]] | None) -> float:
    """Mean Spearman correlation across shared (group, gene) p-values."""
    if cpu is None or gpu is None:
        return float("nan")
    shared_groups = sorted(set(cpu) & set(gpu))
    if not shared_groups:
        return float("nan")
    vals: list[float] = []
    for g in shared_groups:
        shared_genes = sorted(set(cpu[g]) & set(gpu[g]))
        if not shared_genes:
            continue
        a = np.array([cpu[g][k] for k in shared_genes], dtype=np.float64)
        b = np.array([gpu[g][k] for k in shared_genes], dtype=np.float64)
        rho = _spearman(a, b)
        if not np.isnan(rho):
            vals.append(rho)
    if not vals:
        return float("nan")
    return round(float(np.mean(vals)), 6)


def _top_gene_overlap(cpu: dict[str, dict[str, float]] | None,
                      gpu: dict[str, dict[str, float]] | None,
                      n: int = _TOP_N_FOR_OVERLAP) -> float:
    """Median Jaccard of top-N gene sets across shared groups."""
    if cpu is None or gpu is None:
        return float("nan")
    shared_groups = sorted(set(cpu) & set(gpu))
    jaccards: list[float] = []
    for g in shared_groups:
        c = sorted(cpu[g].items(), key=lambda kv: kv[1])[:n]
        gp = sorted(gpu[g].items(), key=lambda kv: kv[1])[:n]
        cs, gs = set(k for k, _ in c), set(k for k, _ in gp)
        if not cs or not gs:
            continue
        jaccards.append(len(cs & gs) / len(cs | gs))
    if not jaccards:
        return float("nan")
    return round(float(np.median(jaccards)), 4)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        return None
    impl, requires_gpu, kind = _VARIANT_IMPLS[key]

    if key.startswith("accel_de__pyscx") and not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        return None

    raw = _load_raw(dataset)
    groupby_key = ("accel_de_groupby", dataset.name)
    if groupby_key not in _fixture_cache:
        # Picking groupby + restricting on a fresh copy so we never
        # mutate the cached raw fixture (other accel_* benchmarks
        # depend on it).
        adata_for_pick = raw.copy()
        groupby, reference, synthetic = _select_groupby(adata_for_pick)
        adata_for_pick, _ = _restrict_to_top_groups(adata_for_pick, groupby, reference)
        _fixture_cache[groupby_key] = (groupby, reference, synthetic, adata_for_pick)
    groupby, reference, synthetic, base_adata = _fixture_cache[groupby_key]

    # GPU 1-vs-rest Wilcoxon: pool is all n_obs cells. Skip cleanly when
    # the dataset exceeds the v1 sort capacity.
    if key == "accel_de__pyscx_wilcoxon_gpu" and base_adata.n_obs > 8192:
        logger.warning(
            "accel_de: skipping %s on %s (n_obs=%d > GPU_DE_BLOCK_SORT_CAPACITY=8192)",
            key, dataset.name, base_adata.n_obs,
        )
        return None

    # GPU pdex_ref: pool is the reference group only.
    if key == "accel_de__pyscx_pdex_ref_gpu":
        n_ref = int((base_adata.obs[groupby].astype(str) == reference).sum())
        if n_ref > 8192:
            logger.warning(
                "accel_de: skipping %s on %s (n_ref=%d > GPU_DE_BLOCK_SORT_CAPACITY=8192)",
                key, dataset.name, n_ref,
            )
            return None

    result = BenchmarkResult(
        benchmark="accel_de",
        format=key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "n_obs": int(base_adata.n_obs),
            "n_vars": int(base_adata.n_vars),
            "groupby": groupby,
            "reference": reference,
            "synthetic_groupby": synthetic,
            "kind": kind,
            "random_seed": RANDOM_SEED,
        },
    )

    # Materialise a CPU-baseline result once for the parity comparison.
    # Cached per (dataset, kind) so we don't pay for it on every GPU run.
    cpu_baseline_key = ("accel_de_cpu_baseline", dataset.name, kind)
    cpu_pvals: dict[str, dict[str, float]] | None = None
    if requires_gpu:
        if cpu_baseline_key not in _fixture_cache:
            try:
                cpu_adata = base_adata.copy()
                if kind == "wilcoxon":
                    _run_pyscx_wilcoxon_cpu(cpu_adata, groupby, reference)
                    _fixture_cache[cpu_baseline_key] = _wilcoxon_pvals_by_gene(cpu_adata)
                else:
                    df = _run_pyscx_pdex_ref_cpu(cpu_adata, groupby, reference)
                    _fixture_cache[cpu_baseline_key] = _pdex_pvals_by_gene(df)
                del cpu_adata
                gc.collect()
            except Exception as e:
                logger.warning("accel_de: CPU baseline materialisation failed for %s/%s: %s",
                               dataset.name, kind, e)
                _fixture_cache[cpu_baseline_key] = None
        cpu_pvals = _fixture_cache[cpu_baseline_key]

    # Warmup runs — discarded.
    for _ in range(N_WARMUP_RUNS):
        warm = base_adata.copy()
        try:
            impl(warm, groupby, reference)
        except Exception as e:
            logger.warning("%s warmup raised: %s", key, e)
        del warm
        gc.collect()

    pval_agreements: list[float] = []
    overlaps: list[float] = []

    for i in range(n_runs):
        gc.collect()
        a = base_adata.copy()
        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        try:
            ret = impl(a, groupby, reference)
        except Exception as e:
            logger.error("%s run %d raised: %s", key, i + 1, e)
            del a
            return None
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        extras: dict[str, float] = {}
        if requires_gpu and cpu_pvals is not None:
            try:
                if kind == "wilcoxon":
                    gpu_pvals = _wilcoxon_pvals_by_gene(a)
                else:
                    gpu_pvals = _pdex_pvals_by_gene(ret)
                rho = _pval_agreement(cpu_pvals, gpu_pvals)
                jac = _top_gene_overlap(cpu_pvals, gpu_pvals)
                if not np.isnan(rho):
                    pval_agreements.append(rho)
                    extras["de_pval_agreement_vs_cpu"] = rho
                if not np.isnan(jac):
                    overlaps.append(jac)
                    extras["de_top_gene_overlap_vs_cpu"] = jac
            except Exception as e:
                logger.warning("accel_de: parity comparison failed for %s run %d: %s",
                               key, i + 1, e)

        result.add_run(
            wall_s=wall,
            user_s=u1 - u0,
            sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs rho=%s jaccard=%s",
            key, i + 1, wall,
            pval_agreements[-1] if pval_agreements else float("nan"),
            overlaps[-1] if overlaps else float("nan"),
        )
        del a
        ret = None
        gc.collect()

    if pval_agreements:
        result.metadata["de_pval_agreement_vs_cpu"] = round(float(np.median(pval_agreements)), 6)
    if overlaps:
        result.metadata["de_top_gene_overlap_vs_cpu"] = round(float(np.median(overlaps)), 4)
    return result
