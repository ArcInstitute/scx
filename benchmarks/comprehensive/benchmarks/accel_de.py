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

## Sort capacity

The GPU path's per-gene sort dispatches by pool size: `≤ 8192` keys takes
the single-block CUB `BlockRadixSort` fast path; larger pools use the tiled
bottom-up merge sort (`tile_block_radix_sort_kernel` +
`merge_pass_per_gene_kernel`). There's no upper limit beyond available
VRAM — census-scale 1-vs-rest Wilcoxon (`pool = n_obs ≈ 1M`) sorts in ~7
merge passes and finishes in a few seconds. Earlier versions of this
module skipped GPU rows above 8192 cells; those guards were removed when
G1.5 landed.
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
from benchmarks.comprehensive.runners.accel_runner import cpu_profile_capture

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
    scx_adata = _as_scx_backed_if_available(adata)
    if scx_adata is not None:
        pyscx.accel.rank_genes_groups(
            scx_adata, groupby, reference=reference, device="gpu",
        )
        _propagate_route(scx_adata, adata)
    else:
        pyscx.accel.rank_genes_groups(
            adata, groupby, reference=reference, device="gpu",
        )
    return "pyscx-gpu-wilcoxon"


def _run_pyscx_pdex_ref_cpu(adata: Any, groupby: str, reference: str) -> Any:
    import pyscx
    return pyscx.accel.pdex_ref(adata, groupby, reference=reference, device="cpu")


def _run_pyscx_pdex_ref_gpu(adata: Any, groupby: str, reference: str) -> Any:
    """Run pdex_ref on GPU.

    When the bench has built an SCX-backed fixture for this dataset (with a
    CSC sidecar), route through ``pyscx.open(scx_path)`` so the dataset
    arrives as a ``ScxBackedSparseDataset`` with ``backed_csc=Some(...)``.
    That makes pyscx's dispatch route to ``pdex_ref_gpu(GpuDeShardInput::Backed
    { csc: Some(...), .. })``, which exercises the v3 CSC-direct kernels (the
    path we actually want to bench; v3 is the unconditional default). The
    in-memory scipy CSR fallback routes through ``pdex_ref_gpu(Csr)`` →
    v3 CSR-fallback, which is NOT the path G4.3 is trying to measure.
    """
    import pyscx
    scx_adata = _as_scx_backed_if_available(adata)
    if scx_adata is not None:
        df = pyscx.accel.pdex_ref(
            scx_adata, groupby, reference=reference, device="gpu",
        )
        _propagate_route(scx_adata, adata)
        return df
    return pyscx.accel.pdex_ref(adata, groupby, reference=reference, device="gpu")


def _as_scx_backed_if_available(adata: Any) -> Any:
    """Open the bench-built SCX fixture (with CSC sidecar) for this adata
    via ``pyscx.open(...)`` if one was built, returning an AnnData whose
    ``.X`` is a ``ScxBackedSparseDataset``. Returns ``None`` when no SCX
    fixture has been built (older bench paths or non-pyscx contexts).

    Path is stashed in ``adata.uns["_bench_scx_with_csc_path"]`` by
    ``_ensure_scx_csc_fixture`` at fixture-build time.
    """
    scx_path = None
    try:
        scx_path = adata.uns.get("_bench_scx_with_csc_path")
    except Exception:
        scx_path = None
    if not scx_path:
        return None
    try:
        import pyscx
        # `pyscx.open()` returns an `Experiment`. To get an AnnData
        # whose `.X` is `ScxBackedSparseDataset` (with `backed_csc=Some(...)`
        # set when the file has a CSC sidecar), we have to go through
        # `Experiment.to_anndata(backed=True)`. Without `backed=True` we
        # get a fully-materialised AnnData with a scipy CSR X, which
        # routes through `pdex_ref_gpu(Csr)` and never exercises CSC.
        exp = pyscx.open(str(scx_path))
        return exp.to_anndata(backed=True)
    except Exception as e:
        logger.warning(
            "accel_de: failed to open SCX fixture %s (%s); "
            "falling back to in-memory AnnData (CSR path)",
            scx_path, e,
        )
        return None


def _propagate_route(src_adata: Any, dst_adata: Any) -> None:
    """Merge the accelerator route metadata that pyscx wrote to ``src_adata.uns``
    into ``dst_adata.uns`` so the runner (which only holds ``dst_adata``) can
    read it. Used when the GPU impl runs DE on an internally-opened SCX-backed
    AnnData distinct from the adata the runner passed in.

    Merges per-op (mirroring the Rust ``write_accel_route`` semantics) rather
    than overwriting, so route entries from other ops on ``dst_adata`` survive,
    and re-assigns the dict so it works on any ``uns`` backing."""
    try:
        accel = src_adata.uns.get("scx_accel")
    except Exception:
        accel = None
    if not accel:
        return
    try:
        merged = dict(dst_adata.uns.get("scx_accel", {}) or {})
        merged.update(accel)
        dst_adata.uns["scx_accel"] = merged
    except Exception:
        pass


def _extract_route(adata: Any, op: str) -> str | None:
    """Read ``adata.uns["scx_accel"][op]["route"]``, or None if absent."""
    try:
        return adata.uns["scx_accel"][op]["route"]
    except Exception:
        return None


def _extract_fallback(adata: Any, op: str) -> str | None:
    """Read ``adata.uns["scx_accel"][op]["fallback_reason"]``, or None."""
    try:
        return adata.uns["scx_accel"][op]["fallback_reason"]
    except Exception:
        return None


def _extract_shards_decoded(adata: Any, op: str) -> int | None:
    """Read ``adata.uns["scx_accel"][op]["shards_decoded"]`` (CSC/CSR-direct v3
    route telemetry), or None when absent / not tracked."""
    try:
        v = adata.uns["scx_accel"][op]["shards_decoded"]
        return None if v is None else int(v)
    except Exception:
        return None


def _extract_resident_csr(adata: Any, op: str) -> bool | None:
    """Read ``adata.uns["scx_accel"][op]["resident_csr"]`` (§9.11 telemetry).

    ``None`` when the route had no residency decision to make (CPU, dense, or
    a CSC-direct route, which prefilters by column range and never re-decodes)
    or when the stamp predates the field.
    """
    try:
        v = adata.uns["scx_accel"][op]["resident_csr"]
        return None if v is None else bool(v)
    except Exception:
        return None


def _ensure_scx_csc_fixture(adata: Any, dataset_name: str) -> Path | None:
    """Materialise the bench's prepared adata (groupby-tagged + subset to
    top groups) as a temp SCX file with a CSC sidecar.

    Required so the GPU pdex_ref / wilcoxon impls can route through
    ``pyscx.open(scx_path)`` → ``ScxBackedSparseDataset`` with
    ``backed_csc=Some(...)``. With only the h5ad-loaded scipy CSR in
    memory, pyscx routes to the in-memory `pdex_ref_gpu(Csr)` arm
    which has no CSC reader, so v3 dispatch always falls back to
    the CSR-direct path.

    Returns the path on success or None if pyscx is missing or the
    conversion failed. Cached on disk under
    ``$SCX_BENCH_TMPDIR/scx_csc_fixtures/`` (default
    ``/tmp/scx_csc_fixtures``); ``SCX_BENCH_REBUILD_CSC=1`` forces a
    rebuild.
    """
    import os
    if not _HAS_PYSCX:
        return None
    base = Path(
        os.environ.get("SCX_BENCH_TMPDIR")
        or os.environ.get("SCX_WORK_DIR", "")
    )
    if base.is_dir():
        out_dir = base / "scx_csc_fixtures"
    else:
        out_dir = Path("/tmp/scx_csc_fixtures")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("accel_de: cannot create %s (%s); skipping CSC fixture",
                       out_dir, e)
        return None
    scx_path = out_dir / f"{dataset_name}.bench_csc.scx"
    rebuild = os.environ.get("SCX_BENCH_REBUILD_CSC", "") in ("1", "true", "TRUE")
    if scx_path.exists() and not rebuild:
        logger.info("accel_de: reusing cached SCX-with-CSC fixture %s", scx_path)
        return scx_path
    if scx_path.exists():
        scx_path.unlink()
    try:
        import pyscx
        logger.info(
            "accel_de: building SCX-with-CSC fixture for %s -> %s "
            "(n_obs=%d n_vars=%d)",
            dataset_name, scx_path, int(adata.n_obs), int(adata.n_vars),
        )
        pyscx.from_anndata(adata, str(scx_path), csc="always")
    except Exception as e:
        logger.warning(
            "accel_de: pyscx.from_anndata(%s, csc='always') failed: %s; "
            "GPU impls will fall back to in-memory CSR path (no CSC bench)",
            dataset_name, e,
        )
        return None
    return scx_path


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
    """Extract per-(target, feature) p-values from a `pdex_ref` DataFrame.

    Container-agnostic on purpose. `pdex_ref` returns **pandas** by default
    (polars only on `output="polars"`), and both callers below sit inside a
    `try/except Exception → logger.warning`, so a container assumption here would
    not fail the run — it would silently drop `de_pval_agreement_vs_cpu` /
    `de_top_gene_overlap_vs_cpu` from every GPU triple, i.e. lose the CPU-vs-GPU
    correctness signal without saying so. Iterating columns works on either
    frame and needs no polars import (which previously scored the metric `nan`
    in a polars-less env).
    """
    if df is None:
        return None
    # Narrow on purpose: a missing/renamed column (KeyError), a non-frame
    # (TypeError) or an unparseable value (ValueError) is all that is worth
    # degrading to None. A broad `except Exception` would re-open the silent-loss
    # hole described above for any *other* bug in this extractor.
    #
    # The whole extraction — including the loop — is inside the guard. A null
    # p-value listifies to None, and `float(None)` raises TypeError; with the
    # loop outside, that one escaped to the caller's own
    # `except Exception -> logger.warning`, which is the silent metric loss this
    # helper exists to prevent. A null p-value has nothing to correlate, so it
    # becomes NaN — `_pval_agreement` already drops non-finite pairs — rather
    # than discarding every other gene's value with it.
    try:
        targets = list(df["target"])
        features = list(df["feature"])
        pvals = list(df["p_value"])
        out: dict[str, dict[str, float]] = {}
        for tgt, feat, pv in zip(targets, features, pvals):
            out.setdefault(str(tgt), {})[str(feat)] = (
                float("nan") if pv is None else float(pv)
            )
        return out
    except (KeyError, TypeError, ValueError):
        return None


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
        # Persist the (groupby-tagged, subset) adata to a temp SCX file
        # with CSC sidecar so the GPU pdex_ref / wilcoxon impls can load
        # it via pyscx.open() and exercise the v3-CSC dispatch path. With
        # only the in-memory scipy CSR, pyscx routes to
        # `pdex_ref_gpu(Csr)` → v3-CSR fallback, which never touches
        # the new CSC kernels.
        scx_csc_path = _ensure_scx_csc_fixture(adata_for_pick, dataset.name)
        if scx_csc_path is not None:
            adata_for_pick.uns["_bench_scx_with_csc_path"] = str(scx_csc_path)
        _fixture_cache[groupby_key] = (groupby, reference, synthetic, adata_for_pick)
    groupby, reference, synthetic, base_adata = _fixture_cache[groupby_key]

    # G1.5: the 8192-cell sort-pool cap has been lifted via tiled merge-sort
    # in `scx_gpu::gpu_de_block_sort` — every GPU variant now runs at every
    # dataset tier. The previous early-skip guards have been removed.

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
        # 2.0 ranking oracle: CPU decode/io/reduction/marshalling breakdown
        # (no-op unless SCX_CPU_PROFILE=1).
        extras: dict[str, Any] = {}
        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        try:
            with cpu_profile_capture(extras):
                ret = impl(a, groupby, reference)
        except Exception as e:
            logger.error("%s run %d raised: %s", key, i + 1, e)
            del a
            return None
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        # Record which accelerator route actually ran (read from adata.uns,
        # written by pyscx). `gpu_dispatch_route` is a human-readable string
        # for the coverage banner; `de_route_csc_direct` is a numeric gate
        # signal (1.0 iff the CSC-direct GPU route ran) so the existing
        # absolute-floor machinery in compare_against_baseline.py can fail the
        # gate when a CSC-direct benchmark silently fell back to CSR.
        op_key = "rank_genes_groups" if kind == "wilcoxon" else "pdex_ref"
        route = _extract_route(a, op_key)
        if route is not None:
            extras["gpu_dispatch_route"] = route
            fallback = _extract_fallback(a, op_key)
            if fallback is not None:
                extras["gpu_dispatch_fallback"] = fallback
            # §B.10 crit. 1 artifact field: how many CSC/CSR shards the v3 driver
            # decoded+uploaded across the call. On the CSC-direct route this is
            # below n_csc_shards × n_gene_chunks (range prefiltering); deterministic
            # validation lives in the scx-accel `csc::{pdex,wilcoxon}` multi-shard
            # tests. Recorded here for the route-marked artifact and dashboards.
            shards_decoded = _extract_shards_decoded(a, op_key)
            if shards_decoded is not None:
                extras["shards_decoded"] = float(shards_decoded)
            if requires_gpu and route.startswith("gpu_csr"):
                # Numeric gate signal for §9.11 device residency, in the same
                # shape as `de_route_csc_direct`: 1.0 when the CSR route held
                # the matrix device-resident across gene chunks, 0.0 on a
                # *silent* fall back to re-decoding every shard per chunk.
                #
                # Only emitted on a CSR route — the CSC-direct route prefilters
                # by column range and has no residency decision, so scoring it
                # would be meaningless rather than N/A. Residency is also
                # correctly declined for a single gene chunk, which at bench
                # scale (tens of thousands of genes, a 500-gene chunk) never
                # happens; if a future fixture makes it happen this reads 0.0
                # and the floor should be revisited rather than the code.
                resident = _extract_resident_csr(a, op_key)
                extras["de_route_resident_csr"] = 1.0 if resident else 0.0
            if requires_gpu and kind == "pdex_ref":
                # Numeric gate signal for the GPU pdex_ref triple. Always
                # emitted (so the absolute-floor gate never sees a missing
                # metric) and only 0.0 on a *silent fallback*: we built a CSC
                # sidecar fixture, yet a non-CSC route was recorded. When no CSC
                # fixture was built the signal is 1.0 = "not applicable / OK".
                #
                # GPU DE v3 is the unconditional default (the SCX_GPU_DE_V3 gate
                # was removed in Phase V1b), so a CSC fixture must dispatch
                # gpu_csc_v3. The expectation is keyed on the fixture alone, not
                # the recorded route — *any* non-gpu_csc_v3 route on a CSC fixture
                # is a silent fallback this gate catches (a gpu_csr_v3 CSC→CSR
                # drop, but also a deeper gpu_csr / cpu_* regression). The
                # separate `*_route_gpu_correct` gate covers GPU→CPU drops.
                csc_fixture = bool(a.uns.get("_bench_scx_with_csc_path"))
                extras["de_route_csc_direct"] = (
                    1.0 if (not csc_fixture or route == "gpu_csc_v3") else 0.0
                )
            if requires_gpu and kind == "wilcoxon":
                # Numeric gate signal for the GPU Wilcoxon triple. The variant
                # is skipped on non-GPU hosts (see the requires_gpu guard
                # upstream), so a recorded route always means GPU was attempted;
                # a cpu_* route here is a silent CPU fallback → 0.0 fails the
                # gate. Holds for both the v1 dense path and the v3 routes.
                extras["wilcoxon_route_gpu_correct"] = (
                    1.0 if route.startswith("gpu_") else 0.0
                )
                # CSC-direct route assertion, mirroring pdex_ref's
                # `de_route_csc_direct`. 1.0 when the CSC-direct route ran (or no
                # CSC fixture this run); 0.0 on a *silent fallback* — a CSC
                # sidecar fixture was built, yet a non-gpu_csc_v3 route was
                # recorded. GPU DE v3 is the unconditional default (Phase V1b),
                # so a CSC fixture must dispatch gpu_csc_v3; the expectation is
                # keyed on the fixture alone, not the recorded route, so a deeper
                # regression out of v3 (gpu_csr_v3 / gpu_csr / cpu_*) also
                # fails rather than scoring N/A.
                csc_fixture = bool(a.uns.get("_bench_scx_with_csc_path"))
                extras["wilcoxon_route_csc_direct"] = (
                    1.0 if (not csc_fixture or route == "gpu_csc_v3") else 0.0
                )

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
    # Surface the route at the dataset level too (route is stable across runs).
    run_routes = [
        r.extra.get("gpu_dispatch_route")
        for r in result.runs
        if r.extra.get("gpu_dispatch_route") is not None
    ]
    if run_routes:
        result.metadata["gpu_dispatch_route"] = run_routes[-1]
    return result
