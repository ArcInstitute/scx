"""CSC dispatch benchmark — Phase L.3.

Compares `prefer_format="csr"` vs `prefer_format="csc"` on the four
accelerator entries that accept the kwarg. Each variant runs a
single operation against the same fixture loaded as a backed
AnnData:

  - `bench_csc__qc_metrics_csr` / `_csc`     (`calculate_qc_metrics`)
  - `bench_csc__hvg_csr`        / `_csc`     (`highly_variable_genes`,
                                              single-batch seurat_v3)
  - `bench_csc__de_csr`         / `_csc`     (`rank_genes_groups`)
  - `bench_csc__de_csr_bounded`              (`rank_genes_groups`, CSR, at
                                              the default shard cache)
  - `bench_csc__de_csc_densify`              (`rank_genes_groups`, CSC, the
                                              densify Wilcoxon kernel)
  - `bench_csc__col_sums_csr` / `_csc`       (`pyscx.accel.col_sums`)
  - `bench_csc__col_var_csr`  / `_csc`       (`pyscx.accel.col_var`)
  - `bench_csc__pdex_ref_csr`   / `_csc`     (`pdex_ref`)
  - `bench_csc__pseudobulk_csr` / `_csc`     (`pseudobulk_dex`)

CSC variants require a CSC sidecar; the runner converts the fixture
to a CSC-equipped SCX file once per dataset and caches the path
across variants.

Records wall time, peak RSS, and (for CSC variants on
`BackedCscReader`) cache hit/miss counts via
`pyscx`'s public `prefer_format` surface — no internal metrics
plumbing needed.

Every variant but one opens its handle with the shard cache sized to the
whole file (`_open_backed`), so the CSR arms never re-decode a shard. That is
the *favourable* regime for CSR, and not the one a real pipeline runs in:
`to_anndata(backed=True)` defaults to a 4-shard cache, and once a DE pass
visits more shards than that, every gene chunk re-decodes every shard.
`bench_csc__de_csr_bounded` is the same CSR DE call on a handle opened at that
default, so the suite measures the regime the sidecar exists to fix beside the
one it merely beats. It is scoped to `tabula_sapiens_100k` (7 CSR shards, so
the default cache thrashes) through `FORMAT_DATASET_SCOPE`: at census scale the
bounded CSR route runs for tens of minutes per repetition, which a warm-up plus
three runs would push past this benchmark's SLURM time budget.

`bench_csc__de_csc` times the default CSC Wilcoxon kernel, the exact-nnz one,
and its `csc_dispatch_correct` is 1.0 only for the route `cpu_csc_nnz`.
`bench_csc__de_csc_densify` is the same call on the densify kernel it replaced
(`SCX_ACCEL_WILCOXON_NNZ=0`), kept as the control that shows what the default
is worth, and floored on the route `cpu_csc`. It runs each repetition in a
fresh process: pyscx reads the gate once per process through a `OnceLock`, so
set in this interpreter after `bench_csc__de_csc` had run it would change
nothing and the arm would time the nnz kernel under the densify name. Both of
these are stricter than the other arms' "csc in route" test, which either
kernel passes.

The `col_*` arms carry no `csc_dispatch_correct`: those functions return an
array rather than annotating an AnnData, so there is no route to read. An
explicit `prefer_format="csc"` raises on a file without a sidecar rather than
falling back, so the arm fails instead of timing the wrong layout.

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
    pseudobulk_n_cpus_cap,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.subproc_arm import run_arm

logger = logging.getLogger(__name__)

_HAS_PYSCX = False
try:
    import pyscx  # noqa: F401
    _HAS_PYSCX = True
except ImportError:
    pass

# Cap pydeseq2's loky worker pool in the pseudobulk variant. Without it,
# DefaultInference forks one Python-interpreter worker per core (~300 MB each),
# OOM-killing the job on many-core nodes. pyscx.accel.pseudobulk_dex derives
# this from SLURM_CPUS_PER_TASK by default; passed explicitly for determinism.
_PSEUDOBULK_N_CPUS = pseudobulk_n_cpus_cap()

# Pinned, not left to the default. This variant exists to catch a silent
# CSC→CSR dispatch fallback, and it does so by reading
# `uns["scx_accel"]["pseudobulk_dex"]["route"]` and expecting `cpu_csc` /
# `cpu_csr`. Since v0.13 the default backend is `"nb_glm"`, whose route string
# names the *engine* (`cpu_nb_glm`) rather than the layout — leaving the gate
# comparing `cpu_nb_glm` against `cpu_nb_glm` for both arms, which can never
# fail. (The layout is still reported on that route, in `csc_available`; this
# pin keeps the gate measuring the same thing it always did.)
_PSEUDOBULK_BACKEND = "pydeseq2"

# Variants that open their handle at `to_anndata(backed=True)`'s default shard
# cache instead of one sized to the file. See the module docstring.
_BOUNDED_CACHE_VARIANTS: frozenset[str] = frozenset({"bench_csc__de_csr_bounded"})

# The one variant that runs every repetition in a fresh process; see the module
# docstring.
_DENSIFY_KEY = "bench_csc__de_csc_densify"

# Per repetition. The densify kernel measured ~20 s at tabula_sapiens_100k and
# ~226 s at census_1m, so this is ~8x the slowest observed run: long enough
# never to cut a real one short, short enough that a hung child frees the
# allocation.
_FRESH_PROCESS_TIMEOUT_S = 30 * 60

# Per-format dataset allow-lists, read by `run_parallel` at cohort-build time.
# Only the bounded arm is scoped; every other variant runs wherever it did.
FORMAT_DATASET_SCOPE: dict[str, frozenset[str]] = {
    "bench_csc__de_csr_bounded": frozenset({"tabula_sapiens_100k"}),
}

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
            name="rank_genes_groups (CSR, default shard cache)",
            key="bench_csc__de_csr_bounded",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="rank_genes_groups (CSC, densify Wilcoxon)",
            key="bench_csc__de_csc_densify",
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
    for op in ("col_sums", "col_var"):
        for layout in ("csr", "csc"):
            out.append(
                FormatVariant(
                    name=f"{op} ({layout.upper()})",
                    key=f"bench_csc__{op}_{layout}",
                    category="accel", runner="accel_runner",
                )
            )
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


def _open_backed(path: Path, *, bounded: bool = False) -> Any:
    # Size the LRU shard cache to the file's shard count. The default
    # cache_shards=4 is < tabula_sapiens_100k's 7 CSR shards, which makes the
    # CSR pdex_ref variant re-decode every shard per gene chunk (hours-long).
    # Caching all shards keeps the per-gene-chunk slab reads from thrashing.
    #
    # `bounded=True` leaves the cache at the default instead — that thrash is
    # exactly what `bench_csc__de_csr_bounded` exists to measure.
    exp = pyscx.open(str(path))
    if bounded:
        return exp.to_anndata(backed=True)
    return exp.to_anndata(backed=True, cache_shards=max(4, exp.shard_count))


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


def _unlabel_undersized_groups(adata: Any, groupby: str) -> bool:
    """Turn every level of ``obs[groupby]`` with fewer than two cells into
    unlabelled (NaN) cells, in place; return whether two or more levels remain.

    `rank_genes_groups` refuses a participating group with fewer than two cells
    (since 0.17), and a categorical also carries every level it was declared
    with, so census's `cell_type` — singleton levels among 125, from a
    census-wide vocabulary — raised before the first gene was tested. Dropping
    unused levels is not enough: a one-cell level is used. Unlabelling rather
    than subsetting keeps those cells where scanpy 1.12 puts them, in the rank
    pool and in every group's "rest", so the call still ranks every cell.
    """
    col = adata.obs[groupby]
    if not hasattr(col, "cat"):
        col = col.astype("category")
    counts = col.value_counts()
    col = col.cat.remove_categories([c for c in col.cat.categories if counts.get(c, 0) < 2])
    adata.obs[groupby] = col
    return len(col.cat.categories) >= 2


def _run_de(adata: Any, prefer: str) -> None:
    # Pick a reasonable groupby column. Most fixtures expose
    # `cell_type` or fall back to a synthetic 50/50 split.
    obs_cols = list(adata.obs.columns)
    candidates = ["cell_type", "leiden", "louvain", "cluster", "perturbation"]
    groupby = next((c for c in candidates if c in obs_cols), None)
    if groupby is not None and not _unlabel_undersized_groups(adata, groupby):
        groupby = None
    if groupby is None:
        # Fall back: build a 2-way synthetic split.
        n = adata.n_obs
        adata.obs["_bench_group"] = (np.arange(n) < n // 2).astype(str)
        groupby = "_bench_group"
    pyscx.accel.rank_genes_groups(
        adata, groupby, gene_chunk_size=500, prefer_format=prefer
    )


def _run_col_sums(adata: Any, prefer: str) -> None:
    pyscx.accel.col_sums(adata.X, prefer_format=prefer)


def _run_col_var(adata: Any, prefer: str) -> None:
    pyscx.accel.col_var(adata.X, prefer_format=prefer)


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
            n_cpus=_PSEUDOBULK_N_CPUS,
            backend=_PSEUDOBULK_BACKEND,
        )
    else:
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby,
            "_bench_cond",
            reference,
            prefer_format="csr",
            n_cpus=_PSEUDOBULK_N_CPUS,
            backend=_PSEUDOBULK_BACKEND,
        )


# Variant key → (runner, prefer_format)
_VARIANT_IMPLS: dict[str, tuple[Callable[[Any, str], None], str]] = {
    "bench_csc__qc_metrics_csr":  (_run_qc_metrics, "csr"),
    "bench_csc__qc_metrics_csc":  (_run_qc_metrics, "csc"),
    "bench_csc__hvg_csr":         (_run_hvg, "csr"),
    "bench_csc__hvg_csc":         (_run_hvg, "csc"),
    "bench_csc__de_csr":          (_run_de, "csr"),
    "bench_csc__de_csc":          (_run_de, "csc"),
    "bench_csc__de_csr_bounded":  (_run_de, "csr"),
    "bench_csc__de_csc_densify":  (_run_de, "csc"),
    "bench_csc__pdex_ref_csr":    (_run_pdex_ref, "csr"),
    "bench_csc__pdex_ref_csc":    (_run_pdex_ref, "csc"),
    "bench_csc__pseudobulk_csr":  (_run_pseudobulk, "csr"),
    "bench_csc__pseudobulk_csc":  (_run_pseudobulk, "csc"),
    "bench_csc__col_sums_csr":    (_run_col_sums, "csr"),
    "bench_csc__col_sums_csc":    (_run_col_sums, "csc"),
    "bench_csc__col_var_csr":     (_run_col_var, "csr"),
    "bench_csc__col_var_csc":     (_run_col_var, "csc"),
}

# Variant key → adata.uns["scx_accel"] op key for route extraction. Ops that
# stamp a route to `adata.uns["scx_accel"][op]["route"]` emit the numeric
# `csc_dispatch_correct` gate signal so a silent CSC→CSR (or CSR→CSC) fallback
# fails the gate. The five AnnData ops all stamp a route; the `col_*` arms are
# absent here because those functions return an array and stamp nothing.
_VARIANT_OP_KEY: dict[str, str] = {
    "bench_csc__qc_metrics_csr": "calculate_qc_metrics",
    "bench_csc__qc_metrics_csc": "calculate_qc_metrics",
    "bench_csc__hvg_csr": "highly_variable_genes",
    "bench_csc__hvg_csc": "highly_variable_genes",
    "bench_csc__de_csr": "rank_genes_groups",
    "bench_csc__de_csc": "rank_genes_groups",
    "bench_csc__de_csr_bounded": "rank_genes_groups",
    "bench_csc__de_csc_densify": "rank_genes_groups",
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


# One repetition of a fresh-process variant. Imports this module's own runner
# so the measured call is the in-process arm's, byte for byte.
_FRESH_WORKER = '''
import gc, json, resource, sys, time
from pathlib import Path
from benchmarks.comprehensive.benchmarks import bench_csc_dispatch as b
path = Path(sys.argv[1])
runner, prefer = b._VARIANT_IMPLS[b._DENSIFY_KEY]
adata = b._open_backed(path)
gc.collect()
ru0 = resource.getrusage(resource.RUSAGE_SELF)
t0 = time.perf_counter()
runner(adata, prefer)
wall = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({
    "wall_s": wall,
    "user_s": ru.ru_utime - ru0.ru_utime,
    "sys_s": ru.ru_stime - ru0.ru_stime,
    "peak_rss_mb": ru.ru_maxrss / 1024.0,
    "route": b._extract_route(adata, "rank_genes_groups"),
}))
'''


def _run_de_csc_densify(csc_path: Path, result: BenchmarkResult, n_runs: int):
    """Every repetition of `bench_csc__de_csc_densify` in its own interpreter,
    with `SCX_ACCEL_WILCOXON_NNZ=0`; see the module docstring.

    `wall_s`, `user_s` and `sys_s` cover the op alone, measured inside the
    worker as the in-process arms measure them. `peak_rss_mb` is the worker's
    whole-process high-water mark, open included — `ru_maxrss` has no delta.
    """
    for i in range(N_WARMUP_RUNS + n_runs):
        outcome = run_arm(
            _FRESH_WORKER, [str(csc_path)], env={"SCX_ACCEL_WILCOXON_NNZ": "0"},
            timeout_s=_FRESH_PROCESS_TIMEOUT_S, label=_DENSIFY_KEY,
        )
        if not outcome.ok or not outcome.records:
            logger.error("%s", outcome.failure_text(_DENSIFY_KEY))
            return None
        if i < N_WARMUP_RUNS:
            continue
        rec = outcome.records[-1]
        route = rec.get("route")
        extras: dict[str, Any] = {"csc_dispatch_correct": 1.0 if route == "cpu_csc" else 0.0}
        if route is not None:
            extras["dispatch_route"] = route
        result.add_run(
            wall_s=rec["wall_s"],
            user_s=rec["user_s"],
            sys_s=rec["sys_s"],
            peak_rss_mb=rec["peak_rss_mb"],
            **extras,
        )
        logger.info("  %s run %d: wall=%.3fs route=%s", _DENSIFY_KEY, i + 1 - N_WARMUP_RUNS,
                    rec["wall_s"], route)
    return result


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
    bounded = key in _BOUNDED_CACHE_VARIANTS

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
            # "default" = `to_anndata(backed=True)`'s own cache; "file" = one
            # slot per CSR shard, so nothing is ever re-decoded.
            "shard_cache": "default" if bounded else "file",
            "fresh_process": key == _DENSIFY_KEY,
        },
    )

    if key == _DENSIFY_KEY:
        return _run_de_csc_densify(csc_path, result, n_runs)

    # Warmup runs — discarded.
    for _ in range(N_WARMUP_RUNS):
        adata = _open_backed(csc_path, bounded=bounded)
        try:
            runner(adata, prefer)
        except Exception as e:
            logger.warning("%s warmup raised: %s", key, e)
        del adata
        gc.collect()

    for i in range(n_runs):
        gc.collect()
        adata = _open_backed(csc_path, bounded=bounded)
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
                if key == "bench_csc__de_csc":
                    # The default CSC Wilcoxon kernel, not merely a CSC one.
                    extras["csc_dispatch_correct"] = 1.0 if route == "cpu_csc_nnz" else 0.0
                elif prefer == "csc":
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
