"""The "laptop test": a full 9-stage analysis under a fixed memory ceiling.

Every component of the standard scverse pipeline (PCA, kNN, Leiden, DE) is
already benchmarked here in isolation. What is not: whether the *whole*
pipeline runs to completion on 1M cells inside the 16 or 32 GB an analyst's
workstation actually has, while the scanpy in-memory path is OOM-killed. That
end-to-end claim is the one the community asks about and the suite cannot
currently make.

Naming note: this benchmark carries no `accel_` prefix but is structurally an
accelerator benchmark — self-contained, never touching the `runners/*_runner.py`
contract surface. `bench_csc_dispatch` is in the same position, and both are
named in `run_parallel._ARM_SHAPED_BENCHMARKS`, which drives the format
pairing, the format-pool trigger and the smoke-gate exclusion from one place.

Memory clamping: the ceiling is the SLURM cgroup, not an in-process `prlimit`.
The harness already works this way — the `SCX_BENCH_OOC_MEM_CAP_GB` block in
`config.estimate_memory_gb` caps the request *below* a benchmark's footprint
precisely to force the out-of-core regime — and `MEMORY_BUDGET_GB` below is
read by an early return in that same function, ahead of its `+50% safety` /
round-to-8-GB tail. Two further adjustments would otherwise raise the ceiling
behind the arm's back, so `_per_job_slurm_params` exempts this benchmark from
both: the `--scale-factor` multiplier (1.3 turns 16 GB into 21) and the
`--mem-gb` floor that `capture_baseline.py` always passes.
`pipeline_completed_int` is a claim about a *specific* ceiling; if the
allocation is not the one on the label, the number means nothing.

`RLIMIT_AS` is deliberately not used as a second belt. An SCX open mmaps the
whole file, so an address-space cap refuses the mapping of a file far larger
than the arm's residency — it would fail the backed streaming arm this
benchmark exists to showcase, on memory that arm never makes resident. The
opt-in in `subproc_arm` is `RLIMIT_DATA`, for local runs outside SLURM.

## An OOM is the measurement, not an error

The scanpy arms are *expected* to die at 16 GB on >=500K cells. That is only a
result if it is recorded, so the arm runs in a child that prints one JSON line
per completed stage with `flush=True`. When the cgroup kills it, the trail
names the stage the ceiling was hit in. The child raises its own
`oom_score_adj` so the kernel picks it rather than the parent, which does no
data work and must survive to write the record.

`pipeline_completed_int` is emitted on **every** run, `0.0` included.
`check_absolute_floors` skips a triple whose result is missing but treats a
result that exists with the metric absent as a violation, so an omitted metric
and a `0.0` are not interchangeable — and a cell that simply vanishes reads
exactly like coverage.

The ceiling has three spellings and they are kept apart, because only one of
them is the claim. A cgroup SIGKILL is `oom_killed`. An allocation that merely
*fails* — MemoryError, or `OSError(ENOMEM)` — is also `oom_killed`, reported by
the worker itself so it names the stage. Running out of wall clock is
`timeout`, and Leiden collapsing to one group (which leaves DE unable to run)
is `degenerate_clustering`; all three give `0.0`, and conflating them would
publish "scanpy hit the ceiling" for a run that did something else. Anything
else is a real bug and fails the cell rather than being laundered into a
result.

Verified end to end on pbmc3k off SLURM, via the
`SCX_BENCH_PIPELINE_MEM_LIMIT_GB` hatch: at a 2 GB ceiling the scanpy arm dies
in the `neighbors` stage and records `pipeline_completed_int = 0.0` with six
stage records recovered; at 3 GB it completes at a 977 MB peak. The pyscx arm
completes all nine stages at a 730 MB peak.

That 2 GB run also shows why the hatch needs its own classifier: under a hard
`RLIMIT_DATA` OpenBLAS aborts the process (exit -6) rather than letting Python
raise, so there is no signal to read. The stderr text match that covers it is
enabled **only** under the hatch — on a real capture the cgroup sends SIGKILL,
which needs no guessing, and a parent that classifies on a string eventually
relabels every unrelated failure carrying it.

## Stage order

The spec's §4.2 lists HVG (`flavor="seurat_v3"`) *after* normalize+log1p.
`seurat_v3` expects raw counts — it is a variance-stabilising rank statistic on
the count distribution — so running it on log-normalised values computes a
different thing. The fixtures carry no counts layer to point `layer=` at, and
materialising one would double the footprint under the very ceiling being
tested.

So HVG moves ahead of normalisation and is run with `subset=False`: it marks
`var["highly_variable"]` on raw counts, normalisation then runs over the full
gene axis (not over a 2000-gene subset, which would change every cell's size
factor), and PCA picks the mask up on its own — `pyscx.accel.pca(mask_var=None)`
auto-consumes `var["highly_variable"]`, and scanpy takes the same restriction
via `mask_var="highly_variable"`. Both arms run the identical order; the count
of stages is unchanged.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    QUERY_N_HVGS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import (
    BenchmarkResult,
    require_runs,
    write_missing_result,
)
from benchmarks.comprehensive.rss import PeakRssSampler, current_rss_mb
from benchmarks.comprehensive.subproc_arm import run_arm

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "pipeline_ooc_constrained__pyscx_16g",
    "pipeline_ooc_constrained__pyscx_32g",
    "pipeline_ooc_constrained__scanpy_16g",
    "pipeline_ooc_constrained__scanpy_32g",
})

# The memory ceiling each arm runs under, in GB, read by
# `config.estimate_memory_gb` so the SLURM allocation *is* the experiment.
# Declared here rather than parsed from the key suffix at the call site: the
# budget is this benchmark's property, and config.py should not have to know
# the spelling of an arm.
MEMORY_BUDGET_GB: dict[str, int] = {
    "pipeline_ooc_constrained__pyscx_16g": 16,
    "pipeline_ooc_constrained__pyscx_32g": 32,
    "pipeline_ooc_constrained__scanpy_16g": 16,
    "pipeline_ooc_constrained__scanpy_32g": 32,
}

#: Datasets on which a fixed memory ceiling is a meaningful experiment.
#:
#: `run_parallel` filters this at cohort-build time, which is the point: the
#: alternative — stubbing an out-of-scope dataset inside `run()` — is a typed
#: result, but only after Chimera has started a task, activated conda and
#: imported pyscx. pbmc3k and smartseq2 are out because a 16 GB ceiling on
#: 2,700 cells measures nothing; pbmc10k stays as the cheap arm that still
#: exercises all nine stages end to end.
_SCOPED_DATASETS = frozenset({
    "pbmc10k", "tabula_sapiens_100k", "census_500k", "census_1m",
})
FORMAT_DATASET_SCOPE: dict[str, frozenset[str]] = {
    key: _SCOPED_DATASETS for key in SUPPORTED_FORMATS
}

#: The nine stages, in execution order. The names are the suffixes of the
#: `stage_wall_s__<stage>` / `stage_peak_rss_mb__<stage>` metric keys, and the
#: order is what lets a killed worker's partial trail name the stage it died
#: in — the first name with no completion line.
STAGES: tuple[str, ...] = (
    "load",
    "qc_metrics",
    "filter",
    "hvg",
    "normalize_log1p",
    "pca",
    "neighbors",
    "umap_leiden",
    "rank_genes_groups",
)

N_COMPS = 50
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 1.0
UMAP_MIN_DIST = 0.5
UMAP_SPREAD = 1.0
MIN_GENES = 200
MIN_CELLS = 3

#: Outcome vocabulary for `outcome_reason`. Kept distinct so "hit the ceiling"
#: is never confused with "ran out of wall clock" or "clustered into one group"
#: — all three give `pipeline_completed_int = 0.0`, and only the first is the
#: result this benchmark is making a claim about.
OUTCOME_COMPLETED = "completed"
OUTCOME_OOM = "oom_killed"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_DEGENERATE = "degenerate_clustering"
OUTCOME_FAILED = "failed"

#: Fallback when the per-job SLURM budget cannot be read. The worker timeout
#: should sit *under* the job's own limit: if SLURM kills the job instead, the
#: parent never runs and the cell is lost rather than recording a partial.
_DEFAULT_TIMEOUT_S = 10 * 3600
_TIMEOUT_MARGIN_S = 600


def pipeline_ooc_constrained_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx backed streaming pipeline (16 GB budget)",
            key="pipeline_ooc_constrained__pyscx_16g",
            category="accel", runner="accel_runner",
            params={"engine": "pyscx", "budget_gb": 16},
        ),
        FormatVariant(
            name="pyscx backed streaming pipeline (32 GB budget)",
            key="pipeline_ooc_constrained__pyscx_32g",
            category="accel", runner="accel_runner",
            params={"engine": "pyscx", "budget_gb": 32},
        ),
        FormatVariant(
            name="scanpy in-memory pipeline (16 GB budget)",
            key="pipeline_ooc_constrained__scanpy_16g",
            category="accel", runner="accel_runner",
            params={"engine": "scanpy", "budget_gb": 16},
        ),
        FormatVariant(
            name="scanpy in-memory pipeline (32 GB budget)",
            key="pipeline_ooc_constrained__scanpy_32g",
            category="accel", runner="accel_runner",
            params={"engine": "scanpy", "budget_gb": 32},
        ),
    ]


def engine_for(key: str) -> str:
    return "pyscx" if "__pyscx_" in key else "scanpy"


# ---------------------------------------------------------------------------
# The pipeline itself. Imported by the worker, so it stays testable in-process.
# ---------------------------------------------------------------------------

class StageReporter:
    """Prints one JSON line per completed stage, immediately.

    The flush is the whole mechanism. Buffered output is lost to SIGKILL, and
    with it the only evidence of how far the arm got — which for the scanpy
    arms is the measurement.
    """

    def __init__(self, emit=print) -> None:
        self._emit = emit
        self.index = 0

    def stage(self, name: str, fn) -> Any:
        import time

        idx = self.index
        self.index += 1
        rss_before = current_rss_mb()
        with PeakRssSampler() as sampler:
            t0 = time.perf_counter()
            value = fn()
            wall = time.perf_counter() - t0
        self._emit(json.dumps({
            "kind": "stage",
            "name": name,
            "index": idx,
            "wall_s": wall,
            "peak_rss_mb": sampler.peak_mb,
            "rss_before_mb": rss_before,
            "rss_after_mb": current_rss_mb(),
        }), flush=True)
        return value


def run_pipeline(engine: str, path: Path, reporter: StageReporter) -> dict[str, Any]:
    """The nine stages, in order, reporting each as it completes.

    Raises `DegenerateClustering` when Leiden yields fewer than two groups:
    `rank_genes_groups` cannot run, and recording the pipeline as complete
    without its last stage would be a claim the run did not earn.
    """
    import time

    import numpy as np

    summary: dict[str, Any] = {"engine": engine}
    t_all = time.perf_counter()

    if engine == "pyscx":
        import pyscx

        adata = reporter.stage(
            "load", lambda: pyscx.open(str(path)).to_anndata(backed=True),
        )
        summary["x_type_after_load"] = type(adata.X).__name__
        reporter.stage("qc_metrics", lambda: pyscx.accel.calculate_qc_metrics(
            adata, qc_vars=None, inplace=True,
        ))

        def _filter() -> None:
            pyscx.accel.filter_cells(adata, min_genes=MIN_GENES)
            pyscx.accel.filter_genes(adata, min_cells=MIN_CELLS)

        reporter.stage("filter", _filter)
        reporter.stage("hvg", lambda: pyscx.accel.highly_variable_genes(
            adata, n_top_genes=min(QUERY_N_HVGS, int(adata.n_vars)),
            flavor="seurat_v3", subset=False,
            span=0.3 if adata.n_obs < 10_000 else 1.0, device="cpu",
        ))

        def _norm() -> None:
            pyscx.accel.normalize_total(adata, target_sum=1e4)
            pyscx.accel.log1p(adata)

        reporter.stage("normalize_log1p", _norm)
        summary["x_type_after_normalize"] = type(adata.X).__name__
        # mask_var is left at None on purpose: pyscx auto-consumes
        # var["highly_variable"], which is the restriction the hvg stage set.
        reporter.stage("pca", lambda: pyscx.accel.pca(
            adata, n_comps=N_COMPS, random_state=RANDOM_SEED, device="cpu",
        ))
        reporter.stage("neighbors", lambda: pyscx.accel.neighbors(
            adata, n_neighbors=N_NEIGHBORS, random_state=RANDOM_SEED, device="cpu",
        ))

        def _embed() -> None:
            pyscx.accel.umap(
                adata, min_dist=UMAP_MIN_DIST, spread=UMAP_SPREAD,
                random_state=RANDOM_SEED, device="cpu",
            )
            pyscx.accel.leiden(
                adata, resolution=LEIDEN_RESOLUTION, key_added="leiden",
                random_state=RANDOM_SEED, device="cpu",
            )

        reporter.stage("umap_leiden", _embed)
        n_clusters = int(adata.obs["leiden"].nunique())
        summary["n_clusters"] = n_clusters
        if n_clusters < 2:
            raise DegenerateClustering(n_clusters)
        reporter.stage("rank_genes_groups", lambda: pyscx.accel.rank_genes_groups(
            adata, groupby="leiden", method="wilcoxon", device="cpu",
        ))
    else:
        import anndata
        import scanpy as sc

        adata = reporter.stage("load", lambda: anndata.read_h5ad(str(path)))
        summary["x_type_after_load"] = type(adata.X).__name__
        reporter.stage("qc_metrics", lambda: sc.pp.calculate_qc_metrics(
            adata, inplace=True, percent_top=None,
        ))

        def _filter() -> None:
            sc.pp.filter_cells(adata, min_genes=MIN_GENES)
            sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

        reporter.stage("filter", _filter)
        reporter.stage("hvg", lambda: sc.pp.highly_variable_genes(
            adata, n_top_genes=min(QUERY_N_HVGS, int(adata.n_vars)),
            flavor="seurat_v3", subset=False,
            span=0.3 if adata.n_obs < 10_000 else 1.0,
        ))

        def _norm() -> None:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        reporter.stage("normalize_log1p", _norm)
        reporter.stage("pca", lambda: sc.pp.pca(
            adata, n_comps=N_COMPS, random_state=RANDOM_SEED,
            mask_var="highly_variable",
        ))
        reporter.stage("neighbors", lambda: sc.pp.neighbors(
            adata, n_neighbors=N_NEIGHBORS, random_state=RANDOM_SEED,
        ))

        def _embed() -> None:
            sc.tl.umap(
                adata, min_dist=UMAP_MIN_DIST, spread=UMAP_SPREAD,
                random_state=RANDOM_SEED,
            )
            sc.tl.leiden(
                adata, resolution=LEIDEN_RESOLUTION, key_added="leiden",
                random_state=RANDOM_SEED, flavor="igraph", n_iterations=2,
                directed=False,
            )

        reporter.stage("umap_leiden", _embed)
        n_clusters = int(adata.obs["leiden"].nunique())
        summary["n_clusters"] = n_clusters
        if n_clusters < 2:
            raise DegenerateClustering(n_clusters)
        reporter.stage("rank_genes_groups", lambda: sc.tl.rank_genes_groups(
            adata, groupby="leiden", method="wilcoxon",
        ))

    summary["total_wall_s"] = time.perf_counter() - t_all
    summary["n_obs_final"] = int(adata.n_obs)
    summary["n_vars_final"] = int(adata.n_vars)
    summary["n_hvg"] = int(np.asarray(adata.var["highly_variable"]).sum())
    del adata
    return summary


class DegenerateClustering(RuntimeError):
    """Leiden produced fewer than two groups, so DE cannot run."""

    def __init__(self, n_clusters: int) -> None:
        super().__init__(f"leiden produced {n_clusters} cluster(s); DE needs >= 2")
        self.n_clusters = n_clusters


_WORKER_SCRIPT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path

    engine, path = sys.argv[1:3]

    from benchmarks.comprehensive.benchmarks.pipeline_ooc_constrained import (
        DegenerateClustering, StageReporter, run_pipeline,
    )

    reporter = StageReporter()
    try:
        summary = run_pipeline(engine, Path(path), reporter)
    except DegenerateClustering as exc:
        print(json.dumps({"kind": "degenerate", "n_clusters": exc.n_clusters}),
              flush=True)
        raise SystemExit(4)
    except MemoryError as exc:
        # The ceiling does not always arrive as SIGKILL. Under an explicit
        # RLIMIT_DATA, and on some cgroup paths, the allocation simply fails
        # and Python raises here. Same outcome, different spelling.
        print(json.dumps({"kind": "oom", "stage_index": reporter.index,
                          "how": "MemoryError", "detail": str(exc)[:200]}),
              flush=True)
        raise SystemExit(5)
    except OSError as exc:
        import errno as _errno

        if exc.errno != _errno.ENOMEM:
            raise
        print(json.dumps({"kind": "oom", "stage_index": reporter.index,
                          "how": "ENOMEM", "detail": str(exc)[:200]}),
              flush=True)
        raise SystemExit(5)
    summary["kind"] = "done"
    print(json.dumps(summary), flush=True)
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _local_mem_limit_bytes() -> int | None:
    """`SCX_BENCH_PIPELINE_MEM_LIMIT_GB`, for reproducing the regime off SLURM.

    Returns `None` on a normal capture. See `run()` for why this is not the
    default, and `subproc_arm` for why it is RLIMIT_DATA and never RLIMIT_AS.
    """
    import os

    raw = os.environ.get("SCX_BENCH_PIPELINE_MEM_LIMIT_GB", "").strip()
    if not raw:
        return None
    try:
        return int(float(raw) * 1024**3)
    except ValueError:
        logger.warning(
            "pipeline_ooc_constrained: ignoring unparseable "
            "SCX_BENCH_PIPELINE_MEM_LIMIT_GB=%r", raw,
        )
        return None


def _timeout_for(dataset: DatasetConfig, key: str) -> int:
    """Stay under the SLURM job's own wall limit so a partial is still recorded.

    If SLURM kills the job, the parent never runs and the cell is lost — which
    reads as "not run" and is skipped by the gate in silence. A worker timeout
    that fires first leaves a partial trail and a recorded `0.0`.
    """
    try:
        from benchmarks.comprehensive.config import estimate_time_minutes

        minutes = estimate_time_minutes(dataset, key, "pipeline_ooc_constrained")
        return max(600, int(minutes) * 60 - _TIMEOUT_MARGIN_S)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pipeline_ooc_constrained: time estimate unavailable (%s)", exc)
        return _DEFAULT_TIMEOUT_S


#: Signatures a native library prints when an allocation fails hard.
#:
#: Consulted **only** under the local `SCX_BENCH_PIPELINE_MEM_LIMIT_GB` hatch,
#: never on a real capture. Under a hard RLIMIT, OpenBLAS and friends abort the
#: process with exit 1 and a message rather than letting Python raise
#: MemoryError — observed as `OpenBLAS error: Memory allocation still failed
#: after 10 retries, giving up.` So the hatch needs a text match to be usable
#: at all.
#:
#: It stays out of the capture path on purpose. `conversion_streaming` learned
#: this the hard way: once the parent classifies on a string, every unrelated
#: failure that happens to carry it is silently relabelled as the expected
#: outcome. On SLURM the ceiling arrives as SIGKILL, which needs no guessing.
_ALLOCATION_ABORT_SIGNATURES = (
    "memory allocation",
    "cannot allocate memory",
    "std::bad_alloc",
    "out of memory",
)


def _looks_like_an_allocation_abort(stderr: str) -> bool:
    low = stderr.lower()
    return any(sig in low for sig in _ALLOCATION_ABORT_SIGNATURES)


def summarize_outcome(
    outcome, budget_gb: int, *, trust_stderr: bool = False,
) -> tuple[dict[str, Any], str]:
    """Turn a worker outcome into the metric block, completed or not.

    Every return path sets `pipeline_completed_int`. The RSS figure on a kill
    is the budget rather than the last sample: the run is known to have reached
    the ceiling, and reporting the largest *observed* stage peak — or worse,
    0.0 — would let a `max: 15360` ceiling pass on a run that blew through it.
    """
    stages = [r for r in outcome.records if r.get("kind") == "stage"]
    done = next((r for r in outcome.records if r.get("kind") == "done"), None)
    degenerate = next(
        (r for r in outcome.records if r.get("kind") == "degenerate"), None
    )

    extras: dict[str, Any] = {}
    for rec in stages:
        name = rec.get("name")
        extras[f"stage_wall_s__{name}"] = float(rec.get("wall_s", float("nan")))
        extras[f"stage_peak_rss_mb__{name}"] = float(
            rec.get("peak_rss_mb", float("nan"))
        )
    seen = {r.get("name") for r in stages}
    first_missing = next((s for s in STAGES if s not in seen), None)
    observed_peak = max((float(r.get("peak_rss_mb", 0.0)) for r in stages), default=0.0)

    if done is not None:
        extras["pipeline_completed_int"] = 1.0
        extras["total_wall_s"] = float(done["total_wall_s"])
        extras["pipeline_peak_rss_mb"] = observed_peak
        extras["n_clusters"] = float(done.get("n_clusters", float("nan")))
        extras["n_obs_final"] = float(done.get("n_obs_final", float("nan")))
        extras["n_vars_final"] = float(done.get("n_vars_final", float("nan")))
        extras["outcome_reason"] = OUTCOME_COMPLETED
        extras["failed_stage"] = ""
        return extras, OUTCOME_COMPLETED

    extras["pipeline_completed_int"] = 0.0
    extras["total_wall_s"] = float(outcome.wall_s)
    extras["failed_stage"] = first_missing or STAGES[-1]
    extras["oom_at_stage_index"] = float(len(stages))

    reported_oom = next((r for r in outcome.records if r.get("kind") == "oom"), None)

    if reported_oom is not None:
        # The worker saw the allocation fail and said so, which is strictly
        # better evidence than inferring it from a signal — it names the stage
        # and the spelling (MemoryError vs ENOMEM).
        extras["outcome_reason"] = OUTCOME_OOM
        extras["oom_detected_as"] = reported_oom.get("how", "MemoryError")
        extras["pipeline_peak_rss_mb"] = float(budget_gb) * 1024.0
        idx = int(reported_oom.get("stage_index", len(stages)))
        extras["oom_at_stage_index"] = float(idx)
        extras["failed_stage"] = STAGES[min(idx, len(STAGES) - 1)]
        return extras, OUTCOME_OOM

    if degenerate is not None:
        reason = OUTCOME_DEGENERATE
        extras["n_clusters"] = float(degenerate.get("n_clusters", float("nan")))
        extras["pipeline_peak_rss_mb"] = observed_peak
    elif outcome.timed_out:
        reason = OUTCOME_TIMEOUT
        extras["pipeline_peak_rss_mb"] = observed_peak
    elif outcome.killed_by_oom:
        reason = OUTCOME_OOM
        # The ceiling, not the last sample: the sampler's final reading lands
        # whenever the poll happened to fire, and a SIGKILL gives no chance to
        # take one at the top. Reporting the budget is the one figure that is
        # certainly a lower bound on what the run demanded.
        extras["pipeline_peak_rss_mb"] = float(budget_gb) * 1024.0
    elif trust_stderr and _looks_like_an_allocation_abort(
        getattr(outcome, "stderr", "") or ""
    ):
        reason = OUTCOME_OOM
        extras["oom_detected_as"] = "native_abort"
        extras["pipeline_peak_rss_mb"] = float(budget_gb) * 1024.0
    else:
        reason = OUTCOME_FAILED
        extras["pipeline_peak_rss_mb"] = observed_peak
    extras["outcome_reason"] = reason
    return extras, reason


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in SUPPORTED_FORMATS:
        logger.warning("pipeline_ooc_constrained: unknown variant %s", key)
        return None
    engine = engine_for(key)
    budget_gb = MEMORY_BUDGET_GB[key]

    if engine == "pyscx":
        source_path = dataset.scx_auto_path
    else:
        source_path = Path(converted_path) if converted_path else dataset.h5ad_path
        if source_path.suffix != ".h5ad":
            source_path = dataset.h5ad_path
    if not source_path.exists():
        write_missing_result(
            benchmark="pipeline_ooc_constrained", format_key=key,
            dataset=dataset.name, missing_reason="fixture_missing",
            notes=f"{source_path} does not exist",
        )
        return None

    timeout_s = _timeout_for(dataset, key)
    result = BenchmarkResult(
        benchmark="pipeline_ooc_constrained",
        format=key,
        dataset=dataset.name,
        scenario={
            "name": "end_to_end_pipeline",
            "engine": engine,
            "budget_gb": budget_gb,
            "residency": "backed" if engine == "pyscx" else "in_memory",
            "cache_state": "cold" if cold_cache else "warm",
            "device": "cpu",
        },
        comparison={
            "subject": {"impl": key},
            "baseline": {"impl": f"pipeline_ooc_constrained__scanpy_{budget_gb}g"},
            "metric": "pipeline_completed_int",
            "status": "pending",
        },
        metadata={
            "cold_cache": cold_cache,
            # No warm-up: a warm-up run of a pipeline whose point is the peak
            # would leave the page cache holding the whole matrix, which is the
            # opposite of the constrained regime being measured — and on
            # census_1m it would double a multi-hour job.
            "n_warmup": 0,
            "budget_gb": budget_gb,
            "source_path": str(source_path),
            "stages": list(STAGES),
            "n_comps": N_COMPS,
            "n_neighbors": N_NEIGHBORS,
            "leiden_resolution": LEIDEN_RESOLUTION,
            "n_hvg_requested": QUERY_N_HVGS,
            "random_seed": RANDOM_SEED,
            "worker_timeout_s": timeout_s,
            "stage_order_note": (
                "seurat_v3 HVG runs on raw counts before normalisation and "
                "marks var['highly_variable'] without subsetting; PCA takes "
                "the mask. Both arms run this identical order."
            ),
        },
    )

    for i in range(n_runs):
        if cold_cache:
            from benchmarks.comprehensive.cache_control import drop_file_cache

            drop_file_cache(source_path)
        outcome = run_arm(
            _WORKER_SCRIPT, [engine, str(source_path)],
            timeout_s=timeout_s,
            # Off SLURM there is no cgroup, so the ceiling has to come from
            # somewhere. RLIMIT_DATA is the local stand-in and is opt-in:
            # `SCX_BENCH_PIPELINE_MEM_LIMIT_GB=1`. It is never set by default
            # because on a real capture the cgroup already is the ceiling, and
            # a second, differently-accounted limit would make the number
            # describe neither.
            mem_limit_bytes=_local_mem_limit_bytes(),
            # The child volunteers as the OOM killer's first pick. Without it
            # the kernel may choose this parent — which does no data work and
            # has to survive to record the refusal — and the cell is lost
            # instead of yielding the measurement.
            oom_first=True,
            label=f"{key} run {i + 1}",
        )
        extras, reason = summarize_outcome(
            outcome, budget_gb,
            # Only the local hatch may classify on stderr; see
            # `_ALLOCATION_ABORT_SIGNATURES`.
            trust_stderr=_local_mem_limit_bytes() is not None,
        )
        if reason == OUTCOME_FAILED:
            # Not an OOM, not a timeout, not a degenerate clustering: a real
            # bug. Publishing it as `pipeline_completed_int = 0.0` would claim
            # the memory ceiling was hit when it was not.
            raise RuntimeError(
                outcome.failure_text(f"pipeline_ooc_constrained {key} run {i + 1}")
            )
        result.add_run(
            wall_s=float(extras["total_wall_s"]),
            peak_rss_mb=float(extras["pipeline_peak_rss_mb"]),
            **extras,
        )
        logger.info(
            "  %s run %d: %s in %.1fs (peak %.0f MB, budget %d GB)%s",
            key, i + 1, reason, extras["total_wall_s"],
            extras["pipeline_peak_rss_mb"], budget_gb,
            "" if reason == OUTCOME_COMPLETED
            else f" — stopped at stage '{extras['failed_stage']}'",
        )
        for stage in STAGES:
            w = extras.get(f"stage_wall_s__{stage}")
            if w is not None:
                logger.info(
                    "      %-20s %8.2fs  peak %8.0f MB", stage, w,
                    extras.get(f"stage_peak_rss_mb__{stage}", float("nan")),
                )

    require_runs(result, str(source_path))
    completed = [r.extra["pipeline_completed_int"] for r in result.runs]
    result.metadata["pipeline_completed_int"] = min(completed)
    result.metadata["outcome_reasons"] = sorted(
        {r.extra["outcome_reason"] for r in result.runs}
    )
    result.overall_passed = bool(min(completed) >= 1.0)
    logger.info(
        "pipeline_ooc_constrained complete: %s / %s — completed=%s, reasons=%s",
        key, dataset.name, min(completed), result.metadata["outcome_reasons"],
    )
    return result
