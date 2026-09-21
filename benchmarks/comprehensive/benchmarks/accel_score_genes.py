"""Streaming gene-set scoring, pyscx vs `scanpy.tl.score_genes`.

`sc.tl.score_genes` bins genes by expression, samples control genes per bin and
subtracts the control mean. It **raises `NotImplementedError` on a backed
AnnData**, because its slicing and binning need dense or eager CSR indexing — so
the operation an analyst reaches for on every marker panel is exactly the one
that forces a full materialisation. `pyscx.accel.score_genes` runs the same
arithmetic over the `for_each_shard_ordered` decode-prefetch engine and works on
a backed `X`. That asymmetry is the benchmark's claim, not a flaw in it.

Three scoring flavours, because they are three different asks: `method="control"`
(scanpy's algorithm, and the parity arm), `method="mean"` (light marker score)
and `method="zscore"` (decoupleR-equivalent standardised score). Note the
spelling: there is no `method="scanpy"` — `"control"` *is* the scanpy method and
is the default.

Parity requires handing SCX scanpy's own control genes. SCX's sampler is
deterministic but is not numpy's, so it draws a different control set and the
scores differ by ~10 % of their range at the defaults; a Spearman floor over two
different control sets measures nothing in particular. The wrinkle is that
**scanpy does not expose the set it drew** — `sc.tl.score_genes` only logs how
many, and writes nothing to `uns` (verified against scanpy 1.12: `uns` is empty
afterwards). So the reference child reaches into two private helpers,
`_check_score_genes_args` and `_score_genes_bins`, exactly as
`pyscx/tests/test_accel_score_genes.py` already does. If either moves, this
records a typed `scanpy_private_api_drift` gap rather than falling back to SCX's
own sampler — a fallback would leave `score_spearman_vs_scanpy` reading near 1.0
while comparing a run against itself.

No normalisation is applied. The fixtures store raw counts and every arm scores
what is stored, so the arms differ only in the scoring kernel. Folding a
normalize+log1p in would measure preprocessing — where the backed arm has a
separate, larger advantage — and attribute it to `score_genes`.

Panels are derived from the dataset rather than hard-coded: genes are ranked by
total counts, the top `PANEL_POOL` taken, and each `K` drawn by a fixed stride
through that pool. A stride rather than a head, so a K=500 panel spans the
expression range instead of being 500 near-identical ceiling genes — the control
binning is expression-matched, and a panel concentrated in one bin exercises one
bin.

Process isolation: one fresh child per run. See
`benchmarks/comprehensive/subproc_arm.py` for why a parent-side `PeakRssSampler`
cannot measure a child, and why two arms in one interpreter contaminate each
other's peak.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
)
from benchmarks.comprehensive.results import (
    BenchmarkResult,
    require_runs,
    write_missing_result,
)
from benchmarks.comprehensive.rss import PeakRssSampler
from benchmarks.comprehensive.subproc_arm import run_arm

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "accel_score_genes__pyscx_cpu_scanpy",
    "accel_score_genes__pyscx_cpu_mean",
    "accel_score_genes__pyscx_cpu_zscore",
    "accel_score_genes__scanpy_cpu",
})

#: `method` is the pyscx spelling; `"control"` is scanpy's algorithm. `parity`
#: marks the one arm whose scores are comparable to the reference — `mean` and
#: `zscore` are different statistics and have no scanpy counterpart, so they
#: carry throughput only.
_ARMS: dict[str, dict[str, Any]] = {
    "accel_score_genes__pyscx_cpu_scanpy": {
        "engine": "pyscx", "method": "control", "source": "scx", "parity": True,
    },
    "accel_score_genes__pyscx_cpu_mean": {
        "engine": "pyscx", "method": "mean", "source": "scx", "parity": False,
    },
    "accel_score_genes__pyscx_cpu_zscore": {
        "engine": "pyscx", "method": "zscore", "source": "scx", "parity": False,
    },
    "accel_score_genes__scanpy_cpu": {
        "engine": "scanpy", "method": "control", "source": "h5ad", "parity": False,
    },
}

PANEL_SIZES: tuple[int, ...] = (25, 100, 500)
#: Genes the panels are drawn from, ranked by total counts.
PANEL_POOL = 2000
CTRL_SIZE = 50
N_BINS = 25
SCORE_RANDOM_STATE = 0

_ARM_TIMEOUT_S = 4 * 3600


def accel_score_genes_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx score_genes (backed, scanpy parity)",
            key="accel_score_genes__pyscx_cpu_scanpy",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx score_genes (backed, mean)",
            key="accel_score_genes__pyscx_cpu_mean",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx score_genes (backed, zscore)",
            key="accel_score_genes__pyscx_cpu_zscore",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="scanpy score_genes (in-memory h5ad)",
            key="accel_score_genes__scanpy_cpu",
            category="accel", runner="accel_runner",
        ),
    ]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def gene_panels(
    var_names: Any, totals: np.ndarray, sizes: tuple[int, ...] = PANEL_SIZES,
) -> dict[int, list[str]]:
    """Deterministic marker panels of each size in *sizes*.

    Rank by total counts, keep the top `PANEL_POOL`, then take each K by a
    stride through that pool. The stride is the point: a head would give a K=500
    panel of the 500 highest-expressed genes, which all land in the same
    expression bin, so the expression-matched control sampling every arm is
    exercising would only ever see one bin.

    Ties are broken by gene index (`kind="stable"`), so the panel does not
    depend on the input order of equal-valued genes.
    """
    names = np.asarray([str(v) for v in var_names])
    totals = np.asarray(totals, dtype=np.float64).ravel()
    if names.shape[0] != totals.shape[0]:
        raise ValueError(
            f"var_names ({names.shape[0]}) and totals ({totals.shape[0]}) disagree"
        )
    order = np.argsort(-totals, kind="stable")
    pool = order[: min(PANEL_POOL, order.shape[0])]
    out: dict[int, list[str]] = {}
    for k in sizes:
        k_eff = min(k, pool.shape[0])
        if k_eff == 0:
            out[k] = []
            continue
        stride = max(1, pool.shape[0] // k_eff)
        picked = pool[::stride][:k_eff]
        out[k] = [str(n) for n in names[picked]]
    return out


def _reference_dir() -> Path:
    base = Path(os.environ.get("SCX_BENCH_TMPDIR") or os.environ.get("SCX_WORK_DIR", ""))
    root = base if base.is_dir() else Path("/tmp")
    return root / "scx_bench_score_reference"


def _reference_tag(dataset: DatasetConfig) -> str:
    """Identity of everything that changes the panels, controls or reference.

    A cached artefact built under a different `ctrl_size`, `n_bins` or panel
    rule is not a reference for this run; reusing one would make
    `score_spearman_vs_scanpy` describe a comparison nobody configured, and it
    would read as a pass.
    """
    payload = json.dumps(
        {
            "dataset": dataset.name,
            "sizes": list(PANEL_SIZES),
            "pool": PANEL_POOL,
            "ctrl_size": CTRL_SIZE,
            "n_bins": N_BINS,
            "random_state": SCORE_RANDOM_STATE,
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Reference: panels + scanpy's own control sets + scanpy's scores. Untimed.
# ---------------------------------------------------------------------------

def build_reference(h5ad_path: Path, out_json: Path, out_npz: Path) -> dict[str, Any]:
    """Everything the timed arms need, computed once per dataset, untimed.

    One eager read produces all three artefacts: the per-gene totals the panels
    are ranked by, the control set scanpy would draw for each panel, and
    scanpy's own scores for the parity comparison. Splitting them would mean
    three eager reads of an 11 GB h5ad for no gain.
    """
    import anndata
    import pandas as pd
    import scanpy as sc
    from scanpy.tools._score_genes import (  # noqa: F401  (import is the probe)
        _check_score_genes_args,
        _score_genes_bins,
    )

    adata = anndata.read_h5ad(str(h5ad_path))
    totals = np.asarray(adata.X.sum(axis=0), dtype=np.float64).ravel()
    panels = gene_panels(adata.var_names, totals)

    controls: dict[str, list[str]] = {}
    scores: dict[str, np.ndarray] = {}
    for k, genes in panels.items():
        if not genes:
            continue
        # Exactly what `score_genes` does before sampling: seed numpy's global
        # RNG, then walk the expression bins. There is no public accessor —
        # `score_genes` logs the count and discards the set.
        np.random.seed(SCORE_RANDOM_STATE)
        gl, pool, get_subset = _check_score_genes_args(
            adata, genes, None, use_raw=False, layer=None,
        )
        control = pd.Index([], dtype="string")
        for r_genes in _score_genes_bins(
            gl, pool, ctrl_as_ref=True, ctrl_size=CTRL_SIZE,
            n_bins=N_BINS, get_subset=get_subset,
        ):
            control = control.union(r_genes)
        controls[str(k)] = [str(g) for g in control]

        sc.tl.score_genes(
            adata, genes, ctrl_size=CTRL_SIZE, n_bins=N_BINS,
            random_state=SCORE_RANDOM_STATE, score_name=f"ref_score_k{k}",
            use_raw=False,
        )
        scores[f"k{k}"] = np.asarray(
            adata.obs[f"ref_score_k{k}"].to_numpy(), dtype=np.float64,
        )

    meta = {
        "panels": {str(k): v for k, v in panels.items()},
        "controls": controls,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "ctrl_size": CTRL_SIZE,
        "n_bins": N_BINS,
        "random_state": SCORE_RANDOM_STATE,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2))
    np.savez_compressed(out_npz, **scores)
    del adata
    return meta


_REFERENCE_SCRIPT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path

    h5ad_path, out_json, out_npz = sys.argv[1:4]
    from benchmarks.comprehensive.benchmarks.accel_score_genes import build_reference

    meta = build_reference(Path(h5ad_path), Path(out_json), Path(out_npz))
    print(json.dumps({"n_panels": len(meta["panels"]),
                      "n_controls": {k: len(v) for k, v in meta["controls"].items()}}),
          flush=True)
""")


def ensure_reference(dataset: DatasetConfig) -> tuple[Path, Path] | None:
    """Panels + controls + scanpy scores for *dataset*, cached, untimed.

    Returns `None` when the source h5ad is absent or scanpy's private
    control-selection helpers have moved. Both are recorded by the caller as a
    typed missing result rather than silently degraded: an arm that fell back to
    SCX's own sampler would publish a near-perfect Spearman against itself.
    """
    h5ad = dataset.h5ad_path
    if not h5ad.exists():
        logger.warning("accel_score_genes: no source h5ad at %s", h5ad)
        return None
    tag = _reference_tag(dataset)
    out_json = _reference_dir() / f"{dataset.name}.{tag}.json"
    out_npz = _reference_dir() / f"{dataset.name}.{tag}.npz"
    if out_json.exists() and out_npz.exists():
        logger.info("accel_score_genes: reusing reference %s", out_json)
        return out_json, out_npz
    out_json.parent.mkdir(parents=True, exist_ok=True)
    logger.info("accel_score_genes: building reference for %s (untimed)", dataset.name)
    outcome = run_arm(
        _REFERENCE_SCRIPT, [str(h5ad), str(out_json), str(out_npz)],
        timeout_s=_ARM_TIMEOUT_S, label="accel_score_genes reference",
    )
    if not outcome.ok or not out_json.exists() or not out_npz.exists():
        logger.warning(
            "accel_score_genes: reference build failed.\n%s",
            outcome.failure_text("accel_score_genes reference"),
        )
        return None
    return out_json, out_npz


def reference_failure_reason(outcome_stderr: str) -> str:
    """Distinguish scanpy API drift from an ordinary failure.

    Worth separating because the remedies differ: drift means updating the two
    private-helper names against a new scanpy, while anything else is a fixture
    or environment problem.
    """
    if "scanpy.tools._score_genes" in outcome_stderr or "_score_genes_bins" in outcome_stderr:
        return "scanpy_private_api_drift"
    return "score_reference_failed"


# ---------------------------------------------------------------------------
# The measured work
# ---------------------------------------------------------------------------

def run_arm_once(
    engine: str,
    method: str,
    source: str,
    path: Path,
    meta: dict[str, Any],
    *,
    dump_npz: Path | None = None,
) -> dict[str, Any]:
    """Score every panel size once, timed per size, in *this* process."""
    import time

    import anndata

    if source == "scx":
        import pyscx

        adata = pyscx.open(str(path)).to_anndata(backed=True)
    else:
        adata = anndata.read_h5ad(str(path))

    rec: dict[str, Any] = {
        "engine": engine, "method": method, "source": source,
        "x_type": type(adata.X).__name__,
        "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
    }
    scores: dict[str, np.ndarray] = {}

    with PeakRssSampler() as outer:
        t_all = time.perf_counter()
        for k_str, genes in sorted(meta["panels"].items(), key=lambda kv: int(kv[0])):
            k = int(k_str)
            if not genes:
                continue
            name = f"score_k{k}"
            with PeakRssSampler() as s:
                t0 = time.perf_counter()
                if engine == "pyscx":
                    import pyscx

                    kwargs: dict[str, Any] = {
                        "score_name": name, "method": method, "device": "cpu",
                    }
                    if method == "control":
                        # The parity route. `ctrl_genes=` rejects `gene_pool=`
                        # and ignores ctrl_size / n_bins / random_state, there
                        # being no sampling left to steer.
                        kwargs["ctrl_genes"] = meta["controls"][k_str]
                    pyscx.accel.score_genes(adata, genes, **kwargs)
                else:
                    import scanpy as sc

                    sc.tl.score_genes(
                        adata, genes, ctrl_size=CTRL_SIZE, n_bins=N_BINS,
                        random_state=SCORE_RANDOM_STATE, score_name=name,
                        use_raw=False,
                    )
                rec[f"score_wall_s__k{k}"] = time.perf_counter() - t0
            rec[f"score_peak_rss_mb__k{k}"] = s.peak_mb
            rec[f"score_cells_per_sec__k{k}"] = (
                adata.n_obs / rec[f"score_wall_s__k{k}"]
                if rec[f"score_wall_s__k{k}"] > 0 else float("nan")
            )
            scores[f"k{k}"] = np.asarray(adata.obs[name].to_numpy(), dtype=np.float64)
        rec["score_wall_s"] = time.perf_counter() - t_all
    rec["score_peak_rss_mb"] = outer.peak_mb

    if dump_npz is not None and scores:
        dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_npz, **scores)
    del adata
    return rec


_WORKER_SCRIPT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path

    engine, method, source, path, meta_json, dump = sys.argv[1:7]
    cold = sys.argv[7] == "1"

    from benchmarks.comprehensive.benchmarks.accel_score_genes import run_arm_once

    if cold:
        from benchmarks.comprehensive.cache_control import drop_file_cache
        policy = drop_file_cache(path)
    else:
        policy = "warm"

    meta = json.loads(Path(meta_json).read_text())
    rec = run_arm_once(engine, method, source, Path(path), meta,
                       dump_npz=Path(dump) if dump else None)
    rec["cache_policy"] = policy
    print(json.dumps(rec), flush=True)
""")


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, or NaN when it is undefined.

    Returned as NaN rather than 0.0 for a constant vector: a zero would read as
    a hard disagreement and fail a `min: 0.999` floor, when the truth is that
    the statistic does not exist. The caller drops NaN instead of recording it,
    so the gate sees a missing metric — which on a result that exists is a
    violation, i.e. still loud, but honestly labelled.
    """
    from scipy.stats import spearmanr

    if a.shape != b.shape or a.size < 2:
        return float("nan")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    rho = spearmanr(a, b).statistic
    return float(rho)


def parity_metrics(arm_npz: Path, ref_npz: Path) -> dict[str, float]:
    ours = np.load(arm_npz)
    ref = np.load(ref_npz)
    out: dict[str, float] = {}
    for key in ours.files:
        if key not in ref.files:
            continue
        a, b = ours[key], ref[key]
        if a.shape != b.shape:
            out[f"score_spearman_vs_scanpy__{key}"] = 0.0
            out[f"score_max_abs_diff__{key}"] = float("inf")
            continue
        rho = spearman(a, b)
        if not np.isnan(rho):
            out[f"score_spearman_vs_scanpy__{key}"] = rho
        both = np.isfinite(a) & np.isfinite(b)
        diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else float("inf")
        out[f"score_max_abs_diff__{key}"] = diff
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    key = format_variant.key
    arm = _ARMS.get(key)
    if arm is None:
        logger.warning("accel_score_genes: unknown variant %s — skipping", key)
        return None

    ref = ensure_reference(dataset)
    if ref is None:
        # Every arm needs the panels, not just the parity one — without them
        # the four arms would score different gene sets and their wall times
        # would not be comparable.
        write_missing_result(
            benchmark="accel_score_genes", format_key=key, dataset=dataset.name,
            missing_reason="score_reference_unavailable",
            notes="panels / scanpy control sets could not be built; see the job "
                  "log for whether the source h5ad is missing or scanpy's "
                  "private control-selection helpers moved",
        )
        return None
    meta_json, ref_npz = ref

    if arm["source"] == "scx":
        source_path = dataset.scx_auto_path
    else:
        source_path = Path(converted_path) if converted_path else dataset.h5ad_path
        if source_path.suffix != ".h5ad":
            source_path = dataset.h5ad_path
    if not source_path.exists():
        write_missing_result(
            benchmark="accel_score_genes", format_key=key, dataset=dataset.name,
            missing_reason="fixture_missing", notes=f"{source_path} does not exist",
        )
        return None

    meta = json.loads(meta_json.read_text())

    result = BenchmarkResult(
        benchmark="accel_score_genes",
        format=key,
        dataset=dataset.name,
        scenario={
            "name": "score_genes",
            "engine": arm["engine"],
            "method": arm["method"],
            "residency": "backed" if arm["source"] == "scx" else "in_memory",
            "cache_state": "cold" if cold_cache else "warm",
            "device": "cpu",
        },
        comparison={
            "subject": {"impl": key},
            "baseline": {"impl": "accel_score_genes__scanpy_cpu"},
            "metric": "score_wall_s",
            "status": "pending",
        },
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "source_path": str(source_path),
            "panel_sizes": list(PANEL_SIZES),
            "panel_pool": PANEL_POOL,
            "ctrl_size": CTRL_SIZE,
            "n_bins": N_BINS,
            # The panels themselves, so a reader can audit what was scored
            # without re-deriving it from a cache that may have been rebuilt.
            "panels": {k: v for k, v in meta["panels"].items()},
            "n_control_genes": {k: len(v) for k, v in meta["controls"].items()},
            "normalization": "none (raw stored values on every arm)",
        },
    )

    dump_path = _reference_dir() / "arms" / f"{key}__{dataset.name}.npz"
    argv_base = [
        arm["engine"], arm["method"], arm["source"], str(source_path), str(meta_json),
    ]

    if not cold_cache:
        for i in range(N_WARMUP_RUNS):
            logger.info("accel_score_genes warm-up %d/%d for %s", i + 1, N_WARMUP_RUNS, key)
            run_arm(_WORKER_SCRIPT, [*argv_base, "", "0"],
                    timeout_s=_ARM_TIMEOUT_S, label=f"{key} warmup")

    for i in range(n_runs):
        outcome = run_arm(
            _WORKER_SCRIPT,
            [*argv_base, str(dump_path) if arm["parity"] else "",
             "1" if cold_cache else "0"],
            timeout_s=_ARM_TIMEOUT_S, label=f"{key} run {i + 1}",
        )
        if not outcome.ok or not outcome.records:
            raise RuntimeError(outcome.failure_text(f"accel_score_genes {key} run {i + 1}"))
        rec = outcome.records[-1]

        extras = {
            k: v for k, v in rec.items()
            if k.startswith(("score_wall_s", "score_peak_rss_mb", "score_cells_per_sec"))
        }
        extras["cache_policy"] = rec.get("cache_policy", "warm")
        extras["x_type"] = rec.get("x_type")
        if arm["parity"] and dump_path.exists():
            try:
                extras.update(parity_metrics(dump_path, ref_npz))
            except Exception as exc:  # noqa: BLE001
                logger.warning("accel_score_genes: parity failed for %s: %s", key, exc)

        result.add_run(
            wall_s=float(rec["score_wall_s"]),
            peak_rss_mb=float(rec["score_peak_rss_mb"]),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs peak=%.1fMB  %s",
            key, i + 1, rec["score_wall_s"], rec["score_peak_rss_mb"],
            " ".join(
                f"k{k}={rec.get(f'score_wall_s__k{k}', float('nan')):.3f}s"
                for k in PANEL_SIZES
            ),
        )

    require_runs(result, str(source_path))

    if arm["parity"]:
        for k in PANEL_SIZES:
            vals = [
                r.extra.get(f"score_spearman_vs_scanpy__k{k}") for r in result.runs
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                result.metadata[f"score_spearman_vs_scanpy__k{k}"] = round(
                    float(np.median(vals)), 6,
                )
        rhos = [
            v for k in PANEL_SIZES
            for v in [result.metadata.get(f"score_spearman_vs_scanpy__k{k}")]
            if v is not None
        ]
        if rhos:
            result.overall_passed = bool(min(rhos) >= 0.999)

    logger.info(
        "accel_score_genes complete: %s / %s — median %.3fs, spearman=%s",
        key, dataset.name, result.median_wall_s or 0.0,
        {k: result.metadata.get(f"score_spearman_vs_scanpy__k{k}") for k in PANEL_SIZES},
    )
    return result
