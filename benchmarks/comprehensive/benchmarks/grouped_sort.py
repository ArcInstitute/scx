"""Grouped-sharding benchmark — ``scx sort --group-by`` / ``scx convert --group-by``.

Measures the F1/F2 grouped-sort + Phase-7.4 convert-time grouping feature. For
every grouping dataset (those present in :data:`GROUP_SPEC`) it times three
scenarios and runs a read-back correctness check:

  * ``sort_group``  — ``pyscx.sort(plain_scx, group_by=, reference=)``: re-shard an
    already-converted ``.scx`` into the reference-first / group-clustered layout.
  * ``convert_one`` — ``pyscx.from_h5ad(group_by=, group_pass="one")``: convert-time
    grouping forced down the one-pass streaming gather.
  * ``convert_two`` — ``pyscx.from_h5ad(group_by=, group_pass="two")``: convert-time
    grouping forced down the two-pass (plain convert + sort) path.

The ``convert_one`` vs ``convert_two`` pair is the density auto-routing story
(CSR favours one-pass, dense favours two-pass). The read-back asserts
``read_group(label)`` partitions the obs axis and ``read_reference()`` is
isolated when a reference label is configured.

This is an **SCX-only** benchmark (``SUPPORTED_FORMATS = {"scx_auto"}``); every
other format variant returns ``None`` so the orchestrator skips it (mirrors
``fragment_ops.py`` / ``correctness.py``). Datasets absent from ``GROUP_SPEC``
also return ``None``.

The gateable memory metric is ``grouped_peak_rss_mb`` (plus sparse
``peak_rss_mb__<scenario>`` keys). It is **not** spelled ``peak_rss_mb``,
because that is a reserved ``add_run`` parameter: it lands on the ``RunRecord``
and never in ``runs[].extra``, which is the only place a threshold can read.
The three ``thresholds.yaml`` ceilings were keyed to the bare name and could
therefore never resolve a value — and, because all three of their datasets sit
outside every tier, the resulting missing-metric violation was never reached
either. Two silences stacked into one that looked like coverage.

``peak_rss_mb`` is a **true in-region peak**, sampled by ``PeakRssSampler`` on a
background thread for the duration of each op. It was an end-of-op
``current_rss_mb()`` reading until the peak-sampler change: a grouped convert
allocates a reference-group buffer, encodes it and frees it, so an
instantaneous sample taken after the spike has been freed cannot see it in
either direction. Numbers recorded before that change are not comparable with
numbers after it.
"""

from __future__ import annotations

import gc
import logging
import tempfile
import time
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

# Grouped sharding is SCX-only. ``scx_auto`` is the single trigger so we don't
# re-measure per codec variant (read by run_parallel.py's cohort builder).
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

# dataset name -> (group_by obs column, reference label or None for clustering-only).
# Datasets not listed are skipped (the benchmark returns None).
GROUP_SPEC: dict[str, tuple[str, str | None]] = {
    "pert_synth_10k": ("perturbation", "control"),
    "nb_glm_synth": ("perturbation", "control"),
    "replogle_k562": ("gene", "non-targeting"),
    "tahoe_c38": ("drug", None),
    "chemogenetic_rgfp": ("target_gene", "non-targeting"),
}

# Labels read back for the parity check (capped at the available groups).
_N_PARITY_LABELS = 3

# The convert scenarios are expensive (a dense one-pass grouped gather is a
# full-width random row read — minutes on a real Perturb-seq file), so cap their
# repeat count well below the suite's N_RUNS_SMALL=5; extra reps add little
# signal and risk the SLURM time budget. The cheaper grouped sort is capped
# higher.
_MAX_SORT_RUNS = 3
_MAX_CONVERT_RUNS = 2


def _gc() -> None:
    gc.collect()


def _time_op(fn, *args, **kwargs):
    """Run *fn*, returning ``(result, wall_s, peak_rss_mb)``.

    The RSS figure is the high-water mark *while fn ran*, not a reading taken
    after it returned. The distinction is the whole point: a grouped convert
    allocates a reference-group buffer, encodes it and frees it, so the
    interesting number is gone by the time ``fn`` returns. ``PeakRssSampler``
    seeds with the entry RSS, so a large object this process is still holding
    from a previous scenario is attributed here too — that is why each caller
    ``_gc()``s first.
    """
    _gc()
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        out = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    return out, wall, sampler.peak_mb


def _materialize_source(dataset: DatasetConfig, workdir: Path) -> Path:
    """Return a source h5ad path for the convert scenarios.

    Real datasets use their on-disk ``h5ad_path``; synthetic perturbation
    datasets are generated in-process via ``_pert_synth`` (the same fixtures
    ``cell_eval_parity_perf`` / ``accel_de_nb_glm`` use) and written to a temp
    h5ad. Both carry the grouping column named in ``GROUP_SPEC``.
    """
    if not dataset.synthetic:
        p = dataset.h5ad_path
        if not p.exists():
            raise FileNotFoundError(
                f"Source h5ad not found for {dataset.name}: {p}. "
                f"Run benchmarks/comprehensive/scripts/prep_grouped_fixtures.py first."
            )
        return p

    from benchmarks.comprehensive.benchmarks import _pert_synth

    params = dict(dataset.synth_params)
    out = workdir / f"{dataset.name}_source.h5ad"
    if "n_donors" in params:
        adata = _pert_synth.make_raw_counts_stratified(**params)
    else:
        real, _pred = _pert_synth.make_paired_adata(**params)
        adata = real
    adata.write_h5ad(out)
    return out


def _check_correctness(
    scx_path: Path, group_col: str, reference: str | None
) -> tuple[int, int, int]:
    """Read the grouped file back and verify the layout.

    Returns ``(correctness_passed_int, reference_isolated_int, n_groups)``.
    ``read_group(label)`` rows must all carry ``label`` in ``group_col``; when a
    reference is configured, ``read_reference()`` must be present and all its
    rows must carry the reference label.
    """
    import pyscx

    exp = pyscx.open(str(scx_path))
    labels = list(exp.group_labels())
    n_groups = len(labels)

    passed = 1
    sample = [lbl for lbl in labels if lbl != reference][:_N_PARITY_LABELS]
    for lbl in sample:
        grp = exp.read_group(lbl)
        if grp.n_obs <= 0 or not all(v == lbl for v in grp.obs[group_col]):
            passed = 0
            break

    reference_isolated = 1
    if reference is not None:
        ref = exp.read_reference()
        if ref is None or ref.n_obs <= 0 or not all(
            v == reference for v in ref.obs[group_col]
        ):
            reference_isolated = 0
    return passed, reference_isolated, n_groups


def _run_sort_group(
    result: BenchmarkResult,
    dataset: DatasetConfig,
    group_col: str,
    reference: str | None,
    plain_scx: Path,
    workdir: Path,
    n_runs: int,
) -> None:
    """Time ``pyscx.sort(group_by=…)`` re-sharding the plain ``.scx``."""
    import pyscx

    ref_arg = [reference] if reference is not None else None
    last_out: Path | None = None
    for i in range(n_runs):
        out = workdir / f"sorted_{i}.scx"
        _, wall, rss = _time_op(
            pyscx.sort,
            str(plain_scx),
            str(out),
            by=[],
            group_by=group_col,
            reference=ref_arg,
        )
        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            scenario="sort_group",
            output_size_bytes=out.stat().st_size,
            # `peak_rss_mb=` above is a *reserved* `add_run` parameter: it lands
            # on the RunRecord, never in `extra`, and thresholds read `extra`
            # only. The three ceilings in thresholds.yaml were keyed to the bare
            # name and so could never read a value. Emitted here under a name
            # that is not reserved — plus the house-style sparse per-scenario
            # key for diagnosis.
            grouped_peak_rss_mb=round(rss, 1),
            **{"peak_rss_mb__sort_group": round(rss, 1)},
        )
        logger.info("  sort_group %d/%d: wall=%.3fs rss=%.1fMB", i + 1, n_runs, wall, rss)
        if last_out is not None:
            last_out.unlink(missing_ok=True)
        last_out = out

    # Correctness on the final sorted output, recorded as a 0-run-extra via a
    # dedicated zero-wall record so the gate can read the metric.
    passed, ref_isolated, n_groups = _check_correctness(last_out, group_col, reference)
    result.metadata["n_groups"] = n_groups
    result.add_run(
        wall_s=0.0,
        peak_rss_mb=0.0,
        scenario="correctness",
        correctness_passed_int=passed,
        reference_isolated_int=ref_isolated,
        n_group_records=n_groups,
    )
    if last_out is not None:
        last_out.unlink(missing_ok=True)


def _run_convert(
    result: BenchmarkResult,
    scenario: str,
    group_pass: str,
    source_h5ad: Path,
    group_col: str,
    reference: str | None,
    workdir: Path,
    n_runs: int,
) -> None:
    """Time ``pyscx.from_h5ad(group_by=…, group_pass=…)`` convert-time grouping."""
    import pyscx

    ref_arg = [reference] if reference is not None else None
    for i in range(n_runs):
        out = workdir / f"{scenario}_{i}.scx"
        _, wall, rss = _time_op(
            pyscx.from_h5ad,
            str(source_h5ad),
            str(out),
            group_by=group_col,
            reference=ref_arg,
            group_pass=group_pass,
        )
        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            scenario=scenario,
            group_pass=group_pass,
            output_size_bytes=out.stat().st_size,
            grouped_peak_rss_mb=round(rss, 1),
            **{f"peak_rss_mb__{scenario}": round(rss, 1)},
        )
        logger.info("  %s %d/%d: wall=%.3fs rss=%.1fMB", scenario, i + 1, n_runs, wall, rss)
        out.unlink(missing_ok=True)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Execute the grouped-sharding benchmark for one (dataset, scx_auto) pair."""
    if format_variant is not None and format_variant.key not in SUPPORTED_FORMATS:
        return None
    spec = GROUP_SPEC.get(dataset.name)
    if spec is None:
        return None  # dataset has no grouping column configured — skip cleanly
    group_col, reference = spec

    result = BenchmarkResult(
        benchmark="grouped_sort",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "group_col": group_col,
            "reference_label": reference,
            "scenarios": ["sort_group", "convert_one", "convert_two"],
        },
    )

    workroot = tempfile.TemporaryDirectory(prefix=f"grouped_sort_{dataset.name}_")
    workdir = Path(workroot.name)
    try:
        source_h5ad = _materialize_source(dataset, workdir)
        result.metadata["source_matrix_format"] = _detect_matrix_format(source_h5ad)

        # sort_group needs a plain (ungrouped) .scx. Real datasets reuse the
        # Phase-A converted file; synthetic datasets are converted in-process.
        if converted_path is not None and Path(converted_path).exists():
            plain_scx = Path(converted_path)
        else:
            import pyscx

            plain_scx = workdir / "plain.scx"
            pyscx.from_h5ad(str(source_h5ad), str(plain_scx))
        result.file_size_bytes = plain_scx.stat().st_size

        sort_runs = min(n_runs, _MAX_SORT_RUNS)
        convert_runs = min(n_runs, _MAX_CONVERT_RUNS)

        logger.info("sort_group: %s (group_by=%s)", dataset.name, group_col)
        _run_sort_group(
            result, dataset, group_col, reference, plain_scx, workdir, sort_runs
        )

        logger.info("convert_one: %s", dataset.name)
        _run_convert(
            result, "convert_one", "one", source_h5ad, group_col, reference, workdir, convert_runs
        )

        logger.info("convert_two: %s", dataset.name)
        _run_convert(
            result, "convert_two", "two", source_h5ad, group_col, reference, workdir, convert_runs
        )
    finally:
        workroot.cleanup()

    _emit_per_scenario_medians(result)
    logger.info("grouped_sort done: %s — %d runs", dataset.name, len(result.runs))
    return result


def _detect_matrix_format(h5ad_path: Path) -> str:
    """Best-effort h5ad X encoding (csr / dense / csc) for metadata."""
    try:
        import h5py

        with h5py.File(h5ad_path, "r") as f:
            x = f["X"]
            if isinstance(x, h5py.Group):
                return str(dict(x.attrs).get("encoding-type", "csr_matrix"))
            return "dense"
    except Exception:
        return "unknown"


def _emit_per_scenario_medians(result: BenchmarkResult) -> None:
    """Aggregate per-scenario medians into metadata for the reporting layer."""
    import statistics

    buckets: dict[str, dict[str, list[float]]] = {}
    for rec in result.runs:
        sc = rec.extra.get("scenario")
        if not sc or sc == "correctness":
            continue
        buckets.setdefault(sc, {}).setdefault("_wall_s", []).append(rec.wall_s)
        buckets[sc].setdefault("_peak_rss_mb", []).append(rec.peak_rss_mb)
        osb = rec.extra.get("output_size_bytes")
        if osb is not None:
            buckets[sc].setdefault("_output_size_bytes", []).append(osb)

    summaries: dict[str, dict[str, float]] = {}
    for sc, b in buckets.items():
        s: dict[str, float] = {}
        for k, vals in b.items():
            if vals:
                s[k.lstrip("_") + "_median"] = round(statistics.median(vals), 6)
        summaries[sc] = s
    result.metadata["per_scenario_medians"] = summaries
