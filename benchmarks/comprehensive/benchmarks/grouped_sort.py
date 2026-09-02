"""Grouped-sharding benchmark — ``scx sort --group-by`` / ``scx convert --group-by``.

Measures the F1/F2 grouped-sort + Phase-7.4 convert-time grouping feature. For
every grouping dataset (those present in :data:`GROUP_SPEC`) it times four
scenarios and runs a read-back correctness check:

  * ``sort_group``  — ``pyscx.sort(plain_scx, group_by=, reference=)``: re-shard an
    already-converted ``.scx`` into the reference-first / group-clustered layout.
  * ``convert_one`` — ``pyscx.from_h5ad(group_by=, group_pass="one")``: convert-time
    grouping forced down the one-pass streaming gather.
  * ``convert_two`` — ``pyscx.from_h5ad(group_by=, group_pass="two")``: convert-time
    grouping forced down the two-pass (plain convert + sort) path.
  * ``convert_sort_by`` — ``pyscx.from_h5ad(sort_by=[group_col])``: sort-on-convert.
    The same permutation machinery as the grouping arms (``compute_sort_perm``
    → ``SortKeyExtractor`` → ``stable_argsort``, then a ``PermutedCsrReader``
    over X and a ``take`` over obs/obsm) without group-edge shard splitting or
    the reference-first rule. Nothing in the suite called ``sort_by=`` before,
    so OPT-CONVERT-8 and OPT-OPS-5 had no metric to register against. It
    reuses ``GROUP_SPEC``'s column as the sort key precisely so it is
    comparable with ``convert_one`` / ``convert_two`` on the same file — but
    note both grouped fixtures carry ``obsm``, which the sort path permutes, so
    the arms are not doing identical work.

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

``grouped_peak_rss_mb`` is **flat, not per-scenario**: every timed run in this
module used to carry it, and the gate medians it across all of them. So an arm
added to this module silently moves what those three ceilings are measured
against. ``convert_sort_by`` therefore omits it and reports only its sparse
``peak_rss_mb__convert_sort_by`` / ``wall_s__convert_sort_by`` keys; a future
arm should do the same unless it genuinely belongs in the pooled ceiling.

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
            **{
                "peak_rss_mb__sort_group": round(rss, 1),
                # `wall_s=` above is reserved and never reaches `extra`, so
                # without this the only gateable timing for this arm is the
                # pooled `median_wall_s` across all four scenarios.
                "wall_s__sort_group": round(wall, 6),
            },
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


def _verify_sorted(out: Path, group_col: str, expect_n_obs: int) -> dict[str, int]:
    """Check the sort arm's output really is sorted, and lost nothing.

    Returns gateable 0/1 ints. Without this the arm records a wall and a peak
    for a file nothing looked at: a regression that silently **ignored**
    ``sort_by`` would convert faster, produce a valid file, and pass any `max`
    ceiling authored on `wall_s__convert_sort_by` — the exact "green number on
    the wrong branch" this module's other guards exist to prevent.

    Two checks:

    * ``sort_by_ordered_int`` — the key column is non-decreasing. Catches
      ``sort_by`` being dropped outright, which is the failure that would
      otherwise read as a speedup.
    * ``sort_by_rows_kept_int`` — ``n_obs`` is unchanged. Catches rows lost in
      the permutation.

    **Not checked per capture: that X followed obs.** A convert that permuted
    the obs axis and left the matrix in source order passes both of the above.
    Catching it per run needs the *source's* per-row X against the output's
    under the same obs label — random row reads into a multi-GB h5ad, on every
    timed run, in an arm whose whole output is a wall-clock number.

    It is checked **once, in the suite**, on a 12-row fixture whose every row
    carries a distinct X sentinel:
    ``test_sort_by_convert_moves_x_with_obs``. That is the right level for it —
    the permutation is `from_h5ad`'s behaviour, identical on every dataset, so
    re-establishing it per capture buys nothing that a converter regression
    would not already trip in the suite. Recorded here so the boundary of what
    these two ints establish is not mistaken for the whole claim.
    """
    import numpy as np
    import pyscx

    exp = pyscx.open(str(out))
    try:
        n_obs = int(exp.n_obs)
        keys = exp.read_obs([group_col])[group_col].astype("string").to_numpy()
        ordered = bool(np.all(keys[:-1] <= keys[1:])) if len(keys) > 1 else True
    finally:
        close = getattr(exp, "close", None)
        if close is not None:
            close()
    return {
        "sort_by_ordered_int": int(ordered),
        "sort_by_rows_kept_int": int(n_obs == expect_n_obs),
    }


def _run_convert(
    result: BenchmarkResult,
    scenario: str,
    group_pass: str | None,
    source_h5ad: Path,
    group_col: str,
    reference: str | None,
    workdir: Path,
    n_runs: int,
    *,
    reorder: str = "group_by",
    expect_n_obs: int = 0,
) -> None:
    """Time convert-time reordering: ``group_by=`` grouping or ``sort_by=`` sort.

    ``reorder="group_by"`` (default) is the original arm:
    ``pyscx.from_h5ad(group_by=…, group_pass=…)``, group-aligned CSR shards
    with the reference group first.

    ``reorder="sort_by"`` runs ``from_h5ad(sort_by=[group_col])`` instead —
    the same permutation machinery (`compute_sort_perm` →
    `SortKeyExtractor` → `stable_argsort`, then a `PermutedCsrReader` over X
    and a `take` over obs/obsm) without the group-edge shard splitting or the
    reference-first rule. `sort_by` is the target of OPT-CONVERT-8 and
    OPT-OPS-5, and nothing in the suite called it. It must be a **list** (a
    bare `str` is a pyo3 `TypeError`) and it forces the streaming route
    regardless of `stream=`.

    The sort arm's output is **verified before it is unlinked**
    (``_verify_sorted``): the key column must be non-decreasing and the row
    count preserved. Nothing else in this module looks at a `sort_by` output,
    and a convert that silently ignored `sort_by` would be *faster* — so a
    `max` ceiling on ``wall_s__convert_sort_by`` would read the regression as
    an improvement.

    Metric note: this arm deliberately does **not** emit
    ``grouped_peak_rss_mb``. That key is flat, not per-scenario — every timed
    run in this module carries it — and it is what the three ceilings in
    `thresholds.yaml` read, medianing across all of them. Emitting it here
    would silently move an existing ceiling's basis, so the sort arm reports
    only its sparse ``peak_rss_mb__convert_sort_by`` / ``wall_s__…`` keys.
    """
    import pyscx

    ref_arg = [reference] if reference is not None else None
    for i in range(n_runs):
        out = workdir / f"{scenario}_{i}.scx"
        if reorder == "sort_by":
            _, wall, rss = _time_op(
                pyscx.from_h5ad,
                str(source_h5ad),
                str(out),
                sort_by=[group_col],
            )
        else:
            _, wall, rss = _time_op(
                pyscx.from_h5ad,
                str(source_h5ad),
                str(out),
                group_by=group_col,
                reference=ref_arg,
                group_pass=group_pass,
            )
        extra: dict[str, object] = {
            f"peak_rss_mb__{scenario}": round(rss, 1),
            f"wall_s__{scenario}": round(wall, 6),
        }
        if reorder == "sort_by":
            # Verified before the output is unlinked. A convert that ignored
            # `sort_by` would be *faster* and would pass a max ceiling.
            extra.update(_verify_sorted(out, group_col, expect_n_obs))
        else:
            # See the docstring: `grouped_peak_rss_mb` is the floored key and
            # is pooled across arms.
            extra["grouped_peak_rss_mb"] = round(rss, 1)
        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            scenario=scenario,
            group_pass=group_pass,
            reorder=reorder,
            output_size_bytes=out.stat().st_size,
            **extra,
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
            "scenarios": [
                "sort_group", "convert_one", "convert_two", "convert_sort_by",
            ],
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

        # Sort-on-convert: the same permutation machinery without grouping.
        # `from_h5ad(sort_by=…)` had no benchmark anywhere in the suite, so
        # OPT-CONVERT-8 and OPT-OPS-5 had nothing to register against. Reuses
        # GROUP_SPEC's column as the sort key so the arm is directly
        # comparable with convert_one / convert_two on the same file.
        logger.info("convert_sort_by: %s (sort_by=[%s])", dataset.name, group_col)
        _run_convert(
            result, "convert_sort_by", None, source_h5ad, group_col, reference,
            workdir, convert_runs, reorder="sort_by",
            expect_n_obs=dataset.n_obs,
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
