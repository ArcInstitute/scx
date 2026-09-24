"""Grouped-sharding read/write benchmark — scx's F1/F2 condition grouping.

Times scx's grouped write plus per-perturbation grouped reads, and verifies the
read-back layout, for every grouping dataset (the :data:`grouped_sort.GROUP_SPEC`
entries):

  * ``grouped_write`` — write a reference-first / group-clustered file directly
    from the source h5ad via ``pyscx.from_h5ad(group_by=, reference=)``.
  * ``read_group``    — read one perturbation's cells back
    (``read_group(label)``), once per sampled label; the median is the headline.
    ``cells_read`` is recorded so cells/s throughput is derivable.
  * ``read_reference`` — read the isolated reference/control cells
    (``read_reference()``), when the dataset configures a reference.
  * ``query_filter``  — the same "read one perturbation" access pattern via the
    lazy query engine + predicate pushdown (the group_by column is
    auto-indexed on grouped convert).
  * ``iter_group_shards`` — a full streaming pass over every (non-reference)
    group, for throughput (cells/s) and true peak RSS.
  * ``correctness``   — a zero-wall record: every read group's rows carry its
    label, and the reference rows all carry the reference label (same semantics
    as ``grouped_sort._check_correctness``).

Supported formats: ``scx_auto`` only — every other variant returns ``None`` so
the orchestrator skips it. Datasets absent from ``GROUP_SPEC`` also return
``None``. Like ``grouped_sort`` this benchmark builds its own grouped files
in-process from the source h5ad (real datasets via
``prep_grouped_fixtures.py``; synthetic via ``_pert_synth``), so it carries no
Phase-A conversion dependency — it lives in ``run_parallel.py``'s
``_NO_CONVERSION`` set.

``peak_rss_mb`` is sampled immediately *after* each op (not a true peak), matching
``grouped_sort``. No perf floors are gated — grouped perf is hardware/density
sensitive; only the correctness ints are hard-gated (see ``thresholds.yaml``).
Numbers land in the report and ``docs/performance/query-and-file-ops.md``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from benchmarks.comprehensive.benchmarks.grouped_sort import (
    GROUP_SPEC,
    _detect_matrix_format,
    _emit_per_scenario_medians,
    _materialize_source,
    _time_op,
)
from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

# scx's grouped sharding. `scx_auto` is the single trigger (read by
# run_parallel.py's cohort builder to skip other format variants).
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

# The float paired-expression fixture (`pert_synth_10k`, `make_paired_adata`)
# is already covered end-to-end by the SCX-only `grouped_sort` (sort_group /
# convert_one / convert_two / convert_sort_by); excluded here to avoid
# duplicating that coverage. The remaining GROUP_SPEC datasets are all raw
# counts: `nb_glm_synth` (synthetic stratified counts), `replogle_k562` (dense
# counts), `tahoe_c38` (CSR counts).
_DATASETS: frozenset[str] = frozenset(GROUP_SPEC) - {"pert_synth_10k"}

# Grouped writes are minutes-scale on real Perturb-seq files; cap reps low.
_MAX_WRITE_RUNS = 2
# Number of distinct non-reference labels read back for the read_group timing.
_MAX_READ_LABELS = 5


# ----------------------------------------------------------------------------
# scx grouped write / open, and per-format-independent helpers.
# ----------------------------------------------------------------------------


def _scx_grouped_write(out: Path, src: Path, col: str, ref: str | None):
    import pyscx

    ref_arg = [ref] if ref is not None else None
    # `csc="off"`: this times the grouped CSR write; the ingest default would
    # add a sidecar build on a >= 50,000 x 5,000 input.
    return _time_op(
        pyscx.from_h5ad, str(src), str(out), group_by=col, reference=ref_arg,
        csc="off",
    )


def _scx_open(out: Path):
    import pyscx

    exp = pyscx.open(str(out))
    labels = [str(x) for x in exp.group_labels()]
    return exp, labels, exp.read_group, exp.read_reference


def _grouped_output_path(workdir: Path, i: int) -> Path:
    return workdir / f"grouped_{i}.scx"


def _iter_all_groups(handle) -> tuple[int, float, float]:
    """Full streaming pass over all (non-reference) groups.

    Returns ``(total_cells, wall_s, true_peak_rss_mb)``. scx exposes
    ``iter_group_shards`` yielding a per-shard object with ``to_anndata``.
    """
    import gc
    import time

    gc.collect()
    total = 0
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        for shard in handle.iter_group_shards():
            total += int(shard.to_anndata().n_obs)
        wall = time.perf_counter() - t0
    return total, wall, sampler.peak_mb


def _check_correctness(
    labels: list[str],
    read_group,
    read_reference,
    group_col: str,
    reference: str | None,
) -> tuple[int, int, int]:
    """Return ``(correctness_passed_int, reference_isolated_int, n_groups)``."""
    n_groups = len(labels)
    passed = 1
    sample = [lbl for lbl in labels if lbl != reference][:_MAX_READ_LABELS]
    for lbl in sample:
        grp = read_group(lbl)
        # Vectorized label check (pandas) — avoids a slow Python-level loop over
        # the obs Series for large groups.
        if grp is None or grp.n_obs <= 0 or not (
            grp.obs[group_col].astype(str) == lbl
        ).all():
            passed = 0
            break

    reference_isolated = 1
    if reference is not None:
        ref = read_reference()
        if ref is None or ref.n_obs <= 0 or not (
            ref.obs[group_col].astype(str) == reference
        ).all():
            reference_isolated = 0
    return passed, reference_isolated, n_groups


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Execute the grouped read/write benchmark for one (dataset, scx_auto) pair."""
    if format_variant is None or format_variant.key not in SUPPORTED_FORMATS:
        return None
    if dataset.name not in _DATASETS:
        return None  # no grouping column configured, or covered by grouped_sort (see _DATASETS)
    spec = GROUP_SPEC.get(dataset.name)
    if spec is None:
        return None
    group_col, reference = spec
    fmt = format_variant.key

    result = BenchmarkResult(
        benchmark="grouped_read",
        format=fmt,
        dataset=dataset.name,
        metadata={
            "group_col": group_col,
            "reference_label": reference,
            "scenarios": [
                "grouped_write", "read_group", "read_reference",
                "query_filter", "iter_group_shards",
            ],
        },
    )

    workroot = tempfile.TemporaryDirectory(prefix=f"grouped_read_{dataset.name}_")
    workdir = Path(workroot.name)
    try:
        source_h5ad = _materialize_source(dataset, workdir)
        result.metadata["source_matrix_format"] = _detect_matrix_format(source_h5ad)

        # --- grouped_write (timed, capped reps) ---
        write_runs = min(n_runs, _MAX_WRITE_RUNS)
        last_out: Path | None = None
        for i in range(write_runs):
            out = _grouped_output_path(workdir, i)
            _, wall, rss = _scx_grouped_write(out, source_h5ad, group_col, reference)
            result.add_run(
                wall_s=wall,
                peak_rss_mb=rss,
                scenario="grouped_write",
                output_size_bytes=out.stat().st_size,
            )
            logger.info(
                "  grouped_write[%s] %d/%d: wall=%.3fs rss=%.1fMB",
                fmt, i + 1, write_runs, wall, rss,
            )
            if last_out is not None and last_out != out:
                last_out.unlink(missing_ok=True)
            last_out = out

        assert last_out is not None
        result.file_size_bytes = last_out.stat().st_size

        # --- read_group (per sampled label) + read_reference + correctness ---
        _handle, labels, read_group, read_reference = _scx_open(last_out)
        sample = [lbl for lbl in labels if lbl != reference][:_MAX_READ_LABELS]
        logger.info(
            "  read_group[%s]: %d labels sampled of %d", fmt, len(sample), len(labels)
        )
        for lbl in sample:
            grp, wall, rss = _time_op(read_group, lbl)
            result.add_run(
                wall_s=wall,
                peak_rss_mb=rss,
                scenario="read_group",
                label=lbl,
                cells_read=int(grp.n_obs) if grp is not None else 0,
            )

        if reference is not None:
            ref_ad, wall, rss = _time_op(read_reference)
            result.add_run(
                wall_s=wall,
                peak_rss_mb=rss,
                scenario="read_reference",
                cells_read=int(ref_ad.n_obs) if ref_ad is not None else 0,
            )

        # --- query_filter: "read one perturbation" via the lazy query engine +
        # predicate pushdown, vs read_group's byte-range read. The group_by
        # column is auto-indexed on grouped convert, so filter_obs pushes
        # down. ---
        for lbl in sample:
            def _q(label=lbl):
                return _handle.query().filter_obs(
                    f"{group_col} == '{label}'"
                ).collect()

            res, wall, rss = _time_op(_q)
            result.add_run(
                wall_s=wall,
                peak_rss_mb=rss,
                scenario="query_filter",
                label=lbl,
                cells_read=int(getattr(res, "n_obs", 0) or 0),
            )

        # --- iter_group_shards: full streaming pass over all groups —
        # throughput (cells/s) + true peak RSS (~one shard resident).
        cells, wall, peak = _iter_all_groups(_handle)
        result.add_run(
            wall_s=wall,
            peak_rss_mb=peak,
            scenario="iter_group_shards",
            cells_read=cells,
        )

        passed, ref_isolated, n_groups = _check_correctness(
            labels, read_group, read_reference, group_col, reference
        )
        result.metadata["n_groups"] = n_groups
        result.add_run(
            wall_s=0.0,
            peak_rss_mb=0.0,
            scenario="correctness",
            correctness_passed_int=passed,
            reference_isolated_int=ref_isolated,
            n_group_records=n_groups,
        )
    finally:
        workroot.cleanup()

    _emit_per_scenario_medians(result)
    logger.info(
        "grouped_read done: %s [%s] — %d runs", dataset.name, fmt, len(result.runs)
    )
    return result
