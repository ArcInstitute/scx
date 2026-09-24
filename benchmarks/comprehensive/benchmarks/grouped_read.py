"""Cross-format grouped-sharding head-to-head — scx vs shardad.

This is the direct comparison between scx's F1/F2 condition-grouped sharding and
shardad's native condition grouping (``.shad``). For every grouping dataset (the
:data:`grouped_sort.GROUP_SPEC` entries) it times, **per format**, a grouped
write plus per-perturbation grouped reads, and verifies the read-back layout:

  * ``grouped_write`` — write a reference-first / group-clustered file directly
    from the source h5ad. scx: ``pyscx.from_h5ad(group_by=, reference=)``;
    shardad: ``write_sharded(group_by=, reference=)``.
  * ``read_group``    — read one perturbation's cells back
    (``read_group(label)``), once per sampled label; the median is the headline.
    ``cells_read`` is recorded so cells/s throughput is derivable.
  * ``read_reference`` — read the isolated reference/control cells
    (``read_reference()``), when the dataset configures a reference.
  * ``correctness``   — a zero-wall record: every read group's rows carry its
    label, and the reference rows all carry the reference label (same semantics
    as ``grouped_sort._check_correctness``, but for *both* formats).

Supported formats: ``scx_auto`` and ``shardad`` — every other variant returns
``None`` so the orchestrator skips it. Datasets absent from ``GROUP_SPEC`` also
return ``None``. Like ``grouped_sort`` this benchmark builds its own grouped
files in-process from the source h5ad (real datasets via
``prep_grouped_fixtures.py``; synthetic via ``_pert_synth``), so it carries no
Phase-A conversion dependency — it lives in ``run_parallel.py``'s
``_NO_CONVERSION`` set.

``peak_rss_mb`` is sampled immediately *after* each op (not a true peak), matching
``grouped_sort``. No perf floors are gated — grouped perf is hardware/density
sensitive; only the correctness ints are hard-gated (see ``thresholds.yaml``).
Head-to-head wall/size numbers land in the report and ``docs/performance/query-and-file-ops.md``.
"""

from __future__ import annotations

import functools
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

# scx grouped sharding vs shardad's native grouping. Both keys trigger the
# benchmark; run_parallel.py's cohort builder reads this to skip other formats.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto", "shardad"})

# grouped_read is a head-to-head, so it only runs on datasets both formats can
# ingest. shardad is an integer-count format (its grouped path-source writer
# narrow-casts X.data to uint32 and raises LossyCastError on float X), so the
# float paired-expression fixture ``pert_synth_10k`` (make_paired_adata) is
# excluded here — it stays covered by the SCX-only ``grouped_sort``. The
# remaining GROUP_SPEC datasets are all raw counts: ``nb_glm_synth`` (synthetic
# stratified counts), ``replogle_k562`` (dense counts), ``tahoe_c38`` (CSR counts).
_DATASETS: frozenset[str] = frozenset(GROUP_SPEC) - {"pert_synth_10k"}

# Grouped writes are minutes-scale on real Perturb-seq files; cap reps low.
_MAX_WRITE_RUNS = 2
# Number of distinct non-reference labels read back for the read_group timing.
_MAX_READ_LABELS = 5


def _workers() -> int:
    """shardad worker count (see shardad_runner._workers)."""
    import os

    env = os.environ.get("RAYON_NUM_THREADS", "").strip()
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass
    return os.cpu_count() or 1


# ----------------------------------------------------------------------------
# Per-format adapters. Each returns callables over a grouped output path.
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


def _shardad_grouped_write(out: Path, src: Path, col: str, ref: str | None):
    """Grouped write for shardad, timed to include the source read (so it's
    comparable to scx's ``from_h5ad``, which also reads + groups + encodes).

    shardad's *backed* (path-source) grouped encoder is integer-count only — it
    hardcodes a uint32 cast and raises ``LossyCastError`` on float or dense X.
    The available real perturbation fixtures are float/normalized (and one is
    dense), so we load the source into an in-memory CSR AnnData and pass that to
    ``write_sharded`` — its in-memory grouped encoder preserves float32 and
    accepts any layout. This is the honest shardad workflow for non-integer /
    dense data; scx streams from the h5ad path instead. The in-memory load is
    inside the timed op so both formats' write timings include the source read.
    """
    from shardad import write_sharded

    ref_arg = [ref] if ref is not None else None
    nworkers = _workers()

    def _do() -> None:
        import anndata
        import scipy.sparse as sp

        a = anndata.read_h5ad(str(src))
        if not sp.isspmatrix_csr(a.X):
            a.X = sp.csr_matrix(a.X)
        write_sharded(
            a, str(out), group_by=col, reference=ref_arg,
            overwrite=True, n_workers=nworkers,
        )

    return _time_op(_do)


def _shardad_open(out: Path, group_col: str):
    import pandas as pd
    from shardad import ShardedArchive

    arch = ShardedArchive(str(out))
    # obs is readable without decoding X; unique group_col values are the labels
    # (the reference cells carry the reference label too, mirroring scx's
    # group_labels() which also includes the reference).
    labels = [str(x) for x in pd.unique(arch.obs[group_col])]
    return arch, labels, arch.read_group, arch.read_reference


def _grouped_output_path(workdir: Path, fmt_key: str, i: int) -> Path:
    ext = "shad" if fmt_key == "shardad" else "scx"
    return workdir / f"grouped_{i}.{ext}"


def _iter_all_groups(fmt: str, handle) -> tuple[int, float, float]:
    """Full streaming pass over all (non-reference) groups.

    Returns ``(total_cells, wall_s, true_peak_rss_mb)``. scx and shardad both
    expose ``iter_group_shards`` yielding a per-shard object with ``to_anndata``;
    shardad takes prefetch/worker knobs for its background-decode pipeline.
    """
    import gc
    import time

    gc.collect()
    total = 0
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        if fmt == "shardad":
            nw = _workers()
            it = handle.iter_group_shards(prefetch=nw, n_workers=nw)
        else:
            it = handle.iter_group_shards()
        for shard in it:
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
    """Execute the grouped read/write head-to-head for one (dataset, format)."""
    if format_variant is None or format_variant.key not in SUPPORTED_FORMATS:
        return None
    if dataset.name not in _DATASETS:
        return None  # no grouping column configured, or float source (see _DATASETS)
    spec = GROUP_SPEC.get(dataset.name)
    if spec is None:
        return None
    group_col, reference = spec
    fmt = format_variant.key

    if fmt == "shardad":
        grouped_write = _shardad_grouped_write
        open_grouped = functools.partial(_shardad_open, group_col=group_col)
    else:
        grouped_write = _scx_grouped_write
        open_grouped = _scx_open

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
        # Both arms read from the same source h5ad. scx streams via from_h5ad;
        # shardad loads an in-memory CSR AnnData inside its timed op (see
        # _shardad_grouped_write) — the write timing includes the source read
        # for both formats.
        write_runs = min(n_runs, _MAX_WRITE_RUNS)
        last_out: Path | None = None
        for i in range(write_runs):
            out = _grouped_output_path(workdir, fmt, i)
            _, wall, rss = grouped_write(out, source_h5ad, group_col, reference)
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
        _handle, labels, read_group, read_reference = open_grouped(last_out)
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

        # --- query_filter (scx only): "read one perturbation" via the lazy
        # query engine + predicate pushdown, vs read_group's byte-range read.
        # The group_by column is auto-indexed on grouped convert, so filter_obs
        # pushes down. shardad has no query engine → scx-only scenario. ---
        if fmt == "scx_auto":
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

        # --- iter_group_shards: full streaming pass over all groups (both
        # formats) — throughput (cells/s) + true peak RSS (~one shard resident).
        cells, wall, peak = _iter_all_groups(fmt, _handle)
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
