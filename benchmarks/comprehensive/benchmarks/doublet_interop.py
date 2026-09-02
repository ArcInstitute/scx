"""Doublet-caller interop benchmark.

Mechanizes what has until now been a hand-driven scratch script: export an SCX
file per batch, run external doublet callers in their own environments, import
their results back to canonical obs columns, reach a consensus, and score
everything — against each other and against labelled truth.

Two things are being measured, and they are kept apart on purpose:

* **The plumbing.** Does the tool's own output survive the round trip onto the
  SCX file and back, joined by key? This is deterministic — a max score delta
  and a call-disagreement count that must be zero — so it gates immediately.
  It is the mechanized form of the Phase-4 exit check, and it is the assertion
  that a positional import would fail while passing every row-count check.
* **The science.** Do the tools agree with each other, and do they find
  labelled doublets? Distributional, so its floors are calibrated from a real
  capture rather than guessed.

Labelled truth comes from **computational injection** (see
``scripts/doublet/inject_doublets.py``): no dataset here carries hashing or
genotype labels, and without labels a called rate is an observation rather than
a measurement. Injected truth is ``DOUBLET-DETECTION.md`` Category D — exact,
but not a substitute for real labels — so its ``evidence_type`` travels with
every number and must never be pooled with a hashing/genotype result.

**SCX-only** (``SUPPORTED_FORMATS = {"scx_auto"}``): the thing under test is
the SCX interop path, and re-running scDblFinder once per codec variant would
burn hours to measure the same tool. Datasets absent from :data:`DOUBLET_SPEC`
return ``None`` and are skipped, mirroring ``grouped_sort.py``.

There is no native-SCX arm yet. The detector lives in
``tasks/DOUBLET-DETECTION.md`` and is unstarted; when it lands it is one more
entry in :data:`TOOLS` plus a runner, which is the whole point of building the
comparison harness first.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult  # noqa: E402
from benchmarks.comprehensive.rss import current_rss_mb as _current_rss_mb  # noqa: E402
from benchmarks.comprehensive.scripts.doublet import metrics as _metrics  # noqa: E402
from benchmarks.comprehensive.scripts.doublet._tool_env import probe_all  # noqa: E402
from benchmarks.comprehensive.scripts.doublet.inject_doublets import (  # noqa: E402
    TRUTH_KIND,
    TRUTH_LABEL,
    InjectionSpec,
    inject_doublets,
)
from benchmarks.comprehensive.scripts.doublet.run_tool import (  # noqa: E402
    INPUT_KIND,
    run_tool,
)

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. Mirrors the runtime
guard at the top of ``run()``."""

# Order matters only for reporting. scDblFinder first because it is the
# baseline the exit criterion names.
TOOLS: tuple[str, ...] = ("scdblfinder", "scrublet", "solo")


@dataclass(frozen=True)
class DoubletDatasetSpec:
    """Per-dataset descriptor.

    Field names follow ``DOUBLET-DETECTION.md`` § Dataset manifest so a real
    hashing- or genotype-labelled file drops in by adding an entry with
    ``category="hashing"`` and a ``label_key``, with no code change here.
    """

    category: str                       # "injected" | "hashing" | "genotype" | ...
    evidence_type: str
    batch_key: str | None = None        # None = the file is a single batch
    max_batches: int | None = None      # bound runtime; records what it dropped
    cell_type_key: str | None = None
    label_key: str | None = None        # set for externally-labelled data
    positive_label_values: tuple[str, ...] = ()
    negative_label_values: tuple[str, ...] = ()
    ambiguous_label_values: tuple[str, ...] = ()
    inject: InjectionSpec | None = None
    tools: tuple[str, ...] = TOOLS
    notes: str = ""


_INJECTED_EVIDENCE = (
    "computational injection (DOUBLET-DETECTION.md Category D): exact labels, "
    "but does not reproduce capture or ambient-RNA artifacts"
)

DOUBLET_SPEC: dict[str, DoubletDatasetSpec] = {
    # pbmc3k / pbmc10k obs carries only `n_counts` — no batch column, so the
    # whole file is one batch. Small enough to be the fast signal.
    "pbmc3k": DoubletDatasetSpec(
        category="injected",
        evidence_type=_INJECTED_EVIDENCE,
        inject=InjectionSpec(rate=0.08, seed=0),
        notes="single-batch 10x PBMC; injected truth only",
    ),
    "pbmc10k": DoubletDatasetSpec(
        category="injected",
        evidence_type=_INJECTED_EVIDENCE,
        inject=InjectionSpec(rate=0.08, seed=0),
        notes="single-batch 10x PBMC; injected truth only",
    ),
    # The real multi-library case. 119 donors would be hours of scDblFinder;
    # 3 is the slice Phase 4 proved by hand (10,506 cells), and `max_batches`
    # records the cap rather than silently truncating.
    "tabula_sapiens_100k": DoubletDatasetSpec(
        category="injected",
        evidence_type=_INJECTED_EVIDENCE,
        batch_key="donor_id",
        max_batches=3,
        cell_type_key="cell_type",
        inject=InjectionSpec(rate=0.08, seed=0, max_source_cells=None),
        notes="CELLxGENE-derived; donors are separate libraries, so donor_id "
              "gives no cross-donor doublet labels — truth is injected",
    ),
}


@dataclass
class _ToolOutcome:
    tool: str
    available: bool
    reason: str | None = None
    key_added: str | None = None
    batches: list[dict] = field(default_factory=list)
    import_report: dict | None = None
    roundtrip: dict | None = None
    truth: dict | None = None
    called_rate_by_batch: dict = field(default_factory=dict)
    tool_s: float = 0.0
    residual_rss_mb: float = 0.0
    """RSS this process still holds after the tool's subprocess exited.

    Deliberately **not** named ``peak_rss_mb``. Every doublet caller runs in its
    own conda environment as an external subprocess (``scripts/doublet/run_tool``),
    and ``current_rss_mb()`` reads ``/proc/self/statm`` of the *parent*, so this
    number never contained the tool's memory at all — it is what the parent
    accumulated (the per-batch ``pd.read_csv`` tables) and still holds. A true
    peak for a child would need ``wait4`` rusage; that is out of scope here.
    """


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Run the doublet interop benchmark. See module docstring."""
    if format_variant.key not in SUPPORTED_FORMATS:
        return None
    spec = DOUBLET_SPEC.get(dataset.name)
    if spec is None:
        logger.info("doublet_interop: %s not in DOUBLET_SPEC, skipping",
                    dataset.name)
        return None

    import pyscx

    workdir = Path(tempfile.mkdtemp(prefix=f"doublet_{dataset.name}_"))
    t_start = time.perf_counter()
    try:
        return _run_in(pyscx, dataset, spec, workdir, converted_path, t_start)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_in(pyscx, dataset, spec, workdir: Path, converted_path, t_start):
    import pandas as pd

    envs = probe_all(list(spec.tools))
    scx_path, dropped = _prepare_input(pyscx, dataset, spec, workdir,
                                       converted_path)

    exp = pyscx.open(str(scx_path))
    obs = exp.read_obs()
    n_obs = len(obs)

    batches = _select_batches(obs, spec)
    # Before anything expensive: can the tools' output be joined back at all?
    key_diagnosis = _check_key_is_usable(pyscx, scx_path)
    export_dir = workdir / "batches"
    export_report = None
    if _needs_h5ad(spec):
        # Phase 6's export half, with its within-batch key guard: a batch whose
        # key is not unique is refused *before* anything is written, because a
        # tool run on that file could not tell two cells apart.
        export_report = pyscx.export_batches(
            str(scx_path), str(export_dir),
            batch_key=spec.batch_key or _SYNTHETIC_BATCH,
            batches=batches,
        )

    outcomes = [
        _run_one_tool(pyscx, tool, spec, scx_path, export_dir, export_report,
                      batches, envs.get(tool, {}), workdir)
        for tool in spec.tools
    ]

    ran = [o for o in outcomes if o.available and o.key_added]
    consensus = None
    if len(ran) >= 2:
        consensus = pyscx.doublet_consensus(
            str(scx_path),
            keys=[o.key_added for o in ran],
            method="majority",
            key_added="consensus",
        )

    obs_final = pyscx.open(str(scx_path)).read_obs()
    truth_label = obs_final[TRUTH_LABEL] if TRUTH_LABEL in obs_final else None
    truth_kind = obs_final[TRUTH_KIND] if TRUTH_KIND in obs_final else None

    for outcome in ran:
        if truth_label is not None:
            outcome.truth = _metrics.truth_metrics(
                obs_final[f"{outcome.key_added}_score"],
                obs_final[f"{outcome.key_added}_predicted"],
                truth_label,
                kinds=truth_kind,
            )
        outcome.called_rate_by_batch = _metrics.called_rate_by_batch(
            obs_final[f"{outcome.key_added}_predicted"],
            obs_final[spec.batch_key] if spec.batch_key else
            pd.Series(["all"] * len(obs_final)),
        )

    pairwise = _pairwise(obs_final, ran)
    consensus_truth = None
    if consensus is not None and truth_label is not None:
        consensus_truth = _metrics.truth_metrics(
            # Majority consensus writes no score; rank by how many tools called.
            obs_final["consensus_n_tools_calling"],
            obs_final["consensus_predicted"],
            truth_label,
            kinds=truth_kind,
        )

    total_wall = time.perf_counter() - t_start
    return _build_result(dataset, spec, n_obs, batches, dropped, envs, outcomes,
                         ran, pairwise, consensus, consensus_truth, total_wall,
                         export_report, key_diagnosis)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

_SYNTHETIC_BATCH = "_doublet_bench_batch"


def _safe_name(value) -> str:
    """Make an obs batch label safe as a filename component."""
    import re

    text = re.sub(r"[^A-Za-z0-9._-]", "_", str(value))
    return text if text.strip("._") else "batch"


def _needs_h5ad(spec) -> bool:
    return any(INPUT_KIND.get(t) == "h5ad" for t in spec.tools)


def _prepare_input(pyscx, dataset, spec, workdir: Path, converted_path):
    """Build the working SCX file, and report what the batch cap dropped.

    Always a copy, never the shared fixture. With injection configured the
    working file is *derived* from the source rather than copied: the synthetic
    rows have to exist on the same axis the tools score and the importer joins
    against.

    Returns ``(path, dropped_batches)``. The dropped list comes from here and
    not from a later pass over the converted file, because the cap is applied
    to the source *before* conversion — a later pass would see a file that only
    ever had the kept batches and report, every single time, that the cap
    dropped nothing.
    """
    import anndata as ad

    out = workdir / "bench.scx"

    if spec.inject is not None:
        source_h5ad = dataset.h5ad_path
        if not source_h5ad.exists():
            raise FileNotFoundError(
                f"doublet_interop needs the source h5ad for injection: "
                f"{source_h5ad}"
            )
        adata = ad.read_h5ad(source_h5ad)
        dropped: list[str] = []
        if spec.batch_key and spec.max_batches:
            # Subset to the capped batches BEFORE injecting, so injected rows
            # are drawn from the cells actually being scored rather than from
            # donors this run will never look at.
            all_batches = _first_batches(adata.obs[spec.batch_key], len(adata))
            keep = all_batches[:spec.max_batches]
            dropped = all_batches[spec.max_batches:]
            adata = adata[adata.obs[spec.batch_key].astype(str).isin(keep)].copy()
        if not spec.batch_key:
            # Single-batch files still go through the per-batch export, so the
            # exported path is the one under test rather than a special case
            # only the large datasets exercise. Stamped on the AnnData BEFORE
            # conversion, deliberately — see `_prepare_input`'s note below.
            adata.obs[_SYNTHETIC_BATCH] = "all"
        injected = workdir / "injected.h5ad"
        inject_doublets(adata, injected, spec.inject,
                        cell_type_key=spec.cell_type_key)
        del adata
        pyscx.from_h5ad(str(injected), str(out))
        injected.unlink(missing_ok=True)
        return out, dropped

    if not spec.batch_key:
        # The batch column would have to be stamped onto an already-converted
        # file with `modify_metadata`, and that trips a pyscx bug: an obs
        # rewritten by `modify_metadata` exports through `to_h5ad` with the
        # real index demoted to a `__index_level_0__` data column and the
        # first string column promoted to obs_names. Every exported cell is
        # then misnamed, and the tools would score cells the import cannot
        # join back. Refuse rather than ship the broken path.
        raise ValueError(
            f"{'this dataset'} has no batch_key and no injection spec, so the "
            "per-batch export has no column to split on. Set a batch_key, or "
            "an InjectionSpec (which stamps one pre-conversion)."
        )
    src = converted_path or dataset.scx_path
    if not Path(src).exists():
        raise FileNotFoundError(f"doublet_interop needs an SCX input: {src}")
    shutil.copy(src, out)
    return out, []


def _first_batches(values, limit: int) -> list[str]:
    import pandas as pd

    seen = pd.Series(values).astype(str).drop_duplicates().tolist()
    return seen[:limit]


def _select_batches(obs, spec) -> list[str]:
    """The batches present on the working file — already capped by
    :func:`_prepare_input`, which is where the cap is applied and recorded."""
    if not spec.batch_key:
        return ["all"]
    if spec.batch_key not in obs.columns:
        raise KeyError(
            f"batch_key {spec.batch_key!r} is not an obs column of this "
            f"dataset; obs has {list(obs.columns)[:20]}"
        )
    return _first_batches(obs[spec.batch_key], len(obs))


# ---------------------------------------------------------------------------
# Per-tool execution
# ---------------------------------------------------------------------------


def _run_one_tool(pyscx, tool, spec, scx_path, export_dir, export_report,
                  batches, env_record, workdir) -> _ToolOutcome:
    import pandas as pd

    if not env_record.get("available", False):
        logger.info("doublet_interop: %s unavailable — %s", tool,
                    env_record.get("reason"))
        return _ToolOutcome(tool=tool, available=False,
                            reason=env_record.get("reason"))

    batch_paths = {}
    if export_report:
        batch_paths = {str(b["batch"]): b.get("path")
                       for b in export_report["batches"]}

    tables, records, wall, residual = [], [], 0.0, 0.0
    for batch in batches:
        # The batch label is obs data: a `donor/1` would make this a path into
        # a directory that does not exist, and the runner would fail on write.
        out_csv = workdir / f"{tool}_{_safe_name(batch)}.csv"
        kwargs = {"out_csv": out_csv, "batch": str(batch),
                  "seed": spec.inject.seed if spec.inject else 0}
        if INPUT_KIND[tool] == "scx":
            kwargs["scx_path"] = scx_path
            kwargs["batch_key"] = spec.batch_key or _SYNTHETIC_BATCH
        else:
            path = batch_paths.get(str(batch))
            if not path:
                raise RuntimeError(
                    f"{tool} needs the exported h5ad for batch {batch!r}, but "
                    f"export_batches wrote none (skipped_reason: "
                    f"{[b.get('skipped_reason') for b in export_report['batches'] if str(b['batch']) == str(batch)]})"
                )
            kwargs["h5ad_path"] = Path(path)

        t0 = time.perf_counter()
        record = run_tool(tool, **kwargs)
        wall += time.perf_counter() - t0
        residual = max(residual, _current_rss_mb())

        if not record.get("available", True):
            return _ToolOutcome(tool=tool, available=False,
                                reason=record.get("reason"))
        records.append(record)
        tables.append(pd.read_csv(out_csv))

    if not tables:
        return _ToolOutcome(tool=tool, available=False,
                            reason="no batch produced a table")

    # One import, not one per batch: `overwrite` REPLACES rather than merges,
    # so importing per-batch tables in turn would keep only the last batch's
    # scores and silently null every earlier one.
    combined = pd.concat(tables, ignore_index=True)
    if "barcode" not in combined.columns:
        raise RuntimeError(
            f"{tool} table has no 'barcode' column; got "
            f"{list(combined.columns)}"
        )
    combined_csv = workdir / f"{tool}_all.csv"
    combined.to_csv(combined_csv, index=False)

    # No explicit `key=`. It would have to name a column present on BOTH
    # sides, and the two sides spell it differently — the tool writes
    # `barcode`, the target obs carries `__index_level_0__`. Each side
    # auto-resolves through its own fallback list, which is exactly what the
    # Phase-4 round-trip did on real data. `_check_key_is_usable` has already
    # refused the case where that resolution would join on a non-unique key.
    report = pyscx.doublet_import(str(scx_path), str(combined_csv), tool=tool)

    obs = pyscx.open(str(scx_path)).read_obs()
    roundtrip = _roundtrip(obs, combined, tool, records[0])

    return _ToolOutcome(
        tool=tool, available=True, key_added=tool, batches=records,
        import_report={k: report[k] for k in (
            "n_obs", "n_matched", "n_target_rows_absent", "n_source_rows_absent",
            "obs_key_column", "canonical_columns", "score_source_column",
            "call_source_column") if k in report},
        roundtrip=roundtrip, tool_s=wall, residual_rss_mb=residual,
    )


def _check_key_is_usable(pyscx, scx_path) -> dict:
    """Refuse before running any tool if the join key would not work.

    The tools write back whatever key they were handed — obs_names for the
    h5ad export, `colnames(sce)` for rscx — and the import auto-resolves the
    target side through its own fallback list. That works only while the obs
    index is genuinely unique across the whole file.

    Two cases this catches that nothing downstream does:

    * Barcodes repeated **across** batches. `export_batches` checks uniqueness
      *within* each batch, so this sails past it and only the global join is
      ambiguous — scores would land on another batch's cells.
    * No unique single column at all, only a composite (the measured case on a
      merged atlas: a 10x-duplicated index where `soma_joinid` alone is
      unique). A composite cannot be carried back through obs_names, so the
      benchmark has no way to rejoin.

    Both would otherwise surface hours later, after the tool time is spent.
    """
    diag = pyscx.diagnose_obs_key(str(scx_path))
    resolved, cardinality = diag.get("resolved_key"), diag.get("resolved_cardinality")
    n_obs = int(pyscx.open(str(scx_path)).n_obs_physical)
    if cardinality is not None and int(cardinality) < n_obs:
        raise RuntimeError(
            f"the obs join key {resolved!r} is not unique across this file "
            f"({cardinality} distinct values over {n_obs} rows), so per-batch "
            f"tool output could not be joined back unambiguously. "
            f"{diag['summary']} A composite-key path is not implemented for "
            "this benchmark; give the dataset a unique obs index first."
        )
    return diag


def _roundtrip(obs, tool_table, tool, record) -> dict:
    """Compare the file's canonical columns with the tool's own output, by key.

    The join here is the assertion. Aligning positionally would compare each
    row with itself and pass unconditionally, which is precisely the bug this
    check exists to catch.
    """
    import pandas as pd

    keyed = obs.copy()
    keyed.index = keyed.index.astype(str)
    src = tool_table.set_index(tool_table["barcode"].astype(str))
    aligned = keyed.reindex(src.index)

    out = _metrics.roundtrip_fidelity(
        aligned[f"{tool}_score"],
        aligned[f"{tool}_predicted"],
        src[record["score_column"]],
        _tool_calls(src[record["call_column"]], tool),
    )
    out["n_source_rows"] = int(len(src))
    out["n_missing_in_file"] = int(pd.isna(aligned[f"{tool}_score"]).sum())
    return out


def _tool_calls(values, tool):
    """Map a tool's native call tokens to booleans for comparison.

    Mirrors the profile table in ``scx-convert/src/doublet.rs`` rather than
    guessing: an unrecognised token raises instead of silently becoming False,
    which would make the round-trip check pass on a tool whose class column the
    importer read differently.
    """
    import pandas as pd

    series = pd.Series(values)
    if series.dtype == bool:
        return series.to_numpy()
    text = series.astype(str).str.strip().str.lower()
    mapping = {"doublet": True, "singlet": False,
               "true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(text) - set(mapping))
    if unknown:
        raise RuntimeError(
            f"{tool}: unrecognised call tokens {unknown[:5]} in its own "
            "output; the profile in scx-convert/src/doublet.rs and this "
            "comparison disagree about what a call is"
        )
    return text.map(mapping).to_numpy()


def _pairwise(obs, ran) -> dict:
    out = {}
    for i, a in enumerate(ran):
        for b in ran[i + 1:]:
            key = f"{a.key_added}__{b.key_added}"
            entry = _metrics.pairwise_call_agreement(
                obs[f"{a.key_added}_predicted"], obs[f"{b.key_added}_predicted"])
            entry.update(_metrics.score_rank_correlation(
                obs[f"{a.key_added}_score"], obs[f"{b.key_added}_score"]))
            out[key] = entry
    return out


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _build_result(dataset, spec, n_obs, batches, dropped, envs, outcomes, ran,
                  pairwise, consensus, consensus_truth, total_wall,
                  export_report, key_diagnosis) -> BenchmarkResult:
    skipped = {o.tool: o.reason for o in outcomes if not o.available}

    metadata = {
        "category": spec.category,
        "evidence_type": spec.evidence_type,
        "batch_key": spec.batch_key,
        "batches_run": [str(b) for b in (batches or [])],
        "batches_dropped_by_cap": dropped,
        "max_batches": spec.max_batches,
        "notes": spec.notes,
        "injection": asdict(spec.inject) if spec.inject else None,
        # Checklist item 5: which environment produced each tool's numbers,
        # read off the interpreter rather than asserted.
        "tool_environments": envs,
        "tools_ran": [o.tool for o in ran],
        "tools_skipped": skipped,
        "per_tool": {
            o.tool: {
                "import": o.import_report,
                "roundtrip": o.roundtrip,
                "truth": o.truth,
                "called_rate_by_batch": o.called_rate_by_batch,
                "batches": o.batches,
                "tool_s": o.tool_s,
            }
            for o in ran
        },
        "pairwise": pairwise,
        "consensus": consensus,
        "consensus_truth": consensus_truth,
        "export": export_report,
        "key_diagnosis": key_diagnosis,
    }
    if dropped:
        # Never let a bounded cohort read as full coverage.
        logger.info("doublet_interop: capped at %d batches; dropped %d (%s...)",
                    spec.max_batches, len(dropped), ", ".join(dropped[:3]))

    # Gateable metrics. The plumbing ones are deterministic and floor now; the
    # science ones are recorded so floors can be calibrated from a real capture
    # rather than guessed.
    extra: dict[str, float] = {
        "n_tools_ran": len(ran),
        "n_tools_skipped": len(skipped),
        "n_obs": n_obs,
        "n_batches_run": len(batches or []),
        "n_batches_dropped": len(dropped),
    }
    if ran:
        extra["roundtrip_max_score_delta"] = max(
            o.roundtrip["max_score_delta"] for o in ran)
        extra["roundtrip_call_disagreements"] = sum(
            o.roundtrip["call_disagreements"] for o in ran)
        extra["roundtrip_unmatched"] = sum(
            o.roundtrip["n_unmatched"] for o in ran)
        extra["import_matched_frac"] = min(
            (o.import_report["n_matched"] / max(o.roundtrip["n_source_rows"], 1))
            for o in ran)
        for o in ran:
            if o.truth and o.truth.get("auroc") is not None:
                extra[f"{o.tool}_auroc"] = float(o.truth["auroc"])
            if o.truth and o.truth.get("recall") is not None:
                extra[f"{o.tool}_recall"] = float(o.truth["recall"])
            if o.truth and o.truth.get("precision") is not None:
                extra[f"{o.tool}_precision"] = float(o.truth["precision"])
    for pair, entry in pairwise.items():
        if entry.get("kappa") is not None:
            extra[f"agreement_kappa__{pair}"] = float(entry["kappa"])
        if entry.get("agreement") is not None:
            extra[f"agreement_raw__{pair}"] = float(entry["agreement"])
        if entry.get("spearman") is not None:
            extra[f"score_spearman__{pair}"] = float(entry["spearman"])
    if consensus_truth and consensus_truth.get("auroc") is not None:
        extra["consensus_auroc"] = float(consensus_truth["auroc"])

    overall_passed = bool(ran) and all(
        o.roundtrip["call_disagreements"] == 0
        and o.roundtrip["max_score_delta"] <= 1e-5
        for o in ran
    )

    result = BenchmarkResult(
        benchmark="doublet_interop",
        format="scx_auto",
        dataset=dataset.name,
        overall_passed=overall_passed,
        comparison={
            "subject": {"impl": "scx_doublet_import"},
            "baseline": {"impl": "+".join(o.tool for o in ran) or "none"},
            "metric": "roundtrip_call_disagreements",
            "value": extra.get("roundtrip_call_disagreements"),
            "threshold": 0,
            "status": "pass" if overall_passed else "fail",
        },
        metadata=metadata,
    )
    # The same number goes into both slots on purpose. `peak_rss_mb` is the
    # harness's fixed field and feeds `summary.json`'s `peak_rss_mb_median`, so
    # zeroing it would erase this benchmark's memory series and publish a
    # fabricated 0 MB. `residual_rss_mb` is the honest name, and being in
    # `extra` it is the one a threshold can key off — see `_ToolOutcome`.
    residual_rss = max([o.residual_rss_mb for o in ran], default=0.0)
    result.add_run(
        wall_s=total_wall,
        peak_rss_mb=residual_rss,
        residual_rss_mb=residual_rss,
        **extra,
    )
    logger.info(
        "doublet_interop %s: %d tools ran, %d skipped, disagreements=%s",
        dataset.name, len(ran), len(skipped),
        extra.get("roundtrip_call_disagreements"),
    )
    return result
