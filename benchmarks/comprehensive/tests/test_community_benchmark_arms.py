"""Guards for the four community benchmarks' measured arms.

`test_community_benchmark_wiring.py` covers registration and scheduling — that
each benchmark resolves to the right cells. This file covers what those cells
then *measure*: that every arm emits the metric keys its floors will name, that
the parity comparisons can actually fail, and that the subprocess-arm plumbing
recovers a partial record trail from a killed worker.

The kill case is the one that needs a test most and is the one an ordinary
happy-path suite never reaches. `pipeline_ooc_constrained`'s whole claim is
"scanpy is OOM-killed here and SCX is not", and the OOM side of that is only a
measurement if the parent survives to write `pipeline_completed_int = 0.0`. A
result that simply vanishes is skipped by `check_absolute_floors` in silence and
reads exactly like coverage.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.comprehensive import subproc_arm
from benchmarks.comprehensive.benchmarks import accel_qc_filter as qcf


# ---------------------------------------------------------------------------
# subproc_arm: the process boundary every RSS-floored arm depends on
# ---------------------------------------------------------------------------

def test_run_arm_collects_every_json_line_in_order():
    src = "import json\nfor i in range(4): print(json.dumps({'stage': i}), flush=True)\n"
    out = subproc_arm.run_arm(src, timeout_s=60, label="test")
    assert out.ok
    assert [r["stage"] for r in out.records] == [0, 1, 2, 3]


def test_run_arm_ignores_non_json_noise_on_stdout():
    """Libraries print warnings to stdout; they must not break record parsing.

    The existing in-tree workers sidestep this by reading `splitlines()[-1]`.
    That is not available here — a killed worker's last line is whatever it got
    through — so the parser has to skip noise rather than rely on position.
    """
    src = (
        "import json\n"
        "print('UserWarning: something is deprecated')\n"
        "print(json.dumps({'stage': 0}), flush=True)\n"
        "print('not json either')\n"
        "print(json.dumps({'stage': 1}), flush=True)\n"
    )
    out = subproc_arm.run_arm(src, timeout_s=60, label="test")
    assert [r["stage"] for r in out.records] == [0, 1]


def test_run_arm_recovers_the_partial_trail_from_a_killed_worker():
    """SIGKILL mid-run: the stages that finished are still the measurement.

    This is the shape of a cgroup OOM kill. `killed_by_oom` has to be True and
    the records already flushed have to survive, or `pipeline_ooc_constrained`
    cannot name the stage the ceiling was hit in.
    """
    src = (
        "import json, os\n"
        "for i in range(3): print(json.dumps({'stage': i}), flush=True)\n"
        "os.kill(os.getpid(), 9)\n"
    )
    out = subproc_arm.run_arm(src, timeout_s=60, label="test")
    assert not out.ok
    assert out.killed_by_oom, f"returncode {out.returncode} not read as a kill"
    assert [r["stage"] for r in out.records] == [0, 1, 2]


def test_run_arm_does_not_raise_on_a_nonzero_exit():
    """A failure is returned as data; the caller decides if it is an error."""
    out = subproc_arm.run_arm("raise SystemExit(3)", timeout_s=60, label="test")
    assert out.returncode == 3
    assert not out.killed_by_oom
    assert "test worker failed" in out.failure_text("test")


def test_a_plain_failure_is_not_mistaken_for_an_oom_kill():
    """Exit 1 must not read as a kill, or a genuine crash becomes a `0.0` datum.

    `pipeline_ooc_constrained` records `pipeline_completed_int = 0.0` for a
    kill and *fails the cell* for anything else. Conflating them would publish
    "scanpy OOMed" for a run that actually died on an ImportError.
    """
    out = subproc_arm.run_arm("raise RuntimeError('boom')", timeout_s=60, label="test")
    assert out.returncode == 1
    assert not out.killed_by_oom


def test_worker_prelude_applies_rlimit_data_and_never_rlimit_as():
    """RLIMIT_AS would refuse the mmap of a file larger than the arm's residency.

    An SCX open mmaps the whole file, so an address-space cap fails the backed
    streaming arm the benchmark exists to showcase, on a file it never makes
    resident. Only RLIMIT_DATA may be set.
    """
    src = (
        "import json, resource\n"
        "print(json.dumps({'data': resource.getrlimit(resource.RLIMIT_DATA)[0],\n"
        "                  'addr': resource.getrlimit(resource.RLIMIT_AS)[0]}), flush=True)\n"
    )
    out = subproc_arm.run_arm(
        src, timeout_s=60, mem_limit_bytes=3 * 1024**3, label="test",
    )
    assert out.ok, out.stderr
    assert out.records[0]["data"] == 3 * 1024**3
    assert out.records[0]["addr"] == resource_unlimited()


def resource_unlimited() -> int:
    import resource

    return resource.RLIM_INFINITY


# ---------------------------------------------------------------------------
# accel_qc_filter
# ---------------------------------------------------------------------------

def test_qc_var_masks_find_real_prefixes_and_report_an_empty_subset():
    masks = qcf.qc_var_masks(
        ["MT-ND1", "MT-CO1", "RPS6", "RPL13", "ACTB", "GAPDH"]
    )
    assert masks["mt"].sum() == 2
    assert masks["ribo"].sum() == 2
    # A fixture with no mitochondrial genes must produce an *empty* mask, not a
    # silently full one: `pct_counts_mt` would then be identically zero on both
    # engines and the parity check would pass while comparing zeros.
    empty = qcf.qc_var_masks(["g0", "g1", "g2"])
    assert empty["mt"].sum() == 0
    assert empty["ribo"].sum() == 0


def test_percent_top_is_clamped_to_the_gene_axis():
    """scanpy's own default raises past the end of the var axis.

    That crash is precisely what pyscx's `percent_top=None` default avoids, and
    a benchmark that hands scanpy the unclamped tuple reintroduces it on every
    fixture with fewer than 500 genes.
    """
    assert qcf.percent_top_for(32738) == (50, 100, 200, 500)
    assert qcf.percent_top_for(250) == (50, 100, 200)
    assert qcf.percent_top_for(30) == (30,)
    assert all(p <= 30 for p in qcf.percent_top_for(30))


def test_percent_top_is_passed_to_both_engines_from_one_constant():
    """The two engines default differently; the arms must not inherit that.

    pyscx defaults `percent_top=None`, scanpy defaults `(50,100,200,500)`; and
    pyscx defaults `inplace=True` where scanpy defaults `inplace=False`. If
    either branch stops passing both explicitly, the scanpy arm silently does
    four extra order statistics, or writes nothing at all, and the speedup is
    measured against work the other engine never did.

    Checked over the AST rather than the text: a grep cannot tell a call from a
    sentence, and this module's docstring discusses both kwargs at length.
    """
    calls = _calls_named(qcf, "calculate_qc_metrics")
    assert len(calls) == 2, f"expected one call per engine, found {len(calls)}"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "percent_top" in kwargs, ast.dump(call)
        assert "inplace" in kwargs, ast.dump(call)
        assert "qc_vars" in kwargs, ast.dump(call)


def _write_npz(tmp_path: Path, name: str, **arrays) -> Path:
    p = tmp_path / name
    np.savez_compressed(p, **arrays)
    return p


def test_parity_is_zero_on_identical_columns(tmp_path):
    cols = dict(
        total_counts=np.array([10.0, 20.0, 30.0]),
        n_genes_by_counts=np.array([3.0, 4.0, 5.0]),
        pct_counts_mt=np.array([1.5, 2.5, 3.5]),
        shape_after=np.array([3, 7], dtype=np.int64),
    )
    a = _write_npz(tmp_path, "a.npz", **cols)
    b = _write_npz(tmp_path, "b.npz", **cols)
    out = qcf.parity_metrics(a, b)
    assert out["qc_metrics_max_abs_diff"] == 0.0
    assert out["filtered_shape_match_int"] == 1.0


def test_parity_catches_a_disagreeing_integer_column(tmp_path):
    """The check has to be able to fail, on the column an integer floor names."""
    base = dict(
        n_genes_by_counts=np.array([3.0, 4.0, 5.0]),
        pct_counts_mt=np.array([1.5, 2.5, 3.5]),
        shape_after=np.array([3, 7], dtype=np.int64),
    )
    a = _write_npz(tmp_path, "a.npz", total_counts=np.array([10.0, 20.0, 30.0]), **base)
    b = _write_npz(tmp_path, "b.npz", total_counts=np.array([10.0, 20.0, 31.0]), **base)
    assert qcf.parity_metrics(a, b)["qc_metrics_max_abs_diff"] == 1.0


def test_parity_masks_the_nan_scanpy_writes_for_an_empty_cell(tmp_path):
    """pyscx reports 0.0 where scanpy reports NaN for a zero-count cell.

    Folding that position in makes `max` return NaN, which the gate reads as a
    missing metric — a violation reported for a numerically perfect run.
    """
    shape = np.array([3, 7], dtype=np.int64)
    a = _write_npz(
        tmp_path, "a.npz", pct_counts_mt=np.array([1.5, 0.0, 3.5]), shape_after=shape,
    )
    b = _write_npz(
        tmp_path, "b.npz",
        pct_counts_mt=np.array([1.5, np.nan, 3.5]), shape_after=shape,
    )
    out = qcf.parity_metrics(a, b)
    assert out["qc_metrics_max_abs_diff"] == 0.0
    assert not np.isnan(out["qc_metrics_max_abs_diff"])


def test_parity_never_reports_a_nan_fold_as_agreement(tmp_path):
    """A NaN that reaches the fold must read as a violation, not as 0.0.

    The integer columns are compared unmasked, so a broken arm that emits NaN
    in `total_counts` reaches `np.max` with it. A running `max(worst, x)` would
    return `worst` — Python's `max` keeps its first argument when the
    comparison is False — and publish perfect agreement for a column that is
    not even a number. `np.max` propagates, and the non-finite result is
    converted to `inf` so the gate sees a violation.
    """
    shape = np.array([3, 7], dtype=np.int64)
    a = _write_npz(
        tmp_path, "a.npz",
        total_counts=np.array([10.0, np.nan, 30.0]), shape_after=shape,
    )
    b = _write_npz(
        tmp_path, "b.npz",
        total_counts=np.array([10.0, 20.0, 30.0]), shape_after=shape,
    )
    assert qcf.parity_metrics(a, b)["qc_metrics_max_abs_diff"] == float("inf")


def test_parity_reports_a_shape_mismatch_rather_than_crashing(tmp_path):
    a = _write_npz(
        tmp_path, "a.npz",
        total_counts=np.array([1.0, 2.0]), shape_after=np.array([2, 7]),
    )
    b = _write_npz(
        tmp_path, "b.npz",
        total_counts=np.array([1.0, 2.0, 3.0]), shape_after=np.array([3, 7]),
    )
    out = qcf.parity_metrics(a, b)
    assert out["filtered_shape_match_int"] == 0.0
    assert out["qc_metrics_max_abs_diff"] == float("inf")


def test_reference_tag_changes_when_the_comparison_changes(monkeypatch):
    """A cached reference is only a reference for the settings that produced it.

    Reusing one computed under a different `percent_top` or subset definition
    would make `qc_metrics_max_abs_diff` describe a comparison nobody asked for,
    and it would read as a pass.
    """
    from benchmarks.comprehensive.config import DATASETS

    ds = DATASETS["pbmc3k"]
    base = qcf._reference_tag(ds)
    monkeypatch.setattr(qcf, "PERCENT_TOP", (50, 100))
    assert qcf._reference_tag(ds) != base
    monkeypatch.setattr(qcf, "PERCENT_TOP", (50, 100, 200, 500))
    monkeypatch.setattr(
        qcf, "QC_VAR_PATTERNS", {"mt": ("MT-",), "ribo": ("RPS",)},
    )
    assert qcf._reference_tag(ds) != base


def test_every_arm_key_has_an_implementation():
    assert set(qcf._ARMS) == set(qcf.SUPPORTED_FORMATS)
    assert {v.key for v in qcf.accel_qc_filter_variants()} == set(qcf.SUPPORTED_FORMATS)


def test_the_backed_arm_reads_the_fixture_that_exists_on_disk():
    """`dataset.scx_path` is `<name>.scx`, which no census fixture has.

    The fixtures are built as `<name>_auto.scx`. The spec's own Task 1.2 names
    `scx_path`; `export_streaming` is still standing in that trap. Checked over
    the AST — the module docstring names the wrong property in prose precisely
    to warn about it, and a text search cannot tell those apart.
    """
    attrs = _attribute_names(qcf)
    assert "scx_auto_path" in attrs
    assert "scx_path" not in attrs


def _module_ast(mod) -> ast.Module:
    return ast.parse(Path(mod.__file__).read_text(), filename=mod.__file__)


def _calls_named(mod, func_name: str) -> list[ast.Call]:
    """Every `ast.Call` whose callee ends in *func_name*, ignoring the receiver."""
    out = []
    for node in ast.walk(_module_ast(mod)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == func_name:
            out.append(node)
    return out


def _attribute_names(mod) -> set[str]:
    """Every attribute actually *accessed* in the module body.

    Comments and docstrings are not attribute accesses, so a property named
    only in prose does not appear here.
    """
    return {
        node.attr
        for node in ast.walk(_module_ast(mod))
        if isinstance(node, ast.Attribute)
    }


# ---------------------------------------------------------------------------
# accel_score_genes
# ---------------------------------------------------------------------------

from benchmarks.comprehensive.benchmarks import accel_score_genes as sg  # noqa: E402


def _totals(n: int) -> np.ndarray:
    """Strictly decreasing totals, so rank order is unambiguous."""
    return np.arange(n, 0, -1, dtype=np.float64)


def test_gene_panels_have_the_requested_sizes():
    names = [f"g{i}" for i in range(5000)]
    panels = sg.gene_panels(names, _totals(5000))
    assert {k: len(v) for k, v in panels.items()} == {25: 25, 100: 100, 500: 500}
    for genes in panels.values():
        assert len(set(genes)) == len(genes), "a panel must not repeat a gene"


def test_gene_panels_are_deterministic():
    names = [f"g{i}" for i in range(5000)]
    totals = _totals(5000)
    assert sg.gene_panels(names, totals) == sg.gene_panels(names, totals)


def test_gene_panels_span_the_pool_rather_than_taking_its_head():
    """A head would put every gene of a K=500 panel in one expression bin.

    The control sampling both engines run is expression-matched, so a panel
    concentrated at the top of the range exercises one bin and the benchmark
    would report the cost of a degenerate case.
    """
    names = [f"g{i}" for i in range(5000)]
    panels = sg.gene_panels(names, _totals(5000))
    ranks = [int(g[1:]) for g in panels[500]]
    # The pool is the top PANEL_POOL genes; a strided draw reaches its far end.
    assert max(ranks) > sg.PANEL_POOL * 0.9, f"panel stops at rank {max(ranks)}"
    assert min(ranks) < sg.PANEL_POOL * 0.1


def test_gene_panels_break_ties_by_position_not_by_input_order():
    names = [f"g{i}" for i in range(100)]
    flat = np.ones(100)
    assert sg.gene_panels(names, flat, sizes=(10,))[10] == \
        sg.gene_panels(names, flat, sizes=(10,))[10]


def test_gene_panels_reject_a_length_mismatch():
    with pytest.raises(ValueError):
        sg.gene_panels([f"g{i}" for i in range(10)], _totals(11))


def test_gene_panels_clamp_to_a_small_gene_axis():
    panels = sg.gene_panels([f"g{i}" for i in range(30)], _totals(30))
    assert len(panels[500]) == 30
    assert len(panels[25]) == 25


def test_spearman_is_nan_not_zero_for_a_constant_vector():
    """A constant vector has no rank correlation; 0.0 would read as a failure.

    `min: 0.999` on a NaN is a missing metric, which the gate calls a violation
    on a result that exists — loud, and honestly labelled. `0.0` would be a
    loud claim that the two engines disagree completely, which is not what
    happened.
    """
    a = np.ones(50)
    b = np.arange(50, dtype=float)
    assert np.isnan(sg.spearman(a, b))
    assert np.isnan(sg.spearman(np.arange(3, dtype=float), np.arange(4, dtype=float)))


def test_score_parity_reports_a_shape_mismatch(tmp_path):
    a = _write_npz(tmp_path, "a.npz", k25=np.arange(5, dtype=float))
    b = _write_npz(tmp_path, "b.npz", k25=np.arange(6, dtype=float))
    out = sg.parity_metrics(a, b)
    assert out["score_spearman_vs_scanpy__k25"] == 0.0
    assert out["score_max_abs_diff__k25"] == float("inf")


def test_score_reference_tag_changes_with_the_sampling_settings(monkeypatch):
    from benchmarks.comprehensive.config import DATASETS

    ds = DATASETS["pbmc3k"]
    base = sg._reference_tag(ds)
    for attr, value in (("CTRL_SIZE", 10), ("N_BINS", 5), ("PANEL_POOL", 500),
                        ("SCORE_RANDOM_STATE", 7)):
        monkeypatch.setattr(sg, attr, value)
        assert sg._reference_tag(ds) != base, f"{attr} does not change the tag"
        monkeypatch.undo()


def test_exactly_one_scoring_arm_carries_parity():
    assert set(sg._ARMS) == set(sg.SUPPORTED_FORMATS)
    assert {v.key for v in sg.accel_score_genes_variants()} == set(sg.SUPPORTED_FORMATS)
    parity = [k for k, v in sg._ARMS.items() if v["parity"]]
    assert parity == ["accel_score_genes__pyscx_cpu_scanpy"]
    # `mean` and `zscore` are different statistics with no scanpy counterpart;
    # comparing either to sc.tl.score_genes would be a floor on a mismatch.
    assert all(v["method"] in ("control", "mean", "zscore") for v in sg._ARMS.values())


def test_there_is_no_method_named_scanpy():
    """The spec calls it `method="scanpy"`; the binding rejects that.

    Accepted values are "control" (which *is* scanpy's algorithm, and the
    default), "mean" and "zscore".
    """
    assert all(v["method"] != "scanpy" for v in sg._ARMS.values())


def test_the_parity_arm_passes_ctrl_genes_under_a_method_guard():
    """`ctrl_genes=` is the only thing that makes a 0.999 Spearman floor mean
    anything.

    Measured on pbmc3k: SCX's own sampler scores 0.92 / 0.73 / 0.81 against the
    scanpy reference at K = 25 / 100 / 500 with a max absolute difference of
    ~5.9 on a score of range 16.4, while `ctrl_genes=` scores 0.9999999 at
    ~1e-14. A silent fallback to the sampler would leave the floor comparing a
    run against a different control set and reporting it as agreement.

    The guard matters too: the binding rejects `ctrl_genes=` for the `mean` and
    `zscore` methods, so setting it unconditionally would raise on two arms.
    """
    assigns = _subscript_assign_guards(sg, "ctrl_genes")
    assert assigns, "ctrl_genes is never assigned into the score_genes kwargs"
    assert all(g == repr("control") for g in assigns), (
        f"ctrl_genes must be set only under a method == 'control' guard; "
        f"found guards {assigns}"
    )


def test_the_prime_call_runs_before_the_timed_region():
    """The first score_genes call in a process pays a fixed cost the rest do not.

    Measured on pbmc3k with scanpy: 1.04 s for the first panel against 0.044 s
    for the next two. Panels are scored smallest-first, so without a discarded
    prime the K=25 column is that fixed cost plus the work and the three sizes
    are not comparable to one another.
    """
    fn = _function_def(sg, "run_arm_once")
    prime = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_prime"
    ]
    withs = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.With)]
    assert prime, "run_arm_once must prime before timing"
    assert withs, "expected a PeakRssSampler `with` block"
    assert max(prime) < max(withs), "the prime must precede the timed region"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _function_def(mod, name: str) -> ast.FunctionDef:
    return next(
        n for n in ast.walk(_module_ast(mod))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _subscript_assign_guards(mod, subscript_key: str) -> list[str]:
    """For each `x["<key>"] = ...`, the literal its enclosing `if` compares to.

    Returns one entry per assignment. An assignment with no enclosing `if`
    yields `None`, which is what makes "set unconditionally" visible rather
    than simply absent from the result.
    """
    tree = _module_ast(mod)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    guards: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        hit = any(
            isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == subscript_key
            for t in node.targets
        )
        if not hit:
            continue
        cur: ast.AST | None = node
        found = None
        while cur is not None:
            cur = parents.get(cur)
            if isinstance(cur, ast.If) and isinstance(cur.test, ast.Compare):
                found = ast.unparse(cur.test.comparators[0])
                break
            if isinstance(cur, (ast.FunctionDef, ast.Module)):
                break
        guards.append(found)
    return guards


# ---------------------------------------------------------------------------
# pipeline_ooc_constrained
# ---------------------------------------------------------------------------

from benchmarks.comprehensive.benchmarks import pipeline_ooc_constrained as poc  # noqa: E402


class _FakeOutcome:
    """Enough of `ArmOutcome` for `summarize_outcome`, with no subprocess."""

    def __init__(self, records, *, returncode=0, timed_out=False, wall_s=12.0,
                 stderr=""):
        self.records = records
        self.returncode = returncode
        self.timed_out = timed_out
        self.wall_s = wall_s
        self.stderr = stderr

    @property
    def killed_by_oom(self):
        return self.returncode in subproc_arm.OOM_RETURNCODES


def _stage_records(n: int) -> list[dict]:
    return [
        {"kind": "stage", "name": poc.STAGES[i], "index": i,
         "wall_s": 1.0 + i, "peak_rss_mb": 100.0 * (i + 1)}
        for i in range(n)
    ]


def test_a_completed_pipeline_reports_one_and_no_failed_stage():
    recs = _stage_records(len(poc.STAGES)) + [{
        "kind": "done", "total_wall_s": 99.0, "n_clusters": 12,
        "n_obs_final": 1000, "n_vars_final": 2000,
    }]
    extras, reason = poc.summarize_outcome(_FakeOutcome(recs), 16)
    assert extras["pipeline_completed_int"] == 1.0
    assert reason == poc.OUTCOME_COMPLETED
    assert extras["failed_stage"] == ""
    assert extras["total_wall_s"] == 99.0
    for stage in poc.STAGES:
        assert f"stage_wall_s__{stage}" in extras
        assert f"stage_peak_rss_mb__{stage}" in extras


def test_an_oom_kill_is_recorded_as_a_zero_and_names_the_stage():
    """The case the whole benchmark rests on, and the one a happy-path suite
    never reaches.

    `check_absolute_floors` skips a triple whose *result* is missing, so a cell
    that simply vanishes when the cgroup kills it reads exactly like coverage.
    The metric has to exist and be `0.0`.
    """
    outcome = _FakeOutcome(_stage_records(3), returncode=-9)
    extras, reason = poc.summarize_outcome(outcome, 16)
    assert reason == poc.OUTCOME_OOM
    assert extras["pipeline_completed_int"] == 0.0
    # Three stages finished, so it died in the fourth.
    assert extras["failed_stage"] == poc.STAGES[3]
    assert extras["oom_at_stage_index"] == 3.0


def test_an_oom_kill_reports_the_budget_not_the_last_sample():
    """A `max: 15360` ceiling must not pass on a run that blew through it.

    SIGKILL gives the sampler no chance to take a final reading, so the largest
    *observed* stage peak can be far below what the run actually demanded — and
    for a kill during the first stage it would be 0.0, which sails under any
    ceiling. The budget is the one figure certain to be a lower bound.
    """
    outcome = _FakeOutcome(_stage_records(2), returncode=-9)
    extras, _ = poc.summarize_outcome(outcome, 16)
    assert extras["pipeline_peak_rss_mb"] == 16 * 1024.0
    assert extras["pipeline_peak_rss_mb"] > max(
        r["peak_rss_mb"] for r in _stage_records(2)
    )


def test_a_kill_before_any_stage_still_yields_a_usable_record():
    outcome = _FakeOutcome([], returncode=137)
    extras, reason = poc.summarize_outcome(outcome, 32)
    assert reason == poc.OUTCOME_OOM
    assert extras["pipeline_completed_int"] == 0.0
    assert extras["failed_stage"] == poc.STAGES[0]
    assert extras["pipeline_peak_rss_mb"] == 32 * 1024.0


def test_a_timeout_is_not_reported_as_an_oom():
    """Three different things give 0.0; only one is the claim being made."""
    outcome = _FakeOutcome(_stage_records(5), returncode=-1, timed_out=True)
    extras, reason = poc.summarize_outcome(outcome, 16)
    assert reason == poc.OUTCOME_TIMEOUT
    assert extras["pipeline_completed_int"] == 0.0
    # Not the budget: nothing here says the ceiling was reached.
    assert extras["pipeline_peak_rss_mb"] == 500.0


def test_a_degenerate_clustering_is_not_reported_as_an_oom():
    recs = _stage_records(8) + [{"kind": "degenerate", "n_clusters": 1}]
    extras, reason = poc.summarize_outcome(
        _FakeOutcome(recs, returncode=4), 16,
    )
    assert reason == poc.OUTCOME_DEGENERATE
    assert extras["pipeline_completed_int"] == 0.0
    assert extras["n_clusters"] == 1.0


def test_an_ordinary_crash_is_flagged_for_the_caller_to_raise():
    """Exit 1 must not be laundered into "scanpy hit the ceiling"."""
    extras, reason = poc.summarize_outcome(
        _FakeOutcome(_stage_records(1), returncode=1), 16,
    )
    assert reason == poc.OUTCOME_FAILED
    assert extras["pipeline_completed_int"] == 0.0


def test_every_outcome_path_emits_the_gated_metric():
    """No branch of `summarize_outcome` may omit `pipeline_completed_int`.

    A result that exists with the metric absent is a violation the gate blames
    on the benchmark; a result that is missing is skipped in silence. Neither
    is the measurement.
    """
    cases = [
        _FakeOutcome(_stage_records(9) + [{"kind": "done", "total_wall_s": 1.0}]),
        _FakeOutcome(_stage_records(3), returncode=-9),
        _FakeOutcome([], returncode=137),
        _FakeOutcome(_stage_records(5), returncode=-1, timed_out=True),
        _FakeOutcome(
            _stage_records(8) + [{"kind": "degenerate", "n_clusters": 1}],
            returncode=4,
        ),
        _FakeOutcome(_stage_records(1), returncode=1),
    ]
    for outcome in cases:
        extras, _ = poc.summarize_outcome(outcome, 16)
        assert "pipeline_completed_int" in extras
        assert "pipeline_peak_rss_mb" in extras
        assert "total_wall_s" in extras


def test_stage_reporter_flushes_each_stage_as_it_finishes():
    """Buffered output is lost to SIGKILL, and with it the whole measurement."""
    emitted = []

    def emit(line, flush=False):
        emitted.append((line, flush))

    rep = poc.StageReporter(emit=emit)
    rep.stage("load", lambda: None)
    rep.stage("qc_metrics", lambda: None)
    assert len(emitted) == 2
    assert all(flush for _line, flush in emitted), "each stage must flush"
    names = [json.loads(line)["name"] for line, _ in emitted]
    assert names == ["load", "qc_metrics"]
    assert [json.loads(line)["index"] for line, _ in emitted] == [0, 1]


def test_hvg_runs_on_raw_counts_before_normalisation_on_both_arms():
    """`seurat_v3` is a statistic of the count distribution.

    The spec's stage order puts it after normalize+log1p, where it computes
    something else. It is moved ahead of normalisation and left unsubset, so
    normalisation still runs over the full gene axis and PCA picks the mask up.
    Both arms must agree, or the two are not the same pipeline.
    """
    assert poc.STAGES.index("hvg") < poc.STAGES.index("normalize_log1p")
    assert poc.STAGES.index("normalize_log1p") < poc.STAGES.index("pca")
    fn = _function_def(poc, "run_pipeline")
    hvg_calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "highly_variable_genes"
    ]
    assert len(hvg_calls) == 2, "one HVG call per engine"
    for call in hvg_calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "flavor" in kw and kw["flavor"].value == "seurat_v3"
        assert "subset" in kw and kw["subset"].value is False, (
            "subsetting here would make normalize_total compute size factors "
            "over 2000 genes instead of the full axis"
        )


def test_the_scoped_datasets_exist_and_exclude_the_meaningless_ones():
    from benchmarks.comprehensive.config import DATASETS

    assert set(poc.FORMAT_DATASET_SCOPE) == set(poc.SUPPORTED_FORMATS)
    for names in poc.FORMAT_DATASET_SCOPE.values():
        assert names, "an empty scope schedules nothing at all"
        unknown = sorted(set(names) - set(DATASETS))
        assert not unknown, f"scope names datasets that do not exist: {unknown}"
        # A 16 GB ceiling on 2,700 cells is not an experiment.
        assert "pbmc3k" not in names


def test_the_orchestrator_reads_the_declared_scope():
    """The scope has to be enforced at cohort-build time, not inside run().

    Stubbing an out-of-scope dataset inside `run()` is a typed result, but only
    after Chimera has started a task, activated conda and imported pyscx.
    """
    import benchmarks.comprehensive.scripts.run_parallel as rp

    scope = rp._bench_format_dataset_scope("pipeline_ooc_constrained")
    assert scope, "run_parallel does not see FORMAT_DATASET_SCOPE"
    key = "pipeline_ooc_constrained__pyscx_16g"
    assert rp._triple_compatible("pipeline_ooc_constrained", "pbmc10k", key)
    assert not rp._triple_compatible("pipeline_ooc_constrained", "pbmc3k", key)


def test_the_budget_is_still_declared_consistently():
    """Phase 0 pinned this; the arm implementation must not have drifted."""
    assert set(poc.MEMORY_BUDGET_GB) == set(poc.SUPPORTED_FORMATS)
    for key, gb in poc.MEMORY_BUDGET_GB.items():
        assert key.endswith(f"_{gb}g")
        assert poc.engine_for(key) in ("pyscx", "scanpy")


def test_a_worker_reported_oom_names_the_stage_and_the_spelling():
    """The ceiling does not always arrive as SIGKILL.

    Under an explicit RLIMIT_DATA, and on some cgroup paths, the allocation
    simply fails and Python raises MemoryError. The worker says so, which is
    better evidence than inferring it from a signal — it names the stage.
    """
    recs = _stage_records(4) + [
        {"kind": "oom", "stage_index": 4, "how": "MemoryError", "detail": "..."},
    ]
    extras, reason = poc.summarize_outcome(_FakeOutcome(recs, returncode=5), 16)
    assert reason == poc.OUTCOME_OOM
    assert extras["pipeline_completed_int"] == 0.0
    assert extras["failed_stage"] == poc.STAGES[4]
    assert extras["oom_detected_as"] == "MemoryError"
    assert extras["pipeline_peak_rss_mb"] == 16 * 1024.0


def test_a_capture_run_never_classifies_on_stderr():
    """A text match is only safe where a misclassification cannot be published.

    `conversion_streaming` learned this: once the parent classifies on a
    string, every unrelated failure carrying it is silently relabelled as the
    expected outcome. On SLURM the ceiling arrives as a signal, so the capture
    path needs no guessing and does none.
    """
    outcome = _FakeOutcome(
        _stage_records(6), returncode=-6,
        stderr="OpenBLAS error: Memory allocation still failed after 10 retries",
    )
    _extras, reason = poc.summarize_outcome(outcome, 16)
    assert reason == poc.OUTCOME_FAILED, (
        "the default path must not read an allocation message as the ceiling"
    )


def test_the_local_hatch_does_classify_a_native_abort_as_the_ceiling():
    """Measured: under RLIMIT_DATA=2G the scanpy arm aborts inside OpenBLAS
    with exit -6 at the `neighbors` stage rather than raising MemoryError.

    Without this the local reproduction of the constrained regime — the only
    way to exercise the OOM path off SLURM — reports a bug instead of a result.
    """
    outcome = _FakeOutcome(
        _stage_records(6), returncode=-6,
        stderr="OpenBLAS error: Memory allocation still failed after 10 retries",
    )
    extras, reason = poc.summarize_outcome(outcome, 16, trust_stderr=True)
    assert reason == poc.OUTCOME_OOM
    assert extras["oom_detected_as"] == "native_abort"
    assert extras["failed_stage"] == poc.STAGES[6]


def test_an_unrelated_crash_under_the_hatch_is_still_a_crash():
    outcome = _FakeOutcome(
        _stage_records(2), returncode=1, stderr="ImportError: no module named x",
    )
    _extras, reason = poc.summarize_outcome(outcome, 16, trust_stderr=True)
    assert reason == poc.OUTCOME_FAILED


def test_the_local_memory_hatch_is_off_by_default(monkeypatch):
    """A real capture must take its ceiling from the cgroup and nowhere else.

    A second, differently-accounted limit on top of the cgroup would make
    `pipeline_peak_rss_mb` describe neither.
    """
    monkeypatch.delenv("SCX_BENCH_PIPELINE_MEM_LIMIT_GB", raising=False)
    assert poc._local_mem_limit_bytes() is None
    monkeypatch.setenv("SCX_BENCH_PIPELINE_MEM_LIMIT_GB", "2")
    assert poc._local_mem_limit_bytes() == 2 * 1024**3
    monkeypatch.setenv("SCX_BENCH_PIPELINE_MEM_LIMIT_GB", "not-a-number")
    assert poc._local_mem_limit_bytes() is None


# ---------------------------------------------------------------------------
# multimodal_atlas_streaming
# ---------------------------------------------------------------------------

from benchmarks.comprehensive.benchmarks import multimodal_atlas_streaming as mas  # noqa: E402


def test_x_sum_uses_a_float64_accumulator():
    """scipy accumulates in the array's own dtype, and float32 is not enough.

    Measured on the 5.2K CITE-seq fixture: the same values read as float32 and
    as uint16 are bit-identical elementwise, but `X.sum()` gives 32,173,180 and
    32,173,181 respectively. That one-unit gap is enough to fail an exact
    `modality_sums_match_int` on every arm and every dataset, reporting the
    narrowing arm as corrupting data it reproduces perfectly.
    """
    import scipy.sparse as sp

    # 2^24 is the last integer float32 represents exactly; past it, adding 1
    # is a no-op, so a float32 accumulator stalls while float64 keeps counting.
    vals = np.concatenate([[2.0**24], np.ones(64)]).astype(np.float32)
    m32 = sp.csr_matrix((vals, np.arange(vals.size), [0, vals.size]), shape=(1, vals.size))
    assert float(m32.sum()) != float(2.0**24 + 64), (
        "premise: float32 accumulation must actually lose these additions"
    )
    assert mas.x_sum_f64(m32) == float(2.0**24 + 64)


def test_x_sum_handles_a_dense_block_too():
    assert mas.x_sum_f64(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)) == 10.0


def test_sums_match_is_exact_and_fails_closed():
    ref = {"rna": 100.0, "atac": 200.0}
    assert mas.sums_match({"rna": 100.0, "atac": 200.0}, ref) == 1.0
    assert mas.sums_match({"rna": 100.0, "atac": 200.5}, ref) == 0.0
    # An arm that produced nothing must not read as agreement.
    assert mas.sums_match({}, ref) == 0.0
    assert mas.sums_match(ref, {}) == 0.0
    # A modality the reference does not know about is a disagreement, not a
    # subset match.
    assert mas.sums_match({"rna": 100.0}, ref) == 0.0
    assert mas.sums_match({"rna": 100.0, "atac": 200.0, "adt": 1.0}, ref) == 0.0


def test_each_arm_owns_a_distinct_rss_metric_name():
    """`peak_rss_mb` is a reserved `add_run` parameter and never reaches extra.

    A floor keyed to it resolves None on every run, which the gate reports as
    a missing-metric violation no benchmark can satisfy. Each arm therefore
    emits its own non-reserved name.
    """
    reserved = {"wall_s", "user_s", "sys_s", "peak_rss_mb"}
    names = [mas._rss_key(a["mode"]) for a in mas._ARMS.values()]
    assert len(set(names)) == len(names), f"arms share an RSS metric name: {names}"
    assert not (set(names) & reserved)


def test_no_structurally_constant_metric_is_emitted():
    """The spec's `value_buffer_reduction_ratio` is 2.0 by construction.

    f32 is four bytes and uint16 is two, whatever the data, so a floor on the
    literal "ratio of value buffer bytes" passes on any build — including one
    where the narrowing was removed and the cast happened after a full f32
    assembly. Read as whole-matrix bytes it instead caps at (4+4)/(2+4) = 1.33
    and can never reach the spec's 1.90. Neither reading is a gate metric, so
    the name is not emitted at all; the measured peak-RSS ratio is.
    """
    src = Path(mas.__file__).read_text()
    tree = ast.parse(src)
    emitted = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert "value_buffer_reduction_ratio" not in emitted
    assert "eager_peak_rss_ratio_f32_over_u16" in emitted


def test_the_narrowing_ratio_is_built_from_measured_peaks():
    fn = _function_def(mas, "run")
    src = ast.unparse(fn)
    assert "eager_peak_rss_ratio_f32_over_u16" in src
    assert "peak_rss_mb" in src
    # The control has to be a second real read, not a computed constant.
    assert "scx_eager_f32" in src, (
        "the u16 arm must run its own f32 control; the gate cannot join two "
        "SLURM jobs, and the two eager arms are two jobs"
    )


def _obs(**cols):
    import pandas as pd

    return pd.DataFrame(cols)


def test_predicate_selection_prefers_batch_and_rejects_a_single_level():
    import pandas as pd

    obs = _obs(
        cell_type=pd.Categorical(["a", "b", "a", "b"]),
        batch=pd.Categorical(["b0", "b1", "b0", "b1"]),
    )
    assert mas.pick_predicate_column(obs) == ("batch", "b0")
    # A one-level column selects every row, so the arm would time a full scan
    # and publish it as pushdown.
    only_one = _obs(batch=pd.Categorical(["b0"] * 4),
                    donor=pd.Categorical(["d0", "d1", "d0", "d1"]))
    assert mas.pick_predicate_column(only_one) == ("donor", "d0")
    assert mas.pick_predicate_column(_obs()) is None
    assert mas.pick_predicate_column(_obs(batch=pd.Categorical(["b0"] * 4))) is None


def test_predicate_selection_skips_a_level_that_would_break_the_expression():
    import pandas as pd

    obs = _obs(
        batch=pd.Categorical(["it's", "other"]),
        donor=pd.Categorical(["d0", "d1"]),
    )
    assert mas.pick_predicate_column(obs) == ("donor", "d0")


def test_predicate_selection_ignores_numeric_columns():
    obs = _obs(n_counts=np.array([1.0, 2.0, 3.0, 4.0]))
    assert mas.pick_predicate_column(obs) is None


def test_the_reference_is_keyed_to_the_fixture_not_just_its_name(tmp_path):
    """A rebuilt atlas is a different matrix.

    Reusing the previous sums would report a mismatch on every arm and blame
    the readers for a stale cache.
    """
    from benchmarks.comprehensive.config import DATASETS

    ds = DATASETS["cite_seq_pbmc"]
    f = tmp_path / "fixture.scx"
    f.write_bytes(b"x" * 100)
    first = mas._reference_path(ds, f)
    f.write_bytes(b"x" * 200)
    assert mas._reference_path(ds, f) != first
    assert mas._reference_path(ds, tmp_path / "absent.scx") != first


def test_every_multimodal_arm_is_implemented_and_declared():
    assert set(mas._ARMS) == set(mas.SUPPORTED_FORMATS)
    assert {v.key for v in mas.multimodal_atlas_streaming_variants()} == set(
        mas.SUPPORTED_FORMATS
    )
    # The SCX arms must read the multimodal .scx, never the `converted_path`
    # the orchestrator hands them — which for every key here is the .h5mu.
    scx_arms = [k for k, v in mas._ARMS.items() if v["source"] == "scx"]
    assert len(scx_arms) == 4
    src = Path(mas.__file__).read_text()
    assert "scx_multimodal_path" in src


def test_the_query_probe_demands_the_flag_it_needs():
    """A build predating modality-scoped query must give a typed gap.

    `resolve_scx_bin`'s probe is mandatory rather than defaulted; without
    `requires`, an old binary whose `query --help` merely exits 0 would be
    accepted and the arm would fail on an opaque usage error.
    """
    assert mas.QUERY_PROBE[0] == "query"
    assert mas.QUERY_REQUIRES == b"--modality"
