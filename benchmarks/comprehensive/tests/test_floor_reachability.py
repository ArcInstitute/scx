"""Every `absolute_floors` benchmark in thresholds.yaml must be schedulable.

## The recurrence this closes

A floor is evaluated only if the candidate snapshot holds a raw result for its
`(benchmark, format, dataset)` triple — `compare_against_baseline.py`'s
`_triple_was_run` skips a scoped-out triple *silently*, which is right when an
operator narrowed the run with `--formats` / `--datasets` and would otherwise
drown the report in ~50 spurious "missing" rows.

The consequence is that a benchmark absent from `ALL_BENCHMARKS` can carry
floors nothing can ever trip. `capture_baseline.py` and `run_parallel.py` both
derive their lists from `ALL_BENCHMARKS`, and `capture_baseline.py` additionally
*rejects* an off-list `--benchmarks` name — so a tiered gate cannot reach such a
floor and an explicit request cannot either. The floor reads as coverage and
provides none.

Three benchmarks were in exactly that state until Phase 5c, and the series hit
the problem three sub-phases running:

  * `conversion_streaming` / `export_streaming` — listed only in
    `scripts/run_all.py::AVAILABLE_BENCHMARKS`, with a `streaming_peak_rss_mb`
    ceiling each. Phase 6c noticed and concluded they could still be named
    explicitly; that is true of `run_parallel.py` and false of the gate, so
    Phase 6d's job died in 2.2 s on `[baseline] unknown --benchmarks`.
  * `accel_eval_metrics` — variants in `config.accel_formats()`, an entry in
    `run_parallel._NO_CONVERSION`, two `*_route_gpu_correct` floors, and no
    entry in the one list that schedules it.

## Scope

This checks *reachability*, not that a floor passes. A benchmark can be
registered and still be scoped out of a given run — that is the operator's
choice and this test says nothing about it. What it rejects is a floor whose
benchmark no supported invocation can produce.

Floors deliberately not yet enforceable belong in thresholds.yaml's "Deferred
floors" comment block, which is prose precisely so it cannot be evaluated; this
test therefore needs no exception list, and that is the strongest form. If one
becomes necessary, add the name here with the blocker written down, not silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS = PROJECT_ROOT / "benchmarks" / "comprehensive" / "thresholds.yaml"


def _floor_benchmarks() -> set[str]:
    raw = yaml.safe_load(THRESHOLDS.read_text())
    floors = raw.get("absolute_floors") or []
    return {f["benchmark"] for f in floors}


def test_thresholds_yaml_is_present_and_has_floors():
    """Guard the guard. Every assertion below is over a set read out of one
    file; if that file moves or its top-level key is renamed, the sets go empty
    and the reachability check passes while checking nothing."""
    assert THRESHOLDS.is_file(), f"{THRESHOLDS} is missing — retarget this test"
    assert len(_floor_benchmarks()) > 20, (
        "thresholds.yaml yielded almost no absolute_floors benchmarks; the "
        "schema likely changed and the check below would be vacuous"
    )


def test_every_floored_benchmark_is_in_all_benchmarks():
    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    unreachable = sorted(_floor_benchmarks() - set(ALL_BENCHMARKS))
    assert not unreachable, (
        f"thresholds.yaml declares absolute_floors for {unreachable}, which are "
        f"not in ALL_BENCHMARKS. capture_baseline.py rejects off-list "
        f"--benchmarks names and both orchestrators derive their lists from it, "
        f"so no supported invocation can produce those triples and the floors "
        f"can never fire. Register the benchmark, or move its floors into the "
        f"'Deferred floors' comment block with the blocker written down."
    )


def test_floored_benchmark_modules_are_importable():
    """A registered name that does not resolve to a module is the same gap one
    layer down: `run_parallel._run_benchmark` imports
    `benchmarks.comprehensive.benchmarks.<name>`, and
    `_bench_supported_formats` swallows an `ImportError` as "no restriction"."""
    import importlib

    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    broken = []
    for name in sorted(_floor_benchmarks() & set(ALL_BENCHMARKS)):
        try:
            mod = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{name}")
        except ImportError as exc:  # pragma: no cover - only on a real break
            broken.append(f"{name}: {exc}")
            continue
        if not hasattr(mod, "run"):
            broken.append(f"{name}: module has no run()")
    assert not broken, "floored benchmarks that cannot be dispatched: " + "; ".join(broken)


@pytest.mark.parametrize("name", ["conversion_streaming", "export_streaming"])
def test_streaming_benchmarks_declare_scx_auto_only(name: str):
    """Both self-gate on `scx_auto` inside `run()`; the module constant is what
    stops `run_parallel` scheduling a job per format in the tier just to have it
    return `None`."""
    import importlib

    mod = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{name}")
    assert isinstance(mod.SUPPORTED_FORMATS, frozenset)
    assert mod.SUPPORTED_FORMATS == frozenset({"scx_auto"})


@pytest.mark.parametrize("name", ["conversion_streaming", "export_streaming"])
def test_orchestrator_pairs_the_streaming_benchmarks_with_scx_auto_only(name: str):
    """Exercise the *filter*, not just the declaration.

    `run_parallel._bench_format_compatible` is the mechanism the module constant
    exists for, and its own test
    (`test_run_parallel_deps.py::test_bench_format_compatible_consults_supported_formats`)
    is red on `main` for an unrelated reason — it asserts a hardcoded format set
    for `compression` that drifted when `scx_shufdelta` and the
    `scx_compact_trial*` variants were added. A declaration test alone would have
    left the behaviour this change depends on uncovered.
    """
    import importlib

    rp = importlib.import_module("benchmarks.comprehensive.scripts.run_parallel")

    assert rp._bench_format_compatible(name, "scx_auto")
    for other in ("h5ad_none", "zarr_zstd", "scx_fast", "tiledb_soma"):
        assert not rp._bench_format_compatible(name, other), (
            f"{name} must not be paired with {other}: the job would run and "
            f"return None, producing a phantom missing_result entry"
        )


def test_conversion_streaming_needs_no_phase_a_conversion():
    """It reads `dataset.h5ad_path` and ignores `converted_path`; its inverse
    consumes a pre-converted file and must stay out of the set."""
    import importlib

    rp = importlib.import_module("benchmarks.comprehensive.scripts.run_parallel")
    assert "conversion_streaming" in rp._NO_CONVERSION
    assert "export_streaming" not in rp._NO_CONVERSION


def test_materialize_arm_is_bounded_above_census_1m():
    """The streaming arm — the one the `streaming_peak_rss_mb` floor gates — runs
    at every tier. The materialize arm loads the whole h5ad (13.6 GB measured at
    1M cells) and is skipped above that, with the reason recorded rather than
    dropped silently."""
    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs

    assert cs._skip_materialize_reason(1_000) is None
    assert cs._skip_materialize_reason(cs.MATERIALIZE_MAX_N_OBS) is None
    reason = cs._skip_materialize_reason(5_000_000)
    assert reason and "streaming arm" in reason


# ---------------------------------------------------------------------------
# The two ways a registered floor is still unevaluable
# ---------------------------------------------------------------------------
#
# Registration gets a benchmark scheduled. Two further silent steps sit between
# "it ran" and "its floor was checked", and job 2834602 hit both at once.


def test_no_benchmark_passes_extra_as_a_dict():
    """`BenchmarkResult.add_run` takes `**extra`.

    `add_run(..., extra={...})` therefore nests the payload under a literal
    `"extra"` key, so `_load_current_raw_metric` — which reads
    `runs[].extra[<metric>]` — never finds the metric and the floor is reported
    as a missing-metric violation or skipped entirely.

    `conversion_streaming` did this at all three call sites while carrying a
    comment explaining that the mirroring existed *for* the gate;
    `export_streaming` has a comment warning against it. A grep is the whole
    check, and it is cheap enough to keep.
    """
    import ast

    bench_dir = PROJECT_ROOT / "benchmarks" / "comprehensive" / "benchmarks"
    assert bench_dir.is_dir(), f"{bench_dir} moved — retarget this test"

    # Parsed, not grepped. A regex over the source flags this test's own
    # explanatory comment — and the comment above the fixed call site contains
    # the literal `extra={`, which is precisely the "a grep cannot tell a call
    # from a sentence" failure. `ast` sees calls only.
    offenders = []
    for path in sorted(bench_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != "add_run":
                continue
            for kw in node.keywords:
                if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "add_run() takes **extra, so these sites nest the payload under a "
        f"literal 'extra' key and their metrics are invisible to the gate: "
        f"{offenders}. Pass them as keyword arguments (or splat with **{{...}})."
    )


def test_archive_honours_the_effective_dataset_list():
    """`--datasets` can name a dataset outside `--tier`.

    `archive_raw_results` used to filter on `tier_cfg["datasets"]` alone, so
    `--tier small --datasets census_1m` ran the benchmarks and archived none of
    the results: census_1m is not in tier small. The candidate snapshot came out
    empty, every floor was skipped as a scoped-out triple, and the gate returned
    0 — a pass over nothing measured, which is the failure mode this whole file
    exists for.
    """
    import importlib

    cb = importlib.import_module("benchmarks.comprehensive.scripts.capture_baseline")

    import json
    import pathlib
    import tempfile

    small = cb.TIERS["small"]
    assert "census_1m" not in small["datasets"], (
        "premise: census_1m must be outside tier small for this test to mean "
        "anything. If the tier changed, pick another out-of-tier dataset."
    )
    # The tier filter alone rejects it, which is the mechanism under test.
    name = "conversion_streaming__scx_streaming_vs_materialize__census_1m.json"
    assert not cb._tier_matches(name, small)

    # Drive `archive_raw_results` itself. Asserting on `_tier_matches` with a
    # hand-widened dict passed with the fix reverted — it never reached the
    # widening inside the function, which is the thing that broke.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        fake_raw = tmp / "raw_src"
        fake_raw.mkdir()
        (fake_raw / name).write_text(json.dumps({
            "schema_version": 2, "benchmark": "conversion_streaming",
            "format": "scx_streaming_vs_materialize", "dataset": "census_1m",
            "runs": [{"wall_s": 1.0, "peak_rss_mb": 1.0,
                      "extra": {"streaming_peak_rss_mb": 800.0}}],
        }))
        snapshot = tmp / "snap"
        orig = cb.RAW_DIR
        try:
            cb.RAW_DIR = fake_raw
            cb.archive_raw_results(small, snapshot, datasets=["census_1m"])
        finally:
            cb.RAW_DIR = orig
        archived = sorted(p.name for p in (snapshot / "raw").glob("*.json"))
    assert archived == [name], (
        f"an out-of-tier --datasets result was not archived (got {archived}); "
        f"the candidate snapshot would be empty and every floor silently skipped"
    )


def test_archive_raw_results_signature_takes_datasets():
    """Guard the fix itself: the parameter has to exist and be wired, or the
    behaviour above silently reverts to tier-only filtering."""
    import importlib
    import inspect

    cb = importlib.import_module("benchmarks.comprehensive.scripts.capture_baseline")
    sig = inspect.signature(cb.archive_raw_results)
    assert "datasets" in sig.parameters, (
        "archive_raw_results lost its `datasets` parameter — an out-of-tier "
        "--datasets run would archive nothing again"
    )
    src = inspect.getsource(cb.main) if hasattr(cb, "main") else cb.__file__
    if isinstance(src, str) and "def main" in src:
        assert "datasets=args.datasets" in src, (
            "archive_raw_results is called without forwarding --datasets"
        )


def test_conversion_streaming_emits_its_floor_metric_at_the_top_of_extra():
    """End-to-end plumbing for the one metric `thresholds.yaml` floors.

    Monkeypatches the subprocess worker so this stays a unit test, and asserts
    what `_load_current_raw_metric` will actually read: `runs[].extra` must carry
    `streaming_peak_rss_mb` as a **top-level** key on the streaming runs. The AST
    guard above catches the `extra={...}` spelling; this catches any other way of
    burying the metric, and it is the assertion that was missing when the floor
    sat unevaluable.
    """
    import pathlib

    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs
    from benchmarks.comprehensive.results import BenchmarkResult

    calls = []

    def fake_arm(h5ad_path, n_runs, scenario, thread_count=None, reader_threads=None):
        calls.append((scenario, reader_threads))
        return [{
            "scenario": scenario, "run_idx": 0, "wall_s": 1.0,
            "peak_rss_mb": 123.0, "reader_threads": reader_threads,
            "structural": {"n_obs": 1, "n_vars": 1, "nnz": 1, "shard_count": 1},
            "output_bytes": 10,
        }]

    original = cs._run_arm_subprocess
    try:
        cs._run_arm_subprocess = fake_arm
        result = cs._run_isolated(
            pathlib.Path("/nonexistent.h5ad"), 1,
            BenchmarkResult(
                benchmark="conversion_streaming",
                format="scx_streaming_vs_materialize",
                dataset="unit",
                metadata={"scenarios": []},
            ),
        )
    finally:
        cs._run_arm_subprocess = original

    # One process per arm, and the GATED arm must be the pinned one — a floor
    # measured at the runner's core count is a property of the runner.
    assert calls == [
        ("streaming", cs.GATED_READER_THREADS),
        ("streaming", None),
        ("materialize", None),
    ], calls

    gated = [r for r in result.runs if r.extra.get("scenario") == "streaming"]
    assert gated, "no gated streaming run recorded"
    for r in gated:
        assert "streaming_peak_rss_mb" in r.extra, (
            f"the floored metric is not a top-level key in extra: {sorted(r.extra)}"
        )
        assert "extra" not in r.extra, "extra is nested one level too deep"
        assert r.extra["reader_threads"] == cs.GATED_READER_THREADS

    # The unpinned arm is recorded under its own key so it cannot be confused
    # with the gated one by `_load_current_raw_metric`.
    unpinned = [
        r for r in result.runs
        if r.extra.get("scenario") == "streaming_default_threads"
    ]
    assert unpinned and all(
        "streaming_default_threads_peak_rss_mb" in r.extra for r in unpinned
    )
    assert all("streaming_peak_rss_mb" not in r.extra for r in unpinned), (
        "the unpinned arm must not emit the gated metric key, or the floor's "
        "median would mix pinned and unpinned measurements"
    )
    assert result.metadata["structural"]["equal"] is True
    assert result.metadata["gated_reader_threads"] == cs.GATED_READER_THREADS


def test_both_streaming_modules_use_the_true_peak_sampler():
    """`*_peak_rss_mb` has to be a peak.

    Both modules previously computed `max(before, after)` of the instantaneous
    RSS while their docstrings argued the under-report was harmless "because the
    streaming path's memory profile is bounded structurally" — i.e. assuming the
    property the floor exists to verify. Six sibling benchmark modules already
    use `PeakRssSampler`.
    """
    bench_dir = PROJECT_ROOT / "benchmarks" / "comprehensive" / "benchmarks"
    for name in ("conversion_streaming.py", "export_streaming.py"):
        text = (bench_dir / name).read_text()
        assert "PeakRssSampler" in text, f"{name} does not use the peak sampler"
        code = "\n".join(
            l for l in text.splitlines()
            if not l.lstrip().startswith("#")
        )
        assert "max(rss_before, rss_after)" not in code, (
            f"{name} still reports max(before, after) as a peak"
        )


@pytest.mark.parametrize(
    "mod", ["conversion_streaming", "export_streaming"]
)
def test_thread_sweep_does_not_emit_the_floored_metric_key(mod: str):
    """The thread-scaling sweep must not reuse the gated key.

    `thresholds.yaml` floors `streaming_peak_rss_mb`, and
    `_load_current_raw_metric` takes the **median of every run carrying it**. The
    sweep emitted that bare key for each thread count, so setting
    `SCX_{CONV,EXPORT}_STREAM_THREAD_COUNTS` turned the pinned rt=4 ceiling into a
    median over whatever counts the operator swept — silently undoing the pinning
    the floor depends on. `submit_streaming_threads_census1m.sh` exercises that
    branch, so it was reachable, not hypothetical. Found by review
    (codex - gpt-5.6-sol) on PR #451.

    Checked on the source rather than by running a sweep: the sweep needs a real
    census-scale fixture and minutes of wall time, and the defect is entirely in
    which key string is written.
    """
    import ast

    path = PROJECT_ROOT / "benchmarks" / "comprehensive" / "benchmarks" / f"{mod}.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    sweep = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_run_with_thread_scaling"),
        None,
    )
    assert sweep is not None, f"{mod}: _run_with_thread_scaling not found — retarget this test"

    # Every f-string key built inside the sweep must be thread-qualified.
    bare = []
    for node in ast.walk(sweep):
        if not isinstance(node, ast.JoinedStr):
            continue
        rendered = "".join(
            v.value if isinstance(v, ast.Constant) else "{}"
            for v in node.values
        )
        if rendered.endswith("_peak_rss_mb") or rendered.endswith("_wall_s"):
            if "_t{}" not in rendered:
                bare.append(f"line {node.lineno}: {rendered}")
    assert not bare, (
        f"{mod}'s thread sweep emits un-qualified metric keys {bare}; the bare "
        f"`streaming_peak_rss_mb` is the floored key and mixing thread counts into "
        f"its median defeats GATED_READER_THREADS"
    )


def test_export_streaming_bounds_its_materialize_arm_too():
    """The ingest twin caps the materialize arm above 1M cells; so must this one.

    `to_h5ad(..., stream=False)` calls `read_all_csr_shards_filtered()` — 13.98 GB
    resident on census_1m, scaling linearly — and `estimate_memory_gb` has no arm
    for either streaming benchmark, so the job falls through to `2x` the h5ad size
    and would be under-provisioned at census_5m/10m. Registration is what makes a
    tiered capture schedule it. The asymmetry was found by review
    (Cursor Agent - Grok 4.6 High) on PR #451.
    """
    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs
    from benchmarks.comprehensive.benchmarks import export_streaming as es

    assert es.MATERIALIZE_MAX_N_OBS == cs.MATERIALIZE_MAX_N_OBS, (
        "the two streaming benchmarks should cap the same arm at the same scale"
    )
    for m in (cs, es):
        assert m._skip_materialize_reason(m.MATERIALIZE_MAX_N_OBS) is None
        reason = m._skip_materialize_reason(m.MATERIALIZE_MAX_N_OBS + 1)
        assert reason and "streaming" in reason


# ---------------------------------------------------------------------------
# Published numbers must match the results tracked to back them
# ---------------------------------------------------------------------------


def test_performance_doc_streaming_table_matches_the_tracked_json():
    """`docs/performance.md`'s census_1m streaming table must equal the medians in
    the tracked raw JSON.

    `docs/benchmark_manifest.md` requires every user-visible number to be backed by
    a tracked result. It was not enough: round 1 force-added the JSON and left the
    prose quoting a *different* capture, with two compounding errors that all three
    round-2 reviewers caught —

    * the GB figures divided MB by 1000, so 5 893 MB was published as "5.79 GB"
      (it is 5.75 GiB); and
    * two of three were the *minimum* of the three runs, not the median.

    The table is now quoted in MB, the same unit the JSON and `thresholds.yaml`
    use, which removes the conversion that produced both. This test is the part
    that keeps them tied together — a manifest rule nothing checks is the same
    shape as a floor nothing can trip.
    """
    import json
    import re
    import statistics

    raw = PROJECT_ROOT / "benchmarks" / "comprehensive" / "results" / "raw"
    doc = (PROJECT_ROOT / "docs" / "performance.md").read_text()

    # (doc row label, results file, scenario key in runs[].extra)
    rows = [
        ("streaming (`pyscx.from_h5ad`), `reader_threads=4`",
         "conversion_streaming__scx_streaming_vs_materialize__census_1m.json",
         "streaming"),
        ("streaming, default parallelism (16 readers)",
         "conversion_streaming__scx_streaming_vs_materialize__census_1m.json",
         "streaming_default_threads"),
        ("materialise (`pyscx.from_anndata`)",
         "conversion_streaming__scx_streaming_vs_materialize__census_1m.json",
         "materialize"),
    ]

    checked = 0
    for label, fname in {(l, f) for l, f, _ in rows}:
        assert (raw / fname).is_file(), (
            f"{fname} is not tracked, so the numbers citing it cannot be audited "
            f"from a fresh checkout — force-add it (see docs/benchmark_manifest.md)"
        )

    for label, fname, scenario in rows:
        d = json.loads((raw / fname).read_text())
        runs = [r for r in d["runs"] if r.get("extra", {}).get("scenario") == scenario]
        assert runs, f"{fname} has no runs for scenario {scenario!r}"
        want_wall = statistics.median(r["wall_s"] for r in runs)
        want_rss = statistics.median(r["peak_rss_mb"] for r in runs)

        # `| <label> | 25.37 s | **2 342 MB** |`, tolerating bold and thin spaces.
        pat = re.escape(label) + r"\s*\|\s*\*{0,2}([\d.]+)\s*s\*{0,2}\s*\|\s*\*{0,2}([\d\s,]+)\s*MB"
        m = re.search(pat, doc)
        assert m, f"no MB-denominated table row for {label!r} in docs/performance.md"
        got_wall = float(m.group(1))
        got_rss = float(m.group(2).replace(" ", "").replace(",", "").replace(" ", ""))

        assert abs(got_wall - want_wall) < 0.05, (
            f"{label}: doc says {got_wall} s, tracked median is {want_wall:.2f} s"
        )
        assert abs(got_rss - want_rss) < 1.0, (
            f"{label}: doc says {got_rss} MB, tracked median is {want_rss:.0f} MB "
            f"(min {min(r['peak_rss_mb'] for r in runs):.0f}, "
            f"max {max(r['peak_rss_mb'] for r in runs):.0f}) — publishing the min "
            f"instead of the median is how this drifted before"
        )
        checked += 1
    assert checked == 3, checked


@pytest.mark.parametrize("mod", ["conversion_streaming", "export_streaming"])
def test_thread_sweep_above_the_cap_is_refused_not_silently_uncapped(mod: str, monkeypatch):
    """Drive `run()` — do not just assert the constants exist.

    Round 1 added the export cap with a test that checked `MATERIALIZE_MAX_N_OBS`
    and `_skip_materialize_reason`, and the sweep path stayed uncapped because
    nothing exercised it: `_run_thread_count_subprocess` unconditionally appends
    the materialize arm. All three round-2 reviewers found the bypass, and round 3
    pointed out the replacement test was still the same source-shape check
    (codex - gpt-5.6-sol, Cursor Agent - Grok 4.6 High).

    So this drives the real entry point with the sweep env var set and a dataset
    above the cap, and asserts it refuses *before* touching any input — which is
    also why the guard sits ahead of input resolution.
    """
    import importlib

    from benchmarks.comprehensive.config import DATASETS, FormatVariant

    m = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{mod}")
    big = next(
        (d for d in DATASETS.values()
         if d.n_obs > m.MATERIALIZE_MAX_N_OBS and not d.multimodal),
        None,
    )
    assert big is not None, "no registered dataset above the cap — retarget this test"

    monkeypatch.setenv(m._THREAD_COUNTS_ENV, "1,4")
    # If the guard were removed, these would be reached and would do real work;
    # blowing up here is a much clearer failure than a 140 GB allocation.
    monkeypatch.setattr(
        m, "_run_with_thread_scaling",
        lambda *a, **k: pytest.fail("sweep ran above the materialize cap"),
    )
    if hasattr(m, "_run_arm_subprocess"):
        monkeypatch.setattr(
            m, "_run_arm_subprocess",
            lambda *a, **k: pytest.fail("a benchmark arm ran above the cap"),
        )

    fmt = FormatVariant("SCX (auto)", "scx_auto", "primary", "scx_runner", {})
    with pytest.raises(ValueError, match=r"thread-scaling|THREAD_COUNTS"):
        m.run(dataset=big, format_variant=fmt, n_runs=1)
