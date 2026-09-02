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

    def fake_arm(h5ad_path, n_runs, scenario, thread_count=None,
                 reader_threads=None, extra_kwargs=None):
        calls.append((scenario, reader_threads, extra_kwargs))
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
    # `dataset_name` is None here, so no `_EXTRA_ARMS` entry applies and the
    # three base arms are all that run.
    assert calls == [
        ("streaming", cs.GATED_READER_THREADS, None),
        ("streaming", None, None),
        ("materialize", None, None),
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


def test_conversion_streaming_extra_arms_are_dataset_scoped_and_pinned():
    """`_EXTRA_ARMS` must reach the worker, and only on the right datasets.

    The two extra arms (`csc_always`, `index_preset_cellxgene`) are the same
    `from_h5ad` call with one conversion option changed, and the whole reason
    they exist is that the default arm passes **no** conversion options — so the
    bound `streaming_peak_rss_mb` enforces is measured in a configuration real
    callers do not always use.

    Two properties, both easy to lose in a refactor and neither visible in the
    result JSON if lost:

    1. the arm's kwargs actually reach `_timed_streaming` (a dropped
       `extra_kwargs` would run the *default* conversion under an arm labelled
       `csc_always`, i.e. measure the wrong thing under the right name); and
    2. the arm is pinned to `GATED_READER_THREADS`, so its number is comparable
       with the default arm's rather than with the runner's core count.

    Scope matters for a third reason recorded in the module: `csc_always` is
    expected to breach its ceiling and so needs a justification, and
    justification suppression is *whole-triple* — so it must not run on a
    dataset that carries a live floor.
    """
    import pathlib

    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs
    from benchmarks.comprehensive.results import BenchmarkResult

    def drive(dataset_name):
        calls = []

        def fake_arm(h5ad_path, n_runs, scenario, thread_count=None,
                     reader_threads=None, extra_kwargs=None):
            calls.append((scenario, reader_threads, extra_kwargs))
            # The parent now raises when an extra arm shows no output effect,
            # so the mock has to model a worker that honoured its kwargs — or
            # the happy path fails for the wrong reason. Each guard's own red
            # case is asserted separately below.
            kw = extra_kwargs or {}
            return [{
                "scenario": scenario, "run_idx": 0, "wall_s": 1.0,
                "peak_rss_mb": 123.0, "reader_threads": reader_threads,
                "structural": {
                    "n_obs": 1, "n_vars": 1, "nnz": 1, "shard_count": 1,
                    "has_csc": 1 if kw.get("csc") else 0,
                },
                # An index preset that took effect writes more bytes than the
                # default arm; that difference is the observable the parent
                # checks, since `Experiment` exposes no `has_obs_index`.
                "output_bytes": 20 if kw.get("index_preset") else 10,
            }]

        original = cs._run_arm_subprocess
        try:
            cs._run_arm_subprocess = fake_arm
            result = cs._run_isolated(
                pathlib.Path("/nonexistent.h5ad"), 1,
                BenchmarkResult(
                    benchmark="conversion_streaming",
                    format="scx_streaming_vs_materialize",
                    dataset=dataset_name or "unit",
                    metadata={"scenarios": []},
                ),
                None,
                dataset_name,
            )
        finally:
            cs._run_arm_subprocess = original
        return calls, result

    # In scope for both extras.
    calls, result = drive("tabula_sapiens_100k")
    by_kwargs = {tuple(sorted((k, repr(v)) for k, v in (kw or {}).items())): rt
                 for _, rt, kw in calls}
    assert (("csc", "'always'"),) in by_kwargs, calls
    assert (("index_preset", "'cellxgene'"),) in by_kwargs, calls
    assert by_kwargs[(("csc", "'always'"),)] == cs.GATED_READER_THREADS
    assert set(result.metadata["extra_arms"]) == {
        "csc_always", "index_preset_cellxgene"
    }
    labels = {r.extra["scenario"] for r in result.runs}
    assert "csc_always" in labels and "index_preset_cellxgene" in labels
    for label in ("csc_always", "index_preset_cellxgene"):
        rows = [r for r in result.runs if r.extra["scenario"] == label]
        assert rows and all(f"{label}_peak_rss_mb" in r.extra for r in rows)
        assert all("streaming_peak_rss_mb" not in r.extra for r in rows), (
            f"{label} must not emit the gated key, or the census floor's median "
            f"would mix arms"
        )

    # census_1m carries the live `streaming_peak_rss_mb` floor, so `csc_always`
    # — which needs a whole-triple justification — must NOT run there.
    calls, result = drive("census_1m")
    assert all((kw or {}).get("csc") is None for _, _, kw in calls), calls
    assert result.metadata["extra_arms"] == ["index_preset_cellxgene"]
    assert "csc_always" in result.metadata["extra_arms_skipped"]

    # A dataset in neither scope runs the three base arms only.
    calls, result = drive("pbmc3k")
    assert [c[0] for c in calls] == ["streaming", "streaming", "materialize"]
    assert result.metadata["extra_arms"] == []


def test_mtx_export_is_scoped_out_of_the_census_tiers():
    """The size cap has to stop the *scheduler*, not just `run()`.

    `mtx_export` returns `None` above `MAX_N_OBS`, but `run_parallel` only drops
    cells at cohort-build time. With `SUPPORTED_FORMATS` alone, a `--tier full`
    capture submitted `mtx_export x {census_500k, census_1m}` — 88 GB and 176 GB
    of requested memory plus their Phase-A conversions — and the job returned
    `None` on its first line. `--tier xl` added census_5m at 864 GB.

    This is the trap `_bench_format_dataset_scope`'s own docstring records
    ("they stubbed out-of-scope datasets *inside* `run()` ... eight wasted GPU
    jobs per full accel capture"). Found by review (codex - gpt-5.6-terra and
    Cursor Agent - Grok 4.6 High, independently).

    Exercises the orchestrator's filter, not only the module constant — a
    declaration test alone would pass with the scope never consulted.
    """
    import importlib

    from benchmarks.comprehensive.benchmarks import mtx_export as mx
    from benchmarks.comprehensive.config import DATASETS

    rp = importlib.import_module("benchmarks.comprehensive.scripts.run_parallel")
    scope = rp._bench_format_dataset_scope("mtx_export")
    assert "scx_auto" in scope, (
        "mtx_export declares no FORMAT_DATASET_SCOPE, so every unimodal "
        "scx_auto dataset is schedulable regardless of MAX_N_OBS"
    )
    allowed = scope["scx_auto"]

    for name in ("pbmc3k", "tabula_sapiens_100k"):
        assert name in allowed, name
    for name in ("census_500k", "census_1m", "census_5m", "census_10m"):
        assert name not in allowed, (
            f"{name} is above MAX_N_OBS={mx.MAX_N_OBS:,} yet still schedulable; "
            f"the job would request "
            f"{__import__('benchmarks.comprehensive.config', fromlist=['x']).estimate_memory_gb(DATASETS[name], 'scx_auto', 'mtx_export')} GB "
            f"and then return None"
        )

    # Derived from MAX_N_OBS, not hand-listed: a literal list would stop
    # matching the cap the first time either changed.
    assert allowed == frozenset(
        n for n, ds in DATASETS.items()
        if not ds.multimodal and ds.n_obs <= mx.MAX_N_OBS
    )


@pytest.fixture
def tiny_mtx_source(tmp_path):
    """`(dataset, format_variant, converted_path)` over a synthesised SCX file.

    Deliberately **not** the staged `pbmc3k_auto.scx`. An earlier version of
    these tests used it and skipped when it was absent, which put the durable
    evidence for the two MTX guards exactly where it was least useful: a
    fixtureless checkout could delete the production `if` and keep the suite
    green. Found by review (codex - gpt-5.6-terra, Antigravity - Gemini 3.7
    Flash, Cursor Agent - Grok 4.6 High).

    120 x 40 integer counts is enough to export, delete from, and re-ingest in
    well under a second, and it keeps the guards under test rather than the
    fixture.
    """
    anndata = pytest.importorskip("anndata")
    pyscx = pytest.importorskip("pyscx")
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")

    from benchmarks.comprehensive.config import ALL_FORMATS, DatasetConfig

    n_obs, n_vars = 120, 40
    rng = np.random.default_rng(0)
    x = sparse.csr_matrix(
        (rng.random((n_obs, n_vars)) < 0.3) * rng.integers(1, 50, (n_obs, n_vars))
    ).astype("float32")
    adata = anndata.AnnData(X=x)
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    adata.var_names = [f"gene{j}" for j in range(n_vars)]
    scx = tmp_path / "tiny.scx"
    pyscx.from_anndata(adata, str(scx))

    ds = DatasetConfig(
        id="TEST", name="tiny_mtx", n_obs=n_obs, n_vars=n_vars,
        protocol="synthetic", source="test fixture", approx_h5ad_mb=1,
    )
    fmt = next(f for f in ALL_FORMATS if f.key == "scx_auto")
    return ds, fmt, scx


def test_mtx_deletion_arm_refuses_an_export_that_ignored_the_keep_mask(tiny_mtx_source):
    """The row-count comparison has to be a *test*, not a one-off measurement.

    Round 1 found that the deletion arm recorded `mtx_rows_written` and compared
    it to nothing, so an export ignoring the deletion vector wrote a normal
    successful result. Round 2 pointed out that the fix shipped with the raise
    but no test — the production `if` could be deleted again and the suite would
    stay green. Found by review (codex - gpt-5.6-terra, Cursor Agent - Grok 4.6
    High).

    The mutant is an export that drops nothing: `_row_count` reporting the full
    obs count after cells were marked deleted.
    """
    from benchmarks.comprehensive.benchmarks import mtx_export as mx

    ds, fmt, converted = tiny_mtx_source
    original = mx._row_count
    try:
        mx._row_count = lambda out_dir: ds.n_obs
        with pytest.raises(RuntimeError, match="keep mask was not applied"):
            mx.run(dataset=ds, format_variant=fmt, n_runs=1,
                   converted_path=converted)
    finally:
        mx._row_count = original


def test_mtx_roundtrip_refuses_an_ingest_that_lost_an_entry(tiny_mtx_source):
    """Same for the round-trip shape/nnz comparison.

    The mutant is an ingest that silently drops one non-zero — patched at the
    reopen rather than by corrupting a file, so the test exercises the
    comparison itself and stays a few seconds long.
    """
    from benchmarks.comprehensive.benchmarks import mtx_export as mx

    pyscx = pytest.importorskip("pyscx")
    ds, fmt, converted = tiny_mtx_source
    src = str(converted)
    real_open = pyscx.open

    class _Lossy:
        """Reports one fewer non-zero than the file holds."""

        def __init__(self, exp):
            self._exp = exp

        def __getattr__(self, name):
            value = getattr(self._exp, name)
            return value - 1 if name == "nnz" else value

    try:
        # Truthful for the source read; lossy for the re-imported output.
        pyscx.open = lambda path: (
            real_open(path) if str(path) == src else _Lossy(real_open(path))
        )
        with pytest.raises(RuntimeError, match="round-trip changed the matrix"):
            mx.run(dataset=ds, format_variant=fmt, n_runs=1,
                   converted_path=converted)
    finally:
        pyscx.open = real_open


def test_csc_always_arm_refuses_a_result_with_no_sidecar():
    """The `csc_always` premise is enforced, not merely recorded.

    `_structural_summary` reports `n_csc_shards`, but nothing reads `metadata`,
    so storing it there was not a check: if `csc="always"` were dropped on
    either subprocess hop, the arm would still emit `csc_always_peak_rss_mb`
    and pass — having timed the *default* conversion under the CSC label. That
    is the failure mode the arm exists to rule out, so it has to raise.
    """
    import pathlib

    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs
    from benchmarks.comprehensive.results import BenchmarkResult

    def fake_arm(h5ad_path, n_runs, scenario, thread_count=None,
                 reader_threads=None, extra_kwargs=None):
        # A worker that ignored the kwargs: right label, no sidecar.
        return [{
            "scenario": scenario, "run_idx": 0, "wall_s": 1.0,
            "peak_rss_mb": 123.0, "reader_threads": reader_threads,
            "structural": {
                "n_obs": 1, "n_vars": 1, "nnz": 1, "shard_count": 1,
                "has_csc": 0,
            },
            "output_bytes": 10,
        }]

    original = cs._run_arm_subprocess
    try:
        cs._run_arm_subprocess = fake_arm
        with pytest.raises(RuntimeError, match="csc_always"):
            cs._run_isolated(
                pathlib.Path("/nonexistent.h5ad"), 1,
                BenchmarkResult(
                    benchmark="conversion_streaming",
                    format="scx_streaming_vs_materialize",
                    dataset="tabula_sapiens_100k",
                    metadata={"scenarios": []},
                ),
                None,
                "tabula_sapiens_100k",
            )
    finally:
        cs._run_arm_subprocess = original


@pytest.mark.parametrize(
    "indexed_bytes, expect",
    [
        (10, "no predicate-index sections"),   # same size as the default arm
        (5, "no predicate-index sections"),    # smaller, the measured pbmc3k shape
        (None, "could not be verified"),       # size missing on one side
    ],
)
def test_index_preset_arm_refuses_a_result_with_no_index_effect(
    indexed_bytes, expect
):
    """`index_preset_cellxgene` has to fail **closed**, like the CSC arm.

    `Experiment` exposes no `has_obs_index`, so the guard compares output sizes:
    an index that took effect writes more bytes than the default arm on the same
    input. Two ways it could be satisfied wrongly, both covered here:

    * the sizes are equal or the indexed arm is *smaller* — which is the shape
      actually measured on pbmc3k, where the preset's columns are absent
      (4,379,713 B against the default arm's 4,379,851); and
    * a size is missing, which an earlier version treated as "nothing to
      compare, carry on". A comparison that could not be made is not a
      comparison that passed.

    Found by review (Antigravity - Gemini 3.7 Flash, Cursor Agent - Grok 4.6
    High): the round-2 fix added the guard and no red test, so deleting the
    `if` left the suite green.
    """
    import pathlib

    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs
    from benchmarks.comprehensive.results import BenchmarkResult

    def fake_arm(h5ad_path, n_runs, scenario, thread_count=None,
                 reader_threads=None, extra_kwargs=None):
        kw = extra_kwargs or {}
        rec = {
            "scenario": scenario, "run_idx": 0, "wall_s": 1.0,
            "peak_rss_mb": 123.0, "reader_threads": reader_threads,
            "structural": {
                "n_obs": 1, "n_vars": 1, "nnz": 1, "shard_count": 1,
                "has_csc": 1 if kw.get("csc") else 0,
            },
        }
        if kw.get("index_preset"):
            if indexed_bytes is not None:
                rec["output_bytes"] = indexed_bytes
        else:
            rec["output_bytes"] = 10
        return [rec]

    original = cs._run_arm_subprocess
    try:
        cs._run_arm_subprocess = fake_arm
        with pytest.raises(RuntimeError, match=expect):
            cs._run_isolated(
                pathlib.Path("/nonexistent.h5ad"), 1,
                BenchmarkResult(
                    benchmark="conversion_streaming",
                    format="scx_streaming_vs_materialize",
                    dataset="tabula_sapiens_100k",
                    metadata={"scenarios": []},
                ),
                None,
                "tabula_sapiens_100k",
            )
    finally:
        cs._run_arm_subprocess = original


def test_sweep_companion_arms_enforce_the_same_premises():
    """The thread-sweep path must not be the unguarded way in.

    `_run_extra_arms_once` was added so a sweep capture does not silently lose
    the non-default arms — and it recorded their timings while discarding
    `structural` / `output_bytes`, which reopened on this path the exact
    "wrong path, right label" hole the default path had just closed. A sweep on
    tabula could emit `csc_always_peak_rss_mb` after a worker reported no
    sidecar. Found by all three reviewers.

    The CSC premise is checkable here because it needs only this arm's own
    record. The index premise is not: it compares against the default arm's
    output size, which the sweep writes *after* these arms run, so the caller
    defers it — covered by `test_index_preset_arm_refuses_a_result_with_no_index_effect`.
    """
    import pathlib

    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs
    from benchmarks.comprehensive.results import BenchmarkResult

    def no_sidecar(h5ad_path, n_runs, scenario, thread_count=None,
                   reader_threads=None, extra_kwargs=None):
        # A worker that ignored `csc="always"`: right label, no sidecar.
        return [{
            "scenario": scenario, "run_idx": 0, "wall_s": 1.0,
            "peak_rss_mb": 123.0, "reader_threads": reader_threads,
            "structural": {
                "n_obs": 1, "n_vars": 1, "nnz": 1, "shard_count": 1,
                "has_csc": 0,
            },
            "output_bytes": 10,
        }]

    result = BenchmarkResult(
        benchmark="conversion_streaming",
        format="scx_streaming_vs_materialize",
        dataset="tabula_sapiens_100k",
        metadata={"scenarios": []},
    )
    original = cs._run_arm_subprocess
    try:
        cs._run_arm_subprocess = no_sidecar
        with pytest.raises(RuntimeError, match="csc_always"):
            cs._run_extra_arms_once(
                pathlib.Path("/nonexistent.h5ad"), result, "tabula_sapiens_100k"
            )
    finally:
        cs._run_arm_subprocess = original


def test_conversion_streaming_worker_threads_extra_kwargs_through():
    """The mocked test above cannot see a broken `_WORKER_SCRIPT`.

    `_run_arm_subprocess` is what the other test replaces, so a worker that
    never parsed `sys.argv[5]` — or parsed it and never passed it to
    `_timed_streaming` — would leave that test green while every extra arm ran
    the default conversion. Checked on the worker source, which is a
    `textwrap`-dedented string and therefore invisible to any import-time check.
    Found by review (Cursor Agent - Grok 4.6 High).
    """
    from benchmarks.comprehensive.benchmarks import conversion_streaming as cs

    script = cs._WORKER_SCRIPT
    assert "extra_kwargs = json.loads(sys.argv[5])" in script, (
        "the worker does not parse the extra-arm kwargs out of argv"
    )
    # And it must reach the timed call, not just be parsed. Sliced to the
    # *matching* close paren: the call is multi-line, so stopping at the first
    # `)` lands inside `Path(h5ad_path)` and the check passes vacuously.
    start = script.index("_timed_streaming(") + len("_timed_streaming(")
    depth, end = 1, start
    while depth:
        if script[end] == "(":
            depth += 1
        elif script[end] == ")":
            depth -= 1
            if depth == 0:
                break
        end += 1
    args = script[start:end]
    assert "extra_kwargs" in args, (
        f"the worker parses extra_kwargs but does not pass them to "
        f"_timed_streaming: args were {args!r}"
    )


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
# Registration, and the rest of the "peak_rss_mb is a peak" family
# ---------------------------------------------------------------------------


def test_every_runnable_benchmark_module_is_registered():
    """The converse of `test_every_floored_benchmark_is_in_all_benchmarks`.

    That test catches a *threshold* whose benchmark cannot be scheduled. This
    one catches the earlier mistake: a benchmark module added without a line in
    `ALL_BENCHMARKS`. Nothing fails at that point — the module imports, its
    tests pass, and `run_parallel.py --benchmarks <name>` even runs it — but
    `capture_baseline.py` rejects the off-list name, so no tiered capture and no
    gate can reach it. `conversion_streaming` and `export_streaming` sat in
    exactly that state for months (see `benchmarks/__init__.py`), and the only
    reason it surfaced was that someone tried to add a threshold.

    A module is "runnable" here if it defines a top-level `run`. Private
    helpers (`_pert_synth`) and `__init__` are excluded by name, which is the
    same convention the package's own docstring uses.
    """
    import ast

    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    bench_dir = PROJECT_ROOT / "benchmarks" / "comprehensive" / "benchmarks"
    assert bench_dir.is_dir(), f"{bench_dir} moved — retarget this test"

    runnable = set()
    for path in sorted(bench_dir.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        if any(
            isinstance(n, ast.FunctionDef) and n.name == "run"
            for n in tree.body
        ):
            runnable.add(path.stem)

    assert len(runnable) > 20, (
        f"only {len(runnable)} runnable benchmark modules found under "
        f"{bench_dir} — the discovery above is probably broken, and the "
        f"assertion below would be vacuous"
    )

    unregistered = sorted(runnable - set(ALL_BENCHMARKS))
    assert not unregistered, (
        f"benchmark modules with a top-level run() that are absent from "
        f"ALL_BENCHMARKS: {unregistered}. capture_baseline.py rejects off-list "
        f"--benchmarks names and both orchestrators derive their lists from it, "
        f"so no tiered capture can schedule these and no threshold on them could "
        f"ever fire. Add them to benchmarks/comprehensive/benchmarks/__init__.py."
    )


def test_no_floor_keys_off_a_reserved_add_run_parameter():
    """A threshold on `wall_s` / `peak_rss_mb` / `user_s` / `sys_s` is unreadable.

    Those four are **named parameters** of `BenchmarkResult.add_run`, so they
    land on the `RunRecord` itself and never reach `runs[].extra` —
    and `_load_current_raw_metric` reads `extra` only. A floor keyed to one of
    them resolves `None` on every run, which the gate reports as a
    missing-metric violation. It cannot be satisfied by any benchmark, however
    correct.

    Three `grouped_sort` ceilings were in exactly that state, and the reason
    nobody noticed is the second half of the trap: their datasets
    (chemogenetic_rgfp / replogle_k562 / tahoe_c38) are outside every tier, so a
    default gate run never produced the triple and `check_absolute_floors`
    skipped it *silently*. An unsatisfiable floor and an unreachable triple
    cancel into something that reads exactly like coverage.

    The fix is a non-reserved name in `extra` (`grouped_peak_rss_mb`, or the
    house-style sparse `peak_rss_mb__<scenario>`), never a rename of the
    parameter.
    """
    import inspect

    from benchmarks.comprehensive.results import BenchmarkResult

    sig = inspect.signature(BenchmarkResult.add_run)
    reserved = {
        name for name, prm in sig.parameters.items()
        if prm.kind is not prm.VAR_KEYWORD and name != "self"
    }
    assert "peak_rss_mb" in reserved and "wall_s" in reserved, (
        f"premise: add_run's named parameters were {sorted(reserved)} — if the "
        f"signature changed, retarget this test rather than deleting it"
    )

    raw = yaml.safe_load(THRESHOLDS.read_text())
    offenders = [
        f"{f['benchmark']}/{f['format']}/{f['dataset']}:{f['metric']}"
        for f in (raw.get("absolute_floors") or [])
        if f["metric"] in reserved
    ]
    assert not offenders, (
        f"these floors key off a reserved add_run parameter and can never read "
        f"a value: {offenders}. Emit the number under a different name in "
        f"`extra` and point the floor at that."
    )


def _bench_ast(name: str):
    import ast

    path = PROJECT_ROOT / "benchmarks" / "comprehensive" / "benchmarks" / f"{name}.py"
    return ast.parse(path.read_text(), filename=str(path)), path


def _calls_named(tree, *names: str) -> list[int]:
    """Line numbers of calls to any of *names*.

    Parsed, not grepped, and for the reason this file already records under
    `test_no_benchmark_passes_extra_as_a_dict`: a regex cannot tell a call from
    a sentence. Each of these modules now carries a docstring *explaining* the
    reading it stopped taking, so a text search matches the explanation and the
    test fails on its own prose. `ast` sees calls only.
    """
    import ast

    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        got = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if got in names:
            out.append(node.lineno)
    return out


def _sampler_context_lines(tree) -> list[int]:
    """Line numbers of `with PeakRssSampler() as ...:` statements."""
    import ast

    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "PeakRssSampler":
                out.append(node.lineno)
    return out


@pytest.mark.parametrize("name", ["grouped_sort", "fragment_ops"])
def test_mutating_op_benchmarks_use_the_true_peak_sampler(name: str):
    """`_time_op` has to bracket the op, not sample after it.

    Both modules reported `current_rss_mb()` taken *after* `fn` returned and
    said so in their own docstrings ("not a true peak — sufficient for detecting
    gross regressions"). It is not sufficient for the ops these benchmarks time:
    `append` reads the whole input CSR before re-encoding, `compact` rewrites
    every section, and a grouped convert holds a reference-group buffer — each
    allocates and frees a transient that is gone by the time a post-op sample
    lands.

    Checked on the source because the alternative is running a grouped convert
    on a real Perturb-seq file, and the defect is entirely in where the sample
    is taken.
    """
    tree, path = _bench_ast(name)

    assert _sampler_context_lines(tree), (
        f"{path.name} never enters a `with PeakRssSampler()` block. Importing "
        f"the name is not using it."
    )
    leftover = _calls_named(tree, "current_rss_mb", "_current_rss_mb")
    assert not leftover, (
        f"{path.name} still calls current_rss_mb() at lines {leftover}; a "
        f"post-op instantaneous reading cannot see a transient, which is the "
        f"only thing these ops allocate."
    )


def test_multimodal_training_samples_each_epoch_not_the_process_lifetime():
    """`ru_maxrss` is a real high-water mark — of the wrong region.

    It is scoped to the whole process, so a peak from the time-to-first-batch
    reps or from a previous epoch's eager `read_h5mu` was re-reported as this
    epoch's. The `max(rss_before, rss_after)` that guarded it could never fire:
    `ru_maxrss` is monotone, so `before <= after` always, and the max was a
    no-op dressed as a safeguard.

    Three epoch runners (SCX, h5mu, zarr) share the shape, so the sampler has to
    appear in all three — asserting merely that the module mentions it would
    pass with two of them still on `ru_maxrss`.
    """
    import ast

    tree, path = _bench_ast("multimodal_training")

    sampler_lines = _sampler_context_lines(tree)
    assert len(sampler_lines) >= 3, (
        f"{path.name} enters `with PeakRssSampler()` only at {sampler_lines}; "
        f"all three epoch runners (scx / h5mu / zarr_mudata) need it"
    )

    maxrss = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "ru_maxrss"
    ]
    assert not maxrss, (
        f"{path.name} still reads ru_maxrss at lines {maxrss}; that is a "
        f"process-lifetime high-water mark, not this epoch's peak"
    )

    monotone_max = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "max"
        and [getattr(a, "id", None) for a in node.args] == ["rss_before", "rss_after"]
    ]
    assert not monotone_max, (
        f"the monotone-ru_maxrss no-op is back at lines {monotone_max}"
    )


def test_full_fixture_path_is_declared_once():
    """The `.raw` + obsm + layer fixture has one home.

    Both arms that read it (`export_streaming`'s `streaming_full`,
    `fragment_ops`' `compact_full`) go through `DatasetConfig.scx_full_path`
    rather than each rebuilding `DATA_DIR / f"{name}_full.scx"`. A second
    spelling is how the two arms would end up measuring different files.
    """
    import ast

    from benchmarks.comprehensive.config import DATASETS

    ds = DATASETS["tabula_sapiens_100k"]
    assert ds.scx_full_path.name == "tabula_sapiens_100k_full.scx"

    bench_dir = PROJECT_ROOT / "benchmarks" / "comprehensive" / "benchmarks"
    offenders = []
    for path in sorted(bench_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            # An f-string that builds the fixture name by hand.
            if isinstance(node, ast.JoinedStr):
                rendered = "".join(
                    v.value if isinstance(v, ast.Constant) else "{}"
                    for v in node.values
                )
                if rendered.endswith("_full.scx"):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"these build the full-fixture path by hand instead of using "
        f"DatasetConfig.scx_full_path: {offenders}"
    )


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


# ---------------------------------------------------------------------------
# The second way a floor becomes unreachable: a justification over it
# ---------------------------------------------------------------------------

#: `(benchmark, format, dataset)` triples where an unscoped, whole-triple
#: suppression is the deliberate answer, with the reason. Everything else must
#: either scope its justification with `metrics:` or drop the floor.
#:
#: Kept as data with a reason per entry rather than as a bare skip list: the
#: point of the guard is that suppressing a floor is a decision someone made
#: on purpose, so each one is written down.
_DELIBERATE_WHOLE_TRIPLE_SUPPRESSIONS: dict[tuple[str, str, str], str] = {
    ("accel_leiden", "accel_leiden__pyscx_gpu", "pbmc3k"): (
        "cuGraph Leiden diverges from leidenalg on real graphs — documented in "
        "CLAUDE.md's Known Limitations. Both the ARI floor and the timing row "
        "are downstream of the same divergence, so the whole triple is the unit."
    ),
    ("accel_hvg", "accel_hvg__pyscx_cpu", "pbmc3k"): (
        "hvg_overlap_vs_scanpy is a deterministic 0.9891 against a 0.99 floor on "
        "the 2700-cell fixture — a borderline tie-break miss, and the only floor "
        "on the triple, so scoping would change nothing."
    ),
    ("cloud_push", "scx_auto", "tabula_sapiens_100k"): (
        "throughput_mbps to the shared GCS bucket is network- and "
        "contention-dependent; the floor and the timing row move together with "
        "cluster load."
    ),
}


def _active_suppressions():
    """The committed justifications, as the gate loads them."""
    import sys

    scripts = PROJECT_ROOT / "benchmarks" / "comprehensive" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import _justifications  # noqa: PLC0415

    just_dir = (
        PROJECT_ROOT / "benchmarks" / "comprehensive" / "results" / "justifications"
    )
    suppression, parsed = _justifications.load_active_triples(just_dir)
    return suppression, parsed, _justifications


def test_no_floor_is_fully_suppressed_by_an_active_justification():
    """A justification must not silently disarm a floor it says nothing about.

    Suppression covers both gate surfaces — relative regressions and absolute
    floors — on the same `(benchmark, format, dataset)`. That is right for a
    triple that is wholly known-bad, and wrong for the commonest real case: a
    *composition* change. `capture_baseline.archive_raw_results` writes one
    `peak_rss_mb_median` and one `median_wall_s` per triple, pooled across
    every run in the file, so adding an arm to an existing benchmark shifts
    both by construction and needs a justification — which, unscoped, then
    switches off every floor on that triple.

    Both committed pooled-median justifications were in that state:

      * `fragment_ops_compact_full_arm_raises_pooled_medians.md` suppressed the
        `peak_rss_mb__compact_full max: 4096` ceiling **added in the same
        commit**. The file's own closing note warned about the hazard without
        noticing the condition was already true.
      * `cellset_gather_s512_raises_pooled_rss.md` suppressed all 15
        `cellset_gather` floors, which `thresholds.yaml`'s Deferred item 7
        calls "NOW ACTIVE … all live".

    16 floors, reading as coverage and providing none — the same shape as
    `test_no_floor_keys_off_a_reserved_add_run_parameter` above, and just as
    invisible: the gate prints "Absolute-floor violations: 0 (N
    justification-suppressed)" and exits 0.

    The fix is a `metrics:` (or per-entry `metric:`) scope naming what the file
    actually explains, never deleting the floor.
    """
    suppression, parsed, _justifications = _active_suppressions()
    raw = yaml.safe_load(THRESHOLDS.read_text())

    by_file: dict[tuple[str, str, str], list[str]] = {}
    for j in parsed:
        if not j.is_active():
            continue
        for triple in j.triples:
            by_file.setdefault(triple, []).append(j.path.name)

    offenders: list[str] = []
    for f in raw.get("absolute_floors") or []:
        triple = (f["benchmark"], f["format"], f["dataset"])
        if triple in _DELIBERATE_WHOLE_TRIPLE_SUPPRESSIONS:
            continue
        # A `None` scope is "every metric on this triple" — precisely the
        # state this test rejects. A triple with a named scope is fine here
        # even when the scope happens to cover some other metric.
        if triple in suppression and suppression[triple] is None:
            offenders.append(
                f"{f['benchmark']}/{f['format']}/{f['dataset']}:{f['metric']} "
                f"(suppressed by {', '.join(sorted(set(by_file.get(triple, ['?']))))})"
            )

    assert not offenders, (
        f"these floors sit under a whole-triple justification and can never "
        f"fire: {sorted(offenders)}. Add a `metrics:` list (or a per-entry "
        f"`metric:`) to the justification naming only what it explains — a "
        f"pooled-median justification should say "
        f"`metrics: [peak_rss_mb_median, median_wall_s]`. If the whole triple "
        f"really is known-bad, add it to "
        f"_DELIBERATE_WHOLE_TRIPLE_SUPPRESSIONS with the reason."
    )


def test_deliberate_suppression_allowlist_has_no_stale_entries():
    """The allowlist above is an exception list, so it must stay honest.

    An entry that no longer corresponds to an active justification *and* a
    floor is dead weight that would silently excuse a future regression on that
    triple. Checked because the alternative — an allowlist nobody prunes — is
    how the exception list becomes the policy.
    """
    suppression, _parsed, _justifications = _active_suppressions()
    raw = yaml.safe_load(THRESHOLDS.read_text())
    floored = {
        (f["benchmark"], f["format"], f["dataset"])
        for f in (raw.get("absolute_floors") or [])
    }

    stale = [
        f"{t} ({reason.split('.')[0][:60]}…)"
        for t, reason in _DELIBERATE_WHOLE_TRIPLE_SUPPRESSIONS.items()
        if t not in floored or suppression.get(t, "absent") is not None
    ]
    assert not stale, (
        f"these allowlist entries no longer describe a whole-triple "
        f"suppression over a live floor: {stale}. Remove them — an exception "
        f"nobody needs is an exception nobody rechecks."
    )


# ---------------------------------------------------------------------------
# PR-02c: the five new arms, each with the premise that makes it mean something
# ---------------------------------------------------------------------------


def test_collate_arm_passes_a_non_empty_encoder_mask():
    """An empty `enc_mask_positions` would measure the wrong branch.

    `collate_cell` builds its withheld-gene `HashSet<i64>` — and pays a probe
    per surviving top-K gene — **only** when `enc_mask_positions` is non-empty.
    An empty array is the perturbation path and takes `None`, skipping the
    allocation OPT-LOADER-4 removes. It is not a small difference: collating
    the same batches with the mask replaced by an empty array measured
    78.7 µs/cell against 120.1, so ~34% of the arm's subject would vanish.

    The trap is that empty is exactly what the only pre-existing Python call
    site (`pyscx/tests/test_fork_safety.py`) passes, so copying that template
    produces a green arm over the wrong code.

    Checked on the source: running the arm needs a real fixture, and the defect
    is entirely in which array is handed to the kernel.
    """
    import ast

    tree, path = _bench_ast("cellset_gather")
    src = path.read_text()

    assert "_COLLATE_MASK_FRACTION" in src, (
        f"{path.name} no longer declares a mask fraction; the collate arm's "
        f"premise is that the mask is non-empty"
    )
    fraction = next(
        (
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                getattr(t, "id", None) == "_COLLATE_MASK_FRACTION"
                for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ),
        None,
    )
    assert isinstance(fraction, (int, float)) and 0.0 < fraction < 1.0, (
        f"_COLLATE_MASK_FRACTION is {fraction!r}; it must withhold some but "
        f"not all query positions. 0 leaves the withheld set empty (the "
        f"HashSet is still built but never hits); 1 masks every gene and every "
        f"cell takes the all-masked fallback instead of the compaction loop."
    )

    # The runtime half: the arm refuses to record a run whose mask withheld
    # nothing, which catches an RNG or dtype change the constant cannot.
    assert "masked_positions == 0" in src, (
        f"{path.name} no longer asserts that the built mask withholds "
        f"positions. The constant above is not sufficient: an all-zero uint8 "
        f"array is non-empty, so the kernel takes the masking branch with an "
        f"empty withheld set and the compaction loop has nothing to skip."
    )

    # `hide_readout` must stay all-zero: a set bit short-circuits the encoder
    # path to a single GENE_MASK token, deleting the sort, mask and crop.
    assert "np.zeros(n_rows, dtype=np.uint8)" in src, (
        f"{path.name}'s collate inputs no longer build an all-zero "
        f"hide_readout; a non-zero entry skips the whole encoder path"
    )


def test_ml_loader_highcard_arm_names_a_categorical_column():
    """The obs-cardinality arm must not be pointed at an integer column.

    OPT-LOADER-6's cost is `PyList::new(py, categories.iter())` — a fresh
    `PyUnicode` per category per batch — in `obs_to_pydict`'s `Categorical`
    arm. An `Int64` column takes the `PyArray1::from_vec` arm instead and pays
    none of it. So the arm's value rests entirely on the column being a
    high-cardinality *dictionary* column.

    The review doc prescribed `obs_columns=["soma_joinid"]`, which on census_1m
    is `int64` with a million distinct values: high cardinality, cheap branch,
    and the arm would have measured nothing while looking exactly right.
    `observation_joinid` is the categorical (957,955 categories, verified
    against `census_1m_auto.scx`).
    """
    import ast

    _tree, path = _bench_ast("ml_loader")
    src = path.read_text()
    spec_src = src[src.index("_OBS_CARDINALITY_SPEC"):]
    spec_src = spec_src[: spec_src.index("\n\n\n")]

    assert "soma_joinid" not in spec_src, (
        "the obs-cardinality arm names `soma_joinid`, which is int64 on "
        "census_1m and takes obs_to_pydict's numpy branch — no category list "
        "is rebuilt, so the arm would measure nothing. Use a categorical "
        "column (`observation_joinid`)."
    )
    assert "observation_joinid" in spec_src, (
        "the high-cardinality arm no longer names a known categorical column"
    )

    # Both halves are needed: the ratio is the signal, and one absolute rate
    # cannot separate "obs projection costs something" from "cardinality costs
    # something".
    for arm in ("raw_obs_lowcard", "raw_obs_highcard"):
        assert arm in spec_src, f"the {arm} arm is gone; the pair is the signal"

    # The declared cardinalities are recorded into runs[].extra so a capture is
    # self-describing rather than requiring a reader to re-probe the fixture.
    mod = ast.parse(src)
    assert any(
        isinstance(node, ast.keyword) and node.arg == "obs_n_categories"
        for node in ast.walk(mod)
    ), "the arm no longer records obs_n_categories into runs[].extra"


def test_sort_by_arm_does_not_emit_the_pooled_grouped_peak_key():
    """`grouped_peak_rss_mb` is flat, so a new arm must not join its median.

    Three `thresholds.yaml` ceilings read `grouped_peak_rss_mb`
    (chemogenetic_rgfp 32000, replogle_k562 16000, tahoe_c38 12000), and
    `_load_current_raw_metric` takes the median across **every** run carrying
    the key. The key is not per-scenario: every timed run in `grouped_sort`
    emitted it. So an arm that also emits it silently changes what those three
    ceilings are measured against — which is the pooled-median trap one level
    down, inside a single benchmark's own extras rather than in `summary.json`.

    The `convert_sort_by` arm therefore reports only its sparse
    `peak_rss_mb__convert_sort_by` / `wall_s__convert_sort_by`.
    """
    _tree, path = _bench_ast("grouped_sort")
    src = path.read_text()

    assert 'reorder="sort_by"' in src or "reorder='sort_by'" in src, (
        f"{path.name} no longer has a sort_by arm"
    )
    # The guard is the conditional emission, not the absence of the key: the
    # grouping arms legitimately still emit it.
    assert 'if reorder != "sort_by":' in src, (
        f"{path.name} emits grouped_peak_rss_mb unconditionally again. The "
        f"sort_by arm must omit it, or the three ceilings' medians shift to "
        f"include an arm they were never calibrated against."
    )
    assert 'f"wall_s__{scenario}"' in src, (
        f"{path.name} does not emit a per-scenario wall key; `wall_s` is a "
        f"reserved add_run parameter and never reaches runs[].extra"
    )


def test_fragment_ops_emits_a_gateable_wall_key_per_operation():
    """Every `fragment_ops` arm needs a `wall_s__<op>` in `extra`.

    The module has always *timed* four in-place mutations, but `wall_s` is a
    reserved `add_run` parameter: it lands on the `RunRecord`, never in
    `runs[].extra`, and the gate reads `extra` only. So the only gateable
    timing was `median_wall_s`, pooled across arms and dominated by whichever
    is cheapest — `rollback`, which is a 4 KB pwrite.

    That is the wrong way round for OPT-FORMAT-1. Four of the five ops commit
    through `commit_in_place` -> `finalize_header_with_checksum`, which streams
    offset 256 -> EOF regardless of how little changed, and `rollback` is the
    purest instrument for it precisely *because* it does almost nothing else.
    """
    import ast

    tree, path = _bench_ast("fragment_ops")

    ops: set[str] = set()
    walls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "operation":
            if isinstance(node.value, ast.Constant):
                ops.add(node.value.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("wall_s__"):
                walls.add(node.value[len("wall_s__"):])

    assert ops, f"found no operation= labels in {path.name}"
    missing = sorted(ops - walls)
    assert not missing, (
        f"these {path.name} arms record no gateable wall: {missing}. Add "
        f"`**{{'wall_s__<op>': round(wall, 6)}}` to their add_run — `wall_s=` "
        f"alone is invisible to every threshold."
    )
    assert "rollback" in walls and "obs_import" in walls, (
        "the two cleanest OPT-FORMAT-1 instruments (rollback, obs_import) must "
        f"keep their wall keys; found {sorted(walls)}"
    )


def test_scx_cli_probe_has_one_home():
    """No benchmark module re-implements the `scx` binary search.

    `$SCX_CLI_BIN` -> `target/release/scx` -> PATH was written out twice, and
    `shuffle_layout`'s own docstring noted the other copy "uses the same
    order". A third was about to be added for the cloud `scx info` arm. The
    shared home is `benchmarks/comprehensive/scx_cli.py`, in the shape
    `rss.py` established for the RSS reader — and the callers keep their own
    *probes*, because what a usable binary means differs per arm (`info
    --json`, `optimize --codec`, a cloud-gated subcommand).

    The same shape as `test_rss_helper.py`'s "no module reimplements the
    reader".
    """
    import ast

    bench_dir = PROJECT_ROOT / "benchmarks" / "comprehensive" / "benchmarks"
    offenders: list[str] = []
    for mod in sorted(bench_dir.glob("*.py")):
        tree = ast.parse(mod.read_text(), filename=str(mod))
        # Parsed, not grepped, and for the reason this file already records
        # under `test_no_benchmark_passes_extra_as_a_dict`: all three of these
        # modules now *explain* the shared probe in a docstring, so a text
        # search matches the explanation and the test fails on its own prose.
        # A re-implementation reads the env var, which only ever appears as a
        # call argument (`os.environ.get("SCX_CLI_BIN")`) or a subscript.
        for node in ast.walk(tree):
            reads = []
            if isinstance(node, ast.Call):
                reads = [a for a in node.args if isinstance(a, ast.Constant)]
            elif isinstance(node, ast.Subscript) and isinstance(
                node.slice, ast.Constant
            ):
                reads = [node.slice]
            if any(a.value == "SCX_CLI_BIN" for a in reads):
                offenders.append(f"{mod.name}:{node.lineno}")
    assert not offenders, (
        f"these modules read $SCX_CLI_BIN directly instead of calling "
        f"`benchmarks.comprehensive.scx_cli.resolve_scx_bin`: {offenders}"
    )

    # And the shared module must actually be what they call.
    from benchmarks.comprehensive import scx_cli

    assert callable(scx_cli.resolve_scx_bin)
    # `info --help` exits 0 on a build with no cloud support at all, so the
    # cloud probe has to use a clap-gated subcommand. Pinned because getting
    # this wrong resolves a binary that then fails on the URL at run time.
    assert scx_cli.CLOUD_PROBE[0] != "info", (
        f"CLOUD_PROBE is {scx_cli.CLOUD_PROBE}; `info` is compiled into every "
        f"build (only its cloud *branch* is feature-gated), so probing it "
        f"cannot detect cloud support"
    )


#: Every metric name PR-02c's five arms emit, keyed by the module that owns it.
#:
#: Pinned because `thresholds.yaml`'s Deferred entries 15-19 spell these names
#: out in their activation recipes, and a rename would leave those recipes
#: describing floors that resolve `None` — the "missing metric" outcome, which
#: is loud, but only after someone has already authored the row and run a
#: capture. Cheaper to fail here.
#:
#: Verified once against the gate's own reader
#: (`compare_against_baseline._load_current_raw_metric`) on real runs of all
#: three modules: every one of the twelve resolved a value. This test keeps the
#: names in place; that check established they are readable at all.
_PR02C_ARM_METRICS: dict[str, tuple[str, ...]] = {
    "cellset_gather": (
        "cellsets_per_sec__collate_rust",
        "us_per_cell__collate",
        "peak_rss_mb__collate_rust",
    ),
    "ml_loader": (
        "batches_per_sec__raw_obs_highcard",
        "batches_per_sec__raw_obs_lowcard",
        "obs_highcard_overhead_ms_per_batch",
        "obs_highcard_slowdown_vs_lowcard",
    ),
    "grouped_sort": (
        "peak_rss_mb__convert_sort_by",
        "wall_s__convert_sort_by",
        "wall_s__sort_group",
    ),
    "fragment_ops": (
        "wall_s__obs_import",
        "peak_rss_mb__obs_import",
        "wall_s__rollback",
    ),
    "cloud_metadata": (
        "wall_s__scx_info_cloud",
        "scx_info_cloud_ok",
    ),
}


@pytest.mark.parametrize("module", sorted(_PR02C_ARM_METRICS))
def test_new_arm_metric_names_are_still_emitted(module: str):
    """The arms must keep the metric names their Deferred recipes cite.

    Several are built by f-string from a scenario name
    (`f"peak_rss_mb__{scenario}"`), so the full key never appears as a literal;
    those are matched through the scenario name plus the prefix pattern.

    **What this pins, exactly: the names, not the wiring.** Three of the five
    modules derive the key from a different kind of constant — a local
    (`co_arm = "collate_rust"`), a spec dict's keys
    (`_OBS_CARDINALITY_SPEC`), an argument at the `_run_convert` call site — so
    there is no single AST shape that means "this name is live". An attempt to
    require the scenario name to appear *as an argument to an arm runner*
    failed two modules that were perfectly correct, which is the signal that
    the proxy was wrong rather than the code.

    The complementary check is empirical and was run once, outside the suite:
    each of these twelve names was resolved through the gate's own reader,
    `compare_against_baseline._load_current_raw_metric`, against real runs of
    `cellset_gather`, `fragment_ops` and `grouped_sort` — all twelve returned a
    value. That established they are readable; this test keeps them from being
    renamed out from under `thresholds.yaml`.
    """
    import ast

    tree, path = _bench_ast(module)
    src = path.read_text()

    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    missing = []
    for metric in _PR02C_ARM_METRICS[module]:
        if metric in literals:
            continue
        prefix, _, scenario = metric.rpartition("__")
        if scenario in literals and f'f"{prefix}__{{' in src:
            continue
        missing.append(metric)

    assert not missing, (
        f"{path.name} no longer emits {missing}. These names are cited by "
        f"thresholds.yaml's Deferred floors (items 15-19); renaming one leaves "
        f"its activation recipe describing a floor that resolves None. Rename "
        f"in both places, or here first."
    )


def test_obs_import_arm_resolves_its_join_key_instead_of_assuming_one():
    """The obs index is not always unique, so the arm must not assume it.

    The first version of this arm hard-coded `key="obs_names"`. It passed at
    pbmc3k, whose 2700 barcodes are distinct, and failed at census_1m with

        duplicate obs join key(s): target obs key 'obs_names' contains
        duplicates: ["0", "1", "2", "3", "4"]. key 'obs_names' has 100000
        distinct values over 1000000 rows. Obs columns that ARE unique and can
        key a join: ["soma_joinid"]. Try key = soma_joinid.

    The CELLxGENE export duplicates its obs index 10x. `diagnose_obs_key`'s own
    docstring names this exact fixture, and it is the API for the job — so the
    key is resolved per file. `soma_joinid` is the *right* answer here, which is
    the mirror image of the `ml_loader` arm above, where it is the wrong one:
    unique enough to key a join, and an int64 that rebuilds no category list.

    A pbmc3k-only smoke could not have caught this, which is the general point:
    a fixture that satisfies an assumption cannot test it.
    """
    import ast

    tree, path = _bench_ast("fragment_ops")
    src = path.read_text()

    assert _calls_named(tree, "diagnose_obs_key"), (
        f"{path.name} no longer calls pyscx.diagnose_obs_key; the obs_import "
        f"arm would be assuming a join key again"
    )

    # `key=` must come from the resolved value, never from a literal. A literal
    # is what reintroduces the bug.
    literal_keys = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and str(getattr(node.func, "attr", "")) == "obs_import"
        for kw in node.keywords
        if kw.arg == "key"
        and isinstance(kw.value, ast.Constant)
    ]
    assert not literal_keys, (
        f"{path.name} passes a literal `key=` to obs_import at lines "
        f"{literal_keys}. Resolve it with diagnose_obs_key — the obs index is "
        f"10x-duplicated on the census fixtures."
    )

    # And the two sides are paired explicitly rather than relying on both
    # resolving the same name, since the CSV's column need not be the target's.
    assert "source_key=[source_key]" in src, (
        f"{path.name} no longer pairs the CSV's key column with the target's "
        f"via source_key=; the two sides can then silently disagree"
    )
