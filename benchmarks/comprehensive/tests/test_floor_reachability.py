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
