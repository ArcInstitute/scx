"""Phase-0 gate instrumentation: the per-regime data-wait fraction.

Phases 5-8 of the ML-loader plan are gated on ``p`` — the fraction of a
training step spent waiting on data — and the only ``p`` on record is
STATE3's 0.6%. These tests pin the three properties that make the new
metric trustworthy enough to gate on:

1. ``_data_wait_fraction`` returns ``None``, never ``0.0``, when there was
   no consumer loop to measure. The distinction is load-bearing: the gate's
   ``_load_current_raw_metric`` *skips* ``None`` and medians over what is
   left, so a ``0.0`` placeholder would be indistinguishable from a real
   measurement of a perfectly-fed loader — the exact conclusion the metric
   exists to test.
2. The R3 null-model step is **off by default**, so every registered
   ``index_plan`` capture is unchanged and no floor or pooled
   ``median_wall_s`` moves.
3. The scatter-route driver still refuses to report a speedup ratio unless
   every arm reached the route it was supposed to.

The ml_loader half is asserted by source inspection rather than execution:
the ``gpu_train`` scenario needs CUDA, and its emitter sits inside a broad
``except Exception`` that degrades a raise to ``scenario_summary["gpu_train"]
= {"error": ...}`` with zero runs — so a GPU-less run cannot tell "not
emitted" from "no GPU", and a typo in a metric name would reach a capture
silently.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = REPO_ROOT / "benchmarks" / "comprehensive" / "benchmarks"
SCRIPTS_DIR = REPO_ROOT / "benchmarks" / "scripts"


# ---------------------------------------------------------------------------
# 1. The helper: None, never 0.0
# ---------------------------------------------------------------------------


def _helper():
    from benchmarks.comprehensive.data_wait import data_wait_fraction

    return data_wait_fraction


def test_data_wait_fraction_is_none_without_steps():
    """No steps, or no timed wall, is *unmeasured* — not "zero wait".

    `compare_against_baseline._load_current_raw_metric` skips `None` and
    medians the rest, so `None` reads as "this scenario did not produce the
    metric" while `0.0` reads as "the loader never stalled". Those are
    opposite claims and only one of them is true here.
    """
    f = _helper()
    assert f([], 1.0) is None
    assert f([0.1, 0.2], 0.0) is None
    assert f([0.1], -1.0) is None


def test_data_wait_fraction_is_bounded_and_monotone():
    f = _helper()
    assert f([0.25, 0.25], 1.0) == pytest.approx(0.5)
    assert f([1.0], 1.0) == pytest.approx(1.0)
    assert 0.0 <= f([0.0, 0.0], 2.0) <= 1.0
    # Monotone in the wait sum at fixed wall.
    assert f([0.1], 1.0) < f([0.4], 1.0) < f([0.9], 1.0)


def test_wait_percentiles_expose_the_tail_not_just_the_middle():
    """p95 alone cannot see the stalls that dominate the wait at scale.

    Measured on census_1m: p50 13 microseconds, p95 799 microseconds, yet the
    steady-state fraction is 0.76 — ~10 s of wait over ~976 steps. Those two
    facts only reconcile if a few tens of steps cost ~200 ms each. A metric
    set that stops at p95 reports "sub-millisecond batches" while three
    quarters of the step budget goes to a tail it does not show, so p99 and
    the max are part of the contract, not extras.
    """
    from benchmarks.comprehensive.data_wait import wait_percentiles

    # 950 fast steps, 49 at ~1 ms, 1 catastrophic stall.
    waits = [1e-5] * 950 + [1e-3] * 49 + [0.2]
    p = wait_percentiles(waits)
    assert set(p) == {"p50_ms", "p95_ms", "p99_ms", "max_ms"}
    assert p["p50_ms"] == pytest.approx(0.01, abs=1e-6)
    assert p["max_ms"] == pytest.approx(200.0)
    # The tail must be visible somewhere above p95.
    assert p["p99_ms"] > p["p95_ms"] or p["max_ms"] > 100 * p["p95_ms"]


def test_wait_percentiles_are_all_none_without_steps():
    from benchmarks.comprehensive.data_wait import wait_percentiles

    assert wait_percentiles([]) == {
        "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None,
    }


def test_steady_state_fraction_excludes_the_first_batch():
    """The headline `p` must not be a time-to-first-batch measurement.

    Measured on tabula: a 98-step `gpu_train` epoch reported
    `data_wait_fraction = 0.85` with `batch_wait_ms_p50 = 0.013` and
    `p95 = 0.015` — 2.04 s of "wait" whose p95 is 15 microseconds. Almost all
    of it is the FIRST `next()`, which pays tokio spin-up and the first shard
    decode, in a timed region only 2.4 s long. Published as-is it says "the
    loader is on the critical path 85% of the time", the opposite of what the
    same run's p95 says and of the 4 microsecond send-wait the D0 profile
    recorded.

    A real epoch amortises that startup over far more steps, so the number
    that answers "is the loader on the critical path" is the steady-state one,
    with time-to-first-batch reported beside it rather than folded into it.
    """
    from benchmarks.comprehensive.data_wait import steady_state_wait

    # One 2 s startup then 97 x 10 microseconds, in a 2.4 s region.
    waits = [2.0] + [1e-5] * 97
    p_all = _helper()(waits, 2.4)
    steady = steady_state_wait(waits, 2.4)
    assert p_all > 0.8, "premise: the all-steps fraction is startup-dominated"
    assert steady["ttfb_s"] == pytest.approx(2.0)
    assert steady["n_steady_steps"] == 97
    # 97 x 1e-5 = 9.7e-4 over a 0.4 s steady region -> ~0.24%.
    assert steady["data_wait_fraction_steady"] < 0.01
    assert steady["data_wait_fraction_steady"] < p_all / 50


def test_steady_state_fraction_is_none_with_one_or_zero_steps():
    """One step is all startup; there is no steady state to report."""
    from benchmarks.comprehensive.data_wait import steady_state_wait

    for waits in ([], [0.5]):
        st = steady_state_wait(waits, 1.0)
        assert st["data_wait_fraction_steady"] is None
        assert st["n_steady_steps"] == 0
    assert steady_state_wait([0.5], 1.0)["ttfb_s"] == pytest.approx(0.5)


def test_steady_state_fraction_handles_a_startup_longer_than_the_wall():
    """A degenerate region must yield `None`, not a negative denominator."""
    from benchmarks.comprehensive.data_wait import steady_state_wait

    st = steady_state_wait([5.0, 0.1], 1.0)
    assert st["data_wait_fraction_steady"] is None


def test_data_wait_fraction_clamps_a_wait_that_exceeds_the_wall():
    """A fraction reported to a gate must never exceed 1.

    `wall_s` is measured with `torch.cuda.synchronize()` inside the timer,
    so `sum(waits) <= wall` holds by construction on the real path — but a
    clock hiccup or a future caller that times the wall differently must not
    be able to publish `p = 1.3` into a table read as a percentage.
    """
    f = _helper()
    assert f([2.0], 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. ml_loader's gpu_train emitter — source guard
# ---------------------------------------------------------------------------


_GPU_TRAIN_KEYS = (
    # The headline `p` is the steady-state one; the all-steps figure rides
    # beside it so a reader can see the startup it excludes.
    "data_wait_fraction_steady__gpu_train",
    "ttfb_s__gpu_train",
    "n_steady_steps__gpu_train",
    "data_wait_fraction__gpu_train",
    "batch_wait_ms_p50__gpu_train",
    "batch_wait_ms_p95__gpu_train",
    "batch_wait_ms_p99__gpu_train",
    "batch_wait_ms_max__gpu_train",
    "data_wait_s__gpu_train",
)


def _gpu_train_add_run_kwargs() -> set[str]:
    """Keyword names of the `add_run(...)` call that carries `scenario="gpu_train"`.

    Parsed rather than grepped: a bare grep cannot tell a keyword argument
    from the same string in a comment or a docstring, and this guard's whole
    job is to prove the key reaches `runs[].extra`.
    """
    tree = ast.parse((BENCH_DIR / "ml_loader.py").read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        scenario = next(
            (
                kw.value.value
                for kw in node.keywords
                if kw.arg == "scenario" and isinstance(kw.value, ast.Constant)
            ),
            None,
        )
        if scenario == "gpu_train":
            found |= kwargs
    return found


@pytest.mark.parametrize("key", _GPU_TRAIN_KEYS)
def test_gpu_train_emits_the_data_wait_keys(key: str):
    kwargs = _gpu_train_add_run_kwargs()
    assert kwargs, "no add_run(scenario='gpu_train', ...) call found in ml_loader.py"
    assert key in kwargs, (
        f"{key!r} is not passed to add_run in ml_loader's gpu_train block — "
        "the phase-0 gate has no R1 `p`. Note the block is wrapped in a broad "
        "`except Exception`, so a missing key does not surface at run time."
    )


def test_gpu_train_timed_loop_measures_next_not_a_bare_for():
    """The timed epoch must drive an explicit iterator.

    `for batch in ds:` gives no seam to time `next()` at, so a wait fraction
    derived from it would be the whole loop. This is the premise assertion
    for the metric: without it the number is not what the docs say it is.
    """
    src = (BENCH_DIR / "ml_loader.py").read_text()
    start = src.index("def _run_gpu_train_epoch")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    assert "iter(ds)" in body and "next(" in body, (
        "_run_gpu_train_epoch's timed region does not drive an explicit "
        "iterator, so per-batch wait cannot be measured"
    )


# ---------------------------------------------------------------------------
# 3. index_plan's R3 null model — off by default
# ---------------------------------------------------------------------------


def test_index_plan_null_model_defaults_off(monkeypatch: pytest.MonkeyPatch):
    """Unset env => 0 ms => registered captures are unchanged.

    `index_plan`'s pooled `median_wall_s` is gated against LATEST, and a
    fixed per-batch cost added unconditionally would read as a timing
    regression on a benchmark whose subject did not change.
    """
    monkeypatch.delenv("SCX_BENCH_R3_NULL_MODEL_MS", raising=False)
    sys.modules.pop("benchmarks.comprehensive.benchmarks.index_plan", None)
    from benchmarks.comprehensive.benchmarks import index_plan as ip

    assert ip._null_model_ms() == 0.0


def test_index_plan_null_model_reads_the_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCX_BENCH_R3_NULL_MODEL_MS", "5")
    sys.modules.pop("benchmarks.comprehensive.benchmarks.index_plan", None)
    from benchmarks.comprehensive.benchmarks import index_plan as ip

    assert ip._null_model_ms() == pytest.approx(5.0)
    # A malformed value must not take the whole capture down hours in.
    monkeypatch.setenv("SCX_BENCH_R3_NULL_MODEL_MS", "not-a-number")
    assert ip._null_model_ms() == 0.0


def test_index_plan_emits_the_data_wait_keys_for_every_scenario():
    """The suffixed keys are written for all scenarios, `None` where unmeasured.

    Same shape as the existing `shard_cache_hit_rate__<scenario>: None`
    placeholder: the gate's threshold key stays stable and a scenario that
    cannot produce the metric is legibly absent rather than zero.
    """
    src = (BENCH_DIR / "index_plan.py").read_text()
    for key in (
        'f"data_wait_fraction__{scenario_name}"',
        'f"data_wait_fraction_steady__{scenario_name}"',
        'f"ttfb_s__{scenario_name}"',
        'f"n_steady_steps__{scenario_name}"',
        'f"batch_wait_ms_p50__{scenario_name}"',
        'f"batch_wait_ms_p95__{scenario_name}"',
        'f"batch_wait_ms_p99__{scenario_name}"',
        'f"batch_wait_ms_max__{scenario_name}"',
        'f"null_model_ms__{scenario_name}"',
    ):
        assert key in src, f"index_plan.py does not emit {key}"


def test_index_plan_workers2_loop_measures_next():
    src = (BENCH_DIR / "index_plan.py").read_text()
    start = src.index("def _run_index_plan_workers2")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    assert "iter(loader)" in body and "next(" in body, (
        "_run_index_plan_workers2 does not drive an explicit iterator, so the "
        "R3 wait fraction cannot be measured"
    )


# ---------------------------------------------------------------------------
# 4. The scatter-route premise gate
# ---------------------------------------------------------------------------


def _driver():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import importlib

    mod = importlib.import_module("bench_cellset_scatter_routes")
    return mod


def _arms(on_block=800, off_full=800, off_block=0, default_full=800, default_block=0):
    return {
        "default": {
            "median_cellsets_per_sec": 500.0,
            "full_shard_groups": default_full,
            "block_index_groups": default_block,
        },
        "off": {
            "median_cellsets_per_sec": 520.0,
            "full_shard_groups": off_full,
            "block_index_groups": off_block,
        },
        "on": {
            "median_cellsets_per_sec": 4.0,
            "full_shard_groups": 0,
            "block_index_groups": on_block,
        },
    }


def test_scatter_premises_pass_on_a_well_formed_ab():
    m = _driver()
    premises = m._premises(_arms())
    assert all(premises.values()), premises
    off_ratio, default_ratio = m._ratios(_arms(), [])
    assert off_ratio == pytest.approx(130.0)
    assert default_ratio == pytest.approx(125.0)


@pytest.mark.parametrize(
    "kwargs,failing",
    [
        ({"on_block": 0}, "on arm reached the block-index route"),
        ({"off_full": 0}, "off arm took the full-shard route"),
        ({"off_block": 5}, "off arm did not reach the block-index route"),
        ({"default_block": 5}, "the default agrees with the explicit False"),
    ],
)
def test_scatter_premise_gate_refuses_a_vacuous_ratio(kwargs, failing):
    """No ratio may be published from two arms that took the same route.

    A speedup computed across a collapsed A/B is a plausible number that
    means nothing, and writing it into the artifact invites it being quoted
    later — so both ratio fields must be `None`, not merely flagged.
    """
    m = _driver()
    arms = _arms(**kwargs)
    premises = m._premises(arms)
    failed = [k for k, ok in premises.items() if not ok]
    assert failing in failed, premises
    assert m._ratios(arms, failed) == (None, None)


def test_scatter_plan_shape_is_part_of_the_subject_not_a_run_parameter():
    """A `grouped` row and a `random` row must not share a manifest triple.

    They measure different things: at the shipped default a random plan over
    census_1m retains no row groups at all while a grouped one hits ~45 %, so
    pooling them would produce a median describing neither — the same defect
    the one-result-per-arm rule exists to prevent. `random` keeps the bare
    `_v4reframed` dataset label so the 2026-08-29 rows stay comparable.
    """
    src = (SCRIPTS_DIR / "bench_cellset_scatter_routes.py").read_text()
    assert 'subject = "v4reframed" if args.plan == "random"' in src
    assert "{stem}_{subject}.json" in src
    assert 'dataset=f"{stem}_{subject}"' in src


def test_scatter_grouped_arm_refuses_when_no_groups_resolve():
    """A `grouped` arm that silently fell back would measure `random` twice."""
    m = _driver()
    src = (SCRIPTS_DIR / "bench_cellset_scatter_routes.py").read_text()
    body = src[src.index("def _plan_factory"):src.index("def _arm(")]
    assert "refusing to run" in body, (
        "_plan_factory accepts an empty group list, so a `grouped` capture "
        "could quietly measure something else"
    )
    assert hasattr(m, "_plan_factory")


def test_scatter_provenance_does_not_eat_the_first_dirty_path():
    """`git status --porcelain` lines must be parsed unstripped.

    Porcelain writes two status columns before the path, and an unstaged
    modification leaves the first blank — ` M path`. Stripping the command's
    whole output removes that leading space from the FIRST line only, so a
    fixed `ln[3:]` silently drops one character from exactly one path
    (`benchmarks/README.md` -> `enchmarks/README.md`) and leaves every other
    one intact, which is why it survived.

    It matters because `docs/benchmark_manifest.md` accepts a dirty capture
    only when the dirt is *documented*, and a corrupted path documents
    nothing — it names a file that does not exist.
    """
    m = _driver()
    porcelain = " M benchmarks/README.md\n M benchmarks/other.py\n?? scratch.md\n"
    tracked, all_paths = m._parse_porcelain(porcelain)
    assert tracked == ["benchmarks/README.md", "benchmarks/other.py"]
    assert all_paths == ["benchmarks/README.md", "benchmarks/other.py", "scratch.md"]


def test_scatter_provenance_parses_staged_and_renamed_entries():
    m = _driver()
    tracked, _ = m._parse_porcelain("M  a.py\nR  old.py -> new.py\nAM b.py\n")
    assert tracked == ["a.py", "new.py", "b.py"]


def test_scatter_driver_checks_the_build_before_measuring():
    """The A/B must refuse a pyscx that predates the row-group LRU.

    Measured, not theoretical: the `scx-bench` conda env carries a pyscx wheel
    built from a branch, reporting the same `__version__` as the checkout while
    lacking `row_group_*` in `cache_metrics()` entirely. A run against it
    reproduces the pre-LRU numbers under a heading that says "current main" —
    and nothing errors.
    """
    m = _driver()
    assert hasattr(m, "_require_row_group_counters")
    src = (SCRIPTS_DIR / "bench_cellset_scatter_routes.py").read_text()
    body = src[src.index("def main()"):]
    call = body.index("_require_row_group_counters(")
    first_arm = body.index('for label, gate in (("default"')
    assert call < first_arm, (
        "the build check runs after the arms — by then the hours are spent"
    )


def test_scatter_ratio_is_none_when_the_on_arm_measured_zero():
    m = _driver()
    arms = _arms()
    arms["on"]["median_cellsets_per_sec"] = 0.0
    assert m._ratios(arms, []) == (None, None)


# ---------------------------------------------------------------------------
# 5. W5 — the obs_open docstring
# ---------------------------------------------------------------------------


def test_obs_open_docstring_states_the_measured_split():
    """The "~87% open" split was retracted; the docstring must not revive it.

    `docs/performance.md` records that the 87% figure came from a warm
    interactive smoke whose first `pyscx.open` absorbed interpreter and pyo3
    init, and that the measured split is the opposite. The docstring is what
    keeps proposing the withdrawn `ScxObsReader` work item.
    """
    doc = (BENCH_DIR / "obs_open.py").read_text()
    head = doc[: doc.index('"""', 3)]
    assert "96" in head and "99.8" in head, (
        "obs_open's module docstring does not state the measured share "
        "(obs read 96-99.8% of per-file cost)"
    )
    assert "the finding that ~87%" not in head, (
        "obs_open's module docstring still presents the retracted "
        '"~87% open + catalog parse" split as a finding'
    )
    # Naming the retracted number is fine — desirable, even, so it is not
    # re-derived — but only inside its own retraction.
    if "87%" in head:
        assert "retracted" in head, (
            "obs_open's docstring mentions the 87% figure without saying it "
            "was retracted, which is how it got quoted as a finding before"
        )
