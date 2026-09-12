"""The GPU decode / power-loop arms, and the labels that prove which one ran.

Hermetic — no GPU, no datasets read, no submitit.

Phase 8 changed two GPU paths that a default gate run does not measure:

  * `accel_to_gpu_anndata` only ever ran an **Scx1** fixture, and Scx1 shards
    never enter the ShufDeltaZstd decode paths — so `SCX_SHUFDELTA_NVCOMP=1`
    against it is a literal no-op and both "arms" would report the same numbers
    under different names.
  * `accel_pca`'s GPU variants take the **resident** power loop on every gate
    dataset, because residency is decided against free VRAM and all three fit
    an 80 GB H100. The streaming operator had no coverage at all.

Each arm therefore carries a label recorded from `uns["scx_accel"]`, not
inferred from the variant name, and `thresholds.yaml` floors it in **both**
directions. These tests assert the wiring that makes those labels true:
the arms read different fixtures, the env reaches the worker, and every floor
names a triple that actually runs — a floor on a triple that does not run is
skipped in silence, which is how a gate exits 0 having measured nothing.
"""

from __future__ import annotations

import os

import pytest
import yaml

from benchmarks.comprehensive.config import PROJECT_ROOT

SCX1 = "accel_to_gpu_anndata__scx1_gpu"
SHUF = "accel_to_gpu_anndata__shufdelta_gpu"
SHUF_NVCOMP = "accel_to_gpu_anndata__shufdelta_gpu_nvcomp"
PCA_RESIDENT = "accel_pca__pyscx_gpu_rand_hh"
PCA_STREAMING = "accel_pca__pyscx_gpu_streaming"
PCA_BACKED = "accel_pca__pyscx_gpu_backed"


@pytest.fixture(scope="module")
def floors() -> list[dict]:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "benchmarks/comprehensive/thresholds.yaml").read_text()
    )
    return raw["absolute_floors"]


def _scope_for(fmt: str) -> frozenset[str] | None:
    """The dataset scope the orchestrator actually honours for a format."""
    from benchmarks.comprehensive.benchmarks import accel_pca as pca
    from benchmarks.comprehensive.benchmarks import accel_to_gpu_anndata as tga

    for mod in (tga, pca):
        scope = getattr(mod, "FORMAT_DATASET_SCOPE", {}).get(fmt)
        if scope is not None:
            return frozenset(scope)
    return None


def _floors_for(floors, fmt, metric):
    return [f for f in floors if f.get("format") == fmt and f.get("metric") == metric]


# ---------------------------------------------------------------------------
# to_gpu_anndata: three arms, and the nvcomp one is not a no-op
# ---------------------------------------------------------------------------


def test_three_decode_arms_are_registered():
    import benchmarks.comprehensive.config as c

    keys = {f.key for f in c.accel_formats()}
    assert {SCX1, SHUF, SHUF_NVCOMP} <= keys


def test_the_nvcomp_arm_reads_a_different_fixture_than_the_scx1_arm():
    """The assertion that makes the nvcomp arm mean anything.

    Scx1 shards are decoded by `decode_framed_scx1_gpu` and never reach
    `decode_framed_shufdelta_gpu`, where `nvcomp_enabled()` is consulted. An
    "nvcomp arm" that pointed at the `_scx1` fixture would run byte-identical
    work to the default arm and report it under a different name.
    """
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import ARMS

    assert ARMS[SHUF_NVCOMP].fixture_attr != ARMS[SCX1].fixture_attr
    assert ARMS[SHUF_NVCOMP].fixture_attr == ARMS[SHUF].fixture_attr
    assert ARMS[SHUF_NVCOMP].env == {"SCX_SHUFDELTA_NVCOMP": "1"}
    assert ARMS[SHUF].env == {}


def test_every_arm_pins_its_codec_and_never_leaves_it_to_auto():
    """`auto` re-decides per shard, so it could move a run onto another arm."""
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import ARMS

    assert {a.codec for a in ARMS.values()} == {"scx1", "shufdelta"}
    assert "auto" not in {a.codec for a in ARMS.values()}


def test_the_arm_env_context_sets_and_restores():
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import (
        ARMS,
        _arm_env,
    )

    before = os.environ.get("SCX_SHUFDELTA_NVCOMP")
    with _arm_env(ARMS[SHUF_NVCOMP].env):
        assert os.environ["SCX_SHUFDELTA_NVCOMP"] == "1"
    assert os.environ.get("SCX_SHUFDELTA_NVCOMP") == before


def test_the_shufdelta_fixture_paths_differ_per_dataset():
    """`fixture_attr` names a real `DatasetConfig` property, not a typo.

    `getattr(dataset, arm.fixture_attr)` would raise inside a SLURM worker,
    hours in, and be recorded as a fixture-prep stub rather than a wiring bug.
    """
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import ARMS
    from benchmarks.comprehensive.config import DATASETS

    ds = DATASETS["pbmc3k"]
    paths = {getattr(ds, arm.fixture_attr) for arm in ARMS.values()}
    assert len(paths) == 2, f"expected scx1 + shufdelta fixtures, got {paths}"
    for arm in ARMS.values():
        assert getattr(ds, arm.fixture_attr).name.endswith(".scx")


# ---------------------------------------------------------------------------
# accel_pca: the streaming arm, and the env that a OnceLock forces into the shell
# ---------------------------------------------------------------------------


def test_the_streaming_pca_arm_is_registered_and_dispatchable():
    import benchmarks.comprehensive.config as c
    from benchmarks.comprehensive.benchmarks.accel_pca import _VARIANT_IMPLS

    assert PCA_STREAMING in {f.key for f in c.accel_formats()}
    assert PCA_STREAMING in _VARIANT_IMPLS
    _, requires_gpu = _VARIANT_IMPLS[PCA_STREAMING]
    assert requires_gpu


def test_the_streaming_arm_composes_the_resident_knob_onto_the_native_pin():
    """Composed, not substituted: the streaming arm is still a *native* GPU arm.

    Dropping `SCX_FORCE_NATIVE_GPU` would route it to rapids-singlecell, whose
    PCA has no residency decision at all — the arm would silently stop
    exercising the operator it exists to measure, and `pca_route_gpu_correct`
    is the floor that would notice.
    """
    from benchmarks.comprehensive.benchmarks.accel_pca import dispatch_env

    before = (
        os.environ.get("SCX_FORCE_NATIVE_GPU"),
        os.environ.get("SCX_GPU_PCA_RESIDENT"),
    )
    with dispatch_env(PCA_STREAMING, requires_gpu=True):
        assert os.environ["SCX_FORCE_NATIVE_GPU"] == "1"
        assert os.environ["SCX_GPU_PCA_RESIDENT"] == "0"
    assert (
        os.environ.get("SCX_FORCE_NATIVE_GPU"),
        os.environ.get("SCX_GPU_PCA_RESIDENT"),
    ) == before


def test_the_resident_arm_does_not_set_the_streaming_knob():
    """Otherwise both arms would be the streaming arm, and the pair would agree."""
    from benchmarks.comprehensive.benchmarks.accel_pca import dispatch_env

    with dispatch_env(PCA_RESIDENT, requires_gpu=True):
        assert os.environ["SCX_FORCE_NATIVE_GPU"] == "1"
        assert "SCX_GPU_PCA_RESIDENT" not in os.environ


def test_resident_csr_reader_is_none_when_the_route_had_no_decision():
    """CPU and rapids routes record `None`; the caller must emit no label then,
    not a fabricated 0.0 that would read as "streamed"."""
    from benchmarks.comprehensive.benchmarks.accel_pca import _extract_resident_csr

    class A:
        def __init__(self, uns):
            self.uns = uns

    assert _extract_resident_csr(A({}), "pca") is None
    assert _extract_resident_csr(A({"scx_accel": {}}), "pca") is None
    assert (
        _extract_resident_csr(A({"scx_accel": {"pca": {"resident_csr": None}}}), "pca")
        is None
    )
    assert (
        _extract_resident_csr(A({"scx_accel": {"pca": {"resident_csr": True}}}), "pca")
        is True
    )
    assert (
        _extract_resident_csr(A({"scx_accel": {"pca": {"resident_csr": False}}}), "pca")
        is False
    )


# ---------------------------------------------------------------------------
# The shell half: a knob read once per process must be exported before it starts
# ---------------------------------------------------------------------------


def test_the_arm_knobs_reach_the_worker_shell():
    """`SCX_GPU_PCA_RESIDENT` is cached in a OnceLock on the Rust side, so an
    in-process context manager is not sufficient on its own — by the time it
    runs, an earlier GPU PCA call in the same process may already have latched
    the value. The export has to be in the job's setup commands."""
    from benchmarks.comprehensive.scripts.run_parallel import _arm_env_exports

    assert _arm_env_exports(PCA_STREAMING) == ["export SCX_GPU_PCA_RESIDENT='0'"]
    assert _arm_env_exports(SHUF_NVCOMP) == ["export SCX_SHUFDELTA_NVCOMP='1'"]
    # …and no other format picks up an arm knob it did not ask for.
    assert _arm_env_exports(SHUF) == []
    assert _arm_env_exports(PCA_RESIDENT) == []
    assert _arm_env_exports(None) == []


def test_arm_exports_are_actually_placed_in_the_setup_commands():
    """`_arm_env_exports` returning the right string is worth nothing if the
    caller drops it. Assert it reaches the list submitit runs."""
    from benchmarks.comprehensive.scripts.run_parallel import _slurm_setup_cmds

    cmds = _slurm_setup_cmds("scx-bench-gpu", PCA_STREAMING)
    assert "export SCX_GPU_PCA_RESIDENT='0'" in cmds
    # Ordered before the conda activation, so the worker's Python sees it.
    knob = cmds.index("export SCX_GPU_PCA_RESIDENT='0'")
    activate = next(i for i, c in enumerate(cmds) if c.startswith("conda activate"))
    assert knob < activate

    assert "export SCX_GPU_PCA_RESIDENT='0'" not in _slurm_setup_cmds(
        "scx-bench-gpu", PCA_RESIDENT
    )


def test_row_group_cache_kill_switch_is_forwarded_to_workers(monkeypatch):
    """The OPT-FORMATIO-1 same-build A/B is `SCX_ROW_GROUP_CACHE=0` on the
    orchestrator. submitit does not propagate the orchestrator's environment,
    and pyscx caches the knob in a `OnceLock` on first read — so the export has
    to be in the worker's setup commands, ahead of the env activation, or the
    "off" arm silently measures the "on" build."""
    from benchmarks.comprehensive.scripts.run_parallel import _slurm_setup_cmds

    monkeypatch.setenv("SCX_ROW_GROUP_CACHE", "0")
    cmds = _slurm_setup_cmds("scx-bench", "scx_compact_trial_g256")
    assert "export SCX_ROW_GROUP_CACHE='0'" in cmds
    knob = cmds.index("export SCX_ROW_GROUP_CACHE='0'")
    activate = next(i for i, c in enumerate(cmds) if c.startswith("conda activate"))
    assert knob < activate

    monkeypatch.delenv("SCX_ROW_GROUP_CACHE", raising=False)
    assert not any(
        c.startswith("export SCX_ROW_GROUP_CACHE=") for c in _slurm_setup_cmds("scx-bench")
    ), "unset on the orchestrator must stay unset (the shipped default) on the worker"

    # The same-build A/B also points the workers at a specific checkout; the
    # `.so` a worker imports is the one the orchestrator's PYTHONPATH names.
    monkeypatch.setenv("PYTHONPATH", "/some/repo/pyscx/python")
    assert "export PYTHONPATH='/some/repo/pyscx/python'" in _slurm_setup_cmds("scx-bench")


def test_both_new_arm_families_route_to_the_gpu_conda_env():
    """nvcomp lives in `scx-bench-gpu/lib`; routing an arm elsewhere makes its
    `dlopen` fail and its decode fall back to the pipeline in silence."""
    from benchmarks.comprehensive.scripts.run_parallel import _env_for_format

    for key in (SHUF, SHUF_NVCOMP, PCA_STREAMING):
        assert _env_for_format(key) == "scx-bench-gpu", key


# ---------------------------------------------------------------------------
# The floors: reachable, and opposed
# ---------------------------------------------------------------------------


def test_every_arm_floor_names_a_triple_that_actually_runs(floors):
    """A floor on a triple the candidate did not run is skipped **silently**
    (`check_absolute_floors` → `_triple_was_run` → `continue`). A typo in a
    format key or a dataset outside the arm's scope therefore does not fail the
    gate; it removes the floor. That is the "gate exits 0 having measured
    nothing" shape, and it is why this is asserted here rather than discovered
    on an H100 four hours in."""
    import benchmarks.comprehensive.config as c
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import ARMS

    known = {f.key for f in c.accel_formats()}
    for fmt in (SHUF, SHUF_NVCOMP, PCA_STREAMING, PCA_BACKED):
        specs = [f for f in floors if f.get("format") == fmt]
        assert specs, f"{fmt} carries no floors — the arm would run ungated"
        assert fmt in known, f"{fmt} is floored but not registered"
        for spec in specs:
            ds = spec["dataset"]
            assert "*" not in ds, "these arms use explicit datasets, not globs"
            # Consult the SCOPE the orchestrator honours, not just `ARMS` —
            # `PCA_STREAMING` is not in `ARMS`, so keying on that alone left its
            # floors unchecked against the scope that decides whether they run
            # at all (Cursor Agent, #474).
            scope = _scope_for(fmt)
            assert scope is not None, f"{fmt} declares no dataset scope"
            assert ds in scope, (
                f"{fmt} is floored on {ds}, which its scope does not run — "
                "the floor would be skipped in silence"
            )


def test_the_two_shufdelta_arms_are_floored_in_opposite_directions(floors):
    """One-sided floors would let a silent nvcomp fallback pass both arms.

    `shard_decode.rs` sets `fully_device_decoded` true only on the nvcomp path.
    nvcomp is runtime-`dlopen`'d; when `libnvcomp.so.5` is missing the decode
    falls back to the Phase-1.5 pipeline without a word. With only the `min`
    half, both arms would then report 0.0 and only the nvcomp floor would fail
    — with only the `max` half, both would report 1.0 and only the pipelined
    one would. The pair is what makes each number's arm checkable."""
    nvcomp = _floors_for(floors, SHUF_NVCOMP, "shufdelta_fully_device_decoded")
    pipelined = _floors_for(floors, SHUF, "shufdelta_fully_device_decoded")

    assert nvcomp and pipelined
    assert all(f.get("min") == 1.0 and "max" not in f for f in nvcomp)
    assert all(f.get("max") == 0.0 and "min" not in f for f in pipelined)
    assert {f["dataset"] for f in nvcomp} == {f["dataset"] for f in pipelined}


def test_the_two_pca_power_loops_are_floored_in_opposite_directions(floors):
    """Same shape, for `SCX_GPU_PCA_RESIDENT`'s OnceLock: a streaming run whose
    env arrived after the latch would otherwise report the resident number
    under the streaming name."""
    resident = _floors_for(floors, PCA_RESIDENT, "pca_resident_csr")
    streaming = _floors_for(floors, PCA_STREAMING, "pca_resident_csr")

    assert resident and streaming
    assert all(f.get("min") == 1.0 and "max" not in f for f in resident)
    assert all(f.get("max") == 0.0 and "min" not in f for f in streaming)
    assert {f["dataset"] for f in resident} == {f["dataset"] for f in streaming}


def test_both_shufdelta_arms_gate_decode_correctness(floors):
    """A fast wrong answer is the failure a route floor cannot see."""
    for fmt in (SHUF, SHUF_NVCOMP):
        specs = _floors_for(floors, fmt, "to_gpu_anndata_decode_correct")
        assert specs, f"{fmt} does not gate decode correctness"
        assert all(f.get("min") == 1.0 for f in specs)


# ---------------------------------------------------------------------------
# Round-1 review fixes (PR #474)
# ---------------------------------------------------------------------------


def _guarded_extras_keys() -> dict[str, str]:
    """Map each `extras[...] = ...` key in `_run_arm` to the `if` test guarding it.

    Structural, via `ast`, because a substring check cannot tell
    `if n_shufdelta_gpu > 0:` from `if True:` once a second guarded block exists
    in the same function — a weakness that let a mutation through on first
    writing.
    """
    import ast
    import inspect
    import textwrap

    from benchmarks.comprehensive.benchmarks import accel_to_gpu_anndata as m

    tree = ast.parse(textwrap.dedent(inspect.getsource(m._run_arm)))
    out: dict[str, str] = {}

    def walk(node, guard):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk(child, ast.unparse(child.test))
                for h in child.orelse:
                    walk(h, guard)
                continue
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    if (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "extras"
                        and isinstance(t.slice, ast.Constant)
                    ):
                        out[t.slice.value] = guard
            walk(child, guard)

    walk(tree, "")
    return out


def test_the_shufdelta_metrics_are_guarded_by_an_actual_shufdelta_decode():
    """Both ShufDeltaZstd metrics must be emitted only when the run decoded
    ShufDeltaZstd shards.

    Scx1 also stamps `transfer_mode == "scx_device_decode_gpu"`, so an
    unconditional `shufdelta_fully_device_decoded` recorded `1.0` next to
    `shufdelta_n_shards_gpu = 0` on `__scx1_gpu` — the name was a lie on the
    variant this benchmark already shipped (Cursor Agent, confirmed in the
    job-2858457 snapshot).
    """
    guards = _guarded_extras_keys()
    for key in ("shufdelta_fully_device_decoded", "shufdelta_all_shards_gpu"):
        assert key in guards, f"{key} is no longer emitted at all"
        assert "n_shufdelta_gpu" in guards[key], (
            f"{key} is emitted under `{guards[key] or '<no guard>'}`, which does "
            "not depend on whether any ShufDeltaZstd shard was decoded"
        )


def test_every_arm_variant_declares_a_dataset_scope_the_orchestrator_reads():
    """Scope must live where cohorts are BUILT, not inside `run()`.

    Stubbing an out-of-scope dataset inside `run()` is a typed result, but only
    after SLURM has started a preemptible GPU task, activated conda and imported
    pyscx. And the streaming PCA arm had no scope at all, so a default capture
    scheduled the forced-slow power loop on census_1m — a different, ungated
    experiment. Both found by Cursor Agent.
    """
    from benchmarks.comprehensive.benchmarks import accel_pca as pca
    from benchmarks.comprehensive.benchmarks import accel_to_gpu_anndata as tga

    assert tga.FORMAT_DATASET_SCOPE[SHUF] == tga.ARMS[SHUF].datasets
    assert tga.FORMAT_DATASET_SCOPE[SHUF_NVCOMP] == tga.ARMS[SHUF_NVCOMP].datasets
    assert PCA_STREAMING in pca.FORMAT_DATASET_SCOPE
    assert "census_1m" not in pca.FORMAT_DATASET_SCOPE[PCA_STREAMING]


def test_the_backed_pca_arm_is_registered_and_dispatchable():
    """The out-of-core GPU PCA path had no benchmark at all.

    Every other `accel_pca` variant runs on the runner's **in-memory** adata,
    which reaches GPU PCA through pyscx's `BorrowedCsrSource` — `n_shards()` is
    hard-coded 1, so the decode-prefetch pipeline takes its sequential fallback
    and the multi-shard column-means pass is unobservable. `__pyscx_gpu_streaming`
    is not an exception: it differs only by `SCX_GPU_PCA_RESIDENT=0` and reads
    the same single-shard source. Measured on pbmc3k, an in-memory
    `device="gpu"` PCA reports `route: rapids_singlecell_gpu` and never enters
    the native path at all.
    """
    import benchmarks.comprehensive.config as c
    from benchmarks.comprehensive.benchmarks import accel_pca as pca

    assert PCA_BACKED in {f.key for f in c.accel_formats()}
    assert PCA_BACKED in pca._VARIANT_IMPLS
    impl, requires_gpu = pca._VARIANT_IMPLS[PCA_BACKED]
    assert requires_gpu, "a CPU host must skip this arm, not run it on the CPU"
    assert impl is pca._run_pyscx_gpu_backed


def test_the_backed_pca_arm_is_scoped_away_from_single_shard_datasets():
    """One shard means the prefetch pipeline takes its sequential fallback and
    this arm measures exactly what the in-memory arms do — under a name that
    claims otherwise. Shard count follows `n_obs` at the writer's default
    `shard_target_rows` (16384): pbmc3k's 2,700 cells are one shard;
    tabula_sapiens_100k gives 7 and census_500k 31."""
    from benchmarks.comprehensive.benchmarks import accel_pca as pca

    scope = pca.FORMAT_DATASET_SCOPE.get(PCA_BACKED)
    assert scope, f"{PCA_BACKED} declares no dataset scope"
    assert "pbmc3k" not in scope, "pbmc3k is single-shard — the arm would be a duplicate"
    assert scope == frozenset({"tabula_sapiens_100k", "census_500k"})


def test_the_backed_pca_arm_floors_its_shard_count_premise(floors):
    """The scope is a claim about shard counts; this is the claim being checked
    at run time. Without it, a change to the writer's default `shard_target_rows`
    silently turns this arm into a second in-memory measurement and every wall
    floor below still passes — faster, on a smaller problem."""
    specs = _floors_for(floors, PCA_BACKED, "pca_backed_n_shards")
    assert specs, "the multi-shard premise is unfloored — the arm can degrade silently"
    scope = _scope_for(PCA_BACKED)
    covered = {s["dataset"] for s in specs}
    assert covered == set(scope), (
        f"pca_backed_n_shards floors cover {covered}, scope runs {set(scope)}"
    )
    for s in specs:
        assert s.get("min", 0) >= 2, "a floor of <2 shards asserts nothing"


def test_the_backed_pca_wall_floor_names_a_metric_the_gate_can_read(floors):
    """`check_absolute_floors` reads **`runs[].extra` only**
    (`_load_current_raw_metric`). `wall_s` is a named `add_run` parameter and so
    lands at the top level of the record, where the floor gate cannot see it — a
    `metric: wall_s` floor resolves to "missing", which counts as a violation on
    every single run. The arm therefore emits its own `pca_backed_wall_s` extra,
    and this pins that the floor names that one."""
    import inspect

    from benchmarks.comprehensive.benchmarks import accel_pca as pca

    specs = [f for f in floors if f.get("format") == PCA_BACKED]
    assert specs, f"{PCA_BACKED} carries no floors — the 1.4x it exists to gate is ungated"
    wall_specs = [s for s in specs if "wall" in s["metric"]]
    assert wall_specs, "no wall ceiling — a regression in the means pass would not fail"
    src = inspect.getsource(pca.run)
    for s in wall_specs:
        assert s["metric"] != "wall_s", (
            "`wall_s` is not readable by the absolute-floor gate; emit an extra"
        )
        assert f'extras["{s["metric"]}"]' in src, (
            f"{s['metric']} is floored but `accel_pca.run` never emits it"
        )
        assert "max" in s, "a wall ceiling is a `max`, not a `min`"


def test_the_backed_pca_wall_ceiling_is_only_where_it_can_fire(floors):
    """The wall ceiling is on tabula_sapiens_100k and deliberately NOT on
    census_500k.

    Measured main vs branch in one job, same host, identical shard counts
    (SLURM 2938504):

        tabula_sapiens_100k   1.615 -> 1.141 s   1.42x   1.25x ceiling = 1.43  FIRES
        census_500k           4.033 -> 3.573 s   1.13x   1.25x ceiling = 4.47  cannot

    A census ceiling at the house 1.25x would sit above the *reverted* wall, so
    deleting the optimisation outright would leave it green. That is worse than
    no floor: it reads as coverage. This test exists so a later "we floored one
    dataset, let's floor the other for symmetry" has to produce a number first.
    """
    walls = [f for f in floors
             if f.get("format") == PCA_BACKED and "wall" in f.get("metric", "")]
    covered = {f["dataset"] for f in walls}
    assert covered == {"tabula_sapiens_100k"}, (
        f"wall ceilings on {covered}; census_500k's 1.13x cannot be caught by a "
        "1.25x ceiling — re-measure before adding one"
    )
    # And the exact floors DO cover both, so census still gates what it can.
    for metric in ("pca_backed_n_shards", "pca_route_gpu_correct"):
        exact = {f["dataset"] for f in _floors_for(floors, PCA_BACKED, metric)}
        assert exact == set(_scope_for(PCA_BACKED)), (
            f"{metric} covers {exact}; every scoped dataset must carry it"
        )


def test_the_backed_pca_arm_refuses_to_run_without_its_fixture():
    """Falling back to the in-memory X would report a single-shard number under
    the multi-shard arm's name — the precise failure the arm exists to prevent.
    It raises instead."""
    import pytest as _pytest

    from benchmarks.comprehensive.benchmarks import accel_pca as pca

    class _Adata:
        uns: dict = {}

    with _pytest.raises(RuntimeError, match="no backed fixture"):
        pca._run_pyscx_gpu_backed(_Adata(), 50, 0)


def test_the_orchestrator_refuses_out_of_scope_triples_before_submitting():
    from benchmarks.comprehensive.scripts.run_parallel import _triple_compatible

    assert not _triple_compatible("accel_pca", "census_1m", PCA_STREAMING)
    assert _triple_compatible("accel_pca", "tabula_sapiens_100k", PCA_STREAMING)
    assert not _triple_compatible("accel_to_gpu_anndata", "census_1m", SHUF)
    assert not _triple_compatible("accel_to_gpu_anndata", "census_1m", SHUF_NVCOMP)
    # An unscoped variant is untouched — the hook must not narrow anything else.
    assert _triple_compatible("accel_pca", "census_1m", PCA_RESIDENT)
    assert _triple_compatible("accel_to_gpu_anndata", "census_1m", SCX1)


def test_the_arm_env_has_one_source_of_truth():
    """`run_parallel` derives the decode arms' env from `ARMS` rather than
    keeping a second copy that a later edit could update alone (Cursor Agent)."""
    from benchmarks.comprehensive.benchmarks import accel_to_gpu_anndata as tga
    from benchmarks.comprehensive.scripts.run_parallel import _format_arm_env

    for key, arm in tga.ARMS.items():
        if arm.env:
            assert _format_arm_env()[key] == arm.env, key


def test_dispatch_env_does_not_mutate_the_environment_before_with_entry():
    """A pre-entered `ExitStack` mutated `os.environ` at call time, so a bare
    call — or an exception between two `enter_context()`s — leaked
    `SCX_FORCE_NATIVE_GPU` into the rest of the process. Every native GPU accel
    module reaches this path. Found by Antigravity and Cursor Agent."""
    from benchmarks.comprehensive.benchmarks.accel_pca import dispatch_env

    before = os.environ.get("SCX_FORCE_NATIVE_GPU")
    cm = dispatch_env(PCA_STREAMING, requires_gpu=True)  # not entered
    assert os.environ.get("SCX_FORCE_NATIVE_GPU") == before, (
        "dispatch_env mutated the environment before `with` entry"
    )
    with cm:
        assert os.environ["SCX_FORCE_NATIVE_GPU"] == "1"
        assert os.environ["SCX_GPU_PCA_RESIDENT"] == "0"
    assert os.environ.get("SCX_FORCE_NATIVE_GPU") == before


def test_the_all_shards_floor_is_stronger_than_at_least_one(floors):
    """`shufdelta_n_shards_gpu >= 1` is satisfied by one GPU shard plus six host
    bounces on tabula's seven-shard fixture — a mostly-host-bounce wall labelled
    as the GPU arm. Both ShufDeltaZstd arms carry the stronger floor too."""
    for fmt in (SHUF, SHUF_NVCOMP):
        specs = _floors_for(floors, fmt, "shufdelta_all_shards_gpu")
        assert specs, f"{fmt} does not gate that EVERY shard took the GPU path"
        assert all(f.get("min") == 1.0 for f in specs)
        assert {f["dataset"] for f in specs} == _scope_for(fmt)


def test_the_arm_env_is_lazy_and_does_not_swallow_an_import_failure():
    """An eager module-level map imported the decode benchmark at orchestrator
    import — probing CUDA on every run, including CPU-only ones — and turned an
    ImportError into an empty map, silently dropping SCX_SHUFDELTA_NVCOMP=1
    (Cursor Agent). It is `lru_cache`d now, and it raises."""
    import inspect

    from benchmarks.comprehensive.scripts import run_parallel as rp

    src = inspect.getsource(rp._format_arm_env)
    assert "except ImportError" not in src, (
        "an import failure must not become an empty map — that drops the arm "
        "knob in silence, which a hard-coded dict could never do"
    )
    assert hasattr(rp._format_arm_env, "cache_info"), "must be lru_cache'd (lazy)"
    # And it still resolves to the same thing the eager form did.
    from benchmarks.comprehensive.benchmarks import accel_to_gpu_anndata as tga

    env = rp._format_arm_env()
    for key, arm in tga.ARMS.items():
        if arm.env:
            assert env[key] == arm.env, key


def test_a_second_residency_arm_in_one_process_is_refused_not_mislabelled():
    """`SCX_GPU_PCA_RESIDENT` is latched by a Rust OnceLock on first use, so the
    serial `run_all.py` path cannot produce both arms — the second would report
    the first's numbers under its own name. The opposing floors diagnose that
    after the fact; refusing the combination stops it being recorded (codex)."""
    import inspect

    from benchmarks.comprehensive.benchmarks import accel_pca as m

    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(m.run)))
    # Structural, not a substring: `if False:` leaves every name in place, which
    # is exactly how a first version of this test let the mutation through.
    guards = [
        ast.unparse(n.test)
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and "pca_residency_arm_latched" in ast.unparse(n)
    ]
    assert guards, "a mislabelled second arm must be refused with a typed stub"
    # At least one enclosing condition must actually consult which arm ran —
    # `if False:` leaves every name in place and satisfies a substring check.
    assert any("_RESIDENCY_ARM_RUN" in g for g in guards), (
        f"the refusal is guarded only by {guards}, none of which depends on "
        "which arm already ran in this process"
    )


def test_a_non_arm_format_does_not_resolve_the_arm_map():
    """Resolving it imports the decode benchmark, which probes CUDA; a
    CPU-only capture must not pay for that (codex)."""
    from benchmarks.comprehensive.scripts import run_parallel as rp

    rp._format_arm_env.cache_clear()
    assert rp._arm_env_exports("scx_auto") == []
    assert rp._format_arm_env.cache_info().misses == 0, (
        "a non-arm format resolved the arm map anyway"
    )
    # …and an arm format still gets its knob.
    assert rp._arm_env_exports(SHUF_NVCOMP) == ["export SCX_SHUFDELTA_NVCOMP='1'"]
