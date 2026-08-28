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


@pytest.fixture(scope="module")
def floors() -> list[dict]:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "benchmarks/comprehensive/thresholds.yaml").read_text()
    )
    return raw["absolute_floors"]


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
    for fmt in (SHUF, SHUF_NVCOMP, PCA_STREAMING):
        specs = [f for f in floors if f.get("format") == fmt]
        assert specs, f"{fmt} carries no floors — the arm would run ungated"
        assert fmt in known, f"{fmt} is floored but not registered"
        for spec in specs:
            ds = spec["dataset"]
            assert "*" not in ds, "these arms use explicit datasets, not globs"
            if fmt in ARMS:
                assert ds in ARMS[fmt].datasets, (
                    f"{fmt} is floored on {ds}, which its arm does not run — "
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
