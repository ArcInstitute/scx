"""Registration guards for the four community-workflow benchmarks.

The community-workflow series registers `accel_qc_filter`,
`accel_score_genes`, `pipeline_ooc_constrained` and
`multimodal_atlas_streaming` into the orchestrator. Each of these tests closes
a gap where the wiring can be *half* present and the failure is a benchmark
that schedules nothing — which reads exactly like a benchmark that ran and
found nothing to do. That is the failure shape `test_floor_reachability.py`'s
module docstring is written about, one layer further out: that file asks
whether a floored benchmark is reachable at all, these ask whether a
registered benchmark resolves to any cells.
"""

from __future__ import annotations

import pytest

NEW_BENCHMARKS = (
    "accel_qc_filter",
    "accel_score_genes",
    "pipeline_ooc_constrained",
    "multimodal_atlas_streaming",
)


def _arm_shaped() -> set[str]:
    import benchmarks.comprehensive.scripts.run_parallel as rp
    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    return {b for b in ALL_BENCHMARKS if rp._is_arm_shaped_benchmark(b)}


def test_the_four_are_registered():
    """Guard the guard: every assertion below is over a set derived from
    `ALL_BENCHMARKS`, so if a rename drops one of these names the rest of this
    file goes quietly vacuous."""
    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    missing = [b for b in NEW_BENCHMARKS if b not in ALL_BENCHMARKS]
    assert not missing, f"unregistered: {missing}"


def test_every_arm_shaped_benchmark_contributes_variants():
    """`config.accel_formats()` is a ladder of `try: ... except ImportError:
    pass` blocks, one per arm-shaped benchmark. A typo in a variant function's
    name, a missing block, or an import error *inside* the module all collapse
    to the same silent `pass` and zero variants — and an arm-shaped benchmark
    with zero variants pairs with zero formats and is scheduled never.

    Nothing else notices. `_bench_format_compatible` returns False for every
    format and the cohort loop simply produces no cells, so the run exits 0
    with the benchmark absent from the results and no line in the log saying
    why."""
    from benchmarks.comprehensive.config import accel_formats

    keys = [f.key for f in accel_formats()]
    assert keys, "accel_formats() returned nothing; the ladder is broken"
    contributing = {k.split("__", 1)[0] for k in keys if "__" in k}
    # `bench_csc_dispatch` is the one arm-shaped benchmark whose format keys do
    # not repeat its name (they are `bench_csc__*`), which is why
    # `_bench_format_compatible` special-cases it.
    expected = _arm_shaped() - {"bench_csc_dispatch"}
    missing = sorted(expected - contributing)
    assert not missing, (
        f"arm-shaped benchmarks contributing no variants to accel_formats(): "
        f"{missing}. Each needs a `<bench>_variants()` and a block in "
        f"config.accel_formats(); without both it pairs with zero formats and "
        f"is scheduled silently never."
    )
    assert any(k.startswith("bench_csc__") for k in keys)
    assert len(keys) == len(set(keys)), "duplicate variant key"


def test_no_variant_key_leaks_to_another_benchmark():
    """The arm-shaped pairing rule is symmetric and both halves are needed.

    `_is_arm_shaped_benchmark` decides that an arm-shaped benchmark takes only
    its own `<bench>__*` keys; `_is_arm_shaped_format` decides that everything
    else takes none of them. Register only the first half and every benchmark
    with no `SUPPORTED_FORMATS` — nine of them, including `compression`,
    `read_full`, `write`, `multimodal_compression` and `multimodal_training` —
    picks up the new private keys.

    The multimodal case has no backstop at all: `multimodal_compression` and
    `multimodal_training` sit on the same side of `_triple_compatible`'s
    multimodal XOR as `multimodal_atlas_streaming`, so that filter does not
    separate them."""
    import benchmarks.comprehensive.scripts.run_parallel as rp
    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS
    from benchmarks.comprehensive.config import ALL_FORMATS, accel_formats

    pool = [f.key for f in ALL_FORMATS] + [f.key for f in accel_formats()]
    for bench in ALL_BENCHMARKS:
        got = {k for k in pool if rp._bench_format_compatible(bench, k)}
        for other in _arm_shaped():
            if other == bench:
                continue
            prefix = "bench_csc__" if other == "bench_csc_dispatch" else f"{other}__"
            leaked = sorted(k for k in got if k.startswith(prefix))
            assert not leaked, f"{bench} claims {other}'s keys: {leaked}"


@pytest.mark.parametrize("bench", NEW_BENCHMARKS)
def test_each_new_benchmark_pairs_with_exactly_its_own_keys(bench):
    import benchmarks.comprehensive.scripts.run_parallel as rp
    from benchmarks.comprehensive.config import ALL_FORMATS, accel_formats

    pool = [f.key for f in ALL_FORMATS] + [f.key for f in accel_formats()]
    got = {k for k in pool if rp._bench_format_compatible(bench, k)}
    want = {k for k in pool if k.startswith(f"{bench}__")}
    assert want, f"{bench} contributed no variants"
    assert got == want, f"{bench}: unexpected {sorted(got - want)}, missing {sorted(want - got)}"


def test_multimodal_atlas_streaming_pairs_only_with_multimodal_datasets():
    import benchmarks.comprehensive.scripts.run_parallel as rp
    from benchmarks.comprehensive.config import DATASETS

    key = "multimodal_atlas_streaming__scx_stream"
    for name, cfg in DATASETS.items():
        assert rp._triple_compatible("multimodal_atlas_streaming", name, key) == cfg.multimodal, name


def test_naming_a_new_benchmark_pulls_its_variants_into_the_format_pool():
    """`config.accel_formats()` is added to the pool only when `need_accel`
    fires. Two of these four names do not start with `accel_`, so the prefix
    test that used to drive `need_accel` would have missed them — and a
    benchmark whose variants are not in the pool resolves to zero cells.

    `multimodal_atlas_streaming` is the sharp case: `need_multimodal` fires for
    it, but that adds `config.MULTIMODAL_FORMATS`, which is not where its
    variants live."""
    import benchmarks.comprehensive.scripts.run_parallel as rp

    for bench in NEW_BENCHMARKS:
        assert rp._is_arm_shaped_benchmark(bench), (
            f"{bench} would not set need_accel, so accel_formats() is never "
            f"added to the pool and it schedules zero cells"
        )
    for bench in NEW_BENCHMARKS:
        assert rp._is_arm_shaped_format(f"{bench}__anything")


def test_path_for_format_resolves_every_pool_key_on_every_dataset():
    """`path_for_format` raises `ValueError` for a key it does not know, and it
    is called unguarded over the dataset x format cross-product in Phase A, in
    `--skip-convert`, and in the dry-run summary. An arm-shaped prefix missing
    from its short-circuit takes the whole orchestrator down before anything is
    submitted."""
    from benchmarks.comprehensive.config import ALL_FORMATS, DATASETS, accel_formats

    for cfg in DATASETS.values():
        for fmt in list(ALL_FORMATS) + accel_formats():
            cfg.path_for_format(fmt.key)


def test_arm_shaped_keys_resolve_to_a_source_phase_a_will_not_try_to_build():
    """Phase A decides whether to submit a conversion by testing
    `path_for_format(...).exists()`, and an arm-shaped key has no runner that
    could build anything (`accel_runner` is deliberately absent from
    `_RUNNER_MAP`, so a submitted conversion dies on `KeyError` and takes every
    dependent task with it). The keys are safe only because they resolve to a
    source file that is already there.

    The multimodal half is the one that has to be right: those keys read as
    multimodal, so Phase A no longer skips them on the multimodal XOR, and
    resolving them to `h5ad_path` — which does not exist for a multimodal
    dataset — would submit exactly that doomed conversion."""
    from benchmarks.comprehensive.config import DATASETS

    for name, cfg in DATASETS.items():
        assert (
            cfg.path_for_format("multimodal_atlas_streaming__scx_stream")
            == cfg.h5mu_path
        ), name
        for bench in ("accel_qc_filter", "accel_score_genes", "pipeline_ooc_constrained"):
            assert cfg.path_for_format(f"{bench}__x") == cfg.h5ad_path, (bench, name)


def test_pipeline_budget_is_declared_consistently_in_all_three_places():
    """The 16 / 32 GB ceiling is spelled three times — the format key's suffix,
    `MEMORY_BUDGET_GB`, and `FormatVariant.params["budget_gb"]` — and
    `estimate_memory_gb` reads only the second. If they drift, the SLURM
    allocation stops matching the arm's label and `pipeline_completed_bool`
    becomes a claim about a ceiling that was never applied."""
    from benchmarks.comprehensive.benchmarks.pipeline_ooc_constrained import (
        MEMORY_BUDGET_GB,
        SUPPORTED_FORMATS,
        pipeline_ooc_constrained_variants,
    )

    assert set(MEMORY_BUDGET_GB) == set(SUPPORTED_FORMATS)
    for variant in pipeline_ooc_constrained_variants():
        from_suffix = int(variant.key.rsplit("_", 1)[-1].removesuffix("g"))
        assert MEMORY_BUDGET_GB[variant.key] == from_suffix, variant.key
        assert variant.params["budget_gb"] == from_suffix, variant.key


def test_the_budget_survives_estimate_and_the_orchestrator_post_processing():
    """`estimate_memory_gb`'s shared tail (+50 % safety, round to 8 GB) and
    `_per_job_slurm_params`'s `* scale` / `max(mem, args.mem_gb)` would each
    silently raise the ceiling. `--scale-factor 1.3` is what this script's own
    `--help` recommends for noisy clusters; `capture_baseline.py` always passes
    `--mem-gb`. Either one turns a 16 GB arm into something else."""
    import argparse

    import benchmarks.comprehensive.scripts.run_parallel as rp
    from benchmarks.comprehensive.benchmarks.pipeline_ooc_constrained import (
        MEMORY_BUDGET_GB,
    )
    from benchmarks.comprehensive.config import DATASETS, estimate_memory_gb

    for key, budget in MEMORY_BUDGET_GB.items():
        for ds in ("pbmc3k", "census_1m"):
            cfg = DATASETS[ds]
            assert estimate_memory_gb(cfg, key, "pipeline_ooc_constrained") == budget, (key, ds)

    args = argparse.Namespace(
        scale_factor=1.3, mem_gb=80, timeout=480, partition="cpu_preemptible",
        cpus=16, no_gpu=False,
    )
    for key, budget in MEMORY_BUDGET_GB.items():
        params = rp._per_job_slurm_params(
            args, "census_1m", key, "pipeline_ooc_constrained", is_conversion=False,
        )
        assert params["mem_gb"] == budget, (key, params["mem_gb"])
