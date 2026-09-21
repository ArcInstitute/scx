"""Atlas-scale multimodal streaming and in-decode dtype narrowing.

The suite's existing multimodal benchmarks run on 5.2K-cell CITE-seq and
11.9K-cell Multiome fixtures. SCX's in-decode narrowing —
`to_mudata(data_dtype={"rna": "uint16", "atac": "uint8"})`, which assembles
directly at the target width through `read_all_csr_shards_for_typed` and never
builds the intermediate f32 CSR — is a 2–4x cut in value-buffer bytes that has
never been demonstrated at the >=500K scale where MuData's all-f32 buffers
actually hurt.

Six arms: SCX backed chunk iteration, SCX eager at uint16, SCX eager at f32
(the like-for-like control that makes the narrowing ratio meaningful), MuData
eager and backed, and modality-scoped CLI pushdown.

Gating: `dataset.multimodal == True`, enforced three ways. The name is in
`run_parallel._MULTIMODAL_BENCHMARKS` and the key prefix in
`_MULTIMODAL_FORMAT_PREFIXES`, so `_triple_compatible`'s three-way XOR pairs it
only with multimodal datasets and only with its own keys; `run()` repeats the
check for direct invocation. `DatasetConfig.path_for_format` resolves these
keys to `h5mu_path` (not `h5ad_path`) so Phase A sees an existing source and
skips conversion, the same no-op the `accel_*` keys get.

Phase-4 trap, recorded here because it is invisible at the call site: because
`path_for_format` resolves these keys to `h5mu_path`, the `converted_path`
Phase B hands `run()` is the **`.h5mu` source**, not an SCX file. The
`scx_stream` / `scx_eager_*` / `scx_query_mod` arms want
`dataset.scx_multimodal_path` and must resolve it themselves rather than
trusting the argument. The `scx_query_mod` arm shells the CLI, and must go
through `scx_cli.resolve_scx_bin` — reading `SCX_CLI_BIN` directly is rejected
by `test_floor_reachability.py::test_scx_cli_probe_has_one_home`.

Fixtures: the 500K Multiome / 1M CITE-seq atlases are Phase 4.1
(`benchmarks/scripts/build_multimodal_atlas.py`). Until they are staged this
benchmark has only the two small real fixtures to run on, and its floors stay
in the Deferred block.

STUB — until Phase 4 implements the arms, `run()` writes a typed missing
result (`missing_reason="phase4_stub"`) and returns `None`, so the gap is
visible in the report instead of looking like a benchmark that ran and found
nothing to do.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "multimodal_atlas_streaming__scx_stream",
    "multimodal_atlas_streaming__scx_eager_u16",
    "multimodal_atlas_streaming__scx_eager_f32",
    "multimodal_atlas_streaming__mudata_h5mu",
    "multimodal_atlas_streaming__mudata_backed",
    "multimodal_atlas_streaming__scx_query_mod",
})


def multimodal_atlas_streaming_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="SCX backed multimodal streaming (64K-cell chunks)",
            key="multimodal_atlas_streaming__scx_stream",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="SCX eager to_mudata (in-decode uint16)",
            key="multimodal_atlas_streaming__scx_eager_u16",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="SCX eager to_mudata (f32 control)",
            key="multimodal_atlas_streaming__scx_eager_f32",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="mudata.read_h5mu (eager f32)",
            key="multimodal_atlas_streaming__mudata_h5mu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="mudata.read_h5mu (backed, chunked)",
            key="multimodal_atlas_streaming__mudata_backed",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="scx query --modality (pushdown)",
            key="multimodal_atlas_streaming__scx_query_mod",
            category="accel", runner="accel_runner",
        ),
    ]


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Phase 4 stub — records nothing yet.

    The multimodal guard returns `None` rather than raising, matching
    `multimodal_read_streaming_vs_inmemory`: the orchestrator already filters
    this pairing out at cohort-build time, so a raise here would only fire on
    direct invocation and would be louder than the situation warrants.
    """
    if not dataset.multimodal:
        logger.info(
            "multimodal_atlas_streaming needs a multimodal dataset; %s is "
            "single-modality (use ml_loader / read_streaming_vs_inmemory)",
            dataset.name,
        )
        return None
    logger.info(
        "multimodal_atlas_streaming is a Phase-4 stub (%s / %s): recording a typed gap",
        dataset.name, format_variant.key,
    )
    # A typed missing result rather than a bare `None`. Both make the
    # orchestrator move on, but `None` is written nowhere and is
    # indistinguishable in the report from a triple that was legitimately not
    # applicable — the suite's recurring "green over nothing measured" shape.
    # This way the coverage gap is visible, and a floor added against a
    # still-stubbed arm fails loudly instead of being skipped in silence
    # (`check_absolute_floors` treats a result that exists with the metric
    # absent as a violation, and a result that does not exist as a skip).
    write_missing_result(
        benchmark="multimodal_atlas_streaming", format_key=format_variant.key,
        dataset=dataset.name, missing_reason="phase4_stub",
        notes="the streaming / narrowing arms are not implemented yet; "
              "only the orchestrator wiring is registered.",
    )
    return None
