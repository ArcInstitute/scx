"""Atlas-scale multimodal streaming and in-decode dtype narrowing.

The suite's existing multimodal benchmarks run on 5.2K-cell CITE-seq and
11.9K-cell Multiome fixtures. SCX's in-decode narrowing —
`to_mudata(data_dtype={"rna": "uint16", "atac": "uint16"})`, which assembles
directly at the target width through `read_all_csr_shards_for_typed` and never
builds the intermediate f32 CSR — has never been demonstrated at the >=500K
scale where MuData's all-f32 buffers actually hurt. The two atlases are built
by `benchmarks/scripts/build_multimodal_atlas.py`.

Six arms: SCX backed chunk iteration, SCX eager at uint16, SCX eager at f32
(the like-for-like control), MuData eager and backed, and modality-scoped CLI
pushdown.

## What the narrowing ratio can and cannot be

The spec asks for `value_buffer_reduction_ratio`, "ratio of f32 buffer bytes to
uint16 buffer bytes", floored at 1.90. Read literally that is **2.0 by
construction** — f32 is four bytes and uint16 is two, whatever the data — so a
floor on it would pass on any build, including one where the narrowing was
removed and the cast happened after a full f32 assembly. And read as
*whole-matrix* bytes it can never reach 1.90: the `indices` array is int32 on
both sides and does not narrow, so `(4+4)/(2+4)` caps the ratio at 1.33.

The claim the arm actually makes is that the f32 CSR is never built, and that
is a statement about **peak RSS**. So `eager_peak_rss_ratio_f32_over_u16` is
measured, by running both reads in two separate children inside the one arm,
and is the number a Phase-5 floor should name. The structural byte counts are
recorded in `metadata` as description, not as a gate metric — emitting a
constant as a gated metric is how a gate goes green over nothing.

## Gating

`dataset.multimodal == True`, enforced three ways. The name is in
`run_parallel._MULTIMODAL_BENCHMARKS` and the key prefix in
`_MULTIMODAL_FORMAT_PREFIXES`, so `_triple_compatible`'s three-way XOR pairs it
only with multimodal datasets and only with its own keys; `run()` repeats the
check for direct invocation.

`DatasetConfig.path_for_format` resolves these keys to `h5mu_path` (not
`h5ad_path`) so Phase A sees an existing source and skips conversion. The
consequence, invisible at the call site: the `converted_path` handed to `run()`
is the **`.h5mu` source**, never an SCX file. The four SCX arms resolve
`dataset.scx_multimodal_path` themselves.

The CLI arm goes through `scx_cli.resolve_scx_bin`; reading `SCX_CLI_BIN`
directly is rejected by `test_floor_reachability::test_scx_cli_probe_has_one_home`.

Process isolation: one fresh child per run, for the reason given in
`benchmarks/comprehensive/subproc_arm.py` — every arm here is floored on peak
RSS, and a parent-side sampler cannot see a child at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    STREAMING_CHUNK_ROWS,
)
from benchmarks.comprehensive.results import (
    BenchmarkResult,
    require_runs,
    write_missing_result,
)
from benchmarks.comprehensive.rss import PeakRssSampler
from benchmarks.comprehensive.scx_cli import resolve_scx_bin
from benchmarks.comprehensive.subproc_arm import run_arm

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "multimodal_atlas_streaming__scx_stream",
    "multimodal_atlas_streaming__scx_eager_u16",
    "multimodal_atlas_streaming__scx_eager_f32",
    "multimodal_atlas_streaming__mudata_h5mu",
    "multimodal_atlas_streaming__mudata_backed",
    "multimodal_atlas_streaming__scx_query_mod",
})

#: `source` picks which file the arm opens — `scx` is
#: `dataset.scx_multimodal_path`, `h5mu` is `dataset.h5mu_path`. Note that
#: `converted_path` is the `.h5mu` for *every* key here, so the SCX arms must
#: not trust it.
_ARMS: dict[str, dict[str, Any]] = {
    "multimodal_atlas_streaming__scx_stream": {"mode": "scx_stream", "source": "scx"},
    "multimodal_atlas_streaming__scx_eager_u16": {"mode": "scx_eager_u16", "source": "scx"},
    "multimodal_atlas_streaming__scx_eager_f32": {"mode": "scx_eager_f32", "source": "scx"},
    "multimodal_atlas_streaming__mudata_h5mu": {"mode": "mudata_eager", "source": "h5mu"},
    "multimodal_atlas_streaming__mudata_backed": {"mode": "mudata_backed", "source": "h5mu"},
    "multimodal_atlas_streaming__scx_query_mod": {"mode": "scx_query", "source": "scx"},
}

NARROW_DTYPE = "uint16"

#: `("query", "--help")` with `--modality` required in its output. The probe is
#: mandatory rather than defaulted in `scx_cli`, and a build predating
#: modality-scoped query would otherwise fail the arm with an opaque usage
#: error instead of a typed skip.
QUERY_PROBE: tuple[str, ...] = ("query", "--help")
QUERY_REQUIRES = b"--modality"

_ARM_TIMEOUT_S = 4 * 3600


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


# ---------------------------------------------------------------------------
# Per-modality sums: the correctness reference
# ---------------------------------------------------------------------------

def _reference_dir() -> Path:
    base = Path(os.environ.get("SCX_BENCH_TMPDIR") or os.environ.get("SCX_WORK_DIR", ""))
    root = base if base.is_dir() else Path("/tmp")
    return root / "scx_bench_multimodal_reference"


def _reference_path(dataset: DatasetConfig, scx_path: Path) -> Path:
    """Keyed on the fixture's identity, not just its name.

    A rebuilt fixture is a different matrix; reusing the old sums would report
    a mismatch on every arm and blame the readers. Size + mtime is enough to
    notice a rebuild without hashing tens of gigabytes.
    """
    try:
        st = scx_path.stat()
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        stamp = "absent"
    tag = hashlib.blake2b(stamp.encode(), digest_size=8).hexdigest()
    return _reference_dir() / f"{dataset.name}.{tag}.json"


def x_sum_f64(x: Any) -> float:
    """Sum a matrix's stored values with a float64 accumulator.

    Not `x.sum()`. scipy accumulates in the array's own dtype, so a float32
    CSR's total is computed in float32 — and on the 5.2K CITE-seq fixture that
    loses exactly one unit out of 32,173,181 against the same values read as
    uint16. The two matrices are bit-identical (verified elementwise); only the
    accumulator differs.

    That is enough to fail an exact `modality_sums_match_int`, permanently, on
    every arm and every dataset — the narrowing arm would be reported as
    corrupting data it reproduces perfectly. Summing both sides in float64
    makes the comparison about the values, which is what it is for.
    """
    data = getattr(x, "data", x)
    return float(np.asarray(data, dtype=np.float64).sum())


def modality_sums_from_scx(scx_path: Path) -> dict[str, float]:
    """Per-modality value totals, read through the plain f32 eager path."""
    import pyscx

    mu = pyscx.open(str(scx_path)).to_mudata()
    return {name: x_sum_f64(mu.mod[name].X) for name in mu.mod}


_REFERENCE_SCRIPT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path

    scx_path, out = sys.argv[1:3]
    from benchmarks.comprehensive.benchmarks.multimodal_atlas_streaming import (
        modality_sums_from_scx,
    )

    sums = modality_sums_from_scx(Path(scx_path))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(sums))
    print(json.dumps({"modalities": sorted(sums)}), flush=True)
""")


def ensure_reference(dataset: DatasetConfig, scx_path: Path) -> dict[str, float] | None:
    """Per-modality sums, computed once per fixture in an untimed child.

    Every arm compares against this rather than against "whichever arm ran
    first": each triple is its own SLURM job, so a peer-to-peer comparison
    would be order-dependent, and the first arm would always have nothing to
    compare to and pass vacuously.
    """
    out = _reference_path(dataset, scx_path)
    if out.exists():
        try:
            return json.loads(out.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("multimodal_atlas_streaming: unreadable reference %s", out)
    logger.info(
        "multimodal_atlas_streaming: computing modality sums for %s (untimed)",
        dataset.name,
    )
    outcome = run_arm(
        _REFERENCE_SCRIPT, [str(scx_path), str(out)],
        timeout_s=_ARM_TIMEOUT_S, label="multimodal_atlas_streaming reference",
    )
    if not outcome.ok or not out.exists():
        logger.warning(
            "multimodal_atlas_streaming: reference failed.\n%s",
            outcome.failure_text("multimodal_atlas_streaming reference"),
        )
        return None
    return json.loads(out.read_text())


def sums_match(observed: dict[str, float], reference: dict[str, float]) -> float:
    """1.0 iff every modality's sum matches the reference exactly.

    Exact, not approximate — but only because every caller sums through
    :func:`x_sum_f64`. These are integer counts under 2^16, so a float64
    accumulator reproduces them exactly on every read path, and a tolerance
    here would hide precisely the narrowing bug the check is for. Summing with
    `X.sum()` instead would compare a float32 accumulation against an exact
    integer one and fail on identical data; see :func:`x_sum_f64`.

    Returns 0.0 when `observed` is empty or names a modality the reference does
    not — never silently "nothing to compare, so it passed".
    """
    if not observed or not reference:
        return 0.0
    if set(observed) != set(reference):
        return 0.0
    return 1.0 if all(observed[k] == reference[k] for k in reference) else 0.0


# ---------------------------------------------------------------------------
# The measured work
# ---------------------------------------------------------------------------

def _iterate_chunks(mod_x: Any, n_obs: int, chunk_rows: int) -> tuple[int, float]:
    """Walk one modality's X in row blocks; return (n_chunks, sum)."""
    total = 0.0
    n_chunks = 0
    for start in range(0, n_obs, chunk_rows):
        stop = min(start + chunk_rows, n_obs)
        block = mod_x[start:stop]
        total += x_sum_f64(block)
        n_chunks += 1
    return n_chunks, total


def run_arm_once(
    mode: str,
    path: Path,
    *,
    chunk_rows: int = STREAMING_CHUNK_ROWS,
    query_modality: str | None = None,
    query_filter: str | None = None,
    scx_bin: str | None = None,
) -> dict[str, Any]:
    """One read of the whole file in *this* process, timed and RSS-sampled."""
    import time

    rec: dict[str, Any] = {"mode": mode}
    sums: dict[str, float] = {}
    data_bytes = 0

    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        if mode == "scx_stream":
            import pyscx

            mu = pyscx.open(str(path)).to_mudata(backed=True)
            n_chunks = 0
            for name in mu.mod:
                ad = mu.mod[name]
                c, s = _iterate_chunks(ad.X, ad.shape[0], chunk_rows)
                n_chunks += c
                sums[name] = s
            rec["n_chunks"] = n_chunks
            rec["n_obs"] = int(mu.n_obs)
        elif mode in ("scx_eager_u16", "scx_eager_f32"):
            import pyscx

            exp = pyscx.open(str(path))
            kwargs = (
                {"data_dtype": NARROW_DTYPE} if mode == "scx_eager_u16" else {}
            )
            mu = exp.to_mudata(**kwargs)
            for name in mu.mod:
                x = mu.mod[name].X
                sums[name] = x_sum_f64(x)
                data_bytes += int(x.data.nbytes)
            rec["n_obs"] = int(mu.n_obs)
            rec["x_dtypes"] = {n: str(mu.mod[n].X.dtype) for n in mu.mod}
        elif mode == "mudata_eager":
            import mudata

            mu = mudata.read_h5mu(str(path))
            for name in mu.mod:
                x = mu.mod[name].X
                sums[name] = x_sum_f64(x)
                data_bytes += int(getattr(x, "data", x).nbytes)
            rec["n_obs"] = int(mu.n_obs)
        elif mode == "mudata_backed":
            import mudata

            mu = mudata.read_h5mu(str(path), backed="r")
            n_chunks = 0
            for name in mu.mod:
                ad = mu.mod[name]
                c, s = _iterate_chunks(ad.X, ad.shape[0], chunk_rows)
                n_chunks += c
                sums[name] = s
            rec["n_chunks"] = n_chunks
            rec["n_obs"] = int(mu.n_obs)
        elif mode == "scx_query":
            import subprocess

            argv = [
                scx_bin, "query", str(path),
                "--modality", query_modality, "--filter", query_filter, "--count",
            ]
            proc = subprocess.run(argv, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"scx query exited {proc.returncode}\n{proc.stderr}"
                )
            rec["query_stdout"] = proc.stdout.strip()[-200:]
            rec["query_modality"] = query_modality
            rec["query_filter"] = query_filter
        else:
            raise ValueError(f"unknown mode {mode!r}")
        wall = time.perf_counter() - t0

    rec["wall_s"] = wall
    rec["peak_rss_mb"] = sampler.peak_mb
    rec["modality_sums"] = sums
    if data_bytes:
        rec["x_data_bytes"] = data_bytes
    if rec.get("n_obs") and mode in ("scx_stream", "mudata_backed"):
        rec["stream_cells_per_sec"] = (
            rec["n_obs"] * max(1, len(sums)) / wall if wall > 0 else float("nan")
        )
    return rec


_WORKER_SCRIPT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path

    mode, path, chunk_rows, modality, predicate, scx_bin = sys.argv[1:7]
    cold = sys.argv[7] == "1"

    from benchmarks.comprehensive.benchmarks.multimodal_atlas_streaming import (
        run_arm_once,
    )

    if cold:
        from benchmarks.comprehensive.cache_control import drop_file_cache
        policy = drop_file_cache(path)
    else:
        policy = "warm"

    rec = run_arm_once(
        mode, Path(path), chunk_rows=int(chunk_rows),
        query_modality=modality or None, query_filter=predicate or None,
        scx_bin=scx_bin or None,
    )
    rec["cache_policy"] = policy
    print(json.dumps(rec), flush=True)
""")


# ---------------------------------------------------------------------------
# The query predicate
# ---------------------------------------------------------------------------

def pick_predicate_column(obs: Any) -> tuple[str, str] | None:
    """`(column, level)` for a selective equality predicate, or None.

    Split out from the I/O so the selection rule is testable without a
    fixture. A single-level column is rejected: the predicate would select
    every row, so the arm would time a full scan and publish it as pushdown.
    """
    preferred = [c for c in ("batch", "donor", "cell_type") if c in obs.columns]
    candidates = preferred + [
        c for c in obs.columns
        if c not in preferred and str(obs[c].dtype) in ("category", "object", "string")
    ]
    for col in candidates:
        values = obs[col].dropna().unique()
        if len(values) < 2:
            continue
        level = str(values[0])
        if "'" in level or '"' in level:
            # The predicate is a shell-quoted string; a level containing a
            # quote would produce a filter expression that parses as something
            # else entirely.
            continue
        return col, level
    return None


def choose_query_predicate(scx_path: Path) -> tuple[str, str, str] | None:
    """`(modality, filter_expr, column)` for the pushdown arm, or None.

    The atlases carry a `batch` column built for this, but the two real
    fixtures carry **no obs columns at all** (verified on
    `cite_seq_pbmc_5k_multimodal.scx`: `read_obs().columns` is empty), so the
    column is discovered rather than assumed. An arm hard-coded to `batch`
    would fail outright on the only multimodal fixtures available until the
    atlases are staged, instead of recording a typed gap.
    """
    import pyscx

    exp = pyscx.open(str(scx_path))
    names = list(exp.modality_names)
    if not names:
        return None
    picked = pick_predicate_column(exp.read_obs())
    if picked is None:
        return None
    col, level = picked
    return names[-1], f"{col} == '{level}'", col


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    key = format_variant.key
    arm = _ARMS.get(key)
    if arm is None:
        logger.warning("multimodal_atlas_streaming: unknown variant %s", key)
        return None

    if not dataset.multimodal:
        # Returning None rather than raising, matching
        # `multimodal_read_streaming_vs_inmemory`: the orchestrator already
        # filters this pairing out at cohort-build time, so a raise would only
        # fire on direct invocation.
        logger.info(
            "multimodal_atlas_streaming needs a multimodal dataset; %s is "
            "single-modality", dataset.name,
        )
        return None

    scx_path = dataset.scx_multimodal_path
    h5mu_path = dataset.h5mu_path
    source_path = scx_path if arm["source"] == "scx" else h5mu_path
    if not source_path.exists():
        hint = (
            "run benchmarks/scripts/build_multimodal_atlas.py, or let Phase A "
            "convert the .h5mu via scx_multimodal_per_modality_auto"
            if arm["source"] == "scx" else
            "run benchmarks/scripts/build_multimodal_atlas.py"
        )
        logger.warning("multimodal_atlas_streaming: %s missing", source_path)
        write_missing_result(
            benchmark="multimodal_atlas_streaming", format_key=key,
            dataset=dataset.name, missing_reason="fixture_missing",
            notes=f"{source_path} does not exist; {hint}",
        )
        return None

    scx_bin = ""
    query_modality = query_filter = query_col = ""
    if arm["mode"] == "scx_query":
        resolved = resolve_scx_bin(QUERY_PROBE, requires=QUERY_REQUIRES)
        if resolved is None:
            write_missing_result(
                benchmark="multimodal_atlas_streaming", format_key=key,
                dataset=dataset.name, missing_reason="no_scx_cli_modality_query",
                notes="no scx binary whose `query --help` advertises --modality; "
                      "build target/release/scx or set SCX_CLI_BIN",
            )
            return None
        scx_bin = resolved
        chosen = choose_query_predicate(scx_path)
        if chosen is None:
            write_missing_result(
                benchmark="multimodal_atlas_streaming", format_key=key,
                dataset=dataset.name, missing_reason="no_selective_obs_column",
                notes="no categorical obs column with >= 2 levels to push down",
            )
            return None
        query_modality, query_filter, query_col = chosen

    reference = ensure_reference(dataset, scx_path) if scx_path.exists() else None

    result = BenchmarkResult(
        benchmark="multimodal_atlas_streaming",
        format=key,
        dataset=dataset.name,
        scenario={
            "name": arm["mode"],
            "engine": "pyscx" if arm["source"] == "scx" else "mudata",
            "cache_state": "cold" if cold_cache else "warm",
            "device": "cpu",
            "chunk_size": STREAMING_CHUNK_ROWS,
        },
        comparison={
            "subject": {"impl": key},
            "baseline": {"impl": "multimodal_atlas_streaming__mudata_h5mu"},
            "metric": "wall_s",
            "status": "pending",
        },
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "source_path": str(source_path),
            "chunk_size": STREAMING_CHUNK_ROWS,
            "modality_names": list(dataset.modality_names),
            "narrow_dtype": NARROW_DTYPE,
            "reference_available": reference is not None,
            "query_modality": query_modality or None,
            "query_filter": query_filter or None,
            "query_column": query_col or None,
        },
    )

    def _argv(mode: str, path: Path) -> list[str]:
        return [
            mode, str(path), str(STREAMING_CHUNK_ROWS),
            query_modality, query_filter, scx_bin,
            "1" if cold_cache else "0",
        ]

    if not cold_cache:
        for i in range(N_WARMUP_RUNS):
            logger.info(
                "multimodal_atlas_streaming warm-up %d/%d for %s",
                i + 1, N_WARMUP_RUNS, key,
            )
            run_arm(_WORKER_SCRIPT, _argv(arm["mode"], source_path),
                    timeout_s=_ARM_TIMEOUT_S, label=f"{key} warmup")

    for i in range(n_runs):
        outcome = run_arm(
            _WORKER_SCRIPT, _argv(arm["mode"], source_path),
            timeout_s=_ARM_TIMEOUT_S, label=f"{key} run {i + 1}",
        )
        if not outcome.ok or not outcome.records:
            raise RuntimeError(
                outcome.failure_text(f"multimodal_atlas_streaming {key} run {i + 1}")
            )
        rec = outcome.records[-1]
        extras: dict[str, Any] = {
            "cache_policy": rec.get("cache_policy", "warm"),
            _rss_key(arm["mode"]): float(rec["peak_rss_mb"]),
        }
        if "stream_cells_per_sec" in rec:
            extras["stream_cells_per_sec"] = float(rec["stream_cells_per_sec"])
        if "n_chunks" in rec:
            extras["n_chunks"] = float(rec["n_chunks"])
        if arm["mode"] == "scx_query":
            extras["modality_query_wall_s"] = float(rec["wall_s"])

        if arm["mode"] == "scx_query":
            # The pushdown arm reads one modality under a predicate, so its
            # totals are deliberately not the whole-file sums. Nothing to
            # compare; the arm is timed, not checked.
            pass
        elif reference is not None:
            extras["modality_sums_match_int"] = sums_match(
                rec.get("modality_sums") or {}, reference,
            )
        else:
            # Omitted, not 0.0. `0.0` asserts that the sums disagree, which is
            # a claim about the reader; what actually happened is that the SCX
            # reference could not be built, so nothing was compared. Both are
            # loud — the gate calls a missing metric on a result that exists a
            # violation — but only one of them is true, and a mudata arm would
            # otherwise be blamed for a missing .scx fixture.
            logger.warning(
                "multimodal_atlas_streaming: no reference for %s; "
                "modality_sums_match_int omitted for %s", dataset.name, key,
            )

        # The narrowing arm measures its own f32 control, in a second child, so
        # the ratio is derivable from one result. The gate cannot join two
        # SLURM jobs, and the two eager arms are two jobs.
        if arm["mode"] == "scx_eager_u16":
            ctrl = run_arm(
                _WORKER_SCRIPT, _argv("scx_eager_f32", source_path),
                timeout_s=_ARM_TIMEOUT_S, label=f"{key} f32 control {i + 1}",
            )
            if ctrl.ok and ctrl.records:
                c = ctrl.records[-1]
                extras["eager_f32_peak_rss_mb"] = float(c["peak_rss_mb"])
                if float(rec["peak_rss_mb"]) > 0:
                    extras["eager_peak_rss_ratio_f32_over_u16"] = (
                        float(c["peak_rss_mb"]) / float(rec["peak_rss_mb"])
                    )
                result.metadata["x_data_bytes__f32"] = c.get("x_data_bytes")
                result.metadata["x_data_bytes__u16"] = rec.get("x_data_bytes")
            else:
                logger.warning(
                    "multimodal_atlas_streaming: f32 control failed; the "
                    "narrowing ratio will be absent for %s", key,
                )

        result.add_run(
            wall_s=float(rec["wall_s"]),
            peak_rss_mb=float(rec["peak_rss_mb"]),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs peak=%.1fMB sums_match=%s%s",
            key, i + 1, rec["wall_s"], rec["peak_rss_mb"],
            extras.get("modality_sums_match_int", "n/a"),
            (f" ratio(f32/u16)={extras['eager_peak_rss_ratio_f32_over_u16']:.2f}"
             if "eager_peak_rss_ratio_f32_over_u16" in extras else ""),
        )
        if i == 0:
            result.metadata["observed_modality_sums"] = rec.get("modality_sums")
            result.metadata["x_dtypes"] = rec.get("x_dtypes")

    require_runs(result, str(source_path))
    matches = [
        r.extra.get("modality_sums_match_int") for r in result.runs
        if r.extra.get("modality_sums_match_int") is not None
    ]
    if matches:
        result.overall_passed = bool(min(matches) >= 1.0)
    logger.info(
        "multimodal_atlas_streaming complete: %s / %s — median %.3fs, passed=%s",
        key, dataset.name, result.median_wall_s or 0.0, result.overall_passed,
    )
    return result


def _rss_key(mode: str) -> str:
    """The peak-RSS metric name each arm owns.

    Per-arm rather than one shared `peak_rss_mb`, because the spec's floors
    name them separately and because `peak_rss_mb` is a reserved `add_run`
    parameter that never reaches `runs[].extra` — a floor on it can never
    resolve.
    """
    return {
        "scx_stream": "streaming_peak_rss_mb",
        "scx_eager_u16": "eager_u16_peak_rss_mb",
        "scx_eager_f32": "eager_f32_peak_rss_mb",
        "mudata_eager": "mudata_eager_peak_rss_mb",
        "mudata_backed": "mudata_backed_peak_rss_mb",
        "scx_query": "query_peak_rss_mb",
    }[mode]
