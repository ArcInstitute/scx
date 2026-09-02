"""
Cloud metadata-open latency benchmark.

Cross-format metadata-only open latency:
  * SCX: ``pyscx.open_cloud(url)`` + touch ``n_obs / n_vars / nnz``
  * Zarr: ``zarr.open(url)`` + touch ``attrs['shape']``
  * TileDB-SOMA: ``Experiment.open(url)`` + touch obs/var counts
  * SLAF: ``SLAFArray(url)`` + touch ``shape``

Each runner's ``read_cloud_metadata(url)`` helper does the minimal open
and touches only schema-level properties; array reads are excluded so the
benchmark measures first-GET / catalog-parse latency, not bandwidth.
Returns ``None`` for formats whose runner does not provide the helper
(silent skip).

A second, SCX-only arm times the **CLI**: ``scx info --json <url>`` as a
subprocess (``scenario="scx_info_cloud"``). It is not a duplicate of the
in-process arm. ``scx info`` fills two columns the library open does not — the
per-shard codec breakdown and the value-encoding summary — by range-reading
every CSR shard's 76-byte header at ``METADATA_SHARD_FETCH_CONCURRENCY = 8``,
which is the concurrency OPT-CLOUD-1 raises. So the two arms differ by
thousands of GETs on an atlas-scale fixture, and only this one moves when that
lands. Requires a ``--features cloud`` build; skips with a recorded reason
otherwise. See ``_run_cli_info_arm``.
"""

from __future__ import annotations

import gc
import json
import logging
import subprocess
import time
from pathlib import Path

from benchmarks.comprehensive.cloud_fixtures import (
    ensure_cloud_fixture,
    ensure_gcp_credentials_or_skip,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCS_TEST_BUCKET,
    N_WARMUP_RUNS,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import current_rss_mb
from benchmarks.comprehensive.scx_cli import CLOUD_PROBE, resolve_scx_bin
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

REQUIRED_CAPABILITIES: frozenset[str] = frozenset({"cloud_metadata"})
"""Runner-capability requirement — read by ``run_parallel.py``'s cohort
builder so incompatible (bench, format) cells never get submitted. Mirrors
the runtime guard at the top of ``run()`` (defense-in-depth for direct
invocation)."""

# A cloud `scx info` range-reads one header per CSR shard at concurrency 8;
# on the largest fixture that is ~16k GETs and measured ~175 s pre-OPT-CLOUD-1.
# The cap is a runaway guard, not a budget.
_CLI_INFO_TIMEOUT_S = 900

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})
"""Format-key allow-list — the only SCX layout with a cloud fixture suffix
(``config._FORMAT_KEY_TO_CLOUD_SUFFIX``). The Scx1 ``compact_trial_g*``
variants declare the ``cloud_metadata`` capability (via ``scx_runner``) but
have no cloud layout, so ``DatasetConfig.cloud_url`` raises ``ValueError``
for them — pinning the allow-list keeps them out of the cohort (a failed
cloud job otherwise breaks the next cohort's ``afterok:`` dependency). Read
by ``run_parallel.py``'s cohort builder; mirrored by the runtime guard in
``run()``."""


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    provider: str = "gcs",
) -> BenchmarkResult | None:
    if format_variant.key not in SUPPORTED_FORMATS:
        return None
    if provider != "gcs":
        raise ValueError(
            f"Only 'gcs' provider is supported in Phase 5 (got {provider!r})"
        )

    runner = make_runner(format_variant)
    if "cloud_metadata" not in runner.capabilities:
        logger.info(
            "Skipping cloud_metadata for %s — runner does not declare cloud_metadata",
            format_variant.key,
        )
        return None
    open_metadata = runner.read_cloud_metadata

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted {format_variant.key} file for {dataset.name}. "
            f"Run conversion first (--formats {format_variant.key})."
        )

    if not ensure_gcp_credentials_or_skip(
        benchmark="cloud_metadata",
        format_key=format_variant.key,
        dataset_name=dataset.name,
    ):
        return None
    cloud_url = ensure_cloud_fixture(
        dataset, format_variant, Path(converted_path), provider=provider,
    )

    result = BenchmarkResult(
        benchmark="cloud_metadata",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "n_warmup": N_WARMUP_RUNS,
            "cold_cache": cold_cache,
        },
    )
    result.file_size_bytes = runner.file_size(Path(converted_path))

    for i in range(N_WARMUP_RUNS):
        logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
        open_metadata(cloud_url)
        gc.collect()

    for i in range(n_runs):
        if cold_cache:
            runner._drop_caches()
        gc.collect()

        logger.info("cloud_metadata run %d/%d ← %s", i + 1, n_runs, cloud_url)
        timing = open_metadata(cloud_url)
        result.add_run(
            wall_s=timing.wall_s,
            user_s=timing.user_s,
            sys_s=timing.sys_s,
            peak_rss_mb=timing.peak_rss_mb,
            **(timing.extra or {}),
        )
        logger.info("  wall=%.6fs", timing.wall_s)

    _run_cli_info_arm(result, dataset, cloud_url)

    logger.info(
        "cloud_metadata complete: %s / %s — median %.6fs",
        format_variant.key,
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result


def _run_cli_info_arm(
    result: BenchmarkResult,
    dataset: DatasetConfig,
    cloud_url: str,
) -> None:
    """Time `scx info <cloud-url>` as a subprocess.

    The in-process arm above measures `pyscx.open_cloud` + touching schema
    properties. `scx info` is a *different* code path with a different cost —
    it fills two columns (the per-shard codec breakdown and the value-encoding
    summary) by range-reading **every** CSR shard's 76-byte header, at
    `METADATA_SHARD_FETCH_CONCURRENCY = 8`. That is what OPT-CLOUD-1 changes,
    and nothing measured it, so a 175 s → ~22 s improvement had nowhere to
    land.

    Its numbers ride the in-process runs rather than adding new ones — see the
    comment at the loop for why, and note that this is what makes the arm
    composition-neutral and so needs no justification file.

    It records no RSS *peak*: the harness blocks in `waitpid` while the child
    works, so there is nothing to sample. Two `harness_*_rss_mb` readings
    bracket the call as a diagnostic; the child's own peak would need `wait4`
    rusage.

    Skips with a recorded reason, never raises. Two independent reasons it can
    be unavailable and they need telling apart:

    * **no cloud-capable binary.** `scx info` accepts a URL on any build, but
      its cloud branch is `#[cfg(feature = "cloud")]` and `cloud` is not in
      `scx-cli`'s default features, so a stock build fails the URL at run time
      with "requires scx-cli built with --features cloud". The probe therefore
      tests a clap-gated cloud subcommand (`pull --help`), not `info --help`,
      which exits 0 either way.
    * **credentials.** Already handled upstream: `run()` returns before
      resolving the fixture when `ensure_gcp_credentials_or_skip` fails, so
      this arm is never reached without them.

    Success is asserted on the *content*, not the exit code alone: `--json`'s
    `n_obs` **and** `n_vars` must both match the dataset's own. An exit-0 that
    printed nothing useful would otherwise be recorded as a fast open, and
    `n_obs` alone accepts any file with the same cell count.
    """
    scx_bin = resolve_scx_bin(CLOUD_PROBE)
    if scx_bin is None:
        reason = (
            "no `scx` binary with cloud support: probed $SCX_CLI_BIN, "
            "target/release/scx and PATH for a build whose clap-gated `pull` "
            "subcommand exists. Build one with "
            "`cargo build -p scx-cli --release --features cloud`."
        )
        logger.warning("  scx-info-cloud arm skipped: %s", reason)
        result.metadata["cli_info_skipped_reason"] = reason
        return

    result.metadata["cli_info_bin"] = scx_bin
    # One CLI call per existing in-process run, and its numbers are attached to
    # that run rather than appended as new ones.
    #
    # Appending was the first design and it pooled two incomparable operations:
    # `BenchmarkResult.median_wall_s` is a median over every run, so three
    # sub-second `open_cloud` catalog reads beside three multi-second `scx
    # info` calls produced a figure between them that described neither. The
    # RSS side was worse — the sampled peak is the *harness's* footprint while
    # a child does the work, so pooling it into `peak_rss_mb_median` mixed in a
    # number that is not about either operation.
    #
    # Attaching instead leaves `median_wall_s`, `wall_s_iqr`, `n_runs` and
    # `peak_rss_mb_median` exactly as the in-process arm left them, and
    # `_load_current_raw_metric` medians the sparse `wall_s__scx_info_cloud`
    # over the runs that carry it — the same gate reading with no composition
    # change at all. That is also why this arm needs no justification file.
    targets = list(result.runs)
    if not targets:
        result.metadata["cli_info_skipped_reason"] = (
            "no in-process runs to attach to; the library arm recorded nothing"
        )
        return
    verdicts: list[int] = []
    for i in range(len(targets)):
        gc.collect()
        entry_rss = current_rss_mb()
        t0 = time.perf_counter()
        # `timeout=` raises rather than returning, and a cloud hang must not
        # take the whole cohort job with it — the docstring above promises this
        # arm never raises. A timed-out run is recorded as ok=0 with its (capped)
        # wall, which is the honest reading: it did take at least that long.
        try:
            proc = subprocess.run(
                [scx_bin, "info", "--json", cloud_url],
                capture_output=True, text=True, timeout=_CLI_INFO_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "  scx info %s exceeded the %ds cap", cloud_url,
                _CLI_INFO_TIMEOUT_S,
            )
            proc = None
        except Exception as e:  # noqa: BLE001
            logger.error("  scx info %s could not run: %s", cloud_url, e)
            proc = None
        wall = time.perf_counter() - t0

        ok = 0
        n_obs_seen = -1
        n_vars_seen = -1
        if proc is not None and proc.returncode == 0:
            try:
                payload = json.loads(proc.stdout)
                n_obs_seen = int(payload.get("n_obs", -1))
                n_vars_seen = int(payload.get("n_vars", -1))
            except Exception:  # noqa: BLE001
                n_obs_seen = -1
                n_vars_seen = -1
            # BOTH axes. `n_obs` alone accepts any file with the same cell
            # count — a stale catalog, the wrong URL resolved, a different
            # fixture — and would record its (fast) wall as a successful open.
            # Multimodal datasets sum n_vars across modalities in the config
            # and are out of this arm's scope anyway (SUPPORTED_FORMATS is
            # single-modality scx_auto).
            ok = 1 if (
                n_obs_seen == dataset.n_obs and n_vars_seen == dataset.n_vars
            ) else 0
        if not ok:
            # Recorded rather than raised — one arm must not fail the cohort —
            # but recorded *loudly*: the wall below timed a failure, and the
            # aggregated `scx_info_cloud_ok` written after the loop is 0 for
            # the whole arm, so a `min: 1.0` floor fails even if the other
            # runs succeeded.
            logger.error(
                "  scx info %s failed (rc=%s, n_obs=%s want %d, "
                "n_vars=%s want %d): %s",
                cloud_url,
                "timeout/spawn" if proc is None else proc.returncode,
                n_obs_seen, dataset.n_obs, n_vars_seen, dataset.n_vars,
                "" if proc is None else (proc.stderr or "").strip()[:400],
            )
        # NOT a peak, and not called one. The work happens in a child process
        # while this one blocks in `waitpid`, so the harness allocates nothing
        # and its RSS is flat for the duration — a `PeakRssSampler` here polls
        # `/proc/self/statm` every 5 ms to rediscover a constant. Two readings
        # bracket it instead, and the name says whose memory it is.
        #
        # Measuring the *child's* peak would need `wait4` rusage, which is the
        # same gap `doublet_interop`'s `residual_rss_mb` documents.
        verdicts.append(ok)
        targets[i].extra.update({
            "wall_s__scx_info_cloud": round(wall, 6),
            "harness_rss_mb__scx_info_cloud": round(current_rss_mb(), 1),
            "harness_entry_rss_mb__scx_info_cloud": round(entry_rss, 1),
            # This run's own verdict, for diagnosis. NOT the gateable key —
            # see the aggregate below.
            "scx_info_cloud_run_ok": ok,
            "scx_info_n_obs": n_obs_seen,
            "scx_info_n_vars": n_vars_seen,
        })
        logger.info(
            "  scx info (cloud) run %d/%d: wall=%.3fs ok=%d",
            i + 1, len(targets), wall, ok,
        )

    # `_load_current_raw_metric` takes the **median** over the runs carrying a
    # key, so a per-run 0/1 written straight to the gateable name hides a
    # minority of failures: `[0, 1, 1]` medians to 1.0 and reads as success.
    # The aggregate is `min`, written onto every carrier, so one failed
    # invocation drives the metric to 0 and a `min: 1.0` floor fails — which is
    # what the arm's own docstring and Deferred item 19 claim. Each run's own
    # verdict stays under `scx_info_cloud_run_ok` for diagnosis.
    if verdicts:
        aggregate = min(verdicts)
        for run_rec in targets:
            if "scx_info_cloud_run_ok" in run_rec.extra:
                run_rec.extra["scx_info_cloud_ok"] = aggregate
        result.metadata["cli_info_runs_ok"] = verdicts
        if aggregate == 0:
            logger.error(
                "  scx info (cloud): %d of %d runs failed; "
                "scx_info_cloud_ok=0", verdicts.count(0), len(verdicts),
            )
