"""Tests for ``cloud_fixtures.ensure_gcp_credentials_or_skip``.

The cloud_* benchmarks need to skip gracefully when GCP credentials aren't
configured — otherwise they crash hard and pollute the dashboard's
``missing_result`` bucket with undifferentiated failures. The helper writes
a typed skip stub instead so the reporting pipeline can classify them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def isolated_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Re-root ``RAW_RESULTS_DIR`` so the test never writes the production
    ``results/raw/`` directory.
    """
    raw = tmp_path / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # ``results.py`` reads ``RAW_RESULTS_DIR`` from ``config`` at call time,
    # so monkeypatching the config module is enough.
    for modname in ("benchmarks.comprehensive.config", "benchmarks.comprehensive.results"):
        if modname in sys.modules:
            del sys.modules[modname]

    import benchmarks.comprehensive.config as cfg
    monkeypatch.setattr(cfg, "RAW_RESULTS_DIR", raw, raising=False)
    import benchmarks.comprehensive.results as results
    monkeypatch.setattr(results, "RAW_RESULTS_DIR", raw, raising=False)

    return raw


def test_skip_when_no_credentials(
    isolated_results_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Returns False and writes a ``no_gcp_credentials`` stub JSON when neither
    ``GOOGLE_APPLICATION_CREDENTIALS`` nor the default key path resolves."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    import benchmarks.comprehensive.cloud_fixtures as cf
    monkeypatch.setattr(cf, "_DEFAULT_KEY_PATH", Path("/tmp/nope_intentionally_missing.json"))

    ok = cf.ensure_gcp_credentials_or_skip(
        benchmark="cloud_push",
        format_key="scx_auto",
        dataset_name="pbmc3k",
    )

    assert ok is False
    stub = isolated_results_dir / "cloud_push__scx_auto__pbmc3k.json"
    assert stub.exists(), f"Skip stub missing at {stub}"

    payload = json.loads(stub.read_text())
    assert payload["missing_reason"] == "no_gcp_credentials"
    assert payload["benchmark"] == "cloud_push"
    assert payload["format"] == "scx_auto"
    assert payload["dataset"] == "pbmc3k"


def test_continue_when_credentials_present(
    isolated_results_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Returns True without writing a stub when a readable key file is set."""
    key_file = tmp_path / "fake_key.json"
    key_file.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_file))

    import benchmarks.comprehensive.cloud_fixtures as cf
    ok = cf.ensure_gcp_credentials_or_skip(
        benchmark="cloud_pull",
        format_key="scx_auto",
        dataset_name="pbmc3k",
    )

    assert ok is True
    stub = isolated_results_dir / "cloud_pull__scx_auto__pbmc3k.json"
    assert not stub.exists(), (
        f"Stub should NOT be written when credentials resolve: {stub}"
    )


def test_falls_back_to_default_key_when_env_unset(
    isolated_results_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When GOOGLE_APPLICATION_CREDENTIALS is unset but ~/.gcp/scx-bench.json
    exists, the helper picks it up and returns True."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    default_key = tmp_path / "default_key.json"
    default_key.write_text("{}")

    import benchmarks.comprehensive.cloud_fixtures as cf
    monkeypatch.setattr(cf, "_DEFAULT_KEY_PATH", default_key)

    ok = cf.ensure_gcp_credentials_or_skip(
        benchmark="cloud_metadata",
        format_key="scx_auto",
        dataset_name="pbmc3k",
    )

    assert ok is True


# ---------------------------------------------------------------------------
# The `scx info <cloud-url>` CLI arm
# ---------------------------------------------------------------------------


def _fake_scx(tmp_path, stdout: str, rc: int = 0, cloud: bool = True):
    """A stand-in `scx` that answers the cloud probe and prints *stdout*.

    Two subcommands, because the arm asks the binary two different questions:
    `pull --help` is the cloud-feature probe (a clap-gated cloud subcommand —
    `info --help` exits 0 on a build with no cloud support at all, so probing
    `info` would resolve a binary that then fails on the URL), and
    `info --json <url>` is the measurement. ``cloud=False`` fails the probe,
    standing in for a stock build.
    """
    import stat

    path = tmp_path / "scx"
    path.write_text(
        "#!/usr/bin/env bash\n"
        + ('if [ "$1" = "pull" ]; then echo "Download from cloud"; exit 0; fi\n'
           if cloud else "")
        + 'if [ "$1" = "info" ]; then cat <<\'JSON\'\n'
        + f"{stdout}\n"
        + "JSON\n"
        + f"exit {rc}; fi\n"
        + "exit 64\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _pin_candidates(monkeypatch, *paths):
    """Make the shared resolver see exactly *paths*.

    Patching `candidates` rather than `$SCX_CLI_BIN` is deliberate. The env
    var is a *preference*, not a pin: a candidate that fails the probe falls
    through to `target/release/scx` and then to PATH, which is the right
    behaviour and which made a first version of the skip test below resolve
    the real cloud-capable binary and issue a live GET against a bucket that
    does not exist. A test must not be able to reach the network by
    misconfiguration.
    """
    from benchmarks.comprehensive import scx_cli

    monkeypatch.setattr(scx_cli, "candidates", lambda: [str(p) for p in paths])


def _cli_arm(monkeypatch, tmp_path, stdout: str, rc: int = 0, n_runs: int = 1):
    """Drive `_run_cli_info_arm` against the fake binary; return the result.

    Seeded with *n_runs* in-process runs, because the arm attaches its metrics
    to those rather than appending its own — that is what keeps it from pooling
    two incomparable operations into `median_wall_s`.
    """
    from benchmarks.comprehensive.benchmarks import cloud_metadata
    from benchmarks.comprehensive.config import DATASETS
    from benchmarks.comprehensive.results import BenchmarkResult

    _pin_candidates(monkeypatch, _fake_scx(tmp_path, stdout, rc))
    dataset = DATASETS["pbmc3k"]
    assert dataset.n_obs == 2700, "premise: pbmc3k is 2700 cells"
    result = BenchmarkResult(
        benchmark="cloud_metadata", format="scx_auto", dataset=dataset.name,
    )
    for _ in range(n_runs):
        result.add_run(wall_s=0.25, peak_rss_mb=110.0, library_open=True)
    cloud_metadata._run_cli_info_arm(
        result, dataset, "gs://example/pbmc3k.scxd/", n_runs=n_runs,
    )
    return result


def test_cli_info_arm_records_ok_on_a_matching_n_obs(monkeypatch, tmp_path):
    """The happy path: exit 0 and the right `n_obs` -> `scx_info_cloud_ok=1`."""
    result = _cli_arm(monkeypatch, tmp_path, '{"n_obs": 2700, "n_vars": 32738}')
    assert len(result.runs) == 1
    extra = result.runs[0].extra
    assert extra["scx_info_cloud_ok"] == 1
    assert extra["scx_info_n_obs"] == 2700
    assert extra["wall_s__scx_info_cloud"] > 0
    # No `scenario` of its own: the metrics ride the in-process run, which is
    # what keeps the arm from shifting the triple's pooled medians.
    assert "scenario" not in extra
    assert extra["library_open"] is True


def test_cli_info_arm_refuses_an_exit_zero_that_answered_the_wrong_file(
    monkeypatch, tmp_path,
):
    """Exit 0 is not enough — the JSON has to describe the right dataset.

    A CLI that exits 0 having printed something useless (a truncated read, a
    stale cached catalog, the wrong URL resolved) would otherwise be recorded
    as a very fast open. The wall in that case is real but measures a failure,
    which is worse than a missing number because it looks like an improvement.
    """
    result = _cli_arm(monkeypatch, tmp_path, '{"n_obs": 1, "n_vars": 1}')
    extra = result.runs[0].extra
    assert extra["scx_info_cloud_ok"] == 0, (
        "an n_obs that disagrees with the dataset must not count as success"
    )
    assert extra["scx_info_n_obs"] == 1, "the observed value is recorded, not hidden"
    # Still recorded, so a `scx_info_cloud_ok >= 1` floor fails on it rather
    # than the triple going quiet.
    assert extra["wall_s__scx_info_cloud"] > 0


def test_cli_info_arm_refuses_unparseable_output(monkeypatch, tmp_path):
    """Garbage on stdout is a failure, not a crash."""
    result = _cli_arm(monkeypatch, tmp_path, "not json at all")
    extra = result.runs[0].extra
    assert extra["scx_info_cloud_ok"] == 0
    assert extra["scx_info_n_obs"] == -1


def test_cli_info_arm_skips_with_a_reason_when_no_cloud_build_exists(
    monkeypatch, tmp_path,
):
    """No cloud-capable binary -> a recorded reason and zero runs.

    Zero runs is the loud outcome for a threshold (missing metric), but only
    if the reason is findable, so it goes into `metadata` rather than a log
    line alone.
    """
    from benchmarks.comprehensive.benchmarks import cloud_metadata
    from benchmarks.comprehensive.config import DATASETS
    from benchmarks.comprehensive.results import BenchmarkResult

    # A binary with no cloud support: `pull` is not compiled in, so the probe
    # rejects it — and `info --help` would have accepted it, which is why the
    # probe is `pull`. It is the *only* candidate, because the resolver
    # correctly falls through to the next one otherwise.
    _pin_candidates(
        monkeypatch, _fake_scx(tmp_path, "{}", rc=0, cloud=False),
    )

    result = BenchmarkResult(
        benchmark="cloud_metadata", format="scx_auto", dataset="pbmc3k",
    )
    cloud_metadata._run_cli_info_arm(
        result, DATASETS["pbmc3k"], "gs://example/pbmc3k.scxd/", n_runs=2,
    )
    assert not result.runs, "a skipped arm must record no runs"
    assert "--features cloud" in result.metadata["cli_info_skipped_reason"]


def test_cli_info_arm_refuses_a_file_with_the_right_n_obs_and_wrong_n_vars(
    monkeypatch, tmp_path,
):
    """`n_obs` alone accepts a different dataset with the same cell count.

    Reproduced by a reviewer: `{"n_obs": 2700, "n_vars": 1}` was recorded as
    `scx_info_cloud_ok=1` for pbmc3k. A stale catalog, the wrong URL resolved,
    or a truncated read can all land there, and the wall would go on record as
    a successful — very fast — open.

    Both axes are compared now, which is also why the happy-path test above
    carries pbmc3k's real `n_vars` (32738) rather than any plausible integer.
    """
    result = _cli_arm(monkeypatch, tmp_path, '{"n_obs": 2700, "n_vars": 1}')
    extra = result.runs[0].extra
    assert extra["scx_info_cloud_ok"] == 0, (
        "an n_vars that disagrees with the dataset must not count as success"
    )
    assert extra["scx_info_n_obs"] == 2700
    assert extra["scx_info_n_vars"] == 1, "the observed value is recorded"


def test_cli_info_arm_adds_no_runs_and_leaves_the_pooled_medians_alone(
    monkeypatch, tmp_path,
):
    """The arm must not append runs — it pools two incomparable operations.

    `BenchmarkResult.median_wall_s` and `capture_baseline._median_rss` take a
    median over **every** run of the triple. An `open_cloud` catalog read is
    sub-second; `scx info` range-reads one header per CSR shard and was
    measured at ~175 s on the largest fixture. Appending the CLI runs put the
    pooled figure between the two, describing neither — and the first version
    of the arm additionally omitted `peak_rss_mb=` entirely, so `add_run`
    supplied `0.0` and the pooled RSS was an average of zeros and real
    samples.

    Attaching to the existing runs gives `_load_current_raw_metric` the same
    median over the sparse key, with `n_runs`, `median_wall_s`, `wall_s_iqr`
    and `peak_rss_mb_median` all untouched.
    """
    result = _cli_arm(
        monkeypatch, tmp_path, '{"n_obs": 2700, "n_vars": 32738}', n_runs=3,
    )
    assert len(result.runs) == 3, (
        f"the arm appended runs ({len(result.runs)} for 3 seeded); that pools "
        f"a multi-second CLI call with a sub-second catalog open"
    )
    assert result.median_wall_s == 0.25, (
        f"pooled median_wall_s moved to {result.median_wall_s}; the in-process "
        f"arm recorded 0.25 s per run and this arm must not shift it"
    )
    for run in result.runs:
        assert run.peak_rss_mb == 110.0, "the seeded RSS must survive untouched"
        assert run.extra["library_open"] is True, "attached, not replaced"
        # The harness's own footprint, labelled as such — not a peak, and not
        # the run's reserved `peak_rss_mb`.
        assert run.extra["harness_rss_mb__scx_info_cloud"] > 0.0
        assert run.extra["harness_entry_rss_mb__scx_info_cloud"] > 0.0
        assert "peak_rss_mb__scx_info_cloud" not in run.extra, (
            "the harness's RSS while it blocks in waitpid is not a peak of "
            "anything; naming it one is what PR-02a had to undo three times"
        )
        assert run.extra["wall_s__scx_info_cloud"] > 0.0
