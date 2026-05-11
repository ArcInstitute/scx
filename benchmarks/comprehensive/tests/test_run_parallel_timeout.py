"""
sacct fallback for TIMEOUT-reaped jobs in `run_parallel.py`.

Covers ``_sacct_head_state`` and ``_job_terminal_head``: when
``submitit.Job.state`` returns empty (job reaped from squeue) the
wait loop must consult sacct instead of falling through to
``.result()`` — a bare ``.result()`` on a reaped TIMEOUT job hangs
because submitit cannot find the result pickle and waits forever
for one to appear.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module import — sidestep heavy benchmark deps by stubbing them
# ---------------------------------------------------------------------------


def _import_run_parallel():
    """Import run_parallel.py with its benchmark/config deps stubbed.

    The module's submitit / scx-bench imports aren't needed for the
    helpers under test; stubbing keeps the test fast and avoids a
    transitive cargo build on test collection.
    """
    project = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project))

    # Minimal submitit stub
    fake_submitit = types.ModuleType("submitit")

    class _FakeAutoExecutor:
        def __init__(self, *_a, **_kw):
            pass

    class _FakeJob:
        pass

    fake_submitit.AutoExecutor = _FakeAutoExecutor
    fake_submitit.Job = _FakeJob
    sys.modules.setdefault("submitit", fake_submitit)

    # Import after stubs are in place
    from benchmarks.comprehensive.scripts import run_parallel  # noqa: E402
    return run_parallel


@pytest.fixture
def rp(monkeypatch: pytest.MonkeyPatch):
    mod = _import_run_parallel()
    # Fresh cache per test
    monkeypatch.setattr(mod, "_SACCT_STATE_CACHE", {}, raising=True)
    return mod


# ---------------------------------------------------------------------------
# Job stubs
# ---------------------------------------------------------------------------


class _StubJob:
    """Job stub with a state sequence to mimic squeue reaping.

    Each ``state`` access pops the next entry; once exhausted, returns
    the last value (or "" if the sequence ended empty). Mirrors how
    submitit.Job.state behaves over the job's lifetime — RUNNING,
    TIMEOUT (briefly visible in squeue), then "" after reaping.
    """

    def __init__(self, job_id: str, states: list[str]):
        self.job_id = job_id
        self._states = list(states)
        self._last = ""

    @property
    def state(self) -> str:
        if self._states:
            self._last = self._states.pop(0)
        return self._last


# ---------------------------------------------------------------------------
# _sacct_head_state
# ---------------------------------------------------------------------------


def test_sacct_head_state_parses_timeout(rp, monkeypatch):
    """sacct's TIMEOUT row → 'TIMEOUT'."""
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: "TIMEOUT\n",
    )
    assert rp._sacct_head_state("9999") == "TIMEOUT"


def test_sacct_head_state_normalises_cancelled(rp, monkeypatch):
    """'CANCELLED by 10024' → 'CANCELLED' (first whitespace-delimited token)."""
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: "CANCELLED by 10024\n",
    )
    assert rp._sacct_head_state("9999") == "CANCELLED"


def test_sacct_head_state_normalises_plus_suffix(rp, monkeypatch):
    """'CANCELLED+0:0' → 'CANCELLED' (split on '+' before whitespace)."""
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: "CANCELLED+0:0\n",
    )
    assert rp._sacct_head_state("9999") == "CANCELLED"


def test_sacct_head_state_caches_first_lookup(rp, monkeypatch):
    """Repeated lookup of the same id hits cache (one shell-out only)."""
    calls: list[tuple] = []

    def _fake(*args, **_kw):
        calls.append(args)
        return "TIMEOUT\n"

    monkeypatch.setattr(subprocess, "check_output", _fake)
    assert rp._sacct_head_state("123") == "TIMEOUT"
    assert rp._sacct_head_state("123") == "TIMEOUT"
    assert len(calls) == 1


def test_sacct_head_state_does_not_cache_empty(rp, monkeypatch):
    """Empty sacct response is not cached — retry on next call."""
    calls: list[int] = []

    def _fake(*_a, **_kw):
        calls.append(1)
        return "\n"

    monkeypatch.setattr(subprocess, "check_output", _fake)
    assert rp._sacct_head_state("123") == ""
    assert rp._sacct_head_state("123") == ""
    assert len(calls) == 2


def test_sacct_head_state_swallows_subprocess_errors(rp, monkeypatch):
    """sacct unreachable → empty string, not raise."""

    def _fake(*_a, **_kw):
        raise subprocess.CalledProcessError(1, "sacct")

    monkeypatch.setattr(subprocess, "check_output", _fake)
    assert rp._sacct_head_state("123") == ""


# ---------------------------------------------------------------------------
# _job_terminal_head — TIMEOUT-reap regression test
# ---------------------------------------------------------------------------


def test_terminal_head_uses_live_state_when_available(rp, monkeypatch):
    """Live job.state='TIMEOUT' is used directly; sacct is NOT consulted."""
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: pytest.fail("sacct should not be called"),
    )
    job = _StubJob("123", states=["TIMEOUT"])
    assert rp._job_terminal_head(job) == "TIMEOUT"


def test_terminal_head_normalises_live_state(rp, monkeypatch):
    """'CANCELLED by 10024' from live state → 'CANCELLED'."""
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: pytest.fail("sacct should not be called"),
    )
    job = _StubJob("123", states=["CANCELLED by 10024"])
    assert rp._job_terminal_head(job) == "CANCELLED"


def test_terminal_head_falls_back_to_sacct_when_state_empty(rp, monkeypatch):
    """Empty live state → consult sacct (the TIMEOUT-reap fix)."""
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: "TIMEOUT\n",
    )
    job = _StubJob("123", states=[""])
    assert rp._job_terminal_head(job) == "TIMEOUT"


def test_terminal_head_timeout_then_reap_sequence(rp, monkeypatch):
    """Simulates the run #6 hang: state RUNNING → TIMEOUT → '' across polls.

    On the iteration that observes empty state, sacct fallback fires
    and returns the historical TIMEOUT — fast-fail predicate matches,
    no .result() hang. This is the regression scenario the fix exists
    for; without sacct fallback, the loop falls through to .result()
    and blocks indefinitely on a reaped job whose result pickle was
    never written.
    """
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: "TIMEOUT\n",
    )
    job = _StubJob("123", states=["RUNNING", "TIMEOUT", ""])
    # Consume the first two states (simulating earlier poll iterations).
    _ = job.state  # RUNNING
    _ = job.state  # TIMEOUT
    # Now on the iteration with empty state, fallback should fire.
    assert rp._job_terminal_head(job) == "TIMEOUT"


def test_terminal_head_returns_empty_when_both_empty(rp, monkeypatch):
    """Neither live state nor sacct has anything → empty string.

    The wait loop's fast-fail predicate then returns False and the
    caller falls through to .result(). Acceptable: this is a job
    slurm has truly lost track of, not a stuck-after-TIMEOUT case.
    """
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: "\n")
    job = _StubJob("123", states=[""])
    assert rp._job_terminal_head(job) == ""


def test_terminal_head_completed_not_treated_as_failure(rp, monkeypatch):
    """sacct says COMPLETED — head is 'COMPLETED', NOT in the failure set.

    The wait loop's fast-fail predicate must not match COMPLETED,
    so the caller falls through to .result() and reads the benchmark
    output normally.
    """
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *_a, **_kw: "COMPLETED\n",
    )
    job = _StubJob("123", states=[""])
    head = rp._job_terminal_head(job)
    assert head == "COMPLETED"
    failure_set = {"CANCELLED", "TIMEOUT", "NODE_FAIL", "FAILED",
                   "OUT_OF_MEMORY", "PREEMPTED"}
    assert head not in failure_set
