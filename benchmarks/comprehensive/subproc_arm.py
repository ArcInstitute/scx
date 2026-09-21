"""Run one benchmark arm in a fresh process and read its self-reported records.

Why a process boundary at all: :class:`benchmarks.comprehensive.rss.PeakRssSampler`
polls ``/proc/self/statm``, so it measures **the process it runs in**. A parent
sampling across a child sees nothing — measured at 0.0 MB across a child that
allocated 500 MB — and ``getrusage(RUSAGE_CHILDREN).ru_maxrss`` is no substitute
because it is a high-water mark over *every* child reaped so far, not a per-child
figure (500 MB child then a 50 MB child still reports 531.6 MB for the second).

So an arm whose peak RSS is gated has to be the thing doing the sampling. On top
of that, ``PeakRssSampler.__enter__`` seeds itself with the entry RSS, and glibc
does not return large freed arenas, so two arms in one interpreter make the
second inherit the first's retained heap. `conversion_streaming` measured exactly
that: 1722 -> 2983 -> 3472 MB across three identical runs of an arm that a fresh
process puts at 1771 MB.

Four modules already solve this privately (``conversion_streaming``,
``export_streaming``, ``parallel_write_scaling``, ``parallel_scaling``). This is
the same shape factored out for the community benchmarks, with two differences
that those four do not need:

* **Every** JSON line on stdout is collected, not just the last. A worker that is
  SIGKILLed part-way leaves a partial trail, and for `pipeline_ooc_constrained`
  that trail is the measurement — it names the stage the memory ceiling was hit
  in. The existing modules take ``splitlines()[-1]`` because their workers print
  once at the end and cannot be killed by design.
* A non-zero exit is **returned, not raised**. An OOM kill is data for the
  "laptop test", not an error; the caller decides which return codes are results
  and which are failures.

The child applies its own resource policy rather than having the parent do it in
a ``preexec_fn``: between ``fork`` and ``exec`` only the forking thread exists,
so anything that can take an allocator lock there can deadlock, and this suite
runs sampler threads. :data:`WORKER_PRELUDE` is prepended to a worker source and
reads the two env vars this module sets.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = ["ArmOutcome", "WORKER_PRELUDE", "run_arm", "OOM_RETURNCODES"]

#: Exit statuses that mean "the kernel killed this, it did not choose to stop".
#: ``subprocess`` reports a signal death as a negative number; a shell in between
#: (or SLURM's own reporting) turns SIGKILL into 137. Both spellings appear
#: depending on whether the worker was execed directly.
OOM_RETURNCODES = frozenset({-9, 137})

_ENV_OOM_FIRST = "SCX_BENCH_ARM_OOM_FIRST"
_ENV_MEM_LIMIT = "SCX_BENCH_ARM_MEM_LIMIT_BYTES"


WORKER_PRELUDE = textwrap.dedent('''\
    # --- injected by benchmarks.comprehensive.subproc_arm -----------------
    # Applied before any heavy import so the policy is in force for the whole
    # arm, and applied by the child itself rather than by a parent preexec_fn
    # (which runs between fork and exec, where an allocator lock held by another
    # thread deadlocks the child).
    import os as _sa_os, resource as _sa_resource, sys as _sa_sys
    if _sa_os.environ.get("SCX_BENCH_ARM_OOM_FIRST") == "1":
        # Volunteer to be the OOM killer's first pick inside this cgroup. The
        # parent does no data work, so without this the kernel may still choose
        # it on score ties and the run is lost instead of recording the refusal.
        try:
            with open("/proc/self/oom_score_adj", "w") as _sa_f:
                _sa_f.write("1000")
        except OSError as _sa_e:
            print("subproc_arm: oom_score_adj unavailable: %s" % _sa_e,
                  file=_sa_sys.stderr)
    _sa_lim = _sa_os.environ.get("SCX_BENCH_ARM_MEM_LIMIT_BYTES")
    if _sa_lim:
        # RLIMIT_DATA, never RLIMIT_AS. An SCX open mmaps the whole file, so an
        # address-space cap refuses the mmap of a file far larger than the
        # arm's actual residency — it would fail the streaming arm it exists to
        # demonstrate. RLIMIT_DATA bounds the heap and private anonymous maps,
        # which is where a materialising arm's memory actually goes.
        _sa_n = int(_sa_lim)
        try:
            _sa_resource.setrlimit(_sa_resource.RLIMIT_DATA, (_sa_n, _sa_n))
        except (OSError, ValueError) as _sa_e:
            print("subproc_arm: RLIMIT_DATA not applied: %s" % _sa_e,
                  file=_sa_sys.stderr)
    del _sa_os, _sa_resource, _sa_sys
    # --- end prelude -----------------------------------------------------
''')


@dataclass
class ArmOutcome:
    """What a worker process reported, and how it ended."""

    records: list[dict[str, Any]] = field(default_factory=list)
    """Every JSON object the worker printed on stdout, in order.

    A worker that emits one line per stage and is then killed leaves the stages
    it finished. Lines that are not JSON (library warnings, deprecation notices)
    are skipped here and preserved verbatim in :attr:`stdout`.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    wall_s: float = 0.0
    timed_out: bool = False

    @property
    def killed_by_oom(self) -> bool:
        """True when the kernel SIGKILLed the worker.

        In a SLURM job this means the cgroup's ``--mem`` was exhausted, which is
        the result `pipeline_ooc_constrained`'s scanpy arms exist to record. It
        is not proof of an OOM in the strict sense — any external SIGKILL reads
        the same — but nothing else in this harness sends one.
        """
        return self.returncode in OOM_RETURNCODES

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def failure_text(self, label: str) -> str:
        """A diagnostic that includes both streams, for raising on a real error."""
        how = (
            f"timed out after {self.wall_s:.0f}s"
            if self.timed_out
            else f"exit={self.returncode}"
        )
        return (
            f"{label} worker failed ({how}); "
            f"{len(self.records)} record(s) recovered.\n"
            f"--- stderr ---\n{self.stderr}\n"
            f"--- stdout ---\n{self.stdout}"
        )


def _collect_records(stdout: str) -> list[dict[str, Any]]:
    """Every JSON object on its own line, in order.

    A worker may print a bare object per stage or a list at the end; both are
    flattened into one sequence so the caller does not care which it chose.
    """
    out: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line[0] not in "[{":
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
        elif isinstance(parsed, list):
            out.extend(p for p in parsed if isinstance(p, dict))
    return out


def run_arm(
    worker_src: str,
    argv: Sequence[str] = (),
    *,
    timeout_s: int,
    env: dict[str, str] | None = None,
    mem_limit_bytes: int | None = None,
    oom_first: bool = False,
    label: str = "arm",
) -> ArmOutcome:
    """Run *worker_src* as a fresh Python process and collect what it printed.

    *worker_src* is prefixed with :data:`WORKER_PRELUDE` and passed to
    ``python -c``; nothing is written to disk. Keep it thin and have it import
    the benchmark module's own helpers, so the measured code stays importable
    and unit-testable in-process.

    *env* entries are overlaid on the parent's environment. *mem_limit_bytes*
    and *oom_first* are passed through to the prelude. Never raises for a
    non-zero exit — inspect :attr:`ArmOutcome.ok` / :attr:`killed_by_oom` and
    call :meth:`ArmOutcome.failure_text` to raise with both streams attached.
    """
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    if oom_first:
        child_env[_ENV_OOM_FIRST] = "1"
    if mem_limit_bytes is not None:
        child_env[_ENV_MEM_LIMIT] = str(int(mem_limit_bytes))

    cmd = [sys.executable, "-c", WORKER_PRELUDE + worker_src, *(str(a) for a in argv)]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=child_env, timeout=timeout_s,
        )
        wall = time.perf_counter() - t0
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        # A timeout still carries whatever the worker managed to flush, which
        # for a staged worker is every stage it completed before hanging.
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        rc, timed_out = -1, True

    outcome = ArmOutcome(
        records=_collect_records(stdout),
        returncode=rc,
        stdout=stdout,
        stderr=stderr,
        wall_s=wall,
        timed_out=timed_out,
    )
    if not outcome.ok:
        logger.warning(
            "%s worker ended abnormally (exit=%s, timed_out=%s) after %.1fs "
            "with %d record(s)",
            label, rc, timed_out, wall, len(outcome.records),
        )
    return outcome
