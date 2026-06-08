#!/usr/bin/env python3
"""Ad-hoc regression gate (local controller, SLURM workers).

This is a **job submitter**, not a benchmark runner. The controller
process (which can run on any CPU-only host: a login node, ``sh_dev``,
a laptop with cluster access — wherever you can reach ``sbatch``)
submits SLURM jobs that execute the benchmarks on real compute nodes.
**You do not need a GPU on the host running this script** — GPU
benchmarks request GPUs from SLURM and run on H100 worker nodes via
``sbatch``. This is the standard Chimera workflow: orchestrate from
``sh_dev``, benchmarks execute under ``sbatch`` on GPU partitions.

Captures a **candidate snapshot** under ``benchmarks/comprehensive/results/<name>/``
via ``capture_baseline.py`` (default name ``candidate_<sha>_<YYYYMMDD>``),
then runs ``compare_against_baseline.py --gate`` to diff that snapshot
against the canonical baseline at ``results/baselines/LATEST``.

Despite its name, ``capture_baseline.py`` does **not** create or update a
baseline — it just writes a snapshot directory. ``results/baselines/LATEST``
is populated only by the separate, manual ``promote_baseline.py`` step on
explicitly chosen reference runs. ``gate_candidate.py`` is read-only with
respect to ``results/baselines/`` and never promotes a snapshot.

Replaces the prior ``gate_candidate.sh`` with structured logging,
pre-flight checks, and a coverage banner so operators can audit which
benchmark axes (format, accel CPU, accel GPU) actually ran.

GPU pre-flight runs on a SLURM compute node when SLURM is detected (the
controller submits a small ``gpu_probe`` job that validates nvidia-smi /
cupy / pyscx.accel on a real GPU instead of the submission host) — so
GPU pre-flight passes from any submission host, GPU or not. Pass
``--probe-partition -`` to force local GPU checks (only use this when
the submission host *is* a GPU node), or ``--no-gpu`` to skip GPU
coverage entirely.

Usage::

    # Default: small tier, full coverage (format + accel CPU + accel GPU)
    python benchmarks/comprehensive/scripts/gate_candidate.py

    # CPU-only on a GPU-equipped host
    python benchmarks/comprehensive/scripts/gate_candidate.py --no-gpu

    # Accel only — fast iteration on accel kernels
    python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only

    # Format only (today's pre-rewrite default behavior)
    python benchmarks/comprehensive/scripts/gate_candidate.py --no-accel

    # Larger tier
    python benchmarks/comprehensive/scripts/gate_candidate.py --tier full

Exit codes (preserved from the bash predecessor)::

    0   gate passed
    1   regression / floor violation / fingerprint drift
    2   missing inputs / pre-flight failure
    130 interrupted (SIGINT)

A timestamped log under ``benchmarks/comprehensive/logs/`` captures the
full subprocess stdout/stderr; a sidecar ``*.summary.json`` records the
phase / exit / elapsed / coverage plan / probe outcome for downstream
tooling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPREHENSIVE = PROJECT_ROOT / "benchmarks" / "comprehensive"
SCRIPTS = COMPREHENSIVE / "scripts"
RESULTS = COMPREHENSIVE / "results"
LATEST = RESULTS / "baselines" / "LATEST"
LOGS = COMPREHENSIVE / "logs"
SUBMITIT_ROOT = LOGS / "submitit" / "gate_probe"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class _ColorFormatter(logging.Formatter):
    """Console formatter with ANSI level coloring (no third-party deps)."""

    _COLORS = {
        "DEBUG": "\x1b[90m",      # gray
        "INFO": "\x1b[36m",       # cyan
        "WARNING": "\x1b[33m",    # yellow
        "ERROR": "\x1b[31m",      # red
        "CRITICAL": "\x1b[1;31m", # bold red
    }
    _RESET = "\x1b[0m"

    def __init__(self, *, color: bool):
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)-7s] [%(phase)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "phase"):
            record.phase = "main"
        line = super().format(record)
        if self._color:
            c = self._COLORS.get(record.levelname, "")
            return f"{c}{line}{self._RESET}" if c else line
        return line


def setup_logging(verbose: int, log_name: str) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_file = LOGS / f"{log_name}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console_level = logging.DEBUG if verbose >= 1 else logging.INFO
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(console_level)
    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    console.setFormatter(_ColorFormatter(color=use_color))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_ColorFormatter(color=False))
    root.addHandler(file_handler)
    return log_file


def phase_logger(phase: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger("gate"), {"phase": phase})


# ---------------------------------------------------------------------------
# Errors + check results
# ---------------------------------------------------------------------------


class GateError(Exception):
    """Raised on a step failure that should produce a structured exit summary."""

    def __init__(self, phase: str, exit_code: int, message: str):
        self.phase = phase
        self.exit_code = exit_code
        self.message = message
        super().__init__(f"[{phase}] {message}")


@dataclasses.dataclass
class CheckResult:
    name: str
    level: str  # "ok" | "warn" | "fail"
    msg: str


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_silent(cmd: list[str], cwd: Path | None = None, timeout: float = 10.0) -> tuple[int, str]:
    """Run a short command, return (rc, combined-stdout-stderr). -1 on missing binary."""
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def run_subprocess(cmd: list[str], phase: str) -> int:
    """Stream a long-running subprocess into the logger and return its exit code.

    stderr is merged into stdout so the file log preserves ordering.
    """
    log = phase_logger(phase)
    log.info("$ %s", " ".join(cmd))
    start = time.monotonic()

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log.info(line.rstrip())
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=10)
        raise

    elapsed = time.monotonic() - start
    log.info("[exit=%d, elapsed=%.1fs]", rc, elapsed)
    return rc


# ---------------------------------------------------------------------------
# GPU probe (runs on a SLURM compute node)
# ---------------------------------------------------------------------------


def gpu_probe() -> dict:
    """Pre-flight probe shipped to a SLURM GPU node via submitit.

    MUST stay self-contained — submitit pickles the function's bytecode
    and ships it to the worker, so any module-level imports from the
    controller (logging adapters, argparse, etc.) would not be available.
    Returns a JSON-serialisable dict the controller translates into
    `CheckResult`s.
    """
    import socket
    import subprocess as _sp
    import sys as _sys

    out: dict = {"hostname": socket.gethostname(), "python": _sys.version.split()[0]}

    # nvidia-smi -L
    try:
        rc = _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        n_gpus = sum(1 for ln in rc.stdout.splitlines() if ln.strip().startswith("GPU "))
        out["nvidia_smi"] = {
            "rc": rc.returncode,
            "stdout": rc.stdout,
            "stderr": rc.stderr,
            "n_gpus": n_gpus,
        }
    except Exception as e:  # pragma: no cover — defensive
        out["nvidia_smi"] = {"rc": -1, "stdout": "", "stderr": f"{type(e).__name__}: {e}", "n_gpus": 0}

    # cupy import + tiny CUDA round-trip — confirms the runtime is live
    try:
        import cupy  # type: ignore[import-not-found]
        s = float(cupy.array([1.0, 2.0, 3.0], dtype=cupy.float32).sum().get())
        out["cupy"] = {"ok": True, "version": cupy.__version__, "round_trip_sum": s}
    except Exception as e:
        out["cupy"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # pyscx.accel import + gpu_info() — confirms the GPU build was loaded
    try:
        from pyscx import accel  # type: ignore[import-not-found]
        info = accel.gpu_info()
        out["pyscx_accel"] = {
            "ok": True,
            "has_all": hasattr(accel, "__all__"),
            "gpu_info": info,
        }
    except Exception as e:
        out["pyscx_accel"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return out


def _probe_setup_cmds(conda_env: str) -> list[str]:
    """Shell setup for the probe's SLURM job.

    Activates the user-supplied conda env (default: `scx-gpu`) so the
    probe can find `cupy` and the GPU-built `pyscx` wheel. Mirrors the
    `unset SLURM_*` + `SLURM_MPI_TYPE=none` housekeeping that
    ``run_parallel.py::_slurm_setup_cmds`` performs.
    """
    conda_base = os.environ.get("CONDA_EXE", "").replace("/bin/conda", "")
    if not conda_base:
        conda_base = str(Path.home() / "miniforge3")
    cmds = [
        "unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true",
        "export SLURM_MPI_TYPE=none",
    ]
    if conda_env:
        cmds.extend([
            f'eval "$({conda_base}/bin/conda shell.bash hook)"',
            f"conda activate {conda_env}",
        ])
    return cmds


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _check_repo_root() -> CheckResult:
    rc, _ = _run_silent(["git", "rev-parse", "--git-dir"], cwd=PROJECT_ROOT)
    if rc != 0:
        return CheckResult(
            "git_checkout", "fail",
            f"{PROJECT_ROOT} is not a git checkout — gate must run from the scx repo",
        )
    return CheckResult("git_checkout", "ok", str(PROJECT_ROOT))


def _check_dirty_tree() -> CheckResult:
    rc, out = _run_silent(["git", "status", "--porcelain"], cwd=PROJECT_ROOT)
    if rc != 0:
        return CheckResult("dirty_tree", "warn", "(git status unreachable)")
    if out.strip():
        n = len(out.strip().splitlines())
        return CheckResult(
            "dirty_tree", "warn",
            f"{n} dirty path(s) — gate result not reproducible from this git_sha",
        )
    return CheckResult("dirty_tree", "ok", "clean")


def _check_helpers() -> CheckResult:
    needed = [
        SCRIPTS / "capture_baseline.py",
        SCRIPTS / "compare_against_baseline.py",
    ]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        return CheckResult(
            "helpers", "fail",
            "missing helper(s): " + ", ".join(str(p.name) for p in missing),
        )
    return CheckResult("helpers", "ok", f"{len(needed)} helpers present")


def _check_baseline(explicit_baseline: str | None) -> CheckResult:
    if explicit_baseline:
        p = Path(explicit_baseline)
        if not p.exists():
            return CheckResult("baseline", "fail", f"--baseline {p} does not exist")
        return CheckResult("baseline", "ok", str(p))
    if not LATEST.exists() and not LATEST.is_symlink():
        return CheckResult(
            "baseline", "fail",
            f"no baseline at {LATEST.relative_to(PROJECT_ROOT)} — run promote_baseline.py first",
        )
    target = LATEST.resolve() if LATEST.is_symlink() else LATEST
    return CheckResult("baseline", "ok", f"LATEST → {target.name}")


def _check_python_interp(python: str) -> CheckResult:
    if not python:
        return CheckResult("python_interp", "fail", "--python is empty")
    resolved = shutil.which(python) or python
    if not Path(resolved).is_file() or not os.access(resolved, os.X_OK):
        return CheckResult("python_interp", "fail", f"{python!r} not executable")
    rc, out = _run_silent([python, "-c", "import sys; print(sys.version)"])
    if rc != 0:
        return CheckResult("python_interp", "fail", f"{python!r} failed to start: {out.strip()[:120]}")
    return CheckResult("python_interp", "ok", out.strip().split()[0])


def _check_nvidia_smi() -> CheckResult:
    rc, out = _run_silent(["nvidia-smi", "-L"])
    if rc != 0:
        return CheckResult(
            "nvidia_smi", "fail",
            "nvidia-smi unavailable but --no-gpu was not set — pass --no-gpu to skip GPU coverage",
        )
    n = sum(1 for ln in out.splitlines() if ln.strip().startswith("GPU "))
    if n == 0:
        return CheckResult(
            "nvidia_smi", "fail",
            "nvidia-smi -L returned no GPUs — pass --no-gpu to skip GPU coverage",
        )
    return CheckResult("nvidia_smi", "ok", f"{n} GPU(s) visible")


def _check_cupy(python: str) -> CheckResult:
    rc, out = _run_silent(
        [python, "-c", "import cupy; print(cupy.__version__)"], timeout=20,
    )
    if rc != 0:
        return CheckResult(
            "cupy", "fail",
            f"{python!r} cannot import cupy — activate scx-bench-gpu or pass --no-gpu",
        )
    return CheckResult("cupy", "ok", f"cupy {out.strip()}")


def _check_pyscx_accel(python: str) -> CheckResult:
    # pyscx exposes `accel` as a runtime attribute set in __init__.py, not a
    # dotted-path submodule — `import pyscx.accel` would always fail.
    rc, out = _run_silent(
        [python, "-c", "from pyscx import accel; assert hasattr(accel, '__all__')"],
        timeout=20,
    )
    if rc != 0:
        msg = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        return CheckResult(
            "pyscx_accel", "fail",
            f"pyscx.accel not importable in {python!r} — run `maturin develop --features gpu` first ({msg[:120]})",
        )
    return CheckResult("pyscx_accel", "ok", "import ok")


def _check_disk_space() -> CheckResult:
    RESULTS.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(RESULTS).free
    free_gib = free / (1024**3)
    if free_gib < 1.0:
        return CheckResult(
            "disk_space", "warn",
            f"only {free_gib:.2f} GiB free at {RESULTS} — capture may fail mid-run",
        )
    return CheckResult("disk_space", "ok", f"{free_gib:.1f} GiB free")


def _check_slurm() -> CheckResult:
    """Informational: submitit.AutoExecutor falls back to local execution
    when sbatch is missing, so absence is fine — we just want the operator
    to know what mode they're in."""
    rc, _ = _run_silent(["sbatch", "--version"])
    if rc != 0:
        return CheckResult(
            "slurm", "warn",
            "sbatch not found — submitit will run jobs locally on this host",
        )
    return CheckResult("slurm", "ok", "SLURM detected")


def _check_cloud_probe(python: str, bucket: str | None = None) -> list[CheckResult]:
    """Probe cloud connectivity before submitting jobs.

    Catches two recurring pre-run failure modes that otherwise surface
    only after hundreds of FAST_FAILs land in the gate report:

      1. **gcsfs not registered with fsspec** — the env has fsspec but
         not gcsfs (or version-mismatched), so the `gs://` protocol fails
         at first use. Surfaces as `Please install gcsfs` from each cloud
         cell. Tier-full gate ran into this on 2026-05-10 worker env →
         24 zarr cloud FAST_FAILs.
      2. **catalog symmetry bug class** — a known-bad cloud `.scxd`
         catalog round-trips at push but fails at pull with
         `failed to fill whole buffer`. Surfaces as 158 cloud FAST_FAILs
         on the second 2026-05-10 gate run.

    Each probe runs in < 30 s; the combined cost is far cheaper than a
    2 h gate that fails late. Returns one CheckResult per probe.
    Skipped silently when ``--probe-cloud`` is not passed.
    """
    # Imported here so the controller doesn't carry these symbols
    # unconditionally; the bucket default lives in config.py.
    if bucket is None:
        rc, out = _run_silent(
            [python, "-c", "from benchmarks.comprehensive.config import GCS_TEST_BUCKET; print(GCS_TEST_BUCKET)"],
            timeout=10,
        )
        if rc != 0:
            return [CheckResult(
                "cloud_probe", "fail",
                f"could not resolve GCS_TEST_BUCKET via {python!r}: {out.strip()[:120]}",
            )]
        bucket = out.strip().splitlines()[-1] if out.strip() else "gs://arc-ctc-nextflow/scx-test"

    checks: list[CheckResult] = []

    # 1. fsspec gs protocol resolution + bucket reachability.
    # `ls` confirms credentials too — IMDS / GOOGLE_APPLICATION_CREDENTIALS
    # must be wired before this returns.
    rc, out = _run_silent(
        [python, "-c", (
            "import fsspec; "
            f"fs = fsspec.filesystem('gs'); "
            f"list(fs.ls({bucket!r}))[:1]"
        )],
        timeout=25,
    )
    if rc != 0:
        msg = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        checks.append(CheckResult(
            "cloud_gcsfs", "fail",
            f"fsspec gs filesystem failed on {bucket} — gcsfs missing/mismatched "
            f"or GCP credentials not configured: {msg[:160]}",
        ))
        # If gcsfs is broken, the open_cloud probe will likely fail for
        # the same reason — skip it to keep the operator's attention on
        # the root cause.
        return checks
    checks.append(CheckResult("cloud_gcsfs", "ok", f"fsspec ls {bucket} ok"))

    # 2. pyscx.open_cloud round-trip on a known small fixture (catches
    # catalog-symmetry / cloud-layout bugs class). pbmc3k is the
    # cheapest possible probe (1 GET on _catalog.bin + a metadata
    # touch); skip with WARN rather than FAIL if it isn't staged.
    probe_url = f"{bucket.rstrip('/')}/pbmc3k.scxd"
    rc, out = _run_silent(
        [python, "-c", (
            f"import pyscx; "
            f"h = pyscx.open_cloud({probe_url!r}); "
            f"_ = (h.n_obs, h.n_vars, h.nnz)"
        )],
        timeout=30,
    )
    if rc != 0:
        msg = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        # Heuristic: distinguish "fixture not staged" (warn — operator
        # forgot setup_cloud_test_data.sh) from "open failed despite
        # fixture existing" (fail — actual catalog/format problem).
        not_staged = (
            "no such" in msg.lower()
            or "not found" in msg.lower()
            or "404" in msg
        )
        if not_staged:
            checks.append(CheckResult(
                "cloud_open", "warn",
                f"probe fixture {probe_url} not staged — open_cloud "
                f"symmetry check skipped. Run setup_cloud_test_data.sh "
                f"to enable.",
            ))
        else:
            checks.append(CheckResult(
                "cloud_open", "fail",
                f"pyscx.open_cloud failed on {probe_url} — likely "
                f"catalog-format bug; re-push fixtures with current pyscx: "
                f"{msg[:160]}",
            ))
        return checks
    checks.append(CheckResult(
        "cloud_open", "ok", f"pyscx.open_cloud {probe_url} ok",
    ))
    return checks


def run_preflight(args: argparse.Namespace) -> tuple[list[CheckResult], dict]:
    """Run pre-flight checks and return ``(checks, probe_info)``.

    ``probe_info`` records the SLURM probe outcome (or ``"skipped"``) so
    the summary JSON can audit which path the gate took.
    """
    log = phase_logger("pre-flight")
    checks: list[CheckResult] = [
        _check_repo_root(),
        _check_dirty_tree(),
        _check_helpers(),
        _check_baseline(args.baseline),
        _check_python_interp(args.python),
        _check_disk_space(),
    ]
    slurm_check = _check_slurm()
    checks.append(slurm_check)

    probe_info: dict = {
        "submitted": False,
        "job_id": None,
        "partition": None,
        "elapsed_s": None,
        "outcome": "skipped",
    }
    slurm_available = slurm_check.level == "ok"
    want_cluster_probe = (
        slurm_available
        and not args.no_gpu
        and args.probe_partition != "-"
    )

    # Graceful skip: if the requested probe conda env doesn't exist on
    # the local filesystem (Weka is shared on Chimera, so the same path
    # the SLURM worker would source is checkable here), don't submit the
    # probe — it would otherwise fail with a cupy ImportError on the
    # worker and surface as a confusing pre-flight FAIL. Operators get a
    # one-line warning + clear remediation instead.
    #
    # When we skip the cluster probe because the env is missing, we
    # ALSO disable the local-host GPU fallback (nvidia-smi / cupy on
    # the submission host) — there's no point checking for a GPU on a
    # CPU-only orchestrator host (login node / sh_dev) when we already
    # know we can't run GPU benchmarks anyway. Treat env-missing as a
    # soft `--no-gpu` for the rest of the pre-flight: log the skip and
    # propagate it through `args.no_gpu` so downstream coverage banners
    # / capture invocation see the same disabled state.
    skip_gpu_due_to_missing_env = False
    if want_cluster_probe and args.probe_conda_env:
        env_path = Path.home() / "miniforge3" / "envs" / args.probe_conda_env
        if not env_path.is_dir():
            log.warning(
                "  gpu_probe : SKIP — conda env %r not found at %s; pass "
                "--no-gpu explicitly to silence this warning, or "
                "`conda env create -f benchmarks/comprehensive/envs/scx-bench-gpu.yml` "
                "to enable GPU coverage. Continuing with GPU coverage "
                "disabled (CPU-only mode).",
                args.probe_conda_env, env_path,
            )
            want_cluster_probe = False
            skip_gpu_due_to_missing_env = True
            args.no_gpu = True  # propagate to capture / banner / coverage
            probe_info["outcome"] = "skipped_env_missing"

    if want_cluster_probe:
        gpu_checks, probe_info = _run_cluster_gpu_probe(args)
        checks.extend(gpu_checks)
    elif not args.no_gpu and not skip_gpu_due_to_missing_env:
        # Local fallback: today's checks, run on the submission host.
        nvidia = _check_nvidia_smi()
        checks.append(nvidia)
        if nvidia.level == "ok":
            checks.append(_check_cupy(args.python))

    # pyscx.accel import is a submission-host concern *only* when there's
    # no cluster probe (the probe already exercises pyscx.accel on the
    # compute node, where the GPU-built wheel actually lives).
    if not args.no_accel and not want_cluster_probe:
        checks.append(_check_pyscx_accel(args.python))

    # Cloud probe — opt-in. When the gate's matrix includes cloud cells,
    # this catches the two recurring pre-run failure modes (missing
    # gcsfs registration, catalog symmetry bug) in < 30 s instead of
    # after 158 FAST_FAILs land in the gate report. Disabled by default
    # because not every gate run exercises cloud (--accel-only,
    # --no-accel, narrow --formats lists).
    if args.probe_cloud:
        checks.extend(_check_cloud_probe(args.python))

    width = max(len(c.name) for c in checks)
    marker = {"ok": "OK   ", "warn": "WARN ", "fail": "FAIL "}
    level_fn = {"ok": log.info, "warn": log.warning, "fail": log.error}
    for c in checks:
        level_fn[c.level]("  %-*s : %s — %s", width, c.name, marker[c.level], c.msg)
    return checks, probe_info


def _run_cluster_gpu_probe(args: argparse.Namespace) -> tuple[list[CheckResult], dict]:
    """Submit ``gpu_probe`` to SLURM and translate its result.

    Returns ``(checks, probe_info)``. On submission failure / timeout the
    probe surfaces as a single ``FAIL`` ``CheckResult`` so the standard
    pre-flight failure path applies.
    """
    log = phase_logger("pre-flight")
    info: dict = {
        "submitted": False,
        "job_id": None,
        "partition": args.probe_partition,
        "elapsed_s": None,
        "outcome": "skipped",
    }

    try:
        import submitit  # type: ignore[import-not-found]
    except ImportError as e:
        msg = f"submitit unavailable ({e}); pass --probe-partition - to fall back to local checks"
        info["outcome"] = "fail"
        return [CheckResult("gpu_probe", "fail", msg)], info

    SUBMITIT_ROOT.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(SUBMITIT_ROOT))
    executor.update_parameters(
        slurm_partition=args.probe_partition,
        slurm_gres="gpu:1",
        cpus_per_task=args.probe_cpus,
        mem_gb=args.probe_mem_gb,
        timeout_min=args.probe_time_min,
        slurm_setup=_probe_setup_cmds(args.probe_conda_env),
    )

    log.info(
        "submitting gpu_probe (partition=%s gres=gpu:1 mem=%dG cpus=%d time=%dm conda=%s)",
        args.probe_partition, args.probe_mem_gb, args.probe_cpus,
        args.probe_time_min, args.probe_conda_env or "(none)",
    )
    start = time.monotonic()
    try:
        job = executor.submit(gpu_probe)
    except Exception as e:  # submission-side failure (sbatch, partition, …)
        info["outcome"] = "fail"
        return [CheckResult("gpu_probe", "fail", f"submission failed: {e}")], info

    info["submitted"] = True
    info["job_id"] = job.job_id
    log.info("gpu_probe submitted: job_id=%s — waiting up to %ds", job.job_id, args.probe_timeout)

    # submitit 1.5.x's Job.result() blocks indefinitely; poll done() with a
    # wall-clock budget instead so a stuck queue can't hang the gate.
    deadline = time.monotonic() + args.probe_timeout
    poll_interval = 5.0
    last_state: str | None = None
    while not job.done():
        if time.monotonic() >= deadline:
            info["elapsed_s"] = round(time.monotonic() - start, 2)
            info["outcome"] = "timeout"
            try:
                state = job.state  # may issue a squeue lookup
            except Exception:
                state = "?"
            tail = _tail_submitit_err(job)
            msg = (
                f"gpu_probe job_id={job.job_id} state={state} did not complete "
                f"within {args.probe_timeout}s — cancel manually with "
                f"`scancel {job.job_id}` if still queued"
            )
            if tail:
                msg = f"{msg}\n  err tail: {tail}"
            return [CheckResult("gpu_probe", "fail", msg)], info
        try:
            current_state = job.state
        except Exception:
            current_state = None
        if current_state and current_state != last_state:
            log.info("gpu_probe state: %s", current_state)
            last_state = current_state
        time.sleep(poll_interval)

    info["elapsed_s"] = round(time.monotonic() - start, 2)
    try:
        result = job.result()
    except Exception as e:
        info["outcome"] = "fail"
        tail = _tail_submitit_err(job)
        msg = f"gpu_probe job_id={job.job_id} returned an error: {e}"
        if tail:
            msg = f"{msg}\n  err tail: {tail}"
        return [CheckResult("gpu_probe", "fail", msg)], info

    info["outcome"] = "ok"
    log.info("gpu_probe completed in %.1fs on host=%s",
             info["elapsed_s"], result.get("hostname", "?"))
    return _translate_probe_result(result), info


def _tail_submitit_err(job, n_lines: int = 20) -> str:
    """Best-effort tail of submitit's stderr for diagnostics."""
    err_path = Path(job.paths.folder) / f"{job.job_id}_log.err"
    if not err_path.is_file():
        return ""
    try:
        lines = err_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n  " + "\n  ".join(lines[-n_lines:])


def _translate_probe_result(r: dict) -> list[CheckResult]:
    """Render the dict returned by ``gpu_probe`` as 3 ``CheckResult``s."""
    out: list[CheckResult] = []

    n = r.get("nvidia_smi") or {}
    rc = n.get("rc", -1)
    n_gpus = n.get("n_gpus", 0)
    if rc != 0 or n_gpus == 0:
        out.append(CheckResult(
            "nvidia_smi", "fail",
            f"compute node returned no GPUs (rc={rc}, n_gpus={n_gpus})",
        ))
    else:
        out.append(CheckResult(
            "nvidia_smi", "ok",
            f"{n_gpus} GPU(s) on probe node {r.get('hostname', '?')}",
        ))

    c = r.get("cupy") or {}
    if c.get("ok"):
        out.append(CheckResult(
            "cupy", "ok",
            f"cupy {c.get('version', '?')} round-trip ok (sum={c.get('round_trip_sum')!s})",
        ))
    else:
        out.append(CheckResult(
            "cupy", "fail",
            f"probe import failed: {c.get('error', '(no detail)')}",
        ))

    p = r.get("pyscx_accel") or {}
    if p.get("ok"):
        info = p.get("gpu_info")
        # A falsy `gpu_info` (None / empty dict) means pyscx imported but
        # was built without `--features gpu` — accel.gpu_info() returns
        # None when the GPU build path is compiled out. Every downstream
        # GPU bench silently skips with `_HAS_PYSCX_GPU = False`, which
        # used to slip past the gate. Treat as a hard fail here.
        if not info:
            out.append(CheckResult(
                "pyscx_accel", "fail",
                f"pyscx imports but gpu_info() returned {info!r} — likely "
                f"built without --features gpu. Rebuild: cd pyscx && "
                f".venv/bin/maturin develop --release --features gpu",
            ))
        else:
            info_s = f" gpu_info={info!s:.80}"
            out.append(CheckResult("pyscx_accel", "ok", f"import + gpu_info ok{info_s}"))
    else:
        out.append(CheckResult(
            "pyscx_accel", "fail",
            f"probe import failed: {p.get('error', '(no detail)')}",
        ))
    return out


# ---------------------------------------------------------------------------
# Defensive worker-pyscx-GPU sanity check
# ---------------------------------------------------------------------------


def _check_worker_gpu_pyscx(
    env_name: str | None, preflight_ran: bool = True
) -> CheckResult:
    """Verify the worker GPU conda env has a pyscx built with ``--features gpu``.

    Background: pyscx's editable .so is shared across conda envs via the
    maturin install path. A rebuild from a non-GPU shell silently
    overwrites the shared .so with a no-GPU build. Every downstream GPU
    benchmark then skips with ``_HAS_PYSCX_GPU = False`` and the gate
    completes "successfully" with zero GPU rows — which we only noticed
    after a 6h full-tier bench in this session.

    This check spawns a ~1s subprocess in the worker GPU env, imports
    pyscx, and inspects ``accel.gpu_info()``. Runs unconditionally when GPU
    is in scope, even when ``--skip-preflight`` is passed (the SLURM
    gpu_probe is the heavier preflight piece — this is the cheap local one
    we always want).

    ``IMPORT_FAIL`` / ``CALL_FAIL`` (a genuinely broken gpu build) are
    always fatal. A falsy ``accel.gpu_info()`` (``GPU_INFO_FALSY``) is
    ambiguous — it means "no GPU on *this* host **or** built without
    ``--features gpu``". When ``preflight_ran`` is True the SLURM gpu_probe
    already validated the build on a real GPU node, so this is just a
    CPU-hosted orchestrator with no local GPU → WARN. Under
    ``--skip-preflight`` (``preflight_ran=False``) no authoritative probe
    ran, so a falsy result must stay FAIL rather than silently let every
    GPU benchmark skip.

    Returns ``CheckResult(level="ok" | "fail" | "warn")``.
    """
    if not env_name:
        return CheckResult(
            "worker_gpu_pyscx", "warn",
            "no GPU worker env resolved — sanity check skipped",
        )

    conda_base = os.environ.get("CONDA_EXE", "").replace("/bin/conda", "")
    if not conda_base:
        conda_base = str(Path.home() / "miniforge3")
    conda_hook = f"{conda_base}/bin/conda"
    if not Path(conda_hook).is_file():
        return CheckResult(
            "worker_gpu_pyscx", "warn",
            f"conda not found at {conda_hook} — sanity check skipped",
        )

    probe = (
        "import sys\n"
        "try:\n"
        "    from pyscx import accel\n"
        "    info = accel.gpu_info()\n"
        "except ImportError as e:\n"
        "    print(f'IMPORT_FAIL: {type(e).__name__}: {e}')\n"
        "    sys.exit(2)\n"
        "except Exception as e:\n"
        "    print(f'CALL_FAIL: {type(e).__name__}: {e}')\n"
        "    sys.exit(2)\n"
        "if not info:\n"
        "    print(f'GPU_INFO_FALSY: {info!r}')\n"
        "    sys.exit(2)\n"
        "print(f'OK: {info!r}')\n"
    )
    cmd = [
        "bash", "-lc",
        f'eval "$({conda_hook} shell.bash hook)" && '
        f"conda activate {env_name} && python -c {shlex.quote(probe)}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return CheckResult(
            "worker_gpu_pyscx", "fail",
            f"subprocess timed out (60s) in env={env_name}",
        )
    # Scan *all* lines for the probe's verdict prefix, not just out[-1]: stderr
    # is concatenated after stdout, so a trailing conda/deprecation warning line
    # would otherwise mask the real result (and re-introduce the false-fail this
    # check is meant to avoid).
    out = (proc.stdout + proc.stderr).strip().splitlines()

    def _find(prefix: str) -> str | None:
        return next((ln for ln in out if ln.startswith(prefix)), None)

    ok_line = _find("OK:")
    if proc.returncode == 0 and ok_line:
        return CheckResult("worker_gpu_pyscx", "ok", f"env={env_name}: {ok_line}")

    # `GPU_INFO_FALSY` means pyscx imported cleanly but accel.gpu_info() was
    # falsy — i.e. either there is no GPU on *this* (orchestrator) host, or the
    # build is missing --features gpu. That is NOT a broken gpu build the way
    # IMPORT_FAIL / CALL_FAIL are. When preflight ran, the SLURM gpu_probe
    # already validated pyscx-gpu on a real GPU node, so this is just a
    # CPU-hosted orchestrator → WARN. Under --skip-preflight no authoritative
    # probe ran, so a falsy result stays FAIL rather than silently skipping
    # every GPU benchmark.
    falsy_line = _find("GPU_INFO_FALSY")
    if falsy_line:
        if preflight_ran:
            return CheckResult(
                "worker_gpu_pyscx", "warn",
                f"env={env_name}: no GPU on orchestrator host ({falsy_line}); "
                "the pre-flight SLURM gpu_probe is the authoritative gpu-build check",
            )
        return CheckResult(
            "worker_gpu_pyscx", "fail",
            f"env={env_name}: {falsy_line} under --skip-preflight; with no SLURM "
            "gpu_probe this is indistinguishable from a build missing --features "
            "gpu, which would silently skip every GPU benchmark. Drop "
            "--skip-preflight to validate the build on a GPU node, or rebuild "
            "with: cd pyscx && /path/to/.venv/bin/maturin develop --release "
            "--features gpu",
        )

    rebuild_hint = (
        "rebuild with: cd pyscx && "
        "/path/to/.venv/bin/maturin develop --release --features gpu"
    )
    err_line = _find("IMPORT_FAIL") or _find("CALL_FAIL") or (out[-1] if out else "")
    return CheckResult(
        "worker_gpu_pyscx", "fail",
        f"env={env_name}: {err_line or 'no output'} (rc={proc.returncode}); {rebuild_hint}",
    )


def _resolve_worker_gpu_env() -> str | None:
    """Return the conda env name workers use for GPU benchmarks.

    Delegates to ``run_parallel._env_for_format`` so the lookup mirrors the
    actual SLURM submission path one-to-one. Falls back to
    ``"scx-bench-gpu"`` when the import fails (e.g. running outside the
    workspace), which is the historical default.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "benchmarks" / "comprehensive" / "scripts"))
        from run_parallel import _env_for_format  # type: ignore[import-not-found]
        return _env_for_format("accel_de__pyscx_pdex_ref_gpu")
    except Exception:
        return "scx-bench-gpu"


# ---------------------------------------------------------------------------
# Coverage banner
# ---------------------------------------------------------------------------


def coverage_plan(args: argparse.Namespace) -> dict:
    if args.accel_only:
        format_y, accel_cpu = False, True
    elif args.no_accel:
        format_y, accel_cpu = True, False
    else:
        format_y, accel_cpu = True, True
    accel_gpu = (not args.no_gpu) and (not args.no_accel)
    return {"format": format_y, "accel_cpu": accel_cpu, "accel_gpu": accel_gpu}


def print_banner(
    args: argparse.Namespace,
    candidate_dir: Path,
    log_file: Path,
    plan: dict,
) -> None:
    log = phase_logger("plan")
    sep = "─" * 68
    yn = lambda b: "✓" if b else "✗"
    baseline = args.baseline or str(LATEST.relative_to(PROJECT_ROOT))
    log.info(sep)
    log.info("  Gate plan")
    log.info("  ─────────")
    log.info("  tier              : %s", args.tier)
    log.info("  candidate         : %s", candidate_dir.relative_to(PROJECT_ROOT))
    log.info("  baseline          : %s", baseline)
    log.info("  log file          : %s", log_file.relative_to(PROJECT_ROOT))
    log.info("  python            : %s", args.python)
    log.info("  format benchmarks : %s", yn(plan["format"]))
    log.info("  accel CPU         : %s", yn(plan["accel_cpu"]))
    log.info("  accel GPU         : %s", yn(plan["accel_gpu"]))
    log.info(sep)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _capture_argv(args: argparse.Namespace, candidate_dir: Path, *, dry_run: bool) -> list[str]:
    cmd = [
        args.python,
        str(SCRIPTS / "capture_baseline.py"),
        "--tier", args.tier,
        "--name", candidate_dir.name,
    ]
    if dry_run:
        cmd.extend(["--mode", "dry-run"])
    if args.skip_convert:
        cmd.append("--skip-convert")
    if args.no_gpu:
        cmd.append("--no-gpu")
    if args.accel_only:
        cmd.append("--accel-only")
    elif args.no_accel:
        cmd.append("--no-accel")
    else:
        cmd.append("--include-accel")
    if args.benchmarks:
        cmd.extend(["--benchmarks", *args.benchmarks])
    if args.datasets:
        cmd.extend(["--datasets", *args.datasets])
    if args.formats:
        cmd.extend(["--formats", *args.formats])
    if args.skip_smoke:
        cmd.append("--skip-smoke")
    if args.partition is not None:
        cmd.extend(["--partition", args.partition])
    return cmd


def run_capture(args: argparse.Namespace, candidate_dir: Path) -> None:
    rc = run_subprocess(_capture_argv(args, candidate_dir, dry_run=False), "capture")
    if rc != 0:
        raise GateError(
            "capture",
            1 if rc == 1 else 2,
            f"capture_baseline.py exited with {rc}; some results may have been archived",
        )


def run_dry_run(args: argparse.Namespace, candidate_dir: Path) -> int:
    """Show the schedule via capture_baseline.py --mode dry-run, then exit."""
    return run_subprocess(_capture_argv(args, candidate_dir, dry_run=True), "dry-run")


def run_gate(args: argparse.Namespace, candidate_dir: Path) -> int:
    cmd = [
        args.python,
        str(SCRIPTS / "compare_against_baseline.py"),
        "--current", str(candidate_dir),
        "--gate",
    ]
    if args.baseline:
        cmd.extend(["--baseline", args.baseline])
    # When the operator narrowed the capture via --benchmarks, the
    # candidate snapshot only covers that subset. Forward the filter to
    # the comparison so missing rows for unrequested benchmarks aren't
    # flagged as DISAPPEARED regressions against the full-surface
    # baseline.
    if args.benchmarks:
        cmd.extend(["--only-benchmarks", *args.benchmarks])
    cmd.extend(args.extra_gate_args)
    return run_subprocess(cmd, "gate")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def write_summary(
    log_file: Path,
    *,
    phase: str,
    exit_code: int,
    elapsed: float,
    candidate_dir: Path | None,
    plan: dict | None,
    cause: str | None,
    probe: dict | None = None,
) -> None:
    summary = {
        "phase": phase,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 2),
        "candidate": str(candidate_dir) if candidate_dir else None,
        "plan": plan,
        "probe": probe,
        "cause": cause,
        "log_file": str(log_file),
    }
    summary_path = log_file.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log = phase_logger("summary")
    if exit_code == 0:
        log.info("PASS in %.1fs — log: %s", elapsed, log_file)
    else:
        log.error(
            "FAIL exit=%d phase=%s elapsed=%.1fs — %s",
            exit_code, phase, elapsed, cause or "(see log)",
        )
        log.error("log: %s", log_file)
    log.info("summary: %s", summary_path)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Coverage defaults to format + accel CPU + accel GPU. Opt out via "
            "--no-gpu / --no-accel / --accel-only. Pre-flight checks fail fast "
            "with exit 2 on missing inputs (baseline, helpers, GPU env)."
        ),
    )
    parser.add_argument("--tier", choices=["small", "full", "xl"], default="small",
                        help="Dataset tier (default: small).")
    parser.add_argument("--name", default=None,
                        help="Candidate snapshot name (default: candidate_<sha>_<YYYYMMDD>).")
    parser.add_argument("--baseline", default=None,
                        help="Pin a specific baseline dir instead of LATEST.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for sub-steps (default: current).")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Reuse an existing candidate dir; skip capture phase.")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip format conversion (assume converted files exist).")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip pre-flight checks. Use only if you know what you're doing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the schedule (capture_baseline.py --mode dry-run) "
                             "and exit. Skips capture and gate.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase console verbosity (-v: DEBUG).")

    coverage = parser.add_mutually_exclusive_group()
    coverage.add_argument("--no-accel", action="store_true",
                          help="Skip all accel_* benchmarks (format only).")
    coverage.add_argument("--accel-only", action="store_true",
                          help="Run only accel_* benchmarks (skip format).")

    parser.add_argument("--no-gpu", action="store_true",
                        help="Drop GPU accel format variants (accel_*__*_gpu*).")
    parser.add_argument("--probe-cloud", action="store_true",
                        help="Run a cloud connectivity probe in pre-flight: "
                             "fsspec gs ls (catches missing/mismatched gcsfs) "
                             "and pyscx.open_cloud against a small staged "
                             "fixture (catches the catalog-symmetry bug "
                             "class). Adds ~10-30s to pre-flight but surfaces "
                             "two recurring failure modes in seconds rather "
                             "than after the full matrix lands as FAST_FAILs. "
                             "Opt-in because --accel-only / --no-accel / "
                             "narrow --formats runs may not exercise cloud.")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Restrict to a subset of the canonical benchmark "
                             "list (e.g. `--benchmarks accel_pca`). Forwarded "
                             "to capture_baseline.py.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Override the tier's dataset list (forwarded to "
                             "capture_baseline.py). Useful with --benchmarks "
                             "for narrow spot-checks.")
    parser.add_argument("--formats", nargs="+", default=None,
                        help="Restrict to a subset of format keys (forwarded "
                             "to capture_baseline.py → run_parallel.py). "
                             "Default: tier-implicit. Use to scope away "
                             "from format runners whose deps aren't in the "
                             "current conda env (e.g. drop `slaf` when not "
                             "running from `scx-bench-slaf`, drop `bpcells` "
                             "when not running from `scx-bench-r`).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip the pre-submit format-runner contract "
                             "check inside run_parallel.py. Useful for narrow "
                             "accel-only runs or when known-broken format "
                             "runners (BPCells/Parquet) are blocking unrelated "
                             "submissions. Forwarded to capture_baseline.py.")

    parser.add_argument(
        "--partition",
        default=None,
        help=(
            "SLURM partition forwarded to capture_baseline.py for the "
            "per-(benchmark, dataset, format) jobs. Defaults to the tier's "
            "configured partition (Chimera's `cpu_preemptible` historically; "
            "overridable via the `SCX_BENCH_PARTITION` env var). Pass an "
            "explicit value when running on a cluster whose partition names "
            "differ — e.g. `--partition preemptible` on Lambda HPC."
        ),
    )

    # SLURM probe settings — controls the cluster-side GPU pre-flight job.
    probe = parser.add_argument_group(
        "GPU pre-flight probe (SLURM)",
        "When SLURM is detected and GPU coverage is requested, gate_candidate "
        "submits a small probe job to a GPU node to validate nvidia-smi / cupy "
        "/ pyscx.accel there instead of on the (often CPU-only) submission host.",
    )
    probe.add_argument("--probe-partition", default="preemptible",
                       help="SLURM partition for the GPU probe job. "
                            "Use '-' to disable cluster probing and force "
                            "local GPU checks (default: preemptible).")
    probe.add_argument("--probe-conda-env", default="scx-bench-gpu",
                       help="Conda env to activate inside the probe job "
                            "(default: scx-bench-gpu — the canonical GPU "
                            "benchmark env per `benchmarks/README.md`). "
                            "Empty string skips activation. The pre-flight "
                            "auto-skips the cluster probe (printing a clear "
                            "warning) when the env doesn't exist on the "
                            "Weka shared filesystem, so a CPU-only host or "
                            "a host that just doesn't have the GPU env "
                            "doesn't get blocked at pre-flight.")
    probe.add_argument("--probe-cpus", type=int, default=2,
                       help="CPUs requested for the probe (default: 2).")
    probe.add_argument("--probe-mem-gb", type=int, default=8,
                       help="Memory (GiB) requested for the probe (default: 8).")
    probe.add_argument("--probe-time-min", type=int, default=10,
                       help="SLURM --time for the probe (default: 10 min).")
    probe.add_argument("--probe-timeout", type=int, default=600,
                       help="Wall-clock budget covering queue wait + probe "
                            "runtime (default: 600s).")

    parser.add_argument("extra_gate_args", nargs=argparse.REMAINDER,
                        help="Extra args after `--` are forwarded to compare_against_baseline.py.")
    return parser.parse_args()


def _git_sha() -> str:
    rc, out = _run_silent(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT)
    return out.strip() if rc == 0 else "unknown"


def _check_conda_env() -> None:
    """Loudly warn if the script is being launched from outside the
    canonical ``scx-bench`` (or sibling) conda env.

    The comprehensive benchmark suite is designed around isolated
    conda envs (``scx-bench``, ``scx-bench-gpu``, ``scx-bench-r``,
    ``scx-bench-slaf``, see ``benchmarks/comprehensive/envs/``).
    ``run_parallel.py``'s ``_slurm_setup_cmds`` only activates a
    conda env on the SLURM workers when the orchestrator's own
    ``CONDA_PREFIX`` contains ``"scx-bench"`` — otherwise every job
    falls back to the dev ``.venv/`` PATH, which lacks
    ``mudata`` / ``slafdb`` / ``BPCells`` / etc., cascading hundreds
    of convert+bench jobs into ``DependencyNeverSatisfied``. The
    warning is non-fatal so operators can still iterate from
    ``.venv/`` for narrow runs (e.g. accel-only on a CPU laptop)
    but the loud reminder prevents accidental hour-long stalls.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if "scx-bench" not in conda_prefix:
        msg = (
            "WARNING: gate_candidate.py was launched from a non-scx-bench "
            "environment (CONDA_PREFIX=%r). The orchestrator's per-SLURM-job "
            "setup will fall back to the dev .venv/ PATH, which is missing "
            "format-runner deps (mudata, slafdb, BPCells, etc.). Most "
            "convert jobs will fail with ImportError, cascading into "
            "DependencyNeverSatisfied chains that stall the gate.\n\n"
            "    To run end-to-end:\n"
            "        conda activate scx-bench\n"
            "        python benchmarks/comprehensive/scripts/gate_candidate.py ...\n\n"
            "    Or scope formats to ones that work in the dev venv:\n"
            "        --formats h5ad_none h5ad_gzip h5ad_lzf zarr_zstd zarr_lz4 \\\n"
            "                  tiledb_soma scx_auto scx_none scx_scx1 scx_zstd \\\n"
            "                  scx_lz4 scx_pcodec h5mu_uncompressed h5mu_gzip \\\n"
            "                  zarr_mudata_zstd scx_multimodal_per_modality_auto \\\n"
            "                  scx_multimodal_uniform_auto"
        ) % (conda_prefix or "(unset)",)
        print("\n" + "=" * 70, file=sys.stderr)
        print(msg, file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.extra_gate_args and args.extra_gate_args[0] == "--":
        args.extra_gate_args = args.extra_gate_args[1:]
    _check_conda_env()

    # Import submitit eagerly (before setup_logging) so its import-time
    # logging.config.dictConfig() side-effect can't close our FileHandler's
    # stream. submitit is optional — if unavailable, GPU pre-flight falls
    # back to local checks and benchmark dispatch will fail later with a
    # clearer error.
    try:
        import submitit  # noqa: F401  (side-effect: dictConfig)
    except ImportError:
        pass

    git_sha = _git_sha()
    today = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    if not args.name:
        args.name = f"candidate_{git_sha}_{today}"
    log_file = setup_logging(args.verbose, f"gate_candidate_{git_sha}_{timestamp}")

    candidate_dir = RESULTS / args.name
    main_log = phase_logger("main")
    main_log.info("scx regression gate starting (git_sha=%s)", git_sha)

    start = time.monotonic()
    plan: dict | None = None
    probe_info: dict | None = None
    phase = "init"
    cause: str | None = None
    rc = 0

    try:
        if not args.skip_preflight:
            phase = "pre-flight"
            checks, probe_info = run_preflight(args)
            failures = [c for c in checks if c.level == "fail"]
            if failures:
                cause = "pre-flight: " + "; ".join(c.name for c in failures)
                raise GateError("pre-flight", 2, cause)

        plan = coverage_plan(args)

        # Always-on cheap GPU sanity check (~1s) when GPU benchmarks are
        # in scope. Catches the "pyscx editable .so rebuilt without
        # --features gpu" case independently of --skip-preflight — that
        # silently breaks every GPU bench with `_HAS_PYSCX_GPU = False`
        # and we only learn after the full gate runs.
        if plan.get("accel_gpu"):
            phase = "worker-gpu-sanity"
            gpu_env = _resolve_worker_gpu_env()
            sanity = _check_worker_gpu_pyscx(
                gpu_env, preflight_ran=not args.skip_preflight
            )
            log = phase_logger("worker-gpu-sanity")
            # Map CheckResult levels to real logging.Logger method names
            # ("warn"/"fail" are not logger methods → AttributeError otherwise).
            log_method = {"ok": "info", "warn": "warning", "fail": "error"}.get(
                sanity.level, "info"
            )
            getattr(log, log_method)(
                "%s [%s]: %s", sanity.name, sanity.level, sanity.msg,
            )
            if sanity.level == "fail":
                cause = f"worker_gpu_pyscx: {sanity.msg}"
                raise GateError("worker-gpu-sanity", 2, cause)

        print_banner(args, candidate_dir, log_file, plan)

        if args.dry_run:
            phase = "dry-run"
            rc = run_dry_run(args, candidate_dir)
            if rc != 0:
                raise GateError("dry-run", 2, f"capture_baseline.py --mode dry-run exited with {rc}")
            elapsed = time.monotonic() - start
            write_summary(
                log_file, phase=phase, exit_code=0, elapsed=elapsed,
                candidate_dir=candidate_dir, plan=plan, probe=probe_info,
                cause="dry-run requested; no capture or gate executed",
            )
            return 0

        if not args.skip_capture:
            phase = "capture"
            run_capture(args, candidate_dir)
        else:
            main_log.info("--skip-capture: reusing %s", candidate_dir)
            if not candidate_dir.is_dir():
                raise GateError("capture", 2, f"{candidate_dir} does not exist")

        phase = "gate"
        rc = run_gate(args, candidate_dir)
        if rc not in (0, 1, 2):
            cause = f"compare_against_baseline.py returned unexpected exit {rc}"
            raise GateError("gate", rc, cause)
        if rc != 0:
            cause = (
                "regression / floor / fingerprint drift" if rc == 1
                else "missing inputs"
            )

    except GateError as e:
        rc = e.exit_code
        cause = e.message
        phase = e.phase
    except KeyboardInterrupt:
        rc = 130
        cause = "interrupted by user (SIGINT)"
        phase_logger(phase).error(cause)

    elapsed = time.monotonic() - start
    write_summary(
        log_file,
        phase=phase,
        exit_code=rc,
        elapsed=elapsed,
        candidate_dir=candidate_dir,
        plan=plan,
        probe=probe_info,
        cause=cause,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
