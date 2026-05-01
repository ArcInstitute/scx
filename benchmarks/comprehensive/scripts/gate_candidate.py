#!/usr/bin/env python3
"""Ad-hoc regression gate (local controller, SLURM workers).

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
cupy / pyscx.accel on a real GPU instead of the submission host). Pass
``--probe-partition -`` to force local GPU checks, or ``--no-gpu`` to skip
GPU coverage entirely.

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

    if want_cluster_probe:
        gpu_checks, probe_info = _run_cluster_gpu_probe(args)
        checks.extend(gpu_checks)
    elif not args.no_gpu:
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
        info_s = f" gpu_info={info!s:.80}" if info else ""
        out.append(CheckResult("pyscx_accel", "ok", f"import + gpu_info ok{info_s}"))
    else:
        out.append(CheckResult(
            "pyscx_accel", "fail",
            f"probe import failed: {p.get('error', '(no detail)')}",
        ))
    return out


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
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Restrict to a subset of the canonical benchmark "
                             "list (e.g. `--benchmarks accel_pca`). Forwarded "
                             "to capture_baseline.py.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Override the tier's dataset list (forwarded to "
                             "capture_baseline.py). Useful with --benchmarks "
                             "for narrow spot-checks.")
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
    probe.add_argument("--probe-conda-env", default="scx-gpu",
                       help="Conda env to activate inside the probe job "
                            "(default: scx-gpu). Empty string skips activation.")
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


def main() -> int:
    args = parse_args()
    if args.extra_gate_args and args.extra_gate_args[0] == "--":
        args.extra_gate_args = args.extra_gate_args[1:]

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
