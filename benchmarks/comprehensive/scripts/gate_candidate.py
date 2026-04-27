#!/usr/bin/env python3
"""Local-only ad-hoc regression gate.

Captures a candidate snapshot via ``capture_baseline.py`` then runs
``compare_against_baseline.py --gate`` against ``results/baselines/LATEST``.
Replaces the prior ``gate_candidate.sh`` with structured logging,
pre-flight checks, and a coverage banner so operators can audit which
benchmark axes (format, accel CPU, accel GPU) actually ran.

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
phase / exit / elapsed / coverage plan for downstream tooling.
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


def run_preflight(args: argparse.Namespace) -> list[CheckResult]:
    log = phase_logger("pre-flight")
    checks: list[CheckResult] = [
        _check_repo_root(),
        _check_dirty_tree(),
        _check_helpers(),
        _check_baseline(args.baseline),
        _check_python_interp(args.python),
        _check_disk_space(),
        _check_slurm(),
    ]
    if not args.no_gpu:
        nvidia = _check_nvidia_smi()
        checks.append(nvidia)
        # Only check cupy if nvidia-smi found a GPU; else two cascading
        # failures confuse the operator.
        if nvidia.level == "ok":
            checks.append(_check_cupy(args.python))
    if not args.no_accel:
        checks.append(_check_pyscx_accel(args.python))

    width = max(len(c.name) for c in checks)
    marker = {"ok": "OK   ", "warn": "WARN ", "fail": "FAIL "}
    level_fn = {"ok": log.info, "warn": log.warning, "fail": log.error}
    for c in checks:
        level_fn[c.level]("  %-*s : %s — %s", width, c.name, marker[c.level], c.msg)
    return checks


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
) -> None:
    summary = {
        "phase": phase,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 2),
        "candidate": str(candidate_dir) if candidate_dir else None,
        "plan": plan,
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
    phase = "init"
    cause: str | None = None
    rc = 0

    try:
        if not args.skip_preflight:
            phase = "pre-flight"
            checks = run_preflight(args)
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
                candidate_dir=candidate_dir, plan=plan,
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
        cause=cause,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
