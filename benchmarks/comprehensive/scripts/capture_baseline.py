#!/usr/bin/env python3
"""
Capture a regression benchmark snapshot into
benchmarks/comprehensive/results/<name>/ (default <name>: baseline_2026_04_17).

This implements step 2 of the "Systemic Mitigation Strategy" in
2026-04-17_CODE-REVIEW.md (§12.4).  The initial baseline is captured once;
every subsequent PR captures its own snapshot into a separately-named
directory and runs compare_against_baseline.py to diff them.

Snapshots live in a gitignored directory tree — distinguish runs by the
--name flag, not by git branch.  Conventional names:
    baseline_<date>                 first reference point on main.
    candidate_<date>_<finding>      a PR's run, tagged with the review
                                    finding it addresses (e.g.
                                    candidate_2026_05_01_h10_chacha).

What gets captured
------------------
  1. environment.json        — git SHA, branch, dirty flag, rust/python versions,
                                library versions, hardware, env vars that affect
                                determinism (RAYON_NUM_THREADS, OMP_NUM_THREADS).
  2. raw/                    — per (benchmark, format, dataset) JSON from the
                                comprehensive suite.  Copied from
                                `results/raw/` after the suite finishes.
  3. fingerprints/           — per-accelerator BLAKE3 hashes of canonical
                                outputs, produced by fingerprint_accelerators.py.
  4. summary.json            — aggregated wall-clock / peak RSS / file-size
                                table keyed by (benchmark, format, dataset),
                                for fast diffing without re-parsing 500+ JSONs.
  5. MANIFEST.sha256         — sha256 of every captured file, so the baseline
                                itself is tamper-evident.

Operating modes
---------------
  submit   — default: run run_parallel.py + fingerprint script, then archive.
  archive  — skip running, just snapshot whatever is currently in `results/raw/`
             and run the fingerprint script.  Useful if the SLURM jobs were
             already launched separately.
  rerun    — force re-submit even if a result file already exists (otherwise
             run_parallel.py preserves existing outputs).

Tier selection
--------------
  --tier small   -> D1-D4 only (pbmc3k, pbmc10k, smartseq2, tabula_sapiens_100k).
                    ~1h wall time on cpu_preemptible.
  --tier full    -> D1-D6 (adds census_500k, census_1m). Needs ~4-6h + ~200 GB mem.
  --tier xl      -> D1-D7 (adds census_5m). Requires cpu_high_mem, 500 GB.

Usage
-----
    # Typical baseline capture for the code-review sprint:
    python benchmarks/comprehensive/scripts/capture_baseline.py \
        --tier full --mode submit

    # Only fingerprints + environment snapshot (quick sanity check):
    python benchmarks/comprehensive/scripts/capture_baseline.py \
        --mode fingerprint-only

    # Archive whatever is in results/raw/ right now without running anything:
    python benchmarks/comprehensive/scripts/capture_baseline.py --mode archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASELINE_NAME = "baseline_2026_04_17"
RESULTS_ROOT = PROJECT_ROOT / "benchmarks" / "comprehensive" / "results"
RAW_DIR = RESULTS_ROOT / "raw"

# `mem_gb` here is the FLOOR forwarded to run_parallel.py — actual memory is
# sized per-(benchmark, dataset, format) by `estimate_memory_gb`. Partition is
# the DEFAULT and is auto-overridden to `cpu_high_mem` for jobs whose estimate
# exceeds 200 GB. (Pre-fix, both values were applied uniformly to every job.)
#
# `DEFAULT_PARTITION` is the SLURM partition used by every tier unless
# overridden via `--partition`. Defaults to Chimera's `cpu_preemptible`
# (the historical config); pass `--partition preemptible` (or any other
# value) for clusters where that partition does not exist (e.g. Lambda
# HPC, which has `preemptible` / `standard` / `large_batch` instead).
# The `SCX_BENCH_PARTITION` env var also overrides the default — useful
# for outer SLURM wrappers that already know the cluster.
DEFAULT_PARTITION = os.environ.get("SCX_BENCH_PARTITION", "cpu_preemptible")

TIERS = {
    "small": {
        "datasets": ["pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k"],
        "partition": DEFAULT_PARTITION,
        "mem_gb": 8,
        "timeout": 240,
    },
    "full": {
        "datasets": [
            "pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k",
            "census_500k", "census_1m",
        ],
        "partition": DEFAULT_PARTITION,
        "mem_gb": 8,
        "timeout": 480,
    },
    "xl": {
        "datasets": [
            "pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k",
            "census_500k", "census_1m", "census_5m",
        ],
        "partition": DEFAULT_PARTITION,
        "mem_gb": 8,
        "timeout": 960,
    },
}

from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

# Canonical benchmark list lives in benchmarks/__init__.py::ALL_BENCHMARKS.
# Alias kept for compatibility with callers that import BENCHMARKS.
BENCHMARKS = list(ALL_BENCHMARKS)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command, return stdout stripped or 'unknown' on failure."""
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def capture_environment() -> dict[str, Any]:
    from benchmarks.comprehensive.sysinfo import collect_system_info

    # Single `git status --porcelain` — an empty output means a clean tree,
    # the sentinel "unknown" means the command failed (not a dirty tree).
    git_status = _run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT)
    env: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT),
        "git_dirty": git_status not in ("", "unknown"),
        "git_describe": _run(["git", "describe", "--always", "--dirty"], cwd=PROJECT_ROOT),
        "determinism_env": {
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "unset"),
            "OMP_NUM_THREADS":   os.environ.get("OMP_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS":   os.environ.get("MKL_NUM_THREADS", "unset"),
        },
    }
    env["system"] = collect_system_info()
    return env


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def submit_benchmarks(
    tier_cfg: dict[str, Any],
    dry_run: bool,
    skip_convert: bool,
    overwrite: bool,
    include_accel: bool = False,
    no_accel: bool = False,
    accel_only: bool = False,
    no_gpu: bool = False,
    benchmarks: list[str] | None = None,
    datasets: list[str] | None = None,
    skip_smoke: bool = False,
) -> int:
    """Invoke run_parallel.py with the selected tier's settings.

    run_parallel.py submits one SLURM job per (benchmark, dataset, format)
    triple and blocks until all of them finish.  Returns its exit code.

    Coverage flags (mutually exclusive: at most one of include_accel / no_accel /
    accel_only is honored, with accel_only > no_accel > include_accel):
      include_accel  forwards --include-accel so accel_* formats are scheduled
                     alongside format benchmarks.
      no_accel       drops every accel_* benchmark from the submitted list.
      accel_only     keeps only accel_* benchmarks (drops compression / read /
                     write / parallel_scaling / memory / ml_loader / cloud_*).
      no_gpu         forwards --no-gpu so accel formats matching *_gpu* are
                     filtered out of the cross-product.
    """
    if accel_only:
        bench_list = [b for b in BENCHMARKS if b.startswith("accel_")]
    elif no_accel:
        bench_list = [b for b in BENCHMARKS if not b.startswith("accel_")]
    else:
        bench_list = list(BENCHMARKS)

    # Optional explicit narrowing — must be a strict subset of bench_list so
    # the gate's coverage banner stays accurate. Rejects unknown names.
    if benchmarks:
        unknown = [b for b in benchmarks if b not in BENCHMARKS]
        if unknown:
            raise SystemExit(f"[baseline] unknown --benchmarks: {unknown}")
        bench_list = [b for b in bench_list if b in benchmarks]
        if not bench_list:
            raise SystemExit(
                f"[baseline] --benchmarks {benchmarks} produced an empty list "
                "after coverage filters (--accel-only / --no-accel)"
            )

    ds_list = list(datasets) if datasets else list(tier_cfg["datasets"])

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmarks" / "comprehensive" / "scripts" / "run_parallel.py"),
        "--datasets", *ds_list,
        "--benchmarks", *bench_list,
        "--partition", tier_cfg["partition"],
        "--mem-gb",    str(tier_cfg["mem_gb"]),
        "--timeout",   str(tier_cfg["timeout"]),
    ]
    if dry_run:
        # In dry-run we just want to print the schedule. The runner contract
        # check (smoke_test_runners) belongs to real submission paths.
        cmd.extend(["--dry-run", "--skip-smoke"])
    elif skip_smoke:
        cmd.append("--skip-smoke")
    if skip_convert:
        cmd.append("--skip-convert")
    if overwrite:
        cmd.append("--overwrite")
    # `accel_only` already restricted bench_list to accel_*; --include-accel is
    # auto-enabled in run_parallel.py when --benchmarks names any accel_* entry.
    # We still pass it explicitly when requested for parity with the gate's
    # coverage banner.
    if include_accel or accel_only:
        cmd.append("--include-accel")
    if no_gpu:
        cmd.append("--no-gpu")

    print("[baseline] submitting benchmarks:")
    print("           " + " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Archival
# ---------------------------------------------------------------------------


def _tier_matches(filename: str, tier_cfg: dict[str, Any]) -> bool:
    """Does this results/raw/<name>.json belong to the current tier?"""
    # Convention: filenames are "<bench>__<format>__<dataset>.json".
    parts = filename.rsplit("__", 2)
    if len(parts) != 3 or not parts[-1].endswith(".json"):
        return False
    dataset = parts[-1][:-len(".json")]
    return dataset in tier_cfg["datasets"]


def archive_raw_results(
    tier_cfg: dict[str, Any],
    baseline_dir: Path,
    *,
    since_mtime: float | None = None,
) -> dict[str, Any]:
    """Copy result JSONs for the selected tier into baseline/raw/.

    ``since_mtime`` (epoch seconds), when set, filters out files written
    before this run started — without it, ``RAW_DIR`` accumulates results
    from every prior invocation and would silently archive a mix of fresh
    and stale results into the candidate snapshot. The gate would then
    diff against that mix and report regressions that have nothing to do
    with the current commit.

    Returns a per-file summary dict suitable for summary.json.
    """
    dst = baseline_dir / "raw"
    dst.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    copied = 0
    skipped = 0
    stale_filtered = 0
    for src in sorted(RAW_DIR.glob("*.json")):
        if not _tier_matches(src.name, tier_cfg):
            continue
        if since_mtime is not None and src.stat().st_mtime < since_mtime:
            stale_filtered += 1
            continue
        shutil.copy2(src, dst / src.name)
        copied += 1

        try:
            data = json.loads(src.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue

        key = f"{data.get('benchmark')}__{data.get('format')}__{data.get('dataset')}"
        runs = data.get("runs") or []
        # Prefer the in-JSON field (results.py v2+) so ad-hoc edits to a raw
        # JSON's runs[] don't silently drift from the recorded median; fall
        # back to recomputing from runs[] for legacy v1 files lacking the
        # field. Same logic for n_runs.
        wall_s_iqr = data.get("wall_s_iqr")
        if wall_s_iqr is None:
            wall_s_iqr = _wall_s_iqr(runs)
        n_runs = data.get("n_runs")
        if n_runs is None:
            n_runs = len(runs)
        summary[key] = {
            "median_wall_s": data.get("median_wall_s"),
            "wall_s_iqr": wall_s_iqr,
            "n_runs": n_runs,
            "peak_rss_mb_median": _median_rss(runs),
            "file_size_bytes": data.get("file_size_bytes"),
            "source_file": src.name,
        }

    if stale_filtered:
        print(
            f"[baseline] archived {copied} result files ({skipped} unparseable, "
            f"{stale_filtered} pre-run files skipped)"
        )
    else:
        print(f"[baseline] archived {copied} result files ({skipped} unparseable)")
    return summary


def _median_rss(runs: list[dict[str, Any]]) -> float | None:
    vals = [r.get("peak_rss_mb") for r in runs if r.get("peak_rss_mb") is not None]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    if n % 2 == 1:
        return float(vals_sorted[n // 2])
    return float((vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2)


def _wall_s_iqr(runs: list[dict[str, Any]]) -> float | None:
    """IQR (p75 - p25) of wall_s across runs, mirroring
    BenchmarkResult.wall_s_iqr — keep the two in sync. Used only as the
    v1 → v2 backfill when the raw JSON predates SCHEMA_VERSION=2 and
    lacks the field. Returns ``None`` for n < 3 so the gate falls back
    to the fixed ``--timing-tolerance`` rather than over-widening on a
    poorly-estimated 2-sample dispersion (see results.py)."""
    import statistics
    walls = [r.get("wall_s") for r in runs if r.get("wall_s") is not None]
    if len(walls) < 3:
        return None
    walls = [float(w) for w in walls]
    q = statistics.quantiles(walls, n=4, method="exclusive")
    return float(q[2] - q[0])


def run_fingerprints(baseline_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmarks" / "comprehensive" / "scripts" / "fingerprint_accelerators.py"),
        "--output-dir", str(baseline_dir / "fingerprints"),
    ]
    print("[baseline] running accelerator fingerprints")
    return subprocess.run(cmd).returncode


def write_manifest(baseline_dir: Path) -> None:
    """sha256 of every file under the baseline directory."""
    manifest = baseline_dir / "MANIFEST.sha256"
    lines: list[str] = []
    for path in sorted(baseline_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        rel = path.relative_to(baseline_dir).as_posix()
        lines.append(f"{h.hexdigest()}  {rel}")
    manifest.write_text("\n".join(lines) + "\n")
    print(f"[baseline] wrote {manifest} ({len(lines)} files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--tier", choices=list(TIERS.keys()), default="full")
    parser.add_argument(
        "--mode",
        choices=["submit", "archive", "fingerprint-only", "dry-run"],
        default="submit",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Assume format conversions already exist on disk.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run all conversions even if outputs exist.",
    )
    parser.add_argument(
        "--skip-fingerprints", action="store_true",
        help="Don't run the fingerprint script (e.g. if pyscx isn't importable).",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_BASELINE_NAME,
        help=(
            "Directory name under benchmarks/comprehensive/results/ that this "
            "snapshot writes to.  The results directory is gitignored, so the "
            "name is the only way runs are distinguished — pick something "
            "descriptive for PR runs, e.g. 'candidate_2026_05_01_h10_chacha'. "
            f"(default: {DEFAULT_BASELINE_NAME})"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Override the parent results dir (default: "
            "benchmarks/comprehensive/results).  --name is resolved relative "
            "to this."
        ),
    )
    # Coverage flags — drive the gate's CPU/GPU/accel mode selection. Mutually
    # exclusive at most one of {--include-accel, --no-accel, --accel-only}.
    coverage = parser.add_mutually_exclusive_group()
    coverage.add_argument(
        "--include-accel", action="store_true",
        help="Schedule accel_* benchmarks alongside format benchmarks.",
    )
    coverage.add_argument(
        "--no-accel", action="store_true",
        help="Drop accel_* benchmarks (format benchmarks only).",
    )
    coverage.add_argument(
        "--accel-only", action="store_true",
        help="Run only accel_* benchmarks (skip compression/read/write/etc.).",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Drop GPU accel format variants (*_gpu*). Use on CPU-only hosts "
             "or to validate CPU-only changes.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Restrict to a subset of the canonical benchmark list "
             "(e.g. `--benchmarks accel_pca`). Must be a strict subset; "
             "unknown names raise. Combined with the coverage flags.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Override the tier's dataset list (e.g. "
             "`--datasets pbmc3k tabula_sapiens_100k`).",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip the pre-submit format-runner contract check. Useful for "
             "narrow accel-only runs or when known-broken format runners "
             "(BPCells / Parquet) are blocking submission of unrelated work.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help=(
            "SLURM partition for the per-(benchmark, dataset, format) jobs "
            "this script dispatches via run_parallel.py. Overrides the "
            "tier's default partition (and the `SCX_BENCH_PARTITION` env "
            f"var). Default: {DEFAULT_PARTITION!r}. Use this when running "
            "on a cluster whose partition names differ from Chimera's "
            "(e.g. Lambda HPC: --partition preemptible)."
        ),
    )
    args = parser.parse_args()

    tier_cfg = dict(TIERS[args.tier])  # shallow copy so we can override
    if args.partition is not None:
        tier_cfg["partition"] = args.partition
    results_root = Path(args.results_dir) if args.results_dir else RESULTS_ROOT
    baseline_dir = results_root / args.name
    baseline_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] target dir: {baseline_dir}")
    print(f"[baseline] tier:       {args.tier}  ({len(tier_cfg['datasets'])} datasets)")
    print(f"[baseline] mode:       {args.mode}")

    # 1. Always snapshot the environment first — it describes what the rest
    #    of the artefacts are about to be captured under.
    env = capture_environment()
    env["snapshot_name"] = args.name
    (baseline_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))
    print(f"[baseline] git_sha: {env['git_sha']} (branch: {env['git_branch']})")
    if env["git_dirty"]:
        print("[baseline] WARNING: working tree is dirty — baseline will not be reproducible")

    # 2. Run the comprehensive suite (unless we're just archiving or
    #    fingerprinting what already exists).
    # Capture start time so the archive step can filter out stale RAW_DIR
    # entries from prior invocations. ``time.time()`` is wall-clock seconds
    # matching ``stat().st_mtime``.
    submission_start: float | None = None

    if args.mode == "submit":
        # Margin: subtract 1s so we don't lose results written within the
        # same second by a fast-running benchmark.
        submission_start = time.time() - 1.0
        rc = submit_benchmarks(
            tier_cfg,
            dry_run=False,
            skip_convert=args.skip_convert,
            overwrite=args.overwrite,
            include_accel=args.include_accel,
            no_accel=args.no_accel,
            accel_only=args.accel_only,
            no_gpu=args.no_gpu,
            benchmarks=args.benchmarks,
            datasets=args.datasets,
            skip_smoke=args.skip_smoke,
        )
        if rc != 0:
            print(
                f"[baseline] ERROR: run_parallel.py exited with {rc}; "
                f"refusing to archive stale RAW_DIR contents. Re-run after "
                f"fixing the underlying failure."
            )
            return rc
    elif args.mode == "dry-run":
        submit_benchmarks(
            tier_cfg, dry_run=True,
            skip_convert=args.skip_convert, overwrite=False,
            include_accel=args.include_accel,
            no_accel=args.no_accel,
            accel_only=args.accel_only,
            no_gpu=args.no_gpu,
            benchmarks=args.benchmarks,
            datasets=args.datasets,
            skip_smoke=args.skip_smoke,
        )
        return 0

    # 3. Archive raw JSON + write summary.json. ``since_mtime`` filters
    #    stale entries from prior runs out of ``RAW_DIR`` so only fresh
    #    results land in the candidate snapshot.
    if args.mode != "fingerprint-only":
        summary = archive_raw_results(
            tier_cfg, baseline_dir, since_mtime=submission_start,
        )
        (baseline_dir / "summary.json").write_text(json.dumps(
            {
                "snapshot_name": args.name,
                "tier": args.tier,
                "datasets": tier_cfg["datasets"],
                "rows": summary,
            },
            indent=2, default=str,
        ))

    # 4. Run accelerator fingerprints.
    if not args.skip_fingerprints:
        fp_rc = run_fingerprints(baseline_dir)
        if fp_rc != 0:
            print(f"[baseline] WARNING: fingerprint_accelerators.py exited with {fp_rc}")

    # 5. Finalize with a sha256 manifest of every captured file.
    write_manifest(baseline_dir)

    print(f"[baseline] DONE — artefacts in {baseline_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
