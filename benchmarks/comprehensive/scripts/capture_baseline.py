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
TIERS = {
    "small": {
        "datasets": ["pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k"],
        "partition": "cpu_preemptible",
        "mem_gb": 8,
        "timeout": 240,
    },
    "full": {
        "datasets": [
            "pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k",
            "census_500k", "census_1m",
        ],
        "partition": "cpu_preemptible",
        "mem_gb": 8,
        "timeout": 480,
    },
    "xl": {
        "datasets": [
            "pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k",
            "census_500k", "census_1m", "census_5m",
        ],
        "partition": "cpu_preemptible",
        "mem_gb": 8,
        "timeout": 960,
    },
}

BENCHMARKS = [
    "compression",
    "write",
    "read_full",
    "read_selective",
    "parallel_scaling",
    "parallel_write_scaling",
    "memory",
    "fragment_ops",
    "cloud_push",
    "cloud_pull",
    "cloud_read",
    "cloud_metadata",
]


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
) -> int:
    """Invoke run_parallel.py with the selected tier's settings.

    run_parallel.py submits one SLURM job per (benchmark, dataset, format)
    triple and blocks until all of them finish.  Returns its exit code.
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmarks" / "comprehensive" / "scripts" / "run_parallel.py"),
        "--datasets", *tier_cfg["datasets"],
        "--benchmarks", *BENCHMARKS,
        "--partition", tier_cfg["partition"],
        "--mem-gb",    str(tier_cfg["mem_gb"]),
        "--timeout",   str(tier_cfg["timeout"]),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if skip_convert:
        cmd.append("--skip-convert")
    if overwrite:
        cmd.append("--overwrite")

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
) -> dict[str, Any]:
    """Copy result JSONs for the selected tier into baseline/raw/.

    Returns a per-file summary dict suitable for summary.json.
    """
    dst = baseline_dir / "raw"
    dst.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    copied = 0
    skipped = 0
    for src in sorted(RAW_DIR.glob("*.json")):
        if not _tier_matches(src.name, tier_cfg):
            continue
        shutil.copy2(src, dst / src.name)
        copied += 1

        try:
            data = json.loads(src.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue

        key = f"{data.get('benchmark')}__{data.get('format')}__{data.get('dataset')}"
        summary[key] = {
            "median_wall_s": data.get("median_wall_s"),
            "peak_rss_mb_median": _median_rss(data.get("runs") or []),
            "file_size_bytes": data.get("file_size_bytes"),
            "source_file": src.name,
        }

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
    args = parser.parse_args()

    tier_cfg = TIERS[args.tier]
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
    if args.mode == "submit":
        rc = submit_benchmarks(
            tier_cfg,
            dry_run=False,
            skip_convert=args.skip_convert,
            overwrite=args.overwrite,
        )
        if rc != 0:
            print(f"[baseline] WARNING: run_parallel.py exited with {rc}; archiving whatever landed")
    elif args.mode == "dry-run":
        submit_benchmarks(tier_cfg, dry_run=True,
                          skip_convert=args.skip_convert, overwrite=False)
        return 0

    # 3. Archive raw JSON + write summary.json.
    if args.mode != "fingerprint-only":
        summary = archive_raw_results(tier_cfg, baseline_dir)
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
