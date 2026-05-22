"""Shared environment configuration for benchmark scripts.

Loads SCX_WORK_DIR and SCX_DATA_DIR from the repo-root .env file
(via python-dotenv when available) and exports them as Path objects.

`python-dotenv` is only a convenience for the dev `.venv/`; on conda
envs (e.g. `scx-gpu`) it may be missing. In that case this module reads
`SCX_WORK_DIR` / `SCX_DATA_DIR` directly from `os.environ` — callers
are responsible for exporting them (e.g. via `source .env` in a SLURM
wrapper before invoking the benchmark script).

Usage:
    from benchmarks.comprehensive.bench_env import WORK_DIR, DATA_DIR
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# F3 (SCX-USER-REPORT-2026-05-21-Tier3): track whether dotenv was actually
# imported so the SCX_WORK_DIR error below can distinguish "auto-load
# failed because python-dotenv isn't installed" from "auto-load ran but
# .env doesn't define SCX_WORK_DIR". Pre-fix, both modes raised the same
# generic error and dev `.venv/` users (no dotenv installed) wasted a
# 4-second job each time figuring out the cause.
_dotenv_imported = False
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    _dotenv_imported = True
except ImportError:
    # `python-dotenv` not installed (common on conda envs). Surfaced
    # explicitly in the error message below if SCX_WORK_DIR is also unset.
    pass

_work = os.environ.get("SCX_WORK_DIR", "")
if not _work:
    env_path = REPO_ROOT / ".env"
    if _dotenv_imported:
        raise RuntimeError(
            f"SCX_WORK_DIR is not set, even after auto-loading {env_path} "
            f"via python-dotenv. Add `SCX_WORK_DIR=...` to {env_path} (see "
            ".env.example) or `export SCX_WORK_DIR=...` in your shell."
        )
    raise RuntimeError(
        f"SCX_WORK_DIR is not set, and python-dotenv is not installed in "
        f"this Python environment ({sys.executable}) so {env_path} was not "
        "auto-loaded. Either install python-dotenv "
        "(`pip install python-dotenv`, or `pip install -e ./pyscx[dev]` "
        "to get all dev tools at once) or `export SCX_WORK_DIR=...` in "
        "your shell before running."
    )

WORK_DIR = Path(_work)
_data = os.environ.get("SCX_DATA_DIR", "")
DATA_DIR = Path(_data) if _data else WORK_DIR / "benchmarks" / "datasets"
