"""Shared environment configuration for benchmark scripts.

Loads SCX_WORK_DIR and SCX_DATA_DIR from the repo-root .env file
(via python-dotenv when available) and exports them as Path objects.

`python-dotenv` is only a convenience for the dev `.venv/`; on conda
envs (e.g. `scx-gpu`) it may be missing. In that case this module reads
`SCX_WORK_DIR` / `SCX_DATA_DIR` directly from `os.environ` — callers
are responsible for exporting them (e.g. via `source .env` in a SLURM
wrapper before invoking the benchmark script).

Usage:
    from bench_env import WORK_DIR, DATA_DIR
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    # `python-dotenv` not installed (common on conda envs). Skip silently;
    # caller is expected to have exported SCX_WORK_DIR directly.
    pass

_work = os.environ.get("SCX_WORK_DIR", "")
if not _work:
    raise RuntimeError(
        "SCX_WORK_DIR is not set. "
        "Create a .env file in the repo root (see .env.example) or export SCX_WORK_DIR."
    )

WORK_DIR = Path(_work)
_data = os.environ.get("SCX_DATA_DIR", "")
DATA_DIR = Path(_data) if _data else WORK_DIR / "benchmarks" / "datasets"
