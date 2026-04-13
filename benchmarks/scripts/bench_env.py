"""Shared environment configuration for benchmark scripts.

Loads SCX_WORK_DIR and SCX_DATA_DIR from the repo-root .env file
(via python-dotenv) and exports them as Path objects.

Usage:
    from bench_env import WORK_DIR, DATA_DIR
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

_work = os.environ.get("SCX_WORK_DIR", "")
if not _work:
    raise RuntimeError(
        "SCX_WORK_DIR is not set. "
        "Create a .env file in the repo root (see .env.example) or export SCX_WORK_DIR."
    )

WORK_DIR = Path(_work)
DATA_DIR = Path(os.environ.get("SCX_DATA_DIR", "")) or WORK_DIR / "benchmarks" / "datasets"
