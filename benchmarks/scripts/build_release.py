"""Ensure pyscx is built in release mode before benchmarking."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYSCX_DIR = PROJECT_ROOT / "pyscx"
MATURIN = str(PROJECT_ROOT / ".venv" / "bin" / "maturin")


def ensure_release_build():
    """Build pyscx with maturin develop --release.

    Call this before importing pyscx in any benchmark script to ensure
    the native extension is compiled with optimizations enabled.
    """
    print("Building pyscx in release mode...")
    result = subprocess.run(
        [MATURIN, "develop", "--release"],
        cwd=PYSCX_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: maturin develop --release failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  pyscx release build complete.")
