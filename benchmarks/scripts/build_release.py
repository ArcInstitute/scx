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
    # `--features` REPLACES maturin's default feature set, so hdf5 (from_h5ad)
    # and gpu must be listed explicitly — omitting them silently drops those
    # capabilities from the shared editable .so across all envs.
    result = subprocess.run(
        [MATURIN, "develop", "--release", "--features", "hdf5,gpu"],
        cwd=PYSCX_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: maturin develop --release failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  pyscx release build complete.")
