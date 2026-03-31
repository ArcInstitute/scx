"""
System information collector for benchmark reproducibility.

Collects CPU model, core count, RAM, OS, kernel, disk type, Python version,
Rust version, and key library versions. Output is a JSON-serializable dict
included in every benchmark result.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def collect_system_info() -> dict[str, Any]:
    """Collect system hardware and software information.

    Returns a dict suitable for embedding in benchmark JSON results.
    """
    info: dict[str, Any] = {
        "hostname": os.uname().nodename,
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "python_version": platform.python_version(),
    }

    # CPU info
    info["cpu"] = _get_cpu_model()
    info["cpu_cores_physical"] = os.cpu_count() or 0

    # RAM
    info["ram_gb"] = _get_ram_gb()

    # Rust version
    info["rust_version"] = _get_rust_version()

    # Key library versions
    info["library_versions"] = _get_library_versions()

    # Storage info (best-effort)
    info["storage"] = _get_storage_info()

    return info


def _get_cpu_model() -> str:
    """Read CPU model name from /proc/cpuinfo (Linux only)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or "unknown"


def _get_ram_gb() -> int:
    """Read total RAM in GB from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / 1_048_576)  # KB → GB
    except (OSError, ValueError):
        pass
    return 0


def _get_rust_version() -> str:
    """Get installed Rust compiler version."""
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _get_library_versions() -> dict[str, str]:
    """Collect versions of key Python libraries.

    Failures are silently recorded as 'not installed'.
    """
    libs = {}
    packages = [
        ("anndata", "anndata"),
        ("scanpy", "scanpy"),
        ("zarr", "zarr"),
        ("h5py", "h5py"),
        ("scipy", "scipy"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("tiledbsoma", "tiledbsoma"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("umap-learn", "umap"),
        ("leidenalg", "leidenalg"),
        ("igraph", "igraph"),
        ("psutil", "psutil"),
    ]

    for display_name, import_name in packages:
        try:
            # Use importlib.metadata for version, avoiding deprecated __version__
            from importlib.metadata import version as get_version
            libs[display_name] = get_version(display_name)
        except Exception:
            try:
                mod = __import__(import_name)
                libs[display_name] = getattr(mod, "__version__", "unknown")
            except ImportError:
                libs[display_name] = "not installed"

    # pyscx — always from source
    try:
        import pyscx
        libs["pyscx"] = getattr(pyscx, "__version__", "dev (from source)")
    except ImportError:
        libs["pyscx"] = "not installed"

    return libs


def _get_storage_info() -> dict[str, str]:
    """Best-effort storage device detection."""
    info: dict[str, str] = {}
    try:
        # Check if /scratch is on NVMe (common HPC pattern)
        result = subprocess.run(
            ["df", "-T", "/scratch"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                info["scratch_device"] = parts[0]
                info["scratch_fstype"] = parts[1]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for NVMe devices
    try:
        result = subprocess.run(
            ["ls", "/dev/nvme0n1"],
            capture_output=True, text=True, timeout=2,
        )
        info["has_nvme"] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["has_nvme"] = False

    return info


def print_system_info() -> None:
    """Print collected system information to stdout."""
    import json
    info = collect_system_info()
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    print_system_info()
