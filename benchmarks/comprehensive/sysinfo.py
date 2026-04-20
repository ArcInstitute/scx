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

    # GPU / CUDA / nvidia-fs (best-effort; only present on GPU nodes). Closes
    # the ROADMAP §5.1 claim that these are captured — Phase I.2.
    gpu = _get_gpu_info()
    if gpu:
        info["gpu"] = gpu

    # GCP compute-node matrix labels (Phase E). The launcher sets these env
    # vars on each VM before invoking the benchmark so the emitted JSON
    # self-labels with the instance type + region. Absent off-cloud, so
    # local runs never carry stale GCP tags.
    gcp_instance = os.environ.get("SCX_BENCH_GCP_INSTANCE")
    gcp_region = os.environ.get("SCX_BENCH_GCP_REGION")
    gcp_zone = os.environ.get("SCX_BENCH_GCP_ZONE")
    if gcp_instance or gcp_region or gcp_zone:
        gcp: dict[str, str] = {}
        if gcp_instance:
            gcp["instance_type"] = gcp_instance
        if gcp_region:
            gcp["region"] = gcp_region
        if gcp_zone:
            gcp["zone"] = gcp_zone
        info["gcp"] = gcp

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


def _get_gpu_info() -> dict[str, Any]:
    """Capture CUDA runtime / driver / nvidia-fs status (GPU nodes only).

    Returns an empty dict when the host has no GPU toolchain — callers
    then skip the ``gpu`` key entirely so CPU-only runs don't carry
    empty-string placeholders.
    """
    info: dict[str, Any] = {}

    # CUDA runtime (nvcc). Missing on nodes that have driver-only installs.
    try:
        out = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if "release" in line:
                    info["cuda_runtime"] = line.strip()
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Driver via nvidia-smi (the common case on compute nodes).
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            first = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",", 1)]
            if parts:
                info["driver_version"] = parts[0]
            if len(parts) > 1:
                info["gpu_name"] = parts[1]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # nvidia-fs (GPUDirect Storage). Presence of the kernel module implies
    # nvidia-fs is loaded; the GPU path under scx-gpu checks this too.
    try:
        mods = Path("/proc/modules")
        if mods.exists():
            info["nvidia_fs_loaded"] = "nvidia_fs" in mods.read_text()
    except OSError:
        pass

    return info


def print_system_info() -> None:
    """Print collected system information to stdout."""
    import json
    info = collect_system_info()
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    print_system_info()
