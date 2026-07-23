"""
Page-cache eviction helpers for cold-cache benchmarking.

Single source of truth for dropping OS page caches so benchmark modules (which
are *not* ``FormatRunner`` subclasses) don't have to reach into
``runners/base.py``'s classmethods. ``runners/base.py::_drop_file_cache``
delegates here so the two paths stay identical.

Two mechanisms:

  * :func:`drop_file_cache` — **unprivileged**, per-file
    ``posix_fadvise(POSIX_FADV_DONTNEED)``. Works on shared SLURM nodes without
    root. This is the one benchmark modules should use.
  * :func:`drop_caches_root` — system-wide ``/proc/sys/vm/drop_caches``; needs
    root, silently unavailable on Chimera worker nodes.

Both return / cooperate with a ``cache_policy`` string tag recorded verbatim in
each run's ``extra`` dict: ``"cold_fadvise"`` / ``"cold_root"`` / ``"warm"``.

``fadvise`` only evicts CLEAN pages, so any recent writes on ``path`` must be
``fsync``'d first — benchmark reads don't write, so this isn't a concern for the
read path.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["drop_file_cache", "drop_caches_root", "COLD_FADVISE", "COLD_ROOT", "WARM"]

COLD_FADVISE = "cold_fadvise"
COLD_ROOT = "cold_root"
WARM = "warm"

_drop_caches_root_warned = False


def drop_file_cache(path: str | Path) -> str:
    """Evict ``path`` from the page cache without root.

    Uses ``os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`` on the given file —
    or every regular file under the directory when ``path`` is a directory
    (covers SCX ``.scxd/`` shard trees, Zarr stores, SOMA experiments, SLAF
    DuckDB dirs, annbatch zarr shards).

    Returns the ``cache_policy`` label: ``"cold_fadvise"`` when at least one
    file was hinted for eviction, else ``"warm"`` (nonexistent path, platform
    without ``POSIX_FADV_DONTNEED``, or kernel refusal).
    """
    path = Path(path)
    if not path.exists():
        return WARM
    try:
        subprocess.run(["sync"], check=False)  # best-effort — failure not surfaced
        files: list[Path] = (
            [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        )
        evicted = 0
        for fp in files:
            try:
                fd = os.open(str(fp), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                evicted += 1
            except (AttributeError, OSError):
                # AttributeError: platform without POSIX_FADV_DONTNEED
                # (macOS, WSL1). OSError: kernel refused the hint (rare).
                pass
            finally:
                os.close(fd)
        return COLD_FADVISE if evicted > 0 else WARM
    except Exception:  # noqa: BLE001 — cache-drop is best-effort
        return WARM


def drop_caches_root() -> bool:
    """Attempt a system-wide page-cache drop (needs root/sysctl).

    Returns ``True`` on success. Logs a warning on the first failure and
    thereafter stays quiet. Prefer :func:`drop_file_cache` on shared SLURM
    nodes without root.
    """
    global _drop_caches_root_warned
    try:
        subprocess.run(["sync"], check=False)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, OSError):
        if not _drop_caches_root_warned:
            logger.warning(
                "Failed to drop page caches system-wide (requires root). "
                "Falling back to per-file posix_fadvise via drop_file_cache()."
            )
            _drop_caches_root_warned = True
        return False
