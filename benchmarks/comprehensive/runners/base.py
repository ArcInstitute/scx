"""
Abstract base class for format-specific benchmark runners.

Every format (h5ad, Zarr, TileDB-SOMA, SCX, BPCells, Parquet) implements
this interface so the orchestrator can treat them uniformly.
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of a single timed operation."""
    wall_s: float
    user_s: float = 0.0
    sys_s: float = 0.0
    peak_rss_mb: float = 0.0
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "wall_s": round(self.wall_s, 6),
            "user_s": round(self.user_s, 6),
            "sys_s": round(self.sys_s, 6),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
        }
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class ConvertResult:
    """Result of a format conversion."""
    wall_s: float
    peak_rss_mb: float
    output_size_bytes: int
    write_throughput_mb_s: float = 0.0
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "wall_s": round(self.wall_s, 6),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "output_size_bytes": self.output_size_bytes,
            "write_throughput_mb_s": round(self.write_throughput_mb_s, 2),
        }
        if self.extra:
            d["extra"] = self.extra
        return d


class FormatRunner(ABC):
    """Abstract base for format-specific benchmark operations.

    Each subclass must implement the core operations (convert, read, subset)
    and the ``name`` / ``key`` properties.

    The base class provides shared helpers for timing, RSS measurement,
    and cold-cache flushing.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable format name (e.g. 'h5ad (gzip)')."""

    @property
    @abstractmethod
    def key(self) -> str:
        """Machine-readable key (e.g. 'h5ad_gzip').

        Used in filenames and JSON results.
        """

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @abstractmethod
    def convert_from_h5ad(self, h5ad_path: str | Path, output_path: str | Path) -> ConvertResult:
        """Convert an h5ad file to this format.

        Parameters
        ----------
        h5ad_path : path to source h5ad file
        output_path : path for the converted output

        Returns
        -------
        ConvertResult with timing, size, and throughput.
        """

    @abstractmethod
    def read_full(self, path: str | Path) -> TimingResult:
        """Read the entire expression matrix into an in-memory CSR.

        Parameters
        ----------
        path : path to the file/directory in this format

        Returns
        -------
        TimingResult with wall-clock time and peak RSS.
        """

    @abstractmethod
    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        """Read a subset of cells and/or genes.

        Parameters
        ----------
        path : path to the file/directory in this format
        cell_indices : which rows (cells) to read; None = all
        gene_indices : which columns (genes) to read; None = all

        Returns
        -------
        TimingResult with wall-clock time and peak RSS.
        """

    @abstractmethod
    def file_size(self, path: str | Path) -> int:
        """Return total on-disk size in bytes.

        For single-file formats, this is ``os.path.getsize()``.
        For directory-based formats (Zarr, SOMA, BPCells), this is
        the recursive size of the directory.
        """

    # ------------------------------------------------------------------
    # Optional operations (not all formats support these)
    # ------------------------------------------------------------------

    def read_backed(self, path: str | Path) -> TimingResult:
        """Open the file in backed/lazy mode (no full read).

        Returns TimingResult measuring open time and idle RSS.
        Raises NotImplementedError for formats without backed mode.
        """
        raise NotImplementedError(f"{self.name} does not support backed mode")

    def read_backed_slice(
        self,
        path: str | Path,
        start: int,
        count: int,
    ) -> TimingResult:
        """Read a contiguous row slice from backed mode.

        Parameters
        ----------
        path : format-specific path
        start : starting row index
        count : number of rows to read

        Returns
        -------
        TimingResult
        """
        raise NotImplementedError(f"{self.name} does not support backed row slices")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rss_mb() -> float:
        """Current RSS in MB (Linux: /proc/self/statm, fallback: ru_maxrss).

        Uses /proc/self/statm to get the *current* resident set size rather
        than the process-lifetime high-water mark (ru_maxrss), so that
        per-operation measurements are not inflated by earlier operations.
        """
        try:
            with open("/proc/self/statm") as f:
                # Field 1 is resident pages
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except (OSError, IndexError, ValueError):
            # Fallback for non-Linux platforms
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    @staticmethod
    def _get_cpu_times() -> tuple[float, float]:
        """Return (user_s, sys_s) from getrusage."""
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime, r.ru_stime

    _drop_caches_warned = False

    @classmethod
    def _drop_caches(cls) -> bool:
        """Attempt to drop OS page caches (requires root/sudo).

        Returns True if successful.  Logs a warning on the first failure.
        """
        try:
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
            return True
        except (PermissionError, OSError):
            if not cls._drop_caches_warned:
                logger.warning(
                    "Failed to drop page caches (requires root). "
                    "Cold-cache benchmark results may be unreliable."
                )
                cls._drop_caches_warned = True
            return False

    @staticmethod
    def _gc_collect() -> None:
        """Force garbage collection."""
        gc.collect()

    @staticmethod
    def _dir_size(path: str | Path) -> int:
        """Recursively compute directory size in bytes."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total

    def timed_run(self, fn, *args, **kwargs) -> tuple[Any, TimingResult]:
        """Run ``fn(*args, **kwargs)`` and return (result, TimingResult).

        Measures wall-clock time, user+sys CPU time, and peak RSS.
        RSS is sampled before and after the operation; the maximum of the
        two is reported as the peak for this operation.
        """
        self._gc_collect()
        rss_before = self._get_rss_mb()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        result = fn(*args, **kwargs)

        wall = time.perf_counter() - t0
        u1, s1 = self._get_cpu_times()
        rss_after = self._get_rss_mb()

        return result, TimingResult(
            wall_s=wall,
            user_s=u1 - u0,
            sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} key={self.key!r}>"
