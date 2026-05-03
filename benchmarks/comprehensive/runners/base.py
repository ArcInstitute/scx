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
import subprocess
import time

from benchmarks.comprehensive.rss import current_rss_mb
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

    # Declared-capability manifest. ``read_selective.py`` and
    # ``smoke_test_runners.py`` trust this set to decide whether a missing
    # optional method is a feature gap (skip quietly) or a contract
    # violation (fail loudly). Runners that advertise ``"filtered_query"``
    # must implement ``read_filtered_query``; ``"backed_mode"`` requires
    # ``read_backed`` + ``read_backed_slice``. Cloud capabilities:
    # ``"cloud_read"`` → ``read_cloud``; ``"cloud_subset"`` →
    # ``read_cloud_subset``; ``"cloud_push"`` → ``push``; ``"cloud_pull"``
    # → ``pull``; ``"cloud_filtered"`` → ``read_cloud_filtered_query``.
    capabilities: frozenset[str] = frozenset()

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
    # Cloud operations (Phase 5 — GCP only)
    # ------------------------------------------------------------------

    def read_cloud(self, cloud_url: str) -> TimingResult:
        """Read the entire expression matrix directly from a cloud URI.

        A runner that advertises ``"cloud_read"`` in its ``capabilities``
        set must override this method.

        Parameters
        ----------
        cloud_url : GCS URI (e.g. ``gs://arc-ctc-nextflow/scx-test/pbmc3k.scxd/``)

        Returns
        -------
        TimingResult
            ``extra`` should include ``"provider"`` (``"gcs"``) and a
            ``"native_mechanism"`` tag so reports can group apples-to-apples.
        """
        raise NotImplementedError(
            f"{self.name} does not support read_cloud"
        )

    def read_cloud_subset(
        self,
        cloud_url: str,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        """Read a subset of cells/genes directly from a cloud URI.

        A runner that advertises ``"cloud_subset"`` in its ``capabilities``
        set must override this method.
        """
        raise NotImplementedError(
            f"{self.name} does not support read_cloud_subset"
        )

    def push(self, local_path: str | Path, cloud_url: str) -> TimingResult:
        """Upload a local file/directory to the given cloud URI.

        A runner that advertises ``"cloud_push"`` in its ``capabilities``
        set must override this method.

        Returns a TimingResult whose ``extra`` dict should include
        ``"bytes_uploaded"`` and ``"throughput_mbps"``.
        """
        raise NotImplementedError(
            f"{self.name} does not support push"
        )

    def pull(self, cloud_url: str, local_path: str | Path) -> TimingResult:
        """Download a cloud URI into a local file/directory.

        A runner that advertises ``"cloud_pull"`` in its ``capabilities``
        set must override this method.

        Returns a TimingResult whose ``extra`` dict should include
        ``"bytes_downloaded"`` and ``"throughput_mbps"``.
        """
        raise NotImplementedError(
            f"{self.name} does not support pull"
        )

    def read_cloud_filtered_query(
        self,
        cloud_url: str,
        predicate: "Predicate",
    ) -> TimingResult:
        """Execute a format-native filtered query against a cloud URI.

        A runner that advertises ``"cloud_filtered"`` in its ``capabilities``
        set must override this method. The default raises so
        ``cloud_filtered.py`` can flag a contract violation instead of
        silently skipping.

        Parameters
        ----------
        cloud_url : GCS URI (e.g. ``gs://arc-ctc-nextflow/scx-test/pbmc3k.soma/``)
        predicate : one of the ``Predicate`` subclasses declared in
            ``benchmarks.comprehensive.queries``.

        Returns
        -------
        TimingResult
            Timing for the native filtered cloud read. The ``extra`` dict
            should include ``"provider"`` (``"gcs"``), ``"native_mechanism"``
            (e.g. ``"tiledb_cloud_value_filter"``, ``"slaf_cloud_sql"``,
            ``"scx_pull_and_filter"``), and ``"predicate"``.
        """
        raise NotImplementedError(
            f"{self.name} does not support read_cloud_filtered_query"
        )

    def read_filtered_query(
        self,
        path: str | Path,
        predicate: "Predicate",
    ) -> TimingResult:
        """Execute a format-native filtered query against obs/var metadata.

        A runner that advertises ``"filtered_query"`` in its ``capabilities``
        set must override this method; the default raises so
        ``read_selective.py`` can flag a contract violation instead of
        silently skipping.

        Parameters
        ----------
        path : format-specific path
        predicate : one of the ``Predicate`` subclasses declared in
            ``benchmarks.comprehensive.queries``.

        Returns
        -------
        TimingResult
            Timing for the native filtered read. The ``extra`` dict should
            include ``"native_mechanism"`` (e.g. ``"scx_pushdown"``,
            ``"slaf_sql"``, ``"slaf_stride_hash"``) so reporting can group
            apples-to-apples.
        """
        raise NotImplementedError(
            f"{self.name} does not support read_filtered_query"
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rss_mb() -> float:
        """Back-compat shim — delegates to
        ``benchmarks.comprehensive.rss.current_rss_mb``."""
        return current_rss_mb()

    @staticmethod
    def _get_cpu_times() -> tuple[float, float]:
        """Return (user_s, sys_s) from getrusage."""
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime, r.ru_stime

    _drop_caches_warned = False

    @classmethod
    def _drop_caches(cls) -> bool:
        """Attempt to drop OS page caches system-wide (requires root/sudo).

        Returns True if successful.  Logs a warning on the first failure.
        Prefer ``_drop_file_cache(path)`` on shared SLURM nodes without
        root — it uses ``posix_fadvise(POSIX_FADV_DONTNEED)`` on the
        specific file(s) and works unprivileged.
        """
        try:
            subprocess.run(["sync"], check=False)
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
            return True
        except (PermissionError, OSError):
            if not cls._drop_caches_warned:
                logger.warning(
                    "Failed to drop page caches (requires root). "
                    "Use `_drop_file_cache(path)` for per-file non-root cold "
                    "reads via posix_fadvise."
                )
                cls._drop_caches_warned = True
            return False

    @classmethod
    def _drop_file_cache(cls, path: str | Path) -> str:
        """Evict ``path`` from the page cache without root (Phase I.3).

        Uses ``os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`` on the
        given file — or every regular file under the directory when ``path``
        is a directory (covers SCX ``.scxd/`` shard trees, Zarr stores,
        SOMA experiments, SLAF DuckDB dirs).

        Returns the ``cache_policy`` label for the per-run ``extra`` dict:

          * ``"cold_fadvise"`` — the non-root path succeeded.
          * ``"cold_root"``   — caller combined this with a successful
            ``_drop_caches()`` (system-wide eviction).
          * ``"warm"``        — no eviction attempted / all attempts failed.

        The returned label is a string tag; callers record it verbatim.
        ``fadvise`` only evicts CLEAN pages — any recent writes on the path
        must be ``fsync``'d first (benchmark reads don't write, so this
        isn't a concern for the read-path).
        """
        path = Path(path)
        if not path.exists():
            return "warm"
        try:
            subprocess.run(["sync"], check=False)  # best-effort — no error surfaced on failure
            files: list[Path] = (
                [path] if path.is_file()
                else [p for p in path.rglob("*") if p.is_file()]
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
                    # AttributeError: pre-Py-3.3 or platform without
                    # POSIX_FADV_DONTNEED (macOS, WSL1). OSError: kernel
                    # refused the hint (rare).
                    pass
                finally:
                    os.close(fd)
            return "cold_fadvise" if evicted > 0 else "warm"
        except Exception:  # noqa: BLE001 — cache-drop is best-effort
            return "warm"

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
