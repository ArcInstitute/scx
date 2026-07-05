"""Shardad format benchmark runner.

Wraps the ``shardad`` package (``~/dev/python/shardad``): a single ``.shad``
container of condition-grouped, narrow-dtype, zstd-compressed CSR shards. Reads
always materialize the full covering rows into an in-memory AnnData (no lazy
query engine / backed mode), so ``capabilities`` is empty — shardad participates
in the format-driven benches (compression / write / read_full / read_selective /
memory / parallel_scaling) and opt-in round-trip / correctness, but not the
filtered-query, cloud, backed, or streaming benches.

Parallelism is Python multiprocessing (``n_workers``), not rayon, so the runner
maps ``RAYON_NUM_THREADS`` (which ``parallel_scaling`` / ``parallel_write_scaling``
set per subprocess) onto ``n_workers`` so those sweeps actually move.

The grouped-sharding head-to-head against scx lives in
``benchmarks/grouped_read.py``; this runner only covers the geometric (ungrouped)
format comparison.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from benchmarks.comprehensive.runners.base import (
    ConvertResult,
    FormatRunner,
    TimingResult,
)

try:
    import shardad  # noqa: F401
    from shardad import ShardedArchive, write_sharded

    _HAS_SHARDAD = True
except ImportError:
    _HAS_SHARDAD = False


_MISSING_MSG = (
    "shardad is not installed in this env. Install it into scx-bench with: "
    "conda activate scx-bench && pip install ~/dev/python/shardad"
)


def _workers() -> int:
    """Resolve the shardad worker count.

    ``parallel_scaling`` / ``parallel_write_scaling`` pin ``RAYON_NUM_THREADS``
    per subprocess to drive SCX's rayon pool; shardad has no rayon, so we map the
    same env onto its ``n_workers`` so the sweep is meaningful. Falls back to the
    CPU count.
    """
    env = os.environ.get("RAYON_NUM_THREADS", "").strip()
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass
    return os.cpu_count() or 1


class ShardadRunner(FormatRunner):
    """Benchmark runner for shardad ``.shad`` archives."""

    capabilities: frozenset[str] = frozenset()

    @property
    def name(self) -> str:
        return "Shardad"

    @property
    def key(self) -> str:
        return "shardad"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_shardad() -> None:
        if not _HAS_SHARDAD:
            raise ImportError(_MISSING_MSG)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(
        self, h5ad_path: str | Path, output_path: str | Path
    ) -> ConvertResult:
        self._require_shardad()
        h5ad_path = str(h5ad_path)
        output_path = str(output_path)
        n_workers = _workers()

        def _convert() -> None:
            # Pass the h5ad *path* (not a loaded AnnData) so shardad streams the
            # source instead of forcing a full in-memory read up front.
            write_sharded(
                h5ad_path, output_path, overwrite=True, n_workers=n_workers
            )

        _, timing = self.timed_run(_convert)

        output_size = os.path.getsize(output_path)
        throughput = 0.0
        if timing.wall_s > 0:
            throughput = (output_size / (1024 * 1024)) / timing.wall_s

        return ConvertResult(
            wall_s=timing.wall_s,
            peak_rss_mb=timing.peak_rss_mb,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
            extra={"n_workers": n_workers},
        )

    def read_full(self, path: str | Path) -> TimingResult:
        self._require_shardad()
        n_workers = _workers()

        def _read() -> None:
            arch = ShardedArchive(str(path))
            adata = arch.to_anndata(n_workers=n_workers)
            # Reference X to be explicit that the CSR is materialized.
            _ = adata.X

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        self._require_shardad()
        n_workers = _workers()

        # shardad selects on the row axis only; a gene projection has no
        # native pushdown, so we read the covering rows and slice columns in
        # memory (honest capability gap, mirrors the h5ad runner).
        if gene_indices is None:
            mechanism = "shardad_row_slice"
        elif cell_indices is None:
            mechanism = "shardad_load_and_slice_genes"
        else:
            mechanism = "shardad_row_slice+slice_genes"

        def _read_subset() -> None:
            arch = ShardedArchive(str(path))
            if cell_indices is not None:
                adata = arch[np.asarray(cell_indices)].to_anndata(n_workers=n_workers)
            else:
                adata = arch.to_anndata(n_workers=n_workers)
            X = adata.X
            if gene_indices is not None:
                X = X[:, gene_indices]
            if sp.issparse(X):
                X.toarray()
            else:
                _ = X.shape

        _, timing = self.timed_run(_read_subset)
        timing.extra = {"mechanism": mechanism}
        return timing

    def file_size(self, path: str | Path) -> int:
        return os.path.getsize(path)
