"""CellStream format benchmark runner.

Wraps the ``cellstream`` package (``~/dev/python/cellstream``): a scatter-immune,
per-cell on-disk store (directory of ``payload.bin`` + ``offsets``/``indptr``
sidecars + ``genes.npy`` + ``obs/`` + JSON manifest) over shardad's codec, built
for random-access row gather during training. Like shardad it materializes covering
rows on read (no lazy query engine / backed pushdown), so ``capabilities`` is empty
— it participates in the format-driven + ml_loader benches, not the filtered-query /
cloud / backed / streaming ones.

Fixtures are written in **source (h5ad) row order** via the low-level
``cellstream.writer.write_store`` — NOT the public ``cellstream.write`` h5ad path,
which sorts by ``(context, perturbation)`` (columns the benchmark h5ads lack) and
would both crash and break row alignment with the ``.scx`` fixture. write_store
writes rows as given, auto-selects the on-disk count dtype, and validates
non-negative integer counts.
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
    import cellstream  # noqa: F401
    from cellstream import format as _cs_fmt
    from cellstream.writer import write_store

    _HAS_CELLSTREAM = True
except ImportError:
    _HAS_CELLSTREAM = False


_MISSING_MSG = (
    "cellstream is not installed in this env. Install it into scx-bench with: "
    "conda activate scx-bench && pip install pyfastpfor && "
    "pip install -e ~/dev/python/cellstream"
)


def _dir_size(path: str) -> int:
    """Total bytes of a cellstream store directory (sum of all files)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


class CellStreamRunner(FormatRunner):
    """Benchmark runner for cellstream directory stores."""

    capabilities: frozenset[str] = frozenset()

    @property
    def name(self) -> str:
        return "CellStream"

    @property
    def key(self) -> str:
        return "cellstream"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_cellstream() -> None:
        if not _HAS_CELLSTREAM:
            raise ImportError(_MISSING_MSG)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(
        self, h5ad_path: str | Path, output_path: str | Path
    ) -> ConvertResult:
        self._require_cellstream()
        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        def _convert() -> None:
            import anndata

            ad = anndata.read_h5ad(h5ad_path)
            X = ad.X.tocsr()
            X.sort_indices()
            genes = np.asarray(ad.var_names.tolist())
            # Preserve source row order (align with the .scx fixture); obs is not
            # needed for the ml_loader gather benchmark, so write an empty obs.
            write_store(
                output_path,
                X,
                {},
                genes,
                codec_name=_cs_fmt.CODEC_PFORDELTA,
            )

        _, timing = self.timed_run(_convert)

        output_size = _dir_size(output_path)
        throughput = 0.0
        if timing.wall_s > 0:
            throughput = (output_size / (1024 * 1024)) / timing.wall_s

        return ConvertResult(
            wall_s=timing.wall_s,
            peak_rss_mb=timing.peak_rss_mb,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
            extra={"codec": "pfordelta"},
        )

    def read_full(self, path: str | Path) -> TimingResult:
        self._require_cellstream()

        def _read() -> None:
            import cellstream

            store = cellstream.open(str(path))
            try:
                X = store.gather_rows(np.arange(store.n_obs))
                # gather_rows returns scipy CSR, but guard the dense case so a
                # future / configured dense return doesn't AttributeError on
                # `.nnz`/`.data` (mirrors read_subset below).
                if sp.issparse(X):
                    _ = X.nnz + int(X.data.sum())
                else:
                    _ = X.shape + (int(X.sum()),)
            finally:
                store.close()

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        self._require_cellstream()

        # cellstream gathers on the row axis only; a gene projection has no native
        # pushdown, so we gather covering rows and slice columns in memory (honest
        # capability gap, mirrors the shardad runner).
        if gene_indices is None:
            mechanism = "cellstream_row_gather"
        elif cell_indices is None:
            mechanism = "cellstream_gather_all+slice_genes"
        else:
            mechanism = "cellstream_row_gather+slice_genes"

        def _read_subset() -> None:
            import cellstream

            store = cellstream.open(str(path))
            try:
                rows = (
                    np.asarray(cell_indices)
                    if cell_indices is not None
                    else np.arange(store.n_obs)
                )
                X = store.gather_rows(rows)
                if gene_indices is not None:
                    X = X[:, gene_indices]
                if sp.issparse(X):
                    _ = X.nnz + int(X.data.sum())
                else:
                    _ = X.shape + (int(X.sum()),)
            finally:
                store.close()

        _, timing = self.timed_run(_read_subset)
        timing.extra = {"mechanism": mechanism}
        return timing

    def file_size(self, path: str | Path) -> int:
        return _dir_size(str(path))
