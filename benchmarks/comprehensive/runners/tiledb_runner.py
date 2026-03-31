"""TileDB-SOMA format benchmark runner."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult

try:
    import tiledbsoma
except ImportError:
    tiledbsoma = None

_TILEDB_MISSING_MSG = (
    "tiledbsoma is not installed. "
    "Install with: pip install tiledbsoma"
)


def _require_tiledbsoma() -> None:
    if tiledbsoma is None:
        raise ImportError(_TILEDB_MISSING_MSG)


class TileDBRunner(FormatRunner):
    """Benchmark runner for TileDB-SOMA format."""

    @property
    def name(self) -> str:
        return "TileDB-SOMA"

    @property
    def key(self) -> str:
        return "tiledb_soma"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(self, h5ad_path: str | Path, output_path: str | Path) -> ConvertResult:
        _require_tiledbsoma()
        import anndata

        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        self._gc_collect()
        t0 = time.perf_counter()

        adata = anndata.read_h5ad(h5ad_path)
        tiledbsoma.io.from_anndata(
            output_path,
            adata,
            measurement_name="RNA",
        )

        wall = time.perf_counter() - t0
        rss = self._get_rss_mb()

        output_size = self._dir_size(output_path)
        throughput = (output_size / (1024 * 1024)) / wall if wall > 0 else 0.0

        return ConvertResult(
            wall_s=wall,
            peak_rss_mb=rss,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
        )

    def read_full(self, path: str | Path) -> TimingResult:
        _require_tiledbsoma()

        def _read():
            with tiledbsoma.Experiment.open(str(path)) as exp:
                query = exp.axis_query("RNA")
                adata = query.to_anndata(X_name="data")
                # Force materialization of the sparse matrix
                X = adata.X
                if hasattr(X, "toarray"):
                    X.toarray()

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        _require_tiledbsoma()

        def _read_subset():
            obs_query = tiledbsoma.AxisQuery(
                coords=(list(cell_indices),)
            ) if cell_indices is not None else tiledbsoma.AxisQuery()

            var_query = tiledbsoma.AxisQuery(
                coords=(list(gene_indices),)
            ) if gene_indices is not None else tiledbsoma.AxisQuery()

            with tiledbsoma.Experiment.open(str(path)) as exp:
                query = exp.axis_query(
                    "RNA",
                    obs_query=obs_query,
                    var_query=var_query,
                )
                adata = query.to_anndata(X_name="data")
                X = adata.X
                if hasattr(X, "toarray"):
                    X.toarray()

        _, timing = self.timed_run(_read_subset)
        return timing

    def file_size(self, path: str | Path) -> int:
        return self._dir_size(path)
