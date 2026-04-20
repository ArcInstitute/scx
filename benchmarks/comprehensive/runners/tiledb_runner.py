"""TileDB-SOMA format benchmark runner."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult

try:
    import tiledbsoma
    import tiledbsoma.io  # noqa: F401 — ensure submodule is loaded
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

    capabilities: frozenset[str] = frozenset({"filtered_query"})

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
                # Access X to force materialization into memory
                _ = adata.X

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

    # ------------------------------------------------------------------
    # Filtered query via TileDB-SOMA ``AxisQuery.value_filter``
    # ------------------------------------------------------------------

    def read_filtered_query(
        self,
        path: str | Path,
        predicate,
    ) -> TimingResult:
        _require_tiledbsoma()
        from benchmarks.comprehensive.queries import (
            EqPredicate,
            GtPredicate,
            RandomSamplePredicate,
        )

        if isinstance(predicate, EqPredicate):
            if isinstance(predicate.value, str):
                value_filter = f"{predicate.column} == '{predicate.value}'"
            else:
                value_filter = f"{predicate.column} == {predicate.value}"
            mechanism = "tiledb_value_filter"
        elif isinstance(predicate, GtPredicate):
            value_filter = f"{predicate.column} > {predicate.threshold}"
            mechanism = "tiledb_value_filter"
        elif isinstance(predicate, RandomSamplePredicate):
            # SOMA has no sampling predicate; fall back to coord selection.
            value_filter = None
            mechanism = "tiledb_random_coords"
        else:
            raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")

        def _filtered():
            with tiledbsoma.Experiment.open(str(path)) as exp:
                if value_filter is not None:
                    obs_query = tiledbsoma.AxisQuery(value_filter=value_filter)
                else:
                    import numpy as np

                    assert isinstance(predicate, RandomSamplePredicate)
                    # Select coords: we need the obs domain to sample from.
                    obs_df = exp.obs.read().concat().to_pandas()
                    n_obs = len(obs_df)
                    rng = np.random.default_rng(predicate.seed)
                    n_take = max(1, int(n_obs * predicate.fraction))
                    sel = np.sort(rng.choice(n_obs, size=n_take, replace=False))
                    obs_query = tiledbsoma.AxisQuery(coords=(sel.tolist(),))

                query = exp.axis_query("RNA", obs_query=obs_query)
                adata = query.to_anndata(X_name="data")
                X = adata.X
                if hasattr(X, "toarray"):
                    X.toarray()

        _, timing = self.timed_run(_filtered)
        timing.extra = {
            "native_mechanism": mechanism,
            "predicate": predicate.name,
        }
        return timing
