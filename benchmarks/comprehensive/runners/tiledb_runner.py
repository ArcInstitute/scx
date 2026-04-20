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

    capabilities: frozenset[str] = frozenset({
        "filtered_query",
        "cloud_read",
        "cloud_subset",
        "cloud_filtered",
    })

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

    @staticmethod
    def _predicate_to_value_filter(predicate) -> tuple[str | None, str]:
        """Translate a ``Predicate`` into a ``(value_filter, mechanism_tag)``.

        ``value_filter`` is ``None`` for random-sample predicates, which must
        fall back to coord-based selection since SOMA exposes no sampling
        primitive. Callers run the coord fallback with the returned mechanism
        tag (``"tiledb_random_coords"``).
        """
        from benchmarks.comprehensive.queries import (
            EqPredicate,
            GtPredicate,
            RandomSamplePredicate,
            sql_literal,
        )

        if isinstance(predicate, EqPredicate):
            return (
                f"{predicate.column} == {sql_literal(predicate.value)}",
                "tiledb_value_filter",
            )
        if isinstance(predicate, GtPredicate):
            return (
                f"{predicate.column} > {predicate.threshold}",
                "tiledb_value_filter",
            )
        if isinstance(predicate, RandomSamplePredicate):
            return None, "tiledb_random_coords"
        raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")

    def _run_filtered_query(
        self,
        uri: str,
        predicate,
        mechanism_prefix: str,
    ) -> TimingResult:
        """Execute ``predicate`` against ``uri`` (local path or gs:// URL).

        Shared implementation used by both ``read_filtered_query`` and
        ``read_cloud_filtered_query`` — the only difference between those two
        is the mechanism tag (``tiledb_`` vs ``tiledb_cloud_``), so hoisting
        the body into one place keeps the two surfaces honest.
        """
        from benchmarks.comprehensive.queries import RandomSamplePredicate

        value_filter, base_mechanism = self._predicate_to_value_filter(predicate)
        mechanism = base_mechanism.replace("tiledb_", mechanism_prefix, 1)

        def _filtered():
            with tiledbsoma.Experiment.open(uri) as exp:
                if value_filter is not None:
                    obs_query = tiledbsoma.AxisQuery(value_filter=value_filter)
                else:
                    assert isinstance(predicate, RandomSamplePredicate)
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

    def read_filtered_query(
        self,
        path: str | Path,
        predicate,
    ) -> TimingResult:
        _require_tiledbsoma()
        return self._run_filtered_query(str(path), predicate, "tiledb_")

    # ------------------------------------------------------------------
    # Cloud operations (Phase C — TileDB-SOMA opens gs:// URIs natively
    # when the environment has the tiledb VFS GCS plugin available)
    # ------------------------------------------------------------------

    def read_cloud(self, cloud_url: str) -> TimingResult:
        _require_tiledbsoma()

        def _read():
            with tiledbsoma.Experiment.open(cloud_url) as exp:
                query = exp.axis_query("RNA")
                adata = query.to_anndata(X_name="data")
                _ = adata.X

        _, timing = self.timed_run(_read)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "soma_open_gs",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_subset(
        self,
        cloud_url: str,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        _require_tiledbsoma()

        def _subset():
            obs_query = tiledbsoma.AxisQuery(
                coords=(list(cell_indices),)
            ) if cell_indices is not None else tiledbsoma.AxisQuery()
            var_query = tiledbsoma.AxisQuery(
                coords=(list(gene_indices),)
            ) if gene_indices is not None else tiledbsoma.AxisQuery()
            with tiledbsoma.Experiment.open(cloud_url) as exp:
                query = exp.axis_query(
                    "RNA", obs_query=obs_query, var_query=var_query
                )
                adata = query.to_anndata(X_name="data")
                X = adata.X
                if hasattr(X, "toarray"):
                    X.toarray()

        _, timing = self.timed_run(_subset)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "soma_axis_query_gs",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_metadata(self, cloud_url: str) -> TimingResult:
        """Open the experiment and touch its obs/var counts only."""
        _require_tiledbsoma()

        def _open():
            with tiledbsoma.Experiment.open(cloud_url) as exp:
                # Touching ``exp.obs.count`` / ``exp.ms['RNA'].var.count``
                # triggers only schema / fragment metadata reads.
                _ = exp.obs.count
                _ = exp.ms["RNA"].var.count

        _, timing = self.timed_run(_open)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "soma_open_gs",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_filtered_query(
        self,
        cloud_url: str,
        predicate,
    ) -> TimingResult:
        """Execute ``predicate`` via ``AxisQuery.value_filter`` on a cloud URI.

        Opens ``tiledbsoma.Experiment.open(cloud_url)`` and pushes the
        predicate down through SOMA's native value-filter surface — same
        code path as the local variant, just with the GCS-opened handle.
        """
        _require_tiledbsoma()
        timing = self._run_filtered_query(cloud_url, predicate, "tiledb_cloud_")
        extra = timing.extra or {}
        extra.update({
            "provider": "gcs",
            "telemetry": "phase_f_deferred",
        })
        timing.extra = extra
        return timing
