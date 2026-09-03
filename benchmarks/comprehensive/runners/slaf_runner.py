"""SLAF (slafdb) format benchmark runner.

Wraps the `slafdb` PyPI package (upstream: https://github.com/slaf-project/slaf).
SLAF stores single-cell data as Lance tables backed by a DuckDB query layer, so
selective reads fall back to SQL ``WHERE`` pushdown via ``SLAFArray.query``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.comprehensive.runners.base import (
    probe_optional,
    ConvertResult,
    FormatRunner,
    TimingResult,
)

if TYPE_CHECKING:
    from benchmarks.comprehensive.queries import Predicate

_HAS_SLAF, _MISSING_MSG = probe_optional(
    "slaf", "slaf:SLAFArray",
    install_hint=(
        "Create the SLAF env with: conda env create -f "
        "benchmarks/comprehensive/envs/scx-bench-slaf.yml"
    ),
)
if _HAS_SLAF:
    import slaf  # noqa: F401
    from slaf import SLAFArray  # noqa: F401


class SlafRunner(FormatRunner):
    """Benchmark runner for SLAF directories."""

    capabilities: frozenset[str] = frozenset({
        "filtered_query",
        "cloud_read",
        "cloud_subset",
        "cloud_filtered",
        "cloud_metadata",
    })

    @property
    def name(self) -> str:
        return "SLAF"

    @property
    def key(self) -> str:
        return "slaf"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_slaf() -> None:
        if not _HAS_SLAF:
            raise ImportError(_MISSING_MSG)

    @staticmethod
    def _open_array(path: str | Path) -> "SLAFArray":
        return SLAFArray(str(path))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(
        self, h5ad_path: str | Path, output_path: str | Path
    ) -> ConvertResult:
        self._require_slaf()
        from slaf.data import SLAFConverter

        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        def _convert() -> None:
            # chunked=True streams the h5ad through the converter without
            # holding the full AnnData in RAM — matches SLAF's documented
            # large-dataset path.
            converter = SLAFConverter(chunked=True)
            converter.convert(h5ad_path, output_path, input_format="h5ad")

        _, timing = self.timed_run(_convert)

        output_size = self._dir_size(output_path)
        throughput = 0.0
        if timing.wall_s > 0:
            throughput = (output_size / (1024 * 1024)) / timing.wall_s

        return ConvertResult(
            wall_s=timing.wall_s,
            peak_rss_mb=timing.peak_rss_mb,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
            extra={"chunked": True},
        )

    def read_full(self, path: str | Path) -> TimingResult:
        self._require_slaf()
        from slaf.integrations.anndata import read_slaf

        def _read() -> None:
            lazy_adata = read_slaf(str(path))
            adata = lazy_adata.compute()
            # Force materialization of X so the CSR conversion is included
            # in the timing — ``compute()`` already returns a scipy CSR
            # inside adata.X, but reference it to be explicit.
            _ = adata.X

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        """Subset read via ``SLAFArray.get_submatrix`` + CSR materialization.

        ``LazyAnnData[row_idx, col_idx]`` has a known issue when the row
        array is large-valued integer indices — SLAF constructs the scipy
        COO with the *original* cell_id as the row coord instead of the
        subset row position. Going through ``get_submatrix`` and
        constructing the CSR by hand side-steps that path.
        """
        self._require_slaf()
        import scipy.sparse as sp

        def _read_subset() -> None:
            slaf_array = self._open_array(path)

            # Build sorted numpy arrays for both IN-clause construction and
            # vectorized ``searchsorted`` index mapping below. ``read_selective``
            # already sorts the inputs, but sorting defensively makes the
            # method safe when called outside the harness too.
            cell_arr = (
                None if cell_indices is None
                else np.sort(np.asarray(cell_indices, dtype=np.int64))
            )
            gene_arr = (
                None if gene_indices is None
                else np.sort(np.asarray(gene_indices, dtype=np.int64))
            )

            # SLAF's high-level ``get_submatrix`` API returns (cell_id,
            # gene_id, value) where cell_id/gene_id are *strings* that
            # may not be unique across the dataset (census_1m has
            # cell_ids that repeat across chunks), making the long-form
            # ambiguous. Go directly through the SQL engine: the
            # ``expression`` table stores ``cell_integer_id`` /
            # ``gene_integer_id`` which are globally unique.
            where_parts: list[str] = []
            if cell_arr is not None:
                where_parts.append(
                    "cell_integer_id IN ("
                    + ",".join(str(int(c)) for c in cell_arr) + ")"
                )
            if gene_arr is not None:
                where_parts.append(
                    "gene_integer_id IN ("
                    + ",".join(str(int(g)) for g in gene_arr) + ")"
                )
            where_clause = (
                " WHERE " + " AND ".join(where_parts) if where_parts else ""
            )
            sql = (
                "SELECT cell_integer_id, gene_integer_id, value "
                "FROM expression" + where_clause
            )
            df = slaf_array.query(sql)

            cell_ids = df["cell_integer_id"].to_numpy()
            gene_ids = df["gene_integer_id"].to_numpy()
            values = df["value"].to_numpy()

            # Map the returned global cell_integer_id / gene_integer_id back
            # to positions in the requested subset. ``cell_arr`` / ``gene_arr``
            # are sorted, so ``np.searchsorted`` does this in vectorized C —
            # dramatically faster than a per-element dict lookup at
            # census scale.
            if cell_arr is not None:
                row = np.searchsorted(cell_arr, cell_ids).astype(
                    np.int64, copy=False
                )
                n_rows = len(cell_arr)
            else:
                row = cell_ids.astype(np.int64, copy=False)
                n_rows = slaf_array.shape[0]
            if gene_arr is not None:
                col = np.searchsorted(gene_arr, gene_ids).astype(
                    np.int64, copy=False
                )
                n_cols = len(gene_arr)
            else:
                col = gene_ids.astype(np.int64, copy=False)
                n_cols = slaf_array.shape[1]

            _ = sp.coo_matrix(
                (values, (row, col)), shape=(n_rows, n_cols)
            ).tocsr()

        _, timing = self.timed_run(_read_subset)
        timing.extra = {"query_approach": "sql_in+csr"}
        return timing

    def file_size(self, path: str | Path) -> int:
        return self._dir_size(path)

    # ------------------------------------------------------------------
    # Filtered query (SQL pushdown)
    # ------------------------------------------------------------------

    @staticmethod
    def _predicate_to_cells_sql(predicate: "Predicate") -> tuple[str, str]:
        """Return ``(cells_sql, mechanism_tag)`` for the given predicate.

        The ``cells_sql`` selects ``cell_integer_id`` for the matching rows;
        the caller then materializes the expression slice. ``stride_hash`` is
        tagged separately from ``sql`` because it's a congruence-class filter
        (every Nth cell), not a true random sample.
        """
        from benchmarks.comprehensive.queries import (
            EqPredicate,
            GtPredicate,
            RandomSamplePredicate,
            sql_literal,
        )

        if isinstance(predicate, EqPredicate):
            return (
                "SELECT cell_integer_id FROM cells "
                f"WHERE {predicate.column} = {sql_literal(predicate.value)}"
            ), "sql"
        if isinstance(predicate, GtPredicate):
            return (
                "SELECT cell_integer_id FROM cells "
                f"WHERE {predicate.column} > {predicate.threshold}"
            ), "sql"
        if isinstance(predicate, RandomSamplePredicate):
            # Polars-SQL doesn't expose DuckDB's SAMPLE clause via
            # SLAFArray.query — emulate with a stride-hash filter on
            # cell_integer_id. Selects a single congruence class
            # (every Nth cell at a fixed offset), NOT a Bernoulli sample.
            modulus = max(int(1 / predicate.fraction), 2)
            target = (predicate.seed * 2654435761) % modulus
            return (
                "SELECT cell_integer_id FROM cells "
                f"WHERE (cell_integer_id % {modulus}) = {target}"
            ), "stride_hash"
        raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")

    def _run_filtered_query(
        self,
        uri: str,
        predicate: "Predicate",
        mechanism_prefix: str,
    ) -> TimingResult:
        """Execute ``predicate`` against ``uri`` (local path or gs:// URL).

        Shared implementation for both ``read_filtered_query`` and
        ``read_cloud_filtered_query``. Resolves the predicate to a set of
        cell_integer_ids then materializes the matching expression slice —
        mirrors the ``read_subset`` path and sidesteps ``get_submatrix``
        (which rejects string cell_id lists).
        """
        sql_cells, base_tag = self._predicate_to_cells_sql(predicate)
        mechanism = f"{mechanism_prefix}{base_tag}"

        def _filtered() -> None:
            slaf_array = SLAFArray(uri)
            cell_rows = slaf_array.query(sql_cells)
            ids = cell_rows["cell_integer_id"].to_list()
            if not ids:
                return
            ids_csv = ",".join(str(int(i)) for i in ids)
            sql_expr = (
                "SELECT cell_integer_id, gene_integer_id, value "
                f"FROM expression WHERE cell_integer_id IN ({ids_csv})"
            )
            _ = slaf_array.query(sql_expr)

        _, timing = self.timed_run(_filtered)
        timing.extra = {
            "native_mechanism": mechanism,
            "predicate": predicate.name,
        }
        return timing

    def read_filtered_query(
        self,
        path: str | Path,
        predicate: "Predicate",
    ) -> TimingResult:
        """Execute ``predicate`` via a SQL ``WHERE`` against SLAF's cells table.

        The resulting cell ids feed a submatrix fetch so the timing reflects
        end-to-end pushdown, not just predicate evaluation.
        """
        self._require_slaf()
        return self._run_filtered_query(str(path), predicate, "slaf_")

    # ------------------------------------------------------------------
    # Cloud operations (Phase C — SLAF opens cloud URIs via its
    # documented cloud-backed DuckDB path; see Phase D.3 for depth)
    # ------------------------------------------------------------------

    def read_cloud(self, cloud_url: str) -> TimingResult:
        self._require_slaf()
        from slaf.integrations.anndata import read_slaf

        def _read() -> None:
            lazy_adata = read_slaf(cloud_url)
            adata = lazy_adata.compute()
            _ = adata.X

        _, timing = self.timed_run(_read)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "slaf_cloud",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_subset(
        self,
        cloud_url: str,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        self._require_slaf()
        import scipy.sparse as sp

        cell_arr = (
            None if cell_indices is None
            else np.sort(np.asarray(cell_indices, dtype=np.int64))
        )
        gene_arr = (
            None if gene_indices is None
            else np.sort(np.asarray(gene_indices, dtype=np.int64))
        )

        def _subset() -> None:
            slaf_array = SLAFArray(cloud_url)
            where_parts: list[str] = []
            if cell_arr is not None:
                where_parts.append(
                    "cell_integer_id IN ("
                    + ",".join(str(int(c)) for c in cell_arr) + ")"
                )
            if gene_arr is not None:
                where_parts.append(
                    "gene_integer_id IN ("
                    + ",".join(str(int(g)) for g in gene_arr) + ")"
                )
            where_clause = (
                " WHERE " + " AND ".join(where_parts) if where_parts else ""
            )
            sql = (
                "SELECT cell_integer_id, gene_integer_id, value "
                "FROM expression" + where_clause
            )
            df = slaf_array.query(sql)
            cell_ids = df["cell_integer_id"].to_numpy()
            gene_ids = df["gene_integer_id"].to_numpy()
            values = df["value"].to_numpy()
            if cell_arr is not None:
                row = np.searchsorted(cell_arr, cell_ids).astype(np.int64, copy=False)
                n_rows = len(cell_arr)
            else:
                row = cell_ids.astype(np.int64, copy=False)
                n_rows = slaf_array.shape[0]
            if gene_arr is not None:
                col = np.searchsorted(gene_arr, gene_ids).astype(np.int64, copy=False)
                n_cols = len(gene_arr)
            else:
                col = gene_ids.astype(np.int64, copy=False)
                n_cols = slaf_array.shape[1]
            _ = sp.coo_matrix(
                (values, (row, col)), shape=(n_rows, n_cols)
            ).tocsr()

        _, timing = self.timed_run(_subset)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "slaf_cloud_sql",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_metadata(self, cloud_url: str) -> TimingResult:
        """Open the SLAF array and touch ``shape`` only."""
        self._require_slaf()

        def _open():
            arr = SLAFArray(cloud_url)
            _ = arr.shape

        _, timing = self.timed_run(_open)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "slaf_open",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_filtered_query(
        self,
        cloud_url: str,
        predicate: "Predicate",
    ) -> TimingResult:
        """Execute ``predicate`` against a SLAF array opened from a cloud URL.

        Mirrors ``read_filtered_query`` but opens the array from the GCS URI.
        SLAF's DuckDB backend resolves object-store reads transparently —
        mechanism tag is ``slaf_cloud_sql`` / ``slaf_cloud_stride_hash`` so
        reports distinguish cloud-pushed queries from local ones.
        """
        self._require_slaf()
        timing = self._run_filtered_query(cloud_url, predicate, "slaf_cloud_")
        extra = timing.extra or {}
        extra.update({
            "provider": "gcs",
            "telemetry": "phase_f_deferred",
        })
        timing.extra = extra
        return timing
