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
    ConvertResult,
    FormatRunner,
    TimingResult,
)

if TYPE_CHECKING:
    from benchmarks.comprehensive.queries import Predicate

try:
    import slaf  # noqa: F401
    from slaf import SLAFArray

    _HAS_SLAF = True
except ImportError:
    _HAS_SLAF = False


_MISSING_MSG = (
    "slafdb is not installed. Create the SLAF env with: "
    "conda env create -f benchmarks/comprehensive/envs/scx-bench-slaf.yml"
)


class SlafRunner(FormatRunner):
    """Benchmark runner for SLAF directories."""

    capabilities: frozenset[str] = frozenset({"filtered_query"})

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

            # Convert to python lists for IN clause building.
            cell_list = (
                None if cell_indices is None
                else np.asarray(cell_indices, dtype=np.int64).tolist()
            )
            gene_list = (
                None if gene_indices is None
                else np.asarray(gene_indices, dtype=np.int64).tolist()
            )

            # SLAF's high-level ``get_submatrix`` API returns (cell_id,
            # gene_id, value) where cell_id/gene_id are *strings* that
            # may not be unique across the dataset (census_1m has
            # cell_ids that repeat across chunks), making the long-form
            # ambiguous. Go directly through the SQL engine: the
            # ``expression`` table stores ``cell_integer_id`` /
            # ``gene_integer_id`` which are globally unique.
            where_parts: list[str] = []
            if cell_list is not None:
                where_parts.append(
                    "cell_integer_id IN ("
                    + ",".join(str(int(c)) for c in cell_list) + ")"
                )
            if gene_list is not None:
                where_parts.append(
                    "gene_integer_id IN ("
                    + ",".join(str(int(g)) for g in gene_list) + ")"
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

            if cell_list is not None:
                cell_lookup = {c: i for i, c in enumerate(cell_list)}
                row = np.fromiter(
                    (cell_lookup[int(c)] for c in cell_ids),
                    dtype=np.int64, count=len(cell_ids),
                )
                n_rows = len(cell_list)
            else:
                row = cell_ids.astype(np.int64, copy=False)
                n_rows = slaf_array.shape[0]
            if gene_list is not None:
                gene_lookup = {g: i for i, g in enumerate(gene_list)}
                col = np.fromiter(
                    (gene_lookup[int(g)] for g in gene_ids),
                    dtype=np.int64, count=len(gene_ids),
                )
                n_cols = len(gene_list)
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
        from benchmarks.comprehensive.queries import (
            EqPredicate,
            GtPredicate,
            RandomSamplePredicate,
        )

        def _filtered() -> None:
            slaf_array = self._open_array(path)
            # Resolve the predicate to a set of cell_integer_ids in one
            # SQL roundtrip, then materialize the matching expression
            # records with a second SQL query joined on that set. This
            # mirrors the ``read_subset`` path and sidesteps SLAF's
            # ``get_submatrix`` (which doesn't accept string cell_id
            # lists).
            if isinstance(predicate, EqPredicate):
                value_literal = (
                    f"'{predicate.value}'"
                    if isinstance(predicate.value, str)
                    else str(predicate.value)
                )
                sql_cells = (
                    "SELECT cell_integer_id FROM cells "
                    f"WHERE {predicate.column} = {value_literal}"
                )
            elif isinstance(predicate, GtPredicate):
                sql_cells = (
                    "SELECT cell_integer_id FROM cells "
                    f"WHERE {predicate.column} > {predicate.threshold}"
                )
            elif isinstance(predicate, RandomSamplePredicate):
                # Polars-SQL doesn't expose DuckDB's SAMPLE clause via
                # ``SLAFArray.query``; emulate with a deterministic hash
                # filter on cell_integer_id.
                modulus = max(int(1 / predicate.fraction), 2)
                # (seed * 2654435761) mod 2^32 — Knuth multiplicative
                # constant; combines seed into the bucket selection.
                target = (predicate.seed * 2654435761) % modulus
                sql_cells = (
                    "SELECT cell_integer_id FROM cells "
                    f"WHERE (cell_integer_id % {modulus}) = {target}"
                )
            else:  # pragma: no cover — exhaustive
                raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")

            cell_rows = slaf_array.query(sql_cells)
            ids = cell_rows["cell_integer_id"].to_list()
            if not ids:
                return  # No matching cells — empty read

            # Materialize the expression slice (long form).
            ids_csv = ",".join(str(int(i)) for i in ids)
            sql_expr = (
                "SELECT cell_integer_id, gene_integer_id, value "
                f"FROM expression WHERE cell_integer_id IN ({ids_csv})"
            )
            _ = slaf_array.query(sql_expr)

        _, timing = self.timed_run(_filtered)
        timing.extra = {
            "native_mechanism": "slaf_sql",
            "predicate": predicate.name,
        }
        return timing
