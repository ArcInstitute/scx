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
        self._require_slaf()
        from slaf.integrations.anndata import read_slaf

        def _read_subset() -> None:
            lazy_adata = read_slaf(str(path))

            cell_sel: Any = slice(None) if cell_indices is None else np.asarray(
                cell_indices, dtype=np.int64
            )
            gene_sel: Any = slice(None) if gene_indices is None else np.asarray(
                gene_indices, dtype=np.int64
            )

            # LazyAnnData.__getitem__ takes a (rows, cols) tuple and returns a
            # new LazyAnnData; compute() materializes a real sc.AnnData.
            sliced = lazy_adata[cell_sel, gene_sel]
            adata = sliced.compute()
            _ = adata.X

        _, timing = self.timed_run(_read_subset)
        timing.extra = {"query_approach": "lazy_slice+compute"}
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
            # Predicate → SQL WHERE clause
            if isinstance(predicate, EqPredicate):
                value_literal = (
                    f"'{predicate.value}'"
                    if isinstance(predicate.value, str)
                    else str(predicate.value)
                )
                sql = (
                    f"SELECT cell_id FROM cells "
                    f"WHERE {predicate.column} = {value_literal}"
                )
            elif isinstance(predicate, GtPredicate):
                sql = (
                    f"SELECT cell_id FROM cells "
                    f"WHERE {predicate.column} > {predicate.threshold}"
                )
            elif isinstance(predicate, RandomSamplePredicate):
                # DuckDB's SAMPLE clause is deterministic under REPEATABLE.
                pct = predicate.fraction * 100.0
                sql = (
                    f"SELECT cell_id FROM cells "
                    f"USING SAMPLE {pct} PERCENT (bernoulli, {predicate.seed})"
                )
            else:  # pragma: no cover — exhaustive
                raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")

            cell_ids_df = slaf_array.query(sql)
            # polars DataFrame → python list for get_submatrix
            cell_ids = cell_ids_df["cell_id"].to_list()
            # get_submatrix returns a Polars DataFrame of (cell_id, gene_id,
            # value) long-form; materializing it is the comparable "read"
            # cost for SLAF's pushdown path.
            _ = slaf_array.get_submatrix(cell_selector=cell_ids)

        _, timing = self.timed_run(_filtered)
        timing.extra = {
            "native_mechanism": "slaf_sql",
            "predicate": predicate.name,
        }
        return timing
