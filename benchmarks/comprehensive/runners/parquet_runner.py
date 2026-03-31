"""Parquet format benchmark runner (CSR arrays stored as Parquet columns)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


_COMPRESSION_NAMES = {
    "zstd": ("Parquet (zstd)", "parquet_zstd"),
    "snappy": ("Parquet (snappy)", "parquet_snappy"),
    "gzip": ("Parquet (gzip)", "parquet_gzip"),
    "none": ("Parquet (uncompressed)", "parquet_none"),
}


class ParquetRunner(FormatRunner):

    def __init__(self, compression: str = "zstd") -> None:
        if not _HAS_PYARROW:
            raise ImportError(
                "pyarrow is required for ParquetRunner. "
                "Install it with: pip install pyarrow"
            )
        if compression not in _COMPRESSION_NAMES:
            raise ValueError(
                f"Unsupported compression {compression!r}; "
                f"expected one of {list(_COMPRESSION_NAMES)}"
            )
        self.compression = compression

    @property
    def name(self) -> str:
        return _COMPRESSION_NAMES[self.compression][0]

    @property
    def key(self) -> str:
        return _COMPRESSION_NAMES[self.compression][1]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(self, h5ad_path: str | Path, output_path: str | Path) -> ConvertResult:
        import anndata
        import scipy.sparse as sp

        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        self._gc_collect()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        adata = anndata.read_h5ad(h5ad_path)
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        elif not sp.isspmatrix_csr(X):
            X = X.tocsr()

        # Build arrow table from CSR components
        table = pa.table({
            "indptr": pa.array(X.indptr),
            "indices": pa.array(X.indices),
            "data": pa.array(X.data),
        })

        # Store matrix shape in file metadata
        metadata = {
            "n_rows": str(X.shape[0]),
            "n_cols": str(X.shape[1]),
        }
        # Merge with any existing schema metadata
        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, **{k.encode(): v.encode() for k, v in metadata.items()}}
        table = table.replace_schema_metadata(merged_meta)

        compression_arg = self.compression if self.compression != "none" else None
        pq.write_table(table, output_path, compression=compression_arg)

        wall = time.perf_counter() - t0
        u1, s1 = self._get_cpu_times()
        rss = self._get_rss_mb()

        output_size = os.path.getsize(output_path)
        throughput = (output_size / (1024 * 1024)) / wall if wall > 0 else 0.0

        return ConvertResult(
            wall_s=wall,
            peak_rss_mb=rss,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
        )

    def _read_csr(self, path: str | Path):
        """Read a Parquet file and reconstruct the CSR matrix."""
        import scipy.sparse as sp

        table = pq.read_table(str(path))

        # Extract shape from file metadata
        meta = table.schema.metadata or {}
        n_rows = int(meta[b"n_rows"])
        n_cols = int(meta[b"n_cols"])

        indptr = table.column("indptr").to_numpy()
        indices = table.column("indices").to_numpy()
        data = table.column("data").to_numpy()

        csr = sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
        return csr

    def read_full(self, path: str | Path) -> TimingResult:
        def _read():
            csr = self._read_csr(path)
            # Force materialization
            csr.toarray()

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        def _read_subset():
            csr = self._read_csr(path)
            X = csr
            if cell_indices is not None:
                X = X[cell_indices]
            if gene_indices is not None:
                X = X[:, gene_indices]
            # Force materialization
            X.toarray()

        _, timing = self.timed_run(_read_subset)
        return timing

    def file_size(self, path: str | Path) -> int:
        return os.path.getsize(path)
