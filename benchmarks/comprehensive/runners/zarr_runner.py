"""
Zarr format runner for the comprehensive benchmark suite.

Supports two compressor variants:
  - zstd: Zarr with Zstandard compression
  - lz4:  Zarr with Blosc-LZ4 compression

Uses numcodecs for compressor configuration (stable across zarr v2/v3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.runners.base import (
    ConvertResult,
    FormatRunner,
    TimingResult,
)


class ZarrRunner(FormatRunner):
    """Benchmark runner for Zarr stores with configurable compression.

    Parameters
    ----------
    compressor : str
        Either ``"zstd"`` or ``"lz4"`` (Blosc-LZ4).
    level : int
        Compression level passed to the underlying codec.
    """

    _VALID_COMPRESSORS = ("zstd", "lz4")

    def __init__(self, compressor: str = "zstd", level: int = 3) -> None:
        if compressor not in self._VALID_COMPRESSORS:
            raise ValueError(
                f"compressor must be one of {self._VALID_COMPRESSORS}, got {compressor!r}"
            )
        self.compressor = compressor
        self.level = level

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        if self.compressor == "zstd":
            return "Zarr (zstd)"
        return "Zarr (blosc-lz4)"

    @property
    def key(self) -> str:
        if self.compressor == "zstd":
            return "zarr_zstd"
        return "zarr_lz4"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_compressor(self) -> Any:
        """Return a numcodecs compressor instance."""
        import numcodecs

        if self.compressor == "zstd":
            return numcodecs.Zstd(level=self.level)
        # lz4 via Blosc
        return numcodecs.Blosc(cname="lz4", clevel=self.level)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(
        self, h5ad_path: str | Path, output_path: str | Path
    ) -> ConvertResult:
        import anndata
        import scipy.sparse as sp
        import zarr

        h5ad_path = Path(h5ad_path)
        output_path = Path(output_path)

        compressor = self._make_compressor()

        def _convert() -> None:
            adata = anndata.read_h5ad(str(h5ad_path))
            X = adata.X

            # Ensure CSR format
            if sp.issparse(X):
                if not sp.isspmatrix_csr(X):
                    X = X.tocsr()
            else:
                X = sp.csr_matrix(X)

            shape = X.shape
            indptr = np.asarray(X.indptr)
            indices = np.asarray(X.indices)
            data = np.asarray(X.data)

            # Write to zarr store (v2 API via numcodecs compressors)
            store = zarr.open(str(output_path), mode="w")
            store.create_dataset(
                "indptr", data=indptr, chunks=(len(indptr),), compressor=compressor
            )
            store.create_dataset(
                "indices", data=indices, chunks=(min(len(indices), 1 << 20),), compressor=compressor
            )
            store.create_dataset(
                "data", data=data, chunks=(min(len(data), 1 << 20),), compressor=compressor
            )
            store.attrs["shape"] = list(shape)

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
        )

    def read_full(self, path: str | Path) -> TimingResult:
        import scipy.sparse as sp
        import zarr

        path = Path(path)

        def _read() -> sp.csr_matrix:
            store = zarr.open(str(path), mode="r")
            indptr = store["indptr"][:]
            indices = store["indices"][:]
            data = store["data"][:]
            shape = tuple(store.attrs["shape"])
            return sp.csr_matrix((data, indices, indptr), shape=shape)

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        import scipy.sparse as sp
        import zarr

        path = Path(path)

        def _read_subset() -> sp.csr_matrix:
            store = zarr.open(str(path), mode="r")
            indptr = store["indptr"][:]
            shape = tuple(store.attrs["shape"])

            if cell_indices is not None:
                cell_idx = np.asarray(cell_indices, dtype=np.intp)
                # For each selected cell, gather the corresponding index/data
                # ranges from the CSR indptr.
                new_indptr = np.empty(len(cell_idx) + 1, dtype=indptr.dtype)
                new_indptr[0] = 0

                # Compute the lengths per selected row
                lengths = indptr[cell_idx + 1] - indptr[cell_idx]
                np.cumsum(lengths, out=new_indptr[1:])

                total_nnz = int(new_indptr[-1])
                new_indices = np.empty(total_nnz, dtype=np.int32)
                new_data = np.empty(total_nnz, dtype=np.float32)

                # Read full indices/data arrays (Zarr doesn't support
                # efficient gather of many small slices).
                all_indices = store["indices"][:]
                all_data = store["data"][:]

                offset = 0
                for i, ci in enumerate(cell_idx):
                    start = int(indptr[ci])
                    end = int(indptr[ci + 1])
                    length = end - start
                    new_indices[offset : offset + length] = all_indices[start:end]
                    new_data[offset : offset + length] = all_data[start:end]
                    offset += length

                n_rows = len(cell_idx)
                n_cols = shape[1]
                mat = sp.csr_matrix(
                    (new_data, new_indices, new_indptr), shape=(n_rows, n_cols)
                )
            else:
                indices_arr = store["indices"][:]
                data_arr = store["data"][:]
                mat = sp.csr_matrix((data_arr, indices_arr, indptr), shape=shape)

            if gene_indices is not None:
                gene_idx = np.asarray(gene_indices, dtype=np.intp)
                mat = mat[:, gene_idx]

            return mat

        _, timing = self.timed_run(_read_subset)
        return timing

    def file_size(self, path: str | Path) -> int:
        return self._dir_size(path)
