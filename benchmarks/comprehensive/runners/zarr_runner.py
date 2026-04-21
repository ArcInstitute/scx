"""
Zarr format runner for the comprehensive benchmark suite.

Supports two compressor variants:
  - zstd: Zarr with Zstandard compression
  - lz4:  Zarr with Blosc-LZ4 compression

When ``backed=True`` the runner treats the dataset as AnnData-on-Zarr
(``anndata.write_zarr`` / ``anndata.read_zarr(..., backed="r")``) rather
than the raw-CSR triple-array layout used by the primary variants. This
powers the ``anndata_zarr_backed`` FormatVariant.
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
    backed : bool
        When True, use the AnnData-on-Zarr layout (``anndata.write_zarr`` /
        ``anndata.read_zarr(..., backed="r")``). The raw-CSR triple-array
        layout used by the primary ``zarr_zstd`` / ``zarr_lz4`` variants
        doesn't preserve obs/var, so this is a separate on-disk layout.
    """

    _VALID_COMPRESSORS = ("zstd", "lz4")

    def __init__(
        self,
        compressor: str = "zstd",
        level: int = 3,
        backed: bool = False,
    ) -> None:
        if compressor not in self._VALID_COMPRESSORS:
            raise ValueError(
                f"compressor must be one of {self._VALID_COMPRESSORS}, got {compressor!r}"
            )
        self.compressor = compressor
        self.level = level
        self.backed = backed
        self.capabilities = (
            frozenset({"cloud_read", "cloud_subset", "backed_mode"})
            if backed
            else frozenset({"cloud_read", "cloud_subset"})
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        if self.backed:
            return "AnnData-on-Zarr (backed)"
        if self.compressor == "zstd":
            return "Zarr (zstd)"
        return "Zarr (blosc-lz4)"

    @property
    def key(self) -> str:
        if self.backed:
            return "anndata_zarr_backed"
        if self.compressor == "zstd":
            return "zarr_zstd"
        return "zarr_lz4"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_compressors(self) -> tuple:
        """Return a tuple of zarr v3 codec instances."""
        from zarr.codecs import BloscCodec, ZstdCodec

        if self.compressor == "zstd":
            return (ZstdCodec(level=self.level),)
        # lz4 via Blosc
        return (BloscCodec(cname="lz4", clevel=self.level),)

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

        compressors = self._make_compressors()
        backed = self.backed

        def _convert_backed() -> None:
            adata = anndata.read_h5ad(str(h5ad_path))
            adata.write_zarr(str(output_path))

        def _convert_raw() -> None:
            adata = anndata.read_h5ad(str(h5ad_path))
            X = adata.X

            if sp.issparse(X):
                if not sp.isspmatrix_csr(X):
                    X = X.tocsr()
            else:
                X = sp.csr_matrix(X)

            shape = X.shape
            indptr = np.asarray(X.indptr)
            indices = np.asarray(X.indices)
            data = np.asarray(X.data)

            store = zarr.open(str(output_path), mode="w")
            store.create_array(
                "indptr", data=indptr, chunks=(len(indptr),), compressors=compressors
            )
            store.create_array(
                "indices", data=indices, chunks=(min(len(indices), 1 << 20),), compressors=compressors
            )
            store.create_array(
                "data", data=data, chunks=(min(len(data), 1 << 20),), compressors=compressors
            )
            store.attrs["shape"] = list(shape)

        _, timing = self.timed_run(_convert_backed if backed else _convert_raw)

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

        if self.backed:
            import anndata

            def _read_backed():
                adata = anndata.read_zarr(str(path))
                _ = adata.X

            _, timing = self.timed_run(_read_backed)
            timing.extra = {"mode": "backed"}
            return timing

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

        if self.backed:
            import anndata

            def _subset_backed():
                adata = anndata.read_zarr(str(path))
                X = adata.X
                if cell_indices is not None:
                    X = X[np.asarray(cell_indices, dtype=np.intp)]
                if gene_indices is not None:
                    X = X[:, np.asarray(gene_indices, dtype=np.intp)]
                if hasattr(X, "toarray"):
                    X.toarray()

            _, timing = self.timed_run(_subset_backed)
            timing.extra = {"mode": "backed"}
            return timing

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

    # ------------------------------------------------------------------
    # Cloud operations
    # ------------------------------------------------------------------

    @staticmethod
    def _open_cloud_store(cloud_url: str) -> tuple[Any, bool]:
        """Open a cloud zarr store, preferring consolidated metadata.

        Returns ``(store, consolidated)``. Falls back to the unconsolidated
        open on ``FileNotFoundError`` / ``KeyError`` / ``ValueError`` — the
        three exception classes zarr-python 3.x raises when the
        ``zarr.json`` consolidated blob is absent.
        """
        import zarr

        try:
            store = zarr.open_consolidated(cloud_url, mode="r")
            return store, True
        except (FileNotFoundError, KeyError, ValueError):
            return zarr.open(cloud_url, mode="r"), False

    def read_cloud(self, cloud_url: str) -> TimingResult:
        if self.backed:
            import anndata

            consolidated_holder: dict = {"value": None}

            def _read_backed_cloud():
                # anndata.read_zarr routes its own consolidated-metadata
                # probe through the underlying zarr store; we surface the
                # flag by probing once up front so the ``extra`` dict is
                # accurate. The probe is cheap (single GET).
                _, consolidated = self._open_cloud_store(cloud_url)
                consolidated_holder["value"] = consolidated
                adata = anndata.read_zarr(cloud_url)
                _ = adata.X

            _, timing = self.timed_run(_read_backed_cloud)
            timing.extra = {
                "provider": "gcs",
                "native_mechanism": "anndata_zarr_cloud",
                "consolidated_metadata": bool(consolidated_holder["value"]),
                "mode": "backed",
                "telemetry": "phase_f_deferred",
            }
            return timing

        import scipy.sparse as sp

        consolidated_holder: dict = {"value": False}

        def _read() -> sp.csr_matrix:
            store, consolidated = self._open_cloud_store(cloud_url)
            consolidated_holder["value"] = consolidated
            indptr = store["indptr"][:]
            indices = store["indices"][:]
            data = store["data"][:]
            shape = tuple(store.attrs["shape"])
            return sp.csr_matrix((data, indices, indptr), shape=shape)

        _, timing = self.timed_run(_read)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "zarr_fsspec",
            "consolidated_metadata": bool(consolidated_holder["value"]),
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_subset(
        self,
        cloud_url: str,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        if self.backed:
            import anndata

            consolidated_holder: dict = {"value": None}

            def _subset_backed_cloud():
                _, consolidated = self._open_cloud_store(cloud_url)
                consolidated_holder["value"] = consolidated
                adata = anndata.read_zarr(cloud_url)
                X = adata.X
                if cell_indices is not None:
                    X = X[np.asarray(cell_indices, dtype=np.intp)]
                if gene_indices is not None:
                    X = X[:, np.asarray(gene_indices, dtype=np.intp)]
                if hasattr(X, "toarray"):
                    X.toarray()

            _, timing = self.timed_run(_subset_backed_cloud)
            timing.extra = {
                "provider": "gcs",
                "native_mechanism": "anndata_zarr_cloud",
                "consolidated_metadata": bool(consolidated_holder["value"]),
                "mode": "backed",
                "telemetry": "phase_f_deferred",
            }
            return timing

        import scipy.sparse as sp

        consolidated_holder: dict = {"value": False}

        def _subset() -> sp.csr_matrix:
            store, consolidated = self._open_cloud_store(cloud_url)
            consolidated_holder["value"] = consolidated
            indptr = store["indptr"][:]
            shape = tuple(store.attrs["shape"])
            all_indices = store["indices"][:]
            all_data = store["data"][:]
            mat = sp.csr_matrix(
                (all_data, all_indices, indptr), shape=shape
            )
            if cell_indices is not None:
                mat = mat[np.asarray(cell_indices, dtype=np.intp)]
            if gene_indices is not None:
                mat = mat[:, np.asarray(gene_indices, dtype=np.intp)]
            return mat

        _, timing = self.timed_run(_subset)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "zarr_fsspec",
            "consolidated_metadata": bool(consolidated_holder["value"]),
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_metadata(self, cloud_url: str) -> TimingResult:
        """Open the store and touch metadata only — no array reads.

        Prefers ``zarr.open_consolidated`` so a single GET fetches the
        consolidated ``zarr.json`` metadata blob. Falls back to plain
        ``zarr.open`` for legacy layouts. The returned ``extra`` records
        ``consolidated_metadata=True|False`` so reports can distinguish
        fast (single-GET) vs slow (per-array GET) cloud metadata opens.
        """
        consolidated_holder: dict = {"value": False}

        def _open():
            store, consolidated = self._open_cloud_store(cloud_url)
            consolidated_holder["value"] = consolidated
            if self.backed:
                # anndata-on-zarr stores shape/obs/var under nested groups
                _ = store.attrs.asdict() if hasattr(store.attrs, "asdict") else dict(store.attrs)
            else:
                _ = tuple(store.attrs.get("shape", ()))

        _, timing = self.timed_run(_open)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": (
                "anndata_zarr_open" if self.backed else "zarr_open"
            ),
            "consolidated_metadata": bool(consolidated_holder["value"]),
            "mode": "backed" if self.backed else "direct",
            "telemetry": "phase_f_deferred",
        }
        return timing
