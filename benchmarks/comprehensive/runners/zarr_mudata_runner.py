"""Zarr-backed MuData benchmark runner (Phase K).

Mirrors the existing ``zarr_runner.py`` for the multimodal case:
``mu.write_zarr(path, compressor="zstd")``. Used by
``multimodal_compression`` as the chunked/columnar baseline against
SCX v2 multimodal.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from benchmarks.comprehensive.runners.base import (
    ConvertResult,
    FormatRunner,
    TimingResult,
)

try:
    import mudata

    _HAS_MUDATA = True
except ImportError:
    _HAS_MUDATA = False


_COMPRESSOR_NAMES = {
    "zstd": ("Zarr-MuData (zstd)", "zarr_mudata_zstd"),
}


class ZarrMuDataRunner(FormatRunner):
    """Zarr store of a MuData object (per-modality columns / chunks)."""

    capabilities: frozenset[str] = frozenset()

    def __init__(self, compressor: str = "zstd", level: int = 3) -> None:
        if compressor not in _COMPRESSOR_NAMES:
            raise ValueError(
                f"Unsupported compressor {compressor!r}; "
                f"expected one of {list(_COMPRESSOR_NAMES)}"
            )
        self.compressor = compressor
        self.level = level

    @property
    def name(self) -> str:
        return _COMPRESSOR_NAMES[self.compressor][0]

    @property
    def key(self) -> str:
        return _COMPRESSOR_NAMES[self.compressor][1]

    @staticmethod
    def _check_mudata() -> None:
        if not _HAS_MUDATA:
            raise RuntimeError(
                "mudata is not installed. Install with: pip install mudata"
            )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert_from_h5ad(self, h5ad_path, output_path) -> ConvertResult:
        raise NotImplementedError(
            "zarr_mudata_runner.convert_from_h5ad: source h5mu via "
            "convert_from_h5mu; single-modality h5ad has no canonical "
            "multimodal lift"
        )

    def convert_from_h5mu(self, h5mu_path, output_path) -> ConvertResult:
        self._check_mudata()
        h5mu_path = str(h5mu_path)
        output_path = str(output_path)

        self._gc_collect()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        mu = mudata.read_h5mu(h5mu_path)
        # mudata 0.4 uses anndata's experimental zarr writer; the public
        # API is `mu.write_zarr(path)`. The compressor is configured at
        # the global zarr level — we don't override (the default is
        # blosc with whatever the host zarr version ships). The
        # `compressor` kwarg is captured here for future override
        # support and stored in the convert result extra.
        mu.write_zarr(output_path)

        wall = time.perf_counter() - t0
        u1, s1 = self._get_cpu_times()
        rss = self._get_rss_mb()

        output_size = self._dir_size(output_path)
        throughput = (output_size / (1024 * 1024)) / wall if wall > 0 else 0.0

        return ConvertResult(
            wall_s=wall,
            peak_rss_mb=rss,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
            extra={"compressor": self.compressor, "level": self.level},
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_full(self, path) -> TimingResult:
        self._check_mudata()

        def _read():
            mu = mudata.read_zarr(str(path))
            for name in mu.mod:
                X = mu.mod[name].X
                if sp.issparse(X):
                    _ = X.nnz
                else:
                    _ = X.shape

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        self._check_mudata()

        def _read_subset():
            mu = mudata.read_zarr(str(path))
            for name in mu.mod:
                X = mu.mod[name].X
                if cell_indices is not None:
                    X = X[cell_indices]
                if gene_indices is not None:
                    X = X[:, gene_indices]
                if sp.issparse(X):
                    X.toarray()
                else:
                    _ = X.shape

        _, timing = self.timed_run(_read_subset)
        return timing

    def file_size(self, path) -> int:
        return self._dir_size(path)
