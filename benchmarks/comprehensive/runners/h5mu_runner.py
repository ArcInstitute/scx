"""h5mu (multimodal HDF5) format benchmark runner.

Phase K: per-modality MuData stored as a single HDF5 file under the
mudata 0.1.0 spec. Variants:

* ``h5mu (uncompressed)`` — `mu.write_h5mu(..., compression=None)`.
* ``h5mu (gzip)``         — `mu.write_h5mu(..., compression="gzip")`.

Used by ``benchmarks.comprehensive.benchmarks.multimodal_compression``
as the HDF5 baseline against SCX v2 multimodal. The convert path
sources from a `.h5mu` (not `.h5ad`); ``convert_from_h5ad`` raises
because there's no canonical 1-modality-MuData lift from h5ad without
arbitrary modality naming.
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


_COMPRESSION_NAMES = {
    None: ("h5mu (uncompressed)", "h5mu_uncompressed"),
    "gzip": ("h5mu (gzip)", "h5mu_gzip"),
}


class H5muRunner(FormatRunner):
    """Multimodal HDF5 baseline (per-modality groups under one .h5mu)."""

    capabilities: frozenset[str] = frozenset({"backed_mode"})

    def __init__(self, compression: str | None = None) -> None:
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
        # Lifting a single h5ad to a 1-modality MuData is ambiguous (no
        # canonical modality name). Multimodal_compression always
        # routes through `convert_from_h5mu`; raise here so accidental
        # passes against a single-modality dataset surface a clear
        # error instead of silently emitting a malformed h5mu.
        raise NotImplementedError(
            "h5mu_runner.convert_from_h5ad: source h5mu via convert_from_h5mu; "
            "single-modality h5ad has no canonical multimodal lift"
        )

    def convert_from_h5mu(self, h5mu_path, output_path) -> ConvertResult:
        self._check_mudata()
        h5mu_path = str(h5mu_path)
        output_path = str(output_path)

        self._gc_collect()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        # Disable HDF5 file locking — see scx_runner.convert_from_h5mu.
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        mu = mudata.read_h5mu(h5mu_path)
        mu.write_h5mu(output_path, compression=self.compression)

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
            extra={"compression": self.compression},
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_full(self, path) -> TimingResult:
        self._check_mudata()

        def _read():
            mu = mudata.read_h5mu(str(path))
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
            mu = mudata.read_h5mu(str(path))
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
        return os.path.getsize(path)
