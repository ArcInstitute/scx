"""H5AD format benchmark runner (uncompressed, gzip, lzf)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import anndata
import numpy as np
import scipy.sparse as sp

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult


_COMPRESSION_NAMES = {
    None: ("h5ad (uncompressed)", "h5ad_none"),
    "gzip": ("h5ad (gzip)", "h5ad_gzip"),
    "lzf": ("h5ad (lzf)", "h5ad_lzf"),
}


class H5adRunner(FormatRunner):

    capabilities: frozenset[str] = frozenset({"filtered_query"})

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

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(self, h5ad_path: str | Path, output_path: str | Path) -> ConvertResult:
        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        self._gc_collect()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        adata = anndata.read_h5ad(h5ad_path)
        adata.write_h5ad(output_path, compression=self.compression)

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

    def read_full(self, path: str | Path) -> TimingResult:
        def _read():
            adata = anndata.read_h5ad(str(path))
            # Access X to force full read into memory (CSR or dense)
            _ = adata.X

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        def _read_subset():
            adata = anndata.read_h5ad(str(path))
            X = adata.X
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

    def file_size(self, path: str | Path) -> int:
        return os.path.getsize(path)

    # ------------------------------------------------------------------
    # Filtered query — h5ad has no pushdown, load + mask is the honest path
    # ------------------------------------------------------------------

    def read_filtered_query(
        self,
        path: str | Path,
        predicate,
    ) -> TimingResult:
        from benchmarks.comprehensive.queries import (
            EqPredicate,
            GtPredicate,
            RandomSamplePredicate,
        )

        def _filtered():
            adata = anndata.read_h5ad(str(path))
            if isinstance(predicate, EqPredicate):
                mask = adata.obs[predicate.column] == predicate.value
                sub = adata[mask.values]
            elif isinstance(predicate, GtPredicate):
                mask = adata.obs[predicate.column] > predicate.threshold
                sub = adata[mask.values]
            elif isinstance(predicate, RandomSamplePredicate):
                rng = np.random.default_rng(predicate.seed)
                n_take = max(1, int(adata.n_obs * predicate.fraction))
                idx = np.sort(rng.choice(adata.n_obs, size=n_take, replace=False))
                sub = adata[idx]
            else:
                raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")
            X = sub.X
            if sp.issparse(X):
                X.toarray()
            else:
                _ = X.shape

        _, timing = self.timed_run(_filtered)
        timing.extra = {
            "native_mechanism": "h5ad_load_and_mask",
            "predicate": predicate.name,
        }
        return timing
