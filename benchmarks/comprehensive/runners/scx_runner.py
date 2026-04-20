"""SCX format benchmark runner (auto, none, scx1, zstd, lz4 codec variants)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult

try:
    import pyscx

    _HAS_PYSCX = True
except ImportError:
    _HAS_PYSCX = False

try:
    import anndata

    _HAS_ANNDATA = True
except ImportError:
    _HAS_ANNDATA = False


_CODEC_NAMES = {
    "auto": ("SCX (auto)", "scx_auto"),
    "none": ("SCX (none)", "scx_none"),
    "scx1": ("SCX (scx1)", "scx_scx1"),
    "zstd": ("SCX (zstd)", "scx_zstd"),
    "lz4": ("SCX (lz4)", "scx_lz4"),
    "pcodec": ("SCX (pcodec)", "scx_pcodec"),
}


class ScxRunner(FormatRunner):
    """Benchmark runner for the SCX format with configurable codec."""

    capabilities: frozenset[str] = frozenset({"filtered_query", "backed_mode"})

    def __init__(self, codec: str = "auto") -> None:
        if codec not in _CODEC_NAMES:
            raise ValueError(
                f"Unsupported codec {codec!r}; "
                f"expected one of {list(_CODEC_NAMES)}"
            )
        self.codec = codec

    @property
    def name(self) -> str:
        return _CODEC_NAMES[self.codec][0]

    @property
    def key(self) -> str:
        return _CODEC_NAMES[self.codec][1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_pyscx() -> None:
        if not _HAS_PYSCX:
            raise RuntimeError(
                "pyscx is not installed. "
                "Install with: cd pyscx && maturin develop"
            )

    @staticmethod
    def _check_anndata() -> None:
        if not _HAS_ANNDATA:
            raise RuntimeError(
                "anndata is not installed. "
                "Install with: pip install anndata"
            )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(self, h5ad_path: str | Path, output_path: str | Path) -> ConvertResult:
        self._check_pyscx()
        self._check_anndata()

        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        self._gc_collect()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        adata = anndata.read_h5ad(h5ad_path)
        pyscx.from_anndata(adata, output_path, codec=self.codec)

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
            extra={"codec": self.codec},
        )

    def read_full(self, path: str | Path) -> TimingResult:
        self._check_pyscx()

        def _read():
            ds = pyscx.open(str(path))
            adata = ds.to_anndata()
            # Force materialization of the expression matrix
            _ = adata.X

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        self._check_pyscx()

        def _read_subset():
            ds = pyscx.open(str(path))
            if cell_indices is not None:
                # Use backed mode for arbitrary cell index subsetting
                adata = ds.to_anndata(backed=True)
                X = adata.X[cell_indices]
                if gene_indices is not None:
                    X = X[:, gene_indices]
            elif gene_indices is not None:
                # Gene-only selection via query API
                q = ds.query().select_genes(gene_indices)
                X = q.collect().to_csr()
            else:
                X = ds.query().collect().to_csr()

        _, timing = self.timed_run(_read_subset)
        timing.extra = {"query_approach": "backed_index+select_genes"}
        return timing

    def file_size(self, path: str | Path) -> int:
        return os.path.getsize(path)

    # ------------------------------------------------------------------
    # Optional: backed mode
    # ------------------------------------------------------------------

    def read_backed(self, path: str | Path) -> TimingResult:
        self._check_pyscx()

        def _open_backed():
            ds = pyscx.open(str(path))
            adata = ds.to_anndata(backed=True)
            return adata

        _, timing = self.timed_run(_open_backed)
        timing.extra = {"mode": "backed"}
        return timing

    def read_backed_slice(
        self,
        path: str | Path,
        start: int,
        count: int,
    ) -> TimingResult:
        self._check_pyscx()

        def _backed_slice():
            ds = pyscx.open(str(path))
            adata = ds.to_anndata(backed=True)
            X_slice = adata.X[start : start + count]
            return X_slice

        _, timing = self.timed_run(_backed_slice)
        timing.extra = {"mode": "backed_slice", "start": start, "count": count}
        return timing

    # ------------------------------------------------------------------
    # Filtered query via SCX catalog pushdown
    # ------------------------------------------------------------------

    def read_filtered_query(
        self,
        path: str | Path,
        predicate,
    ) -> TimingResult:
        self._check_pyscx()
        from benchmarks.comprehensive.queries import (
            EqPredicate,
            GtPredicate,
            RandomSamplePredicate,
        )

        if isinstance(predicate, EqPredicate):
            if isinstance(predicate.value, str):
                expr = f"{predicate.column} == '{predicate.value}'"
            else:
                expr = f"{predicate.column} == {predicate.value}"
        elif isinstance(predicate, GtPredicate):
            expr = f"{predicate.column} > {predicate.threshold}"
        elif isinstance(predicate, RandomSamplePredicate):
            expr = None  # handled below — SCX has no SAMPLE pushdown
        else:
            raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")

        def _filtered():
            ds = pyscx.open(str(path))
            if expr is not None:
                q = ds.query().filter_obs(expr)
                _ = q.collect().to_csr()
            else:
                # Random-sample: materialize via backed index slicing. SCX's
                # catalog pushdown doesn't support Bernoulli sampling, so
                # the honest native mechanism is a random-index read.
                import numpy as np

                assert isinstance(predicate, RandomSamplePredicate)
                n_obs = ds.n_obs
                rng = np.random.default_rng(predicate.seed)
                n_take = max(1, int(n_obs * predicate.fraction))
                cell_idx = np.sort(rng.choice(n_obs, size=n_take, replace=False))
                adata = ds.to_anndata(backed=True)
                _ = adata.X[cell_idx]

        _, timing = self.timed_run(_filtered)
        timing.extra = {
            "native_mechanism": "scx_pushdown"
            if expr is not None
            else "scx_backed_index",
            "predicate": predicate.name,
        }
        return timing
