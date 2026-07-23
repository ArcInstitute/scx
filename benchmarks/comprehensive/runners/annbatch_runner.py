"""annbatch format benchmark runner (data-load Phase 0).

Wraps ``annbatch`` (scverse/Lamin, ``~/dev/python/annbatch``). annbatch's core
value is a **one-time global pre-shuffle** of the source into a sharded zarr
``DatasetCollection``; the per-epoch ``Loader`` then does big contiguous reads
over already-shuffled data. So ``convert_from_h5ad`` here runs that pre-shuffle
(the honest "prepare your data for annbatch" step, amortized across epochs), and
the actual loader throughput is measured in ``benchmarks/ooc_loader.py``.

``capabilities`` is empty — annbatch participates only in the ``ooc_loader``
Phase-0 comparison, not the filtered-query / cloud / backed / streaming benches.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.runners.base import (
    ConvertResult,
    FormatRunner,
    TimingResult,
)

try:
    import annbatch  # noqa: F401
    from annbatch import DatasetCollection, Loader

    _HAS_ANNBATCH = True
except ImportError:
    _HAS_ANNBATCH = False


_MISSING_MSG = (
    "annbatch is not installed in this env. Install it into scx-bench with: "
    "conda activate scx-bench && pip install 'annbatch[zarrs]'"
)

# annbatch pre-shuffle knobs (report §3 / §P1.1). `dataset_size` bounds the
# in-memory shuffle buffer; `shuffle_chunk_size` is the contiguous-block
# granularity. Match annbatch's documented defaults.
_DATASET_SIZE = os.environ.get("SCX_BENCH_ANNBATCH_DATASET_SIZE", "20GB")
_SHUFFLE_CHUNK_SIZE = 1000


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


class AnnbatchRunner(FormatRunner):
    """Benchmark runner for annbatch sharded-zarr DatasetCollections."""

    capabilities: frozenset[str] = frozenset()

    @property
    def name(self) -> str:
        return "annbatch"

    @property
    def key(self) -> str:
        return "annbatch"

    @staticmethod
    def _require() -> None:
        if not _HAS_ANNBATCH:
            raise ImportError(_MISSING_MSG)

    def convert_from_h5ad(
        self, h5ad_path: str | Path, output_path: str | Path
    ) -> ConvertResult:
        """Pre-shuffle the source h5ad into a sharded zarr ``DatasetCollection``
        (``output_path`` must end in ``.zarr``)."""
        self._require()
        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        def _convert() -> None:
            DatasetCollection(output_path).add_adatas(
                [h5ad_path],
                dataset_size=_DATASET_SIZE,
                shuffle_chunk_size=_SHUFFLE_CHUNK_SIZE,
                shuffle=True,
                rng=np.random.default_rng(0),
            )

        _, timing = self.timed_run(_convert)
        output_size = _dir_size(output_path)
        throughput = 0.0
        if timing.wall_s > 0:
            throughput = (output_size / (1024 * 1024)) / timing.wall_s
        return ConvertResult(
            wall_s=timing.wall_s,
            peak_rss_mb=timing.peak_rss_mb,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
            extra={
                "dataset_size": _DATASET_SIZE,
                "shuffle_chunk_size": _SHUFFLE_CHUNK_SIZE,
            },
        )

    def read_full(self, path: str | Path) -> TimingResult:
        """Iterate the pre-shuffled collection once through a CPU ``Loader``
        (scipy CSR batches). Provided to satisfy the runner interface; the
        Phase-0 throughput signal comes from ``ooc_loader``, not here."""
        self._require()

        def _read() -> None:
            import anndata as ad

            collection = DatasetCollection(str(path), mode="r")
            loader = Loader(
                batch_size=1024,
                chunk_size=32,
                preload_nchunks=32,
                shuffle=False,
                preload_to_gpu=False,
                to=None,
            )
            with ad.settings.override(remove_unused_categories=False):
                loader = loader.use_collection(collection)
                total = 0
                for batch in loader:
                    X = batch["X"] if isinstance(batch, dict) else batch
                    total += X.shape[0]
                _ = total

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices=None,
        gene_indices=None,
    ) -> TimingResult:
        # annbatch has no random-subset API (it's a shuffled sequential loader);
        # the honest capability gap. Not exercised in Phase 0.
        raise NotImplementedError(
            "annbatch has no subset read; use ooc_loader for the loader benchmark"
        )

    def file_size(self, path: str | Path) -> int:
        return _dir_size(str(path))
