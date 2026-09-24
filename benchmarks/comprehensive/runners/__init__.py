# Format runners for the comprehensive benchmark suite.

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult
from benchmarks.comprehensive.runners.h5ad_runner import H5adRunner
from benchmarks.comprehensive.runners.h5mu_runner import H5muRunner
from benchmarks.comprehensive.runners.zarr_runner import ZarrRunner
from benchmarks.comprehensive.runners.zarr_mudata_runner import ZarrMuDataRunner
from benchmarks.comprehensive.runners.tiledb_runner import TileDBRunner
from benchmarks.comprehensive.runners.scx_runner import ScxRunner
from benchmarks.comprehensive.runners.bpcells_runner import BPCellsRunner
from benchmarks.comprehensive.runners.parquet_runner import ParquetRunner
from benchmarks.comprehensive.runners.slaf_runner import SlafRunner
from benchmarks.comprehensive.runners.annbatch_runner import AnnbatchRunner

__all__ = [
    "FormatRunner",
    "TimingResult",
    "ConvertResult",
    "H5adRunner",
    "H5muRunner",
    "ZarrRunner",
    "ZarrMuDataRunner",
    "TileDBRunner",
    "ScxRunner",
    "BPCellsRunner",
    "ParquetRunner",
    "SlafRunner",
    "AnnbatchRunner",
    "make_runner",
]


_RUNNER_MAP = {
    "h5ad_runner": H5adRunner,
    "h5mu_runner": H5muRunner,
    "zarr_runner": ZarrRunner,
    "zarr_mudata_runner": ZarrMuDataRunner,
    "tiledb_runner": TileDBRunner,
    "scx_runner": ScxRunner,
    "bpcells_runner": BPCellsRunner,
    "parquet_runner": ParquetRunner,
    "slaf_runner": SlafRunner,
    "annbatch_runner": AnnbatchRunner,
}


def make_runner(fmt) -> FormatRunner:
    """Instantiate a FormatRunner from a FormatVariant."""
    cls = _RUNNER_MAP[fmt.runner]
    params = dict(fmt.params)
    # G4.3: opt-in CSC sidecar write at convert time. Required for the GPU DE
    # CSC-direct path to be exercised; without it, GPU DE falls back to the
    # CSR-direct atomicAdd path on every fixture. Cheap to leave on for non-GPU
    # runs (modest convert-time overhead + larger SCX file). Off by default to
    # preserve back-compat.
    import os
    if cls is ScxRunner and os.environ.get("SCX_BENCH_WITH_CSC", "").strip() in ("1", "true", "TRUE"):
        params.setdefault("csc", "always")
    return cls(**params)
