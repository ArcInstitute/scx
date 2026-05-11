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
}


def make_runner(fmt) -> FormatRunner:
    """Instantiate a FormatRunner from a FormatVariant."""
    cls = _RUNNER_MAP[fmt.runner]
    return cls(**fmt.params)
