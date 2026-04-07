"""
SCX Comprehensive Benchmark Suite — Configuration.

Central configuration for dataset paths, format definitions, benchmark constants,
and system-level settings. All benchmark scripts import from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../scx/
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
COMPREHENSIVE_DIR = BENCHMARKS_DIR / "comprehensive"
RESULTS_DIR = COMPREHENSIVE_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
REPORTS_DIR = RESULTS_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Default data directory — override via SCX_DATA_DIR env var
DATA_DIR = Path(os.environ.get(
    "SCX_DATA_DIR",
    "/scratch/ctc/nickyoungblut/scx/benchmarks/datasets",
))

# Conda environments for benchmarking (isolated from dev .venv/)
# These are created by: bash benchmarks/comprehensive/scripts/install_dependencies.sh
CONDA_ENV_CPU = "scx-bench"        # CPU benchmarks
CONDA_ENV_GPU = "scx-bench-gpu"    # GPU benchmarks (CUDA + RAPIDS)
CONDA_ENV_R = "scx-bench-r"        # R / BPCells benchmarks
ENVS_DIR = COMPREHENSIVE_DIR / "envs"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Metadata for a benchmark dataset."""
    id: str            # Short ID (e.g. "D1")
    name: str          # Filesystem stem (e.g. "pbmc3k")
    n_obs: int         # Expected cell count
    n_vars: int        # Expected gene count
    protocol: str      # e.g. "10x v2 (UMI)"
    source: str        # e.g. "10x Genomics"
    approx_h5ad_mb: int  # Approximate uncompressed h5ad size in MB
    available: bool = True  # Whether the dataset is expected to already exist

    @property
    def h5ad_path(self) -> Path:
        return DATA_DIR / f"{self.name}.h5ad"

    @property
    def h5ad_gzip_path(self) -> Path:
        return DATA_DIR / f"{self.name}_gzip.h5ad"

    @property
    def h5ad_lzf_path(self) -> Path:
        return DATA_DIR / f"{self.name}_lzf.h5ad"

    @property
    def scx_path(self) -> Path:
        return DATA_DIR / f"{self.name}.scx"

    @property
    def zarr_zstd_path(self) -> Path:
        return DATA_DIR / f"{self.name}_zstd.zarr"

    @property
    def zarr_lz4_path(self) -> Path:
        return DATA_DIR / f"{self.name}_lz4.zarr"

    @property
    def soma_path(self) -> Path:
        return DATA_DIR / f"{self.name}.soma"

    @property
    def bpcells_path(self) -> Path:
        return DATA_DIR / f"{self.name}_bpcells"

    @property
    def parquet_path(self) -> Path:
        return DATA_DIR / f"{self.name}.parquet"

    # Per-codec SCX paths for benchmark isolation
    @property
    def scx_auto_path(self) -> Path:
        return DATA_DIR / f"{self.name}_auto.scx"

    @property
    def scx_scx1_path(self) -> Path:
        return DATA_DIR / f"{self.name}_scx1.scx"

    @property
    def scx_zstd_path(self) -> Path:
        return DATA_DIR / f"{self.name}_zstd.scx"

    @property
    def scx_none_path(self) -> Path:
        return DATA_DIR / f"{self.name}_none.scx"

    @property
    def scx_lz4_path(self) -> Path:
        return DATA_DIR / f"{self.name}_lz4.scx"

    def path_for_format(self, format_key: str) -> Path:
        """Return the persistent on-disk path for a given format key."""
        prop = _FORMAT_KEY_TO_PROP.get(format_key)
        if prop is None:
            raise ValueError(f"No persistent path for format key {format_key!r}")
        return getattr(self, prop)


_FORMAT_KEY_TO_PROP: dict[str, str] = {
    "h5ad_none": "h5ad_path",
    "h5ad_gzip": "h5ad_gzip_path",
    "h5ad_lzf": "h5ad_lzf_path",
    "zarr_zstd": "zarr_zstd_path",
    "zarr_lz4": "zarr_lz4_path",
    "tiledb_soma": "soma_path",
    "scx_auto": "scx_auto_path",
    "scx_scx1": "scx_scx1_path",
    "scx_zstd": "scx_zstd_path",
    "scx_none": "scx_none_path",
    "scx_lz4": "scx_lz4_path",
    "bpcells": "bpcells_path",
    "parquet_zstd": "parquet_path",
}


DATASETS: dict[str, DatasetConfig] = {
    "pbmc3k": DatasetConfig(
        id="D1", name="pbmc3k",
        n_obs=2_700, n_vars=32_738,
        protocol="10x v2 (UMI)", source="10x Genomics",
        approx_h5ad_mb=21, available=True,
    ),
    "pbmc10k": DatasetConfig(
        id="D2", name="pbmc10k",
        n_obs=11_769, n_vars=33_538,
        protocol="10x v3 (UMI)", source="10x Genomics",
        approx_h5ad_mb=194, available=True,
    ),
    "smartseq2": DatasetConfig(
        id="D3", name="smartseq2",
        n_obs=50_000, n_vars=61_497,
        protocol="Smart-seq2", source="CELLxGENE Census",
        approx_h5ad_mb=1_070, available=True,
    ),
    "tabula_sapiens_100k": DatasetConfig(
        id="D4", name="tabula_sapiens_100k",
        n_obs=100_000, n_vars=61_497,
        protocol="10x (UMI)", source="CELLxGENE Census",
        approx_h5ad_mb=1_600, available=True,
    ),
    "census_500k": DatasetConfig(
        id="D5", name="census_500k",
        n_obs=500_000, n_vars=61_497,
        protocol="10x (UMI)", source="CELLxGENE Census (blood)",
        approx_h5ad_mb=5_700, available=True,
    ),
    "census_1m": DatasetConfig(
        id="D6", name="census_1m",
        n_obs=1_000_000, n_vars=61_497,
        protocol="10x (UMI)", source="CELLxGENE Census (blood)",
        approx_h5ad_mb=11_400, available=True,
    ),
    "census_5m": DatasetConfig(
        id="D7", name="census_5m",
        n_obs=5_000_000, n_vars=61_497,
        protocol="10x (UMI)", source="CELLxGENE Census (blood)",
        approx_h5ad_mb=86_000, available=True,
    ),
    "census_10m": DatasetConfig(
        id="D8", name="census_10m",
        n_obs=10_000_000, n_vars=61_497,
        protocol="Mixed", source="CELLxGENE Census (blood)",
        approx_h5ad_mb=176_000, available=True,
    ),
}


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

@dataclass
class FormatVariant:
    """A single format variant to benchmark."""
    name: str          # Human-readable (e.g. "h5ad (gzip)")
    key: str           # Machine key (e.g. "h5ad_gzip")
    category: str      # "primary" or "additional"
    runner: str        # Runner module name (e.g. "h5ad_runner")
    params: dict[str, Any] = field(default_factory=dict)


PRIMARY_FORMATS: list[FormatVariant] = [
    FormatVariant("h5ad (uncompressed)", "h5ad_none", "primary", "h5ad_runner",
                  {"compression": None}),
    FormatVariant("h5ad (gzip)", "h5ad_gzip", "primary", "h5ad_runner",
                  {"compression": "gzip"}),
    FormatVariant("h5ad (lzf)", "h5ad_lzf", "primary", "h5ad_runner",
                  {"compression": "lzf"}),
    FormatVariant("Zarr (zstd)", "zarr_zstd", "primary", "zarr_runner",
                  {"compressor": "zstd", "level": 3}),
    FormatVariant("Zarr (blosc-lz4)", "zarr_lz4", "primary", "zarr_runner",
                  {"compressor": "lz4", "level": 5}),
    FormatVariant("TileDB-SOMA", "tiledb_soma", "primary", "tiledb_runner"),
    FormatVariant("SCX (auto)", "scx_auto", "primary", "scx_runner",
                  {"codec": "auto"}),
    FormatVariant("SCX (none)", "scx_none", "primary", "scx_runner",
                  {"codec": "none"}),
    FormatVariant("SCX (scx1)", "scx_scx1", "primary", "scx_runner",
                  {"codec": "scx1"}),
    FormatVariant("SCX (zstd)", "scx_zstd", "primary", "scx_runner",
                  {"codec": "zstd"}),
    FormatVariant("SCX (lz4)", "scx_lz4", "primary", "scx_runner",
                  {"codec": "lz4"}),
]

ADDITIONAL_FORMATS: list[FormatVariant] = [
    FormatVariant("BPCells", "bpcells", "additional", "bpcells_runner"),
    FormatVariant("Parquet (zstd)", "parquet_zstd", "additional", "parquet_runner",
                  {"compression": "zstd"}),
    FormatVariant("AnnData-on-Zarr (backed)", "anndata_zarr_backed", "additional",
                  "zarr_runner", {"backed": True}),
]

ALL_FORMATS = PRIMARY_FORMATS + ADDITIONAL_FORMATS


def get_formats(include_additional: bool = False) -> list[FormatVariant]:
    """Return the list of format variants to benchmark."""
    if include_additional:
        return ALL_FORMATS
    return PRIMARY_FORMATS


# ---------------------------------------------------------------------------
# Benchmark Constants
# ---------------------------------------------------------------------------

# Number of repetitions
N_RUNS_LARGE = 3      # For datasets >= 100K cells
N_RUNS_SMALL = 5      # For datasets < 100K cells
N_WARMUP_RUNS = 1     # Discarded warm-up iterations

# Row-slice query: number of random cells to select
QUERY_N_CELLS = 1_000

# Column projection: number of HVGs to project onto
QUERY_N_HVGS = 2_000

# Random seed for reproducible subsetting
RANDOM_SEED = 42

# Thread counts for parallel scaling benchmarks
THREAD_COUNTS = [1, 2, 4, 8, 16, 32]

# ML loader config
ML_BATCH_SIZE = 1024
ML_MAX_BATCHES = None  # None = iterate full dataset

# RSS sampling interval (ms) for memory time-series
RSS_SAMPLE_INTERVAL_MS = 100

# Accelerator benchmarks
PCA_N_COMPS = [20, 50, 100]
KNN_N_NEIGHBORS = [5, 15, 30]
UMAP_N_EPOCHS = [200, 500]
DE_N_GROUPS = [10, 25, 50]

# Cache configurations for backed-mode benchmarks
BACKED_CACHE_SHARDS = [0, 4, 16, 64]


def n_runs_for_dataset(dataset_name: str) -> int:
    """Return the number of benchmark runs for a given dataset."""
    cfg = DATASETS.get(dataset_name)
    if cfg and cfg.n_obs >= 100_000:
        return N_RUNS_LARGE
    return N_RUNS_SMALL


# ---------------------------------------------------------------------------
# SLURM Defaults
# ---------------------------------------------------------------------------

SLURM_DEFAULTS = {
    "cpu": {
        "partition": "cpu",
        "cpus_per_task": 16,
        "mem_gb": 80,
        "time": "04:00:00",
    },
    "cpu_high_mem": {
        "partition": "cpu_high_mem",
        "cpus_per_task": 16,
        "mem_gb": 500,
        "time": "08:00:00",
    },
    "gpu": {
        "partition": "gpu",
        "cpus_per_task": 16,
        "mem_gb": 128,
        "gpus": 1,
        "time": "04:00:00",
    },
}
