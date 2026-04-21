"""
SCX Comprehensive Benchmark Suite — Configuration.

Central configuration for dataset paths, format definitions, benchmark constants,
and system-level settings. All benchmark scripts import from here.
"""

from __future__ import annotations

import os
import sys
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

sys.path.insert(0, str(PROJECT_ROOT / "benchmarks" / "scripts"))
from bench_env import DATA_DIR

# Expand tildes on path-valued env vars loaded from .env. python-dotenv
# doesn't do this, so `GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/scx-bench.json`
# in .env would otherwise reach downstream consumers (gcsfs, gsutil subprocess,
# pyscx's Rust cloud backend) as a literal `~` — gcsfs in particular silently
# falls back to anonymous auth when the cred file path doesn't exist, which
# surfaces as cryptic "Zstd decompression error: invalid input data" on reads
# (it's actually getting 401-HTML back, not zstd bytes). Expanding once at
# import keeps downstream consumers honest.
for _env_key in ("GOOGLE_APPLICATION_CREDENTIALS",):
    _raw = os.environ.get(_env_key, "")
    if _raw and (_raw.startswith("~") or "~" in _raw):
        os.environ[_env_key] = os.path.expanduser(_raw)

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
    synthetic: bool = False  # If True, materialized on demand by the benchmark
    synth_params: dict[str, Any] = field(default_factory=dict)

    @property
    def h5ad_path(self) -> Path:
        if self.synthetic:
            # Synthetic datasets live under a sibling 'synthetic/' tree.
            return DATA_DIR / "synthetic" / f"{self.name}.h5ad"
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

    @property
    def slaf_path(self) -> Path:
        return DATA_DIR / f"{self.name}.slaf"

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

    @property
    def scx_pcodec_path(self) -> Path:
        return DATA_DIR / f"{self.name}_pcodec.scx"

    @property
    def anndata_zarr_backed_path(self) -> Path:
        return DATA_DIR / f"{self.name}_anndata.zarr"

    def path_for_format(self, format_key: str) -> Path:
        """Return the persistent on-disk path for a given format key."""
        prop = _FORMAT_KEY_TO_PROP.get(format_key)
        if prop is None:
            raise ValueError(f"No persistent path for format key {format_key!r}")
        return getattr(self, prop)

    def cloud_url(self, format_key: str, provider: str = "gcs") -> str:
        """Return the cloud URI for this dataset in a given format.

        Phase 5 is GCP-only; ``provider`` must be ``"gcs"``. The returned URL
        points at the shared test bucket (``GCS_TEST_BUCKET``) with the
        format-appropriate suffix (``.scxd`` for SCX, ``.zarr``, ``.soma``,
        ``.slaf``). Directory-style layouts keep the trailing slash so
        callers can concatenate sub-paths without conditional logic.
        """
        if provider != "gcs":
            raise ValueError(
                f"Only 'gcs' provider is supported (got {provider!r}). "
                "AWS S3 and Azure Blob validation is deferred; see the "
                "Cloud Benchmarks section of benchmarks/README.md."
            )
        suffix = _FORMAT_KEY_TO_CLOUD_SUFFIX.get(format_key)
        if suffix is None:
            raise ValueError(
                f"No cloud layout defined for format key {format_key!r}"
            )
        return f"{GCS_TEST_BUCKET}/{self.name}{suffix}/"


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
    "scx_pcodec": "scx_pcodec_path",
    "bpcells": "bpcells_path",
    "parquet_zstd": "parquet_path",
    "slaf": "slaf_path",
    "anndata_zarr_backed": "anndata_zarr_backed_path",
}


# ---------------------------------------------------------------------------
# Cloud (GCP-only — see benchmarks/README.md "Cloud Benchmarks (GCP)")
# ---------------------------------------------------------------------------

# Shared test bucket. The comprehensive cloud benchmarks validate behavior
# against GCP only; AWS S3 and Azure Blob coverage is deferred.
GCS_TEST_BUCKET = os.environ.get(
    "GCS_TEST_BUCKET", "gs://arc-ctc-nextflow/scx-test"
).rstrip("/")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "c-tc-429521")
# Region the bucket lives in — Phase E launcher pins VM region to this so
# egress stays intra-region. Override via env if the bucket ever moves.
GCP_BUCKET_REGION = os.environ.get("GCP_BUCKET_REGION", "us-central1")

# Rolling-dashboard publish target (Phase G.4). When set, ``publish_dashboard.py``
# rsyncs the HTML snapshot tree to this destination. Empty → the publish
# script is a no-op (so CI can unconditionally invoke it). Typical values:
#   rsync-over-SSH: "user@host:/var/www/scx-bench/"
#   GCS:            "gs://my-bucket/scx-bench/" (requires gsutil, not rsync)
# The publish script picks the protocol by URL prefix.
DASHBOARD_PUBLISH_TARGET = os.environ.get("DASHBOARD_PUBLISH_TARGET", "").strip()

# ---------------------------------------------------------------------------
# GCP compute-node matrix (Phase E)
# ---------------------------------------------------------------------------

# Instance-type profiles for the cloud compute-node matrix. Egress bandwidth
# is the per-VM egress *class* published by Google (not a guaranteed floor);
# reporting uses it to contextualize measured throughput. A VM's region MUST
# match the GCS bucket region — cross-region reads silently incur egress
# charges, so ``ensure_instance_region_matches_bucket`` is called by the
# launcher before any `gcloud compute instances create`.
GCP_INSTANCE_TYPES: dict[str, dict[str, Any]] = {
    "n2-standard-8": {
        "vcpus": 8,
        "mem_gb": 32,
        "gpu": None,
        "egress_gbps": 16,
        "family": "n2",
        "notes": "general-purpose Cascade Lake / Ice Lake; baseline CPU tier",
    },
    "c3-standard-8": {
        "vcpus": 8,
        "mem_gb": 32,
        "gpu": None,
        "egress_gbps": 23,
        "family": "c3",
        "notes": "Sapphire Rapids; higher per-core bandwidth",
    },
    "a3-highgpu-1g": {
        "vcpus": 26,
        "mem_gb": 234,
        "gpu": "H100 80GB x1",
        "egress_gbps": 200,
        "family": "a3",
        "notes": "H100 GPU node; used for GPU cloud benchmarks + high-egress baseline",
    },
}


# GCS pricing (in USD) used by the cost_model benchmark. Values are pinned
# against the published GCS rate card for the Standard storage class in a
# multi-region bucket, plus standard-network egress within the same region
# (intra-region egress to GCE in the same region is $0.00/GB). Update only
# when the rate card changes — this table is the benchmark's single source
# of truth for cents-per-query math.
GCS_PRICING: dict[str, float] = {
    # Class-A operations (PUT / COPY / POST / LIST): $0.05 per 10k
    "class_a_per_10k_usd": 0.05,
    # Class-B operations (GET / HEAD): $0.004 per 10k
    "class_b_per_10k_usd": 0.004,
    # Same-region egress to GCE: $0.00 per GB — the ``ensure_instance_
    # region_matches_bucket`` gate keeps all Phase E/F runs in this regime.
    "egress_same_region_usd_per_gb": 0.00,
    # Cross-continent egress (reference only; we actively avoid this by
    # pinning the VM region to the bucket region).
    "egress_cross_region_usd_per_gb": 0.08,
    # Standard storage (reference only — storage cost dominates at
    # long-term rest, not per-query).
    "storage_standard_usd_per_gb_month": 0.026,
}


def ensure_instance_region_matches_bucket(instance_region: str) -> None:
    """Fail fast when the chosen VM region doesn't match the bucket region.

    Cross-region reads from GCS incur per-GB egress charges that silently
    dominate the benchmark budget, and they also invalidate comparisons
    (network RTT + saturation differ). The launcher calls this before any
    `gcloud compute instances create` invocation.
    """
    if instance_region != GCP_BUCKET_REGION:
        raise RuntimeError(
            f"GCP instance region {instance_region!r} does not match the "
            f"configured bucket region {GCP_BUCKET_REGION!r}. Override "
            f"GCP_BUCKET_REGION if the bucket has moved; do not run "
            f"benchmarks cross-region (silent egress charges)."
        )


# Cloud layout suffixes per format. SCX uses the exploded ``.scxd/`` layout
# on GCS (see docs/cloud.md); others keep their native directory suffix.
_FORMAT_KEY_TO_CLOUD_SUFFIX: dict[str, str] = {
    "scx_auto": ".scxd",
    "scx_scx1": ".scxd",
    "scx_zstd": ".scxd",
    "scx_none": ".scxd",
    "scx_lz4": ".scxd",
    "scx_pcodec": ".scxd",
    "zarr_zstd": ".zarr",
    # zarr_lz4 uses a codec-qualified suffix so it doesn't collide with
    # zarr_zstd on the shared `{dataset}.zarr/` cloud path — ``ensure_cloud_fixture``
    # would otherwise treat the already-uploaded zstd fixture as a cache hit for
    # the lz4 variant and skip the upload, silently producing wrong numbers
    # (cloud_read would decode zstd-compressed chunks as if they were lz4).
    "zarr_lz4": "_lz4.zarr",
    "tiledb_soma": ".soma",
    "slaf": ".slaf",
    "anndata_zarr_backed": "_anndata.zarr",
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
    # Log-normalized variants (normalize_total + log1p) for Pcodec float benchmarks
    "pbmc3k_lognorm": DatasetConfig(
        id="D1_LN", name="pbmc3k_lognorm",
        n_obs=2_700, n_vars=32_738,
        protocol="10x v2 (UMI)", source="10x Genomics (log-normalized)",
        approx_h5ad_mb=21, available=True,
    ),
    "smartseq2_lognorm": DatasetConfig(
        id="D3_LN", name="smartseq2_lognorm",
        n_obs=50_000, n_vars=61_497,
        protocol="Smart-seq2", source="CELLxGENE Census (log-normalized)",
        approx_h5ad_mb=1_070, available=True,
    ),
    "tabula_sapiens_100k_lognorm": DatasetConfig(
        id="D4_LN", name="tabula_sapiens_100k_lognorm",
        n_obs=100_000, n_vars=61_497,
        protocol="10x (UMI)", source="CELLxGENE Census (log-normalized)",
        approx_h5ad_mb=1_600, available=True,
    ),
    # Synthetic perturbation datasets for cell-eval / arc-bench parity
    # benchmarks. The perturbation obs column + gene-name matching constraint
    # (knockdown target lookup) rules out the real census datasets; these are
    # generated on first access by benchmarks.comprehensive.benchmarks._pert_synth.
    "pert_synth_10k": DatasetConfig(
        id="PS1", name="pert_synth_10k",
        n_obs=10_000, n_vars=2_000,
        protocol="synthetic (paired real/pred)",
        source="_pert_synth.make_paired_adata",
        approx_h5ad_mb=40, available=True, synthetic=True,
        synth_params={"n_obs": 10_000, "n_vars": 2_000, "n_perts": 50, "seed": 42},
    ),
    "pert_synth_100k": DatasetConfig(
        id="PS2", name="pert_synth_100k",
        n_obs=100_000, n_vars=2_000,
        protocol="synthetic (paired real/pred)",
        source="_pert_synth.make_paired_adata",
        approx_h5ad_mb=400, available=True, synthetic=True,
        synth_params={"n_obs": 100_000, "n_vars": 2_000, "n_perts": 50, "seed": 42},
    ),
    "pert_synth_500k": DatasetConfig(
        id="PS3", name="pert_synth_500k",
        n_obs=500_000, n_vars=2_000,
        protocol="synthetic (paired real/pred)",
        source="_pert_synth.make_paired_adata",
        approx_h5ad_mb=2_000, available=True, synthetic=True,
        synth_params={"n_obs": 500_000, "n_vars": 2_000, "n_perts": 50, "seed": 42},
    ),
    "pert_synth_1m": DatasetConfig(
        id="PS4", name="pert_synth_1m",
        n_obs=1_000_000, n_vars=2_000,
        protocol="synthetic (paired real/pred)",
        source="_pert_synth.make_paired_adata",
        approx_h5ad_mb=4_000, available=True, synthetic=True,
        synth_params={"n_obs": 1_000_000, "n_vars": 2_000, "n_perts": 50, "seed": 42},
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
    FormatVariant("SCX (pcodec)", "scx_pcodec", "primary", "scx_runner",
                  {"codec": "pcodec"}),
    FormatVariant("SLAF", "slaf", "primary", "slaf_runner"),
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
# Per-job memory estimation (SLURM submitit sizing)
# ---------------------------------------------------------------------------

# Above this many GB of estimated peak RSS, route the job to the high-mem
# partition. cpu_preemptible nodes can sustain up to ~200 GB per task; beyond
# that we need cpu_high_mem.
MEM_HIGH_MEM_THRESHOLD_GB = 200

# Hard ceiling — the largest single-task allocation cpu_high_mem can serve.
# Estimates above this are clamped (with a warning at submit time); jobs that
# legitimately need more would have OOMed under the previous uniform 500 GB
# scheme too.
MEM_CEILING_GB = 500

# Floor for any job (Python + scanpy + pyo3 baseline + scratch).
MEM_FLOOR_GB = 8


def estimate_memory_gb(
    dataset: "DatasetConfig",
    format_key: str,
    benchmark: str,
) -> int:
    """Estimate per-job peak memory in GB for a (benchmark, dataset, format).

    Models the worst-case in-memory footprint with a ~50% safety margin and
    rounds up to the next 8 GB. Used by ``run_parallel.py`` to size each
    submitit job individually instead of allocating a single uniform value
    across all jobs in a tier.
    """
    import math

    n_obs = dataset.n_obs
    n_vars = dataset.n_vars
    h5ad_mb = max(dataset.approx_h5ad_mb, 1)

    # Base footprint: source AnnData object in RAM. Sparse expansion + obs/var
    # metadata + scratch typically runs ~2× the on-disk h5ad size.
    base_mb = h5ad_mb * 2

    # Dense materialization upper bound for an X matrix at f32.
    dense_mb = (n_obs * n_vars * 4) / (1024 * 1024)

    is_h5ad = format_key.startswith("h5ad")
    is_zarr = format_key.startswith("zarr")
    is_dense_path = is_h5ad or is_zarr  # densify on read in scanpy/h5py path

    if benchmark in ("read_full", "memory"):
        if is_dense_path:
            peak_mb = max(base_mb, dense_mb * 1.3)
        else:
            # SCX/SOMA stream sparse: bounded by sparse footprint + scratch.
            peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark in ("write", "parallel_write_scaling"):
        # Both load the source h5ad (sparse) then encode; for dense input
        # paths the encoder may materialize per-shard.
        peak_mb = max(base_mb, dense_mb * 1.0 if is_dense_path else dense_mb * 0.5)
    elif benchmark == "parallel_scaling":
        # Multiple concurrent readers share buffers but inflate scratch.
        peak_mb = max(base_mb * 4, dense_mb * 0.7)
    elif benchmark == "read_selective":
        # Column-projected reads — small in absolute terms.
        peak_mb = base_mb * 0.5
    elif benchmark == "compression":
        # Just measures file sizes — Python overhead only.
        peak_mb = base_mb * 0.25
    elif benchmark == "fragment_ops":
        # SCX-only. pyscx.append reads the entire input CSR into memory
        # (indptr + indices + decoded values) before re-encoding into the
        # target file — dominant footprint is ~sparse-CSR-sized working
        # buffers, not the dense matrix. Also needs scratch room for a few
        # on-disk copies of the base file (for per-run isolation), but those
        # are SCX-compressed and << dense_mb.
        peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark in ("cloud_push", "cloud_pull"):
        # SCX-only. pyscx.push/pull stream section-by-section with a small
        # reorder buffer; dominant footprint is the shard currently
        # encoding/decoding plus the catalog. Well-bounded regardless of
        # dataset size — just need enough for the source CSR read when
        # pack'ing on pull. Keep sized against base to cover catalog parse.
        peak_mb = max(base_mb, 4 * 1024)  # 4 GB ceiling for the streaming path
    elif benchmark == "cloud_read":
        # SCX pull+read needs the dense matrix at end; other runners (zarr,
        # tiledb, slaf) materialize in-memory too. Size like read_full.
        if is_dense_path:
            peak_mb = max(base_mb, dense_mb * 1.3)
        else:
            peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark == "cloud_metadata":
        # Catalog-only open — single GET + a few small parses. Trivial.
        peak_mb = max(base_mb * 0.25, 2 * 1024)  # 2 GB floor for Python baseline
    elif benchmark == "cloud_filtered":
        # Cross-format predicate pushdown — matching cells subset is loaded
        # per predicate; bounded by the largest expected result set.
        peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark == "cloud_reader_vs_pull":
        # Exercises both open_cloud and pull (full) paths; sizing matches
        # the more expensive full-pull path.
        peak_mb = max(base_mb, dense_mb * 0.8)
    elif benchmark == "cost_model":
        # metadata + selective + full_read per layout. Dominant run is
        # full_read; size to that.
        peak_mb = max(base_mb, dense_mb * 0.8)
    elif benchmark == "cloud_large_atlas":
        # Assertion-based — peak RSS MUST stay under the 240 MB bound from
        # docs/cloud.md. Give headroom but not much.
        peak_mb = 8 * 1024  # 8 GB ceiling for safety
    else:
        peak_mb = base_mb

    peak_gb = math.ceil(peak_mb / 1024)
    safety_gb = max(8, int(peak_gb * 0.5))
    total = peak_gb + safety_gb
    # Round up to the next 8 GB so we don't fragment the scheduler with
    # awkward request sizes.
    total = math.ceil(total / 8) * 8
    total = max(total, MEM_FLOOR_GB)
    # Clamp at the cluster's largest single-task allocation. Combinations
    # whose true footprint exceeds this (e.g. dense-h5ad read on census_5m)
    # would OOM either way; the cap keeps submitit from rejecting the job.
    return min(total, MEM_CEILING_GB)


def estimate_time_minutes(
    dataset: "DatasetConfig",
    format_key: str,
    benchmark: str,
) -> int:
    """Estimate per-job wall-clock budget in minutes for a triple.

    Centralizes the time ceiling that was previously tier-uniform in
    ``capture_baseline.py::TIERS``. Returns a value rounded up to 5-min
    increments so scheduler fragmentation stays low. Callers that want
    a conservative envelope multiply by ``--scale-factor``; CI defaults
    to the aggressive 0.9× side, ad-hoc runs use 1.3×.
    """
    import math

    n_obs = dataset.n_obs
    # Scale with n_obs: small datasets complete in minutes, census_10m in
    # hours. Model as a per-benchmark base rate + a per-million-cells term.
    per_million = max(n_obs / 1_000_000, 0.1)

    # Base minutes per benchmark (empirical floors on pbmc3k).
    base_minutes: dict[str, int] = {
        "compression":            3,
        "write":                  10,
        "read_full":              8,
        "read_selective":         10,
        "parallel_scaling":       20,
        "parallel_write_scaling": 25,
        "memory":                 15,
        "fragment_ops":           15,
        "cloud_push":             20,
        "cloud_pull":             20,
        "cloud_read":             20,
        "cloud_metadata":         5,
        "cloud_filtered":         20,
        "cloud_reader_vs_pull":   25,
        "cost_model":             20,
        "cloud_large_atlas":      60,   # 50GB+ pull is not quick
        "ml_loader":              30,
    }
    base = base_minutes.get(benchmark, 15)

    # Per-million-cells multiplier. Cloud ops scale linearly with bytes
    # downloaded; cost_model / cloud_large_atlas scale heavily.
    slope_minutes_per_million = 8
    if benchmark in ("cloud_large_atlas",):
        slope_minutes_per_million = 30
    elif benchmark in ("cloud_push", "cloud_pull", "cloud_read",
                        "cloud_reader_vs_pull", "cost_model"):
        slope_minutes_per_million = 12
    elif benchmark in ("cloud_metadata", "cloud_filtered"):
        slope_minutes_per_million = 4
    elif benchmark in ("compression", "read_selective"):
        slope_minutes_per_million = 2

    total = base + int(slope_minutes_per_million * per_million)
    # Dense-path formats (h5ad / zarr) take longer at census scale.
    if format_key.startswith(("h5ad", "zarr")) and n_obs >= 1_000_000:
        total = int(total * 1.5)

    # Round up to 5-min increments.
    return max(5, math.ceil(total / 5) * 5)


def partition_for_memory(mem_gb: int, default: str = "cpu_preemptible") -> str:
    """Auto-route to ``cpu_high_mem`` when ``mem_gb`` exceeds the preemptible
    cap, otherwise stay on ``default``.
    """
    if mem_gb > MEM_HIGH_MEM_THRESHOLD_GB:
        return "cpu_high_mem"
    return default


# ---------------------------------------------------------------------------
# SLURM Defaults
# ---------------------------------------------------------------------------

SLURM_DEFAULTS = {
    "cpu": {
        "partition": "cpu_preemptible",
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
        "partition": "preemptible",
        "cpus_per_task": 16,
        "mem_gb": 128,
        "gpus": 1,
        "time": "04:00:00",
    },
}
