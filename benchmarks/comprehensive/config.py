"""
SCX Comprehensive Benchmark Suite — Configuration.

Central configuration for dataset paths, format definitions, benchmark constants,
and system-level settings. All benchmark scripts import from here.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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

from benchmarks.comprehensive.bench_env import DATA_DIR

def _expand_path_env_vars() -> None:
    """Expand ``~`` in path-valued env vars loaded from ``.env``.

    ``python-dotenv`` does not tilde-expand values, so entries like
    ``GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/scx-bench.json`` would otherwise
    reach downstream consumers (``gcsfs``, ``gsutil`` subprocess, pyscx's
    Rust cloud backend) as a literal ``~``. ``gcsfs`` in particular silently
    falls back to anonymous auth when the cred file path doesn't exist,
    surfacing as cryptic ``Zstd decompression error: invalid input data`` on
    reads (it's actually getting 401-HTML back, not zstd bytes).

    ``require_gcp_credentials`` also performs this expansion, but callers
    of the cloud stack that bypass it (notably the repro scripts under
    ``scripts/repro_*_cloud_read.py`` — which by design hit the bug paths
    without explicit setup) rely on this early normalization. The function
    is idempotent, so re-invocation from test fixtures is safe.
    """
    for env_key in ("GOOGLE_APPLICATION_CREDENTIALS",):
        raw = os.environ.get(env_key, "")
        if raw and "~" in raw:
            os.environ[env_key] = os.path.expanduser(raw)


_expand_path_env_vars()

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
    # Phase K: multimodal flag + modality names (Phase B vocabulary).
    # Multimodal datasets carry a `.h5mu` source instead of `.h5ad`; the
    # multimodal_compression / multimodal_training benchmarks branch on
    # this flag.
    multimodal: bool = False
    modality_names: tuple[str, ...] = ()

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

    @property
    def shardad_path(self) -> Path:
        return DATA_DIR / f"{self.name}.shad"

    @property
    def cellstream_path(self) -> Path:
        # cellstream stores are directories, not single files.
        return DATA_DIR / f"{self.name}.cellstream"

    @property
    def annbatch_path(self) -> Path:
        # annbatch pre-shuffles into a sharded zarr DatasetCollection
        # (directory). AnnLoader / scDataset read the source h5ad directly and
        # therefore reuse `h5ad_path` (no dedicated fixture / conversion).
        return DATA_DIR / f"{self.name}.annbatch.zarr"

    # Per-codec SCX paths for benchmark isolation
    @property
    def scx_auto_path(self) -> Path:
        return DATA_DIR / f"{self.name}_auto.scx"

    @property
    def scx_full_path(self) -> Path:
        """The `auto`-codec fixture enriched with `.raw`, obsm and a layer.

        Deliberately **not** a `FormatVariant`, and deliberately not routed
        through `path_for_format`. Both of the obvious shapes would produce a
        threshold that a default `gate_candidate.py` run never evaluates:

        * as an `ADDITIONAL_FORMATS` variant — `run_parallel` defaults its format
          pool to `PRIMARY_FORMATS` (+ accel / multimodal), so it would only be
          scheduled under an explicit `--formats`. Adding it to `PRIMARY_FORMATS`
          instead would pair it with every benchmark that declares no
          `SUPPORTED_FORMATS`, which is most of them.
        * as a new `DATASETS` entry — it would sit outside every
          `capture_baseline.TIERS` list, and `gate_candidate.py` forwards
          `--datasets` only when the operator passes it. `check_absolute_floors`
          skips a triple that did not run *silently*, so the threshold would read
          as coverage and provide none. **53 floors across 7 datasets are in
          exactly that state today** — see `thresholds.yaml`'s Deferred floors
          block.

        Instead the two benchmarks that need it (`export_streaming`,
        `fragment_ops`) read this path as an extra *arm* of a triple the default
        gate already schedules.

        Why it has to exist at all: no source h5ad in the suite carries a
        `.raw`, an `obsm` key or a layer, so `census_1m_auto.scx` and
        `tabula_sapiens_100k_auto.scx` have `obsm_keys == []` and
        `layer_names == []`. Every export threshold measured on them is blind to
        the `.raw` / `obsm` / layer handling in `scx-convert`'s export path.

        Since OPT-CONVERT-1, `/raw` streams like `/X` and the layers, so `obsm`
        (with `varm` / `obsp` / `varp`) is the only whole-matrix copy left —
        OPT-CONVERT-4. The arm still matters more than that shrinkage suggests:
        streamed and materialised raw produce byte-identical output, so peak RSS
        here is the only signal that would catch a regression back.

        Built by `benchmarks/scripts/prep_full_fixtures.py`; absent until that
        has been run for the dataset.
        """
        return DATA_DIR / f"{self.name}_full.scx"

    @property
    def scx_fast_path(self) -> Path:
        return DATA_DIR / f"{self.name}_fast.scx"

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
    def scx_shufdelta_path(self) -> Path:
        return DATA_DIR / f"{self.name}_shufdelta.scx"

    @property
    def scx_compact_trial_path(self) -> Path:
        return DATA_DIR / f"{self.name}_compact_trial.scx"

    # F5 follow-up (Phase B): per-row-group-size compact-trial variants. Each G
    # needs a distinct on-disk path — `path_for_format` resolves via a static
    # key→property table, so a single G-parametrized variant would collide on
    # one file and the "exists → skip" cache would serve a stale G.
    @property
    def scx_compact_trial_g128_path(self) -> Path:
        return DATA_DIR / f"{self.name}_compact_trial_g128.scx"

    @property
    def scx_compact_trial_g256_path(self) -> Path:
        return DATA_DIR / f"{self.name}_compact_trial_g256.scx"

    @property
    def scx_compact_trial_g512_path(self) -> Path:
        return DATA_DIR / f"{self.name}_compact_trial_g512.scx"

    @property
    def scx_compact_trial_g1024_path(self) -> Path:
        return DATA_DIR / f"{self.name}_compact_trial_g1024.scx"

    @property
    def anndata_zarr_backed_path(self) -> Path:
        return DATA_DIR / f"{self.name}_anndata.zarr"

    # Phase K: multimodal source + per-format paths.
    @property
    def h5mu_path(self) -> Path:
        return DATA_DIR / f"{self.name}.h5mu"

    @property
    def h5mu_gzip_path(self) -> Path:
        return DATA_DIR / f"{self.name}_gzip.h5mu"

    @property
    def scx_multimodal_path(self) -> Path:
        return DATA_DIR / f"{self.name}_multimodal.scx"

    @property
    def scx_multimodal_uniform_path(self) -> Path:
        """SCX multimodal written with `codec_per_modality=False`
        (Phase K.3.4 sweep variant — every modality routed through
        single-modality `select_codec`)."""
        return DATA_DIR / f"{self.name}_multimodal_uniform.scx"

    @property
    def zarr_mudata_path(self) -> Path:
        return DATA_DIR / f"{self.name}.zarr.mudata"

    def path_for_format(self, format_key: str) -> Path:
        """Return the persistent on-disk path for a given format key.

        Accelerator variants (`accel_*__<impl>`) don't correspond to a
        file format — they run on the source h5ad directly via
        `pyscx.accel.*` / `scanpy.*`. For those keys we return the
        source h5ad path itself, which is what the `accel_*.py`
        benchmark modules already use through `dataset.h5ad_path`.
        """
        if format_key.startswith("accel_") or format_key.startswith("bench_csc__"):
            # `bench_csc_dispatch` variants run on the source h5ad
            # directly; the bench module converts to a CSC-equipped
            # SCX file once per dataset and caches it.
            return self.h5ad_path
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
    "scx_fast": "scx_fast_path",
    "scx_scx1": "scx_scx1_path",
    "scx_zstd": "scx_zstd_path",
    "scx_none": "scx_none_path",
    "scx_lz4": "scx_lz4_path",
    "scx_pcodec": "scx_pcodec_path",
    "scx_shufdelta": "scx_shufdelta_path",
    "scx_compact_trial": "scx_compact_trial_path",
    "scx_compact_trial_g128": "scx_compact_trial_g128_path",
    "scx_compact_trial_g256": "scx_compact_trial_g256_path",
    "scx_compact_trial_g512": "scx_compact_trial_g512_path",
    "scx_compact_trial_g1024": "scx_compact_trial_g1024_path",
    "bpcells": "bpcells_path",
    "parquet_zstd": "parquet_path",
    "slaf": "slaf_path",
    "shardad": "shardad_path",
    "cellstream": "cellstream_path",
    # Data-load Phase 0 competitor loaders. annbatch has its own pre-shuffled
    # zarr fixture; AnnLoader + scDataset read the source h5ad backed.
    "annbatch": "annbatch_path",
    "annloader": "h5ad_path",
    "scdataset": "h5ad_path",
    "anndata_zarr_backed": "anndata_zarr_backed_path",
    # Phase K — multimodal format keys.
    "h5mu_uncompressed": "h5mu_path",
    "h5mu_gzip": "h5mu_gzip_path",
    "zarr_mudata_zstd": "zarr_mudata_path",
    "scx_multimodal_per_modality_auto": "scx_multimodal_path",
    "scx_multimodal_uniform_auto": "scx_multimodal_uniform_path",
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
    "scx_fast": ".scxd",
    "scx_scx1": ".scxd",
    "scx_zstd": ".scxd",
    "scx_none": ".scxd",
    "scx_lz4": ".scxd",
    "scx_pcodec": ".scxd",
    "scx_shufdelta": ".scxd",
    "scx_compact_trial": ".scxd",
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
    # GPU pseudobulk NB-GLM DE fixtures (accel_de_nb_glm). Stratified sparse
    # Perturb-seq: control + (n_perts-1) KOs × n_donors donors → each condition
    # has n_donors≥2 replicates (pdex_nb_glm's stratifier guard). Generated
    # in-process by _pert_synth.make_raw_counts_stratified; n_obs = cells_per ·
    # n_perts · n_donors (cells_per≈60). n_sub per target = 2·n_donors = 6, the
    # documented target regime.
    #
    # density=0.3 (~35% nonzero): an ultra-sparse single-cell density (~0.08)
    # leaves the pseudobulk aggregate with too few counts/gene/block (~3) for the
    # DE anchors to be defined — CPU↔GPU rank Spearman stays 1.0 (the kernels
    # agree) but NB-GLM-vs-pdex_ref concordance is undefined on signal-free genes.
    # Calibrated CPU-side: density 0.3 at cells_per≈60 gives min(Spearman) ≈ 0.98
    # vs pdex_ref with comfortable margin over the 0.95 floor, while staying
    # representative (aggregation does not dominate — the surfaced speedup stays
    # well above 1×). The ultra-sparse regime's aggregation-aware speedup is
    # characterized separately by the standalone bench_nb_glm.py --gpu dev tool.
    #
    # NOT added to capture_baseline TIERS (that would schedule every other
    # benchmark on a fixture with no on-disk h5ad) — exercise via
    # `run_parallel.py --benchmarks accel_de_nb_glm --datasets nb_glm_synth`,
    # mirroring how cell_eval_parity_perf's pert_synth_* floors are run.
    "nb_glm_synth_small": DatasetConfig(
        id="NBG0", name="nb_glm_synth_small",
        n_obs=3_600, n_vars=18_000,
        protocol="synthetic (stratified Perturb-seq, raw counts)",
        source="_pert_synth.make_raw_counts_stratified",
        approx_h5ad_mb=40, available=True, synthetic=True,
        synth_params={"n_obs": 3_600, "n_vars": 18_000, "n_perts": 20,
                      "n_donors": 3, "density": 0.3, "seed": 42},
    ),
    "nb_glm_synth": DatasetConfig(
        id="NBG1", name="nb_glm_synth",
        n_obs=18_000, n_vars=18_000,
        protocol="synthetic (stratified Perturb-seq, raw counts)",
        source="_pert_synth.make_raw_counts_stratified",
        approx_h5ad_mb=200, available=True, synthetic=True,
        synth_params={"n_obs": 18_000, "n_vars": 18_000, "n_perts": 100,
                      "n_donors": 3, "density": 0.3, "seed": 42},
    ),
    "nb_glm_synth_xl": DatasetConfig(
        id="NBG2", name="nb_glm_synth_xl",
        n_obs=45_000, n_vars=18_000,
        protocol="synthetic (stratified Perturb-seq, raw counts)",
        source="_pert_synth.make_raw_counts_stratified",
        approx_h5ad_mb=500, available=True, synthetic=True,
        synth_params={"n_obs": 45_000, "n_vars": 18_000, "n_perts": 250,
                      "n_donors": 3, "density": 0.3, "seed": 42},
    ),
    # Real Perturb-seq fixtures for the grouped-sharding benchmark
    # (grouped_sort). Copied from the read-only loader tree by
    # benchmarks/comprehensive/scripts/prep_grouped_fixtures.py. Each carries a
    # categorical grouping column (see grouped_sort.GROUP_SPEC): `replogle_k562`
    # is dense X grouped by `gene` with a `non-targeting` reference (exercises
    # the dense → two-pass auto route + reference isolation); `tahoe_c38` is CSR
    # X grouped by `drug` (exercises the CSR one-pass route).
    "replogle_k562": DatasetConfig(
        id="GS1", name="replogle_k562",
        n_obs=68_729, n_vars=6_546,
        protocol="Perturb-seq (CRISPRi, K562)",
        source="Replogle2022 — k562_n600.h5ad (dense X, gene + non-targeting)",
        approx_h5ad_mb=3_600, available=True,
    ),
    "tahoe_c38": DatasetConfig(
        id="GS2", name="tahoe_c38",
        n_obs=69_245, n_vars=62_710,
        protocol="Tahoe-100M drug screen slice",
        source="tahoe-100m — c38-n10.h5ad (CSR X, drug)",
        approx_h5ad_mb=2_200, available=True,
    ),
    # Real RAW-COUNT Perturb-seq fixture (integer UMIs, CSR) — shardad's
    # integer-count home turf. group_by `target_gene` (2354 KOs) + `non-targeting`
    # reference. ~909M nnz; the largest grouped fixture (7.3 GB h5ad).
    "chemogenetic_rgfp": DatasetConfig(
        id="GS3", name="chemogenetic_rgfp",
        n_obs=136_051, n_vars=18_151,
        protocol="Perturb-seq (CRISPRi, raw counts, HDAC-inhibitor screen)",
        source="chemogenetic_h1/run1 — RGFP-n5.h5ad (CSR raw counts, target_gene + non-targeting)",
        approx_h5ad_mb=7_000, available=True,
    ),
    # Phase K — multimodal datasets sourced from 10x Genomics public
    # CITE-seq + Multiome libraries. Staged via
    # benchmarks/scripts/download_citeseq_pbmc.py and
    # download_multiome_pbmc.py. n_vars is the *sum* across modalities
    # (no single global var index; each modality has its own).
    "cite_seq_pbmc": DatasetConfig(
        id="K1", name="cite_seq_pbmc_5k",
        n_obs=5_247, n_vars=33_538 + 32,
        protocol="10x v3 (UMI) + Antibody Capture",
        source="10x Genomics — 5k_pbmc_protein_v3",
        approx_h5ad_mb=85, available=True,
        multimodal=True, modality_names=("rna", "adt"),
    ),
    "multiome_pbmc": DatasetConfig(
        id="K2", name="multiome_pbmc_10k",
        n_obs=11_898, n_vars=36_601 + 143_887,
        protocol="10x Multiome ARC v1 (RNA + ATAC)",
        source="10x Genomics — pbmc_granulocyte_sorted_10k",
        approx_h5ad_mb=1_086, available=True,
        multimodal=True, modality_names=("rna", "atac"),
    ),
    # Tiny synthetic h5ad used by the streaming-conversion smoke job.
    # The fixture is generated on-node by `scripts/_synth_h5ad.py` and
    # lives under SCX_DATA_DIR, so it's marked `synthetic=False` (the
    # generator runs outside the harness rather than via the harness's
    # own synth path).
    "streaming_smoke": DatasetConfig(
        id="S1", name="streaming_smoke",
        n_obs=50_000, n_vars=5_000,
        protocol="synthetic (uint8)", source="generated on /scratch",
        approx_h5ad_mb=40, available=True,
    ),
}


# Phase K — convenience export of multimodal-only dataset names so the
# orchestrator can resolve `--datasets multimodal` shorthand and the
# multimodal benchmarks can iterate over the right subset without
# leaking single-modality entries.
MULTIMODAL_DATASETS: list[str] = [
    name for name, ds in DATASETS.items() if ds.multimodal
]


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
    # Decode-speed-max escape hatch (the pre-flip `auto` behavior): heuristic
    # single-encode (Scx1 for low-median integer shards). Gates the decode-max
    # throughput floors that `scx_auto` used to carry.
    FormatVariant("SCX (fast)", "scx_fast", "primary", "scx_runner",
                  {"codec": "fast"}),
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
    FormatVariant("SCX (shufdelta)", "scx_shufdelta", "primary", "scx_runner",
                  {"codec": "shufdelta"}),
    FormatVariant("SCX (compact-trial)", "scx_compact_trial", "primary", "scx_runner",
                  {"codec": "compact-trial", "row_group_rows": 512}),
    FormatVariant("SLAF", "slaf", "primary", "slaf_runner"),
    FormatVariant("Shardad", "shardad", "primary", "shardad_runner"),
    FormatVariant("CellStream", "cellstream", "primary", "cellstream_runner"),
]

ADDITIONAL_FORMATS: list[FormatVariant] = [
    # Data-load Phase 0 competitor loaders. Kept OUT of PRIMARY so they never
    # enter the default cross-format sweep (they are loaders reading h5ad /
    # pre-shuffled zarr, not storage formats — a `read_full`/`compression` run
    # on them would just duplicate the h5ad rows). Only `ooc_loader` allow-lists
    # them (SUPPORTED_FORMATS), and the Phase-0 capture selects them explicitly
    # via `--benchmarks ooc_loader --formats annbatch annloader scdataset ...`.
    # `--formats` resolves against ALL_FORMATS, so ADDITIONAL keys are
    # selectable. annbatch pre-shuffles to sharded zarr (`annbatch_runner`);
    # AnnLoader + scDataset read the source h5ad backed (`h5ad_runner` +
    # `h5ad_path`, so Phase-A conversion is a no-op skip). BioNeMo-SCDL is
    # deferred (isolated NeMo/CUDA env).
    FormatVariant("annbatch", "annbatch", "additional", "annbatch_runner"),
    FormatVariant("AnnLoader (backed)", "annloader", "additional", "h5ad_runner"),
    FormatVariant("scDataset", "scdataset", "additional", "h5ad_runner"),
    FormatVariant("BPCells", "bpcells", "additional", "bpcells_runner"),
    FormatVariant("Parquet (zstd)", "parquet_zstd", "additional", "parquet_runner",
                  {"compression": "zstd"}),
    FormatVariant("AnnData-on-Zarr (backed)", "anndata_zarr_backed", "additional",
                  "zarr_runner", {"backed": True}),
    # F5 follow-up (Phase B): row-group-size sweep for the compact-trial codec.
    # Kept out of PRIMARY (the default no-`--formats` sweep) so they only run
    # when explicitly selected via `--formats scx_compact_trial_g<G>` — used by
    # the `compression` (ratio-vs-G) and `read_scattered` (block-index adoption)
    # benchmarks. `--formats` resolves against ALL_FORMATS so ADDITIONAL keys are
    # selectable. Distinct on-disk paths (see `scx_compact_trial_g<G>_path`).
    FormatVariant("SCX (compact-trial G=128)", "scx_compact_trial_g128", "additional",
                  "scx_runner", {"codec": "compact-trial", "row_group_rows": 128}),
    FormatVariant("SCX (compact-trial G=256)", "scx_compact_trial_g256", "additional",
                  "scx_runner", {"codec": "compact-trial", "row_group_rows": 256}),
    FormatVariant("SCX (compact-trial G=512)", "scx_compact_trial_g512", "additional",
                  "scx_runner", {"codec": "compact-trial", "row_group_rows": 512}),
    FormatVariant("SCX (compact-trial G=1024)", "scx_compact_trial_g1024", "additional",
                  "scx_runner", {"codec": "compact-trial", "row_group_rows": 1024}),
]


# Phase K — multimodal format variants. These consume `.h5mu` rather
# than `.h5ad`, so the multimodal_compression benchmark routes via
# `runner.convert_from_h5mu` instead of `convert_from_h5ad`.
MULTIMODAL_FORMATS: list[FormatVariant] = [
    FormatVariant(
        "h5mu (uncompressed)", "h5mu_uncompressed", "multimodal", "h5mu_runner",
        {"compression": None},
    ),
    FormatVariant(
        "h5mu (gzip)", "h5mu_gzip", "multimodal", "h5mu_runner",
        {"compression": "gzip"},
    ),
    FormatVariant(
        "Zarr-MuData (zstd)", "zarr_mudata_zstd", "multimodal", "zarr_mudata_runner",
        {"compressor": "zstd", "level": 3},
    ),
    FormatVariant(
        "SCX multimodal (per-modality auto)",
        "scx_multimodal_per_modality_auto", "multimodal", "scx_runner",
        {"codec": "auto", "codec_per_modality": True},
    ),
    FormatVariant(
        "SCX multimodal (uniform auto)",
        "scx_multimodal_uniform_auto", "multimodal", "scx_runner",
        {"codec": "auto", "codec_per_modality": False},
    ),
]

ALL_FORMATS = PRIMARY_FORMATS + ADDITIONAL_FORMATS + MULTIMODAL_FORMATS


# ---------------------------------------------------------------------------
# Accelerator variants — Phase 9.1+
#
# These are NOT file-format variants; they're accelerator implementations
# for PCA / kNN / UMAP / Leiden / preprocessing / HVG. Each registers as a
# `FormatVariant` so the per-cell parallel launcher (`run_parallel.py`) can
# schedule one SLURM job per (benchmark × implementation × dataset) — e.g.
# (accel_pca, accel_pca__pyscx_gpu_rand_hh, census_1m). Runner is `noop_runner`
# because these benchmarks don't depend on file conversion.
# ---------------------------------------------------------------------------

def accel_formats() -> list[FormatVariant]:
    """Lazily discover accelerator variants from each `accel_*.py` module.

    Imported on demand (not at `config.py` module-load) so the
    accelerator modules — which import from `config.py` — don't create
    a circular import. Each accelerator benchmark module exports a
    `<bench>_variants()` callable; append it here when it lands.
    """
    out: list[FormatVariant] = []
    try:
        from benchmarks.comprehensive.benchmarks.accel_pca import (
            accel_pca_variants,
        )
        out.extend(accel_pca_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_knn import (
            accel_knn_variants,
        )
        out.extend(accel_knn_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_umap import (
            accel_umap_variants,
        )
        out.extend(accel_umap_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_leiden import (
            accel_leiden_variants,
        )
        out.extend(accel_leiden_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_harmony import (
            accel_harmony_variants,
        )
        out.extend(accel_harmony_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_preprocess import (
            accel_preproc_variants,
        )
        out.extend(accel_preproc_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_hvg import (
            accel_hvg_variants,
        )
        out.extend(accel_hvg_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_pipeline import (
            accel_pipeline_variants,
        )
        out.extend(accel_pipeline_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_de import (
            accel_de_variants,
        )
        out.extend(accel_de_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_de_nb_glm import (
            accel_de_nb_glm_variants,
        )
        out.extend(accel_de_nb_glm_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_eval_metrics import (
            accel_eval_metrics_variants,
        )
        out.extend(accel_eval_metrics_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_r_route import (
            accel_r_route_variants,
        )
        out.extend(accel_r_route_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.bench_csc_dispatch import (
            bench_csc_dispatch_variants,
        )
        out.extend(bench_csc_dispatch_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import (
            accel_to_gpu_anndata_variants,
        )
        out.extend(accel_to_gpu_anndata_variants())
    except ImportError:
        pass
    try:
        from benchmarks.comprehensive.benchmarks.accel_format_pipeline import (
            accel_format_pipeline_variants,
        )
        out.extend(accel_format_pipeline_variants())
    except ImportError:
        pass
    return out


def get_formats(
    include_additional: bool = False,
    include_accel: bool = False,
) -> list[FormatVariant]:
    """Return the list of format variants to benchmark.

    `include_accel` adds accelerator implementations (PCA / kNN / UMAP /
    Leiden / preprocessing / HVG variants). Enable this only when the
    orchestrator is also scheduling `accel_*` benchmark modules.
    """
    out = list(PRIMARY_FORMATS)
    if include_additional:
        out.extend(ADDITIONAL_FORMATS)
    if include_accel:
        out.extend(accel_formats())
    return out


def pseudobulk_n_cpus_cap() -> int:
    """Bounded pydeseq2 worker cap for the pseudobulk DE benchmark paths.

    The ``correctness`` and ``bench_csc_dispatch`` benches pass this explicitly
    to ``pyscx.accel.pseudobulk_dex(n_cpus=...)`` for determinism. Capped at 8
    so a large SLURM allocation never spawns one loky worker (~300 MB Python
    interpreter) per core and OOM-kills the job. Mirrors the env-derived default
    baked into ``pseudobulk_dex`` itself.
    """
    return min(int(os.environ.get("SLURM_CPUS_PER_TASK") or 4), 8)


# ---------------------------------------------------------------------------
# Benchmark Constants
# ---------------------------------------------------------------------------

# Number of repetitions
N_RUNS_LARGE = 3      # For datasets >= 100K cells
N_RUNS_SMALL = 5      # For datasets < 100K cells
N_WARMUP_RUNS = 1     # Discarded warm-up iterations

# Row-slice query: number of random cells to select
QUERY_N_CELLS = 1_000

# Streaming-vs-in-memory benchmark: row-chunk width for the iteration
# loop. 65 536 rows matches the canonical SCX shard target so backed
# iteration usually decodes one shard per chunk on default configs.
STREAMING_CHUNK_ROWS = 65_536

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
# scheme too. Bumped 500 → 1000 GB after the 2026-05-10 tier-full gate ran
# correctness/scx_auto into the prior ceiling on census_500k / census_1m;
# Chimera's high-mem nodes carry ~1–2 TB so 1 TB single-task is reachable.
MEM_CEILING_GB = 1000

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
        # tiledb_soma materialises the full obs DataFrame before
        # applying the predicate's value_filter — on census_1m that's
        # ~1.5 GB of obs plus per-thread working buffers — and OOM-killed
        # both census tier cells on the 2026-05-10 tier-full run. Bump for
        # >=500K cells via the dense-mb proxy (scales with n_obs * n_vars
        # but in practice the obs side dominates).
        if format_key == "tiledb_soma" and n_obs >= 500_000:
            peak_mb = max(peak_mb, dense_mb * 0.6)
    elif benchmark == "compression":
        # Just measures file sizes — Python overhead only.
        peak_mb = base_mb * 0.25
    elif benchmark in ("conversion_streaming", "export_streaming"):
        # Both benchmarks cap their materialize arm above
        # MATERIALIZE_MAX_N_OBS (1M cells), so the worst resident case at any
        # tier is that arm at exactly 1M: the whole CSR triplet, measured at
        # 13.98 GB on census_1m — about 1.2x the source h5ad, which `base_mb`
        # (2x) already covers.
        #
        # The 1.5x on top is `export_streaming`'s `streaming_full` arm. It used
        # to be justified by `.raw` being held whole by `write_raw_to_h5ad`'s
        # `read_all_raw_csr_shards()` — +1579 MB over the plain arm on
        # tabula_sapiens_100k, scaling with nnz at 8 B each. OPT-CONVERT-1 made
        # raw stream, which removed that term: the measured `streaming_full`
        # residual over the plain arm is now ~508 MB on tabula and ~4335 MB on
        # census_1m, from the still-eager `obsm` plus the fact that peak RSS over
        # a multi-matrix export is not the max of independent peaks.
        #
        # The multiplier is DELIBERATELY not retuned here. It sizes a scheduler
        # request, where over-asking costs queue time and under-asking costs an
        # OOM mid-capture; census_1m's `streaming_full` still peaks at ~7.5 GB,
        # which `base_mb` alone does not cover at every tier. Retuning it wants
        # its own measurement across tiers, not a division on one capture.
        #
        # Both fell through to `peak_mb = base_mb` before this branch, which was
        # not a decision: `export_streaming`'s own docstring flagged the absence
        # as the reason its materialize cap had to exist at all.
        peak_mb = max(base_mb * 1.5, 8 * 1024)
    elif benchmark == "build_csc":
        # `write_csc_sidecar` takes the whole CSR by value and builds the
        # column-major transpose beside it, so the working set tracks the sparse
        # footprint and not the `memory_limit` the caller passes — the gap this
        # benchmark exists to measure.
        #
        # Sized from the measurement rather than from `dense_mb`: peak was
        # 2970 MB at tabula_sapiens_100k and 8137 MB at census_500k, against
        # `base_mb` (2x the source h5ad) of 3200 and 11,400 MB. So 1.5x base
        # covers both with room; `dense_mb * 0.6` would have asked for 70 GB at
        # census_500k for an 8 GB peak.
        peak_mb = max(base_mb * 1.5, 8 * 1024)
    elif benchmark == "mtx_export":
        # Both directions materialise: `to_mtx` reads the whole CSR before
        # formatting, `from_mtx` parses the triplet into one. Bounded by the
        # sparse footprint plus the gzip buffers, not by the dense matrix — so
        # sized off `base_mb` (2x the source h5ad) and NOT off `dense_mb`, which
        # an earlier version used while this comment already said not to.
        #
        # The module's `FORMAT_DATASET_SCOPE` caps it at n_obs <= 100,000, so the
        # largest cell is tabula_sapiens_100k (base_mb 3200); measured peak is
        # under 500 MB at pbmc3k and O(nnz) above that.
        peak_mb = max(base_mb, 8 * 1024)
    elif benchmark == "fragment_ops":
        # SCX-only. pyscx.append reads the entire input CSR into memory
        # (indptr + indices + decoded values) before re-encoding into the
        # target file — dominant footprint is ~sparse-CSR-sized working
        # buffers, not the dense matrix. Also needs scratch room for a few
        # on-disk copies of the base file (for per-run isolation), but those
        # are SCX-compressed and << dense_mb.
        #
        # The `compact_full` arm (a rewrite of `<name>_full.scx`, which carries
        # `.raw` + obsm + a layer) rides inside this envelope: it is another
        # rewrite of a file ~2.7x the plain one, and `dense_mb * 0.5` is already
        # 11.7 GB at tabula_sapiens_100k against a measured single-GB peak.
        #
        # The `obs_import` arm's two additions are both small and both
        # absolute, not scaled by the matrix: one obs index materialised as
        # Python strings for the CSV (`read_obs([])`, 1M rows at census_1m,
        # measured 5.0 s and a few hundred MB), and the pandas frame written
        # out beside it. `attach_external_obs` itself never reads X, layers,
        # `var`, the CSC sidecar or `.raw` — it appends obs shards and rehashes
        # the file, so its own peak is a shard, not a matrix. Disk, not memory,
        # is what this arm consumes most of (one more full-file copy per run).
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
        # tiledb, slaf) materialize in-memory too. Size like read_full,
        # then 2.0× to cover the post-loop verification pass which
        # transiently double-materialises cloud + local CSRs to compare.
        if is_dense_path:
            peak_mb = max(base_mb, dense_mb * 2.0)
        else:
            peak_mb = max(base_mb, dense_mb * 1.0)
    elif benchmark == "cloud_metadata":
        # Catalog-only open — single GET + a few small parses. Trivial.
        # The `scx_info_cloud` arm is a *subprocess*, so its footprint is the
        # CLI's own (a catalog plus per-shard headers, tens of MB) and is not
        # additive with this process's peak in any way the sampler here sees.
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
    elif benchmark == "ml_loader":
        # Streaming loaders (SCX/SOMA/SLAF) hold a few batches resident plus
        # the source CSR; full-load loaders (anndata h5ad path, scDataLoader)
        # materialize the entire matrix before iterating. The GPU training
        # scenario adds the model + activations on-device — host RAM stays
        # bounded by the streaming side.
        # The `raw_obs_highcard` arm additionally holds one categorical obs
        # column's global dictionary — ~958k Rust `String`s built once at
        # construction, plus a fresh `PyUnicode` list of the same length per
        # batch. At ~30 B/entry that is ~30 MB resident and ~30 MB churned per
        # batch: real, but two orders below the CSR terms below.
        if is_dense_path:
            peak_mb = max(base_mb * 2, dense_mb * 1.3)
        else:
            peak_mb = max(base_mb * 2, dense_mb * 0.5)
    elif benchmark == "ooc_loader":
        # Same sizing model as ml_loader's streaming path: SCX/SOMA/annbatch/
        # scDataset/AnnLoader hold a few batches + the source CSR resident; the
        # `h5ad_none` full-RAM arm materializes the whole matrix. Sized to the
        # sparse tier; the memory-constrained OOC regime deliberately UNDER-sizes
        # this via SCX_BENCH_OOC_MEM_CAP_GB (applied after the safety margin).
        peak_mb = max(base_mb * 2, dense_mb * 0.5)
    elif benchmark == "cellset_gather":
        # SCX-only S=64 gather: the shared shard cache holds a bounded working
        # set + a few decoded batches. Bounded by the sparse footprint, not the
        # dense matrix. Also subject to the OOC cap below.
        #
        # The `collate_rust` arm holds ten gathered batches resident at once
        # (it times the kernel alone, so the gather has to finish first) plus
        # one batch's stacked output at k_enc=2048 / k_dec=1024. At S=64 that
        # is ~10k cells: ~160 MB of CSR at tabula's ~1950 nnz/cell, and ~130 MB
        # of f32/i64 output tensors. Both are absolute, not a fraction of the
        # file, so they sit inside the existing envelope on every real dataset
        # and only matter on a tiny one — where `base_mb` already dominates.
        peak_mb = max(base_mb, dense_mb * 0.3)
    elif benchmark == "obs_open":
        # Matrix-free obs open (SCX read_obs) or eager-obs h5ad backed. Peak is
        # one obs table (+ per-file scratch across a manifest), never the X
        # matrix — Python baseline dominates.
        peak_mb = max(base_mb * 0.5, 4 * 1024)
    elif benchmark == "shuffle_layout":
        # The rewrite dominates. `scx sort` without a --memory-budget takes the
        # in-memory strategy, which gathers the whole X (nnz*8 + indptr) plus
        # obsm/layers — the same footprint `grouped_sort` sizes for, minus the
        # grouped reference-shard buffer. The TrainingDataset epochs are
        # streaming and bounded well below that. NOT capped by
        # SCX_BENCH_OOC_MEM_CAP_GB: under-sizing the request here would OOM the
        # rewrite rather than force an interesting out-of-core regime.
        peak_mb = max(base_mb * 2, dense_mb * 0.5)
    elif benchmark == "correctness":
        # Correctness keeps the scanpy-reference AnnData + SCX backed view
        # + SLAF round-trip materialization simultaneously while running
        # 14+ scanpy-equivalence checks (PCA, kNN, UMAP, leiden, DE).
        # slurm_validation_suite.sh allocates 80 GB for pbmc3k and 250 GB
        # for the pbmc3k+tabula_sapiens_100k combined run; size like
        # read_full but with a denser safety multiplier on top.
        # Bumped from `(base_mb*2, dense_mb*1.5)` after the 2026-05-09
        # tier-full gate run, then bumped again from `(base_mb*3, dense_mb*2.0)`
        # after the 2026-05-10 run OOM-killed 5 datasets (pbmc10k, smartseq2,
        # tabula_sapiens_100k, census_500k, census_1m) on scx_auto.
        # Stacked-fixture footprint runs ~2.5x dense on the larger datasets:
        # scanpy reference materialises a dense X for the equivalence
        # assertions, SCX backed reader holds its own working buffers, and
        # SLAF round-trip transiently doubles the in-flight cell count.
        # The 12-GB peak floor handles small datasets like pbmc10k (1.5 GB
        # dense) where the multipliers under-shoot — pbmc10k OOM-killed at
        # the 16 GB allocation that the formula otherwise yielded.
        # Run #4 of the 2026-05-10 gate still OOM'd 4 datasets (smartseq2,
        # tabula_sapiens_100k, census_500k, census_1m) at the 2.5x dense
        # multiplier; run #5 OOM'd the same 4 at the 3.5x multiplier.
        # smartseq2 OOM-killed at the 64 GB allocation after only 96s,
        # suggesting peak is well above 5x dense. Bumped to 5x dense to
        # cover (scanpy reference + SCX backed + SLAF round-trip) all
        # materialising simultaneously + PCA/DE working buffers.
        # census_500k / census_1m hit MEM_CEILING_GB at this multiplier;
        # those triples should ride a justification, not a higher cap.
        peak_mb = max(base_mb * 4, dense_mb * 5.0, 12 * 1024)
    elif benchmark == "roundtrip":
        # Holds source AnnData + SCX-materialised AnnData resident at the
        # same time, plus the (a - b) CSR diff scratch and a second copy
        # per side from the f32 cast / canonicalisation in
        # roundtrip._normalize_csr. dense_mb * 2.0 covers two full
        # matrices with margin for the diff buffer; on highly-sparse
        # inputs this is over-estimated (sparse << dense), but
        # over-budgeting beats OOM-then-resubmit.
        peak_mb = max(base_mb * 2, dense_mb * 2.0)
    elif benchmark == "cell_eval_parity_perf":
        # Holds adata_real + adata_pred + raw_adata simultaneously, plus
        # per-op working buffers (clustering_agreement materialises a
        # centroid kNN graph; energy_distance allocates an O(n_perts^2)
        # GEMM staging area). Observed ~14 GB at 100K — size for headroom.
        peak_mb = max(base_mb * 2, dense_mb * 2.5)
    elif benchmark == "accel_preprocess":
        # Keeps raw AnnData + scanpy reference + per-run copies resident.
        # scanpy normalize/log1p are sparse-in-place; pyscx's lazy-chain
        # materialization (a.X[:, :]) and the GPU-eager scipy CSR handoff
        # can densify intermediate buffers. Observed on census_1m: scanpy
        # 44 GB, pyscx OOM at 64 GB. Size to dense × 1.0 → census_1m lands
        # at ~168 GB (below the 200 GB cpu_preemptible ceiling so CPU
        # variants avoid cpu_high_mem; GPU variants stay on preemptible).
        peak_mb = max(base_mb * 2, dense_mb * 1.0)
    elif benchmark == "accel_hvg":
        # Loess fit uses f64 working arrays + per-gene variance accumulators.
        # Observed 33 GB peak on 1M cells.
        peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark == "accel_de":
        # Per-gene chunked dense buffer `n_obs × gene_chunk_size` (CPU)
        # or device upload of the same (GPU). Rank-test scratch is
        # bounded by the chunk size, not the full n_vars. Size like
        # `bench_csc_dispatch` which has the same chunked dense buffer
        # shape on the CSR-DE path.
        peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark == "accel_de_nb_glm":
        # Synthetic stratified fixture (~9K cells × 18K genes sparse) + the
        # aggregated pseudobulk + a few result DataFrames. The GPU variant holds
        # the CPU + GPU result frames simultaneously for the concordance merge.
        # Bounded by the dense materialisation of the small fixture; size like
        # accel_de's chunked-dense tier.
        peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark == "accel_eval_metrics":
        # Paired real/pred AnnData held simultaneously + per-group pseudobulk
        # means + an O(n_perts^2) GEMM staging area for energy_distance. Same
        # working-set shape as cell_eval_parity_perf (which holds the same pair)
        # but without the extra raw_adata copy — size like it, slightly leaner.
        peak_mb = max(base_mb * 2, dense_mb * 2.0)
    elif benchmark == "bench_csc_dispatch":
        # CSC dispatch benchmark — peak RSS dominated by the operation
        # being run (qc / hvg / de / pseudobulk). DE chunked dense
        # buffer is `n_obs × gene_chunk_size`; HVG keeps loess working
        # arrays. Size roughly like accel_hvg.
        peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark in ("accel_pipeline", "accel_format_pipeline"):
        # End-to-end PCA→kNN→UMAP: holds the preprocessed AnnData + scanpy
        # reference embedding/connectivities + per-run copies, plus the PCA
        # sparse/GPU buffers and the kNN/UMAP graphs simultaneously (the union
        # of accel_pca + accel_knn + accel_umap working sets). Size like the
        # PCA/DE chunked-buffer tier. `accel_format_pipeline` runs the same full
        # RAPIDS pipeline per variant plus a fresh load-to-GPU (and a one-shot
        # scx_auto conversion in prep), so it sits in the same tier — sizing it
        # on the generic `accel_*` estimate (dense_mb × 0.1) under-budgets it.
        peak_mb = max(base_mb, dense_mb * 0.5)
    elif benchmark in ("accel_umap", "accel_leiden", "accel_harmony"):
        # Embeddings + kNN graph + leiden graph in RAM. Observed <10 GB on
        # 1M cells.
        peak_mb = max(base_mb, dense_mb * 0.2)
    elif benchmark == "accel_to_gpu_anndata":
        # Sidecar fixture is now prepared by `scx optimize` (a streaming,
        # one-shard-bounded subprocess), so the dense h5ad self-convert no
        # longer dominates. Host peak is two concurrent sparse CSR copies —
        # the `to_anndata` host reference and `adata_gpu.X.get()` held side by
        # side for the byte-exact compare — i.e. ~2 × the on-disk sparse size
        # (≈ base_mb, which is 2 × h5ad). Size to base_mb × 1.5 for scipy
        # temporaries + the optimize subprocess shard buffer; NOT dense_mb,
        # which over-budgeted census_1m at 352 GB. The decoded matrix lives in
        # VRAM (H100 80 GB), not host RAM.
        peak_mb = max(base_mb * 1.5, dense_mb * 0.05)
    elif benchmark.startswith("accel_"):
        # PCA / kNN stream through sparse or GPU buffers. Observed 2-10 GB
        # on 1M cells.
        peak_mb = max(base_mb, dense_mb * 0.1)
    elif benchmark == "multimodal_compression":
        # Five-format sweep on a `.h5mu` source. Loads MuData once + writes
        # multiple variants (h5mu raw/gzip, zarr, two SCX). Like
        # single-modality `compression`, dominated by Python overhead;
        # half of base is generous.
        peak_mb = base_mb * 0.5
    elif benchmark == "multimodal_training":
        # Eager mudata baseline materialises every modality; SCX path
        # streams. CITE-seq peaks at ~590 MB, Multiome at ~5 GB host RSS
        # in the empirical SLURM run; size like ml_loader's sparse path.
        peak_mb = max(base_mb * 2, dense_mb * 0.5)
    elif benchmark in ("grouped_read", "grouped_sort"):
        # Grouped write + per-perturbation reads. The dominant buffer is the
        # single reference shard held in memory during encode: a large reference
        # group (e.g. chemogenetic_rgfp's 127,705 `non-targeting` cells at
        # ~6,700 nnz/cell ≈ 7 GB CSR) plus the one-pass gather source buffers and
        # (shardad) in-memory materialization. `dense_mb*0.5` under-sized RGFP
        # (24 GB → OOM); size to the sparse footprint with generous headroom.
        peak_mb = max(base_mb * 2, dense_mb * 1.5, 48 * 1024)
    else:
        peak_mb = base_mb

    peak_gb = math.ceil(peak_mb / 1024)
    safety_gb = max(8, int(peak_gb * 0.5))
    total = peak_gb + safety_gb
    # Round up to the next 8 GB so we don't fragment the scheduler with
    # awkward request sizes.
    total = math.ceil(total / 8) * 8
    total = max(total, MEM_FLOOR_GB)
    # Data-load Phase 0 out-of-core regime: to force genuine page-cache misses
    # on datasets whose resident footprint would otherwise fit a big node,
    # `SCX_BENCH_OOC_MEM_CAP_GB` caps the `ooc_loader` / `cellset_gather`
    # request BELOW that footprint (the memory-constrained SLURM allocation the
    # report calls for). Only these two benches honor it; everything else is
    # sized for headroom as usual. The cap still respects MEM_FLOOR_GB.
    if benchmark in ("ooc_loader", "cellset_gather"):
        cap_raw = os.environ.get("SCX_BENCH_OOC_MEM_CAP_GB", "").strip()
        if cap_raw:
            try:
                cap_gb = int(float(cap_raw))
            except ValueError:
                cap_gb = 0
            if cap_gb >= MEM_FLOOR_GB and cap_gb < total:
                logger.info(
                    "estimate_memory_gb(%s/%s/%s): capping %d GB -> %d GB "
                    "(SCX_BENCH_OOC_MEM_CAP_GB, forcing out-of-core)",
                    benchmark, format_key, dataset.name, total, cap_gb,
                )
                return cap_gb
    # Clamp at the cluster's largest single-task allocation. Combinations
    # whose true footprint exceeds this (e.g. dense-h5ad read on census_5m)
    # would OOM either way; the cap keeps submitit from rejecting the job.
    # Emit a warning so an orchestrator log search for "clamped" surfaces
    # every cell where the estimate was truncated.
    if total > MEM_CEILING_GB:
        logger.warning(
            "estimate_memory_gb(%s/%s/%s): %d GB clamped to MEM_CEILING_GB=%d GB "
            "(true peak may OOM)",
            benchmark, format_key, dataset.name, total, MEM_CEILING_GB,
        )
        return MEM_CEILING_GB
    return total


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
        # Bumped from 25 → 40 after the 2026-05-09 tier-full gate run
        # timed out 3 `parallel_write_scaling/{smartseq2,census_500k,census_1m}/
        # tiledb_soma` cells at the 25-min SLURM budget. The benchmark loops
        # 6 thread counts × (1 warmup + N timed runs); tiledb_soma writes
        # are ~25 s on smartseq2 and >4 min on census_1m, so the per-cell
        # wallclock fanout dwarfs the prior 25-min floor. tiledb_soma also
        # picks up the 2× format multiplier at 500K+ cells below.
        "parallel_write_scaling": 40,
        "memory":                 15,
        # Grouped sharding runs sort + a forced one-pass + two-pass convert.
        # The dense one-pass grouped gather (random full-width row reads) is the
        # slow scenario — minutes on a real Perturb-seq file even at the
        # in-module _MAX_CONVERT_RUNS=2 cap — so the base is generous.
        # 40 -> 60 for the `convert_sort_by` arm: a fourth convert scenario at
        # the same 2-run cap, i.e. two more whole-file conversions of a
        # multi-GB Perturb-seq h5ad.
        "grouped_sort":           60,
        # Cross-format grouped read/write head-to-head (scx_auto + shardad).
        # Times a grouped write per format plus per-perturbation read_group
        # over a handful of labels; the shardad arm materializes full groups,
        # so keep the base generous like grouped_sort.
        "grouped_read":           40,
        "cloud_push":             20,
        "cloud_pull":             20,
        "cloud_read":             20,
        # 5 -> 25 for the `scx_info_cloud` subprocess arm. The in-process open
        # is a single GET; `scx info` range-reads every CSR shard's header at
        # concurrency 8, which was measured at ~175 s on the largest fixture
        # pre-OPT-CLOUD-1, and the arm runs it n_runs times.
        "cloud_metadata":         25,
        "cloud_filtered":         20,
        "cloud_reader_vs_pull":   25,
        "cost_model":             20,
        "cloud_large_atlas":      60,   # 50GB+ pull is not quick
        # Bumped from 45 → 60 after the 2026-05-09 tier-full gate run
        # timed out `ml_loader/h5ad_gzip/census_1m` at 85 min wallclock.
        # The full 1M-cell h5ad-gzip read is the slowest combination
        # in the suite. Base bumped 60→120 (2026-07-10): the workers2 /
        # gpu_train / competitor-gather scenarios pushed smartseq2/tabula/census
        # past the old 65-min cap (base 60 + 12-min/M slope rounds to 65 for
        # ≤100k-cell sets), timing out the deferred auto_v2 + cellstream floors.
        # 120 base + 12-min/M slope + 1.5× density gives comfortable headroom.
        # 120 -> 165 for the two `raw_obs_*` cardinality arms on census_1m:
        # 2 arms x (1 untimed warm + n_runs timed) epochs. Cheap warm, but a
        # page-cache-cold census_1m epoch measured ~1.15 batches/s (~850 s),
        # and the high-cardinality arm adds a measured 7.86 ms/batch on top.
        "ml_loader":              165,
        "correctness":            60,
        "roundtrip":              10,
        "cell_eval_parity_perf":  60,
        # Default 15-min fall-through clipped all 6 index_plan/scx_auto cells
        # on the 2026-05-10 tier-full run. The workers2 path is ~50% slower
        # after the v2-catalog regression (see
        # results/justifications/index_plan_workers2_v2_catalog.md), and the
        # cell runs workers0 + workers2 + the streaming dataset training path
        # back-to-back. Bumped 15→40 then 40→75 after run #4 still hit the
        # 40-min timeout on smartseq2/tabula_100k/census_500k/census_1m —
        # smartseq2's high n_vars (61497) drives the loader to fall back to
        # per-batch sizes well under the 512 MB internal budget, slowing
        # iteration substantially. Run #5 caught tabula_sapiens_100k (75:11
        # elapsed on a 75-min cap) and census_500k (80:05 on 80-min) just
        # over budget — bump base 75→120 to give the larger datasets clear
        # headroom on top of the 8 min/M slope.
        "index_plan":            120,
        # Phase 9.3 (post-Tier-3 findings 3 + follow-up). Generous bases
        # because (a) longer timeouts don't hurt queue priority on this
        # cluster, (b) over-budgeting once beats serial retries on timeout
        # flakes. scanpy UMAP on census_1m needed >55 min at 3 runs; give
        # CPU-reference cells clear headroom. All lines add base + default
        # slope (8 min/M cells) per (n_obs / 1M) — census_1m hits these
        # ceilings for the CPU reference implementations.
        "accel_pca":              30,
        "accel_knn":              60,
        "accel_umap":            120,
        "accel_leiden":           90,
        # Harmony runs `max_iter` x `max_iter_kmeans` = 60 sub-iterations over
        # an N x n_comps embedding, three arms, and the harmonypy CPU reference
        # is by far the slowest (docs/performance.md: R harmony 80.5 min where
        # scx-accel is minutes). Budgeted alongside accel_leiden rather than at
        # the 15-min fall-through.
        "accel_harmony":          90,
        "accel_preprocess":       60,
        "accel_hvg":              45,
        # V3 task 2.7 — end-to-end PCA→kNN→UMAP residency benchmark runs all
        # three stages back-to-back per variant (the CPU reference is the
        # slowest), so budget ≈ accel_pca + accel_knn + accel_umap (30+60+120)
        # rather than the 15-min fall-through, which would time out the
        # `pyscx_cpu` variant on the larger tiers.
        "accel_pipeline":        210,
        # Same full RAPIDS pipeline per variant as accel_pipeline, plus a fresh
        # load-to-GPU and a one-shot scx_auto conversion during fixture prep;
        # budget like accel_pipeline + headroom so the larger tiers don't hit
        # the 15-min fall-through default.
        "accel_format_pipeline": 240,
        # PR G1 — pdex_ref + Wilcoxon rank_genes_groups. Per-variant
        # work is bounded by the rank-test + sort × n_genes inner loop;
        # the GPU path's BlockRadixSort caps the per-gene pool at 8192
        # cells, so large datasets short-circuit early. Reference
        # scanpy run dominates the CPU path on smaller datasets.
        "accel_de":               45,
        # GPU NB-GLM: the GPU variant runs pdex_nb_glm gpu×reps + cpu×reps + a
        # pdex_ref anchor. The CPU pdex_nb_glm path at the many-perturbation
        # scale (100 perts × 18K genes) costs tens of seconds per call, so the
        # CPU-baseline + concordance triple-run dominates. Generous base.
        "accel_de_nb_glm":        60,
        # GPU eval metrics: the GPU variant runs one op gpu×reps + cpu×reps on a
        # paired real/pred fixture. energy_distance is O(N²) in cells per pert
        # on the CPU baseline, so the CPU triple-run dominates at the larger
        # synthetic tiers; the gate tier (pert_synth_10k) is fast.
        "accel_eval_metrics":     60,
        # CSC dispatch sweep runs eight ops (qc/hvg/de/pseudobulk × csr/csc)
        # against a converted CSC-equipped fixture; one-shot conversion is
        # cached per dataset so the per-variant work is bounded by the op
        # itself.
        "bench_csc_dispatch":     45,
        # Phase K — multimodal benchmarks. Five-format compression sweep
        # finishes in <1 min on real CITE-seq / Multiome fixtures; the
        # training benchmark runs a 100-batch loop + TTFB warm-up, well
        # within 30 minutes even at Multiome's 144K-feature ATAC width.
        "multimodal_compression": 10,
        "multimodal_training":    30,
        # Data-load Phase 0. `ooc_loader` drops the page cache before every
        # timed epoch, so each run re-reads from disk (no warm reuse) across up
        # to 6 loader arms × 2 scenarios × n_runs — generous base + a steeper
        # cold-read slope below. `cellset_gather` is SCX-only, S=64 gather over
        # scaled batch counts. `obs_open` is a fast open→read-one-column probe
        # (its manifest scenario scales with file count, not n_obs).
        "ooc_loader":             90,
        # 45 -> 55 for the `collate_rust` arm: one untimed gather of ten
        # batches (the kernel is timed alone, so they must be resident first)
        # plus 1 + n_runs collate passes over them, ~1 s each.
        "cellset_gather":         55,
        "obs_open":               20,
        # Data-load Phase 1D. Up to seven full file rewrites (two timed
        # `shuffle_write` reps + one per codec variant in the size sweep) plus
        # cache-cold TrainingDataset epochs on two files. The rewrite count is
        # what makes the base generous; see the steep slope below.
        "shuffle_layout":         60,
        # Streaming ingest / export. Each runs 2-4 arms, each in its own
        # subprocess, each n_runs times, and each arm rewrites or re-reads the
        # whole matrix. `export_streaming`'s `streaming_full` arm is the slowest
        # per run — 35-38 s against the plain arm's 13-15 s at
        # tabula_sapiens_100k, because it writes X, raw, two obsm matrices and a
        # layer instead of X alone.
        #
        # Measured end-to-end at tabula_sapiens_100k: ~4 min for all four
        # export arms at N_RUNS_LARGE=3. The base is generous against that on
        # purpose — these run on `cpu_preemptible`, where a job killed at the
        # timeout costs the whole capture and an over-request costs queue
        # position. Both previously fell through to the 15-min default, which
        # does not cover a single census_1m materialize run.
        "conversion_streaming":   90,
        "export_streaming":       90,
        # Five ops x n_runs, and two of them are whole-file rewrites. Measured
        # at tabula_sapiens_100k: `compact` 154-169 s per run, `compact_full`
        # 244-295 s per run (it rewrites a fixture 2.75x the size, carrying a
        # layer and two obsm matrices). At N_RUNS_LARGE=3 those two arms alone
        # are ~21 min, which the 15-min fall-through default does not cover.
        #
        # 45 -> 55 for the `obs_import` arm: it materialises the obs index as
        # Python strings once (`read_obs([])`, measured 5.0 s at census_1m),
        # writes a 1M-row CSV, and then per run copies the whole file, runs an
        # untimed `dry_run` join and an import that rehashes it (19.7 s per run
        # at census_1m). An earlier version of this change added a SECOND
        # `"fragment_ops"` entry earlier in this same dict, which Python
        # silently discarded in favour of this one — so the +10 had no effect
        # at all. `test_no_duplicate_keys_in_config_tables` now catches that.
        "fragment_ops":           55,
        # One in-place sidecar build per run, plus a file copy per run outside
        # the timed region. O(nnz) with a column-major transpose on top.
        # Captured (`results/baselines/LATEST`): 29.3 s per run at
        # tabula_sapiens_100k (195M nnz), 201 s at census_500k (747M nnz) and
        # 1308 s at census_1m (1.40B) — ~0.27 us per non-zero. At
        # N_RUNS_LARGE=3 that is ~1.5 min and ~10 min, so the base alone covers
        # tabula but not census; the slope below is deliberately far more
        # generous than these numbers require (see there for why).
        "build_csc":              45,
        # O(nnz) over a gzipped *text* triplet and neither direction streams —
        # 3.5 us per non-zero on export, against `build_csc`'s captured 0.27,
        # which makes this the slowest per-non-zero path in the suite. Measured
        # on pbmc3k
        # (2,286,884 nnz): to_mtx 287k nnz/s, from_mtx 450k nnz/s. At
        # tabula_sapiens_100k (195M nnz) that is ~11 min per export and ~7 min
        # per ingest, so N_RUNS_LARGE=3 exports + 1 deletion export + 3 ingests
        # is ~67 min. The module caps itself at n_obs <= 100,000 for the same
        # arithmetic — census_500k would be about an hour per export.
        "mtx_export":            150,
    }
    base = base_minutes.get(benchmark, 15)

    # Per-million-cells multiplier. Cloud ops scale linearly with bytes
    # downloaded; cost_model / cloud_large_atlas scale heavily.
    slope_minutes_per_million = 8
    if benchmark in ("cloud_large_atlas",):
        slope_minutes_per_million = 30
    elif benchmark == "ooc_loader":
        # Cold-cache re-reads from disk every epoch; census_5m/10m cold reads
        # dominate. Steeper than the 8/M default so the larger OOC tiers don't
        # clip.
        slope_minutes_per_million = 15
    elif benchmark == "build_csc":
        # NB this slope is per million **cells** (`per_million = n_obs / 1e6`),
        # while the op is O(nnz). The two coincide only at a fixed density, so
        # the number below is calibrated at census density.
        #
        # Captured (`results/baselines/LATEST`, median of 3): 29.3 s per run at
        # tabula_sapiens_100k (100K cells, 195M nnz), 201 s at census_500k
        # (500K cells, 747M nnz) and 1308 s at census_1m (1M cells, 1.40B nnz).
        # At N_RUNS_LARGE=3 plus a per-run `shutil.copy2` of the input (2.8 GB
        # at census_1m), that is ~1.5 / ~10 / ~70 min of real work.
        #
        # **180/M is a deliberate safety margin, not a derivation** — it budgets
        # 135 min for census_500k and 225 for census_1m, i.e. an order of
        # magnitude over the captured wall. It is kept because a bespoke
        # one-run-per-budget sweep on the same datasets measured 259 s at
        # tabula and 1630 s at census_500k — ~9x slower than LATEST, on the
        # same code — and that divergence has never been explained. (Its
        # peak-RSS half, quoted in thresholds.yaml's deferred-floor item 14,
        # DOES agree with LATEST, so it is wall-clock only; a thread-count or
        # page-cache difference is the obvious suspect and is unconfirmed.)
        # Under-provisioning kills a census-scale cell mid-capture, so the
        # slope stays at the pessimistic end until that is settled. Quote
        # LATEST for throughput, never the 259 s / 1630 s figures.
        slope_minutes_per_million = 180
    elif benchmark == "shuffle_layout":
        # Every arm is O(nnz): each rewrite decodes and re-encodes the whole
        # matrix, and the size sweep does that once per codec variant. Steeper
        # than ooc_loader's cold-read slope because a rewrite is read + encode,
        # not read alone.
        slope_minutes_per_million = 25
    elif benchmark in ("cloud_push", "cloud_pull", "cloud_read",
                        "cloud_reader_vs_pull", "cost_model"):
        slope_minutes_per_million = 12
    elif benchmark in ("cloud_metadata", "cloud_filtered"):
        slope_minutes_per_million = 4
    elif benchmark in ("compression", "read_selective"):
        slope_minutes_per_million = 2
    elif benchmark == "ml_loader":
        # 4 CPU scenarios + 1 GPU scenario × n_runs each; epoch wall time
        # scales near-linearly with n_obs for streaming loaders.
        slope_minutes_per_million = 12
    elif benchmark == "correctness":
        # 14+ scanpy-equivalence checks plus SCX backed-mode and SLAF
        # round-trip — UMAP/Leiden/DE pipelines dominate at scale.
        # Empirical: pbmc3k <30 min; tabula_sapiens_100k ~80-120 min.
        slope_minutes_per_million = 240
    elif benchmark == "fragment_ops":
        # append + delete + compact + rollback, each repeated n_runs times.
        # append decodes+re-encodes the full input and compact rewrites the
        # whole file, so wall time scales ~linearly with nnz; the default
        # 8 min/M is too tight at census scale. Size generously so census_1m
        # gets ~2 h and census_5m scales up. (At ≤100K the op is well under the
        # 15-min base, so the small gated datasets — pbmc3k/pbmc10k/smartseq2 —
        # are unaffected.) NB: the census failures observed in the 2026-06-11
        # gate were a per-shard value_encoding bug in pyscx.append (fixed in
        # scx-ops), not a time-limit; this slope is forward-looking headroom.
        slope_minutes_per_million = 120
    elif benchmark == "roundtrip":
        # Two full reads (anndata.read_h5ad + pyscx.open(...).to_anndata())
        # plus a CSR diff. Bounded by anndata's h5ad parse on the source
        # side, which dominates at census scale. Sized at 2× read_full's
        # default 8 min/M slope to cover both reads with headroom — the
        # earlier 4 min/M was anchored on the small (≤100K) datasets in
        # the floor list and would clip if anyone extends roundtrip to
        # 1M+ datasets.
        slope_minutes_per_million = 16
    elif benchmark == "cellset_gather":
        # Scattered cold gather, and since the collate arm landed, ten gathered
        # batches held resident on top. The 8/M default budgeted 60 min for
        # census_500k; the 2026-09-02 tier-full capture ran it 134 minutes and
        # SLURM killed it, taking a floored triple's rows with it — the collate
        # floors in the Deferred block are prescribed on census_500k. Same for
        # `census_1m/scx_fast`.
        #
        # 300/M is set from the one hard datum (>134 min at 0.5M) with margin,
        # not from a completed run: census_500k -> 205 min, census_1m -> 355.
        # Census-only, because `per_million` floors at 0.1: an unguarded 300/M
        # would add 30 minutes to a 2-minute pbmc3k cell, and 652 cohorts each
        # asking for more than they need slows the whole capture through
        # backfill scheduling. Same guard on the three below.
        # Replace it with a measured value once a census cell finishes. These
        # are ceilings and SLURM bills actual usage, so over-budgeting costs
        # scheduling priority while under-budgeting costs the row.
        slope_minutes_per_million = 300 if n_obs >= 500_000 else 8
    elif benchmark == "accel_knn":
        # HNSW build is superlinear in n_obs and the census cells are the whole
        # cost. The 8/M default gave census_500k 65 min and census_1m 70; both
        # timed out on 2026-09-02, on `pyscx_cpu` and `pyscx_gpu_no_rapids`
        # alike (the no-rapids variant is a CPU path by construction).
        # 200/M -> 160 / 260 min.
        slope_minutes_per_million = 200 if n_obs >= 500_000 else 8
    elif benchmark == "accel_harmony":
        # FLOOR-ONLY (`mean_per_pc_r_vs_harmonypy >= 0.99` on CPU+GPU x
        # pbmc3k+census_1m), so a timeout here costs a floor its only evidence.
        # The 8/M default gave census_1m 100 min and the CPU cell timed out
        # there on 2026-09-02. 200/M -> 290 min.
        slope_minutes_per_million = 200 if n_obs >= 500_000 else 8
    elif benchmark == "accel_de":
        # The `scanpy_wilcoxon_cpu` reference arm dominates at census scale — it
        # timed out at the 55 min the 8/M default allowed on census_1m
        # (2026-09-02). The pyscx arms are far quicker, but a cohort's budget
        # has to cover its slowest arm. 200/M -> 245 min.
        slope_minutes_per_million = 200 if n_obs >= 500_000 else 8
    elif benchmark == "cell_eval_parity_perf":
        # cell-eval's edistance reference is O(n_obs^2) pairwise distance
        # × n_perts × n_runs, so wall-time scales near-quadratically with
        # n_obs even at fixed n_perts=50. Empirical: 10K is ~30 min total,
        # 100K is ~150 min for blas_f32 alone. The non-marquee
        # energy_distance variants skip at >= 100K (see _SKIP_RULES).
        # 10K → 72 min, 100K → 180 min, 1M → blas_f32 also skipped at 500K.
        slope_minutes_per_million = 1200

    total = base + int(slope_minutes_per_million * per_million)
    # Dense-path formats (h5ad / zarr) take longer at census scale.
    if format_key.startswith(("h5ad", "zarr")) and n_obs >= 1_000_000:
        total = int(total * 1.5)
    # tiledb_soma writes are I/O-bound and scale faster than the default
    # 8 min/M slope: empirical 41 min on 500K, 82 min on 1M (vs ~25 s on
    # 75K). Without the multiplier, 500K and 1M cells exhaust the
    # SLURM budget before the 6× thread sweep finishes.
    if (
        benchmark == "parallel_write_scaling"
        and format_key == "tiledb_soma"
        and n_obs >= 500_000
    ):
        total = int(total * 2.5)

    # Round up to 5-min increments.
    return max(5, math.ceil(total / 5) * 5)


def partition_for_memory(mem_gb: int, default: str = "cpu_preemptible") -> str:
    """Auto-route to ``cpu_high_mem`` when ``mem_gb`` exceeds the preemptible
    cap, otherwise stay on ``default``.

    Honours `SCX_BENCH_HIGH_MEM_PARTITION` env var (default: ``cpu_high_mem``)
    so clusters without a `cpu_high_mem` partition (e.g. Lambda HPC, where
    the equivalent role is filled by `large_batch`) can redirect the
    auto-promotion target without touching the call sites.
    """
    if mem_gb > MEM_HIGH_MEM_THRESHOLD_GB:
        return os.environ.get("SCX_BENCH_HIGH_MEM_PARTITION", "cpu_high_mem")
    return default


# ---------------------------------------------------------------------------
# SLURM Defaults
# ---------------------------------------------------------------------------

# The GPU partition, overridable because the default starves. Chimera's
# `preemptible` GPU QOS can leave a job PENDING past the gate's own 600 s probe
# timeout, which reads as a pre-flight failure rather than a queue backlog, so a
# capture that has to finish points this at a priority partition
# (`ctc_gpu_priority`, `gpu`). It was a bare literal in two places — here and in
# `run_parallel._per_job_slurm_params`, where `--partition` and
# `SCX_BENCH_PARTITION` both had no effect on GPU cells — and the standing
# workaround was to edit that line and remember to revert it. Default unchanged,
# so nothing moves unless an operator says so.
GPU_PARTITION = os.environ.get("SCX_BENCH_GPU_PARTITION", "preemptible")

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
        "partition": GPU_PARTITION,
        "cpus_per_task": 16,
        "mem_gb": 128,
        "gpus": 1,
        "time": "04:00:00",
    },
}
