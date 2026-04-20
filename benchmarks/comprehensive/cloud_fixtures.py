"""
Cloud fixture management for the comprehensive benchmark suite.

GCP-only (see the "Cloud Benchmarks (GCP)" section of ``benchmarks/README.md``
for the workflow and the service-account bootstrap). This module provides
the minimal surface the cloud benchmarks need:

  * ``require_gcp_credentials`` — fail fast if auth isn't set up
  * ``cloud_path_exists`` — GCS existence probe via ``gsutil ls``
  * ``cloud_url_for`` — thin wrapper over ``DatasetConfig.cloud_url``
  * ``ensure_cloud_fixture`` — upload the local converted file/directory to
    the shared bucket if it isn't already there (lazy, one-time per fixture)

Fixtures live under ``gs://arc-ctc-nextflow/scx-test/`` by default (override
via ``GCS_TEST_BUCKET``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCP_PROJECT,
    GCS_TEST_BUCKET,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

_DEFAULT_KEY_PATH = Path.home() / ".gcp" / "scx-bench.json"


class CloudCredentialsError(RuntimeError):
    """Raised when GCP credentials are missing or unreadable."""


def require_gcp_credentials() -> str:
    """Ensure ``GOOGLE_APPLICATION_CREDENTIALS`` points at a readable key.

    If the env var is unset and ``~/.gcp/scx-bench.json`` exists, the env
    var is populated from that default path. Returns the resolved key path.

    Raises ``CloudCredentialsError`` on any failure so the calling benchmark
    aborts with a clear message instead of getting cryptic 401s mid-run.
    """
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds and _DEFAULT_KEY_PATH.exists():
        creds = str(_DEFAULT_KEY_PATH)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

    if not creds:
        raise CloudCredentialsError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set and no key was found at "
            f"{_DEFAULT_KEY_PATH}. See the 'Cloud Benchmarks (GCP)' section of "
            "benchmarks/README.md for the service-account bootstrap."
        )
    if not Path(creds).is_file():
        raise CloudCredentialsError(
            f"GOOGLE_APPLICATION_CREDENTIALS points at {creds!r} but the file "
            "does not exist."
        )
    return creds


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def cloud_url_for(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    provider: str = "gcs",
) -> str:
    """Return the canonical cloud URL for this ``(dataset, format)`` pair."""
    return dataset.cloud_url(format_variant.key, provider=provider)


def cloud_path_exists(url: str, timeout_s: float = 30.0) -> bool:
    """Check whether a GCS object/directory exists via ``gsutil ls``.

    Returns True if ``gsutil ls`` exits 0 with non-empty stdout. Any other
    outcome — missing binary, permission error, empty listing — is treated
    as "does not exist" so the caller can decide whether to upload.
    """
    try:
        result = subprocess.run(
            ["gsutil", "ls", url],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("gsutil ls %s failed: %s", url, exc)
        return False
    return result.returncode == 0 and result.stdout.strip() != ""


# ---------------------------------------------------------------------------
# Fixture upload
# ---------------------------------------------------------------------------


def _gsutil_rsync(local_dir: Path, cloud_url: str) -> None:
    """Upload a directory to GCS via ``gsutil -m rsync -r``.

    Used for non-SCX formats (Zarr / TileDB / SLAF) whose runners don't
    implement ``push``. Project flag is passed explicitly so billing lands
    on the right account regardless of the caller's default ``gcloud config``.
    """
    if shutil.which("gsutil") is None:
        raise RuntimeError(
            "gsutil is not on PATH — install the Google Cloud SDK or run "
            "from a node where it is available."
        )
    logger.info("Uploading %s → %s via gsutil rsync", local_dir, cloud_url)
    subprocess.run(
        [
            "gsutil",
            "-m",
            "-o", f"GSUtil:default_project_id={GCP_PROJECT}",
            "rsync",
            "-r",
            str(local_dir),
            cloud_url,
        ],
        check=True,
    )


def ensure_cloud_fixture(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    local_path: Path,
    provider: str = "gcs",
) -> str:
    """Return the cloud URL for the fixture, uploading it if absent.

    For SCX formats this imports ``pyscx`` and calls ``pyscx.push`` (produces
    the exploded ``.scxd/`` layout the readers expect). For Zarr / TileDB /
    SLAF the local directory is rsynced with ``gsutil -m``.

    Assumes ``require_gcp_credentials`` has already been called.
    """
    url = cloud_url_for(dataset, format_variant, provider=provider)
    if cloud_path_exists(url):
        logger.info("Fixture present, skipping upload: %s", url)
        return url

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local fixture not found for {dataset.name}/{format_variant.key}: "
            f"{local_path}"
        )

    if format_variant.key.startswith("scx_"):
        import pyscx

        logger.info("Uploading SCX fixture %s → %s via pyscx.push",
                    local_path, url)
        pyscx.push(str(local_path), url)
    else:
        _gsutil_rsync(local_path, url)
    return url


# ---------------------------------------------------------------------------
# Cloud I/O counters (Phase F.2)
# ---------------------------------------------------------------------------


class CloudIOCounters:
    """Lightweight accumulator for cloud-side transfer counters.

    Wraps a block of ``pyscx.pull`` / ``pyscx.pull_filtered`` calls and
    collects their ``bytes_downloaded`` / ``sections_downloaded`` (used as
    a GET-count proxy) into totals. Used by ``cost_model.py`` and
    ``cloud_reader_vs_pull.py`` to report per-operation egress cost.

    This is a Python-level accumulator — it does NOT instrument the
    ``object_store`` crate directly, so the GET count is a proxy (each
    section is one GET today). A Rust-side counting middleware wrapping
    ``Arc<dyn ObjectStore>`` would give exact per-request counters; that
    is tracked as a Phase F.2 follow-up so the Python surface can stay
    stable when the true counters land.
    """

    def __init__(self) -> None:
        self.bytes_downloaded: int = 0
        self.sections_downloaded: int = 0
        self.calls: int = 0

    def record_pull_stats(self, stats: dict) -> None:
        """Fold a single ``pyscx.pull`` (or ``pull_filtered``) stats dict."""
        self.bytes_downloaded += int(stats.get("bytes_downloaded", 0) or 0)
        # ``pull`` returns ``sections_downloaded``; ``pull_filtered`` returns
        # ``downloaded_shards`` + implicit catalog/header/obs/var GETs. For
        # the cost proxy we sum the explicit shard count plus 3 for the
        # catalog/header/obs fetches pull_filtered always issues.
        if "sections_downloaded" in stats:
            self.sections_downloaded += int(stats["sections_downloaded"])
        elif "downloaded_shards" in stats:
            self.sections_downloaded += int(stats["downloaded_shards"]) + 3
        self.calls += 1

    def as_dict(self) -> dict:
        return {
            "bytes_downloaded": self.bytes_downloaded,
            "get_count_proxy": self.sections_downloaded,
            "calls": self.calls,
            "telemetry_source": "pyscx_stats_dict",
        }

    def egress_cost_usd(
        self, *, cross_region: bool = False,
    ) -> float:
        """Compute egress cost in USD for the accumulated bytes.

        Same-region (GCE ↔ GCS in the same region) is free today; pass
        ``cross_region=True`` to price against the cross-continent rate —
        intended as a "what if we went cross-region" sanity check, not a
        real cost for runs pinned to the bucket region.
        """
        from benchmarks.comprehensive.config import GCS_PRICING

        rate_key = (
            "egress_cross_region_usd_per_gb" if cross_region
            else "egress_same_region_usd_per_gb"
        )
        gb = self.bytes_downloaded / (1024 ** 3)
        return gb * GCS_PRICING[rate_key]

    def request_cost_usd(self) -> float:
        """USD for the accumulated GET proxy under the Class-B rate."""
        from benchmarks.comprehensive.config import GCS_PRICING

        return (self.sections_downloaded / 10_000) * GCS_PRICING["class_b_per_10k_usd"]

    def total_cost_usd(self, *, cross_region: bool = False) -> float:
        return self.egress_cost_usd(cross_region=cross_region) + self.request_cost_usd()
