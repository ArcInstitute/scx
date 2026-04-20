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
