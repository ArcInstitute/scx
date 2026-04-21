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
import tempfile
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

    The env var is sourced first from the process environment and then from
    the repo-root ``.env`` (loaded automatically via
    ``benchmarks/scripts/bench_env.py`` at import time, since ``config.py``
    depends on it). Tilde prefixes are expanded — ``.env`` entries like
    ``GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/scx-bench.json`` are supported.
    If the env var is unset and ``~/.gcp/scx-bench.json`` exists, the env
    var is populated from that default path. Returns the resolved key path.

    Raises ``CloudCredentialsError`` on any failure so the calling benchmark
    aborts with a clear message instead of getting cryptic 401s mid-run.
    """
    creds_raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or ""
    creds = os.path.expanduser(creds_raw) if creds_raw else ""

    if not creds and _DEFAULT_KEY_PATH.exists():
        creds = str(_DEFAULT_KEY_PATH)

    if not creds:
        raise CloudCredentialsError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set and no key was found at "
            f"{_DEFAULT_KEY_PATH}. Add the line to the repo-root .env file "
            "(see .env.example) or mint a key via the bootstrap in docs/cloud.md."
        )
    if not Path(creds).is_file():
        raise CloudCredentialsError(
            f"GOOGLE_APPLICATION_CREDENTIALS resolves to {creds!r} but the "
            "file does not exist."
        )

    # Write the expanded absolute path back so subprocess callers (gsutil,
    # pyscx's Rust cloud backend) see a concrete path rather than a tilde
    # they might not expand consistently.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
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


def _upload_lock_path(dataset: DatasetConfig, format_variant: FormatVariant) -> Path:
    """Per-fixture lock file under /tmp so parallel submitit jobs serialize."""
    return Path(tempfile.gettempdir()) / (
        f"scx_bench_cloud_upload__{dataset.name}__{format_variant.key}.lock"
    )


def _is_fixture_complete(url: str, timeout_s: float = 15.0) -> bool:
    """Return True if the fixture URL + its ``.blake3`` sidecar both exist.

    The sidecar is written by ``_write_blake3_sidecar`` AFTER the main
    upload succeeds, so it functions as a "upload complete" marker.
    Distinguishes a partial upload in progress (url populated, no sidecar)
    from a finished upload ready to read (url populated + sidecar present).
    """
    sidecar = url.rstrip("/") + ".blake3"
    try:
        result = subprocess.run(
            ["gsutil", "ls", sidecar],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() != ""


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

    Concurrency: parallel submitit jobs all reach this function at once.
    Without coordination the shape was a TOCTOU race — every job saw
    ``cloud_path_exists(url) == False`` (fixture not yet uploaded) and
    launched its own ``gsutil rsync`` against the same GCS path. gsutil
    doesn't coordinate writes across processes, so chunks got partially
    overwritten (observed: zarr chunk 1 inflated from 585 KB to 1058 KB —
    two overlapping uploads interleaved). An ``fcntl.flock`` on a
    per-fixture lock file under ``/tmp`` makes uploads strictly serial:
    the first job acquires the lock and uploads, later jobs block on the
    lock, then re-check completion and skip. Completion is probed via the
    ``.blake3`` sidecar (written last) rather than the fixture URL itself
    — a partial upload has the URL populated but no sidecar, so readers
    correctly wait rather than race an unfinished upload.

    Assumes ``require_gcp_credentials`` has already been called.
    """
    url = cloud_url_for(dataset, format_variant, provider=provider)
    if _is_fixture_complete(url):
        logger.info("Fixture present and complete, skipping upload: %s", url)
        return url

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local fixture not found for {dataset.name}/{format_variant.key}: "
            f"{local_path}"
        )

    import fcntl
    lock_path = _upload_lock_path(dataset, format_variant)
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r") as lock_fh:
        logger.info(
            "Acquiring upload lock %s (may block on concurrent jobs)…", lock_path,
        )
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        # Re-check after acquiring the lock — a sibling job may have just
        # completed the upload while we were waiting.
        if _is_fixture_complete(url):
            logger.info(
                "Fixture uploaded by another job while we waited: %s", url,
            )
            return url

        if format_variant.key.startswith("scx_"):
            import pyscx

            logger.info("Uploading SCX fixture %s → %s via pyscx.push",
                        local_path, url)
            pyscx.push(str(local_path), url)
        else:
            _gsutil_rsync(local_path, url)

    # Write a BLAKE3 fingerprint sidecar alongside the fixture so
    # `setup_cloud_test_data.sh` and `cloud_fixtures_doctor.py` can detect
    # stale cloud copies without re-uploading to find out.
    try:
        _write_blake3_sidecar(local_path, url)
    except Exception as exc:  # noqa: BLE001 — sidecar is best-effort
        logger.warning("failed to write BLAKE3 sidecar for %s: %s", url, exc)

    return url


# ---------------------------------------------------------------------------
# BLAKE3 fingerprinting (Phase I.4)
# ---------------------------------------------------------------------------


def compute_blake3(path: Path, _chunk_size: int = 1 << 20) -> str:
    """Return the BLAKE3 hex digest of a file or a directory tree.

    For directories, hashes the sorted list of relative paths + per-file
    content — the same algorithm used by ``MANIFEST.sha256`` semantics,
    adapted to BLAKE3 for speed. Stable across file-order changes since
    the recursion is sorted.
    """
    try:
        import blake3  # type: ignore[import-not-found]
        hasher = blake3.blake3()
    except ImportError:
        # Fallback to hashlib.blake2b — 64-bit digest truncated so the
        # string shape matches BLAKE3's 32-byte digest. The harness env
        # has blake3 via pyscx's deps; the fallback is for clean envs.
        import hashlib
        hasher = hashlib.blake2b(digest_size=32)

    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    if path.is_dir():
        for entry in sorted(path.rglob("*")):
            if not entry.is_file():
                continue
            rel = str(entry.relative_to(path)).encode()
            hasher.update(b"\x00" + rel + b"\x00")
            with open(entry, "rb") as f:
                for chunk in iter(lambda: f.read(_chunk_size), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()

    raise FileNotFoundError(f"not a file or directory: {path}")


def _write_blake3_sidecar(local_path: Path, cloud_url: str) -> None:
    """Upload a ``.blake3`` sidecar to ``{cloud_url}/.blake3`` — one-line file.

    Sidecar path: ``<cloud_url trailing slash normalized>.blake3``. The
    doctor script reads this to detect drift. Best-effort; never blocks
    the main upload.
    """
    digest = compute_blake3(local_path)
    sidecar_url = cloud_url.rstrip("/") + ".blake3"
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="scx_blake3_")
    sidecar_local = Path(tmp_dir) / "sidecar"
    sidecar_local.write_text(digest + "\n")
    try:
        subprocess.run(
            ["gsutil", "-q", "cp", str(sidecar_local), sidecar_url],
            check=False, timeout=30,
        )
    finally:
        try:
            sidecar_local.unlink()
            Path(tmp_dir).rmdir()
        except OSError:
            pass


def read_blake3_sidecar(cloud_url: str, timeout_s: float = 15.0) -> str | None:
    """Fetch ``{cloud_url}.blake3`` contents, or ``None`` if absent."""
    sidecar_url = cloud_url.rstrip("/") + ".blake3"
    try:
        out = subprocess.run(
            ["gsutil", "-q", "cat", sidecar_url],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


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
