#!/usr/bin/env python3
"""
GCP auth preflight for the comprehensive cloud benchmarks (Phase I.0).

Verifies:

  1. ``GOOGLE_APPLICATION_CREDENTIALS`` is set (or defaults to
     ``~/.gcp/scx-bench.json`` when that file exists).
  2. The key file parses as JSON and names the expected service account.
  3. A read + write round-trip against
     ``gs://<GCS_TEST_BUCKET>/.healthcheck`` succeeds.

Invoked standalone to debug a misconfigured cluster / VM; also available
as a library via ``run_preflight()`` so callers (nightly or on-demand
gate wrappers) can gate cloud benchmarks on a live auth check.

Exit codes:
  0 — all checks passed.
  1 — a check failed (message on stderr explains which).
  2 — usage / argparse error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    GCP_PROJECT,
    GCS_TEST_BUCKET,
)

logger = logging.getLogger(__name__)

EXPECTED_SERVICE_ACCOUNT = f"scx-bench@{GCP_PROJECT}.iam.gserviceaccount.com"
_DEFAULT_KEY_PATH = Path.home() / ".gcp" / "scx-bench.json"


@dataclass
class PreflightResult:
    ok: bool
    failures: list[str]
    key_path: str | None
    service_account_email: str | None


def _resolve_credentials() -> tuple[str | None, list[str]]:
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    failures: list[str] = []
    if not creds and _DEFAULT_KEY_PATH.exists():
        creds = str(_DEFAULT_KEY_PATH)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    if not creds:
        failures.append(
            "GOOGLE_APPLICATION_CREDENTIALS is not set and no key was found at "
            f"{_DEFAULT_KEY_PATH}. Mint one via the bootstrap steps in "
            "docs/cloud.md."
        )
        return None, failures
    if not Path(creds).is_file():
        failures.append(
            f"GOOGLE_APPLICATION_CREDENTIALS points at {creds!r} but the file "
            "does not exist."
        )
        return None, failures
    return creds, failures


def _check_service_account(key_path: str) -> tuple[str | None, list[str]]:
    try:
        data = json.loads(Path(key_path).read_text())
    except json.JSONDecodeError as exc:
        return None, [f"{key_path} is not valid JSON: {exc}"]
    email = data.get("client_email")
    if not email:
        return None, [f"{key_path} does not include a `client_email` field."]
    if email != EXPECTED_SERVICE_ACCOUNT:
        return email, [
            f"Service account mismatch: {email!r} != expected "
            f"{EXPECTED_SERVICE_ACCOUNT!r}. Point the env var at the scx-bench "
            f"key, or update `GCP_PROJECT` in config.py if the project moved."
        ]
    return email, []


def _healthcheck_roundtrip(bucket: str, timeout_s: float = 30.0) -> list[str]:
    """Write a tiny marker, read it back, delete it — all via gsutil."""
    failures: list[str] = []
    payload = f"scx-bench healthcheck {time.time():.3f}\n"
    url = f"{bucket.rstrip('/')}/.healthcheck-{os.getpid()}"
    tmp_path = Path(tempfile.mkdtemp()) / "hc.txt"
    tmp_path.write_text(payload)
    try:
        up = subprocess.run(
            ["gsutil", "-q", "cp", str(tmp_path), url],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if up.returncode != 0:
            failures.append(
                f"gsutil upload to {url} failed: {up.stderr.strip() or up.stdout.strip()}"
            )
            return failures

        read = subprocess.run(
            ["gsutil", "-q", "cat", url],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if read.returncode != 0:
            failures.append(
                f"gsutil read-back of {url} failed: {read.stderr.strip()}"
            )
        elif read.stdout != payload:
            failures.append(
                f"healthcheck read-back does not match payload "
                f"(expected {payload!r}, got {read.stdout!r})"
            )
    except FileNotFoundError:
        failures.append(
            "gsutil not found on PATH. Install the Google Cloud SDK or run "
            "this preflight on a VM that has it pre-installed."
        )
    except subprocess.TimeoutExpired:
        failures.append(
            f"gsutil healthcheck timed out after {timeout_s}s. Check network "
            f"connectivity to GCS from this host."
        )
    finally:
        # Best-effort cleanup.
        subprocess.run(
            ["gsutil", "-q", "rm", url],
            capture_output=True, timeout=timeout_s,
        )
        try:
            tmp_path.unlink()
            tmp_path.parent.rmdir()
        except OSError:
            pass
    return failures


def run_preflight(*, skip_healthcheck: bool = False) -> PreflightResult:
    """Library entrypoint for the gate / benchmark runners.

    ``skip_healthcheck=True`` is useful for CI where the key validity matters
    but round-tripping a GCS object would cost a spurious request.
    """
    key_path, creds_failures = _resolve_credentials()
    if creds_failures:
        return PreflightResult(
            ok=False, failures=creds_failures,
            key_path=None, service_account_email=None,
        )

    email, sa_failures = _check_service_account(key_path)
    failures = list(sa_failures)
    if failures:
        return PreflightResult(
            ok=False, failures=failures,
            key_path=key_path, service_account_email=email,
        )

    if not skip_healthcheck:
        failures.extend(_healthcheck_roundtrip(GCS_TEST_BUCKET))

    return PreflightResult(
        ok=not failures, failures=failures,
        key_path=key_path, service_account_email=email,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-healthcheck", action="store_true",
        help="Skip the gs:// read/write round-trip (useful in CI).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print the resolved key path + service account on success.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = run_preflight(skip_healthcheck=args.skip_healthcheck)
    if result.ok:
        if args.verbose:
            print(f"OK: credentials at {result.key_path}")
            print(f"OK: service account {result.service_account_email}")
            print(f"OK: bucket {GCS_TEST_BUCKET} round-trip (or skipped)")
        else:
            print(f"GCP preflight OK ({result.service_account_email})")
        return 0

    print("GCP preflight FAILED:", file=sys.stderr)
    for f in result.failures:
        print(f"  - {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
