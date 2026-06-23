#!/usr/bin/env python3
"""Assert that ``.scx`` benchmark fixtures carry decode sidecars.

The per-row scx1 *decode sidecar* (``DecodeMetadataShard``, section id 26) is a
write-time property: it is emitted by default for Scx1 integer-CSR shards within
a 25% overhead budget, but ``.scx`` files written before scx 0.9.1 carry **none**.
Benchmarking the IndexPlan sidecar gather against a sidecar-less fixture shows no
change and looks like a failed fix (STATE-TX-SIDECAR.md §6.3 / T0.3) — so this
script is the pre-benchmark gate: it counts decode sidecars vs. CSR shards and
asserts coverage before any sidecar benchmark runs.

Detection uses ``scx info <path> --json``: the top-level ``n_csr_shards`` field
and the ``sections[]`` array, where ``DecodeMetadataShard`` entries render their
``type`` as the section name ``decode/<shard>`` (see scx-cli/src/info.rs).

Caveat: over-budget / non-integer / CSC shards legitimately lack a sidecar, so
for mixed fixtures full coverage (``n_sidecars == n_csr_shards``) is only
expected when every CSR shard is sidecar-eligible (the integer-count case, e.g.
the Replogle K562 / Census perturbation fixtures used by ``index_plan.py``).

Usage::

    # Assert one fixture has full coverage (exit 1 on failure).
    python benchmarks/scripts/check_sidecar_coverage.py path/to/fixture.scx

    # Report (don't assert) coverage for several fixtures.
    python benchmarks/scripts/check_sidecar_coverage.py --report a.scx b.scx

    # Use a specific scx binary.
    python benchmarks/scripts/check_sidecar_coverage.py --scx-bin target/release/scx f.scx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def sidecar_coverage(scx_path: str, scx_bin: str = "scx") -> tuple[int, int]:
    """Return ``(n_sidecars, n_csr_shards)`` for an ``.scx`` file.

    ``n_sidecars`` counts ``DecodeMetadataShard`` sections (``type`` starting
    with ``"decode/"``); ``n_csr_shards`` is the header CSR shard count. Raises
    ``subprocess.CalledProcessError`` if ``scx info`` fails, or ``KeyError`` if
    the JSON shape is unexpected.
    """
    proc = subprocess.run(
        [scx_bin, "info", scx_path, "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    n_csr = int(data["n_csr_shards"])
    n_sidecars = sum(
        1
        for s in data.get("sections", [])
        if str(s.get("type", "")).startswith("decode/")
    )
    return n_sidecars, n_csr


def assert_full_coverage(scx_path: str, scx_bin: str = "scx") -> tuple[int, int]:
    """Assert ``n_sidecars == n_csr_shards`` (and at least one shard exists)."""
    n_sidecars, n_csr = sidecar_coverage(scx_path, scx_bin=scx_bin)
    if n_csr == 0:
        raise AssertionError(f"{scx_path}: no CSR shards reported")
    if n_sidecars != n_csr:
        raise AssertionError(
            f"{scx_path}: sidecar-less fixture — {n_sidecars}/{n_csr} CSR shards "
            f"carry a decode sidecar. Regenerate under a post-0.9.1 writer "
            f"(benchmarks/scripts/reconvert_fixtures.py) before benchmarking."
        )
    return n_sidecars, n_csr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="one or more .scx fixture paths")
    ap.add_argument("--scx-bin", default="scx", help="scx binary (default: scx on PATH)")
    ap.add_argument(
        "--report",
        action="store_true",
        help="report coverage for every path without failing on partial coverage",
    )
    args = ap.parse_args()

    failures = 0
    for path in args.paths:
        try:
            n_sidecars, n_csr = sidecar_coverage(path, scx_bin=args.scx_bin)
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"ERROR {path}: {e}", file=sys.stderr)
            failures += 1
            continue
        full = n_sidecars == n_csr and n_csr > 0
        status = "OK" if full else "PARTIAL/NONE"
        print(f"{status:12s} {n_sidecars}/{n_csr} sidecars  {path}")
        if not full and not args.report:
            failures += 1

    if failures and not args.report:
        print(
            f"\n{failures} fixture(s) lack full sidecar coverage — "
            f"regenerate before benchmarking (see STATE-TX-SIDECAR.md T0.3).",
            file=sys.stderr,
        )
    return 1 if (failures and not args.report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
