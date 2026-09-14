#!/usr/bin/env python
"""Measure what a `SparseCellSetDataset` manifest costs, per open reader.

This is the Phase-2 (W8) gate capture, and it deliberately measures a
**resource**, not a rate.

# Why not the gate §12.4 asked for

§12.4 gated on a file-descriptor count: "5,000-file manifest constructs and
gathers under `ulimit -n 1024`". That gate passes on unmodified `main`.
`ScxReader::open` mmaps the file and lets the `File` drop, and `scx-loader`
never opens a *watching* reader, so a manifest of N files holds N mappings and
**zero** descriptors. This script records that directly (the `fds_*` fields) so
the claim is checkable rather than asserted, and then measures the term that is
real: resident bytes, over 90% of which is the parsed `FullCatalog`.

# Why an allocation measurement can run interactively

Every number here is a count — `VmRSS` deltas, `/proc/self/maps` lines,
`/proc/self/fd` entries — taken around a constructor that does no timing-
sensitive work. Neighbour load on a shared node moves wall-clock; it does not
move how many bytes a catalog parse allocates. The wall-clock fields are
recorded for context and are explicitly NOT a benchmark claim.

Usage:
    python measure_reader_registry_rss.py OUT.json [--n 1000]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import time

# One entry per arm. `None` is the default (open everything); the bounded arms
# are what the option is for.
LIMITS = [None, 256, 64, 16]


def _rss_kb() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not reported by this kernel")


def _maps() -> int:
    with open("/proc/self/maps") as fh:
        return sum(1 for _ in fh)


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _synthetic(tmp: pathlib.Path) -> str:
    """A file with ~100 CSR shards — catalog entries are what a reader costs."""
    import anndata
    import numpy as np
    import scipy.sparse as sp

    import pyscx

    path = str(tmp / "manyshard.scx")
    x = sp.random(2000, 200, density=0.05, format="csr", random_state=0, dtype=np.float32)
    pyscx.from_anndata(anndata.AnnData(x), path, shard_size=20)
    return path


def measure(path: str, n: int) -> list[dict]:
    """One arm per `reader_limit`, each in a clean-ish allocator state."""
    import pyscx

    rows = []
    for limit in LIMITS:
        gc.collect()
        r0, m0, f0 = _rss_kb(), _maps(), _fds()
        t0 = time.perf_counter()
        ds = pyscx.SparseCellSetDataset(paths=[path] * n, reader_limit=limit)
        elapsed = time.perf_counter() - t0
        r1, m1, f1 = _rss_kb(), _maps(), _fds()
        metrics = ds.cache_metrics()
        rows.append(
            {
                "reader_limit": limit,
                "n_files": n,
                "rss_delta_kb": r1 - r0,
                "rss_kb_per_file": round((r1 - r0) / n, 2),
                "vma_delta": m1 - m0,
                # The whole point of recording these: they do not move.
                "fds_before": f0,
                "fds_after": f1,
                "reader_opens": metrics["reader_opens"],
                "reader_evictions": metrics["reader_evictions"],
                "reader_resident": metrics["reader_resident"],
                "reader_hwm": metrics["reader_hwm"],
                "construct_s": round(elapsed, 4),
            }
        )
        del ds
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1000, help="manifest entries per arm")
    args = ap.parse_args()

    import tempfile

    import pyscx

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="w8-rss-"))
    data = os.environ.get("SCX_DATA_DIR", "/large_storage/arcinfra/projects/scx")
    # Real fixtures first; the synthetic one exists so the capture reproduces
    # from a clone, where the atlas files are not present.
    fixtures = {
        "tabula_sapiens_100k": f"{data}/tabula_sapiens_100k.scx",
        "census_1m_preprocessed": f"{data}/census_1m_preprocessed.scx",
        "synthetic_100shard": _synthetic(tmp),
    }

    out = {
        "provenance": {
            "what": (
                "Resident cost of a SparseCellSetDataset manifest, per open reader, "
                "at each reader_limit. Records file-descriptor and mmap counts "
                "alongside, because the phase's original premise was that "
                "descriptors were the binding resource and they are not."
            ),
            "not_a_throughput_claim": (
                "Every field except construct_s is a count taken around one "
                "constructor. construct_s is context only and carries no claim; "
                "these arms were not run as an isolated SLURM capture."
            ),
            "records_are_reduced": (
                "This file is NOT BenchmarkResult-shaped. It is one row per "
                "(fixture, reader_limit), reduced from a single construction each."
            ),
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
            "host": os.uname().nodename,
            "python": sys.version.split()[0],
            "pyscx": pyscx.__file__,
            "rlimit_nofile_soft": __import__("resource").getrlimit(
                __import__("resource").RLIMIT_NOFILE
            )[0],
            "vm_max_map_count": int(
                pathlib.Path("/proc/sys/vm/max_map_count").read_text().strip()
            ),
        },
        "arms": [],
    }
    for name, path in fixtures.items():
        if not os.path.exists(path):
            print(f"skip {name}: {path} not present", file=sys.stderr)
            continue
        for row in measure(path, args.n):
            row["fixture"] = name
            row["n_csr_shards"] = pyscx.open(path).shard_count
            out["arms"].append(row)
            print(json.dumps(row), flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out['arms'])} arms to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
