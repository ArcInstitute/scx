#!/usr/bin/env python3
"""Phase 7 — Multimodal query pushdown vs `scx subset` micro-benchmark.

Two ways to obtain a *filtered single-modality* result from a multimodal
(CITE-seq–shaped) SCX file:

  - **PUSHDOWN** — ``scx query f --modality M --filter P --output q.scx``.
    The obs predicate resolves against the shared global obs axis; Level-1
    catalog pruning + per-shard decode-and-filter assemble **only the matching
    cells' rows** of modality ``M`` (`QueryPipeline::collect` never holds the
    whole modality in memory), then writes the filtered result.

  - **SUBSET** (the pre-pushdown workaround) — ``scx subset f out.scx
    --modality M --filter P``. `extract_modality` reads the **entire** modality
    X into memory (`read_all_csr_shards_for`), masks rows in Rust, and writes a
    whole new single-modality SCX file.

Both paths are the **same `scx` CLI binary** — an apples-to-apples comparison
of pushdown vs full-materialization with no Python-interpreter RSS baseline to
confound it (the pyscx `query(modality=…).collect()` API does the identical
engine work; it just returns an in-memory AnnData instead of writing a file).

Each path runs in its own subprocess under ``/usr/bin/time -v`` so peak RSS
(Maximum resident set size) is measured for the process that does the work.
Wall is the subprocess elapsed time (min over repeats); RSS is the max.

Build a **release** `scx` binary first (a debug binary inflates timings 4–10×);
the fixture builder uses whatever pyscx is installed in ``--python-bin`` (build
time is not measured):

    cargo build --release -p scx-cli

Then, e.g.:

    ./.venv/bin/python benchmarks/multimodal_query_bench.py \
        --n-cells 200000 --rna-vars 3000 --adt-vars 30 --density 0.05 \
        --filter "cell_type == 'rare'" --scx-bin target/release/scx \
        --fixture-dir /scratch/$(id -u)/scx_mm_query --output out.json

Output goes to stdout (and optional ``--output FILE.json``).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def build_fixture(
    path: Path,
    n_cells: int,
    rna_vars: int,
    adt_vars: int,
    density: float,
    rare_frac: float,
    seed: int,
) -> None:
    """Write a 2-modality (rna/adt) SCX file with a global ``cell_type`` obs
    column. ``cell_type`` is one of {common, mid, rare}; ``rare`` is present in
    ~``rare_frac`` of cells so a selective filter exercises the pushdown win."""
    import anndata
    import mudata
    import scipy.sparse as sp

    import pyscx

    rng = np.random.default_rng(seed)
    rna = sp.random(
        n_cells, rna_vars, density=density, format="csr", dtype=np.float32,
        random_state=rng,
    )
    rna.data = np.ceil(rna.data * 10).astype(np.float32)
    # ADT is small + dense-ish (protein panel).
    adt = sp.random(
        n_cells, adt_vars, density=0.5, format="csr", dtype=np.float32,
        random_state=rng,
    )
    adt.data = np.ceil(adt.data * 10).astype(np.float32)

    rna_ad = anndata.AnnData(X=rna)
    rna_ad.var_names = [f"g{i}" for i in range(rna_vars)]
    adt_ad = anndata.AnnData(X=adt)
    adt_ad.var_names = [f"a{i}" for i in range(adt_vars)]

    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_cells)]
    # cell_type: rare_frac "rare", then split the rest common/mid.
    u = rng.random(n_cells)
    cell_type = np.where(
        u < rare_frac, "rare", np.where(u < 0.5 + rare_frac / 2, "common", "mid")
    )
    mu.obs["cell_type"] = cell_type.tolist()
    pyscx.from_mudata(mu, str(path), codec="auto")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def run_measured(cmd: list[str], env: dict | None = None) -> tuple[float, float, str]:
    """Run ``cmd`` under ``/usr/bin/time -v``. Return (wall_s, peak_rss_mb, tail).

    Wall is Python-timed around the subprocess; peak RSS is parsed from
    ``/usr/bin/time -v`` stderr. Raises on non-zero exit."""
    full = ["/usr/bin/time", "-v", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(full, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )
    m = _RSS_RE.search(proc.stderr)
    if not m:
        raise RuntimeError(f"could not parse Max RSS from /usr/bin/time output:\n{proc.stderr[-2000:]}")
    peak_rss_mb = int(m.group(1)) / 1024.0
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    return wall, peak_rss_mb, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-cells", type=int, default=200_000)
    ap.add_argument("--rna-vars", type=int, default=3000)
    ap.add_argument("--adt-vars", type=int, default=30)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--rare-frac", type=float, default=0.05,
                    help="fraction of cells with cell_type=='rare'")
    ap.add_argument("--modality", default="rna")
    ap.add_argument("--filter", default="cell_type == 'rare'",
                    help="obs predicate applied by both paths")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scx-bin", default="target/release/scx",
                    help="path to the `scx` CLI binary (release recommended)")
    ap.add_argument("--fixture-dir", default=None,
                    help="dir to hold the .scx fixture (default: a tempdir)")
    ap.add_argument("--output", default=None, help="write JSON results here")
    args = ap.parse_args()

    tmp = None
    if args.fixture_dir:
        fixture_dir = Path(args.fixture_dir)
        fixture_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory()
        fixture_dir = Path(tmp.name)

    scx_bin = Path(args.scx_bin)
    if not scx_bin.exists():
        print(f"ERROR: scx binary not found at {scx_bin}; run `cargo build --release -p scx-cli`",
              file=sys.stderr)
        return 2

    tag = f"mm_{args.n_cells}_{args.rna_vars}x{args.adt_vars}_d{args.density}"
    fixture = fixture_dir / f"{tag}.scx"
    if not fixture.exists():
        print(f"[build] fixture {fixture} ({args.n_cells} cells, "
              f"rna {args.rna_vars} / adt {args.adt_vars} vars, density {args.density})...",
              flush=True)
        t0 = time.perf_counter()
        build_fixture(fixture, args.n_cells, args.rna_vars, args.adt_vars,
                      args.density, args.rare_frac, args.seed)
        print(f"[build] done in {time.perf_counter() - t0:.1f}s "
              f"({fixture.stat().st_size / 1e6:.0f} MB)", flush=True)
    else:
        print(f"[build] reusing fixture {fixture}", flush=True)

    out_q = fixture_dir / f"{tag}_query_out.scx"
    out_s = fixture_dir / f"{tag}_subset_out.scx"

    def measure_cli(out: Path, cmd_for: "callable", label: str) -> dict:
        """Run `cmd_for(out)` `repeats` times under /usr/bin/time -v, removing
        `out` before each run so the write path is exercised every iteration."""
        print(f"[run] {label} x{args.repeats}...", flush=True)
        walls, rsses, tail = [], [], ""
        for _ in range(args.repeats):
            if out.exists():
                out.unlink()
            w, r, tail = run_measured(cmd_for(out))
            walls.append(w)
            rsses.append(r)
        return {
            "wall_s_min": min(walls),
            "wall_s_all": walls,
            "peak_rss_mb_max": max(rsses),
            "peak_rss_mb_all": rsses,
            "tail": tail,
        }

    # PUSHDOWN: `scx query --modality --filter --output` — matched rows only.
    query = measure_cli(
        out_q,
        lambda o: [str(scx_bin), "query", str(fixture), "--modality", args.modality,
                   "--filter", args.filter, "--output", str(o)],
        "scx query (pushdown)",
    )
    # SUBSET: full modality materialization + rewrite to a new file.
    subset = measure_cli(
        out_s,
        lambda o: [str(scx_bin), "subset", str(fixture), str(o),
                   "--modality", args.modality, "--filter", args.filter],
        "scx subset (materialize)",
    )

    result = {
        "config": {
            "n_cells": args.n_cells, "rna_vars": args.rna_vars,
            "adt_vars": args.adt_vars, "density": args.density,
            "rare_frac": args.rare_frac, "modality": args.modality,
            "filter": args.filter, "repeats": args.repeats,
            "fixture_mb": round(fixture.stat().st_size / 1e6, 1),
        },
        "pushdown_query": query,
        "subset_workaround": subset,
        "speedup_wall": round(subset["wall_s_min"] / query["wall_s_min"], 2),
        "rss_reduction": round(subset["peak_rss_mb_max"] / query["peak_rss_mb_max"], 2),
    }

    print("\n=== Multimodal query pushdown vs `scx subset` ===")
    cfg = result["config"]
    print(f"fixture: {cfg['n_cells']} cells, rna {cfg['rna_vars']} / adt "
          f"{cfg['adt_vars']} vars, density {cfg['density']}, {cfg['fixture_mb']} MB")
    print(f"filter : {cfg['filter']}  (modality={cfg['modality']})")
    print(f"{'path':<30}{'wall (s)':>12}{'peak RSS (MB)':>16}")
    print(f"{'scx query (pushdown)':<30}{query['wall_s_min']:>12.3f}"
          f"{query['peak_rss_mb_max']:>16.1f}")
    print(f"{'scx subset (materialize)':<30}{subset['wall_s_min']:>12.3f}"
          f"{subset['peak_rss_mb_max']:>16.1f}")
    print(f"\npushdown is {result['speedup_wall']}x faster (wall), "
          f"{result['rss_reduction']}x lower peak RSS")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\n[json] {args.output}")

    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
