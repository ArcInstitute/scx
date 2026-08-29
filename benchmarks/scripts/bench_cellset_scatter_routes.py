#!/usr/bin/env python3
"""`SparseCellSetDataset` scattered-read route A/B.

Prices the two routes the cell-set gather can take, on a **framed** file:

  * ``scatter_block_index=False`` (the shipped default) —
    full-shard decode into the shared LRU, served from cache on reuse.
  * ``scatter_block_index=True`` — decode only the touched row groups,
    bypassing the LRU entirely.

Why this exists as a separate script rather than a `gate_candidate.py` run: the
registered `cellset_gather` fixtures (`tabula_sapiens_100k`, `census_500k`,
`census_1m`) are all ``format_version = 3``. ``block_index_eligible`` requires
``shard_is_framed``, so on those files **both** settings take the full-shard
path and the comparison is not a comparison. This script reframes a *copy* with
``scx optimize --row-group-rows 256`` so the routes actually differ, leaving
every registered fixture and every committed floor untouched.

The measurement itself is deliberately not new: it drives
``cellset_gather._run_gather`` over ``cellset_gather._random_plans``, so plan
generation, the peak-RSS sampler and the cold-cache drop are the benchmark's,
not a second implementation that could disagree with it.

Usage (see the sibling ``.sbatch``; do not run this inline on a login node):

    python benchmarks/scripts/bench_cellset_scatter_routes.py \\
        --source /data/scx-dev/benchmarks/datasets/tabula_sapiens_100k_auto.scx \\
        --workdir $SCRATCH/cellset-routes --out $SCRATCH/cellset-routes/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.comprehensive.benchmarks import cellset_gather as cg  # noqa: E402
from benchmarks.comprehensive.cache_control import drop_file_cache  # noqa: E402


def _scx_info(scx_bin: str, path: str) -> dict:
    out = subprocess.run(
        [scx_bin, "info", path, "--json"], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _reframe(scx_bin: str, src: str, dst: str, row_group_rows: int) -> dict:
    """`scx optimize` a copy to v4, and return the *verified* output info.

    The verification is load-bearing, not belt-and-braces: an unframed output
    makes both arms identical and the whole A/B silently vacuous — the exact
    trap the registered fixtures are already in.
    """
    if os.path.exists(dst):
        os.remove(dst)
    t0 = time.perf_counter()
    subprocess.run(
        [scx_bin, "optimize", "--row-group-rows", str(row_group_rows), src, dst],
        check=True,
    )
    elapsed = time.perf_counter() - t0
    info = _scx_info(scx_bin, dst)
    if int(info.get("format_version", 0)) < 4:
        raise SystemExit(
            f"reframe produced format_version={info.get('format_version')} — "
            "the block-index route is unreachable and the A/B would measure "
            "nothing. Refusing to continue."
        )
    info["_reframe_wall_s"] = round(elapsed, 1)
    return info


def _arm(scx_path: str, n_obs: int, gate, n_runs: int, n_batches: int, cold: bool) -> dict:
    set_size = cg._SET_SIZE_S64
    spb = cg._sets_per_batch(set_size)

    def plans(nb: int):
        return cg._random_plans(n_obs, nb, set_size, spb)

    # Warm the code paths (tokio/rayon) with a tiny pass; timed reads are cold.
    cg._run_gather(scx_path, lambda: plans(cg._WARMUP_BATCHES), scatter_block_index=gate)

    sps, rss, ttfb, routes = [], [], [], []
    for _ in range(n_runs):
        policy = drop_file_cache(scx_path) if cold else "warm"
        out = cg._run_gather(
            scx_path, lambda: plans(n_batches), scatter_block_index=gate
        )
        sps.append(out.n_sets / out.wall_s if out.wall_s > 0 else 0.0)
        rss.append(out.peak_rss_mb)
        ttfb.append(out.ttfb_s)
        routes.append((out.full_shard_groups, out.block_index_groups,
                       out.shard_cache_hit_rate))
    last = routes[-1]
    return {
        "scatter_block_index": gate,
        "cache_policy": policy,
        "n_runs": n_runs,
        "n_batches": n_batches,
        "median_cellsets_per_sec": round(statistics.median(sps), 2),
        "all_cellsets_per_sec": [round(v, 2) for v in sps],
        "median_peak_rss_mb": round(statistics.median(rss), 1),
        "median_ttfb_s": round(statistics.median(ttfb), 4),
        "full_shard_groups": last[0],
        "block_index_groups": last[1],
        "shard_cache_hit_rate": last[2],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="an existing .scx to reframe")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scx-bin", default=str(REPO_ROOT / "target" / "release" / "scx"))
    ap.add_argument("--row-group-rows", type=int, default=256)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--n-batches", type=int, default=cg._DEFAULT_N_BATCHES)
    ap.add_argument("--warm", action="store_true", help="skip the page-cache drop")
    args = ap.parse_args()

    import pyscx

    # `block_index_eligible` ANDs the process-global switch, read once per
    # process via `OnceLock`. With it off, the `on` arm silently takes the
    # full-shard path and the A/B compares one route with itself — a plausible
    # number that means nothing. Refuse rather than measure it.
    env_switch = os.environ.get("SCX_SCATTER_BLOCK_INDEX")
    if env_switch is not None and (env_switch == "0" or env_switch.lower() == "false"):
        raise SystemExit(
            f"SCX_SCATTER_BLOCK_INDEX={env_switch!r} is set: the process-global "
            "kill-switch forces every reader onto the full-shard path, so both "
            "arms of this A/B would measure the same route. Unset it and re-run."
        )

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    framed = str(Path(args.workdir) / (Path(args.source).stem + "_framed.scx"))

    src_info = _scx_info(args.scx_bin, args.source)
    print(f"source: format_version={src_info.get('format_version')} "
          f"n_csr_shards={src_info.get('n_csr_shards')}", flush=True)
    framed_info = _reframe(args.scx_bin, args.source, framed, args.row_group_rows)
    print(f"framed: format_version={framed_info.get('format_version')} "
          f"n_csr_shards={framed_info.get('n_csr_shards')} "
          f"({framed_info['_reframe_wall_s']}s)", flush=True)

    n_obs = int(pyscx.open(framed).n_obs)
    arms = {}
    # `None` first: it measures the *shipped default*, which is the claim under
    # test. `False` is the same setting passed explicitly — if the two disagree
    # the default is not what this PR says it is.
    for label, gate in (("default", None), ("off", False), ("on", True)):
        arms[label] = _arm(framed, n_obs, gate, args.n_runs, args.n_batches,
                           not args.warm)
        print(f"  {label:8s} {arms[label]['median_cellsets_per_sec']:8.2f} sets/s  "
              f"rss={arms[label]['median_peak_rss_mb']:.0f}MB  "
              f"full_shard={arms[label]['full_shard_groups']} "
              f"block_index={arms[label]['block_index_groups']} "
              f"hit_rate={arms[label]['shard_cache_hit_rate']}", flush=True)

    # Premise checks, all three of them, before any ratio is reported. The
    # docstring promises this script refuses a vacuous comparison; verifying the
    # output is v4 is necessary but not sufficient — the routes have to have
    # actually differed at run time.
    capable = arms["on"]["block_index_groups"] > 0
    premises = {
        "on arm reached the block-index route": capable,
        "off arm took the full-shard route": arms["off"]["full_shard_groups"] > 0,
        "off arm did not reach the block-index route":
            arms["off"]["block_index_groups"] == 0,
        "the default agrees with the explicit False": (
            arms["default"]["block_index_groups"] == arms["off"]["block_index_groups"]
            and arms["default"]["full_shard_groups"] == arms["off"]["full_shard_groups"]
        ),
    }
    failed = [k for k, ok in premises.items() if not ok]
    on = arms["on"]["median_cellsets_per_sec"]
    payload = {
        "source": args.source,
        "framed_path": framed,
        "source_info": src_info,
        "framed_info": framed_info,
        "pyscx_version": pyscx.__version__,
        "n_obs": n_obs,
        "set_size": cg._SET_SIZE_S64,
        "arms": arms,
        # The whole point: is the difference attributable to the route at all?
        "block_index_reachable": capable,
        "speedup_default_over_on": (
            round(arms["default"]["median_cellsets_per_sec"] / on, 3) if on > 0 else None
        ),
        "default_matches_off": premises["the default agrees with the explicit False"],
        "premises": premises,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in
                      ("block_index_reachable", "speedup_default_over_on",
                       "default_matches_off")}, indent=2))
    if failed:
        raise SystemExit(
            "refusing to report this A/B — the routes did not differ as required:\n  "
            + "\n  ".join(f"FAILED: {k}" for k in failed)
            + f"\nrun-level counters are in {args.out} for diagnosis."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
