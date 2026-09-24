#!/usr/bin/env python3
"""`SparseCellSetDataset` scattered-read route A/B.

Prices the two routes the cell-set gather can take, on a **framed** file:

  * ``scatter_block_index=False`` (the shipped default) —
    full-shard decode into the shared LRU, served from cache on reuse.
  * ``scatter_block_index=True`` — decode only the touched row groups,
    bypassing the LRU entirely.

Why this exists as a separate script rather than a `gate_candidate.py` run, and
why it reframes: it prices two routes whose arms differ by ~200x, which a
`BenchmarkResult` pooled over both would describe as neither, and it does so on
a *copy* so no registered fixture and no committed floor moves.

⚠️ The original reason given here — that the registered `cellset_gather`
fixtures are ``format_version = 3``, so ``block_index_eligible``'s
``shard_is_framed`` clause makes both settings take the full-shard path — was
true of the v3 / ``scx1`` / pyscx-0.9.1 fixtures the 2026-08-29 capture ran
against, and is **no longer true of the fixtures this script is pointed at**:
``pbmc10k``, ``tabula_sapiens_100k``, ``census_500k``, ``census_1m``,
``pbmc3k``, ``smartseq2`` and ``chemogenetic_rgfp`` are v4 / framed today, and
the block-index route is reachable on them directly — a 256-row scattered plan
over ``pbmc10k_auto.scx`` reports ``full_shard_groups=4,
block_index_groups=0`` at ``scatter_block_index=False`` and
``full_shard_groups=0, block_index_groups=4`` at ``True``.

It is **not** true of every registered fixture, so do not assume it of a new
``--source``: ``census_5m_auto.scx`` is format **v1** (306 shards, 15.1 GB),
and ``tahoe_c38``, ``replogle_k562`` and the ``_lognorm`` fixtures are v3.
None carries a ``BlockIndex``. The reframe below is what makes any of them
comparable, and `_reframe` verifies the output is v4 rather than trusting it.

The reframe is kept anyway, and for a different reason than it started with:
``scx optimize --row-group-rows 256`` pins the row-group geometry the arms are
compared at, and keeps the ``_v4reframed`` dataset label naming the same
subject the 2026-08-29 rows measured. Dropping it would silently change what a
cross-capture ratio is a ratio of.

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
from benchmarks.comprehensive.results import BenchmarkResult  # noqa: E402


def _parse_porcelain(status: str) -> tuple[list[str], list[str]]:
    """``(tracked_paths, all_paths)`` from ``git status --porcelain`` output.

    ⚠️ The input must be **unstripped**. Porcelain writes two status columns
    before the path, and an unstaged modification leaves the first blank
    (`" M path"`). Stripping the command's whole output removes that leading
    space from the *first* line only, so a fixed `ln[3:]` drops one character
    from exactly one path and leaves every other one intact — which is how
    `benchmarks/README.md` was recorded as `enchmarks/README.md` in a published
    manifest row while the other seven paths were correct.

    That is not cosmetic. `docs/benchmark_manifest.md` accepts a dirty capture
    only when the dirt is **documented**, and a corrupted path documents
    nothing: it names a file that does not exist, so a reader cannot check
    whether the dirt was harmless.

    Renames (`R  old -> new`) are recorded as the destination, which is the
    path whose content the capture actually saw.
    """
    tracked: list[str] = []
    every: list[str] = []
    for ln in status.splitlines():
        if len(ln) < 4:
            continue
        path = ln[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        every.append(path)
        if not ln.startswith("??"):
            tracked.append(path)
    return tracked, every


def _checkout_provenance() -> dict:
    """The revision of the **checkout**, plus exactly what was dirty in it.

    One SHA, not one per file. An earlier version reported the last commit
    touching `scx-loader/src/python.rs` as the "loader commit" and the last
    commit touching this script as the "driver commit"; those are two different
    answers to a question that has one — which tree was measured — and they
    disagreed with each other and with `system.provenance.git_sha`.

    `git_dirty` is only useful with the paths attached: `docs/benchmark_manifest.md`
    accepts a dirty capture when the dirtiness is documented, and that is not
    possible unless the capture records what it was.
    """
    def _run(*args: str, strip: bool = True) -> str:
        out = subprocess.run(
            args, capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout
        return out.strip() if strip else out

    try:
        # NOT stripped: see `_parse_porcelain`.
        status = _run("git", "status", "--porcelain", strip=False)
        tracked, dirty = _parse_porcelain(status)
        return {
            "checkout_sha": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty),
            "dirty_paths": dirty,
            # Tracked dirt invalidates a capture; untracked scratch does not.
            "dirty_tracked_paths": tracked,
        }
    except Exception:  # noqa: BLE001
        return {"checkout_sha": None, "dirty": None}


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


def _sized_budget(scx_path: str) -> tuple[int | None, int | None, str]:
    """``(cache_shards, max_memory_mb, why)`` — **never smaller** than the adaptive budget.

    ⚠️ Two failure modes, in opposite directions, and both produce a route
    ratio that is really a configuration ratio.

    **Too small.** On a reframed census_1m (62 shards, ~181 MB decoded each)
    the loader's adaptive budget resolves to ~4.16 GB, affording ~23 shards. A
    1 024-row random plan touches nearly every shard every batch, so the
    full-shard arm ran at **0.14 cellsets/s** at a 0.215 hit rate and 12.5 GB
    RSS — 95 minutes per timed run. `docs/performance/loader-data-load.md`'s data-load 1A
    capture is the same pathology (census_500k, `cache_shards` 16 -> 31: 0.2 ->
    546 cellsets/s).

    **Too small the other way — sizing that SHRINKS the cache.** Sizing to
    exactly `n_shards x shard_decoded_bytes` is *below* the adaptive budget on
    a small file: tabula needs ~1.73 GB by that formula while the auto-tune
    grants 4.23 GB. Handing it 1.73 GB dropped the shard-cache hit rate
    0.9987 -> 0.5408 and the full-shard arm 936 -> 1.12 cellsets/s, which made
    the block-index arm look 6.2x faster when what had actually happened is
    that its competitor was starved. Measured, not hypothetical — it is what
    the first version of this helper did.

    So: `max(adaptive, sized)`. Sizing may only ever *raise* the budget. That
    keeps the asymmetry honest — the block-index route bypasses the shard cache
    entirely, so any under-sizing penalises only the full-shard arm, which is
    the arm the default favours.
    """
    try:
        import pyscx

        budget = pyscx.SparseCellSetDataset([scx_path]).memory_budget()
        adaptive_mb = int(budget["breakdown"]["total_bytes"]) // (1024 * 1024)
    except Exception as e:  # noqa: BLE001
        return None, None, f"could not read the adaptive budget ({e}); leaving it alone"

    n = cg._shard_count(scx_path)
    if not n:
        return None, None, "shard count unknown; leaving the budget adaptive"
    sized_mb = cg._budget_mb_for(scx_path, n)
    if sized_mb is None:
        return None, None, f"{n} shards; could not size, leaving the budget adaptive"
    if sized_mb <= adaptive_mb:
        return None, None, (
            f"{n} shards need {sized_mb} MB, the adaptive budget already grants "
            f"{adaptive_mb} MB — left alone (sizing must never shrink it)"
        )
    return n, sized_mb, (
        f"raised to hold all {n} shards: {sized_mb} MB "
        f"(adaptive would grant only {adaptive_mb} MB)"
    )


def _plan_factory(kind: str, scx_path: str, n_obs: int):
    """A `(n_batches) -> plans` callable for the requested plan shape.

    Both shapes come from `cellset_gather`'s own generators, so this script
    never becomes a second implementation that could disagree with the
    benchmark.

    The two shapes are not interchangeable and the choice is not cosmetic.
    `random` draws each row independently over the whole corpus, which is the
    worst case for row-group retention: the plan's groups are spread over
    every shard. `grouped` draws each set from one covariate group, which is
    what real models issue (`docs/performance/loader.md`: "Real models issue grouped
    sets, not uniform-random ones") and which clusters a set's cells into few
    shards. Measured at the shipped default, the difference decides whether
    the row-group LRU admits the plan at all: on census_1m a random plan
    retains nothing while a grouped one hits ~45 %.
    """
    set_size = cg._SET_SIZE_S64
    spb = cg._sets_per_batch(set_size)
    if kind == "random":
        return lambda nb: cg._random_plans(n_obs, nb, set_size, spb)
    groups = cg._resolve_groups(scx_path, n_obs)
    if not groups:
        raise SystemExit(
            f"refusing to run: no covariate groups resolved for {scx_path}, so "
            "a `grouped` arm would silently fall back to something else"
        )
    return lambda nb: cg._grouped_plans(groups, nb, set_size, spb)


def _arm(scx_path: str, n_obs: int, gate, n_runs: int, n_batches: int, cold: bool,
         plans, cache_shards=None, max_memory_mb=None) -> dict:
    set_size = cg._SET_SIZE_S64

    # Warm the code paths (tokio/rayon) with a tiny pass; timed reads are cold.
    cg._run_gather(scx_path, lambda: plans(cg._WARMUP_BATCHES), scatter_block_index=gate,
                   cache_shards=cache_shards, max_memory_mb=max_memory_mb)

    sps, rss, ttfb, routes, wall, policies = [], [], [], [], [], []
    for _ in range(n_runs):
        policy = drop_file_cache(scx_path) if cold else "warm"
        # Per sample, not per arm: `drop_file_cache` decides run by run and can
        # fall back to "warm", so one overwritten variable hides a warm sample
        # inside a median published as cold.
        policies.append(policy)
        out = cg._run_gather(
            scx_path, lambda: plans(n_batches), scatter_block_index=gate,
            cache_shards=cache_shards, max_memory_mb=max_memory_mb,
        )
        sps.append(out.n_sets / out.wall_s if out.wall_s > 0 else 0.0)
        rss.append(out.peak_rss_mb)
        ttfb.append(out.ttfb_s)
        wall.append(out.wall_s)
        routes.append((out.full_shard_groups, out.block_index_groups,
                       out.shard_cache_hit_rate))
    last = routes[-1]
    if cold and any(pol != "cold_fadvise" for pol in policies):
        raise SystemExit(
            f"arm scatter_block_index={gate}: {policies.count('warm')} of "
            f"{len(policies)} samples fell back to a warm cache "
            f"({policies}). Refusing to publish a median labelled cold."
        )
    return {
        "samples": [
            {"cellsets_per_sec": round(s, 2), "peak_rss_mb": round(r, 1),
             "ttfb_s": round(tt, 4), "wall_s": round(w, 4), "cache_policy": pol}
            for s, r, tt, w, pol in zip(sps, rss, ttfb, wall, policies)
        ],
        "scatter_block_index": gate,
        "cache_shards": cache_shards,
        "max_memory_mb": max_memory_mb,
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


def _require_row_group_counters(scx_path: str) -> list[str]:
    """Refuse to run against a build that predates the row-group LRU.

    This A/B's whole subject is whether OPT-FORMATIO-1 closed the route gap,
    so a build without it answers a different question — and answers it
    plausibly, reproducing the pre-LRU numbers under a heading that says
    "current main".

    The failure is not hypothetical and not loud: the `scx-bench` conda env
    carries a pyscx *wheel* built from a branch, which reports the same
    `__version__` as the checkout. A bare `import pyscx` finds it, and
    `cache_metrics()` simply lacks the `row_group_*` keys rather than erroring.
    Constructing a dataset is enough to see them — no gather needed.
    """
    import pyscx

    ds = pyscx.SparseCellSetDataset([scx_path], cache_shards=4)
    try:
        keys = set(ds.cache_metrics())
    finally:
        ds.close()
    missing = sorted({"row_group_hits", "row_group_misses"} - keys)
    if missing:
        raise SystemExit(
            f"refusing to run: this pyscx ({pyscx.__file__}) has no "
            f"{missing} in cache_metrics(), so it predates OPT-FORMATIO-1's "
            "row-group LRU — the change this A/B exists to measure across. "
            "Put the checkout's build first on PYTHONPATH "
            "(PYTHONPATH=$REPO/pyscx/python:$REPO); note the env wheel reports "
            "the same __version__."
        )
    return sorted(k for k in keys if k.startswith("row_group_"))


def _premises(arms: dict) -> dict[str, bool]:
    """The four conditions under which this A/B is a comparison at all.

    Verifying the reframed output is v4 is necessary but not sufficient — the
    routes have to have actually *differed at run time*, which only the
    per-arm counters can say. Split out of `main` so the contract is
    unit-testable: the failure this guards against is a plausible number, and a
    plausible number is exactly what does not announce itself.
    """
    return {
        "on arm reached the block-index route":
            arms["on"]["block_index_groups"] > 0,
        "off arm took the full-shard route":
            arms["off"]["full_shard_groups"] > 0,
        "off arm did not reach the block-index route":
            arms["off"]["block_index_groups"] == 0,
        "the default agrees with the explicit False": (
            arms["default"]["block_index_groups"] == arms["off"]["block_index_groups"]
            and arms["default"]["full_shard_groups"] == arms["off"]["full_shard_groups"]
        ),
    }


def _ratios(arms: dict, failed: list[str]) -> tuple[float | None, float | None]:
    """``(off/on, default/on)``, or ``(None, None)`` if any premise failed.

    No ratio unless every premise held: a speedup computed from two arms that
    took the same route is a plausible number that means nothing, and writing
    it into the artifact — even alongside a nonzero exit — invites it being
    quoted later.

    Both ratios are returned, each against the arm it is actually computed
    from. One ratio printed next to a table showing the *other* arm's rate is
    how a published number ends up disagreeing with its own operands.
    """
    on = arms["on"]["median_cellsets_per_sec"]
    if failed or on <= 0:
        return None, None
    return (
        round(arms["off"]["median_cellsets_per_sec"] / on, 3),
        round(arms["default"]["median_cellsets_per_sec"] / on, 3),
    )


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
    ap.add_argument(
        "--plan",
        choices=("random", "grouped"),
        default="random",
        help=(
            "plan shape. `random` (default, and what the 2026-08-29 capture "
            "measured) draws rows independently over the corpus — the worst "
            "case for row-group retention. `grouped` draws each set from one "
            "covariate group, which is what real models issue and which "
            "changes whether the LRU admits the plan at all."
        ),
    )
    ap.add_argument(
        "--adaptive-budget",
        action="store_true",
        help=(
            "leave the loader's adaptive budget alone instead of sizing the "
            "cache to the file's shard count. Measures the shipped default "
            "configuration — which on a file whose shards exceed that budget "
            "makes the full-shard arm a cache-thrash measurement rather than a "
            "route measurement (census_1m: 0.14 cellsets/s at a 0.215 hit rate)."
        ),
    )
    ap.add_argument(
        "--raw-subdir",
        default=None,
        help=(
            "write the per-arm BenchmarkResults into results/raw/<SUBDIR>/ "
            "instead of results/raw/. The six file names are fixed by the "
            "benchmark/format/dataset triple, so a re-measure would otherwise "
            "overwrite the tracked rows a published table already cites. Same "
            "shape as results/raw/pr25_row_group_lru_off/."
        ),
    )
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

    rg = _require_row_group_counters(framed)
    print(f"row-group counters present: {rg}", flush=True)

    probe = pyscx.open(framed)
    try:
        n_obs = int(probe.n_obs)
    finally:
        probe.close()
    arms = {}
    # `None` first: it measures the *shipped default*, which is the claim under
    # test. `False` is the same setting passed explicitly — if the two disagree
    # the default is not what this PR says it is.
    plans = _plan_factory(args.plan, framed, n_obs)
    if args.adaptive_budget:
        cache_shards, budget_mb, why = None, None, "--adaptive-budget: the shipped auto-tune"
    else:
        cache_shards, budget_mb, why = _sized_budget(framed)
    print(f"cache: {why}", flush=True)
    for label, gate in (("default", None), ("off", False), ("on", True)):
        arms[label] = _arm(framed, n_obs, gate, args.n_runs, args.n_batches,
                           not args.warm, plans, cache_shards, budget_mb)
        print(f"  {label:8s} {arms[label]['median_cellsets_per_sec']:8.2f} sets/s  "
              f"rss={arms[label]['median_peak_rss_mb']:.0f}MB  "
              f"full_shard={arms[label]['full_shard_groups']} "
              f"block_index={arms[label]['block_index_groups']} "
              f"hit_rate={arms[label]['shard_cache_hit_rate']}", flush=True)

    # Premise checks, all four of them, before any ratio is reported.
    premises = _premises(arms)
    capable = premises["on arm reached the block-index route"]
    failed = [k for k, ok in premises.items() if not ok]
    speedup_off, speedup_default = _ratios(arms, failed)
    payload = {
        "source": args.source,
        "framed_path": framed,
        "source_info": src_info,
        "framed_info": framed_info,
        "pyscx_version": pyscx.__version__,
        "n_obs": n_obs,
        "set_size": cg._SET_SIZE_S64,
        "plan": args.plan,
        "cache_shards": cache_shards,
        "max_memory_mb": budget_mb,
        "cache_sizing": why,
        "arms": arms,
        # The whole point: is the difference attributable to the route at all?
        "block_index_reachable": capable,
        # Both ratios, each against the arm it is actually computed from. One
        # ratio next to a table showing the *other* arm's rate is how a
        # published number ends up disagreeing with its own operands.
        "speedup_off_over_on": speedup_off,
        "speedup_default_over_on": speedup_default,
        "default_matches_off": premises["the default agrees with the explicit False"],
        "premises": premises,
        "provenance": {
            **_checkout_provenance(),
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "argv": sys.argv[1:],
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    if failed:
        raise SystemExit(
            "refusing to report this A/B — the routes did not differ as required:\n  "
            + "\n  ".join(f"FAILED: {k}" for k in failed)
            + f"\nrun-level counters are in {args.out} for diagnosis "
            "(the speedup fields are null, deliberately)."
        )

    # Schema-v2 `BenchmarkResult`s at the canonical raw path/name, per
    # `docs/benchmark_manifest.md`: every `docs/performance/` claim that can be
    # a benchmark/format/dataset triple must be backed by one. `results/raw/` is
    # gitignored, so these are force-added — four tracked files already sit there
    # for exactly this reason.
    #
    # ⚠️ **One result per ARM**, never one pooled result. `median_wall_s` and
    # `wall_s_iqr` are computed over `runs[]`, so folding a 2.8 cellsets/s arm in
    # with a 511 cellsets/s arm produces a median that describes neither and an
    # IQR that is just the gap between them. That is the same pooling defect this
    # PR removed from `cellset_gather.py`; putting it back inside the artifact
    # that *is* the performance contract would be worse, not better.
    stem = Path(args.source).stem.removesuffix("_auto")
    # The plan shape is part of the subject, not a run parameter: a `grouped`
    # row and a `random` row over the same file measure different things and
    # must not share a manifest triple. `random` keeps the bare label so the
    # 2026-08-29 rows stay comparable.
    subject = "v4reframed" if args.plan == "random" else f"v4reframed_{args.plan}"
    raw_dir = REPO_ROOT / "benchmarks" / "comprehensive" / "results" / "raw"
    if args.raw_subdir:
        raw_dir = raw_dir / args.raw_subdir
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label, arm in arms.items():
        fmt = f"scx_v4reframed_{label}"
        man = BenchmarkResult(
            benchmark="cellset_gather_scatter_routes",
            format=fmt,
            dataset=f"{stem}_{subject}",
            file_size_bytes=os.path.getsize(framed),
            metadata={
                "arm": label,
                "plan": args.plan,
                "scatter_block_index": arm["scatter_block_index"],
                "median_cellsets_per_sec": arm["median_cellsets_per_sec"],
                "full_shard_groups": arm["full_shard_groups"],
                "block_index_groups": arm["block_index_groups"],
                "shard_cache_hit_rate": arm["shard_cache_hit_rate"],
                **{k: v for k, v in payload.items() if k != "arms"},
            },
        )
        for s in arm["samples"]:
            man.add_run(
                wall_s=s["wall_s"],
                peak_rss_mb=s["peak_rss_mb"],
                scenario=f"gather_{args.plan}",
                set_size=cg._SET_SIZE_S64,
                cache_policy=s["cache_policy"],
                scatter_block_index=arm["scatter_block_index"],
                cellsets_per_sec=s["cellsets_per_sec"],
                ttfb_first_set_s=s["ttfb_s"],
                full_shard_groups=arm["full_shard_groups"],
                block_index_groups=arm["block_index_groups"],
            )
        out = raw_dir / f"cellset_gather_scatter_routes__{fmt}__{stem}_{subject}.json"
        out.write_text(json.dumps(man.to_dict(), indent=2))
        written.append(str(out.relative_to(REPO_ROOT)))

    print(json.dumps({k: payload[k] for k in
                      ("block_index_reachable", "speedup_off_over_on",
                       "speedup_default_over_on", "default_matches_off")}, indent=2))
    for w in written:
        print(f"manifest: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
