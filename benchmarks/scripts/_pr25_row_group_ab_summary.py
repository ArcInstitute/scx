#!/usr/bin/env python
"""Summarise the PR-25 (OPT-FORMATIO-1) same-build A/B of `read_scattered`.

Reads the two snapshot directories `_run_pr25_row_group_lru_ab.sh` captured —
`SCX_ROW_GROUP_CACHE=0` ("off", the pre-change regime) and the shipped default
("on") — and prints, per (format, dataset), the medians of the per-run metrics
that the row-group LRU is supposed to move: gather latency p50, wall, peak RSS,
and on the "on" arm the row-group hit rate. Adoption is printed for both arms
because the gate's route floors depend on it not moving.

The structured result is written to `--out` (never to stdout — a worker can
share that fd with library `println!`s, and the previous A/B of this shape
died on the resulting JSONDecodeError after both arms had run).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _rows(snapshot: Path, benchmark: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for f in sorted((snapshot / "raw").glob(f"{benchmark}__*.json")):
        d = json.loads(f.read_text())
        out[(d["format"], d["dataset"])] = d
    return out


def _median_extra(d: dict, key: str) -> float | None:
    vals = [r["extra"].get(key) for r in (d.get("runs") or []) if r and r.get("extra")]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _median_top(d: dict, key: str) -> float | None:
    vals = [r.get(key) for r in (d.get("runs") or []) if r]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("off", type=Path, help="snapshot dir captured with SCX_ROW_GROUP_CACHE=0")
    ap.add_argument("on", type=Path, help="snapshot dir captured with the shipped default")
    ap.add_argument("--out", type=Path, required=True, help="where to json.dump the summary")
    ap.add_argument("--benchmark", default="read_scattered")
    args = ap.parse_args()

    off = _rows(args.off, args.benchmark)
    on = _rows(args.on, args.benchmark)
    keys = sorted(set(off) | set(on))
    if not keys:
        raise SystemExit(f"no {args.benchmark} rows under {args.off}/raw or {args.on}/raw")

    summary = {
        "benchmark": args.benchmark,
        "off_snapshot": str(args.off),
        "on_snapshot": str(args.on),
        "cells": [],
    }
    hdr = (
        f"{'format':<24} {'dataset':<20} {'p50 off ms':>11} {'p50 on ms':>10} "
        f"{'p50 off/on':>10} {'wall off/on':>11} {'rss off MB':>10} {'rss on MB':>9} "
        f"{'rg hit%':>8} {'adopt off':>9} {'adopt on':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for fmt, ds in keys:
        a, b = off.get((fmt, ds)), on.get((fmt, ds))
        p50_off = _median_extra(a, "gather_latency_ms_p50") if a else None
        p50_on = _median_extra(b, "gather_latency_ms_p50") if b else None
        wall_off = a.get("median_wall_s") if a else None
        wall_on = b.get("median_wall_s") if b else None
        rss_off = _median_top(a, "peak_rss_mb") if a else None
        rss_on = _median_top(b, "peak_rss_mb") if b else None
        hit_on = _median_extra(b, "row_group_hit_rate") if b else None
        hit_off = _median_extra(a, "row_group_hit_rate") if a else None
        adopt_off = _median_extra(a, "block_index_adoption_rate") if a else None
        adopt_on = _median_extra(b, "block_index_adoption_rate") if b else None
        cell = {
            "format": fmt,
            "dataset": ds,
            "gather_latency_ms_p50": {"off": p50_off, "on": p50_on, "off_over_on": _ratio(p50_off, p50_on)},
            "median_wall_s": {"off": wall_off, "on": wall_on, "off_over_on": _ratio(wall_off, wall_on)},
            "peak_rss_mb": {"off": rss_off, "on": rss_on, "on_minus_off": (rss_on - rss_off) if (rss_on is not None and rss_off is not None) else None},
            "row_group_hit_rate": {"off": hit_off, "on": hit_on},
            "block_index_adoption_rate": {"off": adopt_off, "on": adopt_on},
            "n_runs": {"off": a.get("n_runs") if a else None, "on": b.get("n_runs") if b else None},
        }
        summary["cells"].append(cell)

        def f(v, spec):
            return format(v, spec) if v is not None else "-"

        print(
            f"{fmt:<24} {ds:<20} {f(p50_off, '11.1f')} {f(p50_on, '10.1f')} "
            f"{f(_ratio(p50_off, p50_on), '10.2f')} {f(_ratio(wall_off, wall_on), '11.2f')} "
            f"{f(rss_off, '10.0f')} {f(rss_on, '9.0f')} "
            f"{f(hit_on * 100 if hit_on is not None else None, '8.1f')} "
            f"{f(adopt_off, '9.3f')} {f(adopt_on, '8.3f')}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
