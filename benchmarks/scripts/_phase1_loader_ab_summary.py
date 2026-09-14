#!/usr/bin/env python
"""Summarise the phase-1 two-build loader A/B into a table fit for docs.

Reads the four arm directories `_run_phase1_loader_ab.sh` writes and reports,
per (dataset, metric), each arm's value plus the before/after ratio.

**The ordering check is the point, not a footnote.** Phase 0 measured a 5-7%
position effect on this cluster, so a single before/after pair cannot tell a
real change from where in the job it ran. This prints the two before arms and
the two after arms side by side and computes the spread within each; when the
within-arm spread is comparable to the between-arm gap, it says so rather than
quoting a ratio.

Usage:
    python _phase1_loader_ab_summary.py /path/to/ab_<jobid>
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

# `us_per_cell__collate` is the headline (W2's subject, lower is better); the
# gather rates are the flat-or-better guard (higher is better); peak RSS must
# not move, since W4 charges nothing by default.
LOWER_IS_BETTER = {"us_per_cell__collate", "peak_rss_mb_median", "median_wall_s"}
METRICS = [
    "us_per_cell__collate",
    "cellsets_per_sec__collate_rust",
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
    "peak_rss_mb_median",
]
ARMS = ("before1", "after1", "after2", "before2")


def _metric(row: dict, name: str) -> float | None:
    """Median of a metric across a result's runs, or a top-level scalar."""
    if name in row and isinstance(row[name], (int, float)):
        return float(row[name])
    vals = []
    for run in row.get("runs", []) or []:
        extra = run.get("extra") or {}
        v = extra.get(name, run.get(name))
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return statistics.median(vals) if vals else None


def main(root: pathlib.Path) -> int:
    arms: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        d = root / f"raw-{arm}"
        if not d.is_dir():
            print(f"missing arm directory: {d}", file=sys.stderr)
            return 2
        arms[arm] = {}
        for f in sorted(d.glob("cellset_gather__*.json")):
            ds = f.stem.split("__")[-1]
            arms[arm][ds] = json.loads(f.read_text())

    datasets = sorted(set().union(*(set(a) for a in arms.values())))
    print(f"# phase-1 loader A/B — {root.name}\n")

    for ds in datasets:
        print(f"## {ds}\n")
        print(f"| metric | {' | '.join(ARMS)} | before (med) | after (med) | ratio | verdict |")
        print("|---|" + "---|" * (len(ARMS) + 4))
        for m in METRICS:
            vals = {a: _metric(arms[a].get(ds, {}), m) for a in ARMS}
            if all(v is None for v in vals.values()):
                continue
            befores = [v for a, v in vals.items() if a.startswith("before") and v is not None]
            afters = [v for a, v in vals.items() if a.startswith("after") and v is not None]
            cells = " | ".join(f"{vals[a]:.4g}" if vals[a] is not None else "—" for a in ARMS)
            if not befores or not afters:
                print(f"| `{m}` | {cells} | — | — | — | incomplete |")
                continue
            b, a_ = statistics.median(befores), statistics.median(afters)

            # Within-arm spread, as a fraction of the arm's own level. This is
            # the host/position noise floor for this metric on this run.
            def spread(xs: list[float]) -> float:
                return (max(xs) - min(xs)) / max(xs) if len(xs) > 1 and max(xs) else 0.0

            noise = max(spread(befores), spread(afters))
            if m in LOWER_IS_BETTER:
                ratio, better = (b / a_ if a_ else float("inf")), a_ < b
            else:
                ratio, better = (a_ / b if b else float("inf")), a_ > b
            gap = abs(a_ - b) / max(a_, b) if max(a_, b) else 0.0
            if gap <= noise:
                verdict = f"within noise (spread {noise:.1%} >= gap {gap:.1%})"
            else:
                verdict = ("improved" if better else "REGRESSED") + f" {ratio:.3g}x"
            print(f"| `{m}` | {cells} | {b:.4g} | {a_:.4g} | {ratio:.3g}x | {verdict} |")
        print()

    print("Ratios are before/after for lower-is-better metrics and after/before")
    print("otherwise, so >1 always means the change helped. A gap no larger than")
    print("the within-arm spread is reported as noise, not as a result.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
