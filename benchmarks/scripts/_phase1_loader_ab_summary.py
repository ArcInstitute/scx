#!/usr/bin/env python
"""Summarise the phase-1 two-build loader A/B as a PAIRED comparison.

`_run_phase1_loader_ab.sh` runs the two builds interleaved: each round gathers
once on each arm, back to back, on the same dataset. This reads those rounds and
reports, per (dataset, metric), the median of the WITHIN-ROUND ratios plus a
sign test over them.

**Why paired, and why a sign test.** The first version of this job compared
arm-sized blocks, and on a contended node that failed: neighbour load drifts on
the timescale of a block, so it lands on whichever arm was running and comes
back as a confident ratio. Measured — a quiet node gave ~1% within-arm spread
and clean signals, a busy one 7-34% and verdicts that contradicted it. Pairing
cancels any drift slow compared to one round, because both members move
together. The sign test then answers the question that survives heavy noise:
not "how much faster", but "did `after` win more rounds than chance allows".

A median ratio without the sign count is the trap this exists to avoid — with
enough variance, any two samples have a ratio.

Usage:
    python _phase1_loader_ab_summary.py /path/to/ab_<jobid>
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

# Lower is better for these; every other metric is a rate.
LOWER_IS_BETTER = {"us_per_cell__collate", "peak_rss_mb_median"}
METRICS = [
    "us_per_cell__collate",
    "cellsets_per_sec__collate_rust",
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
    "peak_rss_mb_median",
]


def _binom_two_sided(wins: int, n: int) -> float:
    """P(at least this lopsided | fair coin). Exact, n is small."""
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def write_manifest(root: pathlib.Path, dest: pathlib.Path) -> None:
    """Fold the per-round files into ONE committed artifact.

    The rounds are the evidence — the published figure is a median of
    within-round ratios and a sign test over them, so the pairing has to
    survive — but 48 one-line files is a poor way to carry 48 records. This
    keeps every record, with its arm and round intact, in a single file.
    Nothing is pooled: the "one file per arm, never pooled" rule exists so
    distinct arms are not averaged together, which folding by round does not do.
    """
    rounds = sorted(
        (json.loads(f.read_text()) for f in (root / "rounds").glob("*.json")),
        key=lambda r: (r["dataset"], r["round"], r["arm"]),
    )
    prov = json.loads((root / "provenance.json").read_text()) if (
        root / "provenance.json").exists() else {}
    dest.write_text(json.dumps({"provenance": prov, "rounds": rounds}, indent=1) + "\n")
    print(f"wrote {len(rounds)} rounds to {dest}")


def main(root: pathlib.Path) -> int:
    rounds_dir = root / "rounds"
    if not rounds_dir.is_dir():
        print(f"no rounds directory at {rounds_dir}", file=sys.stderr)
        return 2

    # (dataset, round) -> {arm: record}
    pairs: dict[tuple[str, int], dict[str, dict]] = {}
    for f in sorted(rounds_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        pairs.setdefault((rec["dataset"], rec["round"]), {})[rec["arm"]] = rec

    datasets = sorted({ds for ds, _ in pairs})
    print(f"# phase-1 loader A/B (paired) — {root.name}\n")

    for ds in datasets:
        complete = sorted(
            (rnd, p) for (d, rnd), p in pairs.items()
            if d == ds and "before" in p and "after" in p
        )
        dropped = sum(1 for (d, _), p in pairs.items() if d == ds and len(p) < 2)
        print(f"## {ds} — {len(complete)} complete rounds"
              + (f" ({dropped} incomplete, dropped)" if dropped else ""))
        print()
        print("| metric | before (med) | after (med) | median ratio | after wins | p (sign) | verdict |")
        print("|---|---|---|---|---|---|---|")

        for m in METRICS:
            ratios, befores, afters = [], [], []
            for _, p in complete:
                b, a = p["before"].get(m), p["after"].get(m)
                if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                    continue
                if b == 0 or a == 0:
                    continue
                befores.append(float(b))
                afters.append(float(a))
                # Ratio is oriented so >1 always means `after` is better.
                ratios.append(b / a if m in LOWER_IS_BETTER else a / b)
            if not ratios:
                continue

            med = statistics.median(ratios)
            wins = sum(1 for r in ratios if r > 1.0)
            n = len(ratios)
            p = _binom_two_sided(wins, n)

            if n < 5:
                verdict = f"too few rounds (n={n})"
            elif p > 0.05:
                verdict = f"no reliable difference ({wins}/{n} rounds)"
            elif med > 1:
                verdict = f"IMPROVED {med:.3g}x"
            else:
                verdict = f"REGRESSED {med:.3g}x"

            print(f"| `{m}` | {statistics.median(befores):.4g} "
                  f"| {statistics.median(afters):.4g} | {med:.3g}x "
                  f"| {wins}/{n} | {p:.3f} | {verdict} |")
        print()

    print("Ratios are oriented so >1 always means the change helped, and are taken")
    print("WITHIN each round so drift slow compared to one round cancels. `p` is an")
    print("exact two-sided sign test over the rounds: a median ratio with p > 0.05 is")
    print("a coincidence of variance, not a result, however large it looks.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--manifest":
        write_manifest(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[3]))
        raise SystemExit(0)
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
