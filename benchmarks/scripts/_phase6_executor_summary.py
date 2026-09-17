"""Summarise the phase-6 two-arm A/B for the multi-set batch executor.

One contrast from one set of interleaved rounds:

  * **executor** — `plan` (the W11 batch executor) vs `set` (the per-set walk)

Ratios are oriented so >1 always means the shipped build helped, taken WITHIN
each round so drift slow compared to one round cancels, and `p` is an exact
two-sided sign test with **ties dropped**.

The statistics, the lower-is-better hints and the neutral-counter list are
imported from `_phase5_factors_summary`, not re-derived: two summarisers that
disagree about whether a rising `misses` is an improvement is exactly the class
of error that file's own comments record.

Usage:
    python _phase6_executor_summary.py /path/to/executor_<jobid> [--manifest out.json]
"""

from __future__ import annotations

import pathlib
import statistics
import sys

from _phase5_factors_summary import (  # noqa: E402
    EXCLUDE,
    _binom_two_sided,
    _is_neutral,
    _lower_is_better,
    _load,
    write_manifest,
)

CONTRASTS = (("executor", "set", "plan"),)

# Reading order, matched as a PREFIX. `cellset_gather` emits one key per
# scenario per metric (`cellsets_per_sec__gather_random_s512`), so an equality
# match would sort every one of them into the alphabetical tail — which is how
# a regression sat unread in the phase-5 summariser's own output.
PREFERRED_PREFIXES = (
    "cellsets_per_sec",
    "us_per_cell",
    "unique_row_fraction",
    "peak_rss_mb",
    "ttfb_first_set_s",
    "block_index_groups",
    "full_shard_groups",
)


def _preference(m: str) -> int:
    for i, pre in enumerate(PREFERRED_PREFIXES):
        if m.startswith(pre):
            return i
    return len(PREFERRED_PREFIXES)


def main(root: pathlib.Path) -> int:
    rounds = _load(root)
    by_cell: dict[str, dict[int, dict[str, dict]]] = {}
    for r in rounds:
        by_cell.setdefault(r["cell"], {}).setdefault(r["round"], {})[r["arm"]] = r

    print(f"# phase 6 executor A/B (one build, two arms) — {root.name}\n")

    for cell in sorted(by_cell):
        print(f"## {cell}\n")
        for label, base_arm, new_arm in CONTRASTS:
            complete = sorted(
                (rnd, arms)
                for rnd, arms in by_cell[cell].items()
                if base_arm in arms and new_arm in arms
            )
            if not complete:
                continue
            dropped = len(by_cell[cell]) - len(complete)
            print(
                f"### {label} — `{new_arm}` vs `{base_arm}`, {len(complete)} complete rounds"
                + (f" ({dropped} incomplete, dropped)" if dropped else "")
            )
            print()
            metrics = set()
            for _, arms in complete:
                metrics |= {
                    k
                    for k, v in arms[new_arm].items()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and k not in EXCLUDE
                }
            ordered = sorted(metrics, key=lambda m: (_preference(m), m))

            rows, side = [], []
            for m in ordered:
                ratios, bs, as_, signed = [], [], [], []
                for _, arms in complete:
                    b, a = arms[base_arm].get(m), arms[new_arm].get(m)
                    if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                        continue
                    # A ratio needs both sides strictly POSITIVE; a signed
                    # diagnostic falls through to the side-by-side table.
                    if b <= 0 or a <= 0:
                        signed.append((float(b), float(a)))
                        continue
                    bs.append(float(b))
                    as_.append(float(a))
                    ratios.append(b / a if _lower_is_better(m) else a / b)
                if not ratios:
                    pool = list(zip(bs, as_)) + signed
                    if not pool:
                        continue
                    side.append(
                        (
                            m,
                            statistics.median([b for b, _ in pool]),
                            statistics.median([a for _, a in pool]),
                        )
                    )
                    continue
                med = statistics.median(ratios)
                wins = sum(1 for r in ratios if r > 1.0)
                losses = sum(1 for r in ratios if r < 1.0)
                n = wins + losses
                p = _binom_two_sided(wins, n)
                if _is_neutral(m):
                    verdict = (
                        f"descriptive, no direction ({wins}/{n} higher)"
                        if n
                        else f"identical on all {len(ratios)} rounds"
                    )
                elif n == 0:
                    verdict = f"identical on all {len(ratios)} rounds"
                elif n < 5:
                    verdict = f"too few non-tied rounds (n={n})"
                elif p > 0.05:
                    verdict = f"no reliable difference ({wins}/{n})"
                elif med > 1:
                    verdict = f"IMPROVED {med:.3g}x"
                else:
                    verdict = f"REGRESSED {med:.3g}x"
                rows.append(
                    (m, statistics.median(bs), statistics.median(as_), med, wins, n, p, verdict)
                )

            if rows:
                print(f"| metric | {base_arm} (med) | {new_arm} (med) | ratio | wins | p | verdict |")
                print("|---|---|---|---|---|---|---|")
                for m, b, a, med, w, n, p, v in rows:
                    print(f"| `{m}` | {b:.4g} | {a:.4g} | {med:.3g}x | {w}/{n} | {p:.3f} | {v} |")
                print()
            if side:
                print(f"| metric (no ratio: a zero baseline) | {base_arm} (med) | {new_arm} (med) |")
                print("|---|---|---|")
                for m, b, a in side:
                    print(f"| `{m}` | {b:.4g} | {a:.4g} |")
                print()

    print("Ratios are oriented so >1 always means the shipped build helped, and are")
    print("taken WITHIN each round so drift slow compared to one round cancels. `p` is")
    print("an exact two-sided sign test over the rounds, ties dropped: a median ratio")
    print("with p > 0.05 is a coincidence of variance, not a result, however large.")
    print()
    print("`plan` vs `set` isolates the executor and nothing else — one build, one")
    print("environment variable. The two arms' BATCHES are byte-identical by")
    print("construction, so nothing here is a correctness signal; what moves is the")
    print("work done to produce them.")
    print()
    print("⚠️ `unique_row_fraction__*` is a property of the PLAN, not of the loader,")
    print("and is identical on both arms by construction. It is reported so the rate")
    print("beside it is interpretable: the same `cellsets_per_sec` means different")
    print("things at 0.53 and at 0.99, and only the shared-control arm is below 0.9.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--manifest":
        write_manifest(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[3]))
        raise SystemExit(0)
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
