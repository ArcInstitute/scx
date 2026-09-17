"""Summarise the phase-5 three-arm factor A/B.

Two contrasts from one set of interleaved rounds:

  * **admission**       — `reuse` vs `plan`
  * **parallel decode** — `reuse` vs `serial`

Ratios are oriented so >1 always means the shipped build helped, taken WITHIN
each round so drift slow compared to one round cancels, and `p` is an exact
two-sided sign test over the rounds.

A metric whose baseline arm is 0 on every round (`row_group_hit_rate` under
`plan`, `parallel_group_decodes` under `serial`) is reported as two medians
side by side — a ratio against zero is undefined, not infinite.

Usage:
    python _phase5_factors_summary.py /path/to/factors_<jobid> [--manifest out.json]
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

# Lower is better for anything named as a latency, a time, or a memory figure.
LOWER_HINTS = ("_ms_", "latency", "us_per_cell", "peak_rss", "_s__", "wall")

CONTRASTS = (("admission", "plan", "reuse"), ("parallel_decode", "serial", "reuse"))

# Metrics worth a row, in reporting order. Anything else emitted is still in
# the committed JSON; this is the reading order, not a filter on what was kept.
PREFERRED = (
    "gather_latency_ms_p50",
    "gather_latency_ms_p99",
    "row_group_hit_rate",
    "block_index_adoption_rate",
    "peak_rss_mb_median",
)


def _lower_is_better(m: str) -> bool:
    return any(h in m for h in LOWER_HINTS)


def _binom_two_sided(wins: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2 * tail / 2**n)


def _load(root: pathlib.Path) -> list[dict]:
    folded = root if root.is_file() else root / "factors_ab.json"
    if folded.is_file():
        return json.loads(folded.read_text())["rounds"]
    return [json.loads(p.read_text()) for p in sorted((root / "rounds").glob("*.json"))]


def write_manifest(root: pathlib.Path, dest: pathlib.Path) -> None:
    prov = root / "provenance.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "provenance": json.loads(prov.read_text()) if prov.is_file() else None,
                "rounds": _load(root),
            }
        )
    )
    print(f"wrote {len(_load(root))} rounds to {dest}")


def main(root: pathlib.Path) -> int:
    rounds = _load(root)
    # cell -> round -> arm -> record
    by_cell: dict[str, dict[int, dict[str, dict]]] = {}
    for r in rounds:
        by_cell.setdefault(r["cell"], {}).setdefault(r["round"], {})[r["arm"]] = r

    print(f"# phase 5 factor A/B (one build, three arms) — {root.name}\n")

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
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            ordered = [m for m in PREFERRED if m in metrics] + sorted(
                m for m in metrics if m not in PREFERRED
            )

            rows, side = [], []
            for m in ordered:
                ratios, bs, as_ = [], [], []
                for _, arms in complete:
                    b, a = arms[base_arm].get(m), arms[new_arm].get(m)
                    if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                        continue
                    bs.append(float(b))
                    as_.append(float(a))
                    if b == 0 or a == 0:
                        continue
                    ratios.append(b / a if _lower_is_better(m) else a / b)
                if not bs and not as_:
                    continue
                if not ratios:
                    # A metric one arm zeroes on every round: two medians, no ratio.
                    side.append((m, statistics.median(bs), statistics.median(as_)))
                    continue
                med = statistics.median(ratios)
                wins = sum(1 for r in ratios if r > 1.0)
                n = len(ratios)
                p = _binom_two_sided(wins, n)
                if n < 5:
                    verdict = f"too few rounds (n={n})"
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
    print("an exact two-sided sign test over the rounds: a median ratio with p > 0.05")
    print("is a coincidence of variance, not a result, however large it looks.")
    print()
    print("`reuse` vs `plan` isolates the admission policy; `reuse` vs `serial`")
    print("isolates the chunked parallel decode. The decode is NOT gated by")
    print("SCX_ROW_GROUP_ADMIT, so a two-arm admission A/B holds it constant and")
    print("cannot measure it — which is what the first phase-5 capture did.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--manifest":
        write_manifest(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[3]))
        raise SystemExit(0)
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
