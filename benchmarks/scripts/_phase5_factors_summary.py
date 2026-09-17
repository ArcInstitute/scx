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

# Lower is better for anything named as a latency, a time, a memory figure —
# or a MISS. `row_group_misses` dropping 15 % was reported as "REGRESSED
# 0.849x" by the first version of this file, because "misses" was not in this
# list and the default is higher-is-better. A wrong direction on a real
# improvement is the same class of error as a wrong number.
LOWER_HINTS = (
    "_ms_", "latency", "us_per_cell", "peak_rss", "_s__", "wall",
    "misses", "evictions", "overshoot",
)

# Counters with no direction: they DESCRIBE what a verdict decided rather than
# scoring it. `rejected_group_bytes` falling is neither good nor bad on its own
# — it falls precisely because some bytes were admitted instead — and calling
# either way "IMPROVED" would be inventing a preference the metric does not
# have. Reported with their ratio and no verdict.
NEUTRAL = frozenset({
    "admitted_group_bytes", "rejected_group_bytes", "reuse_admissions",
    "parallel_group_decodes", "block_index_groups", "full_shard_groups",
    "sidecar_groups", "memory_budget_total_mb",
})


def _is_neutral(m: str) -> bool:
    return m.split("__", 1)[0] in NEUTRAL

CONTRASTS = (("admission", "plan", "reuse"), ("parallel_decode", "serial", "reuse"))

# Bookkeeping the runner records per round, not measurements. `round` is the
# round number; the rest are the shape of the run, and a "ratio" over them is
# meaningless even when it is 1.0.
EXCLUDE = frozenset({"round", "n_batches", "n_cells", "n_steady_steps"})

# Metrics worth reading first. Anything else emitted is still reported below
# them; this is the reading order, not a filter on what was kept.
#
# ⚠️ Matched as a PREFIX, not by equality. `read_scattered` emits bare
# `gather_latency_ms_p50` but `index_plan` emits it once per scenario
# (`gather_latency_ms_p50__pyscx_index_plan_random`), and an equality match put
# every one of those in the alphabetically-sorted tail — which is how a
# REGRESSION (index_plan random p50 27.96 -> 29.17 ms, 1/12 rounds, p = 0.006)
# sat in this summariser's own output unread until a reviewer went to the JSON.
# A metric this file was built to surface must not depend on where the benchmark
# chose to put a suffix.
PREFERRED_PREFIXES = (
    "gather_latency_ms_p50",
    "gather_latency_ms_p99",
    "row_group_hit_rate",
    "block_index_adoption_rate",
    "peak_rss_mb",
)


def _preference(m: str) -> int:
    """Sort key: position in `PREFERRED_PREFIXES`, else after all of them."""
    for i, pre in enumerate(PREFERRED_PREFIXES):
        if m.startswith(pre):
            return i
    return len(PREFERRED_PREFIXES)


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
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and k not in EXCLUDE
                }
            ordered = sorted(metrics, key=lambda m: (_preference(m), m))

            rows, side = [], []
            for m in ordered:
                ratios, bs, as_ = [], [], []
                for _, arms in complete:
                    b, a = arms[base_arm].get(m), arms[new_arm].get(m)
                    if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                        continue
                    bs.append(float(b))
                    as_.append(float(a))
                    # ⚠️ A ratio needs both sides strictly POSITIVE. The
                    # committed artifact carries `estimate_overshoot_mb__*` at
                    # about -8,100, and `-8166 / -8138` is a number near 1 whose
                    # direction means nothing — the first version of this file
                    # printed win counts and a verdict for it. Signed
                    # diagnostics fall through to the side-by-side table below.
                    if b <= 0 or a <= 0:
                        continue
                    ratios.append(b / a if _lower_is_better(m) else a / b)
                if not bs and not as_:
                    continue
                if not ratios:
                    # A metric one arm zeroes on every round: two medians, no ratio.
                    side.append((m, statistics.median(bs), statistics.median(as_)))
                    continue
                med = statistics.median(ratios)
                # ⚠️ TIES ARE DROPPED from the sign test, not counted as losses.
                # The first version counted `r > 1.0` as a win and everything
                # else as a loss, so a metric the two arms agree on EXACTLY —
                # `block_index_groups`, a fixed batch count, the round number —
                # came out "REGRESSED 1x at 0/12, p = 0.000". Excluding ties is
                # the standard sign test; a metric with no non-tied round is
                # identical, which is a finding in its own right on the
                # counters that describe work rather than speed.
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
    print("an exact two-sided sign test over the rounds: a median ratio with p > 0.05")
    print("is a coincidence of variance, not a result, however large it looks.")
    print()
    print("`reuse` vs `plan` isolates the admission policy; `reuse` vs `serial`")
    print("isolates the chunked parallel decode. The decode is NOT gated by")
    print("SCX_ROW_GROUP_ADMIT, so a two-arm admission A/B holds it constant and")
    print("cannot measure it — which is what the first phase-5 capture did.")
    print()
    print('"identical on all N rounds" means the two arms agreed EXACTLY every')
    print("round. On a counter that describes work rather than speed — touched")
    print("groups, adoption rate, batch count — that is the control: it says the")
    print("arms did the same work and differ only in how they did it.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--manifest":
        write_manifest(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[3]))
        raise SystemExit(0)
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
