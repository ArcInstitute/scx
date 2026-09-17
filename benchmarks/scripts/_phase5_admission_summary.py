"""Summarise the phase-5 reuse-signal admission A/B.

One build, two arms selected by `SCX_ROW_GROUP_ADMIT`: `plan` is the pre-W10
all-or-nothing verdict and `reuse` is the shipped policy. Ratios are oriented so
>1 always means `reuse` helped, taken WITHIN each round so drift slow compared
to one round cancels, and `p` is an exact two-sided sign test over the rounds.

`row_group_hit_rate__*` is deliberately NOT ratioed: the `plan` arm reads 0.0 by
construction, and a ratio against zero is undefined rather than infinite. It is
reported as the two medians side by side, which is the honest form of "0.000 to
whatever this recovered".

Usage:
    python _phase5_admission_summary.py /path/to/ab_<jobid>
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

# Lower is better for these; every other metric is a rate.
LOWER_IS_BETTER = {
    "us_per_cell__gather_hot_control_cold_tail",
    "us_per_cell__collate",
    "peak_rss_mb__gather_hot_control_cold_tail",
    "peak_rss_mb_median",
}
METRICS = [
    # The arm the policy exists for.
    "cellsets_per_sec__gather_hot_control_cold_tail",
    "us_per_cell__gather_hot_control_cold_tail",
    "peak_rss_mb__gather_hot_control_cold_tail",
    # Everything the policy must NOT move. Measured on the same runs, so
    # "nothing else moved" is a measurement rather than an inference.
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
    "us_per_cell__collate",
    "peak_rss_mb_median",
]
# Reported as two medians, never as a ratio — the `plan` arm is 0.0 by
# construction on every one of these.
SIDE_BY_SIDE = [
    "row_group_hit_rate__gather_hot_control_cold_tail",
    "reuse_admissions__gather_hot_control_cold_tail",
    "admitted_group_bytes__gather_hot_control_cold_tail",
    "rejected_group_bytes__gather_hot_control_cold_tail",
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
        _load_rounds(root), key=lambda r: (r["dataset"], r["round"], r["arm"])
    )
    prov = json.loads((root / "provenance.json").read_text()) if (
        root / "provenance.json").exists() else {}
    dest.write_text(json.dumps({"provenance": prov, "rounds": rounds}, indent=1) + "\n")
    print(f"wrote {len(rounds)} rounds to {dest}")


def _load_rounds(root: pathlib.Path) -> list[dict]:
    """Accept either a live job directory or the committed artifact.

    The published table has to be recomputable from what is in the tree, not
    only from a scratch directory that the job deletes on exit — otherwise the
    committed evidence is unreplayable and `--manifest` is a one-way door.
    """
    if root.is_file():
        return json.loads(root.read_text())["rounds"]
    rounds_dir = root / "rounds"
    if not rounds_dir.is_dir():
        raise SystemExit(
            f"{root} is neither a folded A/B json nor a job dir with rounds/"
        )
    return [json.loads(f.read_text()) for f in sorted(rounds_dir.glob("*.json"))]


def main(root: pathlib.Path) -> int:
    # (dataset, round) -> {arm: record}
    pairs: dict[tuple[str, int], dict[str, dict]] = {}
    for rec in _load_rounds(root):
        pairs.setdefault((rec["dataset"], rec["round"]), {})[rec["arm"]] = rec

    datasets = sorted({ds for ds, _ in pairs})
    print(f"# phase 5 reuse-signal admission A/B (same build, paired) — {root.name}\n")

    for ds in datasets:
        complete = sorted(
            (rnd, p) for (d, rnd), p in pairs.items()
            if d == ds and "plan" in p and "reuse" in p
        )
        dropped = sum(1 for (d, _), p in pairs.items() if d == ds and len(p) < 2)
        print(f"## {ds} — {len(complete)} complete rounds"
              + (f" ({dropped} incomplete, dropped)" if dropped else ""))
        print()
        print("| metric | plan (med) | reuse (med) | median ratio | reuse wins | p (sign) | verdict |")
        print("|---|---|---|---|---|---|---|")

        for m in METRICS:
            ratios, befores, afters = [], [], []
            for _, p in complete:
                b, a = p["plan"].get(m), p["reuse"].get(m)
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

        # The zero-baseline metrics, side by side.
        print("| metric | plan (med) | reuse (med) |")
        print("|---|---|---|")
        for m in SIDE_BY_SIDE:
            bs = [p["plan"].get(m) for _, p in complete]
            as_ = [p["reuse"].get(m) for _, p in complete]
            bs = [float(v) for v in bs if isinstance(v, (int, float))]
            as_ = [float(v) for v in as_ if isinstance(v, (int, float))]
            if not bs and not as_:
                continue
            fb = f"{statistics.median(bs):.4g}" if bs else "-"
            fa = f"{statistics.median(as_):.4g}" if as_ else "-"
            print(f"| `{m}` | {fb} | {fa} |")
        ceil = next((p["reuse"].get("hot_meta") or {}).get("row_group_hit_rate_ceiling")
                    for _, p in complete if p["reuse"].get("hot_meta"))
        if ceil:
            print(f"\nHit-rate ceiling for this arm's shape: **{ceil}** — only the "
                  "control sets can be served from cache, so that is the best any "
                  "admission policy can reach here.")
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
