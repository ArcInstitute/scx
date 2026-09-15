#!/usr/bin/env python3
"""Report the four per-kernel tokenisation arms, which have no A/B partner.

    .venv/bin/python benchmarks/scripts/_phase3_tokenize_summary.py <dir>

`_phase1_loader_ab_summary.py` is the paired table and is reused unchanged for
the six metrics both builds emit. `us_per_cell__{crop,rank,bin,sample}` exist
only on the `after` build, so they have no ratio and that summariser skips them
— correctly, since a ratio table with one arm would be a table of nothing.

This prints the after-arm medians and spread, plus **why** an arm is missing
where it is missing. That distinction is the point: an absent metric on its own
reads as "not captured", and a fixture that cannot exercise a kernel is a fact
worth publishing rather than a gap to leave unexplained.

Takes either a live job directory (with `rounds/`) or the committed folded JSON,
the same as its A/B sibling, so the published numbers are recomputable from what
is in the tree.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

ARMS = ("crop", "rank", "bin", "sample")


def _load_rounds(root: pathlib.Path) -> list[dict]:
    rounds = root / "rounds"
    if rounds.is_dir():
        return [json.loads(p.read_text()) for p in sorted(rounds.glob("*.json"))]
    if root.is_file():
        return json.loads(root.read_text()).get("rounds", [])
    folded = root / "tokenize_ab.json"
    if folded.is_file():
        return json.loads(folded.read_text()).get("rounds", [])
    raise SystemExit(f"no rounds/ directory and no folded JSON under {root}")


def main(root: pathlib.Path) -> int:
    records = [r for r in _load_rounds(root) if r.get("arm") == "after"]
    if not records:
        raise SystemExit("no `after` rounds found; the four arms exist only there")

    datasets = sorted({r["dataset"] for r in records})
    print(f"# per-kernel tokenisation arms (after build only) — {root.name}\n")

    for ds in datasets:
        rows = [r for r in records if r["dataset"] == ds]
        print(f"## {ds} — {len(rows)} rounds\n")

        meta = next((r.get("tokenize") for r in rows if r.get("tokenize")), None)
        if meta and meta.get("applicable"):
            print(
                f"Shapes: k={meta.get('k')}, l_max={meta.get('l_max')}, "
                f"n_bins={meta.get('n_bins')}, sample_n={meta.get('sample_n')}; "
                f"median nnz/cell {meta.get('median_nnz_per_cell')} "
                f"(crop fill ratio {meta.get('crop_fill_ratio')}), "
                f"min distinct values/row {meta.get('min_distinct_values_per_probed_row')}.\n"
            )
        elif meta:
            print(f"All arms not applicable: {meta.get('reason')}\n")
        else:
            print("No tokenize metadata in these records.\n")

        print("| arm | median us/cell | min | max | rounds | status |")
        print("|---|---|---|---|---|---|")
        for arm in ARMS:
            vals = [
                float(r[f"us_per_cell__{arm}"])
                for r in rows
                if isinstance(r.get(f"us_per_cell__{arm}"), (int, float))
            ]
            arm_meta = ((meta or {}).get("arms") or {}).get(arm) or {}
            if not vals:
                reason = arm_meta.get("reason") or arm_meta.get("error")
                status = f"NOT APPLICABLE — {reason}" if reason else "no value recorded"
                print(f"| `{arm}` | — | — | — | 0 | {status} |")
                continue
            print(
                f"| `{arm}` | {statistics.median(vals):.3f} | {min(vals):.3f} "
                f"| {max(vals):.3f} | {len(vals)} | captured |"
            )
        print()

    print("No floor is proposed from these numbers. The phase's gate requires the")
    print("arms to be captured twice before any floor is authored, and no baseline")
    print("carries `cellset_gather` rows at all — the same blocker that has kept")
    print("`us_per_cell__collate` deferred. See thresholds.yaml item 22.")
    print()
    print("An arm marked NOT APPLICABLE was not silently dropped: the fixture cannot")
    print("exercise that kernel's branch, and the reason is recorded rather than")
    print("papered over by shrinking the kernel's parameter until it fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
