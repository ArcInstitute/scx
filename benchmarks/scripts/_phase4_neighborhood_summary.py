#!/usr/bin/env python3
"""Report the two neighbourhood-plan arms, which have no A/B partner.

    .venv/bin/python benchmarks/scripts/_phase4_neighborhood_summary.py <dir>

`gather_neighborhood_graph` / `_coords` do not exist on `main`, so there is no
"before" build and no ratio — the paired summarisers would print a table of
nothing. This prints the medians and spread, the **measured locality of the
plans** beside them, and **why** an arm is missing where it is missing. That
last distinction is the point: an absent metric on its own reads as "not
captured", and a fixture that cannot exercise a builder is a fact worth
publishing rather than a gap to leave unexplained.

Takes either a live job directory (with `rounds/`) or the committed folded
JSON, so the published numbers are recomputable from what is in the tree.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

ARMS = ("graph", "coords")
# Recorded on the same runs. These DO exist on main; they are here so that
# "nothing else moved on this fixture" is a measurement rather than an
# inference from another dataset's capture.
CARRIED = (
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
)


def _load_rounds(root: pathlib.Path) -> list[dict]:
    rounds = root / "rounds"
    if rounds.is_dir():
        return [json.loads(p.read_text()) for p in sorted(rounds.glob("*.json"))]
    if root.is_file():
        return json.loads(root.read_text()).get("rounds", [])
    folded = root / "neighborhood.json"
    if folded.is_file():
        return json.loads(folded.read_text()).get("rounds", [])
    raise SystemExit(f"no rounds/ directory and no folded JSON under {root}")


def _fmt(v: float) -> str:
    """Readable at both ends of the range these metrics span.

    `%.4g` turns 15051.4 into `1.505e+04`, which is exactly the wrong rendering
    for a sets/s figure; a plain `%.4f` turns 0.0037 into `0.0037` but 15051.4
    into a wall of digits. Split on magnitude.
    """
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.5f}"


def _stat(values: list[float]) -> str:
    if not values:
        return "—"
    med = statistics.median(values)
    if len(values) == 1:
        return f"{_fmt(med)} (n=1)"
    return f"{_fmt(med)}  [{_fmt(min(values))}, {_fmt(max(values))}]  (n={len(values)})"


def main(root: pathlib.Path) -> int:
    records = _load_rounds(root)
    if not records:
        raise SystemExit(f"no round records under {root}")

    datasets = sorted({r["dataset"] for r in records})
    print(f"# neighbourhood plan arms (one-armed) — {root.name}\n")

    for ds in datasets:
        rows = [r for r in records if r["dataset"] == ds]
        print(f"## {ds} — {len(rows)} rounds\n")

        meta = next((r.get("neighborhood") for r in rows if r.get("neighborhood")), None)
        if meta and meta.get("applicable"):
            print(
                f"Shapes: k_graph={meta.get('k_graph')}, "
                f"k_coords={meta.get('k_coords')}, "
                f"sets/batch={meta.get('sets_per_batch')}; "
                f"mean graph degree {meta.get('mean_degree')}, "
                f"coordinate extent {meta.get('coord_extent')} "
                f"in {meta.get('coord_dims')}D\n"
            )
        elif meta:
            print(f"Not applicable: {meta.get('reason')}\n")

        for arm in ARMS:
            info = ((meta or {}).get("arms") or {}).get(arm) or {}
            if info.get("applicable") is False:
                print(f"### {arm} — NOT APPLICABLE\n\n  {info.get('reason')}\n")
                continue
            if "error" in info:
                print(f"### {arm} — FAILED\n\n  {info['error']}\n")
                continue

            print(f"### {arm}\n")
            rates = [
                r[f"cellsets_per_sec__gather_neighborhood_{arm}"]
                for r in rows
                if isinstance(r.get(f"cellsets_per_sec__gather_neighborhood_{arm}"), (int, float))
            ]
            us = [
                r[f"us_per_cell__neighborhood_{arm}"]
                for r in rows
                if isinstance(r.get(f"us_per_cell__neighborhood_{arm}"), (int, float))
            ]
            build = [
                r[f"plan_build_s__neighborhood_{arm}"]
                for r in rows
                if isinstance(r.get(f"plan_build_s__neighborhood_{arm}"), (int, float))
            ]
            print(f"  sets/s     : {_stat(rates)}")
            print(f"  us/cell    : {_stat(us)}")
            print(f"  plan build : {_stat(build)} s   (the builder alone, not the gather)")

            loc = info.get("locality") or {}
            if loc:
                print(
                    f"  plans      : {info.get('n_plans')} sets of median size "
                    f"{loc.get('median_set_size')}, batched into "
                    f"{info.get('n_batches')}"
                )
                print(
                    f"  locality   : set radius {loc.get('median_set_radius')}, "
                    f"row-index span {loc.get('median_set_row_index_span')}, "
                    f"{loc.get('mean_shards_touched_per_batch')} shards/batch, "
                    f"duplicate factor {loc.get('batch_duplicate_factor')}"
                )
                print(
                    "               ⚠️ a row-index span near n_obs and every shard "
                    "touched means this is the PESSIMAL scattered read: the "
                    "fixture's obs order is not spatial."
                )
            print()

        print("### carried metrics (these exist on main too)\n")
        for m in CARRIED:
            vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
            print(f"  {m:<46s}: {_stat(vals)}")
        for arm in ARMS:
            key = f"peak_rss_mb__neighborhood_{arm}"
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                print(f"  {key:<46s}: {_stat(vals)}")
        print()

    print(
        "No floor is proposed from this capture; `thresholds.yaml` item 23 "
        "records the three reasons and the recipe."
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(pathlib.Path(sys.argv[1])))
