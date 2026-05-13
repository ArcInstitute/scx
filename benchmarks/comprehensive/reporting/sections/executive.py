"""Chapter 1: Executive Summary (Phase 5 — data-derived counts).

Derives correctness/equivalency counts from the result store so the
executive summary never contradicts the detail tables in Chapter 3.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CalloutBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def _derive_correctness_summary(store: ResultStore) -> str:
    """Build the correctness bullet from live data, matching Chapter 3."""
    from benchmarks.comprehensive.reporting.tables import (
        _recount_correctness,
    )

    results = store.load_all_results_compat()
    scanpy_equiv = [
        r for r in results
        if r.get("harness") == "scanpy_equivalence"
    ]

    total_pass = total_fail = total_skip = 0
    datasets_seen: list[str] = []
    for r in scanpy_equiv:
        ds = r.get("dataset", "unknown")
        if ds not in datasets_seen:
            datasets_seen.append(ds)
        per_test = r.get("results", [])
        if per_test:
            p, f, s = _recount_correctness(per_test)
            total_pass += p
            total_fail += f
            total_skip += s

    n_ds = len(datasets_seen)
    total = total_pass + total_fail + total_skip
    ds_list = ", ".join(datasets_seen[:4])
    if len(datasets_seen) > 4:
        ds_list += f" + {len(datasets_seen) - 4} more"

    if total_fail > 0:
        return (
            f"**Correctness:** {total_pass}/{total} scanpy equivalence tests "
            f"pass across {n_ds} datasets ({ds_list}); "
            f"{total_fail} failed, {total_skip} skipped (dependency absent)."
        )
    skip_note = f", {total_skip} skipped (dependency absent)" if total_skip else ""
    return (
        f"**Correctness:** {total_pass}/{total} scanpy equivalence tests "
        f"pass across {n_ds} datasets ({ds_list}){skip_note}."
    )


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Executive Summary")

    correctness_bullet = _derive_correctness_summary(store)

    c.sections.append(Section(title="Key findings", blocks=[
        TextBlock(
            "SCX is a purpose-built binary format for single-cell RNA-seq "
            "data.  This report evaluates SCX against h5ad (gzip, lzf, "
            "uncompressed), Zarr (zstd, blosc-lz4), and TileDB-SOMA across "
            "7 datasets (2.7K to 5M cells) on compression, read/write "
            "performance, parallel scaling, memory efficiency, ML data "
            "loading, analysis accelerators, and correctness."
        ),
        CalloutBlock(
            f"- {correctness_bullet}\n"
            "- **Compression:** SCX pcodec/zstd achieve the best compression on UMI data...\n"
            "- **Read speed:** SCX is the fastest reader at census scale...\n"
            "- **Write scaling:** Parallel shard encoding...\n"
            "- **Column projection:** SCX dominates...\n"
            "- **ML loader:** SCX TrainingDataset delivers...\n"
            "- **Pipeline:** With Rust-native Leiden...\n"
            "- **GPU:** kNN 9.4x..."
        ),
        TextBlock(
            "*Interpretation guide:* speedup claims are only meaningful where "
            "equivalency/accuracy checks pass or are explicitly marked as "
            "approximate.  See Chapter 3 for full correctness status."
        ),
    ]))
    return c
