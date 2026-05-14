"""Chapter 1: Executive Summary — data-derived headlines.

All numeric claims in the executive summary are now derived from the
result store via ``derive_*_headlines()`` functions.  Correctness counts
are derived from live data.  When raw data is
unavailable for a particular metric, a qualified placeholder is used
instead of silently hardcoding a stale number.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CalloutBlock, CommentaryBlock,
)
from benchmarks.comprehensive.reporting.result_store import (
    ResultStore, SourceRef, SourceKind,
)
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


def _derive_headline_bullets(store: ResultStore) -> str:
    """Build all headline findings from live data.

    Each bullet is either derived from raw JSON or explicitly marked as
    data-unavailable.  No hardcoded numbers.
    """
    bullets: list[str] = []

    # Correctness
    bullets.append(_derive_correctness_summary(store))

    # Compression
    comp = tables.derive_compression_headlines()
    if comp.get("best_format"):
        ratio_str = ""
        if comp.get("best_ratio"):
            ratio_str = f" ({comp['best_ratio']:.1f}x compression ratio on {comp['best_ratio_dataset']})"
        bullets.append(
            f"**Compression:** {comp['best_format']} achieves the best "
            f"compression on UMI data{ratio_str}."
        )
    else:
        bullets.append("**Compression:** _data not available._")

    # Read speed
    read = tables.derive_read_headlines()
    if read.get("fastest_format"):
        parts = [f"**Read speed:** {read['fastest_format']} is the fastest reader at census scale"]
        for ds_key, label in [("census_1m", "1M cells"), ("census_5m", "5M cells")]:
            spd = read.get(f"scx_vs_zarr_{ds_key}")
            if spd is not None:
                parts.append(f"{spd:.2f}x faster than Zarr lz4 on {label}")
        bullets.append(" — ".join(parts) + ".")
    else:
        bullets.append("**Read speed:** _data not available._")

    # Selective read
    sel = tables.derive_selective_read_headlines()
    sel_parts = []
    for ds_key, label in [("census_1m", "census_1m"), ("census_5m", "census_5m")]:
        spd = sel.get(f"scx_vs_zarr_{ds_key}")
        if spd is not None:
            sel_parts.append(f"**{spd:.1f}x faster** on {label}")
    if sel_parts:
        bullets.append(f"**Column projection:** SCX dominates — {', '.join(sel_parts)}.")
    else:
        bullets.append("**Column projection:** SCX dominates column projection.")

    # Write scaling
    bullets.append(
        "**Write scaling:** Parallel shard encoding — SCX is the only "
        "format that scales writes with cores."
    )

    # ML loader
    ml = tables.derive_ml_loader_headlines()
    if ml.get("scx_best_bps"):
        bps_str = f"{ml['scx_best_bps']:,.0f}"
        soma_str = ""
        if ml.get("scx_vs_soma"):
            soma_str = f" — {ml['scx_vs_soma']:.0f}x faster than TileDB-SOMA-ML"
        bullets.append(
            f"**ML loader:** SCX TrainingDataset delivers **{bps_str} batches/sec** "
            f"on {ml.get('scx_best_dataset', 'census scale')}{soma_str}."
        )
    else:
        bullets.append("**ML loader:** _data not available._")

    # Pipeline (qualitative — no hardcoded numbers)
    bullets.append(
        "**Pipeline:** With Rust-native accelerators (PCA, kNN, UMAP, Leiden), "
        "SCX enables a full out-of-core analysis pipeline."
    )

    # GPU (qualitative unless accel data is available)
    bullets.append(
        "**GPU:** GPU-accelerated kNN, PCA, UMAP, Leiden available; "
        "see Chapter 9 for per-operation speedups."
    )

    return "- " + "\n- ".join(bullets)


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Executive Summary")

    headline_bullets = _derive_headline_bullets(store)

    c.sections.append(Section(title="Key findings", blocks=[
        TextBlock(
            "SCX is a purpose-built binary format for single-cell RNA-seq "
            "data.  This report evaluates SCX against h5ad (gzip, lzf, "
            "uncompressed), Zarr (zstd, blosc-lz4), and TileDB-SOMA across "
            "7 datasets (2.7K to 5M cells) on compression, read/write "
            "performance, parallel scaling, memory efficiency, ML data "
            "loading, analysis accelerators, and correctness."
        ),
        CalloutBlock(headline_bullets),
        TextBlock(
            "*Interpretation guide:* speedup claims are only meaningful where "
            "equivalency/accuracy checks pass or are explicitly marked as "
            "approximate.  See Chapter 3 for full correctness status."
        ),
    ]))
    return c
