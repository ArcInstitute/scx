"""Chapter 11: SCX-specific Format Operations.

Includes:
- CSC/CSR dispatch moved to the accelerator chapter (Ch 9).
- This chapter is now purely about fragment/manifest operations
  (append, delete, compact, rollback) — SCX-specific file operations
  that have no direct equivalent in competing formats.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="SCX-specific Format Operations")

    # ── Fragment & manifest operations ────────────────────────────────
    c.sections.append(Section(title="Fragment & Manifest Operations", blocks=[
        TextBlock(
            "Time to execute append/subset operations on SCX fragments.  "
            "These operations have no direct equivalent in h5ad, Zarr, or "
            "TileDB-SOMA."
        ),
        tables.fragment_ops_table(),
    ]))

    # ── Grouped sharding (sort / convert --group-by) ──────────────────
    c.sections.append(Section(title="Grouped Sharding", blocks=[
        TextBlock(
            "Reference-first, group-clustered CSR layout via `scx sort "
            "--group-by` (re-shard an existing file) and `scx convert "
            "--group-by` (write the grouped layout during ingest). The "
            "convert path auto-routes by source density: CSR sources take the "
            "one-pass streaming gather (cheaper); dense sources fall back to a "
            "two-pass plain-convert-then-sort. `read-back correct` verifies "
            "`read_group(label)` partitions the obs axis and the reference "
            "label is isolated. SCX-only."
        ),
        tables.grouped_sharding_table(),
    ]))

    # ── Grouped read/write head-to-head (scx vs shardad) ──────────────
    c.sections.append(Section(title="Grouped Read/Write — scx vs shardad", blocks=[
        TextBlock(
            "Direct comparison of scx's F1/F2 condition-grouped sharding against "
            "shardad's native `.shad` grouping on integer-count perturbation "
            "screens. Each format writes a reference-first, group-clustered file "
            "from the same source, then reads perturbations back with "
            "`read_group(label)` / `read_reference()`. `grouped write wall` and "
            "`grouped file size` compare the write path; `read_group median` / "
            "`cells/s` compare per-perturbation read throughput; `read-back "
            "correct` verifies both formats partition the obs axis and isolate "
            "the reference. The float paired fixture `pert_synth_10k` is omitted "
            "here (shardad rejects float X on its backed grouped writer) — it "
            "stays in the SCX-only grouped sharding table above. shardad's "
            "backed grouped encoder is integer/CSR-only, so for float or dense "
            "sources the shardad arm loads an in-memory CSR AnnData before "
            "writing; that load is inside the timed write, so both formats' "
            "`grouped write wall` includes the source read. NOTE on the write "
            "numbers: (1) scx's `from_h5ad --group-by` streams (out-of-core, "
            "memory-bounded — groups files larger than RAM), whereas the shardad "
            "arm materializes in RAM, so shardad's write lead does not hold at "
            "atlas scale; (2) the current scx grouped-write path carries a ~8x "
            "throughput regression introduced with the F1/F2 grouping rework "
            "(PR #317) — a pre-#317 scx grouped write is ~comparable to shardad "
            "here. shardad's durable edge on these fixtures is integer-count "
            "*compression* (file size), not write speed; `read_group` is the "
            "cleanest read-side comparison."
        ),
        *tables.grouped_read_table(),
    ]))

    # ── Out-of-core peak-RSS boundary (scx streaming vs shardad materialize) ──
    c.sections.append(Section(title="Out-of-Core Peak RSS — scx vs shardad", blocks=[
        TextBlock(
            "scx can iterate the full matrix in bounded memory (backed / "
            "streaming); shardad always materializes the whole matrix into an "
            "in-memory AnnData. Peak RSS (true high-water mark) for a full-data "
            "pass at rising scale: scx streaming stays ~flat while materialize "
            "grows with n_obs — shardad's read ceiling at atlas scale."
        ),
        tables.ooc_rss_boundary_table(),
    ]))

    # ── Format capability matrix (scx vs shardad) ─────────────────────
    c.sections.append(Section(title="Format Capability Matrix", blocks=[
        TextBlock(
            "Where the two formats differ in *capability* (not just speed). The "
            "perf tables above cover the axes both support; this grid records the "
            "features present on one side and absent on the other — shardad is a "
            "focused counts + condition-grouping store, scx a broad platform."
        ),
        tables.capability_matrix_table(),
    ]))

    return c
