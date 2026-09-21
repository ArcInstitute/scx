"""Chapter 10: Specialized Workloads.

Includes:
- Multimodal training-loader tables moved to ml_loader chapter (Ch 8).
- Multimodal compression stays here as it's a format/codec concern.
- Cell-eval and Harmony/LISI correctness tables included.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Specialized Workloads")

    # ── Multimodal compression ────────────────────────────────────────
    c.sections.append(Section(
        title="Multimodal Compression (CITE-seq / Multiome)",
        blocks=[
            TextBlock(
                "File sizes and compression ratios for multimodal datasets. "
                "Training-loader throughput for multimodal data is in "
                "Chapter 8 (ML Data Loading)."
            ),
            tables.multimodal_compression_table(),
            tables.multimodal_compression_ratio_table(),
        ],
    ))

    # ── Atlas-scale multimodal streaming ─────────────────────────────
    c.sections.append(Section(
        title="Atlas-Scale Multimodal Streaming (multimodal_atlas_streaming)",
        blocks=[
            TextBlock(
                "Six ways to read the same multimodal file, on fixtures that "
                "reach atlas scale: `multiome_atlas_500k` (500K cells x 35K "
                "RNA + 150K ATAC, 1.80 B nonzeros) and `citeseq_atlas_1m` "
                "(1M x 30K RNA + 250 ADT, 1.14 B). Each arm runs in its own "
                "process. `data_dtype=` requires `backed=False`, so the "
                "narrowed read and the backed stream are necessarily separate "
                "arms rather than one."
            ),
            tables.community_multimodal_atlas_table(),
            TextBlock(
                "`Sums match` compares every modality's total across arms "
                "through a **float64 accumulator**, not `X.sum()`: scipy "
                "accumulates in the array's own dtype, and on the small "
                "CITE-seq fixture the identical values read as float32 and as "
                "uint16 give 32,173,180 and 32,173,181 — a one-unit gap that "
                "would have reported the narrowing arm as corrupting data it "
                "reproduces exactly. The modality-pushdown arm emits no sums "
                "column (it never materialises a matrix) and is skipped "
                "entirely on the two small fixtures, which carry no obs "
                "columns for a predicate to select on."
            ),
        ],
    ))

    # ── The laptop test ──────────────────────────────────────────────
    c.sections.append(Section(
        title="End-to-End Pipeline Under a Memory Ceiling (pipeline_ooc_constrained)",
        blocks=[
            TextBlock(
                "Nine stages — load, QC, filter, HVG, normalize+log1p, PCA, "
                "kNN, UMAP+Leiden, Wilcoxon DE — run inside a fixed 16 GB or "
                "32 GB ceiling enforced by the SLURM cgroup. The ceiling is "
                "**not** `RLIMIT_AS`: an SCX open mmaps the whole file, so an "
                "address-space cap would fail the backed streaming path on "
                "memory that path never makes resident — it would break the "
                "arm it exists to showcase."
            ),
            tables.community_pipeline_ooc_table(),
            TextBlock(
                "The deliverable here is the **completion column**, not the "
                "wall time. A cell that hits the ceiling records "
                "`pipeline_completed_int = 0.0` and the stage it died in, "
                "rather than vanishing: an absent result is skipped by the "
                "gate in silence and would read as coverage. Four outcomes are "
                "kept apart because only one is the claim — `oom_killed`, "
                "`timeout` and `degenerate_clustering` all report 0.0, while a "
                "genuine bug fails the cell instead of being laundered into a "
                "result. HVG runs on **raw counts before** normalisation: "
                "`seurat_v3` is a statistic of the count distribution, and "
                "both arms run the identical order."
            ),
            tables.community_pipeline_ooc_stage_table(),
            TextBlock(
                "The per-stage split is where the engines actually differ; the "
                "aggregate hides it in both directions."
            ),
        ],
    ))

    # ── Harmony & LISI — validation first, then scaling ──────────────
    harmony_blocks: list = [
        TextBlock(
            "Harmony/LISI validation status (full details in Chapter 3):"
        ),
        tables.harmony_lisi_correctness_summary_table(),
        tables.harmony_validation_table(),
        TextBlock("**Scaling performance:**"),
    ]
    harmony_blocks.extend(tables.harmony_scaling_table())
    harmony_blocks.append(tables.lisi_comparison_table())
    c.sections.append(Section(title="Harmony & LISI", blocks=harmony_blocks))

    # ── Cell-Eval — correctness first, then performance ──────────────
    c.sections.append(Section(
        title="Cell-Eval / Arc-Bench Parity",
        blocks=[
            TextBlock(
                "Cell-eval correctness status (full details in Chapter 3):"
            ),
            tables.cell_eval_correctness_summary_table(),
            TextBlock("**Performance:**"),
            tables.cell_eval_parity_perf_table(),
        ],
    ))

    return c
