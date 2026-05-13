"""Chapter 3: Correctness & Equivalence (Phase 5).

Presents correctness/equivalency first-class, before performance chapters.
Uses proper status classification (Pass/Fail/Skipped/Not applicable) and
surfaces all datasets, not just pbmc3k.

Subsections:
1. Dataset-level correctness summary
2. Scanpy API equivalence (per-dataset detail tables)
3. Pipeline-level biological agreement
4. Round-trip and format integrity
5. Accelerator parity (cross-linked from Chapter 9)
6. Cell-eval / arc-bench parity (cross-linked from Chapter 10)
7. Harmony / LISI validation (cross-linked from Chapter 10)
8. Skipped and non-comparable tests
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, TableBlock, CalloutBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Correctness & Equivalence")

    # ── 1. Dataset-level correctness overview ─────────────────────────
    c.sections.append(Section(
        title="Dataset-level Summary",
        blocks=[
            TextBlock(
                "The following table summarises all correctness and equivalence "
                "harnesses across every benchmarked dataset.  Status uses proper "
                "classification: dependency-skipped tests (e.g. pseudobulk when "
                "pydeseq2 is absent) count as **Skipped**, not Failed."
            ),
            tables.correctness_dataset_summary_table(),
        ],
    ))

    # ── 2. Scanpy API equivalence (per-dataset) ──────────────────────
    detail_blocks = [
        TextBlock(
            "Per-function scanpy equivalence results for every dataset "
            "with validation data.  Each test compares pyscx's output against "
            "scanpy's reference and reports the key parity metric, its observed "
            "value, and the pass/fail threshold."
        ),
    ]
    detail_blocks.extend(tables.correctness_detail_all_datasets())
    c.sections.append(Section(title="Scanpy API Equivalence", blocks=detail_blocks))

    # ── 3. Pipeline-level biological agreement ────────────────────────
    pipeline_blocks = [
        TextBlock(
            "Pipeline agreement compares three preprocessing pathways "
            "(A: scanpy in-memory, B: pyscx eager, C: pyscx lazy/out-of-core) "
            "end-to-end through Leiden clustering and differential expression. "
            "Parity metrics include Leiden ARI, PCA cosine similarity, DE "
            "overlap, and UMAP Procrustes correlation."
        ),
    ]
    pipeline_blocks.extend(tables.pipeline_agreement_table())
    c.sections.append(Section(
        title="Pipeline-level Biological Agreement",
        blocks=pipeline_blocks,
    ))

    # ── 4. Round-trip and format integrity ────────────────────────────
    # The round-trip / backed-equiv harness results flow through
    # correctness_table() which includes backed_equivalence rows.
    c.sections.append(Section(
        title="Round-trip & Format Integrity",
        blocks=[
            TextBlock(
                "Round-trip validation confirms that SCX write → read preserves "
                "matrix data, obs/var metadata, layers, obsm, and uns.  "
                "Backed-equivalence testing verifies that slicing, indexing, "
                "row/column sums, and NNZ operations on the backed reader "
                "match the in-memory AnnData reference."
            ),
            tables.correctness_table(),
        ],
    ))

    # ── 5. Accelerator parity ─────────────────────────────────────────
    c.sections.append(Section(
        title="Accelerator Parity",
        blocks=[
            TextBlock(
                "Accelerator parity shows whether SCX's Rust-native "
                "implementations reproduce scanpy/leidenalg reference results. "
                "Parity metrics (cosine similarity, recall, ARI, etc.) are "
                "extracted from the same benchmark runs that produce timing "
                "data in Chapter 9."
            ),
            tables.accelerator_parity_table(),
        ],
    ))

    # ── 6. Cell-eval / arc-bench parity correctness ──────────────────
    c.sections.append(Section(
        title="Cell-Eval Parity",
        blocks=[
            TextBlock(
                "Cell-eval / arc-bench correctness status for each perturbation "
                "metric operation.  Performance numbers for these operations "
                "appear in Chapter 10; this section focuses on whether the "
                "SCX implementation reproduces the reference values."
            ),
            tables.cell_eval_correctness_summary_table(),
        ],
    ))

    # ── 7. Harmony / LISI validation ──────────────────────────────────
    c.sections.append(Section(
        title="Harmony & LISI Validation",
        blocks=[
            TextBlock(
                "Harmony2 validation compares per-PC Pearson r between SCX's "
                "Rust-native implementation and R harmony.  LISI validation "
                "compares mean LISI values between scx-accel and R lisi.  "
                "Scaling performance appears in Chapter 10."
            ),
            tables.harmony_lisi_correctness_summary_table(),
            tables.harmony_validation_table(),
        ],
    ))

    # ── 8. Skipped and non-comparable tests ──────────────────────────
    c.sections.append(Section(
        title="Skipped & Non-comparable Tests",
        blocks=[
            TextBlock(
                "Tests that could not run due to missing dependencies or "
                "environment constraints are listed below.  These are "
                "categorised as **Skipped** (dependency absent) or "
                "**Not applicable** (test does not apply to the format/dataset "
                "combination) — not as failures.\n\n"
                "- `pseudobulk_dex` / `pseudobulk_dex_stratified`: Skipped when "
                "`pydeseq2` is not installed.\n"
                "- Census-scale cell-eval: O(N²) reference computation deferred "
                "for datasets > 500K cells.\n"
                "- GPU Leiden label stability: documented divergence from "
                "`leidenalg` — not a correctness failure; pin `device=\"cpu\"` "
                "for label-stable downstream work."
            ),
        ],
    ))

    return c
