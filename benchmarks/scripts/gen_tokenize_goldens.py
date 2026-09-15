#!/usr/bin/env python3
"""Generate the per-kernel reference goldens for `scx-loader/src/tokenize/`.

    .venv/bin/python benchmarks/scripts/gen_tokenize_goldens.py

Writes four JSON files into `scx-loader/tests/data/tokenize/` and prints the
blake3 prefix each Rust test must pin. **Not run in CI**; the goldens are
committed and `tokenize/golden_tests.rs` asserts them under plain `cargo test`.

-----------------------------------------------------------------------------
WHAT THIS GENERATOR CAN AND CANNOT CLAIM — read before trusting a number
-----------------------------------------------------------------------------

The convention this repo's other reference generators follow
(`generate_harmony_references.py`) is that **every expected value is one the
reference library PRODUCED**, never one the script derives. This generator
cannot meet that bar, and says so rather than implying otherwise:

  * `scgpt`, `geneformer` and UCE are installed in **no** conda env on this
    machine and in no `.venv`. Swept 2026-09-14 across all 43 envs under
    ~/miniforge3/envs plus the repo `.venv`: zero hits. scGPT additionally pins
    torch<2.0 and flash-attn, which makes an env for it expensive.

So each kernel below is a **verbatim transcription** of the reference's own
function, quoted in full in this file next to the pinned revision it came from.
That is materially weaker than importing the package — a transcription can be
wrong where an import cannot — and is materially stronger than SCX agreeing with
itself: numpy computes the quantiles, the digitize bounds, the argsort and the
weighted draw in every case, and the transcribed wrapper is 10-15 lines.

Three of the references are also stochastic or underdetermined, so exact parity
is not available at all. What each file claims is stamped into its own
`_provenance.claim` field and restated in `docs/tokenize.md`:

  * crop          exact (np.lexsort; deterministic)
  * rank          exact per equal-value run (np.argsort's default kind is
                  quicksort, so the reference's order WITHIN a run is arbitrary)
  * bin           exact for both digitize bounds; the randomised form is checked
                  by bracketing against reference-produced draws
  * sample        distributional only (np.random.choice's stream is numpy's)

To close the gap: create an env with the real packages, replace each
`_reference_*` function below with the imported one, and the claim per file with
"exact". The transcriptions are deliberately kept in one place so that swap is
mechanical.
"""

from __future__ import annotations

import json
import pathlib
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - the generator's own dependency gate
    sys.exit(
        "numpy is required. Run this with the repo venv:\n"
        "    .venv/bin/python benchmarks/scripts/gen_tokenize_goldens.py"
    )

try:
    import blake3 as _blake3

    def digest(text: str) -> str:
        return _blake3.blake3(text.encode()).hexdigest()[:16]
except ImportError:
    _blake3 = None

    def digest(text: str) -> str:  # pragma: no cover - reported, not silent
        return "<blake3 unavailable: pip install blake3 and re-run>"


REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "scx-loader" / "tests" / "data" / "tokenize"

# Pinned revisions the transcriptions below were taken from.
SCGPT_REV = "cebd6fae655b9c585a4807daa3ac31bb764f06b4"
# Geneformer lives on HuggingFace, whose `main` is mutable, so the branch name
# is not a pin. Resolved via the HF API on 2026-09-14
# (`GET /api/models/ctheodoris/Geneformer` -> .sha); an earlier revision of this
# file said "record the resolved commit" and then recorded `main@<date>`, which
# is the thing that sentence warns against.
GENEFORMER_REV = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5 (huggingface.co/ctheodoris/Geneformer)"
UCE_REV = "9c416007be15ad6753dc84af4468c1dc10421ab9 (github.com/snap-stanford/UCE, eval_data.py)"


# ---------------------------------------------------------------------------
# Transcriptions. Each is the reference's own code, adapted only to take its
# inputs as arguments rather than off an AnnData/loom handle.
# ---------------------------------------------------------------------------


def _reference_crop(gene_ids, values, k, n_genes_total):
    """state3 `_sparse_encoder_inputs`' selection, without the query masking.

        order = np.lexsort((gene_id, -selection))   # value DESC, id ASC
        selected = order[:k]

    Deterministic: `lexsort` is stable and the keys are a total order.
    """
    pad, mask_id = n_genes_total + 1, n_genes_total
    ids = np.full(k, pad, dtype=np.int64)
    vals = np.zeros(k, dtype=np.float32)
    mask = np.zeros(k, dtype=np.uint8)
    padf = np.ones(k, dtype=np.uint8)
    sel = np.clip(np.asarray(values, dtype=np.float32), 0, None)
    keep = sel > 0
    if not keep.any():
        ids[0], padf[0] = mask_id, 0
        return ids, vals, mask, padf
    g, v = np.asarray(gene_ids, dtype=np.int32)[keep], sel[keep]
    order = np.lexsort((g, -v))[:k]
    ids[: len(order)] = g[order]
    vals[: len(order)] = v[order]
    padf[: len(order)] = 0
    return ids, vals, mask, padf


def _reference_rank(gene_ids, counts, stats, target_sum, l_max):
    """Geneformer `tokenize_cell` + `rank_genes`, verbatim:

        X_norm = X_view / n_counts * target_sum / norm_factor_vector
        nonzero_mask = np.nonzero(gene_vector)[0]
        sorted_indices = np.argsort(-gene_vector)
        return gene_tokens[sorted_indices]

    `np.argsort`'s default kind is quicksort, which is NOT stable, so the order
    within an equal-value run is an artefact of the partitioning. This returns
    the normalised values alongside the ids so the caller can group the runs.
    """
    g = np.asarray(gene_ids, dtype=np.int32)
    v = np.clip(np.asarray(counts, dtype=np.float32).astype(np.float64), 0, None)
    lib = v.sum()
    if lib <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    norm = v / lib * target_sum / np.asarray(stats, dtype=np.float32)[g].astype(np.float64)
    nz = np.nonzero(norm)[0]
    order = np.argsort(-norm[nz])
    return g[nz][order][:l_max].astype(np.int64), norm[nz][order][:l_max]


def _reference_bin_edges(values, n_bins):
    """scGPT `binning`'s edge computation, verbatim:

        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
    """
    row = np.asarray(values, dtype=np.float32)
    nz = row[row.nonzero()]
    return np.quantile(nz, np.linspace(0, 1, n_bins - 1))


def _reference_digitize_both(values, bins, rng):
    """scGPT `_digitize(x, bins, side="both")`, verbatim:

        left_digits  = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)
        rands = np.random.rand(len(x))
        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)

    This is the default, so scGPT's `binning()` IS randomised. The only change
    here is taking `rng` explicitly so the committed draws are reproducible —
    the reference reads numpy's global RNG, which is precisely why no SCX output
    can match it draw for draw.
    """
    x = np.asarray(values, dtype=np.float32)
    left = np.digitize(x, bins)
    right = np.digitize(x, bins, right=True)
    rands = rng.random(len(x))
    return np.ceil(rands * (right - left) + left).astype(np.int64)


def _reference_sample_weights(counts):
    """UCE `sample_cell_sentences`' weighting, verbatim:

        weights = torch.log1p(counts)
        weights = (weights / torch.sum(weights))
        choice_idx = np.random.choice(np.arange(len(weights)),
                                      size=args.sample_size, p=weights,
                                      replace=True)

    Note `replace=True`. Only the weights are returned: the draw itself is
    numpy's global RNG and cannot be reproduced in Rust.

    ⚠️ **Precision is an OPEN divergence, not a match.** UCE feeds a torch
    tensor to `torch.log1p`, so if those counts are float32 the weights and the
    normalisation are float32 too, and numpy's `choice` then widens `p` to
    float64 for its cumsum — which would move CDF boundaries relative to the
    f64 arithmetic below and in SCX's kernel.

    This generator does NOT try to reproduce that. Reconstructing it (log1p and
    the divide in float32, widened after) was attempted and produces
    probabilities that sum to 1 only within ~1e-7 — outside the 1.49e-8
    tolerance `np.random.choice` itself enforces, i.e. a `p` numpy would
    *reject*. That is evidence the float32 reconstruction is wrong about UCE,
    not evidence UCE is broken, and the package is not installable here to
    settle it.

    So the arithmetic below is float64 throughout, matching SCX's kernel, and
    the resulting numbers are labelled a **float64 reconstruction of UCE's
    formula** rather than UCE's probabilities. Resolving it needs a run of the
    real package. Recorded in `docs/tokenize.md` alongside the RNG divergence.
    """
    w = np.log1p(np.clip(np.asarray(counts, dtype=np.float32).astype(np.float64), 0, None))
    total = w.sum()
    return (w / total) if total > 0 else np.zeros_like(w)


# ---------------------------------------------------------------------------
# Fixtures. Emitted beside the expected values, so the two cannot drift.
# ---------------------------------------------------------------------------

CROP_CASES = [
    ("basic_descending", [3, 5, 9], [1.0, 7.0, 4.0], 3, 100),
    ("all_counts_tie_so_id_decides", [9, 3, 5, 7], [2.0, 2.0, 2.0, 2.0], 4, 100),
    ("two_tie_runs", [9, 3, 8, 2], [1.0, 1.0, 5.0, 5.0], 4, 100),
    ("truncates_to_k", [1, 2, 3, 4], [4.0, 1.0, 3.0, 2.0], 2, 100),
    ("pads_the_tail", [1, 2], [4.0, 3.0], 5, 100),
    ("non_positive_dropped", [1, 2, 3, 4], [0.0, -3.0, 5.0, 2.0], 4, 100),
    ("empty_row", [], [], 3, 100),
    ("no_positive_counts", [1, 2], [0.0, 0.0], 2, 100),
    ("large_vocabulary", [10, 20, 30], [5.0, 5.0, 1.0], 3, 61_000),
]

RANK_CASES = [
    # (name, gene_ids, counts, stats, l_max)
    ("distinct_values", [1, 3, 5], [2.0, 9.0, 4.0], [1.0] * 8, 3),
    ("statistic_reorders", [1, 3], [2.0, 9.0], [1.0, 0.01, 1.0, 10.0], 2),
    ("all_values_tie", list(range(10)), [1.0] * 10, [1.0] * 10, 10),
    ("equal_after_normalisation", [1, 4], [2.0, 8.0], [1.0, 1.0, 1.0, 1.0, 4.0], 2),
    ("truncates_to_l_max", [1, 2, 3, 4], [4.0, 3.0, 2.0, 1.0], [1.0] * 8, 2),
    ("zeros_dropped", [1, 2, 3], [0.0, 5.0, 0.0], [1.0] * 8, 3),
    ("empty_row", [], [], [1.0] * 8, 3),
    ("all_zero_row", [1, 2], [0.0, 0.0], [1.0] * 8, 3),
]

BIN_CASES = [
    ("ascending_values", [1.0, 2.0, 5.0, 11.0], 6),
    ("f32_widening", [0.1, 0.2, 0.3, 0.7], 6),
    ("repeated_values", [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], 5),
    ("all_nonzeros_equal", [4.0, 4.0, 4.0], 6),
    ("single_nonzero", [7.0], 6),
    ("with_zeros", [0.0, 5.0, 0.0, 1.0], 5),
    ("fifty_one_bins", [float(v) for v in range(1, 40)], 51),
]

SAMPLE_CASES = [
    ("two_genes_equal", [3, 8], [5.0, 5.0]),
    ("wide_dynamic_range", [1, 2], [1.0, 1000.0]),
    ("four_genes", [10, 20, 30, 40], [1.0, 3.0, 7.0, 15.0]),
    ("a_zero_weight_gene", [1, 2, 3], [0.0, 5.0, 0.0]),
    ("all_zero", [1, 2], [0.0, 0.0]),
]

BANNER = (
    "GENERATED FILE - do not edit by hand. Regenerate with "
    "`.venv/bin/python benchmarks/scripts/gen_tokenize_goldens.py`, then update "
    "the blake3 prefix in scx-loader/src/tokenize/golden_tests.rs. See the "
    "generator's docstring for what this file's numbers do and do not claim."
)


def write(name: str, claim: str, reference: str, cases: list) -> None:
    payload = {
        "_comment": BANNER,
        "_provenance": {
            "reference": reference,
            "produced_by": "numpy running a verbatim transcription of the reference "
            "kernel; the package itself is installed nowhere on this machine",
            "claim": claim,
            "numpy": np.__version__,
        },
        "cases": cases,
    }
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path = OUT / name
    path.write_text(text)
    print(f"{path.relative_to(REPO)}  blake3[:16] = {digest(text)}  ({len(cases)} cases)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- crop: exact -------------------------------------------------------
    cases = []
    for name, g, v, k, n_genes in CROP_CASES:
        ids, vals, mask, padf = _reference_crop(g, v, k, n_genes)
        cases.append(
            {
                "name": name,
                "gene_ids": g,
                "values": v,
                "k": k,
                "n_genes_total": n_genes,
                "expected": {
                    "ids": ids.tolist(),
                    "values": [float(x) for x in vals],
                    "mask": mask.tolist(),
                    "pad": padf.tolist(),
                },
            }
        )
        # Rule 3: the bar is CHECKED here, not merely printed.
        assert len(ids) == k and len(vals) == k
    write(
        "crop_golden.json",
        "exact: np.lexsort is deterministic and the keys are a total order",
        "state3 _sparse_encoder_inputs (selection half)",
        cases,
    )

    # --- rank: exact per equal-value run -----------------------------------
    cases = []
    for name, g, c, stats, l_max in RANK_CASES:
        ids, vals = _reference_rank(g, c, stats, 1e4, l_max)
        # Group the reference's output into equal-normalised-value runs and sort
        # each by gene id. Within a run the reference's own order is arbitrary
        # (unstable argsort), so the run — not the sequence — is what it pins.
        groups: list[list[int]] = []
        i = 0
        while i < len(ids):
            j = i + 1
            while j < len(ids) and vals[j] == vals[i]:
                j += 1
            groups.append(sorted(int(x) for x in ids[i:j]))
            i = j
        assert sum(len(gp) for gp in groups) == len(ids)
        cases.append(
            {
                "name": name,
                "gene_ids": g,
                "counts": c,
                "gene_stats": stats,
                "target_sum": 1e4,
                "l_max": l_max,
                "expected_len": int(len(ids)),
                # Each inner list is one equal-value run, ids ascending.
                "expected_groups": groups,
                "tie_free": all(len(gp) == 1 for gp in groups),
            }
        )
    assert any(not c["tie_free"] for c in cases), "no tied case: the run logic is untested"
    assert any(c["tie_free"] and c["expected_len"] > 1 for c in cases), "no tie-free case"
    write(
        "rank_golden.json",
        "exact within each equal-value run; np.argsort's default kind is "
        "quicksort, so the reference's order INSIDE a run follows no rule",
        f"Geneformer tokenizer.py @ {GENEFORMER_REV}",
        cases,
    )

    # --- bin: exact bounds + reference-produced randomised draws -----------
    cases = []
    for name, vals, n_bins in BIN_CASES:
        row = np.asarray(vals, dtype=np.float32)
        edges = _reference_bin_edges(vals, n_bins)
        nz = row[row.nonzero()]
        left = np.digitize(nz, edges)
        right = np.digitize(nz, edges, right=True)
        # 24 draws of the reference's OWN randomised digitize, seeded here only
        # so the committed file is reproducible. The Rust test asserts its own
        # left/right bound these, which checks SCX's bounds against the
        # reference's behaviour rather than against SCX.
        rng = np.random.default_rng(20260914)
        draws = [
            _reference_digitize_both(nz, edges, rng).tolist() for _ in range(24)
        ]
        assert all(
            all(right[i] <= d[i] <= left[i] for i in range(len(nz))) for d in draws
        ), f"{name}: the reference's own draws escaped its own bounds"
        cases.append(
            {
                "name": name,
                "values": vals,
                "n_bins": n_bins,
                "edges": [float(e) for e in edges],
                "expected_left": left.tolist(),
                "expected_right": right.tolist(),
                "reference_randomised_draws": draws,
            }
        )
    write(
        "bin_golden.json",
        "exact for the quantile edges and both np.digitize bounds; the "
        "randomised form draws from numpy's global RNG and is checked by "
        "bracketing the reference's own draws",
        f"scGPT preprocess.py @ {SCGPT_REV}",
        cases,
    )

    # --- sample: distributional only ---------------------------------------
    cases = []
    for name, g, c in SAMPLE_CASES:
        p = _reference_sample_weights(c)
        cases.append(
            {
                "name": name,
                "gene_ids": g,
                "counts": c,
                "weight": "log1p",
                "replace": True,
                "expected_probabilities": [float(x) for x in p],
            }
        )
        assert abs(p.sum() - 1.0) < 1e-12 or p.sum() == 0.0
    write(
        "sample_reference.json",
        "DISTRIBUTIONAL ONLY, and a float64 RECONSTRUCTION of UCE's formula "
        "rather than UCE's own numbers. np.random.choice draws from numpy's "
        "global RNG, so no SCX output reproduces its draws. Separately, UCE "
        "computes its weights through torch, which may round them to float32 "
        "before numpy widens p for the cumsum; that would move CDF boundaries "
        "relative to the f64 arithmetic here and in SCX's kernel. A float32 "
        "reconstruction was attempted and produced a p numpy's own choice would "
        "reject (sum off by ~1e-7 against its 1.49e-8 tolerance), so the "
        "precision question is OPEN and needs a run of the real package. The "
        "frozen SCX draws live in sample_golden.json, whose source of truth is "
        "Rust (regenerate with the #[ignore]d test).",
        f"UCE eval_data.py sample_cell_sentences @ {UCE_REV}",
        cases,
    )

    if _blake3 is None:
        print(
            "\nWARNING: the `blake3` module is not installed, so no prefixes were "
            "printed. `uv pip install blake3` and re-run to get them.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
