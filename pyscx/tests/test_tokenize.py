"""`pyscx.tokenize` — the W6 tokenisation kernels over a gathered CSR batch.

Each kernel is checked against a numpy reference that transcribes the model's
own implementation at a pinned revision. Where the reference is stochastic the
claim is weaker and is stated as such rather than faked:

* **scGPT `binning`** (`cebd6fae`) randomises at bin edges via
  `np.random.rand`, so only its two deterministic bounds can be matched
  exactly; the randomised form is checked by *bracketing*.
* **UCE's sampler** draws from `np.random.choice`, so only the distribution is
  checked.
* **Geneformer's `rank_genes`** uses `np.argsort`'s default kind, which is
  quicksort and therefore unstable, so exactness holds on tie-free input and
  equality-up-to-tie-runs otherwise.

See `docs/tokenize.md` for the full divergence list.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyscx
import pyscx.tokenize as tok

from _gil_probe import largest_gap_during

MIN_MEASURABLE_S = 0.05
MAX_GAP_FRACTION = 0.5


def csr(rows: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (indptr, indices, data) triple from per-row (gene, value) pairs."""
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[float] = []
    for i, row in enumerate(rows):
        srt = sorted(row)
        indices += [g for g, _ in srt]
        data += [v for _, v in srt]
        indptr[i + 1] = len(indices)
    return indptr, np.asarray(indices, dtype=np.int32), np.asarray(data, dtype=np.float32)


def random_csr(n_rows: int, n_genes: int, nnz_per_row: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        genes = rng.choice(n_genes, size=nnz_per_row, replace=False)
        vals = rng.integers(1, 200, size=nnz_per_row).astype(np.float32)
        rows.append(list(zip(genes.tolist(), vals.tolist())))
    return csr(rows)


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


def _numpy_top_k(gene_ids, values, k, n_genes_total):
    """np.lexsort((gene_id, -value)) over the positive entries, then take k."""
    pad, mask_id = n_genes_total + 1, n_genes_total
    ids = np.full(k, pad, dtype=np.int64)
    vals = np.zeros(k, dtype=np.float32)
    pad_flags = np.ones(k, dtype=np.uint8)
    mask_flags = np.zeros(k, dtype=np.uint8)
    keep = np.clip(values, 0, None) > 0
    if not keep.any():
        ids[0], pad_flags[0] = mask_id, 0
        return ids, vals, mask_flags, pad_flags
    g, v = gene_ids[keep], np.clip(values[keep], 0, None)
    order = np.lexsort((g, -v))[:k]
    ids[: len(order)] = g[order]
    vals[: len(order)] = v[order]
    pad_flags[: len(order)] = 0
    return ids, vals, mask_flags, pad_flags


def test_top_k_matches_a_numpy_lexsort_reference():
    indptr, indices, data = random_csr(40, n_genes=500, nnz_per_row=30, seed=1)
    k, n_genes = 8, 500
    got = tok.top_k(indptr, indices, data, k, n_genes)
    for r in range(len(indptr) - 1):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        e_ids, e_vals, e_mask, e_pad = _numpy_top_k(indices[lo:hi], data[lo:hi], k, n_genes)
        assert got["ids"][r * k : (r + 1) * k].tolist() == e_ids.tolist()
        np.testing.assert_array_equal(got["values"][r * k : (r + 1) * k], e_vals)
        np.testing.assert_array_equal(got["mask"][r * k : (r + 1) * k], e_mask)
        np.testing.assert_array_equal(got["pad"][r * k : (r + 1) * k], e_pad)


def test_top_k_ties_break_by_gene_id_ascending():
    indptr, indices, data = csr([[(9, 2.0), (3, 2.0), (5, 2.0), (7, 2.0)]])
    got = tok.top_k(indptr, indices, data, 4, 100)
    assert got["ids"].tolist() == [3, 5, 7, 9]


def test_top_k_empty_row_gets_one_gene_mask_token():
    indptr, indices, data = csr([[], [(1, 5.0)]])
    got = tok.top_k(indptr, indices, data, 3, 100)
    assert got["ids"][:3].tolist() == [100, 101, 101]
    assert got["pad"][:3].tolist() == [0, 1, 1]
    assert got["mask"][:3].tolist() == [0, 0, 0]


def test_top_k_sentinels_agree_with_the_helpers():
    assert tok.gene_mask_id(100) == 100
    assert tok.pad_id(100) == 101


# ---------------------------------------------------------------------------
# rank_tokens
# ---------------------------------------------------------------------------


def _numpy_rank(gene_ids, values, stats, l_max, target_sum=1e4):
    """Transcription of Geneformer's tokenize_cell / rank_genes.

    `np.argsort(-v)` is the reference's own call; its default kind is quicksort,
    so this reference's order within an equal-value run is arbitrary too. The
    tests that use it either avoid ties or canonicalise both sides.
    """
    v = np.clip(values.astype(np.float64), 0, None)
    lib = v.sum()
    if lib <= 0:
        return np.empty(0, dtype=np.int64)
    norm = v / lib * target_sum / stats[gene_ids].astype(np.float64)
    nz = np.nonzero(norm)[0]
    return gene_ids[nz][np.argsort(-norm[nz])][:l_max].astype(np.int64)


def test_rank_tokens_matches_geneformer_exactly_on_tie_free_input():
    rng = np.random.default_rng(7)
    n_genes = 400
    # Irrational-ish statistics and distinct counts, so no two normalised values
    # collide: the case where the reference's order IS determined.
    stats = (rng.random(n_genes).astype(np.float32) * 0.9 + 0.05)
    rows = []
    for _ in range(25):
        genes = rng.choice(n_genes, size=40, replace=False)
        vals = rng.choice(np.arange(1, 4000), size=40, replace=False).astype(np.float32)
        rows.append(list(zip(genes.tolist(), vals.tolist())))
    indptr, indices, data = csr(rows)
    l_max = 16
    got = tok.rank_tokens(indptr, indices, data, stats, l_max, "geneformer-test")
    for r in range(len(indptr) - 1):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        want = _numpy_rank(indices[lo:hi], data[lo:hi], stats, l_max)
        n = int(got["lengths"][r])
        assert n == len(want)
        assert got["ids"][r * l_max : r * l_max + n].tolist() == want.tolist()


def test_rank_tokens_agrees_with_geneformer_up_to_tie_runs():
    """With ties present, agreement is only up to the order inside equal-value
    runs — because `np.argsort`'s default is unstable, the reference itself has
    no rule there. Canonicalise both sides by gene id within each run."""
    n_genes = 60
    stats = np.ones(n_genes, dtype=np.float32)
    # Every count is 1, so every normalised value ties.
    genes = list(range(0, 40))
    indptr, indices, data = csr([[(g, 1.0) for g in genes]])
    l_max = 40
    got = tok.rank_tokens(indptr, indices, data, stats, l_max, "v")
    want = _numpy_rank(indices, data, stats, l_max)
    assert sorted(got["ids"][:l_max].tolist()) == sorted(want.tolist())
    # SCX's declared rule is the stronger statement: id-ascending.
    assert got["ids"][:l_max].tolist() == sorted(want.tolist())


def test_rank_tokens_reports_the_norm_identity_and_it_covers_the_inputs():
    indptr, indices, data = csr([[(0, 1.0), (1, 2.0)]])
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    id_a = tok.rank_tokens(indptr, indices, data, a, 2, "v1")["norm_identity"]
    id_a2 = tok.rank_tokens(indptr, indices, data, a, 2, "v1")["norm_identity"]
    id_b = tok.rank_tokens(indptr, indices, data, b, 2, "v1")["norm_identity"]
    id_v2 = tok.rank_tokens(indptr, indices, data, a, 2, "v2")["norm_identity"]
    assert id_a == id_a2
    assert id_a != id_b, "statistics must be part of the identity"
    assert id_a != id_v2, "vocabulary version must be part of the identity"


def test_rank_tokens_refuses_a_non_positive_statistic():
    indptr, indices, data = csr([[(0, 1.0)]])
    with pytest.raises(ValueError, match="finite and strictly positive"):
        tok.rank_tokens(indptr, indices, data, np.array([0.0], dtype=np.float32), 1, "v")


def test_rank_tokens_refuses_a_gene_outside_the_vocabulary():
    indptr, indices, data = csr([[(5, 1.0)]])
    with pytest.raises(ValueError, match="outside the normalisation vocabulary"):
        tok.rank_tokens(indptr, indices, data, np.ones(3, dtype=np.float32), 1, "v")


# ---------------------------------------------------------------------------
# bin_values
# ---------------------------------------------------------------------------


def _scgpt_edges(row_values, n_bins):
    """scGPT's edge computation, verbatim: quantiles of the non-zero values."""
    nz = row_values[row_values.nonzero()]
    return np.quantile(nz, np.linspace(0, 1, n_bins - 1))


def test_bin_values_left_and_right_match_numpy_digitize():
    indptr, indices, data = random_csr(30, n_genes=300, nnz_per_row=25, seed=3)
    n_bins = 11
    left = tok.bin_values(indptr, indices, data, n_bins, tie="left")["bins"]
    right = tok.bin_values(indptr, indices, data, n_bins, tie="right")["bins"]
    for r in range(len(indptr) - 1):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        vals = data[lo:hi]
        edges = _scgpt_edges(vals, n_bins)
        np.testing.assert_array_equal(left[lo:hi], np.digitize(vals, edges))
        np.testing.assert_array_equal(right[lo:hi], np.digitize(vals, edges, right=True))


def test_bin_values_seeded_is_bracketed_by_the_two_bounds():
    """The strongest claim available about scGPT's randomised `_digitize`:
    every draw it can produce lies between the two deterministic bounds."""
    indptr, indices, data = csr([[(i, 4.0) for i in range(12)]])
    n_bins = 9
    left = tok.bin_values(indptr, indices, data, n_bins, tie="left")["bins"]
    right = tok.bin_values(indptr, indices, data, n_bins, tie="right")["bins"]
    seen = set()
    for row in range(150):
        got = tok.bin_values(
            indptr, indices, data, n_bins, tie="seeded", seed=3, file_identity=9,
            rows=np.array([row], dtype=np.uint64),
        )["bins"]
        assert np.all(got >= right) and np.all(got <= left)
        seen.update(got.tolist())
    # Not vacuous: the draws spread over the whole bracket.
    assert seen == set(range(1, n_bins))


def test_bin_values_zeros_stay_in_bin_zero():
    indptr, indices, data = csr([[(0, 0.0), (1, 5.0), (2, 1.0)]])
    got = tok.bin_values(indptr, indices, data, 5, tie="left")["bins"]
    assert got[0] == 0
    assert got[1] > 0 and got[2] > 0


def test_bin_values_fixed_edges_are_used_as_given():
    indptr, indices, data = csr([[(0, 0.5), (1, 1.5), (2, 2.5), (3, 3.5)]])
    edges = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    got = tok.bin_values(indptr, indices, data, 5, edges=edges, tie="left")["bins"]
    assert got.tolist() == [0, 1, 2, 3]


def test_bin_values_rejects_an_unknown_tie():
    indptr, indices, data = csr([[(0, 1.0)]])
    with pytest.raises(ValueError, match="unknown tie"):
        tok.bin_values(indptr, indices, data, 5, tie="middle")


# ---------------------------------------------------------------------------
# sample_genes
# ---------------------------------------------------------------------------


def test_sample_genes_draws_with_replacement_matching_uce():
    indptr, indices, data = csr([[(3, 5.0), (8, 5.0)]])
    got = tok.sample_genes(indptr, indices, data, 64, seed=1, file_identity=2)
    assert int(got["lengths"][0]) == 64
    counts = np.bincount(got["ids"], minlength=9)
    assert counts[3] > 1 and counts[8] > 1, "with replacement, not a permutation"


def test_sample_genes_weights_are_log1p_of_the_counts():
    """The distributional claim, since `np.random.choice`'s draws cannot be
    reproduced. Counts 1 and 1000: log1p weighting gives gene 1 ~9% of the
    draws, linear weighting ~0.1%."""
    n = 100_000
    indptr, indices, data = csr([[(1, 1.0), (2, 1000.0)]])
    got = tok.sample_genes(indptr, indices, data, n, seed=5, file_identity=6)
    p1 = float((got["ids"] == 1).mean())
    expect = np.log1p(1.0) / (np.log1p(1.0) + np.log1p(1000.0))
    assert abs(p1 - expect) < 0.01, f"{p1} vs {expect}"
    lin = tok.sample_genes(
        indptr, indices, data, n, seed=5, file_identity=6, weight="linear"
    )
    assert float((lin["ids"] == 1).mean()) < 0.01


def test_sample_genes_is_keyed_on_content_not_batch_position():
    """The `rows` argument exists so a cell drawn in two different batches gets
    the same sample. Without it the key is the batch position, which changes
    when the plan does."""
    indptr, indices, data = csr([[(1, 4.0), (2, 9.0), (3, 2.0), (4, 7.0)]])
    a = tok.sample_genes(
        indptr, indices, data, 32, seed=1, file_identity=2,
        rows=np.array([77], dtype=np.uint64),
    )["ids"]
    b = tok.sample_genes(
        indptr, indices, data, 32, seed=1, file_identity=2,
        rows=np.array([77], dtype=np.uint64),
    )["ids"]
    c = tok.sample_genes(
        indptr, indices, data, 32, seed=1, file_identity=2,
        rows=np.array([78], dtype=np.uint64),
    )["ids"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_genes_reports_zero_for_a_row_with_no_weight():
    indptr, indices, data = csr([[(1, 0.0), (2, 0.0)], [(3, 5.0)]])
    got = tok.sample_genes(indptr, indices, data, 4, seed=1, file_identity=2)
    assert got["lengths"].tolist() == [0, 4]


def test_sample_genes_rejects_an_unknown_weight():
    indptr, indices, data = csr([[(1, 1.0)]])
    with pytest.raises(ValueError, match="unknown weight"):
        tok.sample_genes(indptr, indices, data, 2, seed=1, file_identity=2, weight="sqrt")


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def test_transform_values_matches_numpy_for_each_mode():
    indptr, indices, data = random_csr(20, n_genes=200, nnz_per_row=15, seed=11)
    np.testing.assert_array_equal(
        tok.transform_values(indptr, data, "pass_through"), np.clip(data, 0, None)
    )
    np.testing.assert_allclose(
        tok.transform_values(indptr, data, "log1p_raw"),
        np.log1p(np.clip(data, 0, None)),
        rtol=1e-6,
    )
    got = tok.transform_values(indptr, data, "normalize_log1p", target_sum=1e4)
    for r in range(len(indptr) - 1):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        v = np.clip(data[lo:hi], 0, None)
        np.testing.assert_allclose(
            got[lo:hi], np.log1p(v * np.float32(1e4 / v.sum())), rtol=1e-5
        )


def test_transform_values_pflog_requires_its_parameters():
    indptr, indices, data = csr([[(1, 3.0)]])
    with pytest.raises(ValueError, match="pflog_alpha"):
        tok.transform_values(indptr, data, "pflog_raw")
    with pytest.raises(ValueError, match="n_measured"):
        tok.transform_values(indptr, data, "pflog_raw", pflog_alpha=0.25)
    out = tok.transform_values(indptr, data, "pflog_raw", pflog_alpha=0.25, n_measured=8)
    assert out.shape == data.shape


def test_library_size_is_the_row_sum_with_negatives_clipped():
    indptr, indices, data = csr([[(1, 3.0), (2, -2.0), (3, 5.0)], [], [(4, 1.0)]])
    np.testing.assert_array_equal(tok.library_size(indptr, data), [8.0, 0.0, 1.0])


def test_measured_mask_marks_panel_positions_the_row_carries():
    indptr, indices, data = csr([[(2, 1.0), (5, 1.0)], [(5, 1.0)]])
    panel = np.array([5, 2, 9], dtype=np.int32)
    got = tok.measured_mask(indptr, indices, panel)
    assert got.tolist() == [1, 1, 0, 1, 0, 0]


# ---------------------------------------------------------------------------
# Boundary behaviour shared by every entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda ip, ix, d: tok.top_k(ip, ix, d, 2, 100),
        lambda ip, ix, d: tok.rank_tokens(ip, ix, d, np.ones(100, dtype=np.float32), 2, "v"),
        lambda ip, ix, d: tok.bin_values(ip, ix, d, 5),
        lambda ip, ix, d: tok.sample_genes(ip, ix, d, 2, 1, 2),
    ],
    ids=["top_k", "rank_tokens", "bin_values", "sample_genes"],
)
def test_a_malformed_indptr_raises_rather_than_panicking(call):
    # Non-monotonic but with the right last element: passes a `last == nnz`
    # check and then slices out of bounds.
    bad = np.array([0, 3, 2], dtype=np.int64)
    ix = np.array([1, 2], dtype=np.int32)
    d = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        call(bad, ix, d)


@pytest.mark.parametrize(
    "call",
    [
        lambda ip, ix, d: tok.top_k(ip, ix, d, 2, 100),
        lambda ip, ix, d: tok.rank_tokens(ip, ix, d, np.ones(100, dtype=np.float32), 2, "v"),
        lambda ip, ix, d: tok.bin_values(ip, ix, d, 5),
        lambda ip, ix, d: tok.sample_genes(ip, ix, d, 2, 1, 2),
        lambda ip, ix, d: tok.transform_values(ip, d, "pass_through"),
    ],
    ids=["top_k", "rank_tokens", "bin_values", "sample_genes", "transform_values"],
)
def test_an_empty_batch_is_accepted(call):
    ip = np.array([0], dtype=np.int64)
    ix = np.array([], dtype=np.int32)
    d = np.array([], dtype=np.float32)
    call(ip, ix, d)


def test_outputs_carry_the_declared_dtypes():
    indptr, indices, data = random_csr(5, n_genes=100, nnz_per_row=10, seed=2)
    assert tok.top_k(indptr, indices, data, 3, 100)["ids"].dtype == np.int64
    assert tok.top_k(indptr, indices, data, 3, 100)["values"].dtype == np.float32
    assert tok.top_k(indptr, indices, data, 3, 100)["pad"].dtype == np.uint8
    r = tok.rank_tokens(indptr, indices, data, np.ones(100, dtype=np.float32), 3, "v")
    assert r["ids"].dtype == np.int64 and r["lengths"].dtype == np.uint32
    assert tok.bin_values(indptr, indices, data, 5)["bins"].dtype == np.int64
    assert tok.library_size(indptr, data).dtype == np.float64
    assert tok.measured_mask(indptr, indices, np.array([1], dtype=np.int32)).dtype == np.uint8


# ---------------------------------------------------------------------------
# GIL release
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big_batch():
    # Sized so each kernel clears MIN_MEASURABLE_S; a batch too small skips on
    # every run and reads as coverage.
    return random_csr(6000, n_genes=20_000, nnz_per_row=600, seed=99)


@pytest.mark.parametrize(
    "name",
    ["top_k", "rank_tokens", "bin_values", "sample_genes"],
)
def test_kernels_release_the_gil(big_batch, name):
    indptr, indices, data = big_batch
    stats = np.ones(20_000, dtype=np.float32)
    ops = {
        "top_k": lambda: tok.top_k(indptr, indices, data, 2048, 20_000),
        "rank_tokens": lambda: tok.rank_tokens(indptr, indices, data, stats, 2048, "v"),
        "bin_values": lambda: tok.bin_values(indptr, indices, data, 51),
        "sample_genes": lambda: tok.sample_genes(indptr, indices, data, 1024, 1, 2),
    }
    duration, largest_gap = largest_gap_during(ops[name])
    if duration < MIN_MEASURABLE_S:
        pytest.skip(f"{name} too fast to measure ({duration * 1000:.1f} ms)")
    assert largest_gap < MAX_GAP_FRACTION * duration, (
        f"{name} held the GIL: largest monitor gap {largest_gap:.4f}s of a "
        f"{duration:.4f}s call"
    )


def test_contract_version_is_reachable_from_both_spellings():
    assert tok.CONTRACT_VERSION == tok.contract_version()
    assert pyscx.tokenize.CONTRACT_VERSION == tok.CONTRACT_VERSION
