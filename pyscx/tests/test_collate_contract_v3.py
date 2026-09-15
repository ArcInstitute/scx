"""Collate contract v3 — per-row decoder query addressing.

`collate_cellset_gathered` grew one optional argument, `query_offsets`. Omitted,
the decoder query is the per-set `[n_sets, k_dec]` panel it has always been and
every **pre-existing field** is byte-identical to contract v2's. The returned
dict is not identical: `target_pad_mask` is a new key on every path, all zeros
on the per-set one. A consumer that unpacks named keys is unaffected; one that
asserts an exact key set is not — which is part of what the bump is for. Supplied, the query is ragged and
indexed per row, `n_measured` becomes per-row, `enc_mask_positions` becomes
parallel to the ragged query, and `target_pad_mask` says which target slots are
padding rather than a real zero.

Appendix A.5's PR S1 is the consumer side of this bump; it asserts
`COLLATE_CELLSET_CONTRACT_VERSION == 3` at setup and has not landed, so a
consumer pinned to v2 fails loudly here rather than silently mis-reading a
batch.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyscx


def one_set_two_rows():
    """Two rows, genes {1, 3, 5}, in one set."""
    return dict(
        indptr=np.array([0, 3, 6], dtype=np.int64),
        indices=np.array([1, 3, 5, 1, 3, 5], dtype=np.int32),
        data=np.array([4.0, 2.0, 6.0, 1.0, 9.0, 3.0], dtype=np.float32),
        set_offsets=np.array([0, 2], dtype=np.int64),
        cell_indices=np.array([0, 1], dtype=np.uint64),
        file_ids=np.zeros(2, dtype=np.uint32),
        role_tags=np.zeros(2, dtype=np.int32),
        hide_readout=np.zeros(2, dtype=np.uint8),
    )


def collate(*, k_dec, query, offsets=None, mask=None, n_measured, k_enc=4, mode="pass_through"):
    b = one_set_two_rows()
    return pyscx.collate_cellset_gathered(
        b["indptr"],
        b["indices"],
        b["data"],
        b["set_offsets"],
        b["cell_indices"],
        b["file_ids"],
        b["role_tags"],
        k_dec,
        np.asarray(query, dtype=np.int32),
        np.asarray(mask if mask is not None else [], dtype=np.uint8),
        b["hide_readout"],
        np.asarray(n_measured, dtype=np.uint32),
        k_enc,
        mode,
        8,
        query_offsets=None if offsets is None else np.asarray(offsets, dtype=np.int64),
    )


def test_contract_version_is_three():
    assert pyscx.COLLATE_CELLSET_CONTRACT_VERSION == 3


def test_the_batch_carries_target_pad_mask():
    out = collate(k_dec=3, query=[1, 3, 5], n_measured=[3])
    assert "target_pad_mask" in out
    assert out["target_pad_mask"].dtype == np.uint8
    assert out["target_pad_mask"].shape == out["target_counts"].shape


def test_target_pad_mask_is_all_zero_on_the_per_set_path():
    """The guard on "v2 callers are unchanged": the new field exists but says
    nothing, because a per-set query is the same width for every row."""
    out = collate(k_dec=3, query=[1, 3, 5], n_measured=[3])
    assert out["target_pad_mask"].tolist() == [0] * 6


def test_per_row_queries_reproduce_the_per_set_call_field_for_field():
    """v2's addressing is a special case of v3's."""
    mask = [1, 0, 0, 0, 0, 1]
    per_set = collate(k_dec=3, query=[1, 3, 5], mask=mask, n_measured=[3])
    per_row = collate(
        k_dec=3,
        query=[1, 3, 5, 1, 3, 5],
        offsets=[0, 3, 6],
        mask=mask,
        n_measured=[3, 3],
    )
    for key, value in per_set.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, per_row[key], err_msg=key)
        else:
            assert value == per_row[key], key


def test_per_row_queries_let_two_rows_of_one_set_differ():
    out = collate(k_dec=1, query=[1, 3], offsets=[0, 1, 2], n_measured=[3, 3])
    assert out["target_counts"].tolist() == [4.0, 9.0]


def test_short_rows_are_padded_and_the_padding_is_marked():
    out = collate(k_dec=2, query=[1, 3, 5], offsets=[0, 2, 3], n_measured=[3, 3])
    assert out["target_counts"].tolist() == [4.0, 2.0, 3.0, 0.0]
    assert out["target_pad_mask"].tolist() == [0, 0, 0, 1]


def test_a_real_zero_target_is_distinguishable_from_padding():
    # Gene 7 is in neither row: a genuine 0.0 with pad 0.
    out = collate(k_dec=2, query=[7, 1, 3], offsets=[0, 2, 3], n_measured=[3, 3])
    assert out["target_counts"].tolist() == [0.0, 4.0, 9.0, 0.0]
    assert out["target_pad_mask"].tolist() == [0, 0, 0, 1]


def test_each_row_consults_its_own_query_panel():
    # Both rows flag their single query position but query different genes, so a
    # lookup that reused the set's first panel would withhold gene 1 from both.
    out = collate(k_dec=1, query=[1, 3], offsets=[0, 1, 2], mask=[1, 1], n_measured=[3, 3])
    assert out["encoder_gene_ids"][0:2].tolist() == [5, 3]
    assert out["encoder_gene_ids"][4:6].tolist() == [5, 1]


@pytest.mark.parametrize(
    "offsets,query,n_measured,why",
    [
        ([0, 3], [1, 3, 5], [3, 3], "wrong length"),
        ([0, 5, 3], [1, 3, 5], [3, 3], "non-monotonic"),
        ([0, 1, 2], [1, 3, 5], [3, 3], "does not span the flat query"),
        ([0, 2, 3], [1, 3, 5], [3], "n_measured still per-set"),
    ],
)
def test_malformed_query_offsets_raise_rather_than_panicking(offsets, query, n_measured, why):
    with pytest.raises(Exception) as excinfo:
        collate(k_dec=3, query=query, offsets=offsets, n_measured=n_measured)
    assert "collate_gathered" in str(excinfo.value), why


def test_k_dec_narrower_than_the_widest_row_query_is_refused():
    with pytest.raises(Exception, match="narrower than the widest per-row query"):
        collate(k_dec=1, query=[1, 3, 5], offsets=[0, 2, 3], n_measured=[3, 3])


# ---------------------------------------------------------------------------
# Panics this entry could reach across the FFI
# ---------------------------------------------------------------------------
#
# `validate_indptr` closed the non-monotonic case in an earlier PR; these three
# are its siblings, each reproduced against the built extension first.


def _raw(**over):
    b = one_set_two_rows()
    args = dict(
        indptr=b["indptr"], indices=b["indices"], data=b["data"],
        set_offsets=b["set_offsets"], cell_indices=b["cell_indices"],
        file_ids=b["file_ids"], role_tags=b["role_tags"], k_dec=1,
        query=np.array([1], dtype=np.int32), mask=np.array([], dtype=np.uint8),
        hide=b["hide_readout"], n_measured=np.array([3], dtype=np.uint32),
        k_enc=4, mode="pass_through", n_genes=8, alpha=None,
    )
    args.update(over)
    return pyscx.collate_cellset_gathered(
        args["indptr"], args["indices"], args["data"], args["set_offsets"],
        args["cell_indices"], args["file_ids"], args["role_tags"], args["k_dec"],
        args["query"], args["mask"], args["hide"], args["n_measured"],
        args["k_enc"], args["mode"], args["n_genes"], pflog_alpha=args["alpha"],
    )


def test_zero_k_enc_or_k_dec_raises_rather_than_panicking_in_rayon():
    # `par_chunks_mut(0)` panics with "chunk_size must not be zero".
    with pytest.raises(Exception, match="must be >= 1"):
        _raw(k_enc=0)
    with pytest.raises(Exception, match="must be >= 1"):
        _raw(k_dec=0)


def test_indices_shorter_than_data_raises_rather_than_slicing_out_of_bounds():
    with pytest.raises(Exception, match="indices"):
        _raw(indices=np.array([1, 3], dtype=np.int32),
             data=np.array([4.0, 2.0, 6.0], dtype=np.float32))


def test_pflog_with_a_zero_n_measured_raises_instead_of_writing_minus_inf():
    # Centring by a zero denominator wrote -inf into every encoder slot; the
    # kernel's own doc claimed the collator rejected it, and it did not.
    with pytest.raises(Exception, match="n_measured >= 1"):
        _raw(mode="pflog_raw", alpha=0.25,
             n_measured=np.array([0], dtype=np.uint32))
    # Anti-vacuity: the same call with a real panel size succeeds.
    out = _raw(mode="pflog_raw", alpha=0.25,
               n_measured=np.array([3], dtype=np.uint32))
    assert np.isfinite(out["encoder_counts"]).all()


def test_malformed_set_offsets_raise_rather_than_misassigning_rows():
    # `set_offsets` was the one prefix array on this entry nothing validated:
    # `[0, 1]` over two rows was accepted and row 1 silently kept set 0.
    with pytest.raises(Exception, match="set_offsets"):
        _raw(set_offsets=np.array([0, 1], dtype=np.int64))
    with pytest.raises(Exception, match="set_offsets"):
        _raw(set_offsets=np.array([1, 2], dtype=np.int64))


def test_an_unsorted_row_raises_instead_of_gathering_zero_targets():
    # A row {0: 5, 2: 7, 1: 9} queried with [0, 1, 2] returned targets
    # [5.0, 0.0, 0.0] instead of [5.0, 9.0, 7.0] — corrupted batches, no error.
    with pytest.raises(Exception, match="strictly ascending"):
        pyscx.collate_cellset_gathered(
            np.array([0, 3], dtype=np.int64),
            np.array([0, 2, 1], dtype=np.int32),
            np.array([5.0, 7.0, 9.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.uint64),
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.int32),
            3,
            np.array([0, 1, 2], dtype=np.int32),
            np.array([], dtype=np.uint8),
            np.zeros(1, dtype=np.uint8),
            np.array([8], dtype=np.uint32),
            3,
            "pass_through",
            8,
        )


def test_passthrough_array_lengths_are_checked():
    with pytest.raises(Exception, match="file_ids"):
        _raw(file_ids=np.zeros(5, dtype=np.uint32))
    with pytest.raises(Exception, match="role_tags"):
        _raw(role_tags=np.zeros(5, dtype=np.int32))


def test_collate_rejects_a_non_positive_vocabulary():
    # The twin of `pyscx.tokenize.top_k`'s guard: `n_genes_total` fixes the
    # sentinels, and the per-row id check never runs on an empty row, so a
    # negative vocabulary emitted `[-5, -4]` as tokens.
    with pytest.raises(Exception, match="n_genes_total must be >= 1"):
        _raw(n_genes=-5)
    with pytest.raises(Exception, match="n_genes_total must be >= 1"):
        _raw(n_genes=0)
