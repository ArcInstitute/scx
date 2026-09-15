"""Collate contract v3 — per-row decoder query addressing.

`collate_cellset_gathered` grew one optional argument, `query_offsets`. Omitted,
the decoder query is the per-set `[n_sets, k_dec]` panel it has always been and
the output is byte-identical to contract v2's. Supplied, the query is ragged and
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
