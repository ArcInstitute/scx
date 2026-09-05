"""File operations integration tests."""

import numpy as np
import pytest
import scipy.sparse as sp


# ───────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────


def test_append_scx_to_scx(query_adata, scx_from_adata, tmp_dir):
    """append() — create two SCX files, append one to other, verify combined n_obs."""
    import pyscx

    path1 = scx_from_adata(query_adata, "base.scx")
    path2 = scx_from_adata(query_adata, "extra.scx")

    original_n = pyscx.open(path1).n_obs
    pyscx.append(path1, path2)

    new_n = pyscx.open(path1).n_obs
    assert new_n == original_n * 2


def test_append_honors_codec(query_adata, scx_from_adata):
    """P0 #2: pyscx.append plumbs the explicit codec string through to scx_ops::append.

    The Rust-level test (test_append_with_explicit_codec_zstd in
    scx-ops/tests/integration.rs) asserts the on-disk shard codec_id;
    this test just confirms the Python string parser reaches Rust
    without panicking and the appended cells round-trip.
    """
    import pyscx

    path1 = scx_from_adata(query_adata, "codec1.scx")
    path2 = scx_from_adata(query_adata, "codec2.scx")
    original_n = pyscx.open(path1).n_obs

    pyscx.append(path1, path2, codec="zstd")
    assert pyscx.open(path1).n_obs == original_n * 2


def test_append_rejects_nonpositive_shard_size(query_adata, scx_from_adata):
    """P0 #1 + P0 #9: pyscx.append / append_from_anndata reject shard_size <= 0
    with ValueError before crossing into Rust. Covers both shard_size=0 and
    negative integers (which previously would have surfaced as OverflowError
    from pyo3 u32 extraction).
    """
    import anndata
    import pandas as pd
    import pyscx

    path1 = scx_from_adata(query_adata, "z1.scx")
    path2 = scx_from_adata(query_adata, "z2.scx")

    # Zero case (P0 #1)
    with pytest.raises(ValueError, match="shard_size must be > 0"):
        pyscx.append(path1, path2, shard_size=0)

    # Negative case (P0 #9 — must be ValueError, not OverflowError)
    with pytest.raises(ValueError, match="shard_size must be > 0"):
        pyscx.append(path1, path2, shard_size=-1)
    with pytest.raises(ValueError, match="shard_size must be > 0"):
        pyscx.append(path1, path2, shard_size=-100)

    # Same checks for append_from_anndata.
    n_new = 4
    n_vars = query_adata.n_vars
    dense = np.zeros((n_new, n_vars), dtype=np.float32)
    dense[0, 0] = 1
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(["T cell"] * n_new)},
        index=[f"z_{i}" for i in range(n_new)],
    )
    var = query_adata.var.copy()
    new_adata = anndata.AnnData(X=x, obs=obs, var=var)

    with pytest.raises(ValueError, match="shard_size must be > 0"):
        pyscx.append_from_anndata(path1, new_adata, shard_size=0)
    with pytest.raises(ValueError, match="shard_size must be > 0"):
        pyscx.append_from_anndata(path1, new_adata, shard_size=-1)


def test_append_from_anndata(query_adata, scx_from_adata):
    """append_from_anndata() — append AnnData, verify all cells present."""
    import anndata
    import pyscx

    path = scx_from_adata(query_adata, "base.scx")
    original_n = pyscx.open(path).n_obs

    # Create a small new AnnData with matching n_vars
    np.random.seed(77)
    n_new = 30
    n_vars = query_adata.n_vars
    dense = np.random.randint(0, 50, size=(n_new, n_vars)).astype(np.float32)
    x = sp.csr_matrix(dense)
    import pandas as pd
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(["T cell"] * n_new),
         "tissue": pd.Categorical(["lung"] * n_new)},
        index=[f"new_{i}" for i in range(n_new)],
    )
    var = query_adata.var.copy()
    new_adata = anndata.AnnData(X=x, obs=obs, var=var)

    pyscx.append_from_anndata(path, new_adata)
    assert pyscx.open(path).n_obs == original_n + n_new


def test_mark_deleted_indices(query_adata, scx_from_adata):
    """mark_deleted() with explicit indices — deleted cells excluded from to_anndata()."""
    import pyscx

    path = scx_from_adata(query_adata, "del.scx")
    original_n = pyscx.open(path).n_obs
    indices_to_delete = [0, 5, 10, 42]

    total = pyscx.mark_deleted(path, indices_to_delete)
    assert total == len(indices_to_delete)

    # Reopen to see filtered results
    adata = pyscx.open(path).to_anndata()
    assert adata.n_obs == original_n - len(indices_to_delete)


def test_n_obs_reflects_deletions(query_adata, scx_from_adata):
    """B3: after mark_deleted, `n_obs` and the repr report the LOGICAL (live)
    count, consistent with `to_anndata().n_obs` / `query().count()`;
    `n_obs_physical` preserves the raw, pre-deletion header count."""
    import pyscx

    path = scx_from_adata(query_adata, "del_nobs.scx")
    exp0 = pyscx.open(path)
    physical = exp0.n_obs
    # No deletions yet: logical == physical.
    assert not exp0.has_deletions
    assert exp0.n_obs_physical == physical

    indices_to_delete = [0, 5, 10, 42]
    pyscx.mark_deleted(path, indices_to_delete)
    n_deleted = len(indices_to_delete)
    logical = physical - n_deleted

    exp = pyscx.open(path)
    assert exp.has_deletions
    assert exp.n_obs_physical == physical               # raw header count
    assert exp.n_obs == logical                          # now logical
    assert exp.n_obs == exp.query().count()              # matches engine
    assert exp.n_obs == exp.to_anndata().n_obs           # matches materialization
    # The AnnData-style repr header reports the logical count, not the physical.
    assert f"= {logical} ×" in repr(exp)
    assert f"= {physical} ×" not in repr(exp)


def test_mark_deleted_rejects_oob_index(query_adata, scx_from_adata):
    """mark_deleted() rejects positive cell indices >= n_obs."""
    import pyscx

    path = scx_from_adata(query_adata, "del_oob.scx")
    n_obs = pyscx.open(path).n_obs
    with pytest.raises(ValueError, match="out of bounds"):
        pyscx.mark_deleted(path, [0, n_obs + 1_000_000])


def test_mark_deleted_boolean_mask(query_adata, scx_from_adata):
    """PyExperiment.mark_deleted(mask) with boolean mask."""
    import pyscx

    path = scx_from_adata(query_adata, "mask_del.scx")
    exp = pyscx.open(path)
    original_n = exp.n_obs

    # Create a mask: delete all NK cells (last 40)
    mask = np.array([False] * 80 + [True] * 40)
    total = exp.mark_deleted(mask)
    assert total == 40

    # Reopen to verify
    adata = pyscx.open(path).to_anndata()
    assert adata.n_obs == original_n - 40


def test_mark_deleted_mask_too_short(query_adata, scx_from_adata):
    """mark_deleted() rejects masks shorter than n_obs."""
    import pyscx

    path = scx_from_adata(query_adata, "mask_short.scx")
    exp = pyscx.open(path)
    short_mask = np.zeros(exp.n_obs - 10, dtype=bool)
    with pytest.raises(ValueError, match="n_obs_physical"):
        exp.mark_deleted(short_mask)


def test_mark_deleted_accepts_a_live_length_mask(query_adata, scx_from_adata):
    """After a first deletion `read_obs()` is the live frame (0.17), so a mask
    derived from it has `n_obs` entries; it is expanded onto the physical axis.
    A physical-length mask keeps working; any other length names both counts."""
    import pyscx

    path = scx_from_adata(query_adata, "mask_live.scx")
    exp = pyscx.open(path)
    n = exp.n_obs_physical
    exp.mark_deleted(np.arange(n) < 5)  # physical rows 0..4
    assert exp.n_obs == n - 5

    obs = exp.read_obs()
    live_mask = np.zeros(len(obs), dtype=bool)
    live_mask[0] = True  # the first LIVE cell = physical row 5
    total = exp.mark_deleted(live_mask)
    assert total == 6
    assert exp.n_obs == n - 6
    assert list(exp.read_obs().index) == list(obs.index[1:])

    with pytest.raises(ValueError) as e:
        exp.mark_deleted(np.zeros(n - 7, dtype=bool))
    assert f"n_obs = {n - 6}" in str(e.value) and f"n_obs_physical = {n}" in str(e.value)


def test_mark_deleted_refuses_a_reordered_series_mask(query_adata, scx_from_adata):
    """A Series from a sorted `read_obs()` frame has the right length and every
    entry on the wrong cell; its index says so. A Series in the file's order
    (and a plain array) is accepted; a non-bool Series is refused."""
    import pandas as pd
    import pyscx

    path = scx_from_adata(query_adata, "mask_series.scx")
    exp = pyscx.open(path)
    exp.mark_deleted(np.arange(exp.n_obs_physical) < 2)
    obs = exp.read_obs()
    flag = pd.Series(np.arange(len(obs)) == 0, index=obs.index)

    with pytest.raises(ValueError, match="different order"):
        exp.mark_deleted(flag.iloc[::-1])
    with pytest.raises(TypeError, match="boolean"):
        exp.mark_deleted(pd.Series(np.arange(len(obs)), index=obs.index))
    assert exp.mark_deleted(flag) == 3
    assert list(exp.read_obs().index) == list(obs.index[1:])


def test_mark_deleted_multiindex_series_is_order_checked_as_a_composite(
    query_adata, scx_from_adata
):
    """A MultiIndex Series neither crashes nor bypasses the order check: its
    levels compare as the composite key the key join builds."""
    import pandas as pd
    import pyscx

    path = scx_from_adata(query_adata, "mask_mi.scx")
    obs = pyscx.open(path).read_obs()
    obs["sample"] = ["s%d" % (i % 3) for i in range(len(obs))]
    obs["barcode"] = list(obs.index)
    pyscx.modify_metadata(path, obs=obs.set_index(["sample", "barcode"]))
    exp = pyscx.open(path)
    exp.mark_deleted(np.arange(exp.n_obs_physical) < 2)
    live = exp.read_obs()
    assert isinstance(live.index, pd.MultiIndex)
    flag = pd.Series(np.arange(len(live)) == 0, index=live.index)

    with pytest.raises(ValueError, match="different order"):
        exp.mark_deleted(flag.iloc[::-1])
    assert exp.mark_deleted(flag) == 3
    assert exp.n_obs == exp.n_obs_physical - 3

    with pytest.raises(ValueError) as e:
        exp.mark_deleted(np.ones(5, dtype=bool))
    assert str(e.value).startswith("mark_deleted: mask has 5 rows"), str(e.value)


def test_mark_deleted_mask_too_long(query_adata, scx_from_adata):
    """mark_deleted() rejects masks longer than n_obs."""
    import pyscx

    path = scx_from_adata(query_adata, "mask_long.scx")
    exp = pyscx.open(path)
    long_mask = np.zeros(exp.n_obs + 10, dtype=bool)
    with pytest.raises(ValueError, match="n_obs_physical"):
        exp.mark_deleted(long_mask)


def test_mark_deleted_mask_exact_length(query_adata, scx_from_adata):
    """mark_deleted() accepts masks of exactly n_obs length."""
    import pyscx

    path = scx_from_adata(query_adata, "mask_exact.scx")
    exp = pyscx.open(path)
    exact_mask = np.zeros(exp.n_obs, dtype=bool)
    exact_mask[0] = True
    exact_mask[-1] = True
    total = exp.mark_deleted(exact_mask)
    assert total == 2


def test_compact(query_adata, tmp_dir):
    """After append + delete, compact produces smaller valid file."""
    import anndata
    import os
    import pandas as pd
    import pyscx

    # Create base file with non-categorical obs to avoid concat issues
    np.random.seed(42)
    n_obs, n_vars = 60, 40
    dense = np.random.randint(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=x, obs=obs, var=var)

    path = str(tmp_dir / "compact_base.scx")
    pyscx.from_anndata(adata, path)

    # Create extra file to append
    adata_extra = adata.copy()
    adata_extra.obs.index = [f"extra_{i}" for i in range(n_obs)]
    adata_extra.obs["cell_id"] = [f"extra_{i}" for i in range(n_obs)]
    path_extra = str(tmp_dir / "compact_extra.scx")
    pyscx.from_anndata(adata_extra, path_extra)

    # Append
    pyscx.append(path, path_extra)

    # Delete some cells
    pyscx.mark_deleted(path, [0, 1, 2, 3, 4])

    original_size = os.path.getsize(path)
    expected_n = pyscx.open(path).to_anndata().n_obs

    compacted = str(tmp_dir / "compacted.scx")
    pyscx.compact(path, compacted)

    compacted_size = os.path.getsize(compacted)
    assert compacted_size <= original_size  # should be smaller or same

    # Verify same cell count
    adata_result = pyscx.open(compacted).to_anndata()
    assert adata_result.n_obs == expected_n


def test_rollback(query_adata, scx_from_adata):
    """append → rollback → n_obs matches original."""
    import pyscx

    path = scx_from_adata(query_adata, "rollback.scx")
    original_n = pyscx.open(path).n_obs

    # Append
    path2 = scx_from_adata(query_adata, "rollback_extra.scx")
    pyscx.append(path, path2)
    assert pyscx.open(path).n_obs == original_n * 2

    # Rollback
    pyscx.rollback(path)
    assert pyscx.open(path).n_obs == original_n


def test_rollback_to_seq(query_adata, scx_from_adata):
    """Multiple appends → rollback to specific version."""
    import pyscx

    path = scx_from_adata(query_adata, "rollback_seq.scx")
    original_n = pyscx.open(path).n_obs

    # Two appends
    path_extra1 = scx_from_adata(query_adata, "rb_extra1.scx")
    pyscx.append(path, path_extra1)
    n_after_first = pyscx.open(path).n_obs

    path_extra2 = scx_from_adata(query_adata, "rb_extra2.scx")
    pyscx.append(path, path_extra2)
    assert pyscx.open(path).n_obs == original_n * 3

    # Rollback to seq 2 (after first append)
    pyscx.rollback(path, to_seq=2)
    assert pyscx.open(path).n_obs == n_after_first


def test_merge(query_adata, scx_from_adata, tmp_dir):
    """Merge 3 files → output has all cells."""
    import pyscx

    p1 = scx_from_adata(query_adata, "merge1.scx")
    p2 = scx_from_adata(query_adata, "merge2.scx")
    p3 = scx_from_adata(query_adata, "merge3.scx")

    output = str(tmp_dir / "merged.scx")
    pyscx.merge([p1, p2, p3], output)

    merged = pyscx.open(output)
    assert merged.n_obs == query_adata.n_obs * 3
    assert merged.n_vars == query_adata.n_vars


def test_merge_mismatched_nvars(synthetic_adata, query_adata, scx_from_adata, tmp_dir):
    """merge() with mismatched n_vars → error."""
    import pyscx

    p1 = scx_from_adata(synthetic_adata, "merge_a.scx")  # 50 vars
    p2 = scx_from_adata(query_adata, "merge_b.scx")  # 40 vars
    output = str(tmp_dir / "bad_merge.scx")

    with pytest.raises((ValueError, RuntimeError)):
        pyscx.merge([p1, p2], output)


def test_merge_single_file(query_adata, scx_from_adata, tmp_dir):
    """merge() with single file → error."""
    import pyscx

    p1 = scx_from_adata(query_adata, "single.scx")
    output = str(tmp_dir / "single_out.scx")

    with pytest.raises(ValueError, match="at least 2"):
        pyscx.merge([p1], output)
