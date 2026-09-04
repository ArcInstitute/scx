"""Tests for pyscx.iter_chunks() — shard-aligned chunk iteration.

Validates:
- Complete row coverage across chunks
- Correct obs metadata slicing per chunk
- Concatenated chunk X matches full materialization
- Deletion vector support
- Fixed-size chunking via chunk_size=int
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def backed_scx_path(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic_adata for chunk iterator tests."""
    import pyscx

    path = str(tmp_dir / "chunk_iter_test.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


def test_iter_chunks_covers_all_rows(backed_scx_path):
    """All rows are covered by the chunks (no gaps, no overlaps)."""
    import pyscx

    adata = pyscx.open(backed_scx_path).to_anndata(backed=True)
    total_rows = 0
    for chunk in pyscx.iter_chunks(adata):
        total_rows += chunk.n_obs

    assert total_rows == adata.n_obs


def test_iter_chunks_obs_metadata(backed_scx_path):
    """Each chunk's obs has the correct subset of rows."""
    import pyscx

    adata = pyscx.open(backed_scx_path).to_anndata(backed=True)

    # Collect all obs indices from chunks
    all_indices = []
    for chunk in pyscx.iter_chunks(adata):
        all_indices.extend(chunk.obs.index.tolist())

    # Should match the full obs
    expected_indices = adata.obs.index.tolist()
    assert all_indices == expected_indices


def test_iter_chunks_x_matches_full(backed_scx_path):
    """Concatenated chunk X matrices match full materialization."""
    import pyscx

    adata = pyscx.open(backed_scx_path).to_anndata(backed=True)

    # Collect all chunks
    chunks = list(pyscx.iter_chunks(adata))
    assert len(chunks) > 0

    # Stack all chunk X matrices
    concatenated = sp.vstack([chunk.X for chunk in chunks])

    # Compare with full materialization
    full_x = adata.X.to_memory()

    np.testing.assert_array_equal(
        concatenated.toarray(), full_x.toarray()
    )


def test_iter_chunks_with_deletions(tmp_dir):
    """Chunks correctly exclude deleted rows."""
    import anndata
    import pyscx

    np.random.seed(99)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "chunk_del.scx")
    pyscx.from_anndata(adata, path)

    # Mark some cells as deleted
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[5] = True
    delete_mask[n_obs - 1] = True
    n_deleted = int(delete_mask.sum())

    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)

    # Backed mode with deletions
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.n_obs == n_obs - n_deleted

    # Chunks should cover exactly the kept rows
    total_rows = 0
    chunk_x_parts = []
    for chunk in pyscx.iter_chunks(adata_backed):
        total_rows += chunk.n_obs
        chunk_x_parts.append(chunk.X)

    assert total_rows == n_obs - n_deleted

    # Concatenated chunks should match full backed materialization
    concatenated = sp.vstack(chunk_x_parts)
    full_x = adata_backed.X.to_memory()
    np.testing.assert_array_equal(
        concatenated.toarray(), full_x.toarray()
    )


def test_iter_chunks_fixed_size(backed_scx_path):
    """chunk_size=int yields fixed-size chunks."""
    import pyscx

    adata = pyscx.open(backed_scx_path).to_anndata(backed=True)

    chunk_size = 25
    chunks = list(pyscx.iter_chunks(adata, chunk_size=chunk_size))

    # All chunks except possibly the last should have exactly chunk_size rows
    for chunk in chunks[:-1]:
        assert chunk.n_obs == chunk_size

    # Last chunk can be smaller
    assert 0 < chunks[-1].n_obs <= chunk_size

    # Total should match
    total = sum(c.n_obs for c in chunks)
    assert total == adata.n_obs


def test_iter_chunks_invalid_chunk_size(backed_scx_path):
    """Invalid chunk_size raises ValueError."""
    import pyscx

    adata = pyscx.open(backed_scx_path).to_anndata(backed=True)

    with pytest.raises(ValueError, match="chunk_size must be"):
        list(pyscx.iter_chunks(adata, chunk_size=-1))

    with pytest.raises(ValueError, match="chunk_size must be"):
        list(pyscx.iter_chunks(adata, chunk_size="invalid"))


def test_iter_chunks_shard_boundaries(backed_scx_path):
    """shard_boundaries() returns valid contiguous ranges."""
    import pyscx

    adata = pyscx.open(backed_scx_path).to_anndata(backed=True)
    boundaries = adata.X.shard_boundaries()

    assert len(boundaries) > 0
    # First shard starts at 0
    assert boundaries[0][0] == 0
    # Last shard ends at n_obs
    assert boundaries[-1][1] == adata.n_obs
    # Contiguous: each shard starts where the previous ended
    for i in range(1, len(boundaries)):
        assert boundaries[i][0] == boundaries[i - 1][1]


def test_shard_boundaries_with_deletions(tmp_dir):
    """shard_boundaries() remaps to user-visible row space with deletion vectors."""
    import anndata
    import pyscx

    np.random.seed(123)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "shard_bounds_dv.scx")
    pyscx.from_anndata(adata, path)

    # Delete rows spread across the file (likely hitting multiple shards)
    pyscx.mark_deleted(path, [0, 1, 5, 10, 50, 75, 99])
    n_deleted = 7
    expected_n_obs = n_obs - n_deleted

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.n_obs == expected_n_obs

    boundaries = adata_backed.X.shard_boundaries()

    # Basic structural invariants
    assert len(boundaries) > 0
    # First shard starts at 0
    assert boundaries[0][0] == 0
    # Last shard ends at n_obs (user-visible)
    assert boundaries[-1][1] == expected_n_obs
    # Contiguous: each shard starts where the previous ended
    for i in range(1, len(boundaries)):
        assert boundaries[i][0] == boundaries[i - 1][1]
    # Total rows across boundaries matches
    total_from_bounds = sum(end - start for start, end in boundaries)
    assert total_from_bounds == expected_n_obs


def test_iter_chunks_obs_metadata_with_deletions(tmp_dir):
    """Each chunk's obs has the correct subset of rows when deletions are present."""
    import anndata
    import pandas as pd
    import pyscx

    np.random.seed(77)
    n_obs, n_vars = 80, 20
    dense = np.random.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    adata = anndata.AnnData(X=x, obs=obs)

    path = str(tmp_dir / "chunks_obs_dv.scx")
    pyscx.from_anndata(adata, path)

    # Delete some rows
    pyscx.mark_deleted(path, [0, 3, 10, 79])

    adata_backed = pyscx.open(path).to_anndata(backed=True)

    # Collect all obs indices from chunks
    all_indices = []
    for chunk in pyscx.iter_chunks(adata_backed):
        all_indices.extend(chunk.obs.index.tolist())

    # Should match the full backed obs (which already excludes deleted rows)
    expected_indices = adata_backed.obs.index.tolist()
    assert all_indices == expected_indices


def test_iter_chunks_fixed_size_with_deletions(tmp_dir):
    """Fixed-size chunking yields correct results with deletion vectors."""
    import anndata
    import pyscx

    np.random.seed(55)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "chunks_fixed_dv.scx")
    pyscx.from_anndata(adata, path)

    # Delete rows across the file
    pyscx.mark_deleted(path, [2, 7, 15, 30, 60, 90])
    n_deleted = 6
    expected_n_obs = n_obs - n_deleted

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.n_obs == expected_n_obs

    chunk_size = 20
    chunks = list(pyscx.iter_chunks(adata_backed, chunk_size=chunk_size))

    # All chunks except possibly the last should have exactly chunk_size rows
    for chunk in chunks[:-1]:
        assert chunk.n_obs == chunk_size

    # Last chunk can be smaller
    assert 0 < chunks[-1].n_obs <= chunk_size

    # Total should match
    total = sum(c.n_obs for c in chunks)
    assert total == expected_n_obs

    # Concatenated chunks should match full backed materialization
    concatenated = sp.vstack([c.X for c in chunks])
    full_x = adata_backed.X.to_memory()
    np.testing.assert_array_equal(concatenated.toarray(), full_x.toarray())


def test_iter_chunks_aligns_a_lazily_transformed_x_to_shards(tmp_dir):
    """A normalize_total → log1p handle has `shard_boundaries()` too; `iter_chunks`
    must use them rather than fall back to fixed 16384-row chunks (which on a
    small file is one chunk — silently defeating the shard alignment)."""
    import anndata
    import pyscx

    np.random.seed(7)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    dense[np.random.random((n_obs, n_vars)) > 0.4] = 0
    adata = anndata.AnnData(X=sp.csr_matrix(dense))
    path = str(tmp_dir / "chunks_lazy.scx")
    pyscx.from_anndata(adata, path, shard_size=25)

    backed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.normalize_total(backed)
    pyscx.accel.log1p(backed)
    assert isinstance(backed.X, pyscx.ScxLazyTransformedDataset)
    boundaries = backed.X.shard_boundaries()
    assert boundaries == [(0, 25), (25, 50), (50, 75), (75, 100)]

    chunks = list(pyscx.iter_chunks(backed, chunk_size="shard"))
    assert [c.n_obs for c in chunks] == [25, 25, 25, 25]
    full = backed.X.to_memory().toarray()
    got = np.vstack([c.X.toarray() if sp.issparse(c.X) else np.asarray(c.X) for c in chunks])
    np.testing.assert_allclose(got, full, rtol=1e-6)
