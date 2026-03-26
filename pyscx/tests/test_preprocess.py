"""Integration tests for pyscx.preprocess() and pyscx.save_layer().

Tests the streaming write-back preprocessing pipeline:
- normalize_total: per-row normalize → write to new SCX → verify row sums
- log1p: element-wise ln(x+1) → verify values match np.log1p
- fused normalize+log1p: combined ops → verify matches scanpy pipeline
- save_layer: write transformed data as named layer → verify layer round-trip
- end-to-end pipeline: preprocess → open result → run scanpy PCA
"""

import numpy as np
import pytest


def test_preprocess_normalize_total(synthetic_adata, tmp_dir):
    """preprocess(["normalize_total"]) produces rows that sum to target_sum."""
    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "normalized.scx")

    pyscx.from_anndata(synthetic_adata, source)
    n_shards = pyscx.preprocess(source, target, ["normalize_total"], target_sum=1e4)
    assert n_shards > 0

    # Load result and verify row sums
    adata_out = pyscx.open(target).to_anndata()
    assert adata_out.X.shape == synthetic_adata.X.shape

    row_sums = np.array(adata_out.X.sum(axis=1)).flatten()
    # Rows with nnz > 0 should sum to target_sum
    nonzero_mask = np.array(synthetic_adata.X.getnnz(axis=1)) > 0
    np.testing.assert_allclose(
        row_sums[nonzero_mask], 1e4, rtol=1e-5
    )
    # Zero rows should remain zero
    assert np.all(row_sums[~nonzero_mask] == 0)


def test_preprocess_log1p(synthetic_adata, tmp_dir):
    """preprocess(["log1p"]) matches np.log1p element-wise."""
    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "log1p.scx")

    pyscx.from_anndata(synthetic_adata, source)
    pyscx.preprocess(source, target, ["log1p"])

    adata_out = pyscx.open(target).to_anndata()
    expected = synthetic_adata.X.copy()
    expected.data = np.log1p(expected.data)

    np.testing.assert_allclose(
        adata_out.X.toarray(), expected.toarray(), rtol=1e-5
    )


def test_preprocess_fused_normalize_log1p(synthetic_adata, tmp_dir):
    """preprocess(["normalize_total", "log1p"]) matches scanpy pipeline."""
    import scanpy as sc

    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "fused.scx")

    pyscx.from_anndata(synthetic_adata, source)
    pyscx.preprocess(source, target, ["normalize_total", "log1p"], target_sum=1e4)

    adata_out = pyscx.open(target).to_anndata()

    # Replicate with scanpy
    adata_ref = synthetic_adata.copy()
    sc.pp.normalize_total(adata_ref, target_sum=1e4)
    sc.pp.log1p(adata_ref)

    np.testing.assert_allclose(
        adata_out.X.toarray(), adata_ref.X.toarray(), rtol=1e-5
    )


def test_preprocess_copies_metadata(synthetic_adata, tmp_dir):
    """preprocess copies obs, var, obsm, uns to the output file."""
    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "meta.scx")

    pyscx.from_anndata(synthetic_adata, source)
    pyscx.preprocess(source, target, ["log1p"])

    adata_out = pyscx.open(target).to_anndata()

    # obs should match
    assert list(adata_out.obs.columns) == list(synthetic_adata.obs.columns)
    assert adata_out.n_obs == synthetic_adata.n_obs
    assert adata_out.n_vars == synthetic_adata.n_vars

    # var should match
    assert list(adata_out.var.columns) == list(synthetic_adata.var.columns)

    # obsm should have the same keys
    assert set(adata_out.obsm.keys()) == set(synthetic_adata.obsm.keys())
    for key in synthetic_adata.obsm:
        np.testing.assert_allclose(
            adata_out.obsm[key], synthetic_adata.obsm[key], rtol=1e-5
        )


def test_save_layer(synthetic_adata, tmp_dir):
    """save_layer writes transformed data as a named layer."""
    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "with_layer.scx")

    pyscx.from_anndata(synthetic_adata, source)
    pyscx.save_layer(
        source, target, "normalized", ["normalize_total", "log1p"], target_sum=1e4
    )

    adata_out = pyscx.open(target).to_anndata()

    # Original X should be preserved
    np.testing.assert_allclose(
        adata_out.X.toarray(), synthetic_adata.X.toarray(), rtol=1e-5
    )

    # The new layer should exist
    assert "normalized" in adata_out.layers

    # Verify layer values match expected pipeline
    import scanpy as sc

    adata_ref = synthetic_adata.copy()
    sc.pp.normalize_total(adata_ref, target_sum=1e4)
    sc.pp.log1p(adata_ref)

    np.testing.assert_allclose(
        adata_out.layers["normalized"].toarray(),
        adata_ref.X.toarray(),
        rtol=1e-5,
    )


def test_preprocess_end_to_end_pipeline(synthetic_adata, tmp_dir):
    """Preprocessed file can be used for downstream scanpy analysis."""
    import scanpy as sc

    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "preprocessed.scx")

    pyscx.from_anndata(synthetic_adata, source)
    pyscx.preprocess(source, target, ["normalize_total", "log1p"], target_sum=1e4)

    # Load preprocessed data and run PCA
    adata = pyscx.open(target).to_anndata()
    sc.pp.pca(adata, n_comps=min(10, adata.n_vars - 1))

    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape[0] == adata.n_obs


def test_preprocess_identity(synthetic_adata, tmp_dir):
    """preprocess([]) with no ops copies data unchanged."""
    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "identity.scx")

    pyscx.from_anndata(synthetic_adata, source)
    pyscx.preprocess(source, target, [])

    adata_out = pyscx.open(target).to_anndata()
    np.testing.assert_allclose(
        adata_out.X.toarray(), synthetic_adata.X.toarray(), rtol=1e-5
    )


def test_preprocess_invalid_op(synthetic_adata, tmp_dir):
    """preprocess with unknown operation raises error."""
    import pyscx

    source = str(tmp_dir / "source.scx")
    target = str(tmp_dir / "bad.scx")

    pyscx.from_anndata(synthetic_adata, source)
    with pytest.raises(RuntimeError, match="Unknown operation"):
        pyscx.preprocess(source, target, ["invalid_op"])
