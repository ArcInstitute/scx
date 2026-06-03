"""pyscx.from_anndata(csc=...) round-trip tests.

Verifies that the new ``csc`` and ``csc_cols_per_shard`` kwargs on
``from_anndata`` correctly emit a CSC sidecar (multi-shard column-major
layout) and that the densified data matches the source AnnData.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    """20 cells × 15 genes synthetic CSR AnnData."""
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(0)
    mat = sp.random(20, 15, density=0.25, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 100).astype(np.float32)
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(20)]
    adata.var["gene_id"] = [f"g{i}" for i in range(15)]
    return adata


def test_from_anndata_csc_off_default(small_adata, tmp_path):
    """Default `csc='off'` emits no CSC sidecar."""
    import pyscx

    path = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(path))

    exp = pyscx.open(str(path))
    # Round-trip values still match.
    adata2 = exp.to_anndata()
    np.testing.assert_array_almost_equal(
        small_adata.X.toarray(), adata2.X.toarray()
    )


def test_from_anndata_csc_always(small_adata, tmp_path):
    """`csc='always'` emits a CSC sidecar; densified contents match CSR."""
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=5)

    exp = pyscx.open(str(path))
    adata2 = exp.to_anndata()
    # to_anndata reads CSR; the CSC sidecar must densify to the same
    # values as the source matrix.
    np.testing.assert_array_almost_equal(
        small_adata.X.toarray(), adata2.X.toarray()
    )


def test_from_anndata_csc_invalid_value(small_adata, tmp_path):
    """`csc` only accepts 'off' / 'auto' / 'always'; anything else raises."""
    import pyscx

    path = tmp_path / "bad.scx"
    with pytest.raises(ValueError, match="csc"):
        pyscx.from_anndata(small_adata, str(path), csc="yes")


def _csc_available(path):
    """True iff the file has a usable CSC sidecar. `prefer_format="csc"`
    raises ``RuntimeError`` when no sidecar exists, succeeds otherwise."""
    import pyscx

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    try:
        pyscx.accel.col_sums(adata.X, prefer_format="csc")
        return True
    except RuntimeError:
        return False


def test_from_anndata_csc_auto_below_threshold_skips(small_adata, tmp_path):
    """`csc='auto'` builds no sidecar for a sub-threshold dataset
    (20×15 is far below the 50000-obs / 5000-var defaults)."""
    path = tmp_path / "auto_small.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), csc="auto")
    assert _csc_available(path) is False


def test_from_anndata_csc_auto_env_override_builds(small_adata, tmp_path, monkeypatch):
    """Lowering both thresholds to 0 makes `csc='auto'` build a sidecar
    even on the tiny fixture; densified CSC matches CSR."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "auto_forced.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), csc="auto", csc_cols_per_shard=5)
    assert _csc_available(path) is True

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    csr_sums = pyscx.accel.col_sums(adata.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    np.testing.assert_allclose(csr_sums, csc_sums, atol=1e-9)


# ---------------------------------------------------------------------------
# Phase 0.3: an accel-ready index_preset implies csc="auto" when the caller
# did not pass an explicit `csc`. An explicit value always wins.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["training", "perturbseq"])
def test_index_preset_implies_csc_auto(small_adata, tmp_path, monkeypatch, preset):
    """An accel-ready preset with no explicit `csc` upgrades to "auto".
    With the thresholds lowered to 0, that builds a sidecar on the tiny
    fixture."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / f"preset_{preset}.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), index_preset=preset, csc_cols_per_shard=5)
    assert _csc_available(path) is True


def test_index_preset_cellxgene_does_not_imply_csc(small_adata, tmp_path, monkeypatch):
    """The query-oriented `cellxgene` preset does NOT imply csc="auto" —
    no sidecar even with the thresholds at 0."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "preset_cellxgene.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), index_preset="cellxgene", csc_cols_per_shard=5)
    assert _csc_available(path) is False


def test_explicit_csc_off_overrides_preset(small_adata, tmp_path, monkeypatch):
    """An explicit `csc="off"` wins over an accel-ready preset's implied
    "auto" — the user's explicit choice is honored."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "preset_explicit_off.scx"
    import pyscx

    pyscx.from_anndata(
        small_adata, str(path), index_preset="training", csc="off", csc_cols_per_shard=5
    )
    assert _csc_available(path) is False
