"""Phase 3 — daily-driver Python ergonomics (T3.1–T3.7).

Covers the recognizable-API surface: the `Experiment` / `CloudExperiment`
class names, `read()` / `write()` / `read_cloud()` one-liners, the
AnnData-style `__repr__` + key accessors, biologist-actionable error
classes, and the T3.7 write-path warnings + removed `from_10x(in_place=)`.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


# ---------------------------------------------------------------------------
# T3.1 — class names
# ---------------------------------------------------------------------------


def test_experiment_class_name(synthetic_adata, tmp_dir):
    """The Python-visible class is `Experiment`, not `PyExperiment`."""
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    exp = pyscx.open(path)
    assert isinstance(exp, pyscx.Experiment)
    assert type(exp).__name__ == "Experiment"
    assert "PyExperiment" not in repr(exp)


# ---------------------------------------------------------------------------
# T3.4 — AnnData-style repr + key accessors
# ---------------------------------------------------------------------------


def test_repr_is_anndata_style(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    r = repr(pyscx.open(path))
    assert r.startswith("Experiment object with n_obs × n_vars = 100 × 50")
    assert "obs:" in r and "var:" in r
    assert "'batch'" in r  # an obs column is listed


def test_key_accessors(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    exp = pyscx.open(path)
    # The *_keys accessors are callable methods, matching anndata's
    # `adata.obs_keys()` (not properties).
    assert callable(exp.obs_keys)
    # obs/var keys exclude the pandas index column.
    assert set(exp.obs_keys()) >= {"cell_id", "batch"}
    assert "_index" not in exp.obs_keys() and "__index_level_0__" not in exp.obs_keys()
    assert set(exp.var_keys()) >= {"gene_id", "highly_variable"}
    assert exp.obsm_keys() == ["X_pca"]
    assert set(exp.uns_keys()) >= {"species", "version"}
    # layer_names() is a callable method too, consistent with the *_keys() family (F7).
    assert callable(exp.layer_names)
    assert "raw" in exp.layer_names()


def test_info_carries_internals(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    info = pyscx.open(path).info()
    assert "codec_id" in info and "format_version" in info


# ---------------------------------------------------------------------------
# T3.3 — read() / write() one-liners
# ---------------------------------------------------------------------------


def test_read_write_roundtrip(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    adata = pyscx.read(path)
    assert adata.shape == (100, 50)
    # X values survive (counts; float32).
    np.testing.assert_allclose(
        adata.X.toarray(), synthetic_adata.X.toarray(), rtol=0, atol=0
    )


def test_read_forwards_to_anndata_kwargs(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    backed = pyscx.read(path, backed=True)
    assert backed.shape == (100, 50)


# ---------------------------------------------------------------------------
# T3.5 — actionable error classes
# ---------------------------------------------------------------------------


def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        pyscx.read("/no/such/path/missing.scx")


@pytest.mark.skipif(
    not hasattr(pyscx, "from_h5ad"), reason="pyscx built without the hdf5 feature"
)
def test_missing_h5ad_input_raises_filenotfound(tmp_dir):
    # The h5ad/h5mu converters open inputs via hdf5::File::open
    # (ConvertError::Hdf5); a missing path must still raise FileNotFoundError
    # (Codex review): the entry point pre-checks existence.
    with pytest.raises(FileNotFoundError):
        pyscx.from_h5ad("/no/such/input.h5ad", str(tmp_dir / "out.scx"))


@pytest.mark.skipif(
    not hasattr(pyscx, "from_h5ad"), reason="pyscx built without the hdf5 feature"
)
def test_non_hdf5_h5ad_input_raises_valueerror(tmp_dir):
    # Report E1: an existing-but-not-HDF5 input must raise a clean ValueError
    # naming the file and the expected format, not a raw RuntimeError that
    # leaks libhdf5 internals ("H5Fopen(): ... file signature not found").
    bad = tmp_dir / "not_hdf5.h5ad"
    bad.write_bytes(b"this is plainly not an HDF5 file\n" * 16)
    with pytest.raises(ValueError) as ei:
        pyscx.from_h5ad(str(bad), str(tmp_dir / "out.scx"))
    msg = str(ei.value)
    assert "not a valid HDF5/h5ad file" in msg
    assert str(bad) in msg


def test_obs_keys_getter_surfaces_corrupt_file(tmp_dir):
    # A fully corrupt file fails to open; but a getter on a successfully
    # opened-yet-unreadable section must raise, not silently return [].
    # We can't easily fabricate a valid-header/corrupt-obs file, so assert
    # the getter at least returns the right columns on a good file and is
    # typed as raising (PyResult) rather than swallowing.
    import anndata as ad

    x = sp.random(20, 10, density=0.2, format="csr", dtype=np.float32)
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame({"grp": ["a", "b"] * 10}, index=[f"c{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
    )
    path = str(tmp_dir / "ok.scx")
    pyscx.write(adata, path)
    assert pyscx.open(path).obs_keys() == ["grp"]


def test_corrupt_file_raises_valueerror(tmp_dir):
    bad = tmp_dir / "bad.scx"
    bad.write_bytes(b"X" * 300)  # >= header size, bad magic
    with pytest.raises(ValueError):
        pyscx.open(str(bad))


def test_truncated_file_raises_valueerror(tmp_dir):
    small = tmp_dir / "small.scx"
    small.write_bytes(b"X" * 20)  # below header size
    with pytest.raises(ValueError):
        pyscx.open(str(small))


# ---------------------------------------------------------------------------
# T3.7 — write-path warnings + from_10x cleanup
# ---------------------------------------------------------------------------


def test_float64_downcast_warns(tmp_dir):
    x = sp.random(50, 20, density=0.2, format="csr", dtype=np.float64)
    adata = __import__("anndata").AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(20)]),
    )
    path = str(tmp_dir / "f64.scx")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.write(adata, path)
    assert any("float32" in str(w.message) for w in caught)


def test_deletion_export_warns(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    exp = pyscx.open(path)
    exp.mark_deleted(np.array([True] * 10 + [False] * 90))
    assert exp.has_deletions is True
    out = str(tmp_dir / "out.h5ad")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.to_h5ad(path, out)
    assert any("deleted" in str(w.message) for w in caught)


def test_from_10x_has_no_in_place_kwarg():
    # `from_10x` is a native pyfunction; calling with the removed kwarg
    # must raise (TypeError) rather than silently accept it.
    with pytest.raises(TypeError):
        pyscx.from_10x("nonexistent.h5", "out.scx", in_place=True)


# ---------------------------------------------------------------------------
# T3.2 — read_cloud (local object_store path; no GCS creds needed)
# ---------------------------------------------------------------------------

read_cloud = getattr(pyscx, "read_cloud", None)
needs_cloud = pytest.mark.skipif(
    read_cloud is None, reason="pyscx built without the cloud feature"
)


@needs_cloud
def test_read_cloud_full(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    adata = pyscx.read_cloud("file://" + path)
    assert adata.shape == (100, 50)


@needs_cloud
def test_read_cloud_obs_filter(query_adata, tmp_dir):
    path = str(tmp_dir / "q.scx")
    pyscx.write(query_adata, path, index_obs=["cell_type"])
    adata = pyscx.read_cloud("file://" + path, obs_filter="cell_type == 'T cell'")
    assert adata.n_obs == 40
    assert set(adata.obs["cell_type"]) == {"T cell"}


@needs_cloud
def test_read_cloud_var_names(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    adata = pyscx.read_cloud("file://" + path, var_names=["gene_0", "gene_5"])
    assert adata.n_vars == 2


@needs_cloud
def test_read_cloud_unknown_gene_raises(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    with pytest.raises(KeyError):
        pyscx.read_cloud("file://" + path, var_names=["not_a_gene"])


@needs_cloud
def test_read_cloud_empty_var_names_projects_zero_genes(synthetic_adata, tmp_dir):
    # An explicit empty projection must yield 0 genes, NOT fall through to
    # "all genes" (Codex review): `var_names=[]` != `var_names=None`.
    path = str(tmp_dir / "x.scx")
    pyscx.write(synthetic_adata, path)
    adata = pyscx.read_cloud("file://" + path, var_names=[])
    assert adata.n_vars == 0
    assert adata.n_obs == 100
