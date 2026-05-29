"""Phase 8: `pyscx.to_h5ad` round-trip tests.

Mirrors `pyscx.from_h5ad`. Streams by default — these tests confirm
the round-trip is bit-exact for integer X values and that the
extracted h5ad reads cleanly via `anndata.read_h5ad`.
"""

from __future__ import annotations

import numpy as np
import pytest


def _dense_from_anndata_x(adata):
    """Return X as a dense float32 array, regardless of sparse/dense."""
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _simple_adata(n_obs=20, n_vars=12):
    """An AnnData fixture whose obs/var only use column dtypes the
    materialising h5ad_write supports today (`Int32`, `Int64`,
    `Float32`, `Float64`, `Utf8`, `Boolean`, `Dictionary<Int32,
    Utf8>`). Avoids `Dictionary<Int8, Utf8>` categoricals which the
    writer currently skips — that's a pre-existing limitation, not
    Phase 8 scope.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0.0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"n_counts": np.arange(n_obs, dtype=np.int32)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"mean_expr": np.linspace(0.1, 1.0, n_vars, dtype=np.float32)},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=x, obs=obs, var=var)


def test_to_h5ad_round_trip_streaming(tmp_dir):
    """`pyscx.from_anndata → pyscx.to_h5ad → anndata.read_h5ad` round-trip."""
    import anndata
    import pyscx

    src = _simple_adata()
    scx_path = str(tmp_dir / "round_trip.scx")
    h5ad_out = str(tmp_dir / "round_trip.h5ad")
    pyscx.from_anndata(src, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_out, stream=True)

    out = anndata.read_h5ad(h5ad_out)
    np.testing.assert_array_equal(
        _dense_from_anndata_x(src),
        _dense_from_anndata_x(out),
    )
    # Whether the obs/var index round-trips to the same labels is an
    # `from_anndata` concern (pyscx writes the index as a regular
    # column); the Phase 8 contract is that `to_h5ad` preserves
    # whatever obs/var SCX has.
    assert out.n_obs == src.n_obs
    assert out.n_vars == src.n_vars


def test_to_h5ad_round_trip_materialising(tmp_dir):
    """`stream=False` falls back to the materialising path; output
    must match the streaming path."""
    import anndata
    import pyscx

    src = _simple_adata()
    scx_path = str(tmp_dir / "rt.scx")
    h5ad_stream = str(tmp_dir / "rt_stream.h5ad")
    h5ad_mat = str(tmp_dir / "rt_mat.h5ad")
    pyscx.from_anndata(src, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_stream, stream=True)
    pyscx.to_h5ad(scx_path, h5ad_mat, stream=False)

    stream = anndata.read_h5ad(h5ad_stream)
    mat = anndata.read_h5ad(h5ad_mat)
    np.testing.assert_array_equal(
        _dense_from_anndata_x(stream),
        _dense_from_anndata_x(mat),
    )


def test_to_h5ad_preserves_numeric_and_string_obs_columns(tmp_dir):
    """Int / Float / Utf8 obs columns are written as untagged HDF5
    datasets; anndata's empty-IOSpec auto-detect path must round
    them back to the right pandas dtype + values."""
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    rng = np.random.default_rng(7)
    n_obs, n_vars = 30, 8
    x = sp.csr_matrix(rng.integers(0, 20, (n_obs, n_vars)).astype(np.float32))
    obs = pd.DataFrame(
        {
            "n_counts": np.arange(n_obs, dtype=np.int32),
            "total_umi": np.arange(n_obs, dtype=np.int64),
            "f32_metric": np.linspace(0.0, 1.0, n_obs, dtype=np.float32),
            "f64_metric": np.linspace(-1.0, 1.0, n_obs, dtype=np.float64),
            "sample_id": [f"sample_{i % 4}" for i in range(n_obs)],
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    src = anndata.AnnData(X=x, obs=obs, var=var)

    scx_path = str(tmp_dir / "numeric.scx")
    out_path = str(tmp_dir / "numeric.h5ad")
    pyscx.from_anndata(src, scx_path)
    pyscx.to_h5ad(scx_path, out_path, stream=True)

    out = anndata.read_h5ad(out_path)
    for col in ("n_counts", "total_umi", "f32_metric", "f64_metric", "sample_id"):
        assert col in out.obs.columns, f"column {col} missing"
        # Compare values regardless of exact dtype (string columns may
        # come back as object/categorical depending on anndata version).
        np.testing.assert_array_equal(
            np.asarray(src.obs[col]),
            np.asarray(out.obs[col]),
            err_msg=f"column {col} drifted",
        )


def test_to_h5ad_preserves_boolean_obs_column(tmp_dir):
    """Boolean obs columns must survive the streaming export and
    read back through `anndata.read_h5ad`. The writer's
    `nullable-boolean` group form is the only h5py shape anndata
    has a registered IOSpec for."""
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    n_obs = 30
    n_vars = 8
    rng = np.random.default_rng(0)
    x = sp.csr_matrix(
        rng.integers(0, 20, size=(n_obs, n_vars)).astype(np.float32)
    )
    obs = pd.DataFrame(
        {
            "is_doublet": pd.array(
                [bool(i % 3 != 0) for i in range(n_obs)], dtype="boolean"
            )
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    src = anndata.AnnData(X=x, obs=obs, var=var)

    scx_path = str(tmp_dir / "bool.scx")
    h5ad_out = str(tmp_dir / "bool.h5ad")
    pyscx.from_anndata(src, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_out, stream=True)

    out = anndata.read_h5ad(h5ad_out)
    assert "is_doublet" in out.obs.columns, "boolean column lost"
    src_vals = src.obs["is_doublet"].to_numpy(dtype=bool)
    out_vals = out.obs["is_doublet"].to_numpy(dtype=bool)
    np.testing.assert_array_equal(src_vals, out_vals)


def test_to_h5ad_multimodal_without_modality_raises(tmp_dir):
    """Plain `to_h5ad` on a multimodal SCX file must raise; the
    user should pass `modality=...` or call `to_h5mu`."""
    mudata = pytest.importorskip("mudata")
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    n_obs = 6
    rna = anndata.AnnData(
        X=sp.csr_matrix(np.eye(n_obs, 4, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    adt = anndata.AnnData(
        X=sp.csr_matrix(np.eye(n_obs, 3, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"a{i}" for i in range(3)]),
    )
    mu = mudata.MuData({"rna": rna, "adt": adt})
    scx_path = str(tmp_dir / "cite.scx")
    h5ad_out = str(tmp_dir / "cite.h5ad")
    pyscx.from_mudata(mu, scx_path)

    with pytest.raises(Exception):
        pyscx.to_h5ad(scx_path, h5ad_out)


def test_to_h5ad_modality_extract(tmp_dir):
    """Multimodal SCX → single-modality h5ad via `modality='rna'`."""
    mudata = pytest.importorskip("mudata")
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    n_obs = 6
    rna = anndata.AnnData(
        X=sp.csr_matrix(np.eye(n_obs, 4, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    adt = anndata.AnnData(
        X=sp.csr_matrix(np.eye(n_obs, 3, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"a{i}" for i in range(3)]),
    )
    mu = mudata.MuData({"rna": rna, "adt": adt})
    scx_path = str(tmp_dir / "cite.scx")
    h5ad_out = str(tmp_dir / "rna.h5ad")
    pyscx.from_mudata(mu, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_out, modality="rna", stream=True)

    out = anndata.read_h5ad(h5ad_out)
    assert out.n_vars == 4  # rna's gene count, not adt's
    np.testing.assert_array_equal(
        _dense_from_anndata_x(rna), _dense_from_anndata_x(out)
    )


def test_to_h5ad_preserves_nullable_int_and_string(tmp_dir):
    """Null entries in nullable-integer and string obs columns survive
    SCX → h5ad export via anndata's `nullable-integer` /
    `nullable-string-array` group encodings; null floats survive as NaN.

    This is the Patch 2 contract: nulls are no longer silently coerced
    to `0` / `""`.
    """
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    n_obs, n_vars = 24, 8
    rng = np.random.default_rng(1)
    x = sp.csr_matrix(rng.integers(0, 10, (n_obs, n_vars)).astype(np.float32))

    qc_int = pd.array(
        [None if i % 5 == 0 else i for i in range(n_obs)], dtype="Int32"
    )
    qc_str = pd.array(
        [None if i % 4 == 0 else f"grp{i % 3}" for i in range(n_obs)],
        dtype="string",
    )
    pct = np.arange(n_obs, dtype=np.float32)
    pct[::3] = np.nan  # explicit NaN floats
    obs = pd.DataFrame(
        {"qc_int": qc_int, "qc_str": qc_str, "pct": pct},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)

    scx_path = str(tmp_dir / "nullable.scx")
    h5ad_out = str(tmp_dir / "nullable.h5ad")
    pyscx.from_anndata(adata, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_out)

    out = anndata.read_h5ad(h5ad_out)

    # Nullable integer: dtype stays a pandas nullable integer and the
    # null mask is preserved (not coerced to 0).
    assert pd.api.types.is_integer_dtype(out.obs["qc_int"].dtype)
    np.testing.assert_array_equal(
        out.obs["qc_int"].isna().to_numpy(),
        np.array([i % 5 == 0 for i in range(n_obs)]),
    )
    for i in range(n_obs):
        if i % 5 != 0:
            assert int(out.obs["qc_int"].iloc[i]) == i

    # Nullable string: null mask preserved (not coerced to "").
    np.testing.assert_array_equal(
        out.obs["qc_str"].isna().to_numpy(),
        np.array([i % 4 == 0 for i in range(n_obs)]),
    )

    # Float nulls survive as NaN (anndata has no nullable-float spec).
    got_pct = np.asarray(out.obs["pct"], dtype=np.float32)
    assert np.isnan(got_pct[::3]).all()
