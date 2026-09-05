"""A pandas DataFrame in `uns` survives both h5ad directions (X6).

Until X6 this file was `test_uns_dataframe_warning.py` and pinned the opposite
contract: ingest **flattened** an `encoding-type: "dataframe"` uns group into a
dict of columns + `_index` and emitted a `flattened_uns_dataframe`
`UserWarning`, losing column order, per-column categorical dtypes and the
`ordered` bit; export wrote pyscx's envelope back out as a raw subgroup
carrying a `__scx_type__` string dataset, which anndata read as a dict. Neither
direction could carry scanpy's own `rank_genes_groups(pts=True)` output.

Both arms now reconstruct the frame, so the tests assert reconstruction and the
warning is gone.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _uns_frame():
    """Everything the flatten path used to lose, in one frame.

    Column order is not alphabetical (`column-order` is the only thing that
    carries it), the index is named, and the categorical is ordered with an
    unused level — `ordered` and the declared category list both live in
    attributes the old walker never read.
    """
    return pd.DataFrame(
        {
            "zscore": np.array([1.5, -0.5, 2.0], dtype=np.float64),
            "count": np.array([3, 0, 7], dtype=np.int64),
            # `int8` and `bool` are here because the first version of the export
            # arm routed the frame through the obs/var column writer, which
            # widened int8 to int32 and spelled bool as `nullable-boolean` —
            # which ingest then dropped. Found by Cursor Agent.
            "narrow": np.array([1, -2, 3], dtype=np.int8),
            "flag": np.array([True, False, True]),
            "label": np.array(["x", "y", "z"], dtype=object),
            "grade": pd.Categorical(
                ["hi", "lo", "hi"], categories=["lo", "mid", "hi"], ordered=True
            ),
        },
        index=pd.Index(["r0", "r1", "r2"], name="row"),
    )


def _adata_with_uns_df():
    import anndata
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    x = sp.csr_matrix(rng.integers(0, 5, size=(6, 3)).astype(np.float32))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(6)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(3)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    adata.uns["scores_df"] = _uns_frame()
    return adata


def test_uns_dataframe_survives_h5ad_ingest(tmp_dir):
    """`from_h5ad` reconstructs the frame instead of flattening it."""
    import pyscx

    src = _adata_with_uns_df()
    h5ad_in = str(tmp_dir / "uns_df.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "uns_df.scx")

    with warnings.catch_warnings():
        # No `flattened_uns_dataframe` — and nothing else either, so a new
        # warning on this path cannot slip in unnoticed.
        warnings.simplefilter("error")
        pyscx.from_h5ad(h5ad_in, scx_path)

    got = pyscx.open(scx_path).to_anndata().uns["scores_df"]
    assert isinstance(got, pd.DataFrame)
    pd.testing.assert_frame_equal(got, _uns_frame(), check_dtype=True)
    assert list(got.columns) == [
        "zscore",
        "count",
        "narrow",
        "flag",
        "label",
        "grade",
    ]
    assert got.index.name == "row"
    assert got["grade"].cat.ordered is True
    # The unused level survives: pruning it would be a row-filter behaviour,
    # and ingest applies no filter.
    assert list(got["grade"].cat.categories) == ["lo", "mid", "hi"]


def test_uns_dataframe_survives_h5ad_export(tmp_dir):
    """`to_h5ad` writes a group anndata reads back as a real DataFrame.

    The exporter used to hand every non-`ndarray`/`scalar` envelope to the
    generic subgroup path, so `uns["rank_genes_groups"]["pts"]` reached anndata
    as a dict and `sc.tl.filter_rank_genes_groups` broke on it.
    """
    import anndata

    import pyscx

    scx_path = str(tmp_dir / "export.scx")
    pyscx.from_anndata(_adata_with_uns_df(), scx_path)
    h5ad_out = str(tmp_dir / "export.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)

    got = anndata.read_h5ad(h5ad_out).uns["scores_df"]
    assert isinstance(got, pd.DataFrame)
    assert list(got.columns) == [
        "zscore",
        "count",
        "narrow",
        "flag",
        "label",
        "grade",
    ]
    assert list(got.index) == ["r0", "r1", "r2"]
    assert got.index.name == "row"
    assert isinstance(got["grade"].dtype, pd.CategoricalDtype)
    assert got["grade"].cat.ordered is True
    assert list(got["grade"]) == ["hi", "lo", "hi"]
    np.testing.assert_allclose(got["zscore"].to_numpy(), [1.5, -0.5, 2.0])
    assert list(got["label"]) == ["x", "y", "z"]
    # Exact dtypes on the h5ad side too: this arm writes plain datasets, not
    # the obs/var encodings, so nothing widens and nothing becomes nullable.
    assert str(got["narrow"].dtype) == "int8"
    assert str(got["flag"].dtype) == "bool"
    assert str(got["count"].dtype) == "int64"


def test_uns_dataframe_h5ad_round_trip_is_stable(tmp_dir):
    """h5ad -> scx -> h5ad -> scx lands on the same frame, not a decaying one.

    A one-way check can hide an asymmetry where each pass loses a little; going
    round twice and comparing to the original is what catches it.
    """
    import pyscx

    src = _adata_with_uns_df()
    first_h5ad = str(tmp_dir / "rt0.h5ad")
    src.write_h5ad(first_h5ad)

    scx_a = str(tmp_dir / "rt_a.scx")
    pyscx.from_h5ad(first_h5ad, scx_a)
    second_h5ad = str(tmp_dir / "rt1.h5ad")
    pyscx.to_h5ad(scx_a, second_h5ad)
    scx_b = str(tmp_dir / "rt_b.scx")
    pyscx.from_h5ad(second_h5ad, scx_b)

    got = pyscx.open(scx_b).to_anndata().uns["scores_df"]
    pd.testing.assert_frame_equal(got, _uns_frame(), check_dtype=True)


def test_scanpy_pts_frames_survive_the_h5ad_round_trip(tmp_dir):
    """The motivating case, written by scanpy itself rather than by hand.

    Scoped to `pts` / `pts_rest` on purpose. The rest of
    `uns["rank_genes_groups"]` — `names`, `scores`, `pvals`, `pvals_adj`,
    `logfoldchanges` — is a set of *compound* (structured) HDF5 arrays, which
    the h5ad reader skips with a `skipped_uns_key` warning telling the user to
    export DE separately, and which the exporter writes as raw `__scx_type__`
    subgroups. That gap predates X6, is independent of it in both directions,
    and is why `sc.tl.filter_rank_genes_groups` is *not* asserted here the way
    it is on the SCX-native round trip in `test_rank_genes_groups_pts.py`:
    it needs `names`, not `pts`.
    """
    sc = pytest.importorskip("scanpy")
    import anndata

    import pyscx

    adata = _adata_with_uns_df()
    del adata.uns["scores_df"]
    adata.obs["grp"] = pd.Categorical(["a", "a", "a", "b", "b", "b"])
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, "grp", method="wilcoxon", pts=True)
    expected = adata.uns["rank_genes_groups"]["pts"]
    assert isinstance(expected, pd.DataFrame)

    # scx -> h5ad: the frame arrives as a frame, not a dict.
    scx_path = str(tmp_dir / "sc_pts.scx")
    pyscx.from_anndata(adata, scx_path)
    h5ad_out = str(tmp_dir / "sc_pts.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)
    exported = anndata.read_h5ad(h5ad_out).uns["rank_genes_groups"]["pts"]
    assert isinstance(exported, pd.DataFrame)
    pd.testing.assert_frame_equal(exported, expected, check_dtype=True)

    # h5ad -> scx: so does scanpy's own file, which used to flatten to a dict.
    sc_h5ad = str(tmp_dir / "sc_native.h5ad")
    adata.write_h5ad(sc_h5ad)
    scx_b = str(tmp_dir / "sc_native.scx")
    with pytest.warns(UserWarning, match="skipped_uns_key"):
        # The compound-array skip above, not anything to do with the frames.
        pyscx.from_h5ad(sc_h5ad, scx_b)
    ingested = pyscx.open(scx_b).to_anndata().uns["rank_genes_groups"]["pts"]
    assert isinstance(ingested, pd.DataFrame)
    pd.testing.assert_frame_equal(ingested, expected, check_dtype=True)


def test_uns_dataframe_strict_uns_aborts_on_an_unrepresentable_column(tmp_dir):
    """`strict_uns=True` means abort on the first unrepresentable entry.

    The frame reader warned-and-omitted regardless of the flag, so a strict
    conversion returned success with a *truncated* frame — the one outcome
    strict mode exists to prevent. Found by codex.
    """
    import anndata
    import scipy.sparse as sp

    import pyscx

    adata = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.uns["t"] = pd.DataFrame(
        {"ok": [1.0, 2.0], "n": pd.array([1, None], dtype="Int64")},
        index=["r0", "r1"],
    )
    h5ad_in = str(tmp_dir / "strict.h5ad")
    adata.write_h5ad(h5ad_in)

    with pytest.raises(Exception, match=r"nullable-integer"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyscx.from_h5ad(h5ad_in, str(tmp_dir / "strict.scx"), strict_uns=True)

    # Lenient is unchanged: warn, omit the column, keep the rest of the frame.
    with pytest.warns(UserWarning, match="unsupported_uns_dataframe_column"):
        pyscx.from_h5ad(h5ad_in, str(tmp_dir / "lenient.scx"))
    got = pyscx.open(str(tmp_dir / "lenient.scx")).to_anndata().uns["t"]
    assert list(got.columns) == ["ok"]


def test_uns_dataframe_reads_fixed_width_string_columns(tmp_dir):
    """Fixed-length HDF5 strings need a different read type from var-length.

    `read_1d::<VarLenUnicode>()` on a fixed-length dataset fails with an opaque
    "no conversion paths found", so a PyTables-written frame (CellRanger,
    CellBender) lost every string column — and a fixed-width *index* lost the
    whole frame. `read_string_dataset` already handled both families. Found by
    Antigravity and codex.
    """
    import anndata
    import h5py
    import scipy.sparse as sp

    import pyscx

    adata = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.uns["t"] = pd.DataFrame({"s": ["placeholder", "placeholder"]}, index=["r0", "r1"])
    h5ad_in = str(tmp_dir / "fixed.h5ad")
    adata.write_h5ad(h5ad_in)
    with h5py.File(h5ad_in, "a") as f:
        g = f["uns/t"]
        del g["s"]
        g.create_dataset("s", data=np.array([b"aa", b"bb"], dtype="S2"))  # FixedAscii
        del g["_index"]
        g.create_dataset("_index", data=np.array([b"r0", b"r1"], dtype="S2"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyscx.from_h5ad(h5ad_in, str(tmp_dir / "fixed.scx"))
    got = pyscx.open(str(tmp_dir / "fixed.scx")).to_anndata().uns["t"]
    assert list(got.columns) == ["s"]
    assert list(got["s"]) == ["aa", "bb"]
    assert list(got.index) == ["r0", "r1"]


def test_uns_dataframe_group_without_an_index_falls_back_to_a_dict(tmp_dir):
    """A malformed frame group must not cost the whole `uns` key.

    Erroring on the missing `_index` dataset made the key a `SkippedUnsKey`, so
    `uns` came back empty — strictly worse than the pre-X6 flatten, which still
    recovered every sibling column as a dict. Found by Cursor Agent.
    """
    import anndata
    import h5py
    import scipy.sparse as sp

    import pyscx

    adata = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.uns["t"] = pd.DataFrame({"a": [1, 2]}, index=["r0", "r1"])
    h5ad_in = str(tmp_dir / "noidx.h5ad")
    adata.write_h5ad(h5ad_in)
    with h5py.File(h5ad_in, "a") as f:
        del f["uns/t/_index"]

    pyscx.from_h5ad(h5ad_in, str(tmp_dir / "noidx.scx"))
    got = pyscx.open(str(tmp_dir / "noidx.scx")).read_uns()["t"]
    assert isinstance(got, dict), "the column data must survive as a dict"
    assert list(got["a"]) == [1, 2]


def test_uns_dataframe_export_declines_loudly_and_keeps_the_data(tmp_dir):
    """A frame h5ad cannot spell is demoted to the raw envelope, with a warning.

    Two things were wrong before. A boolean categorical was silently *dropped*
    while the export reported success (an empty DataFrame reached anndata), and
    an index whose name matched a column aborted the entire `to_h5ad` with an
    opaque HDF5 error. Both now decline the frame — the raw envelope subgroup
    keeps every value — and say so through Python's `warnings`, which the
    export path did not reach at all before. Found by codex and Cursor Agent.
    """
    import anndata
    import scipy.sparse as sp

    import pyscx

    def frame_file(df, name):
        adata = anndata.AnnData(
            X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
        )
        adata.uns["t"] = df
        path = str(tmp_dir / f"{name}.scx")
        pyscx.from_anndata(adata, path)
        return path

    cases = {
        "boolcat": pd.DataFrame(
            {"bc": pd.Categorical([True, False], categories=[False, True], ordered=True)},
            index=["r0", "r1"],
        ),
        # Legal pandas: index.name equals a column name.
        "collide": pd.DataFrame({"gene": [1.0, 2.0]}, index=pd.Index(["r0", "r1"], name="gene")),
        # A null in an object column: a plain h5ad string dataset has no null.
        "nullstr": pd.DataFrame({"s": np.array(["x", None], dtype=object)}, index=["r0", "r1"]),
    }

    for name, df in cases.items():
        scx_path = frame_file(df, name)
        out = str(tmp_dir / f"{name}.h5ad")
        with pytest.warns(UserWarning, match="uns_exported_as_raw_envelope"):
            pyscx.to_h5ad(scx_path, out)
        got = anndata.read_h5ad(out).uns["t"]
        assert isinstance(got, dict), f"{name}: expected the raw-envelope fallback"
        # The fallback is what makes declining better than dropping a column:
        # every part of the envelope is still on disk.
        assert {"__scx_type__", "index", "columns", "data"} <= set(got), name
