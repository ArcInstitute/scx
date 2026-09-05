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
        # Legal pandas: index.name equals a column name. Both land in one HDF5
        # group, so the names would collide.
        "collide": pd.DataFrame({"gene": [1.0, 2.0]}, index=pd.Index(["r0", "r1"], name="gene")),
        # A null in an object column: a plain h5ad string dataset has no null.
        "nullstr": pd.DataFrame({"s": np.array(["x", None], dtype=object)}, index=["r0", "r1"]),
        # `df.index.name = 7` is legal pandas, but h5ad stores the index name as
        # an HDF5 member name — it used to be silently erased to `_index`.
        "intname": pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index(["r0", "r1"], name=7)),
    }

    # NB: a *boolean categorical* is deliberately not here. Round 1 declined it
    # on a false premise; it round-trips, and
    # `test_boolean_categorical_round_trips_through_h5ad` pins that.

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


def test_hdf5_unsafe_uns_keys_are_skipped_not_escaped_or_aborted(tmp_dir):
    """A `uns` key HDF5 cannot carry must not corrupt the file or kill the export.

    HDF5 resolves a name containing `/` as a **path**: `uns["/evil"]` wrote a
    dataset at the file *root*, outside `/uns` entirely, and `to_h5ad` reported
    success — `anndata.read_h5ad` then could not open the file at all. `""`,
    `"."` and `".."` are rejected identifiers and aborted the whole export from
    inside `create_dataset`. All four are legal Python dict keys.

    Pre-existing and general: reproducible with no DataFrame anywhere, which is
    why the guard lives on the generic `uns` walk. Found by codex via a
    DataFrame column.
    """
    import anndata
    import h5py
    import scipy.sparse as sp

    import pyscx

    def make(uns):
        a = anndata.AnnData(
            X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
        )
        a.uns.update(uns)
        return a

    col = lambda name: pd.DataFrame({name: [1.0, 2.0]}, index=["r0", "r1"])  # noqa: E731
    cases = {
        "plain_slash": {"/evil": 1.0},
        "nested_slash": {"grp": {"/evil": 1.0}},
        "plain_empty": {"": 1.0},
        "plain_dot": {".": 1.0},
        "df_slash": {"t": col("/evil")},
        "df_empty": {"t": col("")},
        "df_dot": {"t": col(".")},
    }

    for name, uns in cases.items():
        scx_path = str(tmp_dir / f"{name}.scx")
        pyscx.from_anndata(make(uns), scx_path)
        out = str(tmp_dir / f"{name}.h5ad")
        with pytest.warns(UserWarning, match="skipped_uns_key"):
            pyscx.to_h5ad(scx_path, out)
        with h5py.File(out) as f:
            assert sorted(f.keys()) == ["X", "obs", "uns", "var"], (
                f"{name}: an unsafe key escaped to the file root"
            )
        # The point of the guard: the file is still a readable h5ad.
        anndata.read_h5ad(out)


def test_uns_dataframe_bytes_index_and_categories_are_refused(tmp_dir):
    """The bytes refusal has to cover every object-array path in a frame.

    Round 1 wired it into the column loop only, so an index or a categorical's
    categories still reached the string walker and were UTF-8-*decoded* —
    `[b"r0", b"r1"]` read back as `["r0", "r1"]`. Found by codex and Cursor
    Agent (index) and Antigravity (categories).
    """
    import anndata
    import scipy.sparse as sp

    import pyscx

    def make(df):
        return anndata.AnnData(
            X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
            uns={"t": df},
        )

    with pytest.raises(ValueError, match=r"bytes are not JSON-serializable"):
        pyscx.from_anndata(
            make(pd.DataFrame({"a": [1, 2]}, index=pd.Index([b"r0", b"r1"]))),
            str(tmp_dir / "bidx.scx"),
        )
    with pytest.raises(ValueError, match=r"bytes are not JSON-serializable"):
        pyscx.from_anndata(
            make(pd.DataFrame({"c": pd.Categorical([b"a", b"b"])}, index=["r0", "r1"])),
            str(tmp_dir / "bcat.scx"),
        )


def test_boolean_categorical_round_trips_through_h5ad(tmp_dir):
    """Boolean categories are representable, so they must not be declined.

    Round 1 refused them on the premise that no reader here takes them back.
    Measured false: an anndata-written ordered boolean categorical ingests with
    values, categories and `ordered` intact, so the guard was a false refusal
    that demoted a valid frame to a dict. Found by codex.
    """
    import anndata
    import scipy.sparse as sp

    import pyscx

    df = pd.DataFrame(
        {"bc": pd.Categorical([True, False], categories=[False, True], ordered=True)},
        index=pd.Index(["r0", "r1"], name="row"),
    )
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
        uns={"t": df},
    )

    # scx -> h5ad: a real DataFrame, not a demoted dict.
    scx_path = str(tmp_dir / "boolcat.scx")
    pyscx.from_anndata(adata, scx_path)
    out = str(tmp_dir / "boolcat.h5ad")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no demotion warning
        pyscx.to_h5ad(scx_path, out)
    got = anndata.read_h5ad(out).uns["t"]
    assert isinstance(got, pd.DataFrame)
    assert list(got["bc"]) == [True, False]
    assert list(got["bc"].cat.categories) == [False, True]
    assert got["bc"].cat.ordered is True

    # h5ad -> scx, from anndata's own writer: the direction the false premise
    # claimed was impossible.
    native = str(tmp_dir / "boolcat_native.h5ad")
    adata.write_h5ad(native)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyscx.from_h5ad(native, str(tmp_dir / "boolcat_native.scx"))
    back = pyscx.open(str(tmp_dir / "boolcat_native.scx")).to_anndata().uns["t"]
    pd.testing.assert_frame_equal(back, df, check_dtype=True)


def test_exported_dataframe_children_carry_anndata_encoding_metadata(tmp_dir):
    """anndata reads an unmarked element only under an `OldFormatWarning`.

    The obs/var writer this arm replaced stamped every child; the direct writer
    initially did not, producing eight such warnings per file. Found by codex.
    The three that remain on any exported file (`/obs/_index`, `/uns`,
    `/var/_index`) are pre-existing and unrelated to frames — this asserts the
    frame's own children, not the file total.
    """
    import anndata
    import h5py
    import scipy.sparse as sp

    import pyscx

    adata = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
        uns={"t": _uns_frame()},
    )
    scx_path = str(tmp_dir / "stamped.scx")
    pyscx.from_anndata(adata, scx_path)
    out = str(tmp_dir / "stamped.h5ad")
    pyscx.to_h5ad(scx_path, out)

    with h5py.File(out) as f:
        group = f["uns/t"]
        assert group.attrs["encoding-type"] == "dataframe"
        for key in group:
            assert "encoding-type" in group[key].attrs, f"uns/t/{key} is unmarked"
        for key in group["grade"]:
            assert "encoding-type" in group["grade"][key].attrs, f"grade/{key} unmarked"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anndata.read_h5ad(out)
    offenders = [
        str(w.message) for w in caught if "OldFormat" in type(w.message).__name__
    ]
    assert not [m for m in offenders if "/uns/t" in m], offenders


def test_uns_dataframe_ingest_refuses_path_bearing_attributes(tmp_dir):
    """`_index` / `column-order` are attributes; HDF5 reads them as *paths*.

    Setting a frame's `_index` to `"/obs/_index"` made the reader rebuild the
    frame with obs's barcodes as its index — wrong data, no warning, under
    `strict_uns=True` too. A crafted `column-order` could pull in any dataset in
    the file the same way. The export side got a member-name guard in round 2;
    this is its missing counterpart. Found by codex.
    """
    import anndata
    import h5py
    import scipy.sparse as sp

    import pyscx

    def write_frame(name):
        adata = anndata.AnnData(
            X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
            uns={"t": pd.DataFrame({"a": [1.0, 2.0]}, index=["r0", "r1"])},
        )
        path = str(tmp_dir / f"{name}.h5ad")
        adata.write_h5ad(path)
        return path

    # A path-bearing index name.
    idx = write_frame("pathidx")
    with h5py.File(idx, "a") as f:
        del f["uns/t"].attrs["_index"]
        f["uns/t"].attrs["_index"] = "/obs/_index"

    with pytest.raises(Exception, match=r"not a member of the dataframe group"):
        pyscx.from_h5ad(idx, str(tmp_dir / "pathidx_strict.scx"), strict_uns=True)

    with pytest.warns(UserWarning, match="unsupported_uns_dataframe_column"):
        pyscx.from_h5ad(idx, str(tmp_dir / "pathidx.scx"))
    got = pyscx.open(str(tmp_dir / "pathidx.scx")).read_uns()["t"]
    # Falls back to the dict recurse; obs's barcodes never reach the frame.
    assert isinstance(got, dict)
    assert list(got["a"]) == [1.0, 2.0]
    assert "c0" not in str(got)

    # A path-bearing column entry.
    cols = write_frame("pathcol")
    with h5py.File(cols, "a") as f:
        del f["uns/t"].attrs["column-order"]
        f["uns/t"].attrs["column-order"] = np.array(
            ["a", "/obs/_index"], dtype=h5py.special_dtype(vlen=str)
        )
    with pytest.warns(UserWarning, match="unsupported_uns_dataframe_column"):
        pyscx.from_h5ad(cols, str(tmp_dir / "pathcol.scx"))
    frame = pyscx.open(str(tmp_dir / "pathcol.scx")).to_anndata().uns["t"]
    assert list(frame.columns) == ["a"], "the path-bearing column must be dropped"


def test_uns_dataframe_export_declines_out_of_range_categorical_codes(tmp_dir):
    """Codes were narrowed with `as i32`, which wraps rather than failing.

    An envelope carrying i64 codes `[2**32, 2**32 + 1]` exported silently as the
    first two categories. A raw envelope reaches `set_uns` as an ordinary dict,
    so this needs pyscx's encoder nowhere in the path. Found by codex.
    """
    import base64

    import anndata
    import scipy.sparse as sp

    import pyscx

    adata = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
        uns={"k": 1.0},
    )
    scx_path = str(tmp_dir / "codes.scx")
    pyscx.from_anndata(adata, scx_path)

    def env(codes):
        return {
            "__scx_type__": "pandas.DataFrame",
            "index": {
                "__scx_type__": "pandas.Index",
                "name": None,
                "data": {
                    "__scx_type__": "ndarray", "dtype": "object", "shape": [2],
                    "encoding": "json", "data": ["r0", "r1"],
                },
            },
            "columns": ["c"],
            "data": {
                "c": {
                    "__scx_type__": "categorical", "ordered": False,
                    "codes": {
                        "__scx_type__": "ndarray", "dtype": "<i8", "shape": [2],
                        "encoding": "base64le",
                        "data": base64.b64encode(
                            np.array(codes, dtype="<i8").tobytes()
                        ).decode(),
                    },
                    "categories": {
                        "__scx_type__": "ndarray", "dtype": "object", "shape": [2],
                        "encoding": "json", "data": ["a", "b"],
                    },
                }
            },
        }

    for name, codes in [("wrapping", [2**32, 2**32 + 1]), ("out_of_range", [0, 5])]:
        pyscx.set_uns(scx_path, {"t": env(codes)})
        out = str(tmp_dir / f"codes_{name}.h5ad")
        with pytest.warns(UserWarning, match="uns_exported_as_raw_envelope"):
            pyscx.to_h5ad(scx_path, out)
        got = anndata.read_h5ad(out).uns["t"]
        assert isinstance(got, dict), f"{name}: must decline, not write wrong codes"

    # A valid code set still exports as a real frame — the guard is not blanket.
    pyscx.set_uns(scx_path, {"t": env([1, 0])})
    ok = str(tmp_dir / "codes_ok.h5ad")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyscx.to_h5ad(scx_path, ok)
    assert list(anndata.read_h5ad(ok).uns["t"]["c"]) == ["b", "a"]
