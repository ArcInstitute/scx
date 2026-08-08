"""T1.2: `adata.raw` round-trips through SCX.

`adata.raw` has its OWN (wider) var axis. These tests confirm the file
ingest path (`pyscx.from_h5ad`) preserves it and that both
`pyscx.open(...).to_anndata()` reconstruction and `pyscx.to_h5ad` export
reproduce `adata.raw.X` and `adata.raw.var`.
"""

from __future__ import annotations

import numpy as np
import pytest


def _dense(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _adata_with_raw(n_obs=20, n_vars=12, raw_n_vars=30):
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0.0
    obs = pd.DataFrame(
        {"n_counts": np.arange(n_obs, dtype=np.int32)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"mean_expr": np.linspace(0.1, 1.0, n_vars, dtype=np.float32)},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)

    # Raw: WIDER var axis (raw_n_vars > n_vars), integer counts.
    raw_dense = rng.integers(0, 40, size=(n_obs, raw_n_vars)).astype(np.float32)
    raw_dense[rng.random((n_obs, raw_n_vars)) > 0.4] = 0.0
    raw_var = pd.DataFrame(
        {"gene_symbol": [f"SYM{i}" for i in range(raw_n_vars)]},
        index=[f"raw_gene_{i}" for i in range(raw_n_vars)],
    )
    adata.raw = anndata.AnnData(X=sp.csr_matrix(raw_dense), var=raw_var)
    return adata, raw_dense


def test_raw_reconstructed_in_to_anndata(tmp_dir):
    import anndata  # noqa: F401
    import pyscx

    n_obs, n_vars, raw_n_vars = 20, 12, 30
    src, raw_dense = _adata_with_raw(n_obs, n_vars, raw_n_vars)

    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is not None, "adata.raw must be reconstructed"
    assert out.raw.shape == (n_obs, raw_n_vars)
    assert out.n_vars == n_vars, "main X var axis must be unchanged"
    np.testing.assert_array_equal(raw_dense, _dense(out.raw.X))
    assert list(out.raw.var_names) == [f"raw_gene_{i}" for i in range(raw_n_vars)]


def test_raw_round_trips_to_h5ad(tmp_dir):
    import anndata
    import pyscx

    n_obs, n_vars, raw_n_vars = 20, 12, 30
    src, raw_dense = _adata_with_raw(n_obs, n_vars, raw_n_vars)

    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    h5ad_out = str(tmp_dir / "out.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)

    rt = anndata.read_h5ad(h5ad_out)
    assert rt.raw is not None, "raw must survive scx → h5ad"
    assert rt.raw.shape == (n_obs, raw_n_vars)
    np.testing.assert_array_equal(raw_dense, _dense(rt.raw.X))
    assert list(rt.raw.var_names) == [f"raw_gene_{i}" for i in range(raw_n_vars)]


def test_raw_dropped_with_warning_in_backed_mode(tmp_dir):
    import anndata  # noqa: F401
    import pyscx

    src, _ = _adata_with_raw(20, 12, 30)
    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    # Backed mode does not reconstruct raw's obs axis → drop + warn
    # (human-readable DroppedRaw message), never silent.
    with pytest.warns(UserWarning, match="dropped_raw"):
        out = pyscx.open(scx_path).to_anndata(backed=True)
    assert out.raw is None


def test_no_raw_is_none(tmp_dir):
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(1)
    dense = rng.integers(0, 10, size=(8, 5)).astype(np.float32)
    src = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    h5ad_in = str(tmp_dir / "noraw.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "noraw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is None


# ---------------------------------------------------------------------------
# The in-memory write path (`pyscx.from_anndata` / `pyscx.write`).
#
# `from_h5ad` has always written the raw section family; the in-memory path
# never mentioned raw at all, so the same dataset kept or lost `.raw`
# depending on which door it came through — with no warning either way.
# ---------------------------------------------------------------------------


def test_from_anndata_writes_raw(tmp_dir):
    import pyscx

    n_obs, n_vars, raw_n_vars = 20, 12, 30
    src, raw_dense = _adata_with_raw(n_obs, n_vars, raw_n_vars)

    scx_path = str(tmp_dir / "mem.scx")
    pyscx.from_anndata(src, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is not None, "from_anndata must write adata.raw"
    # Raw keeps its OWN, wider var axis — X's is untouched.
    assert out.raw.shape == (n_obs, raw_n_vars)
    assert out.n_vars == n_vars
    np.testing.assert_array_equal(raw_dense, _dense(out.raw.X))
    assert list(out.raw.var_names) == [f"raw_gene_{i}" for i in range(raw_n_vars)]
    assert list(out.raw.var["gene_symbol"]) == [f"SYM{i}" for i in range(raw_n_vars)]


def test_from_anndata_and_from_h5ad_agree_on_raw(tmp_dir):
    """The asymmetry stated as an assertion rather than a paragraph.

    One fixture, two doors into SCX. Before the fix the h5ad door kept raw
    and the in-memory door silently dropped it; this is the test that would
    have caught that.
    """
    import pyscx

    src, _ = _adata_with_raw(20, 12, 30)

    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    via_h5ad = str(tmp_dir / "via_h5ad.scx")
    pyscx.from_h5ad(h5ad_in, via_h5ad)

    via_mem = str(tmp_dir / "via_mem.scx")
    pyscx.from_anndata(src, via_mem)

    a = pyscx.open(via_h5ad).to_anndata()
    b = pyscx.open(via_mem).to_anndata()

    assert (a.raw is None) == (b.raw is None), "the two doors disagree on raw"
    assert a.raw is not None
    assert a.raw.shape == b.raw.shape
    np.testing.assert_array_equal(_dense(a.raw.X), _dense(b.raw.X))
    assert list(a.raw.var_names) == list(b.raw.var_names)
    assert list(a.raw.var["gene_symbol"]) == list(b.raw.var["gene_symbol"])


def test_scx_backed_route_warns_when_dropping_raw(tmp_dir):
    """`open(f).to_anndata(backed=True)` → `from_anndata` loses raw for good.

    The SCX→SCX routes write layers/obsm from the Python object, and backed
    reconstruction sets `.raw = None`, so there is nothing on the object to
    carry. The loss is real regardless — the source file has raw and the
    output will not — so it must be warned rather than silent. The read-side
    warning is not a substitute: it says "the on-disk raw sections are
    preserved", which is true of the SOURCE and says nothing about the file
    being written here.
    """
    import pyscx

    src, _ = _adata_with_raw(20, 12, 30)
    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    with pytest.warns(UserWarning, match="dropped_raw"):
        backed = pyscx.open(scx_path).to_anndata(backed=True)
    assert backed.raw is None

    out_path = str(tmp_dir / "rewritten.scx")
    with pytest.warns(UserWarning, match="dropped_raw_on_write") as rec:
        pyscx.from_anndata(backed, out_path)

    # The write-side notice must NOT be the read-side one: reusing
    # `DroppedRaw` here would tell the user "the on-disk raw sections are
    # preserved" about a file that has just been written without any.
    msg = str(rec[0].message)
    assert "on-disk raw sections are preserved" not in msg
    assert "the output file will have no raw" in msg
    # And it must be THIS door's remedy. Without this, swapping the two
    # sites' `reason` strings would fail the Rust sort-path test but stay
    # green here, so the cross-door check would only be half-closed.
    assert "convert from the h5ad" in msg.lower()

    assert pyscx.open(out_path).to_anndata().raw is None


# --- new-behaviour tests (not red-first; they cover code the fix adds) ------


@pytest.mark.parametrize("raw_format", ["dense", "csc"])
def test_from_anndata_writes_non_csr_raw(tmp_dir, raw_format):
    """`raw.X` need not be CSR — `ensure_csr` converts, as it does for X."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    n_obs, n_vars, raw_n_vars = 10, 6, 9
    src, _ = _adata_with_raw(n_obs, n_vars, raw_n_vars)

    rng = np.random.default_rng(7)
    raw_dense = rng.integers(0, 20, size=(n_obs, raw_n_vars)).astype(np.float32)
    raw_dense[rng.random((n_obs, raw_n_vars)) > 0.5] = 0.0
    raw_x = raw_dense if raw_format == "dense" else sp.csc_matrix(raw_dense)
    src.raw = anndata.AnnData(
        X=raw_x,
        var=pd.DataFrame(index=[f"raw_gene_{i}" for i in range(raw_n_vars)]),
    )

    scx_path = str(tmp_dir / f"raw_{raw_format}.scx")
    pyscx.from_anndata(src, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is not None
    np.testing.assert_array_equal(raw_dense, _dense(out.raw.X))


class _RawOverride:
    """An AnnData-like that forwards everything except `.raw`.

    Needed because whether `adata.raw = <misaligned>` is even constructible
    depends on the anndata version: 0.12.10 accepts it (and then reports
    `raw.shape[0]` as the parent's `n_obs`, hiding the mismatch), while the
    version CI pins rejects it in `Raw.__init__`. Going through a duck-typed
    wrapper — a shape `from_anndata` already supports — reaches the pyscx
    guard on both, instead of testing which anndata happens to be installed.
    """

    def __init__(self, adata, raw):
        self._adata = adata
        self.raw = raw

    def __getattr__(self, name):
        return getattr(self._adata, name)


class _RawLike:
    """The minimal surface `write_raw_from_anndata` touches on `.raw`."""

    def __init__(self, X, var):
        self.X = X
        self.var = var
        self.varm = {}


def test_from_anndata_rejects_raw_with_mismatched_obs_axis(tmp_dir):
    """A raw whose rows don't line up with X must never reach disk.

    Writing one would produce a file whose raw rows silently belong to
    different cells — and `adata.raw.shape` cannot be used to notice, since
    it reports the parent's `n_obs` rather than `raw.X.shape[0]`.
    """
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    src, _ = _adata_with_raw(20, 12, 30)
    short = sp.csr_matrix(np.ones((17, 30), dtype=np.float32))
    bad = _RawOverride(
        src,
        _RawLike(short, pd.DataFrame(index=[f"raw_gene_{i}" for i in range(30)])),
    )

    with pytest.raises(ValueError, match=r"17 rows but X has 20"):
        pyscx.from_anndata(bad, str(tmp_dir / "bad.scx"))


def test_raw_override_wrapper_is_a_faithful_stand_in(tmp_dir):
    """Guard the guard: the wrapper above must otherwise behave normally.

    If `_RawOverride` silently broke some attribute `from_anndata` needs,
    the rejection test would pass for the wrong reason.
    """
    import pyscx

    src, raw_dense = _adata_with_raw(20, 12, 30)
    wrapped = _RawOverride(src, src.raw)

    scx_path = str(tmp_dir / "wrapped.scx")
    pyscx.from_anndata(wrapped, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is not None
    assert out.shape == (20, 12)
    np.testing.assert_array_equal(raw_dense, _dense(out.raw.X))


def test_from_anndata_without_raw_writes_no_raw_sections(tmp_dir):
    """A raw-free AnnData must not gain a `raw/var` section or a warning."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import warnings as _warnings
    import h5py
    import pyscx

    rng = np.random.default_rng(3)
    dense = rng.integers(0, 10, size=(8, 5)).astype(np.float32)
    src = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    assert src.raw is None

    scx_path = str(tmp_dir / "noraw_mem.scx")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        pyscx.from_anndata(src, scx_path)
    assert not [w for w in caught if "raw" in str(w.message)]

    assert pyscx.open(scx_path).to_anndata().raw is None

    # `has_raw` is derived from the catalog, so a stray raw section would
    # surface as a `/raw` group on export.
    h5ad_out = str(tmp_dir / "noraw_mem.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)
    with h5py.File(h5ad_out, "r") as f:
        assert "raw" not in f


def test_raw_index_dtype_is_keyed_off_raws_own_var_axis(tmp_dir):
    """A raw axis past 65535 must widen raw's indices, not X's.

    `index_dtype` is per-shard. Reusing X's — narrow here, since X has 6
    genes — would truncate raw column indices above 65535 to u16.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    n_obs, n_vars, raw_n_vars = 4, 6, 70_000
    rng = np.random.default_rng(11)
    src = anndata.AnnData(
        X=sp.csr_matrix(rng.integers(1, 9, size=(n_obs, n_vars)).astype(np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    # A handful of nonzeros, deliberately beyond the u16 ceiling.
    cols = np.array([3, 65_535, 65_536, 69_999], dtype=np.int32)
    raw = sp.csr_matrix(
        (
            np.array([1, 2, 3, 4] * n_obs, dtype=np.float32),
            np.tile(cols, n_obs),
            np.arange(0, 4 * (n_obs + 1), 4, dtype=np.int64),
        ),
        shape=(n_obs, raw_n_vars),
    )
    src.raw = anndata.AnnData(
        X=raw, var=pd.DataFrame(index=[f"r{i}" for i in range(raw_n_vars)])
    )

    scx_path = str(tmp_dir / "wideraw.scx")
    pyscx.from_anndata(src, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw.shape == (n_obs, raw_n_vars)
    assert out.n_vars == n_vars
    np.testing.assert_array_equal(np.sort(out.raw.X[0].indices), cols)
    np.testing.assert_array_equal(_dense(out.raw.X), _dense(raw))


def test_lazy_x_route_warns_when_dropping_raw(tmp_dir):
    """`warn_source_raw_dropped` has two call sites; cover the other one.

    The backed and lazy rewrites share the helper, so a regression that
    skipped only the lazy arm would otherwise stay green.
    """
    import pyscx

    src, _ = _adata_with_raw(20, 12, 30)
    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    with pytest.warns(UserWarning, match="dropped_raw"):
        backed = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(backed, target_sum=1e4)
    assert type(backed.X).__name__ == "ScxLazyTransformedDataset"

    with pytest.warns(UserWarning, match="dropped_raw_on_write"):
        pyscx.from_anndata(backed, str(tmp_dir / "lazy_out.scx"))


@pytest.mark.parametrize("raw_n_vars", [65_535, 65_536, 65_537])
def test_raw_index_dtype_boundary(tmp_dir, raw_n_vars):
    """Around the `raw_n_vars <= 65535` branch.

    65537 is the arm that carries the weight: it is the first width whose
    highest column index (65536) does not fit u16, so a wrongly-narrow
    `index_dtype` fails here. At exactly 65536 the branch flips but the
    highest index is still 65535, so that arm is not observable through the
    public API — it is kept to document the flip point, not as evidence.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    n_obs, n_vars = 3, 5
    src = anndata.AnnData(
        X=sp.csr_matrix(np.ones((n_obs, n_vars), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    highest = raw_n_vars - 1  # 65534 (fits u16) / 65535 (the u16 ceiling)
    raw = sp.csr_matrix(
        (
            np.ones(n_obs, dtype=np.float32),
            np.full(n_obs, highest, dtype=np.int32),
            np.arange(n_obs + 1, dtype=np.int64),
        ),
        shape=(n_obs, raw_n_vars),
    )
    src.raw = anndata.AnnData(
        X=raw, var=pd.DataFrame(index=[f"r{i}" for i in range(raw_n_vars)])
    )

    scx_path = str(tmp_dir / f"boundary_{raw_n_vars}.scx")
    pyscx.from_anndata(src, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw.shape == (n_obs, raw_n_vars)
    np.testing.assert_array_equal(out.raw.X[0].indices, np.array([highest]))


def test_raw_with_multiple_layers(tmp_dir):
    """Raw is written after the layers loop; confirm they do not interfere."""
    import pyscx

    n_obs, n_vars, raw_n_vars = 20, 12, 30
    src, raw_dense = _adata_with_raw(n_obs, n_vars, raw_n_vars)
    x_dense = _dense(src.X)
    src.layers["counts"] = src.X.copy()
    src.layers["scaled"] = src.X.copy()

    scx_path = str(tmp_dir / "raw_layers.scx")
    pyscx.from_anndata(src, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert sorted(out.layers.keys()) == ["counts", "scaled"]
    np.testing.assert_array_equal(raw_dense, _dense(out.raw.X))
    np.testing.assert_array_equal(x_dense, _dense(out.X))
    for key in ("counts", "scaled"):
        np.testing.assert_array_equal(x_dense, _dense(out.layers[key]))


def test_empty_raw_matrix_round_trips(tmp_dir):
    """An all-zero raw has no nonzeros to encode; it must still round-trip."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    n_obs, n_vars, raw_n_vars = 6, 4, 9
    src, _ = _adata_with_raw(n_obs, n_vars, raw_n_vars)
    src.raw = anndata.AnnData(
        X=sp.csr_matrix((n_obs, raw_n_vars), dtype=np.float32),
        var=pd.DataFrame(index=[f"r{i}" for i in range(raw_n_vars)]),
    )
    assert src.raw.X.nnz == 0

    scx_path = str(tmp_dir / "emptyraw.scx")
    pyscx.from_anndata(src, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is not None, "an all-zero raw is still a raw"
    assert out.raw.shape == (n_obs, raw_n_vars)
    assert out.raw.X.nnz == 0
    assert list(out.raw.var_names) == [f"r{i}" for i in range(raw_n_vars)]


def test_in_place_contract_extends_to_raw(tmp_dir):
    """`in_place` now governs `adata.raw.X` too, not just X and layers.

    Both directions, because the default is the one callers rely on: a
    default write must leave the caller's raw CSR untouched, and
    `in_place=True` must actually sort it in place.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    def make():
        adata = anndata.AnnData(
            X=sp.csr_matrix(np.eye(4, dtype=np.float32)),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        )
        raw_x = sp.csr_matrix(
            (
                np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
                np.array([2, 0, 1, 3, 1, 0], dtype=np.int32),
                np.array([0, 2, 4, 5, 6], dtype=np.int64),
            ),
            shape=(4, 4),
        )
        assert not raw_x.has_sorted_indices
        adata.raw = anndata.AnnData(
            X=raw_x, var=pd.DataFrame(index=[f"r{i}" for i in range(4)])
        )
        return adata, raw_x

    a, raw_a = make()
    pyscx.from_anndata(a, str(tmp_dir / "default.scx"))
    assert not raw_a.has_sorted_indices, "default write must not mutate raw.X"

    b, raw_b = make()
    pyscx.from_anndata(b, str(tmp_dir / "inplace.scx"), in_place=True)
    assert raw_b.has_sorted_indices, "in_place=True must sort raw.X in place"

    # Either way the stored matrix is canonical and identical.
    left = pyscx.open(str(tmp_dir / "default.scx")).to_anndata().raw.X
    right = pyscx.open(str(tmp_dir / "inplace.scx")).to_anndata().raw.X
    np.testing.assert_array_equal(_dense(left), _dense(right))


def test_from_anndata_warns_on_raw_varm(tmp_dir):
    """SCX has no section for `raw.varm`; drop it loudly, not silently."""
    import pyscx

    src, _ = _adata_with_raw(20, 12, 30)
    src.raw.varm["PCs"] = np.zeros((30, 3), dtype=np.float32)
    assert len(src.raw.varm) == 1

    with pytest.warns(UserWarning, match="dropped_raw_varm"):
        pyscx.from_anndata(src, str(tmp_dir / "rawvarm.scx"))
