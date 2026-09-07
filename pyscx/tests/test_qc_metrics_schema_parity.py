"""`calculate_qc_metrics` returns one schema, whatever `X` is.

Historically this op had two implementations behind one name: a backed or lazy
`X` ran the native streaming kernel, and a scipy/dense `X` was handed to
`sc.pp.calculate_qc_metrics`. They did not write the same columns. The streaming
route never wrote `log1p_n_genes_by_counts` (obs) or `mean_counts` /
`log1p_mean_counts` / `pct_dropout_by_counts` (var), and the scanpy route wrote
`pct_counts_in_top_{50,100,200,500}_genes` that the streaming route had no way
to produce.

Nothing caught that, because every existing assertion in the suite is
*membership* (`"total_counts" in adata.obs`). These tests assert **set
equality**, which is the only shape of assertion that can see a missing column.

The narrow-file test is the user-visible half: because the delegation never set
`percent_top`, scanpy's default `(50, 100, 200, 500)` applied, and scanpy's
`check_ns` raises `IndexError: Positions outside range of features.` for any
file with fewer than 500 genes. Three fixtures elsewhere in this suite were sized
>= 500 genes *because of that*; two of them said so in their comments, and this
change narrows one of those two (`test_accel_route_metadata.py`) to 25 genes to
show the constraint is gone.
"""

import numpy as np
import pytest


N_OBS, N_VARS = 240, 600
NARROW_OBS, NARROW_VARS = 40, 300

# scanpy's own default. Named here so the tests that opt in read the same way a
# migrating script does.
SCANPY_PERCENT_TOP = (50, 100, 200, 500)


def _counts_adata(n_obs, n_vars, seed):
    """Non-integer float32 counts.

    Integers would make every partial sum an exact f64, so a value comparison
    would pass under any accumulation order and could not distinguish our f64
    accumulation from scanpy's f32 pairwise sums.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.RandomState(seed)
    dense = (rng.random_sample((n_obs, n_vars)) * 1e3).astype(np.float32)
    dense[rng.random_sample((n_obs, n_vars)) > 0.35] = 0
    x = sp.csr_matrix(dense)
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_vars)])
    # Every 40th gene is "mitochondrial": a non-empty subset on both fixtures.
    var["mt"] = np.arange(n_vars) % 40 == 0
    adata = anndata.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=var,
    )
    adata.layers["counts"] = x.copy()
    return adata


@pytest.fixture
def qc_adata():
    """600 genes — wide enough that scanpy's default `percent_top` works, so
    the schema comparison is not masked by the narrow-file crash."""
    return _counts_adata(N_OBS, N_VARS, seed=7)


@pytest.fixture
def narrow_adata():
    """300 genes — narrower than scanpy's largest default `percent_top` entry."""
    return _counts_adata(NARROW_OBS, NARROW_VARS, seed=11)


@pytest.fixture
def qc_path(qc_adata, tmp_dir):
    import pyscx

    path = str(tmp_dir / "qc_schema.scx")
    pyscx.from_anndata(qc_adata, path, shard_size=50)
    return path


def _backed(path):
    import pyscx

    adata = pyscx.open(path).to_anndata(backed=True)
    from pyscx import ScxBackedSparseDataset

    assert isinstance(adata.X, ScxBackedSparseDataset), (
        "premise: the backed arm must actually be backed, or it is a second "
        "in-memory arm wearing a hat"
    )
    return adata


def _lazy(path):
    import pyscx

    adata = _backed(path)
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    from pyscx import ScxLazyTransformedDataset

    assert isinstance(adata.X, ScxLazyTransformedDataset), "premise: lazy arm"
    return adata


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _materialized(adata):
    """A plain scipy AnnData holding whatever `adata.X` currently evaluates to."""
    import anndata
    import scipy.sparse as sp

    return anndata.AnnData(
        X=sp.csr_matrix(_dense(adata.X[:])),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )


# ---------------------------------------------------------------------------
# One schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qc_vars", [None, ["mt"]])
def test_every_route_writes_the_same_columns(qc_adata, qc_path, qc_vars):
    """Set equality, not membership: the assertion shape that sees a gap."""
    import pyscx

    mem = qc_adata.copy()
    pyscx.accel.calculate_qc_metrics(mem, qc_vars=qc_vars)

    backed = _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(backed, qc_vars=qc_vars)

    lazy = _lazy(qc_path)
    pyscx.accel.calculate_qc_metrics(lazy, qc_vars=qc_vars)

    base_obs = set(qc_adata.obs.columns)
    base_var = set(qc_adata.var.columns)
    got = {
        name: (set(a.obs.columns) - base_obs, set(a.var.columns) - base_var)
        for name, a in (("in-memory", mem), ("backed", backed), ("lazy", lazy))
    }
    assert got["in-memory"] == got["backed"] == got["lazy"], got


def test_columns_match_scanpy(qc_adata):
    """The column set is scanpy's, at scanpy's own `percent_top=None`."""
    import scanpy as sc

    import pyscx

    ours = qc_adata.copy()
    pyscx.accel.calculate_qc_metrics(ours, qc_vars=["mt"])

    theirs = qc_adata.copy()
    sc.pp.calculate_qc_metrics(
        theirs, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True
    )

    base_obs, base_var = set(qc_adata.obs.columns), set(qc_adata.var.columns)
    assert set(ours.obs.columns) - base_obs == set(theirs.obs.columns) - base_obs
    assert set(ours.var.columns) - base_var == set(theirs.var.columns) - base_var


@pytest.mark.parametrize("route", ["in-memory", "backed"])
def test_values_match_scanpy(qc_adata, qc_path, route):
    """Same numbers, at the tolerance docs/testing.md already claims (1e-5).

    Our f64 accumulation and scanpy's f32 pairwise sums do not agree bit for
    bit; nothing here pretends otherwise.

    Parametrized over the route on purpose. Run on an in-memory `X` alone this
    assertion is vacuous while the delegation exists — it compares scanpy with
    itself, which is exactly how the benchmark suite's own QC parity check came
    to prove nothing.
    """
    import scanpy as sc

    import pyscx

    ours = qc_adata.copy() if route == "in-memory" else _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(ours, qc_vars=["mt"])

    theirs = qc_adata.copy()
    sc.pp.calculate_qc_metrics(
        theirs, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True
    )

    for frame in ("obs", "var"):
        mine, ref = getattr(ours, frame), getattr(theirs, frame)
        added = [c for c in ref.columns if c not in getattr(qc_adata, frame).columns]
        assert added, f"premise: scanpy added no {frame} columns"
        for col in added:
            np.testing.assert_allclose(
                mine[col].to_numpy(dtype=np.float64),
                ref[col].to_numpy(dtype=np.float64),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{frame}['{col}']",
            )


def test_backed_and_in_memory_agree_on_values(qc_adata, qc_path):
    """The two routes are one kernel, so this is exact, not approximate."""
    import pyscx

    mem = qc_adata.copy()
    pyscx.accel.calculate_qc_metrics(mem, qc_vars=["mt"])
    backed = _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(backed, qc_vars=["mt"])

    for frame in ("obs", "var"):
        a, b = getattr(mem, frame), getattr(backed, frame)
        added = [c for c in a.columns if c not in getattr(qc_adata, frame).columns]
        for col in added:
            np.testing.assert_array_equal(
                a[col].to_numpy(dtype=np.float64),
                b[col].to_numpy(dtype=np.float64),
                err_msg=f"{frame}['{col}'] differs between in-memory and backed",
            )


def test_lazy_matches_the_same_transform_materialized(qc_path):
    """The lazy arm's numbers are the transformed matrix's, not the raw one's."""
    import pyscx

    lazy = _lazy(qc_path)
    ref = _materialized(lazy)

    pyscx.accel.calculate_qc_metrics(lazy, qc_vars=["mt"])
    pyscx.accel.calculate_qc_metrics(ref, qc_vars=["mt"])

    for frame, cols in (
        ("obs", ("total_counts", "n_genes_by_counts", "pct_counts_mt")),
        ("var", ("total_counts", "n_cells_by_counts", "mean_counts")),
    ):
        a, b = getattr(lazy, frame), getattr(ref, frame)
        for col in cols:
            np.testing.assert_allclose(
                a[col].to_numpy(dtype=np.float64),
                b[col].to_numpy(dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{frame}['{col}']",
            )


# ---------------------------------------------------------------------------
# The narrow-file crash
# ---------------------------------------------------------------------------


def test_a_file_narrower_than_500_genes_works_in_memory(narrow_adata):
    """Reproduces the delegation's inherited `IndexError`.

    scanpy's default `percent_top=(50, 100, 200, 500)` goes through `check_ns`,
    which raises `IndexError: Positions outside range of features.` when
    `max(ns) > n_vars`. pyscx never set `percent_top`, so any file with fewer
    than 500 genes could not be QC'd in memory at all.
    """
    import pyscx

    assert narrow_adata.n_vars < max(SCANPY_PERCENT_TOP), "premise: narrow fixture"
    pyscx.accel.calculate_qc_metrics(narrow_adata, qc_vars=["mt"])
    assert "total_counts" in narrow_adata.obs
    assert "mean_counts" in narrow_adata.var


def test_a_narrow_file_agrees_with_scanpy(narrow_adata):
    import scanpy as sc

    import pyscx

    ours = narrow_adata.copy()
    pyscx.accel.calculate_qc_metrics(ours, qc_vars=["mt"])
    theirs = narrow_adata.copy()
    sc.pp.calculate_qc_metrics(
        theirs, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True
    )
    np.testing.assert_allclose(
        ours.obs["total_counts"].to_numpy(dtype=np.float64),
        theirs.obs["total_counts"].to_numpy(dtype=np.float64),
        rtol=1e-5,
    )


def test_qc_vars_survives_an_scx_round_trip_on_an_eager_x(qc_path):
    """`open(p).to_anndata()` + `qc_vars=["mt"]` — the plainest SCX workflow.

    SCX round-trips a boolean `var` column back as pandas' nullable `boolean`
    dtype. The scanpy delegation indexed the matrix with `adata.var[v].values`,
    and scipy cannot take a `BooleanArray`, so this raised
    `AttributeError: 'BooleanArray' object has no attribute 'nonzero'` on an
    eager `X` while the backed route handled it. Running one kernel removes the
    asymmetry: `resolve_qc_masks` reads the column itself.
    """
    import pandas as pd

    import pyscx

    eager = pyscx.open(qc_path).to_anndata()
    assert isinstance(eager.var["mt"].dtype, pd.BooleanDtype), (
        "premise: the round-tripped mask must be nullable boolean, or this "
        "test is not exercising the reported failure"
    )
    pyscx.accel.calculate_qc_metrics(eager, qc_vars=["mt"])
    assert "pct_counts_mt" in eager.obs

    backed = _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(backed, qc_vars=["mt"])
    np.testing.assert_array_equal(
        eager.obs["pct_counts_mt"].to_numpy(dtype=np.float64),
        backed.obs["pct_counts_mt"].to_numpy(dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# percent_top
# ---------------------------------------------------------------------------


def test_percent_top_matches_scanpy_on_every_route(qc_adata, qc_path):
    import scanpy as sc

    import pyscx

    ns = (5, 10, 50)

    theirs = qc_adata.copy()
    sc.pp.calculate_qc_metrics(
        theirs, percent_top=list(ns), log1p=True, inplace=True
    )
    expected = {n: theirs.obs[f"pct_counts_in_top_{n}_genes"].to_numpy() for n in ns}

    mem = qc_adata.copy()
    pyscx.accel.calculate_qc_metrics(mem, percent_top=ns)
    backed = _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(backed, percent_top=ns)

    for label, a in (("in-memory", mem), ("backed", backed)):
        for n in ns:
            col = f"pct_counts_in_top_{n}_genes"
            assert col in a.obs, f"{label}: {col} missing"
            np.testing.assert_allclose(
                a.obs[col].to_numpy(dtype=np.float64),
                expected[n].astype(np.float64),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{label}: {col}",
            )


def test_percent_top_defaults_to_off(qc_adata):
    """pyscx's default is `None`, deliberately not scanpy's.

    scanpy's default raises on narrow files and produces four columns nobody
    asked for; the columns appear here only when requested.
    """
    import pyscx

    pyscx.accel.calculate_qc_metrics(qc_adata)
    assert not [c for c in qc_adata.obs.columns if c.startswith("pct_counts_in_top")]


def test_percent_top_covers_a_row_with_fewer_nonzeros_than_n(qc_path, tmp_dir):
    """A row with `nnz <= n` puts all of its mass in the top n: exactly 1.0.

    scanpy reaches the same answer by zero-padding its partition buffer
    (`top_segment_proportions_sparse_csr`); we reach it by clamping the prefix.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    dense = np.zeros((3, 20), dtype=np.float32)
    dense[0, :2] = [3.0, 1.0]      # nnz=2 < n=5
    dense[1, :] = np.arange(1, 21)  # nnz=20 > n=5
    # row 2 stays all-zero
    a = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(3)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(20)]),
    )
    pyscx.accel.calculate_qc_metrics(a, percent_top=(5,))
    got = a.obs["pct_counts_in_top_5_genes"].to_numpy()
    assert got[0] == pytest.approx(100.0), "a short row is entirely its own top-n"
    top5 = sum(sorted(dense[1], reverse=True)[:5])
    assert got[1] == pytest.approx(top5 / dense[1].sum() * 100.0)
    assert got[2] == pytest.approx(0.0), "zero-count row takes pyscx's 0.0 convention"


@pytest.mark.parametrize("bad", [(0,), (-1,), (N_VARS + 1,)])
def test_percent_top_out_of_range_raises_naming_the_width(qc_adata, bad):
    import pyscx

    with pytest.raises(ValueError, match=r"percent_top"):
        pyscx.accel.calculate_qc_metrics(qc_adata, percent_top=bad)


def test_percent_top_is_validated_against_the_visible_width(qc_path):
    """Under a column projection the bound is the *visible* gene count.

    An implementation that reads `n_vars` off the file accepts a value the
    projected matrix cannot serve.
    """
    import pyscx

    backed = _backed(qc_path)
    cols = list(range(9))
    backed.X.set_col_projection(cols)
    backed._var = backed.var.iloc[cols].copy()

    with pytest.raises(ValueError, match=r"percent_top"):
        pyscx.accel.calculate_qc_metrics(backed, percent_top=(20,))

    pyscx.accel.calculate_qc_metrics(backed, percent_top=(5,))
    assert "pct_counts_in_top_5_genes" in backed.obs


def test_percent_top_honours_a_deletion_vector(qc_path):
    """Top-N is per surviving row, and the output has one entry per visible row."""
    import pyscx

    backed = _backed(qc_path)
    before = backed.n_obs
    pyscx.accel.filter_cells(backed, min_counts=100_000)
    assert 0 < backed.n_obs < before, "premise: the deletion vector must bite"

    pyscx.accel.calculate_qc_metrics(backed, percent_top=(10,))
    got = backed.obs["pct_counts_in_top_10_genes"].to_numpy()
    assert got.shape == (backed.n_obs,)

    ref = _materialized(backed)
    pyscx.accel.calculate_qc_metrics(ref, percent_top=(10,), qc_vars=["mt"])
    pyscx.accel.calculate_qc_metrics(backed, percent_top=(10,), qc_vars=["mt"])

    # Every column, not just the top-N one: `mean_counts` and
    # `pct_dropout_by_counts` divide by the row count, and the physical count is
    # the wrong one on a filtered file — a mistake no assertion about the
    # cell axis can see.
    for frame in ("obs", "var"):
        a, b = getattr(backed, frame), getattr(ref, frame)
        for col in a.columns:
            if col == "mt":
                continue
            np.testing.assert_allclose(
                a[col].to_numpy(dtype=np.float64),
                b[col].to_numpy(dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{frame}['{col}'] under a deletion vector",
            )


def test_explicitly_stored_zeros_are_not_counted_as_expressed():
    """`n_genes_by_counts` counts nonzeros, not stored entries.

    The kernel reads a row's count off `indptr`, which is right for SCX shards —
    they never store a zero. A user's scipy matrix can, and `X[mask] = 0` is a
    common way to get one. Counting those would report more expressed genes and
    fewer dropouts than the data has. scanpy sidesteps it by calling
    `eliminate_zeros()` on the caller's matrix; we do it on our own copy, which
    reaches the same numbers without mutating anything the caller still holds.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    dense = np.array(
        [[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 5.0]], dtype=np.float32
    )
    x = sp.csr_matrix(dense)
    x.data[1] = 0.0  # an explicit zero, still stored
    assert x.nnz == 5 and (x.data == 0).sum() == 1, "premise: a stored zero"

    adata = anndata.AnnData(
        X=x,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    pyscx.accel.calculate_qc_metrics(adata)

    np.testing.assert_array_equal(
        adata.obs["n_genes_by_counts"].to_numpy(), np.array([2, 2])
    )
    np.testing.assert_array_equal(
        adata.var["n_cells_by_counts"].to_numpy(), np.array([2, 0, 1, 1])
    )
    np.testing.assert_allclose(
        adata.var["pct_dropout_by_counts"].to_numpy(), [0.0, 100.0, 50.0, 50.0]
    )
    # The caller's matrix is untouched — scanpy would have compacted it.
    assert adata.X.nnz == 5


# ---------------------------------------------------------------------------
# layer=
# ---------------------------------------------------------------------------


def test_layer_matches_running_on_a_sibling_whose_X_is_that_layer(qc_adata, qc_path):
    """`layer="counts"` on every route == the same op on an AnnData whose `X`
    is that layer."""
    import anndata

    import pyscx

    sibling = anndata.AnnData(
        X=qc_adata.layers["counts"].copy(),
        obs=qc_adata.obs.copy(),
        var=qc_adata.var.copy(),
    )
    pyscx.accel.calculate_qc_metrics(sibling, qc_vars=["mt"])

    mem = qc_adata.copy()
    pyscx.accel.calculate_qc_metrics(mem, qc_vars=["mt"], layer="counts")
    backed = _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(backed, qc_vars=["mt"], layer="counts")

    for label, a in (("in-memory", mem), ("backed", backed)):
        for col in ("total_counts", "n_genes_by_counts", "pct_counts_mt"):
            np.testing.assert_allclose(
                a.obs[col].to_numpy(dtype=np.float64),
                sibling.obs[col].to_numpy(dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{label}: obs['{col}'] under layer=",
            )


def test_layer_on_a_backed_file_reads_the_layer_not_x(qc_path):
    """The layer holds different values from `X` after a transform, so this
    tells "read the layer" apart from "silently read X"."""
    import pyscx

    lazy = _lazy(qc_path)  # X is normalized; layers["counts"] is not
    pyscx.accel.calculate_qc_metrics(lazy, layer="counts")
    from_layer = lazy.obs["total_counts"].to_numpy()

    plain = _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(plain)
    assert np.allclose(from_layer, plain.obs["total_counts"].to_numpy(), rtol=1e-6)


def test_unknown_layer_raises_naming_it(qc_adata):
    import pyscx

    with pytest.raises(ValueError, match=r"nope"):
        pyscx.accel.calculate_qc_metrics(qc_adata, layer="nope")


def test_layer_with_csc_is_rejected(qc_path):
    import pyscx

    backed = _backed(qc_path)
    with pytest.raises((ValueError, RuntimeError), match=r"(?i)layer"):
        pyscx.accel.calculate_qc_metrics(backed, layer="counts", prefer_format="csc")


# ---------------------------------------------------------------------------
# The remaining schema details
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["in-memory", "backed"])
def test_an_empty_qc_var_mask_still_gets_its_log1p_column(qc_adata, qc_path, route):
    """The column set must be a function of the kwargs, not of the data.

    The streaming route omitted `log1p_total_counts_<v>` for an all-False mask
    ("an empty mask keeps its historical shape") while scanpy wrote
    `log1p(0) == 0.0`, so the same call produced different columns depending on
    whether any gene happened to match. The in-memory arm passes on either
    implementation; the backed arm is the one that means something.
    """
    import pyscx

    adata = qc_adata.copy() if route == "in-memory" else _backed(qc_path)
    adata.var["none"] = np.zeros(adata.n_vars, dtype=bool)
    with pytest.warns(UserWarning):
        pyscx.accel.calculate_qc_metrics(adata, qc_vars=["none"], log1p=True)

    assert "log1p_total_counts_none" in adata.obs.columns
    assert np.all(adata.obs["log1p_total_counts_none"].to_numpy() == 0.0)


def test_log1p_false_drops_only_the_log1p_columns(qc_adata, qc_path):
    import pyscx

    mem = qc_adata.copy()
    pyscx.accel.calculate_qc_metrics(mem, qc_vars=["mt"], log1p=False)
    backed = _backed(qc_path)
    pyscx.accel.calculate_qc_metrics(backed, qc_vars=["mt"], log1p=False)

    assert not [c for c in mem.obs.columns if c.startswith("log1p_")]
    assert not [c for c in mem.var.columns if c.startswith("log1p_")]
    assert set(mem.obs.columns) == set(backed.obs.columns)
    assert set(mem.var.columns) == set(backed.var.columns)
    # Unconditional in scanpy, so unconditional here.
    assert "mean_counts" in mem.var.columns
    assert "pct_dropout_by_counts" in mem.var.columns


def test_inplace_false_returns_exactly_what_inplace_true_writes(qc_adata):
    import pyscx

    returned = qc_adata.copy()
    obs_df, var_df = pyscx.accel.calculate_qc_metrics(
        returned, qc_vars=["mt"], percent_top=(5,), inplace=False
    )
    assert "total_counts" not in returned.obs.columns

    written = qc_adata.copy()
    pyscx.accel.calculate_qc_metrics(written, qc_vars=["mt"], percent_top=(5,))

    added_obs = [c for c in written.obs.columns if c not in qc_adata.obs.columns]
    assert set(obs_df.columns) == set(added_obs)
    for col in added_obs:
        np.testing.assert_array_equal(
            obs_df[col].to_numpy(dtype=np.float64),
            written.obs[col].to_numpy(dtype=np.float64),
        )
    added_var = [c for c in written.var.columns if c not in qc_adata.var.columns]
    assert set(var_df.columns) == set(added_var)


# ---------------------------------------------------------------------------
# Presentation-ordered gene axis, on the matrix the op actually reads
# ---------------------------------------------------------------------------


def _ordered_fixture(tmp_dir):
    """Three genes with distinguishable values, so a mislabel is visible.

    g0 = [1, 2], g1 = [5, 6], g2 = [10, 20]. Requesting `["g2", "g0"]` with
    `preserve_var_order=True` makes `adata.var` request-ordered while the
    streaming kernels emit the sorted projection, so reading position 0 as `g2`
    is only correct if the op refuses or honours the permutation.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    dense = np.array([[1.0, 5.0, 10.0], [2.0, 6.0, 20.0]], dtype=np.float32)
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    adata.layers["counts"] = sp.csr_matrix(dense)
    path = str(tmp_dir / "presentation_order.scx")
    pyscx.from_anndata(adata, path)
    return path


def _layer_ops():
    """Every op that resolves `layer=` and streams the result.

    Parametrized because the guard has four call sites and one shared helper:
    pinning only the helper leaves a removed or misplaced call site green, which
    is exactly how the hole reached review in the first place.
    """
    import pyscx

    return [
        pytest.param(
            lambda a: pyscx.accel.calculate_qc_metrics(a, layer="counts"),
            id="calculate_qc_metrics",
        ),
        pytest.param(
            lambda a: pyscx.accel.score_genes(
                a, ["g2"], method="mean", layer="counts", device="cpu"
            ),
            id="score_genes",
        ),
        pytest.param(
            lambda a: pyscx.accel.highly_variable_genes(
                a, n_top_genes=1, flavor="seurat_v3", layer="counts", device="cpu"
            ),
            id="highly_variable_genes",
        ),
        pytest.param(
            lambda a: pyscx.accel.pflog(a, layer="counts", store="baseline"),
            id="pflog",
        ),
    ]


@pytest.mark.parametrize("op", _layer_ops())
def test_a_named_layer_cannot_smuggle_a_presentation_order_past_the_guard(tmp_dir, op):
    """The guard reads `adata.X`; a named layer is a different matrix.

    Materialising `X` disarms the `adata.X`-only check while the layer stays
    presentation-ordered, and the streaming kernel then labels the sorted
    projection with a request-ordered `adata.var`: asking for `g2` returned
    `g0`'s values under `g2`'s name. Silently wrong genes, where the same
    request on `X` had always raised.

    `score_genes` is the original reproduction; the other three share the call
    site, and each is asserted separately so removing any one of them is red.
    """
    import pyscx

    path = _ordered_fixture(tmp_dir)
    adata = pyscx.open(path).to_anndata(
        backed=True, var_names=["g2", "g0"], preserve_var_order=True
    )
    assert list(adata.var_names) == ["g2", "g0"], "premise: request order kept"
    adata.X = adata.X.to_memory()  # disarms the adata.X-only guard

    with pytest.raises(RuntimeError, match=r"caller-requested order"):
        op(adata)


def test_a_layer_handle_assigned_to_x_is_caught_too(tmp_dir):
    """`adata.X = adata.layers["counts"]` — a backed layer handle *as* `X`.

    `cast::<ScxBackedSparseDataset>()` does not match `ScxBackedLayerDataset`,
    so the presentation permutation on the layer's inner dataset was invisible
    to the guard.
    """
    import pyscx

    path = _ordered_fixture(tmp_dir)
    adata = pyscx.open(path).to_anndata(
        backed=True, var_names=["g2", "g0"], preserve_var_order=True
    )
    adata.X = adata.layers["counts"]

    with pytest.raises(RuntimeError, match=r"caller-requested order"):
        pyscx.accel.calculate_qc_metrics(adata)


def test_a_sorted_layer_request_is_still_allowed(tmp_dir):
    """The accept side: a sorted projection is not a presentation order.

    Without this the guard could reject everything and both tests above would
    still pass.
    """
    import pyscx

    path = _ordered_fixture(tmp_dir)
    adata = pyscx.open(path).to_anndata(backed=True, var_names=["g0", "g2"])
    assert list(adata.var_names) == ["g0", "g2"], "premise: sorted request"
    pyscx.accel.calculate_qc_metrics(adata, layer="counts")
    np.testing.assert_allclose(
        adata.obs["total_counts"].to_numpy(), [1.0 + 10.0, 2.0 + 20.0]
    )


def test_csc_rejects_a_layer_handle_assigned_to_x(qc_path):
    """The `BackedLayer` half of the CSC reject, which `layer=` does not cover.

    Reached with no `layer=` kwarg at all, so the message must not advise
    passing `layer=None` — the caller already is.
    """
    import pyscx

    backed = _backed(qc_path)
    backed.X = backed.layers["counts"]
    with pytest.raises(RuntimeError, match=r"(?i)layer source") as excinfo:
        pyscx.accel.calculate_qc_metrics(backed, prefer_format="csc")
    assert "layer=None" not in str(excinfo.value), (
        "the advice must not name a kwarg the caller is already passing"
    )


def test_a_dask_x_is_refused_rather_than_silently_computed():
    """`owned_csr` would `compute()` the whole thing before any QC number.

    The removed scanpy delegation covered "neither of our two SCX handles", so a
    dask array used to reach scanpy's own dask arm and stay bounded. Unifying
    the kernel took that away: measured, pyscx computed the entire array where
    scanpy had done per-axis reductions, and the `nnz x 8` large-copy warning
    cannot help because it is sized from the CSR that materialization produces.
    Refusing is the honest contract.
    """
    import anndata
    import pandas as pd

    import pyscx

    da = pytest.importorskip("dask.array")

    dense = np.array([[1.0, 5.0, 10.0], [2.0, 6.0, 20.0]], dtype=np.float32)
    adata = anndata.AnnData(
        X=da.from_array(dense, chunks=(1, 3)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    with pytest.raises(RuntimeError, match=r"(?i)materialize"):
        pyscx.accel.calculate_qc_metrics(adata)

    # The accept side, so the guard cannot pass by refusing everything.
    import scipy.sparse as sp

    ok = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    pyscx.accel.calculate_qc_metrics(ok)
    assert "total_counts" in ok.obs
    dense_ok = anndata.AnnData(
        X=dense.copy(),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    pyscx.accel.calculate_qc_metrics(dense_ok)
    assert "total_counts" in dense_ok.obs
