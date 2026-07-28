"""Tests for `pyscx.to_h5ad(obs_mask=..., min_counts=...)`.

The row filter exists so a raw all-droplet SCX file can be trimmed on its way
to CellBender without materializing the matrix. Two things make it easy to get
subtly wrong, and both are pinned here:

1. **Coordinate system.** The mask is indexed in the *global / physical* obs
   row space (header ``n_obs``), not the post-deletion live space. A mask built
   from a backed ``X.sum(1)`` on a file with deletions is in the wrong space,
   and silently so if we only checked ``len(mask) <= n_obs``.
2. **Composition with deletion vectors.** The mask is ANDed with the deletion
   mask, never substituted for it — a ``True`` entry must not resurrect a
   logically deleted row.
"""

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")


def _n_obs_written(path):
    """Rows in the exported h5ad, straight from the file."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r") as f:
        return f["X"].attrs["shape"][0]


def _x_dense(path):
    return anndata.read_h5ad(path).X.toarray()


@pytest.fixture
def scx_path(synthetic_adata, scx_from_adata):
    return scx_from_adata(synthetic_adata, "masked.scx")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_obs_mask_filters_every_obs_axis_section(
    synthetic_adata, scx_path, tmp_dir
):
    n_obs = synthetic_adata.n_obs
    mask = np.arange(n_obs) % 2 == 0
    out = tmp_dir / "even.h5ad"

    pyscx.to_h5ad(scx_path, out, obs_mask=mask)

    got = anndata.read_h5ad(out)
    assert got.n_obs == int(mask.sum())
    # obsm and layers must track /X, or anndata refuses to open the file at all.
    assert got.obsm["X_pca"].shape[0] == got.n_obs
    assert got.layers["raw"].shape[0] == got.n_obs

    expected = synthetic_adata[mask]
    np.testing.assert_array_equal(got.obs_names.to_numpy(), expected.obs_names.to_numpy())
    np.testing.assert_allclose(got.X.toarray(), expected.X.toarray())


def test_min_counts_equals_the_equivalent_explicit_mask(
    synthetic_adata, scx_path, tmp_dir
):
    counts = np.asarray(synthetic_adata.X.sum(axis=1)).ravel()
    threshold = float(np.median(counts))
    expected_mask = counts >= threshold
    assert 0 < expected_mask.sum() < synthetic_adata.n_obs

    by_counts = tmp_dir / "counts.h5ad"
    by_mask = tmp_dir / "mask.h5ad"
    pyscx.to_h5ad(scx_path, by_counts, min_counts=threshold)
    pyscx.to_h5ad(scx_path, by_mask, obs_mask=expected_mask)

    assert _n_obs_written(by_counts) == int(expected_mask.sum())
    np.testing.assert_allclose(_x_dense(by_counts), _x_dense(by_mask))


def test_min_counts_and_obs_mask_intersect(synthetic_adata, scx_path, tmp_dir):
    counts = np.asarray(synthetic_adata.X.sum(axis=1)).ravel()
    threshold = float(np.median(counts))
    user_mask = np.arange(synthetic_adata.n_obs) % 2 == 0
    expected = int((user_mask & (counts >= threshold)).sum())

    out = tmp_dir / "both.h5ad"
    pyscx.to_h5ad(scx_path, out, obs_mask=user_mask, min_counts=threshold)
    assert _n_obs_written(out) == expected


def test_min_counts_zero_is_a_no_op(synthetic_adata, scx_path, tmp_dir):
    out = tmp_dir / "all.h5ad"
    pyscx.to_h5ad(scx_path, out, min_counts=0)
    assert _n_obs_written(out) == synthetic_adata.n_obs


# ---------------------------------------------------------------------------
# The deletion-vector interaction — the headline regression
# ---------------------------------------------------------------------------


def test_obs_mask_uses_physical_coordinates_and_never_resurrects_deleted_rows(
    synthetic_adata, scx_from_adata, tmp_dir
):
    path = scx_from_adata(synthetic_adata, "deleted.scx")
    n_obs = synthetic_adata.n_obs

    delete = np.zeros(n_obs, dtype=bool)
    delete[:3] = True
    exp = pyscx.open(path)
    exp.mark_deleted(delete)

    exp = pyscx.open(path)
    assert exp.n_obs_physical == n_obs
    assert exp.n_obs == n_obs - 3, "live count drops, physical count does not"

    # All-true except row 5. Rows 0-2 are True here but deleted.
    mask = np.ones(n_obs, dtype=bool)
    mask[5] = False

    out = tmp_dir / "out.h5ad"
    with pytest.warns(UserWarning, match="logically-deleted"):
        pyscx.to_h5ad(path, out, obs_mask=mask)

    got = anndata.read_h5ad(out)
    assert got.n_obs == n_obs - 4, "3 deleted + 1 masked out"
    kept = synthetic_adata.obs_names.to_numpy()[~delete & mask]
    np.testing.assert_array_equal(got.obs_names.to_numpy(), kept)


def test_obs_mask_in_live_coordinates_is_rejected(
    synthetic_adata, scx_from_adata, tmp_dir
):
    """A mask sized to the *live* row count is the natural mistake; it must
    raise rather than silently filtering the wrong rows."""
    path = scx_from_adata(synthetic_adata, "deleted2.scx")
    n_obs = synthetic_adata.n_obs

    delete = np.zeros(n_obs, dtype=bool)
    delete[:3] = True
    pyscx.open(path).mark_deleted(delete)

    live_sized = np.ones(n_obs - 3, dtype=bool)
    with pytest.raises(ValueError, match="physical n_obs"):
        pyscx.to_h5ad(path, tmp_dir / "out.h5ad", obs_mask=live_sized)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [-1, 1])
def test_wrong_length_mask_raises_value_error(synthetic_adata, scx_path, tmp_dir, delta):
    bad = np.ones(synthetic_adata.n_obs + delta, dtype=bool)
    with pytest.raises(ValueError) as exc:
        pyscx.to_h5ad(scx_path, tmp_dir / "out.h5ad", obs_mask=bad)
    assert "n_obs_physical" in str(exc.value)


@pytest.mark.parametrize(
    "bad",
    [
        np.arange(100, dtype=np.int64),
        np.ones(100, dtype=np.float32),
        np.array(["a"] * 100, dtype=object),
    ],
    ids=["int", "float", "object"],
)
def test_non_bool_dtype_raises_rather_than_coercing(scx_path, tmp_dir, bad):
    """An int array silently becoming `!= 0` is exactly the quiet wrong answer
    a row filter must never produce."""
    with pytest.raises(TypeError, match="boolean array"):
        pyscx.to_h5ad(scx_path, tmp_dir / "out.h5ad", obs_mask=bad)


def test_multidimensional_mask_raises(scx_path, tmp_dir):
    with pytest.raises(ValueError, match="1-D"):
        pyscx.to_h5ad(
            scx_path, tmp_dir / "out.h5ad", obs_mask=np.ones((100, 2), dtype=bool)
        )


def test_all_false_mask_raises(synthetic_adata, scx_path, tmp_dir):
    bad = np.zeros(synthetic_adata.n_obs, dtype=bool)
    with pytest.raises(RuntimeError, match="keeps zero"):
        pyscx.to_h5ad(scx_path, tmp_dir / "out.h5ad", obs_mask=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_min_counts_raises(scx_path, tmp_dir, bad):
    with pytest.raises(ValueError, match="min_counts"):
        pyscx.to_h5ad(scx_path, tmp_dir / "out.h5ad", min_counts=bad)


def test_row_filter_requires_streaming(synthetic_adata, scx_path, tmp_dir):
    mask = np.ones(synthetic_adata.n_obs, dtype=bool)
    with pytest.raises(ValueError, match="stream=True"):
        pyscx.to_h5ad(scx_path, tmp_dir / "a.h5ad", obs_mask=mask, stream=False)
    with pytest.raises(ValueError, match="stream=True"):
        pyscx.to_h5ad(scx_path, tmp_dir / "b.h5ad", min_counts=1, stream=False)


# ---------------------------------------------------------------------------
# Accepted mask spellings
# ---------------------------------------------------------------------------


def test_accepts_list_series_and_strided_views(synthetic_adata, scx_path, tmp_dir):
    n_obs = synthetic_adata.n_obs
    base = np.arange(n_obs) % 2 == 0
    expected = int(base.sum())

    spellings = {"list": base.tolist(), "ndarray": base}

    pd = pytest.importorskip("pandas")
    spellings["series"] = pd.Series(base)

    # A strided view: `as_slice()` on the Rust side requires contiguity, so the
    # wrapper must copy.
    wide = np.zeros((n_obs, 2), dtype=bool)
    wide[:, 0] = base
    spellings["strided"] = wide[:, 0]

    for name, mask in spellings.items():
        out = tmp_dir / f"{name}.h5ad"
        pyscx.to_h5ad(scx_path, out, obs_mask=mask)
        assert _n_obs_written(out) == expected, name


# ---------------------------------------------------------------------------
# Export provenance
# ---------------------------------------------------------------------------


def test_filtered_export_records_uns_scx_export(synthetic_adata, scx_path, tmp_dir):
    counts = np.asarray(synthetic_adata.X.sum(axis=1)).ravel()
    threshold = float(np.median(counts))
    out = tmp_dir / "out.h5ad"
    pyscx.to_h5ad(scx_path, out, min_counts=threshold)

    note = anndata.read_h5ad(out).uns["scx_export"]
    assert note["n_obs_source"] == synthetic_adata.n_obs
    assert note["n_obs_written"] == _n_obs_written(out)
    assert note["filter"] == "min_counts"
    assert note["min_counts"] == pytest.approx(threshold)
    assert note["tool"] == "pyscx"


def test_unfiltered_export_records_nothing(scx_path, tmp_dir):
    out = tmp_dir / "out.h5ad"
    pyscx.to_h5ad(scx_path, out)
    assert "scx_export" not in anndata.read_h5ad(out).uns
