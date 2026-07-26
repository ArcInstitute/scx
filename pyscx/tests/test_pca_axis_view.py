"""PCA must run on the backed handle's *view*, not the whole file.

`ScxBackedSparseDataset` is a lazy window: `kept_to_global` selects visible
rows, `col_projection` selects visible columns. `pyscx.accel.pca` used to hand
the streaming kernels the raw `BackedCsrReader` instead of that window, so a
subset backed `X` was silently ignored:

* after `filter_cells`, embeddings were computed for every on-disk row and
  written in ascending-global order against a shorter `obs` — each cell got
  another cell's coordinates;
* after a var subset, PCA ran over every on-disk gene;
* worst, a var subset **plus** `mask_var` (or an auto-consumed
  `var['highly_variable']`, i.e. the ordinary `filter_genes → hvg → pca`
  pipeline) resolved the mask against the *visible* axis and applied it to the
  *global* one — wholly the wrong genes, with every written shape
  self-consistent and so nothing to notice.

The lazy `X` path was always correct; it goes through
`ScxLazyTransformedDataset::as_shard_source()`. These tests pin the backed path
to the same behaviour, comparing against the materialized matrix in every case.

Multi-shard fixtures throughout: a single-shard file cannot exhibit a
row-window bug, which is how the original probe missed it.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

N_OBS = 120
N_VARS = 40
SHARD_SIZE = 25  # → 5 shards
N_COMPS = 4


def _sign_align(ref, other):
    """PCA components are sign-ambiguous; align `other` to `ref`'s signs."""
    out = np.array(other, dtype=np.float64, copy=True)
    for j in range(ref.shape[1]):
        if np.dot(ref[:, j], other[:, j]) < 0:
            out[:, j] = -out[:, j]
    return out


def _assert_same_embedding(got, ref, atol=1e-3):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert got.shape == ref.shape
    np.testing.assert_allclose(_sign_align(ref, got), ref, atol=atol)


@pytest.fixture(scope="module")
def dense_counts():
    rng = np.random.default_rng(0)
    x = sp.random(
        N_OBS, N_VARS, density=0.4, format="csr", dtype=np.float32, random_state=rng
    )
    x.data = (x.data * 20).astype(np.float32).round() + 1.0
    x.eliminate_zeros()
    return x


@pytest.fixture(scope="module")
def multishard_path(tmp_path_factory, dense_counts):
    """A multi-shard SCX file. The row-window bugs are invisible on one shard."""
    import anndata as ad

    adata = ad.AnnData(X=dense_counts.copy())
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.var_names = [f"g{i}" for i in range(N_VARS)]
    adata.obs["cell_id"] = list(adata.obs_names)
    adata.var["gene_id"] = list(adata.var_names)
    path = tmp_path_factory.mktemp("pca_axis_view") / "counts.scx"
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


def _backed(multishard_path):
    return pyscx.open(str(multishard_path)).to_anndata(backed=True)


def _in_memory(matrix, **pca_kwargs):
    """Ground truth: PCA over an ordinary in-memory AnnData."""
    import anndata as ad

    ref = ad.AnnData(X=matrix.copy())
    pyscx.accel.pca(ref, n_comps=N_COMPS, device="cpu", **pca_kwargs)
    return ref


# ---------------------------------------------------------------------------
# Row axis — `kept_to_global`
# ---------------------------------------------------------------------------


def test_filter_cells_then_pca_matches_the_kept_rows(multishard_path, dense_counts):
    """Embeddings must describe the kept cells, not global rows 0..n_kept."""
    adata = _backed(multishard_path)
    row_sums = np.asarray(adata.X.to_memory().sum(axis=1)).ravel()
    threshold = float(np.median(row_sums))
    pyscx.accel.filter_cells(adata, min_counts=threshold)

    kept = np.flatnonzero(row_sums >= threshold)
    assert 0 < adata.n_obs < N_OBS, "fixture must actually drop cells"
    assert adata.n_obs == kept.size
    assert type(adata.X).__name__ == "ScxBackedSparseDataset", (
        "this test is about the backed path; a lazy X would take the "
        "already-correct branch"
    )

    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")

    emb = np.asarray(adata.obsm["X_pca"])
    assert emb.shape == (adata.n_obs, N_COMPS)
    _assert_same_embedding(emb, _in_memory(dense_counts[kept, :]).obsm["X_pca"])


def test_row_subset_embedding_is_not_the_leading_global_rows(
    multishard_path, dense_counts
):
    """The specific misattribution: row i of X_pca must be cell kept[i].

    Streaming the raw reader produced an embedding for global row i. Where the
    kept set is not a prefix, that lands each cell's coordinates on a different
    cell — this asserts the coordinates are *not* the ones the leading global
    rows would have produced.
    """
    adata = _backed(multishard_path)
    row_sums = np.asarray(adata.X.to_memory().sum(axis=1)).ravel()
    threshold = float(np.median(row_sums))
    pyscx.accel.filter_cells(adata, min_counts=threshold)
    kept = np.flatnonzero(row_sums >= threshold)
    assert not np.array_equal(kept, np.arange(kept.size)), (
        "fixture's kept set must not be a prefix — the two row orders would "
        "coincide and the assertion below could not distinguish them"
    )

    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
    emb = np.asarray(adata.obsm["X_pca"], dtype=np.float64)

    prefix = np.asarray(
        _in_memory(dense_counts[: kept.size, :]).obsm["X_pca"], dtype=np.float64
    )
    assert not np.allclose(np.abs(emb), np.abs(prefix), atol=1e-3)


def test_subset_obs_then_pca_matches(multishard_path, dense_counts):
    """Same via `subset_obs`, whose selection is an arbitrary ascending set."""
    adata = _backed(multishard_path)
    keep = np.zeros(N_OBS, dtype=bool)
    keep[np.arange(3, N_OBS, 3)] = True  # spans every shard, not a prefix
    pyscx.accel.subset_obs(adata, keep)
    assert adata.n_obs == int(keep.sum())

    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
    _assert_same_embedding(
        adata.obsm["X_pca"], _in_memory(dense_counts[keep, :]).obsm["X_pca"]
    )


# ---------------------------------------------------------------------------
# Column axis — `col_projection`
# ---------------------------------------------------------------------------


def _var_subset_mask():
    keep = np.zeros(N_VARS, dtype=bool)
    keep[np.arange(0, N_VARS, 2)] = True  # ascending, 20 of 40
    return keep


def test_var_subset_then_pca_uses_only_the_kept_genes(multishard_path, dense_counts):
    adata = _backed(multishard_path)
    keep = _var_subset_mask()
    pyscx.accel.subset_var(adata, keep)
    assert adata.n_vars == int(keep.sum())

    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")

    pcs = np.asarray(adata.varm["PCs"])
    assert pcs.shape == (adata.n_vars, N_COMPS)
    _assert_same_embedding(
        adata.obsm["X_pca"], _in_memory(dense_counts[:, keep]).obsm["X_pca"]
    )


def test_var_subset_plus_mask_var_resolves_in_the_visible_axis(
    multishard_path, dense_counts
):
    """The headline silent case.

    `mask_var` is resolved against `adata.n_vars` — the *visible* axis — so it
    has to be applied to a source whose columns are already the visible ones.
    Applied to the raw reader's global axis instead, visible index j selected
    on-disk gene j and PCA ran on entirely the wrong genes, with `X_pca` and
    `varm['PCs']` both correctly shaped.
    """
    adata = _backed(multishard_path)
    keep = _var_subset_mask()
    pyscx.accel.subset_var(adata, keep)

    hv = np.zeros(adata.n_vars, dtype=bool)
    hv[:10] = True

    pyscx.accel.pca(adata, n_comps=N_COMPS, mask_var=hv, device="cpu")

    # Ground truth: the same ten genes, selected in the visible space.
    ref = _in_memory(dense_counts[:, keep], mask_var=hv)
    _assert_same_embedding(adata.obsm["X_pca"], ref.obsm["X_pca"])
    np.testing.assert_allclose(
        np.asarray(adata.uns["pca"]["variance_ratio"]),
        np.asarray(ref.uns["pca"]["variance_ratio"]),
        rtol=1e-3,
        atol=1e-6,
    )

    # And explicitly *not* the wrong-axis answer: on-disk genes 0..9.
    wrong_axis = np.asarray(
        _in_memory(dense_counts[:, :10]).obsm["X_pca"], dtype=np.float64
    )
    assert not np.allclose(
        np.abs(np.asarray(adata.obsm["X_pca"], dtype=np.float64)),
        np.abs(wrong_axis),
        atol=1e-3,
    )


def test_var_subset_plus_auto_highly_variable(multishard_path, dense_counts):
    """`mask_var=None` auto-consumes `var['highly_variable']` — same axis rule.

    This is the shape of the ordinary pipeline (`filter_genes` → HVG → `pca`),
    which is why the bug reached real workflows.
    """
    adata = _backed(multishard_path)
    keep = _var_subset_mask()
    pyscx.accel.subset_var(adata, keep)

    hv = np.zeros(adata.n_vars, dtype=bool)
    hv[:10] = True
    adata.var["highly_variable"] = hv

    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
    assert bool(adata.uns["pca"]["params"]["use_highly_variable"]) is True

    _assert_same_embedding(
        adata.obsm["X_pca"], _in_memory(dense_counts[:, keep], mask_var=hv).obsm["X_pca"]
    )


def test_composed_obs_and_var_subset(multishard_path, dense_counts):
    adata = _backed(multishard_path)
    rows = np.zeros(N_OBS, dtype=bool)
    rows[np.arange(1, N_OBS, 2)] = True
    cols = _var_subset_mask()

    pyscx.accel.subset_obs(adata, rows)
    pyscx.accel.subset_var(adata, cols)
    assert (adata.n_obs, adata.n_vars) == (int(rows.sum()), int(cols.sum()))

    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
    _assert_same_embedding(
        adata.obsm["X_pca"],
        _in_memory(dense_counts[rows, :][:, cols]).obsm["X_pca"],
    )


# ---------------------------------------------------------------------------
# `col_presentation` — not representable as a ShardSource, so reject
# ---------------------------------------------------------------------------


def test_preserve_var_order_is_rejected(multishard_path):
    """A request-ordered gene axis has no `ShardSource` spelling.

    The source emits columns in sorted on-disk order while `adata.var` stays in
    request order, so `varm['PCs']` would misalign against the gene names.
    Six other streaming accel ops already refuse this; `pca` now does too.
    """
    exp = pyscx.open(str(multishard_path))
    names = [f"g{i}" for i in (9, 2, 30, 5)]  # deliberately unsorted
    adata = exp.to_anndata(backed=True, var_names=names, preserve_var_order=True)
    assert list(adata.var_names) == names

    with pytest.raises(RuntimeError, match="preserve_var_order"):
        pyscx.accel.pca(adata, n_comps=2, device="cpu")


def test_sorted_var_subset_is_not_rejected(multishard_path, dense_counts):
    """The guard must not catch ordinary subsetting.

    `set_col_projection_ordered` leaves `col_presentation` unset when the
    permutation is the identity, and `_subset` only accepts ascending columns —
    so `adata[:, mask]` never trips the guard.
    """
    adata = _backed(multishard_path)
    keep = _var_subset_mask()
    pyscx.accel.subset_var(adata, keep)
    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")  # must not raise
    assert np.asarray(adata.varm["PCs"]).shape == (adata.n_vars, N_COMPS)


# ---------------------------------------------------------------------------
# Non-regression on the common path
# ---------------------------------------------------------------------------


def test_unsubset_backed_pca_is_unchanged(multishard_path, dense_counts):
    """Swapping the raw reader for the view source must not move the answer."""
    adata = _backed(multishard_path)
    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
    _assert_same_embedding(adata.obsm["X_pca"], _in_memory(dense_counts).obsm["X_pca"])


def test_unsubset_backed_pca_with_mask_var_is_unchanged(
    multishard_path, dense_counts
):
    adata = _backed(multishard_path)
    hv = np.zeros(N_VARS, dtype=bool)
    hv[np.arange(0, N_VARS, 3)] = True
    pyscx.accel.pca(adata, n_comps=N_COMPS, mask_var=hv, device="cpu")

    ref = _in_memory(dense_counts, mask_var=hv)
    _assert_same_embedding(adata.obsm["X_pca"], ref.obsm["X_pca"])
    # `PCs` stays on the full var axis with excluded genes zeroed (scanpy).
    pcs = np.asarray(adata.varm["PCs"])
    assert pcs.shape == (N_VARS, N_COMPS)
    assert np.allclose(pcs[~hv], 0.0)


def test_lazy_transformed_path_still_agrees(multishard_path, dense_counts):
    """The lazy branch was always correct; both branches must now agree.

    `normalize_total` / `log1p` turn `X` into a `ScxLazyTransformedDataset`,
    which is why the standard pipeline masked this bug for so long.
    """
    keep = _var_subset_mask()

    backed = _backed(multishard_path)
    pyscx.accel.subset_var(backed, keep)
    pyscx.accel.pca(backed, n_comps=N_COMPS, device="cpu")

    lazy = _backed(multishard_path)
    pyscx.accel.subset_var(lazy, keep)
    pyscx.accel.log1p(lazy, device="cpu")
    assert type(lazy.X).__name__ == "ScxLazyTransformedDataset"
    pyscx.accel.pca(lazy, n_comps=N_COMPS, device="cpu")

    # Different data (log1p), so compare each against its own reference rather
    # than against each other.
    _assert_same_embedding(
        backed.obsm["X_pca"], _in_memory(dense_counts[:, keep]).obsm["X_pca"]
    )
    logged = dense_counts[:, keep].copy()
    logged.data = np.log1p(logged.data)
    _assert_same_embedding(lazy.obsm["X_pca"], _in_memory(logged).obsm["X_pca"])


# ---------------------------------------------------------------------------
# GPU parity
#
# `pca`'s backed dispatch exists twice — a `#[cfg(feature = "gpu")]` branch and
# a CPU one — and both took the raw reader. A backed / lazy `X` is the
# out-of-VRAM moat, so it stays on the *native* GPU streaming path rather than
# routing to rapids-singlecell; these are the only tests that reach the GPU
# branch of the fix.
#
# Everything above pins `device="cpu"` deliberately: on a GPU host an
# *in-memory* reference AnnData routes to rapids-singlecell while a backed `X`
# stays native, so an unpinned comparison measures rapids-vs-native numerics
# rather than the view logic.
# ---------------------------------------------------------------------------


def _gpu_available():
    try:
        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


gpu_only = pytest.mark.skipif(
    not _gpu_available(), reason="CUDA GPU not available"
)


def _cosine_sign_agnostic(a, b):
    """Column-wise |cosine| between two (n, k) embeddings.

    The repo-standard GPU/CPU PCA comparison (see `test_accel_pca_gpu.py`).
    Elementwise `allclose` is the wrong tool here: the two paths run different
    algorithms — `pick_cpu_method` chooses covariance for a narrow var axis
    while `resolve_gpu_method` always yields randomized — so they agree on the
    subspace, not on the digits.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.abs(np.einsum("nk,nk->k", a, b))
    denom = np.clip(np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0), 1e-12, None)
    return num / denom


@gpu_only
def test_gpu_row_subset_matches_cpu(multishard_path):
    keep = np.zeros(N_OBS, dtype=bool)
    keep[np.arange(3, N_OBS, 3)] = True

    ref = _backed(multishard_path)
    pyscx.accel.subset_obs(ref, keep)
    pyscx.accel.pca(ref, n_comps=N_COMPS, device="cpu")

    adata = _backed(multishard_path)
    pyscx.accel.subset_obs(adata, keep)
    pyscx.accel.pca(adata, n_comps=N_COMPS, device="gpu")

    # The row axis is the whole point: one embedding row per *kept* cell.
    assert np.asarray(adata.obsm["X_pca"]).shape == (adata.n_obs, N_COMPS)
    cos = _cosine_sign_agnostic(ref.obsm["X_pca"], adata.obsm["X_pca"])
    assert (cos[:2] >= 0.99).all(), f"top-2 cosine: {cos[:2]}"


@gpu_only
def test_gpu_var_subset_plus_mask_var_matches_cpu(multishard_path, dense_counts):
    keep = _var_subset_mask()
    hv = np.zeros(int(keep.sum()), dtype=bool)
    hv[:10] = True

    ref = _backed(multishard_path)
    pyscx.accel.subset_var(ref, keep)
    pyscx.accel.pca(ref, n_comps=N_COMPS, mask_var=hv, device="cpu")

    adata = _backed(multishard_path)
    pyscx.accel.subset_var(adata, keep)
    pyscx.accel.pca(adata, n_comps=N_COMPS, mask_var=hv, device="gpu")

    assert np.asarray(adata.varm["PCs"]).shape == (adata.n_vars, N_COMPS)
    cos = _cosine_sign_agnostic(ref.obsm["X_pca"], adata.obsm["X_pca"])
    assert (cos[:2] >= 0.99).all(), f"top-2 cosine: {cos[:2]}"

    # Numerics aside, the GPU branch must have read the *visible* genes: the
    # wrong-axis answer (on-disk genes 0..9) is a different subspace entirely.
    wrong_axis = _in_memory(dense_counts[:, :10]).obsm["X_pca"]
    wrong = _cosine_sign_agnostic(wrong_axis, adata.obsm["X_pca"])
    assert wrong[0] < 0.99, f"GPU used the wrong gene axis (cos={wrong[0]:.4f})"


def test_fused_entry_points_reject_preserve_var_order(multishard_path):
    """`pca_neighbors` / `pca_neighbors_umap` guard before any device dispatch.

    The fused GPU path would otherwise skip the check `pca` applies on the
    delegating path, so the guard sits at the top of each entry point — which
    makes it reachable (and testable) without a GPU.
    """
    exp = pyscx.open(str(multishard_path))
    names = [f"g{i}" for i in (9, 2, 30, 5)]
    adata = exp.to_anndata(backed=True, var_names=names, preserve_var_order=True)
    for op in (
        pyscx.accel.pca,
        pyscx.accel.pca_neighbors,
        pyscx.accel.pca_neighbors_umap,
    ):
        with pytest.raises(RuntimeError, match="preserve_var_order"):
            op(adata, n_comps=2, device="cpu")


@gpu_only
def test_gpu_preserve_var_order_is_rejected(multishard_path):
    """Same guards on the GPU device selector."""
    exp = pyscx.open(str(multishard_path))
    names = [f"g{i}" for i in (9, 2, 30, 5)]
    adata = exp.to_anndata(backed=True, var_names=names, preserve_var_order=True)
    for op in (pyscx.accel.pca, pyscx.accel.pca_neighbors):
        with pytest.raises(RuntimeError, match="preserve_var_order"):
            op(adata, n_comps=2, device="gpu")
