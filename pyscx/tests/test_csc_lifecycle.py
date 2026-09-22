"""CSC lifecycle: convert, info, mutating ops.

Walks the user-visible end-to-end CSC story:

1. Convert AnnData → SCX with `csc="always"` and inspect via the
   on-disk file (CSC catalog entries present, has_csc flag set).
2. Mutating ops (append/compact via the underlying scx-ops API)
   drop the CSC sidecar by default.
3. Round-tripping through `pyscx.open` + `to_anndata(backed=True)`
   preserves CSC.

The CLI surface (`scx info`, `scx append --rebuild-csc`) lives in
scx-cli integration tests; this file exercises the same lifecycle
through the pyscx Python API.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(31)
    mat = sp.random(30, 12, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 30).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(30)]
    adata.var["gene_id"] = [f"g{i}" for i in range(12)]
    return adata


def _csc_count_on_disk(path):
    """Open the file with the low-level reader to inspect CSC state.
    Avoids `pyscx.open()` which may apply caching."""
    import pyscx

    exp = pyscx.open(str(path))
    return exp.csc_shard_count if hasattr(exp, "csc_shard_count") else None


# ---------------------------------------------------------------------------
# Convert with csc="always" produces a CSC sidecar.
# ---------------------------------------------------------------------------


def test_from_anndata_csc_always_writes_sidecar(small_adata, tmp_path):
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)

    # Round-trip via to_anndata(backed=True): the CSC sidecar is
    # transparent at the user level (X stays CSR), but `prefer_format
    # ="csc"` works.
    adata = pyscx.open(str(path)).to_anndata(backed=True)

    csr_sums = pyscx.accel.col_sums(adata.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    np.testing.assert_allclose(csr_sums, csc_sums, atol=1e-9)


def test_from_anndata_csc_off_no_sidecar(small_adata, tmp_path):
    """`csc="off"` (the default) leaves CSC requests reaching the
    `as_column_source()` gate and raising — there's no sidecar."""
    import pyscx

    path = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(path))  # csc defaults to "off"

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.col_sums(adata.X, prefer_format="csc")


# ---------------------------------------------------------------------------
# Different csc_cols_per_shard values produce the requested layout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cols_per_shard,expected_n_shards", [(3, 4), (5, 3), (12, 1)])
def test_from_anndata_csc_cols_per_shard(small_adata, tmp_path, cols_per_shard, expected_n_shards):
    import pyscx

    path = tmp_path / f"csc_{cols_per_shard}.scx"
    pyscx.from_anndata(
        small_adata, str(path), csc="always", csc_cols_per_shard=cols_per_shard
    )

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    # CSC dispatch works regardless of shard layout (parity); we
    # don't assert the exact n_shards from Python (no public API
    # for it), but verify dispatch is functional.
    csr_sums = pyscx.accel.col_sums(adata.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    np.testing.assert_allclose(csr_sums, csc_sums, atol=1e-9)
    # Quiet the `expected_n_shards` parameter from breaking; this is
    # documented in the test but not asserted (no Python accessor).
    _ = expected_n_shards


# ---------------------------------------------------------------------------
# CSC + log1p chain: round-trip through the lazy wrapper preserves
# CSC dispatch capability.
# ---------------------------------------------------------------------------


def test_csc_survives_log1p_lazy(small_adata, tmp_path):
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    pyscx.accel.log1p(adata)
    # Lazy wrapper with Log1p — CSC capability gate is open.
    sums_csc = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    materialised = adata.X[:].toarray() if sp.issparse(adata.X[:]) else np.asarray(adata.X[:])
    np.testing.assert_allclose(
        sums_csc, materialised.sum(axis=0).astype(np.float64), atol=1e-5
    )


# ---------------------------------------------------------------------------
# CSC + row-deletion vector: capability gate must close.
# ---------------------------------------------------------------------------


def test_csc_survives_filter_cells(small_adata, tmp_path):
    """A row filter keeps the CSC route, and answers over the live rows.

    This asserted a `RuntimeError` until the CSC read path learned to renumber
    a slab's rows onto the live row space. The refusal was correct while it
    could not: CSC `indices` are global physical rows and a filter renumbers
    the live ones, so the slab and the caller disagreed about what row 3 meant.
    What matters now is not that the call succeeds but that it answers the
    *visible* matrix — a compaction that silently dropped the wrong rows would
    also "succeed".
    """
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    # The threshold has to actually drop one: a filter that keeps every cell is
    # a no-op (see `test_no_op_filter_cells_keeps_csc`) and would leave this
    # passing for the wrong reason.
    dense_all = np.asarray(small_adata.X.todense(), dtype=np.float64)
    threshold = float(np.median(dense_all.sum(axis=1)))
    pyscx.accel.filter_cells(adata, min_counts=threshold)
    assert adata.n_obs < small_adata.n_obs, "fixture must actually drop cells"

    keep = dense_all.sum(axis=1) >= threshold
    assert int(keep.sum()) == adata.n_obs, "the keep mask must match the handle"
    np.testing.assert_allclose(
        np.asarray(pyscx.accel.col_sums(adata.X, prefer_format="csc")),
        dense_all[keep].sum(axis=0),
        rtol=1e-6,
    )


def test_no_op_filter_cells_keeps_csc(small_adata, tmp_path):
    """A filter that drops nothing must not cost anything.

    `filter_cells` used to install a `kept_to_global` unconditionally, and any
    `kept_to_global` — even the identity one — closed the CSC capability gate,
    so a threshold every cell cleared permanently downgraded the file's
    `gpu_csc_v3` CSC-direct DE route with nothing in the data to explain why.
    A row filter no longer closes that gate, so what an identity one would
    cost now is smaller but not nothing: a row-compaction pass over every CSC
    slab that cannot drop a single entry. Same fix, same test.
    """
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    before = pyscx.accel.col_sums(adata.X, prefer_format="csc")

    # Every cell has at least one count, so this keeps all 30.
    pyscx.accel.filter_cells(adata, min_counts=1)
    assert adata.n_obs == small_adata.n_obs

    np.testing.assert_allclose(
        pyscx.accel.col_sums(adata.X, prefer_format="csc"), before, rtol=0
    )


# ---------------------------------------------------------------------------
# Standalone pyscx.build_csc() — the `scx build-csc` CLI mirror: add a CSC
# sidecar to an existing CSR-only file (input -> output).
# ---------------------------------------------------------------------------


def test_build_csc_adds_sidecar_to_csr_only_file(small_adata, tmp_path):
    import pyscx

    src = tmp_path / "csr_only.scx"
    out = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(src))  # csc defaults to "off"

    # Source has no CSC sidecar.
    assert pyscx.open(str(src)).has_csc is False

    pyscx.build_csc(str(src), str(out), csc_cols_per_shard=4)

    # Output gained a CSC sidecar, and the source is untouched.
    assert pyscx.open(str(out)).has_csc is True
    assert pyscx.open(str(src)).has_csc is False

    # Functional parity: CSC-direct col_sums matches the CSR path.
    adata = pyscx.open(str(out)).to_anndata(backed=True)
    csr_sums = pyscx.accel.col_sums(adata.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    np.testing.assert_allclose(csr_sums, csc_sums, atol=1e-9)


def test_build_csc_force_overwrite(small_adata, tmp_path):
    import pyscx

    src = tmp_path / "csr_only.scx"
    out = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(src))
    pyscx.build_csc(str(src), str(out), csc_cols_per_shard=4)

    # Re-building onto an existing output without force must fail.
    with pytest.raises(RuntimeError):
        pyscx.build_csc(str(src), str(out), csc_cols_per_shard=4)

    # force=True succeeds and the sidecar is still present.
    pyscx.build_csc(str(src), str(out), force=True, csc_cols_per_shard=4)
    assert pyscx.open(str(out)).has_csc is True


def test_build_csc_rejects_same_input_output(small_adata, tmp_path):
    """input == output is rejected up front (run_build_csc would delete the
    source before reopening it)."""
    import pyscx

    src = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(src))

    with pytest.raises(ValueError, match="different files"):
        pyscx.build_csc(str(src), str(src), force=True)

    # The error must point at the in-place spelling that now exists, not at
    # `sort(rebuild_csc=True)` (which is what it said before F9).
    with pytest.raises(ValueError, match="output=None"):
        pyscx.build_csc(str(src), str(src), force=True)


def test_build_csc_in_place_default(small_adata, tmp_path):
    """F9: `output=None` (the default) adds the sidecar to `input` in place.

    The CSC store is a sidecar *on* a file everywhere it is documented, and the
    neighbouring in-place ops (`obs_import` / `doublet_import` /
    `cellbender_import`) all mutate — so requiring a distinct `output` read as a
    missing argument rather than a design choice.
    """
    import pyscx

    src = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(src))
    # Anti-vacuous: no sidecar to begin with.
    assert pyscx.open(str(src)).has_csc is False

    before = pyscx.accel.col_sums(
        pyscx.open(str(src)).to_anndata(backed=True).X, prefer_format="csr"
    )

    pyscx.build_csc(str(src), csc_cols_per_shard=4)

    # Same path now carries the sidecar, and the axes are unchanged.
    exp = pyscx.open(str(src))
    assert exp.has_csc is True
    assert (exp.n_obs, exp.n_vars) == (small_adata.n_obs, small_adata.n_vars)

    # No staging file left beside the target.
    assert not list(tmp_path.glob("*.tmp")), "in-place build_csc leaked a temp file"

    # CSC-direct now works on the rebuilt file and agrees with the CSR path.
    adata = exp.to_anndata(backed=True)
    np.testing.assert_allclose(
        pyscx.accel.col_sums(adata.X, prefer_format="csc"), before, atol=1e-9
    )


def test_build_csc_in_place_is_an_append_that_rollback_undoes(small_adata, tmp_path):
    """In-place `build_csc` appends the sidecar through the manifest chain.

    Three consequences, each pinned: an `Experiment` already open on the file
    raises on its next read (the catalog moved; same inode, so this is the
    header-pointer half of the staleness check), `reload()` sees the sidecar,
    and `pyscx.rollback` removes it again with the CSR untouched. The last used
    to be impossible — the build staged a wholly new file and renamed it over
    the target, leaving no previous catalog.
    """
    import pyscx

    src = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(src))
    exp = pyscx.open(str(src))
    assert exp.has_csc is False
    before = exp.to_anndata().X.toarray()
    inode = src.stat().st_ino

    pyscx.build_csc(str(src), csc_cols_per_shard=4)
    assert src.stat().st_ino == inode, "an append, not a rename"

    with pytest.raises(RuntimeError, match=r"changed on disk|replaced on disk"):
        exp.read_obs()
    exp.reload()
    assert exp.has_csc is True

    pyscx.rollback(str(src))
    back = pyscx.open(str(src))
    assert back.has_csc is False
    np.testing.assert_array_equal(back.to_anndata().X.toarray(), before)


def test_build_csc_in_place_rejects_force(small_adata, tmp_path):
    """`force` overwrites an `output`; there is none in the in-place form.

    Refused rather than ignored — accepting it silently would imply a guard
    that does not exist.
    """
    import pyscx

    src = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(src))

    with pytest.raises(ValueError, match="force"):
        pyscx.build_csc(str(src), force=True)

    # A rejected call must not have half-run.
    assert pyscx.open(str(src)).has_csc is False


def test_build_csc_rejects_multimodal(tmp_path):
    """build_csc is not modality-aware; a multimodal input is rejected."""
    import anndata as ad
    import pyscx

    pytest.importorskip("mudata")
    import mudata

    rng = np.random.default_rng(5)
    rna = ad.AnnData(X=sp.csr_matrix(rng.poisson(0.4, (16, 12)).astype(np.float32)))
    rna.var_names = [f"g{i}" for i in range(12)]
    adt = ad.AnnData(X=sp.csr_matrix(rng.poisson(0.4, (16, 4)).astype(np.float32)))
    adt.var_names = [f"a{i}" for i in range(4)]
    mu = mudata.MuData({"rna": rna, "adt": adt})
    mu.obs_names = [f"c{i}" for i in range(16)]

    src = tmp_path / "cite.scx"
    out = tmp_path / "cite_csc.scx"
    pyscx.from_mudata(mu, str(src))

    with pytest.raises(ValueError, match="multimodal"):
        pyscx.build_csc(str(src), str(out))


def test_build_csc_rejects_bad_memory_limit(small_adata, tmp_path):
    import pyscx

    src = tmp_path / "csr_only.scx"
    out = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(src))

    # Decimal "GB" is rejected as ambiguous by the size parser -> ValueError.
    with pytest.raises(ValueError):
        pyscx.build_csc(str(src), str(out), memory_limit="4GB")
