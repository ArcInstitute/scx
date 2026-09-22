"""CSC-first GPU HVG parity + route tests for
`pyscx.accel.highly_variable_genes`.

Exercises the column-major CSC reduce route (`scx_gpu::
gpu_streaming_{mean_var,clip_square_sum}_csc`, recorded as `gpu_csc_v3`) on a
backed SCX file with a CSC sidecar:

* explicit `prefer_format="csc"` + `device="gpu"` → `gpu_csc_v3`;
* default `prefer_format="csr"` + `device="gpu"` auto-detects the sidecar and
  also routes to `gpu_csc_v3` (mirrors GPU DE);
* `device="cpu"` + `prefer_format="csc"` → `cpu_csc`;
* a CSR-only backed file + `device="gpu"` → `gpu_csr` (no regression).

The HVG gene set (and the seurat_v3 var stats) must match the CSR GPU path
within tolerance — the CSC reduce is one block per gene with no `atomicAdd`,
so it should be numerically at least as good as the CSR atomic path.

Skipped cleanly when `pyscx.accel.gpu_available()` is `False`. Intended to run
in its own process (the Chimera GPU test harness invokes GPU test files
isolated; set `SCX_DISABLE_CUDA_GRAPHS=1`).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import anndata as ad

import pyscx


pytestmark = pytest.mark.skipif(
    not pyscx.accel.gpu_available(),
    reason="GPU not available (pyscx not built with gpu feature, or no CUDA device)",
)


def _raw_counts_adata(n_obs: int = 400, n_vars: int = 80, seed: int = 0) -> ad.AnnData:
    """Synthetic raw-count CSR AnnData (seurat_v3 operates on raw counts)."""
    rng = np.random.default_rng(seed)
    mat = sp.random(n_obs, n_vars, density=0.2, format="csr", dtype=np.float32, random_state=rng)
    mat.data = np.ceil(mat.data * 50).astype(np.float32)
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(n_obs)]
    adata.var["gene_id"] = [f"g{j}" for j in range(n_vars)]
    return adata


def _open_backed(path, adata: ad.AnnData, csc: str) -> ad.AnnData:
    """Round-trip through an SCX file (with/without CSC), opened backed."""
    if csc == "off":
        pyscx.from_anndata(adata, str(path))
    else:
        pyscx.from_anndata(adata, str(path), csc=csc, csc_cols_per_shard=16)
    return pyscx.open(str(path)).to_anndata(backed=True)


def _route(adata: ad.AnnData) -> str | None:
    try:
        return adata.uns["scx_accel"]["highly_variable_genes"]["route"]
    except Exception:
        return None


def _hvg_set(adata: ad.AnnData) -> set[str]:
    var = adata.var
    return set(var.index[var["highly_variable"]].tolist())


def test_gpu_csc_explicit_route_and_parity(tmp_path):
    """`prefer_format="csc"` + `device="gpu"` runs the CSC reduce (gpu_csc_v3)
    and matches the CSR GPU path's HVG set + stats."""
    src = _raw_counts_adata()

    a_csc = _open_backed(tmp_path / "csc.scx", src, "always")
    pyscx.accel.highly_variable_genes(
        a_csc, n_top_genes=30, flavor="seurat_v3", device="gpu", prefer_format="csc"
    )
    assert _route(a_csc) == "gpu_csc_v3"

    # CSR GPU reference on a sidecar-free file (atomic kernel, gpu_csr).
    a_csr = _open_backed(tmp_path / "csr.scx", src, "off")
    pyscx.accel.highly_variable_genes(
        a_csr, n_top_genes=30, flavor="seurat_v3", device="gpu"
    )
    assert _route(a_csr) == "gpu_csr"

    # Per-gene mean/variance agree to f64 epsilon.
    np.testing.assert_allclose(
        a_csc.var["means"].to_numpy(), a_csr.var["means"].to_numpy(), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        a_csc.var["variances"].to_numpy(), a_csr.var["variances"].to_numpy(),
        rtol=1e-5, atol=1e-6,
    )
    # Identical HVG selection (same loess fit on identical stats).
    assert _hvg_set(a_csc) == _hvg_set(a_csr)


def test_gpu_csc_autodetect_default_prefer_format(tmp_path):
    """Default `prefer_format="csr"` + `device="gpu"` on a backed dataset with
    a CSC sidecar auto-routes to the CSC reduce (gpu_csc_v3)."""
    src = _raw_counts_adata()
    a = _open_backed(tmp_path / "auto.scx", src, "always")
    pyscx.accel.highly_variable_genes(
        a, n_top_genes=30, flavor="seurat_v3", device="gpu"
    )
    assert _route(a) == "gpu_csc_v3"


def test_cpu_csc_route(tmp_path):
    """`device="cpu"` + `prefer_format="csc"` records the cpu_csc route."""
    src = _raw_counts_adata()
    a = _open_backed(tmp_path / "cpu_csc.scx", src, "always")
    pyscx.accel.highly_variable_genes(
        a, n_top_genes=30, flavor="seurat_v3", device="cpu", prefer_format="csc"
    )
    assert _route(a) == "cpu_csc"


def test_csr_only_backed_gpu_stays_csr(tmp_path):
    """A backed file without a CSC sidecar + `device="gpu"` keeps the CSR
    atomic route (no auto-detect false positive)."""
    src = _raw_counts_adata()
    a = _open_backed(tmp_path / "csr_only.scx", src, "off")
    pyscx.accel.highly_variable_genes(
        a, n_top_genes=30, flavor="seurat_v3", device="gpu"
    )
    assert _route(a) == "gpu_csr"


def test_gpu_csc_on_a_row_filtered_handle(tmp_path):
    """A windowed backed handle reaches the CSC reduce, over its own window.

    `backed_x_has_csc_sidecar` mirrors the `as_column_source` capability gate,
    and that gate used to refuse a row filter — so this auto-route was
    unreachable for a filtered handle and the run fell to the CSR atomic
    kernel. It is reachable now, which makes this the test that says the row
    renumbering reached the GPU staging path as well as the CPU one: the CSC
    mean/var kernel divides by `source.n_obs()`, so an uncompacted slab would
    be wrong in two directions at once — sums over the file's rows, divided by
    the window's count.

    Compared against the CSR GPU route on the same window, which is the only
    oracle that cannot share a bug with the path under test.
    """
    src = _raw_counts_adata()
    keep = np.zeros(src.n_obs, dtype=bool)
    keep[::3] = True  # interleaved, so every survivor shifts by a different amount

    a_csc = _open_backed(tmp_path / "csc_filtered.scx", src, "always")
    pyscx.accel.subset_obs(a_csc, keep)
    assert a_csc.n_obs == int(keep.sum()) < src.n_obs
    pyscx.accel.highly_variable_genes(
        a_csc, n_top_genes=30, flavor="seurat_v3", device="gpu", prefer_format="csc"
    )
    assert _route(a_csc) == "gpu_csc_v3"

    a_csr = _open_backed(tmp_path / "csr_filtered.scx", src, "off")
    pyscx.accel.subset_obs(a_csr, keep)
    pyscx.accel.highly_variable_genes(
        a_csr, n_top_genes=30, flavor="seurat_v3", device="gpu"
    )
    assert _route(a_csr) == "gpu_csr"

    np.testing.assert_allclose(
        a_csc.var["means"].to_numpy(), a_csr.var["means"].to_numpy(), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        a_csc.var["variances"].to_numpy(), a_csr.var["variances"].to_numpy(),
        rtol=1e-5, atol=1e-6,
    )
    assert _hvg_set(a_csc) == _hvg_set(a_csr)
    # And against numpy over the window, so both GPU routes being wrong the
    # same way would still fail.
    dense = np.asarray(src.X.todense(), dtype=np.float64)[keep]
    np.testing.assert_allclose(
        a_csc.var["means"].to_numpy(), dense.mean(axis=0), rtol=1e-5, atol=1e-8
    )


def test_gpu_hvg_does_not_auto_route_a_filtered_handle_to_csc(tmp_path):
    """The auto-route is policy, and a row filter is outside it.

    `backed_x_has_csc_sidecar` decides whether a default
    `prefer_format="csr"` GPU run silently takes the column-major reduce. It
    used to miss a `filter_cells` handle only because the capability gate
    refused one; once that gate opened, the auto-route would have fired for an
    op whose CSC walk is *slower* than the row-major sweep — adding a
    compaction pass over every column shard on top, for a caller who never
    asked for CSC. So the predicate now requires no row filter. The test above
    is the other half: an explicit request is still honoured on the same
    handle.
    """
    src = _raw_counts_adata()
    keep = np.zeros(src.n_obs, dtype=bool)
    keep[::3] = True

    a = _open_backed(tmp_path / "csc_autoroute.scx", src, "always")
    pyscx.accel.subset_obs(a, keep)
    pyscx.accel.highly_variable_genes(
        a, n_top_genes=30, flavor="seurat_v3", device="gpu"
    )
    assert _route(a) == "gpu_csr", (
        f"a filtered handle must not auto-route into the slower CSC reduce, got {_route(a)!r}"
    )
    # And the stamp must not claim the file had no sidecar: declining on policy
    # is a different diagnosis from a sidecar-less file, and the route gates
    # read exactly this field.
    assert a.uns["scx_accel"]["highly_variable_genes"]["csc_available"] is True

    # Premise: the same handle, unfiltered, does auto-route — so the assertion
    # above is the row filter's doing and not a missing sidecar.
    b = _open_backed(tmp_path / "csc_autoroute_unfiltered.scx", src, "always")
    pyscx.accel.highly_variable_genes(
        b, n_top_genes=30, flavor="seurat_v3", device="gpu"
    )
    assert _route(b) == "gpu_csc_v3"
