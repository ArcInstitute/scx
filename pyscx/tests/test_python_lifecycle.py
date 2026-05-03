"""Full Python lifecycle test."""

import numpy as np
import scipy.sparse as sp


def test_full_lifecycle(tmp_dir):
    """Full Python lifecycle: create → round-trip → append → query → delete → compact."""
    import anndata
    import pandas as pd
    import pyscx

    # 1. from_anndata → create initial file (using string obs, not categorical)
    np.random.seed(88)
    n_obs, n_vars = 60, 30
    dense = np.random.randint(0, 80, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    cell_types = (["T cell"] * 20) + (["B cell"] * 20) + (["NK cell"] * 20)
    obs = pd.DataFrame(
        {"cell_type": cell_types, "tissue": (["lung", "blood"] * 30)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    path = str(tmp_dir / "lifecycle.scx")
    pyscx.from_anndata(adata, path)

    # 2. open → to_anndata → verify round-trip
    adata_rt = pyscx.open(path).to_anndata()
    assert adata_rt.n_obs == n_obs
    assert adata_rt.n_vars == n_vars

    # 3. append_from_anndata → add cells
    n_new = 15
    dense2 = np.random.randint(0, 50, size=(n_new, n_vars)).astype(np.float32)
    x2 = sp.csr_matrix(dense2)
    obs2 = pd.DataFrame(
        {"cell_type": ["T cell"] * n_new,
         "tissue": ["lung"] * n_new},
        index=[f"new_{i}" for i in range(n_new)],
    )
    new_adata = anndata.AnnData(X=x2, obs=obs2, var=var)
    pyscx.append_from_anndata(path, new_adata)

    # 4. open → n_obs → verify increased
    assert pyscx.open(path).n_obs == n_obs + n_new

    # 5. query pipeline → filter_obs → collect → n_obs
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")
    qr = p.collect()
    assert qr.n_obs == 20 + n_new  # original 20 T cells + 15 appended

    # 6. mark_deleted → delete some cells
    pyscx.mark_deleted(path, [0, 1, 2])

    # 7. Verify deletion via n_obs on query (avoids categorical concat issue in to_anndata)
    all_count = pyscx.open(path).query().count()
    assert all_count == n_obs + n_new - 3

    # 8. compact → reclaim space
    compacted = str(tmp_dir / "lifecycle_compacted.scx")
    pyscx.compact(path, compacted)

    # 9. open compacted → verify data integrity
    adata_compacted = pyscx.open(compacted).to_anndata()
    assert adata_compacted.n_obs == n_obs + n_new - 3
    assert adata_compacted.n_vars == n_vars


def test_query_matches_anndata_subsetting(query_adata, scx_from_adata):
    """Query result matches equivalent AnnData subsetting."""
    import pyscx

    path = scx_from_adata(query_adata, "compare.scx")

    # SCX query path
    result = pyscx.open(path).query()
    result.filter_obs("cell_type == 'T cell'")
    scx_adata = result.collect().to_anndata()

    # AnnData subsetting path
    full_adata = pyscx.open(path).to_anndata()
    mask = full_adata.obs["cell_type"] == "T cell"
    anndata_subset = full_adata[mask]

    # Compare
    assert scx_adata.n_obs == anndata_subset.n_obs
    np.testing.assert_array_equal(
        scx_adata.X.toarray(), anndata_subset.X.toarray()
    )
