"""F2 — grouped-read API tests (read_group / read_reference / iter_group_shards).

The grouped layout is produced by `scx sort --group-by` (write side is the CLI,
not pyscx), so these tests build an SCX via `pyscx.from_anndata`, then shell out
to the freshly built `scx` binary to sort it with grouping, then exercise the
pyscx read surface.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _scx_binary():
    """Locate the debug `scx` CLI built by `cargo build`."""
    # pyscx/tests/ -> repo root -> target/debug/scx
    root = Path(__file__).resolve().parents[2]
    for cand in (root / "target" / "debug" / "scx", root / "target" / "release" / "scx"):
        if cand.exists():
            return str(cand)
    return None


@pytest.fixture
def screen_adata():
    """A tiny perturbation-screen AnnData: target_gene grouping + a reference
    label, scattered so sorting must cluster them."""
    import anndata

    np.random.seed(7)
    genes = ["nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1"]
    n_obs, n_vars = len(genes), 6
    dense = np.random.randint(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[np.random.random((n_obs, n_vars)) > 0.5] = 0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(genes)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=x, obs=obs, var=var), genes


def _sort_grouped(scx_bin, src, out, extra):
    subprocess.run(
        [scx_bin, "sort", src, out, "--group-by", "target_gene", "--force", *extra],
        check=True,
        capture_output=True,
    )


def test_grouped_reads_roundtrip(screen_adata, scx_from_adata, tmp_dir):
    import pyscx

    scx_bin = _scx_binary()
    if scx_bin is None:
        pytest.skip("scx CLI binary not built (run `cargo build`)")

    adata, genes = screen_adata
    src = scx_from_adata(adata, "screen_src.scx")
    out = str(tmp_dir / "screen_grouped.scx")
    _sort_grouped(scx_bin, src, out, ["--reference", "nt"])

    exp = pyscx.open(out)

    # group_labels covers every label.
    labels = set(exp.group_labels())
    assert {"nt", "MYC", "TP53", "GATA1"} <= labels

    # read_group("MYC") returns exactly the MYC cells.
    myc = exp.read_group("MYC")
    n_myc = sum(1 for g in genes if g == "MYC")
    assert myc.n_obs == n_myc
    assert myc.n_vars == 6
    assert sp.issparse(myc.X)
    assert all(myc.obs["target_gene"] == "MYC")

    # read_reference returns the nt cells.
    ref = exp.read_reference()
    assert ref is not None
    assert ref.n_obs == sum(1 for g in genes if g == "nt")
    assert all(ref.obs["target_gene"] == "nt")

    # iter_group_shards covers every non-reference label exactly once.
    seen = []
    for gs in exp.iter_group_shards():
        ad = gs.to_anndata()
        assert ad.n_obs == (gs.global_stop - gs.global_start)
        seen.extend(gs.labels)
    assert "nt" not in seen
    assert set(seen) == {"MYC", "TP53", "GATA1"}
    assert len(seen) == len(set(seen)), "no label may appear in two shards"

    # Unknown label → KeyError with suggestions.
    with pytest.raises(KeyError):
        exp.read_group("MYCN")


def test_read_group_on_ungrouped_errors(query_adata, scx_from_adata):
    import pyscx

    path = scx_from_adata(query_adata, "ungrouped.scx")
    exp = pyscx.open(path)
    with pytest.raises(ValueError):
        exp.group_labels()
    with pytest.raises(ValueError):
        exp.read_group("anything")
