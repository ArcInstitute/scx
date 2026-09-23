"""`bench_csc_dispatch._run_de` on a groupby with one-cell and unused levels.

Census fixtures' `cell_type` carries both: singleton levels, and levels from a
census-wide vocabulary with no cell at all. `rank_genes_groups` refuses any
participating group below two cells, so before `_unlabel_undersized_groups`
every census DE arm of this benchmark raised before testing a gene.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pyscx = pytest.importorskip("pyscx")
ad = pytest.importorskip("anndata")
sp = pytest.importorskip("scipy.sparse")

from benchmarks.comprehensive.benchmarks import bench_csc_dispatch as b  # noqa: E402


def _backed(tmp_path):
    x = sp.random(60, 12, density=0.4, format="csr", random_state=0, dtype=np.float32)
    x.data = np.ceil(x.data * 10)
    a = ad.AnnData(X=x)
    labels = ["a"] * 29 + ["b"] * 30 + ["solo"]
    a.obs["cell_type"] = pd.Categorical(labels, categories=["a", "b", "solo", "never_used"])
    path = tmp_path / "g.scx"
    pyscx.from_anndata(a, str(path), csc="always")
    backed = pyscx.open(str(path)).to_anndata(backed=True)
    backed.obs["cell_type"] = a.obs["cell_type"].to_numpy()
    backed.obs["cell_type"] = backed.obs["cell_type"].astype(a.obs["cell_type"].dtype)
    return backed


def test_the_premise_raises_without_the_helper(tmp_path):
    backed = _backed(tmp_path)
    with pytest.raises(ValueError, match="Could not calculate statistics"):
        pyscx.accel.rank_genes_groups(backed, "cell_type", device="cpu")


@pytest.mark.parametrize("prefer", ["csr", "csc"])
def test_run_de_unlabels_the_undersized_levels_and_runs(tmp_path, prefer):
    backed = _backed(tmp_path)
    b._run_de(backed, prefer)
    col = backed.obs["cell_type"]
    assert list(col.cat.categories) == ["a", "b"]
    assert int(col.isna().sum()) == 1, "the one-cell level becomes an unlabelled cell"
    assert set(backed.uns["rank_genes_groups"]["names"].dtype.names) == {"a", "b"}


def test_too_few_levels_left_falls_back_to_the_synthetic_split(tmp_path):
    backed = _backed(tmp_path)
    backed.obs["cell_type"] = pd.Categorical(["a"] * 59 + ["solo"])
    b._run_de(backed, "csr")
    assert "_bench_group" in backed.obs.columns
