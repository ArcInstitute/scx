"""T2.3: ordered categoricals survive h5ad → scx → h5ad.

A pandas/anndata ordered factor (e.g. cell-cycle phase) must round-trip
with `ordered=True` and its original category order intact; an unordered
categorical must stay `ordered=False`. The `ordered` bit is carried in
Arrow `Field::metadata` (`scx.categorical.ordered`) through the SCX obs
section and re-emitted by the h5ad writer.

Both export writers are exercised: `shard_size=None` keeps obs as a single
section (eager h5ad writer); a small `shard_size` shards obs (streaming
h5ad writer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _adata_with_ordered(n_obs):
    import anndata
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    x = sp.csr_matrix(rng.integers(0, 5, size=(n_obs, 3)).astype(np.float32))
    # Ordered factor with a deliberately non-alphabetical category order.
    phase = pd.Categorical(
        [["G1", "S", "G2M"][i % 3] for i in range(n_obs)],
        categories=["G1", "S", "G2M"],
        ordered=True,
    )
    # Unordered control.
    batch = pd.Categorical(
        [["b", "a"][i % 2] for i in range(n_obs)],
        categories=["b", "a"],
        ordered=False,
    )
    obs = pd.DataFrame(
        {"phase": phase, "batch": batch},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(3)])
    return anndata.AnnData(X=x, obs=obs, var=var)


@pytest.mark.parametrize(
    "shard_size,label",
    [(None, "eager"), (4, "streaming")],
)
def test_ordered_categorical_round_trips(tmp_dir, shard_size, label):
    import anndata

    import pyscx

    n_obs = 12
    src = _adata_with_ordered(n_obs)

    h5ad_in = str(tmp_dir / f"in_{label}.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / f"ordered_{label}.scx")
    if shard_size is None:
        pyscx.from_h5ad(h5ad_in, scx_path)
    else:
        pyscx.from_h5ad(h5ad_in, scx_path, shard_size=shard_size)

    h5ad_out = str(tmp_dir / f"out_{label}.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)

    out = anndata.read_h5ad(h5ad_out)
    # Ordered factor preserved, with original (non-alphabetical) order.
    assert out.obs["phase"].cat.ordered is True
    assert list(out.obs["phase"].cat.categories) == ["G1", "S", "G2M"]
    # Unordered control stays unordered.
    assert out.obs["batch"].cat.ordered is False


@pytest.mark.parametrize("backed", [False, True])
def test_ordered_categorical_via_to_anndata(tmp_dir, backed):
    """The in-memory `to_anndata()` path also restores the ordered bit
    (read from Arrow field metadata), not just the h5ad export path."""
    import anndata  # noqa: F401

    import pyscx

    src = _adata_with_ordered(12)
    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "ordered.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    out = pyscx.open(scx_path).to_anndata(backed=backed)
    assert out.obs["phase"].cat.ordered is True
    assert list(out.obs["phase"].cat.categories) == ["G1", "S", "G2M"]
    assert out.obs["batch"].cat.ordered is False
