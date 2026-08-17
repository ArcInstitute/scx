"""T2.3: ordered categoricals survive h5ad → scx → h5ad.

A pandas/anndata ordered factor (e.g. cell-cycle phase) must round-trip
with `ordered=True`, its original category order intact, and its declared
levels intact — including levels no cell uses, which are part of the factor
and not an artefact of the data. An unordered categorical must stay
`ordered=False`. The `ordered` bit is carried in Arrow `Field::metadata`
(`scx.categorical.ordered`) through the SCX obs section and re-emitted by
the h5ad writer.

Both export writers are exercised, but **not** by varying `shard_size` on
`from_h5ad`. Measured: `from_h5ad(shard_size=4)` produces
`obs_metadata_shard_count == 0` — `shard_size` shards `X`, while every
`scx-convert` ingest path writes obs as a single section. So a `from_h5ad`
round trip always takes the whole-batch writer no matter what `shard_size`
says. `from_anndata` is the one entry point that emits `ObsMetadataShard`
sections, and it is what the shard-stream arm below uses; that arm asserts
the shard count, because an arm that silently falls back to the other
writer tests nothing and looks like it tests everything.

The fixture is deliberately reorder-sensitive: declared order is neither
alphabetical nor first-appearance order, and one declared level has no
cells. Without both properties an exporter that rebuilt the vocabulary from
the data would pass — which is how it did.
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
    # Ordered factor. Three properties make it able to fail:
    #   - the declared order is not alphabetical,
    #   - the data starts at "G2M", so first-appearance order is a *third*
    #     order distinct from both, and
    #   - "M" is declared and used by no cell.
    phase = pd.Categorical(
        [["G2M", "G1", "S"][i % 3] for i in range(n_obs)],
        categories=["G1", "S", "G2M", "M"],
        ordered=True,
    )
    # Unordered control, also reversed relative to appearance order.
    batch = pd.Categorical(
        [["b", "a"][i % 2] for i in range(n_obs)],
        categories=["a", "b"],
        ordered=False,
    )
    obs = pd.DataFrame(
        {"phase": phase, "batch": batch},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(3)])
    return anndata.AnnData(X=x, obs=obs, var=var)


def _assert_factors_round_tripped(out):
    """Both factors survive whole: order, unused level, and `ordered` bit."""
    assert out.obs["phase"].cat.ordered is True
    assert list(out.obs["phase"].cat.categories) == ["G1", "S", "G2M", "M"]
    # The control's category list is asserted too — an unordered factor's
    # levels are still the user's, and leaving them unchecked is what let the
    # reordering go unnoticed on this column.
    assert out.obs["batch"].cat.ordered is False
    assert list(out.obs["batch"].cat.categories) == ["a", "b"]


def test_ordered_categorical_round_trips_whole_batch(tmp_dir):
    """`from_h5ad` → single obs section → the whole-batch writer."""
    import anndata

    import pyscx

    src = _adata_with_ordered(12)
    h5ad_in = str(tmp_dir / "in_whole.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "ordered_whole.scx")
    pyscx.from_h5ad(h5ad_in, scx_path, shard_size=4)

    # The premise of this arm: `shard_size` does not shard obs on ingest.
    assert pyscx.open(scx_path).obs_metadata_shard_count == 0

    h5ad_out = str(tmp_dir / "out_whole.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)
    _assert_factors_round_tripped(anndata.read_h5ad(h5ad_out))


def test_ordered_categorical_round_trips_shard_stream(tmp_dir):
    """`from_anndata` with a small `shard_size` → `ObsMetadataShard` sections
    → the shard-stream writer, which is the one that used to rebuild the
    vocabulary from the data."""
    import anndata

    import pyscx

    src = _adata_with_ordered(12)
    scx_path = str(tmp_dir / "ordered_sharded.scx")
    pyscx.from_anndata(src, scx_path, shard_size=4)

    # Without this the arm can quietly become a second copy of the one above.
    assert pyscx.open(scx_path).obs_metadata_shard_count > 1

    h5ad_out = str(tmp_dir / "out_sharded.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)
    _assert_factors_round_tripped(anndata.read_h5ad(h5ad_out))


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
    _assert_factors_round_tripped(out)


@pytest.mark.parametrize("backed", [False, True])
def test_ordered_categorical_from_anndata(tmp_dir, backed):
    """B2 regression: the in-memory `from_anndata` write path must also persist
    the ordered flag. Previously `pandas_to_record_batch` went through
    `pyarrow.Table.from_pandas`, which drops the pandas `ordered` bit, so the
    read path correctly found no metadata and returned the factor unordered."""
    import anndata  # noqa: F401

    import pyscx

    src = _adata_with_ordered(12)
    scx_path = str(tmp_dir / "ordered_from_anndata.scx")
    pyscx.from_anndata(src, scx_path)

    out = pyscx.open(scx_path).to_anndata(backed=backed)
    _assert_factors_round_tripped(out)
