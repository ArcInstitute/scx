"""Tests for `pyscx.from_h5ad(..., shard_obs=...)` and the h5mu sibling.

Organization phase 6c (review item ORG-11.16-4). Every `scx-convert` ingest
path used to write obs as one legacy `ObsMetadata` section at any scale, so a
converted file never exercised the streaming h5ad *export* writer — the gap a
categorical-vocabulary bug lived in undetected. Ingest now shards obs on the
same tri-state policy and the same threshold `pyscx.from_anndata` and
`pyscx.optimize(shard_obs=)` use.

⚠️ Note the knob asymmetry these tests pin, so it does not get "tidied" into a
divergence: `from_h5ad` / `from_h5mu` take `shard_obs=` (tri-state) because they
reach `scx-convert`'s ingest, while `from_anndata` / `from_10x` take
`force_legacy_metadata=` (bool) because they build the batch in Python. Both
resolve to the same `n_obs > shard_size` boundary.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx

pytest.importorskip("anndata")


def _write_h5ad(path, n_obs, n_vars=20):
    """An h5ad with an ordered categorical whose declared vocabulary names a
    level no row uses — the payload the export path must not re-derive."""
    import anndata

    phase = pd.Categorical(
        [["G1", "S", "G2M"][i % 3] for i in range(n_obs)],
        categories=["G1", "S", "G2M", "M"],  # "M" declared, never used
        ordered=True,
    )
    adata = anndata.AnnData(
        X=sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32),
        obs=pd.DataFrame(
            {"phase": phase, "n_counts": np.arange(n_obs, dtype=np.int32)},
            index=[f"c{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    adata.write_h5ad(path)
    return adata


def test_from_h5ad_default_shards_obs_above_the_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = os.path.join(tmp, "in.h5ad")
        scx = os.path.join(tmp, "out.scx")
        _write_h5ad(h5ad, n_obs=40)

        # No shard_obs= at all: the default is "auto".
        pyscx.from_h5ad(h5ad, scx, shard_size=10)

        exp = pyscx.open(scx)
        assert exp.obs_metadata_shard_count == 4
        assert exp.n_obs == 40


def test_from_h5ad_shard_obs_off_keeps_a_single_section():
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = os.path.join(tmp, "in.h5ad")
        scx = os.path.join(tmp, "out.scx")
        _write_h5ad(h5ad, n_obs=40)

        pyscx.from_h5ad(h5ad, scx, shard_size=10, shard_obs="off")

        exp = pyscx.open(scx)
        assert exp.obs_metadata_shard_count == 0
        assert exp.n_obs == 40


def test_from_h5ad_shard_obs_always_shards_below_the_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = os.path.join(tmp, "in.h5ad")
        scx = os.path.join(tmp, "out.scx")
        _write_h5ad(h5ad, n_obs=6)

        # 6 rows at a target of 100: "auto" would not shard, "always" must.
        pyscx.from_h5ad(h5ad, scx, shard_size=100, shard_obs="always")

        assert pyscx.open(scx).obs_metadata_shard_count == 1


def test_from_h5ad_invalid_shard_obs_raises():
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = os.path.join(tmp, "in.h5ad")
        _write_h5ad(h5ad, n_obs=6)
        with pytest.raises(ValueError, match="shard_obs"):
            pyscx.from_h5ad(
                h5ad, os.path.join(tmp, "out.scx"), shard_obs="sometimes"
            )


def test_shard_obs_does_not_change_what_obs_says():
    """Both layouts must read back as the same frame — same values, same
    declared categories, same `ordered` bit. Sharding is a storage decision."""
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = os.path.join(tmp, "in.h5ad")
        sharded = os.path.join(tmp, "sharded.scx")
        single = os.path.join(tmp, "single.scx")
        _write_h5ad(h5ad, n_obs=40)

        pyscx.from_h5ad(h5ad, sharded, shard_size=10, shard_obs="always")
        pyscx.from_h5ad(h5ad, single, shard_size=10, shard_obs="off")
        assert pyscx.open(sharded).obs_metadata_shard_count == 4
        assert pyscx.open(single).obs_metadata_shard_count == 0

        a = pyscx.open(sharded).read_obs()
        b = pyscx.open(single).read_obs()
        assert list(a.columns) == list(b.columns)
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True)
        )
        assert list(a.index) == list(b.index)
        # The declared vocabulary, not just the values referenced by rows.
        assert list(a["phase"].cat.categories) == ["G1", "S", "G2M", "M"]
        assert a["phase"].cat.ordered


def test_a_converted_file_round_trips_its_declared_categories_to_h5ad():
    """The coverage ORG-11.16-4 buys: a `from_h5ad` output now exports through
    the *multi-shard* dataframe writer.

    ⚠️ The shard-count assertion is the test. Without it this silently becomes
    another whole-batch export if a default moves, and still passes — the exact
    way an earlier two-arm test in this suite turned out to have one arm.
    """
    import anndata

    with tempfile.TemporaryDirectory() as tmp:
        h5ad = os.path.join(tmp, "in.h5ad")
        scx = os.path.join(tmp, "mid.scx")
        out = os.path.join(tmp, "out.h5ad")
        _write_h5ad(h5ad, n_obs=40)

        pyscx.from_h5ad(h5ad, scx, shard_size=10)
        assert pyscx.open(scx).obs_metadata_shard_count > 1, (
            "this test only means anything against the shard-stream writer"
        )

        pyscx.to_h5ad(scx, out)

        back = anndata.read_h5ad(out)
        assert str(back.obs["phase"].dtype) == "category"
        assert list(back.obs["phase"].cat.categories) == ["G1", "S", "G2M", "M"]
        assert back.obs["phase"].cat.ordered


def test_from_h5mu_shards_the_outer_obs():
    pytest.importorskip("mudata")
    import anndata
    import mudata

    with tempfile.TemporaryDirectory() as tmp:
        h5mu = os.path.join(tmp, "in.h5mu")
        scx = os.path.join(tmp, "out.scx")
        n_obs = 40
        obs_names = [f"c{i}" for i in range(n_obs)]
        mods = {}
        for name, n_vars in (("rna", 20), ("adt", 8)):
            a = anndata.AnnData(
                X=sp.random(
                    n_obs, n_vars, density=0.3, format="csr", dtype=np.float32
                ),
                obs=pd.DataFrame(index=obs_names),
                var=pd.DataFrame(index=[f"{name}_g{i}" for i in range(n_vars)]),
            )
            mods[name] = a
        mdata = mudata.MuData(mods)
        mdata.obs["donor"] = pd.Categorical(
            [["d1", "d2"][i % 2] for i in range(n_obs)]
        )
        mdata.write(h5mu)

        pyscx.from_h5mu(h5mu, scx, shard_size=10)

        exp = pyscx.open(scx)
        assert exp.obs_metadata_shard_count == 4
        assert exp.n_obs == n_obs


def test_force_legacy_metadata_survives_backed_routing():
    """`from_anndata(backed_adata, force_legacy_metadata=True)` must still
    produce a single-section obs.

    A backed `X` routes `from_anndata` through `scx-convert`'s streaming
    ingest — the same path `from_h5ad` takes — so once that path started
    sharding obs by default, the flag became reachable only on the in-memory
    branch and was silently ignored here. The flag exists precisely to keep the
    legacy layout for readers that have not migrated to `obs_shards()`;
    ignoring it on one of the two branches breaks exactly those readers, and
    quietly.
    """
    import anndata

    with tempfile.TemporaryDirectory() as tmp:
        h5ad = os.path.join(tmp, "in.h5ad")
        _write_h5ad(h5ad, n_obs=40)

        for legacy, want in ((False, 4), (True, 0)):
            for mode in ("backed", "memory"):
                src = (
                    anndata.read_h5ad(h5ad, backed="r")
                    if mode == "backed"
                    else anndata.read_h5ad(h5ad)
                )
                out = os.path.join(tmp, f"{mode}_{legacy}.scx")
                pyscx.from_anndata(
                    src, out, shard_size=10, force_legacy_metadata=legacy
                )
                if mode == "backed":
                    src.file.close()
                assert pyscx.open(out).obs_metadata_shard_count == want, (
                    f"{mode} + force_legacy_metadata={legacy} should give "
                    f"{want} obs shards"
                )
