"""Tests for `pyscx.read_h5ad_metadata` and the `obs_override` /
`var_override` / `uns_override` kwargs on `pyscx.from_h5ad`.

These cover the read-mutate-write flow that avoids materialising
`obsm` through `anndata.read_h5ad(backed='r')`:

    meta = pyscx.read_h5ad_metadata("big.h5ad")
    meta.obs["new_col"] = ...
    pyscx.from_h5ad("big.h5ad", "out.scx", obs_override=meta.obs)

and the silent companion fix — plain `pyscx.from_h5ad(path, out)` no
longer opens a backed AnnData under the hood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

_pyscx = pytest.importorskip("pyscx")
if not hasattr(_pyscx, "from_h5ad"):
    pytest.skip("pyscx built without hdf5 feature", allow_module_level=True)
if not hasattr(_pyscx, "read_h5ad_metadata"):
    pytest.skip(
        "pyscx build predates read_h5ad_metadata", allow_module_level=True
    )


@pytest.fixture
def h5ad_path(synthetic_adata, tmp_dir):
    path = str(tmp_dir / "in.h5ad")
    synthetic_adata.write_h5ad(path)
    return path


def test_read_h5ad_metadata_basic(synthetic_adata, h5ad_path):
    """Returns obs/var/uns matching anndata.read_h5ad for the same fixture."""
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    assert meta.n_obs == synthetic_adata.n_obs
    assert meta.n_vars == synthetic_adata.n_vars
    assert meta.x_format in {"csr", "csc", "dense"}

    # obs round-trip: index + columns.
    pd.testing.assert_index_equal(
        meta.obs.index, synthetic_adata.obs.index, check_names=False
    )
    for col in synthetic_adata.obs.columns:
        np.testing.assert_array_equal(
            np.asarray(meta.obs[col]).astype(str),
            np.asarray(synthetic_adata.obs[col]).astype(str),
        )

    # var round-trip: index + columns.
    pd.testing.assert_index_equal(
        meta.var.index, synthetic_adata.var.index, check_names=False
    )
    for col in synthetic_adata.var.columns:
        np.testing.assert_array_equal(
            np.asarray(meta.var[col]).astype(str),
            np.asarray(synthetic_adata.var[col]).astype(str),
        )

    # uns round-trip: keys + scalar values.
    assert isinstance(meta.uns, dict)
    for k, v in synthetic_adata.uns.items():
        assert k in meta.uns, f"uns key {k!r} missing"
        assert str(meta.uns[k]) == str(v)


def test_read_h5ad_metadata_no_obsm_alloc(h5ad_path):
    """The metadata read does not materialise obsm into Python memory.

    The synthetic fixture has `obsm["X_pca"]` (100×10 float32). We
    confirm that the returned object has no obsm attribute / key —
    `read_h5ad_metadata` only exposes obs/var/uns by design.
    """
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    assert not hasattr(meta, "obsm")


def test_read_h5ad_metadata_repr(h5ad_path):
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    r = repr(meta)
    assert "H5adMetadata" in r
    assert "n_obs=" in r
    assert "n_vars=" in r


def test_from_h5ad_obs_override_round_trip(
    synthetic_adata, h5ad_path, tmp_dir
):
    """obs_override is applied at write time and survives a read-back.

    Critically also verifies that obsm and layers (which are NOT
    expressed as overrides) still survive the convert. That's the
    user-facing guarantee this entire entrypoint is built around:
    mutating obs must not drop the big embedding matrices.
    """
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    # Add a caller-side column that wasn't in the original obs.
    meta.obs["pipeline_tag"] = pd.Categorical(
        ["alpha"] * (synthetic_adata.n_obs // 2)
        + ["beta"] * (synthetic_adata.n_obs - synthetic_adata.n_obs // 2)
    )

    out = str(tmp_dir / "with_obs_override.scx")
    pyscx.from_h5ad(h5ad_path, out, obs_override=meta.obs)

    readback = pyscx.open(out).to_anndata()
    assert "pipeline_tag" in readback.obs.columns
    assert set(readback.obs["pipeline_tag"].astype(str).unique()) == {
        "alpha",
        "beta",
    }
    # X is still the original — overrides don't touch X.
    np.testing.assert_array_equal(
        readback.X.toarray() if sp.issparse(readback.X) else np.asarray(readback.X),
        synthetic_adata.X.toarray()
        if sp.issparse(synthetic_adata.X)
        else np.asarray(synthetic_adata.X),
    )
    # obsm survives — this is the regression guard. If obs_override
    # silently dropped obsm we'd have re-introduced the exact bug class
    # this branch exists to fix.
    assert set(readback.obsm.keys()) == set(synthetic_adata.obsm.keys())
    for k in synthetic_adata.obsm:
        np.testing.assert_allclose(
            np.asarray(readback.obsm[k]),
            np.asarray(synthetic_adata.obsm[k]),
        )
    # Layers survive too — they stream from disk and are not surfaced
    # as overrides.
    assert set(readback.layers.keys()) == set(synthetic_adata.layers.keys())


def test_from_h5ad_var_override_round_trip(synthetic_adata, h5ad_path, tmp_dir):
    """var_override happy path: caller mutates var, X / obsm survive."""
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    meta.var["annotation_source"] = pd.Categorical(
        ["ref_v2"] * synthetic_adata.n_vars
    )

    out = str(tmp_dir / "with_var_override.scx")
    pyscx.from_h5ad(h5ad_path, out, var_override=meta.var)

    readback = pyscx.open(out).to_anndata()
    assert "annotation_source" in readback.var.columns
    assert set(readback.var["annotation_source"].astype(str).unique()) == {
        "ref_v2"
    }
    np.testing.assert_array_equal(
        readback.X.toarray() if sp.issparse(readback.X) else np.asarray(readback.X),
        synthetic_adata.X.toarray()
        if sp.issparse(synthetic_adata.X)
        else np.asarray(synthetic_adata.X),
    )
    assert set(readback.obsm.keys()) == set(synthetic_adata.obsm.keys())


def test_from_h5ad_uns_override_replaces_section(
    synthetic_adata, h5ad_path, tmp_dir
):
    """uns_override is a full replacement, matching backed-router semantics.

    Also confirms obsm and layers survive the convert — uns_override
    must not silently drop the matrix sections this entrypoint exists
    to protect.
    """
    import pyscx

    out = str(tmp_dir / "with_uns_override.scx")
    pyscx.from_h5ad(
        h5ad_path,
        out,
        uns_override={"pipeline_version": "1.2.3", "run_id": 42},
    )
    readback = pyscx.open(out).to_anndata()
    # Original uns keys are gone (replace, not merge).
    assert "species" not in readback.uns
    # New keys are present.
    assert readback.uns["pipeline_version"] == "1.2.3"
    assert int(readback.uns["run_id"]) == 42
    # obsm + layers still come from disk.
    assert set(readback.obsm.keys()) == set(synthetic_adata.obsm.keys())
    assert set(readback.layers.keys()) == set(synthetic_adata.layers.keys())


def test_read_h5ad_metadata_accepts_pathlike(h5ad_path):
    """PathLike (pathlib.Path) is coerced via the existing _coerce_path
    helper, matching from_h5ad's ergonomics."""
    import pathlib

    import pyscx

    meta = pyscx.read_h5ad_metadata(pathlib.Path(h5ad_path))
    assert meta.n_obs > 0
    assert meta.n_vars > 0


def test_from_h5ad_obs_override_row_count_mismatch(h5ad_path, tmp_dir):
    """Mismatched obs_override row count is rejected loudly."""
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    # Drop one row — should now mismatch n_obs.
    short_obs = meta.obs.iloc[:-1].copy()
    out = str(tmp_dir / "should_fail.scx")
    with pytest.raises(ValueError, match=r"obs_override has \d+ rows but X has n_obs="):
        pyscx.from_h5ad(h5ad_path, out, obs_override=short_obs)


def test_from_h5ad_var_override_row_count_mismatch(h5ad_path, tmp_dir):
    """Mismatched var_override row count is rejected loudly."""
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    short_var = meta.var.iloc[:-1].copy()
    out = str(tmp_dir / "should_fail.scx")
    with pytest.raises(ValueError, match=r"var_override has \d+ rows but X has n_vars="):
        pyscx.from_h5ad(h5ad_path, out, var_override=short_var)


def test_from_h5ad_override_with_stream_false_rejected(h5ad_path, tmp_dir):
    """stream=False + any override -> ValueError. The non-streaming path
    doesn't apply overrides, so silently dropping them would be a footgun."""
    import pyscx

    meta = pyscx.read_h5ad_metadata(h5ad_path)
    out = str(tmp_dir / "should_fail.scx")
    with pytest.raises(ValueError, match="require stream=True"):
        pyscx.from_h5ad(h5ad_path, out, obs_override=meta.obs, stream=False)


def test_from_h5ad_no_override_does_not_call_anndata_read_h5ad(
    h5ad_path, tmp_dir, monkeypatch
):
    """Plain `pyscx.from_h5ad(path, out)` must NOT trigger
    `anndata.read_h5ad`. This is the companion fix that closes the
    obsm OOM hole even for callers that don't supply overrides.

    Implementation note: we monkey-patch `anndata.read_h5ad` to raise so
    that any internal call would surface as a clean error. The plain
    convert is expected to complete successfully because it routes
    straight to scx-convert's pure-Rust streaming pipeline.
    """
    import pyscx
    import anndata

    sentinel = []

    def boom(*args, **kwargs):
        sentinel.append((args, kwargs))
        raise RuntimeError("anndata.read_h5ad should not be called")

    monkeypatch.setattr(anndata, "read_h5ad", boom)

    out = str(tmp_dir / "pure_rust.scx")
    pyscx.from_h5ad(h5ad_path, out)
    assert sentinel == [], "anndata.read_h5ad was unexpectedly invoked"

    # And the output is readable + has the right shape.
    readback = pyscx.open(out).to_anndata()
    assert readback.n_obs > 0
    assert readback.n_vars > 0
