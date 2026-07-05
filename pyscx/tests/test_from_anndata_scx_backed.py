"""Phase 8b — SCX → SCX streaming writer for backed / lazy `X`.

Covers `pyscx.from_anndata` accepting `adata.X` as either a
`ScxBackedSparseDataset` (returned by
`pyscx.open(path).to_anndata(backed=True)`) or a
`ScxLazyTransformedDataset` (after `pyscx.accel.normalize_total` /
`pyscx.accel.log1p` stacking).

The byte-passthrough branch copies pre-encoded CSR shards from the
source SCX file into the target writer without decode + re-encode.
The decode-encode fallback handles shard-size / codec mismatch,
deletions, column projection, and lazy transforms.
"""

import json
import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def src_scx(synthetic_adata, tmp_dir):
    """Build a baseline SCX file from `synthetic_adata` and return its path."""
    path = str(tmp_dir / "src.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


def _adata_with_layers(n_obs=80, n_vars=30, n_layers=2, seed=11):
    """Build a small AnnData with N CSR layers for round-trip tests."""
    rng = np.random.RandomState(seed)
    dense = rng.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0
    layers = {}
    for k in range(n_layers):
        layer_dense = rng.randint(0, 50, size=(n_obs, n_vars)).astype(np.float32)
        layer_dense[rng.random((n_obs, n_vars)) > 0.4] = 0
        layers[f"layer_{k}"] = sp.csr_matrix(layer_dense)
    return anndata.AnnData(X=sp.csr_matrix(dense), layers=layers)


# ---------------------------------------------------------------------------
# Passthrough byte-equality
# ---------------------------------------------------------------------------


def _assert_passthrough(dst):
    """Assert the destination's provenance records a backed passthrough."""
    prov = pyscx.open(dst).provenance()
    assert any(
        json.loads(p["params_json"]).get("x_source") == "backed"
        and json.loads(p["params_json"]).get("passthrough") is True
        for p in prov
    ), f"expected passthrough provenance, got {prov}"


def test_passthrough_decoded_x_equals_source_unframed(synthetic_adata, tmp_dir):
    """Backed SCX → SCX (matching shard_size + codec, no deletions /
    projection) yields decoded X that equals the source bit-exact — the
    UNFRAMED (v3) source case (built via `row_group_rows=0`)."""
    src_scx = str(tmp_dir / "src_unframed.scx")
    pyscx.from_anndata(synthetic_adata, src_scx, row_group_rows=0)

    adata = pyscx.open(src_scx).to_anndata(backed=True)
    dst = str(tmp_dir / "dst.scx")
    pyscx.from_anndata(adata, dst)

    src_x = pyscx.open(src_scx).to_anndata().X
    dst_x = pyscx.open(dst).to_anndata().X
    assert src_x.shape == dst_x.shape
    np.testing.assert_array_equal(src_x.toarray(), dst_x.toarray())
    _assert_passthrough(dst)


def test_passthrough_decoded_x_equals_source_framed(synthetic_adata, tmp_dir):
    """Backed SCX → SCX passthrough for a default (row-group-framed, v4)
    source. C5 re-enables the byte-copy fast path for pure-framed v4 files
    (shards carry an in-body BlockIndex, no decode sidecar), so the backed
    rewrite of a default `from_anndata` file must take the passthrough path
    and preserve decoded X bit-exact."""
    src_scx = str(tmp_dir / "src_framed.scx")
    # No `row_group_rows` → framing on by default (v4).
    pyscx.from_anndata(synthetic_adata, src_scx)

    adata = pyscx.open(src_scx).to_anndata(backed=True)
    dst = str(tmp_dir / "dst_framed.scx")
    pyscx.from_anndata(adata, dst)

    src_x = pyscx.open(src_scx).to_anndata().X
    dst_x = pyscx.open(dst).to_anndata().X
    assert src_x.shape == dst_x.shape
    np.testing.assert_array_equal(src_x.toarray(), dst_x.toarray())
    _assert_passthrough(dst)


# ---------------------------------------------------------------------------
# Decode-encode on shard-size mismatch
# ---------------------------------------------------------------------------


def test_shard_size_mismatch_falls_back_to_decode(src_scx, tmp_dir):
    """Passing `shard_size` different from the source forces the
    decode-encode path; output X must still match numerically."""
    adata = pyscx.open(src_scx).to_anndata(backed=True)
    dst = str(tmp_dir / "dst_mismatch.scx")
    # Source default is 16384; choose 32 so a 100-row fixture produces
    # multiple shards and the count differs from the source's single
    # shard.
    pyscx.from_anndata(adata, dst, shard_size=32)

    src = pyscx.open(src_scx)
    out = pyscx.open(dst)
    assert out.shard_count != src.shard_count, \
        f"expected differing shard count, got {out.shard_count} vs {src.shard_count}"

    src_x = src.to_anndata().X
    dst_x = out.to_anndata().X
    np.testing.assert_array_equal(src_x.toarray(), dst_x.toarray())

    prov = out.provenance()
    assert any(
        json.loads(p["params_json"]).get("x_source") == "backed"
        and json.loads(p["params_json"]).get("passthrough") is False
        for p in prov
    )


# ---------------------------------------------------------------------------
# obs / var / obsm / uns updates round-trip
# ---------------------------------------------------------------------------


def test_metadata_updates_round_trip(src_scx, tmp_dir):
    """In-memory mutations on obs / var / obsm / uns of a backed
    AnnData are preserved through the SCX → SCX rewrite."""
    adata = pyscx.open(src_scx).to_anndata(backed=True)
    # Mutate
    adata.obs["new_col"] = np.arange(adata.n_obs, dtype=np.int64)
    adata.var["hv_phase8b"] = np.arange(adata.n_vars, dtype=np.int32) % 2 == 0
    adata.obsm["X_phase8b"] = np.random.RandomState(7).randn(
        adata.n_obs, 3
    ).astype(np.float32)
    adata.uns["log1p"] = {"base": None, "phase": "8b"}

    dst = str(tmp_dir / "dst_meta.scx")
    pyscx.from_anndata(adata, dst)

    out = pyscx.open(dst).to_anndata()
    assert "new_col" in out.obs.columns
    np.testing.assert_array_equal(
        out.obs["new_col"].to_numpy(),
        np.arange(adata.n_obs, dtype=np.int64),
    )
    assert "hv_phase8b" in out.var.columns
    np.testing.assert_array_equal(
        out.var["hv_phase8b"].to_numpy().astype(bool),
        (np.arange(adata.n_vars, dtype=np.int32) % 2 == 0),
    )
    assert "X_phase8b" in out.obsm
    assert out.obsm["X_phase8b"].shape == (adata.n_obs, 3)
    assert out.uns.get("log1p", {}).get("phase") == "8b"


# ---------------------------------------------------------------------------
# Lazy decode-encode: normalize_total + log1p
# ---------------------------------------------------------------------------


def test_lazy_normalize_log1p_matches_scanpy(src_scx, tmp_dir):
    """`adata.X = ScxLazyTransformedDataset` (normalize_total + log1p)
    writes a CSR whose decoded values equal
    `sc.pp.log1p(sc.pp.normalize_total(...))` within fp32 tolerance."""
    sc = pytest.importorskip("scanpy")

    # Reference: scanpy on a fresh in-memory copy.
    ref = pyscx.open(src_scx).to_anndata()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)

    # Lazy path: normalize_total + log1p stacked on a backed wrapper.
    lazy_adata = pyscx.open(src_scx).to_anndata(backed=True)
    pyscx.accel.normalize_total(lazy_adata, target_sum=1e4)
    pyscx.accel.log1p(lazy_adata)

    dst = str(tmp_dir / "dst_lazy.scx")
    pyscx.from_anndata(lazy_adata, dst)

    out = pyscx.open(dst).to_anndata()
    np.testing.assert_allclose(
        out.X.toarray(),
        ref.X.toarray() if sp.issparse(ref.X) else np.asarray(ref.X),
        rtol=1e-5,
        atol=1e-6,
    )

    prov = pyscx.open(dst).provenance()
    payload = next(
        json.loads(p["params_json"])
        for p in prov
        if json.loads(p["params_json"]).get("x_source") == "lazy"
    )
    assert payload["passthrough"] is False
    names = [t["name"] for t in payload["lazy_transforms"]]
    assert names == ["normalize_total", "log1p"], names


# ---------------------------------------------------------------------------
# Wrapper accessors
# ---------------------------------------------------------------------------


def test_lazy_transforms_repr_accessor(src_scx):
    """`ScxLazyTransformedDataset.transforms_repr()` returns a
    JSON-serialisable list of `{name, params}`."""
    adata = pyscx.open(src_scx).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    pyscx.accel.log1p(adata)
    out = adata.X.transforms_repr()
    assert isinstance(out, list)
    assert [t["name"] for t in out] == ["normalize_total", "log1p"]
    assert out[0]["params"]["target_sum"] == 1e4
    assert isinstance(out[0]["params"]["row_sums_len"], int)


# ---------------------------------------------------------------------------
# Deletions disable passthrough
# ---------------------------------------------------------------------------


def test_deletions_disable_passthrough(tmp_dir):
    """A source with a deletion vector forces the decode-encode path
    and emits `passthrough=false` provenance; the output omits the
    deleted rows."""
    np.random.seed(99)
    n_obs, n_vars = 80, 25
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    adata = anndata.AnnData(X=sp.csr_matrix(dense))

    src = str(tmp_dir / "src_del.scx")
    pyscx.from_anndata(adata, src)

    # Mark a few cells deleted.
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[7] = True
    delete_mask[n_obs - 1] = True
    n_deleted = int(delete_mask.sum())

    pyscx.open(src).mark_deleted(delete_mask)

    # Backed → from_anndata should observe the deletion vector and
    # fall through to decode-encode.
    backed = pyscx.open(src).to_anndata(backed=True)
    assert backed.n_obs == n_obs - n_deleted

    dst = str(tmp_dir / "dst_del.scx")
    pyscx.from_anndata(backed, dst)

    out = pyscx.open(dst)
    assert out.n_obs == n_obs - n_deleted

    prov = out.provenance()
    payload = next(
        json.loads(p["params_json"])
        for p in prov
        if json.loads(p["params_json"]).get("x_source") == "backed"
    )
    assert payload["passthrough"] is False


# ---------------------------------------------------------------------------
# CSC sidecar drop warning
# ---------------------------------------------------------------------------


def test_csc_sidecar_drop_warning(synthetic_adata, tmp_dir):
    """Source with a CSC sidecar → backed → from_anndata(csc='off')
    emits a `UserWarning` that names the rebuild opt-in."""
    src = str(tmp_dir / "src_csc.scx")
    pyscx.from_anndata(synthetic_adata, src, csc="always", csc_cols_per_shard=10)

    # Sanity: the file actually has a CSC sidecar.
    assert pyscx.open(src).has_csc, "fixture lacks CSC sidecar"

    backed = pyscx.open(src).to_anndata(backed=True)
    dst = str(tmp_dir / "dst_no_csc.scx")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.from_anndata(backed, dst)
    msgs = [str(w.message) for w in caught]
    assert any("CSC sidecar" in m and "csc=\"always\"" in m for m in msgs), \
        f"expected CSC-drop warning, got {msgs}"

    # Output should not have a CSC sidecar.
    assert not pyscx.open(dst).has_csc


# ---------------------------------------------------------------------------
# Layers round-trip through the streaming rewrite
# ---------------------------------------------------------------------------


def test_layers_round_trip_through_streaming_rewrite(tmp_dir):
    """Layers on a backed AnnData must be preserved through the SCX → SCX
    rewrite. Regression test for the bug where both route functions
    returned before the layer-writing block."""
    adata = _adata_with_layers(n_obs=64, n_vars=20, n_layers=2)
    src = str(tmp_dir / "src_layers.scx")
    pyscx.from_anndata(adata, src)
    assert sorted(pyscx.open(src).layer_names()) == ["layer_0", "layer_1"]

    backed = pyscx.open(src).to_anndata(backed=True)
    dst = str(tmp_dir / "dst_layers.scx")
    pyscx.from_anndata(backed, dst)

    out = pyscx.open(dst)
    assert sorted(out.layer_names()) == ["layer_0", "layer_1"]
    out_ad = out.to_anndata()
    src_ad = pyscx.open(src).to_anndata()
    for name in ["layer_0", "layer_1"]:
        np.testing.assert_array_equal(
            out_ad.layers[name].toarray(),
            src_ad.layers[name].toarray(),
        )


def test_layers_round_trip_after_lazy_transforms(tmp_dir):
    """Lazy transforms (normalize_total + log1p) are applied to X only —
    layers must round-trip untransformed."""
    adata = _adata_with_layers(n_obs=64, n_vars=20, n_layers=1)
    src = str(tmp_dir / "src_layers_lazy.scx")
    pyscx.from_anndata(adata, src)

    backed = pyscx.open(src).to_anndata(backed=True)
    pyscx.accel.normalize_total(backed, target_sum=1e4)
    pyscx.accel.log1p(backed)

    dst = str(tmp_dir / "dst_layers_lazy.scx")
    pyscx.from_anndata(backed, dst)

    out = pyscx.open(dst).to_anndata()
    src_layer = pyscx.open(src).to_anndata().layers["layer_0"]
    np.testing.assert_array_equal(
        out.layers["layer_0"].toarray(),
        src_layer.toarray(),
    )


# ---------------------------------------------------------------------------
# Codec / index_dtype header faithfulness
# ---------------------------------------------------------------------------


def test_default_codec_preserves_source_choice(synthetic_adata, tmp_dir):
    """A backed rewrite without an explicit `codec=` should preserve
    the source codec. A `shard_size=` override forces decode-encode
    but the codec choice still comes from the source header."""
    src = str(tmp_dir / "src_scx1.scx")
    pyscx.from_anndata(synthetic_adata, src, codec="scx1")
    src_codec = pyscx.open(src).codec_id
    assert src_codec == 1, f"expected Scx1 (1), got {src_codec}"

    backed = pyscx.open(src).to_anndata(backed=True)
    dst = str(tmp_dir / "dst_default_codec.scx")
    pyscx.from_anndata(backed, dst, shard_size=32)

    out = pyscx.open(dst)
    assert out.codec_id == src_codec, (
        f"default codec should preserve source ({src_codec}); got {out.codec_id}"
    )


def test_pcodec_round_trip_preserves_header_codec(synthetic_adata, tmp_dir):
    """Source written with `codec='pcodec'` (id 4) must round-trip
    through the streaming rewrite with the header codec preserved.
    Regression test for the hardcoded `match` that mapped Pcodec to
    Zstd in the header."""
    src = str(tmp_dir / "src_pcodec.scx")
    pyscx.from_anndata(synthetic_adata, src, codec="pcodec")
    assert pyscx.open(src).codec_id == 4

    backed = pyscx.open(src).to_anndata(backed=True)
    dst = str(tmp_dir / "dst_pcodec.scx")
    pyscx.from_anndata(backed, dst, shard_size=32)

    out = pyscx.open(dst)
    assert out.codec_id == 4, f"expected Pcodec (4) in header, got {out.codec_id}"


def test_column_projection_uses_visible_n_vars(tmp_dir):
    """A backed rewrite with column projection should compute
    `index_dtype` from the user-visible (projected) n_vars, not the
    source's. The visible-shape invariant is what makes the writer
    parity-equal with the in-memory path.

    Projection on a backed AnnData is wired through
    `to_anndata(backed=True, var_names=[...])`, which sets a
    `col_projection` on the X wrapper. `pyscx.from_anndata` then
    sees the projected shape and routes through decode-encode."""
    rng = np.random.RandomState(7)
    n_obs, n_vars = 80, 100
    dense = rng.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0
    gene_names = [f"gene_{i}" for i in range(n_vars)]
    var = pd.DataFrame({"gene_id": gene_names}, index=gene_names)
    adata = anndata.AnnData(X=sp.csr_matrix(dense), var=var)

    src = str(tmp_dir / "src_proj.scx")
    pyscx.from_anndata(adata, src)
    assert pyscx.open(src).index_dtype == 0  # u16 (n_vars=100)

    keep = ["gene_0", "gene_3", "gene_7", "gene_12", "gene_18"]
    backed = pyscx.open(src).to_anndata(backed=True, var_names=keep)
    assert backed.n_vars == 5
    dst = str(tmp_dir / "dst_proj.scx")
    pyscx.from_anndata(backed, dst)

    out = pyscx.open(dst)
    assert out.n_vars == 5
    assert out.index_dtype == 0, (
        f"projected output should use u16 indices (0); got {out.index_dtype}"
    )

    # Numerical sanity: projected X matches the manual numpy projection.
    src_x = pyscx.open(src).to_anndata().X.toarray()
    keep_idx = np.array([0, 3, 7, 12, 18])
    np.testing.assert_array_equal(
        out.to_anndata().X.toarray(),
        src_x[:, keep_idx],
    )
