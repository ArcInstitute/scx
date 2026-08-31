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
import os
import shutil
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

    # The passthrough byte-copies shard-v2 (framed) shards, so the output header
    # MUST be v4 — a v3 stamp over framed shards would let a v3-only reader
    # accept the file and mis-decode each group's local-rebased indptr as global.
    assert pyscx.open(src_scx).format_version == 4, "source is framed v4"
    assert (
        pyscx.open(dst).format_version == 4
    ), "framed-source passthrough must stamp a v4 header over the copied v2 shards"


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


# ---------------------------------------------------------------------------
# Codec inherited from the source's SHARDS, not its file header
# ---------------------------------------------------------------------------


def _mixed_codec_source(tmp_dir, name="mixed_codec.scx", shard_size=100):
    """An SCX file whose CSR shards do NOT all share one codec.

    First half tiny counts, second half large ones, written unframed with a
    small `shard_size`, so `select_codec`'s median rule lands some shards on
    `scx1` and others on `zstd`.
    """
    import pyscx

    rng = np.random.RandomState(7)
    n_obs, n_vars = 800, 60
    lo = rng.randint(1, 4, size=(n_obs // 2, n_vars)).astype(np.float32)
    hi = rng.randint(20000, 60000, size=(n_obs // 2, n_vars)).astype(np.float32)
    dense = np.vstack([lo, hi])
    dense[rng.random_sample((n_obs, n_vars)) > 0.5] = 0.0
    adata = anndata.AnnData(X=sp.csr_matrix(dense))
    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path, shard_size=shard_size, row_group_rows=0)
    return path, n_obs


def test_subset_rewrite_reselects_per_shard_for_a_mixed_codec_source(tmp_dir):
    """A row subset must not force one codec onto every re-encoded shard.

    The file header's `codec_id` is only a default — each shard header
    overrides it (`docs/codec.md` §1) — so it cannot describe a source whose
    shards were chosen adaptively. This path used to inherit it anyway and pin
    every output shard to it. Both arms here are decode-encode (`shard_size`
    overridden away from the source's, so neither can byte-passthrough), which
    is what makes the comparison fair: a passthrough baseline would be copying
    bytes and would show a difference that says nothing about encoding.
    """
    import pyscx

    src, n_obs = _mixed_codec_source(tmp_dir)
    backed = pyscx.open(src).to_anndata(backed=True)

    no_subset = str(tmp_dir / "de_nosubset.scx")
    pyscx.from_anndata(backed, no_subset, shard_size=128, row_group_rows=0)

    keep = np.ones(n_obs, dtype=bool)
    keep[0] = False
    subset = str(tmp_dir / "de_subset.scx")
    pyscx.from_anndata(backed[keep], subset, shard_size=128, row_group_rows=0)

    a, b = os.path.getsize(no_subset), os.path.getsize(subset)
    # Dropping one row of 800 must not change the encoded size materially.
    assert b < a * 1.25, (
        f"subset rewrite {b} is disproportionate to the same rewrite without a "
        f"subset {a} (ratio {b / a:.2f}); a codec is being forced on shards it "
        f"does not suit"
    )


def test_subset_rewrite_preserves_a_uniform_source_codec(tmp_dir, synthetic_adata):
    """The converse guard: when every source shard agrees, keep that codec.

    Resolving against the shards must not become "always re-select" — a source
    written with an explicit codec should still round-trip through a subset
    rewrite with that codec, including `none` (the GDS fast path), which a
    blanket "ignore uninformative headers" rule would have silently compressed.
    """
    import pyscx

    for codec, expected in (("scx1", 1), ("none", 0)):
        src = str(tmp_dir / f"uniform_{codec}.scx")
        pyscx.from_anndata(synthetic_adata, src, codec=codec)
        assert pyscx.open(src).codec_id == expected

        backed = pyscx.open(src).to_anndata(backed=True)
        n_obs = backed.n_obs
        keep = np.ones(n_obs, dtype=bool)
        keep[0] = False
        dst = str(tmp_dir / f"uniform_{codec}_subset.scx")
        pyscx.from_anndata(backed[keep], dst, shard_size=32)

        out = pyscx.open(dst)
        assert out.n_obs == n_obs - 1
        assert out.codec_id == expected, (
            f"a uniform {codec} source should stay {codec} through a subset "
            f"rewrite; got codec_id {out.codec_id}"
        )


@pytest.mark.xfail(
    reason="Separate, pre-existing defect in `select_codec` "
    "(scx-format/src/codec_select.rs): it picks Scx1 on the sampled MEDIAN "
    "alone, and adaptive Rice has no escape code for outliers, so a shard "
    "whose median is small but which holds a few very large values encodes at "
    "O(value / 2**k) bits each. Shifting the row boundaries by one row is "
    "enough to put such a mix in one shard. Not the cause of the reported "
    "post-subset blowup (that was the file-header codec, fixed above) and not "
    "fixed here: the heuristic governs every write in the repo and changing it "
    "needs the benchmark gate.",
    strict=False,
)
def test_low_median_shard_with_large_outliers_does_not_blow_up(tmp_dir):
    """Reproducer for the Rice-coding outlier pathology, kept executable.

    With `shard_size=100` the subset shifts every row by one, so one shard
    ends up holding mostly 1-3 counts plus a few ~50,000 ones. Its median
    still routes it to Scx1, which then spends hundreds of bits per outlier.
    """
    import pyscx

    src, n_obs = _mixed_codec_source(tmp_dir, name="outlier.scx", shard_size=100)
    backed = pyscx.open(src).to_anndata(backed=True)

    no_subset = str(tmp_dir / "outlier_nosubset.scx")
    pyscx.from_anndata(backed, no_subset, shard_size=128, row_group_rows=0)

    keep = np.ones(n_obs, dtype=bool)
    keep[0] = False
    subset = str(tmp_dir / "outlier_subset.scx")
    # shard_size=100 keeps the source's boundaries, so the -1 row shift is
    # what creates the mixed-magnitude shard.
    pyscx.from_anndata(backed[keep], subset, shard_size=100, row_group_rows=0)

    a, b = os.path.getsize(no_subset), os.path.getsize(subset)
    assert b < a * 1.25, f"ratio {b / a:.2f} ({b} vs {a})"


# ---------------------------------------------------------------------------
# The production case: a file header that lies about compressed shards
# ---------------------------------------------------------------------------

# Byte offset of `codec_id` in the file header: magic[4] + format_version[2] +
# header_length[2] + flags[4] + n_obs[8] + n_vars[8] + nnz[8] + n_csr_shards[4]
# + n_csc_shards[4] + shard_target_rows[4]. See `FileHeader::write_to`.
_HEADER_CODEC_ID_OFFSET = 48


def _plant_lying_header(src, dst, claimed_codec=0):
    """Copy `src` to `dst` with the file header's `codec_id` overwritten.

    This is the shape the 2026-08-29 Replogle artifact had and the one no
    current writer can produce any more: header `none` over compressed shards.
    Reproducing it needs a byte poke, which is exactly why the regression
    shipped without a test. The reader dispatches on per-shard headers
    (`docs/codec.md` §1), so the file stays fully readable.
    """
    shutil.copy(src, dst)
    with open(dst, "r+b") as f:
        f.seek(_HEADER_CODEC_ID_OFFSET)
        f.write(bytes([claimed_codec]))
    return dst


def _lying_header_source(tmp_dir, name):
    """A compressed `scx1` file whose header claims `none`."""
    import pyscx

    rng = np.random.RandomState(21)
    n_obs, n_vars = 600, 40
    dense = rng.randint(1, 6, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random_sample((n_obs, n_vars)) > 0.5] = 0.0
    adata = anndata.AnnData(X=sp.csr_matrix(dense))

    honest = str(tmp_dir / f"{name}_honest.scx")
    pyscx.from_anndata(adata, honest, codec="scx1", shard_size=100)
    assert pyscx.open(honest).codec_id == 1, "premise: shards are scx1"

    lying = _plant_lying_header(honest, str(tmp_dir / f"{name}_lying.scx"))
    assert pyscx.open(lying).codec_id == 0, "premise: header now claims none"
    return honest, lying, n_obs


def test_subset_rewrite_ignores_a_header_that_claims_none(tmp_dir):
    """The Replogle case, as a test.

    Header says `none`, every shard is compressed. Inheriting the header wrote
    the whole matrix raw — 2.19 GB became 10.79 GB on the real artifact. The
    honest source and the lying one differ by one header byte, so their
    rewrites must come out the same size.
    """
    import pyscx

    honest, lying, n_obs = _lying_header_source(tmp_dir, "subset")
    keep = np.ones(n_obs, dtype=bool)
    keep[0] = False

    sizes = {}
    for tag, src in (("honest", honest), ("lying", lying)):
        backed = pyscx.open(src).to_anndata(backed=True)
        dst = str(tmp_dir / f"subset_out_{tag}.scx")
        pyscx.from_anndata(backed[keep], dst)
        out = pyscx.open(dst)
        assert out.n_obs == n_obs - 1
        assert out.codec_id != 0, (
            f"{tag}: output header claims `none`; the rewrite inherited a "
            "file-header codec instead of reading the source's shards"
        )
        sizes[tag] = os.path.getsize(dst)

    assert sizes["lying"] < sizes["honest"] * 1.25, (
        f"a one-byte header lie changed the output size: "
        f"lying={sizes['lying']} vs honest={sizes['honest']}"
    )


def test_lazy_rewrite_ignores_a_header_that_claims_none(tmp_dir):
    """Same fixture through the *lazy* route.

    `normalize_total` / `log1p` then `from_anndata` takes
    `route_scx_lazy_to_scx`, which pinned the encode codec to the file header
    exactly as the backed route did. A transformed rewrite of any
    streaming-converted file therefore wrote raw values.
    """
    import pyscx

    honest, lying, _ = _lying_header_source(tmp_dir, "lazy")

    sizes = {}
    for tag, src in (("honest", honest), ("lying", lying)):
        backed = pyscx.open(src).to_anndata(backed=True)
        pyscx.accel.normalize_total(backed, target_sum=1e4)
        assert type(backed.X).__name__ == "ScxLazyTransformedDataset", (
            "premise: the transform should leave X lazy"
        )
        dst = str(tmp_dir / f"lazy_out_{tag}.scx")
        pyscx.from_anndata(backed, dst)
        out = pyscx.open(dst)
        assert out.codec_id != 0, (
            f"{tag}: lazy rewrite output header claims `none` — the lazy route "
            "is still inheriting the source's file header"
        )
        sizes[tag] = os.path.getsize(dst)

    assert sizes["lying"] < sizes["honest"] * 1.25, (
        f"lazy: a one-byte header lie changed the output size: "
        f"lying={sizes['lying']} vs honest={sizes['honest']}"
    )


def test_explicit_codec_is_not_silently_dropped_by_passthrough(tmp_dir):
    """An explicit `codec=` must hold for every shard in the output.

    The passthrough gate compared the request against the *file header*, so on
    a mixed-codec source `codec="scx1"` took the byte-copy path and returned a
    file still holding `zstd` shards — the request silently ignored. `none`
    (the GDS fast path) is the case where that matters most.
    """
    import pyscx

    src, n_obs = _mixed_codec_source(tmp_dir, name="explicit_mixed.scx")
    # The bug needs the requested codec to EQUAL the source's file header while
    # the shards disagree with it: that is what made the gate say "already this
    # codec, copy the bytes". Requesting a codec the header does not name never
    # reached passthrough, so it would not have caught this.
    assert pyscx.open(src).codec_id == 1, "premise: header names scx1"
    backed = pyscx.open(src).to_anndata(backed=True)

    dst = str(tmp_dir / "explicit_forced.scx")
    pyscx.from_anndata(backed, dst, codec="scx1", row_group_rows=0)
    out = pyscx.open(dst)
    assert out.n_obs == n_obs

    # Provenance is the observable: a mixed source cannot honour an explicit
    # codec by copying bytes, so it must decode-encode.
    payload = next(
        json.loads(p["params_json"])
        for p in out.provenance()
        if json.loads(p["params_json"]).get("x_source") == "backed"
    )
    assert payload["passthrough"] is False, (
        "codec='scx1' was requested on a source whose shards are only half "
        "scx1, yet the rewrite byte-copied them; the explicit codec was "
        "silently dropped"
    )


def test_passthrough_does_not_carry_a_lying_header_forward(tmp_dir):
    """Byte-passthrough must not re-mint the false header.

    A no-subset backed rewrite copies shards verbatim and used to copy the
    source's `codec_id` with them, so a `none`-claiming header survived into
    the new file — which then fed the same bug to the next consumer. The copy
    paths now record the codec they copied, so `finish()` stamps the truth.
    """
    import pyscx

    honest, lying, n_obs = _lying_header_source(tmp_dir, "passthrough")

    backed = pyscx.open(lying).to_anndata(backed=True)
    dst = str(tmp_dir / "passthrough_out.scx")
    pyscx.from_anndata(backed, dst)

    out = pyscx.open(dst)
    assert out.n_obs == n_obs
    _assert_passthrough(dst)
    assert out.codec_id == 1, (
        "a verbatim copy of scx1 shards produced a header claiming "
        f"{out.codec_id}; the copy path is not recording the codec it copied"
    )
