"""Phase D + Phase E coverage for the multimodal API.

Phase D — `pyscx.from_mudata` round-trips a CITE-seq MuData object
through SCX and back via `to_mudata()`.

Phase E — `from_mudata` resolves codec="auto" via
`select_codec_for_modality`, so RNA shards land on Scx1 while the
Protein/ADT modality is force-overridden to Zstd.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


# CodecId enum mapping (mirrors scx_codec::CodecId).
SCX1 = 1
ZSTD = 2


@pytest.fixture
def cite_seq_mudata():
    """Tiny CITE-seq fixture with small-integer counts in both
    modalities (RNA + ADT). Both modalities use the same scipy CSR
    layout and similar value distributions, so any per-modality codec
    difference must come from the ModalityType override."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n_obs, rna_n_vars, adt_n_vars = 32, 80, 12

    # RNA: small UMI-like counts (median ≈ 1).
    rna_dense = rng.poisson(lam=0.4, size=(n_obs, rna_n_vars)).astype(np.float32)
    # ADT: similarly small counts so the *only* signal that flips the
    # codec is the modality_type, not the data magnitude.
    adt_dense = rng.poisson(lam=0.4, size=(n_obs, adt_n_vars)).astype(np.float32)

    rna_csr = sp.csr_matrix(rna_dense)
    adt_csr = sp.csr_matrix(adt_dense)

    rna_ad = anndata.AnnData(X=rna_csr)
    rna_ad.var_names = [f"g{i}" for i in range(rna_n_vars)]
    adt_ad = anndata.AnnData(X=adt_csr)
    adt_ad.var_names = [f"a{i}" for i in range(adt_n_vars)]

    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return mu


def test_from_mudata_round_trip(cite_seq_mudata):
    """Phase D: pyscx.from_mudata → pyscx.open(...).to_mudata() round-trips."""
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)

        reader = pyscx.open(path)
        assert reader.is_multimodal
        assert reader.n_modalities == 2
        assert sorted(reader.modality_names) == ["adt", "rna"]

        mu_back = reader.to_mudata()
        assert "rna" in mu_back.mod
        assert "adt" in mu_back.mod
        rna_orig = cite_seq_mudata.mod["rna"].X.toarray()
        rna_back = mu_back.mod["rna"].X.toarray()
        np.testing.assert_array_equal(rna_orig, rna_back)
        adt_orig = cite_seq_mudata.mod["adt"].X.toarray()
        adt_back = mu_back.mod["adt"].X.toarray()
        np.testing.assert_array_equal(adt_orig, adt_back)


def test_from_mudata_per_modality_codec_routing(cite_seq_mudata):
    """Phase E: with codec="auto", RNA picks Scx1 (small UMI median)
    while the Protein/ADT modality is overridden to Zstd by
    `select_codec_for_modality`, even though the underlying data
    distributions are similar.

    Asserts on `modality_info().default_codec_id`, which the h5mu
    pipeline writes as the same codec used for every CSR shard of
    that modality.
    """
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)  # codec="auto" by default

        reader = pyscx.open(path)
        rna_id = reader.modality_id("rna")
        adt_id = reader.modality_id("adt")
        assert rna_id is not None
        assert adt_id is not None

        rna_info = reader.modality_info(rna_id)
        adt_info = reader.modality_info(adt_id)
        assert rna_info is not None
        assert adt_info is not None

        assert rna_info["default_codec_id"] == SCX1, (
            f"RNA default codec should be Scx1 (id=1); got {rna_info['default_codec_id']}"
        )
        assert adt_info["default_codec_id"] == ZSTD, (
            f"ADT default codec should be Zstd (id=2); got {adt_info['default_codec_id']}"
        )
