"""Tests for the direct ``pyscx.accel.nb_glm`` binding (DESeq2-replacement, §4.3).

Operates on already-pseudobulked matrices; returns PyDESeq2-style pandas columns.
"""

import numpy as np
import pytest

import pyscx

PYDESEQ2_COLUMNS = [
    "gene",
    "baseMean",
    "log2FoldChange",
    "lfcSE",
    "stat",
    "pvalue",
    "padj",
    "dispersion",
    "converged",
    "n_iter",
]


def _pseudobulk_fixture():
    """6 pseudobulk samples (3 control, 3 treated) × 5 genes.

    Gene 0 up ~2x in treated, gene 1 down ~0.5x, genes 2–4 flat. Design columns
    are [intercept, treatment].
    """
    counts = np.array(
        [
            # ctrl                      treated
            [100, 200, 50, 80, 120],  # sample 0 (ctrl)
            [110, 190, 55, 78, 115],  # sample 1 (ctrl)
            [95, 210, 48, 82, 125],  # sample 2 (ctrl)
            [205, 100, 52, 79, 118],  # sample 3 (treated): g0 up, g1 down
            [195, 105, 49, 81, 122],  # sample 4 (treated)
            [210, 95, 51, 80, 119],  # sample 5 (treated)
        ],
        dtype=np.float64,
    )
    design = np.array(
        [[1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]], dtype=np.float64
    )
    return counts, design


def test_nb_glm_columns_and_signs():
    counts, design = _pseudobulk_fixture()
    df = pyscx.accel.nb_glm(counts, design, contrast=1)
    assert list(df.columns) == PYDESEQ2_COLUMNS
    assert len(df) == 5
    lfc = df["log2FoldChange"].to_numpy()
    assert lfc[0] > 0.5, f"gene 0 should be up: {lfc[0]}"
    assert lfc[1] < -0.5, f"gene 1 should be down: {lfc[1]}"
    assert abs(lfc[2]) < 0.3, f"gene 2 should be flat: {lfc[2]}"
    # Well-conditioned fit: no non-finite effects/p-values.
    assert np.all(np.isfinite(lfc))
    assert np.all(np.isfinite(df["pvalue"].to_numpy()))
    # p-values in [0, 1]; dispersion non-negative.
    p = df["pvalue"].to_numpy()
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(df["dispersion"].to_numpy() >= 0)


def test_nb_glm_default_contrast_is_last_coefficient():
    counts, design = _pseudobulk_fixture()
    # No contrast → last coefficient (the treatment column) by DESeq2 convention.
    df_default = pyscx.accel.nb_glm(counts, design)
    df_idx1 = pyscx.accel.nb_glm(counts, design, contrast=1)
    np.testing.assert_allclose(
        df_default["log2FoldChange"].to_numpy(),
        df_idx1["log2FoldChange"].to_numpy(),
    )


def test_nb_glm_counts_axis_orientations_agree():
    counts, design = _pseudobulk_fixture()
    df_sbg = pyscx.accel.nb_glm(counts, design, contrast=1, counts_axis="samples_by_genes")
    df_gbs = pyscx.accel.nb_glm(
        counts.T.copy(), design, contrast=1, counts_axis="genes_by_samples"
    )
    np.testing.assert_allclose(
        df_sbg["log2FoldChange"].to_numpy(), df_gbs["log2FoldChange"].to_numpy()
    )


def test_nb_glm_contrast_vector_matches_index():
    counts, design = _pseudobulk_fixture()
    df_idx = pyscx.accel.nb_glm(counts, design, contrast=1)
    df_vec = pyscx.accel.nb_glm(counts, design, contrast=[0.0, 1.0])
    np.testing.assert_allclose(
        df_idx["log2FoldChange"].to_numpy(), df_vec["log2FoldChange"].to_numpy()
    )


def test_nb_glm_gene_names_passthrough():
    counts, design = _pseudobulk_fixture()
    names = [f"ENSG{i}" for i in range(5)]
    df = pyscx.accel.nb_glm(counts, design, contrast=1, gene_names=names)
    assert df["gene"].tolist() == names


def test_nb_glm_too_few_samples_errors():
    # 2 samples, 2 design columns ⇒ no residual df. AccelError → RuntimeError.
    counts = np.array([[10, 20], [30, 40]], dtype=np.float64)
    design = np.array([[1, 0], [1, 1]], dtype=np.float64)
    with pytest.raises(RuntimeError):
        pyscx.accel.nb_glm(counts, design, contrast=1)


def test_nb_glm_input_validation_errors():
    counts, design = _pseudobulk_fixture()  # 6 samples × 5 genes, design [6 × 2]
    # gene_names length mismatch.
    with pytest.raises(ValueError):
        pyscx.accel.nb_glm(counts, design, contrast=1, gene_names=["a", "b"])
    # invalid counts_axis.
    with pytest.raises(ValueError):
        pyscx.accel.nb_glm(counts, design, contrast=1, counts_axis="genes")
    # contrast weight-vector length != n_features (2).
    with pytest.raises(ValueError):
        pyscx.accel.nb_glm(counts, design, contrast=[1.0, 0.0, 0.0])
    # contrast coefficient index out of range.
    with pytest.raises(ValueError):
        pyscx.accel.nb_glm(counts, design, contrast=5)
    # unknown options key fails loudly.
    with pytest.raises(ValueError):
        pyscx.accel.nb_glm(counts, design, contrast=1, options={"max_outer_iter": 5})


def test_nb_glm_options_dict():
    counts, design = _pseudobulk_fixture()
    # Moments dispersion skips the trend/shrinkage post-pass; must still run.
    df = pyscx.accel.nb_glm(
        counts, design, contrast=1, options={"dispersion": "moments"}
    )
    assert list(df.columns) == PYDESEQ2_COLUMNS
    assert df["log2FoldChange"].to_numpy()[0] > 0.5
