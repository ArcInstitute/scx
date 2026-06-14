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
    "cooks",
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


def test_nb_glm_v2_filtering_option_keys_accepted():
    """The DESeq2 results-stage filtering knobs are recognised and toggle behavior."""
    counts, design = _pseudobulk_fixture()
    # All four new keys parse; explicit None for cooks_cutoff keeps the default.
    df = pyscx.accel.nb_glm(
        counts,
        design,
        contrast=1,
        options={
            "cooks_filtering": True,
            "cooks_cutoff": None,
            "independent_filtering": False,
            "independent_filter_alpha": 0.05,
        },
    )
    assert list(df.columns) == PYDESEQ2_COLUMNS
    # The `cooks` column is finite and non-negative for a clean, well-conditioned fit.
    cooks = df["cooks"].to_numpy()
    assert np.all(np.isfinite(cooks))
    assert np.all(cooks >= 0)


def _cooks_outlier_fixture():
    """8 samples (4 ctrl + 4 treated) × 6 genes, clean except a single gross spike
    in gene 3 (one sample). m−p = 6 supports Cook's filtering."""
    rng = np.random.default_rng(0)
    counts = rng.integers(80, 120, size=(8, 6)).astype(np.float64)
    design = np.array([[1, 0]] * 4 + [[1, 1]] * 4, dtype=np.float64)
    counts[5, 3] = 6000.0  # gross outlier
    return counts, design


def test_nb_glm_cooks_outlier_is_filtered():
    """The single-sample spike makes gene 3 the maximum-Cook's-distance gene, and a
    cutoff just below that maximum flags it: its p-value and p_adj become NaN
    (DESeq2 semantics) while a low-Cook's gene stays finite."""
    counts, design = _cooks_outlier_fixture()
    cooks = pyscx.accel.nb_glm(counts, design, contrast=1)["cooks"].to_numpy()
    imax = int(np.argmax(cooks))
    assert imax == 3, f"gene 3 (the spike) should have max Cook's: {cooks}"
    imin = int(np.argmin(cooks))
    # A cutoff just under the max flags the spike gene only.
    cut = cooks[imax] * 0.99
    df = pyscx.accel.nb_glm(
        counts, design, contrast=1, options={"cooks_cutoff": cut}
    )
    pval = df["pvalue"].to_numpy()
    padj = df["padj"].to_numpy()
    assert np.isnan(pval[imax]), "Cook's outlier should have NaN pvalue"
    assert np.isnan(padj[imax]), "Cook's outlier should have NaN padj"
    assert np.isfinite(pval[imin]), "low-Cook's gene should keep a finite pvalue"


def test_nb_glm_cooks_filtering_can_be_disabled():
    """With cooks_filtering=False, no gene's p-value is NaN'd even under a cutoff
    that would otherwise flag the outlier."""
    counts, design = _cooks_outlier_fixture()
    cooks = pyscx.accel.nb_glm(counts, design, contrast=1)["cooks"].to_numpy()
    cut = float(np.max(cooks)) * 0.5  # would flag the spike gene if filtering were on
    on = pyscx.accel.nb_glm(counts, design, contrast=1, options={"cooks_cutoff": cut})
    assert np.isnan(on["pvalue"].to_numpy()).any(), "low cutoff should filter on"
    off = pyscx.accel.nb_glm(
        counts,
        design,
        contrast=1,
        options={"cooks_filtering": False, "cooks_cutoff": cut},
    )
    assert np.all(
        np.isfinite(off["pvalue"].to_numpy())
    ), "cooks_filtering=False ⇒ no Cook's-driven NaN p-values"
