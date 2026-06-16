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


def test_nb_glm_cpu_gpu_agreement():
    """Stage-A GPU fit matches the CPU f64 reference on the small fixture.

    Skips without a CUDA GPU; the GPU CI harness runs it on an H100. The CPU
    fitter is the reference (no external one exists — DE-GPU-ACC.md §10).
    """
    if not pyscx.accel.gpu_available():
        pytest.skip("no CUDA GPU available")
    # A larger synthetic fixture exercises the kernel across many genes.
    rng = np.random.default_rng(7)
    n_samples, n_genes = 6, 300
    base = rng.uniform(40, 120, size=n_genes)
    lfc = rng.uniform(-1.0, 1.0, size=n_genes)
    is_treat = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    mu = base[None, :] * np.exp(np.outer(is_treat, lfc))
    counts = rng.poisson(mu).astype(np.float64)
    design = np.column_stack([np.ones(n_samples), is_treat])

    cpu = pyscx.accel.nb_glm(counts, design, contrast=1, device="cpu")
    gpu = pyscx.accel.nb_glm(counts, design, contrast=1, device="gpu")

    c_lfc = cpu["log2FoldChange"].to_numpy()
    g_lfc = gpu["log2FoldChange"].to_numpy()
    finite = np.isfinite(c_lfc) & np.isfinite(g_lfc)
    rel = np.abs(c_lfc[finite] - g_lfc[finite]) / (np.abs(c_lfc[finite]) + 1e-6)
    # Bound set by nvcc --use_fast_math transcendentals (exp/log) + the Illinois
    # root-find's slightly different GPU convergence path — measured worst-case
    # ~1.5e-4 on this fixture. 2e-3 leaves margin without admitting a real
    # divergence (ranking Spearman is 1.000 regardless — see the pdex test).
    assert rel.max() <= 2e-3, f"max rel log2fc {rel.max()} > 2e-3"

    # p-values: the fast-math log2fc noise (~1.5e-4) is amplified through the
    # Wald normal-tail, so mid-range p-values diverge by ~2e-4 absolute
    # (measured). Rank concordance is the meaningful guard; the elementwise
    # tolerance is generous to tolerate that benign tail amplification.
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    c_p = cpu["pvalue"].to_numpy()
    g_p = gpu["pvalue"].to_numpy()
    pfin = np.isfinite(c_p) & np.isfinite(g_p)
    rho_p = spearmanr(c_p[pfin], g_p[pfin])[0]
    assert rho_p >= 0.999, f"p-value Spearman {rho_p} < 0.999"
    assert np.allclose(c_p[pfin], g_p[pfin], rtol=5e-3, atol=5e-3)
