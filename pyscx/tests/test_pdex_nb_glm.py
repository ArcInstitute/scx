"""Tests for ``pyscx.accel.pdex_nb_glm`` (cell-eval/pdex path, §4.4) and the
``pseudobulk_dex(backend="nb_glm")`` route.

Covers the cell-eval polars schema, the replicate guard, the log1p guard, route
stamping, and Spearman parity vs ``pdex_ref`` on a stratified fixture (§17).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import pyscx

CELL_EVAL_COLUMNS = [
    "target",
    "feature",
    "fold_change",
    "p_value",
    "fdr",
    "log2_fold_change",
    "abs_log2_fold_change",
]

REFERENCE = "control"


def _perturb_adata(seed: int = 0, n_genes: int = 50, n_donors: int = 6, cells_per: int = 50):
    """Synthetic Perturb-seq counts: control + 3 KO perturbations × `n_donors`
    donors (the replicate stratifier). Each KO applies a graded, gene-specific
    multiplicative effect (distinct per gene) plus a mild per-donor batch factor,
    so both pseudobulk NB-GLM and per-cell pdex_ref recover a consistent gene
    ranking (for the Spearman parity check).

    Base expression is kept near-uniform across genes so per-gene statistical
    power is roughly constant: significance then tracks |effect| in both the
    per-cell (pdex_ref) and pseudobulk (NB-GLM) tests, which is what makes the
    `fdr` ranking concordant.
    """
    rng = np.random.default_rng(seed)
    perts = [REFERENCE, "ko_a", "ko_b", "ko_c"]
    donors = [f"d{j}" for j in range(n_donors)]

    # Near-constant base per-gene expression and a graded log2 effect ramp per KO.
    base = rng.uniform(45.0, 55.0, size=n_genes)
    ramp = np.linspace(-1.5, 1.5, n_genes)  # distinct per-gene log2 effects
    effect = {
        REFERENCE: np.ones(n_genes),
        "ko_a": 2.0 ** ramp,
        "ko_b": 2.0 ** (0.6 * ramp[::-1]),
        "ko_c": 2.0 ** (0.8 * ramp),
    }
    donor_factor = {d: rng.uniform(0.92, 1.08, size=n_genes) for d in donors}

    counts_blocks = []
    pert_labels = []
    donor_labels = []
    for p in perts:
        for d in donors:
            mean = base * effect[p] * donor_factor[d]
            block = rng.poisson(mean[None, :], size=(cells_per, n_genes)).astype(np.float32)
            counts_blocks.append(block)
            pert_labels.extend([p] * cells_per)
            donor_labels.extend([d] * cells_per)

    x = np.vstack(counts_blocks)
    obs = pd.DataFrame(
        {"perturbation": pert_labels, "donor": donor_labels},
        index=[f"cell_{i}" for i in range(x.shape[0])],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    return ad.AnnData(X=x, obs=obs, var=var)


def test_pdex_nb_glm_schema():
    pl = pytest.importorskip("polars")
    adata = _perturb_adata()
    df = pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"], min_cells_per_group=1
    )
    assert isinstance(df, pl.DataFrame)
    assert df.columns == CELL_EVAL_COLUMNS
    # Three non-reference targets, each with all genes.
    assert set(df["target"].to_list()) == {"ko_a", "ko_b", "ko_c"}
    assert df.height == 3 * adata.n_vars
    # fold_change == 2 ** log2_fold_change (finite rows).
    sub = df.filter(pl.col("log2_fold_change").is_finite())
    np.testing.assert_allclose(
        sub["fold_change"].to_numpy(),
        2.0 ** sub["log2_fold_change"].to_numpy(),
        rtol=1e-6,
    )


def test_pdex_nb_glm_requires_stratifier():
    adata = _perturb_adata()
    with pytest.raises(ValueError, match="stratify_by"):
        pyscx.accel.pdex_nb_glm(adata, "perturbation", REFERENCE)
    with pytest.raises(ValueError, match="stratify_by"):
        pyscx.accel.pdex_nb_glm(adata, "perturbation", REFERENCE, stratify_by=[])


def test_pdex_nb_glm_rejects_log1p():
    adata = _perturb_adata()
    with pytest.raises(ValueError, match="raw counts"):
        pyscx.accel.pdex_nb_glm(
            adata, "perturbation", REFERENCE, stratify_by=["donor"], is_log1p=True
        )
    # Auto-detect via adata.uns["log1p"].
    adata2 = _perturb_adata()
    adata2.uns["log1p"] = {"base": None}
    with pytest.raises(ValueError, match="raw counts"):
        pyscx.accel.pdex_nb_glm(
            adata2, "perturbation", REFERENCE, stratify_by=["donor"]
        )


def test_pdex_nb_glm_stamps_route():
    pytest.importorskip("polars")  # pdex_nb_glm emits the polars cell-eval schema
    adata = _perturb_adata()
    pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"], min_cells_per_group=1
    )
    info = adata.uns["scx_accel"]["pdex_nb_glm"]
    assert info["route"] == "cpu_nb_glm"
    assert info["fallback_reason"] == "none"


def test_pseudobulk_dex_nbglm_backend():
    adata = _perturb_adata()
    df = pyscx.accel.pseudobulk_dex(
        adata,
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        min_cells_per_group=1,
        backend="nb_glm",
    )
    # PyDESeq2-style pandas schema (same as the pydeseq2 backend).
    for col in ["gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj", "target", "reference"]:
        assert col in df.columns
    assert set(df["target"].unique()) == {"ko_a", "ko_b", "ko_c"}
    assert (df["reference"] == REFERENCE).all()
    # Route stamped as the native NB-GLM CPU path.
    assert adata.uns["scx_accel"]["pseudobulk_dex"]["route"] == "cpu_nb_glm"


def test_pdex_nb_glm_spearman_parity_vs_pdex_ref():
    pl = pytest.importorskip("polars")
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    adata = _perturb_adata()

    nb = pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"], min_cells_per_group=1
    ).to_pandas()
    ref = pyscx.accel.pdex_ref(
        adata, "perturbation", reference=REFERENCE, is_log1p=False, output="polars"
    ).to_pandas()

    key = ["target", "feature"]
    merged = nb.merge(ref, on=key, suffixes=("_nb", "_ref"))
    assert len(merged) == 3 * adata.n_vars

    rho_lfc = spearmanr(merged["log2_fold_change_nb"], merged["log2_fold_change_ref"]).statistic
    rho_fdr = spearmanr(merged["fdr_nb"], merged["fdr_ref"]).statistic
    assert rho_lfc >= 0.95, f"log2fc Spearman {rho_lfc} < 0.95"
    assert rho_fdr >= 0.95, f"fdr Spearman {rho_fdr} < 0.95"
