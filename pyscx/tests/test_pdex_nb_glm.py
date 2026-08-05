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
    """The cell-eval column schema, in the default container (pandas, F6)."""
    adata = _perturb_adata()
    df = pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"], min_cells_per_group=1
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == CELL_EVAL_COLUMNS
    # Three non-reference targets, each with all genes.
    assert set(df["target"]) == {"ko_a", "ko_b", "ko_c"}
    assert len(df) == 3 * adata.n_vars
    # fold_change == 2 ** log2_fold_change (finite rows).
    sub = df[np.isfinite(df["log2_fold_change"])]
    np.testing.assert_allclose(
        sub["fold_change"].to_numpy(),
        2.0 ** sub["log2_fold_change"].to_numpy(),
        rtol=1e-6,
    )


def test_pdex_nb_glm_output_polars():
    """F6: polars is the opt-in for cell-eval, and both containers carry
    identical columns and values."""
    pl = pytest.importorskip("polars")
    adata = _perturb_adata()
    kw = dict(stratify_by=["donor"], min_cells_per_group=1)

    df_default = pyscx.accel.pdex_nb_glm(adata, "perturbation", REFERENCE, **kw)
    df_pl = pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, output="polars", **kw
    )

    assert isinstance(df_default, pd.DataFrame)
    assert isinstance(df_pl, pl.DataFrame)
    assert list(df_pl.columns) == CELL_EVAL_COLUMNS
    pd.testing.assert_frame_equal(
        df_default.reset_index(drop=True),
        df_pl.to_pandas().reset_index(drop=True),
    )
    # A typo is rejected up front, before the fit is paid for.
    with pytest.raises(ValueError, match="expected 'polars' or 'pandas'"):
        pyscx.accel.pdex_nb_glm(
            adata, "perturbation", REFERENCE, output="bogus", **kw
        )


def test_pdex_nb_glm_requires_stratifier():
    adata = _perturb_adata()
    with pytest.raises(ValueError, match="stratify_by"):
        pyscx.accel.pdex_nb_glm(adata, "perturbation", REFERENCE)
    with pytest.raises(ValueError, match="stratify_by"):
        pyscx.accel.pdex_nb_glm(adata, "perturbation", REFERENCE, stratify_by=[])
    # A bare string (not a list) gets a clear message, not the opaque PyO3
    # "Can't extract 'str' to 'Vec'" type error.
    with pytest.raises(ValueError, match="list of obs column"):
        pyscx.accel.pdex_nb_glm(adata, "perturbation", REFERENCE, stratify_by="donor")


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
    adata = _perturb_adata()
    # Explicit device="cpu" is host-independent: route=cpu_nb_glm, the planner
    # records the deliberate CPU choice as user_forced_cpu. (With the default
    # device="auto", the reason is no_cuda on a CPU host but "none" on a GPU
    # host, where it would instead route to gpu_nb_glm_csr.)
    pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"],
        min_cells_per_group=1, device="cpu",
    )
    info = adata.uns["scx_accel"]["pdex_nb_glm"]
    assert info["route"] == "cpu_nb_glm"
    assert info["fallback_reason"] == "user_forced_cpu"


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


def _lfc_by_gene(df, target):
    """Per-gene log2FoldChange for one target, indexed by gene (sorted)."""
    sub = df[df["target"] == target].set_index("gene")["log2FoldChange"]
    return sub.sort_index()


def test_pseudobulk_dex_nbglm_honors_custom_design():
    """§3.11: backend='nb_glm' now honors a `design` formula (formulaic) instead
    of rejecting it — and the design must actually change the fit (never a silent
    no-op vs the fixed intercept+target default)."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    common = dict(
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        min_cells_per_group=1,
        backend="nb_glm",
    )
    default = pyscx.accel.pseudobulk_dex(adata, **common)
    designed = pyscx.accel.pseudobulk_dex(adata, design="~ perturbation + donor", **common)

    # Same schema + target/reference set as the default path.
    for col in ["gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj", "target", "reference"]:
        assert col in designed.columns
    assert set(designed["target"].unique()) == {"ko_a", "ko_b", "ko_c"}
    assert (designed["reference"] == REFERENCE).all()
    assert adata.uns["scx_accel"]["pseudobulk_dex"]["route"] == "cpu_nb_glm"

    # Donor-adjusted, shared-dispersion fit differs from the per-pair default:
    # the supplied design is consumed, not ignored.
    a = _lfc_by_gene(default, "ko_a")
    b = _lfc_by_gene(designed, "ko_a")
    assert np.nanmax(np.abs(a.to_numpy() - b.to_numpy())) > 1e-3


def test_pseudobulk_dex_nbglm_covariate_adjustment_is_real():
    """`~ perturbation + donor` must differ from `~ perturbation`: the covariate
    enters the model (the fixture has a per-donor batch factor)."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    common = dict(
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        min_cells_per_group=1,
        backend="nb_glm",
    )
    no_cov = pyscx.accel.pseudobulk_dex(adata, design="~ perturbation", **common)
    with_cov = pyscx.accel.pseudobulk_dex(adata, design="~ perturbation + donor", **common)
    # The covariate mostly moves *precision* (donor adjustment), so assert on the
    # Wald statistic for large headroom rather than on the point estimate.
    a = no_cov[no_cov["target"] == "ko_a"].set_index("gene")["stat"].sort_index().to_numpy()
    b = with_cov[with_cov["target"] == "ko_a"].set_index("gene")["stat"].sort_index().to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.nanmax(np.abs(a[ok] - b[ok])) > 1.0


def test_pseudobulk_dex_nbglm_reference_coding_sign_flip():
    """`reference` sets the base level: swapping reference and target negates the
    contrast (target-vs-reference), confirming `test_col[T.<target>]` coding."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    base = dict(
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        design="~ perturbation",
        min_cells_per_group=1,
        backend="nb_glm",
    )
    fwd = pyscx.accel.pseudobulk_dex(adata, reference="control", **base)
    rev = pyscx.accel.pseudobulk_dex(adata, reference="ko_a", **base)
    koa_vs_ctrl = _lfc_by_gene(fwd, "ko_a").to_numpy()
    ctrl_vs_koa = _lfc_by_gene(rev, "control").to_numpy()
    # Same model, reparameterized: the contrast flips sign near-exactly (this also
    # doubles as a numeric-stability regression).
    assert np.allclose(koa_vs_ctrl, -ctrl_vs_koa, atol=1e-6, rtol=1e-6)


def test_pseudobulk_dex_nbglm_explicit_contrast():
    """An explicit `contrast` (int index or weight vector) in nbglm_options tests
    a specific coefficient; index and equivalent weight vector agree; the default
    formula is `~ test_col` when only a contrast is given."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    common = dict(
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        min_cells_per_group=1,
        backend="nb_glm",
    )
    # Columns for `~ perturbation` (base=control): [Intercept, ko_a, ko_b, ko_c].
    idx = pyscx.accel.pseudobulk_dex(adata, nbglm_options={"contrast": 1}, **common)
    vec = pyscx.accel.pseudobulk_dex(
        adata, nbglm_options={"contrast": [0.0, 1.0, 0.0, 0.0]}, **common
    )
    # A single explicit contrast → one result block.
    assert idx["target"].nunique() == 1
    assert np.allclose(
        idx.sort_values("gene")["log2FoldChange"].to_numpy(),
        vec.sort_values("gene")["log2FoldChange"].to_numpy(),
        equal_nan=True,
    )


def test_pseudobulk_dex_nbglm_design_missing_test_col():
    """A design that omits `test_col` cannot produce the per-target contrast → a
    clear ValueError (argument validation), not a silent wrong result."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    with pytest.raises(ValueError, match="does not produce a unique treatment"):
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference=REFERENCE,
            design="~ donor",
            min_cells_per_group=1,
            backend="nb_glm",
        )


def test_pseudobulk_dex_nbglm_design_treatment_coded_test_col():
    """The documented advanced form `C(test_col, ...)` resolves via the suffix
    fallback (not only the bare `test_col[T.level]` name)."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    df = pyscx.accel.pseudobulk_dex(
        adata,
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        design="~ C(perturbation) + donor",
        min_cells_per_group=1,
        backend="nb_glm",
    )
    assert set(df["target"].unique()) == {"ko_a", "ko_b", "ko_c"}


def test_pseudobulk_dex_nbglm_design_unknown_column():
    """A formula referencing a column not in `groupby` fails loudly (formulaic)."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    with pytest.raises(Exception, match="nonexistent_col"):  # noqa: B017
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference=REFERENCE,
            design="~ perturbation + nonexistent_col",
            min_cells_per_group=1,
            backend="nb_glm",
        )


def test_pseudobulk_dex_nbglm_design_requires_formulaic(monkeypatch):
    """When a `design` is supplied but formulaic is unavailable, fail loud with an
    actionable ImportError-derived message (catchable via `except ImportError`)."""
    import sys

    adata = _perturb_adata()
    # Poison the import so `py.import("formulaic")` raises ImportError.
    monkeypatch.setitem(sys.modules, "formulaic", None)
    with pytest.raises(ImportError, match="formulaic is required"):
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference=REFERENCE,
            design="~ perturbation",
            min_cells_per_group=1,
            backend="nb_glm",
        )


def test_pdex_nb_glm_design_smoke():
    """pdex_nb_glm(design=...) fits a covariate-adjusted model and emits the
    cell-eval column schema."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    df = pyscx.accel.pdex_nb_glm(
        adata,
        "perturbation",
        REFERENCE,
        stratify_by=["donor"],
        min_cells_per_group=1,
        design="~ perturbation + donor",
    )
    assert list(df.columns) == CELL_EVAL_COLUMNS
    assert set(df["target"].unique()) == {"ko_a", "ko_b", "ko_c"}
    # A custom design runs on CPU (the GPU kernel is p ≤ 8), so the route stamp
    # must honestly say cpu_nb_glm rather than over-claiming a GPU route.
    assert adata.uns["scx_accel"]["pdex_nb_glm"]["route"] == "cpu_nb_glm"


def test_pdex_nb_glm_rejects_explicit_contrast():
    """pdex_nb_glm emits the cell-eval schema keyed by perturbation name, so an
    explicit contrast (whose coefficient-name target has no perturbation to join
    against) is rejected — pointing the user to pseudobulk_dex / accel.nb_glm."""
    pytest.importorskip("formulaic")
    adata = _perturb_adata()
    with pytest.raises(ValueError, match="does not accept an explicit"):
        pyscx.accel.pdex_nb_glm(
            adata,
            "perturbation",
            REFERENCE,
            stratify_by=["donor"],
            min_cells_per_group=1,
            nbglm_options={"contrast": 1},
        )


def test_pseudobulk_dex_nbglm_recovers_true_lfc():
    """Ground-truth correctness (not just "differs"): the fixture builds
    ``effect["ko_a"] = 2.0 ** ramp`` with ``ramp = linspace(-1.5, 1.5, n_genes)``,
    so the true ko_a log2FC *is* ``ramp``. A design-aware fit must recover it —
    this catches monotone bugs (log-base, halved LFC) that a rank correlation
    would miss. Runs in CI (needs only formulaic)."""
    pytest.importorskip("formulaic")
    n_genes = 60
    adata = _perturb_adata(n_genes=n_genes)
    ramp = np.linspace(-1.5, 1.5, n_genes)
    df = pyscx.accel.pseudobulk_dex(
        adata,
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        design="~ perturbation + donor",
        min_cells_per_group=1,
        backend="nb_glm",
    )
    # Reindex the ko_a estimate onto gene order gene_0..gene_{n-1} == ramp order.
    est = df[df["target"] == "ko_a"].set_index("gene")["log2FoldChange"]
    est = est.reindex([f"gene_{j}" for j in range(n_genes)]).to_numpy()
    ok = np.isfinite(est)
    slope, intercept = np.polyfit(ramp[ok], est[ok], 1)
    pearson = np.corrcoef(ramp[ok], est[ok])[0, 1]
    assert pearson > 0.99, pearson
    assert 0.9 < slope < 1.1, slope  # log2 base + unit LFC scale (catches ln / ½×)
    assert np.max(np.abs(est[ok] - ramp[ok])) < 0.25


def test_pseudobulk_dex_nbglm_parity_vs_pydeseq2():
    """The nb_glm formula path and the pydeseq2 backend agree on the same design,
    on the *magnitude* (not just rank) of the effect — DESeq2-*style*, not
    -*identical*, so a loose but scale-sensitive tolerance."""
    pytest.importorskip("formulaic")
    pytest.importorskip("pydeseq2")
    adata = _perturb_adata()
    common = dict(
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        design="~ perturbation + donor",
        min_cells_per_group=1,
    )
    nb = pyscx.accel.pseudobulk_dex(adata, backend="nb_glm", **common)
    dd = pyscx.accel.pseudobulk_dex(adata, backend="pydeseq2", n_cpus=1, **common)
    a = _lfc_by_gene(nb, "ko_a").to_numpy()
    b = _lfc_by_gene(dd, "ko_a").to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    pearson = np.corrcoef(a[ok], b[ok])[0, 1]
    assert pearson > 0.99, pearson
    assert np.max(np.abs(a[ok] - b[ok])) < 0.3


def test_pseudobulk_dex_nbglm_requires_sum_aggregation():
    """§2.5: the NB count likelihood is defined on summed integer counts, so
    the nb_glm backend must reject mean (fractional) aggregation."""
    adata = _perturb_adata()
    with pytest.raises(ValueError, match='requires aggr_method="sum"'):
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference=REFERENCE,
            aggr_method="mean",
            min_cells_per_group=1,
            backend="nb_glm",
        )


def test_nb_glm_rejects_fractional_counts():
    """§2.5: the low-level nb_glm fitter shares the count-contract validator —
    fractional (non-integer) counts are rejected."""
    # 4 samples × 1 gene (default counts_axis="samples_by_genes"); sample 1 is fractional.
    counts = np.array([[10.0], [20.5], [30.0], [40.0]], dtype=np.float64)
    design = np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="integer count"):
        pyscx.accel.nb_glm(counts, design, contrast=1)
    # Integer-valued counts pass the same validator.
    counts_ok = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float64)
    pyscx.accel.nb_glm(counts_ok, design, contrast=1)


def test_pdex_nb_glm_spearman_parity_vs_pdex_ref():
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    adata = _perturb_adata()

    nb = pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"], min_cells_per_group=1
    )
    ref = pyscx.accel.pdex_ref(
        adata, "perturbation", reference=REFERENCE, is_log1p=False
    )

    key = ["target", "feature"]
    merged = nb.merge(ref, on=key, suffixes=("_nb", "_ref"))
    assert len(merged) == 3 * adata.n_vars

    # Index [0] (not `.statistic`) for SciPy < 1.10 backward compatibility.
    rho_lfc = spearmanr(merged["log2_fold_change_nb"], merged["log2_fold_change_ref"])[0]
    rho_fdr = spearmanr(merged["fdr_nb"], merged["fdr_ref"])[0]
    assert rho_lfc >= 0.95, f"log2fc Spearman {rho_lfc} < 0.95"
    assert rho_fdr >= 0.95, f"fdr Spearman {rho_fdr} < 0.95"


# --- GPU (Stage A) -----------------------------------------------------------
#
# No external reference exists (pyDESeq2 OOMs in the comprehensive correctness
# bench), so the CPU f64 fitter is the reference. These skip when no CUDA GPU is
# present; the GPU CI harness (slurm_scx_gpu_tests.sh) runs them on an H100.


def test_pdex_nb_glm_gpu_route_stamped():
    """device="gpu" on a GPU host records the gpu_nb_glm_csr route."""
    if not pyscx.accel.gpu_available():
        pytest.skip("no CUDA GPU available")
    adata = _perturb_adata()
    pyscx.accel.pdex_nb_glm(
        adata,
        "perturbation",
        REFERENCE,
        stratify_by=["donor"],
        min_cells_per_group=1,
        device="gpu",
    )
    info = adata.uns["scx_accel"]["pdex_nb_glm"]
    assert info["route"] == "gpu_nb_glm_csr", info
    assert info["fallback_reason"] == "none"


def test_pdex_nb_glm_cpu_gpu_agreement():
    """GPU matches CPU within per-quantity relative tolerances + rank concordance."""
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    if not pyscx.accel.gpu_available():
        pytest.skip("no CUDA GPU available")
    adata = _perturb_adata()

    cpu = pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"],
        min_cells_per_group=1, device="cpu",
    )
    gpu = pyscx.accel.pdex_nb_glm(
        adata, "perturbation", REFERENCE, stratify_by=["donor"],
        min_cells_per_group=1, device="gpu",
    )

    key = ["target", "feature"]
    m = cpu.merge(gpu, on=key, suffixes=("_cpu", "_gpu"))
    assert len(m) == 3 * adata.n_vars

    # Rank concordance: the headline agreement metric.
    rho = spearmanr(m["log2_fold_change_cpu"], m["log2_fold_change_gpu"])[0]
    assert rho >= 0.999, f"CPU↔GPU log2fc Spearman {rho} < 0.999"

    # Per-quantity relative tolerances.
    finite = np.isfinite(m["log2_fold_change_cpu"]) & np.isfinite(m["log2_fold_change_gpu"])
    rel_lfc = np.abs(
        m["log2_fold_change_cpu"][finite] - m["log2_fold_change_gpu"][finite]
    ) / (np.abs(m["log2_fold_change_cpu"][finite]) + 1e-6)
    # 2e-3: nvcc --use_fast_math transcendentals set a ~1e-4 floor (ranking
    # Spearman stays 1.000 — asserted above).
    assert rel_lfc.max() <= 2e-3, f"max rel log2fc {rel_lfc.max()} > 2e-3"

    pfin = np.isfinite(m["p_value_cpu"]) & np.isfinite(m["p_value_gpu"])
    assert np.allclose(
        m["p_value_cpu"][pfin], m["p_value_gpu"][pfin], rtol=1e-3, atol=1e-4
    )
