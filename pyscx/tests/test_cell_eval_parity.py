"""Cell-eval / arc-bench parity validation tests.

Compares SCX-accelerated metric implementations head-to-head against the
Python reference implementations in cell-eval and arc-bench. Every Tier 1
metric must match the Python reference within specified tolerance on the
same input data before the Rust accelerators can replace the Python codepath.

Run from within the scx-bench-eval conda environment:
    conda activate scx-bench-eval
    cd pyscx && maturin develop --release && cd ..
    pytest pyscx/tests/test_cell_eval_parity.py -v

External dependencies (only available in scx-bench-eval):
    - cell-eval  (/home/nickyoungblut/dev/python/cell-eval)
    - arc-bench  (/home/nickyoungblut/dev/python/arc-bench)
    - pdex, polars, tqdm
"""

import time

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

# Guard: skip entire module if cell-eval / arc-bench are not installed.
cell_eval = pytest.importorskip("cell_eval", reason="cell-eval not installed (need scx-bench-eval env)")
arc_bench = pytest.importorskip("arc_bench", reason="arc-bench not installed (need scx-bench-eval env)")
pl = pytest.importorskip("polars", reason="polars not installed (need scx-bench-eval env)")

import pyscx  # noqa: E402
from cell_eval import PerturbationAnndataPair, score_agg_metrics  # noqa: E402
from cell_eval.metrics._anndata import (  # noqa: E402
    ClusteringAgreement,
    discrimination_score as ce_discrimination_score,
    edistance as ce_edistance,
    mae as ce_mae,
    mae_delta as ce_mae_delta,
    mse as ce_mse,
    mse_delta as ce_mse_delta,
    pearson_delta as ce_pearson_delta,
)
from arc_bench.tools.normalize_transform.core import (  # noqa: E402
    compute_control_baseline,
    compute_knockdown_efficiency,
    compute_log_deviation,
)
from sklearn.metrics import (  # noqa: E402
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# =============================================================================
# Shared synthetic dataset
# =============================================================================

def _make_cell_eval_adata(
    n_obs=500, n_vars=100, n_perts=8, seed=42, as_sparse=True,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Paired real/pred AnnData matching cell-eval conventions.

    - obs column: "perturbation" (matches CANONICAL_PERTURBATION_COL)
    - control label: "control" (matches CANONICAL_CONTROL_LABEL)
    - Perturbation names match gene names (gene_0..gene_{n_perts-2})
      so knockdown_efficiency can look up target genes
    - Normalize-total + log1p applied (cell-eval expects lognorm input)
    - Predicted data = real + Gaussian noise (correlated but imperfect)

    Key constraints:
    - Gene names must match perturbation names for n_perts - 1 entries
      (excluding control) — required for knockdown/discrimination
      target-gene exclusion tests
    - Data must pass cell_eval.utils.guess_is_lognorm() (values in
      [0, 15), has fractional component)
    - At least 20 cells per perturbation (required for stable pseudobulk)
    - Both real and pred must share the same var_names (cell-eval validates
      this in PerturbationAnndataPair.__init__)
    """
    import pandas as pd
    import scanpy as sc

    rng = np.random.default_rng(seed)

    # Gene names: first n_perts-1 genes match perturbation names
    pert_names = ["control"] + [f"gene_{i}" for i in range(n_perts - 1)]
    gene_names = [f"gene_{i}" for i in range(n_vars)]

    # Verify gene names match perturbation names for n_perts-1 entries
    for pname in pert_names[1:]:
        assert pname in gene_names, (
            f"Perturbation '{pname}' not in gene_names — need n_vars >= n_perts-1"
        )

    # Ensure at least 20 cells per perturbation
    cells_per_pert = max(20, n_obs // n_perts)
    n_obs = cells_per_pert * n_perts  # Adjust to be evenly divisible

    # Assign cells to perturbations
    labels = []
    for name in pert_names:
        labels.extend([name] * cells_per_pert)

    # Base expression (count-like integers)
    base = rng.exponential(5.0, size=n_vars).astype(np.float32)

    # Perturbation-specific effects
    deltas = {}
    for name in pert_names[1:]:
        deltas[name] = rng.normal(0, 2, size=n_vars).astype(np.float32)
        # Make the target gene's knockdown visible
        gene_idx = gene_names.index(name)
        deltas[name][gene_idx] = -base[gene_idx] * 0.7  # 70% knockdown

    # Build real expression (raw counts)
    X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
    for i, label in enumerate(labels):
        noise = rng.poisson(0.5, size=n_vars).astype(np.float32)
        if label == "control":
            X_real[i] = np.maximum(base + noise, 0)
        else:
            X_real[i] = np.maximum(base + deltas[label] + noise, 0)
    # Round to integer-ish counts (cell-eval will normalize+log1p)
    X_real = np.round(X_real).astype(np.float32)

    # Build predicted expression (noisy version of real)
    X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
    for i, label in enumerate(labels):
        noise = rng.poisson(0.5, size=n_vars).astype(np.float32)
        if label == "control":
            X_pred[i] = np.maximum(base + noise, 0)
        else:
            pred_delta = deltas[label] + rng.normal(
                0, 0.5, size=n_vars
            ).astype(np.float32)
            X_pred[i] = np.maximum(base + pred_delta + noise, 0)
    X_pred = np.round(X_pred).astype(np.float32)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=gene_names)

    adata_real = ad.AnnData(X=X_real, obs=obs.copy(), var=var.copy())
    adata_pred = ad.AnnData(X=X_pred, obs=obs.copy(), var=var.copy())

    # Normalize + log1p (cell-eval expects lognorm input)
    sc.pp.normalize_total(adata_real)
    sc.pp.log1p(adata_real)
    sc.pp.normalize_total(adata_pred)
    sc.pp.log1p(adata_pred)

    # Self-check: data must pass cell-eval's lognorm detection
    from cell_eval.utils import guess_is_lognorm
    assert guess_is_lognorm(adata_real, validate=True), (
        "Synthetic real data failed guess_is_lognorm — values may be "
        "outside [0, 15) or lack fractional component"
    )
    assert guess_is_lognorm(adata_pred, validate=True), (
        "Synthetic pred data failed guess_is_lognorm"
    )

    # Self-check: var_names must match between real and pred
    assert list(adata_real.var_names) == list(adata_pred.var_names), (
        "var_names mismatch between real and pred AnnData"
    )

    if as_sparse:
        adata_real.X = sp.csr_matrix(adata_real.X)
        adata_pred.X = sp.csr_matrix(adata_pred.X)

    return adata_real, adata_pred


def _make_raw_count_adata(
    n_obs=500, n_vars=100, n_perts=8, seed=42,
) -> ad.AnnData:
    """Raw-count AnnData for knockdown efficiency tests (NOT log1p).

    Perturbation names match gene names so knockdown can find target genes.
    Returns raw integer-ish counts (normalize_total NOT applied).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    pert_names = ["control"] + [f"gene_{i}" for i in range(n_perts - 1)]
    gene_names = [f"gene_{i}" for i in range(n_vars)]

    cells_per_pert = max(20, n_obs // n_perts)
    n_obs = cells_per_pert * n_perts

    labels = []
    for name in pert_names:
        labels.extend([name] * cells_per_pert)

    base = rng.exponential(5.0, size=n_vars).astype(np.float32)

    deltas = {}
    for name in pert_names[1:]:
        deltas[name] = rng.normal(0, 1, size=n_vars).astype(np.float32)
        gene_idx = gene_names.index(name)
        deltas[name][gene_idx] = -base[gene_idx] * 0.7

    X = np.zeros((n_obs, n_vars), dtype=np.float32)
    for i, label in enumerate(labels):
        noise = rng.poisson(0.5, size=n_vars).astype(np.float32)
        if label == "control":
            X[i] = np.maximum(base + noise, 0)
        else:
            X[i] = np.maximum(base + deltas[label] + noise, 0)

    X = np.round(X).astype(np.float32)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


def _build_pair(
    adata_real: ad.AnnData,
    adata_pred: ad.AnnData,
) -> PerturbationAnndataPair:
    """Build a cell-eval PerturbationAnndataPair from real/pred AnnData."""
    return PerturbationAnndataPair(
        real=adata_real,
        pred=adata_pred,
        pert_col="perturbation",
        control_pert="control",
    )


# =============================================================================
# Shared synthetic dataset validation
# =============================================================================

class TestSharedDataset:
    """Validate _make_cell_eval_adata() meets all constraints."""

    def test_gene_names_match_perturbation_names(self):
        """Gene names must match perturbation names for n_perts-1 entries."""
        adata_real, adata_pred = _make_cell_eval_adata(n_perts=8)
        gene_names = list(adata_real.var_names)
        pert_names = sorted(set(adata_real.obs["perturbation"]) - {"control"})
        for pname in pert_names:
            assert pname in gene_names, f"Perturbation '{pname}' not in gene names"

    def test_passes_guess_is_lognorm(self):
        """Data must pass cell_eval.utils.guess_is_lognorm()."""
        from cell_eval.utils import guess_is_lognorm

        adata_real, adata_pred = _make_cell_eval_adata()
        assert guess_is_lognorm(adata_real, validate=True)
        assert guess_is_lognorm(adata_pred, validate=True)

    def test_min_cells_per_perturbation(self):
        """At least 20 cells per perturbation."""
        adata_real, _ = _make_cell_eval_adata()
        counts = adata_real.obs["perturbation"].value_counts()
        for pert, count in counts.items():
            assert count >= 20, f"Perturbation '{pert}' has only {count} cells (<20)"

    def test_var_names_match(self):
        """Both real and pred must share the same var_names."""
        adata_real, adata_pred = _make_cell_eval_adata()
        assert list(adata_real.var_names) == list(adata_pred.var_names)

    def test_obs_column_and_control_label(self):
        """obs column is 'perturbation', control label is 'control'."""
        adata_real, adata_pred = _make_cell_eval_adata()
        assert "perturbation" in adata_real.obs.columns
        assert "perturbation" in adata_pred.obs.columns
        assert "control" in adata_real.obs["perturbation"].values
        assert "control" in adata_pred.obs["perturbation"].values

    def test_same_perturbation_sets(self):
        """Real and pred have identical perturbation label sets."""
        adata_real, adata_pred = _make_cell_eval_adata()
        real_perts = set(adata_real.obs["perturbation"].unique())
        pred_perts = set(adata_pred.obs["perturbation"].unique())
        assert real_perts == pred_perts

    def test_values_in_valid_range(self):
        """Values in [0, 15) with fractional component (lognorm range)."""
        adata_real, adata_pred = _make_cell_eval_adata()
        for adata, name in [(adata_real, "real"), (adata_pred, "pred")]:
            X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
            assert X.min() >= 0, f"{name}: negative values found"
            assert X.max() < 15, f"{name}: max value {X.max():.2f} >= 15"
            # Must have fractional values (not all integer)
            frac, _ = np.modf(X.data if sp.issparse(adata.X) else X)
            assert np.any(frac > 1e-3), f"{name}: no fractional values found"

    def test_cell_eval_pair_creation(self):
        """PerturbationAnndataPair can be created without errors."""
        adata_real, adata_pred = _make_cell_eval_adata()
        pair = _build_pair(adata_real, adata_pred)
        # cell-eval's __post_init__ validates gene alignment, perturbation
        # overlap, control presence, etc. If we get here, all checks passed.
        assert len(pair.perts) > 0
        assert "control" not in pair.perts  # control excluded from perts

    def test_pred_is_correlated_but_imperfect(self):
        """Predicted data should be correlated with real but not identical."""
        adata_real, adata_pred = _make_cell_eval_adata(as_sparse=False)
        X_real = adata_real.X
        X_pred = adata_pred.X

        # Not identical
        assert not np.allclose(X_real, X_pred), "Real and pred should not be identical"

        # But correlated (per-gene correlation should be positive for most genes)
        from scipy.stats import pearsonr
        n_positive = 0
        for g in range(X_real.shape[1]):
            r, _ = pearsonr(X_real[:, g], X_pred[:, g])
            if r > 0:
                n_positive += 1
        frac_positive = n_positive / X_real.shape[1]
        assert frac_positive > 0.5, (
            f"Only {frac_positive:.0%} of genes have positive real-pred correlation"
        )

    def test_raw_count_adata_is_not_lognorm(self):
        """_make_raw_count_adata() should produce raw counts (not lognorm)."""
        from cell_eval.utils import guess_is_lognorm

        adata = _make_raw_count_adata()
        assert not guess_is_lognorm(adata, validate=False), (
            "Raw-count data should not pass guess_is_lognorm"
        )


# =============================================================================
# Pseudobulk means parity
# =============================================================================

class TestPseudobulkParity:
    """Verify pseudobulk means match cell-eval's polars group_by().mean()."""

    def test_pseudobulk_vs_cell_eval(self):
        adata_real, _adata_pred = _make_cell_eval_adata()

        # SCX pseudobulk
        scx_means, scx_groups = pyscx.accel.pseudobulk_means(
            adata_real, "perturbation"
        )

        # cell-eval pseudobulk (polars group_by().mean())
        ce_keys, ce_values = PerturbationAnndataPair._bulk_anndata(
            adata_real, "perturbation"
        )

        # Both should be sorted by group name
        assert sorted(scx_groups) == sorted(list(ce_keys))

        # Compare per-group means
        for i, group in enumerate(scx_groups):
            ce_idx = np.flatnonzero(ce_keys == group)[0]
            np.testing.assert_allclose(
                scx_means[i], ce_values[ce_idx], atol=1e-6,
                err_msg=f"Pseudobulk mismatch for group '{group}'",
            )


# =============================================================================
# Bulk perturbation metrics parity
# =============================================================================

class TestBulkMetricsParity:
    """Verify pearson_delta, mse, mae, mse_delta, mae_delta match cell-eval."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adata_real, self.adata_pred = _make_cell_eval_adata()
        self.pair = _build_pair(self.adata_real, self.adata_pred)

    def test_pearson_delta_vs_cell_eval(self):
        scx_result = pyscx.accel.perturbation_metrics(
            self.adata_real, self.adata_pred, metrics=["pearson_delta"],
        )
        ce_result = ce_pearson_delta(self.pair)

        for pert in ce_result:
            np.testing.assert_allclose(
                scx_result["pearson_delta"][pert], ce_result[pert], atol=1e-6,
                err_msg=f"pearson_delta mismatch for '{pert}'",
            )

    def test_mse_mae_vs_cell_eval(self):
        scx_result = pyscx.accel.perturbation_metrics(
            self.adata_real, self.adata_pred,
            metrics=["mse", "mae", "mse_delta", "mae_delta"],
        )

        ce_mse_result = ce_mse(self.pair)
        ce_mae_result = ce_mae(self.pair)
        ce_mse_delta_result = ce_mse_delta(self.pair)
        ce_mae_delta_result = ce_mae_delta(self.pair)

        for pert in ce_mse_result:
            np.testing.assert_allclose(
                scx_result["mse"][pert], ce_mse_result[pert], atol=1e-6,
                err_msg=f"mse mismatch for '{pert}'",
            )
            np.testing.assert_allclose(
                scx_result["mae"][pert], ce_mae_result[pert], atol=1e-6,
                err_msg=f"mae mismatch for '{pert}'",
            )
            np.testing.assert_allclose(
                scx_result["mse_delta"][pert], ce_mse_delta_result[pert], atol=1e-6,
                err_msg=f"mse_delta mismatch for '{pert}'",
            )
            np.testing.assert_allclose(
                scx_result["mae_delta"][pert], ce_mae_delta_result[pert], atol=1e-6,
                err_msg=f"mae_delta mismatch for '{pert}'",
            )

    def test_perturbation_metrics_agg_vs_cell_eval(self):
        """Compare aggregated (mean across perturbations) metrics."""
        scx_result = pyscx.accel.perturbation_metrics(
            self.adata_real, self.adata_pred,
        )

        ce_results = {
            "pearson_delta": ce_pearson_delta(self.pair),
            "mse": ce_mse(self.pair),
            "mae": ce_mae(self.pair),
            "mse_delta": ce_mse_delta(self.pair),
            "mae_delta": ce_mae_delta(self.pair),
        }

        for metric_name, ce_vals in ce_results.items():
            scx_vals = scx_result[metric_name]
            # Compute mean across perturbations
            ce_mean = np.mean(list(ce_vals.values()))
            scx_mean = np.mean(list(scx_vals.values()))
            np.testing.assert_allclose(
                scx_mean, ce_mean, atol=1e-5,
                err_msg=f"Aggregated {metric_name} mean mismatch",
            )


# =============================================================================
# Energy distance parity
# =============================================================================

class TestEdistanceParity:
    """Verify energy distance matches cell-eval."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adata_real, self.adata_pred = _make_cell_eval_adata(
            n_obs=400, n_vars=20, n_perts=5, seed=42,
        )
        self.pair = _build_pair(self.adata_real, self.adata_pred)

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    def test_edistance_vs_cell_eval(self, dtype):
        """Compare Pearson correlation of e-distance vectors.

        Parametrised over `dtype ∈ {"f32", "f64"}` (Phase 2). Reductions
        accumulate in f64 regardless of input dtype, so both must agree
        with cell-eval within `atol=1e-4`.
        """
        scx_corr = pyscx.accel.energy_distance(
            self.adata_real, self.adata_pred,
            pert_col="perturbation", control="control",
            dtype=dtype,
        )
        ce_corr = ce_edistance(self.pair)

        np.testing.assert_allclose(
            scx_corr, ce_corr, atol=1e-4,
            err_msg=f"e-distance correlation ({dtype}): SCX={scx_corr} vs cell-eval={ce_corr}",
        )

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    def test_edistance_intermediate_values(self, dtype):
        """Compare per-perturbation e-distance vectors.

        This catches cases where Pearson correlation accidentally matches
        but individual e-distances diverge. Uses
        `pyscx.accel.energy_distance_details()` to get per-perturbation
        e-distances and cell-eval's `PerturbationAnndataPair.get_pert_data()`
        plus `edist` to compute the reference values.

        Parametrised over `dtype ∈ {"f32", "f64"}` (Phase 2). Both must hold
        the per-pert e-distance within `atol=1e-4` against the f64 cdist
        reference; if `f32` fails at this tolerance, profile root cause
        before loosening the test.
        """
        from scipy.spatial.distance import cdist

        details = pyscx.accel.energy_distance_details(
            self.adata_real, self.adata_pred,
            pert_col="perturbation", control="control",
            dtype=dtype,
        )

        # Reference: compute e-distance directly on the same dense data
        # used inside cell-eval's edistance metric. Formula:
        #   e = 2 * mean(D(X, Y)) - mean(D(X, X)) - mean(D(Y, Y))
        # where X = pert cells, Y = control cells (Euclidean distances).
        def _edist(x: np.ndarray, y: np.ndarray) -> float:
            dxy = cdist(x, y, metric="euclidean").mean()
            dxx = cdist(x, x, metric="euclidean").mean()
            dyy = cdist(y, y, metric="euclidean").mean()
            return 2.0 * dxy - dxx - dyy

        X_real = self.adata_real.X.toarray() if sp.issparse(self.adata_real.X) else self.adata_real.X
        X_pred = self.adata_pred.X.toarray() if sp.issparse(self.adata_pred.X) else self.adata_pred.X
        labels_real = self.adata_real.obs["perturbation"].to_numpy()
        labels_pred = self.adata_pred.obs["perturbation"].to_numpy()
        ctrl_real = X_real[labels_real == "control"]
        ctrl_pred = X_pred[labels_pred == "control"]

        for pert in details["pert_names"]:
            ref_real = _edist(X_real[labels_real == pert], ctrl_real)
            ref_pred = _edist(X_pred[labels_pred == pert], ctrl_pred)
            np.testing.assert_allclose(
                details["d_real"][pert], ref_real, atol=1e-4,
                err_msg=f"d_real[{pert}] ({dtype}): SCX={details['d_real'][pert]} vs ref={ref_real}",
            )
            np.testing.assert_allclose(
                details["d_pred"][pert], ref_pred, atol=1e-4,
                err_msg=f"d_pred[{pert}] ({dtype}): SCX={details['d_pred'][pert]} vs ref={ref_pred}",
            )


# =============================================================================
# Discrimination score parity
# =============================================================================

def _make_tied_discrimination_adata(
    n_perts_non_control=3, cells_per_group=32,
) -> tuple[ad.AnnData, ad.AnnData]:
    """A fixture whose real effects are IDENTICAL, so every distance ties.

    The shared `_make_cell_eval_adata` fixture (continuous random) has no ties,
    no repeated gene symbol and no zero-norm effect vector, so it cannot see any
    of the three divergences fixed in v0.14.0 (review §7.13). This one is built
    to produce exact ties: every perturbation group has the same profile, so
    every `real_effect[i]` is the same vector and all `P` distances from any
    prediction are equal.

    "Exact" is the load-bearing word, and it is why the values are dyadic
    rationals and the group size is a power of two: a group mean is then
    representable with no rounding in f32 or f64, so the tie survives into both
    implementations rather than being broken by a last-bit difference. Cell-level
    noise would defeat the whole fixture.
    """
    n_genes = 4
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    # Perturbation names must match gene names for target-gene exclusion.
    perts = ["control"] + [f"gene_{i}" for i in range(n_perts_non_control)]

    control_profile = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    # One profile shared by EVERY perturbation group -> identical effects.
    pert_profile = np.array([2.0, 3.0, 1.5, 0.5], dtype=np.float32)

    rows, labels = [], []
    for pert in perts:
        profile = control_profile if pert == "control" else pert_profile
        for _ in range(cells_per_group):
            rows.append(profile)
            labels.append(pert)
    X = np.vstack(rows)

    obs = __import__("pandas").DataFrame({"perturbation": labels})
    obs.index = [f"cell_{i}" for i in range(X.shape[0])]
    var = __import__("pandas").DataFrame(index=gene_names)

    real = ad.AnnData(X=sp.csr_matrix(X), obs=obs.copy(), var=var.copy())
    # Prediction identical to real: distances are all exactly 0.0, the most
    # degenerate tie there is.
    pred = ad.AnnData(X=sp.csr_matrix(X.copy()), obs=obs.copy(), var=var.copy())
    return real, pred


def _make_duplicate_gene_discrimination_adata() -> tuple[ad.AnnData, ad.AnnData]:
    """A fixture with a repeated gene symbol, built so the *number* of excluded
    columns changes the answer.

    `var_names` are not unique in real 10x data, and cell-eval excludes every
    matching column (`np.flatnonzero(genes != p)`). A lookup keyed to one index
    leaves a duplicate of the target column in place, and that surviving copy
    restores the trivial self-match `exclude_target_gene` exists to remove.

    Making that observable takes care, because the obvious fixture does not: if
    the prediction already matches its own real effect on the remaining genes,
    the score is 1.0 with either exclusion width. Here `gene_0`'s prediction is
    built to agree with `gene_0`'s real effect on the two duplicated columns and
    with `gene_1`'s real effect everywhere else. Drop both duplicates and
    `gene_0` is closer to the wrong perturbation (score 0.5); drop only one and
    the surviving column ties it back to rank 0 (score 1.0).

    Dyadic values and a power-of-two group size again, so the pseudobulk means
    are exact and the comparison is `abs=0` rather than approximate.
    """
    import pandas as pd

    gene_names = ["gene_0", "gene_0", "gene_1", "gene_2"]  # gene_0 duplicated
    perts = ["control", "gene_0", "gene_1"]
    real_profiles = {
        "control": [1.5, 1.5, 1.5, 1.5],
        "gene_0": [11.5, 11.5, 1.5, 1.5],
        "gene_1": [1.5, 1.5, 6.5, 6.5],
    }
    pred_profiles = {
        "control": [1.5, 1.5, 1.5, 1.5],
        # Matches gene_0's real effect on the duplicated columns, and gene_1's
        # real effect on the rest.
        "gene_0": [11.5, 11.5, 6.5, 6.5],
        "gene_1": [1.5, 1.5, 6.5, 6.5],
    }

    def build(profiles):
        rows, labels = [], []
        for pert in perts:
            for _ in range(32):
                rows.append(profiles[pert])
                labels.append(pert)
        X = np.array(rows, dtype=np.float32)
        obs = pd.DataFrame({"perturbation": labels})
        obs.index = [f"cell_{i}" for i in range(X.shape[0])]
        return ad.AnnData(
            X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=gene_names)
        )

    return build(real_profiles), build(pred_profiles)


class TestDiscriminationDuplicateGeneParity:
    """`exclude_target_gene` with a repeated gene symbol, against real cell-eval."""

    def test_every_matching_column_is_excluded(self):
        real, pred = _make_duplicate_gene_discrimination_adata()
        pair = _build_pair(real, pred)

        # Premise: the duplicate must survive into cell-eval's own view of the
        # genes, or the fixture is testing nothing about duplicates.
        genes = [str(g) for g in pair.genes]
        assert genes.count("gene_0") == 2, (
            f"premise broken: cell-eval sees genes {genes}, which does not "
            f"contain the duplicated symbol this fixture is built around"
        )

        for metric in ("l1", "l2"):
            scx = pyscx.accel.discrimination_score(
                real, pred, metric=metric, exclude_target_gene=True,
            )
            ce = ce_discrimination_score(
                pair, metric=metric, exclude_target_gene=True,
            )

            # Premise: excluding BOTH copies must demote gene_0. If the reference
            # says 1.0 the fixture has stopped discriminating and a match below
            # would prove nothing.
            assert ce["gene_0"] == pytest.approx(0.5, abs=0), (
                f"premise broken at metric={metric}: cell-eval scored gene_0 "
                f"{ce['gene_0']}, expected 0.5 (both duplicated columns dropped). "
                f"A 1.0 here means the fixture no longer separates the two "
                f"exclusion widths."
            )

            for pert in ce:
                assert scx[pert] == pytest.approx(ce[pert], abs=0), (
                    f"duplicate-gene exclusion mismatch for '{pert}' at "
                    f"metric={metric}: SCX={scx[pert]} vs cell-eval={ce[pert]}. "
                    f"All SCX={scx}, all cell-eval={ce}"
                )


def _make_mixed_tie_discrimination_adata() -> tuple[ad.AnnData, ad.AnnData]:
    """Four perturbations whose L1 distances from a zero prediction are [3,3,1,1].

    Two tied blocks rather than one — the shape on which a stable and an unstable
    argsort disagree, and therefore the shape a total-tie fixture cannot reach.
    """
    import pandas as pd

    gene_names = ["gene_0", "gene_1"]
    perts = ["control", "p0", "p1", "p2", "p3"]
    # Control at the origin; each perturbation's effect has |effect| in gene_0 of
    # 3, 3, 1, 1 and nothing in gene_1. Predictions sit exactly on control, so
    # every prediction's effect is the zero vector and each perturbation sees the
    # same distance vector [3, 3, 1, 1].
    base = 4.5
    real_offsets = {"control": 0.0, "p0": 3.0, "p1": -3.0, "p2": 1.0, "p3": -1.0}

    def build(offsets):
        rows, labels = [], []
        for pert in perts:
            for _ in range(32):
                rows.append([base + offsets[pert], base])
                labels.append(pert)
        X = np.array(rows, dtype=np.float32)
        obs = pd.DataFrame({"perturbation": labels})
        obs.index = [f"cell_{i}" for i in range(X.shape[0])]
        return ad.AnnData(
            X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=gene_names)
        )

    real = build(real_offsets)
    # Prediction: every group sits on the control profile, so every predicted
    # effect is zero.
    pred = build({k: 0.0 for k in real_offsets})
    return real, pred


class TestDiscriminationTieParity:
    """The **total**-tie case, against the real cell-eval rather than a
    hand-derived value.

    Scoped to a total tie deliberately, and the scope is the point. cell-eval
    reads its rank off `np.argsort`, whose default kind is `quicksort` and
    therefore *not* a stable sort — so its tie order is implementation-defined and
    parity with it is only a meaningful claim where the stable and unstable
    readings coincide. Measured on numpy 2.4.4, that is exactly the totally-tied
    case:

        np.argsort([5, 5, 5])                -> [0, 1, 2]   (agrees with stable)
        np.argsort([3, 3, 1, 1])             -> [3, 2, 1, 0]
        np.argsort([3, 3, 1, 1], kind="stable") -> [2, 3, 0, 1]

    `test_mixed_ties_are_not_claimed_to_match` below covers the other side: SCX
    keeps the stable rule, cell-eval does not, and that divergence is documented
    rather than chased. An earlier version of this file claimed parity on ties in
    general, which was false — found by codex in review.
    """

    def test_mixed_ties_are_not_claimed_to_match(self):
        """A mixed tie: assert SCX's stable rule, and assert the reference differs.

        This is the honest complement to the parity test above. It pins two
        things, both of which have to hold for the documentation to be right:

        1. SCX's scores equal the **stable** argsort ranks (its documented rule).
        2. that on this numpy the divergence is real — cell-eval scores this
           fixture differently, which is what makes the "not claimed" scope in
           `docs/scanpy.md` load-bearing rather than defensive.

        The numpy-only half of the canary lives in `test_eval_metrics.py`, which
        does not require `cell_eval` and therefore actually runs in CI.
        """
        # Distances [3, 3, 1, 1] from a zero prediction: two tied blocks.
        #
        # The numpy default-vs-stable canary that used to live here has moved to
        # `test_eval_metrics.py::test_numpy_default_argsort_still_disagrees_with_stable`
        # — this module `importorskip`s `cell_eval`, which CI's Python-bindings
        # image does not have, so a canary here could never fire. Flagged by
        # Cursor Agent in review.
        d = np.array([3.0, 3.0, 1.0, 1.0])

        real, pred = _make_mixed_tie_discrimination_adata()
        scx = pyscx.accel.discrimination_score(
            real, pred, metric="l1", exclude_target_gene=False,
        )
        # SCX's rule: rank = position under a STABLE ascending sort.
        n = len(scx)
        want = {}
        for rank, idx in enumerate(np.argsort(d, kind="stable")):
            want[f"p{idx}"] = 1.0 - rank / n
        for pert, w in want.items():
            assert scx[pert] == pytest.approx(w, abs=0), (
                f"SCX scored '{pert}' {scx[pert]}, expected {w} from its "
                f"documented stable tie rule. All SCX={scx}"
            )

        # And the divergence itself, against the installed reference: this is the
        # fixture the docs cite, so if cell-eval ever agrees here the "not
        # claimed" scope has become unnecessarily broad and should be revisited.
        ce = ce_discrimination_score(
            _build_pair(real, pred), metric="l1", exclude_target_gene=False,
        )
        assert any(
            scx[p] != pytest.approx(ce[p], abs=0) for p in ce
        ), (
            f"cell-eval now agrees with SCX on this mixed tie (SCX={scx}, "
            f"cell-eval={dict(ce)}). The mixed-tie exclusion in docs/scanpy.md "
            f"may be broader than it needs to be — recheck it."
        )

    def test_all_ties_match_cell_eval(self):
        real, pred = _make_tied_discrimination_adata()
        pair = _build_pair(real, pred)

        for metric in ("l1", "l2", "cosine"):
            scx = pyscx.accel.discrimination_score(
                real, pred, metric=metric, exclude_target_gene=False,
            )
            ce = ce_discrimination_score(pair, metric=metric, exclude_target_gene=False)

            # Premise: the fixture must actually be exercising tie-breaking, or
            # this is just another random-data parity test wearing a new name.
            # Under a total tie the reference's scores are exactly the stable
            # argsort positions, `1 - p/P` — all distinct, and *nothing* like the
            # all-1.0 that a rank-by-strictly-smaller-count produces.
            n = len(ce)
            want_stable = {
                pert: 1.0 - i / n for i, pert in enumerate(pair.perts)
            }
            assert ce == pytest.approx(want_stable, abs=0), (
                f"premise broken for metric={metric}: cell-eval returned {ce}, "
                f"not the stable-argsort ranks {want_stable}. Either the fixture "
                f"stopped producing exact ties, or numpy's argsort no longer "
                f"breaks them by index — check which before trusting this file."
            )

            for pert in ce:
                assert scx[pert] == pytest.approx(ce[pert], abs=0), (
                    f"tie-breaking mismatch for '{pert}' at metric={metric}: "
                    f"SCX={scx[pert]} vs cell-eval={ce[pert]}. All SCX={scx}, "
                    f"all cell-eval={ce}"
                )


class TestDiscriminationScoreParity:
    """Verify discrimination score matches cell-eval."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adata_real, self.adata_pred = _make_cell_eval_adata()
        self.pair = _build_pair(self.adata_real, self.adata_pred)

    def test_discrimination_score_l1_vs_cell_eval(self):
        """L1 metric — rank scores should match exactly."""
        scx_result = pyscx.accel.discrimination_score(
            self.adata_real, self.adata_pred, metric="l1",
        )
        ce_result = ce_discrimination_score(self.pair, metric="l1")

        for pert in ce_result:
            assert scx_result[pert] == pytest.approx(ce_result[pert], abs=0), (
                f"discrimination_score_l1 mismatch for '{pert}': "
                f"SCX={scx_result[pert]} vs cell-eval={ce_result[pert]}"
            )

    def test_discrimination_score_l2_cosine_vs_cell_eval(self):
        """L2 and cosine metrics."""
        for metric, ce_metric in [("l2", "l2"), ("cosine", "cosine")]:
            scx_result = pyscx.accel.discrimination_score(
                self.adata_real, self.adata_pred, metric=metric,
            )
            ce_result = ce_discrimination_score(self.pair, metric=ce_metric)

            for pert in ce_result:
                assert scx_result[pert] == pytest.approx(ce_result[pert], abs=0), (
                    f"discrimination_score_{metric} mismatch for '{pert}': "
                    f"SCX={scx_result[pert]} vs cell-eval={ce_result[pert]}"
                )

    def test_discrimination_target_exclusion_parity(self):
        """Verify exclude_target_gene behavior matches cell-eval."""
        # With target gene exclusion (default in cell-eval)
        scx_with = pyscx.accel.discrimination_score(
            self.adata_real, self.adata_pred, metric="l1",
            exclude_target_gene=True,
        )
        ce_with = ce_discrimination_score(
            self.pair, metric="l1", exclude_target_gene=True,
        )

        # Without target gene exclusion
        scx_without = pyscx.accel.discrimination_score(
            self.adata_real, self.adata_pred, metric="l1",
            exclude_target_gene=False,
        )
        ce_without = ce_discrimination_score(
            self.pair, metric="l1", exclude_target_gene=False,
        )

        for pert in ce_with:
            assert scx_with[pert] == pytest.approx(ce_with[pert], abs=0), (
                f"exclude_target_gene=True mismatch for '{pert}'"
            )
            assert scx_without[pert] == pytest.approx(ce_without[pert], abs=0), (
                f"exclude_target_gene=False mismatch for '{pert}'"
            )

        # Sanity check: if exclusion changes SCX scores, it must also change
        # cell-eval scores (i.e., the exclusion effect itself is consistent).
        # On this synthetic dataset the L1 rank score can be identical with
        # and without excluding a single target gene out of 100; that's fine
        # as long as SCX and cell-eval agree on the (possibly zero) delta.
        scx_delta = {p: scx_with[p] != scx_without[p] for p in scx_with}
        ce_delta = {p: ce_with[p] != ce_without[p] for p in ce_with}
        assert scx_delta == ce_delta, (
            f"exclude_target_gene delta pattern differs: SCX={scx_delta} vs cell-eval={ce_delta}"
        )


# =============================================================================
# Knockdown efficiency parity
# =============================================================================

class TestKnockdownParity:
    """Verify knockdown efficiency matches arc-bench."""

    def test_knockdown_vs_arc_bench(self):
        """Raw-count knockdown efficiency."""
        import scanpy as sc

        adata = _make_raw_count_adata()

        # Normalize (NOT log1p) — knockdown is computed before log1p
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm)

        # arc-bench reference
        baseline_ref = compute_control_baseline(
            adata_norm, "perturbation", "control",
        )
        kd_ref = compute_knockdown_efficiency(
            adata_norm, baseline_ref, "perturbation", "control",
        )

        # SCX implementation
        pyscx.accel.knockdown_efficiency(
            adata_norm, pert_col="perturbation", control="control",
        )
        kd_scx = adata_norm.obs["KnockDownEfficiency"].values

        # Compare — NaN positions must match exactly
        nan_ref = np.isnan(kd_ref)
        nan_scx = np.isnan(kd_scx)
        np.testing.assert_array_equal(
            nan_ref, nan_scx,
            err_msg="NaN positions in knockdown efficiency differ",
        )

        # Compare non-NaN values
        mask = ~nan_ref
        np.testing.assert_allclose(
            kd_scx[mask], kd_ref[mask], atol=1e-6,
            err_msg="Knockdown efficiency values differ",
        )

    def test_log_deviation_vs_arc_bench(self):
        """Log deviation after normalize+log1p.

        SCX's knockdown_efficiency expects normalized (NOT log1p'd) data and
        applies log1p internally to both the data and the baseline. The
        arc-bench reference takes an externally-log1p'd AnnData plus a
        separately-log1p'd baseline. We feed each implementation data in
        its expected form, then compare the resulting KnockDownGeneFC arrays.
        """
        import scanpy as sc

        adata = _make_raw_count_adata()

        # Normalize (linear space, shared starting point).
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm)

        # ── SCX path: feeds normalized data, SCX log1p's internally ────
        adata_scx = adata_norm.copy()
        pyscx.accel.knockdown_efficiency(
            adata_scx, pert_col="perturbation", control="control",
        )
        fc_scx = adata_scx.obs["KnockDownGeneFC"].values

        # ── arc-bench path: baseline in linear space, then log1p data ──
        baseline_ref = compute_control_baseline(
            adata_norm, "perturbation", "control",
        )
        baseline_log_ref = np.log1p(baseline_ref)
        sc.pp.log1p(adata_norm)
        fc_ref = compute_log_deviation(
            adata_norm, baseline_log_ref, "perturbation", "control",
        )

        # NaN positions must match exactly (control + missing-gene cells).
        nan_ref = np.isnan(fc_ref)
        nan_scx = np.isnan(fc_scx)
        np.testing.assert_array_equal(
            nan_ref, nan_scx,
            err_msg="NaN positions in log deviation differ",
        )

        mask = ~nan_ref
        np.testing.assert_allclose(
            fc_scx[mask], fc_ref[mask], atol=1e-6,
            err_msg="Log deviation values differ",
        )

    def test_knockdown_missing_gene(self):
        """Perturbation name not in var_names → NaN for those cells."""
        import pandas as pd
        import scanpy as sc

        rng = np.random.default_rng(99)
        n_obs, n_vars = 100, 20
        X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)

        # "missing_gene" is not in var_names
        labels = (["control"] * 50) + (["missing_gene"] * 25) + (["gene_0"] * 25)
        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
        adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
        sc.pp.normalize_total(adata)

        # arc-bench reference
        baseline = compute_control_baseline(adata, "perturbation", "control")
        kd_ref = compute_knockdown_efficiency(
            adata, baseline, "perturbation", "control",
        )

        # SCX
        pyscx.accel.knockdown_efficiency(
            adata, pert_col="perturbation", control="control",
        )
        kd_scx = adata.obs["KnockDownEfficiency"].values

        # "missing_gene" cells should be NaN in both
        missing_mask = np.array(labels) == "missing_gene"
        assert np.all(np.isnan(kd_ref[missing_mask])), "arc-bench should produce NaN for missing gene"
        assert np.all(np.isnan(kd_scx[missing_mask])), "SCX should produce NaN for missing gene"

        # NaN positions should match exactly
        np.testing.assert_array_equal(
            np.isnan(kd_ref), np.isnan(kd_scx),
            err_msg="NaN positions differ for missing gene case",
        )


# =============================================================================
# Clustering agreement parity
# =============================================================================

class TestClusteringAgreementParity:
    """Verify clustering agreement metrics."""

    def test_clustering_agreement_vs_cell_eval(self):
        """Compare clustering agreement scores.

        Note: Exact match not expected due to stochastic Leiden.

        n_perts=30 (was 8): the Phase 3 refactor swapped scanpy's
        igraph-Leiden for scx_accel's Rust-native Leiden, which uses a
        different RB-modularity tie-break. On centroid graphs with ≤ ~10
        nodes the two algorithms can produce different community counts
        at resolution=1.0, which manifests as a wide AMI gap because the
        scoring is permutation-invariant only after both sides actually
        partition. At n_perts ≥ 16 the algorithms agree exactly on this
        synthetic; n_perts=30 picks a comfortable margin and still
        exercises the multi-resolution sweep. The atol stays at 0.15 per
        the spec.
        """
        adata_real, adata_pred = _make_cell_eval_adata(
            n_obs=900, n_vars=50, n_perts=30, seed=42,
        )
        pair = _build_pair(adata_real, adata_pred)

        scx_score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred,
            pert_col="perturbation", control="control", metric="ami",
        )
        ce_scorer = ClusteringAgreement(metric="ami")
        ce_score = ce_scorer(pair)

        # Loose tolerance due to stochastic Leiden
        np.testing.assert_allclose(
            scx_score, ce_score, atol=0.15,
            err_msg=f"Clustering agreement: SCX={scx_score} vs cell-eval={ce_score}",
        )

    def test_clustering_scoring_functions_vs_sklearn(self):
        """Verify AMI/NMI/ARI scoring on identical labels match sklearn."""
        rng = np.random.default_rng(42)
        labels_a = rng.integers(0, 5, size=100).tolist()
        labels_b = rng.integers(0, 5, size=100).tolist()

        # AMI
        scx_ami = pyscx.accel.adjusted_mutual_info(labels_a, labels_b)
        sk_ami = adjusted_mutual_info_score(labels_a, labels_b)
        np.testing.assert_allclose(scx_ami, sk_ami, atol=1e-10, err_msg="AMI mismatch")

        # NMI
        scx_nmi = pyscx.accel.normalized_mutual_info(labels_a, labels_b)
        sk_nmi = normalized_mutual_info_score(labels_a, labels_b)
        np.testing.assert_allclose(scx_nmi, sk_nmi, atol=1e-10, err_msg="NMI mismatch")

        # ARI — default matches sklearn exactly (like NMI/AMI above).
        scx_ari = pyscx.accel.adjusted_rand_index(labels_a, labels_b)
        sk_ari = adjusted_rand_score(labels_a, labels_b)
        np.testing.assert_allclose(
            scx_ari, sk_ari, atol=1e-10,
            err_msg=f"ARI mismatch vs sklearn: SCX={scx_ari} vs sklearn={sk_ari}",
        )
        # rescaled=True gives cell-eval's (ARI + 1) / 2.
        scx_ari_rescaled = pyscx.accel.adjusted_rand_index(labels_a, labels_b, rescaled=True)
        np.testing.assert_allclose(scx_ari_rescaled, (sk_ari + 1) / 2, atol=1e-10)

    def test_clustering_scoring_string_categorical_labels(self):
        """F5: AMI/NMI/ARI accept string / categorical labels (factorized
        internally like sklearn), not just integer codes."""
        import pandas as pd

        rng = np.random.default_rng(7)
        cell_types = np.array(["T cell", "B cell", "NK cell", "Mono"])
        clusters = np.array(["c0", "c1", "c2", "c3", "c4"])
        labels_a = cell_types[rng.integers(0, len(cell_types), size=200)].tolist()
        labels_b = clusters[rng.integers(0, len(clusters), size=200)].tolist()

        # String labels must no longer raise (was: invalid literal for int()).
        scx_ami = pyscx.accel.adjusted_mutual_info(labels_a, labels_b)
        scx_nmi = pyscx.accel.normalized_mutual_info(labels_a, labels_b)
        scx_ari = pyscx.accel.adjusted_rand_index(labels_a, labels_b)

        # Match sklearn on the same raw string labels.
        np.testing.assert_allclose(
            scx_ami, adjusted_mutual_info_score(labels_a, labels_b), atol=1e-10
        )
        np.testing.assert_allclose(
            scx_nmi, normalized_mutual_info_score(labels_a, labels_b), atol=1e-10
        )
        np.testing.assert_allclose(
            scx_ari, adjusted_rand_score(labels_a, labels_b), atol=1e-10
        )

        # Identical string labels → perfect agreement.
        np.testing.assert_allclose(
            pyscx.accel.adjusted_rand_index(labels_a, labels_a), 1.0, atol=1e-10
        )
        np.testing.assert_allclose(
            pyscx.accel.normalized_mutual_info(labels_a, labels_a), 1.0, atol=1e-10
        )

        # A pandas Categorical gives the same result as its string form.
        cat_a = pd.Categorical(labels_a)
        np.testing.assert_allclose(
            pyscx.accel.adjusted_rand_index(cat_a, labels_b),
            scx_ari,
            atol=1e-10,
        )


# =============================================================================
# DE result format bridge parity
# =============================================================================

class TestDEBridgeParity:
    """Verify DE result format bridge."""

    def test_de_dataframe_format(self):
        """Verify the polars DataFrame cell-eval consumes matches its DEResults
        schema. `output="polars"` is explicit: pandas is the default container
        (F6) but `DEResults.data` is typed `pl.DataFrame`."""
        adata_real, _ = _make_cell_eval_adata(n_obs=200, n_vars=50, n_perts=4)

        df = pyscx.accel.rank_genes_groups_df(
            adata_real, "perturbation", reference="control", output="polars",
        )

        required_cols = {
            "target", "feature", "fold_change", "p_value",
            "fdr", "log2_fold_change", "abs_log2_fold_change",
        }
        assert required_cols.issubset(set(df.columns)), (
            f"Missing columns: {required_cols - set(df.columns)}"
        )

        # Verify column types
        assert df["target"].dtype == pl.Utf8
        assert df["feature"].dtype == pl.Utf8
        for col in ["fold_change", "p_value", "fdr", "log2_fold_change", "abs_log2_fold_change"]:
            assert df[col].dtype == pl.Float64, f"{col} should be Float64, got {df[col].dtype}"

    def test_de_dataframe_defaults_to_pandas(self):
        """F6: pandas is the default container and polars the opt-in, with
        identical columns and values either way."""
        import pandas as pd

        adata_real, _ = _make_cell_eval_adata(n_obs=200, n_vars=50, n_perts=4)

        df_default = pyscx.accel.rank_genes_groups_df(
            adata_real, "perturbation", reference="control",
        )
        df_pl = pyscx.accel.rank_genes_groups_df(
            adata_real, "perturbation", reference="control", output="polars",
        )
        df_pd = pyscx.accel.rank_genes_groups_df(
            adata_real, "perturbation", reference="control", output="pandas",
        )
        assert isinstance(df_default, pd.DataFrame)
        assert isinstance(df_pl, pl.DataFrame)
        assert isinstance(df_pd, pd.DataFrame)
        # Same column names + order.
        assert list(df_pd.columns) == list(df_pl.columns)
        # Values identical to the polars result.
        pd.testing.assert_frame_equal(
            df_pd.reset_index(drop=True),
            df_pl.to_pandas().reset_index(drop=True),
        )
        # Unknown output value is rejected.
        with pytest.raises(ValueError):
            pyscx.accel.rank_genes_groups_df(
                adata_real, "perturbation", reference="control", output="bogus",
            )

    def test_de_bridge_feeds_cell_eval_metrics(self):
        """Verify DE bridge output can be consumed by cell-eval DE metrics.

        Note: Exact values don't need to match (SCX Wilcoxon vs pdex),
        just format compatibility.
        """
        adata_real, adata_pred = _make_cell_eval_adata(n_obs=200, n_vars=50, n_perts=4)

        # Compute DE via SCX for both real and pred. `output="polars"` is
        # required, not stylistic: `cell_eval`'s DEResults.data is typed
        # `pl.DataFrame` and its __post_init__ runs `pl.col(...)` expressions, so
        # the default pandas frame is rejected there (F6).
        df_real = pyscx.accel.rank_genes_groups_df(
            adata_real, "perturbation", reference="control", output="polars",
        )
        df_pred = pyscx.accel.rank_genes_groups_df(
            adata_pred, "perturbation", reference="control", output="polars",
        )

        # Feed into cell-eval's DE initialization
        from cell_eval import initialize_de_comparison

        de_comparison = initialize_de_comparison(real=df_real, pred=df_pred)

        # Verify cell-eval can compute DE metrics (no errors)
        from cell_eval._pipeline import MetricPipeline

        pipeline = MetricPipeline(profile="de", break_on_error=False)
        pipeline.compute_de_metrics(de_comparison)
        results = pipeline.get_results()

        # All DE metrics should produce valid float values
        assert results.height > 0, "No DE metric results produced"
        for col in results.columns:
            if col == "perturbation":
                continue
            values = results[col].to_numpy()
            non_null = values[~np.isnan(values.astype(float))]
            assert len(non_null) > 0, f"DE metric '{col}' produced all NaN/null"


# =============================================================================
# Full pipeline integration test
# =============================================================================

class TestFullPipelineParity:
    """Full pipeline integration against cell-eval."""

    # Tolerance table (see docs/scanpy.md "Perturbation evaluation metrics")
    TOLERANCE = {
        "pearson_delta": 1e-6,
        "mse": 1e-6,
        "mae": 1e-6,
        "mse_delta": 1e-6,
        "mae_delta": 1e-6,
        "discrimination_score_l1": 0,  # exact (integer rank)
        "discrimination_score_l2": 0,
        "discrimination_score_cosine": 0,
        "pearson_edistance": 1e-4,
        "clustering_agreement": 0.15,
    }

    def test_full_pipeline_vs_cell_eval(self):
        """Run complete SCX pipeline vs cell-eval MetricPipeline."""
        adata_real, adata_pred = _make_cell_eval_adata()
        pair = _build_pair(adata_real, adata_pred)

        # --- cell-eval pipeline (anndata metrics only, skip DE) ---
        from cell_eval import MetricsEvaluator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = MetricsEvaluator(
                adata_pred=adata_pred,
                adata_real=adata_real,
                control_pert="control",
                pert_col="perturbation",
                outdir=tmpdir,
                skip_de=True,
            )
            ce_results, ce_agg = evaluator.compute(
                profile="anndata", write_csv=False, break_on_error=True,
            )

        # --- SCX pipeline ---
        scx_results = {}

        # Bulk metrics
        bulk = pyscx.accel.perturbation_metrics(adata_real, adata_pred)
        for metric_name, vals in bulk.items():
            scx_results[metric_name] = vals

        # Discrimination score (all 3 metrics)
        for metric in ["l1", "l2", "cosine"]:
            scx_results[f"discrimination_score_{metric}"] = (
                pyscx.accel.discrimination_score(
                    adata_real, adata_pred, metric=metric,
                )
            )

        # Energy distance
        scx_edistance = pyscx.accel.energy_distance(adata_real, adata_pred)

        # Clustering agreement
        scx_clustering = pyscx.accel.clustering_agreement(
            adata_real, adata_pred,
            pert_col="perturbation", control="control", metric="ami",
        )

        # --- Compare per-perturbation results ---
        perts = sorted(set(adata_real.obs["perturbation"]) - {"control"})

        for metric_name in [
            "pearson_delta", "mse", "mae", "mse_delta", "mae_delta",
            "discrimination_score_l1", "discrimination_score_l2",
            "discrimination_score_cosine",
        ]:
            tol = self.TOLERANCE[metric_name]
            if metric_name not in ce_results.columns:
                continue

            for pert in perts:
                ce_row = ce_results.filter(pl.col("perturbation") == pert)
                if ce_row.height == 0:
                    continue
                ce_val = ce_row[metric_name][0]
                scx_val = scx_results[metric_name].get(pert)
                if scx_val is None or ce_val is None:
                    continue
                np.testing.assert_allclose(
                    scx_val, ce_val, atol=tol,
                    err_msg=f"Pipeline mismatch: {metric_name}[{pert}]",
                )

        # e-distance (single correlation value)
        if "pearson_edistance" in ce_results.columns:
            ce_edist = ce_results["pearson_edistance"][0]
            np.testing.assert_allclose(
                scx_edistance, ce_edist, atol=self.TOLERANCE["pearson_edistance"],
                err_msg="Pipeline mismatch: pearson_edistance",
            )

        # Clustering (loose tolerance due to stochastic Leiden)
        if "clustering_agreement" in ce_results.columns:
            ce_clust = ce_results["clustering_agreement"][0]
            np.testing.assert_allclose(
                scx_clustering, ce_clust,
                atol=self.TOLERANCE["clustering_agreement"],
                err_msg="Pipeline mismatch: clustering_agreement",
            )

    def test_full_pipeline_arc_bench_cli_parity(self):
        """Simulate arc_bench.tools.pert_eval.cli._run_standard().

        Validates the end-to-end arc-bench integration:
        1. Clip X to [0, 14]
        2. Run MetricsEvaluator + SCX pipeline
        """
        adata_real, adata_pred = _make_cell_eval_adata()

        # Clip X to [0, 14] as arc-bench does
        if sp.issparse(adata_real.X):
            adata_real.X = adata_real.X.toarray()
        if sp.issparse(adata_pred.X):
            adata_pred.X = adata_pred.X.toarray()

        adata_real.X = np.clip(adata_real.X, 0, 14)
        adata_pred.X = np.clip(adata_pred.X, 0, 14)

        # Run SCX bulk metrics on clipped data
        scx_bulk = pyscx.accel.perturbation_metrics(adata_real, adata_pred)

        # Run cell-eval on clipped data
        pair = _build_pair(adata_real, adata_pred)
        ce_mse_result = ce_mse(pair)
        ce_pearson_result = ce_pearson_delta(pair)

        # Verify parity on clipped data
        for pert in ce_mse_result:
            np.testing.assert_allclose(
                scx_bulk["mse"][pert], ce_mse_result[pert], atol=1e-6,
                err_msg=f"arc-bench parity mismatch: mse[{pert}]",
            )
            np.testing.assert_allclose(
                scx_bulk["pearson_delta"][pert], ce_pearson_result[pert], atol=1e-6,
                err_msg=f"arc-bench parity mismatch: pearson_delta[{pert}]",
            )


# =============================================================================
# Performance comparison
# =============================================================================

class TestPerformanceComparison:
    """Informational speedup comparison (no assertions on speedup)."""

    @pytest.mark.slow
    def test_performance_vs_cell_eval(self):
        """Time both SCX and cell-eval on a larger dataset."""
        adata_real, adata_pred = _make_cell_eval_adata(
            n_obs=10000, n_vars=2000, n_perts=50, seed=42,
        )
        pair = _build_pair(adata_real, adata_pred)

        timings = {}

        # Pseudobulk means
        t0 = time.perf_counter()
        pyscx.accel.pseudobulk_means(adata_real, "perturbation")
        timings["scx_pseudobulk"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        PerturbationAnndataPair._bulk_anndata(adata_real, "perturbation")
        timings["ce_pseudobulk"] = time.perf_counter() - t0

        # Bulk metrics (all 5 bundled)
        t0 = time.perf_counter()
        pyscx.accel.perturbation_metrics(adata_real, adata_pred)
        timings["scx_bulk_metrics"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ce_pearson_delta(pair)
        ce_mse(pair)
        ce_mae(pair)
        ce_mse_delta(pair)
        ce_mae_delta(pair)
        timings["ce_bulk_metrics"] = time.perf_counter() - t0

        # Discrimination score L1
        t0 = time.perf_counter()
        pyscx.accel.discrimination_score(adata_real, adata_pred, metric="l1")
        timings["scx_discrimination_l1"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ce_discrimination_score(pair, metric="l1")
        timings["ce_discrimination_l1"] = time.perf_counter() - t0

        # Print results
        print("\n" + "=" * 60)
        print("Performance Comparison: SCX vs cell-eval")
        print("=" * 60)
        print(f"  Dataset: {adata_real.n_obs} cells × {adata_real.n_vars} genes × {50} perts")
        print()
        for key in sorted(timings):
            print(f"  {key:30s}  {timings[key]:8.3f}s")
        print()

        # Compute speedups
        for name in ["pseudobulk", "bulk_metrics", "discrimination_l1"]:
            scx_t = timings.get(f"scx_{name}", 0)
            ce_t = timings.get(f"ce_{name}", 0)
            if scx_t > 0:
                speedup = ce_t / scx_t
                print(f"  {name:30s}  {speedup:.1f}x speedup")
        print("=" * 60)


# =============================================================================
# Scoring parity
# =============================================================================

class TestScoringParity:
    """Verify score_agg_metrics compatibility."""

    def test_score_agg_metrics_parity(self):
        """Verify normalized scores are compatible with cell-eval's scoring."""
        import tempfile

        adata_real, adata_pred = _make_cell_eval_adata()
        pair = _build_pair(adata_real, adata_pred)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run cell-eval pipeline to get agg results
            evaluator = cell_eval.MetricsEvaluator(
                adata_pred=adata_pred,
                adata_real=adata_real,
                control_pert="control",
                pert_col="perturbation",
                outdir=tmpdir,
                skip_de=True,
            )
            ce_results, ce_agg = evaluator.compute(
                profile="anndata", write_csv=False, break_on_error=True,
            )

        # Build SCX results in the same DataFrame format
        scx_all = pyscx.accel.perturbation_metrics(adata_real, adata_pred)

        # Build a DataFrame matching cell-eval's format
        perts = sorted(set(adata_real.obs["perturbation"]) - {"control"})
        scx_rows = []
        for pert in perts:
            row = {"perturbation": pert}
            for metric_name, vals in scx_all.items():
                row[metric_name] = vals.get(pert, float("nan"))
            scx_rows.append(row)

        scx_results_df = pl.DataFrame(scx_rows)
        scx_agg = scx_results_df.drop("perturbation").describe()

        # Verify aggregated metrics have the same structure
        assert set(scx_agg.columns).issubset(set(ce_agg.columns) | {"statistic"}), (
            "SCX aggregated results have unexpected columns"
        )

        # Compare mean values for overlapping metrics
        for col in scx_agg.columns:
            if col == "statistic" or col not in ce_agg.columns:
                continue
            # Extract mean row
            scx_mean_row = scx_agg.filter(pl.col("statistic") == "mean")
            ce_mean_row = ce_agg.filter(pl.col("statistic") == "mean")
            if scx_mean_row.height == 0 or ce_mean_row.height == 0:
                continue
            scx_val = scx_mean_row[col][0]
            ce_val = ce_mean_row[col][0]
            if scx_val is not None and ce_val is not None:
                np.testing.assert_allclose(
                    float(scx_val), float(ce_val), atol=1e-5,
                    err_msg=f"Aggregated score mismatch for '{col}'",
                )
