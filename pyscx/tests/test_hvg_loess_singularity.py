"""LOESS singularity resilience for per-batch HVG — see SCX-USER-REPORT-2026-05-21 B1.

The pyscx multi-batch `seurat_v3` path runs one `skmisc.loess.fit()` per
batch. On real Census data, batches with very few cells or with
near-collinear log-mean / log-variance trigger a `ValueError` from
`_loess.pyx` ("There are other near singularities as well. ...").

Pre-fix: the exception propagated as an uncaught traceback, killing the
whole HVG call.

Post-fix: the exception is caught per-batch; a `UserWarning` is emitted
naming the failing batch and its cell count; an internal `batch_failed`
flag is set so the batch is skipped in step 5's normalised-variance
computation and in both the cross-batch mean and the median-rank
aggregation. Other batches proceed normally. If *every* batch fails,
the call raises `RuntimeError` rather than silently returning NaN-ranked
HVGs.

These tests use monkey-patching to deterministically force a singular
LOESS on a single batch — synthesising data that *reliably* singularises
`skmisc.loess` across versions is brittle, but the resilience logic is
purely about catching the `PyErr` regardless of how it was raised.
"""

import warnings

import pytest


def _patch_loess_to_raise_on_first_batch(monkeypatch):
    """Force `skmisc.loess.loess(...).fit()` to raise ValueError on the
    first call of this test, then behave normally for subsequent calls.

    Mirrors the real-world crash signature in the SCX-USER-REPORT B1
    traceback:
        File "_loess.pyx", line 922, in _loess.loess.fit
        ValueError: b'There are other near singularities as well. 0.22764'
    """
    import skmisc.loess as loess_mod

    real_loess = loess_mod.loess
    state = {"n_calls": 0}

    class FailingLoess:
        def __init__(self, *args, **kwargs):
            self._inner = real_loess(*args, **kwargs)

        def __getattr__(self, name):
            # `.fit()` is the singular-prone call on real Census batches.
            if name == "fit":
                state["n_calls"] += 1
                if state["n_calls"] == 1:
                    def _raise():
                        raise ValueError(
                            "b'There are other near singularities as well. "
                            "0.22764'"
                        )

                    return _raise
            return getattr(self._inner, name)

    monkeypatch.setattr(loess_mod, "loess", FailingLoess)
    return state


def _patch_loess_to_always_raise(monkeypatch):
    """Force every `skmisc.loess.loess(...).fit()` call to raise — used
    to exercise the all-batches-failed edge case.
    """
    import skmisc.loess as loess_mod

    real_loess = loess_mod.loess

    class AlwaysFailingLoess:
        def __init__(self, *args, **kwargs):
            self._inner = real_loess(*args, **kwargs)

        def __getattr__(self, name):
            if name == "fit":
                def _raise():
                    raise ValueError(
                        "b'There are other near singularities as well. "
                        "0.99999' (forced)"
                    )

                return _raise
            return getattr(self._inner, name)

    monkeypatch.setattr(loess_mod, "loess", AlwaysFailingLoess)


def test_loess_singularity_is_caught_not_raised(
    synthetic_adata, scx_from_adata, monkeypatch
):
    """A singular LOESS in one batch must not raise; HVG completes."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_loess_catch.scx")
    adata = pyscx.open(path).to_anndata(backed=True)
    _patch_loess_to_raise_on_first_batch(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Should NOT raise.
        pyscx.accel.highly_variable_genes(
            adata,
            n_top_genes=10,
            flavor="seurat_v3",
            batch_key="batch",
            device="cpu",
        )

    assert "highly_variable" in adata.var.columns, (
        "HVG did not populate var columns; warnings: "
        + "; ".join(str(w.message) for w in caught)
    )


def test_loess_singularity_emits_warning_naming_batch(
    synthetic_adata, scx_from_adata, monkeypatch
):
    """The catch must emit a UserWarning naming the failing batch index."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_loess_warn.scx")
    adata = pyscx.open(path).to_anndata(backed=True)
    _patch_loess_to_raise_on_first_batch(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.accel.highly_variable_genes(
            adata,
            n_top_genes=10,
            flavor="seurat_v3",
            batch_key="batch",
            device="cpu",
        )

    singular_warnings = [
        w
        for w in caught
        if "skmisc.loess fit failed on batch" in str(w.message)
    ]
    assert singular_warnings, (
        "No LOESS-singularity warning emitted; got: "
        + "; ".join(str(w.message) for w in caught)
    )
    msg = str(singular_warnings[0].message)
    assert "batch index" in msg
    assert "excluded from the per-batch HVG ranking" in msg
    # Verify the upstream error string is echoed so the user can correlate
    # the warning with a specific cause.
    assert "near singularities" in msg


def test_loess_singularity_does_not_block_other_batches(
    synthetic_adata, scx_from_adata, monkeypatch
):
    """Genes from well-conditioned batches must still get ranked normally."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_loess_other_batches.scx")
    adata = pyscx.open(path).to_anndata(backed=True)
    _patch_loess_to_raise_on_first_batch(monkeypatch)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        pyscx.accel.highly_variable_genes(
            adata,
            n_top_genes=10,
            flavor="seurat_v3",
            batch_key="batch",
            device="cpu",
        )

    # Surviving batches must still produce a non-empty HVG set.
    selected_count = int(adata.var["highly_variable"].sum())
    assert selected_count == 10, (
        f"Expected 10 HVGs from surviving batches, got {selected_count}"
    )


def test_failed_batch_excluded_matches_surviving_batches(
    synthetic_adata, scx_from_adata, monkeypatch
):
    """The "failed batch is excluded from ranking" contract.

    A 3-batch run with batch 0's loess fit forced to fail must produce
    the same HVG mask and `highly_variable_rank` as a 2-batch run on
    the other two batches alone.

    Pre-fix this was silently violated: the failed batch's all-zero
    `estimat_var` became `reg_std_sq = 10^0 == 1.0`, so its
    unregularised clipped variance contributed to both `mean_norm_var`
    and the per-batch median-rank aggregation, polluting the result.
    """
    import re

    import numpy as np
    import pyscx

    # ── Run A: 3-batch, batch 0's loess forced to raise on first call ──
    path_a = scx_from_adata(synthetic_adata, "hvg_parity_3batch.scx")
    adata_a = pyscx.open(path_a).to_anndata(backed=True)
    _patch_loess_to_raise_on_first_batch(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.accel.highly_variable_genes(
            adata_a,
            n_top_genes=10,
            flavor="seurat_v3",
            batch_key="batch",
            device="cpu",
        )

    # Parse the per-batch warning to find which batch index failed —
    # robust against any changes in the batch-iteration order.
    sing = [
        str(w.message)
        for w in caught
        if "skmisc.loess fit failed on batch" in str(w.message)
    ]
    assert len(sing) == 1, f"expected 1 singularity warning, got: {sing!r}"
    m = re.search(r"batch index (\d+)", sing[0])
    assert m is not None, f"warning missing batch index: {sing[0]!r}"
    failed_batch_id = int(m.group(1))

    # Batch IDs come from `pandas.Categorical.codes`, so they index the
    # categorical's category order. Look up the failing label that way.
    batch_categories = list(synthetic_adata.obs["batch"].cat.categories)
    failed_label = batch_categories[failed_batch_id]

    # ── Run B: same source, surviving batches only, no loess patch ──
    surviving_mask = synthetic_adata.obs["batch"] != failed_label
    adata_b_src = synthetic_adata[surviving_mask].copy()
    adata_b_src.obs["batch"] = adata_b_src.obs[
        "batch"
    ].cat.remove_unused_categories()
    path_b = scx_from_adata(adata_b_src, "hvg_parity_surviving.scx")
    adata_b = pyscx.open(path_b).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(
        adata_b,
        n_top_genes=10,
        flavor="seurat_v3",
        batch_key="batch",
        device="cpu",
    )

    # ── Parity assertion ──
    mask_a = np.asarray(adata_a.var["highly_variable"])
    mask_b = np.asarray(adata_b.var["highly_variable"])
    np.testing.assert_array_equal(
        mask_a,
        mask_b,
        err_msg=(
            "HVG mask differs between 3-batch-with-1-failed and "
            "2-batch-on-surviving runs — the failed batch is still "
            "influencing selection."
        ),
    )

    # `highly_variable_rank` is NaN outside the top-N; the NaN pattern
    # and the finite ranks must both match.
    rank_a = np.asarray(adata_a.var["highly_variable_rank"])
    rank_b = np.asarray(adata_b.var["highly_variable_rank"])
    np.testing.assert_array_equal(np.isnan(rank_a), np.isnan(rank_b))
    finite = ~np.isnan(rank_a)
    np.testing.assert_array_equal(rank_a[finite], rank_b[finite])


def test_all_batches_failed_raises(
    synthetic_adata, scx_from_adata, monkeypatch
):
    """When every batch's loess fit fails there's nothing left to rank
    against. The call must raise `RuntimeError` rather than silently
    producing NaN-ranked HVGs (which is what dividing the cross-batch
    mean by zero surviving batches would yield).
    """
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_all_failed.scx")
    adata = pyscx.open(path).to_anndata(backed=True)
    _patch_loess_to_always_raise(monkeypatch)

    n_batches = len(synthetic_adata.obs["batch"].cat.categories)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match=r"all \d+ batches failed"):
            pyscx.accel.highly_variable_genes(
                adata,
                n_top_genes=10,
                flavor="seurat_v3",
                batch_key="batch",
                device="cpu",
            )

    # Every batch should have surfaced its per-batch singularity warning
    # before the all-failed error is raised.
    sing = [
        w
        for w in caught
        if "skmisc.loess fit failed on batch" in str(w.message)
    ]
    assert len(sing) == n_batches, (
        f"expected {n_batches} per-batch warnings, got {len(sing)}: "
        + "; ".join(str(w.message) for w in sing)
    )
