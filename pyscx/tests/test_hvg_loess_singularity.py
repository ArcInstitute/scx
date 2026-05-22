"""LOESS singularity resilience for per-batch HVG — see SCX-USER-REPORT-2026-05-21 B1.

The pyscx multi-batch `seurat_v3` path runs one `skmisc.loess.fit()` per
batch. On real Census data, batches with very few cells or with
near-collinear log-mean / log-variance trigger a `ValueError` from
`_loess.pyx` ("There are other near singularities as well. ...").

Pre-fix: the exception propagated as an uncaught traceback, killing the
whole HVG call.

Post-fix: the exception is caught per-batch; a `UserWarning` is emitted
naming the failing batch and its cell count; the batch's `estimat_var`
stays all-zero so it contributes no normalised variance and is excluded
from the median-rank aggregation. Other batches proceed normally.

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
