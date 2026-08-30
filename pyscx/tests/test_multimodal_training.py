"""Phase H — multimodal training loader tests.

Covers:
- `pyscx.MultimodalTrainingDataset` round-trip on a CITE-seq fixture:
  tuple/dict batches whose row indices align across modalities.
- `pyscx.TrainingDataset(path)` on a multimodal file falls back to
  the alphabetically-first modality and emits a UserWarning (Phase H.3
  backward-compat path).
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest


@pytest.fixture
def cite_seq_path():
    """Tiny CITE-seq SCX fixture written via pyscx.from_mudata."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(0)
    n_obs, rna_n_vars, adt_n_vars = 64, 50, 8

    rna_dense = rng.poisson(lam=0.4, size=(n_obs, rna_n_vars)).astype(np.float32)
    adt_dense = rng.poisson(lam=0.4, size=(n_obs, adt_n_vars)).astype(np.float32)
    rna_ad = anndata.AnnData(X=sp.csr_matrix(rna_dense))
    rna_ad.var_names = [f"g{i}" for i in range(rna_n_vars)]
    adt_ad = anndata.AnnData(X=sp.csr_matrix(adt_dense))
    adt_ad.var_names = [f"a{i}" for i in range(adt_n_vars)]
    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(mu, path)
        yield path, n_obs, rna_n_vars, adt_n_vars


def test_multimodal_dataset_dict_batches(cite_seq_path):
    """Phase H.1 / H.2: dict-mode batches expose per-modality X arrays
    and a shared `cell_indices` row ordering."""
    import pyscx

    path, n_obs, rna_n_vars, adt_n_vars = cite_seq_path
    ds = pyscx.MultimodalTrainingDataset(
        path,
        modalities=["rna", "adt"],
        batch_size=16,
        normalize=False,
        log1p=False,
        seed=42,
    )
    assert ds.n_obs == n_obs
    assert ds.modality_names == ["rna", "adt"]
    nv = ds.n_vars
    assert nv["rna"] == rna_n_vars
    assert nv["adt"] == adt_n_vars

    seen_rows = 0
    for batch in ds:
        assert "X" in batch
        assert "cell_indices" in batch
        x_dict = batch["X"]
        assert "rna" in x_dict
        assert "adt" in x_dict
        n_rna = x_dict["rna"].shape[0]
        n_adt = x_dict["adt"].shape[0]
        # Per-modality batches share the same row count.
        assert n_rna == n_adt
        # Per-modality n_vars differs.
        assert x_dict["rna"].shape[1] == rna_n_vars
        assert x_dict["adt"].shape[1] == adt_n_vars
        # cell_indices rank matches the batch row count.
        assert batch["cell_indices"].shape[0] == n_rna
        seen_rows += n_rna
    # Total cells iterated equals n_obs (one full epoch).
    assert seen_rows == n_obs
    ds.close()


def test_multimodal_close_then_iterate_starts_a_fresh_epoch(cite_seq_path):
    """`close()` is re-openable here too — the sibling of
    `TestCloseIsReopenable` in `test_training_loader.py`.

    Asserted nowhere before: all seven `ds.close()` calls in this file are
    end-of-body teardown. It is the pin for ORG-9.10-4's `closed` getter, which
    on this class can only mean "torn down right now".
    """
    import pyscx

    path, n_obs, _, _ = cite_seq_path
    ds = pyscx.MultimodalTrainingDataset(
        path, modalities=["rna", "adt"], batch_size=16, normalize=False,
        log1p=False, seed=42,
    )

    def epoch_rows():
        return sum(b["cell_indices"].shape[0] for b in ds)

    assert ds.closed is False, "a fresh dataset has never been closed"
    assert epoch_rows() == n_obs
    ds.close()
    assert ds.closed is True
    assert epoch_rows() == n_obs
    assert ds.closed is False
    ds.close()


def test_multimodal_uniform_batch_across_wide_and_narrow_modalities(tmp_path):
    """Regression: a wide modality (many genes) and a narrow one must not
    desync their per-batch row ordering.

    Each modality runs its own ``TrainingPipeline`` with a per-modality
    memory budget. Without a uniform-batch guard the wide modality's
    ``batch_size`` is shrunk independently by the memory-budget auto-tuner
    (``compute_memory_budget``) below the narrow modality's, so the two
    pipelines chunk the (identical) shuffled cell order into different
    batch boundaries and the ``__next__`` alignment check raises
    ``RuntimeError``. The loader now pins a uniform effective ``batch_size``
    across modalities, so the epoch iterates cleanly with aligned rows.
    """
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(0)
    n_obs, rna_n_vars, wide_n_vars = 400, 20, 8000
    rna = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.4, size=(n_obs, rna_n_vars)).astype(np.float32))
    )
    rna.var_names = [f"g{i}" for i in range(rna_n_vars)]
    wide = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.4, size=(n_obs, wide_n_vars)).astype(np.float32))
    )
    wide.var_names = [f"p{i}" for i in range(wide_n_vars)]
    mu = mudata.MuData({"rna": rna, "prot": wide})
    mu.obs_names = [f"c{i}" for i in range(n_obs)]
    path = str(tmp_path / "wide.scx")
    pyscx.from_mudata(mu, path)

    # A small explicit budget forces the wide modality's per-modality
    # batch below the requested 256; the narrow modality would otherwise
    # keep 256. The uniform-batch guard reconciles them.
    ds = pyscx.MultimodalTrainingDataset(
        path,
        modalities=["rna", "prot"],
        batch_size=256,
        max_memory_mb=64,
        normalize=False,
        log1p=False,
        seed=1,
    )
    seen = 0
    first_rows = None
    for batch in ds:
        n_rna = batch["X"]["rna"].shape[0]
        n_prot = batch["X"]["prot"].shape[0]
        assert n_rna == n_prot, "per-modality batches must share a row count"
        if first_rows is None:
            first_rows = n_rna
        seen += n_rna
    assert seen == n_obs, "one full epoch over all cells"
    # The wide modality forced the pinned batch below the requested 256,
    # so this run genuinely exercised the uniform-batch reconciliation.
    assert first_rows is not None and first_rows < 256
    ds.close()


def test_multimodal_dataset_tuple_batches(cite_seq_path):
    """Phase H.2: `return_dict=False` yields tuples of X arrays."""
    import pyscx

    path, _, rna_n_vars, adt_n_vars = cite_seq_path
    ds = pyscx.MultimodalTrainingDataset(
        path,
        modalities=["rna", "adt"],
        batch_size=8,
        normalize=False,
        log1p=False,
        return_dict=False,
        seed=42,
    )
    for batch in ds:
        # Tuple mode: (X_rna, X_adt) in the order of `modalities`.
        assert isinstance(batch, tuple)
        assert len(batch) == 2
        rna_x, adt_x = batch
        assert rna_x.shape[0] == adt_x.shape[0]
        assert rna_x.shape[1] == rna_n_vars
        assert adt_x.shape[1] == adt_n_vars
    ds.close()


def test_multimodal_dataset_unknown_modality_raises(cite_seq_path):
    """Constructor raises a clear error when an unknown modality is requested."""
    import pyscx

    path, *_ = cite_seq_path
    with pytest.raises(RuntimeError, match="modality named 'spatial'"):
        pyscx.MultimodalTrainingDataset(path, modalities=["rna", "spatial"])


def test_multimodal_dataset_rejects_single_modality_file(tmp_path):
    """On a single-modality file, MultimodalTrainingDataset directs the user back to TrainingDataset."""
    pytest.importorskip("anndata")
    import anndata
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(0)
    adata = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.3, size=(20, 30)).astype(np.float32))
    )
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.obs_names = [f"c{i}" for i in range(20)]
    path = str(tmp_path / "single.scx")
    pyscx.from_anndata(adata, path)

    with pytest.raises(RuntimeError, match="single-modality"):
        pyscx.MultimodalTrainingDataset(path, modalities=["rna"])


def test_training_dataset_multimodal_warns(cite_seq_path):
    """Phase H.3: TrainingDataset on a multimodal file emits UserWarning
    and falls back to the alphabetically-first modality."""
    import pyscx

    path, n_obs, _, adt_n_vars = cite_seq_path
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = pyscx.TrainingDataset(path, batch_size=16, normalize=False, log1p=False)
        # The warning should fire during construction.
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("multimodal" in m and "MultimodalTrainingDataset" in m for m in msgs), (
            f"expected multimodal-fallback UserWarning, got: {msgs}"
        )
    # Alphabetically-first modality is "adt"; verify the dataset
    # resolved to the ADT n_vars (8), not the RNA n_vars.
    assert ds.n_vars == adt_n_vars
    assert ds.n_obs == n_obs
    ds.close()


def test_training_dataset_explicit_modality_kwarg(cite_seq_path):
    """Phase H.1: explicit `modality=` selects without warning."""
    import pyscx

    path, n_obs, rna_n_vars, _ = cite_seq_path
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = pyscx.TrainingDataset(
            path, modality="rna", batch_size=16, normalize=False, log1p=False
        )
        # No multimodal-fallback warning when modality is explicit.
        multimodal_warnings = [
            str(w.message)
            for w in caught
            if issubclass(w.category, UserWarning)
            and "MultimodalTrainingDataset" in str(w.message)
        ]
        assert not multimodal_warnings, multimodal_warnings
    assert ds.n_vars == rna_n_vars
    assert ds.n_obs == n_obs
    ds.close()


def test_hvg_indices_are_not_range_checked_on_a_shared_multimodal_panel(
    cite_seq_path,
):
    """`MultimodalTrainingDataset` alone skips the HVG range check.

    Everything else — `TrainingDataset` and `IndexPlanDataset` — rejects an
    `hvg_indices` entry `>= n_vars`, because it matches no column and yields
    an output feature that is silently always zero. `MultimodalTrainingDataset`
    cannot: it fans one panel across every selected modality and they have
    different widths, so an RNA-sized panel would be rejected outright by this
    fixture's 8-feature ADT modality.

    This test pins that decision. Without it the opt-out in
    `TrainingPipeline::new` reads as dead code and a later cleanup would
    silently re-enable the check, turning working multimodal calls into
    errors. `docs/training.md` documents the trade-off for users.

    The companion test below pins the *other* half: the opt-out is keyed to
    this one caller, not to "a modality is selected".
    """
    import pyscx

    path, n_obs, rna_n_vars, adt_n_vars = cite_seq_path
    assert adt_n_vars < rna_n_vars  # the panel below is OOR for adt only

    panel = np.array([0, 5, rna_n_vars - 1], dtype=np.uint32)

    ds = pyscx.MultimodalTrainingDataset(
        path,
        modalities=["rna", "adt"],
        batch_size=16,
        hvg_indices=panel,
        normalize=False,
        log1p=False,
        seed=42,
    )
    batch = next(iter(ds))
    # Both modalities report the panel width.
    assert batch["X"]["rna"].shape[1] == len(panel)
    assert batch["X"]["adt"].shape[1] == len(panel)
    # Panel entries 0 and 5 are real adt features; entry 49 is past its 8, so
    # that column is the silently-always-zero one this bypass permits. Assert
    # both halves — an all-zero adt block would pass the dead-column check
    # while proving nothing about the projection.
    adt = batch["X"]["adt"]
    assert np.any(adt[:, :2]), "in-range adt columns should carry data"
    assert not np.any(adt[:, 2]), "the out-of-range column is always zero"
    ds.close()


def test_modality_scoped_training_dataset_still_range_checks_hvg(cite_seq_path):
    """A *scoped* `TrainingDataset` is checked — the opt-out is not `modality_id`.

    Keying the opt-out on "a modality is selected" would have been the obvious
    implementation and is wrong: `TrainingDataset(path, modality="adt")` and the
    implicit alphabetically-first fallback both set `modality_id`, yet each has
    exactly one panel and one unambiguous `n_vars`. Under that keying every
    multimodal `TrainingDataset` silently kept the dead-zero-column behaviour
    while the docs promised it was rejected.

    Both entry points are asserted because they reach `modality_id` by different
    routes (explicit kwarg vs. fallback), and only one of them warns.
    """
    import pyscx

    path, _n_obs, rna_n_vars, adt_n_vars = cite_seq_path
    panel = np.array([0, 5, rna_n_vars - 1], dtype=np.uint32)  # OOR for adt

    # Explicit modality=.
    with pytest.raises(RuntimeError, match="out of range"):
        pyscx.TrainingDataset(
            path,
            modality="adt",
            batch_size=16,
            hvg_indices=panel,
            normalize=False,
            log1p=False,
        )

    # Implicit fallback: no modality= on a multimodal file resolves the
    # alphabetically-first modality, which here is "adt".
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="out of range"):
            pyscx.TrainingDataset(
                path,
                batch_size=16,
                hvg_indices=panel,
                normalize=False,
                log1p=False,
            )

    # And a panel that IS in range for the scoped modality still builds.
    ds = pyscx.TrainingDataset(
        path,
        modality="adt",
        batch_size=16,
        hvg_indices=np.array([0, adt_n_vars - 1], dtype=np.uint32),
        normalize=False,
        log1p=False,
    )
    assert ds.n_output_genes == 2
    ds.close()
