"""F1: ScxDataModule integration tests (Phase2-Step7 §F1)."""

import pytest

torch = pytest.importorskip(
    "torch", reason="torch not available or broken", exc_type=ImportError
)


def test_scx_datamodule_train_dataloader(query_adata, scx_from_adata):
    """ScxDataModule.train_dataloader() returns a working DataLoader."""
    import torch
    from pyscx.scx_integrations.scvi import ScxDataModule

    path = scx_from_adata(query_adata, "scvi_test.scx")

    dm = ScxDataModule(
        scx_path=path,
        batch_size=32,
    )

    loader = dm.train_dataloader()
    assert loader is not None

    batch_count = 0
    for batch in loader:
        batch_count += 1
        assert "X" in batch
        assert isinstance(batch["X"], torch.Tensor)
        assert batch["X"].dtype == torch.float32
        assert batch["X"].ndim == 2
        assert "cell_indices" in batch

    assert batch_count > 0


def test_scx_datamodule_properties(query_adata, scx_from_adata):
    """ScxDataModule exposes n_obs, n_vars, n_output_genes."""
    from pyscx.scx_integrations.scvi import ScxDataModule

    path = scx_from_adata(query_adata, "props.scx")

    dm = ScxDataModule(scx_path=path, batch_size=16)
    assert dm.n_obs == query_adata.n_obs
    assert dm.n_vars == query_adata.n_vars
    assert dm.n_output_genes == query_adata.n_vars  # no HVG projection


def test_scx_datamodule_val_dataloader(query_adata, scx_from_adata):
    """val_dataloader() returns None (not yet supported)."""
    from pyscx.scx_integrations.scvi import ScxDataModule

    path = scx_from_adata(query_adata, "val.scx")
    dm = ScxDataModule(scx_path=path, batch_size=16)
    assert dm.val_dataloader() is None


def test_scx_datamodule_is_lightning_datamodule(query_adata, scx_from_adata):
    """ScxDataModule inherits from pl.LightningDataModule."""
    import lightning.pytorch as pl
    from pyscx.scx_integrations.scvi import ScxDataModule

    path = scx_from_adata(query_adata, "lightning.scx")
    dm = ScxDataModule(scx_path=path, batch_size=16)
    assert isinstance(dm, pl.LightningDataModule)
