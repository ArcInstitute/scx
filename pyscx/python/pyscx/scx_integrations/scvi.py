"""scVI integration for SCX — PyTorch Lightning DataModule.

Provides ``ScxDataModule``, a ready-to-use PyTorch Lightning DataModule
that wraps the SCX ``TrainingDataset`` for scVI model training.

Usage::

    from pyscx.scx_integrations.scvi import ScxDataModule

    dm = ScxDataModule(
        scx_path="experiment.scx",
        batch_size=1024,
        hvg_indices=hvg_array,
    )
    model = scvi.model.SCVI(dm.adata_manager)
    model.train(datamodule=dm)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def _check_lightning():
    """Import pytorch_lightning / lightning, raising ImportError if missing."""
    try:
        import lightning.pytorch as pl

        return pl
    except ImportError:
        pass
    try:
        import pytorch_lightning as pl

        return pl
    except ImportError:
        raise ImportError(
            "PyTorch Lightning is required for ScxDataModule. "
            "Install it with: pip install lightning"
        )


class _ScxIterableDataset(IterableDataset):
    """Thin PyTorch IterableDataset wrapper around pyscx.TrainingDataset.

    Each iteration yields a dict of torch tensors:
      - ``X``: (batch_size, n_genes) float32 dense expression
      - ``cell_indices``: (batch_size,) int64
      - ``obs``: dict of metadata columns (if requested)
    """

    def __init__(self, scx_path: str, batch_size: int, **kwargs: Any):
        super().__init__()
        self.scx_path = scx_path
        self.batch_size = batch_size
        self.kwargs = kwargs
        self._dataset = None

    def _ensure_dataset(self):
        """Lazily create the TrainingDataset on first iteration."""
        if self._dataset is None:
            import pyscx

            self._dataset = pyscx.TrainingDataset(
                path=self.scx_path,
                batch_size=self.batch_size,
                **self.kwargs,
            )

    @property
    def n_obs(self) -> int:
        self._ensure_dataset()
        return self._dataset.n_obs

    @property
    def n_vars(self) -> int:
        self._ensure_dataset()
        return self._dataset.n_vars

    @property
    def n_output_genes(self) -> int:
        self._ensure_dataset()
        return self._dataset.n_output_genes

    def __iter__(self):
        self._ensure_dataset()
        for batch in self._dataset:
            yield {
                "X": torch.from_numpy(batch["X"]),
                "cell_indices": torch.from_numpy(
                    batch["cell_indices"].astype(np.int64)
                ),
            }


class ScxDataModule:
    """PyTorch Lightning DataModule for SCX-backed scVI training.

    Wraps ``pyscx.TrainingDataset`` in a PyTorch ``DataLoader`` compatible
    with scVI's training loop.

    .. warning::

        ``normalize`` and ``log1p`` default to ``True`` (inherited from
        ``TrainingDataset``), so batches are log-normalized by default. **scVI
        and other count-likelihood models require raw integer counts** — pass
        ``normalize=False, log1p=False`` so the loader streams raw counts. The
        on-by-default transforms are kept here only for parity with
        ``TrainingDataset``; they are *not* the right default for scVI.

    Parameters
    ----------
    scx_path : str
        Path to the ``.scx`` file.
    batch_size : int
        Mini-batch size.
    hvg_indices : array-like or None
        Gene indices for HVG projection. None = all genes. Passed through to
        the loader, which sorts and deduplicates the panel — so batch columns
        are in ascending gene-index order regardless of the order given, and
        ``n_output_genes`` can be smaller than ``len(hvg_indices)``. Use
        ``np.unique(hvg_indices)`` to recover the column order.
    normalize : bool
        Apply total-count normalization (default True). Set ``False`` for
        raw-count output (required by scVI — see the warning above).
    log1p : bool
        Apply log1p transformation (default True). Set ``False`` for
        raw-count output (required by scVI — see the warning above).
    target_sum : float
        Normalization target sum (default 1e4).
    seed : int
        RNG seed for reproducibility (default 42).
    **kwargs
        Additional keyword arguments passed to ``TrainingDataset``.
    """

    def __init__(
        self,
        scx_path: str,
        batch_size: int = 1024,
        hvg_indices: Any | None = None,
        normalize: bool = True,
        log1p: bool = True,
        target_sum: float = 1e4,
        seed: int = 42,
        **kwargs: Any,
    ):
        # Dynamically inherit from LightningDataModule if available
        pl = _check_lightning()
        self.__class__ = type(
            "ScxDataModule",
            (pl.LightningDataModule,),
            dict(self.__class__.__dict__),
        )
        pl.LightningDataModule.__init__(self)

        self.scx_path = scx_path
        self.batch_size = batch_size

        # Build kwargs for TrainingDataset
        self._ds_kwargs: dict[str, Any] = {
            "normalize": normalize,
            "log1p": log1p,
            "target_sum": target_sum,
            "seed": seed,
        }
        if hvg_indices is not None:
            self._ds_kwargs["hvg_indices"] = list(
                int(x) for x in hvg_indices
            )
        self._ds_kwargs.update(kwargs)

        self._train_dataset = _ScxIterableDataset(
            scx_path=scx_path,
            batch_size=batch_size,
            **self._ds_kwargs,
        )

    @property
    def n_obs(self) -> int:
        """Total number of observations (cells)."""
        return self._train_dataset.n_obs

    @property
    def n_vars(self) -> int:
        """Total number of variables (genes)."""
        return self._train_dataset.n_vars

    @property
    def n_output_genes(self) -> int:
        """Number of output genes per batch (HVG count if projection active)."""
        return self._train_dataset.n_output_genes

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for training.

        Uses ``batch_size=None`` and ``num_workers=0`` because the Rust
        ``TrainingDataset`` handles batching and threading internally.
        """
        return DataLoader(
            self._train_dataset,
            batch_size=None,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Return validation DataLoader. Currently not supported."""
        return None
