# ML training data loading

> Part of the [SCX + scanpy guide](README.md).

For large-scale model training, SCX provides a high-performance data loader
that bypasses Python I/O entirely. Three dataset types cover different
ML patterns:

- **`TrainingDataset`** — sequential streaming for standard training loops
  (autoencoder, scVI, scGPT). 82× faster than TileDB-SOMA-ML on 1M cells.
- **`IndexPlanDataset`** — paired `(perturbed, control)` cell reads for
  perturbation training, contrastive learning, and donor-matched designs.
- **`MultimodalTrainingDataset`** — cell-aligned multi-assay batches
  (RNA + ADT + ATAC) from a single multimodal SCX file.

```python
import pyscx
import torch

dataset = pyscx.TrainingDataset(
    "atlas.scx",
    batch_size=1024,
    hvg_indices=hvg_array,   # decode only HVGs → less data
    normalize=True,
    log1p=True,
    obs_columns=["cell_type", "batch"],  # metadata in each batch
)

for batch in dataset:
    x = torch.from_numpy(batch["X"]).to(device)
    cell_types = batch["obs"]["cell_type"]  # {"codes": ndarray, "categories": list}
    # model.forward(), loss.backward(), ...

dataset.close()
```

The training loader uses a triple-buffered Rust pipeline (I/O → decode → GPU)
with zero Python on the hot path. For scVI, use the built-in
[`ScxDataModule`](../api/python-training.md#scvi-integration-pyscxscx_integrationsscvi)
PyTorch Lightning DataModule.

See the dedicated [ML Training Guide](../training.md) for end-to-end examples,
train/val split handling, PyTorch DataLoader compatibility, and migration
from h5ad-based training loops.
