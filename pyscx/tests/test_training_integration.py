"""TrainingDataset integration tests."""


def test_training_dataset_iterates_epoch(query_adata, scx_from_adata):
    """TrainingDataset iterates a full epoch without errors."""
    import pyscx

    path = scx_from_adata(query_adata, "train.scx")

    ds = pyscx.TrainingDataset(
        path=path,
        batch_size=32,
    )

    batch_count = 0
    for batch in ds:
        batch_count += 1
        assert "X" in batch
        assert "cell_indices" in batch
        assert "obs" in batch

    assert batch_count > 0


def test_training_after_append(query_adata, scx_from_adata):
    """After append, TrainingDataset sees new cells."""
    import pyscx

    path = scx_from_adata(query_adata, "train_append.scx")
    path2 = scx_from_adata(query_adata, "train_append_extra.scx")

    # Count cells from n_obs before append
    n_before = pyscx.TrainingDataset(path=path, batch_size=1000).n_obs

    # Append
    pyscx.append(path, path2)

    # n_obs after append should be double
    n_after = pyscx.TrainingDataset(path=path, batch_size=1000).n_obs
    assert n_after == n_before * 2


def test_training_after_mark_deleted(query_adata, scx_from_adata):
    """After mark_deleted, TrainingDataset excludes deleted cells."""
    import pyscx

    path = scx_from_adata(query_adata, "train_del.scx")

    n_before = pyscx.TrainingDataset(path=path, batch_size=1000).n_obs

    # Delete 10 cells
    pyscx.mark_deleted(path, list(range(10)))

    # n_obs should reflect deletion
    ds_after = pyscx.TrainingDataset(path=path, batch_size=1000)
    # Count actual cells yielded
    total_after = 0
    for batch in ds_after:
        # X shape is [n_rows, n_genes]
        total_after += batch["X"].shape[0]

    assert total_after == n_before - 10


def test_training_and_query_concurrent(query_adata, scx_from_adata):
    """TrainingDataset and QueryPipeline can open the same file (read-only safety)."""
    import pyscx

    path = scx_from_adata(query_adata, "concurrent.scx")

    # Both should work without errors
    ds = pyscx.TrainingDataset(path=path, batch_size=32)
    result = pyscx.open(path).query().collect()

    assert result.n_obs == query_adata.n_obs

    batch_count = 0
    for batch in ds:
        batch_count += 1
    assert batch_count > 0
