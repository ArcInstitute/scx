# Rust: training loader and cloud (`scx-loader`, `scx-cloud`)

> Part of the [SCX API reference](README.md). See also [docs/training.md](../training.md) and
[docs/cloud.md](../cloud.md).

## scx-loader — Training Data Loader

### `TrainingPipeline`
Triple-buffered Rust pipeline: tokio I/O → rayon decode → Python/GPU.

```rust
let pipeline = TrainingPipeline::new("file.scx", LoaderConfig {
    batch_size: 1024,
    hvg_indices: Some(hvg_array),
    normalize: true,
    log1p: true,
    ..Default::default()
})?;
pipeline.start_epoch()?;
// `next_batch` returns `Result<Option<Batch>>`: `Ok(None)` = clean epoch end,
// `Err(e)` = a mid-epoch I/O/decode fault (propagate it — don't treat as EOF).
while let Some(batch) = pipeline.next_batch()? {
    // batch.x: dense f32 matrix, batch.obs_columns: Vec<Vec<f32>>
}
```

## scx-cloud — Cloud Operations

### `cloud_optimize(input, output) → Result<()>`
Rewrite file with front-of-file catalog for single-read cloud opens.

### `explode(input, output_dir) → Result<()>`
Packed `.scx` → exploded `.scxd` directory (one file per section).

### `pack(input_dir, output) → Result<()>`
Exploded `.scxd` directory → packed `.scx` (cloud-ready by default).

### `pull(source, dest, options) → Result<PullStats>`
Streaming cloud → local packed file. Supports parallel downloads and selective filter.

### `push(source, dest, options) → Result<PushStats>`
Streaming local → cloud exploded directory. Parallel uploads.

### `CloudReader::open_cloud(url) → Result<CloudReader>`
Direct cloud reads without full download. Supports `.scxd/`, cloud-ready `.scx`, and non-cloud-ready `.scx`.
