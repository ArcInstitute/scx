//! Highly variable gene statistics (streaming mean/variance + clipped sums).
//!
//! [`cpu`] holds the single-pass host accumulators; [`gpu`] (behind the `gpu`
//! feature) holds the device-dispatched wrappers over the `scx-gpu` kernels.

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::{
    streaming_clip_square_sum, streaming_clip_square_sum_batched, streaming_mean_var,
    streaming_mean_var_batched, streaming_mean_var_expm1, BatchedHvgStats, HvgStats,
};
#[cfg(feature = "gpu")]
pub use gpu::{
    streaming_clip_square_sum_batched_with_device, streaming_clip_square_sum_with_device,
    streaming_mean_var_batched_with_device, streaming_mean_var_with_device,
};
