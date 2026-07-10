//! GPU training-loader decode (Phase D) — feature-gated skeleton.
//!
//! This module is compiled only under `--features gpu`, which pulls in the
//! hard-cudarc `scx-gpu` crate (CUDA toolchain required to build). The default
//! CPU loader path (`io_stage` → `decode_stage` → `python`) is untouched when
//! this feature is off.
//!
//! # Status: D1 (feature edge) only
//!
//! Per `SHUFDELTA-LOADER-END-TO-END.md` §12, Phase D moves loader decode onto
//! the device so ShufDeltaZstd's smaller PCIe payload becomes a training-
//! throughput lever. Only the feature edge (D1) has landed. The device
//! dense-scatter kernel + `GpuTrainingDataset`/DLPack contract (D2), device
//! HVG/normalize (D3), and multi-block scan occupancy (D4) are **gated on D0**
//! — a profiling deliverable confirming stage-1 codec decode is a real
//! critical-path training ceiling (spec §12.4 / open-question Q3). Do not add
//! kernel or `GpuTrainingDataset` code here until D0 shows the win is real.
//!
//! Device-decode reuse points for D2 (already shipped in `scx-gpu`, today
//! reached only via `to_gpu_anndata`):
//! - `scx_gpu::shufdelta_gpu::decode_indices_frame_to_device`
//! - `scx_gpu::shufdelta_gpu::decode_values_frame_to_device`
//! - `scx_gpu::shufdelta_gpu::decode_framed_shufdelta_gpu_pipelined`

// Touch the dependency so the feature graph is exercised and an accidental
// drop of the `scx-gpu` edge fails the `--features gpu` build. Replaced by real
// device-decode wiring in D2.
#[allow(unused_imports)]
use scx_gpu as _;
