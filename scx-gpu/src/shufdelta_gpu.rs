//! GPU decode transforms for ShufDeltaZstd (codec_id = 5) shards.
//!
//! The CPU host runs zstd (inherently sequential within a frame); these
//! launchers take over the two byte-plane transforms that Phase-0 profiling
//! showed dominate the CPU decode: the per-plane wrapping-u8 delta prefix scan
//! (`undelta_planes_gpu`) and the plane-major → element-major transpose fused
//! with the widen-to-i32/f32 convert (`unshuffle_convert_*_gpu`).
//!
//! The two frame helpers ([`decode_indices_frame_to_device`] /
//! [`decode_values_frame_to_device`]) tie them together: zstd-decompress one
//! sub-stream frame on the host to its intermediate "still shuffled+delta'd"
//! plane bytes (via `scx_codec::zstd_decompress_bounded`), upload those, and
//! run the kernels — so only the narrower pre-convert plane bytes cross PCIe.

use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::safe::{
    CudaEvent, CudaSlice, CudaView, CudaViewMut, LaunchConfig, PinnedHostSlice,
};
use cudarc::driver::PushKernelArg;

use scx_codec::{zstd_decompress_bounded, RowGroupSpan, ValueEncoding};
use scx_format_io::shard::{resolve_block_index, ShardHeader};

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::profile::{self, CodecClass};
use crate::shard_decode::{DeviceDecodeStats, GpuCsr};

/// Compiled PTX for the shufdelta kernels (produced by build.rs via nvcc --ptx).
const SHUFDELTA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/shufdelta.ptx"));

/// In-place per-plane wrapping-u8 inclusive prefix scan (undo byte-delta).
///
/// `buf` is plane-major `[width][n]`; each of the `width` planes is scanned
/// independently. Launches one thread block per plane (`grid_dim.x == width`).
fn undelta_planes_gpu(
    dev: &GpuDevice,
    buf: &mut CudaSlice<u8>,
    n: u32,
    width: u32,
) -> Result<(), GpuError> {
    if n == 0 || width == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(SHUFDELTA_PTX)?;
    let kernel = module
        .load_function("undelta_planes_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load undelta_planes_kernel: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (width, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(buf)
            .arg(&n)
            .arg(&width)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("undelta_planes_kernel: {e}")))?;
    Ok(())
}

/// Unshuffle (plane-major → element-major) fused with widen-to-i32/f32.
///
/// `out_is_float == 0` writes `i32` (indices), `1` writes `f32` (values). The
/// kernel treats `out` as an opaque device pointer, so it is generic over the
/// output element type `T` — the caller allocates `out` as `i32` or `f32` to
/// match `out_is_float` and the pointer is forwarded type-agnostically.
fn launch_unshuffle_convert<T: cudarc::driver::DeviceRepr>(
    dev: &GpuDevice,
    src: &CudaSlice<u8>,
    out: &mut CudaSlice<T>,
    n: u32,
    width: u32,
    out_is_float: u32,
) -> Result<(), GpuError> {
    let module = dev.load_module_cached(SHUFDELTA_PTX)?;
    let kernel = module
        .load_function("unshuffle_convert_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load unshuffle_convert_kernel: {e}")))?;

    let threads: u32 = 256;
    let grid = n.div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(src)
            .arg(out)
            .arg(&n)
            .arg(&width)
            .arg(&out_is_float)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("unshuffle_convert_kernel: {e}")))?;
    Ok(())
}

/// zstd-decompress one sub-stream frame to its intermediate plane bytes and
/// verify the length is exactly `expected` (a short frame would mis-align the
/// in-place undelta and is treated as malformed input, mirroring the codec's
/// `expect_exact_len`).
fn decompress_frame(compressed: &[u8], expected: usize, which: &str) -> Result<Vec<u8>, GpuError> {
    let planes = zstd_decompress_bounded(compressed, expected)
        .map_err(|e| GpuError::InvalidShard(format!("shufdelta {which} zstd decode: {e}")))?;
    if planes.len() != expected {
        return Err(GpuError::InvalidShard(format!(
            "shufdelta {which} decompressed len {} != expected {}",
            planes.len(),
            expected
        )));
    }
    Ok(planes)
}

/// Decode one ShufDeltaZstd **indices** frame to a device `i32` buffer.
///
/// Pipeline: CPU zstd → upload plane bytes → GPU undelta → GPU unshuffle+widen.
/// Returns the device buffer and the number of host→device bytes uploaded.
pub fn decode_indices_frame_to_device(
    dev: &GpuDevice,
    compressed: &[u8],
    nnz: usize,
    index_width: usize,
) -> Result<(CudaSlice<i32>, u64), GpuError> {
    if nnz == 0 {
        return Ok((dev.alloc_zeros::<i32>(0)?, 0));
    }
    let expected = nnz * index_width;
    let planes = decompress_frame(compressed, expected, "indices")?;
    let mut d_planes = dev.htod_copy(&planes)?;
    undelta_planes_gpu(dev, &mut d_planes, nnz as u32, index_width as u32)?;
    let mut out = dev.alloc_zeros::<i32>(nnz)?;
    launch_unshuffle_convert(dev, &d_planes, &mut out, nnz as u32, index_width as u32, 0)?;
    Ok((out, expected as u64))
}

/// Decode one ShufDeltaZstd **integer values** frame to a device `f32` buffer.
///
/// Integer values are shuffle-only (no delta) on encode, so this skips the
/// undelta step. Float value encodings are zstd-only and are **not** handled
/// here — the caller must route them to the host-bounce path.
pub fn decode_values_frame_to_device(
    dev: &GpuDevice,
    compressed: &[u8],
    nnz: usize,
    value_encoding: ValueEncoding,
) -> Result<(CudaSlice<f32>, u64), GpuError> {
    if !value_encoding.is_integer() {
        return Err(GpuError::InvalidShard(
            "float ShufDeltaZstd values have no GPU decode path (host-bounce only)".into(),
        ));
    }
    if nnz == 0 {
        return Ok((dev.alloc_zeros::<f32>(0)?, 0));
    }
    let width = value_encoding.byte_width();
    let expected = nnz * width;
    let planes = decompress_frame(compressed, expected, "values")?;
    let d_planes = dev.htod_copy(&planes)?;
    // `unshuffle_convert_kernel` writes f32 when out_is_float == 1.
    let mut out = dev.alloc_zeros::<f32>(nnz)?;
    launch_unshuffle_convert(dev, &d_planes, &mut out, nnz as u32, width as u32, 1)?;
    Ok((out, expected as u64))
}

// ---------------------------------------------------------------------------
// Phase 1.5: pipelined framed decode (parallel CPU zstd + multi-stream H2D).
//
// Profiling (GPU ShufDeltaZstd decode, Phase 0/1) showed the sequential path's floor
// is single-threaded CPU zstd, and that `htod_copy` (pageable, NULL stream)
// blocks the host. This path fans the per-group zstd out across worker threads
// (bounded channel → backpressure) and overlaps it with GPU work: uploads run
// async on a dedicated copy stream into a 2-slot device staging ring, gated
// against the compute stream (`dev.stream()`) by CUDA events — the same
// copy↔compute handshake as `gpu_shard_source.rs`. Per-group nnz offsets are a
// prefix sum computed up front, so groups may be produced/consumed out of order
// (each kernel writes its own disjoint `combined[base..]` region).
// ---------------------------------------------------------------------------

/// A reused host staging buffer: page-locked when possible (true async H2D),
/// falling back to a pageable `Vec` if `alloc_pinned` fails.
enum HostPlaneBuf {
    Pinned(PinnedHostSlice<u8>),
    Pageable(Vec<u8>),
}

impl HostPlaneBuf {
    fn new(dev: &GpuDevice, cap: usize) -> Self {
        // SAFETY: `alloc_pinned` is unsafe only in that the buffer is
        // uninitialized; we fully overwrite the used prefix before every upload.
        match unsafe { dev.context().alloc_pinned::<u8>(cap.max(1)) } {
            Ok(p) => HostPlaneBuf::Pinned(p),
            Err(_) => HostPlaneBuf::Pageable(vec![0u8; cap.max(1)]),
        }
    }

    /// Copy `src` into the buffer's prefix, returning a host slice of exactly
    /// `src.len()` bytes suitable for an (async, when pinned) H2D upload.
    fn stage<'a>(&'a mut self, src: &[u8]) -> Result<&'a [u8], GpuError> {
        let n = src.len();
        match self {
            HostPlaneBuf::Pinned(p) => {
                let dst = p
                    .as_mut_slice()
                    .map_err(|e| GpuError::CudaError(format!("pinned as_mut_slice: {e}")))?;
                dst[..n].copy_from_slice(src);
                let view = p
                    .as_slice()
                    .map_err(|e| GpuError::CudaError(format!("pinned as_slice: {e}")))?;
                Ok(&view[..n])
            }
            HostPlaneBuf::Pageable(v) => {
                v[..n].copy_from_slice(src);
                Ok(&v[..n])
            }
        }
    }
}

/// Decompressed plane bytes for one row-group, produced by a worker thread.
type GroupPlanes = (usize, Vec<u8>, Vec<u8>); // (group_idx, indices_planes, values_planes)

/// Combined device CSR buffers from the pipelined decode + the H2D byte total.
pub struct PipelinedCsr {
    pub indptr: CudaSlice<i64>,
    pub indices: CudaSlice<i32>,
    pub data: CudaSlice<f32>,
    pub host_uploaded_bytes: u64,
}

/// Pipelined GPU decode of a framed ShufDeltaZstd shard. Returns the combined
/// device CSR buffers plus the host→device byte total. Caller supplies the
/// resolved `spans` and derived dims (mirrors the sequential path's setup).
#[allow(clippy::too_many_arguments)]
pub fn decode_framed_shufdelta_gpu_pipelined(
    dev: &GpuDevice,
    spans: &[RowGroupSpan],
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_width: usize,
) -> Result<PipelinedCsr, GpuError> {
    let value_width = value_encoding.byte_width();

    // Precompute per-group nnz offsets + the full global indptr on the host
    // (tiny; the large index/value frames go to the device). Offsets let the
    // GPU consumer place each group independently, so producers can run ahead
    // and out of order.
    let mut offsets: Vec<usize> = Vec::with_capacity(spans.len());
    let mut combined_indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    combined_indptr.push(0);
    let mut nnz_base = 0usize;
    let t_indptr = profile::start();
    for span in spans {
        offsets.push(nnz_base);
        let g_rows = span.n_rows as usize;
        let g_indptr = scx_codec::decode_row_group_indptr_only(
            scx_codec::CodecId::ShufDeltaZstd,
            span,
            indptr_bytes,
        )?;
        debug_assert_eq!(g_indptr.len(), g_rows + 1);
        for &local in &g_indptr[1..=g_rows] {
            combined_indptr.push(nnz_base as i64 + local);
        }
        nnz_base += span.nnz as usize;
    }
    profile::record_host_decode_since(CodecClass::Generic, t_indptr);
    debug_assert_eq!(nnz_base, nnz);
    debug_assert_eq!(combined_indptr.len(), n_rows + 1);

    let mut combined_indices = dev.alloc_zeros::<i32>(nnz)?;
    let mut combined_data = dev.alloc_zeros::<f32>(nnz)?;
    let mut host_uploaded_bytes = (combined_indptr.len() * 8) as u64;

    if nnz == 0 {
        let d_indptr = dev.htod_copy(&combined_indptr)?;
        dev.synchronize()?;
        return Ok(PipelinedCsr {
            indptr: d_indptr,
            indices: combined_indices,
            data: combined_data,
            host_uploaded_bytes,
        });
    }

    // Dedicated non-blocking copy stream for async H2D; compute stays on the
    // default stream. (new_stream flips the context into multi-stream mode —
    // acceptable here; to_gpu_anndata does no graph capture.)
    let copy_stream = dev
        .context()
        .new_stream()
        .map_err(|e| GpuError::StreamError(format!("new_stream: {e}")))?;
    let compute_stream = dev.stream();

    let max_gnnz = spans.iter().map(|s| s.nnz as usize).max().unwrap_or(0);
    let max_idx_bytes = max_gnnz * index_width;
    let max_val_bytes = max_gnnz * value_width;

    // 2-slot rings: pinned host staging + device staging, one entry per stream
    // in flight. Upload(g+1) into the other slot overlaps compute(g).
    let mut pinned_idx = [
        HostPlaneBuf::new(dev, max_idx_bytes),
        HostPlaneBuf::new(dev, max_idx_bytes),
    ];
    let mut pinned_val = [
        HostPlaneBuf::new(dev, max_val_bytes),
        HostPlaneBuf::new(dev, max_val_bytes),
    ];
    let mut dev_idx = [
        dev.alloc_zeros::<u8>(max_idx_bytes.max(1))?,
        dev.alloc_zeros::<u8>(max_idx_bytes.max(1))?,
    ];
    let mut dev_val = [
        dev.alloc_zeros::<u8>(max_val_bytes.max(1))?,
        dev.alloc_zeros::<u8>(max_val_bytes.max(1))?,
    ];
    let mut slot_events: [Option<CudaEvent>; 2] = [None, None];

    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(spans.len().max(1));
    let counter = AtomicUsize::new(0);
    let n_groups = spans.len();

    // Bounded channel: backpressure caps decompressed groups held in RAM.
    let (tx, rx) = crossbeam_channel::bounded::<Result<GroupPlanes, GpuError>>(2 * n_workers);

    // Wrapped in a block so `consume`'s mutable borrows of the combined
    // buffers / staging rings release when the block ends — before we read the
    // results below.
    let scope_result = {
        let mut consume = |rx: crossbeam_channel::Receiver<Result<GroupPlanes, GpuError>>|
         -> Result<(), GpuError> {
        let mut slot = 0usize;
        for msg in rx.iter() {
            let (g, idx_planes, val_planes) = msg?;
            let g_nnz = spans[g].nnz as usize;
            let base = offsets[g];
            let idx_len = idx_planes.len();
            let val_len = val_planes.len();

            // Host gate: don't overwrite a pinned slot whose prior DMA may
            // still be in flight (device gates only order device streams).
            if let Some(evt) = slot_events[slot].take() {
                evt.synchronize()
                    .map_err(|e| GpuError::CudaError(format!("slot event sync: {e}")))?;
            }

            let t_htod = profile::start();
            let h_idx = pinned_idx[slot].stage(&idx_planes)?;
            copy_stream
                .memcpy_htod(h_idx, &mut dev_idx[slot].slice_mut(0..idx_len))
                .map_err(|e| GpuError::CudaError(format!("h2d indices: {e}")))?;
            let h_val = pinned_val[slot].stage(&val_planes)?;
            copy_stream
                .memcpy_htod(h_val, &mut dev_val[slot].slice_mut(0..val_len))
                .map_err(|e| GpuError::CudaError(format!("h2d values: {e}")))?;
            host_uploaded_bytes += (idx_len + val_len) as u64;

            // copy → compute: kernels must observe the uploaded bytes.
            let upload_event = copy_stream
                .record_event(None)
                .map_err(|e| GpuError::CudaError(format!("record upload event: {e}")))?;
            compute_stream
                .wait(&upload_event)
                .map_err(|e| GpuError::CudaError(format!("compute wait upload: {e}")))?;
            profile::record_htod_since(CodecClass::Generic, t_htod, idx_len + val_len);

            // Transforms on the compute stream (dev.stream()); the staging
            // buffers are oversized, so pass the actual g_nnz — the kernels only
            // touch the [0, width*g_nnz) prefix.
            let t_gpu = profile::start();
            undelta_planes_gpu(dev, &mut dev_idx[slot], g_nnz as u32, index_width as u32)?;
            let out_indices =
                unshuffle_convert_indices(dev, &dev_idx[slot], g_nnz, index_width)?;
            let out_data =
                unshuffle_convert_values(dev, &dev_val[slot], g_nnz, value_width)?;
            let mut idx_dst = combined_indices.slice_mut(base..base + g_nnz);
            compute_stream
                .memcpy_dtod(&out_indices, &mut idx_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod indices: {e}")))?;
            let mut data_dst = combined_data.slice_mut(base..base + g_nnz);
            compute_stream
                .memcpy_dtod(&out_data, &mut data_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod data: {e}")))?;
            profile::record_gpu_decode_since(t_gpu);

            // compute → copy: the next upload reusing this slot must wait for
            // the current compute reads to finish. Stash for the host gate too.
            let compute_event = compute_stream
                .record_event(None)
                .map_err(|e| GpuError::CudaError(format!("record compute event: {e}")))?;
            copy_stream
                .wait(&compute_event)
                .map_err(|e| GpuError::CudaError(format!("copy wait compute: {e}")))?;
            slot_events[slot] = Some(compute_event);

            slot ^= 1;
        }
        Ok(())
    };

        std::thread::scope(|scope| -> Result<(), GpuError> {
            // K producers do CPU zstd only (no CUDA) → thread-safe.
            for _ in 0..n_workers {
                let tx = tx.clone();
                let counter = &counter;
                scope.spawn(move || {
                    loop {
                        let g = counter.fetch_add(1, Ordering::Relaxed);
                        if g >= n_groups {
                            break;
                        }
                        let span = &spans[g];
                        let g_nnz = span.nnz as usize;
                        if g_nnz == 0 {
                            continue;
                        }
                        let idx_frame = &indices_bytes[span.indices.clone()];
                        let val_frame = &values_bytes[span.values.clone()];
                        let idx_exp = g_nnz * index_width;
                        let val_exp = g_nnz * value_width;
                        let decoded = (|| -> Result<GroupPlanes, GpuError> {
                            let ip = decompress_frame(idx_frame, idx_exp, "indices")?;
                            let vp = decompress_frame(val_frame, val_exp, "values")?;
                            Ok((g, ip, vp))
                        })();
                        if tx.send(decoded).is_err() {
                            break; // consumer dropped rx (error / done)
                        }
                    }
                });
            }
            // Drop the main thread's sender so `rx.iter()` terminates once all
            // producers finish; then consume on this (single, CUDA-owning) thread.
            drop(tx);
            consume(rx)
        })
    };
    // On the error path the slot-event drain below is skipped, so an async H2D on
    // `copy_stream` out of the pinned host ring may still be in flight when
    // `pinned_idx`/`pinned_val` drop at end of function → host-side
    // use-after-free. Best-effort sync the copy stream before propagating so any
    // outstanding DMA out of the pinned buffers has completed first.
    if scope_result.is_err() {
        let _ = copy_stream.synchronize();
    }
    scope_result?;

    // Drain outstanding slot events so the pinned buffers are safe to drop.
    for evt in slot_events.iter_mut() {
        if let Some(e) = evt.take() {
            e.synchronize()
                .map_err(|err| GpuError::CudaError(format!("slot drain sync: {err}")))?;
        }
    }

    let d_indptr = dev.htod_copy(&combined_indptr)?;
    dev.synchronize()?;
    Ok(PipelinedCsr {
        indptr: d_indptr,
        indices: combined_indices,
        data: combined_data,
        host_uploaded_bytes,
    })
}

/// `unshuffle_convert` → i32 (indices). Thin wrapper over the shared launcher.
fn unshuffle_convert_indices(
    dev: &GpuDevice,
    src: &CudaSlice<u8>,
    n: usize,
    width: usize,
) -> Result<CudaSlice<i32>, GpuError> {
    let mut out = dev.alloc_zeros::<i32>(n)?;
    launch_unshuffle_convert(dev, src, &mut out, n as u32, width as u32, 0)?;
    Ok(out)
}

/// `unshuffle_convert` → f32 (integer values, no undelta).
fn unshuffle_convert_values(
    dev: &GpuDevice,
    src: &CudaSlice<u8>,
    n: usize,
    width: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let mut out = dev.alloc_zeros::<f32>(n)?;
    launch_unshuffle_convert(dev, src, &mut out, n as u32, width as u32, 1)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase 2: nvcomp full in-VRAM decode (upload compressed, GPU zstd, then the
// same undelta/unshuffle/convert kernels). View-based launchers operate on the
// per-group sub-ranges of the single big nvcomp output buffer (no extra copies).
// ---------------------------------------------------------------------------

/// In-place per-plane undelta on a `CudaViewMut` sub-range (plane-major
/// `[width][n]`). Same kernel as [`undelta_planes_gpu`], view-typed for the
/// nvcomp concat buffer.
fn undelta_planes_view(
    dev: &GpuDevice,
    buf: &mut CudaViewMut<u8>,
    n: u32,
    width: u32,
) -> Result<(), GpuError> {
    if n == 0 || width == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(SHUFDELTA_PTX)?;
    let kernel = module
        .load_function("undelta_planes_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load undelta_planes_kernel: {e}")))?;
    let cfg = LaunchConfig {
        grid_dim: (width, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(buf)
            .arg(&n)
            .arg(&width)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("undelta_planes_kernel (view): {e}")))?;
    Ok(())
}

/// Unshuffle+convert reading a `CudaView` sub-range → newly allocated i32/f32.
fn unshuffle_convert_view<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    dev: &GpuDevice,
    src: &CudaView<u8>,
    n: usize,
    width: usize,
    out_is_float: u32,
) -> Result<CudaSlice<T>, GpuError> {
    let mut out = dev.alloc_zeros::<T>(n)?;
    let module = dev.load_module_cached(SHUFDELTA_PTX)?;
    let kernel = module
        .load_function("unshuffle_convert_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load unshuffle_convert_kernel: {e}")))?;
    let threads: u32 = 256;
    let grid = (n as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    let w_u32 = width as u32;
    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(src)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&w_u32)
            .arg(&out_is_float)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("unshuffle_convert_kernel (view): {e}")))?;
    Ok(out)
}

/// **Phase 2**: full in-VRAM decode of a framed ShufDeltaZstd shard via nvcomp.
///
/// Uploads the **compressed** per-group frames (not decompressed planes), runs
/// nvcomp batched GPU zstd over all non-empty groups (indices + values as two
/// batches), then finishes each group with the existing undelta/unshuffle/
/// convert kernels on its sub-range. Only compressed bytes cross PCIe, so the
/// caller stamps `fully_device_decoded = true`. Requires
/// [`crate::nvcomp::nvcomp_available`]; the dispatcher checks that first.
#[allow(clippy::too_many_arguments)]
pub fn decode_framed_shufdelta_gpu_nvcomp(
    dev: &GpuDevice,
    spans: &[RowGroupSpan],
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_width: usize,
) -> Result<PipelinedCsr, GpuError> {
    let value_width = value_encoding.byte_width();

    // Host-decode the tiny indptr per group + per-group nnz offsets.
    let mut offsets: Vec<usize> = Vec::with_capacity(spans.len());
    let mut combined_indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    combined_indptr.push(0);
    let mut nnz_base = 0usize;
    let t_indptr = profile::start();
    for span in spans {
        offsets.push(nnz_base);
        let g_rows = span.n_rows as usize;
        let g_indptr = scx_codec::decode_row_group_indptr_only(
            scx_codec::CodecId::ShufDeltaZstd,
            span,
            indptr_bytes,
        )?;
        debug_assert_eq!(g_indptr.len(), g_rows + 1);
        for &local in &g_indptr[1..=g_rows] {
            combined_indptr.push(nnz_base as i64 + local);
        }
        nnz_base += span.nnz as usize;
    }
    profile::record_host_decode_since(CodecClass::Generic, t_indptr);
    debug_assert_eq!(nnz_base, nnz);

    let mut combined_indices = dev.alloc_zeros::<i32>(nnz)?;
    let mut combined_data = dev.alloc_zeros::<f32>(nnz)?;
    let mut host_uploaded_bytes = (combined_indptr.len() * 8) as u64;

    if nnz == 0 {
        let d_indptr = dev.htod_copy(&combined_indptr)?;
        dev.synchronize()?;
        return Ok(PipelinedCsr {
            indptr: d_indptr,
            indices: combined_indices,
            data: combined_data,
            host_uploaded_bytes,
        });
    }

    // Gather the non-empty groups' compressed frames.
    let mut gids: Vec<usize> = Vec::new();
    let mut idx_frames: Vec<&[u8]> = Vec::new();
    let mut idx_exp: Vec<usize> = Vec::new();
    let mut val_frames: Vec<&[u8]> = Vec::new();
    let mut val_exp: Vec<usize> = Vec::new();
    for (gi, span) in spans.iter().enumerate() {
        let g_nnz = span.nnz as usize;
        if g_nnz == 0 {
            continue;
        }
        gids.push(gi);
        idx_frames.push(&indices_bytes[span.indices.clone()]);
        idx_exp.push(g_nnz * index_width);
        val_frames.push(&values_bytes[span.values.clone()]);
        val_exp.push(g_nnz * value_width);
    }
    host_uploaded_bytes += idx_frames.iter().map(|f| f.len() as u64).sum::<u64>();
    host_uploaded_bytes += val_frames.iter().map(|f| f.len() as u64).sum::<u64>();

    // GPU batched zstd: compressed frames → concatenated plane buffers.
    let t_gpu = profile::start();
    let (mut d_idx_planes, idx_off) =
        crate::nvcomp::batch_decompress_concat(dev, dev.stream(), &idx_frames, &idx_exp)?;
    let (d_val_planes, val_off) =
        crate::nvcomp::batch_decompress_concat(dev, dev.stream(), &val_frames, &val_exp)?;

    // Per group: undelta (indices) + unshuffle/convert on the group's sub-range
    // → combined buffers at the running nnz offset.
    for (k, &gi) in gids.iter().enumerate() {
        let g_nnz = spans[gi].nnz as usize;
        let base = offsets[gi];
        let io = idx_off[k];
        let ilen = g_nnz * index_width;
        {
            let mut idx_view = d_idx_planes.slice_mut(io..io + ilen);
            undelta_planes_view(dev, &mut idx_view, g_nnz as u32, index_width as u32)?;
        }
        let out_i: CudaSlice<i32> = {
            let idx_view = d_idx_planes.slice(io..io + ilen);
            unshuffle_convert_view(dev, &idx_view, g_nnz, index_width, 0)?
        };
        let mut idx_dst = combined_indices.slice_mut(base..base + g_nnz);
        dev.stream()
            .memcpy_dtod(&out_i, &mut idx_dst)
            .map_err(|e| GpuError::CudaError(format!("dtod indices (nvcomp): {e}")))?;

        let vo = val_off[k];
        let vlen = g_nnz * value_width;
        let out_v: CudaSlice<f32> = {
            let val_view = d_val_planes.slice(vo..vo + vlen);
            unshuffle_convert_view(dev, &val_view, g_nnz, value_width, 1)?
        };
        let mut data_dst = combined_data.slice_mut(base..base + g_nnz);
        dev.stream()
            .memcpy_dtod(&out_v, &mut data_dst)
            .map_err(|e| GpuError::CudaError(format!("dtod data (nvcomp): {e}")))?;
    }
    profile::record_gpu_decode_since(t_gpu);

    let d_indptr = dev.htod_copy(&combined_indptr)?;
    dev.synchronize()?;

    Ok(PipelinedCsr {
        indptr: d_indptr,
        indices: combined_indices,
        data: combined_data,
        host_uploaded_bytes,
    })
}

// ---------------------------------------------------------------------------
// Phase 2.x — nvcomp cross-shard batching.
//
// The per-shard `decode_framed_shufdelta_gpu_nvcomp` pays two
// `batch_decompress_concat` calls (idx + val) *per shard*, each with its own
// temp alloc + host pointer arrays + stream sync + status readback — ~124 nvcomp
// calls on census_1m's 62 shards. Collapsing that into **2** batched calls total
// (all shards' idx frames in one, all val frames in the other) removes ~122
// syncs; more chunks per batch also fills more SMs (the 256-vs-4-frame spike
// finding). Output stays byte-identical: the global nnz offset of each group is
// the prefix sum over all groups in shard-then-group order — the same
// concatenation the per-shard assembly loop produces.
// ---------------------------------------------------------------------------

/// One non-empty row-group in the flattened cross-shard group table. Frame
/// slices borrow the caller's mmap-resident shard bytes.
struct GlobalGroup<'a> {
    /// Compressed indices sub-stream frame for this group.
    idx_frame: &'a [u8],
    /// Exact decompressed indices-plane byte length (`g_nnz * index_width`).
    idx_expected: usize,
    /// Compressed values sub-stream frame for this group.
    val_frame: &'a [u8],
    /// Exact decompressed values-plane byte length (`g_nnz * value_width`).
    val_expected: usize,
    /// Global nnz offset (prefix sum over all prior groups of all prior shards).
    global_nnz_base: usize,
    /// nnz in this group (> 0).
    g_nnz: usize,
    /// Byte width of a column index (2 for u16, 4 for u32).
    index_width: usize,
    /// Byte width of a value (from the shard's `ValueEncoding`).
    value_width: usize,
}

/// Flattened decode plan for a run of framed ShufDeltaZstd shards.
struct ShufdeltaBatchPlan<'a> {
    groups: Vec<GlobalGroup<'a>>,
    /// Global CSR indptr (`total_rows + 1`), rebased across all shards.
    combined_indptr: Vec<i64>,
    total_rows: usize,
    total_nnz: usize,
    n_cols: usize,
}

/// Bounds-checked sub-slice of a shard's byte buffer (mirrors
/// `shard_decode::extract_slice`, module-local so the returned slice keeps the
/// shard's `'a` lifetime for the frame table).
fn slice_of<'a>(
    bytes: &'a [u8],
    rel_offset: u32,
    length: u32,
    name: &str,
    si: usize,
) -> Result<&'a [u8], GpuError> {
    let start = rel_offset as usize;
    let end = start + length as usize;
    if end > bytes.len() {
        return Err(GpuError::InvalidShard(format!(
            "shard {si} {name} slice [{start}..{end}] exceeds shard size {}",
            bytes.len()
        )));
    }
    Ok(&bytes[start..end])
}

/// Cross-shard pre-scan (2x-a): flatten every shard's row groups into one global
/// table + assemble the rebased global indptr. Cheap (header parse + tiny
/// per-group indptr host-decode, no plane decode). Caller guarantees every shard
/// is framed ShufDeltaZstd with an integer value encoding.
fn prescan_shufdelta_shards<'a>(shards: &[&'a [u8]]) -> Result<ShufdeltaBatchPlan<'a>, GpuError> {
    let mut groups: Vec<GlobalGroup<'a>> = Vec::new();
    let mut combined_indptr: Vec<i64> = vec![0];
    let mut total_rows = 0usize;
    let mut total_nnz = 0usize;
    let mut n_cols = 0usize;

    for (si, &shard_bytes) in shards.iter().enumerate() {
        let header = ShardHeader::read_from(&mut Cursor::new(shard_bytes))
            .map_err(|e| GpuError::InvalidShard(format!("shard {si} header: {e}")))?;
        let value_encoding = ValueEncoding::from_u8(header.value_encoding).ok_or_else(|| {
            GpuError::InvalidShard(format!(
                "shard {si} unknown value_encoding: {}",
                header.value_encoding
            ))
        })?;
        let index_width = if header.index_dtype == 0 { 2usize } else { 4 };
        let value_width = value_encoding.byte_width();
        let cols = header.n_minor as usize;
        if si == 0 {
            n_cols = cols;
        } else if cols != n_cols {
            return Err(GpuError::InvalidShard(format!(
                "shard {si} column count {cols} != {n_cols} (shards must agree)"
            )));
        }

        let indptr_bytes = slice_of(
            shard_bytes,
            header.indptr_rel_offset,
            header.indptr_length,
            "indptr",
            si,
        )?;
        let indices_bytes = slice_of(
            shard_bytes,
            header.indices_rel_offset,
            header.indices_length,
            "indices",
            si,
        )?;
        let values_bytes = slice_of(
            shard_bytes,
            header.values_rel_offset,
            header.values_length,
            "values",
            si,
        )?;
        let block_index_bytes = slice_of(
            shard_bytes,
            header.block_index_rel_offset,
            header.block_index_length,
            "block_index",
            si,
        )?;

        let spans = resolve_block_index(&header, block_index_bytes)
            .map_err(|e| GpuError::InvalidShard(format!("shard {si} block index: {e}")))?;

        for span in &spans {
            let g_rows = span.n_rows as usize;
            let g_nnz = span.nnz as usize;
            let g_indptr = scx_codec::decode_row_group_indptr_only(
                scx_codec::CodecId::ShufDeltaZstd,
                span,
                indptr_bytes,
            )?;
            debug_assert_eq!(g_indptr.len(), g_rows + 1);
            for &local in &g_indptr[1..=g_rows] {
                combined_indptr.push(total_nnz as i64 + local);
            }
            if g_nnz > 0 {
                groups.push(GlobalGroup {
                    idx_frame: &indices_bytes[span.indices.clone()],
                    idx_expected: g_nnz * index_width,
                    val_frame: &values_bytes[span.values.clone()],
                    val_expected: g_nnz * value_width,
                    global_nnz_base: total_nnz,
                    g_nnz,
                    index_width,
                    value_width,
                });
            }
            total_nnz += g_nnz;
            total_rows += g_rows;
        }
    }
    debug_assert_eq!(combined_indptr.len(), total_rows + 1);

    Ok(ShufdeltaBatchPlan {
        groups,
        combined_indptr,
        total_rows,
        total_nnz,
        n_cols,
    })
}

/// **Phase 2.x**: full in-VRAM decode of a *run of* framed ShufDeltaZstd shards
/// via nvcomp, batching every group's compressed frame into **2** batched
/// `nvcompBatchedZstdDecompressAsync` calls total (all indices, then all values)
/// — instead of 2 per shard. Produces a single row-stacked device CSR
/// byte-identical to decoding each shard with the per-shard nvcomp path and
/// concatenating. Only compressed bytes cross PCIe, so the returned stats report
/// `fully_device_decoded = true`. The dispatcher guarantees every shard is framed
/// ShufDeltaZstd with an integer value encoding (2x-d) and that nvcomp is loadable.
///
/// **VRAM ceiling (caller must pre-flight):** unlike the pipeline's bounded
/// 2-slot ring, the batch holds the whole modality resident at peak — the
/// compressed blob (Σ all frames), both decompressed plane buffers
/// (`total_nnz·(index_width+value_width)`), and the final CSR
/// (`total_nnz·8 + rows·8`), plus nvcomp temp (≈ 19 GB on census_1m). There is
/// **no internal chunking**, so a caller must gate on free VRAM before calling
/// (pyscx's `to_gpu_anndata` does; see `experiment.rs`). Shard-chunking to bound
/// the transient on smaller cards is a documented follow-on.
pub fn decode_shufdelta_shards_nvcomp_batched(
    dev: &GpuDevice,
    shards: &[&[u8]],
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    let t_host = profile::start();
    let plan = prescan_shufdelta_shards(shards)?;
    profile::record_host_decode_since(CodecClass::Generic, t_host);
    let total_nnz = plan.total_nnz;
    let total_rows = plan.total_rows;
    let n_cols = plan.n_cols;

    let mut combined_indices = dev.alloc_zeros::<i32>(total_nnz)?;
    let mut combined_data = dev.alloc_zeros::<f32>(total_nnz)?;
    let mut host_uploaded_bytes = (plan.combined_indptr.len() * 8) as u64;

    if total_nnz == 0 {
        let d_indptr = dev.htod_copy(&plan.combined_indptr)?;
        dev.synchronize()?;
        return Ok((
            GpuCsr {
                indptr: d_indptr,
                indices: combined_indices,
                data: combined_data,
                shape: (total_rows, n_cols),
            },
            DeviceDecodeStats {
                host_uploaded_bytes,
                device_decoded_bytes: 0,
                fully_device_decoded: true,
                n_shards_shufdelta_gpu: shards.len() as u32,
                ..DeviceDecodeStats::default()
            },
        ));
    }

    // Two batched decompress calls over ALL groups of ALL shards (idx, then val).
    let idx_frames: Vec<&[u8]> = plan.groups.iter().map(|g| g.idx_frame).collect();
    let idx_exp: Vec<usize> = plan.groups.iter().map(|g| g.idx_expected).collect();
    let val_frames: Vec<&[u8]> = plan.groups.iter().map(|g| g.val_frame).collect();
    let val_exp: Vec<usize> = plan.groups.iter().map(|g| g.val_expected).collect();
    host_uploaded_bytes += idx_frames.iter().map(|f| f.len() as u64).sum::<u64>();
    host_uploaded_bytes += val_frames.iter().map(|f| f.len() as u64).sum::<u64>();

    let t_gpu = profile::start();
    let (mut d_idx_planes, idx_off) =
        crate::nvcomp::batch_decompress_concat(dev, dev.stream(), &idx_frames, &idx_exp)?;
    let (d_val_planes, val_off) =
        crate::nvcomp::batch_decompress_concat(dev, dev.stream(), &val_frames, &val_exp)?;

    // Per group: undelta (indices) + unshuffle/convert on the group's plane
    // sub-range → combined buffers at the group's global nnz offset.
    for (k, g) in plan.groups.iter().enumerate() {
        let base = g.global_nnz_base;
        let g_nnz = g.g_nnz;
        let io = idx_off[k];
        let ilen = g_nnz * g.index_width;
        {
            let mut idx_view = d_idx_planes.slice_mut(io..io + ilen);
            undelta_planes_view(dev, &mut idx_view, g_nnz as u32, g.index_width as u32)?;
        }
        let out_i: CudaSlice<i32> = {
            let idx_view = d_idx_planes.slice(io..io + ilen);
            unshuffle_convert_view(dev, &idx_view, g_nnz, g.index_width, 0)?
        };
        let mut idx_dst = combined_indices.slice_mut(base..base + g_nnz);
        dev.stream()
            .memcpy_dtod(&out_i, &mut idx_dst)
            .map_err(|e| GpuError::CudaError(format!("dtod indices (nvcomp batched): {e}")))?;

        let vo = val_off[k];
        let vlen = g_nnz * g.value_width;
        let out_v: CudaSlice<f32> = {
            let val_view = d_val_planes.slice(vo..vo + vlen);
            unshuffle_convert_view(dev, &val_view, g_nnz, g.value_width, 1)?
        };
        let mut data_dst = combined_data.slice_mut(base..base + g_nnz);
        dev.stream()
            .memcpy_dtod(&out_v, &mut data_dst)
            .map_err(|e| GpuError::CudaError(format!("dtod data (nvcomp batched): {e}")))?;
    }
    profile::record_gpu_decode_since(t_gpu);

    let d_indptr = dev.htod_copy(&plan.combined_indptr)?;
    dev.synchronize()?;

    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: combined_indices,
            data: combined_data,
            shape: (total_rows, n_cols),
        },
        DeviceDecodeStats {
            host_uploaded_bytes,
            device_decoded_bytes: (total_nnz as u64) * 8,
            fully_device_decoded: true,
            n_shards_shufdelta_gpu: shards.len() as u32,
            ..DeviceDecodeStats::default()
        },
    ))
}
