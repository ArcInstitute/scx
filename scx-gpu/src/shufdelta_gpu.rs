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

use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::safe::{CudaEvent, CudaSlice, LaunchConfig, PinnedHostSlice};
use cudarc::driver::PushKernelArg;

use scx_codec::{zstd_decompress_bounded, RowGroupSpan, ValueEncoding};

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::profile::{self, CodecClass};

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
// Profiling (GPU-SHUFDELTA-DECODE Phase 0/1) showed the sequential path's floor
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
