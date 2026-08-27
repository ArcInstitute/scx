//! GPU decode transforms for ShufDeltaZstd (codec_id = 5) shards.
//!
//! The CPU host runs zstd (inherently sequential within a frame); these
//! launchers take over the two byte-plane transforms that Phase-0 profiling
//! showed dominate the CPU decode: the per-plane wrapping-u8 delta prefix scan
//! (`undelta_planes`) and the plane-major → element-major transpose fused with
//! the widen-to-i32/f32 convert (`unshuffle_convert_indices` /
//! `unshuffle_convert_values`).
//!
//! Those three are private, so the names are code spans rather than intra-doc
//! links: this is a `pub mod`, and rustdoc warns on a public doc linking to a
//! private item. The names still have to be right — they were `undelta_planes_gpu`
//! and `unshuffle_convert_*_gpu`, one of which this module deleted and the other
//! of which never existed.
//!
//! The two frame helpers ([`decode_indices_frame_to_device`] /
//! [`decode_values_frame_to_device`]) tie them together: zstd-decompress one
//! sub-stream frame on the host to its intermediate "still shuffled+delta'd"
//! plane bytes (via `scx_codec::zstd_decompress_bounded`), upload those, and
//! run the kernels — so only the narrower pre-convert plane bytes cross PCIe.

use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::safe::{
    CudaEvent, CudaFunction, CudaSlice, CudaView, CudaViewMut, LaunchConfig, PinnedHostSlice,
};
use cudarc::driver::PushKernelArg;

use scx_codec::{zstd_decompress_bounded, RowGroupSpan, ValueEncoding};
use scx_format_io::shard::{resolve_block_index, ShardHeader};

use crate::combined_csr::CombinedCsr;
use crate::csr_placement::Placement;
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::profile::{self, CodecClass};
use crate::shard_decode::{check_device_len, DeviceDecodeStats, GpuCsr};

/// Compiled PTX for the shufdelta kernels (produced by build.rs via nvcc --ptx).
const SHUFDELTA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/shufdelta.ptx"));

/// Host-decode each framed row-group's local indptr and fold its `[1..=g_rows]`
/// tail into `combined_indptr`, rebased by a running nnz base that starts at
/// `nnz_base`. Returns `(per-group nnz base offsets in span order, final nnz
/// base)`.
///
/// `combined_indptr` must already hold its leading `0` (or a prior shard's tail
/// — the cross-shard batched path calls this once per shard, passing the
/// running `total_nnz` as `nnz_base`). `offsets[i]` is the global nnz offset of
/// group `i`: exactly where its decoded indices/values are placed in the
/// combined CSR, i.e. the same running base the per-group assembly loops need.
///
/// Pure host work (frame-header parse + tiny per-group indptr decode; the large
/// index/value plane frames are never touched here), so callers keep their own
/// `profile::record_host_decode_since` timing wrapper. Parameterized by `codec`
/// so both the Scx1 (FOR-BP/Rice) and ShufDeltaZstd framed GPU decode paths
/// share it.
pub(crate) fn prescan_framed_group_indptr(
    codec: scx_codec::CodecId,
    spans: &[RowGroupSpan],
    indptr_bytes: &[u8],
    nnz_base: usize,
    combined_indptr: &mut Vec<i64>,
) -> Result<(Vec<usize>, usize), GpuError> {
    let mut offsets: Vec<usize> = Vec::with_capacity(spans.len());
    let mut base = nnz_base;
    for span in spans {
        offsets.push(base);
        let g_rows = span.n_rows as usize;
        let g_indptr = scx_codec::decode_row_group_indptr_only(codec, span, indptr_bytes)?;
        // decode_row_group_indptr_only should return exactly g_rows+1 entries;
        // validate explicitly (not just debug_assert) so a malformed frame that
        // decoded a short indptr errors here instead of panicking the
        // `[1..=g_rows]` slice in a release build.
        if g_indptr.len() != g_rows + 1 {
            return Err(GpuError::InvalidShard(format!(
                "framed indptr: decoded {} entries, expected {}",
                g_indptr.len(),
                g_rows + 1
            )));
        }
        for &local in &g_indptr[1..=g_rows] {
            combined_indptr.push(base as i64 + local);
        }
        base += span.nnz as usize;
    }
    Ok((offsets, base))
}

// ---------------------------------------------------------------------------
// Kernel-launch primitives. Two kernels, two launch bodies.
//
// cudarc's `PushKernelArg` really does have separate concrete impls for
// `&CudaSlice`, `&mut CudaSlice`, `&CudaView` and `&mut CudaViewMut` with no
// blanket `DevicePtr` impl, so a single `.arg(buf)` line cannot be generic over
// all four. This module used to conclude from that that the launchers were
// irreducibly per-type, and grew six functions for the two kernels — one pair
// for callers holding a `CudaSlice` (the framed/pipelined paths) and one for
// callers holding a view into the nvcomp concat buffer.
//
// The conclusion does not follow. `CudaSlice::as_view()` / `as_view_mut()` are
// pure pointer copies — a struct literal, no CUDA call — and the view
// `PushKernelArg` impls are line-for-line identical to the slice ones, event
// bookkeeping included. So the view form is canonical here and a slice-holding
// caller spells `&buf.as_view()`; the kernel name, block size and grid formula
// are each written once, as before, and now so is each launch.
// ---------------------------------------------------------------------------

/// Resolve `undelta_planes_kernel` from the cached shufdelta PTX module.
fn undelta_kernel(dev: &GpuDevice) -> Result<CudaFunction, GpuError> {
    dev.load_module_cached(SHUFDELTA_PTX)?
        .load_function("undelta_planes_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load undelta_planes_kernel: {e}")))
}

/// Resolve `unshuffle_convert_kernel` from the cached shufdelta PTX module.
fn unshuffle_kernel(dev: &GpuDevice) -> Result<CudaFunction, GpuError> {
    dev.load_module_cached(SHUFDELTA_PTX)?
        .load_function("unshuffle_convert_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load unshuffle_convert_kernel: {e}")))
}

/// Launch config for `undelta_planes_kernel`: one 256-thread block per plane
/// (`grid_dim.x == width`).
fn undelta_cfg(width: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (width, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch config for `unshuffle_convert_kernel`: 256-thread blocks over the
/// `n` output elements.
fn unshuffle_cfg(n: u32) -> LaunchConfig {
    let threads: u32 = 256;
    LaunchConfig {
        grid_dim: (n.div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// In-place per-plane wrapping-u8 inclusive prefix scan (undo byte-delta).
///
/// `buf` is plane-major `[width][n]`; each of the `width` planes is scanned
/// independently. Launches one thread block per plane (`grid_dim.x == width`).
///
/// Takes a `CudaViewMut` so the nvcomp paths can pass a sub-range of the concat
/// buffer directly; a caller holding a whole `CudaSlice` passes
/// `&mut buf.as_view_mut()`, which costs a pointer copy.
fn undelta_planes(
    dev: &GpuDevice,
    buf: &mut CudaViewMut<u8>,
    n: usize,
    width: usize,
) -> Result<(), GpuError> {
    if n == 0 || width == 0 {
        return Ok(());
    }
    // Checked for the same reason as in `launch_unshuffle_convert`: the kernel
    // takes both as `u32` and a truncating cast would scan only the wrapped
    // prefix of each plane, leaving the tail still delta-encoded.
    let n = u32::try_from(n).map_err(|_| {
        GpuError::InvalidShard(format!(
            "undelta_planes: {n} elements exceeds the kernel's u32 element count"
        ))
    })?;
    let width = u32::try_from(width).map_err(|_| {
        GpuError::InvalidShard(format!(
            "undelta_planes: plane width {width} exceeds the kernel's u32 width"
        ))
    })?;
    let kernel = undelta_kernel(dev)?;
    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(buf)
            .arg(&n)
            .arg(&width)
            .launch(undelta_cfg(width))
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("undelta_planes_kernel: {e}")))?;
    Ok(())
}

/// Unshuffle (plane-major → element-major) fused with widen-to-i32/f32, into a
/// freshly allocated `n`-element output.
///
/// The kernel treats `out` as an opaque device pointer, so this is generic over
/// the output element type — but `T` and `out_is_float` **must** agree or the
/// kernel writes one type's bits through the other's pointer. Nothing in the
/// signature enforces that, which is why the only two callers are the typed
/// wrappers below: they are what binds `i32` to `0` and `f32` to `1`, and they
/// are the only two places the flag is spelled.
fn launch_unshuffle_convert<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    dev: &GpuDevice,
    src: &CudaView<u8>,
    n: usize,
    width: usize,
    out_is_float: u32,
) -> Result<CudaSlice<T>, GpuError> {
    let mut out = dev.alloc_zeros::<T>(n)?;
    let kernel = unshuffle_kernel(dev)?;
    // Checked, not `as`: the kernel takes `n` and `width` as `u32`, and this
    // function allocates `n` output elements before launching. A truncating
    // cast would leave the kernel processing only the wrapped prefix — or
    // receiving a zero-sized grid — while the returned slice still has length
    // `n`, i.e. a silently short decode. The row-group callers derive `n` from a
    // `u32` nnz and the widths are small codec constants, but the two public
    // whole-frame helpers take `usize`, so the cast is not universally safe and
    // is better failed loudly than proved by inspection of today's callers.
    let n_u32 = u32::try_from(n).map_err(|_| {
        GpuError::InvalidShard(format!(
            "unshuffle_convert: {n} elements exceeds the kernel's u32 element count"
        ))
    })?;
    let w_u32 = u32::try_from(width).map_err(|_| {
        GpuError::InvalidShard(format!(
            "unshuffle_convert: element width {width} exceeds the kernel's u32 width"
        ))
    })?;
    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(src)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&w_u32)
            .arg(&out_is_float)
            .launch(unshuffle_cfg(n_u32))
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("unshuffle_convert_kernel: {e}")))?;
    Ok(out)
}

/// `unshuffle_convert` → `i32` (column indices).
fn unshuffle_convert_indices(
    dev: &GpuDevice,
    src: &CudaView<u8>,
    n: usize,
    width: usize,
) -> Result<CudaSlice<i32>, GpuError> {
    launch_unshuffle_convert(dev, src, n, width, 0)
}

/// `unshuffle_convert` → `f32` (integer values, no undelta — integer values are
/// shuffle-only on encode).
fn unshuffle_convert_values(
    dev: &GpuDevice,
    src: &CudaView<u8>,
    n: usize,
    width: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    launch_unshuffle_convert(dev, src, n, width, 1)
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
    undelta_planes(dev, &mut d_planes.as_view_mut(), nnz, index_width)?;
    let out = unshuffle_convert_indices(dev, &d_planes.as_view(), nnz, index_width)?;
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
    let out = unshuffle_convert_values(dev, &d_planes.as_view(), nnz, width)?;
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
    /// `cap` is `max(span.nnz) × element_width` — derived from the **untrusted**
    /// block index, so the pageable fallback must be fallible. `vec![0u8; cap]`
    /// aborts through `handle_alloc_error` on a hostile `nnz`, the same remote
    /// kill switch [`clamped_reserve`] closes on the reassembly buffers, and it
    /// is reached *before* the fallible `alloc_zeros` staging allocations below.
    /// Clamping is not an option here — [`Self::stage`] indexes `v[..src.len()]`,
    /// so the buffer must actually hold `cap` — hence `try_reserve_exact`, which
    /// returns instead of aborting.
    fn new(dev: &GpuDevice, cap: usize) -> Result<Self, GpuError> {
        let cap = cap.max(1);
        // SAFETY: `alloc_pinned` is unsafe only in that the buffer is
        // uninitialized; we fully overwrite the used prefix before every upload.
        match unsafe { dev.context().alloc_pinned::<u8>(cap) } {
            Ok(p) => Ok(HostPlaneBuf::Pinned(p)),
            Err(_) => {
                let mut v: Vec<u8> = Vec::new();
                v.try_reserve_exact(cap).map_err(|e| {
                    GpuError::OutOfMemory(format!(
                        "pageable host staging buffer of {cap} bytes (pinned alloc failed): {e}"
                    ))
                })?;
                v.resize(cap, 0);
                Ok(HostPlaneBuf::Pageable(v))
            }
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

/// A finished device CSR from one of the framed ShufDeltaZstd decode paths,
/// plus the host→device byte total its caller stamps into
/// [`DeviceDecodeStats`].
///
/// Carries a whole [`GpuCsr`] rather than three loose buffers: they only become
/// a CSR once `CombinedCsr::finish` has established that the placed units cover
/// the matrix and that the three lengths agree, and handing back the parts
/// invited each caller to re-assert that for itself.
pub struct PipelinedCsr {
    pub csr: GpuCsr,
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
    n_cols: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_width: usize,
) -> Result<PipelinedCsr, GpuError> {
    let value_width = value_encoding.byte_width();

    // Precompute per-group nnz offsets + the full global indptr on the host
    // (tiny; the large index/value frames go to the device). Offsets let the
    // GPU consumer place each group independently, so producers can run ahead
    // and out of order.
    let mut combined = CombinedCsr::new(dev, nnz, n_rows, indptr_bytes.len())?;
    let t_indptr = profile::start();
    let (offsets, nnz_final) = prescan_framed_group_indptr(
        scx_codec::CodecId::ShufDeltaZstd,
        spans,
        indptr_bytes,
        0,
        &mut combined.indptr,
    )?;
    profile::record_host_decode_since(CodecClass::Generic, t_indptr);
    check_device_len(nnz_final, nnz, "framed ShufDeltaZstd block-index nnz")?;

    let mut host_uploaded_bytes = (combined.indptr.len() * 8) as u64;

    if nnz == 0 {
        return Ok(PipelinedCsr {
            csr: combined.finish(dev, n_cols, "framed ShufDeltaZstd shard (pipelined)")?,
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
        HostPlaneBuf::new(dev, max_idx_bytes)?,
        HostPlaneBuf::new(dev, max_idx_bytes)?,
    ];
    let mut pinned_val = [
        HostPlaneBuf::new(dev, max_val_bytes)?,
        HostPlaneBuf::new(dev, max_val_bytes)?,
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
            undelta_planes(
                dev,
                &mut dev_idx[slot].as_view_mut(),
                g_nnz,
                index_width,
            )?;
            let out_indices =
                unshuffle_convert_indices(dev, &dev_idx[slot].as_view(), g_nnz, index_width)?;
            let out_data =
                unshuffle_convert_values(dev, &dev_val[slot].as_view(), g_nnz, value_width)?;
            // `place` copies on `dev.stream()`, which *is* `compute_stream`:
            // `GpuDevice::stream` hands back the same `Arc<CudaStream>` bound
            // above, so the ordering against the upload event is unchanged.
            combined.place(
                dev,
                Placement {
                    base,
                    len: g_nnz,
                    op: "pipelined ShufDeltaZstd group",
                    index: g,
                },
                &out_indices,
                &out_data,
            )?;
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

    Ok(PipelinedCsr {
        csr: combined.finish(dev, n_cols, "framed ShufDeltaZstd shard (pipelined)")?,
        host_uploaded_bytes,
    })
}

// ---------------------------------------------------------------------------
// Phase 2: nvcomp full in-VRAM decode (upload compressed, GPU zstd, then the
// same undelta/unshuffle/convert kernels). View-based launchers operate on the
// per-group sub-ranges of the single big nvcomp output buffer (no extra copies).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-group assembly from batched nvcomp plane buffers into the combined CSR.
// Shared by the per-shard (`decode_framed_shufdelta_gpu_nvcomp`, one
// interleaved loop) and cross-shard (`decode_shufdelta_shards_nvcomp_batched`,
// two passes) decode bodies. The two halves are exposed separately because the
// cross-shard path frees the indices plane buffer before allocating the values
// one, so it cannot hold a `&mut d_idx_planes` and `&d_val_planes` borrow at
// once. `g_nnz == 0` groups are never passed (both callers skip them). All ops
// run on `dev.stream()`.
// ---------------------------------------------------------------------------

/// Undelta (in place) + unshuffle/convert one group's **indices** plane
/// sub-range `[idx_off .. idx_off + g_nnz*index_width)`, then `memcpy_dtod` the
/// widened `i32` into `combined[base .. base + g_nnz]` via
/// [`CombinedCsr::place_indices`].
fn assemble_group_indices_view(
    dev: &GpuDevice,
    d_idx_planes: &mut CudaSlice<u8>,
    idx_off: usize,
    at: Placement<'_>,
    index_width: usize,
    combined: &mut CombinedCsr,
) -> Result<(), GpuError> {
    let g_nnz = at.len;
    let ilen = g_nnz * index_width;
    {
        let mut idx_view = d_idx_planes.slice_mut(idx_off..idx_off + ilen);
        undelta_planes(dev, &mut idx_view, g_nnz, index_width)?;
    }
    let out_i = {
        let idx_view = d_idx_planes.slice(idx_off..idx_off + ilen);
        unshuffle_convert_indices(dev, &idx_view, g_nnz, index_width)?
    };
    combined.place_indices(dev, at, &out_i)
}

/// unshuffle/convert one group's **values** plane sub-range (no undelta —
/// values are shuffle-only) → `memcpy_dtod` the widened `f32` into
/// `combined[base .. base + g_nnz]` via [`CombinedCsr::place_values`]. Sibling of
/// [`assemble_group_indices_view`].
fn assemble_group_values_view(
    dev: &GpuDevice,
    d_val_planes: &CudaSlice<u8>,
    val_off: usize,
    at: Placement<'_>,
    value_width: usize,
    combined: &mut CombinedCsr,
) -> Result<(), GpuError> {
    let g_nnz = at.len;
    let vlen = g_nnz * value_width;
    let out_v = {
        let val_view = d_val_planes.slice(val_off..val_off + vlen);
        unshuffle_convert_values(dev, &val_view, g_nnz, value_width)?
    };
    combined.place_values(dev, at, &out_v)
}

/// Assemble one group's indices then values into the combined CSR at nnz offset
/// `base`. Convenience wrapper for the single-shard nvcomp path (one
/// interleaved loop); the cross-shard batched path calls the two halves
/// directly across its separate indices/values passes.
#[allow(clippy::too_many_arguments)]
fn assemble_group_view(
    dev: &GpuDevice,
    d_idx_planes: &mut CudaSlice<u8>,
    d_val_planes: &CudaSlice<u8>,
    idx_off: usize,
    val_off: usize,
    at: Placement<'_>,
    index_width: usize,
    value_width: usize,
    combined: &mut CombinedCsr,
) -> Result<(), GpuError> {
    assemble_group_indices_view(dev, d_idx_planes, idx_off, at, index_width, combined)?;
    assemble_group_values_view(dev, d_val_planes, val_off, at, value_width, combined)
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
    n_cols: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_width: usize,
) -> Result<PipelinedCsr, GpuError> {
    let value_width = value_encoding.byte_width();

    // Host-decode the tiny indptr per group + per-group nnz offsets.
    let mut combined = CombinedCsr::new(dev, nnz, n_rows, indptr_bytes.len())?;
    let t_indptr = profile::start();
    let (offsets, nnz_final) = prescan_framed_group_indptr(
        scx_codec::CodecId::ShufDeltaZstd,
        spans,
        indptr_bytes,
        0,
        &mut combined.indptr,
    )?;
    profile::record_host_decode_since(CodecClass::Generic, t_indptr);
    check_device_len(nnz_final, nnz, "nvcomp ShufDeltaZstd block-index nnz")?;

    let mut host_uploaded_bytes = (combined.indptr.len() * 8) as u64;

    if nnz == 0 {
        return Ok(PipelinedCsr {
            csr: combined.finish(dev, n_cols, "framed ShufDeltaZstd shard (nvcomp)")?,
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
        assemble_group_view(
            dev,
            &mut d_idx_planes,
            &d_val_planes,
            idx_off[k],
            val_off[k],
            Placement {
                base: offsets[gi],
                len: spans[gi].nnz as usize,
                op: "nvcomp ShufDeltaZstd group",
                index: gi,
            },
            index_width,
            value_width,
            &mut combined,
        )?;
    }
    profile::record_gpu_decode_since(t_gpu);

    Ok(PipelinedCsr {
        csr: combined.finish(dev, n_cols, "framed ShufDeltaZstd shard (nvcomp)")?,
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

        // Rebase this shard's group indptrs onto the running global nnz base;
        // `offsets[gi]` is each group's global nnz offset (== the old inline
        // `total_nnz` captured before that group), used as its `global_nnz_base`.
        let (offsets, new_total_nnz) = prescan_framed_group_indptr(
            scx_codec::CodecId::ShufDeltaZstd,
            &spans,
            indptr_bytes,
            total_nnz,
            &mut combined_indptr,
        )?;
        for (gi, span) in spans.iter().enumerate() {
            let g_nnz = span.nnz as usize;
            if g_nnz > 0 {
                groups.push(GlobalGroup {
                    idx_frame: &indices_bytes[span.indices.clone()],
                    idx_expected: g_nnz * index_width,
                    val_frame: &values_bytes[span.values.clone()],
                    val_expected: g_nnz * value_width,
                    global_nnz_base: offsets[gi],
                    g_nnz,
                    index_width,
                    value_width,
                });
            }
            total_rows += span.n_rows as usize;
        }
        total_nnz = new_total_nnz;
    }
    check_device_len(
        combined_indptr.len(),
        total_rows + 1,
        "nvcomp batch plan indptr",
    )?;

    Ok(ShufdeltaBatchPlan {
        groups,
        combined_indptr,
        total_rows,
        total_nnz,
        n_cols,
    })
}

/// Split `group_plane_bytes` (per-group decompressed idx+val plane bytes) into
/// contiguous chunks whose summed bytes fit `budget`, so the nvcomp batched
/// decode's transient host/device footprint stays bounded. Greedy: start a
/// new chunk before a group that would overflow the current one. A single group
/// larger than `budget` forms its own chunk (progress guaranteed). The returned
/// ranges exactly partition `0..group_plane_bytes.len()`.
fn plan_nvcomp_chunks(group_plane_bytes: &[usize], budget: usize) -> Vec<std::ops::Range<usize>> {
    let budget = budget.max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, &b) in group_plane_bytes.iter().enumerate() {
        if i > start && acc.saturating_add(b) > budget {
            chunks.push(start..i);
            start = i;
            acc = 0;
        }
        acc = acc.saturating_add(b);
    }
    if start < group_plane_bytes.len() {
        chunks.push(start..group_plane_bytes.len());
    }
    chunks
}

/// Per-chunk transient-byte budget for [`decode_shufdelta_shards_nvcomp_batched`]
/// . `SCX_SHUFDELTA_NVCOMP_CHUNK_BYTES` overrides it (tests force small
/// chunks); otherwise ~15% of free VRAM — conservative headroom for the nvcomp
/// temp buffer (which scales with the chunk's output) on top of the resident
/// combined CSR. If free VRAM can't be queried, returns `usize::MAX` ⇒ a single
/// chunk ⇒ the pre-M3 single-batch behavior.
fn nvcomp_batch_chunk_budget(dev: &GpuDevice) -> usize {
    if let Ok(v) = std::env::var("SCX_SHUFDELTA_NVCOMP_CHUNK_BYTES") {
        if let Ok(bytes) = v.parse::<usize>() {
            return bytes.max(1);
        }
    }
    match dev.free_memory() {
        Ok((free, _total)) => ((free as f64 * 0.15) as usize).max(1),
        Err(_) => usize::MAX,
    }
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
/// **Memory:** the final combined CSR (`total_nnz·8 + rows·8`) is the resident
/// result and the caller must pre-flight free VRAM for it (pyscx's
/// `to_gpu_anndata` does; see `experiment.rs`). The *transient* decode footprint
/// — the host compressed `blob`, the decompressed plane buffer, and the nvcomp
/// temp — is **bounded per chunk** by [`nvcomp_batch_chunk_budget`]
/// ([`plan_nvcomp_chunks`] splits the groups into runs whose decompressed plane
/// bytes fit the budget). Indices and values are decoded in separate passes per
/// chunk, so at most one plane buffer (not both) is resident at a time. Set
/// `SCX_SHUFDELTA_NVCOMP_CHUNK_BYTES` to force a specific chunk size (tests);
/// otherwise the budget is a fraction of free VRAM measured after the combined
/// buffers are allocated. Output is byte-identical to a single-batch decode.
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

    // The pre-scan already assembled the global indptr — it had to, to compute
    // the per-group global offsets — so hand it over rather than reserving a
    // second one, and let the builder own the placement, the coverage tally and
    // the upload from here.
    let mut combined = CombinedCsr::with_indptr(dev, total_nnz, total_rows, plan.combined_indptr)?;
    let mut host_uploaded_bytes = (combined.indptr.len() * 8) as u64;

    if total_nnz == 0 {
        return Ok((
            combined.finish(dev, n_cols, "nvcomp cross-shard batch")?,
            DeviceDecodeStats {
                host_uploaded_bytes,
                device_decoded_bytes: 0,
                fully_device_decoded: true,
                n_shards_shufdelta_gpu: shards.len() as u32,
                ..DeviceDecodeStats::default()
            },
        ));
    }

    // Account for every group's compressed frames up front (idx + val).
    host_uploaded_bytes += plan
        .groups
        .iter()
        .map(|g| (g.idx_frame.len() + g.val_frame.len()) as u64)
        .sum::<u64>();

    // Chunk the groups so each `batch_decompress_concat` — and its host `blob`,
    // device plane buffer, and nvcomp temp — stays within a VRAM-derived budget
    // (measured now, after the combined output buffers are allocated). Indices
    // and values are decoded in separate passes per chunk, so at most one plane
    // buffer is resident at a time.
    let group_plane_bytes: Vec<usize> = plan
        .groups
        .iter()
        .map(|g| g.idx_expected + g.val_expected)
        .collect();
    let budget = nvcomp_batch_chunk_budget(dev);
    let chunks = plan_nvcomp_chunks(&group_plane_bytes, budget);

    let t_gpu = profile::start();
    for chunk in chunks {
        let chunk_groups = &plan.groups[chunk.clone()];

        // Pass 1: indices for this chunk.
        let idx_frames: Vec<&[u8]> = chunk_groups.iter().map(|g| g.idx_frame).collect();
        let idx_exp: Vec<usize> = chunk_groups.iter().map(|g| g.idx_expected).collect();
        {
            let (mut d_idx_planes, idx_off) =
                crate::nvcomp::batch_decompress_concat(dev, dev.stream(), &idx_frames, &idx_exp)?;
            for (k, g) in chunk_groups.iter().enumerate() {
                assemble_group_indices_view(
                    dev,
                    &mut d_idx_planes,
                    idx_off[k],
                    Placement {
                        base: g.global_nnz_base,
                        len: g.g_nnz,
                        op: "nvcomp ShufDeltaZstd group",
                        index: chunk.start + k,
                    },
                    g.index_width,
                    &mut combined,
                )?;
            }
            // `d_idx_planes` dropped here — its plane buffer frees before the
            // values pass allocates its own.
        }

        // Pass 2: values for this chunk.
        let val_frames: Vec<&[u8]> = chunk_groups.iter().map(|g| g.val_frame).collect();
        let val_exp: Vec<usize> = chunk_groups.iter().map(|g| g.val_expected).collect();
        let (d_val_planes, val_off) =
            crate::nvcomp::batch_decompress_concat(dev, dev.stream(), &val_frames, &val_exp)?;
        for (k, g) in chunk_groups.iter().enumerate() {
            assemble_group_values_view(
                dev,
                &d_val_planes,
                val_off[k],
                Placement {
                    base: g.global_nnz_base,
                    len: g.g_nnz,
                    op: "nvcomp ShufDeltaZstd group",
                    index: chunk.start + k,
                },
                g.value_width,
                &mut combined,
            )?;
        }
    }
    profile::record_gpu_decode_since(t_gpu);

    Ok((
        combined.finish(dev, n_cols, "nvcomp cross-shard batch")?,
        DeviceDecodeStats {
            host_uploaded_bytes,
            device_decoded_bytes: (total_nnz as u64) * 8,
            fully_device_decoded: true,
            n_shards_shufdelta_gpu: shards.len() as u32,
            ..DeviceDecodeStats::default()
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::plan_nvcomp_chunks;

    /// Chunks exactly partition `0..n` in order (no gaps, no overlap, ascending).
    fn assert_partitions(chunks: &[std::ops::Range<usize>], n: usize) {
        let mut expected = 0usize;
        for c in chunks {
            assert_eq!(c.start, expected, "chunk starts contiguous");
            assert!(c.end > c.start, "chunk non-empty");
            expected = c.end;
        }
        assert_eq!(expected, n, "chunks cover 0..n");
    }

    #[test]
    fn plan_nvcomp_chunks_respects_budget() {
        // budget 10: [3,4,5,2,9] → [3,4] (7) | [5,2] (7) | [9] (9)
        let bytes = [3usize, 4, 5, 2, 9];
        let chunks = plan_nvcomp_chunks(&bytes, 10);
        assert_partitions(&chunks, bytes.len());
        assert_eq!(chunks, vec![0..2, 2..4, 4..5]);
        for c in &chunks {
            // Each chunk fits the budget unless it is a single (oversized) group.
            let sum: usize = bytes[c.clone()].iter().sum();
            assert!(sum <= 10 || c.len() == 1, "chunk {c:?} sum {sum} > budget");
        }
    }

    #[test]
    fn plan_nvcomp_chunks_oversized_group_alone() {
        // A group larger than the budget forms its own chunk; progress is made.
        let bytes = [2usize, 100, 3];
        let chunks = plan_nvcomp_chunks(&bytes, 10);
        assert_partitions(&chunks, bytes.len());
        assert_eq!(chunks, vec![0..1, 1..2, 2..3]);
    }

    #[test]
    fn plan_nvcomp_chunks_single_chunk_when_budget_huge() {
        // usize::MAX budget (VRAM unqueryable) ⇒ exactly one chunk (pre-M3 behavior).
        let bytes = [3usize, 4, 5, 2, 9];
        assert_eq!(plan_nvcomp_chunks(&bytes, usize::MAX), vec![0..5]);
    }

    #[test]
    fn plan_nvcomp_chunks_empty() {
        assert!(plan_nvcomp_chunks(&[], 10).is_empty());
    }
}
