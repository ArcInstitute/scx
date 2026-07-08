//! Optional GPU zstd via NVIDIA nvcomp (batched decompress), loaded at runtime.
//!
//! GPU-SHUFDELTA-DECODE Phase 2: instead of running zstd on the CPU and
//! uploading decompressed plane bytes (Phase 1/1.5), upload the **compressed**
//! per-group zstd frames and decompress them **on the device** with nvcomp's
//! batched API, then finish with the existing undelta/unshuffle/convert kernels.
//! Only compressed bytes cross PCIe → `fully_device_decoded = true`.
//!
//! nvcomp is **not** a build/link dependency. It is `dlopen`ed at runtime from
//! `$CONDA_PREFIX/lib/libnvcomp.so.5` (mirroring the cuVS loader in
//! [`crate::gpu_knn`]); when it is absent the decode dispatch falls back to the
//! Phase-1.5 CPU-zstd pipeline. `SCX_DISABLE_NVCOMP=1` forces the fallback
//! (A/B benchmarking + safety valve).
//!
//! ABI pinned to nvcomp 5.1 (`nvcomp/zstd.h`, `shared_types.h`): the decompress
//! opts struct is a by-value 64-byte `{ i32 backend; char reserved[60]; }`;
//! `nvcompStatus_t` / `nvcompDecompressBackend_t` are C ints (`nvcompSuccess=0`,
//! `NVCOMP_DECOMPRESS_BACKEND_DEFAULT=0`). We over-align every buffer to 256
//! bytes (≥ the 8-byte minimum) so no per-algorithm alignment query is needed.

use std::ffi::c_void;
use std::sync::OnceLock;

use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::device::GpuDevice;
use crate::error::GpuError;

/// Over-alignment applied to every compressed-frame start and per-chunk output
/// offset. ≥ nvcomp's 8-byte zstd minimum; 256 also matches CUDA's allocation
/// granularity, so a buffer base + a 256-multiple offset stays aligned.
const NVCOMP_ALIGN: usize = 256;

/// nvcomp 5.1 `nvcompBatchedZstdDecompressOpts_t` — 64 bytes, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
struct ZstdDecompressOpts {
    /// `nvcompDecompressBackend_t`; 0 = `NVCOMP_DECOMPRESS_BACKEND_DEFAULT`.
    backend: i32,
    /// Reserved, must be zero (forward-compat padding to 64 bytes total).
    reserved: [u8; 60],
}

impl ZstdDecompressOpts {
    fn default_opts() -> Self {
        ZstdDecompressOpts {
            backend: 0,
            reserved: [0u8; 60],
        }
    }
}

// nvcomp 5.1 batched-zstd FFI signatures (nvcomp/zstd.h). `nvcompStatus_t` → i32.
type FnGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    opts: ZstdDecompressOpts,
    temp_bytes: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> i32;

#[allow(clippy::too_many_arguments)]
type FnDecompressAsync = unsafe extern "C" fn(
    device_compressed_chunk_ptrs: *const *const c_void,
    device_compressed_chunk_bytes: *const usize,
    device_uncompressed_buffer_bytes: *const usize,
    device_uncompressed_chunk_bytes: *mut usize,
    num_chunks: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_chunk_ptrs: *const *mut c_void,
    opts: ZstdDecompressOpts,
    device_statuses: *mut i32,
    stream: *mut c_void, // cudaStream_t (ABI-compatible with CUstream)
) -> i32;

/// Loaded nvcomp handle with resolved batched-zstd entry points.
struct NvcompLib {
    _lib: libloading::Library,
    get_temp_size: FnGetTempSize,
    decompress_async: FnDecompressAsync,
}

// SAFETY: fn pointers are resolved once and immutable; nvcomp entry points are
// thread-safe (they operate on caller-provided device buffers + stream).
unsafe impl Send for NvcompLib {}
unsafe impl Sync for NvcompLib {}

static NVCOMP_LIB: OnceLock<Option<NvcompLib>> = OnceLock::new();

fn load_nvcomp() -> Option<NvcompLib> {
    if std::env::var("SCX_DISABLE_NVCOMP")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return None;
    }
    // Bare SONAME first (honors LD_LIBRARY_PATH / ld cache), then $CONDA_PREFIX/lib.
    let names = ["libnvcomp.so.5", "libnvcomp.so"];
    let mut search: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(conda) = std::env::var("CONDA_PREFIX") {
        search.push(std::path::PathBuf::from(format!("{conda}/lib")));
    }
    let lib = names
        .iter()
        .find_map(|n| unsafe { libloading::Library::new(n).ok() })
        .or_else(|| {
            for dir in &search {
                for n in &names {
                    let full = dir.join(n);
                    if full.exists() {
                        if let Ok(l) = unsafe { libloading::Library::new(&full) } {
                            return Some(l);
                        }
                    }
                }
            }
            None
        })?;

    unsafe {
        let get_temp_size: FnGetTempSize = *lib
            .get(b"nvcompBatchedZstdDecompressGetTempSizeAsync\0")
            .ok()?;
        let decompress_async: FnDecompressAsync =
            *lib.get(b"nvcompBatchedZstdDecompressAsync\0").ok()?;
        Some(NvcompLib {
            _lib: lib,
            get_temp_size,
            decompress_async,
        })
    }
}

fn nvcomp_lib() -> Option<&'static NvcompLib> {
    NVCOMP_LIB.get_or_init(load_nvcomp).as_ref()
}

/// Whether nvcomp GPU zstd is available (loadable and not disabled). Cached.
pub fn nvcomp_available() -> bool {
    nvcomp_lib().is_some()
}

/// Whether the Phase-2 nvcomp full-in-VRAM decode should be **used** for framed
/// ShufDeltaZstd shards: opt-in via `SCX_SHUFDELTA_NVCOMP=1` **and** nvcomp
/// loadable. Opt-in (not default) because — although nvcomp achieves
/// `fully_device_decoded` route parity and uploads only compressed bytes (~4×
/// less PCIe) — its per-shard call overhead (temp alloc + host pointer arrays +
/// a stream sync per shard) makes `to_gpu_anndata` slower end-to-end than the
/// Phase-1.5 CPU-zstd pipeline on the metadata-bound wall (measured ~1.4× slower
/// on census_1m). The env var is read per call (not cached), so it can be
/// toggled in-process for A/B benchmarking.
pub fn nvcomp_enabled() -> bool {
    std::env::var("SCX_SHUFDELTA_NVCOMP")
        .map(|v| v == "1")
        .unwrap_or(false)
        && nvcomp_available()
}

#[inline]
fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Batch-decompress independent zstd `frames` on the GPU into one new device
/// buffer, each frame's output concatenated at a 256-aligned offset. Returns
/// the device buffer and the per-frame output offsets (use `off..off+expected`).
///
/// `expected[i]` is the exact decompressed size of `frames[i]` (known from the
/// shard: `g_nnz * width`). Fails loud if nvcomp reports any per-chunk error or
/// an actual size != expected.
pub fn batch_decompress_concat(
    dev: &GpuDevice,
    stream: &CudaStream,
    frames: &[&[u8]],
    expected: &[usize],
) -> Result<(CudaSlice<u8>, Vec<usize>), GpuError> {
    let lib =
        nvcomp_lib().ok_or_else(|| GpuError::CudaError("nvcomp not available".to_string()))?;
    assert_eq!(frames.len(), expected.len());
    let n = frames.len();

    // Output layout: 256-aligned per-frame offsets in one buffer.
    let mut out_offsets = Vec::with_capacity(n);
    let mut out_cursor = 0usize;
    for &exp in expected {
        out_offsets.push(out_cursor);
        out_cursor += align_up(exp, NVCOMP_ALIGN);
    }
    let out_total = out_cursor.max(1);

    // Compressed blob: 256-aligned per-frame offsets, copied on the host.
    let mut comp_offsets = Vec::with_capacity(n);
    let mut comp_cursor = 0usize;
    for f in frames {
        comp_offsets.push(comp_cursor);
        comp_cursor += align_up(f.len(), NVCOMP_ALIGN);
    }
    let mut blob = vec![0u8; comp_cursor.max(1)];
    for (i, f) in frames.iter().enumerate() {
        blob[comp_offsets[i]..comp_offsets[i] + f.len()].copy_from_slice(f);
    }

    let comp_bytes: Vec<usize> = frames.iter().map(|f| f.len()).collect();
    let uncomp_caps: Vec<usize> = expected
        .iter()
        .map(|&e| align_up(e, NVCOMP_ALIGN))
        .collect();

    // `d_out` / `d_status` / `d_actual` outlive the guard block below so they
    // can be returned / read after the device-pointer guards (which borrow
    // them) are dropped.
    let mut d_out = dev.alloc_zeros::<u8>(out_total)?;
    let mut d_actual = dev.alloc_zeros::<usize>(n)?;
    let mut d_status = dev.alloc_zeros::<i32>(n)?;

    // The whole async decompress + its stream sync happen inside this block, so
    // all device-pointer guards (incl. the `&mut d_out` borrow) release before
    // we read/return the buffers below.
    {
        let d_blob = dev.htod_copy(&blob)?;
        let (blob_base, _g_blob) = d_blob.device_ptr(stream);
        let (out_base, _g_out) = d_out.device_ptr_mut(stream);

        let comp_ptrs: Vec<u64> = comp_offsets.iter().map(|&o| blob_base + o as u64).collect();
        let uncomp_ptrs: Vec<u64> = out_offsets.iter().map(|&o| out_base + o as u64).collect();
        let d_comp_ptrs = dev.htod_copy(&comp_ptrs)?;
        let d_comp_bytes = dev.htod_copy(&comp_bytes)?;
        let d_uncomp_caps = dev.htod_copy(&uncomp_caps)?;
        let d_uncomp_ptrs = dev.htod_copy(&uncomp_ptrs)?;

        let opts = ZstdDecompressOpts::default_opts();
        let max_expected = expected.iter().copied().max().unwrap_or(0);
        let mut temp_bytes: usize = 0;
        let st = unsafe { (lib.get_temp_size)(n, max_expected, opts, &mut temp_bytes, out_total) };
        if st != 0 {
            return Err(GpuError::CudaError(format!(
                "nvcompBatchedZstdDecompressGetTempSizeAsync failed: status {st}"
            )));
        }
        let mut d_temp = dev.alloc_zeros::<u8>(temp_bytes.max(1))?;

        let (comp_ptrs_dev, _g1) = d_comp_ptrs.device_ptr(stream);
        let (comp_bytes_dev, _g2) = d_comp_bytes.device_ptr(stream);
        let (uncomp_caps_dev, _g3) = d_uncomp_caps.device_ptr(stream);
        let (uncomp_ptrs_dev, _g4) = d_uncomp_ptrs.device_ptr(stream);
        let (actual_dev, _g5) = d_actual.device_ptr_mut(stream);
        let (status_dev, _g6) = d_status.device_ptr_mut(stream);
        let (temp_dev, _g7) = d_temp.device_ptr_mut(stream);

        let st = unsafe {
            (lib.decompress_async)(
                (comp_ptrs_dev as usize) as *const *const c_void,
                (comp_bytes_dev as usize) as *const usize,
                (uncomp_caps_dev as usize) as *const usize,
                (actual_dev as usize) as *mut usize,
                n,
                (temp_dev as usize) as *mut c_void,
                temp_bytes,
                (uncomp_ptrs_dev as usize) as *const *mut c_void,
                opts,
                (status_dev as usize) as *mut i32,
                (stream.cu_stream() as usize) as *mut c_void,
            )
        };
        if st != 0 {
            return Err(GpuError::CudaError(format!(
                "nvcompBatchedZstdDecompressAsync launch failed: status {st}"
            )));
        }

        // Await completion before the guards (and staging buffers) drop.
        stream
            .synchronize()
            .map_err(|e| GpuError::CudaError(format!("nvcomp stream sync: {e}")))?;
    }

    // Verify every chunk succeeded with the exact expected size.
    let statuses = dev.dtoh_copy(&d_status)?;
    let actual = dev.dtoh_copy(&d_actual)?;
    for i in 0..n {
        if statuses[i] != 0 {
            return Err(GpuError::CudaError(format!(
                "nvcomp chunk {i} decode status {} (nonzero = error)",
                statuses[i]
            )));
        }
        if actual[i] != expected[i] {
            return Err(GpuError::CudaError(format!(
                "nvcomp chunk {i} decoded {} bytes != expected {}",
                actual[i], expected[i]
            )));
        }
    }

    Ok((d_out, out_offsets))
}
