//! Full shard GPU decode pipeline.
//!
//! Parses a raw shard byte buffer (header + encoded data), dispatches to
//! the appropriate codec path (GPU-accelerated for Scx1, CPU fallback for
//! None/Zstd), and returns a GPU-resident CSR matrix.

use std::io::Cursor;

use cudarc::driver::safe::CudaSlice;

use scx_codec::delta_golomb::delta_golomb_decode;
use scx_codec::rice::B_VAL;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::shard::{
    resolve_block_index, ShardHeader, DEFAULT_WRITE_SHARD_FORMAT_VERSION, SHARD_HEADER_SIZE,
};

use crate::cast_gpu::{cast_u32_to_f32_gpu, cast_u32_to_i32_gpu};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::forbp_gpu::forbp_decode_gpu;
use crate::profile::{self, CodecClass};
use crate::rice_gpu::rice_decode_gpu;
use crate::shufdelta_gpu::{decode_indices_frame_to_device, decode_values_frame_to_device};

/// GPU-resident CSR matrix.
///
/// Type layout matches scipy CSR conventions: i64 indptr, i32 indices, f32 data.
/// Binary-compatible with cuSPARSE and cupy `__cuda_array_interface__`.
pub struct GpuCsr {
    /// Compressed row pointer array (`n_rows + 1` elements, i64).
    pub indptr: CudaSlice<i64>,
    /// Column indices array (`nnz` elements, i32).
    pub indices: CudaSlice<i32>,
    /// Non-zero values array (`nnz` elements, f32).
    pub data: CudaSlice<f32>,
    /// Matrix dimensions `(n_rows, n_cols)`.
    pub shape: (usize, usize),
}

/// Host→device transfer accounting for a device decode, surfaced for the
/// ACC-RUST-OPT-V4 §4.4 `transfer_mode` / `bytes_uploaded` route metadata.
///
/// Lets a caller distinguish a genuine in-VRAM Scx1 decode (only the tiny indptr
/// uploaded) from a host-decode+HtoD bounce. All Scx1 indices+values (including
/// the >= 128-nnz BitPacker4x rows, Task 4.4b) decode on the device; only a
/// non-Scx1 codec shard decodes wholly on the host.
#[derive(Debug, Clone, Copy)]
pub struct DeviceDecodeStats {
    /// Total host→device bytes: the Scx1 indptr (always uploaded) and the full
    /// payload of any non-Scx1 shard.
    pub host_uploaded_bytes: u64,
    /// Bytes decoded directly on the device (Scx1 indices/values that took the
    /// GPU kernel path).
    pub device_decoded_bytes: u64,
    /// True iff every shard decoded its indices+values on the device (all Scx1)
    /// — i.e. only indptr was uploaded. Drives the `scx_device_decode_gpu`
    /// transfer-mode stamp.
    pub fully_device_decoded: bool,
    /// Shards that took the in-VRAM Scx1 device path (`decode_scx1_gpu` /
    /// `decode_framed_scx1_gpu`).
    pub n_shards_scx1_gpu: u32,
    /// Shards that took a ShufDeltaZstd GPU decode path. This is how per-codec
    /// GPU routing is observed for `compact-trial` (mixed-codec) files.
    /// `fully_device_decoded` is `true` for the Phase-2 nvcomp path (only
    /// compressed bytes cross PCIe) and `false` for the Phase-1/1.5 CPU-zstd
    /// paths (decompressed plane bytes cross PCIe).
    pub n_shards_shufdelta_gpu: u32,
    /// Shards that host-decoded + HtoD-bounced (`decode_host_bounce`).
    pub n_shards_host_bounced: u32,
}

impl Default for DeviceDecodeStats {
    fn default() -> Self {
        // `fully_device_decoded` starts `true` so aggregation can AND it down as
        // shards are merged; a fresh single-shard stat sets it explicitly.
        Self {
            host_uploaded_bytes: 0,
            device_decoded_bytes: 0,
            fully_device_decoded: true,
            n_shards_scx1_gpu: 0,
            n_shards_shufdelta_gpu: 0,
            n_shards_host_bounced: 0,
        }
    }
}

impl DeviceDecodeStats {
    /// Fold a per-shard stat into a running aggregate.
    pub fn merge(&mut self, other: &DeviceDecodeStats) {
        self.host_uploaded_bytes += other.host_uploaded_bytes;
        self.device_decoded_bytes += other.device_decoded_bytes;
        self.fully_device_decoded &= other.fully_device_decoded;
        self.n_shards_scx1_gpu += other.n_shards_scx1_gpu;
        self.n_shards_shufdelta_gpu += other.n_shards_shufdelta_gpu;
        self.n_shards_host_bounced += other.n_shards_host_bounced;
    }
}

/// Decode an entire SCX shard on GPU, producing a [`GpuCsr`].
///
/// Pipeline:
/// 1. Parse shard header (76 bytes, CPU)
/// 2. Extract encoded indptr/indices/values byte slices
/// 3. Dispatch on codec_id:
///    - **Scx1**: Delta-Golomb (CPU) + FOR-BP (GPU) + Rice (GPU)
///    - **None/Zstd**: CPU decode via `decode_shard_scipy` + upload
/// 4. Type-convert to i64/i32/f32 and return GpuCsr
///
/// Produces **bit-identical** output to `scx_codec::decode_shard_scipy`.
pub fn decode_shard_gpu(dev: &GpuDevice, shard_bytes: &[u8]) -> Result<GpuCsr, GpuError> {
    decode_shard_gpu_with_stats(dev, shard_bytes).map(|(csr, _stats)| csr)
}

/// Decode an entire SCX shard on GPU, returning the [`GpuCsr`] plus
/// [`DeviceDecodeStats`] describing the host↔device transfer. Framed Scx1 shards
/// decode in VRAM group-by-group; unframed Scx1 shards decode via the FOR-BP /
/// Rice CPU prescan; all other codecs host-decode + HtoD bounce.
pub fn decode_shard_gpu_with_stats(
    dev: &GpuDevice,
    shard_bytes: &[u8],
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    // 1. Parse 76-byte shard header
    if shard_bytes.len() < SHARD_HEADER_SIZE {
        return Err(GpuError::InvalidShard(format!(
            "shard too small: {} bytes < {} header",
            shard_bytes.len(),
            SHARD_HEADER_SIZE
        )));
    }
    let header = ShardHeader::read_from(&mut Cursor::new(shard_bytes))
        .map_err(|e| GpuError::InvalidShard(format!("shard header: {e}")))?;

    let codec_id = CodecId::from_u8(header.codec_id)
        .ok_or_else(|| GpuError::InvalidShard(format!("unknown codec_id: {}", header.codec_id)))?;
    let value_encoding = ValueEncoding::from_u8(header.value_encoding).ok_or_else(|| {
        GpuError::InvalidShard(format!("unknown value_encoding: {}", header.value_encoding))
    })?;
    let index_dtype_u16 = header.index_dtype == 0;
    let n_rows = header.n_major as usize;
    let n_cols = header.n_minor as usize;
    let nnz = header.nnz as usize;

    // 2. Extract encoded byte slices
    let indptr_bytes = extract_slice(
        shard_bytes,
        header.indptr_rel_offset,
        header.indptr_length,
        "indptr",
    )?;
    let indices_bytes = extract_slice(
        shard_bytes,
        header.indices_rel_offset,
        header.indices_length,
        "indices",
    )?;
    let values_bytes = extract_slice(
        shard_bytes,
        header.values_rel_offset,
        header.values_length,
        "values",
    )?;
    let block_index_bytes = extract_slice(
        shard_bytes,
        header.block_index_rel_offset,
        header.block_index_length,
        "block_index",
    )?;

    // 3. Dispatch.
    //
    // Row-group-framed (shard v2) shards store per-group local-rebased indptrs
    // plus a concatenation of per-group value/index frames. Framed **Scx1**
    // decodes in VRAM group-by-group (each group is a standalone Scx1 sub-shard,
    // so the FOR-BP/Rice kernels run per group frame — Phase E). Framed non-Scx1
    // codecs (None/Zstd/Lz4Shuffle/Pcodec/ShufDeltaZstd) have no GPU decoder, so
    // they host-decode + HtoD bounce via the shared CPU path.
    if header.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION {
        return match codec_id {
            CodecId::Scx1 => decode_framed_scx1_gpu(
                dev,
                &header,
                indptr_bytes,
                indices_bytes,
                values_bytes,
                block_index_bytes,
            ),
            // ShufDeltaZstd: CPU zstd + GPU undelta/unshuffle/convert per group
            // (Phase 1). Float-valued shufdelta falls back to host-bounce inside.
            CodecId::ShufDeltaZstd => decode_framed_shufdelta_gpu(
                dev,
                &header,
                indptr_bytes,
                indices_bytes,
                values_bytes,
                block_index_bytes,
            ),
            _ => decode_host_bounce(
                dev,
                &header,
                indptr_bytes,
                indices_bytes,
                values_bytes,
                block_index_bytes,
            ),
        };
    }

    match codec_id {
        CodecId::Scx1 => decode_scx1_gpu(
            dev,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            value_encoding,
            index_dtype_u16,
            n_rows,
            n_cols,
            nnz,
        ),
        CodecId::ShufDeltaZstd => decode_shufdelta_gpu(
            dev,
            &header,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            block_index_bytes,
        ),
        CodecId::None | CodecId::Zstd | CodecId::Lz4Shuffle | CodecId::Pcodec => {
            decode_host_bounce(
                dev,
                &header,
                indptr_bytes,
                indices_bytes,
                values_bytes,
                block_index_bytes,
            )
        }
    }
}

/// Extract a byte slice from shard data using relative offset and length.
fn extract_slice<'a>(
    shard_bytes: &'a [u8],
    rel_offset: u32,
    length: u32,
    name: &str,
) -> Result<&'a [u8], GpuError> {
    let start = rel_offset as usize;
    let end = start + length as usize;
    if end > shard_bytes.len() {
        return Err(GpuError::InvalidShard(format!(
            "{name} slice [{start}..{end}] exceeds shard size {}",
            shard_bytes.len()
        )));
    }
    Ok(&shard_bytes[start..end])
}

/// GPU-accelerated Scx1 decode path.
///
/// - indptr: Delta-Golomb on CPU (small array, not worth GPU overhead)
/// - indices: FOR-BP on GPU → on-device u32→i32 cast
/// - values: Rice on GPU → on-device u32→f32 cast
#[allow(clippy::too_many_arguments)]
fn decode_scx1_gpu(
    dev: &GpuDevice,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    n_rows: usize,
    n_cols: usize,
    nnz: usize,
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    if !value_encoding.is_integer() {
        return Err(GpuError::InvalidShard(
            "Scx1 codec does not support float value encodings".into(),
        ));
    }

    let mut stats = DeviceDecodeStats::default();

    // indptr: Delta-Golomb on CPU → Vec<u64> → Vec<i64> → upload
    let t_host = profile::start();
    let indptr_u64 =
        delta_golomb_decode(indptr_bytes, n_rows + 1).map_err(scx_codec::CodecError::from)?;
    let indptr_i64: Vec<i64> = indptr_u64.into_iter().map(|v| v as i64).collect();
    profile::record_host_decode_since(CodecClass::Scx1, t_host);

    let t_htod = profile::start();
    let d_indptr = dev.htod_copy(&indptr_i64)?;
    let indptr_bytes_uploaded = (indptr_i64.len() * 8) as u64;
    profile::record_htod_since(CodecClass::Scx1, t_htod, indptr_i64.len() * 8);
    // indptr always round-trips host→device; it is the residual upload of a
    // fully-GPU-decoded Scx1 shard (the "only the tiny indptr" of §2.5).
    stats.host_uploaded_bytes += indptr_bytes_uploaded;

    // indices: FOR-BP on GPU → on-device u32→i32 cast (no host round-trip)
    // values:  Rice on GPU → on-device u32→f32 cast (no host round-trip)
    // Both decode GPU-side; the bucket covers the bitstream upload + kernels.
    // The kernels re-derive per-row/per-block offsets via a GPU prescan.
    let t_gpu = profile::start();
    let (d_indices_u32, _row_lengths) =
        forbp_decode_gpu(dev, indices_bytes, n_rows, index_dtype_u16)?;
    // FOR-BP indices (scalar + BitPacker4x rows, Task 4.4b) and Rice values both
    // decode on the device — the gpu_decode bucket covers the bitstream upload +
    // kernels; only the indptr round-trips through the host.
    stats.device_decoded_bytes += (nnz as u64) * 4;
    let d_indices = cast_u32_to_i32_gpu(dev, &d_indices_u32)?;

    let d_values_u32 = rice_decode_gpu(dev, values_bytes, nnz, B_VAL)?;
    // Rice values always decode on the device.
    stats.device_decoded_bytes += (nnz as u64) * 4;
    let d_data = cast_u32_to_f32_gpu(dev, &d_values_u32)?;
    if t_gpu.is_some() {
        // Synchronize so the elapsed time reflects completed device work, not
        // just async launch latency. Only paid when profiling is enabled.
        dev.synchronize()?;
    }
    profile::record_gpu_decode_since(t_gpu);

    stats.n_shards_scx1_gpu = 1;
    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: d_indices,
            data: d_data,
            shape: (n_rows, n_cols),
        },
        stats,
    ))
}

/// Host-decode + HtoD upload path for shards that do not take the in-VRAM Scx1
/// device kernels: any framed (v2) **non-Scx1** shard (framed Scx1 routes to
/// [`decode_framed_scx1_gpu`]) and any unframed non-Scx1 codec
/// (None / Zstd / Lz4Shuffle / Pcodec / ShufDeltaZstd).
///
/// Decodes on the host via the shared [`scx_format_io::decode_shard_regions_scipy`]
/// — which iterates the row-group block index for framed shards and reads the
/// whole stream for legacy shards — then uploads the assembled CSR. Always a
/// host bounce, so `fully_device_decoded` is `false` and the caller stamps
/// `scx_device_handoff_streamed`.
fn decode_host_bounce(
    dev: &GpuDevice,
    header: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    let n_rows = header.n_major as usize;
    let n_cols = header.n_minor as usize;

    let t_host = profile::start();
    let (indptr, indices, data) = scx_format_io::decode_shard_regions_scipy(
        header,
        indptr_bytes,
        indices_bytes,
        values_bytes,
        block_index_bytes,
    )
    .map_err(|e| GpuError::InvalidShard(format!("host-bounce shard decode: {e}")))?;
    profile::record_host_decode_since(CodecClass::Generic, t_host);

    let t_htod = profile::start();
    let d_indptr = dev.htod_copy(&indptr)?;
    let d_indices = dev.htod_copy(&indices)?;
    let d_data = dev.htod_copy(&data)?;
    let htod_bytes = indptr.len() * 8 + indices.len() * 4 + data.len() * 4;
    profile::record_htod_since(CodecClass::Generic, t_htod, htod_bytes);

    // Host-decoded shards upload the full CSR — never a device decode, so this
    // never counts as `fully_device_decoded`.
    let stats = DeviceDecodeStats {
        host_uploaded_bytes: htod_bytes as u64,
        device_decoded_bytes: 0,
        fully_device_decoded: false,
        n_shards_host_bounced: 1,
        ..DeviceDecodeStats::default()
    };

    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: d_indices,
            data: d_data,
            shape: (n_rows, n_cols),
        },
        stats,
    ))
}

/// In-VRAM device decode of a **framed (v2) Scx1** shard, group by group.
///
/// Each row-group is a standalone Scx1 sub-shard (group-local Delta-Golomb
/// indptr + FOR-BP index frame + Rice value frame), so the existing per-frame
/// prescans (`forbp_decode_gpu` / `rice_decode_gpu`) run verbatim on each group's
/// byte slice; the decoded per-group device buffers are dtod-concatenated into
/// combined nnz-sized buffers (the same idiom as the cross-shard
/// `decode_csr_shards_to_device_with_stats`, applied at group granularity).
/// The global indptr is assembled on the host from each group's local indptr via
/// `scx_codec::decode_row_group_indptr_only` (only the indptr sub-stream frame is
/// decoded on the host; the much larger index/value frames go to the device).
///
/// Only the indptr round-trips to the device, so `fully_device_decoded` is `true`
/// and the caller stamps `scx_device_decode_gpu`.
fn decode_framed_scx1_gpu(
    dev: &GpuDevice,
    header: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    let value_encoding = ValueEncoding::from_u8(header.value_encoding).ok_or_else(|| {
        GpuError::InvalidShard(format!("unknown value_encoding: {}", header.value_encoding))
    })?;
    if !value_encoding.is_integer() {
        return Err(GpuError::InvalidShard(
            "Scx1 codec does not support float value encodings".into(),
        ));
    }
    let index_dtype_u16 = header.index_dtype == 0;
    let n_rows = header.n_major as usize;
    let n_cols = header.n_minor as usize;
    let nnz = header.nnz as usize;

    let spans = resolve_block_index(header, block_index_bytes)
        .map_err(|e| GpuError::InvalidShard(format!("framed Scx1 block index: {e}")))?;

    // Combined nnz-sized device buffers, filled per group at the running offset.
    let mut combined_indices = dev.alloc_zeros::<i32>(nnz)?;
    let mut combined_data = dev.alloc_zeros::<f32>(nnz)?;
    // Host-assembled global indptr (tiny: n_rows + 1 elements).
    let mut combined_indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    combined_indptr.push(0);

    let mut nnz_base: usize = 0;
    for span in &spans {
        let g_rows = span.n_rows as usize;
        let g_nnz = span.nnz as usize;

        // Global indptr: decode only this group's local indptr frame on the host
        // (indices/values frames are never host-decoded), then rebase by nnz_base.
        let g_indptr = scx_codec::decode_row_group_indptr_only(CodecId::Scx1, span, indptr_bytes)?;
        // decode_row_group_indptr_only returns exactly g_rows+1 entries (or an
        // error); assert it so the `[1..=g_rows]` slice can never panic.
        debug_assert_eq!(g_indptr.len(), g_rows + 1);
        for &local in &g_indptr[1..=g_rows] {
            combined_indptr.push(nnz_base as i64 + local);
        }

        if g_nnz > 0 {
            // Decode this group's index + value frames on the device.
            let ix_frame = &indices_bytes[span.indices.clone()];
            let vv_frame = &values_bytes[span.values.clone()];
            let (d_indices_u32, _row_lengths) =
                forbp_decode_gpu(dev, ix_frame, g_rows, index_dtype_u16)?;
            let d_indices = cast_u32_to_i32_gpu(dev, &d_indices_u32)?;
            let d_values_u32 = rice_decode_gpu(dev, vv_frame, g_nnz, B_VAL)?;
            let d_data = cast_u32_to_f32_gpu(dev, &d_values_u32)?;
            // The block index already guarantees Σ nnz == header.nnz; these
            // localize a corrupt frame that decoded a different length before it
            // becomes a mismatched-length dtod copy (mirrors gpu_csr_assemble).
            debug_assert_eq!(d_indices.len(), g_nnz);
            debug_assert_eq!(d_data.len(), g_nnz);

            let mut idx_dst = combined_indices.slice_mut(nnz_base..nnz_base + g_nnz);
            dev.stream()
                .memcpy_dtod(&d_indices, &mut idx_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod indices (framed group): {e}")))?;
            let mut data_dst = combined_data.slice_mut(nnz_base..nnz_base + g_nnz);
            dev.stream()
                .memcpy_dtod(&d_data, &mut data_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod data (framed group): {e}")))?;
        }

        nnz_base += g_nnz;
    }
    debug_assert_eq!(nnz_base, nnz);
    debug_assert_eq!(combined_indptr.len(), n_rows + 1);

    let indptr_bytes_uploaded = (combined_indptr.len() * 8) as u64;
    let d_indptr = dev.htod_copy(&combined_indptr)?;
    // The per-group FOR-BP/Rice/cast kernels and dtod copies are queued async on
    // the stream; sync so a downstream cuPy consumer cannot race unfinished work.
    dev.synchronize()?;

    let stats = DeviceDecodeStats {
        // Only the assembled indptr round-trips host→device; indices+values
        // decode in VRAM (like the unframed Scx1 device path).
        host_uploaded_bytes: indptr_bytes_uploaded,
        device_decoded_bytes: (nnz as u64) * 8,
        fully_device_decoded: true,
        n_shards_scx1_gpu: 1,
        ..DeviceDecodeStats::default()
    };

    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: combined_indices,
            data: combined_data,
            shape: (n_rows, n_cols),
        },
        stats,
    ))
}

/// GPU decode of a **framed (v2) ShufDeltaZstd** shard. Selects the fastest
/// available path (see the body):
///
/// - **Phase 2 (nvcomp)** when nvcomp is loadable — upload compressed frames,
///   GPU zstd, then kernels. Only compressed bytes cross PCIe, so the caller
///   stamps `fully_device_decoded = true` (Scx1 parity).
/// - **Phase 1.5 (pipeline)** — parallel CPU zstd overlapped with async GPU
///   uploads/kernels (≥2 groups); uploads decompressed planes → `false`.
/// - **Phase 1 (sequential)** — per-group CPU zstd + GPU transforms.
///
/// Per-codec GPU routing is tracked via `n_shards_shufdelta_gpu`. Float value
/// encodings (zstd-only, no plane transforms) have no GPU path and fall back to
/// [`decode_host_bounce`].
fn decode_framed_shufdelta_gpu(
    dev: &GpuDevice,
    header: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    let value_encoding = ValueEncoding::from_u8(header.value_encoding).ok_or_else(|| {
        GpuError::InvalidShard(format!("unknown value_encoding: {}", header.value_encoding))
    })?;
    // Float ShufDeltaZstd values are zstd-only (no shuffle/delta) — no GPU
    // transform path; fall back to the host bounce (spec constraint 5).
    if !value_encoding.is_integer() {
        return decode_host_bounce(
            dev,
            header,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            block_index_bytes,
        );
    }
    let index_width = if header.index_dtype == 0 { 2 } else { 4 };
    let n_rows = header.n_major as usize;
    let n_cols = header.n_minor as usize;
    let nnz = header.nnz as usize;

    let spans = resolve_block_index(header, block_index_bytes)
        .map_err(|e| GpuError::InvalidShard(format!("framed ShufDeltaZstd block index: {e}")))?;

    // Decode path selection:
    //  1. Phase 2 — nvcomp full in-VRAM (OPT-IN via `SCX_SHUFDELTA_NVCOMP=1`):
    //     upload compressed frames, GPU zstd, then kernels. Only compressed bytes
    //     cross PCIe → `fully_device_decoded=true` (transfer_mode
    //     `scx_device_decode_gpu`, Scx1 parity). Opt-in, not default: nvcomp's
    //     per-shard overhead makes `to_gpu_anndata` slower end-to-end than the
    //     pipeline on the metadata-bound wall (see `nvcomp::nvcomp_enabled`).
    //  2. Phase 1.5 (DEFAULT) — pipeline the per-group CPU zstd (parallel workers)
    //     with async GPU uploads/kernels (≥2 groups). Uploads decompressed → `false`.
    //  3. Phase 1 — sequential CPU zstd + GPU transforms.
    // `SCX_SHUFDELTA_GPU_SEQUENTIAL=1` forces path 3 (safety valve + A/B toggle).
    let force_sequential = std::env::var("SCX_SHUFDELTA_GPU_SEQUENTIAL")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_sequential && crate::nvcomp::nvcomp_enabled() {
        let pcsr = crate::shufdelta_gpu::decode_framed_shufdelta_gpu_nvcomp(
            dev,
            &spans,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            n_rows,
            nnz,
            value_encoding,
            index_width,
        )?;
        let stats = DeviceDecodeStats {
            host_uploaded_bytes: pcsr.host_uploaded_bytes,
            device_decoded_bytes: (nnz as u64) * 8,
            // Phase 2: only compressed bytes crossed PCIe — full device decode.
            fully_device_decoded: true,
            n_shards_shufdelta_gpu: 1,
            ..DeviceDecodeStats::default()
        };
        return Ok((
            GpuCsr {
                indptr: pcsr.indptr,
                indices: pcsr.indices,
                data: pcsr.data,
                shape: (n_rows, n_cols),
            },
            stats,
        ));
    }
    if spans.len() >= 2 && !force_sequential {
        let pcsr = crate::shufdelta_gpu::decode_framed_shufdelta_gpu_pipelined(
            dev,
            &spans,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            n_rows,
            nnz,
            value_encoding,
            index_width,
        )?;
        let stats = DeviceDecodeStats {
            host_uploaded_bytes: pcsr.host_uploaded_bytes,
            device_decoded_bytes: (nnz as u64) * 8,
            fully_device_decoded: false,
            n_shards_shufdelta_gpu: 1,
            ..DeviceDecodeStats::default()
        };
        return Ok((
            GpuCsr {
                indptr: pcsr.indptr,
                indices: pcsr.indices,
                data: pcsr.data,
                shape: (n_rows, n_cols),
            },
            stats,
        ));
    }

    let mut combined_indices = dev.alloc_zeros::<i32>(nnz)?;
    let mut combined_data = dev.alloc_zeros::<f32>(nnz)?;
    let mut combined_indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    combined_indptr.push(0);

    let mut host_uploaded_bytes: u64 = 0;
    let mut nnz_base: usize = 0;
    for span in &spans {
        let g_rows = span.n_rows as usize;
        let g_nnz = span.nnz as usize;

        let g_indptr =
            scx_codec::decode_row_group_indptr_only(CodecId::ShufDeltaZstd, span, indptr_bytes)?;
        debug_assert_eq!(g_indptr.len(), g_rows + 1);
        for &local in &g_indptr[1..=g_rows] {
            combined_indptr.push(nnz_base as i64 + local);
        }

        if g_nnz > 0 {
            let ix_frame = &indices_bytes[span.indices.clone()];
            let vv_frame = &values_bytes[span.values.clone()];
            let (d_indices, up_i) =
                decode_indices_frame_to_device(dev, ix_frame, g_nnz, index_width)?;
            let (d_data, up_v) =
                decode_values_frame_to_device(dev, vv_frame, g_nnz, value_encoding)?;
            host_uploaded_bytes += up_i + up_v;
            debug_assert_eq!(d_indices.len(), g_nnz);
            debug_assert_eq!(d_data.len(), g_nnz);

            let mut idx_dst = combined_indices.slice_mut(nnz_base..nnz_base + g_nnz);
            dev.stream()
                .memcpy_dtod(&d_indices, &mut idx_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod indices (shufdelta group): {e}")))?;
            let mut data_dst = combined_data.slice_mut(nnz_base..nnz_base + g_nnz);
            dev.stream()
                .memcpy_dtod(&d_data, &mut data_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod data (shufdelta group): {e}")))?;
        }

        nnz_base += g_nnz;
    }
    debug_assert_eq!(nnz_base, nnz);
    debug_assert_eq!(combined_indptr.len(), n_rows + 1);

    host_uploaded_bytes += (combined_indptr.len() * 8) as u64;
    let d_indptr = dev.htod_copy(&combined_indptr)?;
    // Per-group kernels + dtod copies are queued async on the stream; sync so a
    // downstream cuPy consumer cannot race unfinished work.
    dev.synchronize()?;

    let stats = DeviceDecodeStats {
        host_uploaded_bytes,
        device_decoded_bytes: (nnz as u64) * 8,
        // Phase 1: decompressed plane bytes crossed PCIe, so this is not the
        // Scx1 "only indptr uploaded" device decode.
        fully_device_decoded: false,
        n_shards_shufdelta_gpu: 1,
        ..DeviceDecodeStats::default()
    };

    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: combined_indices,
            data: combined_data,
            shape: (n_rows, n_cols),
        },
        stats,
    ))
}

/// GPU decode of an **unframed (v1) ShufDeltaZstd** shard (whole shard as a
/// single zstd frame per sub-stream). Same transforms as the framed path with
/// no per-group loop; the frame helpers produce the full nnz-sized device
/// buffers directly. Float value encodings fall back to [`decode_host_bounce`].
fn decode_shufdelta_gpu(
    dev: &GpuDevice,
    header: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    let value_encoding = ValueEncoding::from_u8(header.value_encoding).ok_or_else(|| {
        GpuError::InvalidShard(format!("unknown value_encoding: {}", header.value_encoding))
    })?;
    let n_rows = header.n_major as usize;
    let n_cols = header.n_minor as usize;
    let nnz = header.nnz as usize;
    if !value_encoding.is_integer() {
        return decode_host_bounce(
            dev,
            header,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            block_index_bytes,
        );
    }
    let index_width = if header.index_dtype == 0 { 2 } else { 4 };

    // Whole-shard indptr on the host (tiny); indices/values decode on device.
    let combined_indptr =
        scx_codec::decode_indptr_only(indptr_bytes, CodecId::ShufDeltaZstd, n_rows)
            .map_err(|e| GpuError::InvalidShard(format!("unframed ShufDeltaZstd indptr: {e}")))?;
    debug_assert_eq!(combined_indptr.len(), n_rows + 1);
    let mut host_uploaded_bytes = (combined_indptr.len() * 8) as u64;

    let (combined_indices, combined_data) = if nnz > 0 {
        let (d_indices, up_i) =
            decode_indices_frame_to_device(dev, indices_bytes, nnz, index_width)?;
        let (d_data, up_v) = decode_values_frame_to_device(dev, values_bytes, nnz, value_encoding)?;
        host_uploaded_bytes += up_i + up_v;
        (d_indices, d_data)
    } else {
        (dev.alloc_zeros::<i32>(0)?, dev.alloc_zeros::<f32>(0)?)
    };

    let d_indptr = dev.htod_copy(&combined_indptr)?;
    dev.synchronize()?;

    let stats = DeviceDecodeStats {
        host_uploaded_bytes,
        device_decoded_bytes: (nnz as u64) * 8,
        fully_device_decoded: false,
        n_shards_shufdelta_gpu: 1,
        ..DeviceDecodeStats::default()
    };

    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: combined_indices,
            data: combined_data,
            shape: (n_rows, n_cols),
        },
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::build_test_shard;
    use scx_codec::{decode_shard_scipy, EncodedShardRef};
    use scx_format_io::shard::ShardHeader;

    /// CPU reference decode for comparison.
    fn cpu_decode(
        indptr_bytes: &[u8],
        indices_bytes: &[u8],
        values_bytes: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        n_rows: usize,
        nnz: usize,
        index_dtype_u16: bool,
    ) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        let encoded = EncodedShardRef {
            indptr_bytes,
            indices_bytes,
            values_bytes,
        };
        decode_shard_scipy(
            &encoded,
            codec_id,
            value_encoding,
            n_rows,
            nnz,
            index_dtype_u16,
        )
        .expect("CPU decode failed")
    }

    #[test]
    fn test_shard_decode_gpu_scx1() {
        let dev = require_gpu!();

        // Build test CSR: 200 rows, ~1000 columns, u16 values
        let n_rows = 200;
        let n_cols: u32 = 1000;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values_u16 = Vec::new();

        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for _ in 0..n_rows {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let nnz_row = (state % 20 + 1) as usize;
            let mut col = 0u32;
            for _ in 0..nnz_row {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                col += (state % 10 + 1) as u32;
                if col >= n_cols {
                    break;
                }
                indices.push(col);
                // UMI-like values: mostly 1-3, occasional larger
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let v = ((state % 5) + 1) as u16;
                values_u16.push(v);
            }
            indptr.push(indices.len() as u64);
        }

        // Convert u16 values to raw LE bytes
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let nnz = indices.len();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        // GPU decode
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        // CPU reference decode
        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            true,
        );

        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }

    /// Build a deterministic CSR with the given per-row nnz counts. Columns are
    /// strictly increasing within each row (FOR-BP requires sorted indices).
    /// Returns `(indptr, indices, values_u16)`.
    fn build_dense_csr(row_nnzs: &[usize], n_cols: u32) -> (Vec<u64>, Vec<u32>, Vec<u16>) {
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values_u16: Vec<u16> = Vec::new();
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for &nnz_row in row_nnzs {
            let mut col = 0u32;
            for _ in 0..nnz_row {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                col += (state % 4 + 1) as u32; // strictly increasing
                assert!(col < n_cols, "test column overflow — widen n_cols");
                indices.push(col);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                values_u16.push((state % 7 + 1) as u16);
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, values_u16)
    }

    fn assert_gpu_cpu_decode_match(
        dev: &GpuDevice,
        indptr: &[u64],
        indices: &[u32],
        values_u16: &[u16],
        n_cols: u32,
    ) {
        let n_rows = indptr.len() - 1;
        let nnz = indices.len();
        let index_dtype_u16 = n_cols <= 65535;
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_test_shard(
            indptr,
            indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        let gpu_csr = decode_shard_gpu(dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            index_dtype_u16,
        );

        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }

    /// FOR-BP SIMD-layout (BitPacker4x) GPU-decode parity (Task 4.4b).
    ///
    /// Rows with nnz >= `scx_codec::forbp::SIMD_THRESHOLD` (128) are bit-packed
    /// by the encoder with BitPacker4x's SIMD layout; `forbp_decode_gpu` decodes
    /// them on-device via the BitPacker4x kernel. The synthetic
    /// `test_shard_decode_gpu_scx1` uses only small rows (<=20 nnz) and never
    /// exercises this path; real data (e.g. pbmc3k cells expressing >=128 genes)
    /// does.
    #[test]
    fn test_shard_decode_gpu_scx1_dense_rows_u16() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        // Mix of dense (>=128 nnz -> SIMD path), exactly-threshold, sparse, empty.
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        assert_gpu_cpu_decode_match(&dev, &indptr, &indices, &values_u16, n_cols);
    }

    /// Same SIMD-layout parity for u32 column indices (n_cols > 65535), which
    /// exercises the wider `frame_min` / `frame_bits` path of the BitPacker4x kernel.
    #[test]
    fn test_shard_decode_gpu_scx1_dense_rows_u32() {
        let dev = require_gpu!();
        let n_cols: u32 = 70_000;
        let row_nnzs = [2usize, 130, 300, 0, 512, 9, 256];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        assert_gpu_cpu_decode_match(&dev, &indptr, &indices, &values_u16, n_cols);
    }

    /// Assemble a full shard-section byte buffer (header + payload + block
    /// index) from an [`encode_one_shard`] result, mirroring the on-disk layout
    /// the writer produces. Used to exercise the GPU framed-shard host-bounce.
    fn framed_section_bytes(
        indptr: &[u64],
        indices: &[u32],
        values_f32: &[f32],
        explicit_codec: CodecId,
        n_cols: u32,
        row_group_rows: u32,
    ) -> Vec<u8> {
        use scx_format_io::modality::ModalityType;
        use scx_format_io::section::SectionType;
        let index_dtype = if n_cols <= 65535 { 0u8 } else { 1u8 };
        let framing = scx_format_io::FramingConfig {
            row_group_rows,
            target_nnz: None,
            trial: false,
        };
        let section = scx_format_io::encode_one_shard(
            indptr,
            indices,
            values_f32,
            Some(explicit_codec),
            index_dtype,
            n_cols,
            0,
            SectionType::CsrShard,
            ModalityType::Rna,
            "X_shard_0".to_string(),
            Some(framing),
        )
        .expect("encode_one_shard (framed) failed");
        // Framing must actually have kicked in (shard v2), else the test would
        // vacuously pass on an unframed shard.
        assert_eq!(
            section.shard_format_version(),
            2,
            "expected framed (v2) shard for codec {explicit_codec:?}"
        );
        let mut buf = section.header_buf.clone();
        buf.extend_from_slice(&section.encoded.indptr_bytes);
        buf.extend_from_slice(&section.encoded.indices_bytes);
        buf.extend_from_slice(&section.encoded.values_bytes);
        buf.extend_from_slice(&section.block_index_bytes);
        buf
    }

    /// GPU decode of a **framed (v2)** shard produces output byte-identical to the
    /// source CSR for both codecs, with the expected `fully_device_decoded` flag:
    /// framed **Scx1** decodes fully in VRAM (`true`, only indptr uploaded), while
    /// framed **ShufDeltaZstd** takes the Phase-1 hybrid GPU path (CPU zstd + GPU
    /// undelta/unshuffle/convert) which uploads decompressed plane bytes, so
    /// `false`. (Per-codec GPU-vs-host routing is asserted via the
    /// `n_shards_*` counters in the dedicated shufdelta tests below.) The
    /// multi-group fixture (14 rows, `row_group_rows = 4` → 4 groups) exercises
    /// the per-group concat path.
    #[test]
    fn test_shard_decode_gpu_framed() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let n_rows = indptr.len() - 1;
        let want_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let want_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();

        for codec in [CodecId::ShufDeltaZstd, CodecId::Scx1] {
            // Scx1 always fully device-decodes. Framed ShufDeltaZstd's DEFAULT
            // path is the Phase-1.5 pipeline (decompressed planes → false); the
            // nvcomp Phase-2 path (fully_device_decoded=true) is opt-in and
            // covered by `test_shard_decode_gpu_shufdelta_nvcomp`.
            let want_device = matches!(codec, CodecId::Scx1);
            let section = framed_section_bytes(&indptr, &indices, &values_f32, codec, n_cols, 4);
            let (gpu_csr, stats) = decode_shard_gpu_with_stats(&dev, &section)
                .unwrap_or_else(|e| panic!("framed {codec:?} GPU decode failed: {e}"));
            let g_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
            let g_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
            let g_data = dev.dtoh_copy(&gpu_csr.data).unwrap();
            assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize), "{codec:?} shape");
            assert_eq!(g_indptr, want_indptr, "{codec:?} indptr mismatch");
            assert_eq!(g_indices, want_indices, "{codec:?} indices mismatch");
            assert_eq!(g_data, values_f32, "{codec:?} data mismatch");
            assert_eq!(
                stats.fully_device_decoded, want_device,
                "{codec:?} expected fully_device_decoded={want_device}"
            );
        }
    }

    /// Assert a framed ShufDeltaZstd shard GPU-decodes (Phase 1: CPU zstd + GPU
    /// undelta/unshuffle/convert) byte-identically to the source CSR, and that
    /// it took the shufdelta GPU path (not host-bounce).
    fn assert_framed_shufdelta_matches(
        dev: &GpuDevice,
        indptr: &[u64],
        indices: &[u32],
        values_u16: &[u16],
        n_cols: u32,
        row_group_rows: u32,
    ) {
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let n_rows = indptr.len() - 1;
        let want_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let want_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();

        let section = framed_section_bytes(
            indptr,
            indices,
            &values_f32,
            CodecId::ShufDeltaZstd,
            n_cols,
            row_group_rows,
        );
        let (gpu_csr, stats) = decode_shard_gpu_with_stats(dev, &section)
            .unwrap_or_else(|e| panic!("framed shufdelta GPU decode failed: {e}"));
        let g_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let g_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let g_data = dev.dtoh_copy(&gpu_csr.data).unwrap();
        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize), "shape");
        assert_eq!(g_indptr, want_indptr, "indptr mismatch");
        assert_eq!(g_indices, want_indices, "indices mismatch");
        assert_eq!(g_data, values_f32, "data mismatch");
        assert_eq!(
            stats.n_shards_shufdelta_gpu, 1,
            "expected the ShufDeltaZstd GPU path"
        );
        assert_eq!(stats.n_shards_host_bounced, 0, "should not host-bounce");
        // Default path is the Phase-1.5 pipeline (decompressed planes crossed
        // PCIe), so not a full in-VRAM decode. The opt-in nvcomp Phase-2 path
        // (fully_device_decoded=true) is covered separately.
        assert!(!stats.fully_device_decoded);
    }

    /// Framed ShufDeltaZstd GPU decode, u16 indices (n_cols <= 65535). The
    /// multi-group fixture (14 rows, `row_group_rows = 4`) has groups with
    /// g_nnz up to ~647, exercising the multi-tile undelta prefix-scan carry.
    #[test]
    fn test_shard_decode_gpu_framed_shufdelta_u16() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        assert_framed_shufdelta_matches(&dev, &indptr, &indices, &values_u16, n_cols, 4);
    }

    /// Framed ShufDeltaZstd GPU decode, u32 indices (n_cols > 65535) — 4-plane
    /// undelta + 4-wide unshuffle.
    #[test]
    fn test_shard_decode_gpu_framed_shufdelta_u32() {
        let dev = require_gpu!();
        let n_cols: u32 = 70_000;
        let row_nnzs = [2usize, 130, 300, 0, 512, 9, 256];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        assert_framed_shufdelta_matches(&dev, &indptr, &indices, &values_u16, n_cols, 4);
    }

    /// Large single row-group (many rows collapsed into one group via a big
    /// `row_group_rows`) so a plane's `g_nnz` spans many 256-element tiles,
    /// exercising multi-hop carry propagation in the single-block undelta kernel.
    #[test]
    fn test_shard_decode_gpu_framed_shufdelta_large_group() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs: Vec<usize> = (0..40).map(|i| 80 + (i % 7) * 10).collect();
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        // row_group_rows = 64 > 40 rows → a single group whose plane_len is the
        // whole shard nnz (~3800), i.e. ~15 tiles of 256.
        assert_framed_shufdelta_matches(&dev, &indptr, &indices, &values_u16, n_cols, 64);
    }

    /// Unframed (v1) ShufDeltaZstd GPU decode vs the CPU reference decode.
    #[test]
    fn test_shard_decode_gpu_unframed_shufdelta() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let n_rows = indptr.len() - 1;
        let nnz = indices.len();
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint16,
            n_cols,
        );
        let (gpu_csr, stats) = decode_shard_gpu_with_stats(&dev, &shard_bytes).unwrap();
        let g_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let g_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let g_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            true,
        );

        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(g_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(g_indices, cpu_indices, "indices mismatch");
        assert_eq!(g_data, cpu_data, "data mismatch");
        assert_eq!(
            stats.n_shards_shufdelta_gpu, 1,
            "expected shufdelta GPU path"
        );
        assert_eq!(stats.n_shards_host_bounced, 0);
    }

    /// Float-valued ShufDeltaZstd (zstd-only, no plane transforms) has no GPU
    /// path and must fall back to the host-bounce, still decoding correctly.
    #[test]
    fn test_shard_decode_gpu_framed_shufdelta_float_fallback() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [3usize, 10, 130, 0, 256, 5];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        // Fractional values force a Float32 value encoding.
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32 + 0.5).collect();
        let want_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let want_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();

        let section = framed_section_bytes(
            &indptr,
            &indices,
            &values_f32,
            CodecId::ShufDeltaZstd,
            n_cols,
            4,
        );
        let (gpu_csr, stats) = decode_shard_gpu_with_stats(&dev, &section).unwrap();
        let g_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let g_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let g_data = dev.dtoh_copy(&gpu_csr.data).unwrap();
        assert_eq!(g_indptr, want_indptr, "indptr mismatch");
        assert_eq!(g_indices, want_indices, "indices mismatch");
        assert_eq!(g_data, values_f32, "data mismatch");
        // Float shufdelta has no GPU transform path → host-bounce.
        assert_eq!(
            stats.n_shards_host_bounced, 1,
            "float shufdelta host-bounces"
        );
        assert_eq!(stats.n_shards_shufdelta_gpu, 0);
    }

    /// `DeviceDecodeStats::merge` folds per-codec counters (compact-trial files
    /// mix Scx1 + ShufDeltaZstd shards). Pure CPU — no GPU required.
    #[test]
    fn test_device_decode_stats_merge_counters() {
        let mut agg = DeviceDecodeStats::default();
        let scx1 = DeviceDecodeStats {
            fully_device_decoded: true,
            n_shards_scx1_gpu: 1,
            ..DeviceDecodeStats::default()
        };
        let shuf = DeviceDecodeStats {
            fully_device_decoded: false,
            n_shards_shufdelta_gpu: 1,
            ..DeviceDecodeStats::default()
        };
        agg.merge(&scx1);
        agg.merge(&shuf);
        assert_eq!(agg.n_shards_scx1_gpu, 1);
        assert_eq!(agg.n_shards_shufdelta_gpu, 1);
        assert_eq!(agg.n_shards_host_bounced, 0);
        // A single host-bounced shard ANDs fully_device_decoded down to false.
        assert!(!agg.fully_device_decoded);
    }

    /// Phase 1.5: the pipelined framed ShufDeltaZstd decode (parallel zstd +
    /// copy-stream/event overlap, default for ≥2 groups) is byte-identical to
    /// the sequential (Phase-1) path forced by `SCX_SHUFDELTA_GPU_SEQUENTIAL=1`,
    /// and both match the source CSR. Both take the shufdelta GPU path.
    #[test]
    fn test_shard_decode_gpu_shufdelta_pipeline_matches_sequential() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let want_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let want_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
        // row_group_rows = 4 → 4 groups → pipeline eligible.
        let section = framed_section_bytes(
            &indptr,
            &indices,
            &values_f32,
            CodecId::ShufDeltaZstd,
            n_cols,
            4,
        );

        let decode = || {
            let (csr, stats) = decode_shard_gpu_with_stats(&dev, &section).unwrap();
            (
                dev.dtoh_copy(&csr.indptr).unwrap(),
                dev.dtoh_copy(&csr.indices).unwrap(),
                dev.dtoh_copy(&csr.data).unwrap(),
                stats,
            )
        };

        // Sequential (Phase 1) baseline. env::set_var is safe on edition 2021 and
        // GPU tests run single-threaded (--test-threads=1).
        std::env::set_var("SCX_SHUFDELTA_GPU_SEQUENTIAL", "1");
        let (seq_ip, seq_ix, seq_d, seq_stats) = decode();
        std::env::remove_var("SCX_SHUFDELTA_GPU_SEQUENTIAL");
        // Pipelined (default).
        let (pipe_ip, pipe_ix, pipe_d, pipe_stats) = decode();

        // Both took the shufdelta GPU path (not host-bounce).
        assert_eq!(seq_stats.n_shards_shufdelta_gpu, 1);
        assert_eq!(pipe_stats.n_shards_shufdelta_gpu, 1);
        assert_eq!(seq_stats.n_shards_host_bounced, 0);
        assert_eq!(pipe_stats.n_shards_host_bounced, 0);
        // Both equal the source CSR.
        assert_eq!(seq_ip, want_indptr, "sequential indptr");
        assert_eq!(seq_ix, want_indices, "sequential indices");
        assert_eq!(seq_d, values_f32, "sequential data");
        // Pipelined == sequential, exactly.
        assert_eq!(pipe_ip, seq_ip, "pipelined indptr != sequential");
        assert_eq!(pipe_ix, seq_ix, "pipelined indices != sequential");
        assert_eq!(pipe_d, seq_d, "pipelined data != sequential");
    }

    /// Many small groups (`row_group_rows = 1`, incl. empty rows) exercise the
    /// producer/consumer channel + backpressure + empty-group skipping in the
    /// pipelined path. Output must stay byte-exact vs source.
    #[test]
    fn test_shard_decode_gpu_shufdelta_many_small_groups() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        // row_group_rows = 1 → one group per row (14 groups, two empty).
        assert_framed_shufdelta_matches(&dev, &indptr, &indices, &values_u16, n_cols, 1);
    }

    /// Phase 2 (opt-in nvcomp): with `SCX_SHUFDELTA_NVCOMP=1` the framed decode
    /// takes the full in-VRAM nvcomp path — `fully_device_decoded=true` (only
    /// compressed bytes cross PCIe) — and is byte-identical to the source CSR.
    /// Skipped when nvcomp is not loadable.
    #[test]
    fn test_shard_decode_gpu_shufdelta_nvcomp() {
        let dev = require_gpu!();
        if !crate::nvcomp::nvcomp_available() {
            eprintln!("nvcomp not available — skipping Phase 2 nvcomp path test");
            return;
        }
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let want_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let want_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
        let section = framed_section_bytes(
            &indptr,
            &indices,
            &values_f32,
            CodecId::ShufDeltaZstd,
            n_cols,
            4,
        );

        // env::set_var is safe on edition 2021; GPU tests run --test-threads=1.
        std::env::set_var("SCX_SHUFDELTA_NVCOMP", "1");
        let (gpu_csr, stats) = decode_shard_gpu_with_stats(&dev, &section)
            .unwrap_or_else(|e| panic!("nvcomp framed shufdelta decode failed: {e}"));
        std::env::remove_var("SCX_SHUFDELTA_NVCOMP");

        let g_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let g_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let g_data = dev.dtoh_copy(&gpu_csr.data).unwrap();
        assert_eq!(g_indptr, want_indptr, "nvcomp indptr mismatch");
        assert_eq!(g_indices, want_indices, "nvcomp indices mismatch");
        assert_eq!(g_data, values_f32, "nvcomp data mismatch");
        // Phase 2: only compressed bytes crossed PCIe → full device decode.
        assert!(
            stats.fully_device_decoded,
            "nvcomp path must report fully_device_decoded"
        );
        assert_eq!(stats.n_shards_shufdelta_gpu, 1);
        assert_eq!(stats.n_shards_host_bounced, 0);
    }

    /// GPU-SHUFDELTA-DECODE **Phase 2.x** (cross-shard nvcomp batching): a run of
    /// framed ShufDeltaZstd shards decoded via the batched path
    /// (`decode_csr_shards_to_device_with_stats` with `SCX_SHUFDELTA_NVCOMP=1`)
    /// is byte-identical to (a) the host decode-and-concat and (b) the per-shard
    /// nvcomp path (`SCX_NVCOMP_NO_BATCH=1`), and reports `fully_device_decoded`
    /// with `n_shards_shufdelta_gpu == n_shards`. Skipped when nvcomp is absent.
    #[test]
    fn test_shufdelta_shards_nvcomp_batched_matches_host() {
        let dev = require_gpu!();
        if !crate::nvcomp::nvcomp_available() {
            eprintln!("nvcomp not available — skipping Phase 2.x batched test");
            return;
        }
        let n_cols: u32 = 4000;
        // Three uneven shards, each multi-group (row_group_rows = 4) with empty,
        // sparse, exactly-threshold and dense (>=128 nnz) rows.
        let shard_row_nnzs: [&[usize]; 3] = [
            &[0usize, 1, 5, 130, 256, 7, 0, 384],
            &[200usize, 3, 128, 129],
            &[512usize, 50, 0, 9, 300, 1],
        ];

        // Build the expected global concat (host reference) + the shard sections.
        let mut sections: Vec<Vec<u8>> = Vec::new();
        let mut want_indptr: Vec<i64> = vec![0];
        let mut want_indices: Vec<i32> = Vec::new();
        let mut want_data: Vec<f32> = Vec::new();
        let mut nnz_base: i64 = 0;
        let mut total_rows: usize = 0;
        for row_nnzs in shard_row_nnzs {
            let (indptr, indices, values_u16) = build_dense_csr(row_nnzs, n_cols);
            let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
            for &p in &indptr[1..] {
                want_indptr.push(nnz_base + p as i64);
            }
            want_indices.extend(indices.iter().map(|&v| v as i32));
            want_data.extend_from_slice(&values_f32);
            nnz_base += *indptr.last().unwrap() as i64;
            total_rows += row_nnzs.len();
            sections.push(framed_section_bytes(
                &indptr,
                &indices,
                &values_f32,
                CodecId::ShufDeltaZstd,
                n_cols,
                4,
            ));
        }
        let refs: Vec<&[u8]> = sections.iter().map(|s| s.as_slice()).collect();

        // env::set_var is safe on edition 2021; GPU tests run --test-threads=1.
        std::env::set_var("SCX_SHUFDELTA_NVCOMP", "1");

        // Batched path (default when nvcomp enabled + uniform framed shufdelta).
        std::env::remove_var("SCX_NVCOMP_NO_BATCH");
        let (batched, b_stats) =
            crate::gpu_csr_assemble::decode_csr_shards_to_device_with_stats(&dev, &refs)
                .unwrap_or_else(|e| panic!("batched nvcomp assembly failed: {e}"));

        // Per-shard path (forced) for a byte-exact A/B on the same input.
        std::env::set_var("SCX_NVCOMP_NO_BATCH", "1");
        let (per_shard, p_stats) =
            crate::gpu_csr_assemble::decode_csr_shards_to_device_with_stats(&dev, &refs)
                .unwrap_or_else(|e| panic!("per-shard nvcomp assembly failed: {e}"));
        std::env::remove_var("SCX_NVCOMP_NO_BATCH");
        std::env::remove_var("SCX_SHUFDELTA_NVCOMP");

        let b_indptr = dev.dtoh_copy(&batched.indptr).unwrap();
        let b_indices = dev.dtoh_copy(&batched.indices).unwrap();
        let b_data = dev.dtoh_copy(&batched.data).unwrap();
        assert_eq!(
            batched.shape,
            (total_rows, n_cols as usize),
            "batched shape"
        );
        assert_eq!(b_indptr, want_indptr, "batched indptr vs host");
        assert_eq!(b_indices, want_indices, "batched indices vs host");
        assert_eq!(b_data, want_data, "batched data vs host");

        // Batched == per-shard, element for element.
        assert_eq!(
            b_indptr,
            dev.dtoh_copy(&per_shard.indptr).unwrap(),
            "batched vs per-shard indptr"
        );
        assert_eq!(
            b_indices,
            dev.dtoh_copy(&per_shard.indices).unwrap(),
            "batched vs per-shard indices"
        );
        assert_eq!(
            b_data,
            dev.dtoh_copy(&per_shard.data).unwrap(),
            "batched vs per-shard data"
        );

        // Both paths are full in-VRAM decodes over all three shufdelta shards.
        assert!(b_stats.fully_device_decoded, "batched fully_device_decoded");
        assert!(
            p_stats.fully_device_decoded,
            "per-shard fully_device_decoded"
        );
        assert_eq!(b_stats.n_shards_shufdelta_gpu, 3, "batched shard count");
        assert_eq!(p_stats.n_shards_shufdelta_gpu, 3, "per-shard shard count");
        assert_eq!(b_stats.n_shards_host_bounced, 0);
    }

    /// GPU-SHUFDELTA-DECODE Phase 2 spike (GATE): nvcomp batched GPU zstd decode
    /// of a shard's per-group indices/values frames is **byte-identical** to the
    /// CPU `zstd_decompress_bounded`, and prints GPU vs **parallel** CPU
    /// throughput. Uses a **large frame count** (256 groups) so nvcomp — which
    /// runs ~one thread block per frame — can actually saturate the SM array
    /// (the spec's Open Question #1), and a warm-up call to exclude one-time
    /// nvcomp init from the timing. Skipped when nvcomp is not loadable.
    #[test]
    fn test_nvcomp_zstd_batch_matches_cpu() {
        let dev = require_gpu!();
        if !crate::nvcomp::nvcomp_available() {
            eprintln!("nvcomp not available — skipping Phase 2 spike");
            return;
        }
        // 16384 rows × 1000 nnz, row_group_rows=64 → 256 groups → 256 frames per
        // sub-stream: enough independent frames to fill an H100's SMs.
        let n_cols: u32 = 30000;
        let row_nnzs = vec![1000usize; 16384];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let section = framed_section_bytes(
            &indptr,
            &indices,
            &values_f32,
            CodecId::ShufDeltaZstd,
            n_cols,
            64,
        );

        let header = ShardHeader::read_from(&mut Cursor::new(&section)).unwrap();
        let index_width = if header.index_dtype == 0 { 2usize } else { 4 };
        let value_width = ValueEncoding::from_u8(header.value_encoding)
            .unwrap()
            .byte_width();
        let indices_bytes =
            &section[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_bytes =
            &section[header.values_rel_offset as usize..][..header.values_length as usize];
        let block_index_bytes = &section[header.block_index_rel_offset as usize..]
            [..header.block_index_length as usize];
        let spans = resolve_block_index(&header, block_index_bytes).unwrap();

        let mut idx_frames: Vec<&[u8]> = Vec::new();
        let mut idx_exp: Vec<usize> = Vec::new();
        let mut val_frames: Vec<&[u8]> = Vec::new();
        let mut val_exp: Vec<usize> = Vec::new();
        for s in &spans {
            if s.nnz == 0 {
                continue;
            }
            idx_frames.push(&indices_bytes[s.indices.clone()]);
            idx_exp.push(s.nnz as usize * index_width);
            val_frames.push(&values_bytes[s.values.clone()]);
            val_exp.push(s.nnz as usize * value_width);
        }

        // Warm up nvcomp (one-time module/context init) so it's excluded below.
        let _ = crate::nvcomp::batch_decompress_concat(&dev, dev.stream(), &idx_frames, &idx_exp)
            .unwrap();

        let run = |frames: &[&[u8]], exp: &[usize], label: &str| {
            let comp_bytes: usize = frames.iter().map(|f| f.len()).sum();
            let dec_bytes: usize = exp.iter().sum();

            // GPU nvcomp batched decode (post-warmup).
            let t = std::time::Instant::now();
            let (d_out, offsets) =
                crate::nvcomp::batch_decompress_concat(&dev, dev.stream(), frames, exp).unwrap();
            let gpu_s = t.elapsed().as_secs_f64();
            let out = dev.dtoh_copy(&d_out).unwrap();

            // Byte-exact vs CPU zstd, per frame.
            for (i, f) in frames.iter().enumerate() {
                let cpu = scx_codec::zstd_decompress_bounded(f, exp[i]).unwrap();
                assert_eq!(
                    &out[offsets[i]..offsets[i] + exp[i]],
                    &cpu[..],
                    "{label} frame {i}: nvcomp GPU != CPU zstd"
                );
            }

            // Parallel CPU baseline (matches the Phase-1.5 producer pool).
            let ncpu = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                .min(frames.len());
            let chunk = frames.len().div_ceil(ncpu);
            let pairs: Vec<(&[u8], usize)> =
                frames.iter().copied().zip(exp.iter().copied()).collect();
            let t = std::time::Instant::now();
            std::thread::scope(|s| {
                for c in pairs.chunks(chunk) {
                    s.spawn(move || {
                        for (f, e) in c {
                            let _ = scx_codec::zstd_decompress_bounded(f, *e).unwrap();
                        }
                    });
                }
            });
            let cpu_s = t.elapsed().as_secs_f64();

            eprintln!(
                "[nvcomp spike] {label}: {} frames, avg {:.0} KB compressed, {:.1} MB decompressed | \
                 GPU {:.2} ms ({:.1} GB/s) vs CPU-{}thread {:.2} ms ({:.1} GB/s)",
                frames.len(),
                comp_bytes as f64 / frames.len() as f64 / 1024.0,
                dec_bytes as f64 / 1e6,
                gpu_s * 1e3,
                dec_bytes as f64 / gpu_s / 1e9,
                ncpu,
                cpu_s * 1e3,
                dec_bytes as f64 / cpu_s / 1e9,
            );
        };

        run(&idx_frames, &idx_exp, "indices");
        run(&val_frames, &val_exp, "values");
    }

    /// A framed (v2) Scx1 shard decoded in-VRAM group-by-group is byte-identical
    /// to the same matrix decoded unframed (v1) on the device — proving the
    /// per-group concat + host-assembled indptr equals the whole-shard path.
    #[test]
    fn test_shard_decode_gpu_framed_scx1_matches_unframed() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();

        // Unframed Scx1 (whole-shard device kernel). build_test_shard takes raw
        // value bytes; encode the u16 values little-endian.
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let unframed = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );
        let (u_csr, u_stats) = decode_shard_gpu_with_stats(&dev, &unframed).unwrap();
        assert!(
            u_stats.fully_device_decoded,
            "unframed Scx1 must device-decode"
        );

        // Framed Scx1 (per-group device decode), 4 rows per group → 4 groups.
        let framed = framed_section_bytes(&indptr, &indices, &values_f32, CodecId::Scx1, n_cols, 4);
        let (f_csr, f_stats) = decode_shard_gpu_with_stats(&dev, &framed).unwrap();
        assert!(
            f_stats.fully_device_decoded,
            "framed Scx1 must device-decode"
        );

        assert_eq!(f_csr.shape, u_csr.shape, "shape framed vs unframed");
        assert_eq!(
            dev.dtoh_copy(&f_csr.indptr).unwrap(),
            dev.dtoh_copy(&u_csr.indptr).unwrap(),
            "indptr framed vs unframed"
        );
        assert_eq!(
            dev.dtoh_copy(&f_csr.indices).unwrap(),
            dev.dtoh_copy(&u_csr.indices).unwrap(),
            "indices framed vs unframed"
        );
        assert_eq!(
            dev.dtoh_copy(&f_csr.data).unwrap(),
            dev.dtoh_copy(&u_csr.data).unwrap(),
            "data framed vs unframed"
        );
    }

    /// A framed Scx1 shard with a **fully-empty row-group** (all-zero-nnz rows
    /// within one group) exercises the `g_nnz == 0` branch: the group's indptr is
    /// extended but no device decode / dtod copy runs. Output stays byte-identical.
    #[test]
    fn test_shard_decode_gpu_framed_scx1_empty_group() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        // row_group_rows = 4 → group 0 = 4 empty rows (nnz 0); group 1 = [3, 1].
        let row_nnzs = [0usize, 0, 0, 0, 3, 1];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let n_rows = indptr.len() - 1;

        let section =
            framed_section_bytes(&indptr, &indices, &values_f32, CodecId::Scx1, n_cols, 4);
        let (gpu_csr, stats) = decode_shard_gpu_with_stats(&dev, &section).unwrap();
        assert!(stats.fully_device_decoded, "framed Scx1 must device-decode");
        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(
            dev.dtoh_copy(&gpu_csr.indptr).unwrap(),
            indptr.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            "indptr with an empty group"
        );
        assert_eq!(
            dev.dtoh_copy(&gpu_csr.indices).unwrap(),
            indices.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            "indices with an empty group"
        );
        assert_eq!(
            dev.dtoh_copy(&gpu_csr.data).unwrap(),
            values_f32,
            "data with an empty group"
        );
    }

    #[test]
    fn test_shard_decode_gpu_none() {
        let dev = require_gpu!();

        // Small CSR: 10 rows, u8 values, no compression
        let n_cols: u32 = 50;
        let indptr = vec![0u64, 3, 5, 5, 8, 10, 12, 15, 17, 20, 22];
        let indices = vec![
            0u32, 10, 20, // row 0
            5, 15, // row 1
            // row 2 empty
            1, 2, 3, // row 3
            30, 40, // row 4
            0, 49, // row 5
            10, 20, 30, // row 6
            25, 35, // row 7
            0, 1, 2, // row 8
            48, 49, // row 9
        ];
        let values_u8: Vec<u8> = (1..=22).collect();
        let nnz = indices.len();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_u8,
            CodecId::None,
            ValueEncoding::Uint8,
            n_cols,
        );

        // GPU decode
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        // CPU reference
        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::None,
            ValueEncoding::Uint8,
            10,
            nnz,
            true,
        );

        assert_eq!(gpu_csr.shape, (10, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }

    #[test]
    fn test_shard_decode_gpu_zstd() {
        let dev = require_gpu!();

        // 50 rows, float32 values (Zstd is the only codec for floats)
        let n_rows = 50;
        let n_cols: u32 = 200;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values_f32 = Vec::new();

        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        for _ in 0..n_rows {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let nnz_row = (state % 10 + 1) as usize;
            let mut col = 0u32;
            for _ in 0..nnz_row {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                col += (state % 20 + 1) as u32;
                if col >= n_cols {
                    break;
                }
                indices.push(col);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                values_f32.push((state % 1000) as f32 / 100.0);
            }
            indptr.push(indices.len() as u64);
        }

        let values_raw: Vec<u8> = values_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let nnz = indices.len();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Zstd,
            ValueEncoding::Float32,
            n_cols,
        );

        // GPU decode
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        // CPU reference
        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::Zstd,
            ValueEncoding::Float32,
            n_rows,
            nnz,
            true,
        );

        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }
}
