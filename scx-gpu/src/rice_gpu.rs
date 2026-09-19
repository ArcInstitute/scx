//! GPU-accelerated Rice decoder.
//!
//! Wraps the `rice_decode_kernel` CUDA kernel with a CPU pre-scan stage
//! that extracts per-block byte offsets and k parameters from the encoded
//! bitstream. The GPU then decodes blocks in parallel (one thread per block).

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// Compiled PTX for the Rice decode kernel (produced by build.rs via nvcc --ptx).
const RICE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/rice_decode.ptx"));

/// Pre-scan the Rice-encoded bitstream on CPU to extract per-block metadata.
///
/// Returns `(block_byte_offsets, block_k_values)` where:
/// - `block_byte_offsets[i]` = byte offset where block i's data starts (after the 1-byte header)
/// - `block_k_values[i]` = Rice k parameter for block i
fn prescan_rice_blocks(
    data: &[u8],
    n_values: usize,
    block_size: usize,
) -> Result<(Vec<u32>, Vec<u8>), GpuError> {
    use scx_codec::bitstream::BitReader;

    let n_blocks = n_values.div_ceil(block_size);
    let mut block_offsets = Vec::with_capacity(n_blocks);
    let mut block_k = Vec::with_capacity(n_blocks);
    let mut reader = BitReader::new(data);
    let mut remaining = n_values;

    for _ in 0..n_blocks {
        let block_len = remaining.min(block_size);

        // Read 1-byte header: k in low nibble. Spec §4 reserves the high
        // nibble; the CPU decoder rejects a non-zero one (`scx_codec::rice`)
        // and this prescan used to mask it off, so a corrupt stream decoded to
        // a different answer on the GPU than on the CPU — silently.
        let block_header = reader
            .read_bits(8)
            .map_err(|e| GpuError::InvalidShard(format!("Rice prescan: truncated header: {e}")))?
            as u8;
        if block_header & 0xF0 != 0 {
            return Err(GpuError::InvalidShard(format!(
                "Rice prescan: block header has non-zero reserved high nibble: 0x{block_header:02x}"
            )));
        }
        let k = block_header & 0x0F;

        // Record the byte offset right after the header byte.
        //
        // No ceiling here, unlike FOR-BP's `check_forbp_substream_len`, and the
        // asymmetry is deliberate: FOR-BP stores a *bit* offset, so it multiplies
        // the position by 8 and overflows `u32` at 512 MiB. This stores a *byte*
        // offset, and `data` is a sub-stream of `header.values_length`, a `u32`
        // field — so `bit_pos / 8 <= data.len() <= u32::MAX` and the narrowing
        // below cannot truncate. Same reasoning covers `bitstream_len`.
        let bit_pos = reader.position();
        debug_assert!(
            bit_pos.is_multiple_of(8),
            "block header should leave reader byte-aligned"
        );
        block_offsets.push((bit_pos / 8) as u32);
        block_k.push(k);

        // Skip past this block's encoded values to find the next block boundary.
        // We must decode all values because Rice codes are variable-length —
        // which means the prescan already holds `q` and `r`, so it can apply the
        // CPU's range check for free.
        for _ in 0..block_len {
            let q = reader.read_unary().map_err(|e| {
                GpuError::InvalidShard(format!("Rice prescan: truncated value: {e}"))
            })?;
            let r = if k > 0 {
                reader.read_bits(k).map_err(|e| {
                    GpuError::InvalidShard(format!("Rice prescan: truncated remainder: {e}"))
                })?
            } else {
                0
            };
            // The kernel reconstructs `(q << k) | r` in **32 bits** and adds 1
            // (`kernels/rice_decode.cu`), with none of the CPU's `checked_shl` /
            // `<= u32::MAX` filter. `read_unary` is bounded only by the stream
            // length, so ~16 KiB of set bits gives `q >= 2^17`; at `k = 15` that
            // is `q << k >= 2^32` and the kernel wraps silently in release while
            // the CPU returns `MalformedInput`. Range-check here, where the
            // quotient is already in hand, rather than in 32-bit device code.
            q.checked_shl(k as u32)
                .map(|qk| qk | r)
                .and_then(|shifted| shifted.checked_add(1))
                .filter(|&v| v <= u32::MAX as u64)
                .ok_or_else(|| {
                    GpuError::InvalidShard(
                        "Rice prescan: value overflows u32 (corrupt stream)".to_string(),
                    )
                })?;
        }

        // Align to byte boundary (blocks are byte-aligned)
        reader.align_to_byte();
        remaining -= block_len;
    }

    Ok((block_offsets, block_k))
}

/// Decode Rice-encoded values on GPU.
///
/// Returns a `CudaSlice<u32>` containing `n_values` decoded values (all >= 1).
/// The bitstream is uploaded to GPU, metadata is pre-scanned on CPU,
/// and the kernel is launched with one thread per Rice block (256 values).
///
/// Bit-identical to `scx_codec::rice::rice_decode` for well-formed input, and
/// it now rejects the same malformed input. `prescan_rice_blocks` runs every
/// CPU-side corruption check before the kernel launches: a stream that runs
/// past the encoded data, a non-zero reserved high nibble in a block header,
/// and a value that would overflow `u32` in the kernel's 32-bit
/// `(q << k) | r`. Each returns `Err`, where the first two used to be the
/// CPU's answer only and the third wrapped silently on the device.
pub fn rice_decode_gpu(
    dev: &GpuDevice,
    data: &[u8],
    n_values: usize,
    block_size: usize,
) -> Result<CudaSlice<u32>, GpuError> {
    if n_values == 0 {
        return dev.alloc_zeros::<u32>(0);
    }
    // CPU pre-scan to find block byte offsets and k parameters
    let (block_offsets, block_k) = prescan_rice_blocks(data, n_values, block_size)?;
    rice_decode_gpu_core(dev, data, n_values, block_size, block_offsets, block_k)
}

/// Shared Rice GPU decode core: takes precomputed per-block byte offsets + k
/// parameters (from the CPU prescan) and launches the kernel (one thread per
/// block).
fn rice_decode_gpu_core(
    dev: &GpuDevice,
    data: &[u8],
    n_values: usize,
    block_size: usize,
    block_offsets: Vec<u32>,
    block_k: Vec<u8>,
) -> Result<CudaSlice<u32>, GpuError> {
    // Load PTX module (cached) and get kernel function
    let module = dev.load_module_cached(RICE_PTX)?;
    let kernel = module
        .load_function("rice_decode_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load rice_decode_kernel: {e}")))?;

    let n_blocks = block_offsets.len() as u32;
    let last_block_len = {
        let rem = n_values % block_size;
        if rem == 0 {
            block_size
        } else {
            rem
        }
    } as u32;
    let block_size_u32 = block_size as u32;
    let bitstream_len = data.len() as u32;

    // Upload to GPU
    let d_bitstream = dev.htod_copy(data)?;
    let d_block_offsets = dev.htod_copy(&block_offsets)?;
    let d_block_k = dev.htod_copy(&block_k)?;
    let mut d_output = dev.alloc_zeros::<u32>(n_values)?;

    // Launch: one thread per Rice block, CUDA blocks of 256 threads
    let threads_per_cuda_block = 256u32;
    let grid_dim = n_blocks.div_ceil(threads_per_cuda_block);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (threads_per_cuda_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(&d_bitstream)
            .arg(&d_block_offsets)
            .arg(&d_block_k)
            .arg(&mut d_output)
            .arg(&n_blocks)
            .arg(&block_size_u32)
            .arg(&last_block_len)
            .arg(&bitstream_len)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("rice_decode_kernel: {e}")))?;

    Ok(d_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_codec::bitstream::BitWriter;
    use scx_codec::rice::{rice_decode, rice_encode, B_VAL, MAX_RICE_K};

    /// Spec §4 reserves the block header's high nibble. The CPU decoder rejects
    /// a non-zero one; this prescan masked it off, so the same corrupt shard
    /// produced an answer on the GPU and an error on the CPU.
    ///
    /// Needs no GPU — the prescan is host-side. Until this PR `rice_gpu.rs` had
    /// no CPU-runnable test at all, so every check in it was exercised only by
    /// the one sbatch job that runs the `#[ignore]`d suites.
    #[test]
    fn prescan_rice_rejects_a_reserved_high_nibble() {
        let values = vec![1u32, 2, 3, 4];
        let mut encoded = rice_encode(&values, B_VAL).unwrap();

        // Premise: the untouched stream prescans, so the rejection below is
        // about the nibble and not about the fixture.
        assert!(
            prescan_rice_blocks(&encoded, values.len(), B_VAL).is_ok(),
            "premise: the fixture prescans before corruption"
        );
        assert_eq!(
            encoded[0] & 0xF0,
            0,
            "premise: byte 0 is the block header and its high nibble is clear"
        );

        encoded[0] |= 0xA0;
        let Err(err) = prescan_rice_blocks(&encoded, values.len(), B_VAL) else {
            panic!("a non-zero reserved nibble must be rejected, not masked off");
        };
        // Match the guard's own words: masked off, this stream still decodes
        // happily, so `is_err()` would not distinguish the two behaviours.
        assert!(
            matches!(&err, GpuError::InvalidShard(m) if m.contains("reserved high nibble")),
            "expected the reserved-nibble guard, got {err:?}"
        );
    }

    /// The kernel reconstructs `(q << k) | r` in 32 bits and wraps silently in
    /// release. The CPU range-checks it (`rice_decode_rejects_u32_overflow`);
    /// this is that fixture, driven through the GPU prescan.
    #[test]
    fn prescan_rice_rejects_a_value_that_overflows_u32() {
        let mut w = BitWriter::new();
        w.write_bits(MAX_RICE_K as u64, 8); // block header: k = 15
                                            // q = 2^17 → q << 15 = 2^32, one past u32::MAX.
        w.write_unary(1u64 << 17);
        w.write_bits(0, MAX_RICE_K);
        w.pad_to_byte();
        let data = w.flush();

        // Premise: the CPU rejects this exact stream, so the two arms are being
        // held to one rule rather than to two different ones.
        assert!(
            rice_decode(&data, 1, B_VAL).is_err(),
            "premise: the CPU decoder rejects this fixture"
        );

        let Err(err) = prescan_rice_blocks(&data, 1, B_VAL) else {
            panic!("a value that overflows u32 must be rejected, not wrapped in the kernel");
        };
        assert!(
            matches!(&err, GpuError::InvalidShard(m) if m.contains("overflows u32")),
            "expected the u32 range check, got {err:?}"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rice_gpu_vs_cpu_all_ones() {
        let dev = require_gpu!();
        let values = vec![1u32; 256];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let cpu_decoded = rice_decode(&encoded, 256, B_VAL).unwrap();

        let d_output = rice_decode_gpu(&dev, &encoded, 256, B_VAL).unwrap();
        let gpu_decoded = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(
            gpu_decoded, cpu_decoded,
            "GPU output must be bit-identical to CPU"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rice_gpu_vs_cpu_typical_umi() {
        let dev = require_gpu!();
        // Typical UMI distribution: ~55% ones, geometric tail
        let mut values = Vec::with_capacity(256);
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for _ in 0..256 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let r = (state % 100) as u32;
            let v = if r < 55 {
                1
            } else if r < 75 {
                2
            } else if r < 88 {
                3
            } else if r < 95 {
                (state % 8 + 4) as u32
            } else {
                (state % 50 + 10) as u32
            };
            values.push(v);
        }

        let encoded = rice_encode(&values, B_VAL).unwrap();
        let cpu_decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();

        let d_output = rice_decode_gpu(&dev, &encoded, values.len(), B_VAL).unwrap();
        let gpu_decoded = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_decoded, cpu_decoded);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rice_gpu_vs_cpu_multi_block() {
        let dev = require_gpu!();
        // 600 values = 3 blocks: 256 + 256 + 88
        let mut values = Vec::with_capacity(600);
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        for _ in 0..600 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push((state % 100 + 1) as u32);
        }

        let encoded = rice_encode(&values, B_VAL).unwrap();
        let cpu_decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();

        let d_output = rice_decode_gpu(&dev, &encoded, values.len(), B_VAL).unwrap();
        let gpu_decoded = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_decoded, cpu_decoded);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rice_gpu_vs_cpu_outlier() {
        let dev = require_gpu!();
        // Most values are 1, with large outliers that produce long unary codes
        let mut values = vec![1u32; 256];
        values[100] = 500;
        values[200] = 1000;

        let encoded = rice_encode(&values, B_VAL).unwrap();
        let cpu_decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();

        let d_output = rice_decode_gpu(&dev, &encoded, values.len(), B_VAL).unwrap();
        let gpu_decoded = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_decoded, cpu_decoded);
    }
}
