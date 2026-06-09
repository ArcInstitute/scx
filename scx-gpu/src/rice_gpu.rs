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

        // Read 1-byte header: k in low nibble
        let k = reader
            .read_bits(8)
            .map_err(|e| GpuError::InvalidShard(format!("Rice prescan: truncated header: {e}")))?
            as u8
            & 0x0F;

        // Record the byte offset right after the header byte.
        let bit_pos = reader.position();
        debug_assert!(
            bit_pos.is_multiple_of(8),
            "block header should leave reader byte-aligned"
        );
        block_offsets.push((bit_pos / 8) as u32);
        block_k.push(k);

        // Skip past this block's encoded values to find the next block boundary.
        // We must decode all values because Rice codes are variable-length.
        for _ in 0..block_len {
            let _q = reader.read_unary().map_err(|e| {
                GpuError::InvalidShard(format!("Rice prescan: truncated value: {e}"))
            })?;
            if k > 0 {
                reader.read_bits(k).map_err(|e| {
                    GpuError::InvalidShard(format!("Rice prescan: truncated remainder: {e}"))
                })?;
            }
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
/// Produces **bit-identical** output to `scx_codec::rice::rice_decode`.
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

/// Rice GPU decode driven by an encoder-emitted decode-metadata sidecar
/// ([`scx_codec::rice::RiceBlockMetadata`]) instead of the CPU
/// `prescan_rice_blocks` pass. The per-block `bit_offset` (to the 1-byte block
/// header) and `k` are exactly what the kernel needs, so the full CPU scan of
/// the bitstream (which otherwise must decode every value to find block
/// boundaries) is skipped. Output is **bit-identical** to [`rice_decode_gpu`].
pub fn rice_decode_gpu_with_metadata(
    dev: &GpuDevice,
    data: &[u8],
    blocks: &[scx_codec::rice::RiceBlockMetadata],
    n_values: usize,
    block_size: usize,
) -> Result<CudaSlice<u32>, GpuError> {
    if n_values == 0 {
        return dev.alloc_zeros::<u32>(0);
    }
    let expected = n_values.div_ceil(block_size);
    if blocks.len() != expected {
        return Err(GpuError::InvalidShard(format!(
            "Rice sidecar: blocks.len() {} != expected {expected} for {n_values} values",
            blocks.len()
        )));
    }
    let mut block_offsets = Vec::with_capacity(blocks.len());
    let mut block_k = Vec::with_capacity(blocks.len());
    for b in blocks {
        if !b.bit_offset.is_multiple_of(8) {
            return Err(GpuError::InvalidShard(
                "Rice sidecar: block bit_offset is not byte-aligned".into(),
            ));
        }
        // Sidecar bit_offset points at the 1-byte block header; the kernel wants
        // the byte offset of the block's data (just past that header).
        let byte_off = b.bit_offset / 8 + 1;
        block_offsets.push(u32::try_from(byte_off).map_err(|_| {
            GpuError::InvalidShard(format!(
                "Rice sidecar: block byte offset {byte_off} exceeds u32"
            ))
        })?);
        block_k.push(b.k);
    }
    rice_decode_gpu_core(dev, data, n_values, block_size, block_offsets, block_k)
}

/// Shared Rice GPU decode core: takes precomputed per-block byte offsets + k
/// parameters (from either the CPU prescan or a decode sidecar) and launches the
/// kernel (one thread per block).
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
    use scx_codec::rice::{rice_decode, rice_encode, B_VAL};

    #[test]
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
