//! GPU CSR → dense conversion with optional HVG (gene subset) projection.
//!
//! Wraps the `sparse_to_dense_kernel` CUDA kernel. The kernel uses warp-
//! cooperative row scatter: each warp (32 threads) handles one row's
//! non-zeros in parallel, with optional column remapping for HVG projection.

use cudarc::driver::safe::{CudaSlice, CudaView};
use cudarc::driver::PushKernelArg;

use crate::device::{flat_launch_1d, GpuDevice};
use crate::error::GpuError;
use crate::shard_decode::GpuCsr;

/// Compiled PTX for the sparse-to-dense kernel (produced by build.rs via nvcc --ptx).
const SPARSE_DENSE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sparse_dense.ptx"));

/// Sentinel value in an HVG projection map marking an input column that is
/// dropped (not present in the selected gene subset). Matches the `0xFFFFFFFF`
/// skip code recognised by `sparse_to_dense_kernel`.
pub const HVG_MAP_SKIP: u32 = 0xFFFF_FFFF;

/// Validate an HVG projection map for use with [`sparse_to_dense_gpu`] and
/// friends. `host_map[col]` is the output column for input column `col`, or
/// [`HVG_MAP_SKIP`] to drop that column.
///
/// `n_input_cols` is the number of input (original) columns the map must cover
/// — i.e. `gpu_csr.shape.1`. The kernel indexes `hvg_map[col]` for every CSR
/// column `col ∈ [0, n_input_cols)`, so a map shorter than this would make the
/// kernel read past the device allocation; the map length is required to match.
///
/// Returns an error if the map length `!= n_input_cols`, if any non-skip entry
/// is `>= n_output_cols`, or if two input columns map to the same output column.
/// The kernel scatters one nonzero per warp lane with a plain (non-atomic)
/// store, so a many-to-one map races on the shared output cell with an undefined
/// winner — see `sparse_dense.cu`. This is a pure host-side check (no device
/// interaction); its allocation is bounded by `host_map.len()`, not by the
/// caller-supplied `n_output_cols` (which could be arbitrarily large on corrupt
/// input).
pub fn validate_hvg_map(
    host_map: &[u32],
    n_output_cols: usize,
    n_input_cols: usize,
) -> Result<(), GpuError> {
    if host_map.len() != n_input_cols {
        return Err(GpuError::KernelLaunchFailed(format!(
            "validate_hvg_map: map length {} != n_input_cols ({}); the kernel \
             reads hvg_map[col] for every input column and would read out of bounds",
            host_map.len(),
            n_input_cols
        )));
    }
    // Collect the mapped (non-skip) output columns, checking the range bound as
    // we go, then sort + scan for duplicates to detect a non-injective map. The
    // scratch is bounded by the number of mapped columns (`<= host_map.len()`),
    // never by `n_output_cols` — so a corrupt, huge `n_output_cols` cannot drive
    // an out-of-memory allocation here.
    let mut mapped: Vec<u32> = Vec::with_capacity(host_map.len());
    for (col, &m) in host_map.iter().enumerate() {
        if m == HVG_MAP_SKIP {
            continue;
        }
        if m as usize >= n_output_cols {
            return Err(GpuError::KernelLaunchFailed(format!(
                "validate_hvg_map: input column {col} maps to output column {m} \
                 but n_output_cols == {n_output_cols}"
            )));
        }
        mapped.push(m);
    }
    mapped.sort_unstable();
    for w in mapped.windows(2) {
        if w[0] == w[1] {
            return Err(GpuError::KernelLaunchFailed(format!(
                "validate_hvg_map: non-injective map — output column {} is \
                 written by more than one input column (kernel scatter would race)",
                w[0]
            )));
        }
    }
    Ok(())
}

/// Validate ([`validate_hvg_map`]) and upload an HVG projection map to the
/// device. This is the sanctioned way to build the `hvg_map` argument for
/// [`sparse_to_dense_gpu`] / [`sparse_to_dense_gpu_into`] /
/// [`sparse_to_dense_gpu_into_view`]: it guarantees the injectivity and input
/// coverage the kernel requires, failing with an error instead of producing a
/// silently corrupted dense matrix (or an out-of-bounds device read) on a
/// malformed map. `n_input_cols` is the CSR's column count (`gpu_csr.shape.1`).
pub fn upload_hvg_map(
    device: &GpuDevice,
    host_map: &[u32],
    n_output_cols: usize,
    n_input_cols: usize,
) -> Result<CudaSlice<u32>, GpuError> {
    validate_hvg_map(host_map, n_output_cols, n_input_cols)?;
    device
        .htod_copy(host_map)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("upload_hvg_map: htod copy: {e}")))
}

/// Convert a GPU-resident CSR matrix to a dense row-major matrix, allocating
/// a fresh zero-initialised output buffer.
///
/// # Arguments
///
/// * `device` — GPU device handle.
/// * `gpu_csr` — GPU-resident CSR matrix (i64 indptr, i32 indices, f32 data).
/// * `hvg_map` — Optional gene projection map on device. `hvg_map[original_col]`
///   gives the output column index, or [`HVG_MAP_SKIP`] to skip. If `None`,
///   identity mapping is used (all columns, `n_output_cols == n_cols`).
///   **Contract:** the map MUST be injective over its non-skip entries (each
///   output column written by at most one input column) — the kernel's
///   warp-cooperative scatter races on a many-to-one map. Build the device map
///   with [`upload_hvg_map`], which validates this.
/// * `n_output_cols` — Number of columns in the dense output. Must equal
///   `gpu_csr.shape.1` when `hvg_map` is `None`, or the number of selected
///   genes when projecting.
///
/// # Returns
///
/// A `CudaSlice<f32>` of size `n_rows × n_output_cols`, row-major, zero-filled
/// for absent entries.
///
/// Thin wrapper around [`sparse_to_dense_gpu_into`] for one-shot callers that
/// don't reuse a scratch buffer. Loop callers should hoist the buffer and call
/// [`sparse_to_dense_gpu_into`] directly.
pub fn sparse_to_dense_gpu(
    device: &GpuDevice,
    gpu_csr: &GpuCsr,
    hvg_map: Option<&CudaSlice<u32>>,
    n_output_cols: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let (n_rows, _n_cols) = gpu_csr.shape;

    if n_rows == 0 || n_output_cols == 0 {
        return device.alloc_zeros::<f32>(0);
    }

    let mut d_output = device.alloc_zeros::<f32>(n_rows * n_output_cols)?;
    sparse_to_dense_gpu_into(device, gpu_csr, hvg_map, n_output_cols, &mut d_output)?;
    Ok(d_output)
}

/// Convert a GPU-resident CSR matrix to a dense row-major matrix, writing into
/// a caller-supplied scratch buffer instead of allocating.
///
/// # Buffer contract
///
/// `scratch` must satisfy `scratch.len() >= n_rows * n_output_cols`. The
/// kernel only writes positions corresponding to nonzeros (and, with
/// `hvg_map`, mapped output columns); empty rows and unmapped columns are
/// left **unchanged**. Callers who reuse the same scratch across multiple
/// CSRs must therefore zero the leading `n_rows * n_output_cols` region
/// before each call to avoid carrying over stale values from a previous
/// shard. `cudarc::driver::safe::CudaStream::memset_zeros` on a
/// `slice_mut(0..n_rows * n_output_cols)` view is the cheapest way to do
/// this.
///
/// Allocator-allocated buffers (e.g. via `alloc_zeros`) are zero-filled by
/// construction; the first call after allocation does not need an explicit
/// zero pass.
///
/// # Arguments
///
/// Same as [`sparse_to_dense_gpu`], plus `scratch` — the caller-owned output
/// buffer.
pub fn sparse_to_dense_gpu_into(
    device: &GpuDevice,
    gpu_csr: &GpuCsr,
    hvg_map: Option<&CudaSlice<u32>>,
    n_output_cols: usize,
    scratch: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let (n_rows, _n_cols) = gpu_csr.shape;
    if n_rows == 0 {
        return Ok(());
    }
    let nnz = gpu_csr.data.len();
    sparse_to_dense_gpu_into_view(
        device,
        &gpu_csr.indptr.slice(..n_rows + 1),
        &gpu_csr.indices.slice(..nnz),
        &gpu_csr.data.slice(..nnz),
        n_rows,
        hvg_map,
        n_output_cols,
        scratch,
    )
}

/// View-based variant of [`sparse_to_dense_gpu_into`].
///
/// Identical kernel; takes borrowed CSR component views (`indptr`,
/// `indices`, `data`) plus an explicit `n_rows` instead of an owned
/// [`GpuCsr`]. This is the path used by consumers that iterate a reusable
/// [`GpuCsrSlot`](crate::staging::GpuCsrSlot) (via
/// [`GpuMatrixSource`](crate::gpu_matrix_source::GpuMatrixSource)) rather than
/// owning a `GpuCsr` per shard. The same buffer contract as
/// [`sparse_to_dense_gpu_into`] applies to `scratch`.
#[allow(clippy::too_many_arguments)]
pub fn sparse_to_dense_gpu_into_view(
    device: &GpuDevice,
    indptr: &CudaView<'_, i64>,
    indices: &CudaView<'_, i32>,
    data: &CudaView<'_, f32>,
    n_rows: usize,
    hvg_map: Option<&CudaSlice<u32>>,
    n_output_cols: usize,
    scratch: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    if n_rows == 0 || n_output_cols == 0 {
        return Ok(());
    }

    let needed = n_rows * n_output_cols;
    if scratch.len() < needed {
        return Err(GpuError::KernelLaunchFailed(format!(
            "sparse_to_dense_gpu_into: scratch buffer too small ({} < {})",
            scratch.len(),
            needed
        )));
    }

    // Load PTX module (cached) and get kernel function
    let module = device.load_module_cached(SPARSE_DENSE_PTX)?;
    let kernel = module
        .load_function("sparse_to_dense_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load sparse_to_dense_kernel: {e}")))?;

    let n_rows_i32 = n_rows as i32;
    let n_output_cols_i32 = n_output_cols as i32;

    // Launch config: one warp (32 threads) per row.
    // CUDA block size = 256 threads = 8 warps per block.
    //
    // Sized as a flat 32-threads-per-row launch, which yields the identical
    // block count as `n_rows / warps_per_block` while counting in u64 and
    // rejecting a grid past the `grid_dim.x` cap instead of truncating it.
    let threads_per_block: u32 = 256;
    let cfg = flat_launch_1d(n_rows as u64 * 32, threads_per_block)?;

    // Build kernel argument for hvg_map (null pointer if None).
    // cudarc expects all arguments to be pushed in order; for a nullable pointer
    // we pass either the device slice or a raw null u64.
    match hvg_map {
        Some(hmap) => unsafe {
            device
                .stream()
                .launch_builder(&kernel)
                .arg(indptr)
                .arg(indices)
                .arg(data)
                .arg(scratch)
                .arg(hmap)
                .arg(&n_rows_i32)
                .arg(&n_output_cols_i32)
                .launch(cfg)
        },
        None => {
            // Pass a null device pointer for hvg_map.
            let null_ptr: u64 = 0;
            unsafe {
                device
                    .stream()
                    .launch_builder(&kernel)
                    .arg(indptr)
                    .arg(indices)
                    .arg(data)
                    .arg(scratch)
                    .arg(&null_ptr)
                    .arg(&n_rows_i32)
                    .arg(&n_output_cols_i32)
                    .launch(cfg)
            }
        }
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("sparse_to_dense_kernel: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_decode::decode_shard_gpu;
    use crate::test_utils::build_test_shard;
    use scx_codec::{CodecId, ValueEncoding};

    /// CPU reference: scatter CSR arrays into a dense matrix.
    fn cpu_sparse_to_dense(
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> Vec<f32> {
        let mut dense = vec![0.0f32; n_rows * n_cols];
        for row in 0..n_rows {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for nz in start..end {
                let col = indices[nz] as usize;
                if col < n_cols {
                    dense[row * n_cols + col] = data[nz];
                }
            }
        }
        dense
    }

    /// CPU reference: scatter CSR arrays into a dense matrix with HVG projection.
    fn cpu_sparse_to_dense_hvg(
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        n_rows: usize,
        hvg_map: &[u32],
        n_output_cols: usize,
    ) -> Vec<f32> {
        let mut dense = vec![0.0f32; n_rows * n_output_cols];
        for row in 0..n_rows {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for nz in start..end {
                let col = indices[nz] as usize;
                if col < hvg_map.len() {
                    let mapped = hvg_map[col];
                    if mapped != 0xFFFFFFFF && (mapped as usize) < n_output_cols {
                        dense[row * n_output_cols + mapped as usize] = data[nz];
                    }
                }
            }
        }
        dense
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_sparse_to_dense_gpu_matches_cpu() {
        let dev = require_gpu!();

        // Build a 5-row, 100-col CSR shard
        let n_cols: u32 = 100;
        let indptr = vec![0u64, 3, 5, 5, 8, 12];
        let indices = vec![
            0u32, 10, 50, // row 0
            5, 15, // row 1
            // row 2 empty
            1, 2, 3, // row 3
            20, 40, 60, 80, // row 4
        ];
        let values_u16: Vec<u16> = (1..=12).collect();
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        // GPU decode to GpuCsr
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        assert_eq!(gpu_csr.shape, (5, 100));

        // GPU sparse → dense (no HVG)
        let d_dense = sparse_to_dense_gpu(&dev, &gpu_csr, None, n_cols as usize).unwrap();
        let gpu_dense = dev.dtoh_copy(&d_dense).unwrap();

        // CPU reference
        let cpu_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let cpu_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
        let cpu_data: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let cpu_dense = cpu_sparse_to_dense(&cpu_indptr, &cpu_indices, &cpu_data, 5, 100);

        assert_eq!(gpu_dense.len(), cpu_dense.len(), "output size mismatch");
        assert_eq!(
            gpu_dense, cpu_dense,
            "GPU dense output must match CPU reference"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_sparse_to_dense_with_hvg() {
        let dev = require_gpu!();

        // Same 5-row, 100-col CSR
        let n_cols: u32 = 100;
        let indptr = vec![0u64, 3, 5, 5, 8, 12];
        let indices = vec![
            0u32, 10, 50, // row 0
            5, 15, // row 1
            // row 2 empty
            1, 2, 3, // row 3
            20, 40, 60, 80, // row 4
        ];
        let values_u16: Vec<u16> = (1..=12).collect();
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        // GPU decode to GpuCsr
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();

        // HVG subset: keep only columns 0, 2, 10, 50, 80 → output cols 0..4
        // Build a mapping: hvg_map[col] = output_col or 0xFFFFFFFF
        let selected_cols: Vec<u32> = vec![0, 2, 10, 50, 80];
        let n_output_cols = selected_cols.len();
        let mut hvg_map = vec![HVG_MAP_SKIP; n_cols as usize];
        for (out_col, &orig_col) in selected_cols.iter().enumerate() {
            hvg_map[orig_col as usize] = out_col as u32;
        }
        let d_hvg_map = upload_hvg_map(&dev, &hvg_map, n_output_cols, n_cols as usize).unwrap();

        // GPU sparse → dense with HVG
        let d_dense = sparse_to_dense_gpu(&dev, &gpu_csr, Some(&d_hvg_map), n_output_cols).unwrap();
        let gpu_dense = dev.dtoh_copy(&d_dense).unwrap();

        // CPU reference
        let cpu_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let cpu_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
        let cpu_data: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let cpu_dense = cpu_sparse_to_dense_hvg(
            &cpu_indptr,
            &cpu_indices,
            &cpu_data,
            5,
            &hvg_map,
            n_output_cols,
        );

        assert_eq!(
            gpu_dense.len(),
            5 * n_output_cols,
            "output size should be n_rows × n_output_cols"
        );
        assert_eq!(gpu_dense, cpu_dense, "HVG-projected dense must match CPU");

        // Verify specific values for row 0 (so the row-major index reduces
        // to a plain column index — leaving the explicit `0 * n_output_cols`
        // here trips `clippy::erasing_op` on toolchain ≥ 1.94).
        // Row 0 has col=0 (val=1.0), col=10 (val=2.0), col=50 (val=3.0)
        // Mapped: col 0→out 0, col 10→out 2, col 50→out 3
        assert_eq!(gpu_dense[0], 1.0, "row0 col0");
        assert_eq!(gpu_dense[2], 2.0, "row0 col10→out2");
        assert_eq!(gpu_dense[3], 3.0, "row0 col50→out3");
        // Unmapped columns should be zero
        assert_eq!(gpu_dense[1], 0.0, "row0 unmapped col2");
    }

    // Host-only: validation runs before any device interaction, so these need
    // no GPU.
    #[test]
    fn test_validate_hvg_map_accepts_injective() {
        // cols 0,2,5 → out 0,1,2; everything else skipped. 10 input columns.
        let mut map = vec![HVG_MAP_SKIP; 10];
        map[0] = 0;
        map[2] = 1;
        map[5] = 2;
        assert!(validate_hvg_map(&map, 3, 10).is_ok());
    }

    #[test]
    fn test_validate_hvg_map_rejects_non_injective() {
        // Two input columns map to the same output column 0 → race.
        let mut map = vec![HVG_MAP_SKIP; 10];
        map[1] = 0;
        map[4] = 0;
        let err = validate_hvg_map(&map, 2, 10).unwrap_err();
        assert!(
            matches!(err, GpuError::KernelLaunchFailed(ref m) if m.contains("non-injective")),
            "expected non-injective error, got {err:?}"
        );
    }

    #[test]
    fn test_validate_hvg_map_rejects_out_of_range() {
        let mut map = vec![HVG_MAP_SKIP; 10];
        map[3] = 5; // 5 >= n_output_cols (2)
        let err = validate_hvg_map(&map, 2, 10).unwrap_err();
        assert!(
            matches!(err, GpuError::KernelLaunchFailed(ref m) if m.contains("n_output_cols")),
            "expected out-of-range error, got {err:?}"
        );
    }

    #[test]
    fn test_validate_hvg_map_rejects_short_map() {
        // Map shorter than the input column count → the kernel would read
        // hvg_map[col] past the device allocation for col >= map.len().
        let map = vec![HVG_MAP_SKIP; 8];
        let err = validate_hvg_map(&map, 4, 10).unwrap_err();
        assert!(
            matches!(err, GpuError::KernelLaunchFailed(ref m) if m.contains("n_input_cols")),
            "expected coverage error, got {err:?}"
        );
    }
}
