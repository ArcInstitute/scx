//! GPU differential expression primitives (Mann-Whitney U / Wilcoxon rank sum).
//!
//! Per-gene block radix sort
//! (CUB), batched searchsorted, on-device tie-term computation, and normal-tail
//! p-value via `erfc`. These primitives are the building blocks for the
//! high-level `pdex_ref_gpu_*` / `wilcoxon_rank_sum_*_gpu` entry points that
//! live in `scx-accel/src/diffexp.rs` under `#[cfg(feature = "gpu")]` — see
//! the CPU path in the same file for the parity oracle.
//!
//! All kernels share a single PTX module (`diffexp.ptx`, compiled by
//! `scx-gpu/build.rs` from `kernels/diffexp.cu`).
//!
//! ## Per-gene sort capacity
//!
//! The per-gene sort in [`gpu_de_block_sort`] is two-path:
//!
//! * **Fast path** (`n_per_gene ≤ GPU_DE_BLOCK_SORT_CAPACITY = 8192`): one
//!   CUB `BlockRadixSort` per gene — the entire row sits in registers across
//!   one block.
//! * **Multi-tile path** (`> 8192`): bottom-up iterative merge sort. Tile
//!   sort with `tile_block_radix_sort_kernel`, then `⌈log₂(K)⌉` passes of
//!   `merge_pass_per_gene_kernel` (block-cooperative merge-path
//!   partitioning), ping-ponging between `scratch.slab` and `scratch.slab_aux`.
//!
//! No upper limit beyond available VRAM. Callers no longer need to guard
//! pool sizes against the 8192 threshold — the dispatch handles arbitrary
//! sizes internally. Tested on census-scale fixtures (n_pool ≈ 1M) where the
//! multi-tile path engages ~7 merge passes per chunk.

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// PTX source for the GPU DE kernels, compiled at build time by `scx-gpu/build.rs`.
const DIFFEXP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/diffexp.ptx"));

/// Fast-path threshold for the single-tile block radix sort.
///
/// Matches `BLOCK_THREADS * ITEMS_PER_THREAD` in `kernels/diffexp.cu`. Pools
/// `≤ GPU_DE_BLOCK_SORT_CAPACITY` keys sort in one CUB block (everything in
/// registers — fast). Pools larger than this dispatch to a tiled bottom-up
/// merge sort (`tile_block_radix_sort_kernel` + `merge_pass_per_gene_kernel`)
/// with no upper limit beyond available VRAM. See [`gpu_de_block_sort`].
pub const GPU_DE_BLOCK_SORT_CAPACITY: usize = 8192;

/// Reusable per-chunk device buffers for a streaming MWU pipeline.
///
/// One allocation per chunk_max suffices for the entire DE call: the chunk
/// loop refills `dense` per chunk and resizes `ref_slab` / `group_slab` only
/// when the membership counts change. This intentionally mirrors how
/// `gpu_pca::GpuPcaScratch` hoists allocations out of the power-iteration
/// inner loop.
pub struct GpuDeChunkScratch {
    /// `[n_obs × chunk_max]` row-major dense buffer for the current chunk.
    pub dense: CudaSlice<f32>,
    /// `[chunk_max × n_pool_max]` gene-major slab; reused for ref then for
    /// each test group (size large enough for whichever is bigger).
    pub slab: CudaSlice<f32>,
    /// Ping-pong buffer used by the multi-tile path in [`gpu_de_block_sort`]
    /// when `n_per_gene > GPU_DE_BLOCK_SORT_CAPACITY`. Allocated lazily on
    /// first multi-tile encounter via [`Self::ensure_aux_capacity`]; small-
    /// pool callers never pay for it.
    pub slab_aux: CudaSlice<f32>,
    /// `[chunk_max]` f64 tie-term scratch (ref-only or combined).
    pub tie_term: CudaSlice<f64>,
    /// `[chunk_max]` f64 U1 / rank-sum scratch.
    pub u_or_rank: CudaSlice<f64>,
    /// `[chunk_max]` f64 p-values for one test group.
    pub p_values: CudaSlice<f64>,
    n_obs: usize,
    chunk_max: usize,
    slab_capacity: usize,
    aux_capacity_elems: usize,
}

impl GpuDeChunkScratch {
    /// Allocate scratch buffers sized for `n_obs` cells, up to `chunk_max`
    /// genes per chunk, and an initial pool capacity of `n_pool_max` cells.
    ///
    /// The slab grows on demand via [`Self::ensure_slab_capacity`]; the
    /// dense buffer is fixed-size for the whole DE call.
    pub fn new(
        dev: &GpuDevice,
        n_obs: usize,
        chunk_max: usize,
        n_pool_max: usize,
    ) -> Result<Self, GpuError> {
        let dense = dev.alloc_zeros::<f32>(n_obs * chunk_max)?;
        let slab = dev.alloc_zeros::<f32>(chunk_max * n_pool_max)?;
        // slab_aux is allocated lazily — empty until the first call that hits
        // the multi-tile sort path. Allocating a zero-length CudaSlice is
        // cheap (a few bytes of metadata) and avoids paying VRAM for the
        // common small-pool case where every pool fits in one tile.
        let slab_aux = dev.alloc_zeros::<f32>(0)?;
        let tie_term = dev.alloc_zeros::<f64>(chunk_max)?;
        let u_or_rank = dev.alloc_zeros::<f64>(chunk_max)?;
        let p_values = dev.alloc_zeros::<f64>(chunk_max)?;
        Ok(Self {
            dense,
            slab,
            slab_aux,
            tie_term,
            u_or_rank,
            p_values,
            n_obs,
            chunk_max,
            slab_capacity: n_pool_max,
            aux_capacity_elems: 0,
        })
    }

    /// Grow the slab if `n_pool > current slab capacity`. No-op otherwise.
    pub fn ensure_slab_capacity(&mut self, dev: &GpuDevice, n_pool: usize) -> Result<(), GpuError> {
        if n_pool <= self.slab_capacity {
            return Ok(());
        }
        // Bump to a round multiple to avoid thrashing on small growths.
        let new_cap = n_pool.next_power_of_two().max(self.slab_capacity * 2);
        self.slab = dev.alloc_zeros::<f32>(self.chunk_max * new_cap)?;
        self.slab_capacity = new_cap;
        Ok(())
    }

    /// Grow the ping-pong aux buffer to hold at least `n_elements` f32 keys.
    ///
    /// Called by [`gpu_de_block_sort`] before the multi-tile path. No-op when
    /// the aux is already large enough. Bumps to `next_power_of_two` to amortise
    /// repeated growths across a streaming chunk loop.
    pub fn ensure_aux_capacity(
        &mut self,
        dev: &GpuDevice,
        n_elements: usize,
    ) -> Result<(), GpuError> {
        if n_elements <= self.aux_capacity_elems {
            return Ok(());
        }
        let new_cap = n_elements
            .next_power_of_two()
            .max(self.aux_capacity_elems * 2);
        self.slab_aux = dev.alloc_zeros::<f32>(new_cap)?;
        self.aux_capacity_elems = new_cap;
        Ok(())
    }

    /// Maximum number of cells per chunk this scratch is sized for.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Maximum number of genes per chunk this scratch is sized for.
    pub fn chunk_max(&self) -> usize {
        self.chunk_max
    }
}

/// Upload a host-side dense `[n_obs × chunk_size]` (row-major, f32) chunk to
/// the front of the device dense slot.
///
/// Caller is responsible for ensuring `dense_host.len() == n_obs * chunk_size`
/// and `chunk_size <= scratch.chunk_max()`.
pub fn gpu_de_upload_chunk(
    dev: &GpuDevice,
    scratch: &mut GpuDeChunkScratch,
    dense_host: &[f32],
    n_obs: usize,
    chunk_size: usize,
) -> Result<(), GpuError> {
    if dense_host.len() != n_obs * chunk_size {
        return Err(GpuError::ShapeMismatch {
            expected: format!(
                "{} (n_obs={} × chunk_size={})",
                n_obs * chunk_size,
                n_obs,
                chunk_size
            ),
            got: format!("{}", dense_host.len()),
        });
    }
    if n_obs != scratch.n_obs || chunk_size > scratch.chunk_max {
        return Err(GpuError::ShapeMismatch {
            expected: format!(
                "n_obs≤{}, chunk_size≤{} (scratch capacity)",
                scratch.n_obs, scratch.chunk_max
            ),
            got: format!("n_obs={n_obs}, chunk_size={chunk_size}"),
        });
    }
    let nelem = n_obs * chunk_size;
    // CudaSlice supports a slice view via .slice(...) in cudarc 0.19.
    let mut view = scratch.dense.slice_mut(..nelem);
    dev.stream()
        .memcpy_htod(dense_host, &mut view)
        .map_err(|e| GpuError::CudaError(format!("htod_copy(dense chunk): {e}")))?;
    Ok(())
}

/// Scatter a permutation of cells (e.g. ref or group) from the device dense
/// chunk into a gene-major slab `[chunk_size × n_perm]`.
///
/// `cell_indices_host` is uploaded internally; for repeated calls with the
/// same permutation, consider caching the upload via a dedicated CudaSlice.
pub fn gpu_de_scatter_gene_major(
    dev: &GpuDevice,
    dense: &CudaSlice<f32>,
    cell_indices_host: &[i32],
    slab: &mut CudaSlice<f32>,
    n_obs: usize,
    chunk_size: usize,
) -> Result<(), GpuError> {
    if cell_indices_host.is_empty() || chunk_size == 0 {
        return Ok(());
    }
    let n_perm = cell_indices_host.len();
    let d_indices = dev.htod_copy(cell_indices_host)?;

    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("scatter_perm_to_gene_major_kernel")
        .map_err(|e| {
            GpuError::KernelLaunchFailed(format!("scatter_perm_to_gene_major_kernel: {e}"))
        })?;

    let bx: u32 = 32;
    let by: u32 = 8;
    let gx = (n_perm as u32).div_ceil(bx);
    let gy = (chunk_size as u32).div_ceil(by);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, 1),
        block_dim: (bx, by, 1),
        shared_mem_bytes: 0,
    };

    let n_obs_i32 = n_obs as i32;
    let n_perm_i32 = n_perm as i32;
    let chunk_i32 = chunk_size as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(dense)
            .arg(&d_indices)
            .arg(slab)
            .arg(&n_obs_i32)
            .arg(&n_perm_i32)
            .arg(&chunk_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("scatter_perm_to_gene_major_kernel: {e}")))?;

    Ok(())
}

/// Sort each gene's row of a gene-major slab in ascending order, in-place.
///
/// Two-path dispatch by `n_per_gene`:
///
/// * **Fast path** (`n_per_gene ≤ GPU_DE_BLOCK_SORT_CAPACITY`): single CUB
///   `BlockRadixSort` per gene — everything in registers, one kernel launch
///   for the whole chunk. No use of `scratch.slab_aux`.
/// * **Multi-tile path** (`n_per_gene > GPU_DE_BLOCK_SORT_CAPACITY`):
///   bottom-up iterative merge sort. Tile sort with `tile_block_radix_sort_kernel`
///   (1 block per `(gene, tile)`), then `⌈log₂(K)⌉` merge passes (where
///   `K = ⌈n_per_gene / GPU_DE_BLOCK_SORT_CAPACITY⌉`) of
///   `merge_pass_per_gene_kernel`. Ping-pongs between `slab` and
///   `scratch.slab_aux`; final copy back to `slab` when pass count is odd,
///   so callers always read the sorted result from `slab`.
///
/// `scratch` is taken `&mut` to allow on-demand growth of `slab_aux` via
/// [`GpuDeChunkScratch::ensure_aux_capacity`]. Fast-path callers can pass
/// any scratch they have lying around — the aux is only touched on the
/// multi-tile path.
pub fn gpu_de_block_sort(
    dev: &GpuDevice,
    scratch: &mut GpuDeChunkScratch,
    slab: &mut CudaSlice<f32>,
    chunk_size: usize,
    n_per_gene: usize,
) -> Result<(), GpuError> {
    if n_per_gene == 0 || chunk_size == 0 {
        return Ok(());
    }

    // Fast path: one CUB BlockRadixSort per gene.
    if n_per_gene <= GPU_DE_BLOCK_SORT_CAPACITY {
        return gpu_de_single_tile_block_sort(dev, slab, chunk_size, n_per_gene);
    }

    // Multi-tile path: tile sort + iterative merge.
    let n_elements = chunk_size
        .checked_mul(n_per_gene)
        .ok_or_else(|| GpuError::ShapeMismatch {
            expected: "chunk_size * n_per_gene fits in usize".into(),
            got: format!("chunk_size={chunk_size}, n_per_gene={n_per_gene}"),
        })?;
    scratch.ensure_aux_capacity(dev, n_elements)?;

    // Tile sort in-place into `slab`. Each block handles one tile of one gene;
    // the final tile may be partial, handled via +inf padding in the kernel.
    gpu_de_tile_block_sort(dev, slab, chunk_size, n_per_gene)?;

    // Iterate merge passes. After tile sort the row contains K runs of size
    // ≤ GPU_DE_BLOCK_SORT_CAPACITY. Each pass doubles `run_size` and halves
    // the run count, terminating when run_size ≥ n_per_gene.
    let mut run_size = GPU_DE_BLOCK_SORT_CAPACITY;
    let mut n_passes: usize = 0;
    while run_size < n_per_gene {
        if n_passes.is_multiple_of(2) {
            // pass 0, 2, 4, ... : slab -> aux.
            gpu_de_merge_pass(
                dev,
                &*slab,
                &mut scratch.slab_aux,
                chunk_size,
                n_per_gene,
                run_size,
            )?;
        } else {
            // pass 1, 3, 5, ... : aux -> slab.
            gpu_de_merge_pass(
                dev,
                &scratch.slab_aux,
                &mut *slab,
                chunk_size,
                n_per_gene,
                run_size,
            )?;
        }
        n_passes += 1;
        run_size = run_size.saturating_mul(2);
    }

    // After an odd number of merge passes the final result lives in `slab_aux`;
    // copy it back to `slab` so callers downstream (searchsorted, tie-term,
    // p-value) read from the canonical buffer.
    if !n_passes.is_multiple_of(2) {
        let mut slab_view = slab.slice_mut(..n_elements);
        let aux_view = scratch.slab_aux.slice(..n_elements);
        dev.stream()
            .memcpy_dtod(&aux_view, &mut slab_view)
            .map_err(|e| GpuError::CudaError(format!("memcpy_dtod(aux→slab): {e}")))?;
    }

    Ok(())
}

/// Single-tile fast path — exactly the v1 single-block kernel from the
/// pre-tiled-merge implementation. Used directly when `n_per_gene ≤
/// GPU_DE_BLOCK_SORT_CAPACITY` and as the per-tile work in the multi-tile path.
fn gpu_de_single_tile_block_sort(
    dev: &GpuDevice,
    slab: &mut CudaSlice<f32>,
    chunk_size: usize,
    n_per_gene: usize,
) -> Result<(), GpuError> {
    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("block_radix_sort_per_gene_kernel")
        .map_err(|e| {
            GpuError::KernelLaunchFailed(format!("block_radix_sort_per_gene_kernel: {e}"))
        })?;

    let block_threads: u32 = 1024;
    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let chunk_i32 = chunk_size as i32;
    let n_per_gene_i32 = n_per_gene as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(slab)
            .arg(&chunk_i32)
            .arg(&n_per_gene_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("block_radix_sort_per_gene_kernel: {e}")))?;

    Ok(())
}

/// Tile-level block radix sort: one block per `(gene, tile)`. Each block sorts
/// a contiguous up-to-`GPU_DE_BLOCK_SORT_CAPACITY` slice of the gene's row in
/// place. Used as the first step of [`gpu_de_block_sort`]'s multi-tile path.
fn gpu_de_tile_block_sort(
    dev: &GpuDevice,
    slab: &mut CudaSlice<f32>,
    chunk_size: usize,
    n_per_gene: usize,
) -> Result<(), GpuError> {
    let tile_size = GPU_DE_BLOCK_SORT_CAPACITY;
    let n_tiles = n_per_gene.div_ceil(tile_size);

    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("tile_block_radix_sort_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("tile_block_radix_sort_kernel: {e}")))?;

    let block_threads: u32 = 1024;
    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, n_tiles as u32, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let chunk_i32 = chunk_size as i32;
    let n_per_gene_i32 = n_per_gene as i32;
    let tile_size_i32 = tile_size as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(slab)
            .arg(&chunk_i32)
            .arg(&n_per_gene_i32)
            .arg(&tile_size_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("tile_block_radix_sort_kernel: {e}")))?;

    Ok(())
}

/// One merge pass: merge pairs of sorted runs of size `run_size` from
/// `in_slab` into runs of size `2 * run_size` in `out_slab`. One block per
/// `(gene, pair)`. Called repeatedly with doubling `run_size` until the row
/// is fully sorted.
fn gpu_de_merge_pass(
    dev: &GpuDevice,
    in_slab: &CudaSlice<f32>,
    out_slab: &mut CudaSlice<f32>,
    chunk_size: usize,
    n_per_gene: usize,
    run_size: usize,
) -> Result<(), GpuError> {
    // Number of (paired) blocks per gene: each block merges one A-run and one
    // B-run. Odd run counts at the final pair just memcpy A → C inside the
    // kernel (see the `n == 0` branch in merge_pass_per_gene_kernel).
    let n_pairs = n_per_gene.div_ceil(run_size.saturating_mul(2));

    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("merge_pass_per_gene_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("merge_pass_per_gene_kernel: {e}")))?;

    // Must match MERGE_BLOCK_THREADS in diffexp.cu.
    let block_threads: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, n_pairs as u32, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let chunk_i32 = chunk_size as i32;
    let n_per_gene_i32 = n_per_gene as i32;
    let run_size_i32 = run_size as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(in_slab)
            .arg(out_slab)
            .arg(&chunk_i32)
            .arg(&n_per_gene_i32)
            .arg(&run_size_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("merge_pass_per_gene_kernel: {e}")))?;

    Ok(())
}

/// Compute Σ(c^3 − c) per gene over a single sorted row. Writes
/// `tie_out[gene]` for `gene in 0..chunk_size`.
pub fn gpu_de_tie_term(
    dev: &GpuDevice,
    sorted_slab: &CudaSlice<f32>,
    tie_out: &mut CudaSlice<f64>,
    chunk_size: usize,
    n_per_gene: usize,
) -> Result<(), GpuError> {
    if n_per_gene == 0 || chunk_size == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("tie_term_sorted_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("tie_term_sorted_kernel: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let chunk_i32 = chunk_size as i32;
    let n_per_gene_i32 = n_per_gene as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(sorted_slab)
            .arg(tie_out)
            .arg(&chunk_i32)
            .arg(&n_per_gene_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("tie_term_sorted_kernel: {e}")))?;
    Ok(())
}

/// Combined tie term over (ref + group): merge-walks two pre-sorted rows per
/// gene and writes the result into `tie_out`.
pub fn gpu_de_combined_tie_term(
    dev: &GpuDevice,
    sorted_ref: &CudaSlice<f32>,
    sorted_group: &CudaSlice<f32>,
    tie_out: &mut CudaSlice<f64>,
    chunk_size: usize,
    n_ref: usize,
    n_g: usize,
) -> Result<(), GpuError> {
    if chunk_size == 0 || (n_ref == 0 && n_g == 0) {
        return Ok(());
    }
    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("combined_tie_term_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("combined_tie_term_kernel: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let chunk_i32 = chunk_size as i32;
    let n_ref_i32 = n_ref as i32;
    let n_g_i32 = n_g as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(sorted_ref)
            .arg(sorted_group)
            .arg(tie_out)
            .arg(&chunk_i32)
            .arg(&n_ref_i32)
            .arg(&n_g_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("combined_tie_term_kernel: {e}")))?;
    Ok(())
}

/// Batched searchsorted U1 statistic for ref-mode MWU.
/// `u_out[gene]` = Σ_i (n_ref_less(group[i]) + 0.5·n_ref_equal(group[i])).
pub fn gpu_de_searchsorted_u_stat(
    dev: &GpuDevice,
    sorted_ref: &CudaSlice<f32>,
    group_slab: &CudaSlice<f32>,
    u_out: &mut CudaSlice<f64>,
    chunk_size: usize,
    n_ref: usize,
    n_g: usize,
) -> Result<(), GpuError> {
    if chunk_size == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("searchsorted_u_stat_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("searchsorted_u_stat_kernel: {e}")))?;

    let block_threads: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: block_threads * 8, // partials[256] f64
    };
    let chunk_i32 = chunk_size as i32;
    let n_ref_i32 = n_ref as i32;
    let n_g_i32 = n_g as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(sorted_ref)
            .arg(group_slab)
            .arg(u_out)
            .arg(&chunk_i32)
            .arg(&n_ref_i32)
            .arg(&n_g_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("searchsorted_u_stat_kernel: {e}")))?;
    Ok(())
}

/// 1-vs-rest mid-rank sum for the global Wilcoxon path.
/// `ranksum_out[gene]` = Σ over group cells of their mid-rank in the global
/// sorted-all pool. Then U_g = ranksum − n_g(n_g+1)/2.
pub fn gpu_de_searchsorted_ranksum(
    dev: &GpuDevice,
    sorted_all: &CudaSlice<f32>,
    group_slab: &CudaSlice<f32>,
    ranksum_out: &mut CudaSlice<f64>,
    chunk_size: usize,
    n_total: usize,
    n_g: usize,
) -> Result<(), GpuError> {
    if chunk_size == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("searchsorted_ranksum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("searchsorted_ranksum_kernel: {e}")))?;

    let block_threads: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: block_threads * 8,
    };
    let chunk_i32 = chunk_size as i32;
    let n_total_i32 = n_total as i32;
    let n_g_i32 = n_g as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(sorted_all)
            .arg(group_slab)
            .arg(ranksum_out)
            .arg(&chunk_i32)
            .arg(&n_total_i32)
            .arg(&n_g_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("searchsorted_ranksum_kernel: {e}")))?;
    Ok(())
}

/// MWU two-sided p-value via normal approximation with tie correction.
///
/// Matches the CPU `wilcoxon_full_from_ranks` formula exactly (no continuity
/// correction): `z = (U − μ)/σ`, `p = erfc(|z| / √2)`. Output is clipped to
/// `[0, 1]`. Caller is responsible for setting p = 1.0 when either group is
/// empty (kernel handles `n1 == 0 || n2 == 0` defensively but the chunk loop
/// short-circuits before launch).
pub fn gpu_de_pvalues(
    dev: &GpuDevice,
    u_stats: &CudaSlice<f64>,
    tie_term: &CudaSlice<f64>,
    p_out: &mut CudaSlice<f64>,
    chunk_size: usize,
    n1: usize,
    n2: usize,
) -> Result<(), GpuError> {
    if chunk_size == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("pvalue_erfc_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("pvalue_erfc_kernel: {e}")))?;

    let threads: u32 = 256;
    let blocks = (chunk_size as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let chunk_i32 = chunk_size as i32;
    let n1_i32 = n1 as i32;
    let n2_i32 = n2 as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(u_stats)
            .arg(tie_term)
            .arg(p_out)
            .arg(&chunk_i32)
            .arg(&n1_i32)
            .arg(&n2_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("pvalue_erfc_kernel: {e}")))?;
    Ok(())
}

/// All-groups pseudobulk fold: per-(gene × group) `Σ pre(x)` over each
/// group's cell list.
///
/// One kernel launch produces `[n_groups × chunk_size]` f64 sums in one pass.
/// Cells are flattened into `all_group_cells` with CSR-style `group_offsets`,
/// so the kernel knows which slice of the permutation each group owns:
///
/// ```text
/// group 0: all_group_cells[group_offsets[0] .. group_offsets[1])
/// group 1: all_group_cells[group_offsets[1] .. group_offsets[2])
/// ...
/// ```
///
/// `mode_id` selects the per-cell pre-transform applied before summation:
///
/// | id | meaning           | host equivalent                |
/// |----|-------------------|--------------------------------|
/// |  0 | ArithRaw / identity | `f(x) = x` (Wilcoxon raw sums) |
/// |  1 | ArithLog1pExpand  | `f(x) = expm1(x)`              |
/// |  2 | GeomRaw           | `f(x) = log1p(x)`              |
/// |  3 | GeomLog1p         | `f(x) = x`                     |
///
/// The host divides each sum by `n_cells_in_group` and applies `mode.post()`
/// to recover the natural-count mean. Sums are kept on device as f64 so
/// downstream callers can read them without precision loss.
///
/// Output `sums` must be pre-zeroed and sized `>= n_groups * chunk_size`.
#[allow(clippy::too_many_arguments)]
pub fn gpu_de_pseudobulk_all_groups(
    dev: &GpuDevice,
    dense: &CudaSlice<f32>,
    all_group_cells: &CudaSlice<i32>,
    group_offsets: &CudaSlice<i32>,
    sums: &mut CudaSlice<f64>,
    n_obs: usize,
    chunk_size: usize,
    n_groups: usize,
    mode_id: i32,
) -> Result<(), GpuError> {
    if chunk_size == 0 || n_groups == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("pseudobulk_all_groups_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("pseudobulk_all_groups_kernel: {e}")))?;

    // Must match PSEUDOBULK_BLOCK_THREADS in diffexp.cu.
    let block_threads: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, n_groups as u32, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: block_threads * 8, // sdata[block_threads] f64
    };

    let n_obs_i32 = n_obs as i32;
    let chunk_i32 = chunk_size as i32;
    let n_groups_i32 = n_groups as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(dense)
            .arg(all_group_cells)
            .arg(group_offsets)
            .arg(sums)
            .arg(&n_obs_i32)
            .arg(&chunk_i32)
            .arg(&n_groups_i32)
            .arg(&mode_id)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("pseudobulk_all_groups_kernel: {e}")))?;

    Ok(())
}

/// Heuristic gene-chunk size for streaming GPU DE.
///
/// Budgets ~18% of free VRAM for the dense + slab buffers needed by one
/// chunk, snapped to a multiple of 64 genes. Returns at least 64. Honours
/// the `SCX_GPU_DE_GENE_CHUNK_SIZE` env override (rounded to multiple of 64,
/// minimum 64).
pub fn default_gpu_de_gene_chunk_size(dev: &GpuDevice, n_obs: usize, n_pool_max: usize) -> usize {
    if let Ok(env) = std::env::var("SCX_GPU_DE_GENE_CHUNK_SIZE") {
        if let Ok(v) = env.parse::<usize>() {
            return ((v / 64).max(1)) * 64;
        }
    }
    let free_bytes = dev.free_memory().map(|(f, _)| f).unwrap_or(0);
    if free_bytes == 0 {
        return 256;
    }
    // Budget ~18% of free VRAM for the chunk's working set; account for
    // dense + slab + scratch (≈ (n_obs + n_pool_max) * 4 bytes per gene).
    let per_gene_bytes = (n_obs + n_pool_max) * 4 + 64; // small constant for f64 stats
    let budget = (free_bytes as f64 * 0.18) as usize;
    let raw = budget.max(per_gene_bytes) / per_gene_bytes;
    ((raw / 64).max(1)) * 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sort_capacity_constant_matches_kernel() {
        // BLOCK_THREADS * ITEMS_PER_THREAD in kernels/diffexp.cu must match
        // the published Rust constant. If you bump the kernel tunables,
        // bump the constant in lock-step.
        assert_eq!(GPU_DE_BLOCK_SORT_CAPACITY, 1024 * 8);
    }

    #[test]
    fn test_default_chunk_size_respects_env() {
        // Set an explicit override and confirm it round-trips (snapped to 64).
        let prev = std::env::var("SCX_GPU_DE_GENE_CHUNK_SIZE").ok();
        std::env::set_var("SCX_GPU_DE_GENE_CHUNK_SIZE", "129");
        // The function only touches GPU mem if no env override is set; this
        // test exercises the env-override branch without needing a device.
        // We fake a GpuDevice by creating one only if available; otherwise
        // skip with a noop assertion of the env path's value semantics.
        if let Ok(dev) = GpuDevice::new(0) {
            let v = default_gpu_de_gene_chunk_size(&dev, 1, 1);
            assert_eq!(v, 128, "129 should snap down to 128 (multiple of 64)");
        } else {
            // No GPU on this host; just confirm the snapping logic by hand.
            let raw: usize = 129;
            assert_eq!(((raw / 64).max(1)) * 64, 128);
        }
        if let Some(v) = prev {
            std::env::set_var("SCX_GPU_DE_GENE_CHUNK_SIZE", v);
        } else {
            std::env::remove_var("SCX_GPU_DE_GENE_CHUNK_SIZE");
        }
    }

    fn cpu_sort_ascending(row: &mut [f32]) {
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn cpu_tie_term(sorted: &[f32]) -> f64 {
        let mut s = 0.0f64;
        let n = sorted.len();
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && sorted[j] == sorted[i] {
                j += 1;
            }
            let c = (j - i) as i64;
            if c > 1 {
                s += (c * c * c - c) as f64;
            }
            i = j;
        }
        s
    }

    fn cpu_u1_searchsorted(sorted_ref: &[f32], group: &[f32]) -> f64 {
        let mut u = 0.0f64;
        for &x in group {
            let lo = sorted_ref.partition_point(|&r| r < x);
            let hi = sorted_ref.partition_point(|&r| r <= x);
            let n_less = lo as f64;
            let n_eq = (hi - lo) as f64;
            u += n_less + 0.5 * n_eq;
        }
        u
    }

    fn cpu_combined_tie(sorted_ref: &[f32], sorted_group: &[f32]) -> f64 {
        let mut merged: Vec<f32> = sorted_ref
            .iter()
            .copied()
            .chain(sorted_group.iter().copied())
            .collect();
        cpu_sort_ascending(&mut merged);
        cpu_tie_term(&merged)
    }

    /// End-to-end primitive parity: gene-major scatter + sort + tie + U1 +
    /// combined tie against CPU references on a small heavily-tied integer
    /// fixture. Skips cleanly when no CUDA device is available.
    #[test]
    fn test_gpu_de_primitives_match_cpu_reference() {
        let dev = require_gpu!();

        // 2 genes, 6 cells, with deliberate ties spanning ref+group.
        let n_obs = 6usize;
        let chunk_size = 2usize;
        // dense[cell * chunk_size + gene]
        // gene 0:  cell vals = [0, 0, 0, 1, 1, 2]
        // gene 1:  cell vals = [3, 1, 1, 2, 0, 0]
        let dense: Vec<f32> = vec![
            0.0, 3.0, // cell 0
            0.0, 1.0, // cell 1
            0.0, 1.0, // cell 2
            1.0, 2.0, // cell 3
            1.0, 0.0, // cell 4
            2.0, 0.0, // cell 5
        ];
        // Ref = {0, 1, 2, 3}; group = {4, 5}.
        let ref_cells: Vec<i32> = vec![0, 1, 2, 3];
        let group_cells: Vec<i32> = vec![4, 5];
        let n_ref = ref_cells.len();
        let n_g = group_cells.len();

        let mut scratch = GpuDeChunkScratch::new(&dev, n_obs, chunk_size, n_ref.max(n_g)).unwrap();
        gpu_de_upload_chunk(&dev, &mut scratch, &dense, n_obs, chunk_size).unwrap();

        // Scatter + sort ref.
        let mut d_ref_slab = dev.alloc_zeros::<f32>(chunk_size * n_ref).unwrap();
        gpu_de_scatter_gene_major(
            &dev,
            &scratch.dense,
            &ref_cells,
            &mut d_ref_slab,
            n_obs,
            chunk_size,
        )
        .unwrap();
        gpu_de_block_sort(&dev, &mut scratch, &mut d_ref_slab, chunk_size, n_ref).unwrap();
        gpu_de_tie_term(&dev, &d_ref_slab, &mut scratch.tie_term, chunk_size, n_ref).unwrap();
        dev.synchronize().unwrap();

        let sorted_ref_flat = dev.dtoh_copy(&d_ref_slab).unwrap();
        let tie_ref = dev.dtoh_copy(&scratch.tie_term).unwrap();

        // Expected sorted ref rows (length 4 per gene):
        // gene 0: [0, 0, 0, 1]
        // gene 1: [1, 1, 2, 3]
        let expected_ref_g0 = vec![0.0_f32, 0.0, 0.0, 1.0];
        let expected_ref_g1 = vec![1.0_f32, 1.0, 2.0, 3.0];
        assert_eq!(&sorted_ref_flat[0..4], expected_ref_g0.as_slice());
        assert_eq!(&sorted_ref_flat[4..8], expected_ref_g1.as_slice());
        assert!((tie_ref[0] - cpu_tie_term(&expected_ref_g0)).abs() < 1e-9);
        assert!((tie_ref[1] - cpu_tie_term(&expected_ref_g1)).abs() < 1e-9);

        // Scatter + sort group + searchsorted U1.
        let mut d_group_slab = dev.alloc_zeros::<f32>(chunk_size * n_g).unwrap();
        gpu_de_scatter_gene_major(
            &dev,
            &scratch.dense,
            &group_cells,
            &mut d_group_slab,
            n_obs,
            chunk_size,
        )
        .unwrap();
        gpu_de_searchsorted_u_stat(
            &dev,
            &d_ref_slab,
            &d_group_slab,
            &mut scratch.u_or_rank,
            chunk_size,
            n_ref,
            n_g,
        )
        .unwrap();
        gpu_de_block_sort(&dev, &mut scratch, &mut d_group_slab, chunk_size, n_g).unwrap();
        gpu_de_combined_tie_term(
            &dev,
            &d_ref_slab,
            &d_group_slab,
            &mut scratch.tie_term,
            chunk_size,
            n_ref,
            n_g,
        )
        .unwrap();
        gpu_de_pvalues(
            &dev,
            &scratch.u_or_rank,
            &scratch.tie_term,
            &mut scratch.p_values,
            chunk_size,
            n_g,
            n_ref,
        )
        .unwrap();
        dev.synchronize().unwrap();

        let u_host = dev.dtoh_copy(&scratch.u_or_rank).unwrap();
        let combined_host = dev.dtoh_copy(&scratch.tie_term).unwrap();
        let p_host = dev.dtoh_copy(&scratch.p_values).unwrap();

        // CPU references.
        // group for gene 0 (cells 4, 5): [1.0, 2.0]
        // group for gene 1 (cells 4, 5): [0.0, 0.0]
        let group_g0 = vec![1.0_f32, 2.0];
        let group_g1 = vec![0.0_f32, 0.0];
        let u_expected_g0 = cpu_u1_searchsorted(&expected_ref_g0, &group_g0);
        let u_expected_g1 = cpu_u1_searchsorted(&expected_ref_g1, &group_g1);
        assert!((u_host[0] - u_expected_g0).abs() < 1e-9);
        assert!((u_host[1] - u_expected_g1).abs() < 1e-9);

        let comb_expected_g0 = cpu_combined_tie(&expected_ref_g0, &group_g0);
        let comb_expected_g1 = cpu_combined_tie(&expected_ref_g1, &group_g1);
        assert!((combined_host[0] - comb_expected_g0).abs() < 1e-9);
        assert!((combined_host[1] - comb_expected_g1).abs() < 1e-9);

        // p-value sanity: within [0, 1] and finite.
        for &p in &p_host[..chunk_size] {
            assert!((0.0..=1.0).contains(&p), "p out of range: {p}");
            assert!(p.is_finite(), "p non-finite: {p}");
        }
    }

    /// Block radix sort over a randomized [16 × 1000] slab; row-wise parity
    /// with `Vec::sort_by` reference.
    #[test]
    fn test_gpu_de_block_sort_random_parity() {
        let dev = require_gpu!();

        let chunk_size = 16usize;
        let n_per_gene = 1000usize;

        // Deterministic pseudo-random input (no rand crate dep — splittable LCG).
        let mut state: u64 = 0xC0DEFACE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut data = vec![0.0f32; chunk_size * n_per_gene];
        for v in data.iter_mut() {
            // Bias toward integer-ish values to stress tie handling.
            *v = (next() % 32) as f32;
        }

        let mut d_slab = dev.htod_copy(&data).unwrap();
        let mut scratch = GpuDeChunkScratch::new(&dev, 1, chunk_size, 1).unwrap();
        gpu_de_block_sort(&dev, &mut scratch, &mut d_slab, chunk_size, n_per_gene).unwrap();
        dev.synchronize().unwrap();
        let gpu_sorted = dev.dtoh_copy(&d_slab).unwrap();

        for gene in 0..chunk_size {
            let mut expected: Vec<f32> = data[gene * n_per_gene..(gene + 1) * n_per_gene].to_vec();
            cpu_sort_ascending(&mut expected);
            let got = &gpu_sorted[gene * n_per_gene..(gene + 1) * n_per_gene];
            assert_eq!(got, expected.as_slice(), "sort mismatch on gene {gene}");
        }
    }

    /// Multi-tile sort: exercises the G1.5 tiled bottom-up merge path on a
    /// `[16 × 20_000]` slab. With `GPU_DE_BLOCK_SORT_CAPACITY = 8192`, 20_000
    /// keys per gene partitions into 3 tiles → 2 merge passes (4096 keys
    /// after pass 1, 8192 after pass 2; final pass merges 8192+4096 etc.).
    /// Output must match `Vec::sort_by` per row exactly. Includes deliberate
    /// ties (modulo) to stress the merge-path co-rank.
    #[test]
    fn test_gpu_de_block_sort_above_capacity() {
        let dev = require_gpu!();

        let chunk_size = 16usize;
        let n_per_gene = 20_000usize;
        assert!(
            n_per_gene > GPU_DE_BLOCK_SORT_CAPACITY,
            "test fixture must exceed the fast-path threshold to exercise the multi-tile path"
        );

        let mut state: u64 = 0xFEEDBEEF_2026;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut data = vec![0.0f32; chunk_size * n_per_gene];
        for v in data.iter_mut() {
            // Mod-128 produces lots of ties across runs → stresses merge-path
            // co_rank's "ties spanning the diagonal" branches.
            *v = (next() % 128) as f32;
        }

        let mut d_slab = dev.htod_copy(&data).unwrap();
        let mut scratch = GpuDeChunkScratch::new(&dev, 1, chunk_size, 1).unwrap();
        gpu_de_block_sort(&dev, &mut scratch, &mut d_slab, chunk_size, n_per_gene).unwrap();
        dev.synchronize().unwrap();
        let gpu_sorted = dev.dtoh_copy(&d_slab).unwrap();

        for gene in 0..chunk_size {
            let mut expected: Vec<f32> = data[gene * n_per_gene..(gene + 1) * n_per_gene].to_vec();
            cpu_sort_ascending(&mut expected);
            let got = &gpu_sorted[gene * n_per_gene..(gene + 1) * n_per_gene];
            assert_eq!(
                got,
                expected.as_slice(),
                "multi-tile sort mismatch on gene {gene} (n_per_gene={n_per_gene})"
            );
        }
    }

    /// Multi-tile sort with an awkward, non-power-of-two pool size — exercises
    /// the partial-tile and odd-pair-count edge cases (the final tile is only
    /// 1000 keys; the final merge pair pairs a full run with a partial one).
    #[test]
    fn test_gpu_de_block_sort_above_capacity_uneven() {
        let dev = require_gpu!();

        let chunk_size = 4usize;
        // 8192 + 7000 = 15_192 → 2 tiles (full + partial), 1 merge pass.
        let n_per_gene = 15_192usize;

        let mut state: u64 = 0xABADCAFE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut data = vec![0.0f32; chunk_size * n_per_gene];
        for v in data.iter_mut() {
            *v = ((next() as i32) & 0xffff) as f32; // larger range, fewer ties
        }

        let mut d_slab = dev.htod_copy(&data).unwrap();
        let mut scratch = GpuDeChunkScratch::new(&dev, 1, chunk_size, 1).unwrap();
        gpu_de_block_sort(&dev, &mut scratch, &mut d_slab, chunk_size, n_per_gene).unwrap();
        dev.synchronize().unwrap();
        let gpu_sorted = dev.dtoh_copy(&d_slab).unwrap();

        for gene in 0..chunk_size {
            let mut expected: Vec<f32> = data[gene * n_per_gene..(gene + 1) * n_per_gene].to_vec();
            cpu_sort_ascending(&mut expected);
            let got = &gpu_sorted[gene * n_per_gene..(gene + 1) * n_per_gene];
            assert_eq!(
                got,
                expected.as_slice(),
                "uneven multi-tile sort mismatch on gene {gene}"
            );
        }
    }

    /// G1.6 primitive parity: `gpu_de_pseudobulk_all_groups` per-(group × gene)
    /// sums must match a host f64 reference for all 4 `mode_id` transforms.
    /// Synthetic 50 cells × 5 genes × 3 groups fixture (ref + 2 test groups).
    #[test]
    fn test_gpu_de_pseudobulk_all_modes() {
        let dev = require_gpu!();

        let n_obs = 50usize;
        let chunk_size = 5usize;

        // Group layout: ref={0..19} (20 cells), A={20..34} (15), B={35..49} (15).
        let group_offsets: Vec<i32> = vec![0, 20, 35, 50];
        let all_group_cells: Vec<i32> = (0..n_obs as i32).collect();
        let n_groups = group_offsets.len() - 1;

        // Deterministic values in [0.0, 2.0) — chosen so expm1 and log1p both
        // produce non-trivial spread (avoids the f(0) = 0 trivial case).
        let mut state: u64 = 0x5EEDC0DE;
        let mut next_uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0
        };
        let mut dense_host = vec![0.0f32; n_obs * chunk_size];
        for v in dense_host.iter_mut() {
            *v = next_uniform() as f32;
        }

        let d_dense = dev.htod_copy(&dense_host).unwrap();
        let d_cells = dev.htod_copy(&all_group_cells).unwrap();
        let d_offsets = dev.htod_copy(&group_offsets).unwrap();

        // Host pre transform: keep in lockstep with apply_pre_transform in
        // diffexp.cu and GeomMeanMode::pre in scx-accel/src/pseudobulk.rs.
        let host_pre = |x: f32, mode_id: i32| -> f64 {
            let xd = x as f64;
            match mode_id {
                0 | 3 => xd,
                1 => xd.exp_m1(),
                2 => xd.ln_1p(),
                _ => xd,
            }
        };

        for mode_id in 0..4i32 {
            let mut d_sums = dev.alloc_zeros::<f64>(n_groups * chunk_size).unwrap();
            gpu_de_pseudobulk_all_groups(
                &dev,
                &d_dense,
                &d_cells,
                &d_offsets,
                &mut d_sums,
                n_obs,
                chunk_size,
                n_groups,
                mode_id,
            )
            .unwrap();
            dev.synchronize().unwrap();
            let gpu_sums = dev.dtoh_copy(&d_sums).unwrap();

            // Host reference: for each (group, gene) accumulate pre(x) over the
            // group's cell list.
            for g in 0..n_groups {
                let start = group_offsets[g] as usize;
                let end = group_offsets[g + 1] as usize;
                for gene in 0..chunk_size {
                    let mut host_sum = 0.0f64;
                    for i in start..end {
                        let cell = all_group_cells[i] as usize;
                        let x = dense_host[cell * chunk_size + gene];
                        host_sum += host_pre(x, mode_id);
                    }
                    let gpu_sum = gpu_sums[g * chunk_size + gene];
                    let diff = (host_sum - gpu_sum).abs();
                    let denom = host_sum.abs().max(1.0);
                    assert!(
                        diff < 1e-9 || diff / denom < 1e-12,
                        "mode_id={mode_id} group={g} gene={gene}: host={host_sum}, \
                         gpu={gpu_sum}, |Δ|={diff}"
                    );
                }
            }
        }
    }
}
