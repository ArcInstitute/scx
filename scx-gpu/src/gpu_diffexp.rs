//! GPU differential expression primitives (Mann-Whitney U / Wilcoxon rank sum).
//!
//! Per-gene block radix sort
//! (CUB), batched searchsorted, on-device tie-term computation, and normal-tail
//! p-value via `erfc`. These primitives are the building blocks for the
//! high-level `pdex_ref_gpu_*` / `wilcoxon_rank_sum_*_gpu` entry points that
//! live in `scx-accel/src/diffexp/gpu.rs` under `#[cfg(feature = "gpu")]` — see
//! `scx-accel/src/diffexp/cpu.rs` for the parity oracle.
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
//!   partitioning), ping-ponging between the gene-major slab being sorted
//!   (`scratch.ref_slab` / a `per_tg_pool_slabs` entry) and `scratch.slab_aux`.
//!
//! No upper limit beyond available VRAM. Callers no longer need to guard
//! pool sizes against the 8192 threshold — the dispatch handles arbitrary
//! sizes internally. Tested on census-scale fixtures (n_pool ≈ 1M) where the
//! multi-tile path engages ~7 merge passes per chunk.

use std::sync::OnceLock;

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_csc_shard_source::GpuCscShardView;
use crate::staging::GpuCsrShardView;

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
/// loop resizes `ref_slab` / `per_tg_pool_slabs` only when the membership
/// counts change. This intentionally mirrors how
/// `gpu_pca::GpuPcaScratch` hoists allocations out of the power-iteration
/// inner loop.
pub struct GpuDeChunkScratch {
    /// Ping-pong buffer used by the multi-tile path in [`gpu_de_block_sort`]
    /// when `n_per_gene > GPU_DE_BLOCK_SORT_CAPACITY`. Allocated lazily on
    /// first multi-tile encounter via [`Self::ensure_aux_capacity`]; small-
    /// pool callers never pay for it.
    ///
    /// The `slab` this used to ping-pong against is gone: no v3 driver ever
    /// read it (nothing called `ensure_slab_capacity`), while
    /// `GpuDeChunkScratch::new` allocated `chunk_max × n_pool_max` f32 eagerly
    /// — ~2 GB at 1 M cells and chunk 500 — *and* the budget charged for it,
    /// so the clamp then shrank the gene chunk to pay for VRAM nothing used.
    /// The sorts ping-pong between `ref_slab` / `per_tg_pool_slabs` and this.
    pub slab_aux: CudaSlice<f32>,
    /// `[chunk_max]` f64 tie-term scratch (ref-only or combined).
    pub tie_term: CudaSlice<f64>,
    /// `[chunk_max]` f64 U1 / rank-sum scratch.
    pub u_or_rank: CudaSlice<f64>,
    /// `[chunk_max]` f64 p-values for one test group.
    pub p_values: CudaSlice<f64>,
    /// `[chunk_max × n_ref_max]` gene-major ref slab. G2 hoisted from a
    /// per-chunk `dev.alloc_zeros` in the pdex_ref GPU driver. Grow-only via
    /// [`Self::ensure_ref_slab_capacity`].
    pub ref_slab: CudaSlice<f32>,
    /// `[n_groups_max × chunk_max]` f64 pseudobulk sums buffer. G2 hoisted
    /// from the per-chunk pseudobulk fold. Grow-only via
    /// [`Self::ensure_sums_capacity`].
    pub sums: CudaSlice<f64>,
    /// `[n_test_groups_max × chunk_max]` f64 per-test-group U / rank-sum
    /// staging buffer. G10.4 hoist: each test group's U output is
    /// `memcpy_dtod`-copied here from `u_or_rank` so the host-side
    /// `dtoh_copy` can batch all groups into a single PCIe transfer at
    /// chunk end. Grow-only via [`Self::ensure_per_group_capacity`].
    pub u_per_group: CudaSlice<f64>,
    /// `[n_test_groups_max × chunk_max]` f64 per-test-group p-value
    /// staging buffer. Same G10.4 pattern as `u_per_group`.
    pub p_per_group: CudaSlice<f64>,
    /// `[n_test_groups_max × chunk_max]` f64 per-test-group combined-tie
    /// staging buffer. Used by the Wilcoxon GPU driver's ref-
    /// mode path, where each tg produces its own combined tie term that
    /// must round-trip to host for the post-pvalue computation. 1-vs-
    /// rest reuses the global pool-tie and so doesn't write here.
    pub tie_per_group: CudaSlice<f64>,
    /// G4 v2: per-test-group gene-major slabs (one slab per test group,
    /// each `[chunk_max × n_g_max]` f32). The v2 streaming path pre-zeros
    /// all of these once per chunk, then a single shard iteration
    /// scatters each shard's CSR rows into ref_slab + every tg slab in
    /// parallel — eliminating the dense→slab gather kernel (one launch
    /// per tg per chunk in v1).
    ///
    /// Empty on v1 paths; grown by [`Self::ensure_per_tg_pool_slabs_capacity`]
    /// in the v2 driver above the chunk loop. The vec length equals
    /// `n_test_groups`.
    pub per_tg_pool_slabs: Vec<CudaSlice<f32>>,
    n_obs: usize,
    chunk_max: usize,
    aux_capacity_elems: usize,
    ref_slab_capacity: usize,
    sums_capacity: usize,
    per_group_capacity: usize,
    per_tg_pool_slabs_capacity: usize,
    alloc_count: u64,
}

/// Checked element-count product for GPU DE device allocations. Mirrors the
/// `gpu_de_block_sort` overflow guard so an atlas-scale dimension fails with a
/// clear `ShapeMismatch` instead of wrapping into an under-sized buffer.
fn de_alloc_elems(rows: usize, cols: usize) -> Result<usize, GpuError> {
    rows.checked_mul(cols)
        .ok_or_else(|| GpuError::ShapeMismatch {
            expected: "rows × cols fits in usize".into(),
            got: format!("rows={rows}, cols={cols}"),
        })
}

impl GpuDeChunkScratch {
    /// Allocate scratch buffers sized for `n_obs` cells and up to `chunk_max`
    /// genes per chunk.
    ///
    /// Everything except the three `[chunk_max]` f64 scalars starts at zero
    /// length and grows through an `ensure_*_capacity` call, so a small DE
    /// never pays for an atlas-scale buffer.
    pub fn new(dev: &GpuDevice, n_obs: usize, chunk_max: usize) -> Result<Self, GpuError> {
        // slab_aux is allocated lazily — empty until the first call that hits
        // the multi-tile sort path. Allocating a zero-length CudaSlice is
        // cheap (a few bytes of metadata) and avoids paying VRAM for the
        // common small-pool case where every pool fits in one tile.
        let slab_aux = dev.alloc_zeros::<f32>(0)?;
        let tie_term = dev.alloc_zeros::<f64>(chunk_max)?;
        let u_or_rank = dev.alloc_zeros::<f64>(chunk_max)?;
        let p_values = dev.alloc_zeros::<f64>(chunk_max)?;
        // G2 grow-on-demand slots — start at zero size so small DE calls
        // never pay for them; first `ensure_*_capacity` call allocates.
        let ref_slab = dev.alloc_zeros::<f32>(0)?;
        let sums = dev.alloc_zeros::<f64>(0)?;
        let u_per_group = dev.alloc_zeros::<f64>(0)?;
        let p_per_group = dev.alloc_zeros::<f64>(0)?;
        let tie_per_group = dev.alloc_zeros::<f64>(0)?;
        Ok(Self {
            slab_aux,
            tie_term,
            u_or_rank,
            p_values,
            ref_slab,
            sums,
            u_per_group,
            p_per_group,
            tie_per_group,
            per_tg_pool_slabs: Vec::new(),
            n_obs,
            chunk_max,
            aux_capacity_elems: 0,
            ref_slab_capacity: 0,
            sums_capacity: 0,
            per_group_capacity: 0,
            per_tg_pool_slabs_capacity: 0,
            alloc_count: 0,
        })
    }

    /// Grow the ping-pong aux buffer to hold at least `n_elements` f32 keys.
    ///
    /// **The caller must size this before a multi-tile sort** — despite the
    /// name, [`gpu_de_block_sort`] does NOT call it. It cannot: it takes `slab`
    /// and `aux` as two separate `&mut CudaSlice<f32>` (so a caller can thread
    /// two disjoint `GpuDeChunkScratch` fields at once) and so holds no handle
    /// to the scratch. An undersized aux is rejected there with
    /// `ShapeMismatch`, not grown. Use [`gpu_de_aux_elems`] to compute
    /// `n_elements`, and read its doc for which sort's `n_per_gene` governs.
    ///
    /// No-op when the aux is already large enough.
    ///
    /// Takes `chunk_size` and `span` separately, and rounds the **span** — not
    /// their product — to `next_power_of_two`, so the allocation is exactly
    /// `chunk_size × p2(span)` and [`gpu_de_per_gene_scratch_bytes`]'s
    /// `p2(n_aux)` charge is the per-gene truth rather than an approximation.
    /// Rounding the product instead (what this did before) made the real buffer
    /// up to **2×** the budgeted one — 4.29 GB against 2.4 GB modelled at
    /// `pool_len = 600_000, chunk = 1000` — so the clamp reported `fits = true`
    /// and the allocator then raised a bare `OutOfMemory` instead of the
    /// dimension-naming error the clamp exists to produce. `n_aux` was the only
    /// f32 term in that model without a `p2()`; `n_ref` and `n_g_max` already
    /// round their spans, and this now matches them.
    pub fn ensure_aux_capacity(
        &mut self,
        dev: &GpuDevice,
        chunk_size: usize,
        span: usize,
    ) -> Result<(), GpuError> {
        if span == 0 {
            return Ok(());
        }
        let n_elements = gpu_de_aux_alloc_elems(chunk_size, span)?;
        if n_elements <= self.aux_capacity_elems {
            return Ok(());
        }
        self.slab_aux = dev.alloc_zeros::<f32>(n_elements)?;
        self.aux_capacity_elems = n_elements;
        self.alloc_count += 1;
        Ok(())
    }

    /// Grow `ref_slab` to hold at least `chunk_max × n_ref` f32 keys.
    /// Called once per pdex_ref GPU driver invocation before the chunk
    /// loop. Same grow-only `next_power_of_two` pattern as the slab.
    pub fn ensure_ref_slab_capacity(
        &mut self,
        dev: &GpuDevice,
        n_ref: usize,
    ) -> Result<(), GpuError> {
        if n_ref <= self.ref_slab_capacity {
            return Ok(());
        }
        let new_cap = n_ref.next_power_of_two().max(self.ref_slab_capacity * 2);
        self.ref_slab = dev.alloc_zeros::<f32>(de_alloc_elems(self.chunk_max, new_cap)?)?;
        self.ref_slab_capacity = new_cap;
        self.alloc_count += 1;
        Ok(())
    }

    /// Grow `sums` to hold at least `n_groups × chunk_max` f64 values.
    /// Called above the chunk loop in pdex_ref / Wilcoxon driver before
    /// any pseudobulk fold.
    pub fn ensure_sums_capacity(
        &mut self,
        dev: &GpuDevice,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        if n_groups <= self.sums_capacity {
            return Ok(());
        }
        let new_cap = n_groups.next_power_of_two().max(self.sums_capacity * 2);
        self.sums = dev.alloc_zeros::<f64>(de_alloc_elems(self.chunk_max, new_cap)?)?;
        self.sums_capacity = new_cap;
        self.alloc_count += 1;
        Ok(())
    }

    /// G10.4: grow `u_per_group` and `p_per_group` to hold at least
    /// `n_test_groups × chunk_max` f64 values each.
    ///
    /// Called above the chunk loop in the pdex_ref / Wilcoxon GPU drivers
    /// so the per-chunk dtoh fan-out
    /// (one transfer per test group) collapses into a single batched
    /// dtoh at chunk end. `memcpy_dtod` from `u_or_rank` /
    /// `p_values` into the per-group slot is `O(chunk_size)` and
    /// stays on-device, so the per-tg dispatch becomes a fire-and-
    /// forget GPU operation.
    pub fn ensure_per_group_capacity(
        &mut self,
        dev: &GpuDevice,
        n_test_groups: usize,
    ) -> Result<(), GpuError> {
        if n_test_groups <= self.per_group_capacity {
            return Ok(());
        }
        let new_cap = n_test_groups
            .next_power_of_two()
            .max(self.per_group_capacity * 2);
        let elems = de_alloc_elems(self.chunk_max, new_cap)?;
        self.u_per_group = dev.alloc_zeros::<f64>(elems)?;
        self.p_per_group = dev.alloc_zeros::<f64>(elems)?;
        self.tie_per_group = dev.alloc_zeros::<f64>(elems)?;
        self.per_group_capacity = new_cap;
        self.alloc_count += 3;
        Ok(())
    }

    /// G4 v2: grow `per_tg_pool_slabs` to a vec of length `n_test_groups`,
    /// each slab `[chunk_max × n_g_max]` f32. Called above the chunk loop
    /// in the v2 driver. Slabs are pre-zeroed by the chunk loop before
    /// each chunk's shard iteration.
    ///
    /// Idempotent — only re-allocates when the test-group count grows or
    /// the per-tg pool size grows.
    pub fn ensure_per_tg_pool_slabs_capacity(
        &mut self,
        dev: &GpuDevice,
        n_test_groups: usize,
        n_g_max: usize,
    ) -> Result<(), GpuError> {
        let n_g_max = n_g_max.max(1);
        let need_grow = n_test_groups > self.per_tg_pool_slabs.len()
            || n_g_max > self.per_tg_pool_slabs_capacity;
        if !need_grow {
            return Ok(());
        }
        let new_cap = n_g_max
            .next_power_of_two()
            .max(self.per_tg_pool_slabs_capacity * 2)
            .max(n_g_max);
        // Always re-allocate the full vec when growing — the per-slab
        // capacity is uniform.
        self.per_tg_pool_slabs.clear();
        for _ in 0..n_test_groups {
            self.per_tg_pool_slabs
                .push(dev.alloc_zeros::<f32>(de_alloc_elems(self.chunk_max, new_cap)?)?);
            self.alloc_count += 1;
        }
        self.per_tg_pool_slabs_capacity = new_cap;
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

    /// Total scratch-buffer grow events since construction. Used by
    /// regression tests (`test_pdex_ref_no_realloc_across_chunks`) to
    /// verify the chunk loop reuses scratch without re-allocating.
    pub fn alloc_count(&self) -> u64 {
        self.alloc_count
    }
}

/// G4 v3 CSR scatter with combined cell_to_group + cell_to_pos tables
/// (post code-review #6). The per-pool lookup goes through the shared
/// `cell_to_group_dev` + `cell_to_pos_dev` device tables rather than K+1
/// per-pool inverse-permutation buffers.
///
/// Caller dispatches K+1 launches per shard (one per pool: `group_id=0`
/// for ref, `group_id=k+1` for tg_k). The whole-block early exit triggers
/// when the row's cell isn't in `this_group_id`; otherwise threads stride
/// the row's nonzeros and write to
/// `slab[gene_local × n_perm + cell_to_pos[cell]]`. No atomicAdd, no race.
///
/// `slab` MUST be pre-zeroed before the FIRST shard's scatter for each
/// pool (zeros are implicit in the CSR representation).
#[allow(clippy::too_many_arguments)]
pub fn gpu_de_scatter_csr_to_gene_major_filtered(
    dev: &GpuDevice,
    view: &GpuCsrShardView<'_>,
    cell_to_group_dev: &CudaSlice<i32>,
    cell_to_pos_dev: &CudaSlice<i32>,
    this_group_id: i32,
    slab: &mut CudaSlice<f32>,
    global_row_offset: usize,
    n_perm: usize,
    chunk_size: usize,
    c0: usize,
    c1: usize,
) -> Result<(), GpuError> {
    if n_perm == 0 || chunk_size == 0 || c0 >= c1 {
        return Ok(());
    }
    let n_shard_rows = view.shape.0;
    if n_shard_rows == 0 {
        return Ok(());
    }
    debug_assert!(
        n_shard_rows <= i32::MAX as usize,
        "n_shard_rows {} exceeds i32::MAX — SCX shard layout invariant violated",
        n_shard_rows
    );
    debug_assert!(
        n_perm <= i32::MAX as usize,
        "n_perm {} exceeds i32::MAX",
        n_perm
    );

    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("csr_shard_to_gene_major_filtered_kernel")
        .map_err(|e| {
            GpuError::KernelLaunchFailed(format!("csr_shard_to_gene_major_filtered_kernel: {e}"))
        })?;

    let bx: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: (n_shard_rows as u32, 1, 1),
        block_dim: (bx, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_shard_rows_i32 = n_shard_rows as i32;
    let global_row_offset_i64 = global_row_offset as i64;
    let n_perm_i32 = n_perm as i32;
    let c0_i32 = c0 as i32;
    let c1_i32 = c1 as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(&view.indptr)
            .arg(&view.indices)
            .arg(&view.data)
            .arg(cell_to_group_dev)
            .arg(cell_to_pos_dev)
            .arg(&this_group_id)
            .arg(slab)
            .arg(&n_shard_rows_i32)
            .arg(&global_row_offset_i64)
            .arg(&n_perm_i32)
            .arg(&c0_i32)
            .arg(&c1_i32)
            .launch(cfg)
    }
    .map_err(|e| {
        GpuError::KernelLaunchFailed(format!("csr_shard_to_gene_major_filtered_kernel: {e}"))
    })?;
    let _ = chunk_size;
    Ok(())
}

/// G4 v3: CSR-direct pseudobulk fold — accumulate `[n_groups × chunk_size]`
/// f64 sums by streaming a CSR shard view directly, skipping the
/// `[n_obs × chunk_size]` dense intermediate that
/// [`gpu_de_pseudobulk_all_groups`] reads from.
///
/// Pairs with [`build_cell_to_group_dev`] (cell→group inverse permutation,
/// uploaded once per DE call) and is launched per-shard inside the
/// `for_each_gpu_shard` loop. `sums` MUST be pre-zeroed by the caller
/// before the first shard's invocation — atomicAdd accumulates.
///
/// `mode_id` selects the per-element pre-transform applied before
/// summation; same encoding as [`gpu_de_pseudobulk_all_groups`]
/// (0=identity, 1=expm1, 2=log1p, 3=identity). The host divides by the
/// group's cell count and applies the matching `mode.post` transform.
///
/// Grid: one block per shard row. Whole-block early-exit when the cell is
/// not in any group (`cell_to_group[global_cell] < 0`). f64 atomicAdd is
/// hardware-native on H100 (CC 6.0+; H100 is 9.0).
#[allow(clippy::too_many_arguments)]
pub fn gpu_de_pseudobulk_csr_direct(
    dev: &GpuDevice,
    view: &GpuCsrShardView<'_>,
    cell_to_group_dev: &CudaSlice<i32>,
    sums: &mut CudaSlice<f64>,
    global_row_offset: usize,
    chunk_size: usize,
    c0: usize,
    c1: usize,
    mode_id: i32,
) -> Result<(), GpuError> {
    if chunk_size == 0 || c0 >= c1 {
        return Ok(());
    }
    let n_shard_rows = view.shape.0;
    if n_shard_rows == 0 {
        return Ok(());
    }
    debug_assert!(
        n_shard_rows <= i32::MAX as usize,
        "n_shard_rows {} exceeds i32::MAX — SCX shard layout invariant violated",
        n_shard_rows
    );

    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("csr_shard_pseudobulk_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("csr_shard_pseudobulk_kernel: {e}")))?;

    let bx: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: (n_shard_rows as u32, 1, 1),
        block_dim: (bx, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_shard_rows_i32 = n_shard_rows as i32;
    let global_row_offset_i64 = global_row_offset as i64;
    let chunk_size_i32 = chunk_size as i32;
    let c0_i32 = c0 as i32;
    let c1_i32 = c1 as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(&view.indptr)
            .arg(&view.indices)
            .arg(&view.data)
            .arg(cell_to_group_dev)
            .arg(sums)
            .arg(&n_shard_rows_i32)
            .arg(&global_row_offset_i64)
            .arg(&chunk_size_i32)
            .arg(&c0_i32)
            .arg(&c1_i32)
            .arg(&mode_id)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("csr_shard_pseudobulk_kernel: {e}")))?;

    Ok(())
}

/// Reject a per-cell device table that does not cover every row the shard can
/// name, before any CSC kernel indexes it.
///
/// The CSC kernels dereference `cell_to_group[row_indices[e]]` /
/// `cell_to_pos[...]` with no device-side bound on the table, so this closes
/// the second half of the out-of-range hazard: `RawGpuCscShardSource` validates
/// row indices against the *source's* `n_obs`, and nothing else compares that
/// against the `n_obs` the caller sized these tables from — the DE drivers take
/// it from their `groups` argument while the source takes it from the file. An
/// O(1) check per launch, so it costs nothing next to the kernel it guards.
fn ensure_cell_table_covers(
    table: &CudaSlice<i32>,
    n_obs: usize,
    name: &str,
) -> Result<(), GpuError> {
    if table.len() < n_obs {
        return Err(GpuError::ShapeMismatch {
            expected: format!("{name} covering the shard's n_obs = {n_obs}"),
            got: format!("{name}.len() = {}", table.len()),
        });
    }
    Ok(())
}

/// G4 v3 (CSC-direct): all-groups pseudobulk fold over one CSC shard.
///
/// One block per chunk gene; threads tree-reduce per-group accumulators in
/// shared memory; thread 0 of each block adds the block result into the
/// running `sums[g * chunk_size + gene_local]`. **No atomicAdd** —
/// cross-shard accumulation is safe via stream ordering (each launch reads
/// the running sum, adds this shard's contribution, writes back).
///
/// `sums` MUST be pre-zeroed by the caller before the first shard's
/// invocation. Blocks whose `gene_global` falls outside the shard's column
/// range `[csc_view.col_start, csc_view.col_end)` early-exit at no cost.
///
/// `mode_id` encoding matches [`gpu_de_pseudobulk_all_groups`] (0=identity,
/// 1=expm1, 2=log1p, 3=identity).
///
/// Adaptive accumulation (no `n_groups` ceiling):
/// * **SMEM-staged** (`csc_shard_pseudobulk_kernel`) — `n_groups × blockDim.x ×
///   8` bytes of shared memory, tree-reduced (fast, atomic-free, deterministic).
///   The wrapper picks the largest `bx ∈ {128, 64, 32}` whose SMEM fits the
///   device opt-in limit (opting into >48 KB via
///   `set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, …)`). On
///   H100 `bx=32` admits ~900 groups (covers census×cell_type).
/// * **Global-atomic** (`csc_shard_pseudobulk_global_kernel`) — used when no
///   `bx` fits (many groups, e.g. perturb-seq guide-level DE). No per-group
///   SMEM; `atomicAdd` into global `sums`, scaling to any `n_groups`. f64
///   atomicAdd makes summation order run-to-run nondeterministic (as on the
///   CSR-direct path). `SCX_GPU_DE_PSEUDOBULK_FORCE_ATOMIC=1` forces this path
///   for testing.
#[allow(clippy::too_many_arguments)]
pub fn gpu_de_pseudobulk_csc_direct(
    dev: &GpuDevice,
    csc_view: &GpuCscShardView<'_>,
    cell_to_group_dev: &CudaSlice<i32>,
    sums: &mut CudaSlice<f64>,
    c0: usize,
    c1: usize,
    chunk_size: usize,
    n_groups: usize,
    mode_id: i32,
) -> Result<(), GpuError> {
    if chunk_size == 0 || c0 >= c1 || n_groups == 0 {
        return Ok(());
    }
    let n_cols_in_shard = csc_view.n_cols();
    if n_cols_in_shard == 0 {
        return Ok(());
    }
    // Skip entirely if the shard's columns don't overlap the chunk.
    if csc_view.col_end <= c0 || csc_view.col_start >= c1 {
        return Ok(());
    }
    debug_assert!(
        n_cols_in_shard <= i32::MAX as usize,
        "n_cols_in_shard {} exceeds i32::MAX",
        n_cols_in_shard
    );
    ensure_cell_table_covers(cell_to_group_dev, csc_view.n_obs, "cell_to_group")?;

    let module = dev.load_module_cached(DIFFEXP_PTX)?;

    // Adaptive accumulation strategy.
    //
    // The SMEM-staged `csc_shard_pseudobulk_kernel` keeps `n_groups × blockDim.x`
    // f64 per-group partials in shared memory and tree-reduces them — fast and
    // *deterministic* (no atomics), but its `n_groups × blockDim.x × 8` SMEM
    // footprint hits the per-block hardware ceiling beyond a few hundred groups.
    // Pick the largest `bx ∈ {128, 64, 32}` whose SMEM fits the device opt-in
    // limit (48 KB default; ~228 KB opt-in on H100 — `bx=32` admits ~900 groups,
    // covering census×cell_type). When none fit (many groups, e.g. perturb-seq
    // guide-level DE), fall to `csc_shard_pseudobulk_global_kernel`, which uses
    // global `atomicAdd` with no per-group SMEM and scales to any `n_groups`;
    // atomic contention is naturally low in that regime. `n_groups` is constant
    // across a DE op, so every shard launch takes the same path.
    let smem_bx = csc_pseudobulk_block_dim(
        n_groups,
        csc_pseudobulk_smem_limit(dev),
        gpu_de_force_atomic_pseudobulk(),
    );
    if smem_bx.is_none() && gpu_de_require_deterministic() {
        return Err(GpuError::UnsupportedLayout(format!(
            "SCX_GPU_DE_REQUIRE_DETERMINISTIC=1, but {n_groups} groups do not fit the \
             deterministic CSC pseudobulk kernel's shared memory on this device \
             ({} B opt-in): the fallback is a global f64 atomicAdd, whose summation \
             order is not reproducible. Use fewer/coarser groups, run on a device with \
             more opt-in shared memory, or unset the variable and accept the atomic \
             reduction (it is stamped as `reduction=\"atomic\"` on uns[\"scx_accel\"]).",
            csc_pseudobulk_smem_limit(dev)
        )));
    }

    let (func, bx, shared_mem_bytes) = match smem_bx {
        Some(bx) => {
            let func = module
                .load_function("csc_shard_pseudobulk_kernel")
                .map_err(|e| {
                    GpuError::KernelLaunchFailed(format!("csc_shard_pseudobulk_kernel: {e}"))
                })?;
            let required = n_groups * bx as usize * std::mem::size_of::<f64>();
            // Opt into >48 KB when required (per-function sticky + idempotent).
            if required > CSC_PSEUDOBULK_DEFAULT_SMEM_LIMIT {
                func.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    required as i32,
                )
                .map_err(|e| {
                    GpuError::KernelLaunchFailed(format!(
                        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={required}) on \
                         csc_shard_pseudobulk_kernel failed: {e}"
                    ))
                })?;
            }
            (func, bx, required as u32)
        }
        None => {
            let func = module
                .load_function("csc_shard_pseudobulk_global_kernel")
                .map_err(|e| {
                    GpuError::KernelLaunchFailed(format!("csc_shard_pseudobulk_global_kernel: {e}"))
                })?;
            (func, 128u32, 0u32)
        }
    };

    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (bx, 1, 1),
        shared_mem_bytes,
    };

    let n_cols_in_shard_i32 = n_cols_in_shard as i32;
    let shard_col_start_i32 = csc_view.col_start as i32;
    let c0_i32 = c0 as i32;
    let chunk_size_i32 = chunk_size as i32;
    let n_groups_i32 = n_groups as i32;
    let n_obs_i32 = i32::try_from(csc_view.n_obs).map_err(|_| GpuError::ShapeMismatch {
        expected: "n_obs <= i32::MAX".to_string(),
        got: format!("n_obs = {}", csc_view.n_obs),
    })?;

    // Both kernels share the same parameter list.
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(&csc_view.col_indptr)
            .arg(&csc_view.row_indices)
            .arg(&csc_view.data)
            .arg(cell_to_group_dev)
            .arg(sums)
            .arg(&n_cols_in_shard_i32)
            .arg(&shard_col_start_i32)
            .arg(&c0_i32)
            .arg(&chunk_size_i32)
            .arg(&n_groups_i32)
            .arg(&mode_id)
            .arg(&n_obs_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("csc_shard_pseudobulk launch: {e}")))?;

    Ok(())
}

/// G4 v3 (CSC-direct, post code-review #6): scatter one CSC shard's
/// nonzeros into a gene-major pool slab `[chunk_size × n_perm]` using
/// combined `cell_to_group + cell_to_pos` tables and a launch-time
/// `this_group_id` filter.
///
/// One block per chunk gene; threads stride the gene's CSC column. The
/// kernel writes only the cells where `cell_to_group[cell] ==
/// this_group_id`, using `cell_to_pos[cell]` as the in-group position.
/// **No atomicAdd, no race** — each `(gene_local, pos)` cell has at most
/// one writer per launch.
///
/// Caller dispatches K+1 launches per shard (one per pool: `group_id=0`
/// for ref, `group_id=k+1` for tg_k), reusing the same combined device
/// tables — total cell-permutation memory is `2 n_obs * 4` bytes
/// regardless of K (vs `(K+1) * n_obs * 4` in the pre-#6 per-pool design).
///
/// `slab` MUST be pre-zeroed before the first shard's scatter.
#[allow(clippy::too_many_arguments)]
pub fn gpu_de_scatter_csc_to_gene_major(
    dev: &GpuDevice,
    csc_view: &GpuCscShardView<'_>,
    cell_to_group_dev: &CudaSlice<i32>,
    cell_to_pos_dev: &CudaSlice<i32>,
    this_group_id: i32,
    slab: &mut CudaSlice<f32>,
    c0: usize,
    c1: usize,
    chunk_size: usize,
    n_perm: usize,
) -> Result<(), GpuError> {
    if chunk_size == 0 || n_perm == 0 || c0 >= c1 {
        return Ok(());
    }
    let n_cols_in_shard = csc_view.n_cols();
    if n_cols_in_shard == 0 {
        return Ok(());
    }
    if csc_view.col_end <= c0 || csc_view.col_start >= c1 {
        return Ok(());
    }
    debug_assert!(
        n_cols_in_shard <= i32::MAX as usize,
        "n_cols_in_shard {} exceeds i32::MAX",
        n_cols_in_shard
    );
    debug_assert!(
        n_perm <= i32::MAX as usize,
        "n_perm {} exceeds i32::MAX",
        n_perm
    );
    ensure_cell_table_covers(cell_to_group_dev, csc_view.n_obs, "cell_to_group")?;
    ensure_cell_table_covers(cell_to_pos_dev, csc_view.n_obs, "cell_to_pos")?;

    let module = dev.load_module_cached(DIFFEXP_PTX)?;
    let func = module
        .load_function("csc_shard_to_gene_major_kernel")
        .map_err(|e| {
            GpuError::KernelLaunchFailed(format!("csc_shard_to_gene_major_kernel: {e}"))
        })?;

    let bx: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (bx, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_cols_in_shard_i32 = n_cols_in_shard as i32;
    let shard_col_start_i32 = csc_view.col_start as i32;
    let c0_i32 = c0 as i32;
    let chunk_size_i32 = chunk_size as i32;
    let n_perm_i32 = n_perm as i32;
    let n_obs_i32 = i32::try_from(csc_view.n_obs).map_err(|_| GpuError::ShapeMismatch {
        expected: "n_obs <= i32::MAX".to_string(),
        got: format!("n_obs = {}", csc_view.n_obs),
    })?;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(&csc_view.col_indptr)
            .arg(&csc_view.row_indices)
            .arg(&csc_view.data)
            .arg(cell_to_group_dev)
            .arg(cell_to_pos_dev)
            .arg(&this_group_id)
            .arg(slab)
            .arg(&n_cols_in_shard_i32)
            .arg(&shard_col_start_i32)
            .arg(&c0_i32)
            .arg(&chunk_size_i32)
            .arg(&n_perm_i32)
            .arg(&n_obs_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("csc_shard_to_gene_major_kernel: {e}")))?;

    Ok(())
}

/// Build a device-resident `cell_to_group` table from the same flat
/// `all_group_cells` + CSR-style `group_offsets` representation consumed by
/// [`gpu_de_pseudobulk_all_groups`]. Each entry of the returned buffer is
/// either `-1` (cell not in any group) or the group's id (`0..n_groups`)
/// — the inverse of the per-group cell list.
///
/// Used by [`gpu_de_pseudobulk_csr_direct`] (G4 v3 driver) to look up which
/// group a CSR shard row belongs to inside the kernel. Built once per DE
/// call and reused across the full chunk loop.
pub fn build_cell_to_group_dev(
    dev: &GpuDevice,
    all_group_cells_host: &[i32],
    group_offsets_host: &[i32],
    n_obs: usize,
) -> Result<CudaSlice<i32>, GpuError> {
    let mut cell_to_group = vec![-1i32; n_obs];
    if group_offsets_host.len() < 2 {
        return dev.htod_copy(&cell_to_group);
    }
    for g in 0..group_offsets_host.len() - 1 {
        let start = group_offsets_host[g] as usize;
        let end = group_offsets_host[g + 1] as usize;
        for &cell in &all_group_cells_host[start..end] {
            if cell >= 0 && (cell as usize) < n_obs {
                cell_to_group[cell as usize] = g as i32;
            }
        }
    }
    dev.htod_copy(&cell_to_group)
}

/// Build the cell→position-within-group inverse permutation used by the
/// post-#6 v3 scatter kernels.
///
/// Mirrors [`build_cell_to_group_dev`] in shape but writes the in-group
/// position (offset within the cell's group, `0..n_g_k - 1`) instead of
/// the group id. Each cell that belongs to a group `g` gets
/// `cell_to_pos[cell] = i - group_offsets[g]` where `i` is the cell's
/// position within `all_group_cells_host[group_offsets[g]..group_offsets[g+1]]`.
/// Cells not in any group keep the `-1` sentinel.
///
/// Used together with `cell_to_group_dev` + a launch-time `this_group_id`
/// to replace the previous K+1 per-pool `cell_to_pool` tables —
/// constant-in-K device memory at `2 × n_obs × 4` bytes.
pub fn build_cell_to_pos_dev(
    dev: &GpuDevice,
    all_group_cells_host: &[i32],
    group_offsets_host: &[i32],
    n_obs: usize,
) -> Result<CudaSlice<i32>, GpuError> {
    let mut cell_to_pos = vec![-1i32; n_obs];
    if group_offsets_host.len() < 2 {
        return dev.htod_copy(&cell_to_pos);
    }
    for g in 0..group_offsets_host.len() - 1 {
        let start = group_offsets_host[g] as usize;
        let end = group_offsets_host[g + 1] as usize;
        for (offset, &cell) in all_group_cells_host[start..end].iter().enumerate() {
            if cell >= 0 && (cell as usize) < n_obs {
                cell_to_pos[cell as usize] = offset as i32;
            }
        }
    }
    dev.htod_copy(&cell_to_pos)
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
/// `slab` is sorted in-place. `aux` is used as the multi-tile ping-pong
/// buffer and must hold at least `chunk_size × n_per_gene` f32 keys (the
/// caller must size it via [`GpuDeChunkScratch::ensure_aux_capacity`] when
/// `n_per_gene > GPU_DE_BLOCK_SORT_CAPACITY`). Fast-path callers can pass
/// any allocation for `aux` — it is not touched on the single-tile path.
///
/// Taking `slab` and `aux` as separate `&mut CudaSlice<f32>` arguments lets
/// callers thread two disjoint fields of `GpuDeChunkScratch` simultaneously
/// (e.g. `&mut scratch.ref_slab` + `&mut scratch.slab_aux`), which is
/// required by the G2 chunk-loop hoisting in pdex_ref / Wilcoxon drivers.
pub fn gpu_de_block_sort(
    dev: &GpuDevice,
    slab: &mut CudaSlice<f32>,
    aux: &mut CudaSlice<f32>,
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
    if aux.len() < n_elements {
        return Err(GpuError::ShapeMismatch {
            expected: format!(
                "aux buffer ≥ chunk_size * n_per_gene = {chunk_size} * {n_per_gene} = \
                 {n_elements} f32 keys — size it with gpu_de_aux_elems() against the \
                 LARGEST n_per_gene in the chunk loop, not the first sort's"
            ),
            got: format!("aux.len() = {}", aux.len()),
        });
    }

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
            gpu_de_merge_pass(dev, &*slab, aux, chunk_size, n_per_gene, run_size)?;
        } else {
            // pass 1, 3, 5, ... : aux -> slab.
            gpu_de_merge_pass(dev, &*aux, slab, chunk_size, n_per_gene, run_size)?;
        }
        n_passes += 1;
        run_size = run_size.saturating_mul(2);
    }

    // After an odd number of merge passes the final result lives in `aux`;
    // copy it back to `slab` so callers downstream (searchsorted, tie-term,
    // p-value) read from the canonical buffer.
    if !n_passes.is_multiple_of(2) {
        let mut slab_view = slab.slice_mut(..n_elements);
        let aux_view = aux.slice(..n_elements);
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

/// Threshold below which the host dispatches to the single-thread tie
/// kernels (`*_simple_kernel`) rather than the 256-thread block-cooperative
/// variants. The block-cooperative kernels carry ~10 µs of fixed overhead
/// (BlockReduce + cross-warp shmem relays + boundary stitch) per launch;
/// for small n_per_gene the work fits in microseconds and the overhead
/// dominates, so the simple kernel wins. Empirically the crossover sits
/// near n_per_gene ≈ 8000 on H100; aligned to the block-radix-sort
/// capacity (`BLOCK_THREADS × ITEMS_PER_THREAD = 8192`) to keep small/large
/// inputs cleanly separated at the same boundary the sort kernel uses.
pub const GPU_DE_TIE_BLOCK_THRESHOLD: usize = 8192;

/// Compute Σ(c^3 − c) per gene over a single sorted row. Writes
/// `tie_out[gene]` for `gene in 0..chunk_size`. Dispatches to the
/// single-thread `_simple_kernel` for `n_per_gene < GPU_DE_TIE_BLOCK_THRESHOLD`
/// and to the block-cooperative kernel above the threshold.
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
    let (kernel_name, block_dim) = if n_per_gene < GPU_DE_TIE_BLOCK_THRESHOLD {
        ("tie_term_sorted_simple_kernel", 1u32)
    } else {
        ("tie_term_sorted_kernel", 256u32)
    };
    let func = module
        .load_function(kernel_name)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("{kernel_name}: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
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
    .map_err(|e| GpuError::KernelLaunchFailed(format!("{kernel_name}: {e}")))?;
    Ok(())
}

/// Combined tie term over (ref + group): merge-walks two pre-sorted rows per
/// gene and writes the result into `tie_out`. Dispatches by `n_ref + n_g`
/// against `GPU_DE_TIE_BLOCK_THRESHOLD` — same crossover as the sorted-row
/// kernel.
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
    let (kernel_name, block_dim) = if n_ref + n_g < GPU_DE_TIE_BLOCK_THRESHOLD {
        ("combined_tie_term_simple_kernel", 1u32)
    } else {
        ("combined_tie_term_kernel", 256u32)
    };
    let func = module
        .load_function(kernel_name)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("{kernel_name}: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (chunk_size as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
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
    .map_err(|e| GpuError::KernelLaunchFailed(format!("{kernel_name}: {e}")))?;
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
/// Matches the CPU pdex_ref path
/// `wilcoxon_full_from_ranks(.., continuity=true)`: continuity-corrected
/// `z = max(|U − μ| − 0.5, 0)/σ`, `p = erfc(z / √2)` (upstream pdex /
/// numba_mwu `use_continuity=True`, scipy's default). This launcher is used
/// only by the pdex_ref GPU sequences; the Wilcoxon GPU path computes its
/// p-value elsewhere and stays uncorrected for scanpy parity. Output is
/// clipped to `[0, 1]`. Caller is responsible for setting p = 1.0 when either
/// group is empty (kernel handles `n1 == 0 || n2 == 0` defensively but the
/// chunk loop short-circuits before launch).
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

/// Default fraction of free VRAM the per-chunk DE working set may occupy.
///
/// The per-target-group pool slabs (`n_test × chunk × n_g_max` f32) dominate
/// that working set, so the gene chunk is clamped to keep the whole set under
/// this fraction. The remaining headroom covers fixed device buffers (indptr,
/// cell→group/pos tables, cuSPARSE/cuBLAS handles) and allocator fragmentation.
const GPU_DE_MEM_BUDGET_FRAC: f64 = 0.6;

/// Lowest gene chunk the budget clamp will fall to before declaring the
/// per-chunk working set un-fittable (caller then surfaces a clear error).
pub const GPU_DE_MIN_GENE_CHUNK: usize = 32;

/// Read the per-chunk DE VRAM budget fraction, honouring the
/// `SCX_GPU_DE_MEM_BUDGET_FRAC` env override. Values outside `(0, 0.95)` (or
/// unparseable) fall back to [`GPU_DE_MEM_BUDGET_FRAC`].
fn gpu_de_mem_budget_frac() -> f64 {
    // Cache the parsed env var: read once per process (it's an environment knob,
    // not expected to change mid-run) to avoid taking the `std::env` global lock
    // on the DE hot path.
    static FRAC: OnceLock<f64> = OnceLock::new();
    *FRAC.get_or_init(|| {
        std::env::var("SCX_GPU_DE_MEM_BUDGET_FRAC")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|f| *f > 0.0 && *f < 0.95)
            .unwrap_or(GPU_DE_MEM_BUDGET_FRAC)
    })
}

/// Whether the test-only `SCX_GPU_DE_PSEUDOBULK_FORCE_ATOMIC` override forces the
/// global-atomic CSC pseudobulk path. Cached on first read (this is called per
/// shard per chunk on the DE hot path, so a per-call `std::env` lock would be
/// wasteful). The override must therefore be set **before** the first DE op in
/// the process; production never sets it, and the parity test selects the atomic
/// path via `n_groups` rather than this flag.
pub fn gpu_de_force_atomic_pseudobulk() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| std::env::var_os("SCX_GPU_DE_PSEUDOBULK_FORCE_ATOMIC").is_some())
}

/// Whether `SCX_GPU_DE_REQUIRE_DETERMINISTIC=1` turns a fall-through to the
/// global-atomic CSC pseudobulk kernel into an error.
///
/// The inverse of [`gpu_de_force_atomic_pseudobulk`], which forces the
/// non-reproducible arm for testing. This one is for a caller who needs the
/// numbers to be reproducible and would rather be told than silently get the
/// atomic kernel because the device's shared memory happened not to fit.
/// Same `OnceLock` caching and the same "set it before the first DE op" rule.
pub fn gpu_de_require_deterministic() -> bool {
    static REQUIRE: OnceLock<bool> = OnceLock::new();
    *REQUIRE.get_or_init(|| std::env::var("SCX_GPU_DE_REQUIRE_DETERMINISTIC").as_deref() == Ok("1"))
}

/// The device's opt-in dynamic shared-memory limit, or the 48 KB hardware
/// default when it cannot be read.
///
/// Split out so [`csc_pseudobulk_block_dim`] takes it as a plain number and the
/// selection rule is testable without a device.
pub fn csc_pseudobulk_smem_limit(dev: &GpuDevice) -> usize {
    dev.max_dynamic_shared_mem_per_block()
        .unwrap_or(CSC_PSEUDOBULK_DEFAULT_SMEM_LIMIT)
}

/// The 48 KB per-block shared-memory floor every CUDA device provides without
/// an opt-in, used when the device attribute cannot be read.
pub const CSC_PSEUDOBULK_DEFAULT_SMEM_LIMIT: usize = 48 * 1024;

/// Which CSC pseudobulk kernel runs: `Some(block_dim)` for the deterministic
/// SMEM tree-reduce, `None` for the global-atomic fallback.
///
/// **This is the single selection rule**, called both by the launch site and by
/// the route stamp, so the provenance record cannot drift from the kernel that
/// actually ran. It is deliberately pure — `smem_limit` and `force_atomic` are
/// passed in rather than read from the device and the environment — so the
/// whole table is unit-testable on a CPU host.
///
/// The choice is **numerically visible and device-dependent**: the tree-reduce
/// has no atomics and is bit-reproducible run to run, the fallback is a global
/// f64 `atomicAdd` and is not. `max_dynamic_shared_mem_per_block` is 227 KB on
/// sm_90 and 163 KB on sm_80, so at `bx = 32` the ceiling is 908 groups on an
/// H100 and 652 on an A100 — the same file, the same call, different numbers
/// on a different card. That is why the answer is stamped (review §8.6) rather
/// than left implicit, and why `SCX_GPU_DE_REQUIRE_DETERMINISTIC=1` exists.
pub fn csc_pseudobulk_block_dim(
    n_groups: usize,
    smem_limit: usize,
    force_atomic: bool,
) -> Option<u32> {
    if force_atomic {
        return None;
    }
    let elem_bytes = std::mem::size_of::<f64>();
    [128u32, 64, 32]
        .into_iter()
        .find(|&bx| n_groups * bx as usize * elem_bytes <= smem_limit)
}

/// Device-scratch bytes consumed *per gene column* of a DE chunk, matching the
/// chunk-linear allocations of the v3 drivers (`GpuDeChunkScratch::new` + the
/// `ensure_*_capacity` grow calls). `next_power_of_two` mirrors the *span*
/// rounding those grow calls apply.
///
/// - `n_aux`      — ping-pong `aux` (f32), [`gpu_de_aux_span`] of the loop's
///   largest sort, which is 0 on the single-tile fast path.
/// - `n_ref`      — `ref_slab` (f32)
/// - `n_g_max`    — the dominant `per_tg_pool_slabs` (f32, ×`n_test`)
/// - `n_test`     — per-test-group `per_tg_pool_slabs` count and the `u/p/tie`
///   f64 staging buffers (×3)
/// - `n_slots`    — pseudobulk `sums` (f64)
///
/// Every term here is genuinely **chunk-linear** — i.e. it is allocated as
/// `chunk_max × <this>` by the corresponding `GpuDeChunkScratch::ensure_*_capacity`
/// call (`u/p/tie_per_group` are `[n_test × chunk_max]` via `ensure_per_group_capacity`;
/// `sums` is `[n_slots × chunk_max]` via `ensure_sums_capacity`; the `+3` covers the
/// `[chunk_max]` `tie_term`/`u_or_rank`/`p_values` scalars). None is a per-op constant,
/// so multiplying this whole value by the chunk size is correct, not an over-count.
///
/// Every f32 term now rounds its **span**, matching what the corresponding
/// `ensure_*_capacity` allocates. `n_aux` was the exception until review §8.13:
/// `ensure_aux_capacity` rounded the already-multiplied `chunk × span`, so the
/// real buffer could be up to 2× this charge and the clamp said `fits = true`
/// on a working set that then OOM'd. The `n_slab` and `group_slab` terms are
/// gone with the buffers themselves — nothing read either one.
pub fn gpu_de_per_gene_scratch_bytes(
    n_aux: usize,
    n_ref: usize,
    n_g_max: usize,
    n_test: usize,
    n_slots: usize,
) -> usize {
    let p2 = |x: usize| x.max(1).next_power_of_two();
    // f32 (4 bytes): aux + ref_slab + per_tg_pool (×n_test).
    //
    // `n_aux` is 0 unless some sort reaches the multi-tile path
    // ([`gpu_de_aux_span`]); `p2` of 0 would charge 1 element for a buffer that
    // is never allocated, so it is folded in raw and rounded only when non-zero.
    let aux_elems = if n_aux == 0 { 0 } else { p2(n_aux) };
    let f32_elems = aux_elems
        .saturating_add(p2(n_ref))
        .saturating_add(n_test.saturating_mul(p2(n_g_max)));
    // f64 (8 bytes): sums + 3×per-group (u/p/tie) + 3 per-gene scalars
    // (tie_term/u_or_rank/p_values).
    let f64_elems = p2(n_slots)
        .saturating_add(3usize.saturating_mul(p2(n_test)))
        .saturating_add(3);
    f32_elems
        .saturating_mul(4)
        .saturating_add(f64_elems.saturating_mul(8))
}

/// Per-gene span of [`GpuDeChunkScratch::slab_aux`]: `n_per_gene_max` when the
/// multi-tile sort path can be reached, and **0** when it cannot.
///
/// [`gpu_de_block_sort`] returns from the single-tile fast path without ever
/// touching `aux`, so at or below [`GPU_DE_BLOCK_SORT_CAPACITY`] the buffer is
/// dead weight — both in VRAM and in the [`gpu_de_per_gene_scratch_bytes`]
/// budget, where charging for it needlessly shrinks the gene chunk.
pub fn gpu_de_aux_span(n_per_gene_max: usize) -> usize {
    if n_per_gene_max > GPU_DE_BLOCK_SORT_CAPACITY {
        n_per_gene_max
    } else {
        0
    }
}

/// Element count [`GpuDeChunkScratch::slab_aux`] must hold for a DE chunk loop.
/// Zero when no sort in the loop can reach the multi-tile path — see
/// [`gpu_de_aux_span`].
///
/// `n_per_gene_max` is the **largest** `n_per_gene` that any
/// [`gpu_de_block_sort`] call in the loop will pass — not merely the first one.
/// Both v3 chunk sequences sort twice: the pool/reference slab, and then (in
/// ref-mode) each test group's own slab. So it is `max(pool_len, n_g_max)`, and
/// sizing it against the pool alone breaks on any reference smaller than the
/// largest test group — the group sort then hits the multi-tile path with an
/// aux built for the pool and fails with `ShapeMismatch`.
///
/// Kept next to [`gpu_de_per_gene_scratch_bytes`] because the two must agree:
/// that function's `n_aux` is the per-gene charge for this same buffer, so pass
/// it [`gpu_de_aux_span`] of the *same* `n_per_gene_max` used here. A caller
/// that widens one and not the other gets a VRAM budget that disagrees with
/// what it allocates — in one direction an under-count, in the other a chunk
/// shrunk to reserve a buffer that is never created.
pub fn gpu_de_aux_elems(chunk_size: usize, n_per_gene_max: usize) -> Result<usize, GpuError> {
    match gpu_de_aux_span(n_per_gene_max) {
        0 => Ok(0),
        span => de_alloc_elems(chunk_size, span),
    }
}

/// Element count [`GpuDeChunkScratch::ensure_aux_capacity`] actually allocates
/// for `(chunk_size, span)`: `chunk_size × next_power_of_two(span)`, and `0`
/// when the span is 0.
///
/// The twin of [`gpu_de_aux_elems`], which is the *requirement*
/// [`gpu_de_block_sort`] checks against (`chunk × span`, unrounded). This is
/// what gets allocated, and it is what [`gpu_de_per_gene_scratch_bytes`]'s
/// `p2(n_aux)` charge must equal once multiplied by the chunk — a free function
/// rather than an inline expression so a CPU-only test can hold the model and
/// the allocation to each other without a device.
pub fn gpu_de_aux_alloc_elems(chunk_size: usize, span: usize) -> Result<usize, GpuError> {
    if span == 0 {
        return Ok(0);
    }
    de_alloc_elems(chunk_size, span.next_power_of_two())
}

/// Pure budget clamp (no device access — unit-testable).
///
/// Returns `(chunk, fits)`: the largest gene chunk ≤ `requested` whose
/// per-chunk scratch (`per_gene_bytes` × chunk) fits `free_bytes × frac`,
/// floored at [`GPU_DE_MIN_GENE_CHUNK`]. `fits` is false only when even the
/// floor chunk exceeds the budget — the caller should then surface a clear
/// dimension-naming error rather than let the allocation OOM.
fn clamp_chunk_for_budget(
    requested: usize,
    per_gene_bytes: usize,
    free_bytes: usize,
    frac: f64,
) -> (usize, bool) {
    let requested = requested.max(1);
    let per_gene_bytes = per_gene_bytes.max(1);
    let budget = (free_bytes as f64 * frac) as usize;
    let max_chunk = (budget / per_gene_bytes).max(1);
    if max_chunk >= requested {
        (requested, true)
    } else if max_chunk >= GPU_DE_MIN_GENE_CHUNK {
        (max_chunk, true)
    } else {
        (GPU_DE_MIN_GENE_CHUNK, false)
    }
}

/// Clamp a requested gene chunk so the per-chunk DE working set fits a budget
/// fraction of free VRAM. Never increases `requested`. When free memory can't
/// be queried, returns `(requested, true)` (don't clamp what we can't measure).
///
/// See [`clamp_chunk_for_budget`] / [`gpu_de_per_gene_scratch_bytes`]. Bounds
/// every chunk-linear scratch allocation (dominated by `per_tg_pool_slabs`),
/// closing the census-scale OOM where a user `gene_chunk_size` (or the auto
/// sizer) ignored the `n_test × n_g_max` term.
pub fn gpu_de_budget_gene_chunk(
    dev: &GpuDevice,
    requested: usize,
    per_gene_bytes: usize,
) -> (usize, bool) {
    let Ok((free_bytes, _total)) = dev.free_memory() else {
        return (requested.max(1), true);
    };
    if free_bytes == 0 {
        return (requested.max(1), true);
    }
    clamp_chunk_for_budget(
        requested,
        per_gene_bytes,
        free_bytes,
        gpu_de_mem_budget_frac(),
    )
}

#[cfg(test)]
#[path = "gpu_diffexp_tests.rs"]
mod tests;
