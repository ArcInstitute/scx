use super::*;

#[test]
fn test_de_alloc_elems_rejects_overflow() {
    // Normal dimensions multiply cleanly.
    assert_eq!(de_alloc_elems(1_000, 50).unwrap(), 50_000);
    // A product that overflows usize is rejected up-front with a clear
    // ShapeMismatch instead of wrapping into an under-sized allocation.
    let err = de_alloc_elems(usize::MAX, 2).unwrap_err();
    assert!(matches!(err, GpuError::ShapeMismatch { .. }), "got {err:?}");
}

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

#[test]
fn test_per_gene_scratch_bytes_dominated_by_per_tg_pool() {
    // The per-target-group pool (n_test × next_pow2(n_g_max) f32) should
    // dominate the per-gene cost. n_g_max=10_000 → next_pow2 = 16_384.
    let n_test = 40;
    let n_g_max = 10_000;
    let bytes = gpu_de_per_gene_scratch_bytes(
        /* n_pool_max */ 16_000,
        /* n_ref */ 16_000,
        n_g_max,
        n_test,
        /* n_slots */ n_test + 1,
    );
    // Lower bound: just the per-tg pool term (n_test × 16384 × 4 bytes).
    let per_tg = n_test * 16_384 * 4;
    assert!(
        bytes >= per_tg,
        "per-gene bytes {bytes} should cover the per-tg pool floor {per_tg}"
    );
    // And it must be the dominant term (> half the total).
    assert!(
        bytes < 2 * per_tg,
        "per-tg pool should dominate; got {bytes}"
    );
}

#[test]
fn test_clamp_chunk_for_budget() {
    // Reproduce the B8 scenario: requested 4000-gene chunk, large per-tg pool,
    // and a partially-occupied GPU whose free VRAM the full chunk would blow
    // past (the report OOM'd because the backed reader / shard decode / index
    // tables already held VRAM, so free ≪ 80 GB).
    let per_gene = gpu_de_per_gene_scratch_bytes(16_000, 16_000, 10_000, 40, 41);
    let free = 8 * 1024 * 1024 * 1024usize; // 8 GB free at DE time

    // 4000 × per_gene at frac=0.6 of 8 GB does not fit → clamps below 4000,
    // but stays well above the floor (fits=true).
    let (chunk, fits) = clamp_chunk_for_budget(4000, per_gene, free, 0.6);
    assert!(fits, "should still fit at a reduced chunk with 8 GB free");
    assert!(
        chunk < 4000,
        "expected clamp below requested 4000, got {chunk}"
    );
    assert!(
        chunk >= GPU_DE_MIN_GENE_CHUNK,
        "clamped chunk {chunk} below the floor"
    );
    // The clamped chunk's working set must fit the budget.
    assert!((chunk * per_gene) as f64 <= free as f64 * 0.6);

    // Ample free VRAM → no clamp, returns the requested chunk unchanged.
    let big_free = 80 * 1024 * 1024 * 1024usize;
    let (chunk2, fits2) = clamp_chunk_for_budget(4000, per_gene, big_free, 0.6);
    assert!(fits2);
    assert_eq!(chunk2, 4000, "ample VRAM should leave the chunk unclamped");

    // Pathological: per-tg pool so large even the floor chunk can't fit on a
    // tiny device → fits=false (caller surfaces a clear error).
    let huge_per_gene = gpu_de_per_gene_scratch_bytes(0, 0, 4_000_000, 200, 201);
    let small_free = 4 * 1024 * 1024 * 1024usize; // 4 GB
    let (chunk3, fits3) = clamp_chunk_for_budget(4000, huge_per_gene, small_free, 0.6);
    assert!(
        !fits3,
        "floor chunk should not fit; per_gene={huge_per_gene}"
    );
    assert_eq!(chunk3, GPU_DE_MIN_GENE_CHUNK);
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
            s += c as f64 * c as f64 * c as f64 - c as f64;
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

/// CPU emulator of the GPU `combined_tie_term_kernel` algorithm (per-thread
/// merge-walk + boundary stitch). Mirrors the kernel logic line-by-line so
/// we can debug algorithmic vs CUDA bugs.
fn cpu_emulator_combined_tie(
    sorted_ref: &[f32],
    sorted_group: &[f32],
    block_threads: usize,
) -> f64 {
    let n_ref = sorted_ref.len();
    let n_g = sorted_group.len();
    let total = n_ref + n_g;
    if total == 0 {
        return 0.0;
    }
    let per_thread = total.div_ceil(block_threads);

    let co_rank = |diag: usize| -> usize {
        let mut i_lo = if diag > n_g { diag - n_g } else { 0 };
        let mut i_hi = diag.min(n_ref);
        while i_lo < i_hi {
            let i = (i_lo + i_hi) / 2;
            let j = diag - i;
            if i > 0 && j < n_g && sorted_ref[i - 1] > sorted_group[j] {
                i_hi = i;
            } else if i < n_ref && j > 0 && sorted_group[j - 1] >= sorted_ref[i] {
                i_lo = i + 1;
            } else {
                return i;
            }
        }
        i_lo
    };

    let mut heads_v = vec![0.0f32; block_threads];
    let mut heads_c = vec![0i64; block_threads];
    let mut tails_v = vec![0.0f32; block_threads];
    let mut tails_c = vec![0i64; block_threads];
    let mut total_inner = 0.0f64;

    for tid in 0..block_threads {
        let mut diag_start = tid * per_thread;
        let mut diag_end = diag_start + per_thread;
        if diag_start > total {
            diag_start = total;
        }
        if diag_end > total {
            diag_end = total;
        }
        let i_start = co_rank(diag_start);
        let j_start = diag_start - i_start;
        let i_end = co_rank(diag_end);
        let j_end = diag_end - i_end;

        let mut i = i_start;
        let mut j = j_start;
        let mut n_runs = 0u32;
        let mut head_value = 0.0f32;
        let mut head_count = 0i64;
        let mut last_value = 0.0f32;
        let mut last_count = 0i64;
        let mut inner_sum = 0.0f64;

        while i < i_end || j < j_end {
            let v = if j >= j_end {
                sorted_ref[i]
            } else if i >= i_end {
                sorted_group[j]
            } else {
                sorted_ref[i].min(sorted_group[j])
            };
            let mut c = 0i64;
            while i < i_end && sorted_ref[i] == v {
                i += 1;
                c += 1;
            }
            while j < j_end && sorted_group[j] == v {
                j += 1;
                c += 1;
            }
            if n_runs == 0 {
                head_value = v;
                head_count = c;
            } else if n_runs >= 2 && last_count > 1 {
                inner_sum +=
                    last_count as f64 * last_count as f64 * last_count as f64 - last_count as f64;
            }
            last_value = v;
            last_count = c;
            n_runs += 1;
        }

        heads_v[tid] = head_value;
        heads_c[tid] = head_count;
        if n_runs == 0 {
            tails_v[tid] = 0.0;
            tails_c[tid] = 0;
        } else {
            tails_v[tid] = last_value;
            tails_c[tid] = last_count;
        }
        total_inner += inner_sum;
    }

    let mut t = 0;
    while t < block_threads && heads_c[t] == 0 {
        t += 1;
    }
    if t >= block_threads {
        return total_inner;
    }

    let mut boundary_sum = 0.0f64;
    let mut open_value;
    let mut open_count;

    if heads_v[t] == tails_v[t] {
        open_value = heads_v[t];
        open_count = heads_c[t];
    } else {
        let c = heads_c[t];
        if c > 1 {
            boundary_sum += c as f64 * c as f64 * c as f64 - c as f64;
        }
        open_value = tails_v[t];
        open_count = tails_c[t];
    }

    for u in (t + 1)..block_threads {
        if heads_c[u] == 0 {
            continue;
        }
        let u_single = heads_v[u] == tails_v[u];

        if heads_v[u] == open_value {
            open_count += heads_c[u];
            if !u_single {
                let c = open_count;
                if c > 1 {
                    boundary_sum += c as f64 * c as f64 * c as f64 - c as f64;
                }
                open_value = tails_v[u];
                open_count = tails_c[u];
            }
        } else {
            let c = open_count;
            if c > 1 {
                boundary_sum += c as f64 * c as f64 * c as f64 - c as f64;
            }
            if !u_single {
                let c2 = heads_c[u];
                if c2 > 1 {
                    boundary_sum += c2 as f64 * c2 as f64 * c2 as f64 - c2 as f64;
                }
                open_value = tails_v[u];
                open_count = tails_c[u];
            } else {
                open_value = heads_v[u];
                open_count = heads_c[u];
            }
        }
    }
    let c = open_count;
    if c > 1 {
        boundary_sum += c as f64 * c as f64 * c as f64 - c as f64;
    }

    total_inner + boundary_sum
}

/// Sweep small fixtures to verify the CPU emulator (which mirrors the
/// kernel algorithm line-for-line) matches brute-force tie computation
/// across a range of block-threads, unique-value counts, and sizes.
/// Catches algorithmic bugs (e.g. non-monotonic `merge_path_co_rank`)
/// independently of CUDA. If this passes, parity issues are downstream
/// of the algorithm.
#[test]
fn test_cpu_emulator_combined_tie_sweeps() {
    // Sweep block-threads × unique-values × sizes. The
    // `merge_path_co_rank` monotonicity bug fixed during G1.9
    // surfaces only on inputs with ties spanning the diagonal AND
    // a partition fine enough that two adjacent threads share a
    // tied boundary — small bt with mod-N keys is where it shows.
    for n_uniq in [5u32, 10, 20] {
        for &(n_ref, n_g) in &[(20usize, 15), (60, 40), (200, 150)] {
            let mut state: u64 = 0xFEEDFACE;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u32
            };
            let mut r: Vec<f32> = (0..n_ref).map(|_| (next() % n_uniq) as f32).collect();
            cpu_sort_ascending(&mut r);
            let mut g: Vec<f32> = (0..n_g).map(|_| (next() % n_uniq) as f32).collect();
            cpu_sort_ascending(&mut g);

            for bt in [4usize, 8, 16, 32, 64] {
                let brute = cpu_combined_tie(&r, &g);
                let emu = cpu_emulator_combined_tie(&r, &g, bt);
                assert_eq!(
                    emu, brute,
                    "n_uniq={n_uniq} n_ref={n_ref} n_g={n_g} bt={bt}: emu disagrees"
                );
            }
        }
    }
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
    scratch
        .ensure_aux_capacity(&dev, chunk_size * n_ref)
        .unwrap();
    gpu_de_block_sort(
        &dev,
        &mut d_ref_slab,
        &mut scratch.slab_aux,
        chunk_size,
        n_ref,
    )
    .unwrap();
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
    scratch.ensure_aux_capacity(&dev, chunk_size * n_g).unwrap();
    gpu_de_block_sort(
        &dev,
        &mut d_group_slab,
        &mut scratch.slab_aux,
        chunk_size,
        n_g,
    )
    .unwrap();
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
    scratch
        .ensure_aux_capacity(&dev, chunk_size * n_per_gene)
        .unwrap();
    gpu_de_block_sort(
        &dev,
        &mut d_slab,
        &mut scratch.slab_aux,
        chunk_size,
        n_per_gene,
    )
    .unwrap();
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

    let mut state: u64 = 0xFEED_BEEF_2026;
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
    scratch
        .ensure_aux_capacity(&dev, chunk_size * n_per_gene)
        .unwrap();
    gpu_de_block_sort(
        &dev,
        &mut d_slab,
        &mut scratch.slab_aux,
        chunk_size,
        n_per_gene,
    )
    .unwrap();
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
    scratch
        .ensure_aux_capacity(&dev, chunk_size * n_per_gene)
        .unwrap();
    gpu_de_block_sort(
        &dev,
        &mut d_slab,
        &mut scratch.slab_aux,
        chunk_size,
        n_per_gene,
    )
    .unwrap();
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

/// G2 regression: `ensure_*_capacity` only allocates on initial grow and
/// on size increase; same / smaller requests are no-ops. The chunk loops
/// in the pdex_ref / Wilcoxon GPU drivers rely on
/// this so they can pre-grow once before the loop and never re-allocate
/// per chunk.
#[test]
fn test_scratch_ensure_capacity_no_realloc_on_same_or_smaller() {
    let dev = require_gpu!();
    let n_obs = 100;
    let chunk_max = 8;
    let n_pool_initial = 10;

    let mut scratch = GpuDeChunkScratch::new(&dev, n_obs, chunk_max, n_pool_initial).unwrap();
    // Construction does not call `ensure_*` — alloc_count starts at zero.
    assert_eq!(scratch.alloc_count(), 0, "fresh scratch has no grow events");

    // First grow of ref_slab beyond the initial slab capacity.
    scratch.ensure_ref_slab_capacity(&dev, 64).unwrap();
    assert_eq!(scratch.alloc_count(), 1, "first ref_slab grow");

    // Same size — must be no-op.
    scratch.ensure_ref_slab_capacity(&dev, 64).unwrap();
    assert_eq!(scratch.alloc_count(), 1, "same ref_slab size: no realloc");

    // Smaller size — also no-op (capacity is monotonic non-decreasing).
    scratch.ensure_ref_slab_capacity(&dev, 32).unwrap();
    assert_eq!(
        scratch.alloc_count(),
        1,
        "smaller ref_slab size: no realloc"
    );

    // First grow of group_slab.
    scratch.ensure_group_slab_capacity(&dev, 50).unwrap();
    assert_eq!(scratch.alloc_count(), 2, "first group_slab grow");

    // Repeat — no-op.
    scratch.ensure_group_slab_capacity(&dev, 50).unwrap();
    assert_eq!(scratch.alloc_count(), 2, "same group_slab: no realloc");

    // First grow of sums.
    scratch.ensure_sums_capacity(&dev, 4).unwrap();
    assert_eq!(scratch.alloc_count(), 3, "first sums grow");

    // First grow of aux.
    scratch.ensure_aux_capacity(&dev, 1024).unwrap();
    assert_eq!(scratch.alloc_count(), 4, "first aux grow");

    // Another grow on ref_slab past current capacity bumps alloc count.
    // ref_slab grew to next_power_of_two(64) = 64, so 96 forces a grow.
    let initial = scratch.alloc_count();
    scratch.ensure_ref_slab_capacity(&dev, 96).unwrap();
    assert!(
        scratch.alloc_count() > initial,
        "growing ref_slab past power-of-two boundary must reallocate"
    );

    // The whole point: a chunk loop that calls ensure_* once up front
    // followed by N iterations of buffer reuse only sees a constant
    // alloc_count.
    let frozen = scratch.alloc_count();
    for _ in 0..16 {
        scratch.ensure_ref_slab_capacity(&dev, 32).unwrap();
        scratch.ensure_group_slab_capacity(&dev, 50).unwrap();
        scratch.ensure_sums_capacity(&dev, 4).unwrap();
        scratch.ensure_aux_capacity(&dev, 1024).unwrap();
    }
    assert_eq!(
        scratch.alloc_count(),
        frozen,
        "16 iterations of same/smaller ensure_* must not grow",
    );
}

/// `gpu_de_scatter_shard_to_dense` reproduces, on device, the dense
/// `[n_obs × sz]` row-major chunk for a given column range directly from
/// a CSR shard source. Three
/// fixtures cover: (a) full column range, (b) middle column subrange,
/// (c) an empty shard interleaved with non-empty shards.
#[test]
fn test_csr_shard_to_dense_chunk_parity() {
    use crate::gpu_shard_source::{GpuShardSource, RawGpuShardSource};
    use scx_format_io::ShardSource;
    use scx_sparse::ScxCsr;

    let dev = require_gpu!();

    // Tiny in-memory shard source for the test. Shard 0 has 3 rows
    // with ties + missing columns; shard 1 has 0 rows (empty);
    // shard 2 has 4 rows with one row entirely outside the chunk.
    struct InMemorySource {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
    }
    impl ShardSource for InMemorySource {
        fn n_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

    // 8 columns; each row written explicitly so the parity comparison
    // also catches col→(col-c0) miscalculation.
    let n_vars = 8usize;
    let shard0 = ScxCsr::new_unchecked(
        (3, n_vars),
        vec![0i64, 3, 5, 7],
        vec![0i32, 3, 6, 1, 5, 2, 7],
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    );
    // Empty shard: indptr length n_rows+1 = 1, no nonzeros. Note the
    // RawGpuShardSource driver itself skips a `csr.n_rows() == 0`
    // shard before invoking the callback, so this shard advances
    // `global_row` by 0 — the next shard's global_row stays correct.
    let shard1 = ScxCsr::new_unchecked((0, n_vars), vec![0i64], vec![], vec![]);
    let shard2 = ScxCsr::new_unchecked(
        (4, n_vars),
        vec![0i64, 2, 2, 4, 6],
        vec![0i32, 4, 2, 7, 1, 3],
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
    );
    let shards = vec![shard0.clone(), shard1.clone(), shard2.clone()];
    // total rows in the dense matrix: 3 + 0 + 4 = 7
    let n_obs = 7usize;
    let src = InMemorySource {
        shards: shards.clone(),
        n_obs,
        n_vars,
    };

    // Reference dense `[n_obs × sz]` built on host for a given column
    // range. Matches what `gpu_de_scatter_shard_to_dense` writes on
    // device.
    let host_reference = |c0: usize, c1: usize| -> Vec<f32> {
        let sz = c1 - c0;
        let mut buf = vec![0.0f32; n_obs * sz];
        let mut global_row = 0usize;
        for shard in &shards {
            let n_rows = shard.n_rows();
            for r in 0..n_rows {
                let s = shard.indptr[r] as usize;
                let e = shard.indptr[r + 1] as usize;
                for k in s..e {
                    let col = shard.indices[k] as usize;
                    if col >= c0 && col < c1 {
                        buf[(global_row + r) * sz + (col - c0)] = shard.data[k];
                    }
                }
            }
            global_row += n_rows;
        }
        buf
    };

    // Run the GPU scatter for a (c0, c1) range against the host
    // reference. The dense buffer is allocated freshly each
    // sub-test to verify the zeroed-prefix contract.
    let run_case = |c0: usize, c1: usize| {
        let sz = c1 - c0;
        let mut gpu_src = RawGpuShardSource::new(&dev, &src).unwrap();
        let mut dense = dev.alloc_zeros::<f32>(n_obs * sz).unwrap();
        // Zero is already the alloc_zeros postcondition; a real
        // chunked driver re-zeros each iteration via memset_zeros.

        let mut global_row = 0usize;
        gpu_src
            .for_each_gpu_shard(|_idx, slot| {
                let view = slot.view();
                let n_rows = view.shape.0;
                crate::gpu_diffexp::gpu_de_scatter_shard_to_dense(
                    &dev, &view, &mut dense, global_row, sz, c0, c1,
                )?;
                global_row += n_rows;
                Ok(())
            })
            .unwrap();
        dev.synchronize().unwrap();

        let mut host_actual = vec![0.0f32; n_obs * sz];
        dev.stream().memcpy_dtoh(&dense, &mut host_actual).unwrap();
        dev.synchronize().unwrap();

        let host_expected = host_reference(c0, c1);
        assert_eq!(
            host_actual, host_expected,
            "shard-to-dense scatter mismatch for c0={c0}, c1={c1}"
        );
    };

    // (a) full column range
    run_case(0, n_vars);
    // (b) middle subrange — excludes col 0 and col 7, includes ties at col 1..6
    run_case(1, 6);
    // (c) narrow subrange — only one shard contributes to col 4
    run_case(4, 5);
    // (d) empty intersection — should leave dense fully zero
    run_case(0, 0); // c0 == c1 short-circuits in the wrapper
}

// ----- G1.9: warp-parallel tie-term kernel parity -----
//
// The kernels are block-cooperative segmented reductions over equal-key
// runs; tie correction is exact-integer arithmetic, so parity vs the
// CPU reference must hold bit-for-bit (no tolerance).

fn run_tie_term_sorted(dev: &GpuDevice, rows: &[Vec<f32>]) -> Vec<f64> {
    let chunk_size = rows.len();
    let n_per_gene = rows[0].len();
    for r in rows {
        assert_eq!(r.len(), n_per_gene);
    }
    let mut flat = Vec::with_capacity(chunk_size * n_per_gene);
    for r in rows {
        flat.extend_from_slice(r);
    }
    let d_slab = dev.htod_copy(&flat).unwrap();
    let mut d_tie = dev.alloc_zeros::<f64>(chunk_size).unwrap();
    gpu_de_tie_term(dev, &d_slab, &mut d_tie, chunk_size, n_per_gene).unwrap();
    dev.synchronize().unwrap();
    dev.dtoh_copy(&d_tie).unwrap()
}

fn run_combined_tie(dev: &GpuDevice, ref_rows: &[Vec<f32>], group_rows: &[Vec<f32>]) -> Vec<f64> {
    let chunk_size = ref_rows.len();
    assert_eq!(group_rows.len(), chunk_size);
    let n_ref = ref_rows[0].len();
    let n_g = group_rows[0].len();
    for r in ref_rows {
        assert_eq!(r.len(), n_ref);
    }
    for g in group_rows {
        assert_eq!(g.len(), n_g);
    }
    let mut ref_flat = Vec::with_capacity(chunk_size * n_ref);
    for r in ref_rows {
        ref_flat.extend_from_slice(r);
    }
    let mut g_flat = Vec::with_capacity(chunk_size * n_g);
    for g in group_rows {
        g_flat.extend_from_slice(g);
    }
    let d_ref = dev.htod_copy(&ref_flat).unwrap();
    let d_g = dev.htod_copy(&g_flat).unwrap();
    let mut d_tie = dev.alloc_zeros::<f64>(chunk_size).unwrap();
    gpu_de_combined_tie_term(dev, &d_ref, &d_g, &mut d_tie, chunk_size, n_ref, n_g).unwrap();
    dev.synchronize().unwrap();
    dev.dtoh_copy(&d_tie).unwrap()
}

/// Strictly increasing row → no ties → tie term must be exactly 0.
#[test]
fn test_tie_term_sorted_no_ties() {
    let dev = require_gpu!();
    let n = 5000usize;
    let row: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let got = run_tie_term_sorted(&dev, std::slice::from_ref(&row));
    assert_eq!(got[0], 0.0);
    assert_eq!(got[0], cpu_tie_term(&row));
}

/// All-tied row of constant value → exactly one run of size `n`,
/// tie term = n³ − n.
#[test]
fn test_tie_term_sorted_all_tied() {
    let dev = require_gpu!();
    let n = 4096usize;
    let row = vec![7.0f32; n];
    let got = run_tie_term_sorted(&dev, std::slice::from_ref(&row));
    let n_f64 = n as f64;
    assert_eq!(got[0], n_f64 * n_f64 * n_f64 - n_f64);
    assert_eq!(got[0], cpu_tie_term(&row));
}

/// Large sorted row with synthetic ties (LCG keys mod 100). Exercises
/// the post-G1.5 `n_per_gene` regime (≫ 8192) where the old single-thread
/// walk was the bottleneck. 4 genes × 50_000 cells covers the multi-block
/// dispatch as well.
#[test]
fn test_tie_term_sorted_large_with_ties() {
    let dev = require_gpu!();
    let chunk_size = 4usize;
    let n = 50_000usize;
    let mut state: u64 = 0xC0DEFACE;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut rows: Vec<Vec<f32>> = (0..chunk_size)
        .map(|_| {
            let mut r: Vec<f32> = (0..n).map(|_| (next() % 100) as f32).collect();
            cpu_sort_ascending(&mut r);
            r
        })
        .collect();
    // Force gene 1 to have a single dominant run that crosses many
    // per-thread slice boundaries.
    for v in rows[1].iter_mut().take(n / 2) {
        *v = 3.0;
    }
    cpu_sort_ascending(&mut rows[1]);

    let got = run_tie_term_sorted(&dev, &rows);
    for (g, r) in rows.iter().enumerate() {
        assert_eq!(got[g], cpu_tie_term(r), "tie mismatch on gene {g}");
    }
}

/// A single run of equal values that straddles the per-thread slice
/// boundaries inside one block. With `TIE_BLOCK_THREADS = 256` and
/// `n_per_gene = 600`, `per_thread = 3`, so a run from position 100 to
/// 400 spans ~100 per-thread slices. Stitching must merge them.
#[test]
fn test_tie_term_sorted_run_spans_tiles() {
    let dev = require_gpu!();
    let n = 600usize;
    // [0..100): strictly increasing; [100..400): constant value 1000; [400..600): strictly increasing
    let mut row = vec![0.0f32; n];
    for (i, v) in row.iter_mut().enumerate().take(100) {
        *v = i as f32;
    }
    for v in row.iter_mut().skip(100).take(300) {
        *v = 1000.0;
    }
    for (k, v) in row.iter_mut().skip(400).enumerate() {
        *v = 1001.0 + k as f32;
    }
    // Already sorted: 0..99 < 1000 (× 300) < 1001..1200.
    let got = run_tie_term_sorted(&dev, std::slice::from_ref(&row));
    // Expected: only the 300-long run contributes: 300³ − 300.
    let expected = 300.0f64 * 300.0 * 300.0 - 300.0;
    assert_eq!(got[0], expected);
    assert_eq!(got[0], cpu_tie_term(&row));
}

/// Combined tie on a large fixture — `n_ref = 30_000`, `n_g = 20_000`,
/// deterministic ties, exercises merge-path partitioning across the
/// full block.
#[test]
fn test_combined_tie_term_large_with_ties() {
    let dev = require_gpu!();
    let chunk_size = 4usize;
    let n_ref = 30_000usize;
    let n_g = 20_000usize;
    let mut state: u64 = 0xFEEDFACE;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut ref_rows: Vec<Vec<f32>> = (0..chunk_size)
        .map(|_| {
            let mut r: Vec<f32> = (0..n_ref).map(|_| (next() % 80) as f32).collect();
            cpu_sort_ascending(&mut r);
            r
        })
        .collect();
    let mut group_rows: Vec<Vec<f32>> = (0..chunk_size)
        .map(|_| {
            let mut g: Vec<f32> = (0..n_g).map(|_| (next() % 80) as f32).collect();
            cpu_sort_ascending(&mut g);
            g
        })
        .collect();
    // Force gene 2 to have one heavily dominant value shared across both
    // streams (most thread slices end up as single-run with the same key).
    for v in ref_rows[2].iter_mut().take(n_ref * 3 / 4) {
        *v = 42.0;
    }
    cpu_sort_ascending(&mut ref_rows[2]);
    for v in group_rows[2].iter_mut().take(n_g * 3 / 4) {
        *v = 42.0;
    }
    cpu_sort_ascending(&mut group_rows[2]);

    let got = run_combined_tie(&dev, &ref_rows, &group_rows);
    for g in 0..chunk_size {
        let expected = cpu_combined_tie(&ref_rows[g], &group_rows[g]);
        assert_eq!(got[g], expected, "combined tie mismatch on gene {g}");
    }
}

/// A run spanning multiple merge-path thread partitions: both ref and
/// group contain the same dominant value across many positions, so
/// adjacent thread slices each see a single-run of the same key.
/// Boundary stitching must merge all of them into one long run.
#[test]
fn test_combined_tie_term_run_spans_threads() {
    let dev = require_gpu!();
    // n_ref + n_g = 600 → per_thread = 3 → many adjacent slices that
    // all see value 5.0.
    let ref_row = vec![5.0f32; 300];
    let mut group_row = vec![5.0f32; 300];
    // Add a few non-tied entries at the ends so head/tail differ from
    // the middle on some thread slices.
    group_row[0] = 0.0;
    group_row[299] = 10.0;
    // Sort to maintain pre-sort invariant required by the kernel.
    let mut group_sorted = group_row.clone();
    cpu_sort_ascending(&mut group_sorted);

    let got = run_combined_tie(
        &dev,
        std::slice::from_ref(&ref_row),
        std::slice::from_ref(&group_sorted),
    );
    let expected = cpu_combined_tie(&ref_row, &group_sorted);
    assert_eq!(got[0], expected);
    // Sanity: there are 598 copies of 5.0 in the combined sort.
    let c = 598i64;
    let manual_tie = c as f64 * c as f64 * c as f64 - c as f64;
    assert_eq!(got[0], manual_tie);
}

/// Empty group: combined tie must equal `tie_term_sorted_kernel` on
/// the ref alone (single-stream fallthrough via merge_path_co_rank).
#[test]
fn test_combined_tie_term_empty_group() {
    let dev = require_gpu!();
    let mut state: u64 = 0xBEEF;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let n_ref = 1024usize;
    let mut ref_row: Vec<f32> = (0..n_ref).map(|_| (next() % 64) as f32).collect();
    cpu_sort_ascending(&mut ref_row);
    let group_row: Vec<f32> = Vec::new();

    let got_combined = run_combined_tie(
        &dev,
        std::slice::from_ref(&ref_row),
        std::slice::from_ref(&group_row),
    );
    let got_sorted = run_tie_term_sorted(&dev, std::slice::from_ref(&ref_row));
    let expected = cpu_tie_term(&ref_row);
    assert_eq!(got_combined[0], expected);
    assert_eq!(got_combined[0], got_sorted[0]);
}

/// Lock the simple↔block-cooperative dispatch crossover for the sorted-row
/// tie kernel. `gpu_de_tie_term` switches at `n_per_gene == 8192`, so
/// 8191 hits the simple kernel and 8192/8193 hit the cooperative one.
/// A regression on either side would not surface against the existing
/// well-above (50k) and well-below (≤4k) tests.
#[test]
fn test_tie_term_sorted_threshold_boundary() {
    let dev = require_gpu!();
    for &n in &[
        GPU_DE_TIE_BLOCK_THRESHOLD - 1,
        GPU_DE_TIE_BLOCK_THRESHOLD,
        GPU_DE_TIE_BLOCK_THRESHOLD + 1,
    ] {
        let mut state: u64 = 0xB0DA_C0DE_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut row: Vec<f32> = (0..n).map(|_| (next() % 64) as f32).collect();
        cpu_sort_ascending(&mut row);
        let got = run_tie_term_sorted(&dev, std::slice::from_ref(&row));
        assert_eq!(got[0], cpu_tie_term(&row), "tie mismatch at n_per_gene={n}");
    }
}

/// Lock the simple↔block-cooperative dispatch crossover for the combined
/// tie kernel. Dispatch is by `n_ref + n_g`. Pairs are chosen so the
/// sum spans the threshold.
#[test]
fn test_combined_tie_term_threshold_boundary() {
    let dev = require_gpu!();
    for &(n_ref, n_g) in &[(4096usize, 4095usize), (4096, 4096), (4097, 4096)] {
        let mut state: u64 = 0xCAFE_F00D_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut r: Vec<f32> = (0..n_ref).map(|_| (next() % 48) as f32).collect();
        cpu_sort_ascending(&mut r);
        let mut g: Vec<f32> = (0..n_g).map(|_| (next() % 48) as f32).collect();
        cpu_sort_ascending(&mut g);
        let got = run_combined_tie(&dev, std::slice::from_ref(&r), std::slice::from_ref(&g));
        assert_eq!(
            got[0],
            cpu_combined_tie(&r, &g),
            "combined tie mismatch at (n_ref={n_ref}, n_g={n_g})"
        );
    }
}

/// Regression for the i64-cube-overflow bug in `c³ − c`. With a single
/// all-tied run of `n = 2_100_000`, `n³` is ~9.26 × 10¹⁸ — strictly
/// above i64::MAX (~9.22 × 10¹⁸). If the kernel ever reverts to
/// `(double)(c * c * c - c)`, the multiplication overflows i64 before
/// the cast and this test catches it. Expected value is computed in
/// f64 (exact integer for `n ≤ 2^53`).
#[test]
fn test_tie_term_sorted_overflow_regression() {
    let dev = require_gpu!();
    let n = 2_100_000usize;
    let row = vec![1.0f32; n];
    let got = run_tie_term_sorted(&dev, std::slice::from_ref(&row));
    let n_f64 = n as f64;
    let expected = n_f64 * n_f64 * n_f64 - n_f64;
    assert_eq!(got[0], expected);
}

/// Same overflow regression for the combined-tie kernel. The merge of
/// two all-`1.0` streams is one run of length `n_ref + n_g = 2_200_000`,
/// which cubes to ~1.06 × 10¹⁹ — well past i64::MAX.
#[test]
fn test_combined_tie_term_overflow_regression() {
    let dev = require_gpu!();
    let n_ref = 1_100_000usize;
    let n_g = 1_100_000usize;
    let ref_row = vec![1.0f32; n_ref];
    let group_row = vec![1.0f32; n_g];
    let got = run_combined_tie(
        &dev,
        std::slice::from_ref(&ref_row),
        std::slice::from_ref(&group_row),
    );
    let total = (n_ref + n_g) as f64;
    let expected = total * total * total - total;
    assert_eq!(got[0], expected);
}
