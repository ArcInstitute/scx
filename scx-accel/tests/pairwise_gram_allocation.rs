//! The gemm distance backend must not allocate the whole `[N_A × N_B]` Gram.
//!
//! `resolve_backend` maps `Auto → Gemm` for euclidean and cosine, and pyscx
//! defaults `backend=None → Auto`, so this is the path every
//! `pyscx.accel.energy_distance` call takes. Its Gram was one
//! `n_a · n_b · size_of::<F>()` allocation — 40 GB at 100 K control cells on the
//! documented default `dtype="f32"`, and `compute_energy_distance` keeps up to
//! `threads` of them live at once inside its per-perturbation `par_iter`.
//!
//! # Why the allocator is the oracle
//!
//! The obvious test — call it and assert the answer is right — is **green
//! against the unfixed code**. Linux with `vm.overcommit_memory = 0` satisfies
//! the request on a large-memory host (this dev node has 1007 GB), the reduction
//! then runs and returns the correct mean. The reservation *is* the defect, so
//! the reservation is what has to be measured.
//!
//! # The one allocation that has to be warmed away first
//!
//! faer's x86 gemm backend keeps a **thread-local scratch arena**
//! (`private_gemm_x86::gemm::MEM`), lazily initialised the first time a given
//! thread runs a gemm. On this host that is a single **440.4 MB** request, and
//! it is fixed-size: it does not scale with `n_a`, `n_b` or `n_dims`, and it is
//! neither caused nor removed by anything in this crate. Arming the recorder
//! before every rayon worker has paid it makes the measurement about faer's
//! allocator rather than about the Gram — which is exactly what happened on the
//! first draft of this test, and is why the warm-up below uses
//! `rayon::broadcast` to force *every* worker through a gemm rather than just
//! calling the kernel once.
//!
//! With that paid, the armed region measures the Gram and nothing else:
//! **67.1 MB** (`4096² × 4`) against the untiled kernel, `≤ 4 MiB` once the rows
//! of `a` are blocked to the budget. Both bounds are asserted below, the lower
//! one so that a kernel which allocated nothing at all could not pass.
//!
//! Structure copied from `scx-format-io/tests/framed_decode_allocation.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use scx_accel::eval_metrics::distances::{mean_pairwise_distance_self, DistanceBackend};
use scx_accel::DistanceMetric;

// ---------------------------------------------------------------------------
// Recording allocator
// ---------------------------------------------------------------------------

static PEAK_REQUEST: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

/// Passes every allocation through to the system allocator, recording the
/// largest single request made while armed. `realloc` is included because
/// `Vec` growth goes through it.
struct RecordingAlloc;

impl RecordingAlloc {
    #[inline]
    fn note(size: usize) {
        if ARMED.load(Ordering::Relaxed) {
            PEAK_REQUEST.fetch_max(size, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for RecordingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: RecordingAlloc = RecordingAlloc;

/// Run `f` with allocation recording on, returning `(result, peak_request_bytes)`.
///
/// The whole file is a single `#[test]`, so no other test thread can pollute the
/// counter — rayon workers inside the kernel allocate too, and they are exactly
/// what we want to see. Arming is deliberately as tight as possible around the
/// call.
fn measure_peak_alloc<T>(f: impl FnOnce() -> T) -> (T, usize) {
    PEAK_REQUEST.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    let out = f();
    ARMED.store(false, Ordering::SeqCst);
    (out, PEAK_REQUEST.load(Ordering::SeqCst))
}

const N: usize = 4096;
const N_DIMS: usize = 32;
/// Forced Gram budget. Small enough that the fix has to tile ~16 ways, large
/// enough that a tile is a real gemm rather than a degenerate one-row call.
const BUDGET_BYTES: usize = 4 << 20;
/// Generous next to the 67.1 MB an untiled kernel requests, and comfortably
/// above the budget plus the kernel's `O(n)` bookkeeping.
const MAX_ALLOWED_REQUEST: usize = 8 << 20;

/// Deterministic points in `[-1, 1)`, no `rand` dependency.
fn points(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n * d)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn gemm_self_distance_does_not_allocate_the_whole_gram() {
    // Set before the first call so the `OnceLock` inside `pairwise_memory_budget`
    // caches this value. Safe here because this binary holds one test.
    std::env::set_var("SCX_ACCEL_PAIRWISE_MEMORY_BUDGET", BUDGET_BYTES.to_string());

    let a = points(N, N_DIMS, 7);

    // Force every rayon worker through a gemm so its thread-local faer arena is
    // already initialised when the recorder arms. `broadcast` runs the closure
    // exactly once per worker, which a plain call cannot guarantee.
    let warm = points(64, N_DIMS, 1);
    let warm_once = || {
        let _ = mean_pairwise_distance_self(
            &warm,
            64,
            N_DIMS,
            DistanceMetric::Euclidean,
            DistanceBackend::Gemm,
        )
        .unwrap();
    };
    rayon::broadcast(|_| warm_once());
    // `broadcast` covers the pool's workers but not this thread, and the
    // calling thread participates in faer's parallel gemm.
    warm_once();

    let (mean, peak) = measure_peak_alloc(|| {
        mean_pairwise_distance_self(
            &a,
            N,
            N_DIMS,
            DistanceMetric::Euclidean,
            DistanceBackend::Gemm,
        )
        .unwrap()
    });

    // The answer still has to be right — a kernel that returns garbage would
    // also allocate nothing.
    assert!(
        mean.is_finite() && mean > 0.0,
        "gemm self-distance returned {mean}, so the allocation bound below would be vacuous"
    );

    // Premise: the recorder must actually be seeing the Gram. If the largest
    // request is far below one block, the allocator is blind to faer's buffer
    // and the upper bound proves nothing.
    assert!(
        peak >= BUDGET_BYTES / 2,
        "largest request was only {peak} bytes — under one {BUDGET_BYTES}-byte Gram \
         block, so this oracle is not observing the Gram and the bound below is vacuous"
    );

    assert!(
        peak <= MAX_ALLOWED_REQUEST,
        "gemm self-distance on {N}x{N_DIMS} f32 made a single {peak}-byte allocation; \
         the {BUDGET_BYTES}-byte Gram budget allows at most {MAX_ALLOWED_REQUEST}. \
         The untiled kernel asks for the whole {}-byte Gram (n^2 * 4).",
        N * N * 4
    );
}
