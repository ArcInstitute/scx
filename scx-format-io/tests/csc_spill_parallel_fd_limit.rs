//! The parallel emit's side of the spill store's descriptor claim.
//!
//! **Its own test binary** for the reason `csc_spill_fd_limit.rs` gives:
//! `setrlimit(RLIMIT_NOFILE)` is process-global.

#![cfg(all(target_os = "linux", feature = "parallel"))]

use scx_format_io::TempDirSpillStore;
use scx_sparse::{CscBuilder, CscBuilderConfig, ScxCsr, MAX_BUCKETS};

/// `scx_sparse`'s internal cap on concurrent spill reads. Hard-coded: this
/// test's claim is "a pool of 32 under a limit well below 32", not the value.
const PERMITTED_SPILL_READS: u64 = 8;

/// A parallel drain holds one store reader per bucket it is walking, so
/// without a cap the descriptor count grows with the rayon pool rather than
/// staying constant. Here 200 spilled groups drain in one batch on a 32-thread
/// pool under a limit of only `PERMITTED_SPILL_READS` descriptors above
/// what the process already holds — which 32 concurrent readers would exceed.
#[test]
fn a_parallel_drain_stays_under_a_descriptor_limit_below_the_pool() {
    let dir = tempfile::tempdir().expect("root");
    let store = TempDirSpillStore::new(Some(dir.path())).expect("store");

    let cfg = CscBuilderConfig {
        cols_per_shard: 1,
        memory_bytes: 1 << 20,
        spill_after_bytes: 0,
        target_buckets: MAX_BUCKETS,
        block_bytes: 8,
    };
    let (n_rows, n_cols) = (64usize, 200usize);
    let mut indptr = vec![0i64];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        for c in 0..n_cols {
            indices.push(c as i32);
            data.push((r * n_cols + c + 1) as f32);
        }
        indptr.push(indices.len() as i64);
    }
    let csr = ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(32)
        .build()
        .expect("pool");

    let mut b = CscBuilder::new(n_rows, n_cols, cfg, Box::new(store)).expect("new");
    b.push_shard(0, &csr).expect("push");
    let mut em = b.finish().expect("finish");
    assert!(
        em.stats().spilled_bytes > 0,
        "premise: the buckets spilled, so the drain reads files"
    );

    let open_fds = || std::fs::read_dir("/proc/self/fd").expect("fds").count() as u64;
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    assert_eq!(unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) }, 0);
    let restore = lim;
    // `read_dir` itself holds one descriptor while counting, and the pool's
    // threads hold none, so this leaves room for the permitted readers and a
    // small margin — well short of 32.
    let tight = libc::rlimit {
        rlim_cur: open_fds() + PERMITTED_SPILL_READS + 4,
        rlim_max: restore.rlim_max,
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &tight) }, 0);

    let result = pool.install(|| em.next_batch(u64::MAX, true));

    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &restore) }, 0);
    let batch = result.expect("a parallel drain under a tight fd limit");
    assert_eq!(batch.len(), n_cols, "every shard in one batch");
    let nnz: usize = batch.iter().map(|a| a.indices.len()).sum();
    assert_eq!(nnz, n_rows * n_cols);
}
