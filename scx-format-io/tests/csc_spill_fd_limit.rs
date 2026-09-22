//! The CSC spill store's descriptor claim, executable.
//!
//! **Its own test binary on purpose.** `setrlimit(RLIMIT_NOFILE)` is
//! process-global, so lowering it inside the library's unit-test binary would
//! break whichever of the other ~500 tests happened to open a file in the
//! window. A `tests/` file gets its own process and this is the only test in
//! it, so the limit it sets is its own.

#![cfg(unix)]

use scx_format_io::TempDirSpillStore;
use scx_sparse::{CscBuilder, CscBuilderConfig, ScxCsr, MAX_BUCKETS};

/// The store is handed whole blocks and opens one file per call, so a
/// many-bucket build must complete under a descriptor limit far below the
/// bucket count. `scx-convert`'s external transpose cannot make this claim —
/// it holds one `BufWriter<File>` per bucket for a whole pass, which is what
/// makes 11,922 buckets an EMFILE at a default `ulimit -n` of 1024 — and it is
/// why there is no descriptor cap in the builder.
#[test]
fn many_buckets_complete_under_a_tiny_descriptor_limit() {
    let dir = tempfile::tempdir().expect("root");
    let store = TempDirSpillStore::new(Some(dir.path())).expect("store");

    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) },
        0,
        "getrlimit"
    );
    let restore = lim;
    // 64 open files is far below `MAX_BUCKETS`, and well below the ~200
    // buckets this build asks for.
    let tight = libc::rlimit {
        rlim_cur: 64,
        rlim_max: restore.rlim_max,
    };
    assert_eq!(
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &tight) },
        0,
        "setrlimit"
    );

    let cfg = CscBuilderConfig {
        cols_per_shard: 1,
        memory_bytes: 1 << 20,
        spill_after_bytes: 0,
        target_buckets: MAX_BUCKETS,
        block_bytes: 8,
    };
    let n_cols = 200;
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for r in 0..8usize {
        for c in 0..n_cols {
            indices.push(c as i32);
            data.push((r * n_cols + c + 1) as f32);
        }
        indptr.push(indices.len() as i64);
    }
    let csr = ScxCsr::new_unchecked((8, n_cols), indptr, indices, data);

    let result = (|| -> Result<usize, String> {
        let mut b = CscBuilder::new(8, n_cols, cfg, Box::new(store)).map_err(|e| e.to_string())?;
        b.push_shard(0, &csr).map_err(|e| e.to_string())?;
        let mut em = b.finish().map_err(|e| e.to_string())?;
        // Premise: the build really did use more buckets than the limit.
        assert!(
            em.stats().n_buckets > 64,
            "{} buckets is not above the 64-descriptor limit",
            em.stats().n_buckets
        );
        let (mut ip, mut ix, mut dt) = (Vec::new(), Vec::new(), Vec::new());
        let mut n = 0usize;
        while em
            .next_shard_into(&mut ip, &mut ix, &mut dt)
            .map_err(|e| e.to_string())?
            .is_some()
        {
            n += ix.len();
        }
        Ok(n)
    })();

    assert_eq!(
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &restore) },
        0,
        "restore rlimit"
    );
    assert_eq!(result.expect("build under a tight fd limit"), 8 * n_cols);
}
