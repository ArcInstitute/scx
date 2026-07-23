//! Tests for the decode-prefetch + budgeted-reduction primitives.
//!
//! Extracted per the repo convention for non-trivial test modules
//! (`#[cfg(test)] #[path = "prefetch_tests.rs"] mod tests;`), keeping
//! white-box `super::*` access.

use super::*;
use scx_sparse::{ScxCsc, ScxCsr};

/// In-memory `ShardSource` that can be told to panic while decoding a chosen
/// shard, to exercise the worker `catch_unwind` path.
struct StubSource {
    shards: Vec<ScxCsr>,
    n_obs: usize,
    n_vars: usize,
    panic_on: Option<usize>,
}

impl StubSource {
    fn new(shards: Vec<ScxCsr>, n_obs: usize, n_vars: usize) -> Self {
        Self {
            shards,
            n_obs,
            n_vars,
            panic_on: None,
        }
    }
}

impl ShardSource for StubSource {
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
        if Some(shard_idx) == self.panic_on {
            panic!("injected decode panic at shard {shard_idx}");
        }
        Ok(self.shards[shard_idx].clone())
    }
}

/// Build `n` single-row shards over `n_vars` columns; row `i` has value
/// `(i + 1)` in column `i % n_vars`. Deterministic and cheap.
fn make_shards(n: usize, n_vars: usize) -> StubSource {
    let shards: Vec<ScxCsr> = (0..n)
        .map(|i| {
            let col = (i % n_vars) as i32;
            ScxCsr::new_unchecked((1, n_vars), vec![0, 1], vec![col], vec![(i + 1) as f32])
        })
        .collect();
    StubSource::new(shards, n, n_vars)
}

#[test]
fn ordered_delivery_is_in_shard_order() {
    let src = make_shards(64, 8);
    let mut seen = Vec::new();
    for_each_shard_ordered(&src, 8, |idx, _csr| {
        seen.push(idx);
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, (0..64).collect::<Vec<_>>());
}

#[test]
fn depth_one_uses_sequential_fallback() {
    let src = make_shards(16, 4);
    let mut seen = Vec::new();
    for_each_shard_ordered(&src, 1, |idx, _csr| {
        seen.push(idx);
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, (0..16).collect::<Vec<_>>());
}

#[test]
fn empty_source_is_noop() {
    let src = StubSource::new(vec![], 0, 4);
    let mut count = 0usize;
    for_each_shard_ordered(&src, 8, |_idx, _csr| {
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 0);
}

/// Source whose shard `0` stalls (spins) until `stall_head_until` *other*
/// shards have decoded — forces the head-of-line-stall case that the
/// spawn-on-consume fix must keep bounded.
struct StallHeadSource {
    n: usize,
    n_vars: usize,
    others_done: std::sync::atomic::AtomicUsize,
    stall_head_until: usize,
}

impl ShardSource for StallHeadSource {
    fn n_shards(&self) -> usize {
        self.n
    }
    fn n_obs(&self) -> usize {
        self.n
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
        use std::sync::atomic::Ordering;
        if shard_idx == 0 {
            while self.others_done.load(Ordering::Acquire) < self.stall_head_until {
                std::thread::yield_now();
            }
        } else {
            self.others_done.fetch_add(1, Ordering::AcqRel);
        }
        Ok(ScxCsr::new_unchecked(
            (1, self.n_vars),
            vec![0, 1],
            vec![0],
            vec![1.0],
        ))
    }
}

/// Regression for the Cursor review: under a head-of-line stall the reorder
/// buffer (decoded-but-unconsumed shards) must stay bounded by `depth`. With the
/// old spawn-on-receive this grew to ~n_shards; spawn-on-consume caps it.
#[test]
fn head_stall_keeps_reorder_buffer_bounded_by_depth() {
    // Skip on a single-thread pool (the prefetch path isn't taken there).
    if rayon::current_num_threads() <= 1 {
        return;
    }
    let depth = 4usize;
    let n = 20usize;
    let src = StallHeadSource {
        n,
        n_vars: 4,
        others_done: std::sync::atomic::AtomicUsize::new(0),
        // Head waits for the other primed shards (depth - 1) to decode.
        stall_head_until: depth - 1,
    };
    MAX_REORDER_BUFFER.with(|m| m.set(0));
    let mut seen = Vec::new();
    for_each_shard_ordered(&src, depth, |idx, _csr| {
        seen.push(idx);
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, (0..n).collect::<Vec<_>>());
    let peak = MAX_REORDER_BUFFER.with(|m| m.get());
    assert!(
        peak <= depth,
        "reorder buffer peaked at {peak} shards, exceeding depth {depth} — \
         the bounded-memory contract is violated"
    );
}

#[test]
fn worker_panic_becomes_error_not_hang() {
    let mut src = make_shards(32, 4);
    src.panic_on = Some(7);
    let err = for_each_shard_ordered(&src, 8, |_idx, _csr| Ok(())).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("panicked") && msg.contains("shard 7"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn consumer_error_propagates_and_stops() {
    let src = make_shards(32, 4);
    let mut n_consumed = 0usize;
    let err = for_each_shard_ordered(&src, 8, |idx, _csr| {
        n_consumed += 1;
        if idx == 5 {
            return Err(AccelError::InvalidInput("stop here".into()));
        }
        Ok(())
    })
    .unwrap_err();
    assert!(format!("{err}").contains("stop here"));
    // Ordered consume stops at the first error (shards 0..=5 consumed).
    assert_eq!(n_consumed, 6);
}

/// The core 2.1 guarantee: ordered decode-prefetch accumulation is
/// **byte-identical** to the sequential loop it replaces.
#[test]
fn ordered_accumulation_is_bit_identical_to_sequential() {
    let n_vars = 16usize;
    let src = make_shards(500, n_vars);

    // Sequential reference (the pre-2.1 loop shape).
    let mut seq_sum = vec![0.0f64; n_vars];
    let mut seq_sq = vec![0.0f64; n_vars];
    for idx in 0..src.n_shards() {
        let csr = src.read_shard(idx).unwrap();
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let c = col as usize;
            let v = val as f64;
            seq_sum[c] += v;
            seq_sq[c] += v * v;
        }
    }

    // Prefetched ordered accumulation.
    let mut pf_sum = vec![0.0f64; n_vars];
    let mut pf_sq = vec![0.0f64; n_vars];
    for_each_shard_ordered(&src, 8, |_idx, csr| {
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let c = col as usize;
            let v = val as f64;
            pf_sum[c] += v;
            pf_sq[c] += v * v;
        }
        Ok(())
    })
    .unwrap();

    for c in 0..n_vars {
        assert_eq!(
            pf_sum[c].to_bits(),
            seq_sum[c].to_bits(),
            "sum col {c} not bit-identical"
        );
        assert_eq!(
            pf_sq[c].to_bits(),
            seq_sq[c].to_bits(),
            "sum_sq col {c} not bit-identical"
        );
    }
}

#[test]
fn budgeted_reduction_matches_sequential_within_tolerance() {
    let n_vars = 16usize;
    let src = make_shards(500, n_vars);

    // Sequential column sums.
    let mut seq = vec![0.0f64; n_vars];
    for idx in 0..src.n_shards() {
        let csr = src.read_shard(idx).unwrap();
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            seq[col as usize] += val as f64;
        }
    }

    // Parallel per-worker reduction, merged. Order differs → tolerance only.
    let par = reduce_shards_budgeted(
        &src,
        4,
        || vec![0.0f64; n_vars],
        |acc: &mut Vec<f64>, _idx, csr| {
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                acc[col as usize] += val as f64;
            }
            Ok(())
        },
        |mut a: Vec<f64>, b: Vec<f64>| {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x += *y;
            }
            a
        },
    )
    .unwrap();

    for c in 0..n_vars {
        assert!(
            (par[c] - seq[c]).abs() < 1e-9,
            "col {c}: parallel {} vs sequential {}",
            par[c],
            seq[c]
        );
    }
}

/// Minimal in-memory `ColumnShardSource` for the CSC prefetch smoke test.
struct StubCscSource {
    shards: Vec<ScxCsc>,
    n_obs: usize,
    n_vars: usize,
}

impl ColumnShardSource for StubCscSource {
    fn n_csc_shards(&self) -> usize {
        self.shards.len()
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn read_csc_shard(&self, idx: usize) -> scx_format_io::Result<ScxCsc> {
        Ok(self.shards[idx].clone())
    }
    fn read_csc_columns(&self, _r: std::ops::Range<u32>) -> scx_format_io::Result<ScxCsc> {
        unreachable!("prefetch does not call read_csc_columns")
    }
    fn csc_shard_col_range(&self, idx: usize) -> Option<(u32, u32)> {
        Some((idx as u32, idx as u32 + 1))
    }
}

#[test]
fn csc_ordered_delivery_is_in_shard_order() {
    // 16 single-column CSC shards over a 4-row axis.
    let shards: Vec<ScxCsc> = (0..16)
        .map(|_| ScxCsc::new_unchecked((4, 1), vec![0, 1], vec![0], vec![1.0]))
        .collect();
    let src = StubCscSource {
        shards,
        n_obs: 4,
        n_vars: 16,
    };
    let mut seen = Vec::new();
    for_each_csc_shard_ordered(&src, 8, |idx, _csc| {
        seen.push(idx);
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, (0..16).collect::<Vec<_>>());
}

#[test]
fn accumulate_shards_default_matches_sequential() {
    // Default mode is StableOrder → bit-identical to sequential.
    let n_vars = 8usize;
    let src = make_shards(200, n_vars);

    let mut seq = vec![0.0f64; n_vars];
    for idx in 0..src.n_shards() {
        let csr = src.read_shard(idx).unwrap();
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            seq[col as usize] += val as f64;
        }
    }

    let acc = accumulate_shards(
        &src,
        4,
        || vec![0.0f64; n_vars],
        |acc: &mut Vec<f64>, _idx, csr| {
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                acc[col as usize] += val as f64;
            }
            Ok(())
        },
        |mut a: Vec<f64>, b: Vec<f64>| {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x += *y;
            }
            a
        },
    )
    .unwrap();

    for c in 0..n_vars {
        assert_eq!(
            acc[c].to_bits(),
            seq[c].to_bits(),
            "col {c} not bit-identical"
        );
    }
}
