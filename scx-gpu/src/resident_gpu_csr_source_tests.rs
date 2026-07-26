//! Tests for [`ResidentGpuCsrSource`](super::ResidentGpuCsrSource).
//!
//! The load-bearing claim is that residency is a *provenance* change and not a
//! numerical one: the consumer callback must observe the identical shard
//! subsequence, with identical indices, shapes and device contents, whether the
//! source is streaming or resident. Everything here compares the two drains
//! element-for-element rather than checking a derived statistic.

use super::*;

use scx_format_io::ShardSource;
use scx_sparse::ScxCsr;

use crate::backed_gpu_matrix_source::BackedGpuMatrixSource;

/// Multi-shard in-memory CSR `ShardSource`. `InMemoryCsrShardSource` is
/// single-shard by construction, and the whole point here is `n_shards > 1`.
struct MultiShardCsr {
    shards: Vec<ScxCsr>,
    n_obs: usize,
    n_vars: usize,
}

impl ShardSource for MultiShardCsr {
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
        self.shards
            .get(shard_idx)
            .cloned()
            .ok_or(scx_format_io::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.shards.len(),
            })
    }
    fn max_shard_rows(&self) -> scx_format_io::Result<usize> {
        Ok(self.shards.iter().map(|s| s.n_rows()).max().unwrap_or(0))
    }
}

/// One shard with `rows` rows over `n_vars` columns, deterministic sparsity.
/// Column indices are strictly increasing per row (the GPU-DE contract that
/// `validate_shard_for_gpu_de` enforces), and values are distinct per nonzero
/// so a mis-ordered comparison cannot pass by coincidence.
fn make_shard(rows: usize, n_vars: usize, seed: u64) -> ScxCsr {
    let mut indptr: Vec<i64> = vec![0];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    let mut state = seed | 1;
    for _ in 0..rows {
        let mut col: i64 = -1;
        let nnz_row = 1 + (state % 5) as usize;
        for _ in 0..nnz_row {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            col += 1 + (state % 7) as i64;
            if col >= n_vars as i64 {
                break;
            }
            indices.push(col as i32);
            data.push((indices.len() as f32) * 0.25 + (seed as f32));
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsr {
        indptr,
        indices,
        data,
        shape: (rows, n_vars),
    }
}

fn fixture() -> MultiShardCsr {
    let n_vars = 64;
    let shards = vec![
        make_shard(7, n_vars, 0x1111),
        make_shard(5, n_vars, 0x2222),
        make_shard(9, n_vars, 0x3333),
    ];
    let n_obs = shards.iter().map(|s| s.n_rows()).sum();
    MultiShardCsr {
        shards,
        n_obs,
        n_vars,
    }
}

/// Everything the callback can observe about one yielded shard.
#[derive(Debug, PartialEq)]
struct Observed {
    idx: usize,
    shape: (usize, usize),
    nnz: usize,
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
}

fn drain(dev: &GpuDevice, source: &mut dyn GpuMatrixSource) -> Vec<Observed> {
    let mut out = Vec::new();
    source
        .for_each_gpu_csr_shard(&mut |idx, slot| {
            let view = slot.view();
            out.push(Observed {
                idx,
                shape: view.shape,
                nnz: view.nnz(),
                indptr: dev.stream().clone_dtoh(&view.indptr).unwrap(),
                indices: dev.stream().clone_dtoh(&view.indices).unwrap(),
                data: dev.stream().clone_dtoh(&view.data).unwrap(),
            });
            Ok(())
        })
        .unwrap();
    out
}

/// The whole contract: a resident drain is indistinguishable from a streaming
/// one at the callback boundary.
#[test]
fn resident_drain_matches_streaming_drain_exactly() {
    let dev = require_gpu!();
    let src = fixture();

    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let expected = drain(&dev, &mut streaming);
    assert_eq!(expected.len(), 3, "premise: all three shards are non-empty");
    assert!(
        expected.iter().any(|o| o.nnz > 0),
        "premise: the fixture has nonzeros"
    );

    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let mut resident = try_build_resident(&dev, &mut streaming, DEFAULT_RESIDENT_MAX_FRAC)
        .unwrap()
        .expect("fixture is tiny — residency must be granted");

    assert_eq!(resident.n_retained_shards(), 3);
    assert_eq!(resident.shape(), (src.n_obs, src.n_vars));
    assert!(resident.available_layouts().contains(LayoutSet::CSR));

    let got = drain(&dev, &mut resident);
    assert_eq!(
        got, expected,
        "resident drain diverged from streaming drain"
    );
}

/// Replay is repeatable — the point of residency is that the *second* pass is
/// free, so the second pass had better be identical to the first.
#[test]
fn resident_source_replays_identically_across_passes() {
    let dev = require_gpu!();
    let src = fixture();
    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let mut resident = try_build_resident(&dev, &mut streaming, DEFAULT_RESIDENT_MAX_FRAC)
        .unwrap()
        .unwrap();

    let first = drain(&dev, &mut resident);
    let second = drain(&dev, &mut resident);
    let third = drain(&dev, &mut resident);
    assert_eq!(first, second);
    assert_eq!(second, third);
}

/// A budget too small to hold even one shard declines cleanly, and the inner
/// source is still usable afterwards — the caller's fallback is a plain
/// streaming pass, so an aborted drain must not have poisoned the staging ring.
#[test]
fn tiny_budget_declines_and_leaves_the_source_usable() {
    let dev = require_gpu!();
    let src = fixture();

    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let declined = try_build_resident(&dev, &mut streaming, 1e-12).unwrap();
    assert!(
        declined.is_none(),
        "a budget of ~0 bytes must decline residency"
    );

    // Same source object, streamed after the aborted drain.
    let after = drain(&dev, &mut streaming);
    let mut fresh = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let expected = drain(&dev, &mut fresh);
    assert_eq!(
        after, expected,
        "an aborted residency drain left the streaming source in a bad state"
    );
}

/// An empty source is declined rather than producing a zero-shard resident
/// source that would silently skip the caller's work.
#[test]
fn empty_source_declines() {
    let dev = require_gpu!();
    let src = MultiShardCsr {
        shards: vec![],
        n_obs: 0,
        n_vars: 0,
    };
    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    assert!(
        try_build_resident(&dev, &mut streaming, DEFAULT_RESIDENT_MAX_FRAC)
            .unwrap()
            .is_none()
    );
}

/// Shards with zero rows are skipped by the streaming driver, so they must be
/// absent from the retained set too — otherwise `global_row` accounting in the
/// consumer would see a different subsequence.
#[test]
fn empty_shards_are_skipped_in_both_drains() {
    let dev = require_gpu!();
    let n_vars = 32;
    let shards = vec![
        make_shard(4, n_vars, 0xAAAA),
        ScxCsr {
            indptr: vec![0],
            indices: vec![],
            data: vec![],
            shape: (0, n_vars),
        },
        make_shard(6, n_vars, 0xBBBB),
    ];
    let n_obs = shards.iter().map(|s| s.n_rows()).sum();
    let src = MultiShardCsr {
        shards,
        n_obs,
        n_vars,
    };

    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let expected = drain(&dev, &mut streaming);
    assert_eq!(expected.len(), 2, "the empty shard is not yielded");
    assert_eq!(
        expected.iter().map(|o| o.idx).collect::<Vec<_>>(),
        vec![0, 2],
        "premise: the streaming driver preserves original shard indices"
    );

    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let mut resident = try_build_resident(&dev, &mut streaming, DEFAULT_RESIDENT_MAX_FRAC)
        .unwrap()
        .unwrap();
    assert_eq!(drain(&dev, &mut resident), expected);
}

/// The resident source retains CSR only; a CSC request must fail loudly rather
/// than silently returning nothing.
#[test]
fn csc_request_is_unsupported() {
    let dev = require_gpu!();
    let src = fixture();
    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();
    let mut resident = try_build_resident(&dev, &mut streaming, DEFAULT_RESIDENT_MAX_FRAC)
        .unwrap()
        .unwrap();
    let err = resident.for_each_gpu_csc_shard_in_range(0..4, &mut |_, _| Ok(()));
    assert!(matches!(err, Err(GpuError::UnsupportedLayout(_))));
}

/// `clone_exact` sizes buffers to the live shard, not to the staging slot's
/// power-of-two capacity — the property that keeps resident VRAM equal to the
/// matrix rather than up to 2× it.
#[test]
fn clone_exact_is_exactly_sized() {
    let dev = require_gpu!();
    // 9 rows / a nnz that is not a power of two, so the staging slot's
    // `ensure_capacity` rounds up and the clone must not.
    let src = MultiShardCsr {
        shards: vec![make_shard(9, 64, 0xC0FFEE)],
        n_obs: 9,
        n_vars: 64,
    };
    let mut streaming = BackedGpuMatrixSource::new(&dev, &src).unwrap();

    let mut staged_capacity = (0usize, 0usize, 0usize);
    let mut cloned_capacity = (0usize, 0usize, 0usize);
    let mut live = (0usize, 0usize);
    streaming
        .for_each_gpu_csr_shard(&mut |_idx, slot| {
            staged_capacity = slot.capacity();
            live = (slot.shape().0, slot.nnz());
            cloned_capacity = slot.clone_exact(&dev).unwrap().capacity();
            Ok(())
        })
        .unwrap();

    assert_eq!(cloned_capacity.0, live.0 + 1, "indptr exactly n_rows + 1");
    assert_eq!(cloned_capacity.1, live.1, "indices exactly nnz");
    assert_eq!(cloned_capacity.2, live.1, "data exactly nnz");
    assert!(
        staged_capacity.1 >= cloned_capacity.1,
        "premise: the staging slot is at least as large as the exact clone \
         (got staged {staged_capacity:?} vs exact {cloned_capacity:?})"
    );
}
