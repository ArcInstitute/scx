//! Pipelined shard streaming for GPU workloads.
//!
//! [`DoubleBufferedShardLoader`] overlaps host-side shard decode with the
//! GPU work that consumes the previous shard. It runs `ShardSource::read_shard`
//! on a background (scoped) worker thread while the main thread uploads the
//! just-finished shard to the device and dispatches the user-provided callback.
//! Because cudarc's kernel launches and `memcpy_htod_async` return promptly,
//! the main thread goes back to `recv` as soon as it has queued the GPU work
//! for the current shard, overlapping worker decode with GPU compute.
//!
//! ## Fallback
//!
//! When `source.n_shards() <= 1`, the loader uses a simple synchronous loop
//! on the main thread — there is no benefit to pipelining a single shard.
//! The constructor never fails due to resource limits; the copy stream is
//! allocated lazily only in the double-buffered path.
//!
//! ## Relationship to G3 / [`crate::gpu_shard_source`]
//!
//! This legacy loader hands out **owned** `GpuCsr` per shard via
//! `upload_csr_to_gpu`. The G3 staging infrastructure (pinned host slots,
//! reusable device CSR slots, cached cuSPARSE descriptors, dedicated
//! copy stream + event handshake) lives behind the
//! [`crate::gpu_shard_source::GpuShardSource`] trait. New device-resident
//! consumers (preprocessing, scVI dataloader) should adopt
//! [`crate::gpu_shard_source::RawGpuShardSource`] /
//! [`crate::gpu_shard_source::GpuPreprocessedShardSource`] directly; the
//! existing `DoubleBufferedShardLoader` callers (HVG, linear operator)
//! continue working on the legacy path and can migrate incrementally.
//!
//! ## Async H→D (deferred per-call-site migration)
//!
//! A dedicated `copy_stream` is created so future iterations can issue the
//! H→D copies from pinned host memory onto that stream (via `memcpy_htod`),
//! record a `CudaEvent`, and have the compute stream wait on the event before
//! running the SpMM. The current implementation uploads on `dev.stream()`
//! (asynchronous anyway but serialized with compute) — sufficient to obtain
//! decode / compute overlap, which is the dominant win on I/O-bound sources.
//! Migrating individual call sites to pinned + dedicated copy stream is
//! handled by adopting `GpuShardSource` (see above), not by mutating this
//! loader's semantics.

use std::sync::mpsc;
use std::sync::Arc;

use cudarc::driver::safe::CudaStream;
use scx_format::ShardSource;
use scx_sparse::ScxCsr;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_pca::upload_csr_to_gpu;
use crate::shard_decode::GpuCsr;

/// Double-buffered shard loader over a [`ShardSource`].
///
/// Create with [`DoubleBufferedShardLoader::new`], then iterate via
/// [`DoubleBufferedShardLoader::for_each_shard`], which invokes the callback
/// on each non-empty shard as a `(shard_idx, &GpuCsr)` pair.
///
/// The source must be `Sync` so it can be borrowed by the worker thread.
/// All current [`ShardSource`] impls (`BackedCsrReader`, `LazyShardSource`,
/// test-only `InMemorySource`) satisfy this.
pub struct DoubleBufferedShardLoader<'a> {
    dev: &'a GpuDevice,
    source: &'a (dyn ShardSource + Sync),
    /// Dedicated stream reserved for async H→D copies from pinned memory.
    /// Currently unused (kept for the Phase 2+ upgrade documented above);
    /// we keep it alive here so creation failure surfaces at construction time.
    #[allow(dead_code)]
    copy_stream: Arc<CudaStream>,
    single_buffered: bool,
}

impl<'a> DoubleBufferedShardLoader<'a> {
    /// Create a loader. Allocates a dedicated copy stream when `n_shards > 1`;
    /// otherwise reuses the compute stream.
    pub fn new(dev: &'a GpuDevice, source: &'a (dyn ShardSource + Sync)) -> Result<Self, GpuError> {
        let single = source.n_shards() <= 1;
        let copy_stream = if single {
            dev.stream().clone()
        } else {
            dev.context()
                .new_stream()
                .map_err(|e| GpuError::StreamError(format!("new_stream: {e}")))?
        };
        Ok(Self {
            dev,
            source,
            copy_stream,
            single_buffered: single,
        })
    }

    /// Whether this loader is operating in single-buffered mode.
    pub fn is_single_buffered(&self) -> bool {
        self.single_buffered
    }

    /// Iterate over all shards, invoking `f(shard_idx, &gpu_csr)` on each
    /// non-empty shard. Shards are yielded in order `0..n_shards`.
    ///
    /// In double-buffered mode, `f`'s GPU work (kernel launches, memcpy) runs
    /// concurrently with the next shard's decode on the worker thread.
    pub fn for_each_shard<F>(&self, mut f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCsr) -> Result<(), GpuError>,
    {
        let n_shards = self.source.n_shards();
        if n_shards == 0 {
            return Ok(());
        }

        if self.single_buffered {
            for i in 0..n_shards {
                let csr = self
                    .source
                    .read_shard(i)
                    .map_err(|e| GpuError::InvalidShard(format!("shard {i}: {e}")))?;
                if csr.n_rows() == 0 {
                    continue;
                }
                let gpu_csr = upload_csr_to_gpu(self.dev, &csr)?;
                f(i, &gpu_csr)?;
            }
            return Ok(());
        }

        // Double-buffered: scoped worker thread feeds decoded shards via a
        // bounded channel. Bound = 1 is enough because worker and main are
        // two-stage pipeline (decode | upload+dispatch); larger buffers don't
        // accelerate an asymptotically-balanced pipeline.
        let source = self.source;
        let dev = self.dev;

        std::thread::scope(|scope| -> Result<(), GpuError> {
            type Msg = Result<(usize, ScxCsr), scx_format::ScxError>;
            let (tx, rx) = mpsc::sync_channel::<Msg>(1);

            scope.spawn(move || {
                for i in 0..n_shards {
                    let out = source.read_shard(i).map(|c| (i, c));
                    // If the main thread bailed out, the receiver is dropped;
                    // stop decoding further shards.
                    if tx.send(out).is_err() {
                        break;
                    }
                }
            });

            while let Ok(msg) = rx.recv() {
                let (i, csr) = match msg {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(GpuError::InvalidShard(format!(
                            "SCX read error during streaming decode: {e}"
                        )))
                    }
                };
                if csr.n_rows() == 0 {
                    continue;
                }
                let gpu_csr = upload_csr_to_gpu(dev, &csr)?;
                f(i, &gpu_csr)?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    // ---- Mock ShardSource used by pipeline tests ----

    struct MockSource {
        n_rows_per_shard: Vec<usize>,
        n_vars: usize,
        /// Sleep injected into `read_shard` to simulate I/O / decode cost.
        decode_delay: Duration,
        /// Records the order in which read_shard is called.
        read_order: Mutex<Vec<usize>>,
        read_count: AtomicUsize,
    }

    impl MockSource {
        fn new(n_rows_per_shard: Vec<usize>, n_vars: usize, decode_delay: Duration) -> Self {
            Self {
                n_rows_per_shard,
                n_vars,
                decode_delay,
                read_order: Mutex::new(Vec::new()),
                read_count: AtomicUsize::new(0),
            }
        }
    }

    impl ShardSource for MockSource {
        fn n_shards(&self) -> usize {
            self.n_rows_per_shard.len()
        }
        fn n_obs(&self) -> usize {
            self.n_rows_per_shard.iter().sum()
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsr> {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            self.read_order.lock().unwrap().push(shard_idx);
            if !self.decode_delay.is_zero() {
                std::thread::sleep(self.decode_delay);
            }
            // Produce a tiny distinctive CSR: one nonzero at (0, shard_idx) with value (shard_idx+1).
            let rows = self.n_rows_per_shard[shard_idx];
            if rows == 0 {
                return Ok(ScxCsr::new_unchecked(
                    (0, self.n_vars),
                    vec![0],
                    vec![],
                    vec![],
                ));
            }
            let mut indptr = vec![0i64; rows + 1];
            indptr[1..].fill(1);
            let col = (shard_idx as i32) % (self.n_vars as i32);
            let indices = vec![col];
            let data = vec![(shard_idx + 1) as f32];
            Ok(ScxCsr::new_unchecked(
                (rows, self.n_vars),
                indptr,
                indices,
                data,
            ))
        }
    }

    #[test]
    fn test_shards_yielded_once_in_order() {
        let dev = require_gpu!();
        let src = MockSource::new(vec![2, 3, 0, 4, 1], 5, Duration::ZERO);

        let loader = DoubleBufferedShardLoader::new(&dev, &src).unwrap();
        let yielded = Mutex::new(Vec::<usize>::new());
        loader
            .for_each_shard(|idx, csr| {
                yielded.lock().unwrap().push(idx);
                // Verify the GPU-side CSR has the expected row count.
                assert_eq!(csr.shape.0, src.n_rows_per_shard[idx]);
                Ok(())
            })
            .unwrap();
        dev.synchronize().unwrap();

        // Empty shard (idx=2) is skipped; all others appear exactly once in order.
        assert_eq!(yielded.into_inner().unwrap(), vec![0, 1, 3, 4]);
        // All shards must have been read (worker ran them all).
        assert_eq!(src.read_count.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_single_buffered_fallback_for_single_shard() {
        let dev = require_gpu!();
        let src = MockSource::new(vec![3], 5, Duration::ZERO);
        let loader = DoubleBufferedShardLoader::new(&dev, &src).unwrap();
        assert!(loader.is_single_buffered());

        let mut calls = 0;
        loader
            .for_each_shard(|idx, _csr| {
                assert_eq!(idx, 0);
                calls += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn test_overlap_beats_sequential() {
        let dev = require_gpu!();
        let n_shards = 4;
        let decode_ms = 60u64;
        let compute_ms = 30u64;

        let src = MockSource::new(vec![2; n_shards], 5, Duration::from_millis(decode_ms));

        let loader = DoubleBufferedShardLoader::new(&dev, &src).unwrap();
        let start = Instant::now();
        loader
            .for_each_shard(|_idx, _csr| {
                std::thread::sleep(Duration::from_millis(compute_ms));
                Ok(())
            })
            .unwrap();
        let db_elapsed = start.elapsed();

        // Sequential baseline wall-time = n_shards * (decode + compute).
        let sequential = Duration::from_millis((decode_ms + compute_ms) * n_shards as u64);
        // Overlap lower bound ≈ decode (first shard) + n_shards * max(decode, compute).
        // Use 85 % of sequential as the assertion threshold — generous, but
        // anything slower than that means decode and compute are NOT overlapping.
        let threshold = sequential.mul_f32(0.85);
        assert!(
            db_elapsed < threshold,
            "double-buffered time {db_elapsed:?} should beat 85 % of sequential {sequential:?} (threshold {threshold:?})"
        );

        // And it must be at least ~decode+compute (can't magically be instant).
        assert!(
            db_elapsed >= Duration::from_millis(decode_ms + compute_ms / 2),
            "pipeline wall-time too small: {db_elapsed:?} — something is mocking wrong"
        );
    }

    #[test]
    fn test_callback_error_halts_iteration() {
        let dev = require_gpu!();
        let src = MockSource::new(vec![2, 3, 4], 5, Duration::ZERO);
        let loader = DoubleBufferedShardLoader::new(&dev, &src).unwrap();

        let mut seen = Vec::new();
        let result = loader.for_each_shard(|idx, _csr| {
            seen.push(idx);
            if idx == 0 {
                Err(GpuError::ShapeMismatch {
                    expected: "stop".into(),
                    got: "now".into(),
                })
            } else {
                Ok(())
            }
        });

        assert!(result.is_err());
        assert_eq!(seen, vec![0]);
    }
}
