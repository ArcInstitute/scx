//! CPU tests for [`drive_shards`]' ordering policy.
//!
//! These deliberately carry **no** `require_gpu!()` and **no** `#[ignore]`.
//! `tests/gpu_test_gating.rs` rule 1 is an *iff* — a test invokes a device gate
//! exactly when it carries the ignore reason — so an ungated, un-ignored test
//! is the correct shape here provided it never names `GpuDevice::new`, which is
//! the whole reason the seam exists: 166 of this crate's tests are unreachable
//! on any CPU host, and the staging lifecycle was entirely inside that set.
//!
//! The fake numbers its events, so *event identity* is assertable — "the
//! host-wait awaited the event this slot's own upload recorded", not merely
//! "an event happened".

use super::*;
use std::cell::RefCell;

/// A decoded shard, reduced to what the driver's policy can observe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FakeShard {
    idx: usize,
    /// Stands in for "no rows" (CSR) / "no nonzeros or no columns" (CSC).
    stageable: bool,
}

/// One observable step of the lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Validate(usize),
    HostWait { slot: usize, awaited: Option<u64> },
    Stage { slot: usize, idx: usize },
    GateCopyToCompute { slot: usize, event: u64 },
    Dispatch(usize),
    GateComputeToCopy { event: u64 },
    Drain { awaited: Vec<u64> },
}

/// Feeds a fixed shard list in plan order, optionally failing one read.
struct RecordingFeeder {
    shards: Vec<FakeShard>,
    /// Shard index whose read fails, standing in for a corrupt shard on disk.
    fail_read_at: Option<usize>,
    reads: RefCell<Vec<usize>>,
}

impl RecordingFeeder {
    fn new(stageable: &[bool]) -> Self {
        Self {
            shards: stageable
                .iter()
                .enumerate()
                .map(|(idx, &stageable)| FakeShard { idx, stageable })
                .collect(),
            fail_read_at: None,
            reads: RefCell::new(Vec::new()),
        }
    }

    fn all_stageable(n: usize) -> Self {
        Self::new(&vec![true; n])
    }

    fn failing_read_at(mut self, idx: usize) -> Self {
        self.fail_read_at = Some(idx);
        self
    }
}

impl ShardFeeder for RecordingFeeder {
    type Shard = FakeShard;

    fn feed(
        &self,
        plan: &StagingPlan,
        consume: ShardConsumer<'_, FakeShard>,
    ) -> Result<(), GpuError> {
        for &idx in &plan.indices {
            if self.fail_read_at == Some(idx) {
                // Mirrors the production classification: the host could not
                // produce the shard, which is a bad input, not a device fault.
                return Err(GpuError::InvalidShard(format!(
                    "shard {idx}: fake read failure"
                )));
            }
            self.reads.borrow_mut().push(idx);
            consume(idx, &self.shards[idx])?;
        }
        Ok(())
    }
}

/// Records the lifecycle instead of performing it.
struct RecordingStager {
    ops: RefCell<Vec<Op>>,
    memo: ValidationMemo,
    next_event: u64,
    /// Per-slot event recorded by that slot's most recent upload gate.
    slot_events: [Option<u64>; RING],
    /// Shard index whose validation fails.
    fail_validate_at: Option<usize>,
    /// Shard index whose dispatch fails.
    fail_dispatch_at: Option<usize>,
    /// If set, `drain` fails once with this message.
    drain_error: Option<String>,
}

impl RecordingStager {
    fn new(n_shards: usize) -> Self {
        Self {
            ops: RefCell::new(Vec::new()),
            memo: ValidationMemo::new(n_shards),
            next_event: 0,
            slot_events: [None; RING],
            fail_validate_at: None,
            fail_dispatch_at: None,
            drain_error: None,
        }
    }

    fn failing_validate_at(mut self, idx: usize) -> Self {
        self.fail_validate_at = Some(idx);
        self
    }

    fn failing_dispatch_at(mut self, idx: usize) -> Self {
        self.fail_dispatch_at = Some(idx);
        self
    }

    fn push(&self, op: Op) {
        self.ops.borrow_mut().push(op);
    }

    fn ops(&self) -> Vec<Op> {
        self.ops.borrow().clone()
    }

    fn fresh_event(&mut self) -> u64 {
        let e = self.next_event;
        self.next_event += 1;
        e
    }
}

impl ShardStager for RecordingStager {
    type Shard = FakeShard;

    fn is_stageable(&self, shard: &FakeShard) -> bool {
        shard.stageable
    }

    fn validate(&mut self, idx: usize, _shard: &FakeShard) -> Result<(), GpuError> {
        self.push(Op::Validate(idx));
        if self.fail_validate_at == Some(idx) {
            return Err(GpuError::InvalidShard(format!(
                "shard {idx}: fake validation failure"
            )));
        }
        Ok(())
    }

    fn memo(&mut self) -> &mut ValidationMemo {
        &mut self.memo
    }

    fn host_wait(&mut self, slot: usize) -> Result<(), GpuError> {
        let awaited = self.slot_events[slot].take();
        self.push(Op::HostWait { slot, awaited });
        Ok(())
    }

    fn stage_and_upload(&mut self, slot: usize, shard: &FakeShard) -> Result<(), GpuError> {
        self.push(Op::Stage {
            slot,
            idx: shard.idx,
        });
        Ok(())
    }

    fn gate_copy_to_compute(&mut self, slot: usize) -> Result<(), GpuError> {
        let event = self.fresh_event();
        self.slot_events[slot] = Some(event);
        self.push(Op::GateCopyToCompute { slot, event });
        Ok(())
    }

    fn dispatch(&mut self, idx: usize, _shard: &FakeShard) -> Result<(), GpuError> {
        self.push(Op::Dispatch(idx));
        if self.fail_dispatch_at == Some(idx) {
            return Err(GpuError::KernelLaunchFailed(format!(
                "shard {idx}: fake dispatch failure"
            )));
        }
        Ok(())
    }

    fn gate_compute_to_copy(&mut self) -> Result<(), GpuError> {
        let event = self.fresh_event();
        self.push(Op::GateComputeToCopy { event });
        Ok(())
    }

    fn drain(&mut self) -> Result<(), GpuError> {
        let mut awaited: Vec<u64> = self.slot_events.iter().flatten().copied().collect();
        awaited.sort_unstable();
        for e in self.slot_events.iter_mut() {
            *e = None;
        }
        self.push(Op::Drain { awaited });
        // Configurable so the error-precedence policy can be tested in both
        // directions. With drain always returning `Ok`, a mutation that gave
        // `drain_result` priority over `feed_result` still passed every test
        // (codex - gpt-5.6-sol).
        match self.drain_error.take() {
            Some(msg) => Err(GpuError::CudaError(msg)),
            None => Ok(()),
        }
    }
}

fn plan_all(n: usize, depth: usize) -> StagingPlan {
    StagingPlan::all(n, depth)
}

fn plan_selected(v: &[usize], depth: usize) -> StagingPlan {
    StagingPlan::selected(v.to_vec(), depth)
}

/// True when some pinned slot is written twice with no host-wait in between —
/// the CPU race the device-side event gates cannot prevent.
///
/// Shared by the property assertion (which requires `false`) and by
/// [`the_legacy_single_shard_fast_path_is_what_this_property_rejects`] (which
/// requires `true` of the shape that shipped), so the two cannot drift into
/// checking different things.
fn restages_a_slot_without_host_waiting(ops: &[Op]) -> bool {
    let mut dirty = [false; RING];
    for op in ops {
        match op {
            Op::Stage { slot, .. } => {
                if dirty[*slot] {
                    return true;
                }
                dirty[*slot] = true;
            }
            Op::HostWait { slot, .. } => dirty[*slot] = false,
            Op::Drain { .. } => dirty = [false; RING],
            _ => {}
        }
    }
    false
}

#[test]
fn the_lifecycle_runs_in_one_fixed_order_per_shard() {
    let feeder = RecordingFeeder::all_stageable(2);
    let mut stager = RecordingStager::new(2);
    drive_shards(&feeder, &mut stager, &plan_all(2, 4)).unwrap();
    assert_eq!(
        stager.ops(),
        vec![
            Op::Validate(0),
            Op::HostWait {
                slot: 0,
                awaited: None
            },
            Op::Stage { slot: 0, idx: 0 },
            Op::GateCopyToCompute { slot: 0, event: 0 },
            Op::Dispatch(0),
            Op::GateComputeToCopy { event: 1 },
            Op::Validate(1),
            Op::HostWait {
                slot: 1,
                awaited: None
            },
            Op::Stage { slot: 1, idx: 1 },
            Op::GateCopyToCompute { slot: 1, event: 2 },
            Op::Dispatch(1),
            Op::GateComputeToCopy { event: 3 },
            Op::Drain {
                awaited: vec![0, 2]
            },
        ]
    );
}

/// Validation precedes staging, and precedes the host-wait.
///
/// The second half is a normalisation: the CSR copy host-waited first. A shard
/// that is about to be rejected should not first block the calling thread on an
/// unrelated DMA.
#[test]
fn validation_precedes_both_the_host_wait_and_the_stage() {
    let feeder = RecordingFeeder::all_stageable(3);
    let mut stager = RecordingStager::new(3);
    drive_shards(&feeder, &mut stager, &plan_all(3, 4)).unwrap();
    let ops = stager.ops();
    for idx in 0..3 {
        let v = ops.iter().position(|o| *o == Op::Validate(idx)).unwrap();
        let s = ops
            .iter()
            .position(|o| matches!(o, Op::Stage { idx: i, .. } if *i == idx))
            .unwrap();
        let hw = ops[v..]
            .iter()
            .position(|o| matches!(o, Op::HostWait { .. }))
            .unwrap()
            + v;
        assert!(v < hw, "shard {idx}: validate must precede the host-wait");
        assert!(v < s, "shard {idx}: validate must precede the stage");
    }
}

#[test]
fn no_slot_is_restaged_without_a_host_wait() {
    let feeder = RecordingFeeder::all_stageable(7);
    let mut stager = RecordingStager::new(7);
    drive_shards(&feeder, &mut stager, &plan_all(7, 4)).unwrap();
    assert!(!restages_a_slot_without_host_waiting(&stager.ops()));
}

/// The §8.18 finding, as a test rather than a claim.
///
/// Both hand-rolled loops special-cased a one-shard drive: no ring, no copy
/// stream, and — the part that matters — **no event recorded**, on the stated
/// grounds that "there is no successor `stage()` that could race the in-flight
/// DMA". True within one drive. False across two: the second drive re-enters
/// with `pinned[0]` still the source of an un-awaited `memcpy_htod_async`.
///
/// The trace below is that shape written out by hand. The property above must
/// reject it; if it does not, the property is decorative.
#[test]
fn the_legacy_single_shard_fast_path_is_what_this_property_rejects() {
    let legacy = vec![
        // First drive: stage, dispatch, return — no event, no drain.
        Op::Validate(0),
        Op::Stage { slot: 0, idx: 0 },
        Op::Dispatch(0),
        // Second drive on the same adapter: straight back into slot 0.
        Op::Validate(0),
        Op::Stage { slot: 0, idx: 0 },
        Op::Dispatch(0),
    ];
    assert!(
        restages_a_slot_without_host_waiting(&legacy),
        "the property does not reject the behaviour it was written to reject"
    );
}

/// Driving the same stager twice must host-wait before re-entering slot 0.
///
/// This is the cross-drive case the fast path got wrong. The uniform path costs
/// one event record and one wait per drive; that is the price.
#[test]
fn a_second_drive_host_waits_before_reusing_the_first_drives_slot() {
    let feeder = RecordingFeeder::all_stageable(1);
    let mut stager = RecordingStager::new(1);
    drive_shards(&feeder, &mut stager, &plan_all(1, 4)).unwrap();
    let after_first = stager.ops().len();
    drive_shards(&feeder, &mut stager, &plan_all(1, 4)).unwrap();
    let ops = stager.ops();
    assert!(!restages_a_slot_without_host_waiting(&ops));
    assert!(
        ops[after_first..]
            .iter()
            .any(|o| matches!(o, Op::HostWait { slot: 0, .. })),
        "the second drive re-entered slot 0 without a host-wait: {ops:?}"
    );
}

/// The slot advances once per staged shard, cycling through the whole ring.
///
/// This is *not* redundant with the host-wait property, and it is worth saying
/// why rather than asserting it: a slot index that never advanced would host-
/// wait before every stage and satisfy that property completely, while
/// serialising the pipeline down to one buffer and destroying the overlap the
/// ring exists for. Only this test sees that.
///
/// It does **not** distinguish `(slot + 1) % RING` from `slot ^= 1`; at
/// `RING == 2` those are the same function, and claiming otherwise would be a
/// bar this fixture cannot carry.
#[test]
fn the_ring_rotates_modulo_ring_width() {
    let feeder = RecordingFeeder::all_stageable(6);
    let mut stager = RecordingStager::new(6);
    drive_shards(&feeder, &mut stager, &plan_all(6, 4)).unwrap();
    let staged: Vec<usize> = stager
        .ops()
        .iter()
        .filter_map(|o| match o {
            Op::Stage { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    assert_eq!(staged, (0..6).map(|n| n % RING).collect::<Vec<_>>());
}

/// A host-wait awaits the event **this slot's own upload** recorded, not
/// whichever event happened to be most recent.
#[test]
fn the_host_wait_awaits_this_slots_own_upload_event() {
    let feeder = RecordingFeeder::all_stageable(5);
    let mut stager = RecordingStager::new(5);
    drive_shards(&feeder, &mut stager, &plan_all(5, 4)).unwrap();
    let ops = stager.ops();
    let mut last_on_slot = [None::<u64>; RING];
    for op in &ops {
        match op {
            Op::HostWait { slot, awaited } => {
                assert_eq!(
                    *awaited, last_on_slot[*slot],
                    "host-wait on slot {slot} awaited {awaited:?}, but that slot's last \
                     upload event was {:?}",
                    last_on_slot[*slot]
                );
                last_on_slot[*slot] = None;
            }
            Op::GateCopyToCompute { slot, event } => last_on_slot[*slot] = Some(*event),
            _ => {}
        }
    }
    // Not vacuous: at least one host-wait actually had an event to await.
    assert!(ops.iter().any(|o| matches!(
        o,
        Op::HostWait {
            awaited: Some(_),
            ..
        }
    )));
}

#[test]
fn drain_runs_exactly_once_and_after_the_last_dispatch() {
    let feeder = RecordingFeeder::all_stageable(4);
    let mut stager = RecordingStager::new(4);
    drive_shards(&feeder, &mut stager, &plan_all(4, 4)).unwrap();
    let ops = stager.ops();
    let drains: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, Op::Drain { .. }))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(drains.len(), 1, "drain must run exactly once");
    let last_dispatch = ops
        .iter()
        .rposition(|o| matches!(o, Op::Dispatch(_)))
        .unwrap();
    assert!(drains[0] > last_dispatch);
    assert_eq!(drains[0], ops.len() - 1, "drain must be the last op");
}

/// A failure anywhere still drains, and still surfaces the original error.
///
/// Leaving a DMA in flight lets the caller drop the pinned buffers underneath
/// it — the error path is exactly where that is most likely.
#[test]
fn drain_runs_on_every_error_path_and_does_not_mask_the_error() {
    // (a) the source fails to produce a shard
    let feeder = RecordingFeeder::all_stageable(5).failing_read_at(3);
    let mut stager = RecordingStager::new(5);
    let err = drive_shards(&feeder, &mut stager, &plan_all(5, 4)).unwrap_err();
    assert!(
        matches!(err, GpuError::InvalidShard(_)),
        "a host-side read failure must classify as InvalidShard, got {err:?}"
    );
    let ops = stager.ops();
    assert!(matches!(ops.last(), Some(Op::Drain { .. })));
    assert!(
        !ops.iter().any(|o| matches!(o, Op::Stage { idx: 3.., .. })),
        "shards at or beyond the failure were staged: {ops:?}"
    );

    // (b) validation rejects a shard
    let feeder = RecordingFeeder::all_stageable(5);
    let mut stager = RecordingStager::new(5).failing_validate_at(2);
    let err = drive_shards(&feeder, &mut stager, &plan_all(5, 4)).unwrap_err();
    assert!(matches!(err, GpuError::InvalidShard(_)));
    assert!(matches!(stager.ops().last(), Some(Op::Drain { .. })));

    // (c) the consumer's kernel launch fails
    let feeder = RecordingFeeder::all_stageable(5);
    let mut stager = RecordingStager::new(5).failing_dispatch_at(1);
    let err = drive_shards(&feeder, &mut stager, &plan_all(5, 4)).unwrap_err();
    assert!(matches!(err, GpuError::KernelLaunchFailed(_)));
    let ops = stager.ops();
    assert!(matches!(ops.last(), Some(Op::Drain { .. })));
    // Both slots still had an upload event outstanding: shard 0's (nothing
    // host-waited it, because shard 1 took the *other* slot) and shard 1's
    // (its dispatch failed after the copy->compute gate recorded one). A drain
    // that only walked the current slot would leave shard 0's DMA in flight.
    assert_eq!(
        ops.last(),
        Some(&Op::Drain {
            awaited: vec![0, 2]
        })
    );
}

/// A rejected shard consumes no slot and does not rotate the ring.
///
/// `GpuPreprocessedShardSource` advances a `global_row_offset` per *dispatched*
/// shard, so a skip that rotated would desynchronise every later shard's row
/// addressing from the ring it is staged through.
#[test]
fn an_unstageable_shard_skips_without_rotating_the_ring() {
    // Shard 1 is empty.
    let feeder = RecordingFeeder::new(&[true, false, true, true]);
    let mut stager = RecordingStager::new(4);
    drive_shards(&feeder, &mut stager, &plan_all(4, 4)).unwrap();
    let ops = stager.ops();
    let staged: Vec<(usize, usize)> = ops
        .iter()
        .filter_map(|o| match o {
            Op::Stage { slot, idx } => Some((*idx, *slot)),
            _ => None,
        })
        .collect();
    // 0 -> slot 0, (1 skipped), 2 -> slot 1, 3 -> slot 0.
    assert_eq!(staged, vec![(0, 0), (2, 1), (3, 0)]);
    // A skipped shard is not validated either — nothing about it is touched.
    assert!(!ops.contains(&Op::Validate(1)));
    assert!(!ops.contains(&Op::Dispatch(1)));
}

/// The memo makes validation once-per-index for the stager's whole lifetime,
/// which is what keeps GPU DE's 123 per-chunk drives from re-scanning O(nnz).
#[test]
fn the_memo_validates_each_index_once_across_passes() {
    let feeder = RecordingFeeder::all_stageable(3);
    let mut stager = RecordingStager::new(3);
    for _ in 0..4 {
        drive_shards(&feeder, &mut stager, &plan_all(3, 4)).unwrap();
    }
    let validates: Vec<usize> = stager
        .ops()
        .iter()
        .filter_map(|o| match o {
            Op::Validate(i) => Some(*i),
            _ => None,
        })
        .collect();
    assert_eq!(validates, vec![0, 1, 2], "a shard was re-validated");
    assert_eq!(stager.memo.validated_count(), 3);
}

/// A shard that fails validation is **not** memoised, so a later pass rejects
/// it again rather than waving it through.
#[test]
fn a_failed_validation_is_not_memoised() {
    let feeder = RecordingFeeder::all_stageable(3);
    let mut stager = RecordingStager::new(3).failing_validate_at(1);
    assert!(drive_shards(&feeder, &mut stager, &plan_all(3, 4)).is_err());
    assert!(drive_shards(&feeder, &mut stager, &plan_all(3, 4)).is_err());
    let validates: Vec<usize> = stager
        .ops()
        .iter()
        .filter_map(|o| match o {
            Op::Validate(i) => Some(*i),
            _ => None,
        })
        .collect();
    // Shard 0 passed on the first drive and is memoised, so the second drive
    // does not re-validate it. Shard 1 failed and is not, so it is scanned
    // again — both halves of the memo's contract in one trace.
    assert_eq!(
        validates,
        vec![0, 1, 1],
        "shard 1 must be re-validated on the second pass, not trusted"
    );
}

/// A selected plan drives exactly its own indices, in its own order, and the
/// ring rotates per *staged* shard rather than per shard index.
#[test]
fn a_selected_plan_drives_only_its_own_indices() {
    let feeder = RecordingFeeder::all_stageable(10);
    let mut stager = RecordingStager::new(10);
    drive_shards(&feeder, &mut stager, &plan_selected(&[2, 5, 9], 4)).unwrap();
    let staged: Vec<(usize, usize)> = stager
        .ops()
        .iter()
        .filter_map(|o| match o {
            Op::Stage { slot, idx } => Some((*idx, *slot)),
            _ => None,
        })
        .collect();
    assert_eq!(staged, vec![(2, 0), (5, 1), (9, 0)]);
    assert_eq!(*feeder.reads.borrow(), vec![2, 5, 9]);
}

#[test]
fn an_empty_plan_touches_nothing() {
    let feeder = RecordingFeeder::all_stageable(4);
    let mut stager = RecordingStager::new(4);
    drive_shards(&feeder, &mut stager, &plan_all(0, 4)).unwrap();
    drive_shards(&feeder, &mut stager, &plan_selected(&[], 4)).unwrap();
    assert!(stager.ops().is_empty(), "an empty plan must not even drain");
    assert!(feeder.reads.borrow().is_empty());
}

/// Drain's error surfaces when the feed succeeded, and does **not** displace a
/// feed error when both fail.
///
/// `drive_shards` states the policy as `feed_result.and(drain_result)`, and
/// `drain_runs_on_every_error_path_and_does_not_mask_the_error` was written to
/// pin it — but that test's fake always drained `Ok`, so it only ever proved
/// "drain runs after a feed failure". A mutation giving `drain_result` priority
/// passed it (codex - gpt-5.6-sol). Both halves are asserted here.
#[test]
fn drain_error_surfaces_alone_but_never_displaces_a_feed_error() {
    // Feed succeeds, drain fails -> the drain error is returned.
    let feeder = RecordingFeeder::all_stageable(3);
    let mut stager = RecordingStager::new(3);
    stager.drain_error = Some("drain blew up".to_string());
    let err = drive_shards(&feeder, &mut stager, &plan_all(3, 3)).unwrap_err();
    assert!(
        format!("{err}").contains("drain blew up"),
        "a drain failure on a clean feed must be surfaced, got: {err}"
    );

    // Feed fails AND drain fails -> the FEED error wins; the drain error must
    // not mask the original diagnosis.
    let feeder = RecordingFeeder::all_stageable(3);
    let mut stager = RecordingStager::new(3);
    stager.fail_validate_at = Some(1);
    stager.drain_error = Some("drain blew up".to_string());
    let err = drive_shards(&feeder, &mut stager, &plan_all(3, 3)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        !msg.contains("drain blew up"),
        "the drain error masked the feed error, got: {msg}"
    );
    // ...and drain still ran, which is the other half of the policy.
    assert!(
        stager
            .ops
            .borrow()
            .iter()
            .any(|o| matches!(o, Op::Drain { .. })),
        "drain must run even when the feed failed"
    );
}

// ── StagingPlan::for_source ─────────────────────────────────────────────
//
// The GPU DE routes reach the same `prefetch` pipeline the CPU ones do, but
// through `for_each_shard_ordered_uncached_selected`, which deliberately does
// not consult `visible_shard_indices`. So the row-projection skip has to be
// asked for when the plan is built, and this is where. Testable on a CPU host:
// the plan is arithmetic over the source's answer, with no device in it.

/// A source that answers a plan, as a row-filtering one does.
struct PlannedSource {
    n_shards: usize,
    plan: Option<Vec<usize>>,
}

impl ShardSource for PlannedSource {
    fn n_shards(&self) -> usize {
        self.n_shards
    }
    fn n_obs(&self) -> usize {
        self.n_shards
    }
    fn n_vars(&self) -> usize {
        1
    }
    fn visible_shard_indices(&self) -> Option<Vec<usize>> {
        self.plan.clone()
    }
    fn read_shard(&self, _shard_idx: usize) -> scx_format_io::Result<scx_sparse::ScxCsr> {
        Ok(scx_sparse::ScxCsr::new_unchecked(
            (1, 1),
            vec![0, 0],
            vec![],
            vec![],
        ))
    }
}

/// Without this the GPU CSR staging path stages every shard of a row-windowed
/// handle, decoding and uploading the ones the window empties only for
/// `drive_shards` to skip them at the far end.
#[test]
fn for_source_honours_a_row_filtering_sources_plan() {
    let src = PlannedSource {
        n_shards: 5,
        plan: Some(vec![0, 4]),
    };
    let plan = StagingPlan::for_source(&src, 4);
    assert_eq!(plan.indices, vec![0, 4]);
    assert_eq!(plan.depth, 4, "the depth is the caller's, not the source's");
}

/// "No filter" must stay "every shard", not become an empty plan — the arm
/// every unsubset GPU DE call takes.
#[test]
fn for_source_stages_every_shard_when_there_is_no_filter() {
    let src = PlannedSource {
        n_shards: 3,
        plan: None,
    };
    assert_eq!(StagingPlan::for_source(&src, 2).indices, vec![0, 1, 2]);
}

/// A plan that keeps nothing stays empty rather than falling back to the
/// whole file: `adata[:0]` stages nothing, and `drive_shards` handles it.
#[test]
fn for_source_keeps_an_empty_plan_empty() {
    let src = PlannedSource {
        n_shards: 3,
        plan: Some(vec![]),
    };
    assert!(StagingPlan::for_source(&src, 2).indices.is_empty());
}
