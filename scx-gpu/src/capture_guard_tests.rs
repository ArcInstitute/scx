//! CPU tests for the capture-region arm/disarm/check state machine.
//!
//! None of these needs a GPU, which is the point of [`super`] being CUDA-free:
//! every test that can reach a real `cuStreamBeginCapture` is
//! `#[ignore = "requires a CUDA GPU"]`, so before this file the contract could
//! only be watched work on an H100.
//!
//! Each rejection test was written against a copy of the function with *its*
//! clause deleted and confirmed to fail there first.

use super::*;

/// The error text, for tests that assert on wording rather than the variant.
fn msg(e: GpuError) -> String {
    match e {
        GpuError::CaptureViolation(s) => s,
        other => panic!("expected CaptureViolation, got {other:?}"),
    }
}

/// The depth this thread starts a test at. Tests in one process share threads
/// via the libtest pool, so a test that leaked an arm would otherwise poison a
/// later one — asserting the floor here turns that into this test's failure.
fn assert_disarmed() {
    assert!(
        !in_capture(),
        "this thread entered the test still armed — a CaptureScope leaked"
    );
}

#[test]
fn outside_a_capture_region_everything_is_allowed() {
    assert_disarmed();
    assert!(check("GpuDevice::alloc_zeros").is_ok());
    assert!(check("GpuDevice::synchronize").is_ok());
}

#[test]
fn inside_a_capture_region_the_check_rejects() {
    assert_disarmed();
    let _scope = enter();
    assert!(in_capture());
    let err = check("GpuDevice::alloc_zeros")
        .expect_err("an allocation inside a capture region must be refused");
    assert!(matches!(err, GpuError::CaptureViolation(_)));
}

/// The op string is the whole diagnostic: a violation surfaces at the harmony
/// call site as a one-line warning, and this is what tells the reader which of
/// the four kernels in the closure grew an allocation.
#[test]
fn the_error_names_the_operation_that_was_refused() {
    assert_disarmed();
    let _scope = enter();
    let text = msg(check("GpuDevice::htod_copy").unwrap_err());
    assert!(
        text.contains("GpuDevice::htod_copy"),
        "the refused op must appear verbatim: {text}"
    );
    // And it must say what to do about it, not merely that it happened.
    assert!(
        text.contains("Hoist it outside the capture"),
        "the message must carry the remedy: {text}"
    );
}

#[test]
fn dropping_the_scope_disarms() {
    assert_disarmed();
    {
        let _scope = enter();
        assert!(in_capture());
    }
    assert!(!in_capture(), "the region must disarm when the guard drops");
    assert!(check("GpuDevice::alloc_zeros").is_ok());
}

/// The realistic disarm path. `capture_graph`'s closure returns `Result`, and
/// production code inside it is threaded with `?` — so the common exit from a
/// capture region is an early return, not a fall-through. If that left the
/// thread armed, every later allocation on it would be rejected and the process
/// would be broken from then on, not just this capture.
#[test]
fn an_early_return_through_the_question_mark_still_disarms() {
    fn armed_then_fails() -> Result<(), GpuError> {
        let _scope = enter();
        assert!(in_capture());
        Err(GpuError::CudaError("kernel said no".into()))?;
        unreachable!("the line above returns")
    }

    assert_disarmed();
    assert!(armed_then_fails().is_err());
    assert!(!in_capture(), "an error exit must still disarm");
}

/// Unwind is the other exit `Drop` has to cover: `capture_graph` calls a
/// caller-supplied closure, and a panic in it must not leave the thread armed
/// for whatever the test harness (or a rayon worker pool) runs next on it.
#[test]
fn a_panic_inside_the_region_still_disarms() {
    assert_disarmed();
    let caught = std::panic::catch_unwind(|| {
        let _scope = enter();
        assert!(in_capture());
        panic!("kernel builder exploded");
    });
    assert!(
        caught.is_err(),
        "the panic must propagate, not be swallowed"
    );
    assert!(!in_capture(), "unwinding must still disarm");
}

/// Depth, not a flag. Nothing nests captures today; the counter exists so that
/// if something ever does, the inner region's exit does not license
/// allocations that the outer one still forbids.
#[test]
fn nesting_disarms_one_level_at_a_time() {
    assert_disarmed();
    let outer = enter();
    {
        let _inner = enter();
        assert!(check("GpuDevice::alloc_zeros").is_err());
    }
    assert!(
        check("GpuDevice::alloc_zeros").is_err(),
        "leaving the inner region must not disarm the outer one"
    );
    drop(outer);
    assert!(check("GpuDevice::alloc_zeros").is_ok());
}

/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` restricts the capturing thread and
/// leaves every other thread alone. The guard mirrors that scope exactly, which
/// matters because the decode and shard-source paths allocate on their own
/// threads while Harmony captures on the caller's — arming globally would
/// reject work CUDA permits.
#[test]
fn another_thread_is_not_armed_by_this_one() {
    assert_disarmed();
    let _scope = enter();
    assert!(in_capture());

    let other = std::thread::spawn(|| (in_capture(), check("GpuDevice::alloc_zeros").is_ok()))
        .join()
        .expect("worker thread panicked");

    assert_eq!(
        other,
        (false, true),
        "a sibling thread must see no capture and be allowed to allocate"
    );
}

/// A `CaptureViolation` says the *code* did something illegal, not that the
/// device failed — but by `is_runtime_failure`'s own stated rule ("this same
/// input, handed to a non-GPU path, would have produced an answer") it belongs
/// on the runtime side, and the alternate route is exactly the fallback the
/// harmony site already takes: run the kernels directly, no capture.
#[test]
fn a_capture_violation_admits_an_alternate_route() {
    let e = GpuError::CaptureViolation("GpuDevice::alloc_zeros".into());
    assert!(e.is_runtime_failure());
    assert!(e.alternate_route_may_succeed());
}
