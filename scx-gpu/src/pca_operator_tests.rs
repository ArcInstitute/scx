//! CPU tests for [`run_power_loop`]'s ordering, on any host.
//!
//! These deliberately carry **no** `require_gpu!()` and **no** `#[ignore]`.
//! `tests/gpu_test_gating.rs` rule 1 is an *iff* — a test invokes a device gate
//! exactly when it carries the ignore reason — so an ungated, un-ignored test is
//! the correct shape here provided it never opens a device, which is the whole
//! reason the seam exists: every one of the 16 tests across `gpu_pca.rs`,
//! `gpu_pca_resident.rs` and `linear_operator.rs` is in the ignored set, and the
//! power loop was entirely inside it.
//!
//! Two fakes, because a call-sequence assertion and a data-flow assertion fail
//! on different bugs. [`RecordingOperator`] catches a step that moved, was
//! dropped, or was added. [`SymbolicOperator`] catches a step that stayed put
//! while reading the wrong buffer — a swap the call sequence cannot see, since
//! both operands of a forward multiply are `n_vars × k`.

use super::*;
use std::cell::RefCell;

/// One observable step of the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Matmat(ForwardOperand),
    Rmatmat,
    Qr(PcaBuf),
}

use ForwardOperand::{Omega, Z as FromZ};
use PcaBuf::{Y as BufY, Z as BufZ};

/// Records the exact sequence of calls, optionally failing the `n`th.
struct RecordingOperator {
    steps: RefCell<Vec<Step>>,
    /// Zero-based index of the call that returns an error, standing in for a
    /// cuSPARSE/cuSOLVER failure part-way through the loop.
    fail_at: Option<usize>,
}

impl RecordingOperator {
    fn new() -> Self {
        Self {
            steps: RefCell::new(Vec::new()),
            fail_at: None,
        }
    }

    fn failing_at(n: usize) -> Self {
        Self {
            steps: RefCell::new(Vec::new()),
            fail_at: Some(n),
        }
    }

    /// Push `step`, then fail if this call was the designated one. The step is
    /// recorded either way: a failing call still *happened*, and the tests below
    /// assert on how far the loop got.
    fn record(&self, step: Step) -> Result<(), GpuError> {
        let mut steps = self.steps.borrow_mut();
        steps.push(step);
        if self.fail_at == Some(steps.len() - 1) {
            return Err(GpuError::KernelLaunchFailed(format!(
                "fake failure at {step:?}"
            )));
        }
        Ok(())
    }

    fn steps(&self) -> Vec<Step> {
        self.steps.borrow().clone()
    }
}

impl PcaOperator for RecordingOperator {
    fn matmat(&mut self, src: ForwardOperand) -> Result<(), GpuError> {
        self.record(Step::Matmat(src))
    }
    fn rmatmat(&mut self) -> Result<(), GpuError> {
        self.record(Step::Rmatmat)
    }
    fn qr(&mut self, buf: PcaBuf) -> Result<(), GpuError> {
        self.record(Step::Qr(buf))
    }
}

/// The sequence [`run_power_loop`] is specified to emit, spelled out
/// independently of the implementation so a change to one has to be made to the
/// other deliberately.
fn expected_steps(n_power_iterations: usize) -> Vec<Step> {
    let mut want = vec![Step::Matmat(Omega), Step::Qr(BufY)];
    for _ in 0..n_power_iterations {
        want.push(Step::Rmatmat);
        want.push(Step::Qr(BufZ));
        want.push(Step::Matmat(FromZ));
        want.push(Step::Qr(BufY));
    }
    want.push(Step::Rmatmat);
    want
}

#[test]
fn emits_the_specified_sequence_for_each_iteration_count() {
    for n in 0..=3 {
        let mut op = RecordingOperator::new();
        run_power_loop(&mut op, n).unwrap();
        assert_eq!(
            op.steps(),
            expected_steps(n),
            "power loop sequence differs at n_power_iterations = {n}"
        );
    }
}

/// `3 + 4N` is the shape the caller's VRAM and time budgets are reasoned
/// against: one seed multiply, one closing multiply, one factorisation of the
/// seed, and four steps per iteration. Asserted as arithmetic rather than as a
/// list so an off-by-one in the *loop bound* — as opposed to a reordering — is
/// named for what it is.
#[test]
fn step_count_is_three_plus_four_per_iteration() {
    for n in 0..=4 {
        let mut op = RecordingOperator::new();
        run_power_loop(&mut op, n).unwrap();
        assert_eq!(
            op.steps().len(),
            3 + 4 * n,
            "n_power_iterations = {n} should emit 3 + 4·{n} steps"
        );
    }
}

/// `n_power_iterations = 0` is a real configuration, not a no-op: seed,
/// orthonormalise, project. Both previous copies handled it by construction
/// (`for _ in 0..0`), and a rewrite that special-cased it would be the easiest
/// place to lose the closing `rmatmat` — which yields an *empty* `B` and a PCA
/// whose SVD is of an uninitialised buffer.
#[test]
fn zero_power_iterations_still_seeds_factorises_and_projects() {
    let mut op = RecordingOperator::new();
    run_power_loop(&mut op, 0).unwrap();
    assert_eq!(
        op.steps(),
        vec![Step::Matmat(Omega), Step::Qr(BufY), Step::Rmatmat]
    );
}

/// Only the seed multiply reads Ω, and it is the first step. Ω and Z are both
/// `n_vars × k` col-major, so reading the wrong one produces a plausible
/// embedding from an uninitialised or stale basis with nothing downstream to
/// catch it — the sequence test above cannot see it, because the *step* is in
/// the right place.
#[test]
fn omega_is_read_once_and_only_by_the_seed_multiply() {
    let mut op = RecordingOperator::new();
    run_power_loop(&mut op, 3).unwrap();
    let steps = op.steps();

    let omega_positions: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s, Step::Matmat(Omega)))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        omega_positions,
        vec![0],
        "Ω must be the operand of exactly the first multiply; got {steps:?}"
    );
    let from_z = steps
        .iter()
        .filter(|s| matches!(s, Step::Matmat(FromZ)))
        .count();
    assert_eq!(from_z, 3, "every in-loop multiply must read Z");
}

/// The loop must end `… qr(Y) → rmatmat`, because the caller reads `Q` from `Y`
/// and `B` from `Z` after it returns. If the trailing `rmatmat` were emitted
/// before the last `qr(Y)`, `B` would be `(X − μ)ᵀ·Y` for an unnormalised `Y` —
/// a full-rank result of the right shape, and wrong.
#[test]
fn the_last_factorisation_of_y_precedes_the_closing_projection() {
    for n in 1..=3 {
        let mut op = RecordingOperator::new();
        run_power_loop(&mut op, n).unwrap();
        let steps = op.steps();
        let last = steps.len() - 1;
        assert_eq!(steps[last], Step::Rmatmat, "n = {n}");
        assert_eq!(steps[last - 1], Step::Qr(BufY), "n = {n}");
    }
}

/// A failure anywhere aborts immediately: the error propagates and no step after
/// the failing one is issued. On a real device a step that ran after a failed
/// SpMM would read a buffer whose contents are undefined.
#[test]
fn an_error_stops_the_loop_at_the_failing_step() {
    let total = expected_steps(2).len();
    for fail_at in 0..total {
        let mut op = RecordingOperator::failing_at(fail_at);
        let err = run_power_loop(&mut op, 2).unwrap_err();
        assert!(
            matches!(err, GpuError::KernelLaunchFailed(_)),
            "failure at step {fail_at} should propagate the operator's error, got {err:?}"
        );
        assert_eq!(
            op.steps().len(),
            fail_at + 1,
            "failure at step {fail_at} should be the last step issued"
        );
        assert_eq!(
            op.steps(),
            expected_steps(2)[..=fail_at],
            "steps before the failure at {fail_at} should be unchanged"
        );
    }
}

// ---------------------------------------------------------------------------
// Data flow
// ---------------------------------------------------------------------------

/// Evaluates the loop symbolically: each buffer holds an expression string, and
/// each step rewrites it exactly as the real operator would.
///
/// This is the half [`RecordingOperator`] cannot cover. A call sequence proves
/// *when* each step ran; this proves *what each step read*. Together they pin
/// the loop to one meaning.
struct SymbolicOperator {
    /// `n_obs × k`.
    y: String,
    /// `n_vars × k`.
    z: String,
}

impl SymbolicOperator {
    fn new() -> Self {
        Self {
            // Deliberately not "0": an unwritten buffer must be distinguishable
            // from a zeroed one, so a loop that projected before seeding shows
            // up as `Xt·<unwritten Y>` rather than as a plausible zero.
            y: "<unwritten Y>".to_string(),
            z: "<unwritten Z>".to_string(),
        }
    }
}

impl PcaOperator for SymbolicOperator {
    fn matmat(&mut self, src: ForwardOperand) -> Result<(), GpuError> {
        let rhs = match src {
            ForwardOperand::Omega => "omega".to_string(),
            ForwardOperand::Z => self.z.clone(),
        };
        self.y = format!("X·{rhs}");
        Ok(())
    }
    fn rmatmat(&mut self) -> Result<(), GpuError> {
        self.z = format!("Xt·{}", self.y);
        Ok(())
    }
    fn qr(&mut self, buf: PcaBuf) -> Result<(), GpuError> {
        match buf {
            PcaBuf::Y => self.y = format!("qr({})", self.y),
            PcaBuf::Z => self.z = format!("qr({})", self.z),
        }
        Ok(())
    }
}

#[test]
fn each_step_reads_the_buffer_the_previous_step_wrote() {
    // n = 0: Q = qr(X·Ω); B = Xᵀ·Q.
    let mut op = SymbolicOperator::new();
    run_power_loop(&mut op, 0).unwrap();
    assert_eq!(op.y, "qr(X·omega)");
    assert_eq!(op.z, "Xt·qr(X·omega)");

    // n = 1: one full round trip through the transpose and back.
    let mut op = SymbolicOperator::new();
    run_power_loop(&mut op, 1).unwrap();
    assert_eq!(op.y, "qr(X·qr(Xt·qr(X·omega)))");
    assert_eq!(op.z, "Xt·qr(X·qr(Xt·qr(X·omega)))");

    // n = 2 — the production default. Every `qr` wraps the expression the step
    // before it produced, and `B` is built from the final `Q`, not an earlier one.
    let mut op = SymbolicOperator::new();
    run_power_loop(&mut op, 2).unwrap();
    assert_eq!(op.y, "qr(X·qr(Xt·qr(X·qr(Xt·qr(X·omega)))))");
    assert_eq!(op.z, format!("Xt·{}", op.y));
}

/// No buffer is ever read before it is written. Stated separately from the
/// expression assertions above because it is the property that survives a
/// change to the loop: whatever the sequence becomes, `Xt·<unwritten Y>` must
/// never appear.
#[test]
fn no_buffer_is_read_before_it_is_written() {
    for n in 0..=3 {
        let mut op = SymbolicOperator::new();
        run_power_loop(&mut op, n).unwrap();
        assert!(
            !op.y.contains("<unwritten") && !op.z.contains("<unwritten"),
            "n = {n} left an unwritten buffer in the result: Y = {}, Z = {}",
            op.y,
            op.z
        );
    }
}
