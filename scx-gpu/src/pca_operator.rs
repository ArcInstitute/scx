//! One randomized-PCA power loop for both the streaming and resident paths.
//!
//! GPU randomized PCA is implemented twice. `CenteredSparseOperator`
//! (`linear_operator.rs`) streams the matrix shard by shard on every multiply;
//! `gpu_pca_resident.rs` holds one device-resident CSR and multiplies against a
//! fixed cuSPARSE descriptor. Both then run the *same* sequence of multiplies
//! and QR factorisations, written out separately — and the two copies drifted:
//! review §8.11 is exactly that, `spmm_policy="deterministic"` honoured by the
//! resident copy and silently dropped by the streaming one, on the out-of-VRAM
//! path that is the flagship use case for streaming in the first place.
//!
//! [`run_power_loop`] is that sequence written once. Like
//! [`drive_shards`](crate::staging_driver::drive_shards) it contains **no CUDA
//! call and names no buffer type**: it emits an ordered series of
//! [`PcaOperator`] calls, and each implementation decides what a multiply means.
//!
//! # Why the seam is here
//!
//! `gpu_pca.rs`, `gpu_pca_resident.rs` and `linear_operator.rs` hold 16 tests
//! between them and **all 16** are `#[ignore = "requires a CUDA GPU"]`. The
//! power loop is pure ordering — no arithmetic of its own — yet no reachable
//! test on a CPU host exercises it, so rewriting it could not be watched red,
//! which is this series' first ground rule. Putting the ordering on one side of
//! a trait boundary and the CUDA on the other lets `pca_operator_tests.rs`
//! drive it with a recording fake, on CPU, with no `#[ignore]`.
//!
//! What the fake cannot prove stays with the GPU suite: that a multiply
//! computes `(X − μ)·V`, that QR returns an orthonormal basis, that the two
//! paths agree numerically, and that the cuSPARSE algorithm actually launched
//! is the one the policy asked for.

// PR A of ORG-8.20-2 lands the seam with no production caller: the ordering is
// testable on CPU from the moment it exists, and the two migrations that consume
// it are separately reviewable. PR B (resident) is the first consumer and
// **deletes this attribute**; a CI branch then asserts it is gone, so "temporary"
// is enforced rather than asserted. If something in here is genuinely unused
// after both migrations, delete it — do not re-silence it.
#![allow(dead_code)]

use crate::error::GpuError;

/// Which of the two power-loop work buffers a step operates on.
///
/// The driver never sees a `CudaSlice`; it names the buffer and the operator
/// resolves it. That is what keeps this module CUDA-free — and it also lets the
/// implementation, not the driver, own the shape: `Y` is always `n_obs × k` and
/// `Z` is always `n_vars × k`, so a `qr` step needs no row count passed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PcaBuf {
    /// `n_obs × k`, col-major. Holds `Y`, then `Q` after each factorisation.
    Y,
    /// `n_vars × k`, col-major. Holds `Z` inside the loop and the final `B`.
    Z,
}

/// Right-hand side of a forward multiply.
///
/// The seed multiply reads the random test matrix `Ω`; every later one reads
/// `Z`. Both are `n_vars × k` col-major, which is exactly why this has to be
/// named rather than inferred — the two are indistinguishable by shape, so a
/// loop that read `Z` on the seed step would produce a plausible embedding from
/// an uninitialised basis and nothing downstream would notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForwardOperand {
    /// The random `n_vars × k` test matrix, used once.
    Omega,
    /// The `Z` buffer, used by every power iteration.
    Z,
}

/// The three operations a randomized-PCA power loop is built from.
///
/// Implemented twice — once streaming per shard, once against a resident
/// descriptor — and driven by exactly one loop. Each method mutates the
/// operator's own buffers; nothing is passed in or returned, so the driver
/// holds no device state.
pub(crate) trait PcaOperator {
    /// `Y ← (X − μ)·src`.
    fn matmat(&mut self, src: ForwardOperand) -> Result<(), GpuError>;

    /// `Z ← (X − μ)ᵀ·Y`.
    fn rmatmat(&mut self) -> Result<(), GpuError>;

    /// `buf ← qr(buf)`, replacing the buffer with an orthonormal basis for its
    /// column space.
    fn qr(&mut self, buf: PcaBuf) -> Result<(), GpuError>;
}

/// Run the randomized-PCA power loop, leaving `Q` in [`PcaBuf::Y`] and
/// `B = (X − μ)ᵀ·Q` in [`PcaBuf::Z`] — the two buffers the caller hands to the
/// downstream SVD.
///
/// The sequence, for `n_power_iterations = N`:
///
/// ```text
/// matmat(Ω) → qr(Y) → [ rmatmat → qr(Z) → matmat(Z) → qr(Y) ] × N → rmatmat
/// ```
///
/// `N = 0` is meaningful and is the degenerate case both previous copies
/// already handled: seed, orthonormalise, project. It is not a no-op, and it is
/// tested.
///
/// Note the loop body's trailing `qr(Y)` and the final `rmatmat`: the last
/// factorisation inside the loop is what leaves `Y` holding `Q` rather than an
/// unnormalised `Y`, and the closing `rmatmat` reads that `Q`. Dropping either
/// still produces a full-rank result of the right shape, which is why the
/// ordering is asserted explicitly rather than inferred from a numerical test.
pub(crate) fn run_power_loop<O: PcaOperator + ?Sized>(
    op: &mut O,
    n_power_iterations: usize,
) -> Result<(), GpuError> {
    // Y = (X − μ)·Ω; Q = qr(Y).
    op.matmat(ForwardOperand::Omega)?;
    op.qr(PcaBuf::Y)?;

    for _ in 0..n_power_iterations {
        // Z = (X − μ)ᵀ·Q; Q_Z = qr(Z).
        op.rmatmat()?;
        op.qr(PcaBuf::Z)?;
        // Y = (X − μ)·Q_Z; Q = qr(Y).
        op.matmat(ForwardOperand::Z)?;
        op.qr(PcaBuf::Y)?;
    }

    // B = (X − μ)ᵀ·Q, into Z.
    op.rmatmat()
}

#[cfg(test)]
#[path = "pca_operator_tests.rs"]
mod tests;
