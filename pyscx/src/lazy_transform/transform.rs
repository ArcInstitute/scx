// Transform enum — per-row lazy operations (NormalizeTotal / Log1p / RowScale).
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use std::sync::Arc;

/// Per-shard transform operations applied lazily during __getitem__.
#[derive(Clone, Debug)]
pub enum Transform {
    /// Divide each row by its precomputed sum, multiply by target_sum.
    /// Semantically equivalent to sc.pp.normalize_total().
    NormalizeTotal {
        row_sums: Arc<Vec<f64>>,
        target_sum: f64,
    },

    /// Element-wise ln(x + 1) on non-zero values.
    /// Semantically equivalent to sc.pp.log1p().
    Log1p,

    /// Per-row multiply by a scalar vector.
    /// Used by normalize_total(inplace=False) which returns X * (target_sum / row_sums).
    RowScale { factors: Arc<Vec<f64>> },

    /// Element-wise multiply by a single matrix-wide scalar.
    /// Used by PFlog (v4): the delta source is `Scale{4α} → Log1p`, i.e.
    /// `log1p(4α·x)`. Depends only on the value itself — no row or column
    /// context at all.
    Scale { factor: f64 },
}

impl Transform {
    /// Returns `true` iff this transform can be applied while reading the
    /// matrix **column-major**, which is what gates CSC dispatch.
    ///
    /// The question is: *is an element's output computable from the element,
    /// its column, and its row index?* Every variant satisfies it, so this is
    /// `true` across the board — but it is written as an exhaustive `match`
    /// rather than a bare `true` so that a future variant has to be considered
    /// here instead of being silently admitted.
    ///
    /// This replaced an `is_column_local()` predicate that asked whether an
    /// element's output depends only on its own column. That is the right
    /// question for a consumer which never learns which row a value came from,
    /// and the wrong one for a CSC reader, which always learns it:
    /// `ScxCsc::indices` *is* the global row index of each nonzero, and the
    /// kernels already materialise it (`scx-accel/src/csc/wilcoxon.rs` computes
    /// `let row = csc.indices[j]` in its inner loop and indexes a dense buffer
    /// with it). `NormalizeTotal` and `RowScale` are not column-local, but they
    /// carry their per-row vectors with them (`row_sums`, `factors`, both at
    /// physical global-row length), so serving them column-major is a lookup at
    /// an index already in hand — not a pass, and not a redesign.
    ///
    /// Both are element-wise **maps**, not reductions, so applying them on the
    /// CSC path is bit-identical to the CSR path rather than merely close:
    /// there is no accumulation whose order could differ.
    /// `shard_source_tests.rs` pins that bit-for-bit.
    ///
    /// This does not relax the other two CSC preconditions, and in particular
    /// an active row-deletion vector is still disqualifying — precisely because
    /// CSC `indices` encode *global* rows and a deletion vector renumbers the
    /// live ones, so a row-indexed transform would read the wrong entry.
    pub fn is_csc_applicable(&self) -> bool {
        match self {
            Transform::Log1p
            | Transform::Scale { .. }
            | Transform::NormalizeTotal { .. }
            | Transform::RowScale { .. } => true,
        }
    }
}

// ---------------------------------------------------------------------------
// ScxLazyTransformedDataset
// ---------------------------------------------------------------------------
