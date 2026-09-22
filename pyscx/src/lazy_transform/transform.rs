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

// ---------------------------------------------------------------------------
// ScxLazyTransformedDataset
// ---------------------------------------------------------------------------
