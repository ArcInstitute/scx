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
}

impl Transform {
    /// Returns `true` iff the transform's output for a given matrix
    /// element depends only on its own column (not on row sums or
    /// per-row factors).
    ///
    /// Used by `LazyShardSource`'s `ColumnShardSource` implementation
    /// and `ScxBackedSparseDataset::as_column_source()` as the
    /// transform-chain compatibility test for CSC dispatch.
    ///
    /// - `Log1p`: `ln(x + 1)` is element-wise, no row context. **true**
    /// - `NormalizeTotal`: divides by per-row sum. **false**
    /// - `RowScale`: multiplies each row by a per-row factor. **false**
    pub fn is_column_local(&self) -> bool {
        matches!(self, Transform::Log1p)
    }
}

// ---------------------------------------------------------------------------
// ScxLazyTransformedDataset
// ---------------------------------------------------------------------------
