// Streaming-matrix abstraction shared across conversion readers.
//
// Phase 0.1 of REAL-WORLD-UX-FEATS. Dense h5ad (Phase 1), CSC
// external-memory transpose (Phase 2), h5mu per-modality (Phase 3),
// and Zarr-backed readers (Phase 4) all implement `CsrShardStream`
// so the writer-side coordinator can drive any of them uniformly.

use super::pipeline::ConvertError;

/// Major axis of the source matrix. Streaming readers always emit
/// CSR shards downstream; `Column` only appears as a marker on
/// readers that perform an internal column-major → row-major
/// transpose before yielding shards (e.g. the Phase 2 CSC reader).
pub enum MajorAxis {
    Row,
    Column,
}

/// A row-range slice of a CSR matrix produced by a
/// [`CsrShardStream`]. `indptr` is shard-local: the first element is
/// always 0 and `indptr.len() == n_rows + 1`. `indices` / `values`
/// are concatenated across the shard's rows.
pub struct StreamedCsrShard {
    pub row_start: u64,
    pub n_rows: u32,
    pub n_cols: u32,
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    /// Optional human-readable label of the source matrix (e.g.
    /// `"X"`, `"layers/spliced"`, `"mod/rna/X"`). Used by the
    /// writer coordinator only for diagnostics; carrying it on the
    /// shard avoids a parallel side-channel.
    pub source_name: Option<String>,
    /// Number of `(row, col)` duplicates that were merged into this
    /// shard's values during canonicalisation (Phase 2 CSC external
    /// transpose). Zero for any reader that doesn't canonicalise.
    /// The writer coordinator surfaces a single
    /// `DuplicateCoordinatesMerged` warning per non-zero shard.
    pub duplicates_merged: u64,
}

/// Cursor over the rows of a single sparse matrix. Drained by the
/// writer coordinator one shard at a time.
pub trait CsrShardStream {
    fn n_obs(&self) -> u64;
    fn n_vars(&self) -> u64;
    /// Human-readable label used in diagnostics and per-shard
    /// section names. Examples: `"X"`, `"layers/spliced"`,
    /// `"mod/rna"`.
    fn source_matrix_name(&self) -> &str;
    /// Pull the next shard of up to `target_rows` rows. Returns
    /// `Ok(None)` once all rows have been emitted.
    fn next_csr_shard(
        &mut self,
        target_rows: usize,
    ) -> Result<Option<StreamedCsrShard>, ConvertError>;
}
