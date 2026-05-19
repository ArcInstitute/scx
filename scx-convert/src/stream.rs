// Streaming-matrix abstraction shared across conversion readers.
//
// Dense h5ad, CSC external-memory transpose, h5mu per-modality, and
// (future) Zarr-backed readers all implement `CsrShardStream` so the
// writer-side coordinator can drive any of them uniformly.

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

    /// If this reader supports parallel row-range reads,
    /// return a `&dyn IndexedCsrShardStream` view of `self`. Default
    /// implementation returns `None`, which routes the writer
    /// coordinator to the sequential path. Readers that can safely
    /// be driven from multiple worker threads (CSR h5ad, dense h5ad,
    /// in-memory CSC) override this to return `Some(self)`.
    fn as_indexed(&self) -> Option<&dyn IndexedCsrShardStream> {
        None
    }
}

/// Sibling of [`CsrShardStream`] for readers that can serve
/// independent row-range reads concurrently from multiple worker
/// threads.
///
/// Implementers must be `Send + Sync` and stateless across calls
/// (no internal cursor). The streaming writer coordinator partitions
/// the matrix into `[(row_start, n_rows)]` ranges via
/// [`crate::pipeline::compute_shard_row_ranges`] and fans them out
/// across a rayon worker pool. The encoded shards funnel through a
/// bounded reorder buffer and are written in shard-index order so
/// the output `.scx` file is byte-identical to the sequential path.
///
/// The CSC external-memory bucket transposer (Phase 2) does not
/// implement this trait — its bucket pipeline is inherently
/// sequential.
pub trait IndexedCsrShardStream: Send + Sync {
    fn n_obs(&self) -> u64;
    fn n_vars(&self) -> u64;
    fn source_matrix_name(&self) -> &str;
    /// Read rows `[row_start, row_start + n_rows)` and return them
    /// as a `StreamedCsrShard`. `row_start + n_rows` must not exceed
    /// `self.n_obs()`; callers (the parallel coordinator) compute
    /// ranges from `compute_shard_row_ranges` so this invariant is
    /// enforced by construction.
    fn read_range(&self, row_start: u64, n_rows: u32) -> Result<StreamedCsrShard, ConvertError>;
}
