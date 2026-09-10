// Streaming-matrix abstraction shared across conversion readers.
//
// Dense h5ad, CSC external-memory transpose, h5mu per-modality, and
// (future) Zarr-backed readers all implement `CsrShardStream` so the
// writer-side coordinator can drive any of them uniformly.

use scx_format_io::modality::ModalityType;

use super::pipeline::ConvertError;

/// Working-set bytes for one CSR shard of `nnz` non-zeros over `n_rows` rows,
/// in two flavours because ingest and export hold different things:
///
/// * `shard_working_set_bytes` — the **ingest** worker's whole phase, 48 B/nnz
///   plus the indptr: the decoded `i32` indices + `f32` values, the encoder's
///   own re-serialised copy of them, and the framed encode's buffers for the
///   two candidates `codec="auto"` runs concurrently. Used by
///   `IndexedCsrShardStream::per_worker_bytes`.
/// * `shard_decode_working_set_bytes` — the **export** side, 16 B/nnz plus the
///   indptr: payload and decoder scratch, no encode term, because
///   `h5ad/stream_write.rs` contains no encode call. Used by
///   `per_shard_export_bytes`.
///
/// They shared one function until the encode charge landed, on the argument
/// that a single model cannot drift. That argument holds only while the two
/// phases are the same, and they are not — the shared model over-derated a
/// budgeted export threefold for memory it never holds.
pub(crate) use crate::budget::{shard_decode_working_set_bytes, shard_working_set_bytes};

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
/// the pipeline coordinator's `compute_shard_row_ranges` and fans them out
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

    /// Hard upper bound on rows the reader can serve in a single
    /// [`read_range`](Self::read_range) call. `None` means no cap
    /// (CSR and in-memory CSC readers). `Some(n)` clamps the
    /// parallel coordinator's partition so the shard-row ranges it
    /// emits never exceed what the reader can handle. The dense
    /// reader returns `Some(max_slab_rows)` when `memory_budget`
    /// is set, matching the sequential `next_csr_shard` clamp.
    fn max_slab_rows(&self) -> Option<u32> {
        None
    }

    /// Per-worker working-set estimate in bytes used by the
    /// memory-budget derate in the parallel-coordinator dispatcher.
    ///
    /// The default impl is a **guess**, and not a bound in either
    /// direction: it assumes a density by modality (`Atac` → 10 %,
    /// everything else → 5 %) and charges
    /// [`crate::budget::WORKER_PHASE_BYTES_PER_NNZ`] per nonzero — the
    /// payload, the reader's rebuild scratch and the framed encode's
    /// buffers. Above that density it under-estimates, and a
    /// dense-stored-as-CSR matrix (density 1.0) under-estimates by 10-20x.
    /// It exists only for readers that cannot cheaply know their nnz.
    /// Readers with a resident `indptr` (CSR h5ad) override this to derive
    /// the max-shard nnz exactly via [`shard_working_set_bytes`];
    /// [`crate::permuted_reader::PermutedCsrReader`] overrides it again
    /// because it reorders rows and the source-aligned maximum does not
    /// describe the shard it emits; dense readers override it to size the
    /// dense slab buffer instead.
    /// The source matrix's row prefix-sum (`indptr`), when the reader holds it
    /// resident and can hand it out without I/O.
    ///
    /// Exists for one caller: `PermutedCsrReader`, which reorders rows and so
    /// cannot price its output shards from the *source*-aligned windows
    /// [`Self::per_worker_bytes`] scans. `None` means "I cannot answer cheaply",
    /// and the adapter falls back to delegating.
    fn source_row_indptr(&self) -> Option<&[i64]> {
        None
    }

    fn per_worker_bytes(&self, shard_target_rows: u32, modality_type: ModalityType) -> u64 {
        let density_den: u64 = match modality_type {
            ModalityType::Atac => crate::budget::PARALLEL_DENSITY_ATAC_DEN,
            _ => crate::budget::PARALLEL_DENSITY_DEFAULT_DEN,
        };
        crate::budget::estimated_worker_bytes(shard_target_rows as u64, self.n_vars(), density_den)
            .max(1)
    }
}
