//! A cursor over the *visible* rows a shard walk delivers.
//!
//! Every streaming consumer that fills an `n_obs`-indexed structure from a
//! shard walk keeps a running count of visible rows. That count is the right
//! cursor — a shard a row projection empties contributes none, so skipping it
//! advances by zero — but it silently trusts the source's
//! [`visible_shard_indices`](crate::ShardSource::visible_shard_indices) plan to
//! agree with the rows actually handed over.
//!
//! This lives here, below both `scx-accel` and `scx-gpu`, because the check was
//! written once per pass and a review found the fourth pass unguarded. Four
//! construction sites today — `scx-accel`'s CPU gene-chunk fill, its `pts`
//! counting pass, its `GpuRowCursor` (which the three GPU DE CSR passes share),
//! and `scx-gpu`'s pseudobulk pass — covering six shard walks between them.

/// A cursor over the visible rows a shard walk delivers, carrying the two
/// checks that make a source's shard plan falsifiable.
///
/// A plan that **under** covers places every row after the gap one shard too
/// early and still returns a fully formed result; one that **over** covers
/// indexes past the end — on the host a panic, on the device an out-of-bounds
/// read, because the GPU DE and pseudobulk kernels index
/// `cell_to_group[global_row + r]` into a buffer of length `n_obs`.
///
/// Errors are `String`, so this couples to no crate's error enum: `scx-accel`
/// wraps them in `AccelError::ShapeError` and `scx-gpu` in
/// `GpuError::InvalidShard`.
#[derive(Debug)]
pub struct VisibleRowCursor {
    pos: usize,
    n_obs: usize,
}

impl VisibleRowCursor {
    pub fn new(n_obs: usize) -> Self {
        Self { pos: 0, n_obs }
    }

    /// Claim `n_rows` rows for a shard, returning the global visible row its
    /// first row occupies. Errors rather than overflowing the output.
    pub fn advance(
        &mut self,
        n_rows: usize,
        shard_idx: usize,
        context: &str,
    ) -> Result<usize, String> {
        // `n_obs - pos`, not `pos + n_rows`: the cursor invariant keeps
        // `pos <= n_obs`, so the subtraction is safe, while the addition could
        // itself overflow on a large `n_rows` — panicking in debug and wrapping
        // in release, inside the check whose job is to prevent exactly that.
        if n_rows > self.n_obs - self.pos {
            return Err(format!(
                "{context}: shard {shard_idx} has {n_rows} rows, exceeding n_obs = {} at row {}",
                self.n_obs, self.pos
            ));
        }
        let start = self.pos;
        self.pos += n_rows;
        Ok(start)
    }

    /// Errors unless the shards visited covered exactly `n_obs` rows.
    pub fn finish(&self, context: &str) -> Result<(), String> {
        if self.pos != self.n_obs {
            return Err(format!(
                "{context}: the shards visited cover {} rows but the source reports n_obs = {}",
                self.pos, self.n_obs
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::VisibleRowCursor;

    #[test]
    fn a_cursor_hands_out_ascending_offsets_and_accepts_exact_coverage() {
        let mut c = VisibleRowCursor::new(24);
        assert_eq!(c.advance(8, 0, "op").unwrap(), 0);
        assert_eq!(c.advance(8, 1, "op").unwrap(), 8);
        assert_eq!(c.advance(8, 2, "op").unwrap(), 16);
        c.finish("op").unwrap();
    }

    /// A skipped shard advances the cursor by nothing, which is why a running
    /// count stays correct under the row-projection skip.
    #[test]
    fn an_empty_shard_does_not_move_the_cursor() {
        let mut c = VisibleRowCursor::new(8);
        assert_eq!(c.advance(0, 0, "op").unwrap(), 0);
        assert_eq!(c.advance(8, 1, "op").unwrap(), 0);
        c.finish("op").unwrap();
    }

    #[test]
    fn over_coverage_errors_before_the_write_rather_than_indexing_past_the_end() {
        let mut c = VisibleRowCursor::new(8);
        assert_eq!(c.advance(8, 0, "op").unwrap(), 0);
        let err = c.advance(1, 1, "op").unwrap_err();
        assert!(
            err.contains("shard 1") && err.contains("exceeding n_obs = 8"),
            "{err}"
        );
    }

    /// The check must not be the thing that overflows. `pos + n_rows` panics in
    /// debug and wraps in release at this input; `n_obs - pos` cannot, because
    /// the cursor never lets `pos` pass `n_obs`.
    #[test]
    fn an_absurd_row_count_errors_rather_than_overflowing_the_check() {
        let mut c = VisibleRowCursor::new(8);
        assert_eq!(c.advance(4, 0, "op").unwrap(), 0);
        let err = c.advance(usize::MAX, 1, "op").unwrap_err();
        assert!(err.contains("exceeding n_obs = 8"), "{err}");
        // And the cursor is unchanged by the rejected claim.
        assert_eq!(c.advance(4, 2, "op").unwrap(), 4);
        c.finish("op").unwrap();
    }

    #[test]
    fn under_coverage_errors_rather_than_returning_a_shifted_result() {
        let mut c = VisibleRowCursor::new(40);
        c.advance(16, 0, "op").unwrap();
        let err = c.finish("op").unwrap_err();
        assert!(
            err.contains("cover 16 rows") && err.contains("n_obs = 40"),
            "{err}"
        );
    }
}
