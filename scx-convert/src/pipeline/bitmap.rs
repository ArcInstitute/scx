//! Detection-bitmap generation: the `auto` eligibility policy and the
//! per-shard writer.
//!
//! `BitmapPolicy::Auto` is three independent thresholds -- density, `n_vars`
//! cap, and a size budget as a percentage of the shard's payload -- and the
//! constants are here beside the only function that reads them.

use scx_format_io::modality::ModalityType;
use scx_format_io::writer::ScxWriter;

use super::error::ConvertError;
use crate::warnings::{ConvertWarning, WarningSink};
use scx_format_io::bitmap::BitmapShard;
use scx_format_io::BitmapPolicy;

/// Phase 5b: density threshold below which `--bitmap=auto` considers a
/// shard "sparse enough" for bitmaps. Above this, the CSR storage is
/// already dense-ish (>30% nonzero) and bitmaps offer little win.
const BITMAP_AUTO_DENSITY_THRESHOLD: f32 = 0.30;
/// Phase 5b: `n_vars` cap for `--bitmap=auto`. Tied to the per-row
/// allocator cost on extremely wide matrices.
const BITMAP_AUTO_N_VARS_CAP: u32 = 1_000_000;
/// Phase 5b: bitmap size budget under `--bitmap=auto`, expressed as a
/// percentage of the encoded CSR shard size. Roaring sizes vary enough
/// that this is checked *after* the build, not before.
const BITMAP_AUTO_SIZE_PERCENT: usize = 15;

/// Outcome of [`maybe_build_bitmap_shard`]. Either a built shard
/// (ready to write) or a structured reason for skipping that the
/// caller forwards to the warning sink.
pub(crate) enum BitmapBuildOutcome {
    Skip { reason: String },
    Built(BitmapShard),
}

/// Pure (no I/O, no sink) bitmap-build helper. Applies
/// the same `--bitmap=auto` density / size gates as the sequential
/// path but returns an outcome instead of writing. The parallel
/// coordinator runs this in a worker thread and the sequential
/// wrapper (`build_and_write_bitmap_for_shard`) routes the outcome
/// through the writer + sink.
#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_build_bitmap_shard(
    indptr: &[u64],
    indices: &[u32],
    row_start: u64,
    n_rows: u32,
    n_vars: u32,
    encoded_csr_size: usize,
    policy: BitmapPolicy,
    modality_type: ModalityType,
) -> Option<BitmapBuildOutcome> {
    if matches!(policy, BitmapPolicy::Off) {
        return None;
    }
    if n_vars > BITMAP_AUTO_N_VARS_CAP && !matches!(policy, BitmapPolicy::Always) {
        return Some(BitmapBuildOutcome::Skip {
            reason: format!("n_vars {n_vars} exceeds auto cap {BITMAP_AUTO_N_VARS_CAP}"),
        });
    }

    let nnz = *indptr.last().unwrap_or(&0);
    let cells = n_rows as u64;
    let density = if cells == 0 || n_vars == 0 {
        0.0_f32
    } else {
        nnz as f32 / (cells as f32 * n_vars as f32)
    };
    if matches!(policy, BitmapPolicy::Auto)
        && density > BITMAP_AUTO_DENSITY_THRESHOLD
        && !matches!(modality_type, ModalityType::Atac)
    {
        return Some(BitmapBuildOutcome::Skip {
            reason: format!(
                "density {density:.3} above auto threshold {BITMAP_AUTO_DENSITY_THRESHOLD}"
            ),
        });
    }

    let shard = BitmapShard::build_from_csr(row_start, n_rows, n_vars, indptr, indices);

    if matches!(policy, BitmapPolicy::Auto) && !matches!(modality_type, ModalityType::Atac) {
        let est = shard.estimated_encoded_size();
        if encoded_csr_size > 0
            && est.saturating_mul(100) > encoded_csr_size.saturating_mul(BITMAP_AUTO_SIZE_PERCENT)
        {
            return Some(BitmapBuildOutcome::Skip {
                reason: format!(
                    "estimated {est} bytes > {BITMAP_AUTO_SIZE_PERCENT}% of CSR shard ({encoded_csr_size})"
                ),
            });
        }
    }

    Some(BitmapBuildOutcome::Built(shard))
}

/// Build (and conditionally write) a detection bitmap for
/// one CSR shard.
///
/// `modality_type` and `modality_name` drive the auto policy
/// (ATAC modalities are eager; everything else compares estimated
/// bitmap size against `encoded_csr_size`).
///
/// Returns whether a bitmap section was actually written so callers
/// can stamp provenance.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_and_write_bitmap_for_shard(
    writer: &mut ScxWriter,
    indptr: &[u64],
    indices: &[u32],
    row_start: u64,
    n_rows: u32,
    n_vars: u32,
    encoded_csr_size: usize,
    policy: BitmapPolicy,
    modality_type: ModalityType,
    modality_name: Option<&str>,
    sink: &mut WarningSink,
) -> Result<bool, ConvertError> {
    let outcome = maybe_build_bitmap_shard(
        indptr,
        indices,
        row_start,
        n_rows,
        n_vars,
        encoded_csr_size,
        policy,
        modality_type,
    );
    match outcome {
        None => Ok(false),
        Some(BitmapBuildOutcome::Skip { reason }) => {
            sink.emit(ConvertWarning::BitmapSkipped {
                modality: modality_name.map(String::from),
                reason,
            });
            Ok(false)
        }
        Some(BitmapBuildOutcome::Built(shard)) => {
            writer
                .write_bitmap_shard(&shard)
                .map_err(ConvertError::from)?;
            Ok(true)
        }
    }
}
