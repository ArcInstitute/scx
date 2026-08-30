//! Loess-failure diagnostics for the seurat_v3 flavor — per-batch
//! singularity detection, the user-facing warning, and the uns record.

use pyo3::types::PyDict;

use super::*;

/// One batch whose `skmisc.loess.fit()` raised a singularity `ValueError`.
pub(crate) struct LoessFailure {
    /// Index into the non-empty batch list — what
    /// `uns["hvg"]["loess_failed_batches"]` is keyed by. Not an index into
    /// `batch_key`'s categories.
    pub batch_idx: usize,
    /// Cells in the batch.
    pub n_cells: usize,
    /// Genes with non-zero variance in this batch, i.e. the number of points the
    /// regression actually got. The gap to `n_vars` is what makes the fit
    /// singular, so it is the number the user needs.
    pub n_fit_points: usize,
    /// The upstream `skmisc` error string.
    pub error: String,
}

/// Build the summary warning text for all batches whose loess fit was singular.
///
/// Split out from the emitter so the two branches — batched and unbatched — can
/// be asserted directly. `failed` is in iteration order; `n_total` is the total
/// batch count; `n_vars` is the gene count the fit ran over.
///
/// A high-cardinality `batch_key` (e.g. a per-dataset id) can produce dozens of
/// tiny, singular batches, and one verbatim warning each buries the signal
/// (user-report F10) — so this coalesces into a single message naming the
/// failed/total count and a representative first failure.
///
/// **The remedies must branch on whether `batch_key` was passed (dogfood E3).**
/// The batched singularity is driven by small / near-collinear batches, so
/// dropping or coarsening `batch_key` is the fix and `filter_genes` cannot help.
/// With no `batch_key` there is exactly one global fit: the batch_key advice is
/// *inapplicable*, "1 of 1 batches" is noise, and the remedy that actually works
/// — `filter_genes(min_cells=10)`, which on the reported case cut 61,497 genes to
/// 9,998 and made the fit succeed — was buried at the end and phrased as a
/// limitation. Leading with inapplicable advice costs the user the one thing
/// that would have worked.
pub(crate) fn hvg_loess_singularity_message(
    first: &LoessFailure,
    n_failed: usize,
    n_total: usize,
    n_vars: usize,
    batch_key: Option<&str>,
    batch_labels: Option<&[String]>,
) -> String {
    let n_zero_var = n_vars.saturating_sub(first.n_fit_points);

    // No batch_key → one global fit. Say so in those terms and lead with the
    // remedy that applies to it.
    //
    // Which remedy that *is* depends on whether there is anything to filter. On
    // the reported case 42,267 of 61,497 genes were expressed in no cell, so
    // `filter_genes(min_cells=10)` cut the gene axis to 9,998 and the fit
    // succeeded. But when nothing is constant, filtering removes nothing and the
    // fit is simply short of points — recommending it there would repeat E3's
    // own mistake of leading with advice that cannot apply.
    if batch_key.is_none() {
        let remedy = if n_zero_var > 0 {
            format!(
                "the other {n_zero_var} are constant (typically expressed in no cell) \
                 and cannot contribute. Try pyscx.accel.filter_genes(min_cells=10) \
                 first — dropping never-expressed genes is usually what makes this fit \
                 well-conditioned. Failing that, switch to flavor=\"seurat\" \
                 post-normalize"
            )
        } else {
            "every gene is non-constant, so filtering cannot add points — this fit is \
             simply too small to regress (a degree-2 loess needs appreciably more). \
             Switch to flavor=\"seurat\" post-normalize, or run seurat_v3 on a gene axis \
             that has not already been narrowed"
                .to_string()
        };
        return format!(
            "highly_variable_genes(flavor=\"seurat_v3\"): the skmisc.loess fit failed \
             (singular / under-determined) — {error}. The fit regresses log-variance on \
             log-mean over the {n_fit_points} of {n_vars} genes that have non-zero \
             variance here; {remedy}. \
             (Recorded in adata.uns[\"hvg\"][\"loess_failed_batches\"].)",
            error = first.error,
            n_fit_points = first.n_fit_points,
        );
    }

    // Batched: name the batch, not just its position. `batches` drops empty
    // groups, so the index cannot be looked up in `batch_key`'s categories
    // (user-report F4). The index is still printed — `uns` is keyed by it.
    let bk = batch_key.expect("checked above");
    let first_detail = match batch_labels {
        Some(labels) => labels
            .get(first.batch_idx)
            .map(|l| format!("({bk}={l:?}, n={} cells)", first.n_cells))
            .unwrap_or_else(|| format!("(n={} cells)", first.n_cells)),
        None => format!("(n={} cells)", first.n_cells),
    };
    // "the remaining 0 proceed normally" is a false statement, and it is exactly
    // what the all-failed case used to print immediately before raising.
    let n_valid = n_total.saturating_sub(n_failed);
    let survivors = if n_valid == 0 {
        "no batch survived, so there is no variance trend to rank against".to_string()
    } else {
        format!("the remaining {n_valid} proceed normally")
    };
    format!(
        "highly_variable_genes(flavor=\"seurat_v3\"): skmisc.loess fit failed on \
         {n_failed} of {n_total} batches — these batches are excluded from the \
         per-batch HVG ranking; {survivors}. First \
         failure: batch index {first_idx} {first_detail}, {n_fit_points} of {n_vars} \
         genes non-constant within the batch — {error}. \
         Common causes: very small batches, near-collinear log-mean / log-variance, \
         or many zero-variance genes within a batch. To avoid this, prefer \
         dropping or coarsening batch_key (a high-cardinality key such as a \
         per-dataset id produces many tiny, singular batches); or switch to \
         flavor=\"seurat\" post-normalize. Note pyscx.accel.filter_genes(min_cells=10) \
         only helps the no-batch_key global fit, not the per-batch singularity. \
         (Full per-batch detail is recorded in adata.uns[\"hvg\"][\"loess_failed_batches\"].)",
        first_idx = first.batch_idx,
        n_fit_points = first.n_fit_points,
        error = first.error,
    )
}

/// Emit a single summary UserWarning for all batches whose `skmisc.loess.fit()`
/// raised a singularity `ValueError`. Text built by
/// [`hvg_loess_singularity_message`]; each failing batch is excluded from the
/// per-batch normalised-variance ranking — semantics identical to a batch with
/// too few non-constant genes.
pub(crate) fn emit_hvg_loess_singularity_warning(
    py: Python<'_>,
    failed: &[LoessFailure],
    n_total: usize,
    n_vars: usize,
    batch_key: Option<&str>,
    batch_labels: Option<&[String]>,
) -> PyResult<()> {
    // `first()` rather than `failed[0]`: the emptiness check and the indexing
    // are the same expression, so `hvg_loess_singularity_message` can take a
    // `&LoessFailure` and the empty case becomes unrepresentable there rather
    // than something a future caller could trip over.
    let Some(first) = failed.first() else {
        return Ok(());
    };
    let msg = hvg_loess_singularity_message(
        first,
        failed.len(),
        n_total,
        n_vars,
        batch_key,
        batch_labels,
    );
    crate::pyimport::import_module(py, "warnings")?.call_method1(
        "warn",
        (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
    )?;
    Ok(())
}

/// Record the per-batch loess-failure detail on
/// `adata.uns["hvg"]["loess_failed_batches"]` as a list of
/// `[batch_idx, n_cells, label]` triples — `[batch_idx, n_cells]` pairs when
/// there is no `batch_key` to label with. (The summary UserWarning names only
/// the first failure.) Mirrors scanpy
/// storing HVG metadata under `uns["hvg"]`.
///
/// Called **once per batched seurat_v3 run**, with `failed` possibly empty.
/// Two invariants this guarantees:
/// - **Merge, don't clobber:** reuse any pre-existing `uns["hvg"]` dict (e.g.
///   metadata a prior `scanpy.pp.highly_variable_genes` wrote) and only set the
///   `loess_failed_batches` key, so sibling metadata survives.
/// - **No stale failures:** the key is rewritten every run to the *current*
///   list, so a clean rerun of the same `AnnData` overwrites a stale list from
///   an earlier failed run (writing `[]` when no batch failed) rather than
///   leaving downstream diagnostics reporting phantom failures.
pub(crate) fn record_hvg_loess_failed_batches(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    failed: &[LoessFailure],
    batch_labels: Option<&[String]>,
) -> PyResult<()> {
    let uns = adata.getattr("uns")?;
    let hvg_dict = match uns.get_item("hvg") {
        // Present and a dict → merge into it; present but not a dict (unexpected)
        // → replace with a fresh dict; absent → fresh dict.
        Ok(existing) => existing
            .cast_into::<PyDict>()
            .unwrap_or_else(|_| PyDict::new(py)),
        Err(_) => PyDict::new(py),
    };
    let failed_list = pyo3::types::PyList::empty(py);
    for f in failed {
        // `[batch_idx, n_cells, label]` — the label is what the user can
        // actually act on (`batch_idx` indexes the non-empty batch list, not
        // `batch_key`'s categories). Omitted when there is no `batch_key`,
        // leaving the historical `[batch_idx, n_cells]` pair.
        let entry = pyo3::types::PyList::empty(py);
        entry.append(f.batch_idx)?;
        entry.append(f.n_cells)?;
        if let Some(label) = batch_labels.and_then(|l| l.get(f.batch_idx)) {
            entry.append(label)?;
        }
        failed_list.append(entry)?;
    }
    hvg_dict.set_item("loess_failed_batches", failed_list)?;
    uns.set_item("hvg", hvg_dict)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{hvg_loess_singularity_message, LoessFailure};

    fn one_failure(n_fit_points: usize) -> Vec<LoessFailure> {
        vec![LoessFailure {
            batch_idx: 0,
            n_cells: 1104,
            n_fit_points,
            error: "b'There are other near singularities as well. 0.22764'".to_string(),
        }]
    }

    /// E3, the reported case: `highly_variable_genes(flavor="seurat_v3")` with
    /// **no** `batch_key` on 1,104 cells where 42,267 of 61,497 genes are
    /// expressed in no cell.
    ///
    /// The old message led with "prefer dropping or coarsening `batch_key`" —
    /// inapplicable, since none was passed — said "1 of 1 batches", and buried
    /// the remedy that actually worked at the end, phrased as a limitation.
    #[test]
    fn unbatched_leads_with_filter_genes_and_never_mentions_batch_key() {
        let msg = hvg_loess_singularity_message(&one_failure(19_230)[0], 1, 1, 61_497, None, None);

        assert!(
            !msg.contains("batch_key"),
            "no batch_key was passed, so advice about it is inapplicable: {msg}"
        );
        // No batch *count* reporting. (The trailing
        // `uns["hvg"]["loess_failed_batches"]` key name is an existing public
        // key and legitimately contains the word, so match the noise phrase
        // rather than the substring.)
        for noise in ["1 of 1", "of 1 batches", "per-batch"] {
            assert!(
                !msg.contains(noise),
                "{noise:?} is noise for a single global fit: {msg}"
            );
        }
        // The remedy that works must lead, not trail.
        let filter_at = msg.find("filter_genes").expect("must name filter_genes");
        let seurat_at = msg.find("flavor=\\\"seurat\\\"").unwrap_or(usize::MAX);
        assert!(
            filter_at < seurat_at,
            "filter_genes is the effective remedy here and must come first: {msg}"
        );
        assert!(
            !msg.contains("only helps"),
            "filter_genes must not be phrased as a limitation when it is THE fix: {msg}"
        );
        // Concrete numbers so the user can see why the fit was under-determined.
        for needle in ["19230", "61497", "42267", "constant"] {
            assert!(msg.contains(needle), "must mention {needle:?}: {msg}");
        }
        // The upstream error string still gets through — scanpy surfaces only
        // this and nothing else, and it is what a search engine matches.
        assert!(msg.contains("near singularities"), "{msg}");
    }

    /// The other half of E3's own lesson, applied to the fix: when *nothing* is
    /// constant there is nothing for `filter_genes` to drop, so recommending it
    /// would be the same "advice that cannot apply" the item is about. Found by
    /// running the real op on a fixture with no never-expressed genes, where the
    /// first version of this message still said "try filter_genes".
    #[test]
    fn unbatched_with_no_constant_genes_does_not_recommend_filter_genes() {
        let msg = hvg_loess_singularity_message(&one_failure(8)[0], 1, 1, 8, None, None);
        assert!(
            !msg.contains("filter_genes"),
            "with 0 constant genes filtering removes nothing and cannot help: {msg}"
        );
        assert!(
            msg.contains("filtering cannot add points"),
            "must say why the obvious remedy is not the remedy: {msg}"
        );
        assert!(
            msg.contains("flavor=\"seurat\""),
            "must still offer the remedy that does apply: {msg}"
        );
        assert!(msg.contains("8 of 8"), "{msg}");
    }

    /// "the remaining 0 proceed normally" is a false statement, and it is exactly
    /// what the all-batches-failed case printed immediately before raising.
    #[test]
    fn all_batches_failed_does_not_claim_survivors() {
        let failed: Vec<LoessFailure> = (0..6)
            .map(|i| LoessFailure {
                batch_idx: i,
                n_cells: 10,
                n_fit_points: 8,
                error: "ValueError: b'svddc failed in l2fit.'".to_string(),
            })
            .collect();
        let msg = hvg_loess_singularity_message(
            &failed[0],
            failed.len(),
            6,
            5000,
            Some("dataset_id"),
            None,
        );
        assert!(
            !msg.contains("remaining 0 proceed normally"),
            "claiming 0 batches proceed normally is false: {msg}"
        );
        assert!(
            msg.contains("no batch survived"),
            "must state the actual outcome: {msg}"
        );
        // And the partial case still reports survivors.
        let partial = hvg_loess_singularity_message(&failed[0], 1, 6, 5000, Some("d"), None);
        assert!(
            partial.contains("the remaining 5 proceed normally"),
            "{partial}"
        );
    }

    /// The batched branch keeps every diagnostic it had: the failed/total count,
    /// the named batch, the surviving-batch count, and the note that
    /// `filter_genes` does *not* help a per-batch singularity.
    #[test]
    fn batched_keeps_the_batch_key_remedy_and_the_filter_genes_caveat() {
        let labels = vec!["dataset_7".to_string()];
        let msg = hvg_loess_singularity_message(
            &one_failure(1_500)[0],
            1,
            92,
            61_497,
            Some("dataset_id"),
            Some(&labels),
        );

        assert!(msg.contains("1 of 92 batches"), "{msg}");
        assert!(
            msg.contains("dataset_id=\"dataset_7\""),
            "the batch must be named, not just indexed: {msg}"
        );
        assert!(msg.contains("the remaining 91 proceed normally"), "{msg}");
        assert!(
            msg.contains("dropping or coarsening batch_key"),
            "this IS the effective remedy for a per-batch singularity: {msg}"
        );
        assert!(
            msg.contains("only helps the no-batch_key global fit"),
            "the caveat is correct here and must survive: {msg}"
        );
        assert!(
            msg.contains("1500 of 61497"),
            "the per-batch non-constant count is new and useful: {msg}"
        );
    }

    /// With a `batch_key` but no resolved labels the index still has to be
    /// printed — it is what `uns["hvg"]["loess_failed_batches"]` is keyed by.
    #[test]
    fn batched_without_labels_still_reports_the_index_and_cell_count() {
        let msg = hvg_loess_singularity_message(
            &one_failure(1_500)[0],
            1,
            4,
            61_497,
            Some("batch"),
            None,
        );
        assert!(msg.contains("batch index 0"), "{msg}");
        assert!(msg.contains("n=1104 cells"), "{msg}");
    }

    /// A fit that got zero usable points must not underflow the
    /// `n_vars - n_fit_points` subtraction.
    #[test]
    fn zero_fit_points_does_not_underflow() {
        let msg = hvg_loess_singularity_message(&one_failure(0)[0], 1, 1, 10, None, None);
        assert!(msg.contains("0 of 10"), "{msg}");
        assert!(msg.contains("other 10 are constant"), "{msg}");
        assert!(
            msg.contains("filter_genes"),
            "10 constant genes CAN be filtered: {msg}"
        );
    }
}
