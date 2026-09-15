//! Top-K crop — the fixed-length encoder selection.
//!
//! Consumers: STATE3 and any fixed-length encoder. This kernel is the one that
//! existed before W6, inlined in `collate_cell`; it is extracted here unchanged
//! and `collate_cell` now calls it. Its semantics are pinned byte-exact by
//! `scx-loader/tests/data/encoder_crop_golden.json`, which is committed
//! identically in this repo and in state3 and whose blake3 prefix is asserted by
//! the kernel tests — so "unchanged" is a checked claim, not an intention.
//!
//! # Tie rule
//!
//! Selection value **descending**, gene id **ascending** on ties — equivalent to
//! `np.lexsort((gene_id, -selection))`. Gene ids are unique within a row, so the
//! order is total and sort stability is irrelevant.
//!
//! The rule is not a parameter. There is exactly one tie rule, one consumer
//! contract that pins it, and no caller that wants another; a `TieRule` enum with
//! a single variant would be a configuration surface for a value nobody changes.
//! If a second rule is ever needed, add it then — and bump the contract version,
//! because a tie rule is part of output identity.
//!
//! # Masking is drop-to-PAD, with no backfill
//!
//! A withheld query gene is **removed** from the crop — its slot becomes PAD —
//! and survivors compact to the left. It is never replaced in place by a
//! GENE_MASK token, and nothing is pulled in from beyond `take` to refill the
//! gap. A single GENE_MASK token appears only in the whole-cell hidden-readout
//! branch and in the all-masked fallback, and those two differ in one bit:
//! the fallback sets `mask[0] = 1`, the degenerate empty row leaves it `0`.

use crate::tokenize::CsrRow;

/// `GENE_MASK` sentinel = `n_genes_total` (state3 `task.py:_special_gene_mask_id`).
#[inline]
pub fn gene_mask_id(n_genes_total: i64) -> i64 {
    n_genes_total
}

/// `PAD` sentinel = `n_genes_total + 1` (state3 `task.py:_special_pad_id`).
#[inline]
pub fn pad_id(n_genes_total: i64) -> i64 {
    n_genes_total + 1
}

/// A set's decoder query panel, sorted for membership tests.
///
/// Built **once per set** (the panel is shared across the set's rows) and
/// consulted per row against that row's own `enc_mask_positions`. This is the
/// only sound way to hoist any part of the withheld-gene test out of the row
/// loop: `query` is sliced per set but the mask bits are sliced per row, so
/// hoisting the *withheld id set* itself would apply row 0's bits to every row
/// of the set.
///
/// Replaces a per-row `HashSet<i64>` that was rebuilt from `k_dec` ids and then
/// SipHash-probed once per surviving top-K gene — measured at ~34 % of collate
/// wall (78.7 vs 120.1 us/cell at 25 % withheld, pbmc3k, k_enc=2048/k_dec=1024).
pub struct SetQueryIndex {
    /// `(gene_id, position_in_query)`, sorted by id then position.
    ///
    /// `position` is the id's offset in the ORIGINAL panel, not its rank here:
    /// `enc_mask_positions` is parallel to `query`, so the bit must be read at
    /// the offset the caller wrote it at.
    sorted: Vec<(i32, u32)>,
}

impl SetQueryIndex {
    /// Sort a copy of one set's query panel. `query` carries no ordering
    /// contract (only `CsrRow::gene_ids` does) and is not deduplicated.
    pub fn new(query: &[i32]) -> Self {
        let mut sorted: Vec<(i32, u32)> = query
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, i as u32))
            .collect();
        // Unstable is fine and faster: the `(id, position)` pairs are distinct,
        // so the order is total.
        sorted.sort_unstable();
        Self { sorted }
    }

    /// Is `gid` withheld from this row's encoder crop?
    ///
    /// True iff **any** position carrying `gid` is flagged — the semantics of
    /// the `HashSet` this replaces, which collected every flagged position's id
    /// (mirroring Python's `query_gene_ids[role_target_mask]`). A repeated id is
    /// reachable on real input, so the fold over the equal-id run is required,
    /// not defensive: stopping at the first match would silently keep a gene the
    /// caller withheld at a later position.
    #[inline]
    fn withholds(&self, gid: i32, maskpos: &[u8]) -> bool {
        let lo = self.sorted.partition_point(|&(g, _)| g < gid);
        self.sorted[lo..]
            .iter()
            .take_while(|&&(g, _)| g == gid)
            // `get`, not an index: the `zip` this replaced stopped at the
            // shorter of `query` / `enc_mask_positions`, so a short mask left
            // the excess positions unflagged. Indexing would panic there
            // instead — a tolerated malformed input turned into a crash on a
            // `pub` kernel. `collate_gathered` validates the length, so this
            // only guards direct callers.
            .any(|&(_, p)| maskpos.get(p as usize).is_some_and(|&m| m != 0))
    }
}

/// One row's encoder mask, inseparable from the panel it indexes.
///
/// A struct rather than two `Option` fields on [`CropIn`] so the pair cannot be
/// supplied half-set: the bits are meaningless without the panel that says
/// which gene each offset refers to, and a mask-without-index would silently
/// withhold nothing — a wrong answer on a cross-repo numeric contract, where a
/// compile error is what is wanted.
#[derive(Clone, Copy)]
pub struct RowMask<'a> {
    /// `[k_dec]` flags over the set's `query`, for THIS row.
    pub positions: &'a [u8],
    /// The set's sorted panel — built once per set, shared by its rows.
    pub index: &'a SetQueryIndex,
}

/// Inputs to one row's crop.
pub struct CropIn<'a> {
    /// The row, whose `values` are the **selection** counts: the `> 0` filter and
    /// the descending order are taken over these, clipped at zero.
    ///
    /// state3 calls them `selection_counts` and passes raw counts even when the
    /// emitted values are transformed, which is why they are separate from
    /// [`emit`](Self::emit) rather than one array used for both.
    pub row: CsrRow<'a>,
    /// Values written into `CropOut::values`, parallel to `row.gene_ids`.
    pub emit: &'a [f32],
    /// This row's withheld-gene mask over its set's query panel. `None` ⇒ no
    /// query-based masking (state3's perturbation path), which also means the
    /// panel is never sorted on that path.
    pub withheld: Option<RowMask<'a>>,
    /// Hide gene identity (readout mask) ⇒ a single `GENE_MASK` token.
    pub hide_readout: bool,
}

/// Fixed parameters for the crop.
#[derive(Clone, Copy, Debug)]
pub struct CropConfig {
    /// Output width. Slots beyond the surviving genes stay PAD.
    pub k: usize,
    /// Vocabulary size, which fixes the two sentinels.
    pub n_genes_total: i64,
}

/// Caller-owned output slices, each length `CropConfig::k`.
pub struct CropOut<'a> {
    pub ids: &'a mut [i64],
    pub values: &'a mut [f32],
    pub mask: &'a mut [u8],
    pub pad: &'a mut [u8],
}

/// Crop one row to its top `k` genes by selection value.
///
/// Every output slice must be length `cfg.k`; `cin.emit` must be parallel to
/// `cin.row.gene_ids`; `cin.withheld`, when present, must carry
/// [`SetQueryIndex::new`] over this row's set's query panel.
///
/// `order` is caller-owned scratch, reused across rows. It is taken as the bare
/// `Vec` rather than as a whole [`Scratch`](crate::tokenize::Scratch) so that a
/// caller can hand this kernel one field while still holding a shared borrow of
/// another — which is exactly what `collate_cell` does, passing `enc_vals` as
/// `cin.emit` out of the same bundle.
pub fn top_k(cin: &CropIn, cfg: &CropConfig, order: &mut Vec<usize>, out: &mut CropOut) {
    let n = cin.row.len();
    let mask_id = gene_mask_id(cfg.n_genes_total);
    let pad = pad_id(cfg.n_genes_total);

    for slot in 0..cfg.k {
        out.ids[slot] = pad;
        out.values[slot] = 0.0;
        out.mask[slot] = 0;
        out.pad[slot] = 1;
    }
    if cin.hide_readout {
        out.ids[0] = mask_id;
        out.mask[0] = 1;
        out.pad[0] = 0;
        return;
    }

    // positive selection uses the selection counts (state3 selection_counts).
    order.clear();
    order.extend((0..n).filter(|&i| cin.row.values[i].max(0.0) > 0.0));
    if order.is_empty() {
        // Degenerate row: one GENE_MASK token, no expr mask (task.py:178).
        out.ids[0] = mask_id;
        out.pad[0] = 0;
        return;
    }

    // lexsort((gene_id, -selection)): selection DESC, gene_id ASC tiebreak.
    // gene_ids are unique within a row, so this is a total order (stability
    // irrelevant); matches np.lexsort exactly for the selected set.
    let (ids, vals) = (cin.row.gene_ids, cin.row.values);
    order.sort_by(|&a, &b| {
        let ra = vals[a].max(0.0);
        let rb = vals[b].max(0.0);
        // unwrap is safe: `max(0.0)` yields a finite, non-NaN f32, so
        // partial_cmp is always `Some`.
        rb.partial_cmp(&ra).unwrap().then(ids[a].cmp(&ids[b]))
    });
    let take = cfg.k.min(order.len());

    // Walk the top-K order, DROPPING withheld genes (their slot is left PAD;
    // no backfill from beyond `take`) and compacting survivors to the left.
    // Mirrors `_sparse_encoder_inputs`' `selected[keep]` (task.py:231-252):
    // withheld genes are absent (PAD), never a GENE_MASK token.
    let withheld = cin.withheld;
    let mut slot = 0usize;
    for &i in order.iter().take(take) {
        let g = ids[i];
        if withheld.is_some_and(|m| m.index.withholds(g, m.positions)) {
            continue;
        }
        out.ids[slot] = g as i64;
        out.values[slot] = cin.emit[i];
        out.pad[slot] = 0;
        slot += 1;
    }

    // All-masked fallback: every selected top-K gene was withheld ⇒ a single
    // active GENE_MASK token at slot 0 (task.py:235-247). Distinct from the
    // degenerate-row branch above, which leaves `mask[0]=0`; here `mask[0]=1`.
    // `values[0]` stays 0 (matches Python `counts[0]`).
    if slot == 0 {
        out.ids[0] = mask_id;
        out.mask[0] = 1;
        out.pad[0] = 0;
    }
    // Slots [slot..k] keep their PAD-init values (id=pad, pad=1, values=0, mask=0).
}

#[cfg(test)]
#[path = "crop_tests.rs"]
mod tests;
