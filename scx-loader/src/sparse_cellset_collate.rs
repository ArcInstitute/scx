//! Per-cell collation kernel for the sparse cell-set loader (state3 "3A hybrid").
//!
//! Ports the RNG-free numerics of state3's `CellSetCollator.task.specify`
//! (`src/state3/data/task.py`) into Rust: count preprocessing, top-K encoder
//! selection, query-position target gather, encoder masking, and library size.
//! The RNG-dependent inputs (the decoder `query_gene_ids` and the per-cell
//! encoder-mask / loss masks) are produced in Python and handed in, so no RNG
//! crosses the boundary — that is what keeps parity tractable.
//!
//! Exactness contract (matches the state3 parity gate): gene ids, masks, and
//! `library_size` (a sum of integer counts < 2^24, order-independent in f32) are
//! reproduced exactly; transcendental-derived encoder values (`log1p`) match the
//! NumPy reference only to ~1 ULP, since NumPy may vectorise `log1p` while Rust
//! calls scalar `libm` — so the harness gates those with a tolerance.
//!
//! Inputs assume a single cell's CSR already in **global vocab, sorted ascending
//! by gene id, with duplicates coalesced** — exactly what `remap_row`
//! (`sparse_cellset.rs`) produces, which mirrors state3's `finalize_csr_row`.
//!
//! # Cross-repo contract (DO NOT drift)
//!
//! The encoder-crop / mask / target semantics here mirror, byte-exact, state3's
//! `_sparse_encoder_inputs` (`state3/src/state3/data/task.py`). In particular the
//! encoder masking policy is **drop-to-PAD**: a withheld query gene is removed from
//! the top-K crop (its slot becomes PAD), NOT replaced in place with a GENE_MASK
//! token. A single GENE_MASK token appears only in the whole-cell hidden-readout
//! and all-masked-fallback branches.
//!
//! The executable contract is the golden-vector fixture committed identically in
//! both repos (`scx-loader/tests/data/encoder_crop_golden.json` ==
//! `state3/tests/data/encoder_crop_golden.json`), asserted by the kernel test here
//! and by state3's `test_encoder_crop_golden`. **Any change to either side must:
//! update both implementations, regenerate the golden fixture, update the kernel
//! tests, and bump `pyscx::COLLATE_CELLSET_CONTRACT_VERSION`** (which state3 asserts
//! at `rust_collate` setup to fail loudly on version skew).
//!
//! ## What the version actually covers — and what it does not
//!
//! This comment used to say the version is bumped "on any contract change", which
//! was not true: #356 changed the accepted preprocess-mode strings — PFlog v2
//! (`pflog1ppf_raw`) to v4 (`pflog_raw` plus a required `pflog_alpha`), a
//! genuinely different transform — and left the version at `1`, so the assertion
//! reported agreement between implementations that disagreed. The claim is
//! narrowed here to something enforceable. `COLLATE_CELLSET_CONTRACT_VERSION`
//! covers, exhaustively:
//!
//! 1. the encoder-crop, masking and target semantics in [`collate_cell`];
//! 2. the accepted [`PreprocessMode`] strings, the parameters each mode
//!    *requires* (e.g. `pflog_raw` requires `pflog_alpha`), and the meaning of
//!    each mode;
//! 3. the §4.4 gather-stage **value** contract — the non-negativity clip and the
//!    optional seeded downsample in `sparse_cellset::SparseCellSetLoader`.
//!
//! It does **not** cover the batch's array *shapes* or key names (a
//! length-validation error surfaces those loudly at the first batch), nor the
//! `pe_mask` omission noted on [`CellOut`], which is an agreed non-emission rather
//! than a version-dependent behaviour.
//!
//! Version history: **v1** initial; **v2** = #356's mode strings (retroactively)
//! + Phase 1B's clip and downsample.

use crate::error::{LoaderError, Result};

/// Per-cell preprocessing mode — mirrors `preprocessing.py:preprocess_cell_counts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreprocessMode {
    PassThrough,
    Log1pRaw,
    PflogRaw,
    NormalizeLog1p,
}

impl PreprocessMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "pass_through" => Ok(Self::PassThrough),
            "log1p_raw" => Ok(Self::Log1pRaw),
            "pflog_raw" => Ok(Self::PflogRaw),
            "normalize_log1p" => Ok(Self::NormalizeLog1p),
            other => Err(LoaderError::ConfigError {
                reason: format!("unknown preprocessing mode {other:?}"),
            }),
        }
    }
}

/// `GENE_MASK` sentinel = `n_genes_total` (`task.py:_special_gene_mask_id`).
#[inline]
fn gene_mask_id(n_genes_total: i64) -> i64 {
    n_genes_total
}

/// `PAD` sentinel = `n_genes_total + 1` (`task.py:_special_pad_id`).
#[inline]
fn pad_id(n_genes_total: i64) -> i64 {
    n_genes_total + 1
}

/// Mutable output slices for one cell (caller-owned; sized `k_enc`/`k_dec`).
///
/// NOTE: `_sparse_encoder_inputs` also produces a `pe_mask` (set at slot 0 in the
/// hidden-readout and all-masked branches). It is intentionally NOT emitted on the
/// rust path and NOT compared by the parity gate. If the model ever consumes
/// `pe_mask` on this path, add it here AND to the golden fixture, the parity
/// comparison, and bump `pyscx::COLLATE_CELLSET_CONTRACT_VERSION`.
pub struct CellOut<'a> {
    pub enc_ids: &'a mut [i64],    // [k_enc]
    pub enc_counts: &'a mut [f32], // [k_enc]
    pub enc_mask: &'a mut [u8],    // [k_enc]
    pub enc_pad: &'a mut [u8],     // [k_enc]
    pub target: &'a mut [f32],     // [k_dec]
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
    /// contract (only `CellIn::gene_ids` does) and is not deduplicated.
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
            .any(|&(_, p)| maskpos[p as usize] != 0)
    }
}

/// One row's encoder mask, inseparable from the panel it indexes.
///
/// A struct rather than two `Option` fields on [`CellIn`] so the pair cannot be
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

/// Immutable, RNG-resolved inputs for one cell.
pub struct CellIn<'a> {
    /// Global gene ids, **sorted ascending, unique** (post-remap/coalesce).
    pub gene_ids: &'a [i32],
    /// Raw counts, parallel to `gene_ids`.
    pub raw: &'a [f32],
    /// Decoder query gene ids for this set (length `k_dec`); shared across the set.
    pub query: &'a [i32],
    /// This row's encoder mask over `query`, with its set's sorted panel.
    /// `None` ⇒ no query-based encoder masking (perturbation path), which also
    /// means the panel is never sorted on that path.
    pub mask: Option<RowMask<'a>>,
    /// Hide gene identity (readout mask) ⇒ single `GENE_MASK` encoder token.
    pub hide_readout: bool,
}

/// Per-run scalars (constant across a batch).
#[derive(Clone, Copy, Debug)]
pub struct CollateConfig {
    pub k_enc: usize,
    pub mode: PreprocessMode,
    pub target_sum: f64,
    /// Measured panel size, for the `pflog` per-cell `center` (denominator `D`).
    pub n_measured: usize,
    /// PFlog (v4) NB overdispersion `α`; required when `mode == PflogRaw`. The
    /// encoder value is `log1p(4α·rc) − center` on raw counts (matrix-wide
    /// pseudocount `1/(4α)`). No per-cell depth. Validated at the caller.
    pub pflog_alpha: Option<f64>,
    pub n_genes_total: i64,
    /// Redefine `library_size` as the sum of raw counts at query positions
    /// (perturbation fixed-query case, `task.py:1729`). Does NOT affect the
    /// preprocessing, which always uses the full-cell raw library size.
    pub lib_size_redef: bool,
}

/// Collate one cell into `out`; returns its `library_size`.
///
/// `out.enc_*` must be length `cfg.k_enc`; `out.target` and (if present)
/// `cin.mask.positions` must be length `cin.query.len()` (= `k_dec`).
///
/// `cin.mask`, when present, must carry
/// [`SetQueryIndex::new(cin.query)`](SetQueryIndex::new) for this row's set.
// Index-based loops below walk several parallel arrays (raw / enc_vals / tgt_vals)
// in lockstep, so explicit indexing is clearer than zipped iterators.
#[allow(clippy::needless_range_loop)]
pub fn collate_cell(cin: &CellIn, cfg: &CollateConfig, out: &mut CellOut) -> f32 {
    let n = cin.gene_ids.len();
    let mask = gene_mask_id(cfg.n_genes_total);
    let pad = pad_id(cfg.n_genes_total);

    // --- library size: sum of raw (clipped >=0); integer counts ⇒ f32-exact. ---
    let lib: f64 = cin.raw.iter().map(|&v| v.max(0.0) as f64).sum();

    // --- per-element preprocessing → (encoder value, target value). ---
    // Parallel to gene_ids. Mirrors preprocess_cell_counts (preprocessing.py:74).
    let mut enc_vals = vec![0f32; n];
    let mut tgt_vals = vec![0f32; n];
    match cfg.mode {
        PreprocessMode::PassThrough => {
            for i in 0..n {
                let rc = cin.raw[i].max(0.0);
                enc_vals[i] = rc;
                tgt_vals[i] = rc;
            }
        }
        PreprocessMode::Log1pRaw => {
            for i in 0..n {
                let rc = cin.raw[i].max(0.0);
                enc_vals[i] = rc.ln_1p();
                tgt_vals[i] = rc;
            }
        }
        PreprocessMode::NormalizeLog1p => {
            let factor = if lib > 0.0 {
                (cfg.target_sum / lib) as f32
            } else {
                1.0
            };
            for i in 0..n {
                let rc = cin.raw[i].max(0.0);
                let scaled = if lib > 0.0 { rc * factor } else { rc };
                let v = scaled.ln_1p();
                enc_vals[i] = v;
                tgt_vals[i] = v;
            }
        }
        PreprocessMode::PflogRaw => {
            // v4: raw-count shifted log `log1p(4α·rc)`, centered by `n_measured`.
            // No per-cell depth (an empty cell → all enc 0 → center 0 → stays 0).
            let four_alpha = 4.0
                * cfg
                    .pflog_alpha
                    .expect("pflog_alpha required for PflogRaw mode (validated at the caller)");
            for i in 0..n {
                let rc = cin.raw[i].max(0.0) as f64;
                enc_vals[i] = (four_alpha * rc).ln_1p() as f32;
            }
            let center =
                (enc_vals.iter().map(|&v| v as f64).sum::<f64>() / cfg.n_measured as f64) as f32;
            for i in 0..n {
                enc_vals[i] -= center;
                tgt_vals[i] = cin.raw[i].max(0.0);
            }
        }
    }

    // --- encoder inputs (top-K), mirroring _sparse_encoder_inputs (task.py:122). ---
    let k_enc = cfg.k_enc;
    for slot in 0..k_enc {
        out.enc_ids[slot] = pad;
        out.enc_counts[slot] = 0.0;
        out.enc_mask[slot] = 0;
        out.enc_pad[slot] = 1;
    }
    if cin.hide_readout {
        out.enc_ids[0] = mask;
        out.enc_mask[0] = 1;
        out.enc_pad[0] = 0;
    } else {
        // positive selection uses RAW counts (selection_counts=raw_counts).
        let pos: Vec<usize> = (0..n).filter(|&i| cin.raw[i].max(0.0) > 0.0).collect();
        if pos.is_empty() {
            // Degenerate cell: one GENE_MASK token, no expr mask (task.py:178).
            out.enc_ids[0] = mask;
            out.enc_pad[0] = 0;
        } else {
            // lexsort((gene_id, -selection)): selection DESC, gene_id ASC tiebreak.
            // gene_ids are unique within a cell, so this is a total order (stability
            // irrelevant); matches np.lexsort exactly for the selected set.
            let mut order = pos;
            order.sort_by(|&a, &b| {
                let ra = cin.raw[a].max(0.0);
                let rb = cin.raw[b].max(0.0);
                // unwrap is safe: `max(0.0)` yields a finite, non-NaN f32, so
                // partial_cmp is always `Some`.
                rb.partial_cmp(&ra)
                    .unwrap()
                    .then(cin.gene_ids[a].cmp(&cin.gene_ids[b]))
            });
            let take = k_enc.min(order.len());

            // Encoder masking (obs): a gene is withheld from the crop when one of
            // this cell's role query positions carrying it is flagged in
            // `enc_mask_positions` (mirrors `mask_gene_ids =
            // query_gene_ids[role_target_mask]`, task.py:1154). `None` ⇒
            // perturbation path (no masking).
            //
            // The panel is searched through the set's `SetQueryIndex` rather
            // than collected into a per-row `HashSet`: the sort is amortised
            // over the set's rows, and the crop loop below does a binary search
            // instead of a SipHash probe per surviving gene.
            let masked = cin.mask;

            // Walk the top-K order, DROPPING withheld genes (their slot is left PAD;
            // no backfill from beyond `take`) and compacting survivors to the left.
            // Mirrors `_sparse_encoder_inputs`' `selected[keep]` (task.py:231-252):
            // withheld genes are absent (PAD), never a GENE_MASK token.
            let mut slot = 0usize;
            for &i in order.iter().take(take) {
                let g = cin.gene_ids[i];
                if masked.is_some_and(|m| m.index.withholds(g, m.positions)) {
                    continue;
                }
                out.enc_ids[slot] = g as i64;
                out.enc_counts[slot] = enc_vals[i];
                out.enc_pad[slot] = 0;
                slot += 1;
            }

            // All-masked fallback: every selected top-K gene was withheld ⇒ a single
            // active GENE_MASK token at slot 0 (task.py:235-247). Distinct from the
            // degenerate-cell branch above, which leaves `enc_mask[0]=0`; here
            // `enc_mask[0]=1`. `enc_counts[0]` stays 0 (matches Python `counts[0]`).
            if slot == 0 {
                out.enc_ids[0] = mask;
                out.enc_mask[0] = 1;
                out.enc_pad[0] = 0;
            }
            // Slots [slot..k_enc] keep their PAD-init values (id=pad, enc_pad=1,
            // counts=0, enc_mask=0).
        }
    }

    // --- target gather at query positions (_gather_counts_at, task.py:210). ---
    // gene_ids sorted ascending unique ⇒ exact-match binary search.
    let k_dec = cin.query.len();
    for q in 0..k_dec {
        let qg = cin.query[q];
        out.target[q] = match cin.gene_ids.binary_search(&qg) {
            Ok(p) => tgt_vals[p],
            Err(_) => 0.0,
        };
    }

    // --- library size (possibly redefined to query-position raw sum). ---
    if cfg.lib_size_redef {
        let mut s = 0f64;
        for q in 0..k_dec {
            if let Ok(p) = cin.gene_ids.binary_search(&cin.query[q]) {
                s += cin.raw[p].max(0.0) as f64;
            }
        }
        s as f32
    } else {
        lib as f32
    }
}

#[cfg(test)]
#[path = "sparse_cellset_collate_tests.rs"]
mod tests;
