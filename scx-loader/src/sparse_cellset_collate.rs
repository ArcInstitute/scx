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
//!    optional seeded downsample in `sparse_cellset::SparseCellSetLoader`;
//! 4. **how the decoder query is addressed** — per set, or per row via
//!    `query_offsets` — and which arrays that addressing makes per-row
//!    (`n_measured`, `enc_mask_positions`);
//! 5. the **set of keys** the batch emits, so an added output like
//!    `target_pad_mask` is a version-visible change.
//!
//! Items 4 and 5 were added at v3. Before that this list said the version does
//! **not** cover array shapes or key names — which was true while the only
//! addressing was per-set, and became false the moment v3 existed *for*
//! addressing and added an output key. Leaving it would have made the bump look
//! like a false mismatch against the scope text in the same file.
//!
//! It still does not cover the batch's array *lengths* (a length-validation
//! error surfaces those loudly at the first batch), nor the `pe_mask` omission
//! noted on [`CellOut`], which is an agreed non-emission rather than a
//! version-dependent behaviour.
//!
//! Version history:
//!
//! - **v1** initial.
//! - **v2** = #356's mode strings (retroactively) + Phase 1B's clip and
//!   downsample.
//! - **v3** = per-row query addressing (`query_offsets`, with per-row
//!   `n_measured` and a ragged `enc_mask_positions`) plus the `target_pad_mask`
//!   output. A call that omits `query_offsets` produces byte-identical values
//!   in every **pre-existing field**; the returned payload is not identical,
//!   because `target_pad_mask` is a new key on every path (all zeros on the
//!   per-set one). A consumer that unpacks named keys is unaffected; one that
//!   asserts an exact key set is not.

use crate::error::{LoaderError, Result};
use crate::tokenize::crop::{self, CropConfig, CropIn, CropOut};
use crate::tokenize::transform;
use crate::tokenize::{CsrRow, Scratch};

// The withheld-gene panel and its per-row mask moved to `tokenize::crop` with the
// crop itself — they are the crop's inputs and nothing else consults them. Re-
// exported here so every existing `sparse_cellset_collate::{RowMask, SetQueryIndex}`
// import keeps resolving; there is one type, not two.
pub use crate::tokenize::crop::{RowMask, SetQueryIndex};

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

    /// Does this mode's decoder target repeat the encoder value, or is it the
    /// raw count?
    ///
    /// The two log-on-raw modes keep the target in raw counts — the model
    /// predicts counts and only *reads* a transform — while the two modes that
    /// rescale the whole cell use the same value on both sides. Stating the rule
    /// once here is what lets the transform dispatch be a table: otherwise each
    /// arm has to remember its own target convention, which is how
    /// `pflog1ppf_raw` and `pflog_raw` came to share a contract version.
    #[inline]
    pub fn target_is_encoder_value(&self) -> bool {
        match self {
            Self::PassThrough | Self::NormalizeLog1p => true,
            Self::Log1pRaw | Self::PflogRaw => false,
        }
    }
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
///
/// `scratch` is caller-owned and reused across rows; hand one per rayon worker
/// through `for_each_init`. It replaced three `Vec`s this function used to
/// allocate per cell — the two preprocessed-value arrays and the selection
/// index — which at the collate arm's pinned shape is three allocations per
/// cell, 64 cells per set.
pub fn collate_cell(
    cin: &CellIn,
    cfg: &CollateConfig,
    scratch: &mut Scratch,
    out: &mut CellOut,
) -> f32 {
    let n = cin.gene_ids.len();

    // --- library size: sum of raw (clipped >=0); integer counts ⇒ f32-exact. ---
    let lib = transform::library_size(cin.raw);

    // --- per-element preprocessing → (encoder value, target value). ---
    // Parallel to gene_ids. Mirrors preprocess_cell_counts (preprocessing.py:74).
    scratch.reset_values(n);
    // Destructured so the crop below can borrow `enc_vals` immutably while it
    // mutates `order`; the two are disjoint fields of one bundle.
    let Scratch {
        order,
        enc_vals,
        tgt_vals,
        ..
    } = scratch;
    match cfg.mode {
        PreprocessMode::PassThrough => transform::pass_through(cin.raw, enc_vals),
        PreprocessMode::Log1pRaw => transform::log1p_raw(cin.raw, enc_vals),
        PreprocessMode::NormalizeLog1p => {
            transform::normalize_log1p(cin.raw, enc_vals, cfg.target_sum, lib)
        }
        PreprocessMode::PflogRaw => transform::pflog_raw(
            cin.raw,
            enc_vals,
            cfg.pflog_alpha
                .expect("pflog_alpha required for PflogRaw mode (validated at the caller)"),
            cfg.n_measured,
        ),
    }
    if cfg.mode.target_is_encoder_value() {
        tgt_vals.copy_from_slice(enc_vals);
    } else {
        transform::pass_through(cin.raw, tgt_vals);
    }

    // --- encoder inputs (top-K), mirroring _sparse_encoder_inputs (task.py:122). ---
    // Selection is over RAW counts (state3 `selection_counts=raw_counts`) while the
    // emitted values are the preprocessed ones — which is why `CropIn` carries both.
    crop::top_k(
        &CropIn {
            row: CsrRow {
                gene_ids: cin.gene_ids,
                values: cin.raw,
            },
            emit: enc_vals,
            withheld: cin.mask,
            hide_readout: cin.hide_readout,
        },
        &CropConfig {
            k: cfg.k_enc,
            n_genes_total: cfg.n_genes_total,
        },
        order,
        &mut CropOut {
            ids: &mut out.enc_ids[..],
            values: &mut out.enc_counts[..],
            mask: &mut out.enc_mask[..],
            pad: &mut out.enc_pad[..],
        },
    );

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
