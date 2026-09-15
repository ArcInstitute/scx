//! Model-agnostic tokenisation kernels over a gathered CSR batch (W6).
//!
//! Every token model in the largest family — Geneformer, scGPT and its lineage,
//! UCE, TranscriptFormer, scPRINT, C2S, STATE3 — runs the same handful of numeric
//! steps per cell: rank the genes, bin the values, crop to a fixed length, sample
//! a subsequence, normalise. Before this module SCX offered exactly one of them
//! (the top-K crop) and only through STATE-shaped `CellIn` / `CollateConfig`
//! types, so every consumer re-implemented the rest in a Python loop.
//!
//! These kernels take borrowed CSR slices plus caller-supplied [`Scratch`], write
//! into caller-owned output slices and allocate nothing per row. A model's
//! tokeniser is then a configuration of kernels rather than a loop.
//!
//! # What the contract version covers
//!
//! [`TOKENIZE_CONTRACT_VERSION`] is exported to Python as
//! `pyscx.tokenize.CONTRACT_VERSION` and is asserted by consumers at setup. It
//! covers, exhaustively:
//!
//! 1. the numeric semantics of every kernel in this module — ordering, tie
//!    rules, bin-edge computation, weight transforms, and what each returns;
//! 2. the parameters each kernel *requires*, and the meaning of each;
//! 3. the seed derivation for the RNG-driven kernels ([`sample`], and [`bin`]'s
//!    [`BinTie::SeededUniform`](bin::BinTie::SeededUniform)).
//!
//! It does **not** cover array shapes or Python key names — a length-validation
//! error surfaces those loudly at the first call.
//!
//! # Where these diverge from the reference tokenisers
//!
//! Each kernel's semantics were taken from a pinned revision of the model that
//! defines them, and three of those references cannot be reproduced exactly by
//! any deterministic Rust implementation. The divergences are declared per
//! kernel in this module's docs and collected in `docs/tokenize.md`; they are
//! **not** incidental, and a consumer that needs draw-for-draw agreement with
//! the reference must keep using the reference.
//!
//! # Epoch-dependent randomness stays with the consumer
//!
//! scGPT's per-epoch masking and any random crop are the model's business, not
//! this module's. The kernels here are RNG-free except where the reference
//! itself samples, and there the draw is keyed on content identity (file, row)
//! rather than on epoch or worker, so a resumed run does not silently switch
//! streams — the rule §3 of the storage contract states.

pub mod bin;
pub mod crop;
pub mod rank;
pub mod sample;
pub mod transform;

/// Version of the kernel contract described in this module's docs.
///
/// Exported as `pyscx.tokenize.CONTRACT_VERSION`. Bump on any change to what a
/// consumer mirroring these kernels must match.
pub const TOKENIZE_CONTRACT_VERSION: u32 = 1;

/// One cell's CSR row in the global vocabulary.
///
/// `gene_ids` is **sorted ascending and unique** — what `remap_row`
/// (`sparse_cellset.rs`) produces, mirroring state3's `finalize_csr_row`. Kernels
/// rely on it for exact-match binary search and for the gene-id tiebreak being a
/// total order; they do not re-verify it, because the verification would cost
/// more than the kernels do.
#[derive(Clone, Copy)]
pub struct CsrRow<'a> {
    pub gene_ids: &'a [i32],
    pub values: &'a [f32],
}

impl<'a> CsrRow<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.gene_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.gene_ids.is_empty()
    }
}

/// Per-thread reusable buffers for the collate path's kernel chain.
///
/// `collate_cell` used to allocate three `Vec`s per cell — the two preprocessed-
/// value arrays and the selection index. At the `cellset_gather` collate arm's
/// pinned shape that is three allocations per cell for 64 cells per set; Phase
/// 1's gate note named "the merge that would collect the rest needs per-row
/// scratch" as the work that belongs here. Hand one of these to `rayon`'s
/// `for_each_init` and it is created once per worker, not once per row.
///
/// Kernels take the individual buffers they need rather than this bundle — see
/// [`crop::top_k`] — so there is no field here that no caller reads. [`bin`] and
/// [`sample`] want an `f64` buffer that this chain never allocates, and take
/// their own.
#[derive(Default)]
pub struct Scratch {
    /// Indices into a row, used for selection and ordering.
    pub(crate) order: Vec<usize>,
    /// Per-element encoder values, parallel to the row.
    pub(crate) enc_vals: Vec<f32>,
    /// Per-element target values, parallel to the row.
    pub(crate) tgt_vals: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resize `enc_vals` / `tgt_vals` to `n`, zero-filled.
    ///
    /// `clear` + `resize` rather than `resize` alone: a shorter previous row
    /// would otherwise leave its tail visible, and every caller here writes
    /// every element before reading it, so the zero-fill is belt-and-braces
    /// that also makes the buffers byte-identical to the `vec![0f32; n]` they
    /// replace.
    #[inline]
    pub(crate) fn reset_values(&mut self, n: usize) {
        self.enc_vals.clear();
        self.enc_vals.resize(n, 0.0);
        self.tgt_vals.clear();
        self.tgt_vals.resize(n, 0.0);
    }
}
