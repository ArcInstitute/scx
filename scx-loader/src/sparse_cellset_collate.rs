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

use std::collections::HashSet;

use crate::error::{LoaderError, Result};

/// Per-cell preprocessing mode — mirrors `preprocessing.py:preprocess_cell_counts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreprocessMode {
    PassThrough,
    Log1pRaw,
    Pflog1ppfRaw,
    NormalizeLog1p,
}

impl PreprocessMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "pass_through" => Ok(Self::PassThrough),
            "log1p_raw" => Ok(Self::Log1pRaw),
            "pflog1ppf_raw" => Ok(Self::Pflog1ppfRaw),
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
    /// `[k_dec]` per-cell encoder-mask positions over `query` (obs role mask).
    /// `None` ⇒ no query-based encoder masking (perturbation path).
    pub enc_mask_positions: Option<&'a [u8]>,
    /// Hide gene identity (readout mask) ⇒ single `GENE_MASK` encoder token.
    pub hide_readout: bool,
}

/// Per-run scalars (constant across a batch).
#[derive(Clone, Copy, Debug)]
pub struct CollateConfig {
    pub k_enc: usize,
    pub mode: PreprocessMode,
    pub target_sum: f64,
    /// Measured panel size, for the `pflog1ppf` per-cell `center`.
    pub n_measured: usize,
    pub n_genes_total: i64,
    /// Redefine `library_size` as the sum of raw counts at query positions
    /// (perturbation fixed-query case, `task.py:1729`). Does NOT affect the
    /// preprocessing, which always uses the full-cell raw library size.
    pub lib_size_redef: bool,
}

/// Collate one cell into `out`; returns its `library_size`.
///
/// `out.enc_*` must be length `cfg.k_enc`; `out.target` and (if present)
/// `cin.enc_mask_positions` must be length `cin.query.len()` (= `k_dec`).
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
        PreprocessMode::Pflog1ppfRaw => {
            if lib <= 0.0 {
                for i in 0..n {
                    enc_vals[i] = 0.0;
                    tgt_vals[i] = cin.raw[i].max(0.0);
                }
            } else {
                let libf = lib as f32;
                for i in 0..n {
                    let rc = cin.raw[i].max(0.0);
                    enc_vals[i] = (rc / libf).ln_1p();
                }
                let center = (enc_vals.iter().map(|&v| v as f64).sum::<f64>()
                    / cfg.n_measured as f64) as f32;
                for i in 0..n {
                    enc_vals[i] -= center;
                    tgt_vals[i] = cin.raw[i].max(0.0);
                }
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
            for (slot, &i) in order.iter().take(take).enumerate() {
                out.enc_ids[slot] = cin.gene_ids[i] as i64;
                out.enc_counts[slot] = enc_vals[i];
                out.enc_pad[slot] = 0;
            }
            // Encoder masking (obs): hide top-K genes that fall in the cell's
            // role query subset (np.isin, task.py:197).
            if let Some(maskpos) = cin.enc_mask_positions {
                let masked: HashSet<i64> = cin
                    .query
                    .iter()
                    .zip(maskpos.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(&g, _)| g as i64)
                    .collect();
                if !masked.is_empty() {
                    for slot in 0..take {
                        if masked.contains(&out.enc_ids[slot]) {
                            out.enc_ids[slot] = mask;
                            out.enc_mask[slot] = 1;
                        }
                    }
                }
            }
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
