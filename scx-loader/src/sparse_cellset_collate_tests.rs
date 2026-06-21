use super::*;

const N_GENES: i64 = 100;
const MASK: i64 = N_GENES; // 100
const PAD: i64 = N_GENES + 1; // 101

struct Buf {
    ids: Vec<i64>,
    counts: Vec<f32>,
    mask: Vec<u8>,
    pad: Vec<u8>,
    target: Vec<f32>,
}

fn cfg(mode: PreprocessMode, k_enc: usize) -> CollateConfig {
    CollateConfig {
        k_enc,
        mode,
        target_sum: 1e4,
        n_measured: 8,
        n_genes_total: N_GENES,
        lib_size_redef: false,
    }
}

fn run(
    gene_ids: &[i32],
    raw: &[f32],
    query: &[i32],
    enc_mask_positions: Option<&[u8]>,
    hide: bool,
    cfg: &CollateConfig,
) -> (Buf, f32) {
    let mut b = Buf {
        ids: vec![0; cfg.k_enc],
        counts: vec![0.0; cfg.k_enc],
        mask: vec![0; cfg.k_enc],
        pad: vec![0; cfg.k_enc],
        target: vec![0.0; query.len()],
    };
    let cin = CellIn {
        gene_ids,
        raw,
        query,
        enc_mask_positions,
        hide_readout: hide,
    };
    let lib = {
        let mut out = CellOut {
            enc_ids: &mut b.ids,
            enc_counts: &mut b.counts,
            enc_mask: &mut b.mask,
            enc_pad: &mut b.pad,
            target: &mut b.target,
        };
        collate_cell(&cin, cfg, &mut out)
    };
    (b, lib)
}

fn approx(a: f32, b: f32) {
    assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs(), "approx {a} vs {b}");
}

#[test]
fn log1p_raw_topk_order_and_gather() {
    let c = cfg(PreprocessMode::Log1pRaw, 4);
    let (b, lib) = run(&[1, 3, 5], &[2.0, 4.0, 1.0], &[3, 5, 7, 1], None, false, &c);
    // top-K by raw DESC: gene3(4), gene1(2), gene5(1); slot 3 is PAD.
    assert_eq!(b.ids, vec![3, 1, 5, PAD]);
    approx(b.counts[0], 4.0f32.ln_1p());
    approx(b.counts[1], 2.0f32.ln_1p());
    approx(b.counts[2], 1.0f32.ln_1p());
    assert_eq!(b.counts[3], 0.0);
    assert_eq!(b.pad, vec![0, 0, 0, 1]);
    assert_eq!(b.mask, vec![0, 0, 0, 0]);
    // target = RAW (log1p_raw): query [3,5,7,1] -> [4,1,0,2]; exact.
    assert_eq!(b.target, vec![4.0, 1.0, 0.0, 2.0]);
    assert_eq!(lib, 7.0);
}

#[test]
fn topk_truncates_to_k_enc() {
    let c = cfg(PreprocessMode::Log1pRaw, 2);
    let (b, _) = run(&[1, 3, 5], &[2.0, 4.0, 1.0], &[3], None, false, &c);
    assert_eq!(b.ids, vec![3, 1]); // gene5 dropped
    assert_eq!(b.pad, vec![0, 0]);
}

#[test]
fn topk_tiebreak_by_gene_id_ascending() {
    let c = cfg(PreprocessMode::Log1pRaw, 2);
    let (b, _) = run(&[2, 5], &[3.0, 3.0], &[2], None, false, &c);
    assert_eq!(b.ids, vec![2, 5]); // equal selection -> gene id ascending
}

#[test]
fn encoder_masking_hides_query_genes() {
    let c = cfg(PreprocessMode::Log1pRaw, 4);
    // mask query positions for ids {3, 1}.
    let (b, _) = run(
        &[1, 3, 5],
        &[2.0, 4.0, 1.0],
        &[3, 5, 7, 1],
        Some(&[1, 0, 0, 1]),
        false,
        &c,
    );
    // top-K order is gene3, gene1, gene5; gene3 & gene1 masked.
    assert_eq!(b.ids, vec![MASK, MASK, 5, PAD]);
    assert_eq!(b.mask, vec![1, 1, 0, 0]);
}

#[test]
fn hide_readout_single_mask_token() {
    let c = cfg(PreprocessMode::Log1pRaw, 4);
    let (b, _) = run(&[1, 3, 5], &[2.0, 4.0, 1.0], &[3, 1], None, true, &c);
    assert_eq!(b.ids, vec![MASK, PAD, PAD, PAD]);
    assert_eq!(b.mask, vec![1, 0, 0, 0]);
    assert_eq!(b.pad, vec![0, 1, 1, 1]);
    // target still gathered for hidden cells: [4, 2].
    assert_eq!(b.target, vec![4.0, 2.0]);
}

#[test]
fn empty_cell_single_mask_no_expr() {
    let c = cfg(PreprocessMode::Log1pRaw, 3);
    let (b, lib) = run(&[], &[], &[3, 5], None, false, &c);
    assert_eq!(b.ids, vec![MASK, PAD, PAD]);
    assert_eq!(b.mask, vec![0, 0, 0]); // no expr mask for degenerate cell
    assert_eq!(b.pad, vec![0, 1, 1]);
    assert_eq!(b.target, vec![0.0, 0.0]);
    assert_eq!(lib, 0.0);
}

#[test]
fn no_positive_counts_single_mask() {
    let c = cfg(PreprocessMode::Log1pRaw, 2);
    let (b, lib) = run(&[2], &[0.0], &[2], None, false, &c);
    assert_eq!(b.ids, vec![MASK, PAD]);
    assert_eq!(b.mask, vec![0, 0]);
    assert_eq!(b.target, vec![0.0]); // gene2 present but raw 0
    assert_eq!(lib, 0.0);
}

#[test]
fn pass_through_keeps_raw_in_encoder() {
    let c = cfg(PreprocessMode::PassThrough, 3);
    let (b, _) = run(&[1, 3], &[2.0, 5.0], &[1, 3], None, false, &c);
    assert_eq!(b.ids, vec![3, 1, PAD]);
    assert_eq!(b.counts[0], 5.0); // raw, not log1p
    assert_eq!(b.counts[1], 2.0);
    assert_eq!(b.target, vec![2.0, 5.0]);
}

#[test]
fn library_size_redefinition_sums_query_positions() {
    let mut c = cfg(PreprocessMode::Log1pRaw, 4);
    c.lib_size_redef = true;
    // full lib = 7; redefined = raw at {gene3, gene1} = 4 + 2 = 6.
    let (_, lib) = run(&[1, 3, 5], &[2.0, 4.0, 1.0], &[3, 1], None, false, &c);
    assert_eq!(lib, 6.0);
}

#[test]
fn pflog1ppf_target_is_raw_and_encoder_centered() {
    let c = cfg(PreprocessMode::Pflog1ppfRaw, 4);
    let (b, lib) = run(&[1, 3, 5], &[2.0, 4.0, 1.0], &[3, 5, 7, 1], None, false, &c);
    // target = RAW (exact); library = full sum.
    assert_eq!(b.target, vec![4.0, 1.0, 0.0, 2.0]);
    assert_eq!(lib, 7.0);
    // encoder values: log1p(raw/lib) - center, finite, ordered by raw desc.
    let libf = 7.0f32;
    let lp: Vec<f32> = [4.0f32, 2.0, 1.0]
        .iter()
        .map(|&r| (r / libf).ln_1p())
        .collect();
    let center = (lp.iter().map(|&v| v as f64).sum::<f64>() / 8.0) as f32;
    approx(b.counts[0], lp[0] - center);
    approx(b.counts[1], lp[1] - center);
    approx(b.counts[2], lp[2] - center);
}

#[test]
fn normalize_log1p_encoder_and_target_are_normalized() {
    // NormalizeLog1p: both encoder and target carry log1p(raw * target_sum/lib).
    // Encoder top-K order still uses RAW counts (selection is on raw, desc).
    let c = cfg(PreprocessMode::NormalizeLog1p, 4); // target_sum=1e4
    let (b, lib) = run(
        &[10, 20, 30],
        &[1.0, 2.0, 3.0],
        &[10, 20, 30, 40],
        None,
        false,
        &c,
    );

    assert_eq!(lib, 6.0); // sum of raw
    let factor = (1e4f64 / 6.0) as f32;
    let v = |r: f32| (r * factor).ln_1p();
    // top-K by raw desc → gene 30 (raw 3), 20 (raw 2), 10 (raw 1), then PAD.
    assert_eq!(b.ids[0], 30);
    assert_eq!(b.ids[1], 20);
    assert_eq!(b.ids[2], 10);
    assert_eq!(b.ids[3], PAD);
    approx(b.counts[0], v(3.0));
    approx(b.counts[1], v(2.0));
    approx(b.counts[2], v(1.0));
    // target gathered at query positions [10,20,30,40] = normalized-log1p (not raw),
    // 0 for the absent gene 40.
    approx(b.target[0], v(1.0));
    approx(b.target[1], v(2.0));
    approx(b.target[2], v(3.0));
    assert_eq!(b.target[3], 0.0);
}

#[test]
fn normalize_log1p_zero_library_uses_factor_one() {
    // All-zero counts → lib == 0 → factor falls back to 1.0 (no div-by-zero),
    // scaled == raw, log1p(0) == 0; degenerate cell emits one GENE_MASK token.
    let c = cfg(PreprocessMode::NormalizeLog1p, 4);
    let (b, lib) = run(&[10, 20], &[0.0, 0.0], &[10, 20, 30], None, false, &c);

    assert_eq!(lib, 0.0);
    assert_eq!(b.ids[0], MASK); // no positive counts → single mask token
    assert_eq!(b.pad[0], 0);
    assert_eq!(b.target, vec![0.0, 0.0, 0.0]); // log1p(0)=0 everywhere
}
