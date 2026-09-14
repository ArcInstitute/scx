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
        pflog_alpha: Some(1.0),
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
    // The index is the caller's job and is built from `query` here exactly as
    // `collate_gathered` builds it per set, so every test drives the real
    // pairing rather than a test-only shortcut.
    let idx = enc_mask_positions.map(|_| SetQueryIndex::new(query));
    let cin = CellIn {
        gene_ids,
        raw,
        query,
        mask: enc_mask_positions.map(|positions| RowMask {
            positions,
            index: idx.as_ref().unwrap(),
        }),
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
fn encoder_masking_drops_withheld_genes_to_pad() {
    // Contract: `task.py::_sparse_encoder_inputs` DROPS withheld query genes from
    // the top-K crop (slot becomes PAD), it does NOT replace them in place with a
    // GENE_MASK token.
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
    // top-K order is gene3, gene1, gene5; gene3 & gene1 dropped -> only gene5 survives.
    assert_eq!(b.ids, vec![5, PAD, PAD, PAD]);
    assert_eq!(b.mask, vec![0, 0, 0, 0]);
    assert_eq!(b.pad, vec![0, 1, 1, 1]);
    // The surviving gene keeps its count; dropped slots are zero (no count leak).
    approx(b.counts[0], 1.0f32.ln_1p());
    assert_eq!(b.counts[1], 0.0);
    assert_eq!(b.counts[2], 0.0);
    assert_eq!(b.counts[3], 0.0);
}

#[test]
fn encoder_masking_drops_one_keeps_rest() {
    // Partial mask dropping some-but-not-all: drop gene3, keep gene1 & gene5,
    // compacted left with a trailing PAD.
    let c = cfg(PreprocessMode::Log1pRaw, 4);
    let (b, _) = run(
        &[1, 3, 5],
        &[2.0, 4.0, 1.0],
        &[3, 5, 7, 1],
        Some(&[1, 0, 0, 0]),
        false,
        &c,
    );
    assert_eq!(b.ids, vec![1, 5, PAD, PAD]);
    assert_eq!(b.mask, vec![0, 0, 0, 0]);
    assert_eq!(b.pad, vec![0, 0, 1, 1]);
    approx(b.counts[0], 2.0f32.ln_1p());
    approx(b.counts[1], 1.0f32.ln_1p());
}

#[test]
fn encoder_masking_all_topk_masked_emits_single_mask_token() {
    // All selected top-K genes withheld -> all-masked fallback (task.py:235-247):
    // a single active GENE_MASK at slot 0 with expr_mask[0]=1 (distinct from the
    // degenerate-cell branch, which leaves expr_mask[0]=0).
    let c = cfg(PreprocessMode::Log1pRaw, 4);
    let (b, _) = run(
        &[1, 3, 5],
        &[2.0, 4.0, 1.0],
        &[3, 5, 7, 1],
        Some(&[1, 1, 0, 1]), // mask genes {3, 5, 1} -> all of top-K
        false,
        &c,
    );
    assert_eq!(b.ids, vec![MASK, PAD, PAD, PAD]);
    assert_eq!(b.mask, vec![1, 0, 0, 0]);
    assert_eq!(b.pad, vec![0, 1, 1, 1]);
    assert_eq!(b.counts[0], 0.0); // fallback token carries no count
}

#[test]
fn encoder_masking_outside_topk_has_no_effect() {
    // Mask set overlaps only genes that are not in the top-K -> crop unchanged.
    let c = cfg(PreprocessMode::Log1pRaw, 4);
    let (b, _) = run(
        &[1, 3, 5],
        &[2.0, 4.0, 1.0],
        &[7, 9, 2, 8], // none of these are stored genes 1/3/5
        Some(&[1, 1, 1, 1]),
        false,
        &c,
    );
    assert_eq!(b.ids, vec![3, 1, 5, PAD]);
    assert_eq!(b.mask, vec![0, 0, 0, 0]);
    assert_eq!(b.pad, vec![0, 0, 0, 1]);
}

#[test]
fn encoder_masking_no_backfill_beyond_take() {
    // No-backfill boundary: take = min(k_enc, n_positive) = 2 < 4 positive genes.
    // top-K (raw desc) = [gene3(4), gene7(3)]; mask the KEPT gene3. The drop must
    // NOT pull in the take-th gene (gene1) — survivor is just gene7 + trailing PAD.
    let c = cfg(PreprocessMode::Log1pRaw, 2);
    let (b, _) = run(
        &[1, 3, 5, 7],
        &[2.0, 4.0, 1.0, 3.0],
        &[3, 1, 5, 7],
        Some(&[1, 0, 0, 0]), // mask gene3
        false,
        &c,
    );
    assert_eq!(b.ids, vec![7, PAD]);
    assert_eq!(b.mask, vec![0, 0]);
    assert_eq!(b.pad, vec![0, 1]);
    approx(b.counts[0], 3.0f32.ln_1p());
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
fn pflog_target_is_raw_and_encoder_centered() {
    // cfg pins pflog_alpha = 1.0 → four_alpha = 4.0. v4: enc = log1p(4α·rc) − center
    // (raw counts, no /lib).
    let c = cfg(PreprocessMode::PflogRaw, 4);
    let (b, lib) = run(&[1, 3, 5], &[2.0, 4.0, 1.0], &[3, 5, 7, 1], None, false, &c);
    // target = RAW (exact); library = full sum (still reported for other uses).
    assert_eq!(b.target, vec![4.0, 1.0, 0.0, 2.0]);
    assert_eq!(lib, 7.0);
    // encoder values: log1p(4α·raw) - center, ordered by raw desc.
    let four_alpha = 4.0f64;
    let lp: Vec<f32> = [4.0f32, 2.0, 1.0]
        .iter()
        .map(|&r| (four_alpha * r as f64).ln_1p() as f32)
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
fn golden_vectors_match_state3_reference() {
    // Executable cross-repo contract: every vector here is generated from state3's
    // `_sparse_encoder_inputs` (PassThrough) by `state3/tests/_gen_encoder_crop_golden.py`
    // and committed byte-identically in both repos. The kernel MUST reproduce each
    // one exactly. See the module doc comment.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/encoder_crop_golden.json"
    );
    let raw =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read golden fixture {path}: {e}"));

    // "Byte-identical in both repos" was enforced by nothing but discipline: this
    // test and state3's `test_encoder_crop_golden` would each pass happily against
    // a *different* fixture, which is the exact drift the golden exists to catch.
    // Pin the digest so a one-sided regeneration fails loudly here. state3's
    // generator has a `--check` mode that verifies both copies at once.
    let digest = blake3::hash(raw.as_bytes()).to_hex().to_string();
    assert_eq!(
        &digest[..16],
        ENCODER_GOLDEN_BLAKE3_PREFIX,
        "encoder_crop_golden.json changed. If deliberate: regenerate with \
         `python tests/_gen_encoder_crop_golden.py --scx-repo <scx>` in state3, verify both \
         copies match (`--check`), update this constant, and bump \
         pyscx::COLLATE_CELLSET_CONTRACT_VERSION."
    );

    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();

    let as_i32 = |v: &serde_json::Value| -> Vec<i32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    };
    let as_f32 = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let as_u8 = |v: &serde_json::Value| -> Vec<u8> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u8)
            .collect()
    };
    let as_i64 = |v: &serde_json::Value| -> Vec<i64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect()
    };

    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let gene_ids = as_i32(&case["cell_gene_ids"]);
        let raw_counts = as_f32(&case["raw_counts"]);
        let query = as_i32(&case["query"]);
        let positions = as_u8(&case["enc_mask_positions"]);
        let k_enc = case["k_enc"].as_u64().unwrap() as usize;
        let n_genes_total = case["n_genes_total"].as_i64().unwrap();
        let hide = case["hide_readout"].as_bool().unwrap();

        let c = CollateConfig {
            k_enc,
            // PassThrough: encoder counts == raw counts, so the masking/crop contract
            // is asserted with exact equality (no log1p tolerance).
            mode: PreprocessMode::PassThrough,
            target_sum: 1e4,
            n_measured: gene_ids.len().max(1),
            pflog_alpha: None,
            n_genes_total,
            lib_size_redef: false,
        };
        let (b, _) = run(&gene_ids, &raw_counts, &query, Some(&positions), hide, &c);

        let exp = &case["expected"];
        assert_eq!(
            b.ids,
            as_i64(&exp["encoder_gene_ids"]),
            "ids mismatch [{name}]"
        );
        assert_eq!(
            b.counts,
            as_f32(&exp["encoder_counts"]),
            "counts mismatch [{name}]"
        );
        assert_eq!(
            b.mask,
            as_u8(&exp["encoder_mask"]),
            "mask mismatch [{name}]"
        );
        assert_eq!(
            b.pad,
            as_u8(&exp["encoder_pad_mask"]),
            "pad mismatch [{name}]"
        );
    }
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

/// blake3 prefix of `tests/data/encoder_crop_golden.json`. The state3 copy must be
/// byte-identical; `state3/tests/_gen_encoder_crop_golden.py --check` verifies both.
const ENCODER_GOLDEN_BLAKE3_PREFIX: &str = "e7b2ef4fceecf514";

/// One gene id at two query positions, flagged at only one of them.
///
/// `query` is not deduplicated anywhere: the only in-repo producer draws it
/// with `replace=True` whenever a set holds fewer than `k_dec` distinct genes
/// (`benchmarks/.../cellset_gather.py::_collate_inputs`), so a repeated id is
/// reachable on real input. The set-of-ids semantics this pins is "withheld iff
/// **any** flagged position carries that id" — the `filter(m != 0).map(g)`
/// collect, mirroring Python's `query_gene_ids[role_target_mask]`.
///
/// A lookup that stops at the first matching position instead of folding over
/// the whole equal-id run silently answers "kept" here.
#[test]
fn duplicate_query_ids_withhold_if_any_flagged_position() {
    let c = cfg(PreprocessMode::PassThrough, 3);
    // order = [1(3), 3(2), 5(1)]; take = 3, so the whole row is selected.
    let (b, _) = run(
        &[1, 3, 5],
        &[3.0, 2.0, 1.0],
        // gene 3 twice: position 0 unflagged, position 1 flagged.
        &[3, 3, 1],
        Some(&[0, 1, 0]),
        false,
        &c,
    );
    assert_eq!(b.ids, vec![1, 5, PAD]);
    assert_eq!(b.counts, vec![3.0, 1.0, 0.0]);
    assert_eq!(b.mask, vec![0, 0, 0]);
    assert_eq!(b.pad, vec![0, 0, 1]);
}

/// An unsorted query panel: the mask bit lives at the id's ORIGINAL position.
///
/// `query` carries no ordering contract (only `gene_ids` does), and 8 of the 9
/// golden cases are in fact unsorted. A lookup that sorts the panel must carry
/// each id's original index with it and read `enc_mask_positions` there — using
/// the id's rank in the sorted order instead reads a different row's bit and,
/// on this input, answers "kept" for a gene that was withheld.
#[test]
fn unsorted_query_positions_index_the_mask_at_the_original_offset() {
    let c = cfg(PreprocessMode::PassThrough, 3);
    // query [5,1,3] sorts to [1,3,5] with original positions [1,2,0]; the flag
    // is at position 2, which is rank 1. Reading rank instead of position finds
    // the unset bit at index 1.
    let (b, _) = run(
        &[1, 3, 5],
        &[3.0, 2.0, 1.0],
        &[5, 1, 3],
        Some(&[0, 0, 1]),
        false,
        &c,
    );
    assert_eq!(b.ids, vec![1, 5, PAD]);
    assert_eq!(b.counts, vec![3.0, 1.0, 0.0]);
    assert_eq!(b.pad, vec![0, 0, 1]);
}

/// A mask shorter than the query panel must not panic.
///
/// The `zip` this kernel used before W2 stopped at the shorter of `query` and
/// `enc_mask_positions`, so a short mask silently left the excess positions
/// unflagged. `collate_cell` is `pub` and re-exported from the crate root, so a
/// direct caller can pass a mismatched pair — and indexing `maskpos[p]` would
/// turn that tolerated input into a crash. `collate_gathered` validates the
/// length, so this covers only the direct-caller path, which is exactly the one
/// with no validation in front of it.
#[test]
fn a_mask_shorter_than_the_query_panel_leaves_the_excess_unflagged() {
    let c = cfg(PreprocessMode::PassThrough, 3);
    // query has 3 positions; the mask supplies only the first two. Gene 5 sits
    // at position 2 and is therefore beyond the mask: unflagged, so kept.
    let (b, _) = run(
        &[1, 3, 5],
        &[3.0, 2.0, 1.0],
        &[1, 3, 5],
        Some(&[0, 1]),
        false,
        &c,
    );
    assert_eq!(
        b.ids,
        vec![1, 5, PAD],
        "gene 3 withheld, gene 5 past the mask"
    );
    assert_eq!(b.pad, vec![0, 0, 1]);
}
