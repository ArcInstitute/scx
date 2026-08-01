//! Tests for the seeded per-row count downsampler.
//!
//! The load-bearing ones are not "does it sample" but: is the draw invariant to
//! scheduling and to manifest order, does the key actually mix, and does the
//! no-op branch still integerise. Each of those encodes a decision that would
//! otherwise be re-derived (or silently regressed) later.

use super::*;

/// Convenience: run one row and return the resulting `(indices, counts)`.
fn run(
    idx: &[i32],
    dat: &[f32],
    method: DownsampleMethod,
    target: u64,
    seed: u64,
    ident: u64,
    row: u64,
) -> (Vec<i32>, Vec<f32>) {
    let cfg = DownsampleConfig {
        target_library_size: target,
        method,
        seed,
        file_identities: vec![ident],
    };
    let mut i = idx.to_vec();
    let mut d = dat.to_vec();
    downsample_row(&mut i, &mut d, &cfg, ident, row);
    (i, d)
}

fn lib(d: &[f32]) -> u64 {
    d.iter().map(|&v| v as u64).sum()
}

// --------------------------------------------------------------------------
// Method parsing
// --------------------------------------------------------------------------

#[test]
fn parses_the_two_accepted_methods() {
    assert_eq!(
        DownsampleMethod::parse("binomial").unwrap(),
        DownsampleMethod::Binomial
    );
    assert_eq!(
        DownsampleMethod::parse("multinomial").unwrap(),
        DownsampleMethod::Multinomial
    );
}

#[test]
fn rejects_an_unknown_method_by_name() {
    let err = DownsampleMethod::parse("hypergeometric").unwrap_err();
    let msg = err.to_string();
    // The offending spelling must appear, or the operator cannot see their typo.
    assert!(msg.contains("hypergeometric"), "unhelpful message: {msg}");
    assert!(
        msg.contains("binomial"),
        "should list the accepted set: {msg}"
    );
}

#[test]
fn rejects_a_zero_target() {
    let cfg = DownsampleConfig {
        target_library_size: 0,
        method: DownsampleMethod::Binomial,
        seed: 1,
        file_identities: vec![],
    };
    assert!(cfg.validate().is_err());
}

// --------------------------------------------------------------------------
// The mixer: the whole point is that components do not trade off against
// each other the way the crate's existing additive convention does.
// --------------------------------------------------------------------------

#[test]
fn mixer_does_not_collide_across_components() {
    // Under the crate's existing additive scheme (`seed + epoch * PHI`), the
    // pairs below would alias. Assert they do not.
    let m = DownsampleMethod::Binomial;
    let base = row_seed(10, m, 20, 30);
    assert_ne!(base, row_seed(11, m, 20, 29));
    assert_ne!(base, row_seed(11, m, 19, 30));
    assert_ne!(base, row_seed(10, m, 21, 29));
    assert_ne!(base, row_seed(9, m, 21, 30));
    // And the method tag separates the streams.
    assert_ne!(base, row_seed(10, DownsampleMethod::Multinomial, 20, 30));
}

#[test]
fn mixer_is_sensitive_to_every_component() {
    let m = DownsampleMethod::Multinomial;
    let base = row_seed(7, m, 7, 7);
    assert_ne!(base, row_seed(8, m, 7, 7), "seed ignored");
    assert_ne!(base, row_seed(7, m, 8, 7), "file identity ignored");
    assert_ne!(base, row_seed(7, m, 7, 8), "row ignored");
}

/// Anti-tautology guard for the mixer tests above: an *additive* key — the
/// convention this module deliberately departs from — must actually fail them.
/// Without this, `mixer_does_not_collide_across_components` could be passing for
/// reasons unrelated to the mixing (e.g. if it were comparing distinct inputs).
#[test]
fn an_additive_key_would_collide_where_the_mixer_does_not() {
    let additive = |seed: u64, ident: u64, row: u64| {
        seed.wrapping_add(ident.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(row)
    };
    // The exact pair the real mixer keeps distinct.
    assert_eq!(additive(10, 20, 30), additive(11, 20, 29));
    assert_ne!(
        row_seed(10, DownsampleMethod::Binomial, 20, 30),
        row_seed(11, DownsampleMethod::Binomial, 20, 29)
    );
}

// --------------------------------------------------------------------------
// File identity
// --------------------------------------------------------------------------

#[test]
fn file_identity_is_alias_invariant_for_the_same_file() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("a.scx");
    std::fs::write(&f, b"x").unwrap();

    let direct = file_identity(f.to_str().unwrap());
    // Same file reached through a `..` component.
    let round_about = dir.path().join("sub").join("..").join("a.scx");
    std::fs::create_dir_all(dir.path().join("sub")).unwrap();
    let via_dotdot = file_identity(round_about.to_str().unwrap());

    assert_eq!(
        direct, via_dotdot,
        "two spellings of one file must key identically"
    );
}

#[test]
fn file_identity_differs_across_files_and_is_stable() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    std::fs::write(&a, b"x").unwrap();
    std::fs::write(&b, b"x").unwrap();

    let ia = file_identity(a.to_str().unwrap());
    let ib = file_identity(b.to_str().unwrap());
    assert_ne!(ia, ib);
    assert_eq!(ia, file_identity(a.to_str().unwrap()), "not stable");
}

#[test]
fn file_identity_falls_back_for_a_nonexistent_path() {
    // Deterministic even when canonicalize fails — just not alias-invariant.
    let i1 = file_identity("/definitely/not/here/x.scx");
    let i2 = file_identity("/definitely/not/here/x.scx");
    assert_eq!(i1, i2);
    assert_ne!(i1, file_identity("/definitely/not/here/y.scx"));
}

// --------------------------------------------------------------------------
// Rounding, clipping, no-op branch
// --------------------------------------------------------------------------

#[test]
fn rounds_half_to_even_like_np_rint() {
    // Ties-to-even: 2.5 -> 2, 3.5 -> 4. `f32::round()` would give 3 and 4, so a
    // naive port disagrees on exactly the first element.
    let (_, d) = run(
        &[0, 1],
        &[2.5, 3.5],
        DownsampleMethod::Binomial,
        // target above the library size (2 + 4 = 6) -> no-op branch, so the
        // rounding is what is under test, not the sampler.
        1000,
        1,
        1,
        0,
    );
    assert_eq!(d, vec![2.0, 4.0]);
    // Guard the premise: the naive rounding really does differ here.
    assert_eq!(2.5f32.round(), 3.0);
}

#[test]
fn no_op_branch_still_integerises_every_cell() {
    // Mirrors downsample.py:78-79 — enabling downsampling rints the whole corpus,
    // not just the cells above target. Surprising, deliberate, pinned.
    let (i, d) = run(
        &[3, 5, 7],
        &[1.4, 2.6, 3.0],
        DownsampleMethod::Multinomial,
        1_000_000,
        42,
        99,
        0,
    );
    assert_eq!(i, vec![3, 5, 7]);
    assert_eq!(d, vec![1.0, 3.0, 3.0]);
}

#[test]
fn clips_negatives_before_sampling() {
    // A negative must never reach the binomial sampler (numpy raises on n<0).
    // Here it is clipped to 0 and then pruned as a zero count.
    let (i, d) = run(
        &[1, 2, 3],
        &[-5.0, 10.0, 4.0],
        DownsampleMethod::Binomial,
        4,
        7,
        7,
        0,
    );
    assert!(
        !i.contains(&1),
        "the negative entry should not survive: {i:?}"
    );
    assert!(d.iter().all(|&v| v >= 0.0), "negative leaked: {d:?}");
}

#[test]
fn clip_negatives_does_not_prune() {
    // The standalone clip keeps structure: the reference prunes only inside its
    // downsample branch, so nnz must be identical with and without a clip.
    let mut d = vec![-1.0f32, 0.0, 3.0];
    clip_negatives(&mut d);
    assert_eq!(d, vec![0.0, 0.0, 3.0]);
}

#[test]
fn empty_and_all_zero_rows_are_left_alone() {
    let (i, d) = run(&[], &[], DownsampleMethod::Binomial, 10, 1, 1, 0);
    assert!(i.is_empty() && d.is_empty());

    // All-zero: library_size == 0 takes the no-op branch, and write_back prunes
    // the zeros, leaving an empty row.
    let (i, d) = run(
        &[1, 2],
        &[0.0, 0.0],
        DownsampleMethod::Multinomial,
        10,
        1,
        1,
        0,
    );
    assert!(i.is_empty(), "zeros should be pruned: {i:?}");
    assert!(d.is_empty());
}

// --------------------------------------------------------------------------
// Sampler behaviour
// --------------------------------------------------------------------------

#[test]
fn multinomial_hits_the_target_exactly() {
    let dat: Vec<f32> = (1..=20).map(|k| (k * 7) as f32).collect();
    let idx: Vec<i32> = (0..20).collect();
    for row in 0..25u64 {
        let (_, d) = run(&idx, &dat, DownsampleMethod::Multinomial, 100, 5, 11, row);
        assert_eq!(lib(&d), 100, "row {row} missed the target");
    }
}

#[test]
fn binomial_hits_the_target_only_in_expectation() {
    let dat: Vec<f32> = (1..=20).map(|k| (k * 7) as f32).collect();
    let idx: Vec<i32> = (0..20).collect();
    let libs: Vec<u64> = (0..40u64)
        .map(|row| {
            let (_, d) = run(&idx, &dat, DownsampleMethod::Binomial, 100, 5, 11, row);
            lib(&d)
        })
        .collect();
    // Never upsamples, and the mean is near target while individual draws are not
    // pinned to it — if every draw were exactly 100 we would have written a
    // multinomial by mistake.
    assert!(libs.iter().all(|&l| l <= 1470), "upsampled: {libs:?}");
    let mean = libs.iter().sum::<u64>() as f64 / libs.len() as f64;
    assert!((mean - 100.0).abs() < 15.0, "mean {mean} far from target");
    assert!(
        libs.iter().any(|&l| l != 100),
        "every binomial draw hit target exactly — suspiciously multinomial"
    );
}

#[test]
fn never_upsamples_under_either_method() {
    for method in [DownsampleMethod::Binomial, DownsampleMethod::Multinomial] {
        let (_, d) = run(&[1, 2], &[3.0, 4.0], method, 1000, 3, 3, 0);
        assert_eq!(lib(&d), 7, "{method:?} inflated a below-target row");
    }
}

#[test]
fn pruning_keeps_the_row_canonical() {
    // A long row squeezed hard must lose entries, keep ascending indices, and stay
    // parallel.
    let idx: Vec<i32> = (0..64).map(|k| k * 3).collect();
    let dat: Vec<f32> = vec![2.0; 64];
    let (i, d) = run(&idx, &dat, DownsampleMethod::Multinomial, 5, 9, 9, 1);
    assert_eq!(i.len(), d.len());
    assert!(i.len() < 64, "nothing pruned from a 128 -> 5 downsample");
    assert!(i.windows(2).all(|w| w[0] < w[1]), "order broken: {i:?}");
    assert!(i.iter().all(|g| idx.contains(g)), "invented a gene id");
    assert_eq!(lib(&d), 5);
}

// --------------------------------------------------------------------------
// Determinism and invariance — the properties the design exists for
// --------------------------------------------------------------------------

#[test]
fn same_key_gives_the_same_draw() {
    let idx: Vec<i32> = (0..30).collect();
    let dat: Vec<f32> = (1..=30).map(|k| (k * 5) as f32).collect();
    let a = run(&idx, &dat, DownsampleMethod::Multinomial, 40, 123, 456, 78);
    let b = run(&idx, &dat, DownsampleMethod::Multinomial, 40, 123, 456, 78);
    assert_eq!(a, b);
}

#[test]
fn changing_the_seed_changes_the_draw() {
    // Anti-tautology for `same_key_gives_the_same_draw`: without this, an
    // implementation that ignored the seed entirely would pass it.
    let idx: Vec<i32> = (0..30).collect();
    let dat: Vec<f32> = (1..=30).map(|k| (k * 5) as f32).collect();
    let a = run(&idx, &dat, DownsampleMethod::Multinomial, 40, 1, 456, 78);
    let b = run(&idx, &dat, DownsampleMethod::Multinomial, 40, 2, 456, 78);
    assert_ne!(a, b, "seed had no effect");
}

#[test]
fn different_rows_and_files_draw_differently() {
    let idx: Vec<i32> = (0..30).collect();
    let dat: Vec<f32> = (1..=30).map(|k| (k * 5) as f32).collect();
    let base = run(&idx, &dat, DownsampleMethod::Multinomial, 40, 1, 100, 0);
    assert_ne!(
        base,
        run(&idx, &dat, DownsampleMethod::Multinomial, 40, 1, 100, 1),
        "row ignored"
    );
    assert_ne!(
        base,
        run(&idx, &dat, DownsampleMethod::Multinomial, 40, 1, 200, 0),
        "file identity ignored"
    );
}

#[test]
fn the_two_methods_draw_independent_streams() {
    let idx: Vec<i32> = (0..30).collect();
    let dat: Vec<f32> = (1..=30).map(|k| (k * 5) as f32).collect();
    let b = run(&idx, &dat, DownsampleMethod::Binomial, 40, 1, 1, 0);
    let m = run(&idx, &dat, DownsampleMethod::Multinomial, 40, 1, 1, 0);
    assert_ne!(b, m);
}

#[test]
fn the_draw_is_independent_of_batch_position() {
    // Scheduling invariance: `read_rows_with` fires callbacks in shard-grouped
    // order and the collate rayon loop is unordered, so a row's counts must depend
    // only on its own identity. Emulate two different visitation orders over the
    // same set of rows and require identical per-row results.
    let idx: Vec<i32> = (0..24).collect();
    let dat: Vec<f32> = (1..=24).map(|k| (k * 9) as f32).collect();
    let cfg = DownsampleConfig {
        target_library_size: 33,
        method: DownsampleMethod::Multinomial,
        seed: 77,
        file_identities: vec![0xAAAA],
    };

    let sample = |rows: &[u64]| -> Vec<(u64, Vec<f32>)> {
        rows.iter()
            .map(|&r| {
                let mut i = idx.clone();
                let mut d = dat.clone();
                downsample_row(&mut i, &mut d, &cfg, 0xAAAA, r);
                (r, d)
            })
            .collect()
    };

    let forward = sample(&[0, 1, 2, 3, 4, 5]);
    let mut shuffled = sample(&[4, 0, 5, 2, 1, 3]);
    shuffled.sort_by_key(|(r, _)| *r);
    assert_eq!(forward, shuffled, "draw depended on visitation order");
}

#[test]
fn identity_for_falls_back_to_a_constant_never_to_file_id() {
    // The fallback must NOT be `file_id`. That would be construction-order keying
    // — the scheme this module exists to avoid — and an earlier revision of this
    // test pinned exactly that footgun. A constant keys on `(seed, method, row)`
    // alone, matching the standalone `downsample_counts_csr` convention so the two
    // entry points cannot disagree; the loader refuses an empty table outright
    // when there is more than one file.
    let cfg = DownsampleConfig {
        target_library_size: 10,
        method: DownsampleMethod::Binomial,
        seed: 1,
        file_identities: vec![0xDEAD],
    };
    assert_eq!(cfg.identity_for(0), 0xDEAD);
    assert_eq!(cfg.identity_for(3), 0, "must not fall back to the file_id");
    assert_eq!(
        cfg.identity_for(7),
        0,
        "fallback must be constant across ids"
    );
}

#[test]
fn multinomial_never_empties_a_row_that_had_counts() {
    // `target >= 1` is validated, and on the sampling path `library_size > target`,
    // so exactly `target` counts are redistributed — at least one survives the
    // zero-prune. Worth pinning because "the sampler emptied a cell" is the failure
    // a caller would notice last: the row simply looks unexpressed downstream.
    let idx: Vec<i32> = (0..40).collect();
    let dat: Vec<f32> = vec![25.0; 40];
    for row in 0..40u64 {
        let (i, d) = run(&idx, &dat, DownsampleMethod::Multinomial, 1, 3, 5, row);
        assert!(!i.is_empty(), "row {row} emptied at target=1");
        assert_eq!(lib(&d), 1, "row {row} did not conserve the target");
    }
}

// --------------------------------------------------------------------------
// Cross-repo golden
// --------------------------------------------------------------------------

#[test]
fn golden_vectors_are_stable() {
    // Executable cross-repo contract for the downsample primitive. Unlike
    // `encoder_crop_golden.json`, whose source of truth is state3's Python, the
    // source of truth here is *this* implementation — state3 adopts the fixture
    // and asserts against it, because the Rust RNG is the reference now.
    //
    // Regenerate with:
    //   cargo test -p scx-loader downsample -- --ignored regenerate_golden --nocapture
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/downsample_golden.json"
    );
    let raw =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read golden fixture {path}: {e}"));

    // The two repos' copies are kept byte-identical by discipline alone — each
    // side's test would otherwise pass happily against a *different* fixture.
    // Pin the digest so a one-sided regeneration fails loudly here.
    let digest = blake3::hash(raw.as_bytes()).to_hex().to_string();
    assert_eq!(
        &digest[..16],
        GOLDEN_BLAKE3_PREFIX,
        "downsample_golden.json changed. If deliberate: regenerate, copy verbatim into \
         state3/tests/data/, update GOLDEN_BLAKE3_PREFIX in BOTH repos, and bump \
         pyscx::COLLATE_CELLSET_CONTRACT_VERSION."
    );

    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    // Without this, adding a case to GOLDEN_CASES and forgetting to regenerate
    // leaves the new case silently untested — the loop below only ever sees what
    // is on disk.
    assert_eq!(
        doc["cases"].as_array().unwrap().len(),
        GOLDEN_CASES.len(),
        "GOLDEN_CASES and the fixture disagree on case count — regenerate the fixture"
    );
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let idx: Vec<i32> = case["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let dat: Vec<f32> = case["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let method = DownsampleMethod::parse(case["method"].as_str().unwrap()).unwrap();
        let (gi, gd) = run(
            &idx,
            &dat,
            method,
            case["target_library_size"].as_u64().unwrap(),
            case["seed"].as_u64().unwrap(),
            case["file_identity"].as_u64().unwrap(),
            case["row"].as_u64().unwrap(),
        );

        let exp_i: Vec<i32> = case["expected_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let exp_d: Vec<f32> = case["expected_counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(gi, exp_i, "indices mismatch [{name}]");
        assert_eq!(gd, exp_d, "counts mismatch [{name}]");
    }
}

/// blake3 prefix of `tests/data/downsample_golden.json`. Mirrored in state3's
/// `tests/test_downsample_golden.py`; both must be updated together.
const GOLDEN_BLAKE3_PREFIX: &str = "b33ed3d0e6d9aac5";

/// The golden cases, shared by the assertion above and the regenerator below so
/// the two cannot drift. `(name, indices, counts, method, target, seed, ident, row)`
type GoldenCase = (
    &'static str,
    &'static [i32],
    &'static [f32],
    &'static str,
    u64,
    u64,
    u64,
    u64,
);

/// Chosen to exercise each documented branch, not to be pretty: both samplers,
/// the `lib <= target` no-op, ties-to-even rounding, a clipped negative, an
/// all-zero row, an empty row, and a hard squeeze that forces pruning.
const GOLDEN_CASES: &[GoldenCase] = &[
    (
        "binomial_basic",
        &[1, 5, 9, 13, 17],
        &[10.0, 20.0, 30.0, 40.0, 50.0],
        "binomial",
        30,
        1234,
        0xA1B2_C3D4,
        7,
    ),
    (
        "multinomial_exact_target",
        &[1, 5, 9, 13, 17],
        &[10.0, 20.0, 30.0, 40.0, 50.0],
        "multinomial",
        30,
        1234,
        0xA1B2_C3D4,
        7,
    ),
    (
        "multinomial_hard_squeeze_prunes",
        &[0, 3, 6, 9, 12, 15, 18, 21],
        &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "multinomial",
        3,
        99,
        0x0BAD_F00D,
        0,
    ),
    (
        "below_target_is_a_noop_but_rints",
        &[2, 4],
        &[1.4, 2.6],
        "binomial",
        1000,
        7,
        0x1111,
        0,
    ),
    (
        "ties_round_to_even",
        &[2, 4, 6],
        &[2.5, 3.5, 4.5],
        "multinomial",
        1000,
        7,
        0x1111,
        0,
    ),
    (
        "negative_is_clipped_then_pruned",
        &[1, 2, 3],
        &[-4.0, 8.0, 12.0],
        "binomial",
        5,
        21,
        0x2222,
        3,
    ),
    (
        "all_zero_row",
        &[1, 2],
        &[0.0, 0.0],
        "multinomial",
        10,
        1,
        0x3333,
        0,
    ),
    ("empty_row", &[], &[], "binomial", 10, 1, 0x3333, 0),
    (
        "same_cell_different_seed",
        &[1, 5, 9, 13, 17],
        &[10.0, 20.0, 30.0, 40.0, 50.0],
        "multinomial",
        30,
        4321,
        0xA1B2_C3D4,
        7,
    ),
];

/// Regenerate `tests/data/downsample_golden.json` and print its blake3 prefix.
///
/// Ignored by default (it writes into the source tree). Run with:
///   `cargo test -p scx-loader regenerate_downsample_golden -- --ignored --nocapture`
/// then paste the printed prefix into `GOLDEN_BLAKE3_PREFIX` here **and** into
/// state3's mirror, and copy the file verbatim into `state3/tests/data/`.
#[test]
#[ignore = "writes into the source tree; run explicitly to regenerate"]
fn regenerate_downsample_golden() {
    let mut cases = Vec::new();
    for &(name, idx, dat, method_s, target, seed, ident, row) in GOLDEN_CASES {
        let method = DownsampleMethod::parse(method_s).unwrap();
        let (gi, gd) = run(idx, dat, method, target, seed, ident, row);
        cases.push(serde_json::json!({
            "name": name,
            "indices": idx,
            "counts": dat,
            "method": method_s,
            "target_library_size": target,
            "seed": seed,
            "file_identity": ident,
            "row": row,
            "expected_indices": gi,
            "expected_counts": gd,
        }));
    }
    let doc = serde_json::json!({
        "_comment": "Source of truth: scx-loader::downsample (Rust ChaCha8). Regenerate via \
                     `cargo test -p scx-loader regenerate_downsample_golden -- --ignored` and copy \
                     verbatim into state3/tests/data/. Keep both copies byte-identical, update \
                     GOLDEN_BLAKE3_PREFIX in both repos, and bump \
                     pyscx.COLLATE_CELLSET_CONTRACT_VERSION on any semantic change.",
        "cases": cases,
    });
    let text = serde_json::to_string_pretty(&doc).unwrap() + "\n";
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/downsample_golden.json"
    );
    std::fs::write(path, &text).unwrap();
    let digest = blake3::hash(text.as_bytes()).to_hex().to_string();
    println!("wrote {path} ({} cases)", GOLDEN_CASES.len());
    println!("GOLDEN_BLAKE3_PREFIX = \"{}\"", &digest[..16]);
}

// --------------------------------------------------------------------------
// Non-finite inputs — the clip's whole purpose is that the gather and the
// collate kernel agree on what a row contains, so it has to agree on NaN too.
// --------------------------------------------------------------------------

#[test]
fn clip_matches_the_kernels_max_semantics_on_nan() {
    // The kernel reads every count as `raw.max(0.0)`, and `f32::max` returns the
    // non-NaN operand — so the kernel sees 0 for a NaN. A `< 0.0` test (the
    // obvious way to write a clip) leaves NaN untouched, which would have left the
    // gather emitting NaN while the kernel computed with 0.
    let mut d = vec![f32::NAN, -1.0, 2.0];
    clip_negatives(&mut d);
    assert_eq!(d[0], 0.0, "NaN must clip to 0, as `raw.max(0.0)` does");
    assert_eq!(d[1], 0.0);
    assert_eq!(d[2], 2.0);
    // Premise: this is genuinely what the kernel's own expression yields.
    assert_eq!(f32::NAN.max(0.0), 0.0);
}

#[test]
fn non_finite_counts_are_treated_as_zero_not_saturated() {
    // `inf as u64` saturates to u64::MAX in Rust, which would hand the sampler a
    // nonsense trial count and quietly produce garbage. Treat it as corrupt input
    // worth 0 instead.
    let (i, d) = run(
        &[1, 2, 3],
        &[f32::INFINITY, f32::NAN, 6.0],
        DownsampleMethod::Multinomial,
        3,
        1,
        1,
        0,
    );
    assert_eq!(i, vec![3], "only the finite entry should survive: {i:?}");
    assert_eq!(d, vec![3.0]);
}

#[test]
fn negative_infinity_clips_like_any_negative() {
    let (i, d) = run(
        &[1, 2],
        &[f32::NEG_INFINITY, 8.0],
        DownsampleMethod::Multinomial,
        4,
        1,
        1,
        0,
    );
    assert_eq!(i, vec![2]);
    assert_eq!(d, vec![4.0]);
}
