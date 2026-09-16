//! Tests for the neighbourhood plan builders.
//!
//! The coordinate builder is checked against an O(n²) brute-force reference
//! that applies the *same* declared tie rule — (squared distance ascending,
//! then row ascending) — and computes each distance with the same summation
//! order, so equality is exact rather than tolerance-bounded.

use super::*;
use scx_format_io::PairwiseRows;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cfg(include_center: bool) -> NeighborhoodConfig {
    NeighborhoodConfig {
        include_center,
        file_id: 0,
    }
}

/// Build a `PairwiseRows` covering `[0, n)` from `(row, col, weight)` triples.
fn rows_from(triples: &[(u64, u64, f32)], n: usize) -> PairwiseRows {
    let mut per_row: Vec<Vec<(i64, f32)>> = vec![Vec::new(); n];
    for &(r, c, v) in triples {
        per_row[r as usize].push((c as i64, v));
    }
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in per_row.iter_mut() {
        row.sort_by_key(|&(c, _)| c);
        for &(c, v) in row.iter() {
            indices.push(c);
            data.push(v);
        }
        indptr.push(indices.len() as i64);
    }
    PairwiseRows {
        indptr,
        indices,
        data,
        row_start: 0,
        n_rows: n as u64,
        n_cols: n as u64,
    }
}

/// A ring graph: `r` is joined to `r+1` (weight 2.0) and `r+2` (weight 1.0).
fn ring(n: usize) -> Vec<(u64, u64, f32)> {
    let mut out = Vec::new();
    for r in 0..n as u64 {
        out.push((r, (r + 1) % n as u64, 2.0));
        out.push((r, (r + 2) % n as u64, 1.0));
    }
    out
}

fn set_rows(p: &SparseCellSetPlan) -> Vec<u64> {
    p.rows.clone()
}

/// Deterministic pseudo-random points — a small LCG rather than a dependency,
/// so the fixture is identical on every machine.
fn points(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Quantised so exact ties occur naturally, which is the case the tie
        // rule exists for.
        out.push(((x >> 33) % 64) as f32);
    }
    out
}

fn d2(coords: &[f32], d: usize, a: u64, b: u64) -> f32 {
    let mut s = 0.0f32;
    for j in 0..d {
        let diff = coords[a as usize * d + j] - coords[b as usize * d + j];
        s += diff * diff;
    }
    s
}

/// Brute-force reference: every kept point, same tie rule, same distance
/// summation order.
fn reference_neighbors(
    coords: &[f32],
    d: usize,
    keep: Option<&[bool]>,
    center: u64,
    query: CoordQuery,
) -> Vec<u64> {
    let n = coords.len() / d;
    let mut cand: Vec<(f32, u64)> = (0..n as u64)
        .filter(|&r| r != center && keep.is_none_or(|m| m[r as usize]))
        .map(|r| (d2(coords, d, center, r), r))
        .collect();
    if let CoordQuery::Radius(rad) = query {
        cand.retain(|&(dd, _)| dd <= rad * rad);
    }
    // Deliberately NOT `cmp_candidate`: a reference that reuses the subject's
    // comparator cannot falsify a change to it. Spelled out from the declared
    // rule instead — squared distance ascending, then row ascending.
    cand.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    if let CoordQuery::Knn(k) = query {
        cand.truncate(k);
    }
    cand.into_iter().map(|(_, r)| r).collect()
}

fn assert_coords_match_reference(
    coords: &[f32],
    d: usize,
    keep: Option<&[bool]>,
    query: CoordQuery,
) {
    let built = plans_from_coords(coords, d, keep, query, cfg(true)).unwrap();
    let n = coords.len() / d;
    let expected_centers: Vec<u64> = (0..n as u64)
        .filter(|&r| keep.is_none_or(|m| m[r as usize]))
        .collect();
    assert_eq!(built.centers, expected_centers);
    for (plan, &center) in built.plans.iter().zip(&built.centers) {
        let want = reference_neighbors(coords, d, keep, center, query);
        let got = set_rows(plan);
        assert_eq!(got[0], center, "centre must be position 0");
        assert_eq!(&got[1..], &want[..], "centre {center}");
        assert_eq!(plan.role_tags[0], 0, "the centre's role tag is 0");
        assert!(plan.role_tags[1..].iter().all(|&t| t == 1));
        assert_eq!(plan.set_offsets, vec![0, got.len() as i64]);
        assert!(plan.file_ids.iter().all(|&f| f == 0));
    }
}

// ---------------------------------------------------------------------------
// Step 1 — graph-driven
// ---------------------------------------------------------------------------

#[test]
fn the_role_tag_values_are_pinned() {
    // These two numbers are the contract with the consumer, not an internal
    // encoding: a set's centre is 0 and a neighbour is 1. Asserting through the
    // constants would pass however they were defined.
    assert_eq!(ROLE_CENTER, 0);
    assert_eq!(ROLE_NEIGHBOR, 1);
}

#[test]
fn graph_plans_carry_every_stored_edge_in_column_order() {
    let n = 8;
    let rows = rows_from(&ring(n), n);
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut out);
    assert_eq!(out.len(), n);
    for (i, plan) in out.plans.iter().enumerate() {
        let c = i as u64;
        let mut want = vec![c];
        let mut nbrs = vec![(c + 1) % n as u64, (c + 2) % n as u64];
        nbrs.sort();
        want.extend(nbrs);
        assert_eq!(set_rows(plan), want, "centre {c}");
        assert_eq!(
            plan.role_tags,
            vec![0, 1, 1],
            "centre first, then neighbours"
        );
    }
}

#[test]
fn graph_top_k_by_weight_breaks_ties_by_column_ascending() {
    // Row 0's edges all weigh the same, so only the declared tie rule decides
    // which two of {1, 2, 3, 4} survive.
    let triples = vec![(0u64, 4u64, 5.0f32), (0, 3, 5.0), (0, 2, 5.0), (0, 1, 5.0)];
    let rows = rows_from(&triples, 5);
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, Some(2), WeightOrder::Desc, cfg(true), &mut out);
    assert_eq!(set_rows(&out.plans[0]), vec![0, 1, 2]);
}

#[test]
fn graph_top_k_prefers_the_heavier_edge() {
    // Distinct weights: the rule must be weight-descending, not column order.
    let triples = vec![(0u64, 1u64, 0.1f32), (0, 2, 0.9), (0, 3, 0.5)];
    let rows = rows_from(&triples, 4);
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, Some(2), WeightOrder::Desc, cfg(true), &mut out);
    // Chosen by weight (2 then 3), emitted in ascending row order.
    assert_eq!(set_rows(&out.plans[0]), vec![0, 2, 3]);
}

#[test]
fn a_deleted_centre_yields_no_set() {
    let n = 6;
    let rows = rows_from(&ring(n), n);
    let mut keep = vec![true; n];
    keep[2] = false;
    keep[4] = false;
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &rows,
        Some(&keep),
        None,
        WeightOrder::Desc,
        cfg(true),
        &mut out,
    );
    assert_eq!(out.centers, vec![0, 1, 3, 5]);
    assert_eq!(out.len(), 4);
}

#[test]
fn a_deleted_neighbour_is_dropped_from_every_set() {
    let n = 6;
    let rows = rows_from(&ring(n), n);
    let mut keep = vec![true; n];
    keep[3] = false;
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &rows,
        Some(&keep),
        None,
        WeightOrder::Desc,
        cfg(true),
        &mut out,
    );
    for plan in &out.plans {
        assert!(!plan.rows.contains(&3), "row 3 survived in {:?}", plan.rows);
    }
    // Centre 1 is joined to 2 and 3; only 2 survives, so the set is short, not
    // backfilled.
    let p1 = &out.plans[out.centers.iter().position(|&c| c == 1).unwrap()];
    assert_eq!(set_rows(p1), vec![1, 2]);
}

#[test]
fn a_self_loop_does_not_duplicate_the_centre() {
    let triples = vec![(0u64, 0u64, 1.0f32), (0, 1, 1.0)];
    let rows = rows_from(&triples, 3);
    let mut with = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut with);
    assert_eq!(set_rows(&with.plans[0]), vec![0, 1]);

    // Without the centre, the self-loop IS an ordinary edge and stays — the
    // de-dup is about not emitting the same row twice, not about self-loops.
    let mut without = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &rows,
        None,
        None,
        WeightOrder::Desc,
        cfg(false),
        &mut without,
    );
    assert_eq!(set_rows(&without.plans[0]), vec![0, 1]);
    assert!(without.plans[0].role_tags.iter().all(|&t| t == 1));
}

#[test]
fn an_isolated_centre_is_a_singleton_or_empty() {
    let rows = rows_from(&[(1u64, 2u64, 1.0f32)], 3);
    let mut with = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut with);
    assert_eq!(set_rows(&with.plans[0]), vec![0]);
    assert_eq!(with.plans[0].set_offsets, vec![0, 1]);

    let mut without = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &rows,
        None,
        None,
        WeightOrder::Desc,
        cfg(false),
        &mut without,
    );
    assert!(without.plans[0].rows.is_empty());
    assert_eq!(without.plans[0].set_offsets, vec![0, 0]);
}

#[test]
fn chunked_graph_builds_equal_a_single_whole_range_build() {
    let n = 12;
    let triples = ring(n);
    let whole = rows_from(&triples, n);
    let mut single = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &whole,
        None,
        None,
        WeightOrder::Desc,
        cfg(true),
        &mut single,
    );

    // Same graph, read as three row ranges — the shape the streaming driver
    // produces.
    let mut chunked = NeighborhoodPlans::default();
    for (start, end) in [(0usize, 5usize), (5, 9), (9, 12)] {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in start..end {
            let lo = whole.indptr[r] as usize;
            let hi = whole.indptr[r + 1] as usize;
            indices.extend_from_slice(&whole.indices[lo..hi]);
            data.extend_from_slice(&whole.data[lo..hi]);
            indptr.push(indices.len() as i64);
        }
        let chunk = PairwiseRows {
            indptr,
            indices,
            data,
            row_start: start as u64,
            n_rows: (end - start) as u64,
            n_cols: n as u64,
        };
        plans_from_graph_chunk(
            &chunk,
            None,
            None,
            WeightOrder::Desc,
            cfg(true),
            &mut chunked,
        );
    }
    assert_eq!(chunked.centers, single.centers);
    assert_eq!(chunked.plans.len(), single.plans.len());
    for (a, b) in chunked.plans.iter().zip(&single.plans) {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.role_tags, b.role_tags);
        assert_eq!(a.set_offsets, b.set_offsets);
    }
}

// ---------------------------------------------------------------------------
// Step 2 — coordinate-driven
// ---------------------------------------------------------------------------

#[test]
fn coords_knn_matches_brute_force() {
    for (n, k) in [(50usize, 4usize), (200, 8), (200, 1), (37, 12)] {
        let coords = points(n, 2, n as u64 * 7 + k as u64);
        assert_coords_match_reference(&coords, 2, None, CoordQuery::Knn(k));
    }
}

#[test]
fn coords_radius_matches_brute_force() {
    for radius in [3.0f32, 10.0, 40.0] {
        let coords = points(150, 2, radius as u64 + 3);
        assert_coords_match_reference(&coords, 2, None, CoordQuery::Radius(radius));
    }
}

#[test]
fn coords_ties_break_by_row_ascending() {
    // Four points exactly one unit from the centre, in descending row order in
    // the array. Only the declared tie rule fixes which two `k = 2` takes.
    let coords: Vec<f32> = vec![
        0.0, 0.0, // 0: the centre
        1.0, 0.0, // 1
        -1.0, 0.0, // 2
        0.0, 1.0, // 3
        0.0, -1.0, // 4
    ];
    let built = plans_from_coords(&coords, 2, None, CoordQuery::Knn(2), cfg(true)).unwrap();
    assert_eq!(set_rows(&built.plans[0]), vec![0, 1, 2]);
    assert_coords_match_reference(&coords, 2, None, CoordQuery::Knn(2));
}

#[test]
fn coords_three_dimensional_matches_brute_force() {
    let coords = points(120, 3, 99);
    assert_coords_match_reference(&coords, 3, None, CoordQuery::Knn(5));
    assert_coords_match_reference(&coords, 3, None, CoordQuery::Radius(12.0));
}

#[test]
fn coords_deletions_drop_centres_and_neighbours() {
    let n = 80;
    let coords = points(n, 2, 11);
    let keep: Vec<bool> = (0..n).map(|i| i % 5 != 0).collect();
    assert_coords_match_reference(&coords, 2, Some(&keep), CoordQuery::Knn(6));
    assert_coords_match_reference(&coords, 2, Some(&keep), CoordQuery::Radius(9.0));
    let built = plans_from_coords(&coords, 2, Some(&keep), CoordQuery::Knn(6), cfg(true)).unwrap();
    for plan in &built.plans {
        assert!(plan.rows.iter().all(|&r| keep[r as usize]));
    }
}

#[test]
fn k_greater_than_the_available_points_returns_a_short_set() {
    let coords: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let built = plans_from_coords(&coords, 2, None, CoordQuery::Knn(50), cfg(true)).unwrap();
    // Three points: every set is the centre plus the other two. No backfill.
    for plan in &built.plans {
        assert_eq!(plan.rows.len(), 3);
    }
}

#[test]
fn a_single_point_file_has_an_empty_neighbourhood() {
    let coords: Vec<f32> = vec![4.0, 7.0];
    let built = plans_from_coords(&coords, 2, None, CoordQuery::Knn(3), cfg(true)).unwrap();
    assert_eq!(built.len(), 1);
    assert_eq!(set_rows(&built.plans[0]), vec![0]);
    let bare = plans_from_coords(&coords, 2, None, CoordQuery::Knn(3), cfg(false)).unwrap();
    assert!(bare.plans[0].rows.is_empty());
}

#[test]
fn every_point_identical_still_answers() {
    // The grid collapses to one cell; the answer is brute force, which is
    // correct but is also what the benchmark arm asserts it is not timing.
    let coords: Vec<f32> = vec![5.0; 2 * 10];
    assert_coords_match_reference(&coords, 2, None, CoordQuery::Knn(3));
    assert_coords_match_reference(&coords, 2, None, CoordQuery::Radius(1.0));
}

#[test]
fn a_collinear_layout_still_answers() {
    // Zero extent on one axis: the grid is 1 cell wide in y.
    let coords: Vec<f32> = (0..40).flat_map(|i| [i as f32, 3.0]).collect();
    assert_coords_match_reference(&coords, 2, None, CoordQuery::Knn(4));
    assert_coords_match_reference(&coords, 2, None, CoordQuery::Radius(2.5));
}

#[test]
fn all_points_deleted_builds_nothing() {
    let coords = points(20, 2, 5);
    let keep = vec![false; 20];
    let built = plans_from_coords(&coords, 2, Some(&keep), CoordQuery::Knn(3), cfg(true)).unwrap();
    assert!(built.is_empty());
}

#[test]
fn non_finite_coordinates_are_refused() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let coords: Vec<f32> = vec![0.0, 0.0, 1.0, bad, 2.0, 2.0];
        let err = plans_from_coords(&coords, 2, None, CoordQuery::Knn(1), cfg(true))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no grid cell"), "{err}");
    }
    // Unless the offending point is deleted, in which case it is not a point.
    let coords: Vec<f32> = vec![0.0, 0.0, 1.0, f32::NAN, 2.0, 2.0];
    let keep = vec![true, false, true];
    assert!(plans_from_coords(&coords, 2, Some(&keep), CoordQuery::Knn(1), cfg(true)).is_ok());
}

#[test]
fn malformed_coordinate_arguments_are_refused() {
    let coords: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];
    for (d, q, needle) in [
        (0usize, CoordQuery::Knn(1), "dimensionality"),
        (3, CoordQuery::Knn(1), "not a multiple"),
        (2, CoordQuery::Knn(0), "k must be >= 1"),
        (2, CoordQuery::Radius(0.0), "radius must be finite"),
        (2, CoordQuery::Radius(f32::NAN), "radius must be finite"),
    ] {
        let err = plans_from_coords(&coords, d, None, q, cfg(true))
            .unwrap_err()
            .to_string();
        assert!(err.contains(needle), "{needle} not in {err}");
    }
    let err = plans_from_coords(&coords, 2, Some(&[true]), CoordQuery::Knn(1), cfg(true))
        .unwrap_err()
        .to_string();
    assert!(err.contains("keep mask"), "{err}");
}

// ---------------------------------------------------------------------------
// Plan shape and batching
// ---------------------------------------------------------------------------

#[test]
fn set_offsets_is_a_valid_indptr_for_collate() {
    let n = 10;
    let rows = rows_from(&ring(n), n);
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut out);
    let batched = batch_plans(&out.plans, 4, None).unwrap();
    for plan in batched.iter().chain(out.plans.iter()) {
        assert_eq!(plan.set_offsets[0], 0);
        assert_eq!(
            *plan.set_offsets.last().unwrap() as usize,
            plan.rows.len(),
            "set_offsets must end at total_rows"
        );
        assert!(plan.set_offsets.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(plan.file_ids.len(), plan.rows.len());
        assert_eq!(plan.role_tags.len(), plan.rows.len());
    }
}

#[test]
fn batch_plans_rebases_offsets_and_preserves_order() {
    let n = 9;
    let rows = rows_from(&ring(n), n);
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut out);
    let batched = batch_plans(&out.plans, 4, None).unwrap();
    assert_eq!(batched.len(), 3, "the last batch is short, not dropped");
    assert_eq!(batched[2].set_offsets.len(), 2, "one set in the tail batch");

    // Concatenating the batches must reproduce the input rows exactly, in order.
    let flat: Vec<u64> = batched.iter().flat_map(|p| p.rows.clone()).collect();
    let expect: Vec<u64> = out.plans.iter().flat_map(|p| p.rows.clone()).collect();
    assert_eq!(flat, expect);

    // And each batch's offsets must delimit its own sets.
    for b in &batched {
        for w in b.set_offsets.windows(2) {
            assert!(w[1] >= w[0]);
        }
        assert_eq!(*b.set_offsets.last().unwrap() as usize, b.rows.len());
    }
}

#[test]
fn batch_plans_shuffle_is_seed_reproducible_and_seed_sensitive() {
    let n = 16;
    let rows = rows_from(&ring(n), n);
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut out);

    let a = batch_plans(&out.plans, 4, Some(7)).unwrap();
    let b = batch_plans(&out.plans, 4, Some(7)).unwrap();
    let c = batch_plans(&out.plans, 4, Some(8)).unwrap();
    let flat = |v: &Vec<SparseCellSetPlan>| -> Vec<u64> {
        v.iter().flat_map(|p| p.rows.clone()).collect()
    };
    assert_eq!(flat(&a), flat(&b), "same seed, same order");
    assert_ne!(flat(&a), flat(&c), "a different seed must reorder");

    // A shuffle reorders sets, never a set's members.
    let mut seen: Vec<Vec<u64>> = Vec::new();
    for batch in &a {
        for w in batch.set_offsets.windows(2) {
            seen.push(batch.rows[w[0] as usize..w[1] as usize].to_vec());
        }
    }
    let mut want: Vec<Vec<u64>> = out.plans.iter().map(|p| p.rows.clone()).collect();
    seen.sort();
    want.sort();
    assert_eq!(seen, want);
}

#[test]
fn batch_plans_refuses_a_multi_set_plan_so_sets_per_batch_means_sets() {
    // `sets_per_batch` used to chunk *plans*, so on a multi-set input it
    // silently meant something else: re-batching an already-batched list at 2
    // produced outputs of six sets each. A knob that means two things
    // depending on its input is worse than one that refuses, and nothing
    // needed the multi-set mode.
    let n = 9;
    let rows = rows_from(&ring(n), n);
    let mut singles = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &rows,
        None,
        None,
        WeightOrder::Desc,
        cfg(true),
        &mut singles,
    );

    let once = batch_plans(&singles.plans, 3, None).unwrap();
    assert_eq!(once.len(), 3);
    for b in &once {
        assert_eq!(b.set_offsets.len() - 1, 3, "three sets per batch, as asked");
    }

    let err = batch_plans(&once, 2, None).unwrap_err().to_string();
    assert!(err.contains("not a single set"), "{err}");
    assert!(err.contains("sets_per_batch counts sets"), "{err}");
}

#[test]
fn batch_plans_refuses_a_zero_batch_size() {
    let err = batch_plans(&[], 0, None).unwrap_err().to_string();
    assert!(err.contains("sets_per_batch"), "{err}");
}

#[test]
fn batch_plans_refuses_a_plan_that_is_not_exactly_one_set() {
    // Every input must be `[0, rows.len()]`. That is what makes the rebase
    // correct — it appends exactly one offset per input — and what makes
    // `sets_per_batch` count sets rather than plans.
    let good = SparseCellSetPlan {
        file_ids: vec![0; 3],
        rows: vec![1, 2, 3],
        role_tags: vec![0, 1, 1],
        set_offsets: vec![0, 3],
    };
    assert!(batch_plans(std::slice::from_ref(&good), 2, None).is_ok());

    for (bad, needle) in [
        (
            SparseCellSetPlan {
                set_offsets: vec![1, 3],
                ..good.clone()
            },
            "not a single set",
        ),
        (
            SparseCellSetPlan {
                set_offsets: vec![0, 2],
                ..good.clone()
            },
            "not a single set",
        ),
        (
            SparseCellSetPlan {
                set_offsets: vec![0, 2, 3],
                ..good.clone()
            },
            "not a single set",
        ),
        (
            SparseCellSetPlan {
                role_tags: vec![0, 1],
                ..good.clone()
            },
            "role_tags",
        ),
    ] {
        let err = batch_plans(&[bad], 2, None).unwrap_err().to_string();
        assert!(err.contains(needle), "{needle} not in {err}");
    }
}

#[test]
fn file_id_is_stamped_on_every_row() {
    let n = 5;
    let rows = rows_from(&ring(n), n);
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &rows,
        None,
        None,
        WeightOrder::Desc,
        NeighborhoodConfig {
            include_center: true,
            file_id: 3,
        },
        &mut out,
    );
    assert!(out.plans.iter().all(|p| p.file_ids.iter().all(|&f| f == 3)));
}

// ---------------------------------------------------------------------------
// Drivers — against a real on-disk spatial fixture
// ---------------------------------------------------------------------------

mod fixture {
    use arrow::array::{Float32Array, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// A small synthetic spatial file: a `g x g` lattice of cells with
    /// `obsm["spatial"]` coordinates and an `obsp["connectivities"]` graph
    /// joining each cell to its right and lower lattice neighbours.
    ///
    /// Built at run time rather than committed: `scx-loader/tests/data/` holds
    /// JSON goldens only, and a binary fixture there would be the first.
    pub struct Spatial {
        pub path: std::path::PathBuf,
        pub n: usize,
        pub coords: Vec<f32>,
    }

    pub fn obs_batch(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("spot_{i}")).collect();
        arrow::array::RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cell_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    pub fn var_batch(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        arrow::array::RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "gene_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    /// `obsm["spatial"]` as **Int64**, which is what scanpy writes for a real
    /// Visium sample — so the narrowing path is what the tests exercise.
    fn coord_batch(coords: &[f32], start: usize, n: usize) -> arrow::array::RecordBatch {
        let xs: Vec<i64> = (start..start + n).map(|r| coords[r * 2] as i64).collect();
        let ys: Vec<i64> = (start..start + n)
            .map(|r| coords[r * 2 + 1] as i64)
            .collect();
        arrow::array::RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("0", DataType::Int64, false),
                Field::new("1", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(xs)),
                Arc::new(Int64Array::from(ys)),
            ],
        )
        .unwrap()
    }

    fn coo(triples: &[(u64, u64, f32)], n: usize) -> arrow::array::RecordBatch {
        let schema = Schema::new(vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ])
        .with_metadata(HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]));
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from(
                    triples.iter().map(|t| t.0 as i32).collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    triples.iter().map(|t| t.1 as i32).collect::<Vec<_>>(),
                )),
                Arc::new(Float32Array::from(
                    triples.iter().map(|t| t.2).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    /// The same lattice plus a deletion vector over `deleted` physical rows —
    /// so the drivers' `drop_deleted` has something to act on. Written through
    /// the section writer rather than through `scx-ops::mark_deleted`, which
    /// `scx-loader` does not depend on.
    pub fn write_with_deletions(
        dir: &tempfile::TempDir,
        g: usize,
        n_shards: usize,
        deleted: &[u32],
    ) -> Spatial {
        write_inner(dir, g, n_shards, true, Some(deleted), "spatial_deleted.scx")
    }

    /// `sharded = false` writes the legacy single-section forms — the shape
    /// `scx sort` re-emits, which is why it has to be covered.
    pub fn write(dir: &tempfile::TempDir, g: usize, n_shards: usize, sharded: bool) -> Spatial {
        let name = if sharded {
            "spatial_sharded.scx"
        } else {
            "spatial_legacy.scx"
        };
        write_inner(dir, g, n_shards, sharded, None, name)
    }

    fn write_inner(
        dir: &tempfile::TempDir,
        g: usize,
        n_shards: usize,
        sharded: bool,
        deleted: Option<&[u32]>,
        file_name: &str,
    ) -> Spatial {
        let n = g * g;
        let coords: Vec<f32> = (0..n)
            .flat_map(|i| [(i % g) as f32 * 10.0, (i / g) as f32 * 10.0])
            .collect();
        let mut edges: Vec<(u64, u64, f32)> = Vec::new();
        for i in 0..n {
            let (x, y) = (i % g, i / g);
            if x + 1 < g {
                edges.push((i as u64, (i + 1) as u64, 2.0));
                edges.push(((i + 1) as u64, i as u64, 2.0));
            }
            if y + 1 < g {
                edges.push((i as u64, (i + g) as u64, 1.0));
                edges.push(((i + g) as u64, i as u64, 1.0));
            }
        }
        edges.sort_by_key(|&(r, c, _)| (r, c));

        let path = dir.path().join(file_name);
        let mut w = ScxWriter::new(
            &path,
            FileHeader::new_single_modality(n as u64, 4, 0, 16384, 0, 0),
        )
        .unwrap();
        w.write_obs(&obs_batch(n)).unwrap();
        w.write_var(&var_batch(4)).unwrap();
        if sharded {
            let per = n.div_ceil(n_shards);
            for s in 0..n_shards {
                let start = s * per;
                let rows = per.min(n.saturating_sub(start));
                w.write_obsm_shard(
                    "spatial",
                    s as u32,
                    start as u64,
                    rows as u64,
                    n as u64,
                    &coord_batch(&coords, start, rows),
                )
                .unwrap();
                let in_shard: Vec<_> = edges
                    .iter()
                    .copied()
                    .filter(|t| (t.0 as usize) >= start && (t.0 as usize) < start + rows)
                    .collect();
                w.write_obsp_shard_coo(
                    "connectivities",
                    s as u32,
                    start as u64,
                    rows as u64,
                    n as u64,
                    &coo(&in_shard, n),
                )
                .unwrap();
            }
        } else {
            w.write_obsm("spatial", &coord_batch(&coords, 0, n))
                .unwrap();
            w.write_obsp("connectivities", &coo(&edges, n)).unwrap();
        }
        if let Some(rows) = deleted {
            let mut dv = scx_format_io::deletion_vectors::DeletionVectors::new();
            let mut bitmap = roaring::RoaringBitmap::new();
            for &r in rows {
                bitmap.insert(r);
            }
            dv.deletions.insert(0, bitmap);
            w.write_deletion_vectors(&dv).unwrap();
        }
        w.finish().unwrap();
        Spatial { path, n, coords }
    }
}

#[test]
fn the_graph_driver_reads_the_same_plans_however_it_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture::write(&dir, 7, 4, true);
    let whole = build_graph_plans(
        &fx.path,
        "connectivities",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        4096,
    )
    .unwrap();
    assert_eq!(whole.len(), fx.n);
    for chunk in [1u64, 2, 3, 7, 13] {
        let got = build_graph_plans(
            &fx.path,
            "connectivities",
            false,
            None,
            WeightOrder::Desc,
            cfg(true),
            chunk,
        )
        .unwrap();
        assert_eq!(got.centers, whole.centers, "chunk_rows={chunk}");
        for (a, b) in got.plans.iter().zip(&whole.plans) {
            assert_eq!(a.rows, b.rows, "chunk_rows={chunk}");
        }
    }
    // And the plans are the lattice: an interior spot has four neighbours.
    let interior = whole.plans[fixture_interior_index(7)].rows.len();
    assert_eq!(interior, 5, "centre plus four lattice neighbours");
}

fn fixture_interior_index(g: usize) -> usize {
    (g / 2) * g + g / 2
}

#[test]
fn the_graph_driver_reads_a_legacy_unsharded_file() {
    // `scx sort` re-emits obsp through `write_obsp`, so a sorted file always
    // takes this branch.
    let dir = tempfile::tempdir().unwrap();
    let sharded = fixture::write(&dir, 6, 3, true);
    let legacy = fixture::write(&dir, 6, 3, false);
    let a = build_graph_plans(
        &sharded.path,
        "connectivities",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        8,
    )
    .unwrap();
    let b = build_graph_plans(
        &legacy.path,
        "connectivities",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        8,
    )
    .unwrap();
    assert_eq!(a.centers, b.centers);
    for (x, y) in a.plans.iter().zip(&b.plans) {
        assert_eq!(x.rows, y.rows);
        assert_eq!(x.role_tags, y.role_tags);
    }
}

#[test]
fn the_coordinate_driver_narrows_int64_coordinates_and_matches_the_reference() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture::write(&dir, 8, 4, true);
    let (coords, d) = read_coords(&fx.path, "spatial").unwrap();
    assert_eq!(d, 2);
    assert_eq!(
        coords, fx.coords,
        "int64 pixel coordinates must narrow exactly"
    );

    let built =
        build_coord_plans(&fx.path, "spatial", false, CoordQuery::Knn(4), cfg(true)).unwrap();
    assert_eq!(built.len(), fx.n);
    assert_coords_match_reference(&fx.coords, 2, None, CoordQuery::Knn(4));

    // On a lattice of pitch 10, radius 10 is exactly the four rook neighbours
    // of an interior spot — an equality the grid must not round.
    let radius = build_coord_plans(
        &fx.path,
        "spatial",
        false,
        CoordQuery::Radius(10.0),
        cfg(true),
    )
    .unwrap();
    assert_eq!(radius.plans[fixture_interior_index(8)].rows.len(), 5);
}

#[test]
fn the_two_builders_agree_on_a_lattice_where_they_must() {
    // The stored graph IS the rook adjacency, and radius 10 on a pitch-10
    // lattice is the same relation. Two independent paths, one answer.
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture::write(&dir, 6, 2, true);
    let g = build_graph_plans(
        &fx.path,
        "connectivities",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        16,
    )
    .unwrap();
    let c = build_coord_plans(
        &fx.path,
        "spatial",
        false,
        CoordQuery::Radius(10.0),
        cfg(true),
    )
    .unwrap();
    assert_eq!(g.centers, c.centers);
    for (a, b) in g.plans.iter().zip(&c.plans) {
        assert_eq!(a.rows, b.rows, "graph and coordinate plans disagree");
    }
}

#[test]
fn the_drivers_read_the_file_s_own_deletion_vectors() {
    // `drop_deleted` goes IN and the mask is read off the reader the driver
    // already opened — a caller-supplied mask (or a second `ScxReader::open`
    // for it) is two snapshots of one path that a replacement between them can
    // put out of step. So the coverage that matters here is that the two
    // settings actually differ on a file that HAS deletions, and that they
    // agree on one that does not.
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture::write_with_deletions(&dir, 5, 2, &[0, 7, 12]);

    let dropped = build_graph_plans(
        &fx.path,
        "connectivities",
        true,
        None,
        WeightOrder::Desc,
        cfg(true),
        8,
    )
    .unwrap();
    let kept_all = build_graph_plans(
        &fx.path,
        "connectivities",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        8,
    )
    .unwrap();
    assert_eq!(kept_all.len(), fx.n, "nothing dropped without drop_deleted");
    assert_eq!(dropped.len(), fx.n - 3);
    assert!(!dropped.centers.iter().any(|&c| [0u64, 7, 12].contains(&c)));
    for plan in &dropped.plans {
        assert!(!plan.rows.iter().any(|r| [0u64, 7, 12].contains(r)));
    }
    // The deleted rows ARE still reachable when the caller says so.
    assert!(kept_all.plans.iter().any(|p| p.rows.contains(&7)));

    let c_dropped =
        build_coord_plans(&fx.path, "spatial", true, CoordQuery::Knn(3), cfg(true)).unwrap();
    assert_eq!(c_dropped.centers, dropped.centers);
    for plan in &c_dropped.plans {
        assert!(!plan.rows.iter().any(|r| [0u64, 7, 12].contains(r)));
    }

    // A file with no deletions: both settings must agree exactly.
    let clean = fixture::write(&dir, 5, 2, true);
    let a = build_graph_plans(
        &clean.path,
        "connectivities",
        true,
        None,
        WeightOrder::Desc,
        cfg(true),
        8,
    )
    .unwrap();
    let b = build_graph_plans(
        &clean.path,
        "connectivities",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        8,
    )
    .unwrap();
    assert_eq!(a.centers, b.centers);
    for (x, y) in a.plans.iter().zip(&b.plans) {
        assert_eq!(x.rows, y.rows);
    }
}

#[test]
fn the_graph_driver_refuses_a_zero_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture::write(&dir, 3, 1, true);
    let err = build_graph_plans(
        &fx.path,
        "connectivities",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        0,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("chunk_rows"), "{err}");
}

#[test]
fn a_missing_key_names_the_section_rather_than_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture::write(&dir, 3, 1, true);
    assert!(build_graph_plans(
        &fx.path,
        "distances",
        false,
        None,
        WeightOrder::Desc,
        cfg(true),
        8
    )
    .is_err());
    assert!(read_coords(&fx.path, "X_umap").is_err());
}

// ---------------------------------------------------------------------------
// Review round 1 — the defects the reviewers found
// ---------------------------------------------------------------------------

#[test]
fn a_wide_embedding_is_refused_rather_than_searched_forever() {
    // `walk_ring` enumerates the (2r+1)^d offsets of a Chebyshev shell and
    // rejects the out-of-range ones at the leaf, so d sits in an exponent. At
    // d = 50 — an ordinary `obsm["X_pca"]`, and a plausible slip for
    // `obsm["spatial"]` — ring 1 is 3^50 visits and the call never returns.
    // This test asserts the refusal; it deliberately does NOT call the builder
    // at d = 50, because a regression here would hang the suite rather than
    // fail it.
    assert_eq!(MAX_COORD_DIMS, 3);
    for d in [4usize, 8, 50] {
        let coords = vec![0.0f32; 4 * d];
        let err = plans_from_coords(&coords, d, None, CoordQuery::Knn(1), cfg(true))
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be 1..=3"), "d={d}: {err}");
        assert!(
            err.contains("graph builder"),
            "the error must name the way out: {err}"
        );
    }
    // 1, 2 and 3 still work.
    for d in [1usize, 2, 3] {
        let coords: Vec<f32> = (0..(6 * d)).map(|i| i as f32).collect();
        assert!(plans_from_coords(&coords, d, None, CoordQuery::Knn(2), cfg(true)).is_ok());
    }
}

#[test]
fn weight_order_decides_which_end_of_a_distance_graph_k_keeps() {
    // The failure this exists for: `obsp["connectivities"]` is an affinity
    // (larger = closer) and `obsp["distances"]` is a metric (larger = farther),
    // so one `weight_order` against the other graph returns each cell's k
    // FARTHEST neighbours — structurally valid, and the opposite question.
    let triples = vec![
        (0u64, 1u64, 1.0f32), // nearest under a distance reading
        (0, 2, 5.0),
        (0, 3, 9.0), // farthest under a distance reading
    ];
    let rows = rows_from(&triples, 4);

    let mut desc = NeighborhoodPlans::default();
    plans_from_graph_chunk(
        &rows,
        None,
        Some(2),
        WeightOrder::Desc,
        cfg(true),
        &mut desc,
    );
    assert_eq!(
        set_rows(&desc.plans[0]),
        vec![0, 2, 3],
        "desc keeps the heaviest"
    );

    let mut asc = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, Some(2), WeightOrder::Asc, cfg(true), &mut asc);
    assert_eq!(
        set_rows(&asc.plans[0]),
        vec![0, 1, 2],
        "asc keeps the lightest"
    );
}

#[test]
fn weight_order_ties_still_break_by_column_ascending_in_both_directions() {
    let triples = vec![(0u64, 4u64, 5.0f32), (0, 3, 5.0), (0, 2, 5.0), (0, 1, 5.0)];
    let rows = rows_from(&triples, 5);
    for order in [WeightOrder::Desc, WeightOrder::Asc] {
        let mut out = NeighborhoodPlans::default();
        plans_from_graph_chunk(&rows, None, Some(2), order, cfg(true), &mut out);
        assert_eq!(set_rows(&out.plans[0]), vec![0, 1, 2], "{order:?}");
    }
}

#[test]
fn non_finite_weights_sort_last_and_never_displace_a_finite_edge() {
    // A distance graph carries `inf` for "not connected", and a NaN is someone
    // else's upstream problem. Neither may be chosen over a real edge, and
    // neither may make the comparator non-total — `slice::sort_by` is entitled
    // to panic on one that is not.
    for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let triples = vec![(0u64, 1u64, bad), (0, 2, 2.0f32), (0, 3, 1.0), (0, 4, bad)];
        let rows = rows_from(&triples, 5);
        for order in [WeightOrder::Desc, WeightOrder::Asc] {
            let mut out = NeighborhoodPlans::default();
            plans_from_graph_chunk(&rows, None, Some(2), order, cfg(true), &mut out);
            let got = set_rows(&out.plans[0]);
            assert!(
                !got.contains(&1) && !got.contains(&4),
                "{bad} at {order:?} was selected over a finite edge: {got:?}"
            );
            assert_eq!(got.len(), 3, "centre plus the two finite edges");
        }
        // With no k there is no ranking, so every stored edge survives.
        let mut all = NeighborhoodPlans::default();
        plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut all);
        assert_eq!(set_rows(&all.plans[0]), vec![0, 1, 2, 3, 4]);
    }
}

#[test]
fn a_negative_column_is_dropped_rather_than_wrapping_into_a_huge_row() {
    // `BackedPairwiseReader` refuses one, but this function is public and takes
    // any `PairwiseRows`: `c as u64` on a negative is a row index above 2^63
    // that a `None` keep mask waves through, and the gather then fails on it
    // far from here.
    let rows = PairwiseRows {
        indptr: vec![0, 3],
        indices: vec![-1, 1, 2],
        data: vec![1.0, 1.0, 1.0],
        row_start: 0,
        n_rows: 1,
        n_cols: 4,
    };
    let mut out = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut out);
    assert_eq!(set_rows(&out.plans[0]), vec![0, 1, 2]);
}

#[test]
fn k_larger_than_the_population_returns_everything_available() {
    // ⚠️ This pins the ANSWER, not the cost, and the distinction is worth
    // stating: the accompanying fix — an extra exit when every available point
    // is already in hand — is a pure optimisation, and removing it leaves this
    // test green. Without it, `out.len() >= k` is never satisfied for
    // `k >= members - 1` and every centre scans the whole bounding box to
    // `max_ring`; the cell count is capped at `MAX_CELLS_PER_POINT * n`, so
    // that is O(n) ring visits per centre in 2-D — invisible at the twelve
    // points here and quadratic at 10^5. Seeing it fail would take either
    // ring-count instrumentation kept for a test, or a fixture large enough to
    // hang the suite; neither is worth it for a change that cannot alter an
    // answer. Recorded rather than left for a reader to assume the mutation
    // was checked.
    let coords: Vec<f32> = (0..12).flat_map(|i| [i as f32 * 5000.0, 0.0]).collect();
    let built = plans_from_coords(&coords, 2, None, CoordQuery::Knn(1000), cfg(true)).unwrap();
    for plan in &built.plans {
        assert_eq!(plan.rows.len(), 12, "every other point, and no more");
    }
    assert_coords_match_reference(&coords, 2, None, CoordQuery::Knn(1000));
}

#[test]
fn the_two_builders_order_a_set_differently_and_both_are_pinned() {
    // Found by smoking the real fixture, not by a reviewer: the graph builder
    // emits neighbours column-ascending and the coordinate builder emits them
    // nearest-first, so `rows[1]` means different things on the two paths.
    // Both are deliberate; this is what stops one drifting into the other
    // unnoticed, which would silently change what `rows[1]` is for a consumer
    // reading it as "the nearest".
    //
    // ⚠️ The layout is irregular on purpose. The first version of this test
    // used the square lattice every other test here uses — where a centre's
    // four rook neighbours are EQUIDISTANT, so the coordinate builder's
    // (distance, row) tie rule reproduces column order exactly and the two
    // paths agree. The test asserted they differ and failed. A fixture whose
    // neighbours are all the same distance apart cannot see an ordering
    // difference at all.
    //
    // Points on a line at 0, 100, 10, 200, 20: from centre 0 the distances are
    // 100, 10, 200, 20, so nearest-first is rows [2, 4, 1, 3] while
    // column-ascending is [1, 2, 3, 4].
    let coords: Vec<f32> = vec![0.0, 100.0, 10.0, 200.0, 20.0];
    let triples: Vec<(u64, u64, f32)> = (1..5u64).map(|c| (0u64, c, 1.0f32)).collect();
    let rows = rows_from(&triples, 5);

    let mut g = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut g);
    let grows = set_rows(&g.plans[0]);
    assert_eq!(grows, vec![0, 1, 2, 3, 4], "graph: column-ascending");

    let c = plans_from_coords(&coords, 1, None, CoordQuery::Knn(4), cfg(true)).unwrap();
    let crows = set_rows(&c.plans[0]);
    assert_eq!(crows, vec![0, 2, 4, 1, 3], "coords: nearest-first");

    // Same members, different order — which is the whole point.
    let (mut gs, mut cs) = (grows.clone(), crows.clone());
    gs.sort();
    cs.sort();
    assert_eq!(gs, cs);
    assert_ne!(grows, crows);
}

#[test]
fn a_null_coordinate_is_refused_rather_than_read_from_the_buffer() {
    // `values()` on a nullable Arrow column hands back the raw buffer, so a
    // null slot reads as whatever is in it. Round 1 added this guard to the
    // COO decoder and not here, where it matters more: a bogus-but-finite
    // value sails past the `is_finite` check and puts the cell in a
    // neighbourhood it is not in, while a non-finite bit pattern raises "NaN
    // coordinate" and names the wrong defect.
    //
    // Written at this layer rather than in pyscx because `from_anndata` does
    // not round-trip a nullable `obsm` — the writer is what would have to
    // change for the Python route to exist at all.
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let n = 6usize;
    let path = dir.path().join("nullcoord.scx");
    let mut w = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(n as u64, 4, 0, 16384, 0, 0),
    )
    .unwrap();
    w.write_obs(&fixture::obs_batch(n)).unwrap();
    w.write_var(&fixture::var_batch(4)).unwrap();
    let schema = Schema::new(vec![
        Field::new("0", DataType::Float32, true),
        Field::new("1", DataType::Float32, true),
    ]);
    let xs: Vec<Option<f32>> = (0..n).map(|i| Some(i as f32)).collect();
    let mut ys: Vec<Option<f32>> = (0..n).map(|i| Some(i as f32)).collect();
    ys[2] = None;
    let batch = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(xs)),
            Arc::new(Float32Array::from(ys)),
        ],
    )
    .unwrap();
    w.write_obsm("spatial", &batch).unwrap();
    w.finish().unwrap();

    let err = read_coords(&path, "spatial").unwrap_err().to_string();
    assert!(err.contains("null entries"), "{err}");
    assert!(
        err.contains("column 1"),
        "the error must name the column: {err}"
    );
}

#[test]
fn a_non_finite_weight_is_never_emitted_as_a_neighbour_even_when_k_is_short() {
    // The round-1 test had exactly `k` finite edges, so it could only prove
    // that a non-finite edge does not DISPLACE a finite one. With fewer finite
    // edges than `k` the `inf` was still emitted — contradicting the contract
    // that a distance graph's "not connected" is never a near neighbour.
    for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let triples = vec![(0u64, 1u64, bad), (0, 2, 2.0f32), (0, 3, bad)];
        let rows = rows_from(&triples, 4);
        for order in [WeightOrder::Desc, WeightOrder::Asc] {
            let mut out = NeighborhoodPlans::default();
            // k = 3, but only ONE finite edge exists.
            plans_from_graph_chunk(&rows, None, Some(3), order, cfg(true), &mut out);
            assert_eq!(
                set_rows(&out.plans[0]),
                vec![0, 2],
                "{bad} at {order:?} was emitted to fill out k"
            );
        }
    }
    // Without `k` there is no ranking and every stored edge passes through,
    // which is the documented behaviour and stays that way.
    let triples = vec![(0u64, 1u64, f32::INFINITY), (0, 2, 2.0f32)];
    let rows = rows_from(&triples, 3);
    let mut all = NeighborhoodPlans::default();
    plans_from_graph_chunk(&rows, None, None, WeightOrder::Desc, cfg(true), &mut all);
    assert_eq!(set_rows(&all.plans[0]), vec![0, 1, 2]);
}
