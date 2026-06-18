use super::*;
use std::collections::BTreeSet;

fn rr(start: u64, end: u64) -> RowRange {
    RowRange { start, end }
}

/// Materialize a RowSet to a set of individual rows for model comparison.
fn to_set(rs: &RowSet) -> BTreeSet<u64> {
    rs.iter_rows().collect()
}

/// Assert RowSet invariants: sorted, non-empty, strictly-gapped (coalesced).
fn assert_canonical(rs: &RowSet) {
    let ranges = rs.ranges();
    for r in ranges {
        assert!(r.start < r.end, "empty range {:?}", r);
    }
    for w in ranges.windows(2) {
        assert!(
            w[0].end < w[1].start,
            "non-coalesced / unsorted: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn coalesces_overlapping_and_touching() {
    let rs = RowSet::from_ranges(vec![rr(0, 5), rr(5, 10), rr(3, 4), rr(20, 25)]);
    assert_canonical(&rs);
    assert_eq!(rs.ranges(), &[rr(0, 10), rr(20, 25)]);
    assert_eq!(rs.cardinality(), 15);
}

#[test]
fn drops_empty_ranges() {
    let rs = RowSet::from_ranges(vec![rr(5, 5), rr(10, 8), rr(1, 3)]);
    assert_eq!(rs.ranges(), &[rr(1, 3)]);
}

#[test]
fn universe_and_empty() {
    assert!(RowSet::empty().is_empty());
    assert!(RowSet::universe(0).is_empty());
    assert_eq!(RowSet::universe(7).ranges(), &[rr(0, 7)]);
}

#[test]
fn union_intersect_difference_basic() {
    let a = RowSet::from_ranges(vec![rr(0, 10), rr(20, 30)]);
    let b = RowSet::from_ranges(vec![rr(5, 25)]);

    let u = a.union(&b);
    assert_canonical(&u);
    assert_eq!(u.ranges(), &[rr(0, 30)]);

    let i = a.intersect(&b);
    assert_canonical(&i);
    assert_eq!(i.ranges(), &[rr(5, 10), rr(20, 25)]);

    let d = a.difference(&b);
    assert_canonical(&d);
    assert_eq!(d.ranges(), &[rr(0, 5), rr(25, 30)]);
}

#[test]
fn complement_round_trips() {
    let a = RowSet::from_ranges(vec![rr(2, 4), rr(7, 9)]);
    let c = a.complement(10);
    assert_eq!(c.ranges(), &[rr(0, 2), rr(4, 7), rr(9, 10)]);
    // complement of complement == original (within universe)
    assert_eq!(c.complement(10), a);
}

#[test]
fn take_first_splits_ranges() {
    let a = RowSet::from_ranges(vec![rr(0, 5), rr(100, 110)]);
    assert_eq!(a.take_first(0), RowSet::empty());
    assert_eq!(a.take_first(3).ranges(), &[rr(0, 3)]);
    assert_eq!(a.take_first(5).ranges(), &[rr(0, 5)]);
    assert_eq!(a.take_first(7).ranges(), &[rr(0, 5), rr(100, 102)]);
    assert_eq!(a.take_first(1000), a);
}

// ---- Property tests against a BTreeSet<u64> model ----

/// Deterministic LCG so tests need no external rng / no Math.random ban issues.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_rowset(rng: &mut Lcg, universe: u64) -> (RowSet, BTreeSet<u64>) {
    let n_ranges = rng.below(6);
    let mut ranges = Vec::new();
    for _ in 0..n_ranges {
        let start = rng.below(universe);
        let len = rng.below(universe / 4 + 1);
        ranges.push(rr(start, (start + len).min(universe)));
    }
    let rs = RowSet::from_ranges(ranges);
    let set = to_set(&rs);
    (rs, set)
}

#[test]
fn property_ops_match_model() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let universe = 64u64;
    for _ in 0..2000 {
        let (a, sa) = random_rowset(&mut rng, universe);
        let (b, sb) = random_rowset(&mut rng, universe);
        assert_canonical(&a);
        assert_canonical(&b);

        let u = a.union(&b);
        assert_canonical(&u);
        assert_eq!(to_set(&u), sa.union(&sb).copied().collect::<BTreeSet<_>>());

        let i = a.intersect(&b);
        assert_canonical(&i);
        assert_eq!(
            to_set(&i),
            sa.intersection(&sb).copied().collect::<BTreeSet<_>>()
        );

        let d = a.difference(&b);
        assert_canonical(&d);
        assert_eq!(
            to_set(&d),
            sa.difference(&sb).copied().collect::<BTreeSet<_>>()
        );

        let c = a.complement(universe);
        assert_canonical(&c);
        let full: BTreeSet<u64> = (0..universe).collect();
        assert_eq!(
            to_set(&c),
            full.difference(&sa).copied().collect::<BTreeSet<_>>()
        );

        assert_eq!(a.cardinality(), sa.len() as u64);

        let k = rng.below(universe + 5);
        let t = a.take_first(k);
        assert_canonical(&t);
        let expected: BTreeSet<u64> = sa.iter().copied().take(k as usize).collect();
        assert_eq!(to_set(&t), expected);
    }
}

#[test]
fn shard_range_to_global_maps_and_guards() {
    // obs_shard_ranges: shard 0 -> [0,100), shard 1 -> [100,250), shard 3 -> [250,400)
    let table = vec![(0u32, 0u64, 100u64), (1, 100, 250), (3, 250, 400)];

    let g = shard_range_to_global(
        &ShardRange {
            shard_id: 1,
            row_start: 10,
            row_end: 20,
        },
        &table,
    )
    .unwrap();
    assert_eq!(g, rr(110, 120));

    // shard 2 is absent (gap) -> None (stale-index safe fallback)
    assert!(shard_range_to_global(
        &ShardRange {
            shard_id: 2,
            row_start: 0,
            row_end: 5,
        },
        &table,
    )
    .is_none());

    // shard 3 maps relative to its own row_start
    let g3 = shard_range_to_global(
        &ShardRange {
            shard_id: 3,
            row_start: 0,
            row_end: 150,
        },
        &table,
    )
    .unwrap();
    assert_eq!(g3, rr(250, 400));
}

#[test]
fn shard_range_to_global_rejects_out_of_bounds_local_range() {
    // shard 0 -> [0,100). A corrupt/stale local range that exceeds the shard
    // must never yield an invalid (start >= end) RowRange — it returns None so
    // the caller falls back to the full scan.
    let table = vec![(0u32, 0u64, 100u64), (1, 100, 250)];

    // local row_start beyond the shard end (start would be 100, clamped end 100)
    assert!(shard_range_to_global(
        &ShardRange {
            shard_id: 0,
            row_start: 120,
            row_end: 130,
        },
        &table,
    )
    .is_none());

    // local range overshoots the shard end: clamp end into the shard, keep the
    // valid prefix.
    let clamped = shard_range_to_global(
        &ShardRange {
            shard_id: 0,
            row_start: 90,
            row_end: 250,
        },
        &table,
    )
    .unwrap();
    assert_eq!(clamped, rr(90, 100));
    assert!(clamped.start < clamped.end, "RowRange invariant must hold");
}
