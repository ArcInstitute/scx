// Row-set algebra over global obs-row ranges.
//
// The query engine's obs-filter / X-materialization pipeline is entirely
// global-row based (`obs_mask`, `ShardInfo`, deletion vectors,
// `materialize_filtered_obs`). When an obs predicate can be resolved directly
// from the `PredicateIndex` (see `predicate::eval_rowset`), we represent the
// matching rows as a [`RowSet`] — a sorted, disjoint, coalesced set of
// half-open global-row ranges — instead of decoding every obs shard to build an
// O(n_obs) boolean mask.
//
// All set operations are linear two-cursor merges over the sorted range slices,
// so cost is proportional to the number of *ranges*, not the number of rows.
// This is what lets `limit(N)` and `count()` short-circuit at atlas scale.

use crate::index::ShardRange;

/// A half-open range of global obs rows: `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowRange {
    pub start: u64,
    pub end: u64,
}

impl RowRange {
    #[inline]
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// A sorted, disjoint, coalesced set of global obs-row ranges.
///
/// Invariants (upheld by every constructor and operation):
/// - ranges are sorted ascending by `start`,
/// - ranges are non-empty (`start < end`),
/// - adjacent ranges have a strictly positive gap (`prev.end < next.start`) —
///   touching or overlapping ranges are always coalesced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowSet {
    ranges: Vec<RowRange>,
}

impl RowSet {
    /// The empty row-set.
    pub fn empty() -> Self {
        RowSet { ranges: Vec::new() }
    }

    /// Build a row-set from an arbitrary list of ranges. Empty ranges are
    /// dropped; the rest are sorted and coalesced into the canonical form.
    pub fn from_ranges(mut ranges: Vec<RowRange>) -> Self {
        ranges.retain(|r| !r.is_empty());
        ranges.sort_unstable_by_key(|r| r.start);
        Self::coalesce_sorted(ranges)
    }

    /// Build a row-set from ranges already sorted ascending by `start`. Empty
    /// ranges are dropped and overlapping/touching ranges coalesced.
    pub fn from_sorted_ranges(ranges: Vec<RowRange>) -> Self {
        let filtered: Vec<RowRange> = ranges.into_iter().filter(|r| !r.is_empty()).collect();
        debug_assert!(
            filtered.windows(2).all(|w| w[0].start <= w[1].start),
            "from_sorted_ranges given unsorted input"
        );
        Self::coalesce_sorted(filtered)
    }

    /// Coalesce a sorted (by start), non-empty list of ranges in place.
    fn coalesce_sorted(sorted: Vec<RowRange>) -> Self {
        let mut out: Vec<RowRange> = Vec::with_capacity(sorted.len());
        for r in sorted {
            match out.last_mut() {
                // touching or overlapping -> extend
                Some(last) if r.start <= last.end => {
                    if r.end > last.end {
                        last.end = r.end;
                    }
                }
                _ => out.push(r),
            }
        }
        RowSet { ranges: out }
    }

    /// The full universe `[0, n_obs)`.
    pub fn universe(n_obs: u64) -> Self {
        if n_obs == 0 {
            RowSet::empty()
        } else {
            RowSet {
                ranges: vec![RowRange {
                    start: 0,
                    end: n_obs,
                }],
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Total number of rows covered.
    pub fn cardinality(&self) -> u64 {
        self.ranges.iter().map(|r| r.len()).sum()
    }

    pub fn ranges(&self) -> &[RowRange] {
        &self.ranges
    }

    /// Union of two row-sets (linear merge).
    pub fn union(&self, other: &Self) -> Self {
        let mut merged: Vec<RowRange> = Vec::with_capacity(self.ranges.len() + other.ranges.len());
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            if self.ranges[i].start <= other.ranges[j].start {
                merged.push(self.ranges[i]);
                i += 1;
            } else {
                merged.push(other.ranges[j]);
                j += 1;
            }
        }
        merged.extend_from_slice(&self.ranges[i..]);
        merged.extend_from_slice(&other.ranges[j..]);
        Self::coalesce_sorted(merged)
    }

    /// Intersection of two row-sets (linear two-cursor walk).
    pub fn intersect(&self, other: &Self) -> Self {
        let mut out: Vec<RowRange> = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            let a = self.ranges[i];
            let b = other.ranges[j];
            let start = a.start.max(b.start);
            let end = a.end.min(b.end);
            if start < end {
                out.push(RowRange { start, end });
            }
            // advance the range that ends first
            if a.end <= b.end {
                i += 1;
            } else {
                j += 1;
            }
        }
        // inputs are already disjoint+sorted, so `out` is too.
        RowSet { ranges: out }
    }

    /// Set difference `self \ other` (linear two-cursor walk).
    pub fn difference(&self, other: &Self) -> Self {
        let mut out: Vec<RowRange> = Vec::new();
        let mut j = 0;
        for &a in &self.ranges {
            let mut cur = a.start;
            // advance `other` cursor past ranges that end before `a` starts
            while j < other.ranges.len() && other.ranges[j].end <= a.start {
                j += 1;
            }
            let mut k = j;
            while k < other.ranges.len() && other.ranges[k].start < a.end {
                let b = other.ranges[k];
                if b.start > cur {
                    out.push(RowRange {
                        start: cur,
                        end: b.start.min(a.end),
                    });
                }
                if b.end > cur {
                    cur = b.end;
                }
                if cur >= a.end {
                    break;
                }
                k += 1;
            }
            if cur < a.end {
                out.push(RowRange {
                    start: cur,
                    end: a.end,
                });
            }
        }
        // `out` is sorted and disjoint by construction.
        RowSet { ranges: out }
    }

    /// Complement against the universe `[0, n_obs)`.
    pub fn complement(&self, n_obs: u64) -> Self {
        RowSet::universe(n_obs).difference(self)
    }

    /// Keep only the first `n` rows in ascending order (the `limit(N)`
    /// short-circuit). Returns at most `n` rows; ranges are split as needed.
    pub fn take_first(&self, n: u64) -> Self {
        if n == 0 {
            return RowSet::empty();
        }
        let mut out: Vec<RowRange> = Vec::new();
        let mut remaining = n;
        for &r in &self.ranges {
            if remaining == 0 {
                break;
            }
            let take = r.len().min(remaining);
            out.push(RowRange {
                start: r.start,
                end: r.start + take,
            });
            remaining -= take;
        }
        RowSet { ranges: out }
    }

    /// Iterate individual global rows in ascending order.
    pub fn iter_rows(&self) -> impl Iterator<Item = u64> + '_ {
        self.ranges.iter().flat_map(|r| r.start..r.end)
    }
}

/// Convert an obs-shard-local [`ShardRange`] to a global [`RowRange`] using the
/// catalog's obs-shard row-range table `(shard_idx, row_start, row_end)` sorted
/// by `shard_idx`. Returns `None` if the range's `shard_id` is not present in
/// the table (a stale index referencing a shard the catalog no longer maps) so
/// the caller can fall back to a full obs scan rather than emit wrong rows.
pub fn shard_range_to_global(
    sr: &ShardRange,
    obs_shard_ranges: &[(u32, u64, u64)],
) -> Option<RowRange> {
    let pos = obs_shard_ranges
        .binary_search_by_key(&sr.shard_id, |(idx, _, _)| *idx)
        .ok()?;
    let (_, shard_row_start, shard_row_end) = obs_shard_ranges[pos];
    let start = shard_row_start + sr.row_start as u64;
    // Defensive: a stale/corrupt local range must never exceed the shard, and
    // the result must satisfy the `RowRange` invariant `start < end`. Clamp the
    // end into the shard, then reject any empty/out-of-bounds range as a
    // coverage gap (`None`) — the caller treats that as "not index-resolvable"
    // and falls back to the full obs scan rather than emit an invalid range.
    let end = (shard_row_start + sr.row_end as u64).min(shard_row_end);
    if start >= end {
        return None;
    }
    Some(RowRange { start, end })
}

#[cfg(test)]
#[path = "rowset_tests.rs"]
mod tests;
