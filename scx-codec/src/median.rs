// Shared floor-median utilities for codec parameter selection.
//
// Consolidates the previously-duplicated `floor_median` functions from
// `rice.rs`, `delta_golomb.rs`, and `scx-format/src/codec_select.rs`
// into a single canonical implementation.
//
// Uses `select_nth_unstable` for O(N) average-case performance instead
// of a full O(N log N) sort, since we only need a single order statistic.

/// Generic floor-median implementation.
///
/// - Empty slice → `T::default()`.
/// - Odd length → the middle value.
/// - Even length → the **lower** of the two middle values ("floor median").
///
/// Uses `select_nth_unstable` (introselect, O(N) average) instead of a
/// full sort. The index `(n - 1) / 2` selects the lower-middle for even
/// lengths and the exact middle for odd lengths.
fn floor_median_impl<T: Ord + Clone + Default>(values: &[T]) -> T {
    match values.len() {
        0 => T::default(),
        1 => values[0].clone(),
        n => {
            let mut tmp = values.to_vec();
            let (_, val, _) = tmp.select_nth_unstable((n - 1) / 2);
            val.clone()
        }
    }
}

/// Compute the floor median of a slice of `u32` values.
///
/// - Empty slice → 0.
/// - Odd length → the middle value.
/// - Even length → the **lower** of the two middle values ("floor median").
///
/// The input slice is not modified; a temporary copy is used.
pub fn floor_median_u32(values: &[u32]) -> u32 {
    floor_median_impl(values)
}

/// Compute the floor median of a slice of `u64` values.
///
/// Same semantics as [`floor_median_u32`] but for `u64` data
/// (used by Delta-Golomb-Rice for indptr delta medians).
pub fn floor_median_u64(values: &[u64]) -> u64 {
    floor_median_impl(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- u32 tests ---

    #[test]
    fn test_floor_median_u32_empty() {
        assert_eq!(floor_median_u32(&[]), 0);
    }

    #[test]
    fn test_floor_median_u32_single() {
        assert_eq!(floor_median_u32(&[42]), 42);
    }

    #[test]
    fn test_floor_median_u32_even() {
        // [1, 2, 3, 4] → lower middle = sorted[1] = 2
        assert_eq!(floor_median_u32(&[4, 1, 3, 2]), 2);
    }

    #[test]
    fn test_floor_median_u32_odd() {
        // [1, 2, 3] → middle = sorted[1] = 2
        assert_eq!(floor_median_u32(&[3, 1, 2]), 2);
    }

    #[test]
    fn test_floor_median_u32_five_elements() {
        // [1, 2, 3, 4, 5] → sorted[2] = 3
        let vals = vec![5, 1, 3, 2, 4];
        assert_eq!(floor_median_u32(&vals), 3);
        // Verify input was not modified
        assert_eq!(vals, vec![5, 1, 3, 2, 4]);
    }

    #[test]
    fn test_floor_median_u32_four_elements() {
        // [10, 20, 30, 40] → sorted[1] = 20 (lower middle)
        assert_eq!(floor_median_u32(&[10, 20, 30, 40]), 20);
    }

    #[test]
    fn test_floor_median_u32_two_elements() {
        // [5, 10] → (2-1)/2 = 0 → sorted[0] = 5 (lower)
        assert_eq!(floor_median_u32(&[10, 5]), 5);
    }

    // --- u64 tests ---

    #[test]
    fn test_floor_median_u64_empty() {
        assert_eq!(floor_median_u64(&[]), 0);
    }

    #[test]
    fn test_floor_median_u64_single() {
        assert_eq!(floor_median_u64(&[42]), 42);
    }

    #[test]
    fn test_floor_median_u64_even() {
        assert_eq!(floor_median_u64(&[4, 1, 3, 2]), 2);
    }

    #[test]
    fn test_floor_median_u64_odd() {
        assert_eq!(floor_median_u64(&[3, 1, 2]), 2);
    }
}
