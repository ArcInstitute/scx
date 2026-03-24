//! Range coalescing for cloud reads from packed SCX files.
//!
//! Implements SPEC §12.3. Merges adjacent or nearby byte ranges into
//! larger reads to minimize the number of cloud GET requests.

/// Given a set of (offset, length) byte ranges, merge adjacent or nearby
/// ranges separated by less than `gap_threshold` bytes into single reads.
///
/// Input ranges should be sorted by offset. The output ranges cover the
/// same bytes plus any small gaps between them.
pub fn coalesce_ranges(sections: &[(u64, u64)], gap_threshold: usize) -> Vec<(u64, u64)> {
    if sections.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(sections.len());
    let (mut current_start, mut current_len) = sections[0];

    for &(offset, length) in &sections[1..] {
        let current_end = current_start + current_len;
        let gap = offset.saturating_sub(current_end);

        if gap <= gap_threshold as u64 {
            // Merge: extend current range to cover this section
            current_len = (offset + length) - current_start;
        } else {
            // Gap too large: emit current range and start a new one
            result.push((current_start, current_len));
            current_start = offset;
            current_len = length;
        }
    }

    // Emit the last range
    result.push((current_start, current_len));
    result
}

/// Default gap threshold: 256 KB. Gaps smaller than this are included
/// in a single read rather than issuing separate requests.
pub const DEFAULT_GAP_THRESHOLD: usize = 256 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        assert_eq!(coalesce_ranges(&[], 1024), vec![]);
    }

    #[test]
    fn test_single_range() {
        let ranges = vec![(100, 200)];
        assert_eq!(coalesce_ranges(&ranges, 1024), vec![(100, 200)]);
    }

    #[test]
    fn test_adjacent_ranges_merged() {
        // Two adjacent ranges with no gap
        let ranges = vec![(0, 100), (100, 200)];
        assert_eq!(coalesce_ranges(&ranges, 1024), vec![(0, 300)]);
    }

    #[test]
    fn test_small_gap_merged() {
        // Gap of 50 bytes, threshold 1024
        let ranges = vec![(0, 100), (150, 200)];
        assert_eq!(coalesce_ranges(&ranges, 1024), vec![(0, 350)]);
    }

    #[test]
    fn test_large_gap_not_merged() {
        // Gap of 2000 bytes, threshold 1024
        let ranges = vec![(0, 100), (2100, 200)];
        assert_eq!(coalesce_ranges(&ranges, 1024), vec![(0, 100), (2100, 200)]);
    }

    #[test]
    fn test_mixed_gaps() {
        // Three ranges: first two close, third far
        let ranges = vec![(0, 100), (150, 100), (10000, 200)];
        assert_eq!(coalesce_ranges(&ranges, 1024), vec![(0, 250), (10000, 200)]);
    }

    #[test]
    fn test_all_within_threshold() {
        let ranges = vec![(0, 100), (200, 100), (400, 100), (600, 100)];
        assert_eq!(coalesce_ranges(&ranges, 1024), vec![(0, 700)]);
    }

    #[test]
    fn test_zero_threshold() {
        // Only truly adjacent ranges merge
        let ranges = vec![(0, 100), (100, 100), (300, 100)];
        assert_eq!(coalesce_ranges(&ranges, 0), vec![(0, 200), (300, 100)]);
    }
}
