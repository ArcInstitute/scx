//! Tests for the deterministic column-block planner.

use super::*;

/// Every block count must tile the column axis exactly — no gap, no overlap,
/// no column dropped. A gap here is silent data loss in the reduction: the
/// covariance entries whose `lo` fell in the gap would simply never be summed.
#[test]
fn blocks_cover_the_column_axis_exactly() {
    for n in 0..24usize {
        let weights: Vec<u64> = (0..n).map(|c| (c as u64 % 5) + 1).collect();
        for n_blocks in 1..=(n + 3).max(1) {
            let blocks = plan_blocks(&weights, n_blocks);
            assert_eq!(
                blocks.len(),
                n_blocks,
                "n={n} n_blocks={n_blocks}: wrong block count"
            );
            let mut cursor = 0usize;
            for b in &blocks {
                assert_eq!(b.start, cursor, "n={n} n_blocks={n_blocks}: not contiguous");
                assert!(
                    b.end >= b.start,
                    "n={n} n_blocks={n_blocks}: inverted range"
                );
                cursor = b.end;
            }
            assert_eq!(cursor, n, "n={n} n_blocks={n_blocks}: does not reach n");
        }
    }
}

/// The split follows weight, not column count. A skewed weight vector must
/// produce blocks of visibly unequal width, or the planner is really an
/// equal-width split wearing a work model.
#[test]
fn blocks_balance_by_weight_not_column_count() {
    // Column 0 carries almost all the work; the rest are cheap.
    let mut weights = vec![1u64; 40];
    weights[0] = 1_000;

    let blocks = plan_blocks(&weights, 4);
    assert_eq!(
        blocks[0],
        0..1,
        "the heavy column must get a block to itself"
    );
    let widths: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
    assert!(
        widths.iter().any(|&w| w != widths[0]),
        "skewed weights produced equal-width blocks {widths:?} — the work model \
         is not being used"
    );
}

/// A zero-weight axis (an empty shard) still has to tile, and must not collapse
/// every column onto the last block.
#[test]
fn a_zero_weight_axis_splits_evenly() {
    let blocks = plan_blocks(&vec![0u64; 12], 4);
    assert_eq!(blocks, vec![0..3, 3..6, 6..9, 9..12]);
}

#[test]
fn sorted_subrange_finds_the_half_open_window() {
    let row = [0i32, 3, 4, 7, 9];
    assert_eq!(sorted_subrange(&row, 0, 10), (0, 5));
    assert_eq!(sorted_subrange(&row, 3, 8), (1, 4)); // 3, 4, 7
    assert_eq!(sorted_subrange(&row, 4, 4), (2, 2)); // empty window
    assert_eq!(sorted_subrange(&row, 10, 20), (5, 5)); // past the end
    assert_eq!(sorted_subrange(&row, 5, 7), (3, 3)); // gap between stored cols
    assert_eq!(sorted_subrange(&[], 0, 5), (0, 0));
}

/// Contiguous windows over a row must partition it: every nonzero belongs to
/// exactly one block. This is the property the reduction's correctness rests on.
#[test]
fn contiguous_windows_partition_a_row() {
    let row = [0i32, 1, 5, 6, 9, 11];
    for n_blocks in 1..=6usize {
        let blocks = plan_blocks(&vec![1u64; 12], n_blocks);
        let mut seen = Vec::new();
        for b in &blocks {
            let (lo, hi) = sorted_subrange(&row, b.start, b.end);
            seen.extend_from_slice(&row[lo..hi]);
        }
        assert_eq!(seen, row, "n_blocks={n_blocks}: windows did not partition");
    }
}

#[test]
fn strictly_increasing_rejects_unsorted_and_duplicate_columns() {
    let sorted = ScxCsr::new_unchecked((2, 4), vec![0, 2, 4], vec![0, 3, 1, 2], vec![1.0; 4]);
    assert!(rows_strictly_increasing(&sorted));

    let unsorted = ScxCsr::new_unchecked((2, 4), vec![0, 2, 4], vec![3, 0, 1, 2], vec![1.0; 4]);
    assert!(!rows_strictly_increasing(&unsorted));

    let duplicated = ScxCsr::new_unchecked((1, 4), vec![0, 2], vec![2, 2], vec![1.0; 2]);
    assert!(!rows_strictly_increasing(&duplicated));

    // An empty row is trivially increasing.
    let empty_row = ScxCsr::new_unchecked((2, 4), vec![0, 0, 1], vec![3], vec![1.0]);
    assert!(rows_strictly_increasing(&empty_row));
}

#[test]
fn split_by_blocks_yields_disjoint_slices_covering_the_buffer() {
    let blocks = plan_blocks(&vec![1u64; 7], 3);
    let mut buf = vec![0u32; 7 * 4];
    {
        let parts = split_by_blocks(&mut buf, &blocks, 4);
        assert_eq!(parts.len(), 3);
        for (b, part) in blocks.iter().zip(parts) {
            assert_eq!(part.len(), b.len() * 4);
            part.fill(b.start as u32 + 1);
        }
    }
    // Every element was written by exactly one block, in block order.
    let mut expected = Vec::new();
    for b in &blocks {
        expected.extend(std::iter::repeat_n(b.start as u32 + 1, b.len() * 4));
    }
    assert_eq!(buf, expected);
}

#[test]
fn block_count_never_exceeds_the_column_axis() {
    assert_eq!(block_count(1), 1);
    assert_eq!(block_count(0), 1);
    assert!(block_count(4096) >= 1);
    assert!(block_count(3) <= 3);
}
