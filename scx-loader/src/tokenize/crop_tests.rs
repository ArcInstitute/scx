use super::*;

const N_GENES: i64 = 100;
const MASK: i64 = N_GENES;
const PAD: i64 = N_GENES + 1;

struct Buf {
    ids: Vec<i64>,
    values: Vec<f32>,
    mask: Vec<u8>,
    pad: Vec<u8>,
}

/// Drive the kernel the way `collate_cell` does: selection values and emitted
/// values are separate arrays, and the panel index is built from `query` exactly
/// as `collate_gathered` builds it per set.
#[allow(clippy::too_many_arguments)]
fn run(
    gene_ids: &[i32],
    selection: &[f32],
    emit: &[f32],
    k: usize,
    query: &[i32],
    maskpos: Option<&[u8]>,
    hide_readout: bool,
    order: &mut Vec<usize>,
) -> Buf {
    let mut b = Buf {
        ids: vec![0; k],
        values: vec![0.0; k],
        mask: vec![0; k],
        pad: vec![0; k],
    };
    let idx = maskpos.map(|_| SetQueryIndex::new(query));
    let cin = CropIn {
        row: CsrRow {
            gene_ids,
            values: selection,
        },
        emit,
        withheld: maskpos.map(|positions| RowMask {
            positions,
            index: idx.as_ref().unwrap(),
        }),
        hide_readout,
    };
    top_k(
        &cin,
        &CropConfig {
            k,
            n_genes_total: N_GENES,
        },
        order,
        &mut CropOut {
            ids: &mut b.ids,
            values: &mut b.values,
            mask: &mut b.mask,
            pad: &mut b.pad,
        },
    );
    b
}

fn plain(gene_ids: &[i32], counts: &[f32], k: usize) -> Buf {
    run(
        gene_ids,
        counts,
        counts,
        k,
        &[],
        None,
        false,
        &mut Vec::new(),
    )
}

#[test]
fn orders_by_value_descending() {
    let b = plain(&[3, 5, 9], &[1.0, 7.0, 4.0], 3);
    assert_eq!(b.ids, vec![5, 9, 3]);
    assert_eq!(b.values, vec![7.0, 4.0, 1.0]);
    assert_eq!(b.pad, vec![0, 0, 0]);
}

#[test]
fn ties_break_by_gene_id_ascending() {
    // Every count equal, so the tiebreak is the ONLY thing deciding the order.
    // This is the test the cross-repo golden cannot do: all nine of its cases
    // carry distinct positive counts, so a flipped comparator leaves it green.
    let b = plain(&[9, 3, 5, 7], &[2.0, 2.0, 2.0, 2.0], 4);
    assert_eq!(b.ids, vec![3, 5, 7, 9]);
}

#[test]
fn ties_break_by_id_only_within_an_equal_value_run() {
    // Two runs at different values: the runs stay value-ordered, ids sort inside.
    let b = plain(&[9, 3, 8, 2], &[1.0, 1.0, 5.0, 5.0], 4);
    assert_eq!(b.ids, vec![2, 8, 3, 9]);
}

#[test]
fn truncates_to_k_keeping_the_largest() {
    let b = plain(&[1, 2, 3, 4], &[4.0, 1.0, 3.0, 2.0], 2);
    assert_eq!(b.ids, vec![1, 3]);
    assert_eq!(b.pad, vec![0, 0]);
}

#[test]
fn unfilled_slots_stay_pad() {
    let b = plain(&[1, 2], &[4.0, 3.0], 5);
    assert_eq!(b.ids, vec![1, 2, PAD, PAD, PAD]);
    assert_eq!(b.pad, vec![0, 0, 1, 1, 1]);
    assert_eq!(b.values, vec![4.0, 3.0, 0.0, 0.0, 0.0]);
    assert_eq!(b.mask, vec![0, 0, 0, 0, 0]);
}

#[test]
fn non_positive_selection_values_are_not_selected() {
    let b = plain(&[1, 2, 3, 4], &[0.0, -3.0, 5.0, 2.0], 4);
    assert_eq!(b.ids, vec![3, 4, PAD, PAD]);
}

#[test]
fn emitted_values_come_from_emit_not_from_selection() {
    // The separation `CropIn` exists for: state3 selects on raw counts while
    // emitting a transform of them. Collapsing the two arrays would emit the
    // selection counts and pass every test that uses one array for both.
    let mut order = Vec::new();
    let b = run(
        &[1, 2, 3],
        &[1.0, 9.0, 5.0],
        &[10.0, 20.0, 30.0],
        3,
        &[],
        None,
        false,
        &mut order,
    );
    assert_eq!(b.ids, vec![2, 3, 1], "order still follows selection");
    assert_eq!(b.values, vec![20.0, 30.0, 10.0], "values follow emit");
}

#[test]
fn empty_row_emits_one_inactive_gene_mask() {
    let b = plain(&[], &[], 3);
    assert_eq!(b.ids, vec![MASK, PAD, PAD]);
    assert_eq!(b.pad, vec![0, 1, 1]);
    // Distinct from the all-withheld fallback below: mask[0] stays 0 here.
    assert_eq!(b.mask, vec![0, 0, 0]);
}

#[test]
fn a_row_with_no_positive_counts_emits_one_inactive_gene_mask() {
    let b = plain(&[1, 2], &[0.0, 0.0], 2);
    assert_eq!(b.ids, vec![MASK, PAD]);
    assert_eq!(b.mask, vec![0, 0]);
    assert_eq!(b.pad, vec![0, 1]);
}

#[test]
fn hide_readout_emits_one_active_gene_mask_and_skips_the_sort() {
    let mut order = vec![999usize; 4];
    let b = run(
        &[1, 2, 3],
        &[5.0, 4.0, 3.0],
        &[5.0, 4.0, 3.0],
        3,
        &[],
        None,
        true,
        &mut order,
    );
    assert_eq!(b.ids, vec![MASK, PAD, PAD]);
    assert_eq!(b.mask, vec![1, 0, 0]);
    assert_eq!(b.pad, vec![0, 1, 1]);
    assert_eq!(
        order,
        vec![999usize; 4],
        "hide_readout short-circuits before the selection, so scratch is untouched"
    );
}

#[test]
fn withheld_genes_drop_to_pad_and_survivors_compact_left() {
    // query = [2], withheld. Gene 2 is inside the top-3, so its slot disappears
    // and 1 / 3 move left; nothing is pulled in from beyond `take`.
    let b = run(
        &[1, 2, 3],
        &[9.0, 8.0, 7.0],
        &[9.0, 8.0, 7.0],
        3,
        &[2],
        Some(&[1]),
        false,
        &mut Vec::new(),
    );
    assert_eq!(b.ids, vec![1, 3, PAD]);
    assert_eq!(b.values, vec![9.0, 7.0, 0.0]);
    assert_eq!(b.pad, vec![0, 0, 1]);
}

#[test]
fn no_backfill_beyond_take() {
    // k = 2 selects genes 1 and 2; withholding 2 must NOT pull gene 3 in.
    let b = run(
        &[1, 2, 3],
        &[9.0, 8.0, 7.0],
        &[9.0, 8.0, 7.0],
        2,
        &[2],
        Some(&[1]),
        false,
        &mut Vec::new(),
    );
    assert_eq!(b.ids, vec![1, PAD]);
    assert_eq!(b.pad, vec![0, 1]);
}

#[test]
fn all_withheld_emits_one_active_gene_mask() {
    let b = run(
        &[1, 2],
        &[9.0, 8.0],
        &[9.0, 8.0],
        3,
        &[1, 2],
        Some(&[1, 1]),
        false,
        &mut Vec::new(),
    );
    assert_eq!(b.ids, vec![MASK, PAD, PAD]);
    // mask[0] = 1 here, unlike the degenerate empty row.
    assert_eq!(b.mask, vec![1, 0, 0]);
    assert_eq!(b.pad, vec![0, 1, 1]);
    assert_eq!(b.values, vec![0.0, 0.0, 0.0]);
}

#[test]
fn an_unflagged_query_position_withholds_nothing() {
    let b = run(
        &[1, 2, 3],
        &[9.0, 8.0, 7.0],
        &[9.0, 8.0, 7.0],
        3,
        &[2],
        Some(&[0]),
        false,
        &mut Vec::new(),
    );
    assert_eq!(b.ids, vec![1, 2, 3]);
}

#[test]
fn duplicate_query_ids_withhold_if_any_position_is_flagged() {
    // `query` is not deduplicated, and the one in-repo producer draws it with
    // replacement. Stopping at the first match would keep a withheld gene.
    let b = run(
        &[1, 2],
        &[9.0, 8.0],
        &[9.0, 8.0],
        2,
        &[2, 2],
        Some(&[0, 1]),
        false,
        &mut Vec::new(),
    );
    assert_eq!(b.ids, vec![1, PAD], "the flagged second occurrence counts");
}

#[test]
fn unsorted_query_positions_index_the_mask_at_the_original_offset() {
    // Panel [5, 1]: the flag is at offset 1, which is gene 1 in the ORIGINAL
    // panel even though sorting puts gene 1 first.
    let b = run(
        &[1, 5],
        &[9.0, 8.0],
        &[9.0, 8.0],
        2,
        &[5, 1],
        Some(&[0, 1]),
        false,
        &mut Vec::new(),
    );
    assert_eq!(b.ids, vec![5, PAD]);
}

#[test]
fn a_mask_shorter_than_the_panel_leaves_the_excess_unflagged() {
    // Tolerated rather than a panic: this is a `pub` kernel and
    // `collate_gathered` validates the length in front of it.
    let b = run(
        &[1, 2],
        &[9.0, 8.0],
        &[9.0, 8.0],
        2,
        &[1, 2],
        Some(&[0]),
        false,
        &mut Vec::new(),
    );
    assert_eq!(b.ids, vec![1, 2]);
}

#[test]
fn reusing_one_scratch_across_rows_is_not_observable() {
    // The fail-first arm for the per-row-allocation removal: a kernel that read
    // a stale `order` would differ here, and the previous row is deliberately
    // longer so a missing `clear()` leaves extra indices behind.
    let rows: [(&[i32], &[f32]); 3] = [
        (&[1, 2, 3, 4, 5], &[5.0, 4.0, 3.0, 2.0, 1.0]),
        (&[7, 8], &[1.0, 2.0]),
        (&[], &[]),
    ];
    let mut shared = Vec::new();
    for (ids, vals) in rows {
        let fresh = run(ids, vals, vals, 4, &[], None, false, &mut Vec::new());
        let reused = run(ids, vals, vals, 4, &[], None, false, &mut shared);
        assert_eq!(fresh.ids, reused.ids);
        assert_eq!(fresh.values, reused.values);
        assert_eq!(fresh.mask, reused.mask);
        assert_eq!(fresh.pad, reused.pad);
    }
}

#[test]
fn sentinels_are_derived_from_the_vocabulary_size() {
    assert_eq!(gene_mask_id(100), 100);
    assert_eq!(pad_id(100), 101);
}
