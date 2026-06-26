//! Unit tests for the F1 group planner (`plan_group_shards`).

use super::*;

fn labels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("g{i}")).collect()
}

#[test]
fn empty_input_yields_empty_plan() {
    let plan = plan_group_shards(&[], &[], &[], &[], 10, 8, 100);
    assert_eq!(plan.n_shards, 0);
    assert!(plan.shard_starts.is_empty());
    assert!(plan.records.is_empty());
    assert_eq!(plan.reference_shard, None);
}

#[test]
fn row_count_mode_reference_first_then_binpack() {
    // labels: g0=reference, g1/g2/g3 = groups.
    // emission order: ref(0,1), g1(2,3,4), g2(5), g3(6,7,8,9)
    let group_of_new = [0, 0, 1, 1, 1, 2, 3, 3, 3, 3];
    let ref_of_new = [
        true, true, false, false, false, false, false, false, false, false,
    ];
    let labs = labels(4);
    // row-count mode (empty per_row_nnz), target 3 rows, max 100.
    let plan = plan_group_shards(&group_of_new, &ref_of_new, &labs, &[], 3, 0, 100);

    assert_eq!(plan.reference_shard, Some(0));
    assert_eq!(plan.n_shards, 4);
    assert_eq!(plan.shard_starts, vec![2, 5, 6]);

    let recs = &plan.records;
    assert_eq!(recs.len(), 4);
    assert_eq!(
        recs[0],
        GroupRecord {
            label: "g0".into(),
            shard: 0,
            row_start: 0,
            row_stop: 2,
            role: Role::Reference
        }
    );
    assert_eq!(recs[1].label, "g1");
    assert_eq!(
        (recs[1].shard, recs[1].row_start, recs[1].row_stop),
        (1, 2, 5)
    );
    assert_eq!(recs[2].label, "g2");
    assert_eq!(
        (recs[2].shard, recs[2].row_start, recs[2].row_stop),
        (2, 5, 6)
    );
    assert_eq!(recs[3].label, "g3");
    assert_eq!(
        (recs[3].shard, recs[3].row_start, recs[3].row_stop),
        (3, 6, 10)
    );
    // Every group is non-reference and exactly one shard.
    for r in &recs[1..] {
        assert_eq!(r.role, Role::Group);
    }
}

#[test]
fn never_splits_an_oversized_group() {
    // No reference. g0 has 5 rows; target 3, so it exceeds the budget but must
    // stay whole in its own shard (never split mid-group).
    let group_of_new = [0, 0, 0, 0, 0, 1, 1];
    let ref_of_new = [false; 7];
    let labs = labels(2);
    let plan = plan_group_shards(&group_of_new, &ref_of_new, &labs, &[], 3, 0, 100);

    assert_eq!(plan.reference_shard, None);
    // g0 (5 rows) alone in shard 0; g1 in shard 1.
    assert_eq!(plan.records.len(), 2);
    assert_eq!(
        (
            plan.records[0].shard,
            plan.records[0].row_start,
            plan.records[0].row_stop
        ),
        (0, 0, 5)
    );
    assert_eq!(
        (
            plan.records[1].shard,
            plan.records[1].row_start,
            plan.records[1].row_stop
        ),
        (1, 5, 7)
    );
    assert_eq!(plan.shard_starts, vec![5]);
    assert_eq!(plan.n_shards, 2);
}

#[test]
fn reference_isolated_in_shard_zero() {
    // Two reference labels packed first into shard 0, never mixed with groups.
    let group_of_new = [0, 1, 2, 2, 3];
    let ref_of_new = [true, true, false, false, false];
    let labs = labels(4);
    let plan = plan_group_shards(&group_of_new, &ref_of_new, &labs, &[], 100, 0, 1000);

    assert_eq!(plan.reference_shard, Some(0));
    // g0,g1 reference → shard 0; g2,g3 groups (fit budget) → shard 1.
    assert_eq!(plan.records[0].shard, 0);
    assert_eq!(plan.records[0].role, Role::Reference);
    assert_eq!(plan.records[1].shard, 0);
    assert_eq!(plan.records[1].role, Role::Reference);
    assert_eq!(plan.records[2].shard, 1);
    assert_eq!(plan.records[3].shard, 1);
    // reference rows only in shard 0.
    for r in plan.records.iter().filter(|r| r.role == Role::Reference) {
        assert_eq!(r.shard, 0);
    }
    assert_eq!(plan.shard_starts, vec![2]);
}

#[test]
fn split_label_yields_two_records() {
    // ReferenceSpec::Column case: label g1 has both reference and group rows.
    // Emission order: reference region first (g1 ref rows), then non-ref by
    // label (g1 group rows, then g2).
    let group_of_new = [1, 1, 1, 1, 2];
    let ref_of_new = [true, true, false, false, false];
    let labs = labels(3);
    let plan = plan_group_shards(&group_of_new, &ref_of_new, &labs, &[], 100, 0, 1000);

    // g1 appears twice: once reference (shard 0), once group (shard 1).
    let g1_recs: Vec<&GroupRecord> = plan.records.iter().filter(|r| r.label == "g1").collect();
    assert_eq!(g1_recs.len(), 2);
    assert!(g1_recs
        .iter()
        .any(|r| r.role == Role::Reference && r.shard == 0));
    assert!(g1_recs
        .iter()
        .any(|r| r.role == Role::Group && r.shard == 1));
    // Each block uniform in role; ranges contiguous & covering [0,5).
    assert_eq!(
        plan.records[0],
        GroupRecord {
            label: "g1".into(),
            shard: 0,
            row_start: 0,
            row_stop: 2,
            role: Role::Reference
        }
    );
    assert_eq!(
        plan.records[1],
        GroupRecord {
            label: "g1".into(),
            shard: 1,
            row_start: 2,
            row_stop: 4,
            role: Role::Group
        }
    );
    assert_eq!(
        plan.records[2],
        GroupRecord {
            label: "g2".into(),
            shard: 1,
            row_start: 4,
            row_stop: 5,
            role: Role::Group
        }
    );
}

#[test]
fn byte_mode_sizes_by_nnz_times_width() {
    // No reference. Two groups; per-row nnz drives packing, not row count.
    // g0 rows have nnz [10,10] => 20 nnz; g1 rows [1,1,1] => 3 nnz.
    // bytes_per_nnz=8 => g0=160 bytes, g1=24 bytes. target 170 bytes:
    // g0 alone is 160; adding g1 (184>170) seals → g1 in shard 1.
    let group_of_new = [0, 0, 1, 1, 1];
    let ref_of_new = [false; 5];
    let per_row_nnz = [10u64, 10, 1, 1, 1];
    let labs = labels(2);
    let plan = plan_group_shards(
        &group_of_new,
        &ref_of_new,
        &labs,
        &per_row_nnz,
        170,
        8,
        100_000,
    );

    assert_eq!(plan.records.len(), 2);
    assert_eq!(plan.records[0].shard, 0);
    assert_eq!(plan.records[1].shard, 1);
    assert_eq!(plan.shard_starts, vec![2]);

    // Widen the target so both fit in one shard.
    let plan2 = plan_group_shards(
        &group_of_new,
        &ref_of_new,
        &labs,
        &per_row_nnz,
        1000,
        8,
        100_000,
    );
    assert_eq!(plan2.n_shards, 1);
    assert!(plan2.shard_starts.is_empty());
    assert_eq!(plan2.records[0].shard, 0);
    assert_eq!(plan2.records[1].shard, 0);
}

#[test]
fn records_cover_all_rows_contiguously_and_within_one_shard() {
    let group_of_new = [0, 0, 1, 2, 2, 2, 3];
    let ref_of_new = [true, true, false, false, false, false, false];
    let labs = labels(4);
    let plan = plan_group_shards(&group_of_new, &ref_of_new, &labs, &[], 2, 0, 1000);

    // Contiguous cover of [0, n).
    let n = group_of_new.len() as u64;
    let mut expected = 0u64;
    for r in &plan.records {
        assert_eq!(r.row_start, expected, "records must be contiguous");
        assert!(r.row_stop > r.row_start);
        expected = r.row_stop;
    }
    assert_eq!(expected, n);

    // Shard ids are non-decreasing and each shard_start matches a record start.
    let mut prev_shard = 0u32;
    for r in &plan.records {
        assert!(r.shard >= prev_shard);
        prev_shard = r.shard;
    }
    for &start in &plan.shard_starts {
        assert!(plan.records.iter().any(|r| r.row_start == start));
    }
    assert_eq!(plan.shard_starts.len() as u32 + 1, plan.n_shards);
}

#[test]
fn sidecar_json_shape_matches_contract() {
    let group_of_new = [0, 1, 1];
    let ref_of_new = [true, false, false];
    let labs = vec!["non-targeting".to_string(), "MYC".to_string()];
    let plan = plan_group_shards(&group_of_new, &ref_of_new, &labs, &[], 100, 0, 1000);
    let v = plan.to_sidecar_json("target_gene", &["non-targeting".to_string()]);

    assert_eq!(v["group_by"], "target_gene");
    assert_eq!(v["reference_shard"], 0);
    assert_eq!(v["reference_labels"][0], "non-targeting");
    let recs = v["records"].as_array().unwrap();
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0]["role"], "reference");
    assert_eq!(recs[0]["label"], "non-targeting");
    assert_eq!(recs[1]["role"], "group");
    assert_eq!(recs[1]["label"], "MYC");
}
