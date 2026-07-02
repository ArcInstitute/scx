//! Unit tests for the F2 `GroupIndex` reader.

use super::*;

fn payload() -> Vec<u8> {
    let v = serde_json::json!({
        "group_by": "target_gene",
        "reference_shard": 0,
        "reference_labels": ["non-targeting"],
        "records": [
            {"label": "non-targeting", "shard": 0, "row_start": 0,  "row_stop": 50, "role": "reference"},
            {"label": "MYC",           "shard": 1, "row_start": 50, "row_stop": 60, "role": "group"},
            {"label": "TP53",          "shard": 1, "row_start": 60, "row_stop": 65, "role": "group"},
            {"label": "GATA1",         "shard": 2, "row_start": 65, "row_stop": 90, "role": "group"},
        ]
    });
    serde_json::to_vec(&v).unwrap()
}

#[test]
fn parses_payload_and_fields() {
    let gi = GroupIndex::from_bytes(&payload()).unwrap();
    assert_eq!(gi.group_by, "target_gene");
    assert_eq!(gi.reference_shard, Some(0));
    assert_eq!(gi.reference_labels, vec!["non-targeting".to_string()]);
    assert_eq!(gi.records().len(), 4);
    assert_eq!(gi.labels(), vec!["GATA1", "MYC", "TP53", "non-targeting"]);
}

#[test]
fn record_lookup_and_ranges() {
    let gi = GroupIndex::from_bytes(&payload()).unwrap();
    let myc = gi.record("MYC").unwrap();
    assert_eq!((myc.row_start, myc.row_stop, myc.shard), (50, 60, 1));
    assert_eq!(myc.role, GroupRole::Group);
    assert!(gi.record("nope").is_none());
}

#[test]
fn reference_range_is_leading_contiguous_region() {
    let gi = GroupIndex::from_bytes(&payload()).unwrap();
    assert_eq!(gi.reference_range(), Some((0, 50)));
}

#[test]
fn split_label_non_reference_wins() {
    // "shared" appears as both reference and group → record() returns the group.
    let v = serde_json::json!({
        "group_by": "g",
        "reference_shard": 0,
        "reference_labels": [],
        "records": [
            {"label": "shared", "shard": 0, "row_start": 0,  "row_stop": 10, "role": "reference"},
            {"label": "shared", "shard": 1, "row_start": 10, "row_stop": 20, "role": "group"},
        ]
    });
    let gi = GroupIndex::from_bytes(&serde_json::to_vec(&v).unwrap()).unwrap();
    let rec = gi.record("shared").unwrap();
    assert_eq!(
        rec.role,
        GroupRole::Group,
        "non-reference must win on collision"
    );
    assert_eq!(rec.shard, 1);
    assert_eq!(gi.reference_range(), Some((0, 10)));
}

#[test]
fn close_matches_resemble_difflib() {
    let gi = GroupIndex::from_bytes(&payload()).unwrap();
    let m = gi.close_matches("MYCN", 5);
    assert!(
        m.contains(&"MYC".to_string()),
        "MYC must be a close match for MYCN, got {m:?}"
    );
    // A wildly different string yields nothing above cutoff.
    assert!(gi.close_matches("zzzzzzzz", 5).is_empty());
}

#[test]
fn shard_handles_exclude_reference_and_localize() {
    let gi = GroupIndex::from_bytes(&payload()).unwrap();
    let handles = gi.shard_handles();
    // Shards 1 and 2 (reference shard 0 excluded).
    assert_eq!(handles.len(), 2);
    assert_eq!(handles[0].shard_index, 1);
    assert_eq!((handles[0].global_start, handles[0].global_stop), (50, 65));
    // MYC local [0,10), TP53 local [10,15) within shard 1.
    assert_eq!(
        handles[0].groups,
        vec![("MYC".to_string(), 0, 10), ("TP53".to_string(), 10, 15)]
    );
    assert_eq!(handles[1].shard_index, 2);
    assert_eq!((handles[1].global_start, handles[1].global_stop), (65, 90));
}

#[test]
fn shard_handle_range_resolves_global_offsets() {
    // 7.1d: GroupShardHandle::range maps a label to its global [start, stop).
    let gi = GroupIndex::from_bytes(&payload()).unwrap();
    let handles = gi.shard_handles();
    // Shard 1 holds MYC (global 50..60) and TP53 (global 60..65).
    assert_eq!(handles[0].range("MYC"), Some((50, 60)));
    assert_eq!(handles[0].range("TP53"), Some((60, 65)));
    // A label not in this shard returns None.
    assert_eq!(handles[0].range("GATA1"), None);
    assert_eq!(handles[1].range("GATA1"), Some((65, 90)));
}

#[test]
fn malformed_payload_errors() {
    assert!(GroupIndex::from_bytes(b"not json").is_err());
    assert!(GroupIndex::from_bytes(b"{}").is_err());
}

#[test]
fn missing_record_field_is_a_hard_error() {
    // 7.1e: every wire field is required by construction. A record missing
    // `shard` (or any other field) must fail to deserialize rather than
    // silently defaulting.
    for missing in ["shard", "row_start", "row_stop", "label", "role"] {
        let mut rec = serde_json::Map::new();
        for (k, v) in [
            ("label", serde_json::json!("MYC")),
            ("shard", serde_json::json!(1)),
            ("row_start", serde_json::json!(0)),
            ("row_stop", serde_json::json!(10)),
            ("role", serde_json::json!("group")),
        ] {
            if k != missing {
                rec.insert(k.to_string(), v);
            }
        }
        let v = serde_json::json!({
            "group_by": "g",
            "reference_shard": serde_json::Value::Null,
            "reference_labels": [],
            "records": [serde_json::Value::Object(rec)],
        });
        let bytes = serde_json::to_vec(&v).unwrap();
        assert!(
            GroupIndex::from_bytes(&bytes).is_err(),
            "record missing '{missing}' must error"
        );
    }
}
