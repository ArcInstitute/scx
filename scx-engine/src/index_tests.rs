use super::*;
use arrow::array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::io::Cursor;
use std::sync::Arc;

use crate::error::EngineError;

// -------------------------------------------------------------------
// C2 Tests: Serialization round-trips
// -------------------------------------------------------------------

#[test]
fn roundtrip_empty_index() {
    let index = PredicateIndex {
        version: 1,
        columns: vec![],
    };
    let mut buf = Vec::new();
    index.write_to(&mut buf).unwrap();

    let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(decoded, index);
}

// -------------------------------------------------------------------
// Phase 5b — v1 / v2 encoding boundary tests
// -------------------------------------------------------------------

/// Below the v1 narrow-counter ceiling: writer must pick v1
/// (one-byte version = 1) and round-trip cleanly through the v1 reader.
#[test]
fn write_to_picks_v1_when_counters_fit() {
    let index = PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "cell_type".to_string(),
            entries: vec![CategoricalEntry {
                value: "B cell".to_string(),
                shard_ranges: vec![ShardRange {
                    shard_id: 0,
                    row_start: 0,
                    row_end: 100,
                }],
            }],
        })],
    };
    let mut buf = Vec::new();
    index.write_to(&mut buf).unwrap();
    assert_eq!(buf[0], 1, "small index must serialise as v1");

    let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(decoded.version, 1);
    assert_eq!(decoded.columns.len(), 1);
}

/// Above the v1 per-value range count ceiling (u16::MAX = 65,535):
/// writer must auto-route to v2 (one-byte version = 2) and the
/// round-tripped values must match. This is the most plausibly-hit
/// v2 trigger — a single ubiquitous categorical value crossing
/// `u16::MAX` shard ranges (≈ 1.07B rows at 16384-row shards).
#[test]
fn write_to_auto_picks_v2_when_shard_ranges_exceed_u16() {
    let n_ranges = (u16::MAX as usize) + 5;
    let shard_ranges: Vec<ShardRange> = (0..n_ranges)
        .map(|i| ShardRange {
            shard_id: i as u32,
            row_start: 0,
            row_end: 16_384,
        })
        .collect();
    let index = PredicateIndex {
        version: 1, // hint ignored — auto-routing wins
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "cell_type".to_string(),
            entries: vec![CategoricalEntry {
                value: "B cell".to_string(),
                shard_ranges,
            }],
        })],
    };
    let mut buf = Vec::new();
    index.write_to(&mut buf).unwrap();
    assert_eq!(buf[0], 2, "over-u16 shard_ranges must promote to v2");

    let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(decoded.version, 2);
    match &decoded.columns[0] {
        IndexedColumn::Categorical(cat) => {
            assert_eq!(cat.entries[0].shard_ranges.len(), n_ranges);
            assert_eq!(
                cat.entries[0].shard_ranges[n_ranges - 1].shard_id,
                (n_ranges - 1) as u32
            );
        }
        _ => panic!("expected Categorical"),
    }

    // Full structural round-trip: any field-order / width regression
    // in `read_from_v2` for the categorical branch fails here.
    // `write_to` ignores `self.version`; the on-disk byte is the
    // source of truth, so set the expected version to 2 to compare.
    let expected = PredicateIndex {
        version: 2,
        columns: index.columns.clone(),
    };
    assert_eq!(decoded, expected);
}

/// Above the v1 numeric column name length ceiling (u16::MAX): writer
/// must auto-route to v2 and the round-tripped numeric B+ tree
/// (internal pages + leaf pages + entries) must compare structurally
/// equal. Covers the numeric branch of `read_from_v2`
/// (lines 644–696) — `write_to_auto_picks_v2_when_shard_ranges_exceed_u16`
/// only exercises the categorical branch.
#[test]
fn roundtrip_numeric_index_v2() {
    // Tiny but structurally complete B+ tree: 1 internal page pointing
    // at 2 leaf pages, each with 2 entries. Forced onto the v2 path by
    // a numeric column name longer than u16::MAX.
    let long_name = "n".repeat((u16::MAX as usize) + 1);
    let index = PredicateIndex {
        version: 1, // hint ignored — auto-routing wins
        columns: vec![IndexedColumn::Numeric(NumericIndex {
            column_name: long_name,
            fanout: 64,
            internal_pages: vec![InternalPage {
                n_keys: 1,
                keys: vec![5.0],
                children: vec![0, 1],
            }],
            leaf_pages: vec![
                LeafPage {
                    entries: vec![
                        NumericLeafEntry {
                            min_value: 0.0,
                            max_value: 2.5,
                            shard_id: 0,
                            row_start: 0,
                            row_end: 100,
                        },
                        NumericLeafEntry {
                            min_value: 2.5,
                            max_value: 5.0,
                            shard_id: 0,
                            row_start: 100,
                            row_end: 200,
                        },
                    ],
                },
                LeafPage {
                    entries: vec![
                        NumericLeafEntry {
                            min_value: 5.0,
                            max_value: 7.5,
                            shard_id: 1,
                            row_start: 0,
                            row_end: 50,
                        },
                        NumericLeafEntry {
                            min_value: 7.5,
                            max_value: 10.0,
                            shard_id: 1,
                            row_start: 50,
                            row_end: 150,
                        },
                    ],
                },
            ],
        })],
    };

    let mut buf = Vec::new();
    index.write_to(&mut buf).unwrap();
    assert_eq!(buf[0], 2, "over-u16 numeric column_name must promote to v2");

    let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
    let expected = PredicateIndex {
        version: 2,
        columns: index.columns.clone(),
    };
    assert_eq!(decoded, expected);
}

/// `requires_v2_encoding` is the routing oracle for `write_to`. Each
/// widened v1→v2 field should independently trip the oracle so a future
/// regression that narrows one but not the others is caught.
///
/// **Coverage:** the oracle has 10 distinct return-`true` paths. Five
/// are exercised below (cheap to materialise); the other five gate on
/// `Vec::len() > u32::MAX`, which would require allocating > 4B
/// elements and is infeasible to construct in a unit test. The
/// inspected-only triggers, with the line in `requires_v2_encoding`
/// that handles each:
///
/// - `cat.entries.len() > u32::MAX` (inspected at `requires_v2_encoding`, line 180)
/// - `leaf_pages.len() > u32::MAX` (line 196)
/// - `internal_pages.len() > u32::MAX` (line 199)
/// - per-leaf-page `entries.len() > u32::MAX` (line 204)
/// - summed numeric `total_entries > u32::MAX` (line 209)
///
/// If `requires_v2_encoding` is refactored, audit those five paths
/// manually and treat them as code-review coverage, not test coverage.
#[test]
fn requires_v2_encoding_triggers_on_each_widened_field() {
    // Baseline: empty index stays v1.
    assert!(!requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: vec![],
    }));

    let cat_entry = |value: String, n_ranges: usize| CategoricalEntry {
        value,
        shard_ranges: (0..n_ranges)
            .map(|i| ShardRange {
                shard_id: i as u32,
                row_start: 0,
                row_end: 0,
            })
            .collect(),
    };

    // a) categorical value length > u16::MAX → v2.
    assert!(requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "c".to_string(),
            entries: vec![cat_entry("x".repeat((u16::MAX as usize) + 1), 0)],
        })],
    }));

    // b) per-value shard_ranges count > u16::MAX → v2 (the most
    //    plausibly-hit trigger at multi-billion-row obs scale).
    assert!(requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "c".to_string(),
            entries: vec![cat_entry("v".to_string(), (u16::MAX as usize) + 1)],
        })],
    }));

    // c) categorical column name length > u16::MAX → v2.
    assert!(requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "n".repeat((u16::MAX as usize) + 1),
            entries: vec![],
        })],
    }));

    // d) total columns count > u16::MAX → v2. Cheap because each
    //    column is an empty Categorical placeholder.
    let many_columns: Vec<IndexedColumn> = (0..(u16::MAX as usize) + 1)
        .map(|_| {
            IndexedColumn::Categorical(CategoricalIndex {
                column_name: String::new(),
                entries: Vec::new(),
            })
        })
        .collect();
    assert!(requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: many_columns,
    }));

    // e) numeric column name length > u16::MAX → v2. Independent of
    //    the categorical paths above.
    assert!(requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Numeric(NumericIndex {
            column_name: "n".repeat((u16::MAX as usize) + 1),
            fanout: 64,
            internal_pages: Vec::new(),
            leaf_pages: Vec::new(),
        })],
    }));

    // f) negative: small categorical stays v1.
    assert!(!requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "small".to_string(),
            entries: vec![cat_entry("v".to_string(), 4)],
        })],
    }));

    // g) negative: small numeric stays v1.
    assert!(!requires_v2_encoding(&PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Numeric(NumericIndex {
            column_name: "score".to_string(),
            fanout: 64,
            internal_pages: Vec::new(),
            leaf_pages: vec![LeafPage {
                entries: vec![NumericLeafEntry {
                    min_value: 0.0,
                    max_value: 1.0,
                    shard_id: 0,
                    row_start: 0,
                    row_end: 1,
                }],
            }],
        })],
    }));
}

/// Reader must reject an unknown version byte cleanly (defense against
/// future format changes / corruption / fuzz inputs).
#[test]
fn read_from_rejects_unknown_version() {
    let buf: [u8; 1] = [99];
    let err = PredicateIndex::read_from(&mut Cursor::new(&buf[..]))
        .expect_err("unknown version must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown predicate index version"),
        "expected unknown-version error, got: {msg}"
    );
}

/// Regression: a 10-byte malformed input that declared
/// `n_cat_entries = 0x2d000000` (~755 million) used to trigger a
/// ~36 GB `Vec::with_capacity` and OOM the process. Found by the
/// Phase 9 `fuzz_predicate_index` libfuzzer target on its first
/// 10-second run. The reader now rejects the input with
/// `InvalidData` via [`MAX_CATEGORICAL_ENTRIES`].
#[test]
fn read_from_rejects_oversized_categorical_count() {
    // version | n_columns(LE) | name_len(LE) | column_type | n_entries(LE)
    //   0x01  |   0x000a      |   0x0000     |    0x00     |  0x2d000000
    // Version pinned to 1 so the v1 dispatcher (which owns
    // `MAX_CATEGORICAL_ENTRIES`) runs; unknown version bytes are
    // rejected separately by `read_from`.
    let crash_input: [u8; 10] = [0x01, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2d];
    let err = PredicateIndex::read_from(&mut Cursor::new(&crash_input[..]))
        .expect_err("oversized n_entries must be rejected, not allocated");
    let msg = format!("{err}");
    assert!(
        msg.contains("categorical entries") && msg.contains("exceeds maximum"),
        "expected 'categorical entries ... exceeds maximum' error, got: {msg}"
    );
}

#[test]
fn roundtrip_categorical_index() {
    let index = PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "cell_type".to_string(),
            entries: vec![
                CategoricalEntry {
                    value: "B cell".to_string(),
                    shard_ranges: vec![
                        ShardRange {
                            shard_id: 0,
                            row_start: 10,
                            row_end: 20,
                        },
                        ShardRange {
                            shard_id: 2,
                            row_start: 0,
                            row_end: 15,
                        },
                    ],
                },
                CategoricalEntry {
                    value: "NK cell".to_string(),
                    shard_ranges: vec![
                        ShardRange {
                            shard_id: 1,
                            row_start: 5,
                            row_end: 30,
                        },
                        ShardRange {
                            shard_id: 3,
                            row_start: 0,
                            row_end: 50,
                        },
                        ShardRange {
                            shard_id: 4,
                            row_start: 10,
                            row_end: 40,
                        },
                    ],
                },
                CategoricalEntry {
                    value: "T cell".to_string(),
                    shard_ranges: vec![
                        ShardRange {
                            shard_id: 0,
                            row_start: 0,
                            row_end: 10,
                        },
                        ShardRange {
                            shard_id: 1,
                            row_start: 0,
                            row_end: 5,
                        },
                        ShardRange {
                            shard_id: 2,
                            row_start: 15,
                            row_end: 50,
                        },
                    ],
                },
            ],
        })],
    };
    let mut buf = Vec::new();
    index.write_to(&mut buf).unwrap();

    let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(decoded, index);
}

#[test]
fn roundtrip_numeric_index() {
    let index = PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Numeric(NumericIndex {
            column_name: "n_genes".to_string(),
            fanout: 4,
            internal_pages: vec![InternalPage {
                n_keys: 1,
                keys: vec![500.0],
                children: vec![0, 1],
            }],
            leaf_pages: vec![
                LeafPage {
                    entries: vec![
                        NumericLeafEntry {
                            min_value: 100.0,
                            max_value: 200.0,
                            shard_id: 0,
                            row_start: 0,
                            row_end: 50,
                        },
                        NumericLeafEntry {
                            min_value: 200.0,
                            max_value: 500.0,
                            shard_id: 1,
                            row_start: 0,
                            row_end: 100,
                        },
                    ],
                },
                LeafPage {
                    entries: vec![
                        NumericLeafEntry {
                            min_value: 500.0,
                            max_value: 800.0,
                            shard_id: 2,
                            row_start: 0,
                            row_end: 80,
                        },
                        NumericLeafEntry {
                            min_value: 800.0,
                            max_value: 5000.0,
                            shard_id: 3,
                            row_start: 0,
                            row_end: 200,
                        },
                    ],
                },
            ],
        })],
    };
    let mut buf = Vec::new();
    index.write_to(&mut buf).unwrap();

    let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(decoded, index);
}

#[test]
fn roundtrip_mixed_index() {
    let index = PredicateIndex {
        version: 1,
        columns: vec![
            IndexedColumn::Categorical(CategoricalIndex {
                column_name: "cell_type".to_string(),
                entries: vec![CategoricalEntry {
                    value: "T cell".to_string(),
                    shard_ranges: vec![ShardRange {
                        shard_id: 0,
                        row_start: 0,
                        row_end: 10,
                    }],
                }],
            }),
            IndexedColumn::Numeric(NumericIndex {
                column_name: "n_genes".to_string(),
                fanout: 4,
                internal_pages: vec![],
                leaf_pages: vec![LeafPage {
                    entries: vec![NumericLeafEntry {
                        min_value: 100.0,
                        max_value: 5000.0,
                        shard_id: 0,
                        row_start: 0,
                        row_end: 100,
                    }],
                }],
            }),
        ],
    };
    let mut buf = Vec::new();
    index.write_to(&mut buf).unwrap();
    let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(decoded, index);
}

// -------------------------------------------------------------------
// C3 Tests: Index construction
// -------------------------------------------------------------------

fn make_obs_batch() -> RecordBatch {
    // 12 rows, 2 shards: shard 0 = rows 0..6, shard 1 = rows 6..12
    let schema = Schema::new(vec![
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("n_genes", DataType::Int32, false),
    ]);
    let cell_types = StringArray::from(vec![
        "T cell", "B cell", "T cell", "NK cell", "B cell", "T cell", "NK cell", "T cell", "B cell",
        "NK cell", "NK cell", "T cell",
    ]);
    let n_genes = Int32Array::from(vec![
        200, 300, 150, 450, 500, 250, 600, 100, 350, 700, 800, 400,
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(cell_types), Arc::new(n_genes)],
    )
    .unwrap()
}

#[test]
fn streaming_builder_matches_batch_auto_detect() {
    // Same obs batch, two construction paths:
    //   (a) batch mode via build_obs_predicate_index_bytes
    //   (b) streaming mode via ObsPredicateIndexBuilder with the
    //       batch split into two shards (rows 0..6, rows 6..12)
    // Both must produce identical serialised bytes for the same
    // shard_row_ranges.
    let batch = make_obs_batch();
    let shard_row_ranges: Vec<(u64, u64)> = vec![(0, 6), (6, 12)];
    let options = PredicateIndexBuildOptions {
        forced_columns: Vec::new(),
        preset_columns: Vec::new(),
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };

    let mut batch_outcomes = Vec::new();
    let mut batch_indexed_names = Vec::new();
    let batch_bytes = build_obs_predicate_index_bytes(
        &batch,
        &shard_row_ranges,
        &options,
        &mut batch_outcomes,
        &mut batch_indexed_names,
    )
    .unwrap()
    .expect("batch-mode auto-detect should index both columns");

    let shard_a = batch.slice(0, 6);
    let shard_b = batch.slice(6, 6);
    let mut builder = ObsPredicateIndexBuilder::new(batch.schema(), &options).unwrap();
    builder.push_shard(&shard_a, 0).unwrap();
    builder.push_shard(&shard_b, 6).unwrap();
    let mut stream_outcomes = Vec::new();
    let mut stream_indexed_names = Vec::new();
    let stream_bytes = builder
        .finish(
            &shard_row_ranges,
            &mut stream_outcomes,
            &mut stream_indexed_names,
        )
        .unwrap()
        .expect("streaming auto-detect should index both columns");

    assert_eq!(
        batch_bytes, stream_bytes,
        "streaming builder must produce byte-identical output to batch path"
    );
    assert_eq!(batch_indexed_names, stream_indexed_names);
    assert!(stream_outcomes.is_empty());
}

#[test]
fn streaming_builder_named_forced_column() {
    // Force the `cell_type` column under named mode and verify the
    // streaming and batch builders agree.
    let batch = make_obs_batch();
    let shard_row_ranges: Vec<(u64, u64)> = vec![(0, 6), (6, 12)];
    let options = PredicateIndexBuildOptions {
        forced_columns: vec!["cell_type".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };

    let mut batch_outcomes = Vec::new();
    let mut batch_indexed_names = Vec::new();
    let batch_bytes = build_obs_predicate_index_bytes(
        &batch,
        &shard_row_ranges,
        &options,
        &mut batch_outcomes,
        &mut batch_indexed_names,
    )
    .unwrap()
    .unwrap();

    let mut builder = ObsPredicateIndexBuilder::new(batch.schema(), &options).unwrap();
    builder.push_shard(&batch.slice(0, 6), 0).unwrap();
    builder.push_shard(&batch.slice(6, 6), 6).unwrap();
    let mut stream_outcomes = Vec::new();
    let mut stream_indexed_names = Vec::new();
    let stream_bytes = builder
        .finish(
            &shard_row_ranges,
            &mut stream_outcomes,
            &mut stream_indexed_names,
        )
        .unwrap()
        .unwrap();

    assert_eq!(batch_bytes, stream_bytes);
    assert_eq!(batch_indexed_names, vec!["cell_type".to_string()]);
    assert_eq!(stream_indexed_names, vec!["cell_type".to_string()]);
}

#[test]
fn streaming_builder_rejects_out_of_order_shards() {
    let batch = make_obs_batch();
    let options = PredicateIndexBuildOptions {
        forced_columns: vec!["cell_type".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let mut builder = ObsPredicateIndexBuilder::new(batch.schema(), &options).unwrap();
    builder.push_shard(&batch.slice(0, 6), 0).unwrap();
    // Wrong offset — second shard should start at row 6, not 100.
    let err = builder.push_shard(&batch.slice(6, 6), 100).unwrap_err();
    assert!(matches!(err, EngineError::Generic(_)), "got: {err:?}");
}

#[test]
fn build_categorical_index_correct_entries() {
    let batch = make_obs_batch();
    let col = batch.column(0);
    let shard_ranges = vec![(0u64, 6u64), (6, 12)];

    let index = build_categorical_index(col, "cell_type", &shard_ranges);
    assert_eq!(index.column_name, "cell_type");
    assert_eq!(index.entries.len(), 3); // B cell, NK cell, T cell (sorted)
    assert_eq!(index.entries[0].value, "B cell");
    assert_eq!(index.entries[1].value, "NK cell");
    assert_eq!(index.entries[2].value, "T cell");

    // B cell appears in rows 1, 4 (shard 0) and 8 (shard 1)
    let b_cell = &index.entries[0];
    assert!(b_cell.shard_ranges.iter().any(|r| r.shard_id == 0));
    assert!(b_cell.shard_ranges.iter().any(|r| r.shard_id == 1));
}

#[test]
fn build_numeric_index_valid_btree() {
    let batch = make_obs_batch();
    let col = batch.column(1);
    let shard_ranges = vec![(0u64, 6u64), (6, 12)];

    let index = build_numeric_index(col, "n_genes", &shard_ranges, 4);
    assert_eq!(index.column_name, "n_genes");
    assert!(!index.leaf_pages.is_empty());

    // All leaf entries should have valid min <= max
    for page in &index.leaf_pages {
        for entry in &page.entries {
            assert!(entry.min_value <= entry.max_value);
        }
    }
}

#[test]
fn build_indexes_auto_detect() {
    let batch = make_obs_batch();
    let shard_ranges = vec![(0u64, 6u64), (6, 12)];

    // Empty indexed_columns → auto-detect
    let index = build_indexes(&batch, &shard_ranges, &[]).unwrap();
    assert!(!index.columns.is_empty());
    // Both cell_type and n_genes should be indexed (both have <1000 unique)
    assert_eq!(index.columns.len(), 2);
}

#[test]
fn build_indexes_high_cardinality_skipped() {
    // Create a column with >10K unique values
    let n = 11_000;
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    let ids = Int32Array::from((0..n).collect::<Vec<i32>>());
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids)]).unwrap();
    let shard_ranges = vec![(0u64, n as u64)];

    let index = build_indexes(&batch, &shard_ranges, &["id".to_string()]).unwrap();
    // Should be skipped because >10K unique
    assert!(index.columns.is_empty());
}

#[test]
fn build_indexes_empty_batch() {
    let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
    let batch = RecordBatch::new_empty(Arc::new(schema));
    let shard_ranges: Vec<(u64, u64)> = vec![];

    let index = build_indexes(&batch, &shard_ranges, &[]).unwrap();
    // Empty batch auto-detect should produce empty or index with empty entries
    assert!(
        index.columns.is_empty()
            || index.columns.iter().all(|c| match c {
                IndexedColumn::Categorical(cat) => cat.entries.is_empty(),
                IndexedColumn::Numeric(num) => num.leaf_pages.is_empty(),
            })
    );
}

#[test]
fn build_indexes_explicit_columns() {
    let batch = make_obs_batch();
    let shard_ranges = vec![(0u64, 6u64), (6, 12)];

    let index = build_indexes(&batch, &shard_ranges, &["cell_type".to_string()]).unwrap();
    assert_eq!(index.columns.len(), 1);
    match &index.columns[0] {
        IndexedColumn::Categorical(cat) => {
            assert_eq!(cat.column_name, "cell_type");
        }
        _ => panic!("expected categorical"),
    }
}

// --- Phase 5a ---

#[test]
fn index_preset_columns_known_names() {
    for name in ["cellxgene", "perturbseq", "training"] {
        let p = index_preset_columns(name).expect("known preset");
        assert!(!p.obs_columns.is_empty());
    }
    assert!(index_preset_columns("unknown").is_none());
}

#[test]
fn accel_ready_presets_imply_csc_auto() {
    // DE/pseudobulk-heavy presets imply a CSC sidecar.
    assert!(preset_implies_csc_auto("training"));
    assert!(preset_implies_csc_auto("perturbseq"));
    // Query/browse-oriented and unknown presets do not.
    assert!(!preset_implies_csc_auto("cellxgene"));
    assert!(!preset_implies_csc_auto("unknown"));
}

#[test]
fn resolve_csc_policy_explicit_wins_else_preset_default() {
    // An explicit value always wins, including "off" over an
    // accel-ready preset.
    assert_eq!(resolve_csc_policy(Some("off"), Some("training")), "off");
    assert_eq!(resolve_csc_policy(Some("always"), None), "always");
    // Unset + accel-ready preset upgrades to "auto".
    assert_eq!(resolve_csc_policy(None, Some("training")), "auto");
    assert_eq!(resolve_csc_policy(None, Some("perturbseq")), "auto");
    // Unset + query-oriented / unknown / no preset stays "off".
    assert_eq!(resolve_csc_policy(None, Some("cellxgene")), "off");
    assert_eq!(resolve_csc_policy(None, Some("unknown")), "off");
    assert_eq!(resolve_csc_policy(None, None), "off");
}

#[test]
fn build_obs_predicate_index_bytes_forced_missing_errors() {
    let batch = make_obs_batch();
    let shard_ranges = vec![(0u64, 12u64)];
    let opts = PredicateIndexBuildOptions {
        forced_columns: vec!["does_not_exist".to_string()],
        preset_columns: vec![],
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes =
        build_obs_predicate_index_bytes(&batch, &shard_ranges, &opts, &mut outcomes, &mut names)
            .unwrap();
    assert!(bytes.is_none());
    assert_eq!(outcomes.len(), 1);
    match &outcomes[0] {
        BuildOutcome::ForcedColumnError { column, reason } => {
            assert_eq!(column, "does_not_exist");
            assert_eq!(reason, &SkipReason::MissingColumn);
        }
        _ => panic!("expected ForcedColumnError"),
    }
}

#[test]
fn build_obs_predicate_index_bytes_preset_missing_warns() {
    let batch = make_obs_batch();
    let shard_ranges = vec![(0u64, 12u64)];
    let opts = PredicateIndexBuildOptions {
        forced_columns: vec![],
        preset_columns: vec!["cell_type".to_string(), "tissue".to_string()],
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes =
        build_obs_predicate_index_bytes(&batch, &shard_ranges, &opts, &mut outcomes, &mut names)
            .unwrap();
    // cell_type exists in the fixture so an index is produced.
    assert!(bytes.is_some());
    assert_eq!(names, vec!["cell_type".to_string()]);
    // tissue is missing → preset skip with typed MissingColumn reason
    assert!(outcomes.iter().any(|o| matches!(
        o,
        BuildOutcome::PresetSkipped { column, reason }
            if column == "tissue" && *reason == SkipReason::MissingColumn
    )));
}

/// Forced column whose cardinality exceeds `high_cardinality_threshold`
/// must surface as a `ForcedColumnError` with the typed
/// `SkipReason::HighCardinality` discriminant. The Display impl is
/// also exercised so callers re-using it for free-form messages
/// stay stable.
#[test]
fn build_obs_predicate_index_bytes_forced_high_cardinality_errors() {
    // 100 rows, 100 unique values in `cell_id`. Setting
    // `high_cardinality_threshold = 10` is enough to reject it.
    let n_rows: usize = 100;
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    let ids: Vec<String> = (0..n_rows).map(|i| format!("cell_{i}")).collect();
    let id_array = StringArray::from(ids);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(id_array)]).unwrap();
    let shard_ranges = vec![(0u64, n_rows as u64)];
    let opts = PredicateIndexBuildOptions {
        forced_columns: vec!["cell_id".to_string()],
        preset_columns: vec![],
        auto_threshold: 1000,
        high_cardinality_threshold: 10,
    };
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes =
        build_obs_predicate_index_bytes(&batch, &shard_ranges, &opts, &mut outcomes, &mut names)
            .unwrap();
    // No column ended up indexed (the only forced one was rejected).
    assert!(bytes.is_none());
    assert!(names.is_empty());
    assert_eq!(outcomes.len(), 1);
    match &outcomes[0] {
        BuildOutcome::ForcedColumnError { column, reason } => {
            assert_eq!(column, "cell_id");
            match reason {
                SkipReason::HighCardinality {
                    n_unique,
                    threshold,
                } => {
                    assert_eq!(*n_unique, n_rows);
                    assert_eq!(*threshold, 10);
                }
                other => panic!("expected HighCardinality reason, got {other:?}"),
            }
            // Display impl should mention both numbers so callers
            // that stringify for warnings get useful output.
            let s = reason.to_string();
            assert!(s.contains("100"), "expected '100' in {s}");
            assert!(s.contains("10"), "expected '10' in {s}");
        }
        _ => panic!("expected ForcedColumnError"),
    }
}

// -------------------------------------------------------------------
// Dictionary value types
//
// `pd.Categorical([1, 2, 3])` reaches Arrow as `Dictionary(_, Int64)`
// and `pd.Categorical([True, False])` as `Dictionary(_, Boolean)`.
// Both used to be classified as *categorical*, whose builder can only
// read string values out of a dictionary — so the column was indexed
// with zero entries, silently.
// -------------------------------------------------------------------

/// One obs batch with an integer-valued and a boolean-valued
/// categorical alongside a string one, keyed `Int32` exactly as
/// `widen_dictionary_keys` normalises them on disk.
fn make_dictionary_obs_batch() -> RecordBatch {
    use arrow::array::{BooleanArray, DictionaryArray, Int64Array};
    use arrow::datatypes::Int32Type;

    // 8 rows: batch ∈ {10, 20, 30}, flag ∈ {true, false}, cell_type ∈ {A, B}
    let batch_keys = Int32Array::from(vec![0, 1, 2, 0, 1, 2, 0, 1]);
    let batch_values: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30]));
    let batch_col: ArrayRef =
        Arc::new(DictionaryArray::<Int32Type>::try_new(batch_keys, batch_values).unwrap());

    let flag_keys = Int32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1]);
    let flag_values: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
    let flag_col: ArrayRef =
        Arc::new(DictionaryArray::<Int32Type>::try_new(flag_keys, flag_values).unwrap());

    let ct_keys = Int32Array::from(vec![0, 0, 1, 1, 0, 1, 0, 1]);
    let ct_values: ArrayRef = Arc::new(StringArray::from(vec!["A", "B"]));
    let ct_col: ArrayRef =
        Arc::new(DictionaryArray::<Int32Type>::try_new(ct_keys, ct_values).unwrap());

    let schema = Schema::new(vec![
        Field::new("batch", batch_col.data_type().clone(), true),
        Field::new("flag", flag_col.data_type().clone(), true),
        Field::new("cell_type", ct_col.data_type().clone(), true),
    ]);
    RecordBatch::try_new(Arc::new(schema), vec![batch_col, flag_col, ct_col]).unwrap()
}

/// An integer-valued categorical must land on the **numeric** index
/// with real leaf entries — not on the categorical index, which can
/// only extract string values and therefore produced an entry-less
/// `CategoricalIndex` that no query could ever hit.
#[test]
fn integer_categorical_is_indexed_numerically() {
    let batch = make_dictionary_obs_batch();
    let shard_ranges = vec![(0u64, 8u64)];
    let opts = PredicateIndexBuildOptions {
        forced_columns: vec!["batch".to_string()],
        preset_columns: vec![],
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes =
        build_obs_predicate_index_bytes(&batch, &shard_ranges, &opts, &mut outcomes, &mut names)
            .unwrap()
            .expect("forced integer categorical must produce an index");
    assert!(
        outcomes.is_empty(),
        "an integer categorical is indexable; got {outcomes:?}"
    );

    let index = PredicateIndex::read_from(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(index.columns.len(), 1);
    match &index.columns[0] {
        IndexedColumn::Numeric(num) => {
            assert_eq!(num.column_name, "batch");
            let n_entries: usize = num.leaf_pages.iter().map(|p| p.entries.len()).sum();
            assert!(
                n_entries > 0,
                "numeric index for an integer categorical must carry leaf entries"
            );
        }
        IndexedColumn::Categorical(cat) => panic!(
            "integer categorical was indexed as categorical with {} entries \
             (zero entries is the bug)",
            cat.entries.len()
        ),
    }
    assert_eq!(index.indexed_kind("batch"), Some(IndexKind::Numeric));
}

/// A boolean-valued categorical is no more indexable than a plain
/// `Boolean` column. It must say so as a typed outcome instead of
/// writing an empty categorical index and reporting success.
#[test]
fn boolean_categorical_is_reported_unsupported() {
    let batch = make_dictionary_obs_batch();
    let shard_ranges = vec![(0u64, 8u64)];
    let opts = PredicateIndexBuildOptions {
        forced_columns: vec!["flag".to_string()],
        preset_columns: vec![],
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes =
        build_obs_predicate_index_bytes(&batch, &shard_ranges, &opts, &mut outcomes, &mut names)
            .unwrap();
    assert!(bytes.is_none(), "nothing indexable → no index section");
    assert_eq!(outcomes.len(), 1, "expected one outcome, got {outcomes:?}");
    match &outcomes[0] {
        BuildOutcome::ForcedColumnError { column, reason } => {
            assert_eq!(column, "flag");
            assert!(
                matches!(reason, SkipReason::UnsupportedDtype(_)),
                "expected UnsupportedDtype, got {reason:?}"
            );
        }
        other => panic!("expected ForcedColumnError, got {other:?}"),
    }
}

/// Auto-detect must make the same three calls: numeric for the integer
/// categorical, categorical for the string one, and skip the boolean.
#[test]
fn auto_detect_routes_each_dictionary_value_type() {
    let batch = make_dictionary_obs_batch();
    let shard_ranges = vec![(0u64, 8u64)];
    let index = build_indexes(&batch, &shard_ranges, &[]).unwrap();
    assert_eq!(index.indexed_kind("batch"), Some(IndexKind::Numeric));
    assert_eq!(
        index.indexed_kind("cell_type"),
        Some(IndexKind::Categorical)
    );
    assert_eq!(index.indexed_kind("flag"), None);
    // The string categorical must still carry its values.
    assert_eq!(
        index
            .categorical_eq("cell_type", "A")
            .map(|r| !r.is_empty()),
        Some(true)
    );
}

/// The streaming builder shares the extractors with the batch builder
/// and must agree with it on all three value types — otherwise a
/// row-sharded atlas silently indexes differently from a small file.
#[test]
fn streaming_builder_agrees_on_dictionary_value_types() {
    let batch = make_dictionary_obs_batch();
    let opts = PredicateIndexBuildOptions {
        forced_columns: vec!["batch".to_string()],
        preset_columns: vec![],
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut builder = ObsPredicateIndexBuilder::new(batch.schema(), &opts).unwrap();
    builder.push_shard(&batch.slice(0, 4), 0).unwrap();
    builder.push_shard(&batch.slice(4, 4), 4).unwrap();
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes = builder
        .finish(&[(0, 4), (4, 8)], &mut outcomes, &mut names)
        .unwrap();
    assert!(outcomes.is_empty(), "got {outcomes:?}");
    let index = PredicateIndex::read_from(&mut Cursor::new(
        &bytes.expect("streaming builder must emit an index"),
    ))
    .unwrap();
    assert_eq!(index.indexed_kind("batch"), Some(IndexKind::Numeric));
    let n_entries: usize = match &index.columns[0] {
        IndexedColumn::Numeric(num) => num.leaf_pages.iter().map(|p| p.entries.len()).sum(),
        _ => 0,
    };
    assert!(n_entries > 0, "streaming numeric index must not be empty");
}

/// Index construction over a dictionary column must be linear in rows.
///
/// `dictionary_key_at` is called once per non-null row by both builders, so
/// resolving a key by any means that touches the whole key column makes the
/// build quadratic. `AnyDictionaryArray::normalized_keys()` is exactly such
/// a means — it allocates and fills a `Vec<usize>` over the entire column on
/// every call — and using it regressed *every* string categorical, not only
/// the newly supported numeric ones (measured 0.080 s / 0.269 s / 1.067 s at
/// 16k / 32k / 64k rows).
///
/// A wall-clock ratio would be flaky, so this asserts an absolute bound at a
/// row count where the two curves are orders of magnitude apart: linear is
/// milliseconds, while quadratic extrapolates to ~17 s from the numbers
/// above. Ten seconds separates them with room for a slow machine.
#[test]
fn dictionary_index_build_is_not_quadratic_in_rows() {
    use arrow::array::DictionaryArray;
    use arrow::datatypes::Int32Type;

    const N: usize = 256_000;
    let keys = Int32Array::from_iter_values((0..N).map(|i| (i % 4) as i32));
    let values: ArrayRef = Arc::new(StringArray::from(vec!["A", "B", "C", "D"]));
    let col: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap());

    let start = std::time::Instant::now();
    let cat = build_categorical_index(&col, "ct", &[(0u64, N as u64)]);
    let elapsed = start.elapsed();

    assert_eq!(cat.entries.len(), 4, "all four categories must be indexed");
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "building a {N}-row dictionary index took {elapsed:?}; a per-row scan of \
         the key column makes this quadratic"
    );
}

/// `widen_dictionary_keys` normalises on-disk keys to `Int32`, but the
/// builders also run on caller-supplied batches. Every Arrow key width
/// must decode; the string extractor used to handle only
/// `Int8`/`Int16`/`Int32` and returned `None` — an empty index — for
/// the rest.
#[test]
fn dictionary_keys_of_every_width_decode() {
    use arrow::array::{DictionaryArray, Int64Array, UInt8Array};
    use arrow::datatypes::{Int64Type, UInt8Type};

    let values: ArrayRef = Arc::new(StringArray::from(vec!["A", "B"]));

    let i64_col: ArrayRef = Arc::new(
        DictionaryArray::<Int64Type>::try_new(
            Int64Array::from(vec![0i64, 1, 0, 1]),
            values.clone(),
        )
        .unwrap(),
    );
    let u8_col: ArrayRef = Arc::new(
        DictionaryArray::<UInt8Type>::try_new(UInt8Array::from(vec![1u8, 0, 1, 0]), values.clone())
            .unwrap(),
    );

    for (label, col) in [("Int64 keys", i64_col), ("UInt8 keys", u8_col)] {
        let cat = build_categorical_index(&col, "c", &[(0u64, 4u64)]);
        assert_eq!(
            cat.entries.len(),
            2,
            "{label}: expected both categories, got {:?}",
            cat.entries.iter().map(|e| &e.value).collect::<Vec<_>>()
        );
    }
}

// Per-function message tests relocated from the production file (T5.6).
// `super::super::` reaches `index` (these gained one nesting level).
#[cfg(test)]
mod forced_column_missing_message_tests {
    use super::super::forced_column_missing_message;

    #[test]
    fn lists_available_columns() {
        let avail = vec![
            "total_counts".to_string(),
            "n_genes_by_counts".to_string(),
            "pct_counts_mt".to_string(),
        ];
        let msg = forced_column_missing_message("obs", "nonexistent_column", &avail);
        assert!(
            msg.contains("forced obs index column 'nonexistent_column'"),
            "{msg}"
        );
        assert!(msg.contains("missing column."), "{msg}");
        assert!(msg.contains("total_counts"), "{msg}");
        assert!(msg.contains("n_genes_by_counts"), "{msg}");
        assert!(!msg.contains("Did you mean"), "{msg}");
    }

    #[test]
    fn suggests_typo() {
        let avail = vec!["total_counts".to_string(), "n_genes_by_counts".to_string()];
        let msg = forced_column_missing_message("obs", "totl_counts", &avail);
        assert!(msg.contains("Did you mean 'total_counts'?"), "{msg}");
    }

    #[test]
    fn no_suggestion_when_far() {
        let avail = vec!["foo".to_string(), "bar".to_string()];
        let msg = forced_column_missing_message("obs", "cell_type", &avail);
        assert!(!msg.contains("Did you mean"), "{msg}");
        assert!(msg.contains("foo") && msg.contains("bar"), "{msg}");
    }

    #[test]
    fn handles_empty_available() {
        let msg = forced_column_missing_message("obs", "total_counts", &[]);
        assert!(
            msg.contains("Available obs columns: [] (this h5ad has no obs metadata)"),
            "{msg}"
        );
        assert!(!msg.contains("Did you mean"), "{msg}");
    }

    // N2-2026-05-21-Tier2: under the 64-column cap, lists with ≤ 64
    // columns show in full regardless of strsim — so users who type
    // scanpy-vocab columns (e.g. `total_counts`) on a Census atlas can
    // still see `raw_sum` in the rendered list. Pre-N2 this test
    // asserted `, ….` truncation; that gate moved to the > 64 path.
    #[test]
    fn shows_all_columns_when_under_cap() {
        let avail: Vec<String> = (0..20).map(|i| format!("col_{i}")).collect();
        let msg = forced_column_missing_message("var", "missing", &avail);
        assert!(
            !msg.contains(", \u{2026}"),
            "20 cols ≤ 64 cap should show all, not truncate: {msg}"
        );
        assert!(msg.contains("col_0") && msg.contains("col_19"), "{msg}");
    }

    // E1-2026-05-20-Tier2: with a strsim near-match the full list still
    // renders (no behavioural change from N2 — only the gating logic
    // changed, this case continues to show all).
    #[test]
    fn shows_all_columns_when_strsim_suggestion_present() {
        // 28 obs columns mirroring the census_500k.scx layout; `raw_sum`
        // is at index 11 (past the 8-column cutoff). The strsim match
        // for `raw_summ` should be `raw_sum`, which triggers show-all.
        let mut avail: Vec<String> = (0..28).map(|i| format!("col_{i:02}")).collect();
        avail[11] = "raw_sum".to_string();
        let msg = forced_column_missing_message("obs", "raw_summ", &avail);
        assert!(
            msg.contains("Did you mean 'raw_sum'?"),
            "suggestion gate: {msg}"
        );
        assert!(
            !msg.contains(", \u{2026}"),
            "show-all path should NOT emit the truncation marker: {msg}"
        );
        assert!(msg.contains("col_00"), "should show first column: {msg}");
        assert!(
            msg.contains("col_27"),
            "should show LAST column (past 8-col preview): {msg}"
        );
        assert!(msg.contains("raw_sum"), "{msg}");
    }

    // N2-2026-05-21-Tier2: even without a strsim suggestion, lists ≤ 64
    // columns show in full. This is the scanpy-vocab failure mode the
    // pre-N2 logic produced: `total_counts` on a Census schema has no
    // near-match, so `raw_sum` was hidden in the 8-column preview.
    #[test]
    fn previews_all_when_under_cap_regardless_of_strsim() {
        let avail: Vec<String> = (0..28).map(|i| format!("col_{i:02}")).collect();
        let msg = forced_column_missing_message("obs", "totally_unrelated", &avail);
        assert!(
            !msg.contains("Did you mean"),
            "no suggestion expected: {msg}"
        );
        assert!(
            !msg.contains(", \u{2026}"),
            "28 cols ≤ 64 cap should show all, not truncate: {msg}"
        );
        assert!(
            msg.contains("col_27"),
            "should show LAST column even with no strsim winner: {msg}"
        );
    }

    // The 64-column cap guards the worst-case payload size: an atlas
    // with > 64 obs columns falls back to the 8-column preview even
    // when a suggestion is present.
    #[test]
    fn cap_at_64_columns_falls_back_to_preview() {
        let mut avail: Vec<String> = (0..70).map(|i| format!("col_{i:03}")).collect();
        avail[50] = "raw_sum".to_string();
        let msg = forced_column_missing_message("obs", "raw_summ", &avail);
        assert!(msg.contains("Did you mean 'raw_sum'?"), "{msg}");
        assert!(
            msg.contains(", \u{2026}"),
            "should truncate past the 64-col cap: {msg}"
        );
        assert!(
            !msg.contains("col_050"),
            "the suggested column lives at index 50 — past the 8-col preview: {msg}"
        );
    }
}

#[cfg(test)]
mod forced_columns_missing_message_tests {
    use super::super::forced_columns_missing_message;

    fn census_obs_columns() -> Vec<String> {
        vec![
            "soma_joinid".to_string(),
            "dataset_id".to_string(),
            "cell_type".to_string(),
            "raw_sum".to_string(),
            "tissue".to_string(),
            "disease".to_string(),
        ]
    }

    #[test]
    fn aggregates_multiple_misses_with_suggestions() {
        // F6-2026-05-20-Tier2: both `raw_summ` and `cell_typ` should be
        // surfaced in a single error, each with its own strsim hint.
        let missing = vec!["raw_summ".to_string(), "cell_typ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &census_obs_columns());
        assert!(
            msg.contains("2 forced obs index columns are missing"),
            "should aggregate count + axis: {msg}"
        );
        assert!(
            msg.contains("- 'raw_summ': did you mean 'raw_sum'?"),
            "should include first typo + suggestion: {msg}"
        );
        assert!(
            msg.contains("- 'cell_typ': did you mean 'cell_type'?"),
            "should include second typo + suggestion: {msg}"
        );
        assert!(
            msg.contains("Available obs columns"),
            "should include available-columns footer: {msg}"
        );
    }

    #[test]
    fn aggregates_misses_without_near_suggestion() {
        let missing = vec![
            "totally_unrelated".to_string(),
            "another_unrelated".to_string(),
        ];
        let msg = forced_columns_missing_message("obs", &missing, &census_obs_columns());
        assert!(
            msg.contains("2 forced obs index columns are missing"),
            "{msg}"
        );
        // Both names without suggestions should appear on their own bullets.
        assert!(msg.contains("- 'totally_unrelated'"), "{msg}");
        assert!(msg.contains("- 'another_unrelated'"), "{msg}");
        assert!(
            !msg.contains("Did you mean") && !msg.contains("did you mean"),
            "no near match in either case: {msg}"
        );
    }

    #[test]
    fn single_miss_delegates_to_singular_helper() {
        // The singular path keeps the historical wording so we don't
        // gratuitously change byte-for-byte output of the single-miss
        // surface (the most common one in practice).
        let missing = vec!["raw_summ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &census_obs_columns());
        assert!(
            msg.contains("forced obs index column 'raw_summ': missing column."),
            "should use the singular wording: {msg}"
        );
        assert!(msg.contains("Did you mean 'raw_sum'?"), "{msg}");
        // The aggregate header MUST NOT appear when N == 1.
        assert!(
            !msg.contains("forced obs index columns are missing"),
            "should not emit the aggregate header for single miss: {msg}"
        );
    }

    #[test]
    fn empty_available_columns_is_handled() {
        let missing = vec!["raw_summ".to_string(), "cell_typ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &[]);
        assert!(
            msg.contains("2 forced obs index columns are missing"),
            "{msg}"
        );
        assert!(
            msg.contains("[] (this h5ad has no obs metadata)"),
            "should render the empty-axis footer: {msg}"
        );
    }

    // E1-2026-05-20-Tier2 + N2-2026-05-21-Tier2: aggregate path renders
    // the full column list when ≤ 64 columns regardless of strsim. This
    // case kept its assertion identical post-N2; only the rationale
    // narrowed (length cap, not strsim winner, is the gate).
    #[test]
    fn aggregate_shows_all_columns_when_any_miss_has_suggestion() {
        let mut avail: Vec<String> = (0..28).map(|i| format!("col_{i:02}")).collect();
        avail[11] = "raw_sum".to_string();
        avail[14] = "cell_type".to_string();
        let missing = vec!["raw_summ".to_string(), "cell_typ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &avail);
        assert!(
            msg.contains("did you mean 'raw_sum'?") && msg.contains("did you mean 'cell_type'?"),
            "both suggestions should render: {msg}"
        );
        assert!(
            !msg.contains(", \u{2026}"),
            "show-all path should NOT truncate: {msg}"
        );
        assert!(
            msg.contains("col_27"),
            "should show LAST column (past 8-col preview): {msg}"
        );
    }
}

#[cfg(test)]
mod column_not_found_message_tests {
    use super::super::column_not_found_message;

    // N2-2026-05-21-Tier2: a runtime `filter_obs("total_counts >= 500")`
    // on a CELLxGENE Census `.scx` must surface `raw_sum` even though
    // normalized Levenshtein("total_counts", "raw_sum") ≈ 0.083 — well
    // below the 0.6 strsim threshold. Pre-N2 the 29-column list was
    // truncated to 8 and the user never saw `raw_sum`.
    #[test]
    fn shows_full_list_when_no_strsim_winner_under_cap() {
        // 29 obs columns mirroring `census_500k_0521.scx`. `raw_sum` is
        // at index 23 (way past the 8-column preview), and no Census
        // column is within 0.6 normalized-Levenshtein of `total_counts`.
        let avail: Vec<String> = vec![
            "soma_joinid",
            "dataset_id",
            "assay",
            "assay_ontology_term_id",
            "cell_type",
            "cell_type_ontology_term_id",
            "development_stage",
            "development_stage_ontology_term_id",
            "disease",
            "disease_ontology_term_id",
            "donor_id",
            "is_primary_data",
            "observation_joinid",
            "self_reported_ethnicity",
            "self_reported_ethnicity_ontology_term_id",
            "sex",
            "sex_ontology_term_id",
            "suspension_type",
            "tissue",
            "tissue_ontology_term_id",
            "tissue_type",
            "tissue_general",
            "tissue_general_ontology_term_id",
            "raw_sum",
            "nnz",
            "raw_mean_nnz",
            "raw_variance_nnz",
            "n_measured_vars",
            "__index_level_0__",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let msg = column_not_found_message("obs", "total_counts", &avail);
        assert!(
            !msg.contains("Did you mean"),
            "no strsim winner expected for total_counts on Census schema: {msg}"
        );
        assert!(
            msg.contains("raw_sum"),
            "user must be able to see `raw_sum` to recover from the \
             scanpy-vocab mismatch: {msg}"
        );
        assert!(
            !msg.contains(", \u{2026}"),
            "29 cols ≤ 64 cap should show all, not truncate: {msg}"
        );
    }
}

// -------------------------------------------------------------------
// Query-time lookups: categorical_eq / indexed_kind / index_covers_all_obs
// -------------------------------------------------------------------
mod query_lookup_tests {
    use super::*;
    use crate::index::{index_covers_all_obs, IndexKind};

    fn cat(column_name: &str, entries: Vec<(&str, Vec<ShardRange>)>) -> IndexedColumn {
        IndexedColumn::Categorical(CategoricalIndex {
            column_name: column_name.to_string(),
            entries: entries
                .into_iter()
                .map(|(value, shard_ranges)| CategoricalEntry {
                    value: value.to_string(),
                    shard_ranges,
                })
                .collect(),
        })
    }

    fn sr(shard_id: u32, row_start: u32, row_end: u32) -> ShardRange {
        ShardRange {
            shard_id,
            row_start,
            row_end,
        }
    }

    fn sample_index() -> PredicateIndex {
        // entries MUST be sorted lexicographically by value (build_indexes invariant)
        PredicateIndex {
            version: 1,
            columns: vec![cat(
                "cell_type",
                vec![
                    ("B cell", vec![sr(0, 0, 5), sr(1, 2, 4)]),
                    ("T cell", vec![sr(0, 5, 10)]),
                ],
            )],
        }
    }

    #[test]
    fn categorical_eq_hit_miss_and_not_indexed() {
        let idx = sample_index();
        // hit
        assert_eq!(
            idx.categorical_eq("cell_type", "B cell"),
            Some(&[sr(0, 0, 5), sr(1, 2, 4)][..])
        );
        // indexed column, absent value -> Some(empty) (exact empty row-set)
        assert_eq!(idx.categorical_eq("cell_type", "NK cell"), Some(&[][..]));
        // column not indexed -> None (residual)
        assert_eq!(idx.categorical_eq("tissue", "blood"), None);
    }

    #[test]
    fn indexed_kind_reports_kind() {
        let idx = sample_index();
        assert_eq!(idx.indexed_kind("cell_type"), Some(IndexKind::Categorical));
        assert_eq!(idx.indexed_kind("tissue"), None);
    }

    #[test]
    fn covers_all_obs_true_when_index_reaches_n_obs() {
        let idx = sample_index();
        // obs shard 0 -> [0,10), shard 1 -> [10,20)
        let ranges = vec![(0u32, 0u64, 10u64), (1, 10, 20)];
        // index references up to shard 1 row_end 4 -> global 14; shard 0 row_end 10 -> 10
        // max covered = 14
        assert!(index_covers_all_obs(&idx, &ranges, 14));
        // stale: n_obs beyond covered (appended rows) -> false (full-scan fallback)
        assert!(!index_covers_all_obs(&idx, &ranges, 20));
    }

    #[test]
    fn covers_all_obs_false_for_empty_index() {
        let idx = PredicateIndex {
            version: 1,
            columns: vec![],
        };
        assert!(!index_covers_all_obs(&idx, &[(0, 0, 10)], 10));
    }
}

// ===========================================================================
// Numeric index size — one leaf entry per shard, not one per row
// ===========================================================================
//
// The numeric B+ tree's leaf entries are read by exactly two consumers, and
// both fold them to shard granularity on the spot: `derive_shard_column_stats`
// (per-shard `ColumnStat::MinMax`, which is what Level-1 pruning reads) and
// `max_covered_global_row` (per-shard max `row_end`, which is what
// `index_covers_all_obs` reads). `eval_rowset` returns `None` for every
// numeric operator and there is no range-lookup method, so nothing else ever
// sees a leaf. Emitting one leaf per *row* therefore buys nothing and costs
// 28 B/row on disk plus a whole-axis accumulator in memory.

/// The value at obs row `row` of the fixture column, for `n` rows.
///
/// `STRIDE` is coprime with every `n` used below, so this is a permutation of
/// `0..n` — all-distinct values in an order unrelated to row order. That
/// ordering is the whole point of the fixture. The old builder sorted by
/// value and merged two leaf entries only when rows adjacent in *value* order
/// were also adjacent in *row* order, so a monotonically increasing column
/// collapses to one entry per shard all by itself and hides the defect
/// completely. A real `total_counts` / `pct_counts_mt` column looks like this
/// one, not like a sorted range.
const STRIDE: usize = 7919;

fn fixture_value(row: usize, n: usize) -> f64 {
    ((row * STRIDE) % n) as f64 * 0.5
}

fn distinct_numeric_obs_batch(n: usize) -> RecordBatch {
    let col: ArrayRef = Arc::new(Float64Array::from_iter_values(
        (0..n).map(|i| fixture_value(i, n)),
    ));
    let schema = Schema::new(vec![Field::new("n_counts", DataType::Float64, true)]);
    RecordBatch::try_new(Arc::new(schema), vec![col]).unwrap()
}

/// The exact `(min, max)` of the fixture's values over a global row range.
fn fixture_min_max(rows: std::ops::Range<usize>, n: usize) -> (f64, f64) {
    rows.map(|r| fixture_value(r, n))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        })
}

fn n_counts_opts() -> PredicateIndexBuildOptions {
    PredicateIndexBuildOptions {
        forced_columns: vec!["n_counts".to_string()],
        preset_columns: vec![],
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    }
}

/// Push `batch` through the streaming builder in fixed-size chunks, then
/// finish against `shard_row_ranges`. `push_rows` is deliberately decoupled
/// from the shard ranges: the two partitions agree on the convert path but
/// not on the merge one, and the difference is load-bearing.
fn streaming_numeric_index(
    batch: &RecordBatch,
    push_rows: usize,
    shard_row_ranges: &[(u64, u64)],
) -> PredicateIndex {
    let mut builder = ObsPredicateIndexBuilder::new(batch.schema(), &n_counts_opts()).unwrap();
    let mut offset = 0usize;
    while offset < batch.num_rows() {
        let take = push_rows.min(batch.num_rows() - offset);
        builder
            .push_shard(&batch.slice(offset, take), offset as u64)
            .unwrap();
        offset += take;
    }
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes = builder
        .finish(shard_row_ranges, &mut outcomes, &mut names)
        .unwrap()
        .expect("a forced numeric column must produce an index");
    assert!(
        outcomes.is_empty(),
        "unexpected build outcomes: {outcomes:?}"
    );
    PredicateIndex::read_from(&mut Cursor::new(&bytes)).unwrap()
}

fn batch_numeric_index(batch: &RecordBatch, shard_row_ranges: &[(u64, u64)]) -> PredicateIndex {
    let mut outcomes = Vec::new();
    let mut names = Vec::new();
    let bytes = build_obs_predicate_index_bytes(
        batch,
        shard_row_ranges,
        &n_counts_opts(),
        &mut outcomes,
        &mut names,
    )
    .unwrap()
    .expect("a forced numeric column must produce an index");
    assert!(
        outcomes.is_empty(),
        "unexpected build outcomes: {outcomes:?}"
    );
    PredicateIndex::read_from(&mut Cursor::new(&bytes)).unwrap()
}

fn numeric_leaf_count(index: &PredicateIndex) -> usize {
    index
        .columns
        .iter()
        .map(|c| match c {
            IndexedColumn::Numeric(num) => num
                .leaf_pages
                .iter()
                .map(|p| p.entries.len())
                .sum::<usize>(),
            IndexedColumn::Categorical(_) => panic!("expected a numeric column"),
        })
        .sum()
}

/// Per-shard `(min, max)` as Level-1 pruning will actually see it.
fn min_max_per_shard(index: &PredicateIndex, n_shards: usize) -> Vec<Option<(f64, f64)>> {
    use scx_format_io::catalog::ColumnStat;
    derive_shard_column_stats(index, n_shards)
        .into_iter()
        .map(|stats| {
            stats.into_iter().find_map(|s| match s {
                ColumnStat::MinMax { min, max, .. } => Some((min, max)),
                _ => None,
            })
        })
        .collect()
}

const N_ROWS: usize = 200_000;
const N_SHARDS: usize = 4;
const SHARD_ROWS: usize = N_ROWS / N_SHARDS;

fn aligned_shard_ranges() -> Vec<(u64, u64)> {
    (0..N_SHARDS)
        .map(|s| ((s * SHARD_ROWS) as u64, ((s + 1) * SHARD_ROWS) as u64))
        .collect()
}

/// The finding. A 200k-row distinct numeric column across 4 shards must
/// produce **4** leaf entries — one per shard, which is all either consumer
/// can use. Before the fix it produced one per row.
#[test]
fn numeric_index_emits_one_leaf_entry_per_shard() {
    let batch = distinct_numeric_obs_batch(N_ROWS);
    let ranges = aligned_shard_ranges();

    let streaming = numeric_leaf_count(&streaming_numeric_index(&batch, SHARD_ROWS, &ranges));
    assert_eq!(
        streaming, N_SHARDS,
        "streaming builder emitted {streaming} leaf entries for {N_ROWS} rows across \
         {N_SHARDS} shards — the leaves are per-row, and every consumer folds them \
         to per-shard anyway"
    );

    let batched = numeric_leaf_count(&batch_numeric_index(&batch, &ranges));
    assert_eq!(
        batched, N_SHARDS,
        "batch builder emitted {batched} leaf entries for {N_ROWS} rows across \
         {N_SHARDS} shards"
    );
}

/// The disk half of the same finding, pinned independently of the entry
/// count: a `NumericLeafEntry` is 28 B on the wire, so a per-row index over
/// 200k rows is ~5.6 MB. Per-shard it is a few hundred bytes. Scaled to a
/// 50M-cell atlas the difference is ~1.4 GB per indexed numeric column.
#[test]
fn numeric_index_section_does_not_scale_with_rows() {
    let batch = distinct_numeric_obs_batch(N_ROWS);
    let ranges = aligned_shard_ranges();
    let mut builder = ObsPredicateIndexBuilder::new(batch.schema(), &n_counts_opts()).unwrap();
    let mut offset = 0usize;
    while offset < N_ROWS {
        builder
            .push_shard(&batch.slice(offset, SHARD_ROWS), offset as u64)
            .unwrap();
        offset += SHARD_ROWS;
    }
    let bytes = builder
        .finish(&ranges, &mut Vec::new(), &mut Vec::new())
        .unwrap()
        .expect("a forced numeric column must produce an index");
    assert!(
        bytes.len() < 4096,
        "numeric index section is {} B for {N_ROWS} rows across {N_SHARDS} shards; \
         it must be bounded by the shard count, not the row count",
        bytes.len()
    );
}

/// The two builders must fold to the same per-shard stats. The batch builder
/// has the whole column and the shard ranges in hand and stays exact; the
/// streaming one summarises. With the push partition aligned to the shard
/// partition — the convert / append / modify_metadata shape — they must agree
/// exactly, or a row-sharded atlas prunes differently from a small file.
#[test]
fn streaming_and_batch_numeric_indexes_agree_on_shard_stats() {
    let batch = distinct_numeric_obs_batch(N_ROWS);
    let ranges = aligned_shard_ranges();
    let streaming = min_max_per_shard(
        &streaming_numeric_index(&batch, SHARD_ROWS, &ranges),
        N_SHARDS,
    );
    let batched = min_max_per_shard(&batch_numeric_index(&batch, &ranges), N_SHARDS);
    assert_eq!(
        streaming, batched,
        "streaming and batch per-shard MinMax diverge"
    );
}

/// …and that shared answer must be the *exact* per-shard range, not merely a
/// sound superset. A fix that widened everywhere would still satisfy the
/// no-false-prune property while quietly destroying pruning power.
#[test]
fn numeric_index_min_max_is_exact_when_pushes_align() {
    let batch = distinct_numeric_obs_batch(N_ROWS);
    let ranges = aligned_shard_ranges();
    let got = min_max_per_shard(
        &streaming_numeric_index(&batch, SHARD_ROWS, &ranges),
        N_SHARDS,
    );
    let want: Vec<Option<(f64, f64)>> = (0..N_SHARDS)
        .map(|s| {
            Some(fixture_min_max(
                s * SHARD_ROWS..(s + 1) * SHARD_ROWS,
                N_ROWS,
            ))
        })
        .collect();
    assert_eq!(got, want);
}

/// The merge shape: the shard partition is only known at `finish`
/// (`merge.rs` builds `output_shard_row_ranges` while the builder is already
/// accumulating), so pushes need not align with it. A summary block that
/// straddles a shard boundary contributes its bounds to both shards.
///
/// That widening is only allowed in the safe direction. Level-1 pruning skips
/// a shard when the probe value falls outside `[min, max]`, so a **wider**
/// bound can only fail to prune — never prune a shard that holds a match.
/// This asserts both halves: containment (no false prune) and a bound on how
/// far the widening can reach (no collapse into a useless global range).
#[test]
fn straddling_pushes_widen_min_max_but_never_prune_a_matching_shard() {
    // Boundaries deliberately co-prime with the summary block size, so every
    // one of them falls strictly inside a block.
    const N: usize = 20_004;
    const BLOCK: usize = NUMERIC_BLOCK_ROWS as usize;
    let bounds = [0usize, 5001, 10002, 15003, N];
    let ranges: Vec<(u64, u64)> = bounds
        .windows(2)
        .map(|w| (w[0] as u64, w[1] as u64))
        .collect();
    let batch = distinct_numeric_obs_batch(N);
    // One push covering every shard — what a caller that does not split looks
    // like. (No in-tree writer pushes this way any more; see `push_shard_split`.)
    let index = streaming_numeric_index(&batch, N, &ranges);
    let got = min_max_per_shard(&index, ranges.len());

    for (s, &(start, end)) in ranges.iter().enumerate() {
        let (start, end) = (start as usize, end as usize);
        let (min, max) = got[s].unwrap_or_else(|| panic!("shard {s} lost its MinMax entirely"));
        // No false prune: every value physically in this shard is inside the
        // recorded bounds, so `Eq` on any of them cannot skip the shard.
        let (true_min, true_max) = fixture_min_max(start..end, N);
        assert!(
            min <= true_min && max >= true_max,
            "shard {s} bounds [{min}, {max}] exclude its own values \
             [{true_min}, {true_max}] — Level-1 pruning would skip a shard \
             that holds a match"
        );
        // Bounded widening: at most one summary block's worth of neighbouring
        // rows can leak in from either side.
        let (reach_min, reach_max) =
            fixture_min_max(start.saturating_sub(BLOCK - 1)..(end + BLOCK - 1).min(N), N);
        assert!(
            min >= reach_min && max <= reach_max,
            "shard {s} bounds [{min}, {max}] reach past one summary block \
             beyond [{reach_min}, {reach_max}]"
        );
    }
}

/// Coverage must stay exactly as sound as it was. `index_covers_all_obs` is
/// what decides whether a post-`append` index may be trusted for row-set
/// pushdown, so a summary that claimed rows it has no value for would let a
/// stale index answer a query. Under-claiming is safe (full-scan fallback);
/// over-claiming is not.
///
/// Here the last three rows are null, so the index reaches global row 9 of 12
/// — one past the last valued row — and no further.
#[test]
fn numeric_index_coverage_stops_at_the_last_valued_row() {
    let col: ArrayRef = Arc::new(Float64Array::from(
        (0..12)
            .map(|i| if i < 9 { Some(i as f64) } else { None })
            .collect::<Vec<_>>(),
    ));
    let schema = Schema::new(vec![Field::new("n_counts", DataType::Float64, true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![col]).unwrap();
    let ranges = vec![(0u64, 6u64), (6, 12)];
    let index = streaming_numeric_index(&batch, 6, &ranges);

    let obs_ranges = [(0u32, 0u64, 6u64), (1u32, 6u64, 12u64)];
    assert!(
        index_covers_all_obs(&index, &obs_ranges, 9),
        "the index must cover every row that carries a value"
    );
    assert!(
        !index_covers_all_obs(&index, &obs_ranges, 10),
        "the index must not claim coverage of a trailing null tail"
    );
}

/// The conversion entry points restore `push_shard`'s contract on their
/// callers' behalf.
///
/// The numeric accumulator summarises *within* a push, so a caller that hands
/// the whole obs axis to one `push_shard` call while telling `finish` there
/// are several shards would give every shard the same widened `[min, max]` —
/// sound, but it erases exactly the Level-1 pruning the index exists for.
/// Both conversion entry points did that: the batch one wraps obs in
/// `iter::once`, and `compact` / `sort` push *input* shards against *output*
/// shard ranges that a reshape has moved. Every writer whose pushes can
/// straddle uses `push_shard_split` instead; merge's sorted path still calls
/// `push_shard`, correctly, because it flushes obs and X at the same size.
///
/// `n_counts` is 10..13 in shard 0 and 1000..1003 in shard 1, so an unsplit
/// push shows up immediately as shard 0 claiming a max of 1003. That both
/// conversion entry points route through the splitter is covered end-to-end
/// by `tests/pushdown_skip.rs::numeric_filter_skips_shard_out_of_range`.
#[test]
fn a_multi_shard_push_is_split_before_it_reaches_the_accumulator() {
    use arrow::array::Int64Array;

    let n_counts: ArrayRef = Arc::new(Int64Array::from(vec![
        10i64, 11, 12, 13, 1000, 1001, 1002, 1003,
    ]));
    let schema = Schema::new(vec![Field::new("n_counts", DataType::Int64, false)]);
    let obs = RecordBatch::try_new(Arc::new(schema), vec![n_counts]).unwrap();
    let ranges = [(0u64, 4u64), (4u64, 8u64)];

    let mut builder = ObsPredicateIndexBuilder::new(obs.schema(), &n_counts_opts()).unwrap();
    // The whole axis in a single push, against a two-shard range table.
    builder.push_shard_split(&obs, 0, &ranges).unwrap();
    let bytes = builder
        .finish(&ranges, &mut Vec::new(), &mut Vec::new())
        .unwrap()
        .expect("a forced numeric column must produce an index");
    let index = PredicateIndex::read_from(&mut Cursor::new(&bytes)).unwrap();

    assert_eq!(
        min_max_per_shard(&index, 2),
        vec![Some((10.0, 13.0)), Some((1000.0, 1003.0))],
        "a push spanning both shards was summarised as one block, so both \
         shards recorded the whole column's range and neither can be pruned"
    );
}

/// The fold must be linear in spans, not spans × shards.
///
/// `numeric_leaves_from_spans` maps each span onto the shards it overlaps, and
/// the batch builder emits **one span per valued row** — so an exhaustive
/// per-span scan of the range table is the whole column times the shard count.
/// The ranges are ascending and disjoint at every call site (checked once, not
/// assumed), and spans arrive in row order, so a cursor collapses that to
/// O(spans + shards).
///
/// A wall-clock ratio would be flaky, so this asserts an absolute bound at a
/// size where the two curves are orders of magnitude apart: with the cursor
/// this is milliseconds, while the exhaustive scan is 4e9 iterations and was
/// measured at over a minute on this machine.
#[test]
fn numeric_leaf_fold_is_not_quadratic_in_rows_times_shards() {
    const N: usize = 200_000;
    const SHARDS: usize = 20_000;
    const ROWS_PER_SHARD: usize = N / SHARDS;

    let batch = distinct_numeric_obs_batch(N);
    let ranges: Vec<(u64, u64)> = (0..SHARDS)
        .map(|s| {
            (
                (s * ROWS_PER_SHARD) as u64,
                ((s + 1) * ROWS_PER_SHARD) as u64,
            )
        })
        .collect();

    let start = std::time::Instant::now();
    let index = build_numeric_index(batch.column(0), "n_counts", &ranges, 64);
    let elapsed = start.elapsed();

    let entries: usize = index.leaf_pages.iter().map(|p| p.entries.len()).sum();
    assert_eq!(entries, SHARDS, "one leaf entry per shard, still");
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "folding {N} spans over {SHARDS} shards took {elapsed:?}; the per-span \
         scan of the range table is quadratic"
    );
}

/// …and so must the splitter, which the fold's own perf test does not reach.
///
/// `push_shard_split` computes its cut points from the range table on every
/// call, and callers push roughly one batch per shard — so scanning the whole
/// table per push is quadratic in shard count before the leaf fold even runs.
/// The overlapping ranges are a contiguous window when the table is
/// well-formed and ascending, which is the case this bounds.
#[test]
fn push_shard_split_is_not_quadratic_in_shard_count() {
    const N: usize = 200_000;
    const SHARDS: usize = 20_000;
    const ROWS: usize = N / SHARDS;

    let batch = distinct_numeric_obs_batch(N);
    let ranges: Vec<(u64, u64)> = (0..SHARDS)
        .map(|s| ((s * ROWS) as u64, ((s + 1) * ROWS) as u64))
        .collect();

    let start = std::time::Instant::now();
    let mut builder = ObsPredicateIndexBuilder::new(batch.schema(), &n_counts_opts()).unwrap();
    // One push per shard — the shape every in-tree streaming writer has.
    for (s, &(lo, hi)) in ranges.iter().enumerate() {
        builder
            .push_shard_split(&batch.slice(s * ROWS, (hi - lo) as usize), lo, &ranges)
            .unwrap();
    }
    let bytes = builder
        .finish(&ranges, &mut Vec::new(), &mut Vec::new())
        .unwrap()
        .expect("a forced numeric column must produce an index");
    let elapsed = start.elapsed();

    let index = PredicateIndex::read_from(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(numeric_leaf_count(&index), SHARDS);
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "{SHARDS} pushes against a {SHARDS}-range table took {elapsed:?}; the \
         splitter is scanning the whole table per push"
    );
}

/// A malformed range table must fall back to the exhaustive scan, not be
/// silently classified as fast-path-safe.
///
/// The fast path stops walking once a range starts past the span, which is
/// only sound if the ranges really are ascending *and* each is well-formed.
/// An inverted range like `(100, 50)` satisfies "my end precedes the next
/// range's start" while sitting in the wrong place entirely, so a predicate
/// that checks only adjacent pairs would break out of the walk and lose the
/// valid overlap behind it. No in-tree table is malformed — this pins the
/// stated fallback contract of the public helper.
#[test]
fn a_malformed_range_table_still_finds_every_overlap() {
    let col: ArrayRef = Arc::new(Float64Array::from(
        (0..80).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    let schema = Schema::new(vec![Field::new("n_counts", DataType::Float64, true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![col]).unwrap();

    // Inverted first range, valid second. `end <= next.start` holds (50 <= 60).
    //
    // The span has to be multi-row to reach the break: a one-row span (what the
    // batch builder emits) makes the cursor skip *past* the inverted range and
    // find the valid one anyway. So this goes through the streaming builder,
    // whose single 80-row push is one summary block spanning both ranges.
    let malformed = [(100u64, 50u64), (60, 70)];
    let index = streaming_numeric_index(&batch, 80, &malformed);

    // The fallback cannot make a malformed table meaningful — the surviving
    // block still summarises all 80 rows, so the bound is conservative. What
    // it must not do is *lose* the overlap: without the validity check the
    // walk breaks at the inverted range and shard 1 gets no entry at all, so
    // no `MinMax` stat and no coverage.
    let per_shard = min_max_per_shard(&index, malformed.len());
    let (min, max) = per_shard[1].expect(
        "the valid range behind the inverted one was skipped entirely — the \
         fast path accepted a table it cannot walk",
    );
    assert!(
        min <= 60.0 && max >= 69.0,
        "shard 1 covers rows 60..70 (values 60..69) but recorded [{min}, {max}]"
    );
}
