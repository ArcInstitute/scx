use super::*;
use arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::io::Cursor;
use std::sync::Arc;

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
