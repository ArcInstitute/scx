//! Row reordering vs. neighbourhood plans.
//!
//! §12.6's reorder case: after `scx sort`, plans built from the sorted file's
//! `obsp` / `obsm` must gather the same **cells** — by `obs_names` — as plans
//! built from the original, even though every physical row index moved.
//!
//! Two things this test is named for, because both were wrong in the phase's
//! premises and both change what it covers:
//!
//! 1. **`scx sort` re-emits `obsp` and `obsm` UNSHARDED** — `sort_engine`
//!    calls `write_obsm` / `write_obsp`, not the `_shard_` writers. So the
//!    "after" half exercises the legacy single-section read branch, not the
//!    sharded one. That branch is exactly where the row-count trap lives (an
//!    unsharded COO section's Arrow row count is nnz, not `n_obs`), so this is
//!    worth covering — but it is not additional coverage of the sharded path.
//! 2. **`scx sort` applies deletion vectors**, so a sorted output has none.
//!    The reorder case and the deletion case therefore cannot share a fixture,
//!    and this file does not try to.
//!
//! This crate is where the test lives because it is the only one that depends
//! on both `scx-loader` (the builder) and `scx-ops` (the sort).

use arrow::array::{Array, Float32Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_loader::neighborhood::{
    build_coord_plans, build_graph_plans, CoordQuery, NeighborhoodConfig, WeightOrder,
};
use std::collections::HashMap;
use std::sync::Arc;

const G: usize = 6;
const N: usize = G * G;
const PITCH: f32 = 10.0;

fn cfg() -> NeighborhoodConfig {
    NeighborhoodConfig {
        include_center: true,
        file_id: 0,
    }
}

/// obs carries the cell id and a `sort_key`. The permutation actually used is
/// `SortOptions::shuffle`, not this key, and that is not arbitrary: the first
/// version of this test sorted on a reversed key, which on a square lattice is
/// a 180-degree rotation — a **symmetry of the rook graph**, so every plan came
/// back with identical row numbers and the "same cells by name" comparison was
/// vacuous. The anti-vacuity assertion below is what caught it. A seeded
/// shuffle has no such symmetry.
fn obs_batch() -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..N).map(|i| format!("spot_{i:03}")).collect();
    // Reverse order, so sorting ascending on `sort_key` exactly reverses obs.
    let keys: Vec<i32> = (0..N as i32).map(|i| N as i32 - 1 - i).collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("sort_key", DataType::Int32, false),
        ])),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(keys)),
        ],
    )
    .unwrap()
}

fn var_batch() -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..4).map(|i| format!("gene_{i}")).collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn coords() -> Vec<f32> {
    (0..N)
        .flat_map(|i| [(i % G) as f32 * PITCH, (i / G) as f32 * PITCH])
        .collect()
}

fn coord_batch() -> arrow::array::RecordBatch {
    let c = coords();
    let xs: Vec<i64> = (0..N).map(|r| c[r * 2] as i64).collect();
    let ys: Vec<i64> = (0..N).map(|r| c[r * 2 + 1] as i64).collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("0", DataType::Int64, false),
            Field::new("1", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(xs)),
            Arc::new(Int64Array::from(ys)),
        ],
    )
    .unwrap()
}

/// Rook adjacency of the lattice.
fn edges() -> Vec<(i32, i32, f32)> {
    let mut out = Vec::new();
    for i in 0..N {
        let (x, y) = (i % G, i / G);
        if x + 1 < G {
            out.push((i as i32, (i + 1) as i32, 1.0));
            out.push(((i + 1) as i32, i as i32, 1.0));
        }
        if y + 1 < G {
            out.push((i as i32, (i + G) as i32, 1.0));
            out.push(((i + G) as i32, i as i32, 1.0));
        }
    }
    out.sort_by_key(|&(r, c, _)| (r, c));
    out
}

fn coo_batch() -> arrow::array::RecordBatch {
    let e = edges();
    let schema = Schema::new(vec![
        Field::new("row", DataType::Int32, false),
        Field::new("col", DataType::Int32, false),
        Field::new("data", DataType::Float32, false),
    ])
    .with_metadata(HashMap::from([
        ("n_rows".to_string(), N.to_string()),
        ("n_cols".to_string(), N.to_string()),
    ]));
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(e.iter().map(|t| t.0).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(e.iter().map(|t| t.1).collect::<Vec<_>>())),
            Arc::new(Float32Array::from(
                e.iter().map(|t| t.2).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn write_fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("spatial.scx");
    let mut w = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N as u64, 4, N as u64, 16384, 0, 0),
    )
    .unwrap();
    w.write_obs(&obs_batch()).unwrap();
    w.write_var(&var_batch()).unwrap();
    // One nonzero per row, so the file is a valid CSR the sort can permute.
    let indptr: Vec<u64> = (0..=N as u64).collect();
    let indices: Vec<u32> = (0..N as u32).map(|r| r % 4).collect();
    let values: Vec<u8> = (0..N).map(|r| (r % 200 + 1) as u8).collect();
    w.write_csr_shard(
        &indptr,
        &indices,
        &values,
        scx_codec::CodecId::None,
        scx_codec::ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    w.write_obsm("spatial", &coord_batch()).unwrap();
    w.write_obsp("connectivities", &coo_batch()).unwrap();
    w.finish().unwrap();
    path
}

fn cell_ids(path: &std::path::Path) -> Vec<String> {
    let reader = ScxReader::open(path).unwrap();
    let obs = reader.read_obs().unwrap();
    let col = obs
        .column_by_name("cell_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    (0..col.len()).map(|i| col.value(i).to_string()).collect()
}

/// A plan's rows rendered as the cell names they name, so the comparison is
/// about identity rather than about row numbers.
fn named_sets(
    plans: &scx_loader::neighborhood::NeighborhoodPlans,
    names: &[String],
) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = plans
        .plans
        .iter()
        .zip(&plans.centers)
        .map(|(p, &c)| {
            let mut members: Vec<String> =
                p.rows.iter().map(|&r| names[r as usize].clone()).collect();
            // Set membership is the claim; a reorder legitimately changes the
            // within-set order because the rows are sorted by index.
            members.sort();
            (names[c as usize].clone(), members)
        })
        .collect();
    out.sort();
    out
}

fn reorder(input: &std::path::Path, output: &std::path::Path) {
    let opts = scx_ops::SortOptions {
        shuffle: Some(20260915),
        ..Default::default()
    };
    scx_ops::sort(input, output, &opts).unwrap();
}

#[test]
fn graph_plans_name_the_same_cells_after_a_row_reorder() {
    let dir = tempfile::tempdir().unwrap();
    let src = write_fixture(&dir);
    let sorted = dir.path().join("sorted.scx");
    reorder(&src, &sorted);

    let before_names = cell_ids(&src);
    let after_names = cell_ids(&sorted);
    assert_ne!(
        before_names, after_names,
        "the fixture's sort key must actually permute obs, or this test proves nothing"
    );

    let before = build_graph_plans(
        &src,
        "connectivities",
        true,
        None,
        WeightOrder::Desc,
        cfg(),
        4096,
    )
    .unwrap();
    let after = build_graph_plans(
        &sorted,
        "connectivities",
        true,
        None,
        WeightOrder::Desc,
        cfg(),
        4096,
    )
    .unwrap();
    assert_eq!(before.len(), N);

    // Anti-vacuity: the *row numbers* must differ, or "the same cells" is a
    // claim about two identical plan lists and the test proves nothing.
    let raw = |p: &scx_loader::neighborhood::NeighborhoodPlans| -> Vec<Vec<u64>> {
        p.plans.iter().map(|x| x.rows.clone()).collect()
    };
    assert_ne!(
        raw(&before),
        raw(&after),
        "the reorder did not move any row"
    );

    assert_eq!(
        named_sets(&before, &before_names),
        named_sets(&after, &after_names)
    );
}

#[test]
fn coordinate_plans_name_the_same_cells_after_a_row_reorder() {
    let dir = tempfile::tempdir().unwrap();
    let src = write_fixture(&dir);
    let sorted = dir.path().join("sorted.scx");
    reorder(&src, &sorted);

    let q = CoordQuery::Radius(PITCH);
    let before = build_coord_plans(&src, "spatial", true, q, cfg()).unwrap();
    let after = build_coord_plans(&sorted, "spatial", true, q, cfg()).unwrap();
    let raw = |p: &scx_loader::neighborhood::NeighborhoodPlans| -> Vec<Vec<u64>> {
        p.plans.iter().map(|x| x.rows.clone()).collect()
    };
    assert_ne!(
        raw(&before),
        raw(&after),
        "the reorder did not move any row"
    );
    assert_eq!(
        named_sets(&before, &cell_ids(&src)),
        named_sets(&after, &cell_ids(&sorted))
    );
}

#[test]
fn the_sorted_output_really_is_the_legacy_unsharded_form() {
    // The premise the two tests above rest on, asserted rather than assumed:
    // if `scx sort` ever starts re-sharding obsp/obsm, those tests quietly stop
    // covering the legacy branch and this one says so.
    let dir = tempfile::tempdir().unwrap();
    let src = write_fixture(&dir);
    let sorted = dir.path().join("sorted.scx");
    reorder(&src, &sorted);
    let reader = ScxReader::open(&sorted).unwrap();
    let types: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.name.starts_with("obsp/") || e.name.starts_with("obsm/"))
        .map(|e| (e.name.clone(), e.section_type))
        .collect();
    assert!(!types.is_empty(), "sort dropped obsp/obsm entirely");
    for (name, ty) in types {
        assert!(
            matches!(
                ty,
                scx_format_io::section::SectionType::ObspEmbedding
                    | scx_format_io::section::SectionType::ObsmEmbedding
            ),
            "{name} came back as {ty:?}, not the legacy single-section form"
        );
    }
}
