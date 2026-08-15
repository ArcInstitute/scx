//! An obs push that spans several CSR shards must not blur their numeric
//! bounds together.
//!
//! The numeric predicate index summarises *within* a `push_shard` call, so a
//! caller that pushes a batch covering several of the shard ranges it later
//! hands `finish` gives each of those shards the same widened `[min, max]`.
//! That cannot produce a wrong answer — Level-1 pruning excludes a shard only
//! when the probe falls *outside* the recorded bounds, so a wider bound can
//! only fail to prune — but it switches off the pruning the index exists for,
//! silently and with no warning.
//!
//! Almost no in-tree caller pushes at CSR-shard granularity naturally:
//! `modify_metadata` and merge chunk obs by `shard_target_rows`, append's
//! convert-on-append path pushes the entire pre-append axis in one call, and
//! the batch conversion entry point pushes the whole axis too. The fixture
//! here is the `modify_metadata` shape, because it needs no configuration
//! trickery to diverge: a file whose CSR shards are *uneven* has a shard
//! partition that no fixed chunk size can match.
//!
//! Assertions are against ground truth computed from the obs frame, never
//! against a second query path.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::QueryPipeline;
use scx_format_io::catalog::ColumnStat;
use scx_format_io::header::FileHeader;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_ops::{modify_metadata, MetadataPatch};
use tempfile::TempDir;

const N_OBS: usize = 400;
const N_VARS: usize = 8;

/// **Uneven** CSR shards. `shard_target_rows` in the header is 100, so
/// `modify_metadata` chunks obs at 100/100/100/100 while the index is keyed to
/// 100/150/150 — the second chunk straddles the 250 boundary. Nothing about
/// this is exotic: `compact` produces uneven live-row shards whenever
/// deletions are unevenly distributed.
const CSR_SHARDS: [usize; 3] = [100, 150, 150];
const CHUNK_ROWS: u32 = 100;

/// Monotonic, so a shard's true `[min, max]` is exactly its endpoints and any
/// widening is visible as a specific wrong number rather than a vague drift.
fn n_counts(i: usize) -> i64 {
    100 + i as i64
}

fn obs_frame() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("n_counts", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                (0..N_OBS).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..N_OBS).map(n_counts).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn var_frame() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            (0..N_VARS).map(|i| format!("g{i}")).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn write_fixture(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("uneven.scx");
    let header = FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, CHUNK_ROWS, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs_frame()).unwrap();
    writer.write_var(&var_frame()).unwrap();

    let mut start = 0usize;
    for &n in &CSR_SHARDS {
        let indptr: Vec<u64> = (0..=n as u64).collect();
        let indices: Vec<u32> = (0..n).map(|r| (r % N_VARS) as u32).collect();
        let values: Vec<u8> = (0..n).map(|r| ((r % 255) + 1) as u8).collect();
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                start as u64,
            )
            .unwrap();
        start += n;
    }
    assert_eq!(start, N_OBS);
    writer.finish().unwrap();
    path
}

/// The per-shard numeric bounds Level-1 pruning will actually read, in shard
/// order.
fn min_max_per_shard(path: &Path) -> Vec<Option<(f64, f64)>> {
    let reader = ScxReader::open(path).unwrap();
    let mut entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == 0)
        .filter_map(|e| e.stats.as_ref())
        .collect();
    entries.sort_by_key(|s| s.row_start);
    entries
        .iter()
        .map(|s| {
            s.column_stats.iter().find_map(|c| match c {
                ColumnStat::MinMax { min, max, .. } => Some((*min, *max)),
                _ => None,
            })
        })
        .collect()
}

/// Rebuild the obs index in place. `modify_metadata` chunks obs by the
/// header's `shard_target_rows` and finishes over the CSR shard ranges.
fn rebuild_index(path: &Path) {
    modify_metadata(
        path,
        &MetadataPatch {
            obs: Some(obs_frame()),
            index: scx_engine::ConversionPredicateIndexOptions {
                index_obs: vec!["n_counts".to_string()],
                index_var: Vec::new(),
                index_preset: None,
                index_auto_threshold: 1000,
            },
            ..Default::default()
        },
    )
    .unwrap();
}

/// The bounds must describe each CSR shard's own rows and no others.
///
/// Shard 1 spans rows 100..250 (values 200..349). The obs chunk covering rows
/// 200..300 straddles its far edge, so a push that is not split on the CSR
/// boundary folds values up to 399 into shard 1 and reports `[200, 399]`.
#[test]
fn uneven_csr_shards_get_their_own_numeric_bounds_not_the_chunks() {
    let dir = TempDir::new().unwrap();
    let path = write_fixture(&dir);
    rebuild_index(&path);

    let mut want = Vec::new();
    let mut start = 0usize;
    for &n in &CSR_SHARDS {
        want.push(Some((
            n_counts(start) as f64,
            n_counts(start + n - 1) as f64,
        )));
        start += n;
    }

    assert_eq!(
        min_max_per_shard(&path),
        want,
        "a CSR shard's recorded bounds cover rows that belong to its neighbour — \
         the obs chunk spanning the shard boundary was summarised as one block"
    );
}

/// …and the consequence, at the level a user sees it: a shard that cannot
/// match must still be skipped.
///
/// `n_counts > 360` is satisfied only by rows 261.. (shard 2). Shard 1's true
/// maximum is 349, so it must be pruned; with bounds blurred to `[200, 399]`
/// it is scanned instead.
#[test]
fn a_shard_that_cannot_match_is_still_pruned_after_an_index_rebuild() {
    let dir = TempDir::new().unwrap();
    let path = write_fixture(&dir);
    rebuild_index(&path);

    let want = (0..N_OBS).filter(|&i| n_counts(i) > 360).count();
    let got = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("n_counts > 360")
        .unwrap()
        .count()
        .unwrap();

    assert_eq!(
        got.matched_rows, want,
        "wrong rows, not merely wrong pruning"
    );
    assert_eq!(
        got.total_shards, 3,
        "fixture must present all three CSR shards"
    );
    assert_eq!(
        got.skipped_shards, 2,
        "shards 0 (max 199) and 1 (max 349) cannot satisfy `> 360` and must both \
         be pruned; {} were skipped",
        got.skipped_shards
    );
}

/// The convert-on-append shape: a legacy file whose obs is a single section
/// gets its whole pre-append obs axis fed to the builder in **one** push,
/// against a range table of the old CSR shards plus the newly appended ones.
///
/// That is the structurally misaligned case — no chunk size can make a single
/// whole-axis push match a multi-shard range table — and it is the one the
/// `modify_metadata` fixture above cannot reach, because that path chunks.
#[test]
fn convert_on_append_gives_each_old_csr_shard_its_own_numeric_bounds() {
    use scx_ops::{append_with_index_options, AppendOptions};
    use std::num::NonZeroU32;

    let dir = TempDir::new().unwrap();
    let path = write_fixture(&dir);

    // 40 new rows, values far above every existing one, so a shard that
    // absorbed them would be obvious.
    const N_NEW: usize = 40;
    let new_obs = {
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("n_counts", DataType::Int64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    (0..N_NEW).map(|i| format!("new_{i}")).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    (0..N_NEW).map(|i| 9000 + i as i64).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };
    let indptr: Vec<u64> = (0..=N_NEW as u64).collect();
    let indices: Vec<u32> = (0..N_NEW).map(|r| (r % N_VARS) as u32).collect();
    let values: Vec<u8> = (0..N_NEW).map(|r| ((r % 255) + 1) as u8).collect();

    append_with_index_options(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &AppendOptions {
            shard_target_rows: NonZeroU32::new(100).unwrap(),
            ..Default::default()
        },
        &scx_engine::ConversionPredicateIndexOptions {
            index_obs: vec!["n_counts".to_string()],
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 1000,
        },
    )
    .unwrap();

    // The three pre-existing CSR shards keep their own bounds; the appended
    // shard carries only the new values.
    let mut want: Vec<Option<(f64, f64)>> = Vec::new();
    let mut start = 0usize;
    for &n in &CSR_SHARDS {
        want.push(Some((
            n_counts(start) as f64,
            n_counts(start + n - 1) as f64,
        )));
        start += n;
    }
    want.push(Some((9000.0, (9000 + N_NEW - 1) as f64)));

    assert_eq!(
        min_max_per_shard(&path),
        want,
        "the single whole-axis push was summarised as one block, so every old \
         CSR shard recorded the whole pre-append column's range"
    );
}
