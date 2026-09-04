//! Cross-op invariant: **deleting a cell survives every operation.**
//!
//! This file exists because the same defect was found independently in four
//! crates. Each op had its own hand-written notion of "which sections survive
//! me", and `SectionType::DeletionVectors` fell out of several of them — so a
//! cell that had been marked deleted quietly came back, with no warning, no
//! error, and an output that was internally self-consistent while being wrong.
//!
//! A per-op test in each crate catches its own op. What none of them can catch
//! is the *next* op to be written, so this asserts the invariant once, over a
//! table, in a crate that can see every op at once.
//!
//! Two behaviours are correct, and the table records which each op has:
//!
//! * **Carry** — the op preserves the obs row space 1:1, so it copies the
//!   deletion-vector section through and the rows stay physically present.
//!   `optimize` is the reference (`optimize.rs`: "optimize does not apply them —
//!   that is `compact`'s job").
//! * **Apply** — the op materializes a new row space, so it drops the deleted
//!   rows and emits no deletion vector. `compact` and `sort` are the references.
//!
//! What is *never* correct is a third outcome: more live cells out than in.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::reader::ScxReader;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;

const N_OBS: usize = 12;
const N_VARS: usize = 6;
const DELETED: [u64; 3] = [1, 4, 9];
const LIVE: usize = N_OBS - DELETED.len();

fn write_fixture(dir: &Path, name: &str, n_obs: usize, prefix: &str) -> PathBuf {
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(n_obs as u64, N_VARS as u64, 0, 10_000, 0, 0),
    )
    .unwrap();

    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..n_obs)
                    .map(|i| format!("{prefix}_cell_{i}"))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..n_obs)
                    .map(|i| if i % 2 == 0 { "T cell" } else { "B cell" })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
    writer
        .write_var(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "gene_id",
                    DataType::Utf8,
                    false,
                )])),
                vec![Arc::new(StringArray::from(
                    (0..N_VARS).map(|i| format!("gene_{i}")).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();

    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for row in 0..n_obs {
        indices.push(((row * 2) % N_VARS) as u32);
        indices.push(((row * 2 + 1) % N_VARS) as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();
    path
}

fn deleted_fixture(dir: &Path, name: &str, prefix: &str) -> PathBuf {
    let path = write_fixture(dir, name, N_OBS, prefix);
    scx_ops::mark_deleted(&path, &DELETED).unwrap();
    path
}

/// The number of cells a reader would actually hand a user.
fn live_cells(path: &Path) -> usize {
    let reader = ScxReader::open(path).unwrap();
    let csr = reader.read_all_csr_shards_filtered().unwrap();
    let obs = reader.read_obs_filtered().unwrap();
    assert_eq!(
        csr.shape.0,
        obs.num_rows(),
        "{}: X and obs must agree on the live cell count — an output where they \
         disagree is worse than one that is merely stale",
        path.display()
    );
    csr.shape.0
}

/// The flag and the section must agree in both directions.
///
/// A flag set with no section is a file that claims deletions it cannot name; a
/// section with no flag is one whose deletions no reader will look for. The
/// writer re-derives the flag from the catalog on `finish()`, which is exactly
/// why an op that forgets to write the section produces the *second* shape —
/// silent resurrection, with no dangling flag to notice it by.
fn assert_flag_matches_section(path: &Path, label: &str) {
    let reader = ScxReader::open(path).unwrap();
    let has_section = reader
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::DeletionVectors);
    assert_eq!(
        reader.header().has_deletion_vectors(),
        has_section,
        "{label}: has_deletion_vectors() must equal 'the catalog holds a \
         DeletionVectors section'"
    );
    if has_section {
        assert!(
            reader.read_deletion_vectors().unwrap().is_some(),
            "{label}: the section must be readable"
        );
    }
}

fn score_batch(n: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "score",
            DataType::Float32,
            true,
        )])),
        vec![Arc::new(Float32Array::from(
            (0..n).map(|i| i as f32).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// `attach_external_obs(positional=True)` with an `n`-row frame — `LIVE` rows is
/// the live-length arm, `N_OBS` the physical one.
fn attach_positional_scores(path: &Path, n: usize) {
    scx_ops::attach_external_obs(
        path,
        &scx_ops::ExternalObsData {
            row_keys: Vec::new(),
            row_annotations: score_batch(n),
            row_embeddings: Vec::new(),
            uns: Default::default(),
            source_checksum: None,
            source_name: None,
        },
        &scx_ops::AttachObsOptions {
            join_key: scx_ops::ObsJoinKey::Positional,
            ..Default::default()
        },
    )
    .unwrap();
}

/// `modify_metadata(obs=)` with `obs` plus one new `score` column, in whichever
/// row space `obs` came in.
fn replace_obs_with_score(path: &Path, obs: RecordBatch) {
    let n = obs.num_rows();
    let mut fields: Vec<Arc<Field>> = obs.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new("score", DataType::Float32, true)));
    let mut columns = obs.columns().to_vec();
    columns.push(score_batch(n).column(0).clone());
    let obs = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    scx_ops::modify_metadata(
        path,
        &scx_ops::MetadataPatch {
            obs: Some(obs),
            ..Default::default()
        },
    )
    .unwrap();
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Policy {
    /// Rows stay; the deletion-vector section comes with them.
    Carry,
    /// Rows are dropped; the output carries no deletion vector.
    Apply,
}

/// Every op that writes an SCX file, run over the same deleted fixture.
///
/// The assertion is deliberately about *live* cells rather than physical rows:
/// it is the one statement that is true of both policies, and it is what a user
/// actually observes.
#[test]
fn every_op_preserves_the_live_cell_count() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    type Op = (&'static str, Policy, fn(&Path, &Path) -> PathBuf);
    let ops: Vec<Op> = vec![
        ("build_csc", Policy::Carry, |src, out| {
            scx_ops::run_build_csc(src, out, "4G", false, 5000, None).unwrap();
            out.to_path_buf()
        }),
        ("build_csc_in_place", Policy::Carry, |src, out| {
            std::fs::copy(src, out).unwrap();
            scx_ops::rebuild_csc_inplace(out, 5000, "4G", None).unwrap();
            out.to_path_buf()
        }),
        ("optimize", Policy::Carry, |src, out| {
            scx_ops::optimize(src, out, None, Default::default()).unwrap();
            out.to_path_buf()
        }),
        ("streaming_preprocess", Policy::Carry, |src, out| {
            let config = scx_engine::PreprocessConfig {
                normalize_target_sum: Some(1e4),
                log1p: true,
            };
            scx_engine::streaming_preprocess(src, out, &config).unwrap();
            out.to_path_buf()
        }),
        ("streaming_save_layer", Policy::Carry, |src, out| {
            let config = scx_engine::PreprocessConfig {
                log1p: true,
                ..Default::default()
            };
            scx_engine::streaming_save_layer(src, out, "norm", &config).unwrap();
            out.to_path_buf()
        }),
        // The two in-place obs writers accept a frame in either row space
        // (pyscx 0.17: `read_obs()` is live-length, `read_obs(logical=False)`
        // physical). Both must carry the deletion vector and neither may
        // resurrect a row — the live-length arm in particular writes a
        // physical-length obs with nulls at the deleted rows.
        ("attach_obs_positional_live", Policy::Carry, |src, out| {
            std::fs::copy(src, out).unwrap();
            attach_positional_scores(out, LIVE);
            out.to_path_buf()
        }),
        (
            "attach_obs_positional_physical",
            Policy::Carry,
            |src, out| {
                std::fs::copy(src, out).unwrap();
                attach_positional_scores(out, N_OBS);
                out.to_path_buf()
            },
        ),
        ("modify_metadata_obs_live", Policy::Carry, |src, out| {
            std::fs::copy(src, out).unwrap();
            let obs = ScxReader::open(out).unwrap().read_obs_filtered().unwrap();
            assert_eq!(obs.num_rows(), LIVE);
            replace_obs_with_score(out, obs);
            out.to_path_buf()
        }),
        ("modify_metadata_obs_physical", Policy::Carry, |src, out| {
            std::fs::copy(src, out).unwrap();
            let obs = ScxReader::open(out).unwrap().read_obs().unwrap();
            assert_eq!(obs.num_rows(), N_OBS);
            replace_obs_with_score(out, obs);
            out.to_path_buf()
        }),
        ("compact", Policy::Apply, |src, out| {
            scx_ops::compact(src, out).unwrap();
            out.to_path_buf()
        }),
        ("sort", Policy::Apply, |src, out| {
            let opts = scx_ops::SortOptions {
                by: vec!["cell_id".to_string()],
                ..Default::default()
            };
            scx_ops::sort(src, out, &opts).unwrap();
            out.to_path_buf()
        }),
    ];

    for (name, policy, run) in ops {
        let src = deleted_fixture(dir, &format!("{name}_in.scx"), "A");
        assert_eq!(live_cells(&src), LIVE, "{name}: fixture sanity");

        let out_path = dir.join(format!("{name}_out.scx"));
        let out = run(&src, &out_path);

        assert_eq!(
            live_cells(&out),
            LIVE,
            "{name}: a deleted cell came back (or an extra one vanished)"
        );
        assert_flag_matches_section(&out, name);

        let reader = ScxReader::open(&out).unwrap();
        match policy {
            Policy::Carry => {
                assert_eq!(
                    reader.n_obs() as usize,
                    N_OBS,
                    "{name}: declared carry, so the physical rows must all still be there"
                );
                let dv = reader
                    .read_deletion_vectors()
                    .unwrap()
                    .unwrap_or_else(|| panic!("{name}: declared carry but wrote no DV section"));
                assert_eq!(dv.total_deleted(), DELETED.len() as u64, "{name}");
                for row in DELETED {
                    assert!(
                        dv.is_deleted_global(row as u32),
                        "{name}: row {row} must still be marked deleted"
                    );
                }
            }
            Policy::Apply => {
                assert_eq!(
                    reader.n_obs() as usize,
                    LIVE,
                    "{name}: declared apply, so the deleted rows must be gone"
                );
                assert!(
                    !reader.header().has_deletion_vectors(),
                    "{name}: declared apply, so nothing is left to delete"
                );
            }
        }
    }
}

/// Merge is its own case: it concatenates row spaces, so a carried deletion
/// vector has to be *remapped*, not copied. Both inputs carry deletions so that
/// a dropped offset shows up as the wrong cells being deleted rather than
/// merely as the wrong count.
#[test]
fn merge_preserves_the_live_cell_count_of_every_input() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let a = deleted_fixture(dir, "merge_a.scx", "A");
    let b = deleted_fixture(dir, "merge_b.scx", "B");

    let out = dir.join("merged.scx");
    scx_ops::merge(&[a.as_path(), b.as_path()], &out).unwrap();

    assert_eq!(live_cells(&out), LIVE * 2, "both inputs' deletions survive");
    assert_flag_matches_section(&out, "merge");

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs() as usize, N_OBS * 2, "carried, not applied");
    let dv = reader.read_deletion_vectors().unwrap().unwrap();
    for row in DELETED {
        assert!(dv.is_deleted_global(row as u32), "input A's row {row}");
        assert!(
            dv.is_deleted_global(row as u32 + N_OBS as u32),
            "input B's row {row} must land at {} + {row}",
            N_OBS
        );
    }
    assert_eq!(dv.total_deleted(), DELETED.len() as u64 * 2);
}

/// The whole point of a deletion vector is that the *right* cells disappear.
/// A count-only assertion passes just as happily when an op deletes three
/// arbitrary cells, so pin the identities too.
#[test]
fn the_surviving_cells_are_the_ones_that_were_not_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let src = deleted_fixture(dir, "ids_in.scx", "A");

    let expected: Vec<String> = (0..N_OBS)
        .filter(|i| !DELETED.contains(&(*i as u64)))
        .map(|i| format!("A_cell_{i}"))
        .collect();

    let read_ids = |path: &Path| -> Vec<String> {
        let obs = ScxReader::open(path).unwrap().read_obs_filtered().unwrap();
        let col = obs.column_by_name("cell_id").unwrap();
        let s = arrow::compute::cast(col, &DataType::Utf8).unwrap();
        let a = s
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        (0..a.len()).map(|i| a.value(i).to_string()).collect()
    };

    assert_eq!(read_ids(&src), expected, "fixture");

    let carried = dir.join("ids_csc.scx");
    scx_ops::run_build_csc(&src, &carried, "4G", false, 5000, None).unwrap();
    assert_eq!(read_ids(&carried), expected, "build-csc (carry)");

    let applied = dir.join("ids_compact.scx");
    scx_ops::compact(&src, &applied).unwrap();
    assert_eq!(read_ids(&applied), expected, "compact (apply)");
}
