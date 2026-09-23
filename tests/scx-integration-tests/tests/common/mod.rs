//! The all-families fixture: one SCX file carrying every section family an op
//! can decide about.
//!
//! Shared between `section_carry.rs`, which asserts what each op does with each
//! family, and `testkit_against_real_ops.rs`, which asserts that the digest
//! harness still works when the file is this rich. Both need the same file for
//! the same reason: a fixture with only X and obs makes most of what a rewrite
//! op does invisible.
//!
//! Not every helper is used by every consumer, hence the blanket `dead_code`
//! allow — the alternative is per-item attributes that go stale.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float32Array, Int32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::writer::ScxWriter;

pub const N_OBS: usize = 8;
pub const N_VARS: usize = 6;
/// Two X shards, so `compact` and `sort` have a row layout to actually change.
/// With one shard several ops are trivially identity and the test proves less.
pub const SHARD_ROWS: usize = 4;
pub const RAW_N_VARS: usize = 9;

/// What to vary in the all-families fixture.
///
/// A struct rather than a run of positional `bool`s. Every knob here exists to
/// isolate exactly one carry decision, and `build_all_families(dir, name, true,
/// false, true)` at a call site does not say which — which matters, because
/// merge validates var identity and requires identical layers across inputs, so
/// two fixtures compared by a merge test **have to** differ in one thing only.
#[derive(Clone, Copy, Default)]
pub struct FixtureShape {
    /// Omit `varm`. For `merge_succeeds_when_only_a_later_input_has_varm`.
    pub without_varm: bool,
    /// Write a second `obsm` key under this name. For merge's key-symmetry
    /// tests, which need two inputs whose obsm key sets differ by one.
    pub extra_obsm_key: Option<&'static str>,
    /// Write the obsp graph under this key instead of `connectivities`.
    pub obsp_key: Option<&'static str>,
    /// Store the obsp graph's `data` column as `Float64` instead of `Float32`.
    ///
    /// `remap_obsp_coo_to_dim` preserves the source dtype, so two inputs that
    /// differ here produce shards that cannot be concatenated under one schema.
    pub obsp_f64_values: bool,
    /// Omit `adata.raw`.
    ///
    /// Only for `append`, which refuses a file with `has_raw()` set
    /// (`append.rs:592`, `OpsError::RawUnsupported`): append extends X's obs
    /// axis and not raw's, so proceeding would leave `raw.n_obs < n_obs` with
    /// `has_raw` still set. Every other op takes the fixture whole.
    pub without_raw: bool,
    /// Store an **explicit zero** as X row 0's first value.
    ///
    /// `is_canonical_csr` treats a stored `0.0` as non-canonical, so this is
    /// what makes `canonicalize_csr` actually rewrite the shard instead of
    /// short-circuiting — and `BitmapShard::build_from_csr` keys off the stored
    /// index regardless of its value, so the bitmap written below records a
    /// gene that canonicalisation is about to remove.
    pub explicit_zero_in_x: bool,
}

/// Every family the format can carry, in one file.
///
/// Built by hand rather than by running an op, so it does not inherit any op's
/// idea of what a file contains — which is the thing under test.
pub fn fixture_all_families(dir: &Path, name: &str) -> PathBuf {
    build_all_families(dir, name, FixtureShape::default())
}

/// The same file with **no `adata.raw`**, and otherwise the same shape.
///
/// For `append` only — see [`FixtureShape::without_raw`]. Keeping every other
/// family means an append digest still covers layers, obsm/varm, obsp/varp,
/// bitmaps, the group index, the predicate indexes and a deletion vector; only
/// the one family the op refuses is absent.
pub fn fixture_all_families_without_raw(dir: &Path, name: &str) -> PathBuf {
    build_all_families(
        dir,
        name,
        FixtureShape {
            without_raw: true,
            ..Default::default()
        },
    )
}

/// The same file with **no `varm`**, and otherwise byte-for-byte the same shape.
///
/// For `merge_succeeds_when_only_a_later_input_has_varm`: merge validates var
/// identity and requires every input to carry the same layers, so the no-varm
/// side cannot be some other fixture — it has to differ in exactly one family.
pub fn fixture_all_families_without_varm(dir: &Path, name: &str) -> PathBuf {
    build_all_families(
        dir,
        name,
        FixtureShape {
            without_varm: true,
            ..FixtureShape::default()
        },
    )
}

/// The same file plus one more `obsm` key.
///
/// Merge takes its obsm key set from input 0 only and drops any key an input
/// lacks, both silently; this is the fixture that makes either asymmetry
/// visible, depending on which side of the merge it is placed.
pub fn fixture_all_families_with_extra_obsm(dir: &Path, name: &str, key: &'static str) -> PathBuf {
    build_all_families(
        dir,
        name,
        FixtureShape {
            extra_obsm_key: Some(key),
            ..FixtureShape::default()
        },
    )
}

/// The same file with `cell_type` stored as a categorical (`Dictionary(Int8,
/// Utf8)` + the `scx.categorical.ordered` stamp) instead of plain `Utf8`.
///
/// For the `attach_obs` output-identity arm only. The in-place obs writers
/// carry a dictionary column through as a dictionary, and a digest over a
/// plain-string fixture cannot see whether they still do — but flipping the
/// default fixture would move every other arm's digest for a reason that has
/// nothing to do with those ops. The obs batch is the one thing this variant
/// changes, so it is passed in rather than selected by a knob on
/// [`FixtureShape`].
pub fn fixture_all_families_with_categorical_obs(dir: &Path, name: &str) -> PathBuf {
    build_all_families_with_obs(dir, name, FixtureShape::default(), categorical_obs_batch())
}

/// The categorical-obs fixture with **no `adata.raw`**.
///
/// For the `append_categorical` output-identity arm: `append` refuses a file
/// carrying `raw` (`OpsError::RawUnsupported`), and the categorical variant is
/// what makes the arm able to see whether the rows an append adds keep their
/// dictionary encoding.
pub fn fixture_all_families_without_raw_with_categorical_obs(dir: &Path, name: &str) -> PathBuf {
    build_all_families_with_obs(
        dir,
        name,
        FixtureShape {
            without_raw: true,
            ..Default::default()
        },
        categorical_obs_batch(),
    )
}

/// [`appendable_rows`] with `cell_type` as the same `Dictionary(Int8, Utf8)`
/// the categorical fixture carries — declared levels, order and the
/// `scx.categorical.ordered` stamp included, `"NK cell"` still declared and
/// still unused.
///
/// A plain-`Utf8` appended batch is accepted too (`validate_obs_schema`
/// compares through `effective_type`), but it would leave the appended shards
/// plain and the digest would pin the reconciliation rather than the write.
pub fn appendable_rows_categorical(n_new: usize) -> (RecordBatch, Vec<u64>, Vec<u32>, Vec<u8>) {
    use arrow::array::{Array, DictionaryArray, Int8Array};
    use arrow::datatypes::Int8Type;

    let (plain, indptr, indices, values) = appendable_rows(n_new);
    let dict = DictionaryArray::<Int8Type>::try_new(
        Int8Array::from((0..n_new).map(|i| (i % 2) as i8).collect::<Vec<_>>()),
        Arc::new(StringArray::from(vec!["T cell", "B cell", "NK cell"])),
    )
    .unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let fields: Vec<Field> = plain
        .schema()
        .fields()
        .iter()
        .map(|f| {
            if f.name() == "cell_type" {
                Field::new("cell_type", dict.data_type().clone(), true).with_metadata(md.clone())
            } else {
                f.as_ref().clone()
            }
        })
        .collect();
    let columns = plain
        .columns()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if plain.schema().field(i).name() == "cell_type" {
                Arc::new(dict.clone()) as arrow::array::ArrayRef
            } else {
                c.clone()
            }
        })
        .collect();
    let obs = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    (obs, indptr, indices, values)
}

/// The same file with an explicit zero stored in X, so that canonicalisation
/// has something to do and the detection bitmap written alongside is left
/// describing a matrix that no longer exists.
pub fn fixture_with_explicit_zero_in_x(dir: &Path, name: &str) -> PathBuf {
    build_all_families(
        dir,
        name,
        FixtureShape {
            explicit_zero_in_x: true,
            ..FixtureShape::default()
        },
    )
}

fn build_all_families(dir: &Path, name: &str, shape: FixtureShape) -> PathBuf {
    build_all_families_with_obs(dir, name, shape, obs_batch())
}

fn build_all_families_with_obs(
    dir: &Path,
    name: &str,
    shape: FixtureShape,
    obs: RecordBatch,
) -> PathBuf {
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();

    // --- obs / var -------------------------------------------------------
    writer.write_obs(&obs).unwrap();
    let var = var_batch(N_VARS, "gene");
    writer.write_var(&var).unwrap();

    // --- X, in two shards ------------------------------------------------
    let mut shard_ranges: Vec<(u64, u64)> = Vec::new();
    for (shard_idx, row_start) in (0..N_OBS).step_by(SHARD_ROWS).enumerate() {
        let (indptr, indices, mut values) = csr_rows(row_start, SHARD_ROWS, N_VARS, 1);
        // Row 0's first stored value only. One is enough — `is_canonical_csr`
        // rejects the whole matrix on the first zero it finds — and keeping it
        // to one shard means the *other* shard's bitmap stays correct, so a
        // test can tell "the carry was gated" from "the carry was removed".
        if shape.explicit_zero_in_x && row_start == 0 {
            values[0] = 0;
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
        shard_ranges.push((row_start as u64, (row_start + SHARD_ROWS) as u64));

        // --- detection bitmaps, keyed to this shard's local rows ---------
        let shard = scx_format_io::bitmap::BitmapShard::build_from_csr(
            row_start as u64,
            SHARD_ROWS as u32,
            N_VARS as u32,
            &indptr,
            &indices,
        );
        writer.write_bitmap_shard(&shard).unwrap();
        let _ = shard_idx;
    }

    // --- a layer ---------------------------------------------------------
    let (l_indptr, l_indices, l_values) = csr_rows(0, N_OBS, N_VARS, 3);
    let layer_shard = scx_format_io::ShardBuffers::new(
        &l_indptr,
        &l_indices,
        &l_values,
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer
        .write_layer_csr_shard("spliced", 0, 0, layer_shard)
        .unwrap();

    // --- obsm / varm -----------------------------------------------------
    writer.write_obsm("X_pca", &dense_embedding(N_OBS)).unwrap();
    if let Some(key) = shape.extra_obsm_key {
        writer.write_obsm(key, &dense_embedding(N_OBS)).unwrap();
    }
    if !shape.without_varm {
        writer.write_varm("PCs", &dense_embedding(N_VARS)).unwrap();
    }

    // --- obsp / varp -----------------------------------------------------
    writer
        .write_obsp_shard_coo(
            shape.obsp_key.unwrap_or("connectivities"),
            0,
            0,
            N_OBS as u64,
            N_OBS as u64,
            &coo_batch_typed(N_OBS, shape.obsp_f64_values),
        )
        .unwrap();
    writer.write_varp("gene_corr", &coo_i32(N_VARS)).unwrap();

    // --- uns -------------------------------------------------------------
    writer
        .write_uns(&serde_json::json!({ "carry_fixture": true }))
        .unwrap();

    // --- adata.raw (its own, wider var axis) -----------------------------
    if !shape.without_raw {
        writer.set_raw_n_vars(RAW_N_VARS as u64);
        let (r_indptr, r_indices, r_values) = csr_rows(0, N_OBS, RAW_N_VARS, 7);
        writer
            .write_raw_csr_shard(
                &r_indptr,
                &r_indices,
                &r_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.write_raw_var(&var_batch(RAW_N_VARS, "raw")).unwrap();
    }

    // --- the grouped-sort sidecar ----------------------------------------
    writer
        .write_group_index(
            serde_json::json!({
                "group_by": "cell_type",
                "reference_shard": 0,
                "reference_labels": ["T cell"],
                "records": [],
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();

    // --- predicate indexes ------------------------------------------------
    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["cell_type".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let (mut outcomes, mut cols) = (Vec::new(), Vec::new());
    let obs_bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &shard_ranges,
        &opts,
        &mut outcomes,
        &mut cols,
    )
    .unwrap()
    .expect("the fixture's cell_type column must produce an obs predicate index");
    writer.write_obs_predicate_index(&obs_bytes).unwrap();

    let var_opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["gene_kind".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let (mut v_outcomes, mut v_cols) = (Vec::new(), Vec::new());
    let var_bytes = scx_engine::build_var_predicate_index_bytes(
        &var,
        &[(0, N_VARS as u64)],
        &var_opts,
        &mut v_outcomes,
        &mut v_cols,
    )
    .unwrap()
    .expect("the fixture's gene_kind column must produce a var predicate index");
    writer.write_var_predicate_index(&var_bytes).unwrap();

    writer.finish().unwrap();

    // --- deletion vectors -------------------------------------------------
    // Via the real op: a hand-written deletion vector would not be exercising
    // the same section the ops read.
    //
    // **No CSC sidecar.** An earlier version of this comment said the fixture
    // added one "via its real op" alongside the deletion vector; it never did.
    // The reason it gave next — that the only thing writing a CSC sidecar is
    // `build-csc`, "which would strip varm/obsp/varp/raw/bitmaps/group-index
    // from this fixture on the way" — was true before Phase 5b and is false
    // now: all six are `Carry::Verbatim`, which is exactly what
    // `section_carry.rs::build_csc_carries_what_optimize_carries` runs this
    // fixture through build-csc to prove. So the sidecar is simply absent
    // rather than unobtainable. `XCsc` and `LayerCsc` stay excluded from
    // `the_fixture_carries_every_family_an_op_can_decide_about`, and the CSC
    // `Dropped` arms stay pinned by the pre-existing CLI tests rather than here
    // — adding one here would make every op's digest carry a sidecar it has
    // nothing to say about. The rewrite ops' carry arms use
    // `fixture_all_families_with_csc` instead.
    scx_ops::mark_deleted(&path, &[2, 5]).unwrap();
    path
}

/// The fixture plus a **CSR-backed** obsp graph (`ObspCsrShard`), which is a
/// different section family from the COO graphs `read_all_obsp` returns.
///
/// Separate rather than folded into `fixture_all_families` because it is the
/// input that distinguishes the two: `optimize` re-encodes it in its shard loop,
/// while `compact` and `sort` never read it at all. A fixture carrying only COO
/// obsp cannot tell those apart.
pub fn fixture_with_csr_obsp(dir: &Path, name: &str) -> PathBuf {
    // Plain `N_VARS` (6) against `N_OBS` (8): the ordinary shape, more cells
    // than genes. It used to be a local `N_OBS + 4` because the writer stamped
    // an obsp shard's minor extent from `n_vars`, so a fixture with fewer genes
    // than cells could not hold a valid obs x obs graph — it failed with
    // `ShardIndexOutOfRange` before any op saw it. `ScxWriter::shard_n_minor`
    // now resolves the obs axis (OPT-FORMATIO-4); keeping the realistic shape
    // here, and in the `optimize_csr_obsp` golden arm this feeds, is what stops
    // the dodge from creeping back.
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();
    writer.write_obs(&obs_batch()).unwrap();
    writer.write_var(&var_batch(N_VARS, "gene")).unwrap();
    for row_start in (0..N_OBS).step_by(SHARD_ROWS) {
        let (indptr, indices, values) = csr_rows(row_start, SHARD_ROWS, N_VARS, 1);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    // obs x obs, so the column extent is N_OBS — canonical CSR, one entry per
    // row, which `scx validate --deep` requires of this section type.
    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for row in 0..N_OBS {
        indices.push(((row + 1) % N_OBS) as u32);
        values.push((row + 1) as u8);
        indptr.push(indptr.last().unwrap() + 1);
    }
    let shard = scx_format_io::ShardBuffers::new(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer
        .write_obsp_shard("connectivities", 0, 0, shard)
        .unwrap();
    writer.finish().unwrap();
    path
}

/// New rows to hand `scx_ops::append`, schema-compatible with the fixture's obs.
///
/// `append` validates the incoming obs against the target's schema, so this
/// cannot be a generic builder — the three columns and their nullability have
/// to match [`obs_batch`] exactly. Row identifiers continue from `N_OBS` so a
/// digest of the appended file distinguishes "the new rows landed" from "the
/// old rows were rewritten".
pub fn appendable_rows(n_new: usize) -> (RecordBatch, Vec<u64>, Vec<u32>, Vec<u8>) {
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
            Field::new("n_counts", DataType::UInt32, false),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..n_new)
                    .map(|i| format!("cell_{}", N_OBS + i))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..n_new)
                    .map(|i| if i % 2 == 0 { "T cell" } else { "B cell" })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                (0..n_new)
                    .map(|i| (100 + N_OBS + i) as u32)
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    // Salt 1 matches X's, so the appended rows continue the same value series
    // the existing shards use rather than forming a distinguishable block.
    let (indptr, indices, values) = csr_rows(N_OBS, n_new, N_VARS, 1);
    (obs, indptr, indices, values)
}

/// [`obs_batch`] with `cell_type` as `Dictionary(Int8, Utf8)` — the shape a
/// pandas categorical lands in — plus the `scx.categorical.ordered` stamp and a
/// declared level no row uses, so a digest over it sees the dictionary, the
/// stamp and the unused level (and not a vocabulary rebuilt from the data).
fn categorical_obs_batch() -> RecordBatch {
    use arrow::array::{Array, DictionaryArray, Int8Array};
    use arrow::datatypes::Int8Type;

    let base = obs_batch();
    let dict = DictionaryArray::<Int8Type>::try_new(
        Int8Array::from((0..N_OBS).map(|i| (i % 2) as i8).collect::<Vec<_>>()),
        Arc::new(StringArray::from(vec!["T cell", "B cell", "NK cell"])),
    )
    .unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let fields: Vec<Field> = base
        .schema()
        .fields()
        .iter()
        .map(|f| {
            if f.name() == "cell_type" {
                Field::new("cell_type", dict.data_type().clone(), true).with_metadata(md.clone())
            } else {
                f.as_ref().clone()
            }
        })
        .collect();
    let columns = base
        .columns()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if base.schema().field(i).name() == "cell_type" {
                Arc::new(dict.clone()) as arrow::array::ArrayRef
            } else {
                c.clone()
            }
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn obs_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
            Field::new("n_counts", DataType::UInt32, false),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..N_OBS).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..N_OBS)
                    .map(|i| if i % 2 == 0 { "T cell" } else { "B cell" })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                (0..N_OBS).map(|i| (100 + i) as u32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn var_batch(n: usize, prefix: &str) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("gene_kind", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..n).map(|i| format!("{prefix}_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..n)
                    .map(|i| if i % 3 == 0 { "mito" } else { "nuclear" })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// Two nonzeros per row, deterministic, with `salt` separating X from the layer
/// and from raw so a mixed-up carry shows as wrong values rather than as a pass.
fn csr_rows(
    row_start: usize,
    n_rows: usize,
    n_cols: usize,
    salt: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        let row = row_start + r;
        let (c0, c1) = ((row * 2) % n_cols, (row * 2 + 1) % n_cols);
        let (lo, hi) = if c0 <= c1 { (c0, c1) } else { (c1, c0) };
        indices.push(lo as u32);
        indices.push(hi as u32);
        values.push(((row + salt) % 255 + 1) as u8);
        values.push(((row + salt + 1) % 255 + 1) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

fn dense_embedding(n_rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("0", DataType::Float32, false),
            Field::new("1", DataType::Float32, false),
        ])),
        vec![
            Arc::new(Float32Array::from(
                (0..n_rows).map(|i| i as f32).collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n_rows).map(|i| -(i as f32)).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// obs×obs COO in the Int64 form `write_obsp_shard_coo` takes.
fn coo_batch(n: usize) -> RecordBatch {
    coo_batch_typed(n, false)
}

/// [`coo_batch`] with control over the `data` column's width.
fn coo_batch_typed(n: usize, f64_values: bool) -> RecordBatch {
    let (data_type, data): (DataType, Arc<dyn arrow::array::Array>) = if f64_values {
        (
            DataType::Float64,
            Arc::new(arrow::array::Float64Array::from(
                (0..n).map(|i| (i + 1) as f64).collect::<Vec<_>>(),
            )),
        )
    } else {
        (
            DataType::Float32,
            Arc::new(Float32Array::from(
                (0..n).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
            )),
        )
    };
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
            Field::new("data", data_type, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(
                (0..n as i64)
                    .map(|r| (r + 1) % n as i64)
                    .collect::<Vec<_>>(),
            )),
            data,
        ],
    )
    .unwrap()
}

/// var×var COO in the Int32 form `write_varp` documents.
fn coo_i32(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(
                (0..n as i32)
                    .map(|r| (r + 1) % n as i32)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// A multimodal file whose **only** pairwise graph is scoped to a modality.
///
/// This is the input the global/per-modality scope split exists for.
/// `compact_multimodal` detects per-modality obsp/varp, warns that it cannot
/// round-trip them ("no per-modality pairwise reader"), and drops them — so a
/// file-wide "obsp is remapped" policy turns a compact that had always
/// succeeded into a hard failure.
pub fn fixture_multimodal_per_modality_obsp(dir: &Path, name: &str) -> PathBuf {
    const RNA_VARS: usize = 8;
    const ADT_VARS: usize = 4;
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, RNA_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();
    writer.write_obs(&obs_batch()).unwrap();

    let ids: Vec<(u8, usize)> = [
        ("rna", ModalityType::Rna, RNA_VARS),
        ("adt", ModalityType::Protein, ADT_VARS),
    ]
    .into_iter()
    .map(|(name, ty, n_vars)| {
        let id = writer
            .add_modality(name, ty, CodecId::None, ValueEncoding::Uint8, false)
            .unwrap();
        writer.write_var_for(id, &var_batch(n_vars, name)).unwrap();
        writer.set_modality_n_vars(id, n_vars as u64).unwrap();
        (id, n_vars)
    })
    .collect();

    for &(id, n_vars) in &ids {
        let (indptr, indices, values) = csr_rows(0, N_OBS, n_vars, 1);
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(id, 0, shard).unwrap();
    }

    // The point of the fixture: an obsp graph on the rna modality and none at
    // file scope, so a scope-blind audit sees "obsp present in, absent out".
    let rna_id = ids[0].0;
    writer
        .with_modality::<_, (), scx_format_io::ScxError>(rna_id, |w| {
            w.write_obsp("connectivities", &coo_batch(N_OBS))?;
            Ok(())
        })
        .unwrap();

    writer.finish().unwrap();
    path
}

/// A multimodal file carrying pairwise graphs at **both** scopes.
///
/// `fixture_multimodal_per_modality_obsp` deliberately has only the
/// modality-scoped one, so it cannot distinguish "the global carry ran" from
/// "nothing ran". This one can: the file-scope graph must survive a merge and
/// the modality-scoped one must not, in the same output.
///
/// That pair is the whole reason `SectionScope` exists. A file with one of each
/// is also the case where a scope-blind audit is *most* wrong — the surviving
/// global copy vouches for the lost per-modality one and the loss goes
/// unremarked.
pub fn fixture_multimodal_both_obsp_scopes(dir: &Path, name: &str) -> PathBuf {
    const RNA_VARS: usize = 8;
    const ADT_VARS: usize = 4;
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, RNA_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();
    writer.write_obs(&obs_batch()).unwrap();

    // File scope, before any modality is registered — obs is shared across
    // modalities, so an obs x obs graph over it is well defined.
    writer
        .write_obsp("global_connectivities", &coo_batch(N_OBS))
        .unwrap();

    let ids: Vec<(u8, usize)> = [
        ("rna", ModalityType::Rna, RNA_VARS),
        ("adt", ModalityType::Protein, ADT_VARS),
    ]
    .into_iter()
    .map(|(name, ty, n_vars)| {
        let id = writer
            .add_modality(name, ty, CodecId::None, ValueEncoding::Uint8, false)
            .unwrap();
        writer.write_var_for(id, &var_batch(n_vars, name)).unwrap();
        writer.set_modality_n_vars(id, n_vars as u64).unwrap();
        (id, n_vars)
    })
    .collect();

    for &(id, n_vars) in &ids {
        let (indptr, indices, values) = csr_rows(0, N_OBS, n_vars, 1);
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(id, 0, shard).unwrap();
    }

    let rna_id = ids[0].0;
    writer
        .with_modality::<_, (), scx_format_io::ScxError>(rna_id, |w| {
            w.write_obsp("rna_connectivities", &coo_batch(N_OBS))?;
            Ok(())
        })
        .unwrap();

    writer.finish().unwrap();
    path
}

/// A file whose `obsm` is **row-sharded**, so a rewrite has something to
/// collapse.
///
/// `fixture_all_families` writes obsm as one legacy section, which cannot tell
/// "the layout was preserved" from "the layout was rebuilt into one section".
pub fn fixture_with_sharded_obsm(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();
    writer.write_obs(&obs_batch()).unwrap();
    writer.write_var(&var_batch(N_VARS, "gene")).unwrap();
    for row_start in (0..N_OBS).step_by(SHARD_ROWS) {
        let (indptr, indices, values) = csr_rows(row_start, SHARD_ROWS, N_VARS, 1);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
        writer
            .write_obsm_shard(
                "X_pca",
                (row_start / SHARD_ROWS) as u32,
                row_start as u64,
                SHARD_ROWS as u64,
                N_OBS as u64,
                &dense_embedding(SHARD_ROWS),
            )
            .unwrap();
    }
    writer.finish().unwrap();
    path
}

/// The all-families fixture with its `obsp` graph under a different key, or
/// with a `Float64` `data` column instead of `Float32`.
///
/// Two knobs, one fixture, because a merge test needs two inputs that differ in
/// **exactly one** thing — merge validates var identity and requires identical
/// layers, so the two sides cannot be unrelated files.
pub fn fixture_all_families_obsp(
    dir: &Path,
    name: &str,
    key: &'static str,
    f64_values: bool,
) -> PathBuf {
    build_all_families(
        dir,
        name,
        FixtureShape {
            obsp_key: Some(key),
            obsp_f64_values: f64_values,
            ..FixtureShape::default()
        },
    )
}

/// A file with two obsp keys where one key's name is a **prefix** of the
/// other's shard naming: `g` and `g_shard_x`.
///
/// `obsp/g_shard_0` (a shard of `g`) and `obsp/g_shard_x_shard_0` (a shard of
/// `g_shard_x`) both start with `obsp/g_shard_`, so a lookup for `g` that only
/// tests the prefix swallows the other key's shard as well.
pub fn fixture_with_colliding_obsp_keys(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();
    writer.write_obs(&obs_batch()).unwrap();
    writer.write_var(&var_batch(N_VARS, "gene")).unwrap();
    for row_start in (0..N_OBS).step_by(SHARD_ROWS) {
        let (indptr, indices, values) = csr_rows(row_start, SHARD_ROWS, N_VARS, 1);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    for key in ["g", "g_shard_x"] {
        writer
            .write_obsp_shard_coo(key, 0, 0, N_OBS as u64, N_OBS as u64, &coo_batch(N_OBS))
            .unwrap();
    }
    writer.finish().unwrap();
    path
}

/// The obsm analogue of [`fixture_with_colliding_obsp_keys`]: keys `g` and
/// `g_shard_x`, both **sharded**, plus a legacy single-section key whose own
/// name contains `_shard_`.
pub fn fixture_with_colliding_obsm_keys(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();
    writer.write_obs(&obs_batch()).unwrap();
    writer.write_var(&var_batch(N_VARS, "gene")).unwrap();
    for row_start in (0..N_OBS).step_by(SHARD_ROWS) {
        let (indptr, indices, values) = csr_rows(row_start, SHARD_ROWS, N_VARS, 1);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    for key in ["g", "g_shard_x"] {
        for (idx, row_start) in (0..N_OBS).step_by(SHARD_ROWS).enumerate() {
            writer
                .write_obsm_shard(
                    key,
                    idx as u32,
                    row_start as u64,
                    SHARD_ROWS as u64,
                    N_OBS as u64,
                    &dense_embedding(SHARD_ROWS),
                )
                .unwrap();
        }
    }
    // A legacy single section whose key contains the shard marker. The reader
    // lists it; the merge key scan used to drop it.
    writer
        .write_obsm("legacy_shard_name", &dense_embedding(N_OBS))
        .unwrap();
    writer.finish().unwrap();
    path
}

/// [`fixture_all_families`] carrying an X CSC sidecar `cols_per_shard` columns
/// wide, added by the real op (`rebuild_csc_inplace`, i.e. `scx build-csc`).
///
/// For the rewrite ops' carry arms: they build a new sidecar from their own
/// output iff the input had one, so the input has to have one. It is a
/// separate fixture rather than a change to `fixture_all_families` because the
/// sidecar would otherwise move every other op's digest.
pub fn fixture_all_families_with_csc(dir: &Path, name: &str, cols_per_shard: usize) -> PathBuf {
    let path = fixture_all_families(dir, name);
    scx_ops::rebuild_csc_inplace(
        &path,
        cols_per_shard,
        "4G",
        scx_ops::framing_for_csc_rebuild(&path),
        None,
    )
    .unwrap();
    path
}
