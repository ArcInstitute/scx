//! Sharded obs on ingest (organization phase 6c, review item ORG-11.16-4).
//!
//! Every `scx-convert` ingest path used to write obs as one unsharded
//! `ObsMetadata` section at any scale, while `pyscx.from_anndata` and
//! `scx optimize --shard-obs` sharded above `shard_target_rows`. Two
//! consequences, in the order the evidence supports them:
//!
//! 1. **The coverage hole.** With convert output always single-section, every
//!    h5ad export *in this test suite* took the whole-batch driver, so the
//!    shard-stream writer's multi-shard behaviours — declared-vocabulary union
//!    across shards, per-shard hyperslab appends, cross-shard category ordering
//!    — were reachable only from `from_anndata`-produced files. §11.1 lived in
//!    exactly that gap.
//! 2. **Layout divergence.** The same h5ad produced a different on-disk obs
//!    layout depending on which entry point read it.
//!
//! What this does **not** fix, measured rather than assumed: the >2 GB obs
//! string column. That panic is on the *read* side
//! (`h5ad/read.rs`'s `StringArray::from`), strictly before any of this, and
//! slicing an already-materialized batch is zero-copy so peak RSS is unchanged.
//! `obs_section_is_written_wide` below pins the half of that finding which
//! **is** already handled, so the claim rests on a test rather than on a
//! reading of the writer.

use super::convert_tests_common::*;

use arrow::array::{Array, DictionaryArray, StringArray};
use arrow::datatypes::Int32Type;
use hdf5::types::VarLenUnicode;
use scx_format_io::ObsShardPolicy;

use crate::h5mu::pipeline::{h5mu_to_scx, h5mu_to_scx_streaming};
use crate::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ingest options with an explicit obs-shard policy and a small
/// `shard_target_rows`, so a modest fixture crosses the `Auto` threshold
/// without a large-`n_obs` test.
fn opts_with(policy: ObsShardPolicy, shard_target_rows: u32, stream: bool) -> ConvertOptions {
    ConvertOptions {
        shard_target_rows,
        stream,
        obs_shard_policy: policy,
        ..ConvertOptions::default()
    }
}

/// Append an anndata `categorical` group (encoding-version 0.2.0) to an
/// existing h5ad's `/obs`, with a **declared** vocabulary that deliberately
/// names a level no row uses.
///
/// The unused level is the payload: it is what distinguishes "the writer
/// re-derived the vocabulary from the data" from "the writer preserved what
/// pandas declared", and it survives an unfiltered export by the rule 6b
/// settled on (prune only under a row filter).
fn append_declared_categorical(
    path: &Path,
    col: &str,
    declared: &[&str],
    codes_for_row: impl Fn(usize) -> i32,
    n_obs: usize,
    ordered: bool,
) {
    let file = hdf5::File::append(path).unwrap();
    let obs = file.group("obs").unwrap();
    let grp = obs.create_group(col).unwrap();

    grp.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("categorical"))
        .unwrap();
    grp.new_attr::<VarLenUnicode>()
        .create("encoding-version")
        .unwrap()
        .write_scalar(&vlu("0.2.0"))
        .unwrap();
    grp.new_attr::<bool>()
        .create("ordered")
        .unwrap()
        .write_scalar(&ordered)
        .unwrap();

    let cats: Vec<VarLenUnicode> = declared.iter().map(|s| vlu(s)).collect();
    grp.new_dataset::<VarLenUnicode>()
        .shape([cats.len()])
        .create("categories")
        .unwrap()
        .write(&cats)
        .unwrap();

    let codes: Vec<i32> = (0..n_obs).map(&codes_for_row).collect();
    grp.new_dataset::<i32>()
        .shape([n_obs])
        .create("codes")
        .unwrap()
        .write(&codes)
        .unwrap();

    // These fixtures carry no `column-order` attribute, so
    // `read_dataframe_group` enumerates `member_names()` and picks the new
    // group up on its own. (hdf5-metno exposes no attribute deletion, so a
    // fixture that *did* declare a column order could not be extended here.)
    assert!(
        obs.attr("column-order").is_err(),
        "fixture gained a `column-order` attribute; this helper would be \
         silently dropping '{col}' from the frame"
    );
}

/// The exported h5ad's declared category list for `/obs/<col>`, in declared
/// order, plus its `ordered` bit.
fn exported_categories(h5ad: &Path, col: &str) -> (Vec<String>, bool) {
    let f = hdf5::File::open(h5ad).unwrap();
    let g = f.group("obs").unwrap().group(col).unwrap();
    let cats: Vec<VarLenUnicode> = g.dataset("categories").unwrap().read_1d().unwrap().to_vec();
    let ordered = g
        .attr("ordered")
        .ok()
        .and_then(|a| a.read_scalar::<bool>().ok())
        .unwrap_or(false);
    (cats.iter().map(|s| s.to_string()).collect(), ordered)
}

/// Render one obs column of a `read_obs()` batch as comparable strings, so two
/// storage layouts can be compared without caring which Arrow encoding each
/// arrived in.
fn column_as_strings(batch: &arrow::record_batch::RecordBatch, idx: usize) -> Vec<Option<String>> {
    let arr = batch.column(idx);
    if let Some(d) = arr.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
        let values = d.values();
        let values = values.as_any().downcast_ref::<StringArray>().unwrap();
        return (0..d.len())
            .map(|i| {
                if d.is_null(i) {
                    None
                } else {
                    Some(values.value(d.keys().value(i) as usize).to_string())
                }
            })
            .collect();
    }
    let fmt = arrow::util::display::ArrayFormatter::try_new(
        arr.as_ref(),
        &arrow::util::display::FormatOptions::default(),
    )
    .unwrap();
    (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(fmt.value(i).to_string())
            }
        })
        .collect()
}

/// Compare two `read_obs()` results field by field: schema names in order,
/// every value, and — for categoricals — the full declared dictionary and the
/// `ordered` bit. A dictionary-unification bug in `assemble_sharded_metadata`
/// surfaces here and nowhere else.
fn assert_obs_equivalent(
    a: &arrow::record_batch::RecordBatch,
    b: &arrow::record_batch::RecordBatch,
) {
    assert_eq!(a.num_rows(), b.num_rows(), "row count");
    let (sa, sb) = (a.schema(), b.schema());
    let names_a: Vec<&str> = sa.fields().iter().map(|f| f.name().as_str()).collect();
    let names_b: Vec<&str> = sb.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names_a, names_b, "column names / order");

    for (i, name) in names_a.iter().enumerate() {
        assert_eq!(
            column_as_strings(a, i),
            column_as_strings(b, i),
            "values differ in column '{name}'"
        );
        assert_eq!(
            a.schema().field(i).metadata(),
            b.schema().field(i).metadata(),
            "field metadata (e.g. the categorical `ordered` bit) differs in column '{name}'"
        );
        // The declared dictionary, not just the values it happens to reference.
        let dict_of = |batch: &arrow::record_batch::RecordBatch| -> Option<Vec<String>> {
            let col = batch.column(i);
            col.as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .map(|d| {
                    let v = d.values();
                    let v = v.as_any().downcast_ref::<StringArray>().unwrap();
                    (0..v.len()).map(|k| v.value(k).to_string()).collect()
                })
        };
        assert_eq!(
            dict_of(a),
            dict_of(b),
            "declared category list differs in column '{name}'"
        );
    }
}

// ---------------------------------------------------------------------------
// Every ingest entry point emits sharded obs above the threshold
// ---------------------------------------------------------------------------

#[test]
fn eager_h5ad_ingest_shards_obs_above_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 40, 6, "csr", false);

    h5ad_to_scx(
        &h5ad,
        &scx,
        &opts_with(ObsShardPolicy::Auto, 10, false),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(
        reader.obs_metadata_shard_count(),
        4,
        "40 rows at shard_target_rows=10 must be 4 ObsMetadataShard sections"
    );
    assert_eq!(reader.read_obs().unwrap().num_rows(), 40);
}

#[test]
fn streaming_h5ad_ingest_shards_obs_above_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 40, 6, "csr", false);

    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts_with(ObsShardPolicy::Auto, 10, true),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 4);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 40);
}

#[test]
fn tenx_ingest_shards_obs_above_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    let scx = dir.path().join("out.scx");
    create_test_tenx_h5(&tenx, 40, 6);

    tenx_to_scx(
        &tenx,
        &scx,
        &opts_with(ObsShardPolicy::Auto, 10, false),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 4);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 40);
}

#[test]
fn eager_h5mu_ingest_shards_outer_obs_above_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("in.h5mu");
    let scx = dir.path().join("out.scx");
    create_test_h5mu(&h5mu, 40, 6, 4);

    h5mu_to_scx(
        &h5mu,
        &scx,
        &opts_with(ObsShardPolicy::Auto, 10, false),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 4);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 40);
}

#[test]
fn streaming_h5mu_ingest_shards_outer_obs_above_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("in.h5mu");
    let scx = dir.path().join("out.scx");
    create_test_h5mu(&h5mu, 40, 6, 4);

    h5mu_to_scx_streaming(
        &h5mu,
        &scx,
        &opts_with(ObsShardPolicy::Auto, 10, true),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 4);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 40);
}

// ---------------------------------------------------------------------------
// The policy, on the same boundary `optimize` and `from_anndata` use
// ---------------------------------------------------------------------------

#[test]
fn shard_obs_off_keeps_a_single_section_however_large() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 40, 6, "csr", false);

    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts_with(ObsShardPolicy::Off, 10, true),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(
        reader.obs_metadata_shard_count(),
        0,
        "`off` must preserve the legacy single ObsMetadata section"
    );
    assert_eq!(reader.read_obs().unwrap().num_rows(), 40);
}

#[test]
fn shard_obs_always_shards_a_sub_threshold_file() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 6, 4, "csr", false);

    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        // 6 rows at a target of 100: `Auto` would not shard, `Always` must.
        &opts_with(ObsShardPolicy::Always, 100, true),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 1);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 6);
}

/// `Auto`'s boundary is a strict `>`, matching `from_anndata`'s `obs_rows > step`
/// and `ObsShardPolicy::auto_uses_strict_gt_threshold`. Exactly at the target,
/// a file stays single-section — a `>=` would silently re-shape every file whose
/// row count happens to land on a shard boundary.
#[test]
fn auto_does_not_shard_at_exactly_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    for (n_obs, target, want_shards) in [(20usize, 20u32, 0usize), (21, 20, 2)] {
        let h5ad = dir.path().join(format!("in_{n_obs}.h5ad"));
        let scx = dir.path().join(format!("out_{n_obs}.scx"));
        create_test_h5ad(&h5ad, n_obs, 4, "csr", false);

        h5ad_to_scx_streaming(
            &h5ad,
            &scx,
            &opts_with(ObsShardPolicy::Auto, target, true),
            &StreamingOverrides::default(),
            &mut WarningSink::log(),
        )
        .unwrap();

        let reader = ScxReader::open(&scx).unwrap();
        assert_eq!(
            reader.obs_metadata_shard_count(),
            want_shards,
            "n_obs={n_obs} at shard_target_rows={target}"
        );
    }
}

// ---------------------------------------------------------------------------
// The two layouts must agree on what obs *is*
// ---------------------------------------------------------------------------

/// A sharded convert and an `--shard-obs off` convert of the same input must
/// read back as the same obs table — same columns, same values, same declared
/// category lists, same `ordered` bits. This is where a dictionary
/// widen/unify bug in `assemble_sharded_metadata` would land: slicing gives
/// every shard the *same* dictionary, so the concat path has to dedup N copies
/// back down to one.
#[test]
fn sharded_and_single_section_obs_read_back_identically() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    create_test_h5ad_with_cell_type(&h5ad, 40, 6);
    append_declared_categorical(
        &h5ad,
        "phase",
        &["G1", "S", "G2M", "M"],
        |i| (i % 3) as i32,
        40,
        true,
    );

    let sharded = dir.path().join("sharded.scx");
    let single = dir.path().join("single.scx");
    h5ad_to_scx_streaming(
        &h5ad,
        &sharded,
        &opts_with(ObsShardPolicy::Auto, 10, true),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx_streaming(
        &h5ad,
        &single,
        &opts_with(ObsShardPolicy::Off, 10, true),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let ra = ScxReader::open(&sharded).unwrap();
    let rb = ScxReader::open(&single).unwrap();
    assert_eq!(ra.obs_metadata_shard_count(), 4);
    assert_eq!(
        rb.obs_metadata_shard_count(),
        0,
        "control arm must be single-section"
    );

    let (oa, ob) = (ra.read_obs().unwrap(), rb.read_obs().unwrap());
    // Without this the comparison is happy to agree that neither side has the
    // categorical it exists to compare.
    assert!(
        oa.schema().field_with_name("phase").is_ok(),
        "fixture's declared categorical never reached obs; the test would \
         compare two frames that agree by both missing it"
    );
    assert_obs_equivalent(&oa, &ob);
    // var is deliberately untouched by this phase.
    assert_eq!(ra.var_metadata_shard_count(), 0);
    assert_eq!(rb.var_metadata_shard_count(), 0);
}

// ---------------------------------------------------------------------------
// The coverage this phase exists to buy
// ---------------------------------------------------------------------------

/// The point of ORG-11.16-4: a `scx convert` output now exports through the
/// **multi-shard** h5ad dataframe writer, so its declared-vocabulary handling
/// is on the common test path rather than reachable only from
/// `from_anndata`-produced files.
///
/// ⚠️ The shard-count assertion is load-bearing, not decoration. 6b shipped a
/// two-arm pyscx test whose arms were secretly the same arm, because nothing
/// asserted which writer each one reached. Without the assertion below this
/// test silently degrades into another whole-batch export the moment a default
/// moves, and it would still pass.
#[test]
fn a_converted_file_exports_through_the_multi_shard_writer() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("mid.scx");
    let out = dir.path().join("out.h5ad");
    create_test_h5ad(&h5ad, 40, 6, "csr", false);
    // "M" is declared but no row uses it: codes cycle over the first three.
    append_declared_categorical(
        &h5ad,
        "phase",
        &["G1", "S", "G2M", "M"],
        |i| (i % 3) as i32,
        40,
        true,
    );

    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts_with(ObsShardPolicy::Auto, 10, true),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert!(
        reader.obs_metadata_shard_count() > 1,
        "this test only means anything against the shard-stream writer; got {} obs shards",
        reader.obs_metadata_shard_count()
    );
    drop(reader);

    scx_to_h5ad(&scx, &out, &mut WarningSink::log()).unwrap();

    let (cats, ordered) = exported_categories(&out, "phase");
    assert_eq!(
        cats,
        vec!["G1", "S", "G2M", "M"],
        "declared order and the unused level must survive an unfiltered export"
    );
    assert!(ordered, "the `ordered` bit must survive");
}

// ---------------------------------------------------------------------------
// The §11.3 half that is already handled
// ---------------------------------------------------------------------------

/// Review §11.3 lists "`write_obs` hits Arrow IPC's 2 GB narrow-offset ceiling"
/// alongside the read-side panic. It does not: `ScxWriter::write_arrow_ipc`
/// upcasts `Utf8 → LargeUtf8` before encoding, so a single obs section is
/// already immune on the write side regardless of sharding.
///
/// This pins the mechanism at a size a test can afford. Reading the writer and
/// asserting the conclusion is what the PR body must not do; this is the
/// evidence that replaces it.
#[test]
fn obs_section_is_written_wide() {
    use arrow::datatypes::DataType;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 8, 4, "csr", false);

    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        // Single-section: the claim under test is about `write_obs`, not shards.
        &opts_with(ObsShardPolicy::Off, 100, true),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 0);

    // Read the section's own Arrow IPC bytes. `read_obs()` runs
    // `downcast_large_types` on the way out, so it cannot answer this — it
    // would report `Utf8` for a payload that is `LargeUtf8` on disk.
    let entry = reader
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == scx_format_io::section::SectionType::ObsMetadata)
        .expect("single-section obs");
    let bytes = reader.section_bytes(entry).unwrap();
    let ipc = arrow::ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
    let schema = ipc.schema();
    let string_cols: Vec<(&str, &DataType)> = schema
        .fields()
        .iter()
        .map(|f| (f.name().as_str(), f.data_type()))
        .filter(|(_, dt)| matches!(dt, DataType::Utf8 | DataType::LargeUtf8))
        .collect();
    assert!(
        !string_cols.is_empty(),
        "fixture has no string obs column, so this asserts nothing; schema was {:?}",
        schema.fields()
    );
    for (name, dt) in string_cols {
        assert_eq!(
            dt,
            &DataType::LargeUtf8,
            "obs column '{name}' was written narrow. `write_arrow_ipc` must widen \
             Utf8 before IPC, or a >2 GB obs column would overflow i32 offsets on \
             the write side too"
        );
    }
}
