//! The hdf5-gated half of `scx doublet-import`'s source handling, driven
//! through the real binary.
//!
//! File-level gated on purpose: hdf5-gated tests live in hdf5-gated test
//! binaries so CI's hdf5 lane selects them with `--test doublet_import_hdf5`
//! (the dedup-guard's "test-hdf5 lane selection lists" step pins exactly that)
//! and no per-name recovery invocation exists for a gated test hiding inside
//! an ungated binary. The ungated sibling `doublet_import_cli.rs` carries the
//! `cfg(not(feature = "hdf5"))` counterpart and everything libhdf5-free.
#![cfg(feature = "hdf5")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;

fn scx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scx"))
}

fn var_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(ids))],
    )
    .unwrap()
}

fn simple_fixture(dir: &Path, name: &str) -> PathBuf {
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "barcode",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec![
            "AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1",
        ]))],
    )
    .unwrap();
    let n_obs = obs.num_rows();
    let n_vars = 2;
    let path = dir.join(name);
    let mut w = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 16384, 0, 0),
    )
    .unwrap();
    w.write_obs(&obs).unwrap();
    w.write_var(&var_batch(n_vars)).unwrap();
    w.write_csr_shard(
        &vec![0u64; n_obs + 1],
        &[],
        &[],
        CodecId::None,
        ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    w.finish().unwrap();
    path
}

fn write_text(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

/// With `--features hdf5` the extension no longer decides — the file's
/// contents do. A file named `.h5ad` that is not HDF5 fails as an unreadable
/// file, naming the format it tried. (Moved from `doublet_import_cli.rs`,
/// where it sat `#[cfg(feature = "hdf5")]` inside an ungated binary that CI's
/// hdf5 lane had to recover by exact test name.)
#[test]
fn a_non_hdf5_h5ad_source_fails_as_an_unreadable_file() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let h5 = write_text(dir.path(), "scrublet_out.h5ad", "not really hdf5");

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            h5.to_str().unwrap(),
            "--tool",
            "scrublet",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("as HDF5"), "{msg}");
}
