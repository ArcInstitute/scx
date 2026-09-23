use super::*;
use crate::catalog::FullCatalogEntry;
use crate::encoder::{encode_one_shard, EncodeShardOptions};
use crate::header::FileHeader;
use crate::reader::ScxReader;
use crate::section::SectionType;
use crate::writer::ScxWriter;
use std::path::Path;

const N_OBS: usize = 30;
const N_VARS: usize = 20;
const SHARD_ROWS: usize = 10;

/// Three 10-row shards. `wide` puts values above `u8::MAX` in the last one,
/// so the sidecar's encoding has to widen past the first shard's.
fn shard(i: usize, wide: bool) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for r in i * SHARD_ROWS..(i + 1) * SHARD_ROWS {
        for c in 0..N_VARS {
            if (r * 7 + c * 3) % 5 == 0 {
                indices.push(c as u32);
                let v = ((r + c) % 200 + 1) as f32;
                values.push(if wide && i == 2 { v * 300.0 } else { v });
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

#[derive(Clone, Copy)]
enum Via {
    /// `write_csr_shard`: the writer encodes.
    Buffers,
    /// `write_preencoded_shard`: encoded elsewhere.
    PreEncoded,
    /// `write_csr_shard_raw_copy`: a complete section's bytes.
    Section,
    /// `copy_section_verbatim`: a section plus its source catalog entry.
    Verbatim,
}

fn header(v4: bool) -> FileHeader {
    let mut h = FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, 10, 0, 0);
    if v4 {
        h.format_version = crate::header::CURRENT_FORMAT_VERSION;
    }
    h
}

fn preencoded(i: usize, v4: bool, wide: bool) -> crate::writer::PreEncodedSection {
    let (indptr, indices, values) = shard(i, wide);
    let mut opts = EncodeShardOptions::new(
        format!("X_shard_{i}"),
        SectionType::CsrShard,
        N_VARS as u64,
        (i * SHARD_ROWS) as u64,
        0,
    );
    opts.framing = v4.then(FramingConfig::default);
    encode_one_shard(&indptr, &indices, &values, &opts).unwrap()
}

fn write_x(writer: &mut ScxWriter, via: Via, v4: bool, wide: bool) {
    for i in 0..N_OBS / SHARD_ROWS {
        match via {
            Via::Buffers => {
                let (indptr, indices, values) = shard(i, wide);
                let enc = scx_codec::detect_value_encoding(&values);
                let raw = scx_codec::values_to_raw_bytes(&values, enc).unwrap();
                writer
                    .write_csr_shard(
                        &indptr,
                        &indices,
                        &raw,
                        CodecId::Zstd,
                        enc,
                        (i * SHARD_ROWS) as u64,
                    )
                    .unwrap();
            }
            Via::PreEncoded => writer
                .write_preencoded_shard(preencoded(i, v4, wide))
                .unwrap(),
            Via::Section | Via::Verbatim => {
                let p = preencoded(i, v4, wide);
                let mut bytes = p.header_buf.clone();
                bytes.extend_from_slice(&p.encoded.indptr_bytes);
                bytes.extend_from_slice(&p.encoded.indices_bytes);
                bytes.extend_from_slice(&p.encoded.values_bytes);
                bytes.extend_from_slice(&p.block_index_bytes);
                if matches!(via, Via::Section) {
                    writer
                        .write_csr_shard_raw_copy(&bytes, p.stats, p.nnz)
                        .unwrap();
                } else {
                    let entry = FullCatalogEntry {
                        name: p.name.clone(),
                        offset: 0,
                        length: bytes.len() as u64,
                        section_type: SectionType::CsrShard,
                        checksum: crate::checksum::blake3_hash(&bytes),
                        modality_id: 0,
                        stats: Some(p.stats),
                    };
                    writer.copy_section_verbatim(&entry, &bytes).unwrap();
                }
            }
        }
    }
}

fn opts(cols: usize) -> CscBuildOptions {
    CscBuildOptions {
        cols_per_shard: cols,
        ..Default::default()
    }
}

fn build(dir: &Path, name: &str, via: Via, v4: bool, wide: bool) -> PathBuf {
    let mut w = ScxWriter::new(dir.join(name), header(v4)).unwrap();
    if v4 {
        w.set_framing(Some(FramingConfig::default()));
    }
    w.enable_csc_sidecar(opts(7)).unwrap();
    write_x(&mut w, via, v4, wide);
    w.emit_csc_sidecar().unwrap().expect("a sidecar");
    w.finish().unwrap()
}

fn csc_entries(r: &ScxReader) -> Vec<(String, u64, [u8; 32])> {
    r.catalog()
        .csc_shards_sorted()
        .into_iter()
        .map(|e: &FullCatalogEntry| (e.name.clone(), e.length, e.checksum))
        .collect()
}

fn dense_from_csr(r: &ScxReader) -> Vec<f32> {
    let csr = r.read_all_csr_shards().unwrap();
    let mut d = vec![0f32; N_OBS * N_VARS];
    for row in 0..N_OBS {
        for k in csr.indptr[row] as usize..csr.indptr[row + 1] as usize {
            d[row * N_VARS + csr.indices[k] as usize] = csr.data[k];
        }
    }
    d
}

fn dense_from_csc(r: &ScxReader) -> Vec<f32> {
    let csc = r.read_all_csc_shards().unwrap();
    let mut d = vec![0f32; N_OBS * N_VARS];
    for col in 0..N_VARS {
        for k in csc.indptr[col] as usize..csc.indptr[col + 1] as usize {
            d[csc.indices[k] as usize * N_VARS + col] = csc.data[k];
        }
    }
    d
}

/// The sidecar is X's transpose, and it does not matter which writer method
/// put X on disk: the byte paths give the same CSC bytes. A feed missing from
/// any one path leaves that file with a short or empty sidecar.
#[test]
fn every_x_write_path_feeds_the_same_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    for v4 in [false, true] {
        let mut seen = Vec::new();
        for (tag, via) in [
            ("buffers", Via::Buffers),
            ("pre", Via::PreEncoded),
            ("section", Via::Section),
            ("verbatim", Via::Verbatim),
        ] {
            let path = build(dir.path(), &format!("{tag}_{v4}.scx"), via, v4, true);
            let r = ScxReader::open(&path).unwrap();
            assert!(r.header().has_csc(), "{tag}/v4={v4}: no sidecar");
            assert_eq!(
                r.header().n_csc_shards,
                3,
                "{tag}/v4={v4}: 20 cols at 7 per shard"
            );
            assert_eq!(dense_from_csc(&r), dense_from_csr(&r), "{tag}/v4={v4}");
            assert_eq!(
                r.catalog().csc_build_generation,
                r.catalog().data_generation,
                "{tag}/v4={v4}: a same-pass sidecar is fresh"
            );
            // The last shard's values exceed u8, so the sidecar widened.
            let sh = r
                .read_shard_header(r.catalog().csc_shards_sorted()[0])
                .unwrap();
            assert_eq!(
                sh.value_encoding,
                ValueEncoding::Uint16 as u8,
                "{tag}/v4={v4}"
            );
            // Framed iff the file is v4.
            assert_eq!(
                sh.shard_format_version > crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION,
                v4,
                "{tag}: framing must follow the file"
            );
            seen.push(csc_entries(&r));
        }
        // Buffers and PreEncoded encode with different candidate codecs
        // (`write_csr_shard` is handed Zstd, `encode_one_shard` picks its
        // own), and the sidecar takes the first shard's codec — so compare the
        // two byte paths, which share their encoded shards, for bytes.
        assert_eq!(seen[1], seen[2], "v4={v4}: pre-encoded vs raw-copy sidecar");
        assert_eq!(seen[1], seen[3], "v4={v4}: pre-encoded vs verbatim sidecar");
    }
}

/// Layers and `adata.raw` are written through the same methods as X; only X
/// is fed. A layer fed into the builder would restart at row 0 and fail the
/// row-order check.
#[test]
fn layer_and_raw_shards_are_not_fed() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ScxWriter::new(dir.path().join("l.scx"), header(false)).unwrap();
    w.enable_csc_sidecar(opts(7)).unwrap();
    w.set_raw_n_vars(N_VARS as u64);
    for i in 0..N_OBS / SHARD_ROWS {
        let (indptr, indices, values) = shard(i, false);
        let raw = scx_codec::values_to_raw_bytes(&values, ValueEncoding::Uint8).unwrap();
        let row = (i * SHARD_ROWS) as u64;
        w.write_csr_shard(
            &indptr,
            &indices,
            &raw,
            CodecId::Zstd,
            ValueEncoding::Uint8,
            row,
        )
        .unwrap();
        let buf = crate::writer::ShardBuffers::new(
            &indptr,
            &indices,
            &raw,
            CodecId::Zstd,
            ValueEncoding::Uint8,
        );
        w.write_layer_csr_shard("counts", i as u32, row, buf)
            .unwrap();
        w.write_raw_csr_shard(
            &indptr,
            &indices,
            &raw,
            CodecId::Zstd,
            ValueEncoding::Uint8,
            row,
        )
        .unwrap();
    }
    w.emit_csc_sidecar().unwrap().expect("a sidecar");
    let r = ScxReader::open(w.finish().unwrap()).unwrap();
    assert_eq!(dense_from_csc(&r), dense_from_csr(&r));
}

/// A shard out of row order is refused at the write, and so is an X shard
/// after the sidecar was emitted — either would leave CSR and CSC disagreeing.
#[test]
fn an_out_of_order_or_late_x_shard_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (indptr, indices, values) = shard(0, false);
    let raw = scx_codec::values_to_raw_bytes(&values, ValueEncoding::Uint8).unwrap();

    let mut w = ScxWriter::new(dir.path().join("a.scx"), header(false)).unwrap();
    w.enable_csc_sidecar(opts(7)).unwrap();
    let err = w
        .write_csr_shard(
            &indptr,
            &indices,
            &raw,
            CodecId::Zstd,
            ValueEncoding::Uint8,
            10,
        )
        .unwrap_err();
    assert!(err.to_string().contains("row order"), "{err}");

    let mut w = ScxWriter::new(dir.path().join("b.scx"), header(false)).unwrap();
    w.enable_csc_sidecar(opts(7)).unwrap();
    w.write_csr_shard(
        &indptr,
        &indices,
        &raw,
        CodecId::Zstd,
        ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    // 10 rows pushed of 30: the builder refuses to emit a short sidecar.
    assert!(w.emit_csc_sidecar().is_err());

    let mut w = ScxWriter::new(dir.path().join("c.scx"), header(false)).unwrap();
    w.enable_csc_sidecar(opts(7)).unwrap();
    write_x(&mut w, Via::Buffers, false, false);
    w.emit_csc_sidecar().unwrap();
    let err = w
        .write_csr_shard(
            &indptr,
            &indices,
            &raw,
            CodecId::Zstd,
            ValueEncoding::Uint8,
            30,
        )
        .unwrap_err();
    assert!(err.to_string().contains("after the same-pass CSC"), "{err}");
}

/// Enabling after X has started, twice, or on a multimodal writer is refused.
#[test]
fn enable_is_refused_where_it_cannot_see_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ScxWriter::new(dir.path().join("a.scx"), header(false)).unwrap();
    write_x(&mut w, Via::Buffers, false, false);
    assert!(w.enable_csc_sidecar(opts(7)).is_err());

    let mut w = ScxWriter::new(dir.path().join("b.scx"), header(false)).unwrap();
    w.enable_csc_sidecar(opts(7)).unwrap();
    assert!(w.enable_csc_sidecar(opts(7)).is_err());

    let mut w = ScxWriter::new(dir.path().join("c.scx"), header(false)).unwrap();
    w.add_modality(
        "rna",
        crate::modality::ModalityType::Rna,
        CodecId::None,
        ValueEncoding::Uint8,
        false,
    )
    .unwrap();
    assert!(w.enable_csc_sidecar(opts(7)).is_err());

    // And the reverse order: a modality registered after the sink exists (or
    // after it was emitted) is refused too.
    for emitted in [false, true] {
        let mut w =
            ScxWriter::new(dir.path().join(format!("d{emitted}.scx")), header(false)).unwrap();
        w.enable_csc_sidecar(opts(7)).unwrap();
        if emitted {
            write_x(&mut w, Via::Buffers, false, false);
            w.emit_csc_sidecar().unwrap();
        }
        let err = w
            .add_modality(
                "rna",
                crate::modality::ModalityType::Rna,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap_err();
        assert!(err.to_string().contains("single-modality"), "{err}");
    }
}

/// `finish()` emits a sidecar the caller never emitted.
#[test]
fn finish_emits_a_pending_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ScxWriter::new(dir.path().join("f.scx"), header(false)).unwrap();
    w.enable_csc_sidecar(opts(7)).unwrap();
    write_x(&mut w, Via::Buffers, false, false);
    let r = ScxReader::open(w.finish().unwrap()).unwrap();
    assert!(r.header().has_csc());
    assert_eq!(dense_from_csc(&r), dense_from_csr(&r));
}

/// Framing a sidecar on a ≤ v3 output is refused, as `build-csc` refuses it.
#[test]
fn framing_a_sidecar_on_a_v3_output_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ScxWriter::new(dir.path().join("v3.scx"), header(false)).unwrap();
    w.enable_csc_sidecar(CscBuildOptions {
        framing: Some(FramingConfig::default()),
        ..opts(7)
    })
    .unwrap();
    write_x(&mut w, Via::Buffers, false, false);
    let err = w.emit_csc_sidecar().unwrap_err();
    assert!(err.to_string().contains("v4"), "{err}");
}

/// No X shard, no sidecar — as on every other path.
#[test]
fn an_empty_matrix_carries_no_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = header(false);
    h.n_obs = 0;
    let mut w = ScxWriter::new(dir.path().join("e.scx"), h).unwrap();
    w.enable_csc_sidecar(opts(7)).unwrap();
    assert!(w.emit_csc_sidecar().unwrap().is_none());
    let r = ScxReader::open(w.finish().unwrap()).unwrap();
    assert!(!r.header().has_csc());
}
