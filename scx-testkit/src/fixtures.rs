//! Fixtures whose digest actually covers something.
//!
//! Nearly every SCX test fixture in the tree writes `CodecId::None` /
//! `ValueEncoding::Uint8` unframed shards, which means it exercises no codec at
//! all. An identity claim asserted on one of those is close to vacuous: the
//! encoder path it pins is the one that does nothing.
//!
//! [`mixed_codec_file`] writes three CSR shards tiling the obs axis, one per
//! encoder path — unframed integer, row-group-framed integer, and float
//! (`Pcodec`) — so a single digest covers all three.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::encoder::FramingConfig;
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;
use scx_format_io::Result;

/// One shard of [`mixed_codec_file`]: a row range and how to encode it.
#[derive(Debug, Clone, Copy)]
struct ShardSpec {
    rows: (usize, usize),
    codec: CodecId,
    encoding: ValueEncoding,
    /// `Some(G)` writes a row-group-framed (shard v2) shard.
    framed: Option<u32>,
}

/// Knobs for [`mixed_codec_file_with`].
///
/// Every field exists so the harness's own premise tests can prove the digest
/// detects that specific change — and, for `provenance_timestamp`, that it
/// deliberately does not. A fixture whose perturbations are untested is a
/// fixture nobody has shown can fail.
#[derive(Debug, Clone)]
pub struct FixtureOpts {
    pub n_obs: usize,
    pub n_vars: usize,
    /// Unix seconds stamped into the provenance entry. Varying **only** this
    /// must leave the digest unchanged — that is the whole premise of
    /// excluding `Provenance`.
    pub provenance_timestamp: i64,
    /// Write order of the three shard specs. Permuting it changes which row
    /// range lands in `X_shard_0`, so the digest must change.
    pub shard_order: [usize; 3],
    /// Codec for the framed integer shard (spec 1). `Scx1` by default;
    /// `Zstd` is the perturbation.
    pub framed_codec: CodecId,
    /// Flip the low mantissa bit of the first float value. The smallest
    /// change a float shard can carry, and invisible to any comparison that
    /// only checks lengths.
    pub perturb_float_bit: bool,
}

impl Default for FixtureOpts {
    fn default() -> Self {
        Self {
            n_obs: 24,
            n_vars: 40,
            // Fixed, not `SystemTime::now()`: a fixture that stamps the clock
            // cannot be used to prove anything about clock-independence.
            provenance_timestamp: 1_700_000_000,
            shard_order: [0, 1, 2],
            framed_codec: CodecId::Scx1,
            perturb_float_bit: false,
        }
    }
}

/// Write a three-shard mixed-codec fixture at `path`.
pub fn mixed_codec_file(path: &Path) -> Result<PathBuf> {
    mixed_codec_file_with(path, &FixtureOpts::default())
}

/// Write a three-shard mixed-codec fixture with explicit options.
pub fn mixed_codec_file_with(path: &Path, opts: &FixtureOpts) -> Result<PathBuf> {
    let (n_obs, n_vars) = (opts.n_obs, opts.n_vars);
    assert!(
        n_obs >= 12 && n_obs % 3 == 0,
        "n_obs must be a multiple of 3"
    );
    let third = n_obs / 3;

    let specs = [
        // Unframed, no codec — the shape ~300 existing fixtures use.
        ShardSpec {
            rows: (0, third),
            codec: CodecId::None,
            encoding: ValueEncoding::Uint8,
            framed: None,
        },
        // Row-group-framed integer (shard v2). G = 4 against 8 rows, so the
        // block index has several entries: a fixture with one row group would
        // pin the framed layout no more than an unframed one does.
        ShardSpec {
            rows: (third, 2 * third),
            codec: opts.framed_codec,
            encoding: ValueEncoding::Uint32,
            framed: Some(4),
        },
        // Float — the only path that reaches Pcodec.
        ShardSpec {
            rows: (2 * third, n_obs),
            codec: CodecId::Pcodec,
            encoding: ValueEncoding::Float32,
            framed: None,
        },
    ];

    let mut header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 10_000, 0, 0);
    // Framed shards are a v4 feature and the writer leaves the bump to the
    // caller (see `ScxWriter::set_framing`).
    header.format_version = header.format_version.max(4);
    let mut writer = ScxWriter::new(path, header)?;
    writer.write_obs(&obs(n_obs))?;
    writer.write_var(&var(n_vars))?;

    let mut order = opts.shard_order;
    order.sort_unstable();
    assert_eq!(
        order,
        [0, 1, 2],
        "shard_order must be a permutation of 0..3"
    );

    for &i in &opts.shard_order {
        let spec = specs[i];
        writer.set_framing(spec.framed.map(|g| FramingConfig {
            row_group_rows: g,
            ..Default::default()
        }));
        let (indptr, indices, values) = rows(spec, n_vars, opts.perturb_float_bit);
        writer.write_csr_shard(
            &indptr,
            &indices,
            &values,
            spec.codec,
            spec.encoding,
            spec.rows.0 as u64,
        )?;
    }
    writer.set_framing(None);

    writer.write_provenance(vec![ProvenanceEntry {
        timestamp: opts.provenance_timestamp,
        action: "scx-testkit fixture".to_string(),
        tool: "scx-testkit".to_string(),
        params_json: "{}".to_string(),
        input_checksums: Vec::new(),
    }])?;

    writer.finish()?;
    Ok(path.to_path_buf())
}

/// Deterministic 3-nnz-per-row CSR for one shard, encoded per its spec.
fn rows(spec: ShardSpec, n_vars: usize, perturb_float: bool) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let (start, end) = spec.rows;
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in start..end {
        // Ascending and distinct within the row: a CSR shard with unsorted or
        // duplicated column indices is structurally invalid, and `Scx1`
        // delta-encodes them.
        let mut cols: Vec<u32> = (0..3)
            .map(|k| ((row * 7 + k * 11) % n_vars) as u32)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        indices.extend_from_slice(&cols);

        for k in 0..cols.len() {
            // Strictly positive: a stored CSR nonzero is a count, and `Scx1`'s
            // Rice stage rejects a zero outright (`rice_encode` shifts by one).
            // A fixture with a zero in it fails to write rather than producing
            // something to digest.
            match spec.encoding {
                ValueEncoding::Uint8 => values.push(1 + ((row + k) % 250) as u8),
                ValueEncoding::Uint32 => {
                    values.extend_from_slice(&(1 + ((row + k) % 9) as u32).to_le_bytes())
                }
                ValueEncoding::Float32 => {
                    // Non-integral, so the encoder cannot re-select an integer
                    // encoding, and f32-exact so the value is not itself a
                    // source of run-to-run difference.
                    let v = (row as f32) + 0.25 * (k as f32 + 1.0);
                    let bits = if perturb_float && values.is_empty() {
                        v.to_bits() ^ 1
                    } else {
                        v.to_bits()
                    };
                    values.extend_from_slice(&f32::from_bits(bits).to_le_bytes());
                }
                other => panic!("fixture does not encode {other:?}"),
            }
        }
        indptr.push(indptr.last().unwrap() + cols.len() as u64);
    }
    (indptr, indices, values)
}

fn obs(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| ["T cell", "B cell", "NK cell"][i % 3])
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap()
}

fn var(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            ids.iter().map(String::as_str).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}
