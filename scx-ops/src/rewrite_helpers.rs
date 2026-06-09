// Shared helpers for copying auxiliary sections between SCX files.
//
// Used by build_csc and upgrade to avoid duplicating layer/obsm/uns/
// predicate-index/provenance copy logic.

use scx_codec::{CodecId, ValueEncoding};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

/// Copy all auxiliary sections (layers, obsm, uns, predicate indices) from
/// reader to writer, then append a new provenance entry.
pub fn copy_auxiliary_sections(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    copy_layers(reader, writer)?;
    copy_obsm(reader, writer)?;
    copy_uns(reader, writer)?;
    copy_predicate_indices(reader, writer)?;
    append_provenance(reader, writer, action, params_json)?;
    Ok(())
}

/// Copy all layer CSR shards from reader to writer, preserving per-shard codec.
fn copy_layers(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let layer_names = reader.layer_names();
    for layer_name in &layer_names {
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format::FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();

        let mut sorted_entries = layer_shard_entries;
        sorted_entries.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));

        for (shard_idx, entry) in sorted_entries.iter().enumerate() {
            let sh = reader.read_shard_header(entry)?;
            let ve = ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(format!("unknown value encoding: {}", sh.value_encoding))?;
            let ci =
                CodecId::from_u8(sh.codec_id).ok_or(format!("unknown codec: {}", sh.codec_id))?;

            let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;
            let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            let mut raw_values = Vec::new();
            for &v in &data {
                ve.encode_f32(&mut raw_values, v)?;
            }
            writer.write_layer_csr_shard(
                &indptr_u64,
                &indices_u32,
                &raw_values,
                ci,
                ve,
                row_start,
                layer_name,
                shard_idx as u32,
            )?;
        }
    }
    Ok(())
}

/// Copy obsm sections from reader to writer.
fn copy_obsm(reader: &ScxReader, writer: &mut ScxWriter) -> Result<(), Box<dyn std::error::Error>> {
    if reader.header().has_obsm() {
        let all_obsm = reader.read_all_obsm()?;
        for (name, batch) in &all_obsm {
            writer.write_obsm(name, batch)?;
        }
    }
    Ok(())
}

/// Copy uns section from reader to writer.
fn copy_uns(reader: &ScxReader, writer: &mut ScxWriter) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }
    Ok(())
}

/// Copy predicate index sections from reader to writer.
pub(crate) fn copy_predicate_indices(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(Some(data)) = reader.read_obs_predicate_index_bytes() {
        writer.write_obs_predicate_index(data)?;
    }
    if let Ok(Some(data)) = reader.read_var_predicate_index_bytes() {
        writer.write_var_predicate_index(data)?;
    }
    Ok(())
}

/// Read existing provenance, append a new entry, and write to writer.
pub(crate) fn append_provenance(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut prov_entries = if let Ok(prov) = reader.read_provenance() {
        prov.operations
    } else {
        Vec::new()
    };
    prov_entries.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: action.to_string(),
        tool: format!("scx-cli {}", env!("CARGO_PKG_VERSION")),
        params_json: params_json.to_string(),
        input_checksums: vec![],
    });
    writer.write_provenance(prov_entries)?;
    Ok(())
}
