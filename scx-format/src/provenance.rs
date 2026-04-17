// Provenance section read/write (docs/format.md (Provenance))

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::error::Result;

/// A single provenance operation recorded during file creation/modification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceEntry {
    /// Unix timestamp (seconds since epoch).
    pub timestamp: i64,
    /// Action name (e.g. "convert", "subset", "merge").
    pub action: String,
    /// Tool name (e.g. "scx-cli 0.1.0").
    pub tool: String,
    /// JSON-encoded parameters.
    pub params_json: String,
    /// BLAKE3 checksums of input files.
    pub input_checksums: Vec<[u8; 32]>,
}

/// Provenance section containing a version byte and a list of operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Provenance format version (always 1).
    pub version: u8,
    /// Ordered list of operations.
    pub operations: Vec<ProvenanceEntry>,
}

impl Provenance {
    /// Serialize provenance to a writer using field-by-field LE encoding.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u8(self.version)?;
        w.write_u32::<LittleEndian>(self.operations.len() as u32)?;

        for entry in &self.operations {
            w.write_i64::<LittleEndian>(entry.timestamp)?;

            let action_bytes = entry.action.as_bytes();
            w.write_u16::<LittleEndian>(action_bytes.len() as u16)?;
            w.write_all(action_bytes)?;

            let tool_bytes = entry.tool.as_bytes();
            w.write_u16::<LittleEndian>(tool_bytes.len() as u16)?;
            w.write_all(tool_bytes)?;

            let params_bytes = entry.params_json.as_bytes();
            w.write_u32::<LittleEndian>(params_bytes.len() as u32)?;
            w.write_all(params_bytes)?;

            w.write_u8(entry.input_checksums.len() as u8)?;
            for checksum in &entry.input_checksums {
                w.write_all(checksum)?;
            }
        }

        Ok(())
    }

    /// Deserialize provenance from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let version = r.read_u8()?;
        let n_operations = r.read_u32::<LittleEndian>()? as usize;

        let mut operations = Vec::with_capacity(n_operations);
        for _ in 0..n_operations {
            let timestamp = r.read_i64::<LittleEndian>()?;

            let action_len = r.read_u16::<LittleEndian>()? as usize;
            let mut action_bytes = vec![0u8; action_len];
            r.read_exact(&mut action_bytes)?;
            let action = String::from_utf8(action_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let tool_len = r.read_u16::<LittleEndian>()? as usize;
            let mut tool_bytes = vec![0u8; tool_len];
            r.read_exact(&mut tool_bytes)?;
            let tool = String::from_utf8(tool_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let params_len = r.read_u32::<LittleEndian>()? as usize;
            let mut params_bytes = vec![0u8; params_len];
            r.read_exact(&mut params_bytes)?;
            let params_json = String::from_utf8(params_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let n_checksums = r.read_u8()? as usize;
            let mut input_checksums = Vec::with_capacity(n_checksums);
            for _ in 0..n_checksums {
                let mut checksum = [0u8; 32];
                r.read_exact(&mut checksum)?;
                input_checksums.push(checksum);
            }

            operations.push(ProvenanceEntry {
                timestamp,
                action,
                tool,
                params_json,
                input_checksums,
            });
        }

        Ok(Provenance {
            version,
            operations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn provenance_round_trip() {
        let prov = Provenance {
            version: 1,
            operations: vec![
                ProvenanceEntry {
                    timestamp: 1710000000,
                    action: "convert".to_string(),
                    tool: "scx-cli 0.1.0".to_string(),
                    params_json: r#"{"input":"test.h5ad"}"#.to_string(),
                    input_checksums: vec![[0xAB; 32]],
                },
                ProvenanceEntry {
                    timestamp: 1710001000,
                    action: "subset".to_string(),
                    tool: "pyscx 0.1.0".to_string(),
                    params_json: "{}".to_string(),
                    input_checksums: vec![[0xCD; 32], [0xEF; 32]],
                },
            ],
        };

        let mut buf = Vec::new();
        prov.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = Provenance::read_from(&mut cursor).unwrap();
        assert_eq!(decoded, prov);
    }

    #[test]
    fn provenance_empty() {
        let prov = Provenance {
            version: 1,
            operations: vec![],
        };

        let mut buf = Vec::new();
        prov.write_to(&mut buf).unwrap();
        // 1 byte version + 4 bytes count = 5 bytes
        assert_eq!(buf.len(), 5);

        let mut cursor = Cursor::new(&buf);
        let decoded = Provenance::read_from(&mut cursor).unwrap();
        assert_eq!(decoded, prov);
    }

    #[test]
    fn provenance_entry_with_multiple_checksums() {
        let entry = ProvenanceEntry {
            timestamp: 1710000000,
            action: "merge".to_string(),
            tool: "scx-cli 0.1.0".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![[0x11; 32], [0x22; 32], [0x33; 32]],
        };

        let prov = Provenance {
            version: 1,
            operations: vec![entry],
        };

        let mut buf = Vec::new();
        prov.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = Provenance::read_from(&mut cursor).unwrap();
        assert_eq!(decoded.operations[0].input_checksums.len(), 3);
        assert_eq!(decoded, prov);
    }
}
