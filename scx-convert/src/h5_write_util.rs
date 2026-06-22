//! Small shared helpers for the HDF5 writer paths (h5ad + h5mu).

use hdf5::types::VarLenUnicode;

/// Convert `&str` → HDF5 `VarLenUnicode`. HDF5's variable-length string
/// parser rejects embedded NUL bytes; rather than panicking on such input
/// we strip the NULs and retry (finding 8.2).
pub(crate) fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap_or_else(|_| {
        let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
        cleaned
            .parse::<VarLenUnicode>()
            .expect("cleaned string should have no NUL bytes")
    })
}
