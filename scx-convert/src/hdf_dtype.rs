//! Shared HDF5 numeric-dtype enum used by all typed readers in this crate.
//!
//! Used by the typed readers in `h5ad/read.rs` (whole-dataset reads of
//! categorical codes and CSR `indptr` / `indices` / `data`) and
//! `h5ad/stream.rs` (slice-read twins for the streaming CSR path), as
//! well as the dense-X reader in `h5ad/dense_stream.rs`. Centralising the
//! `TypeDescriptor → enum` mapping keeps every reader honest about
//! which HDF5 widths it accepts and surfaces unsupported dtypes via a
//! single error path.

use crate::pipeline::ConvertError;
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HdfNumericDtype {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl HdfNumericDtype {
    /// Map an HDF5 `TypeDescriptor` to one of the ten supported numeric
    /// widths. Compound, string, and other non-numeric descriptors fall
    /// through to `ConvertError::UnsupportedDtype` so the caller can
    /// attach dataset context.
    pub(crate) fn from_descriptor(desc: &TypeDescriptor) -> Result<Self, ConvertError> {
        Ok(match desc {
            TypeDescriptor::Float(FloatSize::U2) => Self::F16,
            TypeDescriptor::Float(FloatSize::U4) => Self::F32,
            TypeDescriptor::Float(FloatSize::U8) => Self::F64,
            TypeDescriptor::Integer(IntSize::U1) => Self::I8,
            TypeDescriptor::Integer(IntSize::U2) => Self::I16,
            TypeDescriptor::Integer(IntSize::U4) => Self::I32,
            TypeDescriptor::Integer(IntSize::U8) => Self::I64,
            TypeDescriptor::Unsigned(IntSize::U1) => Self::U8,
            TypeDescriptor::Unsigned(IntSize::U2) => Self::U16,
            TypeDescriptor::Unsigned(IntSize::U4) => Self::U32,
            TypeDescriptor::Unsigned(IntSize::U8) => Self::U64,
            other => return Err(ConvertError::UnsupportedDtype(format!("{other:?}"))),
        })
    }

    pub(crate) fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    // Used by the readers in `h5ad/read.rs` and `h5ad/stream.rs` to
    // attach a source-dtype tag to `ConvertError::IndexOverflow`.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ConvertError;

    #[test]
    fn from_descriptor_round_trips_every_supported_width() {
        let cases = [
            (
                TypeDescriptor::Float(FloatSize::U2),
                HdfNumericDtype::F16,
                "f16",
                2,
            ),
            (
                TypeDescriptor::Float(FloatSize::U4),
                HdfNumericDtype::F32,
                "f32",
                4,
            ),
            (
                TypeDescriptor::Float(FloatSize::U8),
                HdfNumericDtype::F64,
                "f64",
                8,
            ),
            (
                TypeDescriptor::Integer(IntSize::U1),
                HdfNumericDtype::I8,
                "i8",
                1,
            ),
            (
                TypeDescriptor::Integer(IntSize::U2),
                HdfNumericDtype::I16,
                "i16",
                2,
            ),
            (
                TypeDescriptor::Integer(IntSize::U4),
                HdfNumericDtype::I32,
                "i32",
                4,
            ),
            (
                TypeDescriptor::Integer(IntSize::U8),
                HdfNumericDtype::I64,
                "i64",
                8,
            ),
            (
                TypeDescriptor::Unsigned(IntSize::U1),
                HdfNumericDtype::U8,
                "u8",
                1,
            ),
            (
                TypeDescriptor::Unsigned(IntSize::U2),
                HdfNumericDtype::U16,
                "u16",
                2,
            ),
            (
                TypeDescriptor::Unsigned(IntSize::U4),
                HdfNumericDtype::U32,
                "u32",
                4,
            ),
            (
                TypeDescriptor::Unsigned(IntSize::U8),
                HdfNumericDtype::U64,
                "u64",
                8,
            ),
        ];
        for (desc, want, want_name, want_bytes) in cases {
            let got = HdfNumericDtype::from_descriptor(&desc).unwrap_or_else(|e| {
                panic!("from_descriptor({desc:?}) returned Err: {e}");
            });
            assert_eq!(got, want, "variant mismatch for {desc:?}");
            assert_eq!(got.name(), want_name);
            assert_eq!(got.size_bytes(), want_bytes);
        }
    }

    #[test]
    fn from_descriptor_rejects_unsupported() {
        // Boolean is a real `TypeDescriptor` variant that's not in our
        // numeric set. Compound types are harder to construct without
        // an open HDF5 file; one representative rejection is enough.
        let err = HdfNumericDtype::from_descriptor(&TypeDescriptor::Boolean).unwrap_err();
        match err {
            ConvertError::UnsupportedDtype(msg) => {
                assert!(msg.contains("Boolean"), "expected dtype name in msg: {msg}");
            }
            other => panic!("expected UnsupportedDtype, got {other:?}"),
        }
    }
}
