// Materialization plan — container + numeric dtype selection for reads.
//
// A pure descriptor: it records *what* a reader asked for (container + numeric
// dtypes). The actual fail-loud casting gate lives in `scx-codec`
// (`checked_cast_*`) and the typed dense scatter lives in
// `ScxCsr::to_dense_dtype`.

/// Output container for a materialized matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    /// scipy `csr_matrix` (the default, zero-copy path for f32/i32).
    Csr,
    /// Row-major dense 2-D array (no scipy CSR intermediate at Phase 2).
    Dense,
}

/// Numeric dtype of the materialized values (and of a dense buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueDtype {
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
}

impl ValueDtype {
    /// numpy dtype name for this value dtype.
    pub fn numpy_name(self) -> &'static str {
        match self {
            ValueDtype::F16 => "float16",
            ValueDtype::F32 => "float32",
            ValueDtype::F64 => "float64",
            ValueDtype::I8 => "int8",
            ValueDtype::I16 => "int16",
            ValueDtype::I32 => "int32",
            ValueDtype::I64 => "int64",
            ValueDtype::U8 => "uint8",
            ValueDtype::U16 => "uint16",
            ValueDtype::U32 => "uint32",
        }
    }
}

/// Numeric dtype of the CSR column indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexDtype {
    I16,
    I32,
    I64,
}

impl IndexDtype {
    /// numpy dtype name for this index dtype.
    pub fn numpy_name(self) -> &'static str {
        match self {
            IndexDtype::I16 => "int16",
            IndexDtype::I32 => "int32",
            IndexDtype::I64 => "int64",
        }
    }
}

/// A resolved materialization plan: container + dtypes + cast policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializePlan {
    pub container: Container,
    /// Value dtype (X.data / dense-value dtype).
    pub data_dtype: ValueDtype,
    /// Column-indices dtype (CSR only; ignored for `Container::Dense`).
    pub index_dtype: IndexDtype,
    /// F4 gate: `false` fails loud on any narrowing that loses data.
    pub allow_lossy: bool,
}

impl MaterializePlan {
    /// The exact current default: scipy CSR with `f32` values and `i32` indices.
    ///
    /// This is the byte-identical, zero-copy path that must remain untouched.
    pub fn default_csr_f32() -> Self {
        MaterializePlan {
            container: Container::Csr,
            data_dtype: ValueDtype::F32,
            index_dtype: IndexDtype::I32,
            allow_lossy: false,
        }
    }

    /// `true` when this plan selects the default zero-copy CSR/f32/i32 path.
    ///
    /// `allow_lossy` is irrelevant here: the default path performs no narrowing,
    /// so the gate can never trip regardless of the flag.
    pub fn is_default_csr_f32(&self) -> bool {
        self.container == Container::Csr
            && self.data_dtype == ValueDtype::F32
            && self.index_dtype == IndexDtype::I32
    }
}

impl Default for MaterializePlan {
    fn default() -> Self {
        Self::default_csr_f32()
    }
}

// --- Typed materialization buffers ------------------------------------------
//
// `ValueBuffer` / `IndexBuffer` are the runtime carriers that the in-assembly
// narrow read path fills directly at the target width, so a narrow read never
// allocates the intermediate full-matrix `f32` CSR. They mirror `ValueDtype` /
// `IndexDtype` one-to-one, so a resolved `MaterializePlan` maps to a buffer arm
// mechanically. `TypedCsr` / `TypedDense` are the assembled outputs the reader
// returns and pyscx maps to numpy (zero-copy per arm). Constructed by the
// in-assembly typed reader (`scx_format_io::read_all_csr_shards_typed` and
// siblings).

/// A value buffer at one of the numpy-representable dtypes (mirrors `ValueDtype`).
#[derive(Debug, Clone)]
pub enum ValueBuffer {
    F16(Vec<half::f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl ValueBuffer {
    /// Allocate an empty buffer of `dtype` with capacity for `nnz` values.
    pub fn with_capacity(dtype: ValueDtype, nnz: usize) -> Self {
        match dtype {
            ValueDtype::F16 => ValueBuffer::F16(Vec::with_capacity(nnz)),
            ValueDtype::F32 => ValueBuffer::F32(Vec::with_capacity(nnz)),
            ValueDtype::F64 => ValueBuffer::F64(Vec::with_capacity(nnz)),
            ValueDtype::I8 => ValueBuffer::I8(Vec::with_capacity(nnz)),
            ValueDtype::I16 => ValueBuffer::I16(Vec::with_capacity(nnz)),
            ValueDtype::I32 => ValueBuffer::I32(Vec::with_capacity(nnz)),
            ValueDtype::I64 => ValueBuffer::I64(Vec::with_capacity(nnz)),
            ValueDtype::U8 => ValueBuffer::U8(Vec::with_capacity(nnz)),
            ValueDtype::U16 => ValueBuffer::U16(Vec::with_capacity(nnz)),
            ValueDtype::U32 => ValueBuffer::U32(Vec::with_capacity(nnz)),
        }
    }

    /// Allocate a zero-filled buffer of `dtype` with exactly `n` values.
    ///
    /// The in-assembly reader fills the buffer by writing per-shard slices at
    /// their running nnz offset, so it needs a *sized* (not just capacity)
    /// buffer. Zero is the natural fill for numeric dtypes.
    pub fn zeroed(dtype: ValueDtype, n: usize) -> Self {
        match dtype {
            ValueDtype::F16 => ValueBuffer::F16(vec![half::f16::default(); n]),
            ValueDtype::F32 => ValueBuffer::F32(vec![0.0f32; n]),
            ValueDtype::F64 => ValueBuffer::F64(vec![0.0f64; n]),
            ValueDtype::I8 => ValueBuffer::I8(vec![0i8; n]),
            ValueDtype::I16 => ValueBuffer::I16(vec![0i16; n]),
            ValueDtype::I32 => ValueBuffer::I32(vec![0i32; n]),
            ValueDtype::I64 => ValueBuffer::I64(vec![0i64; n]),
            ValueDtype::U8 => ValueBuffer::U8(vec![0u8; n]),
            ValueDtype::U16 => ValueBuffer::U16(vec![0u16; n]),
            ValueDtype::U32 => ValueBuffer::U32(vec![0u32; n]),
        }
    }

    /// The `ValueDtype` this buffer carries.
    pub fn dtype(&self) -> ValueDtype {
        match self {
            ValueBuffer::F16(_) => ValueDtype::F16,
            ValueBuffer::F32(_) => ValueDtype::F32,
            ValueBuffer::F64(_) => ValueDtype::F64,
            ValueBuffer::I8(_) => ValueDtype::I8,
            ValueBuffer::I16(_) => ValueDtype::I16,
            ValueBuffer::I32(_) => ValueDtype::I32,
            ValueBuffer::I64(_) => ValueDtype::I64,
            ValueBuffer::U8(_) => ValueDtype::U8,
            ValueBuffer::U16(_) => ValueDtype::U16,
            ValueBuffer::U32(_) => ValueDtype::U32,
        }
    }

    /// Number of values held.
    pub fn len(&self) -> usize {
        match self {
            ValueBuffer::F16(v) => v.len(),
            ValueBuffer::F32(v) => v.len(),
            ValueBuffer::F64(v) => v.len(),
            ValueBuffer::I8(v) => v.len(),
            ValueBuffer::I16(v) => v.len(),
            ValueBuffer::I32(v) => v.len(),
            ValueBuffer::I64(v) => v.len(),
            ValueBuffer::U8(v) => v.len(),
            ValueBuffer::U16(v) => v.len(),
            ValueBuffer::U32(v) => v.len(),
        }
    }

    /// `true` when the buffer holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A CSR column-index buffer (mirrors `IndexDtype`).
#[derive(Debug, Clone)]
pub enum IndexBuffer {
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl IndexBuffer {
    /// Allocate an empty buffer of `dtype` with capacity for `nnz` indices.
    pub fn with_capacity(dtype: IndexDtype, nnz: usize) -> Self {
        match dtype {
            IndexDtype::I16 => IndexBuffer::I16(Vec::with_capacity(nnz)),
            IndexDtype::I32 => IndexBuffer::I32(Vec::with_capacity(nnz)),
            IndexDtype::I64 => IndexBuffer::I64(Vec::with_capacity(nnz)),
        }
    }

    /// Allocate a zero-filled buffer of `dtype` with exactly `n` indices
    /// (sized for per-shard slice fills; see [`ValueBuffer::zeroed`]).
    pub fn zeroed(dtype: IndexDtype, n: usize) -> Self {
        match dtype {
            IndexDtype::I16 => IndexBuffer::I16(vec![0i16; n]),
            IndexDtype::I32 => IndexBuffer::I32(vec![0i32; n]),
            IndexDtype::I64 => IndexBuffer::I64(vec![0i64; n]),
        }
    }

    /// The `IndexDtype` this buffer carries.
    pub fn dtype(&self) -> IndexDtype {
        match self {
            IndexBuffer::I16(_) => IndexDtype::I16,
            IndexBuffer::I32(_) => IndexDtype::I32,
            IndexBuffer::I64(_) => IndexDtype::I64,
        }
    }

    /// Number of indices held.
    pub fn len(&self) -> usize {
        match self {
            IndexBuffer::I16(v) => v.len(),
            IndexBuffer::I32(v) => v.len(),
            IndexBuffer::I64(v) => v.len(),
        }
    }

    /// `true` when the buffer holds no indices.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A CSR materialized directly at the target dtypes. `indptr` stays `i64`
/// (scipy-canonical); only `indices` and `values` narrow.
#[derive(Debug, Clone)]
pub struct TypedCsr {
    pub shape: (usize, usize),
    pub indptr: Vec<i64>,
    pub indices: IndexBuffer,
    pub values: ValueBuffer,
}

/// A row-major dense buffer materialized directly at the target value dtype.
#[derive(Debug, Clone)]
pub struct TypedDense {
    pub shape: (usize, usize),
    pub values: ValueBuffer,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `ValueDtype` arm this crate knows about (kept in sync with the enum).
    const ALL_VALUE_DTYPES: [ValueDtype; 10] = [
        ValueDtype::F16,
        ValueDtype::F32,
        ValueDtype::F64,
        ValueDtype::I8,
        ValueDtype::I16,
        ValueDtype::I32,
        ValueDtype::I64,
        ValueDtype::U8,
        ValueDtype::U16,
        ValueDtype::U32,
    ];

    const ALL_INDEX_DTYPES: [IndexDtype; 3] = [IndexDtype::I16, IndexDtype::I32, IndexDtype::I64];

    #[test]
    fn value_buffer_with_capacity_roundtrips_dtype() {
        for dt in ALL_VALUE_DTYPES {
            let buf = ValueBuffer::with_capacity(dt, 8);
            // Arm ↔ dtype mapping is total: the constructed arm reports its dtype.
            assert_eq!(buf.dtype(), dt, "value arm mismatch for {dt:?}");
            // Capacity-only allocation holds no values yet.
            assert_eq!(buf.len(), 0);
            assert!(buf.is_empty());
            // The buffer's dtype names the same numpy dtype as the descriptor.
            assert_eq!(buf.dtype().numpy_name(), dt.numpy_name());
        }
    }

    #[test]
    fn index_buffer_with_capacity_roundtrips_dtype() {
        for dt in ALL_INDEX_DTYPES {
            let buf = IndexBuffer::with_capacity(dt, 8);
            assert_eq!(buf.dtype(), dt, "index arm mismatch for {dt:?}");
            assert_eq!(buf.len(), 0);
            assert!(buf.is_empty());
            assert_eq!(buf.dtype().numpy_name(), dt.numpy_name());
        }
    }

    #[test]
    fn zeroed_is_sized_and_roundtrips_dtype() {
        for dt in ALL_VALUE_DTYPES {
            let buf = ValueBuffer::zeroed(dt, 5);
            assert_eq!(buf.dtype(), dt, "value arm mismatch for {dt:?}");
            assert_eq!(buf.len(), 5, "zeroed must be sized, not capacity-only");
            assert!(!buf.is_empty());
        }
        for dt in ALL_INDEX_DTYPES {
            let buf = IndexBuffer::zeroed(dt, 5);
            assert_eq!(buf.dtype(), dt, "index arm mismatch for {dt:?}");
            assert_eq!(buf.len(), 5);
        }
        // Spot-check the fill value is actually zero.
        match ValueBuffer::zeroed(ValueDtype::U16, 3) {
            ValueBuffer::U16(v) => assert_eq!(v, vec![0u16; 3]),
            _ => panic!("wrong arm"),
        }
    }

    #[test]
    fn typed_csr_and_dense_construct_from_buffers() {
        // Smoke: buffers compose into the assembled output containers.
        let csr = TypedCsr {
            shape: (2, 3),
            indptr: vec![0, 1, 2],
            indices: IndexBuffer::I16(vec![0, 2]),
            values: ValueBuffer::U16(vec![5, 7]),
        };
        assert_eq!(csr.shape, (2, 3));
        assert_eq!(csr.indices.len(), 2);
        assert_eq!(csr.values.len(), 2);
        assert_eq!(csr.values.dtype(), ValueDtype::U16);

        let dense = TypedDense {
            shape: (1, 4),
            values: ValueBuffer::F16(vec![half::f16::from_f32(1.0); 4]),
        };
        assert_eq!(dense.shape, (1, 4));
        assert_eq!(dense.values.len(), 4);
        assert_eq!(dense.values.dtype(), ValueDtype::F16);
    }
}
