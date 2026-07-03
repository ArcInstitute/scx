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
