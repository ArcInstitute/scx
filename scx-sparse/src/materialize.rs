// Materialization plan — container + numeric dtype selection for reads.
//
// A pure descriptor: it records *what* a reader asked for (container + numeric
// dtypes). The actual fail-loud casting gate lives in `scx-codec`
// (`checked_cast_*`) and the typed dense scatter lives in
// `ScxCsr::to_dense_dtype`.

use std::ops::Range;

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
    /// Width of one value in bytes.
    ///
    /// What a materialized buffer costs per element, which is what a
    /// memory estimate for an eager read needs — the numpy name is not enough,
    /// since a caller sizing an allocation has to resolve it back to a width.
    pub fn size_bytes(self) -> u64 {
        match self {
            ValueDtype::I8 | ValueDtype::U8 => 1,
            ValueDtype::F16 | ValueDtype::I16 | ValueDtype::U16 => 2,
            ValueDtype::F32 | ValueDtype::I32 | ValueDtype::U32 => 4,
            ValueDtype::F64 | ValueDtype::I64 => 8,
        }
    }

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

    /// A mutable view of `range` at this buffer's dtype. See
    /// [`IndexBuffer::slice_mut`].
    pub fn slice_mut(&mut self, range: Range<usize>) -> ValueSliceMut<'_> {
        match self {
            ValueBuffer::F16(v) => ValueSliceMut::F16(&mut v[range]),
            ValueBuffer::F32(v) => ValueSliceMut::F32(&mut v[range]),
            ValueBuffer::F64(v) => ValueSliceMut::F64(&mut v[range]),
            ValueBuffer::I8(v) => ValueSliceMut::I8(&mut v[range]),
            ValueBuffer::I16(v) => ValueSliceMut::I16(&mut v[range]),
            ValueBuffer::I32(v) => ValueSliceMut::I32(&mut v[range]),
            ValueBuffer::I64(v) => ValueSliceMut::I64(&mut v[range]),
            ValueBuffer::U8(v) => ValueSliceMut::U8(&mut v[range]),
            ValueBuffer::U16(v) => ValueSliceMut::U16(&mut v[range]),
            ValueBuffer::U32(v) => ValueSliceMut::U32(&mut v[range]),
        }
    }

    /// Carve the buffer into consecutive, disjoint mutable chunks of `sizes`.
    /// See [`IndexBuffer::chunks_mut`], including why a partial carve returns
    /// `None` rather than the chunks it managed to take.
    pub fn chunks_mut(&mut self, sizes: &[usize]) -> Option<Vec<ValueSliceMut<'_>>> {
        match self {
            ValueBuffer::F16(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::F16)
                    .collect(),
            ),
            ValueBuffer::F32(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::F32)
                    .collect(),
            ),
            ValueBuffer::F64(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::F64)
                    .collect(),
            ),
            ValueBuffer::I8(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::I8)
                    .collect(),
            ),
            ValueBuffer::I16(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::I16)
                    .collect(),
            ),
            ValueBuffer::I32(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::I32)
                    .collect(),
            ),
            ValueBuffer::I64(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::I64)
                    .collect(),
            ),
            ValueBuffer::U8(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::U8)
                    .collect(),
            ),
            ValueBuffer::U16(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::U16)
                    .collect(),
            ),
            ValueBuffer::U32(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(ValueSliceMut::U32)
                    .collect(),
            ),
        }
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

    /// A mutable view of `range` at this buffer's dtype.
    ///
    /// # Panics
    ///
    /// When `range` is out of bounds, exactly as slice indexing does — this is
    /// the shape the range-taking cast helpers already had.
    pub fn slice_mut(&mut self, range: Range<usize>) -> IndexSliceMut<'_> {
        match self {
            IndexBuffer::I16(v) => IndexSliceMut::I16(&mut v[range]),
            IndexBuffer::I32(v) => IndexSliceMut::I32(&mut v[range]),
            IndexBuffer::I64(v) => IndexSliceMut::I64(&mut v[range]),
        }
    }

    /// Carve the buffer into consecutive, disjoint mutable chunks of `sizes`.
    ///
    /// The chunks borrow disjoint regions of one allocation, so they can be
    /// filled concurrently — which is what lets the typed whole-matrix
    /// assembler decode shards on a rayon pool instead of in a `for` loop, the
    /// same carve-up `assemble_row_major` does for the `f32` path.
    ///
    /// `None` when `sizes` does not sum to exactly [`len`](Self::len): a
    /// short sum would leave a tail no chunk owns and a long one cannot be
    /// satisfied. Both are caller bugs, and both are silent data corruption if
    /// the carve is allowed to succeed partially.
    pub fn chunks_mut(&mut self, sizes: &[usize]) -> Option<Vec<IndexSliceMut<'_>>> {
        match self {
            IndexBuffer::I16(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(IndexSliceMut::I16)
                    .collect(),
            ),
            IndexBuffer::I32(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(IndexSliceMut::I32)
                    .collect(),
            ),
            IndexBuffer::I64(v) => Some(
                split_into_chunks_mut(v, sizes)?
                    .into_iter()
                    .map(IndexSliceMut::I64)
                    .collect(),
            ),
        }
    }
}

/// A mutable slice of an [`IndexBuffer`], at one dtype (mirrors `IndexDtype`).
///
/// Produced by [`IndexBuffer::slice_mut`] / [`IndexBuffer::chunks_mut`] so a
/// consumer matches the dtype **once** and then writes through a plain
/// `&mut [T]`, rather than re-matching per element or per shard.
#[derive(Debug)]
pub enum IndexSliceMut<'a> {
    I16(&'a mut [i16]),
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
}

/// A mutable slice of a [`ValueBuffer`], at one dtype (mirrors `ValueDtype`).
///
/// See [`IndexSliceMut`].
#[derive(Debug)]
pub enum ValueSliceMut<'a> {
    F16(&'a mut [half::f16]),
    F32(&'a mut [f32]),
    F64(&'a mut [f64]),
    I8(&'a mut [i8]),
    I16(&'a mut [i16]),
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
    U8(&'a mut [u8]),
    U16(&'a mut [u16]),
    U32(&'a mut [u32]),
}

/// Split `buf` into consecutive, disjoint mutable chunks of `sizes`.
///
/// The carving primitive behind [`IndexBuffer::chunks_mut`] /
/// [`ValueBuffer::chunks_mut`], and used directly for the `indptr` array (which
/// is always `i64` and so needs no dtype enum). Returns `None` unless `sizes`
/// sums to exactly `buf.len()` — see `chunks_mut` for why a partial carve is
/// not an acceptable degradation.
pub fn split_into_chunks_mut<'a, T>(buf: &'a mut [T], sizes: &[usize]) -> Option<Vec<&'a mut [T]>> {
    let mut out = Vec::with_capacity(sizes.len());
    let mut rest = buf;
    for &n in sizes {
        let (head, tail) = rest.split_at_mut_checked(n)?;
        out.push(head);
        rest = tail;
    }
    rest.is_empty().then_some(out)
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

impl TypedCsr {
    /// Construct without validation, mirroring [`crate::ScxCsr::new_unchecked`]
    /// — same five invariants, checked in debug builds only.
    ///
    /// Worth having even though it is only ever called with data this crate just
    /// built: every `TypedCsr` in the tree was a bare struct literal before, so
    /// the typed path had **none** of the invariant checks its `f32` twin has had
    /// since it was written, on either the whole-matrix or the query assembly.
    pub fn new_unchecked(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: IndexBuffer,
        values: ValueBuffer,
    ) -> Self {
        debug_assert_eq!(
            indptr.len(),
            shape.0 + 1,
            "TypedCsr::new_unchecked: indptr.len() must equal shape.0 + 1"
        );
        debug_assert!(
            indptr.first().copied() == Some(0),
            "TypedCsr::new_unchecked: indptr[0] must be 0"
        );
        debug_assert!(
            indptr.windows(2).all(|w| w[0] <= w[1]),
            "TypedCsr::new_unchecked: indptr must be monotone non-decreasing"
        );
        debug_assert_eq!(
            indices.len(),
            values.len(),
            "TypedCsr::new_unchecked: indices.len() must equal values.len()"
        );
        debug_assert_eq!(
            values.len() as i64,
            *indptr.last().unwrap_or(&0),
            "TypedCsr::new_unchecked: values.len() must equal indptr.last()"
        );
        Self {
            shape,
            indptr,
            indices,
            values,
        }
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.shape.0
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.shape.1
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
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

    #[test]
    fn split_into_chunks_mut_tiles_the_buffer() {
        let mut buf = [0u8; 6];
        let chunks = split_into_chunks_mut(&mut buf, &[1, 0, 3, 2]).expect("sizes sum to len");
        assert_eq!(chunks.len(), 4);
        for (i, c) in chunks.into_iter().enumerate() {
            c.fill(i as u8 + 1);
        }
        // The zero-length chunk writes nothing; the rest tile the buffer in order.
        assert_eq!(buf, [1, 3, 3, 3, 4, 4]);
    }

    #[test]
    fn split_into_chunks_mut_refuses_a_partial_carve() {
        let mut buf = [0u8; 4];
        // Short: a tail no chunk owns.
        assert!(split_into_chunks_mut(&mut buf, &[1, 2]).is_none());
        // Long: cannot be satisfied at all.
        assert!(split_into_chunks_mut(&mut buf, &[3, 3]).is_none());
        // Exact still works, so the refusals above are not vacuous.
        assert!(split_into_chunks_mut(&mut buf, &[3, 1]).is_some());
    }

    #[test]
    fn buffer_chunks_mut_carve_every_arm() {
        for dtype in ALL_VALUE_DTYPES {
            let mut buf = ValueBuffer::zeroed(dtype, 5);
            let chunks = buf
                .chunks_mut(&[2, 3])
                .unwrap_or_else(|| panic!("{dtype:?} carve"));
            assert_eq!(chunks.len(), 2, "{dtype:?}");
            assert!(buf.chunks_mut(&[2, 2]).is_none(), "{dtype:?}");
        }
        for dtype in [IndexDtype::I16, IndexDtype::I32, IndexDtype::I64] {
            let mut buf = IndexBuffer::zeroed(dtype, 5);
            assert_eq!(buf.chunks_mut(&[1, 4]).unwrap().len(), 2, "{dtype:?}");
            assert!(buf.chunks_mut(&[1, 1]).is_none(), "{dtype:?}");
        }
    }

    /// `size_bytes` must agree with what `zeroed` actually allocates — the two
    /// are the only places the width of a `ValueDtype` is written down, and an
    /// estimate built on a stale one is wrong in the direction nobody checks.
    #[test]
    fn value_dtype_size_bytes_matches_what_zeroed_allocates() {
        for dtype in ALL_VALUE_DTYPES {
            let buf = ValueBuffer::zeroed(dtype, 4);
            let bytes = match &buf {
                ValueBuffer::F16(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::F32(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::F64(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::I8(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::I16(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::I32(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::I64(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::U8(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::U16(v) => std::mem::size_of_val(&v[..]),
                ValueBuffer::U32(v) => std::mem::size_of_val(&v[..]),
            };
            assert_eq!(bytes as u64, 4 * dtype.size_bytes(), "{dtype:?}");
        }
    }

    #[test]
    fn chunks_mut_writes_land_where_slice_mut_would() {
        // The two views must address the same regions: the assembler carves
        // once and the query engine slices per range, and they fill the same
        // buffers.
        let mut a = IndexBuffer::zeroed(IndexDtype::I64, 5);
        for (i, c) in a.chunks_mut(&[2, 3]).unwrap().into_iter().enumerate() {
            match c {
                IndexSliceMut::I64(v) => v.fill(i as i64 + 1),
                _ => panic!("wrong arm"),
            }
        }
        let mut b = IndexBuffer::zeroed(IndexDtype::I64, 5);
        for (i, range) in [0..2, 2..5].into_iter().enumerate() {
            match b.slice_mut(range) {
                IndexSliceMut::I64(v) => v.fill(i as i64 + 1),
                _ => panic!("wrong arm"),
            }
        }
        match (&a, &b) {
            (IndexBuffer::I64(x), IndexBuffer::I64(y)) => assert_eq!(x, y),
            _ => panic!("wrong arms"),
        }
    }
}
