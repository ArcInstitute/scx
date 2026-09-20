// ScxCsr struct + operations (docs/api.md (in-memory data model))

/// Errors from CSR construction and validation.
#[derive(Debug, thiserror::Error)]
pub enum CsrError {
    #[error("indptr length {got} != shape.0 + 1 ({expected})")]
    IndptrLength { got: usize, expected: usize },

    #[error("indptr is not monotonically non-decreasing at index {index}")]
    IndptrNotMonotonic { index: usize },

    #[error("indptr[0] must be 0, got {0}")]
    IndptrNonZeroStart(i64),

    #[error("indptr value {value} at position {position} is negative")]
    IndptrNegative { value: i64, position: usize },

    #[error("indices.len() ({indices}) != data.len() ({data})")]
    IndicesDataMismatch { indices: usize, data: usize },

    #[error("nnz mismatch: indptr[last]={indptr_nnz}, indices.len()={actual_nnz}")]
    NnzMismatch { indptr_nnz: i64, actual_nnz: usize },

    #[error("nnz range end {nnz_end} exceeds backing array length {backing_len}")]
    NnzOutOfRange { nnz_end: usize, backing_len: usize },

    #[error("index {index} out of range [0, {n_cols}) at position {position}")]
    IndexOutOfRange {
        index: i32,
        n_cols: usize,
        position: usize,
    },

    #[error("dimension overflow: {rows} * {cols} exceeds usize")]
    DimensionOverflow { rows: usize, cols: usize },

    #[error("dense array length {got} != expected {expected} (n_rows * n_cols)")]
    DenseLengthMismatch { got: usize, expected: usize },

    #[error("n_cols {0} exceeds i32::MAX, cannot represent as i32 column indices")]
    ColumnOverflow(usize),

    #[error("row_slice bounds invalid: start={start}, end={end}, n_rows={n_rows}")]
    RowSliceOutOfBounds {
        start: usize,
        end: usize,
        n_rows: usize,
    },

    #[error(
        "non-canonical CSR: an axis of {extent} cells holds {nnz} stored entries, \
         so at least one coordinate is stored more than once (docs/format.md \
         § v3 canonical CSR invariant requires strictly increasing per-row \
         column indices); run `scx validate --deep` to locate it"
    )]
    NonCanonicalAxis { extent: usize, nnz: usize },

    #[error(
        "column statistic slices disagree: sq_devs={sq_devs}, col_nnz={col_nnz}, \
         col_means={col_means} (all three must be one entry per aggregated column)"
    )]
    StatsLengthMismatch {
        sq_devs: usize,
        col_nnz: usize,
        col_means: usize,
    },
}

/// Number of implicit zeros on an axis of `extent` cells holding `nnz` stored
/// entries, or [`CsrError::NonCanonicalAxis`] if `nnz` exceeds `extent`.
///
/// # Why this can fail, and why here
///
/// In a canonical CSR it cannot: `docs/format.md` § "v3 canonical CSR
/// invariant" requires per-row column indices to be strictly increasing and to
/// lie in `[0, n_minor)`, so an axis of `extent` cells stores at most `extent`
/// entries. `nnz > extent` therefore *proves* the same coordinate is stored
/// twice.
///
/// Ordinary reads do not detect that. The decode seam bounds index **values**
/// against the shard's minor extent — see the "Who upholds invariant 5" note on
/// [`ScxCsr::new_unchecked`] — but nothing on the read path checks their
/// **ordering or uniqueness**; only `scx validate --deep` does
/// (`scx-format-io`'s `validate_canonical_csr_entry`). Neither does
/// [`ScxCsr::new`]. So a duplicate survives all the way to whichever statistic
/// first infers a cardinality from a dimension — which is this function, and
/// which is why the check belongs here rather than at the consumers.
///
/// # What this does NOT detect
///
/// **The pigeonhole argument runs one way only.** `nnz > extent` implies a
/// duplicate; a duplicate does *not* imply `nnz > extent`. A sparse row with a
/// few repeats stays well under its extent and passes here unchallenged — a
/// 1×3 row storing indices `[0, 0]` with values `[1, 2]` has `nnz = 2 ≤ 3`, so
/// `row_var` returns `2/3`, where canonicalizing the duplicate (summing to
/// `[3, 0, 0]`) gives `2.0`. That is the common shape of a duplicate in real
/// data, and it is silently wrong here. Pinned by
/// `duplicates_under_the_extent_are_not_detected`.
///
/// So this is **not** a uniqueness check and must not be described as one. It
/// catches exactly the overfull-axis case — which is the case that used to
/// wrap a `usize` subtraction and return `8.3e19`. Real uniqueness enforcement
/// needs an ordering pass at the decode seam, which `scx validate --deep`
/// does and ordinary reads deliberately do not (measured cost, see
/// `docs/conventions.md`).
///
/// This is **not** the redundant per-consumer bounds check that
/// `docs/conventions.md` warns against: that rule is about re-validating the
/// *same* axis the seam already validated. This is a different invariant
/// (cardinality, not index range), and the seam's guarantee does not imply it.
///
/// Getting it wrong is not a near-miss. Before this function existed,
/// `extent - nnz` was written inline as a `usize` subtraction: in release it
/// wrapped to ~1.8e19 and was multiplied by `mean²`, so a 1×2 row storing three
/// entries reported a variance of `8.301034833169298e19`.
pub fn implicit_zero_count(extent: usize, nnz: usize) -> Result<usize, CsrError> {
    extent
        .checked_sub(nnz)
        .ok_or(CsrError::NonCanonicalAxis { extent, nnz })
}

/// Finalize a per-column population variance from partial centered sums.
///
/// `sq_devs[c]` is `Σ (x − μ_c)²` over the **stored** entries of column `c`
/// (as produced by [`ScxCsr::col_var_partial`]); this folds in the implicit
/// zeros, each of which contributes `μ_c²`, and divides by `extent`.
///
/// `extent` is the number of rows the columns span — the full `n_obs`, or the
/// kept-row count when a deletion vector or row mask is applied. All three
/// slices must be the same length: the number of columns being aggregated over,
/// which is `n_vars` for a whole-matrix reduction and the projection width for
/// a column-projected one.
///
/// Errors with [`CsrError::NonCanonicalAxis`] when any `col_nnz[c] > extent`;
/// see [`implicit_zero_count`] for why that is corruption rather than an
/// unusual-but-valid shape, **and for the duplicates it cannot see**.
///
/// Errors with [`CsrError::StatsLengthMismatch`] if the three slices disagree.
/// That check runs on every profile, not just under `debug_assertions`: this
/// function is public, and a `zip` over mismatched slices would otherwise
/// silently truncate to the shortest — returning a short `Vec` that a caller
/// indexing by column would then read as a different column's variance.
///
/// Kept deliberately in the two-pass centered form rather than the algebraically
/// equivalent `Σx²/n − μ²`: the latter is cancellation-prone, which
/// `scx-accel`'s `accumulate_centered_ss` documents as the reason it forms
/// `(x − μ)` directly.
pub fn finalize_implicit_zero_variance(
    sq_devs: &[f64],
    col_nnz: &[usize],
    col_means: &[f64],
    extent: usize,
) -> Result<Vec<f64>, CsrError> {
    if sq_devs.len() != col_nnz.len() || sq_devs.len() != col_means.len() {
        return Err(CsrError::StatsLengthMismatch {
            sq_devs: sq_devs.len(),
            col_nnz: col_nnz.len(),
            col_means: col_means.len(),
        });
    }
    // Validate the counts *before* the zero-extent short-circuit. An axis of
    // zero cells holding stored entries is still non-canonical, and returning
    // `Ok(zeros)` for it would contradict this function's own contract.
    for &nnz in col_nnz {
        implicit_zero_count(extent, nnz)?;
    }
    if extent == 0 {
        return Ok(vec![0.0; sq_devs.len()]);
    }
    let mut variances = Vec::with_capacity(sq_devs.len());
    for ((&sq_dev, &nnz), &mean) in sq_devs.iter().zip(col_nnz).zip(col_means) {
        let n_zeros = implicit_zero_count(extent, nnz)?;
        variances.push((sq_dev + n_zeros as f64 * mean * mean) / extent as f64);
    }
    Ok(variances)
}

/// A CSR sparse matrix with scipy-compatible dtypes.
///
/// Uses `i64` indptr, `i32` indices, and `f32` data to match scipy's
/// CSR layout for zero-copy interop via numpy/PyO3.
#[derive(Debug, Clone)]
pub struct ScxCsr {
    /// (n_rows, n_cols)
    pub shape: (usize, usize),
    /// Row pointer array (length = n_rows + 1). scipy-compatible i64.
    pub indptr: Vec<i64>,
    /// Column indices (length = nnz). scipy-compatible i32.
    pub indices: Vec<i32>,
    /// Non-zero values (length = nnz). scipy-compatible f32.
    pub data: Vec<f32>,
}

impl ScxCsr {
    /// Create a new ScxCsr with full validation.
    ///
    /// Equivalent to building the struct and calling [`Self::validate`], which
    /// is the single formulation of these checks.
    pub fn new(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> Result<Self, CsrError> {
        let csr = Self {
            shape,
            indptr,
            indices,
            data,
        };
        csr.validate()?;
        Ok(csr)
    }

    /// Check [`Self::new_unchecked`]'s invariants 1-5 against an already-built
    /// matrix. Invariant 6 (`shape.0 > 0` or `indptr == [0]`) follows from 1
    /// and 2 and needs no separate test.
    ///
    /// # Who this is for
    ///
    /// **Callers that build a CSR from outside a reader.** Anything arriving
    /// through `scx-format-io` has already been validated at the decode seam
    /// (`docs/conventions.md` § "Validate shard payload at the decode seam,
    /// not at the consumer"), and re-checking it there is redundant work in a
    /// hot loop — do not add a call on a decode path. What this exists for is
    /// the other direction: an `ScxCsr` assembled from an R `dgCMatrix`, from
    /// caller-supplied numpy arrays, or by hand in a test, where nothing has
    /// checked anything and `new_unchecked`'s `debug_assert`s vanish in
    /// release.
    ///
    /// # What it does NOT check
    ///
    /// **Per-row column indices strictly increasing.** That rule is not one of
    /// `new_unchecked`'s six at all — it comes from `docs/format.md` § "v3
    /// canonical CSR invariant", and it is the one structural rule ordinary
    /// reads deliberately do not verify: an ordering pass at the decode seam
    /// measured +5.1-6.4% of per-shard decode, so it lives in
    /// `scx validate --deep`
    /// (`scx_format_io::ScxReader::validate_canonical_csr_entry`) and, on the
    /// write side, in [`crate::is_canonical_csr`], which also rejects a stored
    /// `0.0`. Call one of those if you need ordering or uniqueness; do not
    /// "complete" this function by adding the pass, which would reverse a
    /// measured decision.
    ///
    /// Note the asymmetry with [`crate::ScxCsc::new`], which *does* reject a
    /// duplicate row index within a column: a CSC sidecar is built once by
    /// `scx build-csc` rather than decoded per read, so it can afford the
    /// `HashSet` per column. Left as it is, deliberately.
    pub fn validate(&self) -> Result<(), CsrError> {
        // The step numbers below are this function's own, not
        // `new_unchecked`'s: steps 4 and 5 are the two halves of its invariant
        // 4, so step 6 here is its invariant 5.
        //
        // 1. indptr length
        let expected_len = self.shape.0 + 1;
        if self.indptr.len() != expected_len {
            return Err(CsrError::IndptrLength {
                got: self.indptr.len(),
                expected: expected_len,
            });
        }

        // 2. indptr[0] == 0 (scipy convention)
        if self.indptr[0] != 0 {
            return Err(CsrError::IndptrNonZeroStart(self.indptr[0]));
        }

        // 3. monotonically non-decreasing
        for i in 1..self.indptr.len() {
            if self.indptr[i] < self.indptr[i - 1] {
                return Err(CsrError::IndptrNotMonotonic { index: i });
            }
        }

        // 4. indices.len() == data.len()
        if self.indices.len() != self.data.len() {
            return Err(CsrError::IndicesDataMismatch {
                indices: self.indices.len(),
                data: self.data.len(),
            });
        }

        // 5. nnz match
        let indptr_nnz = *self.indptr.last().unwrap(); // safe: len >= 1
        if indptr_nnz as usize != self.indices.len() {
            return Err(CsrError::NnzMismatch {
                indptr_nnz,
                actual_nnz: self.indices.len(),
            });
        }

        // 6. all indices in [0, n_cols)
        for (pos, &idx) in self.indices.iter().enumerate() {
            if idx < 0 || idx as usize >= self.shape.1 {
                return Err(CsrError::IndexOutOfRange {
                    index: idx,
                    n_cols: self.shape.1,
                    position: pos,
                });
            }
        }

        Ok(())
    }

    /// Create a new ScxCsr without public validation.
    ///
    /// # Invariants the caller must uphold
    ///
    /// 1. `indptr.len() == shape.0 + 1`
    /// 2. `indptr[0] == 0`
    /// 3. `indptr` is monotone non-decreasing
    /// 4. `indices.len() == data.len() == *indptr.last().unwrap() as usize`
    /// 5. Every `indices[k]` is in `[0, shape.1)`
    /// 6. `shape.0 > 0` OR `indptr == vec![0]` (empty matrices must still have
    ///    a valid single-element indptr)
    ///
    /// Violating any invariant causes out-of-bounds reads in downstream
    /// operations (SpMM, column projection, iteration). These are logic bugs,
    /// not memory-safety UB. A future refactor could upgrade this constructor
    /// to an `unsafe fn` if the accessors ever switch to unchecked indexing.
    ///
    /// ⚠️ **A violation does not always announce itself.** This comment used to
    /// promise that breaking an invariant "will trigger bounds-check panics
    /// rather than silently return wrong answers". That is false for invariant
    /// 5 on the dense paths: [`Self::to_dense`] and its typed twin in
    /// `scx-format-io` write `dense[row_base + col as usize]`, where an
    /// out-of-range `col` runs off the end of one row into the next — a
    /// plausible, wrong matrix and no panic — and a negative one becomes ~2^64
    /// and wraps the add in release.
    ///
    /// In debug builds, invariants 1–4 are checked via `debug_assert!`; if
    /// that fires the caller has a bug.
    ///
    /// # Who upholds invariant 5
    ///
    /// Not this constructor, and for a long time nobody: the readers passed
    /// decoded shard data straight in, and a shard payload is **not** covered by
    /// the catalog checksum (`ScxReader::read_shard_from_entry` documents that
    /// omission). "Decoded from checksummed shards" was never the guarantee it
    /// sounds like.
    ///
    /// It is enforced now, once per shard, at the `scx-format-io` decode seam —
    /// on the scipy path by the bound handed to `scx_codec::decode_shard_scipy`,
    /// on the native path by its own pass — and the seam first requires the
    /// shard header's declared width to match the catalog's authenticated one,
    /// so a corrupt payload cannot widen its own bound to smuggle an index
    /// through. A CSR that comes out of a reader satisfies invariant 5 against
    /// the width the catalog records.
    ///
    /// One gap survives that: a **v1** catalog does not record a minor extent
    /// (`col_start`/`col_end` are v2-only and read as `0`), so on those files
    /// there is nothing to reconcile against and the header's `n_minor` stands
    /// alone. Same for a legacy multimodal shard, whose `n_minor` is `0` —
    /// "undeclared", not "zero columns" — where the bound is skipped entirely.
    ///
    /// # What is deliberately *not* on the list: uniqueness
    ///
    /// Nothing here — nor in [`Self::new`], the fully-validating constructor —
    /// requires a row's column indices to be *distinct*. Invariant 5 bounds each
    /// index's **value**; it says nothing about how many times a value may
    /// appear. `docs/format.md` § "v3 canonical CSR invariant" does require
    /// strictly increasing per-row indices, but on the read path only
    /// `scx validate --deep` checks it, so a duplicate reaches this constructor
    /// intact.
    ///
    /// That gap used to be load-bearing in the wrong direction: five statistics
    /// in this file inferred an implicit-zero count as `extent - nnz`, which a
    /// duplicate makes negative. They now go through [`implicit_zero_count`] and
    /// report [`CsrError::NonCanonicalAxis`] instead of wrapping.
    ///
    /// ⚠️ That is **narrower than a uniqueness check**, and this note used to
    /// overstate it. What those statistics reject is an axis holding *more
    /// stored entries than it has cells* — the pigeonhole case, which is
    /// exactly the one that used to wrap. A duplicate that leaves the row under
    /// its extent (the common shape: a sparse row with a couple of repeats)
    /// passes, and the statistic returns a number computed from the duplicated
    /// entries. See [`implicit_zero_count`] § "What this does NOT detect".
    ///
    /// Callers constructing a CSR from anywhere *other* than a reader still owe
    /// the invariant themselves, and one known caller does not yet discharge it:
    /// `rscx::interop::write_csc_shards_from_csr_r` builds a CSR from
    /// R-supplied index vectors and does not bound them against `n_vars`
    /// (`scx_sparse::validate_csr_arrays` is called only on rscx's *export*
    /// side). The consequence is now a detectable one rather than a silent one —
    /// such a file is rejected when read back instead of materialising a wrong
    /// dense matrix — but the write side should validate. Tracked separately;
    /// not part of the reader-side work that added this note.
    ///
    /// Use when the data is known to be valid.
    pub fn new_unchecked(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(
            indptr.len(),
            shape.0 + 1,
            "ScxCsr::new_unchecked: indptr.len() must equal shape.0 + 1"
        );
        debug_assert!(
            indptr.first().copied() == Some(0),
            "ScxCsr::new_unchecked: indptr[0] must be 0"
        );
        debug_assert!(
            indptr.windows(2).all(|w| w[0] <= w[1]),
            "ScxCsr::new_unchecked: indptr must be monotone non-decreasing"
        );
        debug_assert_eq!(
            indices.len(),
            data.len(),
            "ScxCsr::new_unchecked: indices.len() must equal data.len()"
        );
        debug_assert_eq!(
            indices.len() as i64,
            *indptr.last().unwrap_or(&0),
            "ScxCsr::new_unchecked: indices.len() must equal indptr.last()"
        );
        Self {
            shape,
            indptr,
            indices,
            data,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.shape.0
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.shape.1
    }

    /// Extract a contiguous slice of rows `[start..end)` as a new ScxCsr.
    pub fn row_slice(&self, start: usize, end: usize) -> Result<ScxCsr, CsrError> {
        if start > end || end > self.n_rows() {
            return Err(CsrError::RowSliceOutOfBounds {
                start,
                end,
                n_rows: self.n_rows(),
            });
        }

        let nnz_start = self.indptr[start] as usize;
        let nnz_end = self.indptr[end] as usize;

        // Rebase indptr to start from 0
        let base = self.indptr[start];
        let indptr: Vec<i64> = self.indptr[start..=end].iter().map(|&v| v - base).collect();
        let indices = self.indices[nnz_start..nnz_end].to_vec();
        let data = self.data[nnz_start..nnz_end].to_vec();

        Ok(ScxCsr::new_unchecked(
            (end - start, self.shape.1),
            indptr,
            indices,
            data,
        ))
    }

    /// Convert to a dense row-major matrix (`f32`).
    pub fn to_dense(&self) -> Result<Vec<f32>, CsrError> {
        self.to_dense_dtype(&self.data)
    }

    /// Scatter already-typed values into a dense row-major `(n_rows, n_cols)` buffer.
    ///
    /// `values` must be the per-nonzero data in row-major CSR order (`len == nnz`),
    /// aligned with `self.indices` — typically the result of casting `self.data` to
    /// the caller's requested dtype via `scx_codec::checked_cast_values`. This keeps
    /// the fail-loud cast gate in one place (scx-codec) while the scatter stays here.
    /// Unwritten cells are left at `T::default()` (zero for numeric types).
    pub fn to_dense_dtype<T: Copy + Default>(&self, values: &[T]) -> Result<Vec<T>, CsrError> {
        let expected_nnz = self.indices.len();
        if values.len() != expected_nnz {
            return Err(CsrError::IndicesDataMismatch {
                indices: expected_nnz,
                data: values.len(),
            });
        }
        let (n_rows, n_cols) = self.shape;
        let total = n_rows
            .checked_mul(n_cols)
            .ok_or(CsrError::DimensionOverflow {
                rows: n_rows,
                cols: n_cols,
            })?;
        // Invariant 5 (`indices[k] ∈ [0, n_cols)`) is a *caller* obligation on
        // `new_unchecked`, and this is the one method where breaking it does not
        // announce itself: `dense[row_base + col]` with an out-of-range column
        // runs off the end of one row into the next and returns a plausible,
        // wrong matrix — no panic, no error. (A negative index becomes ~2^64 and
        // wraps the add in release.) Everywhere else an out-of-range index
        // addresses a `Vec` sized by the same axis, so it panics.
        //
        // The readers bound decoded indices at the decode seam, but against the
        // *shard header's* `n_minor`, while this scatters into a buffer sized by
        // `shape.1`. Those agree on any file a writer produced — they are still
        // two different numbers, and this is a public API reachable from callers
        // that never went through a reader at all.
        //
        // One pass, against an allocation of `n_rows × n_cols` that this function
        // is about to make anyway.
        if let Some((position, &index)) = self
            .indices
            .iter()
            .enumerate()
            .find(|&(_, &c)| c < 0 || c as usize >= n_cols)
        {
            return Err(CsrError::IndexOutOfRange {
                index,
                n_cols,
                position,
            });
        }

        let mut dense = vec![T::default(); total];
        for row in 0..n_rows {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let row_base = row * n_cols;
            for (&col, &val) in self.indices[start..end].iter().zip(&values[start..end]) {
                dense[row_base + col as usize] = val;
            }
        }
        Ok(dense)
    }

    // --- Aggregation helpers (used by backed mode) ---

    /// Compute per-row sums: `data[indptr[r]..indptr[r+1]].sum()` for each row.
    ///
    /// Returns `f64` for precision when summing many `f32` values.
    pub fn row_sums(&self) -> Vec<f64> {
        let mut sums = Vec::with_capacity(self.shape.0);
        for r in 0..self.shape.0 {
            let start = self.indptr[r] as usize;
            let end = self.indptr[r + 1] as usize;
            let s: f64 = self.data[start..end].iter().map(|&v| v as f64).sum();
            sums.push(s);
        }
        sums
    }

    /// Compute per-column sums, accumulated into a `n_cols`-length vector.
    pub fn col_sums(&self) -> Vec<f64> {
        let mut sums = vec![0.0f64; self.shape.1];
        for (&col, &val) in self.indices.iter().zip(self.data.iter()) {
            sums[col as usize] += val as f64;
        }
        sums
    }

    /// Compute per-column sums and per-column sum-of-squares in a single pass.
    ///
    /// Returns `(col_sums, col_sum_sq)`, each an `n_cols`-length `f64` vector.
    /// Only stored (non-zero) entries contribute (implicit zeros add 0). Fusing
    /// both reductions avoids a second pass over the nonzeros when a caller needs
    /// the column mean *and* the (centered) total variance — the latter via
    /// `total_variance_from_col_sq(col_sum_sq, means, n_obs)`.
    pub fn col_sums_and_sum_sq(&self) -> (Vec<f64>, Vec<f64>) {
        let mut sums = vec![0.0f64; self.shape.1];
        let mut sum_sq = vec![0.0f64; self.shape.1];
        for (&col, &val) in self.indices.iter().zip(self.data.iter()) {
            let v = val as f64;
            let c = col as usize;
            sums[c] += v;
            sum_sq[c] += v * v;
        }
        (sums, sum_sq)
    }

    /// Compute per-column sums and per-column NNZ counts in a single pass.
    ///
    /// Returns `(col_sums, col_nnz)`. Identical to calling [`Self::col_sums`]
    /// and [`Self::col_nnz`] separately — the accumulation order over the
    /// nonzeros is unchanged, so the sums are bit-identical — but touches the
    /// `indices` / `data` arrays once instead of twice.
    ///
    /// Callers that need both (QC metrics' gene axis, `filter_genes` with a
    /// cell *and* a count threshold) should prefer this: at the streaming layer
    /// the second call is a second full shard decode, not just a second scan.
    pub fn col_sums_and_nnz(&self) -> (Vec<f64>, Vec<u32>) {
        let mut sums = vec![0.0f64; self.shape.1];
        let mut counts = vec![0u32; self.shape.1];
        for (&col, &val) in self.indices.iter().zip(self.data.iter()) {
            let c = col as usize;
            sums[c] += val as f64;
            counts[c] += 1;
        }
        (sums, counts)
    }

    /// Compute per-row NNZ counts: `indptr[r+1] - indptr[r]` for each row.
    pub fn row_nnz(&self) -> Vec<i64> {
        let mut counts = Vec::with_capacity(self.shape.0);
        for r in 0..self.shape.0 {
            counts.push(self.indptr[r + 1] - self.indptr[r]);
        }
        counts
    }

    /// Compute per-column NNZ counts.
    pub fn col_nnz(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.shape.1];
        for &col in &self.indices {
            counts[col as usize] += 1;
        }
        counts
    }

    /// Compute per-row sum of squared values: `sum(val² for val in row)`.
    ///
    /// Only stored (non-zero) entries contribute — implicit zeros add 0² = 0.
    /// Returns `f64` for precision when squaring and summing `f32` values.
    pub fn row_sum_of_squares(&self) -> Vec<f64> {
        let mut sums = Vec::with_capacity(self.shape.0);
        for r in 0..self.shape.0 {
            let start = self.indptr[r] as usize;
            let end = self.indptr[r + 1] as usize;
            let s: f64 = self.data[start..end]
                .iter()
                .map(|&v| {
                    let v64 = v as f64;
                    v64 * v64
                })
                .sum();
            sums.push(s);
        }
        sums
    }

    // --- Variance helpers ---

    /// Compute per-row variance (population variance, ddof=0).
    ///
    /// For each row, computes `mean = sum(data) / n_cols`, then accumulates
    /// `(val - mean)²` for stored entries and `n_zeros * mean²` for implicit zeros.
    /// Returns `f64` for precision.
    ///
    /// Errors with [`CsrError::NonCanonicalAxis`] if any row stores more entries
    /// than the matrix has columns — see [`implicit_zero_count`].
    pub fn row_var(&self) -> Result<Vec<f64>, CsrError> {
        let n_cols = self.shape.1;
        let mut variances = Vec::with_capacity(self.shape.0);
        for r in 0..self.shape.0 {
            let start = self.indptr[r] as usize;
            let end = self.indptr[r + 1] as usize;
            let nnz = end - start;

            // Validate before the zero-extent short-circuit, matching
            // `finalize_implicit_zero_variance`: a (1, 0) shape holding a
            // stored entry is non-canonical, and returning 0.0 for it would
            // contradict the contract this method advertises.
            let n_zeros = implicit_zero_count(n_cols, nnz)?;
            if n_cols == 0 {
                variances.push(0.0);
                continue;
            }

            // Compute mean
            let sum: f64 = self.data[start..end].iter().map(|&v| v as f64).sum();
            let mean = sum / n_cols as f64;

            // Accumulate (val - mean)² for stored entries
            let mut var_sum: f64 = self.data[start..end]
                .iter()
                .map(|&v| {
                    let diff = v as f64 - mean;
                    diff * diff
                })
                .sum();

            // Add contribution from implicit zeros: n_zeros * mean²
            var_sum += n_zeros as f64 * mean * mean;

            variances.push(var_sum / n_cols as f64);
        }
        Ok(variances)
    }

    /// Compute per-column sum-of-squared-deviations from given means.
    ///
    /// For each column, accumulates `(val - mean[col])²` for stored entries.
    /// The caller must also account for implicit zeros: each zero contributes
    /// `mean[col]²` (this is done at the `BackedCsrReader` level by tracking
    /// per-column NNZ and total row count across shards).
    ///
    /// `col_means` must have length == `n_cols`.
    pub fn col_var_partial(&self, col_means: &[f64]) -> Vec<f64> {
        let mut sq_devs = vec![0.0f64; self.shape.1];
        for (&col, &val) in self.indices.iter().zip(self.data.iter()) {
            let c = col as usize;
            let diff = val as f64 - col_means[c];
            sq_devs[c] += diff * diff;
        }
        sq_devs
    }

    // --- Max/Min helpers ---

    /// Compute per-row max, accounting for implicit zeros.
    ///
    /// When a row has fewer stored entries than `n_cols`, the max is
    /// `max(stored_max, 0.0)`. For rows with no stored entries and `n_cols > 0`,
    /// returns `0.0` (all entries are implicit zeros).
    ///
    /// Errors with [`CsrError::NonCanonicalAxis`] if any row stores more entries
    /// than the matrix has columns. Note that this one never *wrapped* — the
    /// `nnz < n_cols` gate below simply read false and skipped the implicit-zero
    /// correction, so a row storing `-5.0` three times in two columns reported a
    /// max of `-5.0` when one of its two cells is an implicit `0.0`. Silently
    /// wrong in debug and release alike; see [`implicit_zero_count`].
    pub fn row_max(&self) -> Result<Vec<f64>, CsrError> {
        let n_cols = self.shape.1;
        let mut maxes = Vec::with_capacity(self.shape.0);
        for r in 0..self.shape.0 {
            let start = self.indptr[r] as usize;
            let end = self.indptr[r + 1] as usize;
            let nnz = end - start;

            let n_zeros = implicit_zero_count(n_cols, nnz)?;
            if n_cols == 0 {
                maxes.push(f64::NEG_INFINITY);
                continue;
            }

            if nnz == 0 {
                // All entries are implicit zeros
                maxes.push(0.0);
                continue;
            }

            let stored_max = self.data[start..end]
                .iter()
                .map(|&v| v as f64)
                .fold(f64::NEG_INFINITY, f64::max);

            if n_zeros > 0 {
                // Has implicit zeros
                maxes.push(stored_max.max(0.0));
            } else {
                maxes.push(stored_max);
            }
        }
        Ok(maxes)
    }

    /// Compute per-column max, accounting for implicit zeros.
    ///
    /// `total_n_obs` is the total number of rows **across the full dataset**
    /// (not just this shard). The parameter used to be called `n_obs`, which
    /// was ambiguous at call sites where the CSR object holds a single shard:
    /// callers must pass the *global* row count so the implicit-zero
    /// correction compares stored-nnz against the right denominator.
    /// When the column has fewer stored entries than `total_n_obs`, returns
    /// `max(stored_max, 0.0)`.
    ///
    /// Errors with [`CsrError::NonCanonicalAxis`]; see [`Self::row_max`].
    pub fn col_max(&self, total_n_obs: usize) -> Result<Vec<f64>, CsrError> {
        let mut maxes = vec![f64::NEG_INFINITY; self.shape.1];
        let mut col_counts = vec![0usize; self.shape.1];

        for (&col, &val) in self.indices.iter().zip(self.data.iter()) {
            let c = col as usize;
            let v = val as f64;
            maxes[c] = maxes[c].max(v);
            col_counts[c] += 1;
        }

        for c in 0..self.shape.1 {
            if implicit_zero_count(total_n_obs, col_counts[c])? > 0 {
                // Has implicit zeros — max is at least 0.0
                if maxes[c] == f64::NEG_INFINITY {
                    maxes[c] = 0.0; // all entries in this shard contribute nothing
                } else {
                    maxes[c] = maxes[c].max(0.0);
                }
            }
            // If col_counts[c] == total_n_obs, all entries are stored; keep stored_max
            // If col_counts[c] == 0 and total_n_obs == 0, keep NEG_INFINITY (degenerate)
        }
        Ok(maxes)
    }

    /// Compute per-row min, accounting for implicit zeros.
    ///
    /// When a row has fewer stored entries than `n_cols`, the min is
    /// `min(stored_min, 0.0)`.
    ///
    /// Errors with [`CsrError::NonCanonicalAxis`]; see [`Self::row_max`].
    pub fn row_min(&self) -> Result<Vec<f64>, CsrError> {
        let n_cols = self.shape.1;
        let mut mins = Vec::with_capacity(self.shape.0);
        for r in 0..self.shape.0 {
            let start = self.indptr[r] as usize;
            let end = self.indptr[r + 1] as usize;
            let nnz = end - start;

            let n_zeros = implicit_zero_count(n_cols, nnz)?;
            if n_cols == 0 {
                mins.push(f64::INFINITY);
                continue;
            }

            if nnz == 0 {
                // All entries are implicit zeros
                mins.push(0.0);
                continue;
            }

            let stored_min = self.data[start..end]
                .iter()
                .map(|&v| v as f64)
                .fold(f64::INFINITY, f64::min);

            if n_zeros > 0 {
                // Has implicit zeros
                mins.push(stored_min.min(0.0));
            } else {
                mins.push(stored_min);
            }
        }
        Ok(mins)
    }

    /// Compute per-column min, accounting for implicit zeros.
    ///
    /// `total_n_obs` is the total number of rows **across the full dataset**
    /// (not just this shard); see [`Self::col_max`] for the rationale on
    /// the renaming. When the column has fewer stored entries than
    /// `total_n_obs`, returns `min(stored_min, 0.0)`.
    ///
    /// Errors with [`CsrError::NonCanonicalAxis`]; see [`Self::row_max`].
    pub fn col_min(&self, total_n_obs: usize) -> Result<Vec<f64>, CsrError> {
        let mut mins = vec![f64::INFINITY; self.shape.1];
        let mut col_counts = vec![0usize; self.shape.1];

        for (&col, &val) in self.indices.iter().zip(self.data.iter()) {
            let c = col as usize;
            let v = val as f64;
            mins[c] = mins[c].min(v);
            col_counts[c] += 1;
        }

        for c in 0..self.shape.1 {
            if implicit_zero_count(total_n_obs, col_counts[c])? > 0 {
                if mins[c] == f64::INFINITY {
                    mins[c] = 0.0;
                } else {
                    mins[c] = mins[c].min(0.0);
                }
            }
        }
        Ok(mins)
    }
}

/// Total variance from pre-computed column sum-of-squares.
///
/// Uses the identity: `Var(X_j) = (Σ x²_j − n·μ_j²) / (n−1)`.
/// Sums variance contributions across all columns to get total variance.
///
/// - `col_sum_sq`: per-column sum-of-squares (e.g. from [`ScxCsr::col_sums_and_sum_sq`])
/// - `means`: column means (if centering was applied)
/// - `n_obs`: number of observations
pub fn total_variance_from_col_sq(col_sum_sq: &[f64], means: Option<&[f64]>, n_obs: usize) -> f64 {
    let total = if let Some(mu) = means {
        col_sum_sq
            .iter()
            .zip(mu.iter())
            .map(|(&sq, &m)| sq - n_obs as f64 * m * m)
            .sum::<f64>()
    } else {
        col_sum_sq.iter().sum::<f64>()
    };
    total / (n_obs as f64 - 1.0).max(1.0)
}

/// Merge multiple `ScxCsr` values into one, rebasing indptr.
///
/// All CSRs must have the same number of columns (`n_vars`).
pub fn concatenate_csr(csrs: &[ScxCsr], n_vars: usize) -> Result<ScxCsr, CsrError> {
    if csrs.is_empty() {
        return Ok(ScxCsr::new_unchecked((0, n_vars), vec![0], vec![], vec![]));
    }

    if csrs.len() == 1 {
        return Ok(csrs[0].clone());
    }

    // Pre-compute total sizes for allocation
    let total_rows: usize = csrs.iter().map(|c| c.n_rows()).sum();
    let total_nnz: usize = csrs.iter().map(|c| c.nnz()).sum();

    let mut merged_indptr = Vec::with_capacity(total_rows + 1);
    let mut merged_indices = Vec::with_capacity(total_nnz);
    let mut merged_data = Vec::with_capacity(total_nnz);
    let mut cumulative_nnz: i64 = 0;

    for (i, csr) in csrs.iter().enumerate() {
        if i == 0 {
            merged_indptr.extend_from_slice(&csr.indptr);
        } else {
            // Skip first element (0) and offset by cumulative nnz
            for &v in &csr.indptr[1..] {
                merged_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += *csr.indptr.last().unwrap_or(&0);
        merged_indices.extend_from_slice(&csr.indices);
        merged_data.extend_from_slice(&csr.data);
    }

    Ok(ScxCsr::new_unchecked(
        (total_rows, n_vars),
        merged_indptr,
        merged_indices,
        merged_data,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_csr() -> ScxCsr {
        // 3x5 matrix:
        // row 0: [0, 5, 0, 10, 0]
        // row 1: [1, 0, 3, 0, 7]
        // row 2: [0, 0, 2, 0, 0]
        ScxCsr::new(
            (3, 5),
            vec![0, 2, 5, 6],
            vec![1, 3, 0, 2, 4, 2],
            vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn csr_new_and_accessors() {
        let csr = sample_csr();
        assert_eq!(csr.n_rows(), 3);
        assert_eq!(csr.n_cols(), 5);
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr.shape, (3, 5));
    }

    #[test]
    fn csr_empty() {
        let csr = ScxCsr::new((0, 0), vec![0], vec![], vec![]).unwrap();
        assert_eq!(csr.n_rows(), 0);
        assert_eq!(csr.n_cols(), 0);
        assert_eq!(csr.nnz(), 0);
    }

    /// `to_dense_dtype` is the scatter where breaking invariant 5 is silent,
    /// and `to_dense` delegates to it — so this guard covers every densify call
    /// site, including `pyscx`'s query / `obs_filter` path
    /// (`convert/interop.rs::csr_to_scipy_typed`), which the typed twin in
    /// `scx-format-io` does not reach.
    ///
    /// Pins the corruption arithmetic, not just the error: with the guard
    /// removed, `dense[row_base + col]` for row 1 / column 12 of a 3x10 matrix
    /// writes flat index 22 — row 2, column 2 — and returns a well-formed wrong
    /// matrix. An `is_err()` assertion would pass against a build that panicked
    /// instead and would not characterise the defect.
    #[test]
    fn to_dense_dtype_rejects_an_out_of_range_column() {
        let csr = ScxCsr::new_unchecked(
            (3, 10),
            vec![0, 1, 3, 4],
            vec![0, 12, 3, 9],
            vec![11.0, 22.0, 33.0, 44.0],
        );
        match csr.to_dense() {
            Err(CsrError::IndexOutOfRange {
                index,
                n_cols,
                position,
            }) => assert_eq!((index, n_cols, position), (12, 10, 1)),
            other => panic!("expected IndexOutOfRange, got {other:?}"),
        }

        // Where the unguarded write would have landed: `row_base + col` for
        // row 1 of a 10-column matrix is 10 + 12 = 22, i.e. row 2, column 2.
        let (row, n_cols, col) = (1usize, 10usize, 12usize);
        assert_eq!(row * n_cols + col, 22);
        assert_eq!((row * n_cols + col) / n_cols, 2, "lands in row 2");

        // A negative column is the other half: `col as usize` makes it ~2^64 and
        // the add wraps in release rather than panicking.
        let neg = ScxCsr::new_unchecked((2, 4), vec![0, 1, 2], vec![0, -1], vec![1.0, 2.0]);
        assert!(matches!(
            neg.to_dense(),
            Err(CsrError::IndexOutOfRange { index: -1, .. })
        ));
    }

    /// Control: the guard must not reject a full-width matrix. Without this an
    /// off-by-one (`>` for `>=`) would pass the test above.
    #[test]
    fn to_dense_dtype_accepts_the_largest_legal_column() {
        let csr = ScxCsr::new_unchecked((1, 4), vec![0, 2], vec![0, 3], vec![7.0, 9.0]);
        assert_eq!(csr.to_dense().unwrap(), vec![7.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn to_dense_dtype_matches_to_dense() {
        let csr = sample_csr();
        // Cast the f32 data to u16 (all values fit) and scatter.
        let data_u16: Vec<u16> = csr.data.iter().map(|&v| v as u16).collect();
        let dense_u16 = csr.to_dense_dtype(&data_u16).unwrap();
        let dense_f32 = csr.to_dense().unwrap();
        assert_eq!(dense_u16.len(), dense_f32.len());
        for (d16, d32) in dense_u16.iter().zip(dense_f32.iter()) {
            assert_eq!(*d16 as f32, *d32);
        }
        // Spot-check row-major layout: row 0 col 1 = 5, col 3 = 10.
        assert_eq!(dense_u16[1], 5);
        assert_eq!(dense_u16[3], 10);
    }

    #[test]
    fn to_dense_dtype_rejects_length_mismatch() {
        let csr = sample_csr();
        let bad: Vec<u16> = vec![1, 2, 3]; // nnz is 6
        assert!(csr.to_dense_dtype(&bad).is_err());
    }

    // 12.7: row_slice
    #[test]
    fn row_slice_middle_rows() {
        let csr = sample_csr();
        let sliced = csr.row_slice(1, 3).unwrap();
        assert_eq!(sliced.shape, (2, 5));
        assert_eq!(sliced.indptr, vec![0, 3, 4]);
        assert_eq!(sliced.indices, vec![0, 2, 4, 2]);
        assert_eq!(sliced.data, vec![1.0, 3.0, 7.0, 2.0]);
    }

    #[test]
    fn row_slice_single_row() {
        let csr = sample_csr();
        let sliced = csr.row_slice(0, 1).unwrap();
        assert_eq!(sliced.shape, (1, 5));
        assert_eq!(sliced.indptr, vec![0, 2]);
        assert_eq!(sliced.indices, vec![1, 3]);
        assert_eq!(sliced.data, vec![5.0, 10.0]);
    }

    #[test]
    fn row_slice_empty() {
        let csr = sample_csr();
        let sliced = csr.row_slice(1, 1).unwrap();
        assert_eq!(sliced.shape, (0, 5));
        assert_eq!(sliced.indptr, vec![0]);
        assert_eq!(sliced.nnz(), 0);
    }

    // 12.8: to_dense
    #[test]
    fn to_dense_known() {
        let csr = sample_csr();
        let dense = csr.to_dense().unwrap();
        #[rustfmt::skip]
        let expected = vec![
            0.0, 5.0, 0.0, 10.0, 0.0,
            1.0, 0.0, 3.0,  0.0, 7.0,
            0.0, 0.0, 2.0,  0.0, 0.0,
        ];
        assert_eq!(dense, expected);
    }

    // 12.9: empty matrix
    #[test]
    fn empty_matrix_with_shape() {
        let csr = ScxCsr::new((5, 10), vec![0, 0, 0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(csr.n_rows(), 5);
        assert_eq!(csr.n_cols(), 10);
        assert_eq!(csr.nnz(), 0);
        let dense = csr.to_dense().unwrap();
        assert_eq!(dense.len(), 50);
        assert!(dense.iter().all(|&v| v == 0.0));
    }

    // 12.10: single-row and single-column
    #[test]
    fn single_row_matrix() {
        let csr = ScxCsr::new((1, 4), vec![0, 2], vec![1, 3], vec![5.0, 9.0]).unwrap();
        assert_eq!(csr.to_dense().unwrap(), vec![0.0, 5.0, 0.0, 9.0]);
    }

    #[test]
    fn single_column_matrix() {
        let csr = ScxCsr::new((3, 1), vec![0, 1, 1, 1], vec![0], vec![7.0]).unwrap();
        assert_eq!(csr.to_dense().unwrap(), vec![7.0, 0.0, 0.0]);
    }

    // 12.11: mismatched lengths
    #[test]
    fn error_indptr_length() {
        let err = ScxCsr::new((3, 5), vec![0, 2, 5], vec![], vec![]).unwrap_err();
        assert!(matches!(
            err,
            CsrError::IndptrLength {
                got: 3,
                expected: 4
            }
        ));
    }

    #[test]
    fn error_indices_data_mismatch() {
        let err = ScxCsr::new(
            (1, 5),
            vec![0, 2],
            vec![1, 3],
            vec![5.0], // only 1, but indices has 2
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CsrError::IndicesDataMismatch {
                indices: 2,
                data: 1
            }
        ));
    }

    #[test]
    fn error_nnz_mismatch() {
        let err = ScxCsr::new(
            (1, 5),
            vec![0, 3], // claims 3 nnz
            vec![1, 2], // but only 2
            vec![1.0, 2.0],
        )
        .unwrap_err();
        assert!(matches!(err, CsrError::NnzMismatch { .. }));
    }

    // 12.12: non-monotonic indptr
    #[test]
    fn error_indptr_not_monotonic() {
        let err = ScxCsr::new(
            (2, 5),
            vec![0, 3, 1], // decreases at index 2
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap_err();
        assert!(matches!(err, CsrError::IndptrNotMonotonic { index: 2 }));
    }

    #[test]
    fn error_indptr_negative_start() {
        let err = ScxCsr::new((1, 5), vec![-1, 0], vec![], vec![]).unwrap_err();
        assert!(matches!(err, CsrError::IndptrNonZeroStart(-1)));
    }

    #[test]
    fn error_indptr_nonzero_start() {
        let err = ScxCsr::new((1, 5), vec![5, 7], vec![1, 3], vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(err, CsrError::IndptrNonZeroStart(5)));
    }

    // 12.13: out-of-range index
    #[test]
    fn error_index_out_of_range() {
        let err = ScxCsr::new(
            (1, 5),
            vec![0, 2],
            vec![1, 5], // 5 >= n_cols(5)
            vec![1.0, 2.0],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CsrError::IndexOutOfRange {
                index: 5,
                n_cols: 5,
                position: 1
            }
        ));
    }

    #[test]
    fn error_negative_index() {
        let err = ScxCsr::new((1, 5), vec![0, 1], vec![-1], vec![1.0]).unwrap_err();
        assert!(matches!(err, CsrError::IndexOutOfRange { index: -1, .. }));
    }

    #[test]
    fn error_row_slice_start_greater_than_end() {
        let csr = sample_csr();
        let err = csr.row_slice(2, 1).unwrap_err();
        assert!(matches!(
            err,
            CsrError::RowSliceOutOfBounds {
                start: 2,
                end: 1,
                n_rows: 3
            }
        ));
    }

    #[test]
    fn error_row_slice_end_exceeds_n_rows() {
        let csr = sample_csr();
        let err = csr.row_slice(0, 4).unwrap_err();
        assert!(matches!(
            err,
            CsrError::RowSliceOutOfBounds {
                start: 0,
                end: 4,
                n_rows: 3
            }
        ));
    }

    #[test]
    fn test_to_dense_dimension_overflow() {
        // Create a CSR with dimensions that overflow usize when multiplied.
        // Build via struct literal to bypass the debug_assert invariants in
        // `new_unchecked` — the point of this test is to exercise the
        // dimension-overflow path in `to_dense`, not the constructor.
        let huge = usize::MAX / 2 + 1;
        let csr = ScxCsr {
            shape: (huge, 2),
            indptr: vec![],
            indices: vec![],
            data: vec![],
        };
        let err = csr.to_dense().unwrap_err();
        assert!(matches!(err, CsrError::DimensionOverflow { .. }));
    }

    // --- Aggregation tests: row_sums / col_sums / row_nnz / col_nnz ---

    #[test]
    fn test_row_sums() {
        let csr = sample_csr();
        let sums = csr.row_sums();
        assert_eq!(sums, vec![15.0, 11.0, 2.0]);
    }

    #[test]
    fn test_col_sums() {
        let csr = sample_csr();
        let sums = csr.col_sums();
        // col 0: 1, col 1: 5, col 2: 3+2=5, col 3: 10, col 4: 7
        assert_eq!(sums, vec![1.0, 5.0, 5.0, 10.0, 7.0]);
    }

    #[test]
    fn test_col_sums_and_sum_sq() {
        let csr = sample_csr();
        let (sums, sum_sq) = csr.col_sums_and_sum_sq();
        // sums match the standalone col_sums().
        assert_eq!(sums, csr.col_sums());
        // col 0: 1², col 1: 5², col 2: 3²+2², col 3: 10², col 4: 7²
        assert_eq!(sum_sq, vec![1.0, 25.0, 13.0, 100.0, 49.0]);
    }

    #[test]
    fn test_col_sums_and_nnz() {
        let csr = sample_csr();
        let (sums, counts) = csr.col_sums_and_nnz();
        // Bit-identical to the two standalone kernels it replaces.
        assert_eq!(sums, csr.col_sums());
        assert_eq!(counts, csr.col_nnz());
    }

    #[test]
    fn test_row_nnz() {
        let csr = sample_csr();
        let nnz = csr.row_nnz();
        assert_eq!(nnz, vec![2, 3, 1]);
    }

    #[test]
    fn test_col_nnz() {
        let csr = sample_csr();
        let nnz = csr.col_nnz();
        assert_eq!(nnz, vec![1, 1, 2, 1, 1]);
    }
    // --- Sum of squares tests ---

    #[test]
    fn test_row_sum_of_squares() {
        let csr = sample_csr();
        // row 0: [0, 5, 0, 10, 0] → 5² + 10² = 25 + 100 = 125
        // row 1: [1, 0, 3, 0, 7]  → 1² + 3² + 7² = 1 + 9 + 49 = 59
        // row 2: [0, 0, 2, 0, 0]  → 2² = 4
        let sq = csr.row_sum_of_squares();
        assert_eq!(sq, vec![125.0, 59.0, 4.0]);
    }

    // --- Variance tests ---

    #[test]
    fn test_row_var() {
        let csr = sample_csr();
        // row 0: [0, 5, 0, 10, 0] → mean=3.0, var = (9+4+9+49+9)/5 = 16.0
        // row 1: [1, 0, 3, 0, 7]  → mean=2.2, var = (1.44+4.84+0.64+4.84+23.04)/5 = 6.96
        // row 2: [0, 0, 2, 0, 0]  → mean=0.4, var = (0.16+0.16+2.56+0.16+0.16)/5 = 0.64
        let var = csr.row_var().unwrap();
        assert!((var[0] - 16.0).abs() < 1e-10);
        assert!((var[1] - 6.96).abs() < 1e-10);
        assert!((var[2] - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_col_var_partial() {
        let csr = sample_csr();
        // n_obs = 3
        // col means: [1/3, 5/3, 5/3, 10/3, 7/3]
        let col_sums = csr.col_sums();
        let col_means: Vec<f64> = col_sums.iter().map(|&s| s / 3.0).collect();
        let partial = csr.col_var_partial(&col_means);

        // For col 0: stored entries: [1] at rows [1], implicit zeros at rows [0,2]
        // partial only sums (val - mean)² for stored entries:
        // (1 - 1/3)² = (2/3)² = 4/9 ≈ 0.4444
        assert!((partial[0] - 4.0 / 9.0).abs() < 1e-10);
    }

    // --- Max tests ---

    #[test]
    fn test_row_max() {
        let csr = sample_csr();
        let maxes = csr.row_max().unwrap();
        // row 0: [0, 5, 0, 10, 0] → max = 10 (has zeros, max(10, 0) = 10)
        // row 1: [1, 0, 3, 0, 7]  → max = 7  (has zeros, max(7, 0) = 7)
        // row 2: [0, 0, 2, 0, 0]  → max = 2  (has zeros, max(2, 0) = 2)
        assert_eq!(maxes, vec![10.0, 7.0, 2.0]);
    }

    #[test]
    fn test_row_max_all_zeros() {
        let csr = ScxCsr::new((1, 3), vec![0, 0], vec![], vec![]).unwrap();
        let maxes = csr.row_max().unwrap();
        assert_eq!(maxes, vec![0.0]);
    }

    #[test]
    fn test_col_max() {
        let csr = sample_csr();
        let maxes = csr.col_max(3).unwrap();
        // col 0: [0,1,0] → stored: [1], has zeros → max(1,0)=1
        // col 1: [5,0,0] → stored: [5], has zeros → max(5,0)=5
        // col 2: [0,3,2] → stored: [3,2], has zeros → max(3,0)=3
        // col 3: [10,0,0] → stored: [10], has zeros → max(10,0)=10
        // col 4: [0,7,0] → stored: [7], has zeros → max(7,0)=7
        assert_eq!(maxes, vec![1.0, 5.0, 3.0, 10.0, 7.0]);
    }

    // --- Min tests ---

    #[test]
    fn test_row_min() {
        let csr = sample_csr();
        let mins = csr.row_min().unwrap();
        // All rows have implicit zeros, so min = min(stored_min, 0.0) = 0.0
        assert_eq!(mins, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_row_min_all_zeros() {
        let csr = ScxCsr::new((1, 3), vec![0, 0], vec![], vec![]).unwrap();
        let mins = csr.row_min().unwrap();
        assert_eq!(mins, vec![0.0]);
    }

    #[test]
    fn test_col_min() {
        let csr = sample_csr();
        let mins = csr.col_min(3).unwrap();
        // All columns have at least one implicit zero, so min = min(stored_min, 0.0) = 0.0
        assert_eq!(mins, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_col_min_negative_values() {
        // 2x3 matrix: row0=[−5, 0, 3], row1=[0, −2, 0]
        let csr = ScxCsr::new((2, 3), vec![0, 2, 3], vec![0, 2, 1], vec![-5.0, 3.0, -2.0]).unwrap();
        let mins = csr.col_min(2).unwrap();
        // col 0: [-5, 0] → min(-5, 0) = -5 (has implicit zero)
        // col 1: [0, -2] → min(-2, 0) = -2 (has implicit zero)
        // col 2: [3, 0]  → min(3, 0) = 0   (has implicit zero)
        assert_eq!(mins, vec![-5.0, -2.0, 0.0]);
    }

    #[test]
    fn test_col_max_negative_only() {
        // 2x2 dense matrix (no implicit zeros): [[-1, -3], [-2, -4]]
        let csr = ScxCsr::new(
            (2, 2),
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![-1.0, -3.0, -2.0, -4.0],
        )
        .unwrap();
        let maxes = csr.col_max(2).unwrap();
        // col 0: [-1, -2] → no implicit zeros → max = -1
        // col 1: [-3, -4] → no implicit zeros → max = -3
        assert_eq!(maxes, vec![-1.0, -3.0]);
    }

    // --- Non-canonical input: nnz > extent ---
    //
    // Every statistic in this file that folds in implicit zeros has to know how
    // many there are, and derives it from `extent - nnz`. A duplicate coordinate
    // inverts those operands. `ScxCsr::new` accepts such a matrix — it validates
    // index *range*, not uniqueness — so these fixtures are constructible
    // through the fully-validating constructor, which is the point.

    /// 1x2 holding three entries summing to 6.0, so `mean == 3.0`.
    /// Column 1 is stored twice: `nnz == 3` against `n_cols == 2`.
    fn non_canonical_row() -> ScxCsr {
        ScxCsr::new((1, 2), vec![0, 3], vec![0, 1, 1], vec![1.0, 2.0, 3.0])
            .expect("new() validates index range, not uniqueness")
    }

    #[test]
    fn implicit_zero_count_rejects_more_entries_than_cells() {
        assert_eq!(implicit_zero_count(5, 2).unwrap(), 3);
        assert_eq!(implicit_zero_count(4, 4).unwrap(), 0);
        assert_eq!(implicit_zero_count(0, 0).unwrap(), 0);
        assert!(matches!(
            implicit_zero_count(2, 3),
            Err(CsrError::NonCanonicalAxis { extent: 2, nnz: 3 })
        ));
    }

    #[test]
    fn row_var_reports_non_canonical_instead_of_a_wrapped_count() {
        // Before the fix this did not fail — it *answered*. `n_cols - nnz` is a
        // usize subtraction, so 2 - 3 wrapped to 18446744073709551615, which
        // times mean² (9.0) over n_cols (2) gave 8.301034833169298e19 in
        // release. Debug builds panicked with "attempt to subtract with
        // overflow" instead, so the same defect had two different symptoms and
        // neither was an error.
        assert!(matches!(
            non_canonical_row().row_var(),
            Err(CsrError::NonCanonicalAxis { extent: 2, nnz: 3 })
        ));
    }

    #[test]
    fn extrema_report_non_canonical_instead_of_skipping_the_zero_correction() {
        // The extrema never wrapped: they gate on `nnz < n_cols`, which simply
        // reads false when nnz is larger, silently skipping the implicit-zero
        // correction. This fixture stores -5.0 three times in column 0 of a
        // 1x2 matrix, so column 1 is an implicit zero and the true row max is
        // 0.0 — but row_max() returned -5.0, in *both* profiles. That makes
        // this the case that proves the extrema fix independently of
        // overflow-checks.
        let csr = ScxCsr::new((1, 2), vec![0, 3], vec![0, 0, 0], vec![-5.0, -5.0, -5.0]).unwrap();
        assert!(matches!(
            csr.row_max(),
            Err(CsrError::NonCanonicalAxis { extent: 2, nnz: 3 })
        ));
        assert!(matches!(
            csr.row_min(),
            Err(CsrError::NonCanonicalAxis { extent: 2, nnz: 3 })
        ));
        // Column axis: column 0 holds 3 entries against total_n_obs == 1.
        assert!(matches!(
            csr.col_max(1),
            Err(CsrError::NonCanonicalAxis { extent: 1, nnz: 3 })
        ));
        assert!(matches!(
            csr.col_min(1),
            Err(CsrError::NonCanonicalAxis { extent: 1, nnz: 3 })
        ));
    }

    #[test]
    fn finalize_implicit_zero_variance_matches_the_inline_form_it_replaced() {
        // The shape the five call sites in scx-format-io and pyscx used to
        // spell out. Two columns over 4 rows: col 0 stores 3 entries, col 1
        // stores 1.
        let sq_devs = [2.5f64, 0.75];
        let col_nnz = [3usize, 1];
        let col_means = [1.25f64, 0.5];
        let got = finalize_implicit_zero_variance(&sq_devs, &col_nnz, &col_means, 4).unwrap();
        let want: Vec<f64> = (0..2)
            .map(|c| {
                let n_zeros = (4 - col_nnz[c]) as f64;
                (sq_devs[c] + n_zeros * col_means[c] * col_means[c]) / 4.0
            })
            .collect();
        assert_eq!(got, want, "must stay bit-identical on canonical input");

        // extent == 0 short-circuits rather than dividing by zero.
        assert_eq!(
            finalize_implicit_zero_variance(&[0.0, 0.0], &[0, 0], &[0.0, 0.0], 0).unwrap(),
            vec![0.0, 0.0]
        );

        // And a column claiming more stored entries than there are rows is
        // reported, not folded in as ~1.8e19 zeros.
        assert!(matches!(
            finalize_implicit_zero_variance(&[1.0], &[5], &[1.0], 4),
            Err(CsrError::NonCanonicalAxis { extent: 4, nnz: 5 })
        ));
    }

    #[test]
    fn finalize_implicit_zero_variance_validates_on_every_profile() {
        // `pub` fn: mismatched slices must be an error, not a `debug_assert!`
        // that compiles out and lets `zip` truncate to the shortest input. A
        // short return would be read by a caller as a *different* column's
        // variance, which is worse than the panic it replaced.
        assert!(matches!(
            finalize_implicit_zero_variance(&[1.0, 2.0], &[1], &[1.0, 2.0], 4),
            Err(CsrError::StatsLengthMismatch {
                sq_devs: 2,
                col_nnz: 1,
                col_means: 2
            })
        ));

        // The zero-extent short-circuit must not smuggle a bad count past the
        // contract: an axis of 0 cells holding 3 entries is non-canonical, and
        // used to return `Ok([0.0])` because the early return came first.
        assert!(matches!(
            finalize_implicit_zero_variance(&[0.0], &[3], &[0.0], 0),
            Err(CsrError::NonCanonicalAxis { extent: 0, nnz: 3 })
        ));
        // ...while a genuinely empty axis still short-circuits.
        assert_eq!(
            finalize_implicit_zero_variance(&[0.0], &[0], &[0.0], 0).unwrap(),
            vec![0.0]
        );
    }

    #[test]
    fn zero_extent_does_not_bypass_the_row_statistics_contract() {
        // A (1, 0) shape holding a stored entry is non-canonical: zero cells
        // cannot hold one entry. The `n_cols == 0` early return used to come
        // first, so these answered 0.0 / ±inf instead — the same ordering bug
        // that was fixed in `finalize_implicit_zero_variance` and initially
        // missed here.
        let bad = ScxCsr {
            shape: (1, 0),
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
        };
        assert!(matches!(
            bad.row_var(),
            Err(CsrError::NonCanonicalAxis { extent: 0, nnz: 1 })
        ));
        assert!(matches!(
            bad.row_max(),
            Err(CsrError::NonCanonicalAxis { extent: 0, nnz: 1 })
        ));
        assert!(matches!(
            bad.row_min(),
            Err(CsrError::NonCanonicalAxis { extent: 0, nnz: 1 })
        ));

        // A genuinely empty (0-column, 0-entry) row still short-circuits.
        let empty = ScxCsr::new((1, 0), vec![0, 0], vec![], vec![]).unwrap();
        assert_eq!(empty.row_var().unwrap(), vec![0.0]);
        assert_eq!(empty.row_max().unwrap(), vec![f64::NEG_INFINITY]);
        assert_eq!(empty.row_min().unwrap(), vec![f64::INFINITY]);
    }

    #[test]
    fn duplicates_under_the_extent_are_not_detected() {
        // Pins the LIMIT of `implicit_zero_count`, so nobody reads the guard as
        // a uniqueness check. Pigeonhole runs one way: `nnz > extent` proves a
        // duplicate, but a duplicate that leaves the row under its extent is
        // invisible — and that is the common shape in real data.
        //
        // 1x3 storing column 0 twice: nnz = 2 <= n_cols = 3, so no error.
        let dup = ScxCsr::new((1, 3), vec![0, 2], vec![0, 0], vec![1.0, 2.0]).unwrap();
        let var = dup
            .row_var()
            .expect("under-extent duplicate is NOT rejected");

        // What it returns is computed from both stored entries plus one
        // implicit zero: mean = 3/3 = 1, so (0 + 1 + 1)/3.
        assert!((var[0] - 2.0 / 3.0).abs() < 1e-12, "got {}", var[0]);

        // Canonicalizing the duplicate (summing to dense [3, 0, 0]) gives 2.0.
        // The gap between 0.667 and 2.0 is the silent-corruption window this
        // guard does not close; closing it needs an ordering pass at the decode
        // seam (`scx validate --deep`), deliberately not on the read path.
        let canonical = ScxCsr::new((1, 3), vec![0, 1], vec![0], vec![3.0]).unwrap();
        assert!((canonical.row_var().unwrap()[0] - 2.0).abs() < 1e-12);
    }
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    /// Build by struct literal, the way `new_unchecked` callers and the R /
    /// numpy interop paths do — bypassing `new` entirely. That is the shape
    /// `validate` exists for, and it is not reachable through the `new` error
    /// tests above, which can only produce a matrix `new` already rejected.
    fn literal(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> ScxCsr {
        ScxCsr {
            shape,
            indptr,
            indices,
            data,
        }
    }

    #[test]
    fn validate_accepts_a_well_formed_matrix() {
        // The accept side first: without it every assertion below would pass
        // against a `validate` that rejected everything.
        literal((2, 4), vec![0, 2, 3], vec![0, 3, 1], vec![1.0, 2.0, 3.0])
            .validate()
            .expect("a well-formed CSR must validate");
        // Degenerate but legal: no rows, no columns, no entries.
        literal((0, 0), vec![0], vec![], vec![])
            .validate()
            .expect("the empty matrix is well formed");
    }

    #[test]
    fn validate_catches_each_structural_invariant() {
        assert!(matches!(
            literal((3, 5), vec![0, 2, 5], vec![], vec![])
                .validate()
                .unwrap_err(),
            CsrError::IndptrLength {
                got: 3,
                expected: 4
            }
        ));
        assert!(matches!(
            literal((1, 5), vec![5, 7], vec![1, 3], vec![1.0, 2.0])
                .validate()
                .unwrap_err(),
            CsrError::IndptrNonZeroStart(5)
        ));
        assert!(matches!(
            literal((2, 5), vec![0, 3, 1], vec![0, 1, 2], vec![1.0, 2.0, 3.0])
                .validate()
                .unwrap_err(),
            CsrError::IndptrNotMonotonic { index: 2 }
        ));
        assert!(matches!(
            literal((1, 5), vec![0, 2], vec![0, 1], vec![1.0])
                .validate()
                .unwrap_err(),
            CsrError::IndicesDataMismatch {
                indices: 2,
                data: 1
            }
        ));
        assert!(matches!(
            literal((1, 5), vec![0, 3], vec![0, 1], vec![1.0, 2.0])
                .validate()
                .unwrap_err(),
            CsrError::NnzMismatch {
                indptr_nnz: 3,
                actual_nnz: 2
            }
        ));
        assert!(matches!(
            literal((1, 2), vec![0, 1], vec![7], vec![1.0])
                .validate()
                .unwrap_err(),
            CsrError::IndexOutOfRange {
                index: 7,
                n_cols: 2,
                position: 0
            }
        ));
        // A negative index is the same invariant, and the one that becomes a
        // huge `usize` rather than a panic when it is not caught.
        assert!(matches!(
            literal((1, 5), vec![0, 1], vec![-1], vec![1.0])
                .validate()
                .unwrap_err(),
            CsrError::IndexOutOfRange { index: -1, .. }
        ));
    }

    /// The documented boundary, pinned so that "completing" `validate` with an
    /// ordering pass is a test failure rather than a silent reversal of a
    /// measured decision.
    ///
    /// Both matrices below violate `docs/format.md` § "v3 canonical CSR
    /// invariant" — one stores a column twice, the other stores its columns
    /// out of order — and `validate` accepts both, because ordering and
    /// uniqueness are `scx validate --deep`'s job on the read side and
    /// `is_canonical_csr`'s on the write side.
    #[test]
    fn validate_deliberately_accepts_a_non_canonical_row() {
        // `is_canonical_csr` speaks the on-disk widths, not scipy's.
        fn canonical(csr: &ScxCsr) -> bool {
            let indptr: Vec<u64> = csr.indptr.iter().map(|&v| v as u64).collect();
            let indices: Vec<u32> = csr.indices.iter().map(|&v| v as u32).collect();
            crate::validate::is_canonical_csr(&indptr, &indices, &csr.data)
        }

        let duplicate = literal((1, 3), vec![0, 2], vec![0, 0], vec![1.0, 2.0]);
        duplicate.validate().expect("structurally valid");
        assert!(
            !canonical(&duplicate),
            "the canonical check is the one that rejects a duplicate"
        );

        let unsorted = literal((1, 3), vec![0, 2], vec![2, 0], vec![1.0, 2.0]);
        unsorted.validate().expect("structurally valid");
        assert!(
            !canonical(&unsorted),
            "the canonical check is the one that rejects an unsorted row"
        );
    }
}
