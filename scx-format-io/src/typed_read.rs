//! In-assembly narrow whole-matrix reader.
//!
//! Assembles a whole-matrix CSR (or dense) **directly at the caller's target
//! dtype**, never allocating the intermediate full-matrix `f32` [`ScxCsr`] the
//! default read path builds. Each shard is still decoded one-at-a-time, but to
//! its *native* stream (integer shards → `u32`, float shards → `f32`) and cast
//! straight into the target-width buffer via the fail-loud cast gate
//! (`scx_codec::checked_cast_*_into`). Two consequences:
//!
//! - **Peak-RSS drop** — a narrow `data_dtype="uint16"` read allocates the value
//!   buffer at 2 B/nnz, not 4 B/nnz (f32) + a 2 B/nnz cast copy.
//! - **Exact integer reads > 2²⁴** — integer→integer narrows never touch `f32`,
//!   so counts above `2²⁴` materialize losslessly (where the f32 path rounds).
//!
//! The default (`csr`/`f32`/`i32`) plan is served by the untouched zero-copy
//! path in `reader.rs`; callers gate on [`MaterializePlan::is_default_csr_f32`]
//! and only route non-default plans here.

use std::ops::Range;

use scx_codec::{
    checked_cast_f32_into, checked_cast_u32_into, guard_decode_loss_for, ShardValuesNative,
};
use scx_sparse::{
    IndexBuffer, IndexDtype, MaterializePlan, TypedCsr, TypedDense, ValueBuffer, ValueDtype,
};

use crate::catalog::FullCatalogEntry;
use crate::error::{Result, ScxError};
use crate::reader::ScxReader;

impl ScxReader {
    /// Read the whole `X` matrix as a [`TypedCsr`] materialized directly at
    /// `plan`'s value/index dtypes (deletion-vector filtered, mirroring
    /// [`read_all_csr_shards_filtered`](ScxReader::read_all_csr_shards_filtered)).
    pub fn read_all_csr_shards_typed(&self, plan: &MaterializePlan) -> Result<TypedCsr> {
        let shards = self.catalog().shards_sorted();
        let csr = self.assemble_shards_typed(&shards, self.n_vars() as usize, plan)?;
        #[cfg(feature = "deletion-vectors")]
        {
            self.filter_typed_csr_rows_by_deletion_vectors(csr, crate::deletion_vectors::DV_GLOBAL)
        }
        #[cfg(not(feature = "deletion-vectors"))]
        {
            Ok(csr)
        }
    }

    /// Read the whole `X` matrix as a row-major [`TypedDense`] at `plan`'s value
    /// dtype. Builds the (deletion-filtered) [`TypedCsr`] then scatters into a
    /// zeroed dense buffer; `index_dtype` is irrelevant for dense output.
    pub fn read_all_csr_shards_dense_typed(&self, plan: &MaterializePlan) -> Result<TypedDense> {
        // Assemble the CSR with i32 indices (used only transiently for scatter).
        let csr_plan = MaterializePlan {
            container: scx_sparse::Container::Csr,
            data_dtype: plan.data_dtype,
            index_dtype: IndexDtype::I32,
            allow_lossy: plan.allow_lossy,
        };
        let csr = self.read_all_csr_shards_typed(&csr_plan)?;
        scatter_typed_csr_to_dense(&csr)
    }

    /// Read all `adata.raw` CSR shards as a [`TypedCsr`] (unfiltered, mirroring
    /// [`read_all_raw_csr_shards`](ScxReader::read_all_raw_csr_shards)).
    pub fn read_all_raw_csr_shards_typed(&self, plan: &MaterializePlan) -> Result<TypedCsr> {
        let shards = self.catalog().raw_csr_shards_sorted();
        let n_cols = self.raw_n_vars().unwrap_or(0);
        self.assemble_shards_typed(&shards, n_cols, plan)
    }

    /// Read a single modality's whole `X` matrix as a [`TypedCsr`] materialized
    /// directly at `plan`'s value/index dtypes.
    ///
    /// **Unfiltered** by construction — it delegates straight to the
    /// [`assemble_shards_typed`](Self::assemble_shards_typed) core, mirroring the
    /// f32 sibling [`read_all_csr_shards_for`](ScxReader::read_all_csr_shards_for).
    /// This is an intentional asymmetry with the global
    /// [`read_all_csr_shards_typed`](Self::read_all_csr_shards_typed) (which IS
    /// deletion-vector filtered): the per-modality path stays unfiltered because
    /// `to_mudata` reads each modality's X unfiltered to keep its row count in
    /// lockstep with the unfiltered global obs axis (see `pyscx::mudata`). The
    /// deletion-filtered per-modality read is
    /// [`read_all_csr_shards_for_filtered`](ScxReader::read_all_csr_shards_for_filtered)
    /// (f32); the global filtered typed read is
    /// [`read_all_csr_shards_typed`](Self::read_all_csr_shards_typed).
    ///
    /// `n_cols` comes from the modality table's `n_vars` (matching
    /// `read_all_csr_shards_for`'s preferred shape); a missing `modality_info`
    /// is an error here (the typed assembler needs `n_cols` up front, whereas the
    /// f32 path can fall back to the assembled shard extent).
    ///
    /// `plan.index_dtype` narrows the returned [`TypedCsr`]'s index buffer, but is
    /// effectively a **no-op for a Python CSR** consumer: scipy upcasts `int16 →
    /// int32` on `csr_matrix` construction (see `docs/api.md`).
    pub fn read_all_csr_shards_for_typed(
        &self,
        modality_id: u8,
        plan: &MaterializePlan,
    ) -> Result<TypedCsr> {
        let shards = self.catalog().csr_shards_for_modality(modality_id);
        let n_cols = self
            .modality_info(modality_id)
            .map(|i| i.n_vars as usize)
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "modality_info({modality_id}) is None — modality table corrupt \
                     or file is single-modality"
                ))
            })?;
        self.assemble_shards_typed(&shards, n_cols, plan)
    }

    /// Read a layer as a [`TypedCsr`] (unfiltered, mirroring
    /// [`read_layer`](ScxReader::read_layer)).
    pub fn read_layer_typed(&self, name: &str, plan: &MaterializePlan) -> Result<TypedCsr> {
        let mut shards = self.legacy_layer_shards(name);
        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!("layer '{name}'")));
        }
        shards.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));
        self.assemble_shards_typed(&shards, self.n_vars() as usize, plan)
    }

    /// Shared sequential typed assembly: pre-size from catalog stats, decode each
    /// shard to native, cast into the target-width slices. Unfiltered (callers
    /// apply deletion vectors as needed).
    fn assemble_shards_typed(
        &self,
        shards: &[&FullCatalogEntry],
        n_cols: usize,
        plan: &MaterializePlan,
    ) -> Result<TypedCsr> {
        if shards.is_empty() {
            return Ok(TypedCsr {
                shape: (0, n_cols),
                indptr: vec![0i64],
                indices: IndexBuffer::zeroed(plan.index_dtype, 0),
                values: ValueBuffer::zeroed(plan.data_dtype, 0),
            });
        }

        // Pre-compute per-shard (n_rows, nnz) from catalog stats and the max
        // integer value across in-scope shards — all without decoding.
        let mut shard_sizes: Vec<(usize, usize)> = Vec::with_capacity(shards.len());
        let mut max_value: u32 = 0;
        for e in shards {
            let stats = e.stats.as_ref().ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "shard entry '{}' at offset {} has no stats block",
                    e.name, e.offset
                ))
            })?;
            let n_rows = stats.row_end.checked_sub(stats.row_start).ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "shard '{}' has row_end {} < row_start {}",
                    e.name, stats.row_end, stats.row_start
                ))
            })? as usize;
            shard_sizes.push((n_rows, stats.nnz as usize));
            max_value = max_value.max(stats.value_max);
        }
        let total_rows: usize = shard_sizes.iter().map(|(r, _)| *r).sum();
        let total_nnz: usize = shard_sizes.iter().map(|(_, n)| *n).sum();

        // O(1) pre-decode guard: fail loud before the big allocation when the
        // target dtype cannot represent `max_value` (integer shards only; float
        // shards report value_max = 0 → never fires). The per-element gate in
        // the fill helpers is the real guarantee.
        guard_value_dtype(max_value, plan.data_dtype, plan.allow_lossy)?;

        let mut indptr = vec![0i64; total_rows + 1];
        let mut indices = IndexBuffer::zeroed(plan.index_dtype, total_nnz);
        let mut values = ValueBuffer::zeroed(plan.data_dtype, total_nnz);

        let mut cum_rows = 0usize;
        let mut cum_nnz = 0usize;

        for (i, entry) in shards.iter().enumerate() {
            let (n_rows, nnz) = shard_sizes[i];
            let (shard_ip, shard_ix, shard_vals) = self.read_shard_from_entry_native(entry)?;

            // Decoded-vs-catalog length checks (returned errors, not asserts —
            // a stat-drifted catalog must not panic the slice writes below).
            if shard_ip.len() != n_rows + 1 {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {i} indptr length mismatch: catalog stats say {}, decoded {}",
                    n_rows + 1,
                    shard_ip.len()
                )));
            }
            if shard_ix.len() != nnz {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {i} indices length mismatch: catalog stats say {nnz}, decoded {}",
                    shard_ix.len()
                )));
            }
            if native_values_len(&shard_vals) != nnz {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {i} data length mismatch: catalog stats say {nnz}, decoded {}",
                    native_values_len(&shard_vals)
                )));
            }

            let range = cum_nnz..cum_nnz + nnz;
            fill_index_slice(&mut indices, range.clone(), &shard_ix, plan.allow_lossy)?;
            fill_value_slice(&mut values, range, &shard_vals, plan.allow_lossy)?;

            // Rebase indptr with the cumulative nnz offset.
            if i == 0 {
                indptr[0..n_rows + 1].copy_from_slice(&shard_ip);
            } else {
                let nnz_off_i64 = cum_nnz as i64;
                for j in 0..n_rows {
                    indptr[cum_rows + 1 + j] = shard_ip[j + 1] + nnz_off_i64;
                }
            }

            cum_rows += n_rows;
            cum_nnz += nnz;
        }

        let n_rows = indptr.len().saturating_sub(1);
        Ok(TypedCsr {
            shape: (n_rows, n_cols),
            indptr,
            indices,
            values,
        })
    }

    /// Deletion-vector row compaction over a [`TypedCsr`] (typed twin of
    /// `filter_csr_rows_by_deletion_vectors`), using the keep mask for
    /// `modality_id` (global-only for `DV_GLOBAL`).
    #[cfg(feature = "deletion-vectors")]
    fn filter_typed_csr_rows_by_deletion_vectors(
        &self,
        csr: TypedCsr,
        modality_id: u8,
    ) -> Result<TypedCsr> {
        let keep = match self.deletion_keep_mask_for(modality_id)? {
            Some(keep) => keep,
            None => return Ok(csr),
        };
        // Same guard as the untyped twin, and it has to be the same guard: the
        // two paths are each other's oracle in `typed_read_applies_deletion_vectors`,
        // so a check on one side only would make them disagree on exactly the
        // malformed files where agreement is the evidence.
        crate::deletion_vectors::check_keep_mask_covers_csr(keep.len(), csr.shape.0)?;
        let mut new_indptr = vec![0i64];
        for (row, &is_kept) in keep.iter().enumerate() {
            if !is_kept {
                continue;
            }
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            let prev = *new_indptr.last().unwrap();
            new_indptr.push(prev + (end - start) as i64);
        }
        let new_n_rows = new_indptr.len() - 1;
        let indices = compact_index_buffer(&csr.indices, &csr.indptr, &keep);
        let values = compact_value_buffer(&csr.values, &csr.indptr, &keep);
        Ok(TypedCsr {
            shape: (new_n_rows, csr.shape.1),
            indptr: new_indptr,
            indices,
            values,
        })
    }
}

/// The `value_max` guard, dispatched from the runtime `ValueDtype` to the
/// generic `guard_decode_loss_for::<T>` (scx-codec can't see `ValueDtype`).
fn guard_value_dtype(max_value: u32, dtype: ValueDtype, allow_lossy: bool) -> Result<()> {
    match dtype {
        ValueDtype::F16 => guard_decode_loss_for::<half::f16>(max_value, allow_lossy)?,
        ValueDtype::F32 => guard_decode_loss_for::<f32>(max_value, allow_lossy)?,
        ValueDtype::F64 => guard_decode_loss_for::<f64>(max_value, allow_lossy)?,
        ValueDtype::I8 => guard_decode_loss_for::<i8>(max_value, allow_lossy)?,
        ValueDtype::I16 => guard_decode_loss_for::<i16>(max_value, allow_lossy)?,
        ValueDtype::I32 => guard_decode_loss_for::<i32>(max_value, allow_lossy)?,
        ValueDtype::I64 => guard_decode_loss_for::<i64>(max_value, allow_lossy)?,
        ValueDtype::U8 => guard_decode_loss_for::<u8>(max_value, allow_lossy)?,
        ValueDtype::U16 => guard_decode_loss_for::<u16>(max_value, allow_lossy)?,
        ValueDtype::U32 => guard_decode_loss_for::<u32>(max_value, allow_lossy)?,
    }
    Ok(())
}

fn native_values_len(vals: &ShardValuesNative) -> usize {
    match vals {
        ShardValuesNative::U32(v) => v.len(),
        ShardValuesNative::F32(v) => v.len(),
    }
}

/// Cast a shard's native `u32` indices into `dst[range]` at the target index dtype.
fn fill_index_slice(
    dst: &mut IndexBuffer,
    range: Range<usize>,
    src: &[u32],
    allow_lossy: bool,
) -> Result<()> {
    match dst {
        IndexBuffer::I16(v) => checked_cast_u32_into::<i16>(src, &mut v[range], allow_lossy)?,
        IndexBuffer::I32(v) => checked_cast_u32_into::<i32>(src, &mut v[range], allow_lossy)?,
        IndexBuffer::I64(v) => checked_cast_u32_into::<i64>(src, &mut v[range], allow_lossy)?,
    }
    Ok(())
}

/// Cast a shard's native values (u32 for integer shards, f32 for float shards)
/// into `dst[range]` at the target value dtype. One line per value dtype; both
/// source variants are generated by the macro.
fn fill_value_slice(
    dst: &mut ValueBuffer,
    range: Range<usize>,
    src: &ShardValuesNative,
    allow_lossy: bool,
) -> Result<()> {
    macro_rules! value_arms {
        ($( $arm:ident => $ty:ty ),* $(,)?) => {
            match (dst, src) {
                $(
                    (ValueBuffer::$arm(v), ShardValuesNative::U32(s)) => {
                        checked_cast_u32_into::<$ty>(s, &mut v[range.clone()], allow_lossy)?
                    }
                    (ValueBuffer::$arm(v), ShardValuesNative::F32(s)) => {
                        checked_cast_f32_into::<$ty>(s, &mut v[range.clone()], allow_lossy)?
                    }
                )*
            }
        };
    }
    value_arms!(
        F16 => half::f16, F32 => f32, F64 => f64,
        I8 => i8, I16 => i16, I32 => i32, I64 => i64,
        U8 => u8, U16 => u16, U32 => u32,
    );
    Ok(())
}

/// Total nnz retained by the keep-mask — used to pre-size the compacted
/// buffers so the copy doesn't repeatedly reallocate (the common case is few
/// deletions on a large matrix, where the kept total is close to the original).
///
/// # Caller obligation
///
/// `keep.len() + 1 == indptr.len()`. Today's only caller has already been
/// through [`check_keep_mask_covers_csr`](crate::deletion_vectors::check_keep_mask_covers_csr),
/// which is where a malformed file is turned into an error; the assert exists
/// so a future direct caller that skips it trips in tests rather than in
/// somebody's file.
#[cfg(feature = "deletion-vectors")]
fn kept_nnz(indptr: &[i64], keep: &[bool]) -> usize {
    debug_assert_eq!(
        keep.len() + 1,
        indptr.len(),
        "keep mask and indptr disagree; call check_keep_mask_covers_csr first"
    );
    let mut n = 0usize;
    for (row, &k) in keep.iter().enumerate() {
        if k {
            n += (indptr[row + 1] - indptr[row]) as usize;
        }
    }
    n
}

/// Compact an index buffer to the kept rows (deletion-vector filter).
///
/// Same caller obligation as [`kept_nnz`], which it calls: `keep` must cover
/// exactly the rows `indptr` describes.
#[cfg(feature = "deletion-vectors")]
fn compact_index_buffer(src: &IndexBuffer, indptr: &[i64], keep: &[bool]) -> IndexBuffer {
    let cap = kept_nnz(indptr, keep);
    macro_rules! compact {
        ($arm:ident, $v:expr) => {{
            let mut out = Vec::with_capacity(cap);
            for (row, &k) in keep.iter().enumerate() {
                if !k {
                    continue;
                }
                let s = indptr[row] as usize;
                let e = indptr[row + 1] as usize;
                out.extend_from_slice(&$v[s..e]);
            }
            IndexBuffer::$arm(out)
        }};
    }
    match src {
        IndexBuffer::I16(v) => compact!(I16, v),
        IndexBuffer::I32(v) => compact!(I32, v),
        IndexBuffer::I64(v) => compact!(I64, v),
    }
}

/// Compact a value buffer to the kept rows (deletion-vector filter).
///
/// Same caller obligation as [`kept_nnz`], which it calls.
#[cfg(feature = "deletion-vectors")]
fn compact_value_buffer(src: &ValueBuffer, indptr: &[i64], keep: &[bool]) -> ValueBuffer {
    let cap = kept_nnz(indptr, keep);
    macro_rules! compact {
        ($arm:ident, $v:expr) => {{
            let mut out = Vec::with_capacity(cap);
            for (row, &k) in keep.iter().enumerate() {
                if !k {
                    continue;
                }
                let s = indptr[row] as usize;
                let e = indptr[row + 1] as usize;
                out.extend_from_slice(&$v[s..e]);
            }
            ValueBuffer::$arm(out)
        }};
    }
    match src {
        ValueBuffer::F16(v) => compact!(F16, v),
        ValueBuffer::F32(v) => compact!(F32, v),
        ValueBuffer::F64(v) => compact!(F64, v),
        ValueBuffer::I8(v) => compact!(I8, v),
        ValueBuffer::I16(v) => compact!(I16, v),
        ValueBuffer::I32(v) => compact!(I32, v),
        ValueBuffer::I64(v) => compact!(I64, v),
        ValueBuffer::U8(v) => compact!(U8, v),
        ValueBuffer::U16(v) => compact!(U16, v),
        ValueBuffer::U32(v) => compact!(U32, v),
    }
}

/// Scatter a [`TypedCsr`] (with `i32` indices) into a row-major dense
/// [`TypedDense`] at the same value dtype. Mirrors `ScxCsr::to_dense_dtype`.
fn scatter_typed_csr_to_dense(csr: &TypedCsr) -> Result<TypedDense> {
    let (n_rows, n_cols) = csr.shape;
    let total = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| ScxError::InvalidCatalog(format!("dense dim overflow {n_rows}×{n_cols}")))?;
    let cols: &[i32] = match &csr.indices {
        IndexBuffer::I32(v) => v,
        _ => {
            return Err(ScxError::InvalidCatalog(
                "dense scatter expects i32 CSR indices".to_string(),
            ))
        }
    };
    let indptr = &csr.indptr;

    // The decode seam already bounded every index against its **shard's**
    // `n_minor`; this bounds against the **assembled matrix's** `n_cols`, which
    // comes from the file header's `n_vars`. They agree on any file a writer
    // produced, so this is normally a no-op — but they are two different numbers
    // and a shard header claiming `n_minor > n_vars` would slip an index through
    // the seam into the write below.
    //
    // Worth the one pass here specifically, and nowhere else: this is the site
    // where a violation does *not* announce itself. `dense[base + col]` with an
    // out-of-range `col` runs off the end of one row into the next and returns a
    // plausible, wrong matrix — no panic, no error. Every other consumer indexes
    // a `Vec` sized by the same axis it validates against, so it panics instead.
    // Cost is negligible against the `n_rows × n_cols` allocation this function
    // already makes.
    if let Some((position, &bad)) = cols
        .iter()
        .enumerate()
        .find(|&(_, &c)| c < 0 || c as usize >= n_cols)
    {
        return Err(ScxError::ShardIndexOutOfRange {
            index: bad as u32,
            position,
            n_minor: n_cols as u32,
        });
    }

    macro_rules! scatter {
        ($arm:ident, $vals:expr, $ty:ty) => {{
            let mut dense = vec![<$ty>::default(); total];
            for row in 0..n_rows {
                let s = indptr[row] as usize;
                let e = indptr[row + 1] as usize;
                let base = row * n_cols;
                for k in s..e {
                    dense[base + cols[k] as usize] = $vals[k];
                }
            }
            ValueBuffer::$arm(dense)
        }};
    }
    let values = match &csr.values {
        ValueBuffer::F16(v) => scatter!(F16, v, half::f16),
        ValueBuffer::F32(v) => scatter!(F32, v, f32),
        ValueBuffer::F64(v) => scatter!(F64, v, f64),
        ValueBuffer::I8(v) => scatter!(I8, v, i8),
        ValueBuffer::I16(v) => scatter!(I16, v, i16),
        ValueBuffer::I32(v) => scatter!(I32, v, i32),
        ValueBuffer::I64(v) => scatter!(I64, v, i64),
        ValueBuffer::U8(v) => scatter!(U8, v, u8),
        ValueBuffer::U16(v) => scatter!(U16, v, u16),
        ValueBuffer::U32(v) => scatter!(U32, v, u32),
    };
    Ok(TypedDense {
        shape: (n_rows, n_cols),
        values,
    })
}

#[cfg(test)]
#[path = "typed_read_tests.rs"]
mod tests;
