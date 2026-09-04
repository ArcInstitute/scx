//! Deletion vectors and detection bitmaps: the reads that honour a
//! logically-deleted row.
//!
//! Feature-gated on `deletion-vectors` in its entirety — at the `mod filtered;`
//! declaration in `reader/mod.rs`, so no item in here carries its own `#[cfg]`
//! and none can be added without one by accident.

use super::*;

impl ScxReader {
    /// Phase 5b: read a single detection-bitmap shard by index, scoped
    /// to a modality (`modality_id == 0` for unimodal files). Shards
    /// are ordered by `row_start`, matching the CSR shard order.
    pub fn read_bitmap_shard_for(
        &self,
        modality_id: u8,
        shard_idx: usize,
    ) -> Result<crate::bitmap::BitmapShard> {
        let entries = self.full_catalog.bitmap_shards_for_modality(modality_id);
        let entry = entries.get(shard_idx).ok_or_else(|| {
            ScxError::SectionNotFound(format!(
                "X/bitmap/shard_{shard_idx} (modality_id={modality_id})"
            ))
        })?;
        let slice = self.section_bytes(entry)?;
        crate::bitmap::BitmapShard::read_from(&mut Cursor::new(slice), slice.len())
    }

    /// Phase 5b: unimodal helper — equivalent to
    /// `read_bitmap_shard_for(0, shard_idx)`.
    pub fn read_bitmap_shard(&self, shard_idx: usize) -> Result<crate::bitmap::BitmapShard> {
        self.read_bitmap_shard_for(0, shard_idx)
    }

    /// Number of bitmap shards available for a modality (0 if none).
    pub fn bitmap_shard_count(&self, modality_id: u8) -> usize {
        self.full_catalog
            .bitmap_shards_for_modality(modality_id)
            .len()
    }

    /// Read deletion vectors if present. Returns `Ok(None)` if the file has no
    /// DV flag set. A legacy v1 (per-shard) section is folded to v2 global
    /// obs-row indices here, so callers only ever see v2.
    pub fn read_deletion_vectors(
        &self,
    ) -> Result<Option<crate::deletion_vectors::DeletionVectors>> {
        if !self.header.has_deletion_vectors() {
            return Ok(None);
        }
        let entry = self
            .full_catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::DeletionVectors)
            .ok_or_else(|| ScxError::SectionNotFound("deletion_vectors".to_string()))?;
        let slice = self.section_bytes(entry)?;
        let mut dv = crate::deletion_vectors::DeletionVectors::read_from(
            &mut Cursor::new(slice),
            slice.len(),
        )?;
        // Fold a legacy v1 (per-shard, shard-local) section to the v2 global
        // representation so every downstream consumer sees the v2 shape.
        dv.fold_v1_to_global(&self.full_catalog);
        Ok(Some(dv))
    }

    /// Build the whole-cell (global) obs-indexed keep mask (`true` = retained)
    /// implied by this file's deletion vectors, or `None` when the file has no
    /// deletion vectors or nothing is deleted.
    ///
    /// This is the single entry point that the reader's CSR filter, the
    /// h5ad/h5mu streaming export, `scx compact`, and the `pyscx` obs filter
    /// all share, so the row-keep semantics stay identical everywhere. Delegates
    /// to [`Self::deletion_keep_mask_for`] with the global modality key.
    pub fn deletion_keep_mask(&self) -> Result<Option<Vec<bool>>> {
        self.deletion_keep_mask_for(crate::deletion_vectors::DV_GLOBAL)
    }

    /// Modality-aware keep mask (`true` = retained): global (whole-cell)
    /// deletions plus, for `modality_id >= 1`, that modality's scoped deletions.
    /// Returns `None` when no deletion applies to the requested modality.
    ///
    /// `modality_id == 0` (`DV_GLOBAL`) is the whole-cell mask that
    /// [`Self::deletion_keep_mask`] exposes and that every global/single-modality
    /// read uses. Because shipped writers only populate the global bitmap and it
    /// applies identically to every modality, this is currently equal to the
    /// global mask for all `modality_id`; scoped (`>= 1`) deletion is reserved.
    pub fn deletion_keep_mask_for(&self, modality_id: u8) -> Result<Option<Vec<bool>>> {
        if !self.header.has_deletion_vectors() {
            return Ok(None);
        }
        let dv = match self.read_deletion_vectors()? {
            Some(dv) => dv,
            None => return Ok(None),
        };
        let applies = dv.total_deleted() > 0
            || (modality_id != crate::deletion_vectors::DV_GLOBAL
                && dv
                    .deletions
                    .get(&modality_id)
                    .is_some_and(|b| !b.is_empty()));
        if !applies {
            return Ok(None);
        }
        Ok(Some(dv.build_keep_mask(self.n_obs() as usize, modality_id)))
    }

    /// Read obs metadata with deletion vectors applied — the obs half of
    /// [`Self::read_all_csr_shards_filtered`], and the counterpart every caller
    /// that materializes a matrix for a user needs.
    ///
    /// [`Self::read_obs`] is *physical*: it returns `header.n_obs` rows whether
    /// or not any are logically deleted. Pairing it with a filtered CSR yields
    /// an obs frame longer than the matrix, which is worse than either half
    /// being wrong alone — so the two must always move together. When the file
    /// has no deletion vectors (or nothing is deleted) this returns exactly what
    /// `read_obs()` returns, with no copy.
    pub fn read_obs_filtered(&self) -> Result<RecordBatch> {
        let obs = self.read_obs()?;
        match self.deletion_keep_mask()? {
            Some(mask) => filter_batch_by_keep_mask(&obs, &mask),
            None => Ok(obs),
        }
    }

    /// Column-projected twin of [`Self::read_obs_filtered`]:
    /// [`Self::read_obs_keys`] with the deletion keep mask applied, so a
    /// projected read stays projected (and dictionary-compacted) in the logical
    /// row space. Filters rows only — a dictionary's declared values are kept
    /// whether or not a live row uses them, matching `read_obs_filtered`. No
    /// copy when nothing is deleted.
    pub fn read_obs_keys_filtered(&self, col_names: &[String]) -> Result<RecordBatch> {
        let obs = self.read_obs_keys(col_names)?;
        match self.deletion_keep_mask()? {
            Some(mask) => filter_batch_by_keep_mask(&obs, &mask),
            None => Ok(obs),
        }
    }

    /// [`Self::obs_categorical_many`] in the **logical** row space: the
    /// physical fold, then the deletion keep mask compacts each column's codes
    /// ([`crate::filter_codes_by_keep_mask`]). Categories are unchanged — an
    /// unreferenced level stays, exactly as in the physical result. The mask is
    /// built once for every column.
    pub fn obs_categorical_many_filtered(
        &self,
        cols: &[String],
    ) -> Result<Vec<(Vec<i32>, Vec<String>)>> {
        let out = self.obs_categorical_many(cols)?;
        match self.deletion_keep_mask()? {
            Some(mask) => out
                .into_iter()
                .map(|(codes, cats)| {
                    Ok((
                        crate::categorical::filter_codes_by_keep_mask(codes, &mask)?,
                        cats,
                    ))
                })
                .collect(),
            None => Ok(out),
        }
    }

    /// [`Self::obs_categorical`] in the logical row space; see
    /// [`Self::obs_categorical_many_filtered`].
    pub fn obs_categorical_filtered(&self, col: &str) -> Result<(Vec<i32>, Vec<String>)> {
        Ok(self
            .obs_categorical_many_filtered(std::slice::from_ref(&col.to_string()))?
            .pop()
            .expect("one column in ⇒ one column out"))
    }

    /// Read all CSR shards with deletion vectors applied.
    /// Deleted rows are excluded from the returned ScxCsr.
    /// If no deletion vectors are present, returns the same result as `read_all_csr_shards()`.
    pub fn read_all_csr_shards_filtered(&self) -> Result<ScxCsr> {
        let csr = self.read_all_csr_shards()?;
        self.filter_csr_rows_by_deletion_vectors(csr, crate::deletion_vectors::DV_GLOBAL)
    }

    /// Per-modality counterpart of [`Self::read_all_csr_shards_filtered`].
    /// Assembles the modality's CSR via [`Self::read_all_csr_shards_for`] and
    /// applies the modality-aware deletion-vector keep mask
    /// ([`Self::deletion_keep_mask_for`]). The global (whole-cell) bitmap always
    /// applies and is obs-indexed and shared across modalities (h5mu invariant),
    /// so each modality's filtered CSR has `n_obs - n_deleted` rows; a scoped
    /// (`modality_id >= 1`) bitmap, when present, additionally drops that
    /// modality's rows.
    pub fn read_all_csr_shards_for_filtered(&self, modality_id: u8) -> Result<ScxCsr> {
        let csr = self.read_all_csr_shards_for(modality_id)?;
        self.filter_csr_rows_by_deletion_vectors(csr, modality_id)
    }

    /// Read a named layer with deletion vectors applied.
    /// Deleted rows are excluded from the returned ScxCsr.
    /// If no deletion vectors are present, returns the same result as `read_layer()`.
    ///
    /// Layers must share X's row count (an AnnData invariant); the whole-cell
    /// row-keep mask is applied to the layer CSR.
    pub fn read_layer_filtered(&self, name: &str) -> Result<ScxCsr> {
        let csr = self.read_layer(name)?;
        self.filter_csr_rows_by_deletion_vectors(csr, crate::deletion_vectors::DV_GLOBAL)
    }

    /// Apply deletion vectors to an already-assembled CSR (X or layer), using
    /// the keep mask for `modality_id` (global-only for `DV_GLOBAL`). The mask
    /// is obs-indexed, so the input CSR must share X's row count.
    fn filter_csr_rows_by_deletion_vectors(&self, csr: ScxCsr, modality_id: u8) -> Result<ScxCsr> {
        // The keep mask is obs-indexed; the input CSR (X or a layer) shares
        // X's row count, so the mask aligns with its rows.
        let keep = match self.deletion_keep_mask_for(modality_id)? {
            Some(keep) => keep,
            None => return Ok(csr),
        };
        crate::deletion_vectors::check_keep_mask_covers_csr(keep.len(), csr.shape.0)?;

        // Filter CSR rows
        let mut new_indptr = vec![0i64];
        let mut new_indices = Vec::new();
        let mut new_data = Vec::new();

        for (row, &is_kept) in keep.iter().enumerate() {
            if !is_kept {
                continue;
            }
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            new_indices.extend_from_slice(&csr.indices[start..end]);
            new_data.extend_from_slice(&csr.data[start..end]);
            let prev = *new_indptr.last().unwrap();
            new_indptr.push(prev + (end - start) as i64);
        }

        let new_n_rows = new_indptr.len() - 1;
        Ok(ScxCsr::new_unchecked(
            (new_n_rows, csr.shape.1),
            new_indptr,
            new_indices,
            new_data,
        ))
    }
}
