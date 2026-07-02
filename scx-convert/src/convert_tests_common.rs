//! Shared fixtures, helpers, and re-exported imports for the
//! scx-convert integration tests (split out of the former single
//! `tests.rs` in T5.6). Subject modules glob-import this.
//!
//! Re-exports are `pub(crate) use` so subject modules reach them
//! through the glob; `unused_imports`/`dead_code` are allowed
//! because any individual helper/import may be used by only a
//! subset of the subject modules.
#![allow(dead_code, unused_imports)]

pub(crate) use std::num::NonZeroU32;

pub(crate) use std::path::Path;

pub(crate) use arrow::array::Array;

pub(crate) use arrow::datatypes::DataType;

pub(crate) use hdf5::types::VarLenUnicode;

pub(crate) use scx_codec::{CodecId, ValueEncoding};

pub(crate) use scx_format_io::reader::ScxReader;

pub(crate) use crate::h5ad::csc_transpose::csc_to_csr;

pub(crate) use super::detect::{
    detect_input_format, detect_matrix_format, InputFormat, MatrixFormat,
};

pub(crate) use scx_codec::value_encoding::is_integer_data;

pub(crate) use super::dtype::detect_value_encoding;

pub(crate) use super::pipeline::{
    h5ad_to_scx, scx_to_h5ad, tenx_to_scx, ConvertError, ConvertOptions,
};
pub(crate) use crate::GroupPass;

pub(crate) use super::stream::{CsrShardStream, StreamedCsrShard};

pub(crate) use super::warnings::WarningSink;

pub(crate) use crate::h5ad::read::read_x_matrix;

pub(crate) use crate::h5ad::stream::{open_layer_streaming, open_x_streaming};

pub(crate) use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

pub(crate) use scx_format_io::section::SectionType as FmtSectionType;

// Integration tests for h5ad/10x conversion
// All tests gated behind #[cfg(feature = "hdf5")] (in mod.rs)

/// Convert &str to VarLenUnicode, stripping NUL bytes if present.
pub(crate) fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap_or_else(|_| {
        let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
        cleaned
            .parse::<VarLenUnicode>()
            .expect("cleaned string should have no NUL bytes")
    })
}

// -----------------------------------------------------------------------
// Test helpers: create synthetic h5ad and 10x files
// -----------------------------------------------------------------------

/// Create a minimal h5ad file with known data.
pub(crate) fn create_test_h5ad(
    path: &Path,
    n_obs: usize,
    n_vars: usize,
    matrix_format: &str, // "csr", "csc", or "dense"
    include_extras: bool,
) {
    let file = hdf5::File::create(path).unwrap();

    // Build CSR data
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for row in 0..n_obs {
        // Each row has 2-3 nonzeros with integer values, sorted by column
        let nnz_in_row = 2 + (row % 2);
        let mut row_entries: Vec<(usize, f32)> = Vec::new();
        for j in 0..nnz_in_row {
            let col = (row * 3 + j) % n_vars;
            let val = ((row * 7 + j * 3 + 1) % 200 + 1) as f32;
            row_entries.push((col, val));
        }
        // Sort by column and deduplicate
        row_entries.sort_by_key(|&(c, _)| c);
        row_entries.dedup_by_key(|e| e.0);
        for (col, val) in &row_entries {
            indices.push(*col as i32);
            data.push(*val);
        }
        indptr.push(data.len() as i64);
    }

    match matrix_format {
        "csr" => {
            let x = file.create_group("X").unwrap();
            x.new_dataset::<i64>()
                .shape([indptr.len()])
                .create("indptr")
                .unwrap()
                .write(&indptr)
                .unwrap();
            x.new_dataset::<i32>()
                .shape([indices.len()])
                .create("indices")
                .unwrap()
                .write(&indices)
                .unwrap();
            x.new_dataset::<f32>()
                .shape([data.len()])
                .create("data")
                .unwrap()
                .write(&data)
                .unwrap();

            let enc = vlu("csr_matrix");
            x.new_attr::<VarLenUnicode>()
                .create("encoding-type")
                .unwrap()
                .write_scalar(&enc)
                .unwrap();

            let shape = [n_obs as i64, n_vars as i64];
            x.new_attr::<i64>()
                .shape([2])
                .create("shape")
                .unwrap()
                .write(&shape)
                .unwrap();
        }
        "csc" => {
            // Convert CSR to CSC for writing
            let (csc_indptr, csc_indices, csc_data) =
                csr_to_csc(&indptr, &indices, &data, n_obs, n_vars);

            let x = file.create_group("X").unwrap();
            x.new_dataset::<i64>()
                .shape([csc_indptr.len()])
                .create("indptr")
                .unwrap()
                .write(&csc_indptr)
                .unwrap();
            x.new_dataset::<i32>()
                .shape([csc_indices.len()])
                .create("indices")
                .unwrap()
                .write(&csc_indices)
                .unwrap();
            x.new_dataset::<f32>()
                .shape([csc_data.len()])
                .create("data")
                .unwrap()
                .write(&csc_data)
                .unwrap();

            let enc = vlu("csc_matrix");
            x.new_attr::<VarLenUnicode>()
                .create("encoding-type")
                .unwrap()
                .write_scalar(&enc)
                .unwrap();

            let shape = [n_obs as i64, n_vars as i64];
            x.new_attr::<i64>()
                .shape([2])
                .create("shape")
                .unwrap()
                .write(&shape)
                .unwrap();
        }
        "dense" => {
            // Build dense matrix from CSR
            let mut dense = vec![0.0f32; n_obs * n_vars];
            for row in 0..n_obs {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                for idx in start..end {
                    let col = indices[idx] as usize;
                    dense[row * n_vars + col] = data[idx];
                }
            }
            let nd_arr = ndarray::Array2::from_shape_vec((n_obs, n_vars), dense).unwrap();
            file.new_dataset::<f32>()
                .shape([n_obs, n_vars])
                .create("X")
                .unwrap()
                .write(&nd_arr)
                .unwrap();
        }
        _ => panic!("unknown matrix format: {matrix_format}"),
    }

    // Write obs
    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();

    let idx_name = vlu("_index");
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&idx_name)
        .unwrap();

    // Add numeric column
    let numeric_col: Vec<i32> = (0..n_obs).map(|i| (i * 10) as i32).collect();
    obs.new_dataset::<i32>()
        .shape([n_obs])
        .create("n_counts")
        .unwrap()
        .write(&numeric_col)
        .unwrap();

    // Write var
    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("gene_{i}"))).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("_index")
        .unwrap()
        .write(&var_index)
        .unwrap();

    let var_idx_name = vlu("_index");
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&var_idx_name)
        .unwrap();

    if include_extras {
        // obsm
        let obsm = file.create_group("obsm").unwrap();
        let pca_data: Vec<f32> = (0..n_obs * 2).map(|i| i as f32 * 0.1).collect();
        let nd_pca = ndarray::Array2::from_shape_vec((n_obs, 2), pca_data).unwrap();
        obsm.new_dataset::<f32>()
            .shape([n_obs, 2])
            .create("X_pca")
            .unwrap()
            .write(&nd_pca)
            .unwrap();

        // uns
        let uns = file.create_group("uns").unwrap();
        let species = vlu("human");
        uns.new_dataset::<VarLenUnicode>()
            .shape(())
            .create("species")
            .unwrap()
            .write_scalar(&species)
            .unwrap();
        uns.new_dataset::<i64>()
            .shape(())
            .create("version")
            .unwrap()
            .write_scalar(&2i64)
            .unwrap();

        // layers
        let layers = file.create_group("layers").unwrap();
        let raw_layer = layers.create_group("raw").unwrap();
        raw_layer
            .new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        raw_layer
            .new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        raw_layer
            .new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();

        let enc = vlu("csr_matrix");
        raw_layer
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&enc)
            .unwrap();
        let shape = [n_obs as i64, n_vars as i64];
        raw_layer
            .new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&shape)
            .unwrap();
    }
}

/// Convert CSR to CSC (helper for test data creation).
pub(crate) fn csr_to_csc(
    csr_indptr: &[i64],
    csr_indices: &[i32],
    csr_data: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    // Use the same scatter algorithm but row↔col swapped
    csc_to_csr(csr_indptr, csr_indices, csr_data, n_cols, n_rows).unwrap()
}

pub(crate) fn create_test_tenx_h5(path: &Path, n_cells: usize, n_genes: usize) {
    let file = hdf5::File::create(path).unwrap();
    let matrix = file.create_group("matrix").unwrap();

    // Shape: [n_genes, n_cells] (10x convention)
    let shape = [n_genes as i64, n_cells as i64];
    matrix
        .new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&shape)
        .unwrap();

    // Build CSC data (genes as major axis)
    let mut csc_indptr = vec![0i64];
    let mut csc_indices = Vec::new();
    let mut csc_data = Vec::new();

    for col in 0..n_cells {
        // Each cell has 2 nonzero gene counts
        let g0 = (col * 2) % n_genes;
        let g1 = (col * 2 + 1) % n_genes;
        let mut row_indices = vec![g0, g1];
        row_indices.sort();
        row_indices.dedup();
        for &g in &row_indices {
            csc_indices.push(g as i32);
            csc_data.push(((col + g + 1) % 100 + 1) as f32);
        }
        csc_indptr.push(csc_data.len() as i64);
    }

    matrix
        .new_dataset::<i64>()
        .shape([csc_indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&csc_indptr)
        .unwrap();
    matrix
        .new_dataset::<i32>()
        .shape([csc_indices.len()])
        .create("indices")
        .unwrap()
        .write(&csc_indices)
        .unwrap();
    matrix
        .new_dataset::<f32>()
        .shape([csc_data.len()])
        .create("data")
        .unwrap()
        .write(&csc_data)
        .unwrap();

    // Barcodes
    let barcodes: Vec<VarLenUnicode> = (0..n_cells).map(|i| vlu(&format!("AAAA-{i}"))).collect();
    matrix
        .new_dataset::<VarLenUnicode>()
        .shape([n_cells])
        .create("barcodes")
        .unwrap()
        .write(&barcodes)
        .unwrap();

    // Features
    let features = matrix.create_group("features").unwrap();
    let gene_ids: Vec<VarLenUnicode> = (0..n_genes).map(|i| vlu(&format!("ENSG{i:08}"))).collect();
    let gene_names: Vec<VarLenUnicode> = (0..n_genes).map(|i| vlu(&format!("Gene{i}"))).collect();
    let feature_types: Vec<VarLenUnicode> = (0..n_genes).map(|_| vlu("Gene Expression")).collect();

    features
        .new_dataset::<VarLenUnicode>()
        .shape([n_genes])
        .create("id")
        .unwrap()
        .write(&gene_ids)
        .unwrap();
    features
        .new_dataset::<VarLenUnicode>()
        .shape([n_genes])
        .create("name")
        .unwrap()
        .write(&gene_names)
        .unwrap();
    features
        .new_dataset::<VarLenUnicode>()
        .shape([n_genes])
        .create("feature_type")
        .unwrap()
        .write(&feature_types)
        .unwrap();
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

/// Append an `adata.raw` group (`raw/X` CSR + `raw/var`) to an existing
/// h5ad. `raw_n_vars` is the raw matrix's OWN (wider) gene axis. Returns
/// the canonical raw CSR `(indptr, indices, data)` for comparison.
pub(crate) fn add_raw_group(
    path: &Path,
    n_obs: usize,
    raw_n_vars: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let file = hdf5::File::open_rw(path).unwrap();

    let mut indptr = vec![0i64];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for row in 0..n_obs {
        let a = (row % raw_n_vars) as i32;
        let b = (raw_n_vars - 1 - (row % raw_n_vars)) as i32;
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if lo == hi {
            indices.push(lo);
            data.push((row + 1) as f32);
        } else {
            indices.push(lo);
            data.push((row + 1) as f32);
            indices.push(hi);
            data.push((row + 2) as f32);
        }
        indptr.push(data.len() as i64);
    }

    let raw = file.create_group("raw").unwrap();
    let rx = raw.create_group("X").unwrap();
    rx.new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&indptr)
        .unwrap();
    rx.new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")
        .unwrap()
        .write(&indices)
        .unwrap();
    rx.new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(&data)
        .unwrap();
    rx.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("csr_matrix"))
        .unwrap();
    rx.new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_obs as i64, raw_n_vars as i64])
        .unwrap();

    let rv = raw.create_group("var").unwrap();
    let rv_index: Vec<VarLenUnicode> = (0..raw_n_vars)
        .map(|i| vlu(&format!("raw_gene_{i}")))
        .collect();
    rv.new_dataset::<VarLenUnicode>()
        .shape([raw_n_vars])
        .create("_index")
        .unwrap()
        .write(&rv_index)
        .unwrap();
    rv.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    (indptr, indices, data)
}

// -----------------------------------------------------------------------
// Phase D: h5mu / MuData round-trip tests
// -----------------------------------------------------------------------

/// Create a minimal h5mu file with `n_obs` cells across two
/// modalities (rna with `rna_n_vars` features and adt with
/// `adt_n_vars` features). Mirrors `create_test_h5ad` for the per-
/// modality `/mod/{name}/X` and `/mod/{name}/var` blocks plus an
/// outer `/obs`.
pub(crate) fn create_test_h5mu(path: &Path, n_obs: usize, rna_n_vars: usize, adt_n_vars: usize) {
    let file = hdf5::File::create(path).unwrap();

    // Outer obs.
    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    // /mod group.
    let mod_group = file.create_group("mod").unwrap();

    let write_modality = |group: &hdf5::Group, n_vars: usize| {
        // Build trivial CSR data: one nonzero per row at column (row % n_vars).
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in 0..n_obs {
            indices.push((row % n_vars) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }

        let x = group.create_group("X").unwrap();
        x.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        x.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        x.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();

        // var
        let var = group.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> =
            (0..n_vars).map(|i| vlu(&format!("feat_{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_index)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    };

    let rna = mod_group.create_group("rna").unwrap();
    write_modality(&rna, rna_n_vars);
    let adt = mod_group.create_group("adt").unwrap();
    write_modality(&adt, adt_n_vars);
}

/// Like [`create_test_h5mu`] but writes a second modality
/// `dense_adt` whose `/X` is a 2D dense **f64** dataset with
/// `encoding-type="array"`. Exercises the dense-non-f32 sampling
/// path in `h5mu::pipeline::sample_modality_values`.
#[cfg(test)]
pub(crate) fn create_test_h5mu_with_dense_f64_modality(
    path: &Path,
    n_obs: usize,
    rna_n_vars: usize,
    dense_n_vars: usize,
) {
    let file = hdf5::File::create(path).unwrap();

    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    let mod_group = file.create_group("mod").unwrap();

    // Sparse `rna` modality (existing layout).
    let rna = mod_group.create_group("rna").unwrap();
    {
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in 0..n_obs {
            indices.push((row % rna_n_vars) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }
        let x = rna.create_group("X").unwrap();
        x.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        x.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        x.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, rna_n_vars as i64])
            .unwrap();

        let var = rna.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> =
            (0..rna_n_vars).map(|i| vlu(&format!("rna_{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([rna_n_vars])
            .create("_index")
            .unwrap()
            .write(&var_index)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    }

    // Dense f64 `dense_adt` modality.
    let dense = mod_group.create_group("dense_adt").unwrap();
    {
        let dense_values: Vec<f64> = (0..n_obs * dense_n_vars)
            .map(|i| if i % 3 == 0 { 0.0 } else { (i as f64) * 0.5 })
            .collect();
        let nd = ndarray::Array2::from_shape_vec((n_obs, dense_n_vars), dense_values).unwrap();
        let x_ds = dense
            .new_dataset::<f64>()
            .shape([n_obs, dense_n_vars])
            .create("X")
            .unwrap();
        x_ds.write(&nd).unwrap();
        x_ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("array"))
            .unwrap();

        let var = dense.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> = (0..dense_n_vars)
            .map(|i| vlu(&format!("adt_{i}")))
            .collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([dense_n_vars])
            .create("_index")
            .unwrap()
            .write(&var_index)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    }
}

// --- Phase F coverage ---------------------------------------------------

// -----------------------------------------------------------------------
// Phase 1 — streaming reader (XStreamReader / open_x_streaming /
// open_layer_streaming). Verifies the row-range reader yields the same
// data as the existing bulk `read_x_matrix` path and refuses
// unsupported on-disk layouts (CSC, dense).
// -----------------------------------------------------------------------

/// Drive `next_shard` to exhaustion, returning the concatenated
/// (indptr, indices, values) in their full-matrix layout. Re-bases
/// the per-shard local indptr to a global running total.
pub(crate) fn drain_streaming(
    reader: &mut crate::h5ad::stream::XStreamReader,
    target_rows: usize,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let n_obs = reader.n_obs;
    let mut full_indptr: Vec<u64> = Vec::with_capacity(n_obs + 1);
    full_indptr.push(0);
    let mut full_indices: Vec<u32> = Vec::new();
    let mut full_values: Vec<f32> = Vec::new();
    let mut row_count = 0usize;

    while let Some(shard) = reader.next_shard(target_rows) {
        let shard = shard.expect("shard read failed");
        assert_eq!(
            shard.row_start, row_count,
            "row_start must equal cumulative row count"
        );
        let base = *full_indptr.last().unwrap();
        for &v in &shard.indptr[1..] {
            full_indptr.push(base + v);
        }
        full_indices.extend_from_slice(&shard.indices);
        full_values.extend_from_slice(&shard.values);
        row_count += shard.n_rows;
    }
    assert_eq!(row_count, n_obs);
    (full_indptr, full_indices, full_values)
}

// -----------------------------------------------------------------------
// Phase 8 — end-to-end streaming pipeline (h5ad_to_scx_streaming).
//
// These differ from the Phase 1 streaming reader tests above: they
// exercise the full converter (open input → write obs/var → stream
// shards → write metadata → finish) and assert the resulting SCX
// file matches the materialising path. CSC sidecar parity and
// multi-layer round-trips are the load-bearing cases.
// -----------------------------------------------------------------------

pub(crate) fn streaming_opts(shard_size: u32) -> ConvertOptions {
    ConvertOptions {
        shard_target_rows: shard_size,
        codec: None,
        csc: super::pipeline::CscPolicy::Off,
        csc_cols_per_shard: 5000,
        tool: "scx".into(),
        ..ConvertOptions::default()
    }
}

// -----------------------------------------------------------------------
// Phase 1 — dense h5ad streaming + wild-h5ad hardening.
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// Phase 1 fixture helpers
// -----------------------------------------------------------------------

/// Write a 2D f32 dense `/X` with obs/var index but no extras.
#[cfg(test)]
pub(crate) fn write_dense_h5ad(path: &Path, n_obs: usize, n_vars: usize, dense: &[f32]) {
    assert_eq!(dense.len(), n_obs * n_vars);
    let file = hdf5::File::create(path).unwrap();
    let nd = ndarray::Array2::from_shape_vec((n_obs, n_vars), dense.to_vec()).unwrap();
    file.new_dataset::<f32>()
        .shape([n_obs, n_vars])
        .create("X")
        .unwrap()
        .write(&nd)
        .unwrap();

    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("gene_{i}"))).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("_index")
        .unwrap()
        .write(&var_index)
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
}

/// Write a CSC-encoded h5ad with hand-supplied indptr/indices/data
/// (no obs/var extras). Used by the unsorted-rows, duplicate, and
/// explicit-zero CSC tests.
#[cfg(test)]
pub(crate) fn write_csc_h5ad(
    path: &Path,
    n_obs: usize,
    n_vars: usize,
    col_indptr: &[i64],
    row_indices: &[i32],
    data: &[f32],
) {
    let file = hdf5::File::create(path).unwrap();
    let x = file.create_group("X").unwrap();
    x.new_dataset::<i64>()
        .shape([col_indptr.len()])
        .create("indptr")
        .unwrap()
        .write(col_indptr)
        .unwrap();
    x.new_dataset::<i32>()
        .shape([row_indices.len()])
        .create("indices")
        .unwrap()
        .write(row_indices)
        .unwrap();
    x.new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(data)
        .unwrap();
    x.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("csc_matrix"))
        .unwrap();
    x.new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_obs as i64, n_vars as i64])
        .unwrap();

    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("c_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g_{i}"))).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("_index")
        .unwrap()
        .write(&var_index)
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
}

// -----------------------------------------------------------------------
// Phase 2 — CSC-on-disk h5ad streaming.
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// Phase 3 — streaming h5mu / MuData conversion.
// -----------------------------------------------------------------------

/// Write a CSR `/X` group with `indptr`/`indices`/`data` children
/// AND a `shape` attribute, but deliberately omit `encoding-type`.
/// Used to test the inference path.
#[cfg(test)]
pub(crate) fn write_csr_h5ad_without_encoding_type(path: &Path, n_obs: usize, n_vars: usize) {
    let file = hdf5::File::create(path).unwrap();
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n_obs {
        let col = row % n_vars;
        indices.push(col as i32);
        data.push((row as f32) + 1.0);
        indptr.push(data.len() as i64);
    }
    let x = file.create_group("X").unwrap();
    x.new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&indptr)
        .unwrap();
    x.new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")
        .unwrap()
        .write(&indices)
        .unwrap();
    x.new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(&data)
        .unwrap();
    x.new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_obs as i64, n_vars as i64])
        .unwrap();
    // NB: NO `encoding-type` attr → triggers the inference path.

    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("gene_{i}"))).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("_index")
        .unwrap()
        .write(&var_index)
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
}

// -----------------------------------------------------------------------
// Phase 5a: conversion-time predicate indexes
// -----------------------------------------------------------------------

/// Build a minimal h5ad fixture that includes a low-cardinality
/// categorical obs column ("cell_type") and a low-cardinality
/// categorical var column ("feature_type") so we can exercise both
/// the obs and var force-index paths without relying on the existing
/// fixture's `_index` / `n_counts` columns.
pub(crate) fn create_test_h5ad_with_cell_type(path: &Path, n_obs: usize, n_vars: usize) {
    create_test_h5ad(path, n_obs, n_vars, "csr", false);
    let file = hdf5::File::append(path).unwrap();
    let obs = file.group("obs").unwrap();
    let labels = ["A", "B", "A", "B", "A"]; // small palette
    let col: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(labels[i % labels.len()])).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("cell_type")
        .unwrap()
        .write(&col)
        .unwrap();

    let var = file.group("var").unwrap();
    let feature_types = ["Gene Expression", "Antibody Capture"];
    let var_col: Vec<VarLenUnicode> = (0..n_vars)
        .map(|i| vlu(feature_types[i % feature_types.len()]))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("feature_type")
        .unwrap()
        .write(&var_col)
        .unwrap();
}

// -----------------------------------------------------------------------
// Phase 5b: detection bitmaps
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// Phase 8: streaming SCX → h5ad / h5mu
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// Parallel streaming reader.
//
// The parallel coordinator must produce byte-identical output to the
// sequential path. These tests run the same fixture through both and
// compare the resulting `.scx` files at the file-content level, plus
// exercise the memory-budget derate / refuse decisions and the
// per-worker error wrapping.
// -----------------------------------------------------------------------

/// Drain helper: read the entire output file into bytes for byte-equal
/// comparisons. Used by the byte-identity tests below.
pub(crate) fn read_file_bytes(p: &Path) -> Vec<u8> {
    std::fs::read(p).expect("read scx output")
}

// -----------------------------------------------------------------------
// Regression tests for the four follow-up defects to PR #104.
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// Phase 8d — parallel streaming reader on the SCX → h5ad/h5mu export
// path. Mirrors the ingest tests above: sequential vs parallel must
// produce identical output, deletion vectors must round-trip, the
// memory budget derate must fire, and oversized shards must refuse.
// -----------------------------------------------------------------------

/// Read /X/{indptr,indices,data} from an h5ad file at `path`. Helper
/// for the export-parallel byte-equality tests.
pub(crate) fn read_h5ad_x_triplet(path: &Path) -> (Vec<i64>, Vec<i32>, Vec<f32>, Vec<i64>) {
    let f = hdf5::File::open(path).unwrap();
    let g = f.group("X").unwrap();
    let indptr: Vec<i64> = g.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
    let indices: Vec<i32> = g.dataset("indices").unwrap().read_1d().unwrap().to_vec();
    let data: Vec<f32> = g.dataset("data").unwrap().read_1d().unwrap().to_vec();
    let shape: Vec<i64> = g.attr("shape").unwrap().read_1d().unwrap().to_vec();
    (indptr, indices, data, shape)
}

/// Convert h5ad → SCX with a small shard_size so the SCX file has
/// many shards (the export-parallel path needs ≥ a handful of shards
/// to exercise the rolling-window).
pub(crate) fn make_multishard_scx(scx_path: &Path, n_obs: usize, n_vars: usize, shard_size: u32) {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("src.h5ad");
    create_test_h5ad(&h5ad, n_obs, n_vars, "csr", false);
    let opts = ConvertOptions {
        shard_target_rows: shard_size,
        ..ConvertOptions::default()
    };
    h5ad_to_scx(&h5ad, scx_path, &opts, &mut WarningSink::log()).unwrap();
}

// -----------------------------------------------------------------------
// `write_dataframe_group_at` must honour the pandas `index_columns`
// schema metadata so that `pyscx.from_h5ad → pyscx.to_h5ad` preserves
// `var_names` / `obs_names` instead of silently swapping them with the
// first non-index column.
// -----------------------------------------------------------------------

pub(crate) fn build_var_batch_with_pandas_metadata(
    index_columns: &[&str],
    index_field_name: &str,
    index_values: &[&str],
) -> arrow::record_batch::RecordBatch {
    use arrow::array::StringArray;
    use arrow::datatypes::{Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut md = HashMap::new();
    md.insert(
        "pandas".to_string(),
        format!(
            "{{\"index_columns\":[{}]}}",
            index_columns
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("gene_ids", DataType::Utf8, false),
            Field::new(index_field_name, DataType::Utf8, false),
        ])
        .with_metadata(md),
    );
    let gene_ids = Arc::new(StringArray::from(vec!["ENSG1", "ENSG2"]));
    let symbols = Arc::new(StringArray::from(index_values.to_vec()));
    arrow::record_batch::RecordBatch::try_new(schema, vec![gene_ids, symbols]).unwrap()
}

// -----------------------------------------------------------------------
// Process_predicate_index_outcomes batches a fully-missing
// preset into a single PresetNoColumnsMatched warning.
// -----------------------------------------------------------------------

// E2-2026-05-20 `forced_column_missing_message` rendering tests live
// alongside the helper in `scx-engine/src/index.rs`. The helper moved
// out of `scx-convert` so `pyscx` (which only depends on `scx-engine`
// unconditionally; `scx-convert` is `hdf5`-gated) can call it from
// CPU-only builds. The previous duplicate tests here were removed in
// the CI fix-up for PR #113.

// -----------------------------------------------------------------------
// Process_predicate_index_outcomes aggregates
// multiple ForcedColumnError(MissingColumn) into a single error so the
// user can see ALL typos in a single run, instead of fixing one per
// invocation.
// -----------------------------------------------------------------------

// ---------------------------------------------------------------------
// Task 6a: streaming obs/var HDF5 hyperslab writes over sharded obs.
// ---------------------------------------------------------------------
