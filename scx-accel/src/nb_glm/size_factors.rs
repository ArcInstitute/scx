//! DESeq2-style median-ratio size factors (spec §7.1).
//!
//! Computed once as a host-side pre-pass; cheap relative to fitting. Operates on
//! the gene-major `[n_genes × n_samples]` count buffer.

/// Compute median-ratio size factors for pseudobulk counts.
///
/// 1. Per gene, the geometric mean across samples over strictly-positive counts.
/// 2. Per sample, the ratios `count / geo_mean` over genes with a positive geo
///    mean and a positive count.
/// 3. Size factor = median ratio per sample.
/// 4. Normalize so the size factors have geometric mean 1.
///
/// Falls back to library-size factors (per-sample total, normalized to geometric
/// mean 1) when too few genes have a valid geometric mean to be reliable.
///
/// `counts_gene_major[g * n_samples + s]`. Returns a length-`n_samples` vector of
/// strictly-positive factors.
pub fn median_ratio_size_factors(
    counts_gene_major: &[f64],
    n_genes: usize,
    n_samples: usize,
) -> Vec<f64> {
    debug_assert_eq!(counts_gene_major.len(), n_genes * n_samples);

    // Per-gene geometric mean over strictly-positive counts. A gene with any
    // zero count is excluded (the classic DESeq2 reference-gene rule).
    let mut gene_geo_mean = vec![0.0_f64; n_genes];
    let mut n_valid_genes = 0usize;
    for g in 0..n_genes {
        let row = &counts_gene_major[g * n_samples..(g + 1) * n_samples];
        let mut log_sum = 0.0;
        let mut all_positive = true;
        for &c in row {
            if c > 0.0 {
                log_sum += c.ln();
            } else {
                all_positive = false;
                break;
            }
        }
        if all_positive {
            gene_geo_mean[g] = (log_sum / n_samples as f64).exp();
            n_valid_genes += 1;
        }
    }

    // Too few reference genes → library-size fallback.
    if n_valid_genes < n_samples.max(1) {
        return library_size_factors(counts_gene_major, n_genes, n_samples);
    }

    // Per-sample median of count / geo_mean over valid genes.
    let mut factors = vec![1.0_f64; n_samples];
    let mut ratios = Vec::with_capacity(n_valid_genes);
    for s in 0..n_samples {
        ratios.clear();
        for g in 0..n_genes {
            let gm = gene_geo_mean[g];
            if gm > 0.0 {
                let c = counts_gene_major[g * n_samples + s];
                if c > 0.0 {
                    ratios.push(c / gm);
                }
            }
        }
        factors[s] = if ratios.is_empty() {
            1.0
        } else {
            median(&mut ratios)
        };
    }

    normalize_to_geometric_mean_one(&mut factors);
    factors
}

/// Library-size factors: per-sample column totals normalized to geometric mean 1.
fn library_size_factors(counts_gene_major: &[f64], n_genes: usize, n_samples: usize) -> Vec<f64> {
    let mut factors = vec![0.0_f64; n_samples];
    for g in 0..n_genes {
        let row = &counts_gene_major[g * n_samples..(g + 1) * n_samples];
        for (s, &c) in row.iter().enumerate() {
            factors[s] += c;
        }
    }
    // Guard against all-zero samples so factors stay strictly positive.
    for f in factors.iter_mut() {
        if *f <= 0.0 {
            *f = 1.0;
        }
    }
    normalize_to_geometric_mean_one(&mut factors);
    factors
}

/// Scale a vector of positive values so their geometric mean is exactly 1.
fn normalize_to_geometric_mean_one(factors: &mut [f64]) {
    if factors.is_empty() {
        return;
    }
    let log_mean: f64 = factors.iter().map(|f| f.ln()).sum::<f64>() / factors.len() as f64;
    let geo_mean = log_mean.exp();
    if geo_mean > 0.0 && geo_mean.is_finite() {
        for f in factors.iter_mut() {
            *f /= geo_mean;
        }
    }
}

/// Median of a slice (sorts in place; caller-owned scratch buffer).
fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n == 0 {
        return 1.0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}
