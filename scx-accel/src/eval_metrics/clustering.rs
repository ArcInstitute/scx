//! Clustering agreement metrics: AMI, NMI, ARI.
//!
//! Implements the standard information-theoretic and combinatorial clustering
//! metrics used by `cell-eval`'s `ClusteringAgreement` class. These operate
//! on two integer label vectors (e.g., cluster assignments) and quantify how
//! well they agree.
//!
//! All implementations follow the sklearn formulas:
//! - AMI: [`adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)
//! - NMI: [`normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
//! - ARI: [`adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
//!
//! ## cell-eval convention
//!
//! cell-eval rescales ARI from [-1, 1] to [0, 1] via `(ARI + 1) / 2`.
//! The [`adjusted_rand_index_rescaled`] function applies this rescaling.

use std::collections::HashMap;

use crate::error::AccelError;

/// Build a contingency table from two label vectors.
///
/// Returns a flat `[n_a × n_b]` count matrix (row-major), along with
/// the number of unique labels in each vector.
///
/// Labels are mapped to contiguous indices 0..n_unique for each vector
/// independently.
///
/// # Errors
///
/// Returns `AccelError::InvalidInput` if the two label vectors have
/// different lengths.
pub fn build_contingency_table(
    labels_a: &[u32],
    labels_b: &[u32],
) -> crate::Result<(Vec<u64>, u32, u32)> {
    if labels_a.len() != labels_b.len() {
        return Err(AccelError::InvalidInput(format!(
            "label vectors must have the same length: {} vs {}",
            labels_a.len(),
            labels_b.len()
        )));
    }
    let n = labels_a.len();

    // Map labels to contiguous indices. Uses HashMap::entry with `len()`
    // as a monotonic counter: the first unseen label gets index 0, the
    // second gets 1, etc.
    let mut map_a: HashMap<u32, u32> = HashMap::new();
    let mut map_b: HashMap<u32, u32> = HashMap::new();

    for i in 0..n {
        let na = map_a.len() as u32;
        map_a.entry(labels_a[i]).or_insert(na);
        let nb = map_b.len() as u32;
        map_b.entry(labels_b[i]).or_insert(nb);
    }

    let n_a = map_a.len() as u32;
    let n_b = map_b.len() as u32;

    // Build contingency matrix.
    let mut table = vec![0u64; (n_a as usize) * (n_b as usize)];

    for i in 0..n {
        let ri = map_a[&labels_a[i]];
        let ci = map_b[&labels_b[i]];
        table[ri as usize * n_b as usize + ci as usize] += 1;
    }

    Ok((table, n_a, n_b))
}

/// Compute row and column marginals from a contingency table.
fn marginals(table: &[u64], n_a: u32, n_b: u32) -> (Vec<u64>, Vec<u64>) {
    let nb = n_b as usize;
    let mut row_sums = vec![0u64; n_a as usize];
    let mut col_sums = vec![0u64; nb];

    for i in 0..n_a as usize {
        for j in 0..nb {
            let v = table[i * nb + j];
            row_sums[i] += v;
            col_sums[j] += v;
        }
    }

    (row_sums, col_sums)
}

/// Compute Shannon entropy from a count vector: H = -sum(p * log(p)) where p = count/total.
#[inline]
fn entropy(counts: &[u64], total: f64) -> f64 {
    if total == 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.ln();
        }
    }
    h
}

/// Compute mutual information between two clusterings.
///
/// MI(U, V) = sum_{i,j} (n_ij / N) * ln(N * n_ij / (a_i * b_j))
fn mutual_information(table: &[u64], n_a: u32, n_b: u32, n: f64) -> f64 {
    let (row_sums, col_sums) = marginals(table, n_a, n_b);
    let nb = n_b as usize;

    let mut mi = 0.0;
    for i in 0..n_a as usize {
        for j in 0..nb {
            let nij = table[i * nb + j];
            if nij > 0 && row_sums[i] > 0 && col_sums[j] > 0 {
                mi += (nij as f64 / n)
                    * (n * nij as f64 / (row_sums[i] as f64 * col_sums[j] as f64)).ln();
            }
        }
    }
    mi
}

/// Compute expected mutual information under the hypergeometric model.
///
/// Uses the exact formula from Vinh, Epps, Bailey (2010):
/// E[MI] = sum_{i,j} sum_{n_ij=max(1, a_i+b_j-N)}^{min(a_i,b_j)}
///         (n_ij / N) * ln(N * n_ij / (a_i * b_j)) *
///         (a_i! * b_j! * (N-a_i)! * (N-b_j)!) / (N! * n_ij! * (a_i-n_ij)! * (b_j-n_ij)! * (N-a_i-b_j+n_ij)!)
///
/// For computational efficiency, we use log-factorials to avoid overflow.
///
/// ## Complexity
///
/// O(K_a × K_b × max_range) where K_a and K_b are the number of unique
/// labels in each vector and max_range = max(min(a_i, b_j) - max(1, a_i+b_j-N)).
/// For centroid-level labels this is typically 50–200, but for large label
/// arrays with many clusters (K > 1000) this can become expensive. The
/// log-factorial table is allocated as O(N).
fn expected_mutual_information(
    n_a: u32,
    n_b: u32,
    row_sums: &[u64],
    col_sums: &[u64],
    n: u64,
) -> f64 {
    if n == 0 {
        return 0.0;
    }

    let nf = n as f64;

    // Precompute log-factorials up to `N`. In a standard contingency table
    // `sum(row_sums) == sum(col_sums) == N`, so `N` is the tight upper bound
    // on every intermediate index (`log_fact[ai]`, `log_fact[bj]`,
    // `log_fact[ai - nij]`, `log_fact[n - ai - bj + nij]`, …). A smaller
    // table keyed on `max(sum(row_sums), sum(col_sums))` would save memory
    // only for malformed confusion matrices; well-formed input already
    // reaches the full `N`.
    let max_val = n as usize + 1;
    let mut log_fact = vec![0.0f64; max_val + 1];
    for i in 2..=max_val {
        log_fact[i] = log_fact[i - 1] + (i as f64).ln();
    }

    let log_n_fact = log_fact[n as usize];

    let mut emi = 0.0;

    for &ai in row_sums.iter().take(n_a as usize) {
        if ai == 0 {
            continue;
        }
        for &bj in col_sums.iter().take(n_b as usize) {
            if bj == 0 {
                continue;
            }

            // Range of valid n_ij values under hypergeometric model.
            let nij_min = (ai + bj).saturating_sub(n).max(1);
            let nij_max = ai.min(bj);

            if nij_min > nij_max {
                continue;
            }

            // Precompute the log of the "outer" hypergeometric term:
            // log(a_i! * b_j! * (N-a_i)! * (N-b_j)!) - log(N!)
            let log_outer = log_fact[ai as usize]
                + log_fact[bj as usize]
                + log_fact[(n - ai) as usize]
                + log_fact[(n - bj) as usize]
                - log_n_fact;

            for nij in nij_min..=nij_max {
                // Term: (nij / N) * ln(N * nij / (ai * bj))
                let log_term = (nf * nij as f64 / (ai as f64 * bj as f64)).ln();
                let term_val = (nij as f64 / nf) * log_term;

                // Hypergeometric probability:
                // log(P) = log_outer - log(nij! * (ai-nij)! * (bj-nij)! * (N-ai-bj+nij)!)
                let n_ai_bj_nij = n as i64 - ai as i64 - bj as i64 + nij as i64;
                if n_ai_bj_nij < 0 {
                    continue;
                }
                let log_inner = log_fact[nij as usize]
                    + log_fact[(ai - nij) as usize]
                    + log_fact[(bj - nij) as usize]
                    + log_fact[n_ai_bj_nij as usize];

                let log_prob = log_outer - log_inner;
                emi += term_val * log_prob.exp();
            }
        }
    }

    emi
}

/// Adjusted Mutual Information (AMI).
///
/// AMI = (MI - E[MI]) / (mean(H_a, H_b) - E[MI])
///
/// Where MI is mutual information, E[MI] is expected MI under chance,
/// and H_a, H_b are the entropies of the two label vectors.
///
/// Matches sklearn's `adjusted_mutual_info_score()` with the default
/// `average_method='arithmetic'`, i.e. the normalizer is the arithmetic
/// mean of the two entropies: `(H_a + H_b) / 2`.
pub fn adjusted_mutual_info(labels_a: &[u32], labels_b: &[u32]) -> f64 {
    let n = labels_a.len();
    if n == 0 {
        return 0.0;
    }

    // build_contingency_table cannot fail here because both slices have the
    // same length (they come from the same caller). unwrap is safe.
    let (table, n_a, n_b) =
        build_contingency_table(labels_a, labels_b).expect("AMI: label vectors have same length");
    let (row_sums, col_sums) = marginals(&table, n_a, n_b);
    let nf = n as f64;

    let mi = mutual_information(&table, n_a, n_b, nf);
    let h_a = entropy(&row_sums, nf);
    let h_b = entropy(&col_sums, nf);

    // sklearn default: arithmetic mean for AMI normalization
    let mean_h = (h_a + h_b) / 2.0;

    // Edge case: if both entropies are zero, AMI is defined as 1.0
    // (both clusterings have a single cluster, which trivially agree).
    if mean_h == 0.0 {
        return 1.0;
    }

    let emi = expected_mutual_information(n_a, n_b, &row_sums, &col_sums, n as u64);
    let denominator = mean_h - emi;

    if denominator.abs() < 1e-15 {
        // When denominator is ~0, AMI is 1.0 if MI == EMI, otherwise 0.0.
        if (mi - emi).abs() < 1e-15 {
            return 1.0;
        }
        return 0.0;
    }

    (mi - emi) / denominator
}

/// Normalized Mutual Information (NMI).
///
/// NMI = 2 * MI / (H_a + H_b)
///
/// Matches sklearn's `normalized_mutual_info_score()` with `average_method='arithmetic'`.
pub fn normalized_mutual_info(labels_a: &[u32], labels_b: &[u32]) -> f64 {
    let n = labels_a.len();
    if n == 0 {
        return 0.0;
    }

    let (table, n_a, n_b) =
        build_contingency_table(labels_a, labels_b).expect("NMI: label vectors have same length");
    let (row_sums, col_sums) = marginals(&table, n_a, n_b);
    let nf = n as f64;

    let mi = mutual_information(&table, n_a, n_b, nf);
    let h_a = entropy(&row_sums, nf);
    let h_b = entropy(&col_sums, nf);

    // Arithmetic mean normalization (sklearn default)
    let mean_h = (h_a + h_b) / 2.0;

    if mean_h == 0.0 {
        return 1.0; // Both single-cluster: perfect agreement
    }

    // NMI = MI / mean(H_a, H_b)
    // Note: sklearn normalizes as 2*MI/(H_a+H_b) which equals MI/mean(H_a,H_b)
    mi / mean_h
}

/// Adjusted Rand Index (ARI).
///
/// ARI = (sum_ij C(n_ij,2) - [sum_i C(a_i,2) * sum_j C(b_j,2)] / C(n,2))
///       / (0.5 * [sum_i C(a_i,2) + sum_j C(b_j,2)] - [sum_i C(a_i,2) * sum_j C(b_j,2)] / C(n,2))
///
/// Matches sklearn's `adjusted_rand_score()`.
pub fn adjusted_rand_index(labels_a: &[u32], labels_b: &[u32]) -> f64 {
    let n = labels_a.len();
    if n == 0 {
        return 0.0;
    }

    let (table, n_a, n_b) =
        build_contingency_table(labels_a, labels_b).expect("ARI: label vectors have same length");
    let (row_sums, col_sums) = marginals(&table, n_a, n_b);
    let nb = n_b as usize;

    // C(k, 2) = k * (k - 1) / 2
    let comb2 = |k: u64| -> i64 { (k as i64) * (k as i64 - 1) / 2 };

    // Sum of C(n_ij, 2) over all pairs
    let mut sum_comb_nij: i64 = 0;
    for i in 0..n_a as usize {
        for j in 0..nb {
            sum_comb_nij += comb2(table[i * nb + j]);
        }
    }

    // Sum of C(a_i, 2) and C(b_j, 2)
    let sum_comb_a: i64 = row_sums.iter().map(|&a| comb2(a)).sum();
    let sum_comb_b: i64 = col_sums.iter().map(|&b| comb2(b)).sum();

    let comb_n = comb2(n as u64);

    if comb_n == 0 {
        // Only 0 or 1 observations
        return 0.0;
    }

    // Expected index: E = sum_comb_a * sum_comb_b / comb_n
    let expected = (sum_comb_a as f64 * sum_comb_b as f64) / comb_n as f64;

    // Maximum index: max_index = 0.5 * (sum_comb_a + sum_comb_b)
    let max_index = 0.5 * (sum_comb_a + sum_comb_b) as f64;

    let denominator = max_index - expected;

    if denominator.abs() < 1e-15 {
        // Edge case: all labels in one cluster or each point in its own cluster.
        return if sum_comb_nij as f64 == expected {
            1.0
        } else {
            0.0
        };
    }

    (sum_comb_nij as f64 - expected) / denominator
}

/// ARI rescaled to [0, 1] following cell-eval convention: (ARI + 1) / 2.
pub fn adjusted_rand_index_rescaled(labels_a: &[u32], labels_b: &[u32]) -> f64 {
    (adjusted_rand_index(labels_a, labels_b) + 1.0) / 2.0
}

/// Clustering agreement metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusteringMetric {
    /// Adjusted Mutual Information (sklearn default: arithmetic mean).
    Ami,
    /// Normalized Mutual Information (sklearn default: arithmetic mean).
    Nmi,
    /// Adjusted Rand Index, rescaled to [0, 1] via (ARI + 1) / 2.
    Ari,
}

impl ClusteringMetric {
    /// Parse a metric name string.
    pub fn parse(name: &str) -> Option<ClusteringMetric> {
        match name.to_lowercase().as_str() {
            "ami" => Some(ClusteringMetric::Ami),
            "nmi" => Some(ClusteringMetric::Nmi),
            "ari" => Some(ClusteringMetric::Ari),
            _ => None,
        }
    }

    /// Compute the metric between two label vectors.
    pub fn score(&self, labels_a: &[u32], labels_b: &[u32]) -> f64 {
        match self {
            ClusteringMetric::Ami => adjusted_mutual_info(labels_a, labels_b),
            ClusteringMetric::Nmi => normalized_mutual_info(labels_a, labels_b),
            ClusteringMetric::Ari => adjusted_rand_index_rescaled(labels_a, labels_b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_agreement() {
        let labels = vec![0, 0, 1, 1, 2, 2];
        let ami = adjusted_mutual_info(&labels, &labels);
        let nmi = normalized_mutual_info(&labels, &labels);
        let ari = adjusted_rand_index(&labels, &labels);

        assert!(
            (ami - 1.0).abs() < 1e-10,
            "AMI for perfect agreement should be 1.0, got {ami}"
        );
        assert!(
            (nmi - 1.0).abs() < 1e-10,
            "NMI for perfect agreement should be 1.0, got {nmi}"
        );
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "ARI for perfect agreement should be 1.0, got {ari}"
        );
    }

    #[test]
    fn test_single_cluster_each() {
        // Both assigning all points to one cluster.
        let a = vec![0, 0, 0, 0];
        let b = vec![1, 1, 1, 1]; // different label value, but same partition

        let ami = adjusted_mutual_info(&a, &b);
        let nmi = normalized_mutual_info(&a, &b);
        let ari = adjusted_rand_index(&a, &b);

        // Single cluster on both sides: trivially perfect agreement.
        assert!(
            (ami - 1.0).abs() < 1e-10,
            "AMI for single-cluster should be 1.0, got {ami}"
        );
        assert!(
            (nmi - 1.0).abs() < 1e-10,
            "NMI for single-cluster should be 1.0, got {nmi}"
        );
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "ARI for single-cluster should be 1.0, got {ari}"
        );
    }

    #[test]
    fn test_known_values_ami() {
        // Known test case: labels_true = [0,0,0,1,1,1], labels_pred = [0,0,1,1,2,2]
        // Exact sklearn value: adjusted_mutual_info_score(a, b) = 0.298792458170890
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 1, 1, 2, 2];

        let ami = adjusted_mutual_info(&a, &b);
        assert!(
            (ami - 0.298792458170890).abs() < 1e-10,
            "AMI should be 0.298792458170890, got {ami}"
        );
    }

    #[test]
    fn test_known_values_nmi() {
        // Exact sklearn value: normalized_mutual_info_score(a, b) = 0.515803742979389
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 1, 1, 2, 2];

        let nmi = normalized_mutual_info(&a, &b);
        assert!(
            (nmi - 0.515803742979389).abs() < 1e-10,
            "NMI should be 0.515803742979389, got {nmi}"
        );
    }

    #[test]
    fn test_known_values_ari() {
        // Exact sklearn value: adjusted_rand_score(a, b) = 0.242424242424242
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 1, 1, 2, 2];

        let ari = adjusted_rand_index(&a, &b);
        assert!(
            (ari - 0.242424242424242).abs() < 1e-10,
            "ARI should be 0.242424242424242, got {ari}"
        );
    }

    #[test]
    fn test_ari_rescaled() {
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 1, 1, 2, 2];

        let ari = adjusted_rand_index(&a, &b);
        let rescaled = adjusted_rand_index_rescaled(&a, &b);

        assert!(
            (rescaled - (ari + 1.0) / 2.0).abs() < 1e-15,
            "Rescaled ARI should be (ARI + 1) / 2"
        );
    }

    #[test]
    fn test_empty_labels() {
        let empty: Vec<u32> = vec![];
        assert_eq!(adjusted_mutual_info(&empty, &empty), 0.0);
        assert_eq!(normalized_mutual_info(&empty, &empty), 0.0);
        assert_eq!(adjusted_rand_index(&empty, &empty), 0.0);
    }

    #[test]
    fn test_two_elements() {
        // Minimal case: two elements
        let a = vec![0, 1];
        let b = vec![0, 1];
        let ari = adjusted_rand_index(&a, &b);
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "ARI for identical 2-element labels should be 1.0, got {ari}"
        );
    }

    #[test]
    fn test_clustering_metric_enum() {
        assert_eq!(ClusteringMetric::parse("ami"), Some(ClusteringMetric::Ami));
        assert_eq!(ClusteringMetric::parse("NMI"), Some(ClusteringMetric::Nmi));
        assert_eq!(ClusteringMetric::parse("ARI"), Some(ClusteringMetric::Ari));
        assert_eq!(ClusteringMetric::parse("unknown"), None);

        let a = vec![0, 0, 1, 1];
        let b = vec![0, 0, 1, 1];
        assert!((ClusteringMetric::Ami.score(&a, &b) - 1.0).abs() < 1e-10);
        assert!((ClusteringMetric::Nmi.score(&a, &b) - 1.0).abs() < 1e-10);
        // ARI rescaled: (1.0 + 1) / 2 = 1.0
        assert!((ClusteringMetric::Ari.score(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_larger_dataset() {
        // 20 elements, 4 clusters in 'a', 3 clusters in 'b'
        let a: Vec<u32> = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];
        let b: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 2, 2, 2, 1, 1, 2, 2, 2];

        let ami = adjusted_mutual_info(&a, &b);
        let nmi = normalized_mutual_info(&a, &b);
        let ari = adjusted_rand_index(&a, &b);

        // Just check they produce reasonable values
        assert!(ami > -0.5 && ami < 1.0, "AMI out of range: {ami}");
        assert!(nmi >= 0.0 && nmi <= 1.0, "NMI out of range: {nmi}");
        assert!(ari >= -1.0 && ari <= 1.0, "ARI out of range: {ari}");
    }

    #[test]
    fn test_contingency_table() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![0, 1, 0, 1, 0, 1];

        let (table, n_a, n_b) = build_contingency_table(&a, &b).unwrap();
        assert_eq!(n_a, 3);
        assert_eq!(n_b, 2);

        // Each (i, j) combination has exactly 1 element
        assert_eq!(table.len(), 6);
        for &v in &table {
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn test_contingency_table_length_mismatch() {
        let a = vec![0, 0, 1];
        let b = vec![0, 1];

        let result = build_contingency_table(&a, &b);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("same length"),
            "Error should mention length: {err_msg}"
        );
    }
}
