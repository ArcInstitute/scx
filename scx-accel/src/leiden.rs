//! Leiden community detection algorithm for network community detection.
//!
//! Provides a Rust-native implementation of the Leiden algorithm
//! (Traag, Waltman & van Eck, 2019) with the Reichardt-Bornholdt (RB) configuration
//! model quality function and conflict-free parallel batching via rayon.
//!
//! Core algorithm adapted from single-clustering
//! (BSD 3-Clause License, Copyright 2025 Ian F. Diks)
//! <https://github.com/SingleRust/single-clustering>
//!
//! Simplified for SCX: concrete `f64` types, single-partition single-layer,
//! RB configuration model only, `AllNeighComms` strategy.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::error::{AccelError, Result};

// ─── Public Types ─────────────────────────────────────────────────────

/// Result of Leiden community detection.
pub struct LeidenResult {
    /// Community label for each node (0-indexed, contiguous).
    pub membership: Vec<usize>,
    /// Partition quality (RB modularity).
    pub modularity: f64,
    /// Number of communities found.
    pub n_communities: usize,
}

/// Configuration for the Leiden algorithm.
#[derive(Debug, Clone)]
pub struct LeidenConfig {
    /// Maximum outer-loop iterations (default 100).
    pub max_iterations: usize,
    /// Convergence tolerance — unused in current impl but reserved (default 1e-6).
    pub tolerance: f64,
    /// Random seed for reproducibility. `None` for OS entropy.
    pub seed: Option<u64>,
    /// Resolution parameter γ — higher values yield more communities (default 1.0).
    pub resolution: f64,
    /// Whether to refine partition for well-connected communities (default true).
    pub refine_partition: bool,
    /// Whether to consider moving nodes to empty communities (default true).
    pub consider_empty_community: bool,
    /// Use parallel (conflict-free batched) local moving instead of sequential.
    /// Sequential (default, `false`) matches C++ leidenalg exactly.
    /// Parallel (`true`) is faster on large graphs but converges to a
    /// different local optimum due to stale-read approximation.
    pub parallel: bool,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            seed: Some(42),
            resolution: 1.0,
            refine_partition: true,
            consider_empty_community: true,
            parallel: false,
        }
    }
}

// ─── LeidenGraph ──────────────────────────────────────────────────────

/// Internal CSR storage for the Leiden graph.
#[allow(dead_code)]
struct LeidenGraphData {
    node_ptrs: Vec<usize>,
    neighbors: Vec<usize>,
    weights: Vec<f64>,
    node_weights: Vec<f64>,
    degrees: Vec<usize>,
    strengths: Vec<f64>,
    total_weight: f64,
    edge_count: usize,
}

/// Arc-wrapped CSR graph — cheap to clone (reference counted).
#[derive(Clone)]
struct LeidenGraph {
    data: Arc<LeidenGraphData>,
}

/// Zero-cost iterator over (neighbor, weight) pairs using pointer arithmetic.
struct NeighborIterator {
    neighbor_ptr: *const usize,
    weight_ptr: *const f64,
    remaining: usize,
}

impl Iterator for NeighborIterator {
    type Item = (usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        unsafe {
            let neighbor = *self.neighbor_ptr;
            let weight = *self.weight_ptr;
            self.neighbor_ptr = self.neighbor_ptr.add(1);
            self.weight_ptr = self.weight_ptr.add(1);
            self.remaining -= 1;
            Some((neighbor, weight))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for NeighborIterator {}

// Safety: the pointers point into an Arc-owned Vec that outlives the iterator.
unsafe impl Send for NeighborIterator {}
unsafe impl Sync for NeighborIterator {}

impl LeidenGraph {
    /// Build a Leiden graph directly from an SCX-format symmetric CSR matrix.
    ///
    /// `indptr` (length n_nodes+1, i64), `indices` (i32), `data` (f64) describe
    /// the full symmetric adjacency — each undirected edge appears twice.
    pub fn from_csr(indptr: &[i64], indices: &[i32], data: &[f64], n_nodes: usize) -> Self {
        let nnz = indices.len();

        // Convert types
        let node_ptrs: Vec<usize> = indptr.iter().map(|&p| p as usize).collect();
        let mut neighbors: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        let mut weights: Vec<f64> = data.to_vec();

        // Ensure neighbors are sorted within each row (required for binary search).
        for node in 0..n_nodes {
            let start = node_ptrs[node];
            let end = node_ptrs[node + 1];
            if end - start > 1 {
                let sorted = neighbors[start..end].windows(2).all(|w| w[0] <= w[1]);
                if !sorted {
                    let mut pairs: Vec<(usize, f64)> =
                        (start..end).map(|i| (neighbors[i], weights[i])).collect();
                    pairs.sort_unstable_by_key(|&(n, _)| n);
                    for (i, &(n, w)) in pairs.iter().enumerate() {
                        neighbors[start + i] = n;
                        weights[start + i] = w;
                    }
                }
            }
        }

        // Compute derived values.
        let mut degrees = Vec::with_capacity(n_nodes);
        let mut strengths = vec![0.0f64; n_nodes];
        let node_weights = vec![1.0f64; n_nodes];
        let mut total_weight = 0.0f64;

        for node in 0..n_nodes {
            let start = node_ptrs[node];
            let end = node_ptrs[node + 1];
            degrees.push(end - start);
            for i in start..end {
                strengths[node] += weights[i];
                // Self-loops contribute twice to strength in igraph's convention
                // (GraphHelper.cpp:418-419: both _strength_in[to] and
                // _strength_in[from] are incremented, which is the same node).
                if neighbors[i] == node {
                    strengths[node] += weights[i];
                }
                // Count each undirected edge once (upper triangle).
                if node <= neighbors[i] {
                    total_weight += weights[i];
                }
            }
        }

        Self {
            data: Arc::new(LeidenGraphData {
                node_ptrs,
                neighbors,
                weights,
                node_weights,
                degrees,
                strengths,
                total_weight,
                edge_count: nnz / 2,
            }),
        }
    }

    /// Build from an edge list — used by `aggregate`.
    ///
    /// Each `(from, to, weight)` edge is stored in both directions unless `from == to`
    /// (self-loop stored once).
    fn from_edges(edges: &[(usize, usize, f64)], node_weights: Vec<f64>) -> Self {
        let num_nodes = node_weights.len();

        // Count degrees.
        let mut degrees = vec![0usize; num_nodes];
        for &(from, to, _) in edges {
            degrees[from] += 1;
            if from != to {
                degrees[to] += 1;
            }
        }

        // Build adjacency lists.
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_nodes];
        for &(from, to, w) in edges {
            adj[from].push((to, w));
            if from != to {
                adj[to].push((from, w));
            }
        }

        // Flatten into CSR.
        let total_degree: usize = degrees.iter().sum();
        let mut node_ptrs = Vec::with_capacity(num_nodes + 1);
        let mut neighbors = Vec::with_capacity(total_degree);
        let mut weights = Vec::with_capacity(total_degree);
        let mut strengths = vec![0.0f64; num_nodes];
        let mut total_weight = 0.0f64;

        node_ptrs.push(0);
        for (node, adj_list) in adj.into_iter().enumerate() {
            let mut sorted = adj_list;
            sorted.sort_unstable_by_key(|&(n, _)| n);
            for (neighbor, w) in sorted {
                neighbors.push(neighbor);
                weights.push(w);
                strengths[node] += w;
                // Self-loops contribute twice to strength (igraph convention).
                if neighbor == node {
                    strengths[node] += w;
                }
                if node <= neighbor {
                    total_weight += w;
                }
            }
            node_ptrs.push(neighbors.len());
        }

        Self {
            data: Arc::new(LeidenGraphData {
                node_ptrs,
                neighbors,
                weights,
                node_weights,
                degrees,
                strengths,
                total_weight,
                edge_count: edges.len(),
            }),
        }
    }

    #[inline]
    fn neighbors(&self, node: usize) -> NeighborIterator {
        let start = self.data.node_ptrs[node];
        let end = self.data.node_ptrs[node + 1];
        NeighborIterator {
            neighbor_ptr: unsafe { self.data.neighbors.as_ptr().add(start) },
            weight_ptr: unsafe { self.data.weights.as_ptr().add(start) },
            remaining: end - start,
        }
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.data.node_weights.len()
    }

    #[inline]
    fn strength(&self, node: usize) -> f64 {
        self.data.strengths[node]
    }

    #[inline]
    fn total_weight(&self) -> f64 {
        self.data.total_weight
    }

    #[inline]
    fn self_loop_weight(&self, node: usize) -> f64 {
        let start = self.data.node_ptrs[node];
        let end = self.data.node_ptrs[node + 1];
        if start >= end {
            return 0.0;
        }
        // Fast check for common case (self-loop at front).
        if self.data.neighbors[start] == node {
            return self.data.weights[start];
        }
        // Binary search since neighbors are sorted.
        match self.data.neighbors[start..end].binary_search(&node) {
            Ok(pos) => self.data.weights[start + pos],
            Err(_) => 0.0,
        }
    }

    /// Create an aggregated graph where groups become super-nodes.
    fn aggregate(&self, grouping: &Grouping) -> Self {
        let new_node_count = grouping.group_count();

        // Accumulate node weights per group.
        let mut new_node_weights = vec![0.0f64; new_node_count];
        for node in 0..self.node_count() {
            new_node_weights[grouping.get_group(node)] += self.data.node_weights[node];
        }

        // Accumulate edge weights: self-loops (intra-group) and cross-group edges.
        let mut self_loop_weights: HashMap<usize, f64> = HashMap::new();
        let mut edge_memo: HashMap<(usize, usize), f64> = HashMap::new();

        for node in 0..self.node_count() {
            let start = self.data.node_ptrs[node];
            let end = self.data.node_ptrs[node + 1];
            for i in start..end {
                let neighbor = self.data.neighbors[i];
                let w = self.data.weights[i];
                // Upper triangle only to avoid double-counting.
                if node <= neighbor {
                    let g1 = grouping.get_group(node);
                    let g2 = grouping.get_group(neighbor);
                    if g1 == g2 {
                        *self_loop_weights.entry(g1).or_insert(0.0) += w;
                    } else {
                        let key = if g1 < g2 { (g1, g2) } else { (g2, g1) };
                        *edge_memo.entry(key).or_insert(0.0) += w;
                    }
                }
            }
        }

        let mut edges = Vec::with_capacity(self_loop_weights.len() + edge_memo.len());
        for (&group, &w) in &self_loop_weights {
            if w > 0.0 {
                edges.push((group, group, w));
            }
        }
        for (&(g1, g2), &w) in &edge_memo {
            edges.push((g1, g2, w));
        }

        Self::from_edges(&edges, new_node_weights)
    }
}

// ─── Grouping ─────────────────────────────────────────────────────────

/// Community membership tracking with eagerly-maintained group sizes
/// and a reuse pool of empty community IDs (matching C++ libleidenalg's
/// `_empty_communities` vector in MutableVertexPartition).
#[derive(Clone)]
struct Grouping {
    assignments: Vec<usize>,
    group_count: usize,
    group_sizes: Vec<usize>,
    /// Stack of community IDs that have zero members and can be reused.
    /// Maintained by `set_group()`: pushed when a community becomes empty,
    /// removed when a community gains its first member.
    empty_communities: Vec<usize>,
}

impl Grouping {
    /// Each node in its own group: 0, 1, 2, …, n-1.
    fn create_isolated(n: usize) -> Self {
        Self {
            assignments: (0..n).collect(),
            group_count: n,
            group_sizes: vec![1; n],
            empty_communities: Vec::new(),
        }
    }

    /// Build from an explicit assignment array.
    fn from_assignments(input: &[usize]) -> Self {
        let max_group = input.iter().copied().max().unwrap_or(0);
        let group_count = max_group + 1;
        let mut group_sizes = vec![0usize; group_count];
        for &g in input {
            group_sizes[g] += 1;
        }
        // Collect initially-empty groups.
        let empty_communities: Vec<usize> =
            (0..group_count).filter(|&g| group_sizes[g] == 0).collect();
        let mut grouping = Self {
            assignments: input.to_vec(),
            group_count,
            group_sizes,
            empty_communities,
        };
        grouping.normalize_groups();
        grouping
    }

    #[inline]
    fn get_group(&self, node: usize) -> usize {
        self.assignments[node]
    }

    #[inline]
    fn set_group(&mut self, node: usize, group: usize) {
        let old = self.assignments[node];
        if old == group {
            return;
        }
        self.group_sizes[old] -= 1;
        // Old community became empty — add to reuse pool.
        if self.group_sizes[old] == 0 {
            self.empty_communities.push(old);
        }
        if group >= self.group_sizes.len() {
            self.group_sizes.resize(group + 1, 0);
        }
        // Target community was empty — remove from reuse pool.
        if self.group_sizes[group] == 0 {
            if let Some(pos) = self.empty_communities.iter().rposition(|&g| g == group) {
                self.empty_communities.swap_remove(pos);
            }
        }
        self.group_sizes[group] += 1;
        self.assignments[node] = group;
        if group >= self.group_count {
            self.group_count = group + 1;
        }
    }

    #[inline]
    fn group_count(&self) -> usize {
        self.group_count
    }

    #[inline]
    #[allow(dead_code)]
    fn node_count(&self) -> usize {
        self.assignments.len()
    }

    #[inline]
    fn group_size(&self, group: usize) -> usize {
        if group < self.group_sizes.len() {
            self.group_sizes[group]
        } else {
            0
        }
    }

    /// Renumber groups to be contiguous 0..k-1, eliminating empty groups.
    fn normalize_groups(&mut self) {
        let mut new_ids = vec![usize::MAX; self.group_count];
        let mut next_id = 0;
        // Only assign IDs to non-empty groups.
        for (g, new_id) in new_ids.iter_mut().enumerate() {
            if g < self.group_sizes.len() && self.group_sizes[g] > 0 {
                *new_id = next_id;
                next_id += 1;
            }
        }
        for g in self.assignments.iter_mut() {
            debug_assert!(new_ids[*g] != usize::MAX, "node in empty group");
            *g = new_ids[*g];
        }
        self.group_count = next_id;
        // Rebuild group_sizes.
        self.group_sizes = vec![0; self.group_count];
        for &g in &self.assignments {
            self.group_sizes[g] += 1;
        }
        // All empty groups were eliminated by renumbering.
        self.empty_communities.clear();
    }

    /// Returns `Vec<Vec<usize>>` — members of each group.
    fn get_group_members(&self) -> Vec<Vec<usize>> {
        let mut groups = vec![Vec::new(); self.group_count];
        for (node, &g) in self.assignments.iter().enumerate() {
            groups[g].push(node);
        }
        groups
    }
}

// ─── RBPartition ──────────────────────────────────────────────────────

/// Reichardt-Bornholdt configuration model partition.
///
/// Q = Σ_c [ 2·w_in(c) − γ · k_c² / (2m) ]
///
/// Caches `community_strengths` (k_c) incrementally — updated in O(1) per
/// `move_node`, rebuilt in O(n) after bulk operations.
#[derive(Clone)]
struct RBPartition {
    graph: LeidenGraph,
    grouping: Grouping,
    resolution: f64,
    two_m: f64,
    /// Per-node strength (weighted degree), never changes.
    node_strengths: Vec<f64>,
    /// Per-community sum of node strengths — maintained incrementally.
    community_strengths: Vec<f64>,
}

impl RBPartition {
    fn new(graph: LeidenGraph, grouping: Grouping, resolution: f64) -> Self {
        let two_m = 2.0 * graph.total_weight();
        let n = graph.node_count();

        // Pre-compute node strengths (immutable).
        let node_strengths: Vec<f64> = (0..n).map(|i| graph.strength(i)).collect();

        // Build community strengths.
        let nc = grouping.group_count();
        let mut community_strengths = vec![0.0f64; nc];
        for node in 0..n {
            community_strengths[grouping.get_group(node)] += node_strengths[node];
        }

        Self {
            graph,
            grouping,
            resolution,
            two_m,
            node_strengths,
            community_strengths,
        }
    }

    /// Singleton partition: each node in its own community.
    fn new_singleton(graph: LeidenGraph, resolution: f64) -> Self {
        let grouping = Grouping::create_isolated(graph.node_count());
        Self::new(graph, grouping, resolution)
    }

    /// Partition with explicit membership.
    fn new_with_membership(graph: LeidenGraph, membership: &[usize], resolution: f64) -> Self {
        let grouping = Grouping::from_assignments(membership);
        Self::new(graph, grouping, resolution)
    }

    // ── Quality ──

    /// RB quality for undirected graphs:
    /// Q = Σ_c [ 2·w_in(c) − γ · k_c² / (2m) ]
    ///
    /// Matches C++ libleidenalg RBConfigurationVertexPartition::quality()
    /// (RBConfigurationVertexPartition.cpp:128-161) for undirected graphs:
    ///   mod = Σ_c [ w_in(c) - γ * K_out(c) * K_in(c) / (4 * total_weight) ]
    ///   q = 2 * mod
    /// Since K_out == K_in == k_c and 4*total_weight == 2*two_m:
    ///   q = Σ_c [ 2*w_in(c) - γ * k_c^2 / two_m ]
    fn quality(&self) -> f64 {
        if self.two_m == 0.0 {
            return 0.0;
        }
        let mut q = 0.0;
        let members = self.grouping.get_group_members();
        for (c, group_members) in members.iter().enumerate() {
            // Sum internal weights (each edge counted once: node <= neighbor).
            let mut w_in = 0.0;
            for &node in group_members {
                for (neighbor, w) in self.graph.neighbors(node) {
                    if self.grouping.get_group(neighbor) == c && node <= neighbor {
                        w_in += w;
                    }
                }
            }
            let k_c = self.community_strengths[c];
            q += 2.0 * w_in - self.resolution * k_c * k_c / self.two_m;
        }
        q
    }

    /// Quality change from moving `node` to `new_community` (thread-safe, &self).
    ///
    /// Matches C++ libleidenalg RBConfigurationVertexPartition::diff_move()
    /// for undirected graphs (RBConfigurationVertexPartition.cpp:40-119).
    ///
    /// Formula (undirected, where w_to == w_from, k_out == k_in):
    ///   diff_old = 2 * (w_to_old - γ * k_i * k_old / two_m)
    ///   diff_new = 2 * (w_to_new + self_weight - γ * k_i * (k_new + k_i) / two_m)
    ///   diff = diff_new - diff_old
    ///
    /// Note: w_to_old/w_to_new use halved self-loop weights (from weight_to_comm).
    /// k_old includes node's own strength; k_new does not (node hasn't moved yet).
    #[inline]
    fn diff_move(&self, node: usize, new_community: usize) -> f64 {
        let old_comm = self.grouping.get_group(node);
        if new_community == old_comm {
            return 0.0;
        }
        if self.two_m == 0.0 {
            return 0.0;
        }

        let k_i = self.node_strengths[node];
        let self_weight = self.graph.self_loop_weight(node);

        let w_to_old = self.weight_to_comm(node, old_comm);
        let w_to_new = if new_community < self.grouping.group_count() {
            self.weight_to_comm(node, new_community)
        } else {
            0.0
        };

        let k_old = if old_comm < self.community_strengths.len() {
            self.community_strengths[old_comm]
        } else {
            0.0
        };
        let k_new = if new_community < self.community_strengths.len() {
            self.community_strengths[new_community]
        } else {
            0.0
        };

        // Match C++ RBConfigurationVertexPartition::diff_move (undirected case).
        // C++ sums separate "to" and "from" directional terms; for undirected
        // these are identical, yielding a factor of 2.
        let diff_old = 2.0 * (w_to_old - self.resolution * k_i * k_old / self.two_m);
        let diff_new =
            2.0 * (w_to_new + self_weight - self.resolution * k_i * (k_new + k_i) / self.two_m);

        diff_new - diff_old
    }

    /// Sum of edge weights from `node` to members of `community`.
    ///
    /// Self-loops are counted at full weight. Unlike igraph (which stores
    /// self-loops twice in the undirected neighbor list and halves them),
    /// our CSR stores each self-loop once, so no halving is needed.
    #[inline]
    fn weight_to_comm(&self, node: usize, community: usize) -> f64 {
        let mut w = 0.0;
        for (neighbor, edge_w) in self.graph.neighbors(node) {
            if self.grouping.get_group(neighbor) == community {
                w += edge_w;
            }
        }
        w
    }

    // ── Mutation ──

    /// Move node to a new community — O(1) community_strengths update.
    fn move_node(&mut self, node: usize, new_community: usize) {
        let old = self.grouping.get_group(node);
        if old == new_community {
            return;
        }
        let k = self.node_strengths[node];
        self.community_strengths[old] -= k;
        if new_community >= self.community_strengths.len() {
            self.community_strengths.resize(new_community + 1, 0.0);
        }
        self.community_strengths[new_community] += k;
        self.grouping.set_group(node, new_community);
    }

    fn set_membership(&mut self, membership: &[usize]) {
        for (node, &comm) in membership.iter().enumerate() {
            self.grouping.set_group(node, comm);
        }
        self.rebuild_community_strengths();
    }

    fn renumber_communities(&mut self) {
        self.grouping.normalize_groups();
        self.rebuild_community_strengths();
    }

    fn rebuild_community_strengths(&mut self) {
        let nc = self.grouping.group_count();
        self.community_strengths = vec![0.0; nc];
        for node in 0..self.graph.node_count() {
            let c = self.grouping.get_group(node);
            self.community_strengths[c] += self.node_strengths[node];
        }
    }

    fn add_empty_community(&mut self) {
        let new_id = self.grouping.group_count;
        self.community_strengths.push(0.0);
        self.grouping.group_sizes.push(0);
        self.grouping.group_count += 1;
        self.grouping.empty_communities.push(new_id);
    }

    /// Return a reusable empty community ID, or create a new one.
    /// Matches C++ `MutableVertexPartition::get_empty_community()`.
    fn get_empty_community(&mut self) -> usize {
        if let Some(&id) = self.grouping.empty_communities.last() {
            id
        } else {
            self.add_empty_community();
            self.grouping.group_count - 1
        }
    }

    // ── Accessors ──

    #[inline]
    fn membership(&self, node: usize) -> usize {
        self.grouping.get_group(node)
    }

    fn membership_vector(&self) -> Vec<usize> {
        self.grouping.assignments.clone()
    }

    #[inline]
    fn community_count(&self) -> usize {
        self.grouping.group_count()
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[inline]
    fn group_size(&self, community: usize) -> usize {
        self.grouping.group_size(community)
    }

    #[inline]
    fn resolution(&self) -> f64 {
        self.resolution
    }
}

// ─── Parallel Evaluation ──────────────────────────────────────────────

/// A proposed node move.
#[derive(Debug, Clone, Copy)]
struct ProposedMove {
    node: usize,
    to_comm: usize,
}

/// Greedily extracts conflict-free batches: no two nodes in a batch are neighbors.
struct ConflictFreeBatcher {
    max_batch_size: usize,
}

impl ConflictFreeBatcher {
    fn new(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }

    /// Partition `nodes` into conflict-free batches, skipping stable nodes.
    fn create_batches(
        &self,
        nodes: &[usize],
        graph: &LeidenGraph,
        is_stable: &[bool],
    ) -> Vec<Vec<usize>> {
        let mut batches = Vec::new();
        let mut remaining: Vec<usize> = nodes.iter().copied().filter(|&n| !is_stable[n]).collect();

        while !remaining.is_empty() {
            let (batch, leftover) = self.extract_batch(&remaining, graph);
            if batch.is_empty() {
                break;
            }
            batches.push(batch);
            remaining = leftover;
        }
        batches
    }

    fn extract_batch(&self, candidates: &[usize], graph: &LeidenGraph) -> (Vec<usize>, Vec<usize>) {
        let mut batch = Vec::new();
        let mut leftover = Vec::new();
        let mut locked = vec![false; graph.node_count()];

        for &node in candidates {
            if self.has_conflict(node, graph, &locked) {
                leftover.push(node);
            } else {
                batch.push(node);
                self.mark_locked(node, graph, &mut locked);
                if batch.len() >= self.max_batch_size {
                    // Remaining candidates go to leftover.
                    let pos = candidates.iter().position(|&n| n == node).unwrap();
                    leftover.extend_from_slice(&candidates[pos + 1..]);
                    break;
                }
            }
        }
        (batch, leftover)
    }

    #[inline]
    fn has_conflict(&self, node: usize, graph: &LeidenGraph, locked: &[bool]) -> bool {
        if locked[node] {
            return true;
        }
        for (neighbor, _) in graph.neighbors(node) {
            if locked[neighbor] {
                return true;
            }
        }
        false
    }

    #[inline]
    fn mark_locked(&self, node: usize, graph: &LeidenGraph, locked: &mut [bool]) {
        locked[node] = true;
        for (neighbor, _) in graph.neighbors(node) {
            locked[neighbor] = true;
        }
    }
}

/// Evaluate a batch of nodes in parallel, returning beneficial moves.
fn evaluate_batch(
    batch: &[usize],
    partition: &RBPartition,
    empty_comm: Option<usize>,
) -> Vec<ProposedMove> {
    batch
        .par_iter()
        .filter_map(|&node| evaluate_node(node, partition, empty_comm))
        .collect()
}

/// Evaluate one node: find best community to move to.
fn evaluate_node(
    node: usize,
    partition: &RBPartition,
    empty_comm: Option<usize>,
) -> Option<ProposedMove> {
    let current_comm = partition.membership(node);

    // Collect neighbor communities (AllNeighComms).
    let mut comms = HashSet::new();
    for (neighbor, _) in partition.graph.neighbors(node) {
        comms.insert(partition.membership(neighbor));
    }
    let candidates: Vec<usize> = comms.into_iter().collect();

    let epsilon = 10.0 * f64::EPSILON;
    let mut best_comm = current_comm;
    let mut best_improv = epsilon;

    for &comm in &candidates {
        let improv = partition.diff_move(node, comm);
        if improv > best_improv {
            best_comm = comm;
            best_improv = improv;
        }
    }

    // Consider moving to an empty community.
    if let Some(ec) = empty_comm {
        if partition.group_size(current_comm) > 1 {
            let improv = partition.diff_move(node, ec);
            if improv > best_improv {
                best_comm = ec;
                best_improv = improv;
            }
        }
    }

    if best_comm != current_comm && best_improv > 0.0 {
        Some(ProposedMove {
            node,
            to_comm: best_comm,
        })
    } else {
        None
    }
}

// ─── LeidenOptimizer ─────────────────────────────────────────────────

/// Orchestrates the Leiden hierarchical optimization loop.
struct LeidenOptimizer {
    config: LeidenConfig,
    rng: ChaCha8Rng,
}

impl LeidenOptimizer {
    fn new(config: LeidenConfig) -> Self {
        let rng = match config.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Run the full Leiden algorithm on `partition` (modified in place).
    /// Returns total quality improvement.
    fn optimize(&mut self, partition: &mut RBPartition) -> Result<f64> {
        let n = partition.node_count();
        if n == 0 {
            return Ok(0.0);
        }

        let mut collapsed = partition.clone();
        let mut aggregate_map: Vec<usize> = (0..n).collect();
        let mut is_first_iteration = true;
        let mut total_improvement = 0.0;

        for _iter in 0..self.config.max_iterations {
            // ── Phase 1: Local moving ──
            let improvement = if self.config.parallel {
                self.move_nodes_parallel(&mut collapsed)?
            } else {
                self.move_nodes_sequential(&mut collapsed)?
            };
            total_improvement += improvement;

            // ── Map optimized communities back to original partition ──
            if is_first_iteration && aggregate_map.iter().enumerate().all(|(i, &v)| i == v) {
                let membership = collapsed.membership_vector();
                partition.set_membership(&membership);
            } else {
                for (node, &agg_node) in aggregate_map.iter().enumerate().take(n) {
                    if agg_node < collapsed.node_count() {
                        let new_comm = collapsed.membership(agg_node);
                        partition.move_node(node, new_comm);
                    }
                }
            }

            // ── Phase 2: Refinement and collapse ──
            let new_collapsed = if self.config.refine_partition {
                self.refine_and_collapse(&collapsed, &mut aggregate_map, n)?
            } else {
                self.simple_collapse(&collapsed)
            };

            // ── Convergence check ──
            let should_continue = new_collapsed.node_count() < collapsed.node_count()
                && collapsed.node_count() > collapsed.community_count();

            collapsed = new_collapsed;
            is_first_iteration = false;

            // Ask glibc to return freed heap pages to the OS. Without this,
            // the repeated HashMap allocations in aggregate() fragment the
            // heap, and glibc keeps all pages mapped, growing RSS to 50+ GB
            // on graphs with 100K+ nodes.
            #[cfg(target_os = "linux")]
            unsafe {
                libc::malloc_trim(0);
            }

            if !should_continue {
                break;
            }
        }

        partition.renumber_communities();
        Ok(total_improvement)
    }

    /// Parallel local moving: conflict-free batched evaluation + sequential apply.
    fn move_nodes_parallel(&mut self, partition: &mut RBPartition) -> Result<f64> {
        let n = partition.node_count();
        let graph = partition.graph.clone(); // Arc clone — cheap.

        let mut total_improv = 0.0;
        let mut is_stable = vec![false; n];

        // Initial queue: all nodes, shuffled.
        let mut nodes: Vec<usize> = (0..n).collect();
        nodes.shuffle(&mut self.rng);
        let mut pending: VecDeque<usize> = nodes.into();

        let batcher = ConflictFreeBatcher::new(10_000);

        // Safety limit: parallel batching with stale reads can cause oscillation
        // where nodes cycle between communities. Bound queue passes to prevent
        // infinite loops. The C++ sequential move_nodes converges naturally;
        // this limit approximates the same bound for the parallel variant.
        while !pending.is_empty() {
            let current: Vec<usize> = pending.drain(..).collect();
            let batches = batcher.create_batches(&current, &graph, &is_stable);

            let mut made_move = false;
            for batch in batches {
                // Pre-compute empty community ID for this batch (shared across
                // all parallel evaluations). Uses reuse pool if available.
                let empty_comm = if self.config.consider_empty_community {
                    Some(partition.get_empty_community())
                } else {
                    None
                };
                let proposed = evaluate_batch(&batch, partition, empty_comm);

                for m in proposed {
                    // Verify the move is still beneficial after sequential application
                    // of earlier moves in this batch (stale-read guard).
                    let current_diff = partition.diff_move(m.node, m.to_comm);
                    if current_diff <= 0.0 {
                        continue;
                    }

                    // Ensure community exists.
                    while partition.community_count() <= m.to_comm {
                        partition.add_empty_community();
                    }

                    total_improv += current_diff;
                    partition.move_node(m.node, m.to_comm);
                    is_stable[m.node] = true;
                    made_move = true;

                    // Mark neighbors unstable.
                    for (neighbor, _) in graph.neighbors(m.node) {
                        if is_stable[neighbor] && partition.membership(neighbor) != m.to_comm {
                            is_stable[neighbor] = false;
                            pending.push_back(neighbor);
                        }
                    }
                }
            }
            // If no moves were made in this pass, the queue will only contain
            // nodes that were already stable or had no beneficial moves.
            if !made_move {
                break;
            }
        }

        partition.renumber_communities();
        Ok(total_improv)
    }

    /// Sequential local moving — matches C++ libleidenalg `Optimiser::move_nodes()`
    /// (Optimiser.cpp:489-749).
    ///
    /// Nodes are shuffled into a deque. Each node is popped, its best neighbor
    /// community evaluated, and if beneficial the node is moved immediately.
    /// Neighbors of moved nodes are marked unstable and re-queued. The loop
    /// continues until the queue is empty (all nodes stable).
    fn move_nodes_sequential(&mut self, partition: &mut RBPartition) -> Result<f64> {
        let n = partition.node_count();
        let graph = partition.graph.clone();

        let mut total_improv = 0.0;
        let mut is_stable = vec![false; n];

        // Initial queue: all nodes, shuffled (matching C++ lines 535-541).
        let mut nodes: Vec<usize> = (0..n).collect();
        nodes.shuffle(&mut self.rng);
        let mut vertex_order: VecDeque<usize> = nodes.into();

        let epsilon = 10.0 * f64::EPSILON;
        // Reuse HashSet across iterations to avoid heap fragmentation from
        // repeated allocations (glibc never returns freed pages to the OS).
        let mut comms = HashSet::new();

        while let Some(v) = vertex_order.pop_front() {
            let v_comm = partition.membership(v);

            // Collect neighbor communities (AllNeighComms, matching C++ lines 578-591).
            comms.clear();
            for (neighbor, _) in graph.neighbors(v) {
                let nc = partition.membership(neighbor);
                if nc != v_comm {
                    comms.insert(nc);
                }
            }

            let mut best_comm = v_comm;
            let mut best_improv = epsilon;

            for comm in &comms {
                let improv = partition.diff_move(v, *comm);
                if improv > best_improv {
                    best_comm = *comm;
                    best_improv = improv;
                }
            }

            // Consider moving to an empty community (matching C++ lines 615-634).
            // Uses get_empty_community() to reuse IDs instead of creating new ones.
            if self.config.consider_empty_community && partition.group_size(v_comm) > 1 {
                let empty_comm = partition.get_empty_community();
                let improv = partition.diff_move(v, empty_comm);
                if improv > best_improv {
                    best_comm = empty_comm;
                    best_improv = improv;
                }
            }

            // Mark node as stable (matching C++ line 672).
            is_stable[v] = true;

            // Move node if beneficial (matching C++ lines 675-734).
            if best_comm != v_comm {
                // Ensure community exists (only needed if get_empty_community
                // returned a brand-new ID that hasn't been registered yet).
                while partition.community_count() <= best_comm {
                    partition.add_empty_community();
                }

                total_improv += best_improv;
                partition.move_node(v, best_comm);

                // Mark neighbors as unstable and re-queue (matching C++ lines 717-731).
                for (neighbor, _) in graph.neighbors(v) {
                    if is_stable[neighbor] && partition.membership(neighbor) != best_comm {
                        is_stable[neighbor] = false;
                        vertex_order.push_back(neighbor);
                    }
                }
            }
        }

        partition.renumber_communities();
        Ok(total_improv)
    }

    /// Constrained merge for the refinement phase.
    ///
    /// Starting from singleton communities, merges single-node communities into
    /// neighboring communities, constrained to stay within the same community
    /// from `constrained_partition`.
    fn merge_nodes_constrained(
        &mut self,
        partition: &mut RBPartition,
        constrained_partition: &RBPartition,
    ) -> Result<f64> {
        let n = partition.node_count();
        let constrained_membership = constrained_partition.membership_vector();

        let mut total_improv = 0.0;
        let mut vertex_order: Vec<usize> = (0..n).collect();
        vertex_order.shuffle(&mut self.rng);

        for v in vertex_order {
            let v_comm = partition.membership(v);

            // Merge behaviour: only consider singleton communities.
            if partition.group_size(v_comm) != 1 {
                continue;
            }

            // Collect constrained candidates (AllNeighComms within same constrained group).
            let v_constrained = constrained_membership[v];
            let mut comms = HashSet::new();
            for (neighbor, _) in partition.graph.neighbors(v) {
                if constrained_membership[neighbor] == v_constrained {
                    comms.insert(partition.membership(neighbor));
                }
            }

            let mut best_comm = v_comm;
            let mut best_improv = 0.0;

            for comm in comms {
                let improv = partition.diff_move(v, comm);
                if improv >= best_improv && comm != v_comm {
                    best_comm = comm;
                    best_improv = improv;
                }
            }

            if best_comm != v_comm {
                total_improv += best_improv;
                partition.move_node(v, best_comm);
            }
        }

        partition.renumber_communities();
        Ok(total_improv)
    }

    /// Refine the collapsed partition then aggregate into a coarser graph.
    fn refine_and_collapse(
        &mut self,
        collapsed_partition: &RBPartition,
        aggregate_map: &mut [usize],
        original_n: usize,
    ) -> Result<RBPartition> {
        // Start with singletons on the same graph.
        let graph = collapsed_partition.graph.clone();
        let mut sub_partition = RBPartition::new_singleton(graph, collapsed_partition.resolution());

        // Constrained refinement.
        self.merge_nodes_constrained(&mut sub_partition, collapsed_partition)?;

        // Update aggregate mapping: original node → refined community.
        for item in aggregate_map.iter_mut().take(original_n) {
            if *item < sub_partition.node_count() {
                *item = sub_partition.membership(*item);
            }
        }

        // Collapse: aggregate the network by refined communities.
        let collapsed_network = collapsed_partition.graph.aggregate(&sub_partition.grouping);

        // Build membership for the new collapsed partition.
        let refined_membership = sub_partition.membership_vector();
        let mut new_membership = vec![0usize; collapsed_network.node_count()];
        for (v, &refined_comm) in refined_membership
            .iter()
            .enumerate()
            .take(collapsed_partition.node_count())
        {
            let original_comm = collapsed_partition.membership(v);
            if refined_comm < new_membership.len() {
                new_membership[refined_comm] = original_comm;
            }
        }

        Ok(RBPartition::new_with_membership(
            collapsed_network,
            &new_membership,
            collapsed_partition.resolution(),
        ))
    }

    /// Non-refined collapse: aggregate directly by current communities.
    fn simple_collapse(&self, collapsed_partition: &RBPartition) -> RBPartition {
        let collapsed_network = collapsed_partition
            .graph
            .aggregate(&collapsed_partition.grouping);
        RBPartition::new_singleton(collapsed_network, collapsed_partition.resolution())
    }
}

// ─── Public API ───────────────────────────────────────────────────────

/// Run Leiden community detection on a CSR adjacency graph.
///
/// # Arguments
/// * `indptr`  — CSR row pointers (i64, length n_nodes+1)
/// * `indices` — CSR column indices (i32)
/// * `weights` — CSR edge weights (f64, e.g. fuzzy connectivities in \[0,1\])
/// * `n_nodes` — Number of nodes
/// * `resolution` — Resolution parameter γ (default 1.0; higher → more communities)
/// * `seed`    — Random seed for reproducibility
/// * `max_iterations` — Number of outer Leiden iterations (matching leidenalg's
///   `n_iterations` parameter). If 0, runs until convergence (matching scanpy's
///   default `n_iterations=-1`). If > 0, runs exactly that many outer iterations.
///   Each outer iteration is one complete hierarchical Leiden pass
///   (move → refine → aggregate → repeat until graph stops collapsing).
/// * `parallel` — Use parallel (conflict-free batched) local moving. Default `false`
///   uses sequential moving that matches C++ leidenalg exactly.
#[allow(clippy::too_many_arguments)]
pub fn leiden(
    indptr: &[i64],
    indices: &[i32],
    weights: &[f64],
    n_nodes: usize,
    resolution: f64,
    seed: u64,
    max_iterations: usize,
    parallel: bool,
) -> Result<LeidenResult> {
    // ── Validate input ──
    if indptr.len() != n_nodes + 1 {
        return Err(AccelError::InvalidInput(format!(
            "indptr length {} != n_nodes + 1 ({})",
            indptr.len(),
            n_nodes + 1
        )));
    }
    if indices.len() != weights.len() {
        return Err(AccelError::InvalidInput(
            "indices and weights must have equal length".into(),
        ));
    }
    if n_nodes == 0 {
        return Ok(LeidenResult {
            membership: vec![],
            modularity: 0.0,
            n_communities: 0,
        });
    }
    if n_nodes == 1 {
        return Ok(LeidenResult {
            membership: vec![0],
            modularity: 0.0,
            n_communities: 1,
        });
    }

    // ── Build graph ──
    let graph = LeidenGraph::from_csr(indptr, indices, weights, n_nodes);

    // ── Configure ──
    // Inner hierarchy limit: how many aggregate cycles per pass. The C++ leidenalg
    // typically converges in 3-8 inner iterations (100K→30K→7K→1K→200→50→25→19).
    // Cap at 10 to prevent excessive memory from allocator fragmentation.
    //
    // consider_empty_community re-enabled: get_empty_community() now reuses
    // empty community IDs (matching C++ MutableVertexPartition::get_empty_community())
    // instead of creating new IDs, preventing community-count explosion and OOM.
    let config = LeidenConfig {
        max_iterations: 10,
        seed: Some(seed),
        resolution,
        parallel,
        ..LeidenConfig::default()
    };

    // ── Run with outer re-optimization loop ──
    // Matches Python leidenalg's Optimiser.optimise_partition() (Optimiser.py:299-310):
    //   while continue_iteration:
    //       diff_inc = _c_leiden._Optimiser_optimise_partition(...)
    //       if n_iterations < 0: continue_iteration = (diff_inc > 0)
    //       else: continue_iteration = itr < n_iterations
    let mut partition = RBPartition::new_singleton(graph, resolution);
    let mut optimizer = LeidenOptimizer::new(config);

    // Outer re-optimization loop: re-runs the full Leiden hierarchy from
    // the converged partition. Matches leidenalg's default n_iterations=2
    // (leidenalg/functions.py:20, Optimiser.py:252). Each optimize() call
    // does a full hierarchical pass: move → refine → aggregate → converge.
    let outer_limit = if max_iterations == 0 {
        2
    } else {
        max_iterations
    };

    for _outer in 0..outer_limit {
        optimizer.optimize(&mut partition)?;
    }

    let membership = partition.membership_vector();
    let modularity = partition.quality();
    let n_communities = partition.community_count();

    Ok(LeidenResult {
        membership,
        modularity,
        n_communities,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric CSR from an edge list: (u, v, w).
    fn edges_to_csr(edges: &[(usize, usize, f64)], n: usize) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
        // Build adjacency lists.
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for &(u, v, w) in edges {
            adj[u].push((v, w));
            if u != v {
                adj[v].push((u, w));
            }
        }
        for list in &mut adj {
            list.sort_by_key(|&(n, _)| n);
        }
        let mut indptr = vec![0i64; n + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for (node, list) in adj.iter().enumerate() {
            for &(neighbor, w) in list {
                indices.push(neighbor as i32);
                data.push(w);
            }
            indptr[node + 1] = indices.len() as i64;
        }
        (indptr, indices, data)
    }

    #[test]
    fn test_two_clusters() {
        // Two well-separated cliques connected by a weak bridge.
        //
        //  Cluster A: 0-1, 0-2, 1-2  (weight 1.0)
        //  Cluster B: 3-4, 3-5, 4-5  (weight 1.0)
        //  Bridge:    2-3             (weight 0.01)
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (3, 4, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
            (2, 3, 0.01),
        ];
        let n = 6;
        let (indptr, indices, data) = edges_to_csr(&edges, n);

        let result = leiden(&indptr, &indices, &data, n, 1.0, 42, 0, false).unwrap();

        assert_eq!(result.membership.len(), n);
        // RB quality can be slightly negative even for good partitions.
        assert_eq!(result.n_communities, 2, "should find 2 communities");

        // Nodes within each clique should share a community.
        assert_eq!(result.membership[0], result.membership[1]);
        assert_eq!(result.membership[0], result.membership[2]);
        assert_eq!(result.membership[3], result.membership[4]);
        assert_eq!(result.membership[3], result.membership[5]);

        // The two clusters should be in different communities.
        assert_ne!(result.membership[0], result.membership[3]);
    }

    #[test]
    fn test_resolution_parameter() {
        // A ring of 6 cliques connected sequentially — higher resolution
        // should yield more communities than lower resolution.
        let mut edges = Vec::new();
        let clique_size = 4;
        let n_cliques = 6;
        let n = clique_size * n_cliques;

        // Intra-clique edges (strong).
        for c in 0..n_cliques {
            let base = c * clique_size;
            for i in 0..clique_size {
                for j in i + 1..clique_size {
                    edges.push((base + i, base + j, 1.0));
                }
            }
        }
        // Inter-clique bridges (weak).
        for c in 0..n_cliques {
            let next = (c + 1) % n_cliques;
            edges.push((c * clique_size, next * clique_size, 0.05));
        }

        let (indptr, indices, data) = edges_to_csr(&edges, n);

        let low_res = leiden(&indptr, &indices, &data, n, 0.5, 42, 0, false).unwrap();
        let high_res = leiden(&indptr, &indices, &data, n, 2.0, 42, 0, false).unwrap();

        assert!(
            high_res.n_communities >= low_res.n_communities,
            "higher resolution should give >= communities: high={} low={}",
            high_res.n_communities,
            low_res.n_communities
        );
    }

    #[test]
    fn test_deterministic_with_seed() {
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (2, 3, 0.5),
            (3, 4, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
        ];
        let n = 6;
        let (indptr, indices, data) = edges_to_csr(&edges, n);

        let r1 = leiden(&indptr, &indices, &data, n, 1.0, 123, 0, false).unwrap();
        let r2 = leiden(&indptr, &indices, &data, n, 1.0, 123, 0, false).unwrap();

        assert_eq!(
            r1.membership, r2.membership,
            "same seed should give same result"
        );
        assert!((r1.modularity - r2.modularity).abs() < 1e-12);
    }

    #[test]
    fn test_empty_graph() {
        let result = leiden(&[0], &[], &[], 0, 1.0, 42, 0, false).unwrap();
        assert_eq!(result.n_communities, 0);
    }

    #[test]
    fn test_single_node() {
        let result = leiden(&[0, 0], &[], &[], 1, 1.0, 42, 0, false).unwrap();
        assert_eq!(result.membership, vec![0]);
        assert_eq!(result.n_communities, 1);
    }

    #[test]
    fn test_disconnected_components() {
        // Two disconnected cliques with no bridge.
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (3, 4, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
        ];
        let n = 6;
        let (indptr, indices, data) = edges_to_csr(&edges, n);

        let result = leiden(&indptr, &indices, &data, n, 1.0, 42, 0, false).unwrap();

        assert_eq!(result.n_communities, 2);
        assert_eq!(result.membership[0], result.membership[1]);
        assert_eq!(result.membership[0], result.membership[2]);
        assert_eq!(result.membership[3], result.membership[4]);
        assert_eq!(result.membership[3], result.membership[5]);
        assert_ne!(result.membership[0], result.membership[3]);
    }

    #[test]
    fn test_leiden_graph_from_csr() {
        // Simple triangle: 0-1, 1-2, 0-2 all weight 1.0
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let n = 3;
        let (indptr, indices, data) = edges_to_csr(&edges, n);

        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);

        assert_eq!(graph.node_count(), 3);
        // Total weight = 3 edges × 1.0 = 3.0
        assert!((graph.total_weight() - 3.0).abs() < 1e-10);
        // Each node has degree 2, strength 2.0
        for i in 0..3 {
            assert!((graph.strength(i) - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_grouping_normalize() {
        let g = Grouping::from_assignments(&[0, 0, 5, 5, 5]);
        // After normalization, groups should be 0 and 1 (no gap).
        assert_eq!(g.group_count(), 2);
        assert_eq!(g.get_group(0), 0);
        assert_eq!(g.get_group(2), 1);
    }

    #[test]
    fn test_well_separated_clusters_positive_modularity() {
        // Two K5 cliques connected by a very weak bridge.
        // At resolution γ=0.5, the RB quality is clearly positive.
        let mut edges = Vec::new();
        // Clique A: nodes 0–4 (complete K5, weight 1.0)
        for i in 0..5 {
            for j in i + 1..5 {
                edges.push((i, j, 1.0));
            }
        }
        // Clique B: nodes 5–9 (complete K5, weight 1.0)
        for i in 5..10 {
            for j in i + 1..10 {
                edges.push((i, j, 1.0));
            }
        }
        // Weak bridge between cliques.
        edges.push((4, 5, 0.001));

        let n = 10;
        let (indptr, indices, data) = edges_to_csr(&edges, n);

        let result = leiden(&indptr, &indices, &data, n, 0.5, 42, 0, false).unwrap();

        assert_eq!(result.n_communities, 2, "should find 2 communities");
        assert!(
            result.modularity > 0.0,
            "modularity should be positive at γ=0.5, got {}",
            result.modularity
        );

        // All nodes in each clique share a community.
        for i in 1..5 {
            assert_eq!(result.membership[0], result.membership[i]);
        }
        for i in 6..10 {
            assert_eq!(result.membership[5], result.membership[i]);
        }
        assert_ne!(result.membership[0], result.membership[5]);
    }

    #[test]
    fn test_expected_partition_synthetic() {
        // Three K4 cliques connected by weak bridges — the partition is
        // unambiguous: each clique forms its own community.
        //
        //   Clique 0: {0,1,2,3}   Clique 1: {4,5,6,7}   Clique 2: {8,9,10,11}
        //   Bridges: 3–4 (0.01), 7–8 (0.01)
        let mut edges = Vec::new();
        for c in 0..3 {
            let base = c * 4;
            for i in 0..4 {
                for j in i + 1..4 {
                    edges.push((base + i, base + j, 1.0));
                }
            }
        }
        edges.push((3, 4, 0.01));
        edges.push((7, 8, 0.01));

        let n = 12;
        let (indptr, indices, data) = edges_to_csr(&edges, n);

        let result = leiden(&indptr, &indices, &data, n, 1.0, 42, 0, false).unwrap();

        assert_eq!(result.n_communities, 3, "should find 3 communities");

        // Collect community sets.
        let mut comm_members: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (node, &comm) in result.membership.iter().enumerate() {
            comm_members.entry(comm).or_default().push(node);
        }

        // Each community should be one of the 3 cliques.
        let mut groups: Vec<Vec<usize>> = comm_members.into_values().collect();
        for g in &mut groups {
            g.sort();
        }
        groups.sort();

        assert_eq!(
            groups,
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
            "partition should match expected cliques"
        );
    }

    #[test]
    fn test_rb_partition_quality() {
        // 3 disconnected triangles: optimal partition should have positive quality.
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
            (6, 7, 1.0),
            (7, 8, 1.0),
            (6, 8, 1.0),
        ];
        let n = 9;
        let (indptr, indices, data) = edges_to_csr(&edges, n);
        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);

        // Optimal: each triangle is its own community.
        let p_opt =
            RBPartition::new_with_membership(graph.clone(), &[0, 0, 0, 1, 1, 1, 2, 2, 2], 1.0);
        // Suboptimal: all in one community.
        let p_all = RBPartition::new_with_membership(graph, &[0, 0, 0, 0, 0, 0, 0, 0, 0], 1.0);

        assert!(
            p_opt.quality() > 0.0,
            "3-community quality should be positive, got {}",
            p_opt.quality()
        );
        assert!(
            p_opt.quality() > p_all.quality(),
            "3-community partition should beat all-in-one"
        );
    }

    #[test]
    fn test_diff_move_triangle_no_self_loops() {
        // Triangle: 0-1, 1-2, 0-2 all weight 1.0, all in community 0.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let n = 3;
        let (indptr, indices, data) = edges_to_csr(&edges, n);
        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);

        // Ensure community 1 exists for the move target.
        let mut partition = RBPartition::new_with_membership(graph, &[0, 0, 0], 1.0);
        partition.add_empty_community();

        // Moving node 0 to empty community 1 should be harmful.
        // two_m = 6.0, k_i = 2.0, self_weight = 0.0
        // w_to_old = 2.0 (neighbors 1,2 in comm 0), w_to_new = 0.0
        // k_old = 6.0 (includes node 0), k_new = 0.0
        // diff_old = 2*(2.0 - 1.0*2.0*6.0/6.0) = 2*(2.0 - 2.0) = 0.0
        // diff_new = 2*(0.0 + 0.0 - 1.0*2.0*2.0/6.0) = 2*(-2/3) = -4/3
        // diff = -4/3 - 0 = -4/3
        let diff = partition.diff_move(0, 1);
        assert!(
            diff < -1.0,
            "moving node out of triangle should be strongly negative, got {}",
            diff
        );
        assert!(
            (diff - (-4.0 / 3.0)).abs() < 1e-10,
            "diff_move should be -4/3, got {}",
            diff
        );
    }

    #[test]
    fn test_diff_move_with_self_loops() {
        // Simulates an aggregated graph: two super-nodes with self-loops.
        // Super-node 0: self-loop weight 3.0
        // Super-node 1: self-loop weight 3.0
        // Edge 0-1: weight 0.01
        let edges = vec![(0, 0, 3.0), (1, 1, 3.0), (0, 1, 0.01)];
        let n = 2;
        let (indptr, indices, data) = edges_to_csr(&edges, n);
        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);

        // Verify strengths match igraph convention (self-loops counted twice).
        assert!(
            (graph.strength(0) - 6.01).abs() < 1e-10,
            "strength should count self-loop twice: expected 6.01, got {}",
            graph.strength(0)
        );

        // Singleton partition: each node in its own community.
        let partition = RBPartition::new_singleton(graph, 1.0);

        // Moving node 0 to community of node 1 should be NEGATIVE
        // (the two super-nodes represent well-separated clusters).
        let diff = partition.diff_move(0, 1);
        assert!(
            diff < 0.0,
            "merging super-nodes should be negative, got {}",
            diff
        );

        // Verify exact value:
        // two_m = 12.02, k_i = 6.01, k_old = 6.01, k_new = 6.01
        // w_to_old = 3.0 (self-loop at full weight), w_to_new = 0.01, sw = 3.0
        // diff_old = 2*(3.0 - 1.0*6.01*6.01/12.02) = 2*(3.0 - 3.005) = -0.01
        // diff_new = 2*(0.01 + 3.0 - 1.0*6.01*12.02/12.02) = 2*(3.01 - 6.01) = -6.0
        // diff = -6.0 - (-0.01) = -5.99
        let expected = -5.99;
        assert!(
            (diff - expected).abs() < 0.02,
            "diff_move expected ~{}, got {}",
            expected,
            diff
        );
    }

    #[test]
    fn test_quality_matches_diff_move() {
        // Two triangles with weak bridge: verify quality delta equals diff_move.
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (3, 4, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
            (2, 3, 0.01),
        ];
        let n = 6;
        let (indptr, indices, data) = edges_to_csr(&edges, n);
        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);

        // Start with all in one community.
        let mut partition = RBPartition::new_with_membership(graph, &[0, 0, 0, 0, 0, 0], 1.0);
        partition.add_empty_community(); // ensure community 1 exists
        let q_before = partition.quality();

        // Compute diff_move for moving node 3 to new community 1.
        let diff = partition.diff_move(3, 1);

        // Actually move the node.
        partition.move_node(3, 1);
        let q_after = partition.quality();

        let actual_improvement = q_after - q_before;
        assert!(
            (actual_improvement - diff).abs() < 1e-10,
            "quality delta ({}) should match diff_move ({})",
            actual_improvement,
            diff
        );
    }

    #[test]
    fn test_quality_cpp_value() {
        // Single triangle in one community at resolution 1.0.
        // C++ quality: q = 2*w_in - gamma*k_c^2/two_m
        //            = 2*3.0 - 1.0*36.0/6.0 = 6.0 - 6.0 = 0.0
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let n = 3;
        let (indptr, indices, data) = edges_to_csr(&edges, n);
        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);
        let partition = RBPartition::new_with_membership(graph, &[0, 0, 0], 1.0);

        // RB modularity of a complete graph in one community at resolution 1.0 is 0.
        let q = partition.quality();
        assert!(
            q.abs() < 1e-10,
            "triangle in one community should have quality ~0.0, got {}",
            q
        );
    }

    #[test]
    fn test_outer_loop_improves_or_matches_single_pass() {
        // Ring of 6 K4 cliques with weak bridges — complex enough that
        // multiple outer iterations may find better partitions.
        let mut edges = Vec::new();
        let clique_size = 4;
        let n_cliques = 6;
        let n = clique_size * n_cliques;

        for c in 0..n_cliques {
            let base = c * clique_size;
            for i in 0..clique_size {
                for j in i + 1..clique_size {
                    edges.push((base + i, base + j, 1.0));
                }
            }
        }
        for c in 0..n_cliques {
            let next = (c + 1) % n_cliques;
            edges.push((c * clique_size, next * clique_size, 0.05));
        }

        let (indptr, indices, data) = edges_to_csr(&edges, n);

        // Single outer pass (max_iterations=1).
        let single = leiden(&indptr, &indices, &data, n, 1.0, 42, 1, false).unwrap();

        // Run until convergence (max_iterations=0 → n_iterations=-1 behavior).
        let converged = leiden(&indptr, &indices, &data, n, 1.0, 42, 0, false).unwrap();

        // The converged result should have quality >= single pass.
        assert!(
            converged.modularity >= single.modularity - 1e-10,
            "converged quality ({}) should be >= single pass quality ({})",
            converged.modularity,
            single.modularity
        );
    }
}
