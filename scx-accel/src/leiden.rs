//! Leiden community detection algorithm for network community detection.
//!
//! Provides a Rust-native community-detection implementation in the Leiden
//! family (Traag, Waltman & van Eck, 2019) with the Reichardt-Bornholdt (RB)
//! configuration model quality function and conflict-free parallel batching via
//! rayon.
//!
//! NOTE: the refinement phase moves singleton nodes under a constrained-partition
//! rule but does NOT implement the paper's node/candidate well-connectedness
//! admissibility conditions or `theta`-randomized selection — it is closer to
//! Louvain with constrained refinement than to full Leiden (tracked for
//! completion; see `LeidenConfig::refine_partition`). Do not describe the output
//! as "well-connected communities".
//!
//! Core algorithm adapted from single-clustering
//! (BSD 3-Clause License, Copyright 2025 Ian F. Diks)
//! <https://github.com/SingleRust/single-clustering>
//!
//! Simplified for SCX: concrete `f64` types, single-partition single-layer,
//! RB configuration model only, `AllNeighComms` strategy.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::error::{AccelError, Result};

// ─── Public Types ─────────────────────────────────────────────────────

/// Normalize an RB quality score onto the per-edge scale: `quality / 2m`.
///
/// The single definition of the relationship between [`LeidenResult::quality`]
/// and [`LeidenResult::modularity`], so the production path and the tests that
/// pin it against igraph cannot drift apart. `two_m == 0` (an edgeless graph)
/// has no meaningful normalization and yields 0.0, matching
/// [`RBPartition::quality`]'s own degenerate answer.
fn rb_modularity(quality: f64, two_m: f64) -> f64 {
    if two_m == 0.0 {
        0.0
    } else {
        quality / two_m
    }
}

/// Result of Leiden community detection.
pub struct LeidenResult {
    /// Community label for each node (0-indexed, contiguous).
    pub membership: Vec<usize>,
    /// **Normalized** generalized RB modularity, `quality / 2m`.
    ///
    /// At `resolution = 1.0` this **is** Newman modularity — the quantity in
    /// `[-0.5, 1]` that igraph / leidenalg / cuGraph report, and the one that is
    /// comparable across graphs.
    ///
    /// At `resolution != 1.0` the γ term does not cancel, so this is the
    /// *generalized* RB objective on a per-edge scale and **is not bounded by
    /// `[-0.5, 1]`**: on Zachary's Karate Club it is `-0.996055` at γ = 20 and
    /// `-2.490138` at γ = 50, where leidenalg's own `modularity` property
    /// reports `-0.049803` for the same partitions. Normalizing is still the
    /// right thing — it is what makes the number comparable with the cuGraph
    /// backend, which writes into this same key — but do not read a γ ≠ 1 value
    /// as Newman modularity.
    pub modularity: f64,
    /// The raw, **un-normalized** RB quality
    /// `Σ_c [2·w_in(c) − γ·k_c²/2m]` — leidenalg's internal
    /// `RBConfigurationVertexPartition.quality()`. It scales with the graph's
    /// total edge weight (~10⁶ on a 1M-edge graph), so it is comparable only
    /// between partitions of the *same* graph. Retained because it is what the
    /// optimizer's `diff_move` deltas sum to.
    pub quality: f64,
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
    /// Whether to run the refinement phase (singleton-start constrained
    /// moving) between local-moving passes (default true).
    ///
    /// NOTE: this refinement moves singleton nodes greedily but does NOT
    /// implement the paper's node/candidate **well-connectedness admissibility
    /// conditions** or `theta`-randomized selection, so it is not the full
    /// Leiden well-connected-community guarantee — closer to Louvain with
    /// constrained refinement. (Tracked for completion; do not describe the
    /// output as "well-connected communities".)
    pub refine_partition: bool,
    /// Whether to consider moving nodes to empty communities (default true).
    pub consider_empty_community: bool,
    /// Use parallel (conflict-free batched) local moving instead of sequential.
    /// Sequential (default, `false`) reproduces C++ libleidenalg's move-node
    /// *ordering* (not a guarantee of overall-algorithm parity — see
    /// `refine_partition`).
    /// Parallel (`true`) reaches a different partition, because batching reads
    /// a partition that later moves in the same batch may have stale. What it
    /// now shares with the sequential path is the *queue discipline*: a node
    /// that declines an evaluation stays eligible for re-evaluation when a
    /// neighbour moves. (It did not before — decliners were retired after a
    /// single evaluation, costing ~22 % of the RB quality sequential reaches.)
    ///
    /// Neither path guarantees that no beneficial move remains when local
    /// moving returns. Both requeue a neighbour only when it is not already in
    /// the destination community, so a node whose neighbourhood changes after
    /// its last evaluation can be left holding one.
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
    strengths: Vec<f64>,
    total_weight: f64,
}

/// Arc-wrapped CSR graph — cheap to clone (reference counted).
#[derive(Clone)]
struct LeidenGraph {
    data: Arc<LeidenGraphData>,
}

/// Zero-cost iterator over (neighbor, weight) pairs using slice iterators.
/// The borrow checker proves safety; no raw pointers or `unsafe impl Send/Sync`
/// needed. Generates identical codegen to the prior pointer-arithmetic form
/// (confirmed via SIMD benchmarks — slice iterators are lowered to the same
/// pointer increments after autovectorization).
struct NeighborIterator<'a> {
    neighbors: std::slice::Iter<'a, usize>,
    weights: std::slice::Iter<'a, f64>,
}

impl<'a> Iterator for NeighborIterator<'a> {
    type Item = (usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match (self.neighbors.next(), self.weights.next()) {
            (Some(&n), Some(&w)) => Some((n, w)),
            _ => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.neighbors.size_hint()
    }
}

impl<'a> ExactSizeIterator for NeighborIterator<'a> {}

impl LeidenGraph {
    /// Build a Leiden graph directly from an SCX-format symmetric CSR matrix.
    ///
    /// `indptr` (length n_nodes+1, i64), `indices` (i32), `data` (f64) describe
    /// the full symmetric adjacency — each undirected edge appears twice.
    pub fn from_csr(indptr: &[i64], indices: &[i32], data: &[f64], n_nodes: usize) -> Self {
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
        let mut strengths = vec![0.0f64; n_nodes];
        let node_weights = vec![1.0f64; n_nodes];
        let mut total_weight = 0.0f64;

        for node in 0..n_nodes {
            let start = node_ptrs[node];
            let end = node_ptrs[node + 1];
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
                strengths,
                total_weight,
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
                strengths,
                total_weight,
            }),
        }
    }

    #[inline]
    fn neighbors(&self, node: usize) -> NeighborIterator<'_> {
        let start = self.data.node_ptrs[node];
        let end = self.data.node_ptrs[node + 1];
        NeighborIterator {
            neighbors: self.data.neighbors[start..end].iter(),
            weights: self.data.weights[start..end].iter(),
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

        // Accumulate collapsed edge weights per ordered group pair `(a, b)` with
        // `a <= b` (self-loop when `a == b`) via a sort/merge instead of two
        // HashMaps — the group ids are dense `0..group_count`, so this avoids the
        // per-collapse HashMap allocation churn (the source of the heap
        // fragmentation the outer loop's `malloc_trim` compensates for).
        //
        // Byte-identical to the previous map-based path: pairs are collected in
        // graph-visitation order (node ascending, then neighbor within row), and
        // a **stable** sort preserves that order within each equal key, so the
        // merge-sum below accumulates each pair's weight in exactly the order the
        // HashMap's `+=` did. Emitting `a <= b` (self-loop stored once) matches
        // what `from_edges` expects.
        let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(self.data.neighbors.len());
        for node in 0..self.node_count() {
            let start = self.data.node_ptrs[node];
            let end = self.data.node_ptrs[node + 1];
            for i in start..end {
                let neighbor = self.data.neighbors[i];
                // Upper triangle only to avoid double-counting.
                if node <= neighbor {
                    let w = self.data.weights[i];
                    let g1 = grouping.get_group(node);
                    let g2 = grouping.get_group(neighbor);
                    let (a, b) = if g1 <= g2 { (g1, g2) } else { (g2, g1) };
                    pairs.push((a, b, w));
                }
            }
        }
        pairs.sort_by_key(|&(a, b, _)| (a, b)); // stable — preserves visitation order

        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(pairs.len());
        for (a, b, w) in pairs {
            match edges.last_mut() {
                Some(last) if last.0 == a && last.1 == b => last.2 += w,
                _ => edges.push((a, b, w)),
            }
        }
        // Drop zero-weight self-loops (the old path skipped `w <= 0.0` self-loops;
        // cross edges were always pushed). Preserves the exact edge set.
        edges.retain(|&(a, b, w)| a != b || w > 0.0);

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
    /// Set of community IDs that have zero members and can be reused.
    /// Maintained by `set_group()`: inserted when a community becomes empty,
    /// removed when a community gains its first member.
    ///
    /// Uses a `BTreeSet` so reuse order is deterministic (smallest-id first)
    /// and `remove()` is O(log n). The prior `Vec` form relied on `.last()`
    /// with `rposition`/`swap_remove` for LIFO semantics, which produced a
    /// different tiebreak order on every insert sequence. The Leiden
    /// fingerprint shifts accordingly; the accelerator is still deterministic
    /// under a fixed seed.
    empty_communities: std::collections::BTreeSet<usize>,
}

impl Grouping {
    /// Each node in its own group: 0, 1, 2, …, n-1.
    fn create_isolated(n: usize) -> Self {
        Self {
            assignments: (0..n).collect(),
            group_count: n,
            group_sizes: vec![1; n],
            empty_communities: std::collections::BTreeSet::new(),
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
        let empty_communities: std::collections::BTreeSet<usize> =
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
            self.empty_communities.insert(old);
        }
        if group >= self.group_sizes.len() {
            self.group_sizes.resize(group + 1, 0);
        }
        // Target community was empty — remove from reuse pool.
        if self.group_sizes[group] == 0 {
            self.empty_communities.remove(&group);
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
    /// Per-node self-loop weight, precomputed once (the graph is immutable during
    /// optimization). Caches `graph.self_loop_weight(node)` so the local-move hot
    /// loop reads it in O(1) instead of re-binary-searching per candidate.
    self_loop_weights: Vec<f64>,
    /// Per-community sum of node strengths — maintained incrementally.
    community_strengths: Vec<f64>,
}

impl RBPartition {
    fn new(graph: LeidenGraph, grouping: Grouping, resolution: f64) -> Self {
        let two_m = 2.0 * graph.total_weight();
        let n = graph.node_count();

        // Per-node self-loop weight is immutable during optimization — compute
        // once. Node strength is read directly from the graph (`graph.strength`,
        // an O(1) slice index) rather than kept as a duplicate copy.
        let self_loop_weights: Vec<f64> = (0..n).map(|i| graph.self_loop_weight(i)).collect();

        // Build community strengths.
        let nc = grouping.group_count();
        let mut community_strengths = vec![0.0f64; nc];
        for node in 0..n {
            community_strengths[grouping.get_group(node)] += graph.strength(node);
        }

        Self {
            graph,
            grouping,
            resolution,
            two_m,
            self_loop_weights,
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

        let w_to_old = self.weight_to_comm(node, old_comm);
        let w_to_new = if new_community < self.grouping.group_count() {
            self.weight_to_comm(node, new_community)
        } else {
            0.0
        };

        self.diff_move_precomputed(node, new_community, w_to_old, w_to_new)
    }

    /// Quality delta for moving `node` to `new_community`, given the
    /// already-computed edge weights from `node` to its current community
    /// (`w_to_old`) and to `new_community` (`w_to_new`).
    ///
    /// This is the arithmetic core of [`diff_move`]; callers in the local-move
    /// hot loops build a `community → weight` map in a single neighbor pass and
    /// supply the two weights directly, avoiding the per-candidate
    /// [`weight_to_comm`] re-scan (O(deg²) → O(deg) per node).
    #[inline]
    fn diff_move_precomputed(
        &self,
        node: usize,
        new_community: usize,
        w_to_old: f64,
        w_to_new: f64,
    ) -> f64 {
        let old_comm = self.grouping.get_group(node);
        if new_community == old_comm {
            return 0.0;
        }
        if self.two_m == 0.0 {
            return 0.0;
        }

        let k_i = self.graph.strength(node);
        let self_weight = self.self_loop_weights[node];

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
        let k = self.graph.strength(node);
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
            self.community_strengths[c] += self.graph.strength(node);
        }
    }

    fn add_empty_community(&mut self) {
        let new_id = self.grouping.group_count;
        self.community_strengths.push(0.0);
        self.grouping.group_sizes.push(0);
        self.grouping.group_count += 1;
        self.grouping.empty_communities.insert(new_id);
    }

    /// Return a reusable empty community ID, or create a new one.
    /// Smallest-id-first reuse (BTreeSet iteration order).
    fn get_empty_community(&mut self) -> usize {
        if let Some(&id) = self.grouping.empty_communities.iter().next() {
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

    /// Partition `nodes` into conflict-free batches.
    ///
    /// Every node handed in is batched. This used to also filter out nodes with
    /// `is_stable[n]`, which was a *second* reader of that flag under a
    /// different meaning than the requeue guard's ("already optimal" vs "not in
    /// the queue") — the disagreement that produced the retirement bug. Under
    /// the restored invariant the filter is a no-op anyway: `pending` is grown
    /// only by the requeue arm, which clears the bit before pushing. The caller
    /// asserts that.
    fn create_batches(&self, mut remaining: Vec<usize>, graph: &LeidenGraph) -> Vec<Vec<usize>> {
        let mut batches = Vec::new();
        let mut locked = vec![false; graph.node_count()];

        while !remaining.is_empty() {
            let (batch, leftover) = self.extract_batch(&remaining, graph, &mut locked);
            if batch.is_empty() {
                break;
            }
            batches.push(batch);
            remaining = leftover;
            locked.fill(false);
        }
        batches
    }

    fn extract_batch(
        &self,
        candidates: &[usize],
        graph: &LeidenGraph,
        locked: &mut [bool],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut batch = Vec::new();
        let mut leftover = Vec::new();

        for &node in candidates {
            if self.has_conflict(node, graph, locked) {
                leftover.push(node);
            } else {
                batch.push(node);
                self.mark_locked(node, graph, locked);
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

    // Accumulate edge weight to each neighbor community in a single pass
    // (AllNeighComms). The map keys are the candidate communities and the
    // values are `weight_to_comm`, so each candidate's diff is O(1) below
    // instead of re-scanning the neighbor list (O(deg) vs O(deg²) per node).
    // One fresh per-node allocation, same as the prior per-node HashSet (the
    // `&RBPartition` shared borrow under `par_iter` rules out a reused buffer).
    // The map carries a heavier per-entry footprint than the old set and adds
    // an O(d log d) sort below, but both are dwarfed by dropping the O(deg²) scan.
    let mut comm_weights: HashMap<usize, f64> = HashMap::new();
    for (neighbor, edge_w) in partition.graph.neighbors(node) {
        *comm_weights
            .entry(partition.membership(neighbor))
            .or_insert(0.0) += edge_w;
    }
    let w_to_old = comm_weights.get(&current_comm).copied().unwrap_or(0.0);

    // Deterministic candidate order so the map's internal ordering never leaks
    // into the move sequence (stronger than the prior randomized-HashSet order).
    let mut candidates: Vec<usize> = comm_weights.keys().copied().collect();
    candidates.sort_unstable();

    let epsilon = 10.0 * f64::EPSILON;
    let mut best_comm = current_comm;
    let mut best_improv = epsilon;

    for &comm in &candidates {
        let w_to_new = comm_weights.get(&comm).copied().unwrap_or(0.0);
        let improv = partition.diff_move_precomputed(node, comm, w_to_old, w_to_new);
        if improv > best_improv {
            best_comm = comm;
            best_improv = improv;
        }
    }

    // Consider moving to an empty community (no members → w_to_new = 0).
    if let Some(ec) = empty_comm {
        if partition.group_size(current_comm) > 1 {
            let improv = partition.diff_move_precomputed(node, ec, w_to_old, 0.0);
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

            // Ask glibc to return freed heap pages to the OS. Each collapse
            // iteration allocates and drops large transient buffers in
            // `aggregate` (the `pairs`/`edges` Vecs, and the per-node adjacency
            // Vecs in `from_edges`); without trimming, glibc keeps the freed
            // pages mapped and RSS climbs to 50+ GB on graphs with 100K+ nodes.
            // (The earlier per-collapse HashMaps were replaced by a sort/merge
            // over pre-sized Vecs, which reduces — but does not eliminate — this
            // transient churn, so the trim is retained conservatively.)
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

        // Termination: every applied move below clears `epsilon` against the
        // *live* partition, so RB quality rises strictly on each move and is
        // bounded above, and `pending` only grows when a move is applied. The
        // `!made_move` break is therefore a backstop, not the mechanism — a
        // pass that applies nothing pushes nothing, so `pending` is already
        // empty and the `while` condition ends the loop on its own.
        let epsilon = 10.0 * f64::EPSILON;
        while !pending.is_empty() {
            let current: Vec<usize> = pending.drain(..).collect();
            debug_assert!(
                current.iter().all(|&n| !is_stable[n]),
                "queue invariant: everything in `pending` must be unstable"
            );
            let batches = batcher.create_batches(current, &graph);

            let mut made_move = false;
            for batch in batches {
                // This batch is leaving the queue. Mirror the sequential
                // reference (`move_nodes_sequential` below, C++
                // libleidenalg `Optimiser::move_nodes()`:672): `is_stable` is
                // an *in-queue* bit, not a local-optimum bit — `false` means
                // "sitting in the queue", which is exactly what the requeue
                // guard below tests. A node therefore becomes stable the
                // moment it is evaluated, whether or not it moves. Marking
                // only movers left a decliner at `false` after it had already
                // been drained out of `pending`, so the guard could never fire
                // for it and `create_batches` never saw it again: it was
                // retired after a single evaluation.
                //
                // Marked per batch, not per pass, so a node still awaiting its
                // own batch keeps reading as queued and is not enqueued twice.
                // `ConflictFreeBatcher` locks each admitted node *and its
                // neighbours*, so batch members are pairwise at graph distance
                // >= 3 and a move applied in this batch can never have a
                // batch-mate as a neighbour.
                for &node in &batch {
                    is_stable[node] = true;
                }

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
                    // of earlier moves in this batch (stale-read guard). Same floor
                    // as `evaluate_node` and the sequential path, so a float-noise
                    // "improvement" can never be applied.
                    let current_diff = partition.diff_move(m.node, m.to_comm);
                    if current_diff <= epsilon {
                        continue;
                    }

                    // Ensure community exists.
                    while partition.community_count() <= m.to_comm {
                        partition.add_empty_community();
                    }

                    total_improv += current_diff;
                    partition.move_node(m.node, m.to_comm);
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
            // Backstop only — see the termination note above. A pass that
            // applies no move pushes nothing, so `pending` is already empty.
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
        // Reuse the scratch map across iterations to avoid heap fragmentation
        // from repeated allocations (glibc never returns freed pages to the OS).
        let mut comm_weights: HashMap<usize, f64> = HashMap::new();
        let mut candidates: Vec<usize> = Vec::new();

        while let Some(v) = vertex_order.pop_front() {
            let v_comm = partition.membership(v);

            // Accumulate edge weight to each neighbor community in a single pass
            // (AllNeighComms, matching C++ lines 578-591). The map values are
            // `weight_to_comm`, so each candidate's diff below is O(1) instead
            // of a per-candidate neighbor re-scan (O(deg) vs O(deg²) per node).
            comm_weights.clear();
            for (neighbor, edge_w) in graph.neighbors(v) {
                *comm_weights
                    .entry(partition.membership(neighbor))
                    .or_insert(0.0) += edge_w;
            }
            let w_to_old = comm_weights.get(&v_comm).copied().unwrap_or(0.0);

            // Deterministic candidate order so the map's internal ordering never
            // leaks into the move sequence. The `v_comm` entry (if present) is a
            // no-op: `diff_move_precomputed` returns 0 for new == old.
            candidates.clear();
            candidates.extend(comm_weights.keys().copied());
            candidates.sort_unstable();

            let mut best_comm = v_comm;
            let mut best_improv = epsilon;

            for &comm in &candidates {
                let w_to_new = comm_weights.get(&comm).copied().unwrap_or(0.0);
                let improv = partition.diff_move_precomputed(v, comm, w_to_old, w_to_new);
                if improv > best_improv {
                    best_comm = comm;
                    best_improv = improv;
                }
            }

            // Consider moving to an empty community (matching C++ lines 615-634).
            // Uses get_empty_community() to reuse IDs instead of creating new ones.
            if self.config.consider_empty_community && partition.group_size(v_comm) > 1 {
                let empty_comm = partition.get_empty_community();
                // Empty community has no members → w_to_new = 0.
                let improv = partition.diff_move_precomputed(v, empty_comm, w_to_old, 0.0);
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
        // Reuse scratch across iterations to avoid per-vertex heap allocation.
        let mut comm_weights: HashMap<usize, f64> = HashMap::new();
        let mut candidates: Vec<usize> = Vec::new();

        for v in vertex_order {
            let v_comm = partition.membership(v);

            // Merge behaviour: only consider singleton communities.
            if partition.group_size(v_comm) != 1 {
                continue;
            }

            // Single neighbor pass: accumulate full edge weight per community
            // (matching `weight_to_comm` — computed over ALL neighbors), and
            // collect the constrained candidate set (AllNeighComms within the
            // same constrained group). The constraint filters which candidates
            // are *considered*, not how the weights are computed.
            let v_constrained = constrained_membership[v];
            comm_weights.clear();
            candidates.clear();
            for (neighbor, edge_w) in partition.graph.neighbors(v) {
                let nc = partition.membership(neighbor);
                *comm_weights.entry(nc).or_insert(0.0) += edge_w;
                if constrained_membership[neighbor] == v_constrained {
                    candidates.push(nc);
                }
            }
            // Deterministic, deduplicated candidate order.
            candidates.sort_unstable();
            candidates.dedup();
            let w_to_old = comm_weights.get(&v_comm).copied().unwrap_or(0.0);

            let mut best_comm = v_comm;
            let mut best_improv = 0.0;

            for &comm in &candidates {
                let w_to_new = comm_weights.get(&comm).copied().unwrap_or(0.0);
                let improv = partition.diff_move_precomputed(v, comm, w_to_old, w_to_new);
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
///   uses sequential moving that reproduces C++ libleidenalg's move-node
///   *ordering* (the refinement omits the paper's well-connectedness
///   admissibility conditions — see `LeidenConfig::refine_partition`).
///   `true` reaches a different partition; see `LeidenConfig::parallel` for
///   what it does and does not share with the sequential path.
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
            quality: 0.0,
            n_communities: 0,
        });
    }
    if n_nodes == 1 {
        return Ok(LeidenResult {
            membership: vec![0],
            modularity: 0.0,
            quality: 0.0,
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
    // the converged partition. Each optimize() call does a full hierarchical
    // pass: move → refine → aggregate → converge.
    //
    // When max_iterations == 0 (from n_iterations=-1, convergence mode):
    // repeat until no improvement, matching leidenalg's Optimiser.py:299-310.
    // When max_iterations > 0: run exactly that many outer passes. This is
    // the caller-supplied n_iterations (leidenalg's own n_iterations default
    // is 2); this function takes it as a required argument with no default.
    if max_iterations == 0 {
        // Converge: repeat until no improvement.
        // Safety cap at 100 to prevent infinite loops on pathological inputs.
        for _ in 0..100 {
            let improvement = optimizer.optimize(&mut partition)?;
            if improvement <= 0.0 {
                break;
            }
        }
    } else {
        for _ in 0..max_iterations {
            optimizer.optimize(&mut partition)?;
        }
    }

    let membership = partition.membership_vector();
    // One `quality()` call, not two. It allocates the community-member structure
    // and scans the whole adjacency, so deriving `modularity` from a second call
    // would pay an extra O(V + E) pass on every CPU Leiden — including every
    // resolution of a `clustering_agreement` sweep.
    let quality = partition.quality();
    let modularity = rb_modularity(quality, partition.two_m);
    let n_communities = partition.community_count();

    Ok(LeidenResult {
        membership,
        modularity,
        quality,
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

    /// OPT-3.1: the `community → weight` map built in one neighbor pass (the
    /// local-move hot loops) must reproduce `weight_to_comm`'s per-community
    /// scan exactly. Guards the precomputed-weight path against drift.
    #[test]
    fn test_comm_weights_map_matches_weight_to_comm() {
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
        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);
        let mut partition = RBPartition::new_singleton(graph, 1.0);

        // Build a non-trivial assignment: communities {0,1,2} and {3,4,5}.
        let c0 = partition.membership(0);
        let c3 = partition.membership(3);
        partition.move_node(1, c0);
        partition.move_node(2, c0);
        partition.move_node(4, c3);
        partition.move_node(5, c3);

        for node in 0..n {
            // Replicate the production accumulation pass.
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            for (neighbor, edge_w) in partition.graph.neighbors(node) {
                *comm_weights
                    .entry(partition.membership(neighbor))
                    .or_insert(0.0) += edge_w;
            }
            // Every accumulated entry must equal the scanning implementation.
            for (&comm, &w) in &comm_weights {
                let scanned = partition.weight_to_comm(node, comm);
                assert!(
                    (w - scanned).abs() < 1e-12,
                    "node {node} comm {comm}: map {w} != weight_to_comm {scanned}"
                );
            }
            // And communities absent from the map carry zero weight.
            for comm in [c0, c3] {
                if !comm_weights.contains_key(&comm) {
                    assert_eq!(partition.weight_to_comm(node, comm), 0.0);
                }
            }
        }
    }

    /// OPT-3.1: lock the exact output labels on a fixed-seed fixture so the
    /// sorted-candidate-order change is regression-guarded. Labels are
    /// bit-identical to the pre-optimization output on this tie-free fixture.
    #[test]
    fn test_local_move_label_golden() {
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

        let result = leiden(&indptr, &indices, &data, n, 1.0, 123, 0, false).unwrap();
        assert_eq!(result.membership, vec![0, 0, 0, 1, 1, 1]);
    }

    /// OPT-3.1: exercise the parallel local-move path (`evaluate_node`, which
    /// uses a fresh per-node weight map under `par_iter`). It must recover the
    /// two-clique structure and be deterministic across runs.
    #[test]
    fn test_parallel_local_move_recovers_clusters() {
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

        let r1 = leiden(&indptr, &indices, &data, n, 1.0, 42, 0, true).unwrap();
        let r2 = leiden(&indptr, &indices, &data, n, 1.0, 42, 0, true).unwrap();

        assert_eq!(
            r1.membership, r2.membership,
            "parallel path must be deterministic"
        );
        assert_eq!(r1.n_communities, 2, "should recover 2 communities");
        assert_eq!(r1.membership[0], r1.membership[1]);
        assert_eq!(r1.membership[0], r1.membership[2]);
        assert_eq!(r1.membership[3], r1.membership[4]);
        assert_eq!(r1.membership[3], r1.membership[5]);
        assert_ne!(r1.membership[0], r1.membership[3]);
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
    fn test_aggregate_collapses_to_expected_graph() {
        // 4 nodes, edges 0-1 (2.0), 1-2 (1.0), 2-3 (3.0). Group {0,1}→0, {2,3}→1.
        // Collapse: group-0 self-loop = intra edge 0-1 = 2.0; group-1 self-loop =
        // intra edge 2-3 = 3.0; cross 0↔1 = edge 1-2 = 1.0.
        let g = LeidenGraph::from_edges(&[(0, 1, 2.0), (1, 2, 1.0), (2, 3, 3.0)], vec![1.0; 4]);
        let grouping = Grouping::from_assignments(&[0, 0, 1, 1]);
        let agg = g.aggregate(&grouping);

        // Independent expected: feed the hand-derived collapsed edges through the
        // same canonicalizing `from_edges`. Node weights sum per group (all 1.0).
        let expected =
            LeidenGraph::from_edges(&[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)], vec![2.0, 2.0]);

        assert_eq!(agg.data.node_ptrs, expected.data.node_ptrs);
        assert_eq!(agg.data.neighbors, expected.data.neighbors);
        assert_eq!(agg.data.weights.len(), expected.data.weights.len());
        for (a, b) in agg.data.weights.iter().zip(&expected.data.weights) {
            assert_eq!(a.to_bits(), b.to_bits(), "edge weight {a} != expected {b}");
        }
        for (a, b) in agg.data.strengths.iter().zip(&expected.data.strengths) {
            assert_eq!(a.to_bits(), b.to_bits(), "strength {a} != expected {b}");
        }
        for (a, b) in agg
            .data
            .node_weights
            .iter()
            .zip(&expected.data.node_weights)
        {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(
            agg.data.total_weight.to_bits(),
            expected.data.total_weight.to_bits()
        );
    }

    #[test]
    fn test_aggregate_merge_sums_multiple_edges_per_pair() {
        // Stress the merge-sum: multiple original edges collapse into a single
        // (a,b) pair, so the sort/merge must accumulate ≥3 contributions per
        // key. Groups {0,1,2}->0, {3,4,5}->1. Weights are exact-summable
        // (powers of two) so the expected totals are unambiguous.
        //   group-0 self-loop (intra {0,1,2}): 1 + 2 + 4 = 7
        //   group-1 self-loop (intra {3,4,5}): 8 + 16 = 24
        //   cross 0<->1: 0.5 + 1.5 + 3.0 = 5
        let edges = [
            (0, 1, 1.0),
            (1, 2, 2.0),
            (0, 2, 4.0),
            (3, 4, 8.0),
            (4, 5, 16.0),
            (2, 3, 0.5),
            (2, 4, 1.5),
            (1, 5, 3.0),
        ];
        let g = LeidenGraph::from_edges(&edges, vec![1.0; 6]);
        let grouping = Grouping::from_assignments(&[0, 0, 0, 1, 1, 1]);
        let agg = g.aggregate(&grouping);
        let expected =
            LeidenGraph::from_edges(&[(0, 0, 7.0), (0, 1, 5.0), (1, 1, 24.0)], vec![3.0, 3.0]);

        assert_eq!(agg.data.node_ptrs, expected.data.node_ptrs);
        assert_eq!(agg.data.neighbors, expected.data.neighbors);
        for (a, b) in agg.data.weights.iter().zip(&expected.data.weights) {
            assert_eq!(a.to_bits(), b.to_bits(), "merged edge weight {a} != {b}");
        }
        for (a, b) in agg.data.strengths.iter().zip(&expected.data.strengths) {
            assert_eq!(a.to_bits(), b.to_bits(), "strength {a} != {b}");
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

    /// `edges_to_csr` plus the node count, so a fixture can hand its CSR
    /// straight to `leiden()`.
    fn edges_to_csr_with_n(
        edges: &[(usize, usize, f64)],
        n: usize,
    ) -> (Vec<i64>, Vec<i32>, Vec<f64>, usize) {
        let (indptr, indices, data) = edges_to_csr(edges, n);
        (indptr, indices, data, n)
    }

    // ── Review §7.10 — reported modularity is normalized ──────────────

    /// Zachary's Karate Club (igraph `Graph.Famous("Zachary")`): 34 nodes,
    /// 78 unit-weight edges, so `two_m = 156`.
    fn karate_club_csr() -> (Vec<i64>, Vec<i32>, Vec<f64>, usize) {
        const EDGES: [(usize, usize); 78] = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 17),
            (0, 19),
            (0, 21),
            (0, 31),
            (1, 2),
            (1, 3),
            (1, 7),
            (1, 13),
            (1, 17),
            (1, 19),
            (1, 21),
            (1, 30),
            (2, 3),
            (2, 7),
            (2, 27),
            (2, 28),
            (2, 32),
            (2, 9),
            (2, 8),
            (2, 13),
            (3, 7),
            (3, 12),
            (3, 13),
            (4, 6),
            (4, 10),
            (5, 6),
            (5, 10),
            (5, 16),
            (6, 16),
            (8, 30),
            (8, 32),
            (8, 33),
            (9, 33),
            (13, 33),
            (14, 32),
            (14, 33),
            (15, 32),
            (15, 33),
            (18, 32),
            (18, 33),
            (19, 33),
            (20, 32),
            (20, 33),
            (22, 32),
            (22, 33),
            (23, 25),
            (23, 27),
            (23, 32),
            (23, 33),
            (23, 29),
            (24, 25),
            (24, 27),
            (24, 31),
            (25, 31),
            (26, 29),
            (26, 33),
            (27, 33),
            (28, 31),
            (28, 33),
            (29, 32),
            (29, 33),
            (30, 32),
            (30, 33),
            (31, 32),
            (31, 33),
            (32, 33),
        ];
        let n = 34;
        let edges: Vec<(usize, usize, f64)> = EDGES.iter().map(|&(u, v)| (u, v, 1.0)).collect();
        edges_to_csr_with_n(&edges, n)
    }

    /// `2m` for the Karate Club: 78 unit-weight edges.
    const KARATE_TWO_M: f64 = 156.0;

    /// Zachary's observed factions — the canonical ground-truth 2-split.
    const KARATE_FACTIONS: [usize; 34] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1,
    ];

    /// The direct oracle: `igraph.Graph.Famous("Zachary").modularity(factions)`
    /// is `0.37146614069691`. Checked against igraph 1.0.0; leidenalg's own
    /// `partition.modularity` equals `partition.quality() / (2 * ecount)` on the
    /// same graph, which is the identity this normalization implements.
    const KARATE_FACTION_MODULARITY: f64 = 0.371_466_140_696_91;

    #[test]
    fn modularity_matches_igraph_on_the_karate_factions() {
        let (indptr, indices, data, n) = karate_club_csr();
        let graph = LeidenGraph::from_csr(&indptr, &indices, &data, n);
        let partition = RBPartition::new_with_membership(graph, &KARATE_FACTIONS, 1.0);
        let m = rb_modularity(partition.quality(), partition.two_m);
        assert!(
            (m - KARATE_FACTION_MODULARITY).abs() < 1e-12,
            "modularity {m} != igraph's {KARATE_FACTION_MODULARITY}"
        );
        // And the un-normalized value is exactly `2m` times it — the whole
        // content of §7.10 is that the two were being confused.
        assert!(
            (partition.quality() - m * KARATE_TWO_M).abs() < 1e-9,
            "quality {} != modularity {m} × two_m {KARATE_TWO_M}",
            partition.quality()
        );
        assert!(
            partition.quality() > 1.0,
            "premise: the raw quality is out of modularity's range ({})",
            partition.quality()
        );
    }

    /// The `[-0.5, 1]` claim holds **at γ = 1**, which is where the reported
    /// value is Newman modularity. `leiden_at_non_unit_resolution_is_not_newman_modularity`
    /// pins what happens away from 1, so neither half of the contract can regress
    /// silently into the other.
    #[test]
    fn leiden_reports_modularity_in_the_modularity_range_at_unit_resolution() {
        let (indptr, indices, data, n) = karate_club_csr();
        let result = leiden(&indptr, &indices, &data, n, 1.0, 42, 2, false).unwrap();

        assert!(
            (-0.5..=1.0).contains(&result.modularity),
            "reported modularity {} is outside [-0.5, 1] — that is the raw RB \
             quality, not modularity",
            result.modularity
        );
        assert!(
            result.modularity > 0.35,
            "modularity {} is far below leidenalg's 0.41979 optimum",
            result.modularity
        );
        assert!(
            (result.quality - result.modularity * KARATE_TWO_M).abs() < 1e-9,
            "quality {} != modularity {} × two_m {KARATE_TWO_M}",
            result.quality,
            result.modularity
        );
    }

    /// Separate from the range check above on purpose: that one guards the
    /// §7.10 normalization, this one guards the *search*.
    ///
    /// `leidenalg.find_partition(Zachary, RBConfigurationVertexPartition,
    /// resolution_parameter=1.0, seed=42, n_iterations=-1)` reports
    /// `quality() = 65.48717948717949`, `modularity = 0.41978961209730437`,
    /// 4 communities (leidenalg 0.10.x / igraph 1.0.0). SCX reaches the same
    /// partition, so the agreement is exact rather than approximate — measured,
    /// not assumed. A change to the Leiden search reds this and leaves the
    /// normalization test alone, which is the point of keeping them apart.
    #[test]
    fn leiden_reaches_leidenalgs_karate_optimum() {
        const LEIDENALG_QUALITY: f64 = 65.487_179_487_179_49;
        const LEIDENALG_MODULARITY: f64 = 0.419_789_612_097_304_37;
        let (indptr, indices, data, n) = karate_club_csr();
        let result = leiden(&indptr, &indices, &data, n, 1.0, 42, 2, false).unwrap();
        assert!(
            (result.quality - LEIDENALG_QUALITY).abs() < 1e-9,
            "RB quality {} != leidenalg's {LEIDENALG_QUALITY}",
            result.quality
        );
        assert!(
            (result.modularity - LEIDENALG_MODULARITY).abs() < 1e-12,
            "modularity {} != leidenalg's {LEIDENALG_MODULARITY}",
            result.modularity
        );
        assert_eq!(result.n_communities, 4);
    }

    /// Away from γ = 1 the reported value is the **generalized** RB objective on
    /// a per-edge scale, not Newman modularity, and it is not bounded by
    /// `[-0.5, 1]`.
    ///
    /// Measured with leidenalg 0.11.0 / igraph 1.0.0 on this graph, over the same
    /// partitions leidenalg itself finds:
    ///
    /// | γ | `quality / 2m` | `partition.modularity` (Newman) |
    /// |---|---|---|
    /// | 1  | `0.419790` | `0.419790` |
    /// | 20 | `-0.996055` | `-0.049803` |
    /// | 50 | `-2.490138` | `-0.049803` |
    ///
    /// Pinned here because the docs previously promised `[-0.5, 1]`
    /// unconditionally. Asserting only that the value is out of range, not its
    /// exact magnitude: SCX's search need not find leidenalg's partition at a
    /// resolution that drives the graph to singletons, and pinning the magnitude
    /// would make this a test of the search rather than of the scale.
    #[test]
    fn leiden_at_non_unit_resolution_is_not_newman_modularity() {
        let (indptr, indices, data, n) = karate_club_csr();
        let result = leiden(&indptr, &indices, &data, n, 50.0, 42, 2, false).unwrap();
        assert!(
            result.modularity < -0.5,
            "at γ = 50 the generalized RB objective should fall below Newman's \
             lower bound; got {} (n_communities {})",
            result.modularity,
            result.n_communities
        );
        // The identity itself still holds — it is only the *bound* that does not.
        assert!(
            (result.quality - result.modularity * KARATE_TWO_M).abs() < 1e-9,
            "quality {} != modularity {} × two_m {KARATE_TWO_M}",
            result.quality,
            result.modularity
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

    /// Deterministic planted-partition graph: 4 blocks of 12 nodes, unit
    /// weights, intra-block edge probability 0.45 and inter-block 0.06.
    fn planted_graph(seed: u64) -> (LeidenGraph, usize) {
        use rand::Rng;
        const N_COMM: usize = 4;
        const SZ: usize = 12;
        let n = N_COMM * SZ;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                let p = if u / SZ == v / SZ { 0.45 } else { 0.06 };
                if rng.gen::<f64>() < p {
                    edges.push((u, v, 1.0));
                }
            }
        }
        let (indptr, indices, data) = edges_to_csr(&edges, n);
        (LeidenGraph::from_csr(&indptr, &indices, &data, n), n)
    }

    /// Run one local-moving pass from a **non-singleton** start membership and
    /// report `(final quality, nodes that still have a beneficial move)`.
    ///
    /// The non-singleton start is the whole point, and it is what makes this
    /// deterministic rather than what makes it possible. At `resolution = 1.0`
    /// a **non-isolated** node whose community is still a singleton, with no
    /// self-loop, has a strictly improving move: `diff_move` reduces to
    /// `2·(w(v,C) − γ·k_v·k_C/2m)`, and summing over the communities holding
    /// v's neighbours gives `≥ 2·k_v²/2m > 0`. So the *first* evaluation of
    /// every such node moves it, and an all-singleton first pass produces no
    /// decliner to retire. (An isolated node — `k_v == 0` — is the exception,
    /// and is also the case that loses nothing by being retired.)
    ///
    /// That bound says nothing about later passes. Once pass 1 has moved
    /// nodes, communities are no longer singletons, the *leave* term is live,
    /// and a node requeued by a neighbour's move can decline and be retired —
    /// at the first level, not only at the aggregate levels where
    /// `refine_and_collapse` hands down a `new_with_membership` partition. So
    /// the public `leiden(..., parallel = true)` path does reach the bug; it
    /// just reaches it after a shuffle, a batching pass and a requeue, which
    /// is not a fixture you can pin. Injecting the start membership puts a
    /// decliner in front of the first batch on purpose.
    ///
    /// It is also where the signal is. End to end through `leiden()` the
    /// multilevel loop re-runs local moving on every level and recovers most
    /// of the damage: over 60 planted graphs across five block shapes, the
    /// worst pre-fix parallel-vs-sequential modularity ratio was 0.986 (0.9983
    /// after), against the 0.78 this sweep sees on a single pass. That is why
    /// the regression lives here and not on the public entry point.
    fn local_move_outcome(
        graph: &LeidenGraph,
        membership: &[usize],
        seed: u64,
        parallel: bool,
    ) -> (f64, Vec<usize>) {
        let n = membership.len();
        let cfg = LeidenConfig {
            resolution: 1.0,
            seed: Some(seed),
            parallel,
            ..Default::default()
        };
        let consider_empty = cfg.consider_empty_community;
        let mut optimizer = LeidenOptimizer::new(cfg);
        let mut partition = RBPartition::new_with_membership(graph.clone(), membership, 1.0);
        if parallel {
            optimizer.move_nodes_parallel(&mut partition).unwrap();
        } else {
            optimizer.move_nodes_sequential(&mut partition).unwrap();
        }
        // Offer the stranded-node probe the same candidate set the optimizer
        // had, empty community included — the empty-community arm also returns
        // `None`, so it produces decliners too and would otherwise be invisible
        // here while `leiden()` leaves it on.
        let empty_comm = consider_empty.then(|| partition.get_empty_community());
        let stranded = (0..n)
            .filter(|&v| evaluate_node(v, &partition, empty_comm).is_some())
            .collect();
        (partition.quality(), stranded)
    }

    /// 32 seeded planted graphs, each started from a scrambled (non-singleton)
    /// membership, run through both local-moving paths.
    ///
    /// Bit-reproducible: the graphs and memberships come from a seeded
    /// `ChaCha8Rng`, `evaluate_batch` is an order-preserving
    /// `par_iter().collect()`, and the apply loop that follows is serial.
    /// Verified identical at `RAYON_NUM_THREADS` 1, 2, 4, 8 and 16, which is
    /// why the assertions below sit close to the measured values rather than
    /// leaving flake margin.
    ///
    /// Caveat on coverage: `consider_empty_community` is left at its
    /// `leiden()` default of `true` so the optimizer sees the same candidate
    /// set production does, but on this fixture that arm never changes an
    /// outcome — the totals are identical with it off. The empty-community
    /// decline path is exercised by the production code change, not by this
    /// sweep.
    fn parallel_vs_sequential_sweep() -> (f64, f64, usize, usize) {
        use rand::Rng;
        let (mut q_par, mut q_seq) = (0.0, 0.0);
        let (mut stranded_par, mut stranded_seq) = (0usize, 0usize);
        for seed in 0u64..32 {
            let (graph, n) = planted_graph(seed);
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
            let membership: Vec<usize> = (0..n).map(|_| rng.gen_range(0..4)).collect();

            let (qp, sp) = local_move_outcome(&graph, &membership, seed, true);
            let (qs, ss) = local_move_outcome(&graph, &membership, seed, false);
            q_par += qp;
            q_seq += qs;
            stranded_par += sp.len();
            stranded_seq += ss.len();
        }
        (q_par, q_seq, stranded_par, stranded_seq)
    }

    /// §7.9: `move_nodes_parallel` used to write `is_stable[node] = true` only
    /// on the applied-move branch. A node that declined therefore kept
    /// `is_stable == false` — which under this invariant means "still queued" —
    /// while having already been drained out of `pending`, so the neighbour
    /// requeue guard could never fire for it, and the `create_batches` of the
    /// time, which skipped stable nodes, never saw it again. (That filter is
    /// gone; it was the second, contradictory reader of the flag.) It was retired after exactly one
    /// evaluation, against unbounded re-evaluation in the sequential
    /// reference.
    ///
    /// Measured on this sweep before the fix: total RB quality 3434.6 against
    /// sequential's 4404.9 — the parallel path was giving up **22 %** of the
    /// quality the same move kernel reaches sequentially. After the fix,
    /// 4376.6 (99.4 %).
    #[test]
    fn test_parallel_local_move_quality_matches_sequential() {
        let (q_par, q_seq, _, _) = parallel_vs_sequential_sweep();
        let ratio = q_par / q_seq;
        // Measured 0.9936. The bar sits just under it rather than at the ~0.95
        // a "clearly better than the 0.78 regression" reading would suggest:
        // the sweep is bit-reproducible (see above), so a loose bar buys no
        // robustness and would pass a large partial regression.
        assert!(
            ratio >= 0.99,
            "parallel local moving reached only {:.2}% of sequential RB quality \
             ({q_par:.1} vs {q_seq:.1}); measured 99.36%, pre-fix 77.97%",
            ratio * 100.0
        );
    }

    /// The same sweep, read as the mechanism rather than the outcome: count
    /// the nodes left holding a beneficial move once local moving has
    /// returned. The sequential reference is not at zero either — its requeue
    /// guard also skips a neighbour already sitting in the destination
    /// community — so the bar is "within an additive 8 of sequential", not
    /// "none".
    ///
    /// Pre-fix: 153 stranded on the parallel path against sequential's 32.
    /// Post-fix: 34 against 32.
    #[test]
    fn test_parallel_local_move_reevaluates_decliners() {
        let (_, _, stranded_par, stranded_seq) = parallel_vs_sequential_sweep();
        // Measured 34 against sequential's 32. An additive slack of 8, not a
        // 2x multiple: the sweep is bit-reproducible, and `2 * stranded_seq`
        // would still pass at 64 stranded nodes — most of the way back to the
        // pre-fix 153.
        assert!(
            stranded_par <= stranded_seq + 8,
            "parallel local moving stranded {stranded_par} nodes with a beneficial move \
             still available against sequential's {stranded_seq}; measured 34 vs 32, \
             pre-fix 153 vs 32"
        );
    }
}
