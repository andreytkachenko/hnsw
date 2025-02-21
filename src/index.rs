use std::collections::{BinaryHeap, HashSet};

use crate::layer::{DistEntry, HnswLayer};

pub trait Scalar: simsimd::SpatialSimilarity + std::fmt::Debug + Copy + Clone + 'static {}
impl<T: simsimd::SpatialSimilarity + std::fmt::Debug + Copy + Clone + 'static> Scalar for T {}

pub struct HnswIndex<S: Scalar> {
    // Слои графа
    pub layers: Vec<HnswLayer<S>>,

    dimensions: usize,

    // Макс. число связей
    m: usize,

    // Параметр построения
    ef_construction: usize,

    // Коэффициент для выбора слоя
    ml: f64,

    // Total count of nodes in the index
    total_nodes: usize,
}

impl<S: Scalar> HnswIndex<S> {
    pub fn new(dimensions: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            layers: vec![HnswLayer::new(dimensions)],
            m,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            total_nodes: 0,
            dimensions,
        }
    }

    /// insert_node inserts a new node into the world.
    /// id must be fully unique in the World
    // 1. pick the level at which to insert the node
    // 2. find the M nearest neighbors for the node at the chosen level
    // 3. connect the new node to the neighbors and on all lower levels
    // 4. recursively connect the new node to the neighbors' neighbors
    // 5. if the new node has no connections, add it to the graph at level 0
    pub fn insert<V: AsRef<[S]>>(&mut self, id: u64, vector: V) {
        let vector = vector.as_ref();

        if self.layers.len() < self.calculate_max_level(self.total_nodes + 1) {
            self.layers.push(HnswLayer::new(self.dimensions));
        }

        let node_level = self.pick_node_level();
        let node = self.layers[node_level].create_node(id, vector, self.ef_construction, self.m);

        // If this is the first node, initialize it as the entrypoint for all levels
        if self.layers.len() == 1 && self.layers[0].nodes.len() == 1 {
            return;
        }

        // Start from the top-level entry point
        let mut current_layer = 0;
        let mut current_node = 0;
        for lid in 0..node_level - 1 {
            let candidates =
                self.layers[lid].search(vector.as_ref(), current_node, self.ef_construction, false);

            // Pick the closest candidate as the new entry point for the next level down
            if let Some(closest) = candidates.min_by(|&node_a, &node_b| {
                let dist_a = self.layers[lid].dist_to(vector, node_a.1);
                let dist_b = self.layers[lid].dist_to(vector, node_b.1);

                dist_a.partial_cmp(&dist_b).unwrap()
            }) {
                current_layer = lid;
                current_node = closest.1;
            }
        }

        // Now we are at the correct insertion level (node_level), perform a local search here
        let results = self.layers[current_layer]
            .search(vector, current_node, self.ef_construction, false)
            .map(|DistEntry(_, id)| id);

        self.layers[node_level].create_connections(node, results);

        // prune the node if it has more than M connections
        self.layers[node_level].prune_connections(node, self.m);
    }

    /// beam_search performs a beam search for the k nearest neighbours to the query vector
    pub fn search<Q: AsRef<[S]>>(&self, query: Q, ef: usize) -> impl Iterator<Item = (f32, u64)> {
        let query = query.as_ref();

        let mut candidates: BinaryHeap<DistEntry<usize>> = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut final_candidates = BinaryHeap::new();

        candidates.push(self.layers[0].dist_to(query, 0));

        for layer in self.layers.iter() {
            loop {
                let Some(DistEntry(dist, candidate)) = candidates.pop() else {
                    break;
                };

                if visited.contains(&candidate) {
                    continue;
                }
                visited.insert(candidate);

                // Add the current candidate to final candidates
                final_candidates.push(DistEntry(-dist, layer.node_ids[candidate]));

                // Combine search results on the level
                for DistEntry(_, node) in layer.search(query, candidate, ef, false) {
                    if !visited.contains(&node) {
                        candidates.push(layer.dist_to(query, node));
                    }
                }
            }
        }

        // Return the best candidates we've found
        std::iter::from_fn(move || {
            final_candidates
                .pop()
                .map(|DistEntry(dist, id)| (-dist.into_inner(), id))
        })
    }

    ///
    /// pick_node_level picks the level at which a new node should be inserted
    /// based on the probabalistic insertion strategy.
    pub(crate) fn pick_node_level(&self) -> usize {
        let mut level = 0;

        while rand::random::<f32>() < (1.0 - self.ml as f32) && level < self.layers.len() - 1 {
            level += 1;
        }

        level
    }

    /// calculate_max_level calculates the maximum level of the HNSW graph based
    /// on the number of nodes and the maximum number of connections per node
    #[inline]
    fn calculate_max_level(&self, n: usize) -> usize {
        // p = 1/m
        // max_level ≈ log(n)/log(m)
        (((n as f64).ln() * self.ml).ceil() as usize).max(1)
    }
}
