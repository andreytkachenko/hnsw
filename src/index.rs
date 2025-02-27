use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    f32,
};

use leapfrog::Value;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;

use crate::DistEntry;
use boxcar as bc;

#[derive(Debug, Clone, Copy)]
pub struct Wrap(pub f32);

impl Value for Wrap {
    fn is_redirect(&self) -> bool {
        self.0 == f32::INFINITY
    }

    fn is_null(&self) -> bool {
        self.0 == f32::NAN
    }

    fn redirect() -> Self {
        Wrap(f32::INFINITY)
    }

    fn null() -> Self {
        Wrap(f32::NAN)
    }
}

pub trait Scalar: simsimd::SpatialSimilarity + std::fmt::Debug + Copy + Clone + 'static {}
impl<T: simsimd::SpatialSimilarity + std::fmt::Debug + Copy + Clone + 'static> Scalar for T {}

pub struct CandidateHeap {
    visited: HashSet<usize>,
    inner: BinaryHeap<DistEntry<usize>>,
    ef: usize,
}

impl CandidateHeap {
    #[inline]
    pub fn push(&mut self, item: DistEntry<usize>) -> Option<DistEntry<usize>> {
        if self.visited.contains(&item.1) {
            return None;
        }

        self.inner.push(item);

        if self.inner.len() > self.ef {
            self.inner.pop()
        } else {
            None
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &DistEntry<usize>> {
        self.inner.iter()
    }

    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = DistEntry<usize>> {
        self.inner.into_iter()
    }

    pub fn new(ef: usize) -> Self {
        Self {
            inner: BinaryHeap::with_capacity(ef + 1),
            ef,
            visited: HashSet::new(),
        }
    }
}

pub struct HnswIndex<S: Scalar> {
    /// Node Id -> Index map
    pub nodes: HashMap<u64, usize>,

    /// Node Index -> Id map
    pub ids: bc::Vec<u64>,

    /// Node vector slices strided by dimension
    pub vectors: RwLock<Vec<S>>,

    /// Node vector slices strided by dimension
    ///    let conn_idx = connections[LayerIdx][NodeIdx];
    pub connections: bc::Vec<bc::Vec<(usize, usize)>>,

    /// Node cached connections distances (structure exact to connection)
    pub distance_cache: leapfrog::LeapMap<(usize, usize), Wrap>,

    /// Pending nodes to prue
    pub pending_indexes: bc::Vec<usize>,

    /// Level entrypoints nodes
    pub level_entrypoints: bc::Vec<usize>,

    pub level_ids: bc::Vec<bc::Vec<usize>>,

    /// Dimensions used as stride in calculation of subslice in node_vector
    dimensions: usize,

    // Макс. число связей
    m: usize,

    // Параметр построения
    ef_construction: usize,

    // Коэффициент для выбора слоя
    ml: f64,
}

impl<S: Scalar> HnswIndex<S> {
    pub fn new(dimensions: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            m,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            dimensions,
            nodes: HashMap::new(),
            ids: bc::Vec::new(),
            vectors: RwLock::new(Vec::new()),
            connections: bc::vec![],
            distance_cache: leapfrog::LeapMap::new(),
            pending_indexes: bc::Vec::new(),
            level_entrypoints: bc::Vec::new(),
            level_ids: bc::Vec::new(),
        }
    }

    fn create_node(&self, id: u64, vector: &[S]) -> usize {
        assert_eq!(vector.len(), self.dimensions);

        if self.nodes.contains_key(&id) {
            panic!("node with id `{}` already exists!", id);
        }

        let mut guard = self.vectors.write();
        guard.extend_from_slice(vector);

        let index = self.ids.push(id);
        let conn_idx = self.connections.push(bc::vec![]);
        debug_assert_eq!(conn_idx, index);

        index
    }

    /// insert_node inserts a new node into the world.
    /// id must be fully unique in the World
    // 1. pick the level at which to insert the node
    // 2. find the M nearest neighbors for the node at the chosen level
    // 3. connect the new node to the neighbors and on all lower levels
    // 4. recursively connect the new node to the neighbors' neighbors
    // 5. if the new node has no connections, add it to the graph at level 0
    pub fn insert<V: AsRef<[S]>>(&self, id: u64, vector: V) {
        let vector = vector.as_ref();

        let node = self.create_node(id, vector);
        let levels = self.calculate_max_level(self.ids.count());

        if self.level_entrypoints.count() < levels {
            self.level_entrypoints.push(node);
            self.level_ids.push(bc::Vec::new());
        }

        let node_level = self.pick_node_level();

        for l in 0..node_level {
            self.level_ids[l].push(node);
        }

        // stage for pruning
        self.pending_indexes.push(node);

        // If this is the first node exit, otherwise initialize it as the entrypoint for all levels
        if self.ids.count() == 1 {
            return;
        }

        let gobal_entrypoint = self.entrypoint();

        let entrypoint = if node_level == self.level_entrypoints.count() - 1 {
            gobal_entrypoint
        } else {
            self.search_inner(vector, node_level + 1, 1, false)
                .map(|DistEntry(_, id)| id)
                .next()
                .unwrap_or(gobal_entrypoint)
        };

        let mut heap = CandidateHeap::new(self.ef_construction);
        // Now we are at the correct insertion level (node_level), perform a local search here
        self.search_layer(
            &mut heap,
            node_level,
            vector,
            entrypoint,
            self.ef_construction,
            false,
        );

        self.add_connections(
            node,
            node_level,
            heap.into_iter().map(|DistEntry(_, idx)| idx),
        );
    }

    #[inline]
    fn entrypoint(&self) -> usize {
        self.level_entrypoints[self.level_entrypoints.count() - 1]
    }

    pub fn search_layer(
        &self,
        heap: &mut CandidateHeap,
        layer: usize,
        query: &[S],
        entrypoint: usize,
        ef: usize,
        debug: bool,
    ) -> DistEntry<usize> {
        let mut visited = HashSet::with_capacity(ef * ef * 4);
        let mut candidates = BinaryHeap::with_capacity(ef);

        let mut min_node = self.dist_to(query, entrypoint);
        visited.insert(entrypoint);
        candidates.push(Reverse(min_node));

        while let Some(Reverse(candidate)) = candidates.pop() {
            if debug {
                println!("#{} {}", candidate.1, candidate.0,);
            }

            if candidate < min_node {
                min_node = candidate;
            }

            if let Some(c) = heap.push(candidate) {
                if c == candidate {
                    continue;
                }
            }

            for (_, &(lvl, neighbor)) in &self.connections[candidate.1] {
                if lvl == layer && !visited.contains(&neighbor) {
                    visited.insert(neighbor);

                    candidates.push(Reverse(self.dist_to(query, neighbor)));
                }
            }
        }

        min_node
    }

    /// search_inner performs a beam search for the nearest neighbours to the query vector
    fn search_inner(
        &self,
        query: &[S],
        level: usize,
        ef: usize,
        debug: bool,
    ) -> impl Iterator<Item = DistEntry<usize>> {
        let mut candidate = self.entrypoint();
        let mut heap = CandidateHeap::new(ef);

        for layer in (level..self.level_entrypoints.count()).rev() {
            candidate = self
                .search_layer(&mut heap, layer, query, candidate, ef, debug)
                .1;
        }

        // Return the best candidates we've found
        heap.into_iter()
    }

    pub fn search<Q: AsRef<[S]>>(
        &self,
        query: Q,
        ef: usize,
    ) -> impl Iterator<Item = (f32, u64)> + '_ {
        self.search_inner(query.as_ref(), 0, ef, true)
            .map(|DistEntry(dist, idx)| (dist.into_inner(), self.ids[idx]))
    }

    ///
    /// pick_node_level picks the level at which a new node should be inserted
    /// based on the probabalistic insertion strategy.
    pub(crate) fn pick_node_level(&self) -> usize {
        let mut level = 0;

        while rand::random::<f32>() < (1.0 - self.ml as f32)
            && level < self.level_entrypoints.count() - 1
        {
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

    #[inline]
    pub fn dist_to(&self, query: &[S], entry: usize) -> DistEntry<usize> {
        let guard = self.vectors.read();
        let offset = entry * self.dimensions;

        DistEntry(
            OrderedFloat(S::l2(query, &guard[offset..offset + self.dimensions]).unwrap() as f32),
            entry,
        )
    }

    pub fn prune_connections(&mut self) {
        let mut heaps = vec![BinaryHeap::with_capacity(self.m + 1); self.level_entrypoints.count()];

        for (_, &node_idx) in self.pending_indexes.iter() {
            for (_, &(lvl, neighbour)) in &self.connections[node_idx] {
                let dist = OrderedFloat(self.distance(neighbour, node_idx));
                heaps[lvl].push(DistEntry(dist, neighbour));
                if heaps[lvl].len() > self.m {
                    heaps.pop();
                }
            }

            let node_onnections = self.connections.get_mut(node_idx).unwrap();
            node_onnections.clear();

            for (lvl, heap) in heaps.iter().enumerate() {
                for &DistEntry(_, idx) in heap.iter() {
                    node_onnections.push((lvl, idx));
                }
            }
        }

        self.pending_indexes.clear();
    }

    fn add_connections(&self, node: usize, level: usize, others: impl Iterator<Item = usize>) {
        for idx in others {
            if self.connections[node]
                .iter()
                .find(|(_, &e)| e == (level, idx))
                .is_none()
            {
                self.connections[node].push((level, idx));
            }

            if self.connections[idx]
                .iter()
                .find(|(_, &e)| e == (level, node))
                .is_none()
            {
                self.connections[idx].push((level, node));
            }

            let _ = self.distance(idx, node);
        }
    }

    #[inline]
    fn distance(&self, node: usize, other: usize) -> f32 {
        let key = if node > other {
            (node, other)
        } else {
            (other, node)
        };

        if self.distance_cache.contains_key(&key) {
            self.distance_cache.get(&key).unwrap().value().unwrap().0
        } else {
            let node_offset = node * self.dimensions;
            let other_offset = other * self.dimensions;
            let guard = self.vectors.read();
            let a = &guard[node_offset..node_offset + self.dimensions];
            let b = &guard[other_offset..other_offset + self.dimensions];
            S::l2(a, b).unwrap() as f32
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{index::HnswIndex, DistEntry};

    #[test]
    fn test_layer_search() {
        let index: HnswIndex<f32> = HnswIndex::new(2, 8, 4);

        for i in 0..32 {
            for j in 0..32 {
                index.insert(i * 32 + j, &[i as f32, j as f32]);
            }
        }

        for lvl in 0..index.level_entrypoints.count() {
            println!("layer #{lvl}");
            println!("  entrypoint: {}", index.level_entrypoints[lvl]);
            println!("  nodes: {}", index.level_ids[lvl].count());
            println!();
        }

        let mut res = index
            .search_inner(&[31.5f32, 31.5f32], 0, 4, true)
            .map(|DistEntry(_, id)| id)
            .collect::<Vec<_>>();

        res.sort();
        println!("{:?}", res);

        assert_eq!(res, vec![990, 991, 1022, 1023]);
    }
}
