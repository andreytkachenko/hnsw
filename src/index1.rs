use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    f32,
    sync::atomic::{AtomicU8, Ordering},
};

use leapfrog::Value;
use ordered_float::OrderedFloat;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use crate::DistEntry;
use boxcar as bc;

type NodeId = u32;

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
    visited: HashSet<NodeId>,
    inner: BinaryHeap<DistEntry<NodeId>>,
    ef: u32,
}

impl CandidateHeap {
    #[inline]
    pub fn push(&mut self, item: DistEntry<NodeId>) -> Option<DistEntry<NodeId>> {
        if self.visited.contains(&item.1) {
            return None;
        }

        self.inner.push(item);

        if self.inner.len() > self.ef as usize {
            self.inner.pop()
        } else {
            None
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &DistEntry<NodeId>> {
        self.inner.iter()
    }

    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = DistEntry<NodeId>> {
        self.inner.into_iter()
    }

    pub fn new(ef: u32) -> Self {
        Self {
            inner: BinaryHeap::with_capacity(ef as usize + 1),
            ef,
            visited: HashSet::new(),
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.inner.clear();
        self.visited.clear();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    // Оцениваемое кол-во элементов (нужно для более равномерного распределения слоев)
    pub min_count: usize,

    // Макс. число связей
    pub m: u32,

    // Параметр построения
    pub ef_construction: u32,

    /// Dimensions used as stride in calculation of subslice in node_vector
    pub dimensions: u32,
}

pub struct HnswIndex<S: Scalar> {
    /// Node Id -> Index map
    pub nodes: HashMap<u64, NodeId>,

    /// Node Index -> Id map
    pub ids: bc::Vec<u64>,

    /// Node vector slices strided by dimension
    pub vectors: bc::Vec<RwLock<Vec<S>>>,
    pub vector_offsets: bc::Vec<usize>,

    /// Node vector slices strided by dimension
    ///    let conn_idx = connections[LayerIdx][NodeIdx];
    pub connections: bc::Vec<bc::Vec<(u32, NodeId)>>,
    pub parent_connections: bc::Vec<NodeId>,

    /// Node cached connections distances (structure exact to connection)
    pub distance_cache: leapfrog::LeapMap<(NodeId, NodeId), Wrap>,

    /// Pending nodes to prue
    pub pending_indexes: bc::Vec<NodeId>,

    pub level_ids: bc::Vec<bc::Vec<NodeId>>,
    pub node_levels: bc::Vec<u8>,

    pub max_level: AtomicU8,

    pub config: HnswConfig,

    // Коэффициент для выбора слоя
    ml: f64,
}

impl<S: Scalar> HnswIndex<S> {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            ml: 1.0 / (config.m as f64).ln(),
            nodes: HashMap::new(),
            ids: bc::Vec::new(),
            vectors: bc::vec![RwLock::new(Vec::new())],
            vector_offsets: bc::vec![],
            connections: bc::vec![],
            parent_connections: bc::vec![],
            distance_cache: leapfrog::LeapMap::new(),
            pending_indexes: bc::Vec::new(),
            level_ids: bc::vec![bc::Vec::new()],
            node_levels: bc::Vec::new(),
            max_level: AtomicU8::new(0),
            config,
        }
    }

    #[inline]
    fn push_vector(&self, lvl: u8, vec: &[S]) -> usize {
        assert_eq!(vec.len(), self.config.dimensions as usize);

        let mut guard = self.vectors[lvl as usize].write();
        let offset = guard.len();
        guard.extend_from_slice(vec);
        drop(guard);

        self.vector_offsets.push(offset)
    }

    #[inline]
    fn get_vector(&self, node: NodeId) -> MappedRwLockReadGuard<'_, [S]> {
        let lvl = self.node_levels[node as usize];
        let offset = self.vector_offsets[node as usize];

        RwLockReadGuard::map(self.vectors[lvl as usize].read(), |x| {
            &x[offset..offset + self.config.dimensions as usize]
        })
    }

    fn create_node(&self, id: u64, vector: &[S], level: u8) -> NodeId {
        if self.nodes.contains_key(&id) {
            panic!("node with id `{}` already exists!", id);
        }

        let index = self.ids.push(id) as NodeId;
        let lvl_index = self.node_levels.push(level) as NodeId;
        let vec_index = self.push_vector(level, vector) as NodeId;
        let conn_idx = self.connections.push(bc::vec![]) as NodeId;

        debug_assert_eq!(conn_idx, index);
        debug_assert_eq!(vec_index, index);
        debug_assert_eq!(lvl_index, index);

        self.level_ids[level as usize].push(index);

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

        self.update_max_level();

        let node_level = self.pick_node_level();
        let node = self.create_node(id, vector, node_level);

        // stage for pruning
        self.pending_indexes.push(node);

        // If this is the first node exit, otherwise initialize it as the entrypoint for all levels
        if self.ids.count() == 1 {
            return;
        }

        let gobal_entrypoint = 0;

        let entrypoint = if node_level == self.max_level.load(Ordering::Relaxed) {
            gobal_entrypoint
        } else {
            self.search_inner(vector, node_level + 1, 1, false)
                .map(|DistEntry(_, id)| id)
                .next()
                .unwrap_or(gobal_entrypoint)
        };

        let mut heap = CandidateHeap::new(self.config.ef_construction);
        // Now we are at the correct insertion level (node_level), perform a local search here
        self.search_layer(
            Some(&mut heap),
            node_level,
            vector,
            entrypoint,
            self.config.ef_construction,
            false,
        );
        self.add_connections(
            node,
            node_level as _,
            heap.iter().map(|&DistEntry(_, idx)| idx),
        );

        if node_level > 0 {
            self.search_layer(None, node_level - 1, vector, entrypoint, 1, false);

            self.add_connections(
                node,
                node_level as _,
                heap.into_iter().map(|DistEntry(_, idx)| idx),
            );
        }
    }

    pub fn search_layer(
        &self,
        heap: Option<&mut CandidateHeap>,
        layer: u8,
        query: &[S],
        entrypoint: NodeId,
        ef: u32,
        debug: bool,
    ) -> DistEntry<NodeId> {
        if debug {
            println!("layer #{layer} ({entrypoint}):");
            println!("{:?}", self.level_ids[layer as usize]);
            println!()
        }

        let mut visited = HashSet::with_capacity(ef as usize * ef as usize * 4);
        let mut candidates = BinaryHeap::with_capacity(ef as usize);

        let mut min_node = DistEntry(OrderedFloat(f32::INFINITY), entrypoint);
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
                    if debug {
                        println!("x");
                    }

                    break;
                }
            }

            for (_, &(lvl, neighbor)) in &self.connections[candidate.1 as usize] {
                if lvl == layer as u32 && !visited.contains(&neighbor) {
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
        level: u8,
        ef: u32,
        debug: bool,
    ) -> impl Iterator<Item = DistEntry<NodeId>> {
        let mut candidate = 0;
        let mut heap = CandidateHeap::new(ef as u32);

        if debug {
            println!("layers: {}", self.max_level.load(Ordering::Relaxed));
        }

        for layer in (level..=self.max_level.load(Ordering::Relaxed)).rev() {
            candidate = self
                .search_layer(Some(&mut heap), layer, query, candidate, ef, debug)
                .1;
        }

        // Return the best candidates we've found
        heap.into_iter()
    }

    pub fn search<Q: AsRef<[S]>>(
        &self,
        query: Q,
        ef: u32,
    ) -> impl Iterator<Item = (f32, u64)> + '_ {
        self.search_inner(query.as_ref(), 0, ef, true)
            .map(|DistEntry(dist, idx)| (dist.into_inner(), self.ids[idx as usize]))
    }

    ///
    /// pick_node_level picks the level at which a new node should be inserted
    /// based on the probabalistic insertion strategy.
    pub(crate) fn pick_node_level(&self) -> u8 {
        let mut level = 0;

        while rand::random::<f32>() < (1.0 - self.ml as f32)
            && level < self.max_level.load(Ordering::Relaxed)
            && self.level_ids[level as usize].count() > 0
        {
            level += 1;
        }

        level
    }

    /// calculate_max_level calculates the maximum level of the HNSW graph based
    /// on the number of nodes and the maximum number of connections per node
    #[inline]
    fn update_max_level(&self) {
        // p = 1/m
        // max_level ≈ log(n)/log(m)
        let estimated_count = usize::max(self.ids.count(), self.config.min_count);

        let max_level =
            (((estimated_count as f64).ln() * self.ml).ceil() as usize).max(1) as u8 - 1;

        while self.max_level.load(Ordering::Relaxed) < max_level {
            println!("level up from {}", self.max_level.load(Ordering::Relaxed));

            self.level_ids.push(bc::Vec::new());
            self.vectors.push(RwLock::new(Vec::new()));
            self.max_level.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[inline]
    pub fn dist_to(&self, query: &[S], entry: NodeId) -> DistEntry<NodeId> {
        DistEntry(
            OrderedFloat(S::l2(query, &self.get_vector(entry)).unwrap() as f32),
            entry,
        )
    }

    pub fn prune_connections(&mut self) {
        let mut heaps = vec![
            BinaryHeap::with_capacity(self.config.m as usize + 1);
            self.max_level.load(Ordering::Relaxed) as usize + 1
        ];

        for (_, &node_idx) in self.pending_indexes.iter() {
            for (_, &(lvl, neighbour)) in &self.connections[node_idx as usize] {
                let dist = OrderedFloat(self.distance(neighbour, node_idx));
                heaps[lvl as usize].push(DistEntry(dist, neighbour));
                if heaps[lvl as usize].len() > self.config.m as usize {
                    heaps.pop();
                }
            }

            let node_onnections = self.connections.get_mut(node_idx as usize).unwrap();
            node_onnections.clear();

            for (lvl, heap) in heaps.iter().enumerate() {
                for &DistEntry(_, idx) in heap.iter() {
                    node_onnections.push((lvl as _, idx));
                }
            }
        }

        self.pending_indexes.clear();
    }

    fn add_connections(&self, node: NodeId, level: u32, others: impl Iterator<Item = NodeId>) {
        for idx in others {
            if self.connections[node as usize]
                .iter()
                .find(|(_, &e)| e == (level, idx))
                .is_none()
            {
                self.connections[node as usize].push((level, idx));
            }

            if self.connections[idx as usize]
                .iter()
                .find(|(_, &e)| e == (level, node))
                .is_none()
            {
                self.connections[idx as usize].push((level, node));
            }

            let _ = self.distance(idx, node);
        }
    }

    #[inline]
    fn distance(&self, node: NodeId, other: NodeId) -> f32 {
        let key = if node > other {
            (node, other)
        } else {
            (other, node)
        };

        if self.distance_cache.contains_key(&key) {
            self.distance_cache.get(&key).unwrap().value().unwrap().0
        } else {
            let a = self.get_vector(node);
            let b = self.get_vector(other);

            S::l2(&a, &b).unwrap() as f32
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        index1::{HnswConfig, HnswIndex},
        DistEntry,
    };

    #[test]
    fn test_layer_search() {
        let index: HnswIndex<f32> = HnswIndex::new(HnswConfig {
            min_count: 1024,
            m: 4,
            ef_construction: 4,
            dimensions: 2,
        });

        for i in 0..32 {
            for j in 0..32 {
                index.insert(i * 32 + j, &[i as f32, j as f32]);
            }
        }

        for lvl in 0..=index.max_level.load(std::sync::atomic::Ordering::Relaxed) {
            println!("layer #{lvl}");
            println!("  nodes: {}", index.level_ids[lvl as usize].count());
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
