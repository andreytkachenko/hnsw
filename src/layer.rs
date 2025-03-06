use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    sync::atomic::{AtomicU32, Ordering},
};

use arrayvec::ArrayVec;
use boxcar as bc;
use ordered_float::OrderedFloat;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use crate::{
    heap::{DistanceCache as _, DistanceOrderedHeap, Heap, MappedHeap, Wrap},
    index::HnswConfig,
    DistEntry, Scalar, NODE_MAX_NEIGHBOURS,
};

#[derive(Debug)]
pub struct HnswNode {
    pub neighbours: RwLock<ArrayVec<u32, NODE_MAX_NEIGHBOURS>>,
    pub delegate: AtomicU32,
}

pub struct HnswLayer<S: Scalar> {
    pub ids: bc::Vec<u64>,

    /// Layer level
    pub level: u32,

    /// Layer vectors data
    pub vectors: RwLock<Vec<S>>,

    /// HnswNode Collection indexes matched with vectors
    pub nodes: bc::Vec<HnswNode>,

    /// Node Index -> Id map
    pub node_ids: bc::Vec<u64>,

    /// Distance cache
    dist_cache: leapfrog::LeapMap<(u32, u32), Wrap>,
    dimensions: usize,

    /// Number of connection
    m: u32,
}

impl<S: Scalar> HnswLayer<S> {
    pub fn new(level: u32, config: &HnswConfig) -> Self {
        // estimated count nodes on the layer level: exp(level * ln(M)) same as 2^(level * log2(M))
        let capacity = 1usize << (level * u32::ilog2(config.m));

        Self {
            ids: bc::Vec::with_capacity(capacity),
            level,
            vectors: RwLock::new(Vec::with_capacity(capacity * config.dimensions as usize)),
            nodes: bc::Vec::with_capacity(capacity),
            node_ids: bc::Vec::with_capacity(capacity),
            dimensions: config.dimensions as _,
            m: config.m,
            dist_cache: leapfrog::LeapMap::with_capacity(capacity * config.m as usize),
        }
    }

    #[inline]
    fn get_vector(&self, node: u32) -> MappedRwLockReadGuard<'_, [S]> {
        RwLockReadGuard::map(self.vectors.read(), |x| {
            let offset = node as usize * self.dimensions;

            &x[offset..offset + self.dimensions]
        })
    }

    #[inline]
    pub fn create_node(&self, id: u64, vector: &[S], entrypoint: u32) -> u32 {
        assert_eq!(vector.len(), self.dimensions);

        let mut guard = self.vectors.write();
        guard.extend_from_slice(vector);
        self.ids.push(id);

        let index = self.nodes.push(HnswNode {
            neighbours: RwLock::new(ArrayVec::new()),
            delegate: AtomicU32::new(0),
        }) as u32;

        drop(guard);

        if self.nodes.count() > 1 {
            let mut guard = self.nodes[index as usize].neighbours.write();

            let mut heap = DistanceOrderedHeap::with_cache(
                &mut *guard,
                index,
                &self.dist_cache,
                |_, node| self.dist_to(vector, node).0.into_inner(),
                Some(self.m),
                None,
            );

            self.search_inner(vector, entrypoint, &mut heap, false);

            for (id, dist) in heap.iter() {
                self.dist_cache.put((index, id), dist);
                self.node_add_connection(id, DistEntry(OrderedFloat(dist), index));
            }
        }

        index
    }

    #[inline]
    pub fn dist_to(&self, query: &[S], entry: u32) -> DistEntry<u32> {
        DistEntry(
            OrderedFloat(S::l2(query, &self.get_vector(entry)).unwrap() as f32),
            entry,
        )
    }

    pub fn dist_between(&self, node: u32, other: u32) -> DistEntry<u32> {
        let lock = self.vectors.read();
        let node_offset = node as usize * self.dimensions;
        let other_offset = other as usize * self.dimensions;
        let dist = S::l2(
            &lock[node_offset..node_offset + self.dimensions],
            &lock[other_offset..other_offset + self.dimensions],
        )
        .unwrap();

        DistEntry(OrderedFloat(dist as f32), other)
    }

    pub(crate) fn search_inner(
        &self,
        query: &[S],
        entrypoint: u32,
        heap: &mut impl Heap<u32>,
        debug: bool,
    ) -> DistEntry<u32> {
        log::debug!("layer #{} ({entrypoint}):", self.level);
        log::debug!("layer node count: {:?}", self.nodes.count());

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();

        let mut min_node = self.dist_to(query, entrypoint);
        candidates.push(Reverse(min_node));

        while let Some(Reverse(candidate)) = candidates.pop() {
            if debug {
                println!("#{} {}", candidate.1, candidate.0,);
            }

            if candidate < min_node {
                min_node = candidate;
            }

            if let Some(c) = heap.push(candidate) {
                if c.0 <= candidate.0 {
                    log::trace!("layer search exit");

                    break;
                }
            }

            for &neighbor in self.nodes[candidate.1 as usize]
                .neighbours
                .read()
                .as_slice()
            {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);

                    candidates.push(Reverse(self.dist_to(query, neighbor)));
                }
            }
        }

        min_node
    }

    pub fn search(&self, query: &[S], entrypoint: u32, heap: &mut impl Heap<u64>, debug: bool) {
        let mut mapped_heap = MappedHeap::new(heap, |x| self.ids[x as usize]);
        self.search_inner(query, entrypoint, &mut mapped_heap, debug);
    }

    fn node_add_connection(&self, entry: u32, neighbour: DistEntry<u32>) {
        let mut guard = self.nodes[entry as usize].neighbours.write();
        let mut heap = DistanceOrderedHeap::with_cache(
            &mut *guard,
            entry,
            &self.dist_cache,
            |_, node| self.dist_between(entry, node).0.into_inner(),
            Some(self.m),
            None,
        );

        heap.push(neighbour);
    }

    #[inline]
    pub(crate) fn _set_delegate(&self, node: u32, delegate: u32) {
        self.nodes[node as usize]
            .delegate
            .store(delegate, Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn get_delegate(&self, node: u32) -> u32 {
        self.nodes[node as usize].delegate.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod test {
    use std::f32;

    use arrayvec::ArrayVec;

    use crate::heap::DistanceOrderedHeap;

    use super::HnswLayer;

    #[test]
    fn test_layer() {
        let config = crate::index::HnswConfig {
            estimate_count: 1024,
            m: 4,
            ef_construction: 4,
            dimensions: 2,
        };

        let layer: HnswLayer<f32> = HnswLayer::new(0, &config);

        for i in 0..32 {
            for j in 0..32 {
                layer.create_node(i * 32 + j, &[i as f32, j as f32], 0);
            }
        }

        let mut res: ArrayVec<u32, 4> = ArrayVec::new();
        let mut heap = DistanceOrderedHeap::new(&mut res, 4, f32::INFINITY);

        let best_one = layer.search_inner(&[31.5f32, 31.5f32], 0, &mut heap, false);

        res.sort();

        assert_eq!(best_one.1, 1023);
        assert_eq!(res.as_slice(), &[990, 991, 1022, 1023]);
    }
}
