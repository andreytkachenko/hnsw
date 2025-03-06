use core::f32;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

use arrayvec::ArrayVec;
use boxcar as bc;
use ordered_float::OrderedFloat;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use crate::{
    DistEntry, NODE_MAX_NEIGHBOURS, Scalar,
    heap::{DistanceCache as _, DistanceOrderedHeap, Heap, MappedHeap, Wrap},
    index::HnswConfig,
};

#[derive(Debug)]
pub struct HnswNode {
    pub neighbours: RwLock<ArrayVec<u32, NODE_MAX_NEIGHBOURS>>,
    pub delegate: u32,
}

pub struct HnswLayer<S: Scalar> {
    pub(crate) ids: bc::Vec<u64>,

    pub(crate) map: leapfrog::LeapMap<u64, u32>,

    /// Layer level
    pub(crate) level: u32,

    /// Layer vectors data
    pub(crate) vectors: RwLock<Vec<S>>,

    /// HnswNode Collection indexes matched with vectors
    pub(crate) nodes: bc::Vec<HnswNode>,

    /// Deleted nodes (needed to ignore and  connections)
    pub(crate) deleted: leapfrog::LeapMap<u32, u8>,

    /// Distance cache
    dist_cache: leapfrog::LeapMap<(u32, u32), Wrap>,
    dimensions: usize,

    /// Number of connection
    m: u32,
    n: u32,
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
            dimensions: config.dimensions as _,
            m: config.m,
            dist_cache: leapfrog::LeapMap::with_capacity(capacity * config.m as usize),
            deleted: leapfrog::LeapMap::new(),
            map: leapfrog::LeapMap::new(),
            n: config.m,
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
    pub fn create_node(&self, id: u64, vector: &[S], entrypoint: u32, delegate: u32) -> u32 {
        assert_eq!(vector.len(), self.dimensions);

        let mut guard = self.vectors.write();
        guard.extend_from_slice(vector);
        self.ids.push(id);

        let index = self.nodes.push(HnswNode {
            neighbours: RwLock::new(ArrayVec::new()),
            delegate,
        }) as u32;

        drop(guard);

        if self.nodes.count() > 1 {
            let mut guard = self.nodes[index as usize].neighbours.write();

            let mut heap = DistanceOrderedHeap::with_cache(
                &mut *guard,
                index,
                &self.dist_cache,
                |_, node| self.dist_to(vector, node).0.into_inner(),
                self.m,
                f32::INFINITY,
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
        let mut countdown = self.n;

        let mut min_node = self.dist_to(query, entrypoint);
        candidates.push(Reverse(min_node));

        while let Some(Reverse(candidate)) = candidates.pop() {
            if debug {
                println!("#{} {}", self.ids[candidate.1 as usize], candidate.0,);
            }

            if candidate <= min_node {
                min_node = candidate;
                countdown = self.n;
            } else {
                countdown -= 1;
                if countdown == 0 {
                    println!("exit loop");
                    break;
                }
            }

            heap.push(candidate);

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
            self.m,
            f32::INFINITY,
        );

        heap.push(neighbour);
    }

    #[inline]
    pub(crate) fn get_delegate(&self, node: u32) -> u32 {
        self.nodes[node as usize].delegate
    }

    pub(crate) fn remove(&mut self, id: u64) {
        let Some(item) = self.map.get(&id).and_then(|mut x| x.value()) else {
            return;
        };

        self.deleted.insert(item, 1);
        for &i in self.nodes[item as usize].neighbours.read().iter() {
            if i != item {
                let mut guard = self.nodes[item as usize].neighbours.write();
                DistanceOrderedHeap::new(&mut *guard, 1, f32::INFINITY).remove(item);
            }
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
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
                layer.create_node(i * 32 + j, &[i as f32, j as f32], 0, 0);
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
