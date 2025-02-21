use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    hash::{Hash, Hasher},
};

use ordered_float::OrderedFloat;

use crate::index::Scalar;

#[derive(Debug, Clone, Copy)]
pub struct DistEntry<T>(pub OrderedFloat<f32>, pub T);

impl<T: Eq> Eq for DistEntry<T> {}
impl<T: PartialEq> PartialEq for DistEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl<T: PartialEq> PartialOrd for DistEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}
impl<T: Eq> Ord for DistEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T: Hash> Hash for DistEntry<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.1.hash(state);
    }
}

pub struct HnswLayer<S: Scalar> {
    /// Node Id to node index mapping
    pub nodes: HashMap<u64, usize>,

    /// Node Ids
    pub node_ids: Vec<u64>,

    /// Node vector slices strided by dimension
    pub node_vectors: Vec<S>,

    /// Node Connections with distace
    pub node_connections: Vec<HashSet<DistEntry<usize>>>,

    /// Dimensions used as stride in calculation of subslice in node_vector
    pub dimension: usize,
}

impl<S: Scalar + std::fmt::Debug> HnswLayer<S> {
    #[inline]
    pub(crate) fn create_node(&mut self, id: u64, vector: &[S], ef: usize, m: usize) -> usize
    where
        S: Clone,
    {
        if self.nodes.contains_key(&id) {
            panic!("node with id `{}` already exists!", id);
        }

        let neighbours = self.search(vector, 0, ef, false).collect::<HashSet<_>>();
        let index = self.node_ids.len();

        for &DistEntry(dist, neighbour_id) in &neighbours {
            self.node_connections[neighbour_id].insert(DistEntry(dist, index));
            self.prune_connections(neighbour_id, m);
        }

        self.node_ids.push(id);
        self.node_connections.push(neighbours);
        self.node_vectors.extend_from_slice(vector);

        index
    }

    pub(crate) fn search(
        &self,
        query: &[S],
        entry: usize,
        ef: usize,
        debug: bool,
    ) -> impl Iterator<Item = DistEntry<usize>> {
        let mut heap = BinaryHeap::with_capacity(ef);

        if !self.node_ids.is_empty() {
            let estimated_hops = self.node_ids.len().ilog2() as usize;

            let mut visited = HashSet::with_capacity(estimated_hops * 4);
            let mut candidates = BinaryHeap::with_capacity(estimated_hops);

            visited.insert(entry);
            candidates.push(Reverse(self.dist_to(query, entry)));

            while let Some(Reverse(candidate)) = candidates.pop() {
                if debug {
                    println!(
                        "#{} {} {}",
                        candidate.1,
                        candidate.0,
                        self.node_connections[candidate.1].len()
                    );
                }

                heap.push(candidate);
                if heap.len() > ef {
                    heap.pop();
                }

                for &DistEntry(_, neighbor) in &self.node_connections[candidate.1] {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);

                        candidates.push(Reverse(self.dist_to(query, neighbor)));
                    }
                }
            }
        }

        heap.into_iter()
    }

    #[inline]
    pub fn dist_to(&self, query: &[S], entry: usize) -> DistEntry<usize> {
        let vec = self.node_vector(entry);

        DistEntry(OrderedFloat(S::l2(query, vec).unwrap() as f32), entry)
    }

    #[inline]
    fn node_vector(&self, entry: usize) -> &[S] {
        let offset = entry * self.dimension;

        &self.node_vectors[offset..offset + self.dimension]
    }

    pub(crate) fn prune_connections(&mut self, node_idx: usize, m: usize) {
        let node_connections_len = self.node_connections[node_idx].len();
        if node_connections_len <= m {
            return;
        }

        let mut heap = BinaryHeap::from_iter(self.node_connections[node_idx].iter().cloned());
        for _ in 0..node_connections_len - m {
            if let Some(e) = heap.pop() {
                self.node_connections[node_idx].remove(&e);
            }
        }
    }

    pub(crate) fn create_connections(&mut self, node: usize, others: impl Iterator<Item = usize>) {
        let offset = node * self.dimension;
        let vector = &self.node_vectors[offset..offset + self.dimension];

        for idx in others {
            let dist = self.dist_to(vector, idx);

            self.node_connections[node].insert(dist);
        }
    }
}

impl<S: Scalar> HnswLayer<S> {
    pub fn new(dimension: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            node_ids: Vec::new(),
            node_vectors: Vec::new(),
            node_connections: Vec::new(),
            dimension,
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::layer::DistEntry;

    use super::HnswLayer;

    #[test]
    fn test_layer_search() {
        let mut layer: HnswLayer<f32> = HnswLayer::new(2);

        for i in 0..32 {
            for j in 0..32 {
                layer.create_node(i * 32 + j, &[i as f32, j as f32], 8, 8);
            }
        }

        let mut res = layer
            .search(&[31.5f32, 31.5f32], 0, 4, true)
            .map(|DistEntry(_, id)| id)
            .collect::<Vec<_>>();
        res.sort();

        assert_eq!(res, vec![990, 991, 1022, 1023]);
    }
}
