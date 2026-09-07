use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    Distance, Scalar,
    heap::{DistanceOrderedHeap, MinEntryHeap},
    layer::HnswLayer,
};

use boxcar as bc;

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    // Оцениваемое кол-во элементов (нужно для более равномерного распределения слоев)
    pub estimate_count: usize,

    // Макс. число связей
    pub m: u32,

    // Параметр построения
    pub ef_construction: u32,

    /// Dimensions used as stride in calculation of subslice in node_vector
    pub dimensions: u32,

    /// Search beam width (ef_search); the query explores at least this many candidates
    pub ef_search: u32,
}

pub struct HnswIndex<S: Scalar, D: Distance<S>> {
    pub(crate) layers: bc::Vec<HnswLayer<S, D>>,
    pub(crate) config: HnswConfig,
    total_nodes: AtomicUsize,

    // Cached value of 1 / ln(M)
    ml: f64,

    // Distance metric
    mt: D,
}

impl<S: Scalar, D: Distance<S>> HnswIndex<S, D> {
    pub fn new(config: HnswConfig, mt: D) -> Self {
        Self {
            layers: bc::vec![HnswLayer::new(0, &config, mt.clone())],
            ml: 1.0 / (config.m as f64).ln(),
            config,
            total_nodes: AtomicUsize::new(0),
            mt,
        }
    }

    /// Insert a node into the index.
    ///
    /// The node is placed in every layer `0..=node_level` (standard HNSW), so it
    /// stays reachable while traversing any lower layer.
    pub fn insert<V: AsRef<[S]>>(&self, id: u64, vector: V) {
        let vector = vector.as_ref();
        let node_level = self.pick_node_level();

        // Greedy descent (ef = 1) from the topmost layer down to `node_level + 1`:
        // the closest node found at each layer becomes the entry point for the next one.
        let mut ep_ext: Option<u64> = None;
        if self.total_nodes.load(Ordering::Relaxed) > 0 {
            for lc in (node_level as usize + 1)..self.layers.count() {
                let layer = &self.layers[lc];
                if layer.is_empty() {
                    continue;
                }

                let entrypoint = ep_ext.and_then(|ext| layer.find(ext)).unwrap_or(0);
                let mut scratch = MinEntryHeap::default();
                let best_one = layer.search_inner(vector, entrypoint, &mut scratch, 1, false);
                ep_ext = Some(layer.ids[best_one.1 as usize]);
            }
        }

        // Create the node in all layers it belongs to, top-down.
        for lc in (0..=node_level as usize).rev() {
            let layer = &self.layers[lc];
            let entrypoint = ep_ext.and_then(|ext| layer.find(ext)).unwrap_or(0);
            layer.create_node(id, vector, entrypoint);
        }

        self.total_nodes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove(&mut self, id: u64) {
        for level in 0..self.layers.count() {
            self.layers.get_mut(level).unwrap().remove(id);
        }
    }

    /// Search for the `k` nearest items within `max_dist`.
    ///
    /// Returns `(external_id, distance)` pairs sorted by distance ascending.
    pub fn search(&self, query: &[S], k: u32, max_dist: f32) -> Vec<(u64, f32)> {
        if self.total_nodes.load(Ordering::Relaxed) == 0 {
            return Vec::new();
        }

        let ef = self.config.ef_search.max(k).max(1);

        // Greedy descent (ef = 1) from the topmost layer down to layer 1.
        let mut ep_ext: Option<u64> = None;
        for lc in (1..self.layers.count()).rev() {
            let layer = &self.layers[lc];
            if layer.is_empty() {
                continue;
            }

            let entrypoint = ep_ext.and_then(|ext| layer.find(ext)).unwrap_or(0);
            let mut scratch = MinEntryHeap::default();
            let best_one = layer.search_inner(query, entrypoint, &mut scratch, 1, false);
            ep_ext = Some(layer.ids[best_one.1 as usize]);
        }

        // Full search at layer 0 with the query beam width.
        let layer0 = &self.layers[0];
        let entrypoint = ep_ext.and_then(|ext| layer0.find(ext)).unwrap_or(0);

        let mut res = Vec::new();
        let mut heap = DistanceOrderedHeap::new(&mut res, k, max_dist);
        layer0.search_inner(query, entrypoint, &mut heap, ef, false);

        // `res` holds internal node indices of layer 0; translate to external ids.
        let mut out: Vec<(u64, f32)> = std::iter::zip(heap.into_inner(), res)
            .map(|(dist, idx)| (layer0.ids[idx as usize], dist.into_inner()))
            .collect();

        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// pick_node_level picks the level at which a new node should be inserted
    /// based on the probabalistic insertion strategy.
    pub(crate) fn pick_node_level(&self) -> u32 {
        let total_count = self.total_nodes.load(Ordering::Relaxed);
        let max_levels = (((total_count as f64).ln() * self.ml).ceil() as usize).max(1) as u32;

        let mut level = 0;

        while rand::random::<f32>() < (1.0 - self.ml as f32) && level < max_levels - 1 {
            level += 1;
        }

        if self.layers.get(level as usize).is_none() {
            log::debug!("ADD LAYER: {}", self.layers.count());

            self.layers.push(HnswLayer::new(level, &self.config, self.mt.clone()));
        }

        level
    }
}

#[cfg(test)]
mod test {
    use crate::index::HnswIndex;
    use crate::Euclidian;

    #[test]
    fn test_index() {
        let config = crate::index::HnswConfig {
            estimate_count: 1024,
            m: 4,
            ef_construction: 4,
            dimensions: 2,
            ef_search: 32,
        };

        let index: HnswIndex<f32, Euclidian> = HnswIndex::new(config, Euclidian);

        for i in 0..32 {
            for j in 0..32 {
                index.insert(i * 32 + j, [i as f32, j as f32]);
            }
        }

        let res = index.search(&[31.5f32, 31.5f32], 4, f32::INFINITY);

        assert_eq!(res.len(), 4);
        let ids: Vec<u64> = res.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&990), "ids: {ids:?}");
        assert!(ids.contains(&991), "ids: {ids:?}");
        assert!(ids.contains(&1022), "ids: {ids:?}");
        assert!(ids.contains(&1023), "ids: {ids:?}");

        // results must be sorted by distance ascending
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_search_empty() {
        let config = crate::index::HnswConfig {
            estimate_count: 64,
            m: 4,
            ef_construction: 4,
            dimensions: 2,
            ef_search: 16,
        };

        let index: HnswIndex<f32, Euclidian> = HnswIndex::new(config, Euclidian);
        assert!(index.search(&[0.0f32, 0.0], 4, f32::INFINITY).is_empty());
    }

    #[test]
    fn test_search_returns_true_nearest_1d() {
        let config = crate::index::HnswConfig {
            estimate_count: 64,
            m: 8,
            ef_construction: 16,
            dimensions: 1,
            ef_search: 32,
        };

        let index: HnswIndex<f32, Euclidian> = HnswIndex::new(config, Euclidian);
        for i in 0..20u64 {
            index.insert(i, [i as f32]);
        }

        // query 10.5: true nearest are 10, 11 (dist 0.5), then 9, 12 (dist 1.5)
        let res = index.search(&[10.5f32], 3, f32::INFINITY);
        let ids: Vec<u64> = res.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&10), "ids: {ids:?}");
        assert!(ids.contains(&11), "ids: {ids:?}");
        for (_, d) in &res {
            assert!(*d <= 1.5 + f32::EPSILON);
        }

        // query 0.0: true nearest are 0, 1, 2
        let res = index.search(&[0.0f32], 3, f32::INFINITY);
        let ids: Vec<u64> = res.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&0), "ids: {ids:?}");
        assert!(ids.contains(&1), "ids: {ids:?}");
        assert!(ids.contains(&2), "ids: {ids:?}");
    }
}
