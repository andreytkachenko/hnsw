use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{heap::DistanceOrderedHeap, layer::HnswLayer, Scalar};

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
}

pub struct HnswIndex<S: Scalar> {
    pub layers: bc::Vec<HnswLayer<S>>,
    pub config: HnswConfig,
    total_nodes: AtomicUsize,

    // Cached value of 1 / ln(M)
    #[allow(dead_code)]
    ml: f64,
}

impl<S: Scalar> HnswIndex<S> {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            layers: bc::vec![HnswLayer::new(0, &config)],
            ml: 1.0 / (config.m as f64).ln(),
            config,
            total_nodes: AtomicUsize::new(0),
        }
    }

    pub fn insert<V: AsRef<[S]>>(&self, id: u64, vector: V) {
        let vector = vector.as_ref();

        self.update_max_level();

        let _node_level = self.pick_node_level();
        let node_level = 0;
        let entrypoint = 0;

        self.layers[node_level as usize].create_node(id, vector, entrypoint);
        self.total_nodes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove(&mut self, id: u64) {
        let _ = id;
        // todo
    }

    pub fn search(&self, query: &[S], k: u32, max_dist: f32) -> impl Iterator<Item = (u64, f32)> {
        let mut res = Vec::new();
        let mut heap = DistanceOrderedHeap::new(&mut res, k, max_dist);

        let mut entrpoint = 0;
        for level in (0..self.layers.count()).rev().rev() {
            let layer = &self.layers[level];

            let best_one = layer.search_inner(query, entrpoint, &mut heap, false);

            entrpoint = layer.get_delegate(best_one.1);
        }

        std::iter::empty()
    }

    /// pick_node_level picks the level at which a new node should be inserted
    /// based on the probabalistic insertion strategy.
    pub(crate) fn pick_node_level(&self) -> u8 {
        let mut level = 0;

        while rand::random::<f32>() < (1.0 - self.ml as f32) && level < self.layers.count() as u8 {
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
        let estimated_count = usize::max(
            self.total_nodes.load(Ordering::Relaxed),
            self.config.estimate_count,
        );

        let _max_levels = (((estimated_count as f64).ln() * self.ml).ceil() as usize).max(1) as u8;

        // if self.layers.count() < max_levels as usize {
        //     self.layers
        //         .push(HnswLayer::new(max_levels as u32 - 1, &self.config));
        // }
    }
}
