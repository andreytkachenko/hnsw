use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    Scalar,
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
}

pub struct HnswIndex<S: Scalar> {
    pub(crate) layers: bc::Vec<HnswLayer<S>>,
    pub(crate) config: HnswConfig,
    total_nodes: AtomicUsize,

    // Cached value of 1 / ln(M)
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
        println!("insert {id}");

        let vector = vector.as_ref();
        let node_level = self.pick_node_level();
        let mut delegate = 0;
        let mut node_closest = 0;
        if !self.layers[node_level as usize].is_empty() {
            let mut noop = MinEntryHeap::default();
            println!("node_level: {node_level}");

            let delegate_level = node_level as i32 - 1;
            println!("delegate_level: {delegate_level}");

            let mut entrypoint = 0;
            for level in (node_level.saturating_sub(1)..self.layers.count() as u32).rev() {
                let layer = &self.layers[level as usize];
                let best_one = layer.search_inner(vector, entrypoint, &mut noop, true);

                println!(
                    "lvl {level}: {} {}",
                    layer.ids.get(best_one.1 as usize).unwrap(),
                    best_one.0
                );

                if level == node_level {
                    node_closest = best_one.1;
                }

                if level as i32 == delegate_level {
                    delegate = best_one.1;
                }

                entrypoint = layer.get_delegate(best_one.1);
            }
        }

        self.layers[node_level as usize].create_node(id, vector, node_closest, delegate);
        self.total_nodes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove(&mut self, id: u64) {
        for level in 0..self.layers.count() {
            self.layers.get_mut(level).unwrap().remove(id);
        }
    }

    pub fn search(&self, query: &[S], k: u32, max_dist: f32) -> impl Iterator<Item = (u64, f32)> {
        let mut res = Vec::new();
        let mut heap = DistanceOrderedHeap::new(&mut res, k, max_dist);

        let mut entrpoint = 0;
        for level in (0..self.layers.count()).rev() {
            let layer = &self.layers[level];
            let best_one = layer.search_inner(query, entrpoint, &mut heap, false);

            entrpoint = layer.get_delegate(best_one.1);
        }

        std::iter::zip(heap.into_inner(), res) //
            .map(|(dist, idx)| (idx as u64, dist.into_inner()))
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
            println!("ADD LAYER: {}", self.layers.count());

            self.layers.push(HnswLayer::new(level, &self.config));
        }

        level
    }
}

#[cfg(test)]
mod test {
    use core::f32;

    use arrayvec::ArrayVec;

    use crate::index::HnswIndex;

    #[test]
    fn test_index() {
        let config = crate::index::HnswConfig {
            estimate_count: 1024,
            m: 4,
            ef_construction: 4,
            dimensions: 2,
        };

        let index: HnswIndex<f32> = HnswIndex::new(config);

        for i in 0..32 {
            for j in 0..32 {
                index.insert(i * 32 + j, [i as f32, j as f32]);
            }
        }

        println!("insertion done");

        for (_, layer) in &index.layers {
            println!("{} {}", layer.level, layer.nodes.count());
            println!("{:?}", layer.ids);
            println!()
        }

        let mut res: ArrayVec<_, 4> = index
            .search(&[31.5f32, 31.5f32], 4, f32::INFINITY)
            .collect();

        res.sort_by_key(|(id, _)| *id);

        assert_eq!(res[0].0, 990);
        assert_eq!(res[1].0, 991);
        assert_eq!(res[2].0, 1022);
        assert_eq!(res[3].0, 1023);
    }
}
