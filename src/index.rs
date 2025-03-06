use crate::{layer::HnswLayer, Scalar};

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

    // Cached value of 1 / ln(M)
    #[allow(dead_code)]
    ml: f64,
}
