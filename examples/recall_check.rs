use vsearch::{Euclidian, index::{HnswConfig, HnswIndex}};

fn main() {
    // 1-D line
    let config = HnswConfig { estimate_count: 64, m: 8, ef_construction: 16, dimensions: 1, ef_search: 32 };
    let index: HnswIndex<f32, Euclidian> = HnswIndex::new(config, Euclidian);
    for i in 0..20u64 { index.insert(i, [i as f32]); }
    let res = index.search(&[10.5f32], 3, f32::INFINITY);
    println!("1d query=10.5: {res:?}");
    let res = index.search(&[0.0f32], 3, f32::INFINITY);
    println!("1d query=0.0: {res:?}");

    // 2-D grid, corner + center queries
    let config = HnswConfig { estimate_count: 1024, m: 16, ef_construction: 32, dimensions: 2, ef_search: 64 };
    let index: HnswIndex<f32, Euclidian> = HnswIndex::new(config, Euclidian);
    for i in 0..32u64 { for j in 0..32u64 { index.insert(i*32+j, [i as f32, j as f32]); } }
    let res = index.search(&[31.5f32, 31.5f32], 4, f32::INFINITY);
    println!("2d corner: {res:?}");
    let res = index.search(&[15.5f32, 15.5f32], 4, f32::INFINITY);
    println!("2d center: {res:?}");
}
