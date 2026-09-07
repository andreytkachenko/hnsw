use core::f32;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::Rng;
use std::hint::black_box;
use vsearch::{Euclidian, index::HnswIndex};

fn index_create(data: &[f32], n: usize) -> HnswIndex<f32, Euclidian> {
    let index = HnswIndex::new(
        vsearch::index::HnswConfig {
            estimate_count: n as _,
            m: 16,
            ef_construction: 16,
            dimensions: 128,
            ef_search: 32,
        },
        Euclidian,
    );

    for idx in 0..n {
        let offset = idx * 128;
        index.insert(idx as _, &data[offset..offset + 128]);
    }

    index
}

fn index_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();
    let count = 100_000;
    let mut data = Vec::with_capacity(count * 128);
    let mut query = Vec::with_capacity(128);

    for _ in 0..count * 128 {
        data.push(rng.random());
    }

    for _ in 0..128 {
        query.push(rng.random());
    }

    c.bench_function("index create 1000", |b| {
        b.iter(|| index_create(black_box(&data), black_box(1000)))
    });

    c.bench_function("index search 100_000", |b| {
        let index = index_create(black_box(&data), black_box(count));

        b.iter(|| index.search(black_box(&query), black_box(8), black_box(f32::INFINITY)))
    });
}

criterion_group!(benches, index_benchmark);
criterion_main!(benches);
