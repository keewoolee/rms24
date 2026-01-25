use criterion::{criterion_group, criterion_main, Criterion};

fn hint_generation(_c: &mut Criterion) {
    // TODO: Add benchmarks once hints module is implemented
}

criterion_group!(benches, hint_generation);
criterion_main!(benches);
