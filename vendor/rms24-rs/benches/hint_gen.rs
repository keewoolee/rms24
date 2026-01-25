//! Hint generation benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rms24::{client::Client, params::Params};

fn bench_cpu_hint_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("hint_gen");

    for &num_entries in &[1_000u64, 10_000, 100_000] {
        let params = Params::new(num_entries, 40, 4);
        let db = vec![0u8; num_entries as usize * 40];

        group.throughput(Throughput::Elements(num_entries));
        group.bench_function(format!("cpu_{}", num_entries), |b| {
            b.iter(|| {
                let mut client = Client::new(params.clone());
                client.generate_hints(black_box(&db));
            });
        });
    }

    group.finish();
}

fn bench_prf_operations(c: &mut Criterion) {
    use rms24::prf::Prf;

    let prf = Prf::new([0u8; 32]);

    c.bench_function("prf_select", |b| {
        b.iter(|| prf.select(black_box(0), black_box(0)));
    });

    c.bench_function("prf_offset", |b| {
        b.iter(|| prf.offset(black_box(0), black_box(0)));
    });

    c.bench_function("prf_select_vector_100", |b| {
        b.iter(|| prf.select_vector(black_box(0), black_box(100)));
    });
}

fn bench_median_cutoff(c: &mut Criterion) {
    use rms24::hints::find_median_cutoff;

    let values: Vec<u32> = (0..1000).collect();

    c.bench_function("find_median_cutoff_1000", |b| {
        b.iter(|| find_median_cutoff(black_box(&values)));
    });
}

criterion_group!(benches, bench_cpu_hint_gen, bench_prf_operations, bench_median_cutoff);
criterion_main!(benches);
