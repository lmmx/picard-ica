// benches/benchmark.rs

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use picard::{Picard, PicardConfig};

/// Generate synthetic ICA data with Laplacian sources.
fn generate_data(n_features: usize, n_samples: usize, seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n_features, n_samples));
    let mut state = seed;

    // Generate Laplacian sources
    for i in 0..n_features {
        for j in 0..n_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            data[[i, j]] = if u < 0.5 {
                (2.0 * u).ln()
            } else {
                -(2.0 * (1.0 - u)).ln()
            };
        }
    }

    // Generate random mixing matrix
    let mut mixing = Array2::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in 0..n_features {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            mixing[[i, j]] = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        }
    }

    mixing.dot(&data)
}

fn bench_picard_default(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_default");

    for n_samples in [1000, 5000, 10000] {
        for n_features in [10, 50, 100] {
            let data = generate_data(n_features, n_samples, 42);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}features_{}samples", n_features, n_samples),
                    n_features,
                ),
                &data,
                |b, data| b.iter(|| Picard::fit(black_box(data))),
            );
        }
    }

    group.finish();
}

fn bench_picard_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_n_components");

    let n_features = 50;
    let n_samples = 5000;
    let data = generate_data(n_features, n_samples, 42);

    for n_components in [5, 10, 25, 50] {
        let config = PicardConfig::builder()
            .n_components(n_components)
            .max_iter(100)
            .random_state(42)
            .build();

        group.bench_with_input(
            BenchmarkId::new("components", n_components),
            &data,
            |b, data| b.iter(|| Picard::fit_with_config(black_box(data), &config)),
        );
    }

    group.finish();
}

fn bench_picard_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_ortho_comparison");

    let n_features = 50;
    let n_samples = 5000;
    let data = generate_data(n_features, n_samples, 42);

    for ortho in [false, true] {
        let name = if ortho { "picard_o" } else { "picard" };
        let config = PicardConfig::builder()
            .n_components(25)
            .ortho(ortho)
            .max_iter(100)
            .random_state(42)
            .build();

        group.bench_with_input(BenchmarkId::new(name, n_features), &data, |b, data| {
            b.iter(|| Picard::fit_with_config(black_box(data), &config))
        });
    }

    group.finish();
}

fn bench_picard_extended(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard_extended_comparison");

    let n_features = 50;
    let n_samples = 5000;
    let data = generate_data(n_features, n_samples, 42);

    for extended in [false, true] {
        let name = if extended { "extended" } else { "standard" };
        let config = PicardConfig::builder()
            .n_components(25)
            .extended(extended)
            .max_iter(100)
            .random_state(42)
            .build();

        group.bench_with_input(BenchmarkId::new(name, n_features), &data, |b, data| {
            b.iter(|| Picard::fit_with_config(black_box(data), &config))
        });
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(30)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_picard_default, bench_picard_components, bench_picard_ortho, bench_picard_extended
}
criterion_main!(benches);
