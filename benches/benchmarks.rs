use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use picard::Picard;

fn generate_data(n_features: usize, n_samples: usize, seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n_features, n_samples));
    let mut state = seed;

    for i in 0..n_features {
        for j in 0..n_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            // Laplace distribution
            data[[i, j]] = if u < 0.5 {
                (2.0 * u).ln()
            } else {
                -(2.0 * (1.0 - u)).ln()
            };
        }
    }

    // Mix with random matrix
    let mut mixing = Array2::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in 0..n_features {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            mixing[[i, j]] = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        }
    }

    mixing.dot(&data)
}

fn bench_picard(c: &mut Criterion) {
    let mut group = c.benchmark_group("picard");

    for n_samples in [1000, 5000, 10000] {
        for n_features in [10, 50, 100] {
            let data = generate_data(n_features, n_samples, 42);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}x{}", n_features, n_samples),
                    format!("{}comp", n_features / 2),
                ),
                &data,
                |b, data| {
                    b.iter(|| {
                        Picard::builder(n_features / 2)
                            .max_iter(100)
                            .random_seed(42)
                            .fit(black_box(data))
                    })
                },
            );
        }
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .measurement_time(std::time::Duration::from_secs(15))
        .sample_size(40)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_picard
}
criterion_main!(benches);
