use constraint_solver::Matrix;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::ThreadPoolBuilder;
use std::hint::black_box;
use std::sync::Once;

static LOG_RAYON_THREADS: Once = Once::new();

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f64(&mut self) -> f64 {
        let v = self.next_u32() as f64 / u32::MAX as f64;
        2.0 * v - 1.0
    }
}

fn random_matrix(rows: usize, cols: usize, rng: &mut LcgRng) -> Matrix {
    let mut m = Matrix::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m[(i, j)] = rng.next_f64();
        }
    }
    m
}

/// Square sizes for multiplication and transpose throughput measurements.
const DENSE_SIZES: &[usize] = &[64, 128, 256, 512];
/// Base dimensions for canonical Jacobi-SVD least-squares workloads.
const LEAST_SQUARES_SIZES: &[usize] = &[8, 16, 32, 64];
/// Aspect ratio used by the tall and wide least-squares fixtures.
const LEAST_SQUARES_RATIO: usize = 2;
/// Dimensions for products whose output is small but whose shared inner
/// dimension makes the total arithmetic large enough to parallelize.
const SKINNY_MATMUL_SHAPES: &[(usize, usize, usize)] = &[(4, 16_384, 4)];

fn assert_f64_close(a: f64, b: f64, abs_tol: f64, rel_tol: f64) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        diff <= abs_tol + rel_tol * scale,
        "value mismatch: {a} vs {b} (diff {diff})"
    );
}

fn assert_matrix_close(a: &Matrix, b: &Matrix, abs_tol: f64, rel_tol: f64) {
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.cols(), b.cols());
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            assert_f64_close(a[(i, j)], b[(i, j)], abs_tol, rel_tol);
        }
    }
}

fn bench_thread_count() -> usize {
    std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|count| *count > 0)
        })
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
}

fn bench_matmul(c: &mut Criterion) {
    let thread_count = bench_thread_count();
    let pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .expect("failed to build rayon thread pool");
    LOG_RAYON_THREADS.call_once(|| {
        eprintln!("rayon threads: {}", thread_count);
    });

    let mut group = c.benchmark_group("matmul");
    for &size in DENSE_SIZES {
        let mut rng = LcgRng::new(123 + size as u64);
        let a = random_matrix(size, size, &mut rng);
        let b = random_matrix(size, size, &mut rng);
        let c_serial = a.try_mul_with_parallel(&b, false).unwrap();
        let c_parallel = pool.install(|| a.try_mul_with_parallel(&b, true).unwrap());
        assert_matrix_close(&c_serial, &c_parallel, 1e-10, 1e-8);
        group.bench_with_input(BenchmarkId::new("serial", size), &size, |bench, _| {
            bench.iter(|| {
                let result = a.try_mul_with_parallel(black_box(&b), false).unwrap();
                black_box(result)
            })
        });
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| {
                let result = pool.install(|| a.try_mul_with_parallel(black_box(&b), true).unwrap());
                black_box(result)
            })
        });
    }

    // Benchmark the work-estimation regression independently from square
    // products. The old rows*output-columns threshold classified this shape as
    // sixteen operations even though it performs 262,144 multiply-adds.
    for &(rows, inner, cols) in SKINNY_MATMUL_SHAPES {
        let mut rng = LcgRng::new(0x51_1a_7e + inner as u64);
        let a = random_matrix(rows, inner, &mut rng);
        let b = random_matrix(inner, cols, &mut rng);
        let serial = a.try_mul_with_parallel(&b, false).unwrap();
        let parallel = pool.install(|| a.try_mul_with_parallel(&b, true).unwrap());
        assert_matrix_close(&serial, &parallel, 1e-10, 1e-8);

        let shape = format!("{rows}x{inner}x{cols}");
        group.bench_with_input(
            BenchmarkId::new("skinny/serial", &shape),
            &shape,
            |bench, _| {
                bench.iter(|| {
                    let result = a.try_mul_with_parallel(black_box(&b), false).unwrap();
                    black_box(result)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("skinny/parallel", &shape),
            &shape,
            |bench, _| {
                bench.iter(|| {
                    let result =
                        pool.install(|| a.try_mul_with_parallel(black_box(&b), true).unwrap());
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let thread_count = bench_thread_count();
    let pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .expect("failed to build rayon thread pool");

    let mut group = c.benchmark_group("transpose");
    for &size in DENSE_SIZES {
        let mut rng = LcgRng::new(456 + size as u64);
        let a = random_matrix(size, size, &mut rng);
        let t_serial = a.transpose_with_parallel(false);
        let t_parallel = pool.install(|| a.transpose_with_parallel(true));
        assert_matrix_close(&t_serial, &t_parallel, 0.0, 0.0);
        group.bench_with_input(BenchmarkId::new("serial", size), &size, |bench, _| {
            bench.iter(|| {
                let result = a.transpose_with_parallel(false);
                black_box(result)
            })
        });
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| {
                let result = pool.install(|| a.transpose_with_parallel(true));
                black_box(result)
            })
        });
    }
    group.finish();
}

/// Measure the canonical SVD solve across full-rank and deficient shapes.
fn bench_least_squares(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_squares_svd");

    for &size in LEAST_SQUARES_SIZES {
        let mut rng = LcgRng::new(789 + size as u64);
        let long_dimension = size * LEAST_SQUARES_RATIO;

        // Full-rank tall and wide inputs exercise both direct and transposed SVD
        // orientations without maintaining a second factorization benchmark.
        let tall = random_matrix(long_dimension, size, &mut rng);
        let tall_rhs = random_matrix(long_dimension, 1, &mut rng);
        let tall_check = tall.solve_least_squares(&tall_rhs).unwrap();
        assert_eq!(tall_check.info.rank, size);
        group.bench_with_input(
            BenchmarkId::new("tall", format!("{long_dimension}x{size}")),
            &tall,
            |bench, matrix| {
                bench.iter(|| {
                    let result = matrix.solve_least_squares(black_box(&tall_rhs)).unwrap();
                    black_box(result);
                })
            },
        );

        let wide = random_matrix(size, long_dimension, &mut rng);
        let wide_rhs = random_matrix(size, 1, &mut rng);
        let wide_check = wide.solve_least_squares(&wide_rhs).unwrap();
        assert_eq!(wide_check.info.rank, size);
        group.bench_with_input(
            BenchmarkId::new("wide", format!("{size}x{long_dimension}")),
            &wide,
            |bench, matrix| {
                bench.iter(|| {
                    let result = matrix.solve_least_squares(black_box(&wide_rhs)).unwrap();
                    black_box(result);
                })
            },
        );

        // An exact duplicate column verifies rank truncation in the same timed
        // implementation used for ordinary full-rank solves.
        let mut deficient = random_matrix(size, size, &mut rng);
        let deficient_rhs = random_matrix(size, 1, &mut rng);
        for row in 0..size {
            deficient[(row, size - 1)] = deficient[(row, 0)];
        }
        let deficient_check = deficient.solve_least_squares(&deficient_rhs).unwrap();
        assert!(deficient_check.info.rank < size);
        group.bench_with_input(
            BenchmarkId::new("rank_deficient", size),
            &deficient,
            |bench, matrix| {
                bench.iter(|| {
                    let result = matrix
                        .solve_least_squares(black_box(&deficient_rhs))
                        .unwrap();
                    black_box(result);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_transpose, bench_least_squares);
criterion_main!(benches);
