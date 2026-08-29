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

const SIZES: &[usize] = &[64, 128, 256, 512, 1024];
const QR_RATIO: usize = 4;
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
    for &size in SIZES {
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
    for &size in SIZES {
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

fn bench_qr(c: &mut Criterion) {
    let thread_count = bench_thread_count();
    let pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .expect("failed to build rayon thread pool");

    let mut group = c.benchmark_group("qr");
    for &size in SIZES {
        let mut rng = LcgRng::new(789 + size as u64);
        let m = size * QR_RATIO;
        let n = size;
        let tall = random_matrix(m, n, &mut rng);
        let tall_b = random_matrix(m, 1, &mut rng);
        let (x_serial, info_serial) = tall
            .solve_least_squares_with_info_with_parallel(&tall_b, false)
            .unwrap();
        let (x_parallel, info_parallel) = pool.install(|| {
            tall.solve_least_squares_with_info_with_parallel(&tall_b, true)
                .unwrap()
        });
        assert_eq!(info_serial.rank, info_parallel.rank);
        assert_f64_close(info_serial.cond_est, info_parallel.cond_est, 1e-8, 1e-6);
        assert_matrix_close(&x_serial, &x_parallel, 1e-8, 1e-6);
        group.bench_with_input(
            BenchmarkId::new("tall/serial", format!("{m}x{n}")),
            &tall,
            |bench, a| {
                bench.iter(|| {
                    let (x, info) = a
                        .solve_least_squares_with_info_with_parallel(black_box(&tall_b), false)
                        .unwrap();
                    black_box(x);
                    black_box(info);
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("tall/parallel", format!("{m}x{n}")),
            &tall,
            |bench, a| {
                bench.iter(|| {
                    let (x, info) = pool.install(|| {
                        a.solve_least_squares_with_info_with_parallel(black_box(&tall_b), true)
                            .unwrap()
                    });
                    black_box(x);
                    black_box(info);
                })
            },
        );

        let wide = random_matrix(n, m, &mut rng);
        let wide_b = random_matrix(n, 1, &mut rng);
        let (x_serial, info_serial) = wide
            .solve_least_squares_with_info_with_parallel(&wide_b, false)
            .unwrap();
        let (x_parallel, info_parallel) = pool.install(|| {
            wide.solve_least_squares_with_info_with_parallel(&wide_b, true)
                .unwrap()
        });
        assert_eq!(info_serial.rank, info_parallel.rank);
        assert_f64_close(info_serial.cond_est, info_parallel.cond_est, 1e-8, 1e-6);
        assert_matrix_close(&x_serial, &x_parallel, 1e-8, 1e-6);
        group.bench_with_input(
            BenchmarkId::new("wide/serial", format!("{n}x{m}")),
            &wide,
            |bench, a| {
                bench.iter(|| {
                    let (x, info) = a
                        .solve_least_squares_with_info_with_parallel(black_box(&wide_b), false)
                        .unwrap();
                    black_box(x);
                    black_box(info);
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("wide/parallel", format!("{n}x{m}")),
            &wide,
            |bench, a| {
                bench.iter(|| {
                    let (x, info) = pool.install(|| {
                        a.solve_least_squares_with_info_with_parallel(black_box(&wide_b), true)
                            .unwrap()
                    });
                    black_box(x);
                    black_box(info);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_transpose, bench_qr);
criterion_main!(benches);
