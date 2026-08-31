use constraint_solver::Matrix;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

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
/// dimension still makes the arithmetic substantial.
const SKINNY_MATMUL_SHAPES: &[(usize, usize, usize)] = &[(4, 16_384, 4)];

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &size in DENSE_SIZES {
        let mut rng = LcgRng::new(123 + size as u64);
        let a = random_matrix(size, size, &mut rng);
        let b = random_matrix(size, size, &mut rng);
        group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, _| {
            bench.iter(|| {
                let result = a.try_mul(black_box(&b)).unwrap();
                black_box(result)
            })
        });
    }

    // Measure long-inner-dimension products independently from square products.
    for &(rows, inner, cols) in SKINNY_MATMUL_SHAPES {
        let mut rng = LcgRng::new(0x51_1a_7e + inner as u64);
        let a = random_matrix(rows, inner, &mut rng);
        let b = random_matrix(inner, cols, &mut rng);
        let shape = format!("{rows}x{inner}x{cols}");
        group.bench_with_input(BenchmarkId::new("skinny", &shape), &shape, |bench, _| {
            bench.iter(|| {
                let result = a.try_mul(black_box(&b)).unwrap();
                black_box(result)
            })
        });
    }
    group.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");
    for &size in DENSE_SIZES {
        let mut rng = LcgRng::new(456 + size as u64);
        let a = random_matrix(size, size, &mut rng);
        group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, _| {
            bench.iter(|| {
                let result = a.transpose();
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

        // Make the square matrix strictly diagonally dominant so the timed group
        // deterministically measures a full-rank solve rather than relying on a
        // probabilistic property of its generated coefficients.
        let mut square = random_matrix(size, size, &mut rng);
        let square_rhs = random_matrix(size, 1, &mut rng);
        for diagonal in 0..size {
            square[(diagonal, diagonal)] += 2.0 * size as f64;
        }
        let square_check = square.solve_least_squares(&square_rhs).unwrap();
        assert_eq!(square_check.info.rank, size);
        group.bench_with_input(
            BenchmarkId::new("square", size),
            &square,
            |bench, matrix| {
                bench.iter(|| {
                    let result = matrix.solve_least_squares(black_box(&square_rhs)).unwrap();
                    black_box(result);
                })
            },
        );

        // Full-rank tall and wide inputs exercise both direct and transposed SVD
        // orientations under the same bounded dimension schedule.
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
