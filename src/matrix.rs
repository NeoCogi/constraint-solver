/*
MIT License

Copyright (c) 2026 Raja Lehtihet & Wael El Oraiby

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 16_384;

fn should_parallelize(rows: usize, cols: usize) -> bool {
    rows.saturating_mul(cols) >= PARALLEL_THRESHOLD
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    DimensionMismatch {
        operation: &'static str,
        left: (usize, usize),
        right: (usize, usize),
    },
}

#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresQrInfo {
    pub rank: usize,
    pub cond_est: f64,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch {
                operation,
                left,
                right,
            } => write!(
                f,
                "Matrix dimension mismatch for {}: left is {}x{}, right is {}x{}",
                operation, left.0, left.1, right.0, right.1
            ),
        }
    }
}

impl std::error::Error for MatrixError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err("Data length doesn't match dimensions".to_string());
        }
        Ok(Matrix { data, rows, cols })
    }

    pub fn identity(size: usize) -> Self {
        let mut m = Matrix::new(size, size);
        for i in 0..size {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn transpose(&self) -> Matrix {
        self.transpose_with_parallel(false)
    }

    pub fn transpose_with_parallel(&self, parallel: bool) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        if parallel && should_parallelize(self.rows, self.cols) {
            result
                .data
                .par_chunks_mut(self.rows)
                .enumerate()
                .for_each(|(j, row)| {
                    for i in 0..self.rows {
                        row[i] = self[(i, j)];
                    }
                });
            return result;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)];
            }
        }
        result
    }

    pub fn lu_decomposition(&self) -> Result<(Matrix, Matrix, Vec<usize>), String> {
        self.lu_decomposition_with_tolerance(1e-12)
    }

    pub fn lu_decomposition_with_tolerance(
        &self,
        singular_relative_epsilon: f64,
    ) -> Result<(Matrix, Matrix, Vec<usize>), String> {
        if self.rows != self.cols {
            return Err("Matrix must be square for LU decomposition".to_string());
        }
        if !singular_relative_epsilon.is_finite() || singular_relative_epsilon < 0.0 {
            return Err("Invalid singular tolerance".to_string());
        }

        let n = self.rows;
        let mut l = Matrix::identity(n);
        let mut u = self.clone();
        let mut pivot = (0..n).collect::<Vec<_>>();

        for k in 0..n {
            let mut max_val = 0.0;
            let mut max_row = k;
            for i in k..n {
                let val = u[(i, k)].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            let mut row_norm: f64 = 0.0;
            for j in k..n {
                row_norm = row_norm.max(u[(max_row, j)].abs());
            }

            if max_val <= singular_relative_epsilon * row_norm {
                return Err("Matrix is singular".to_string());
            }

            if max_row != k {
                pivot.swap(k, max_row);
                for j in 0..n {
                    let temp = u[(k, j)];
                    u[(k, j)] = u[(max_row, j)];
                    u[(max_row, j)] = temp;
                }
                for j in 0..k {
                    let temp = l[(k, j)];
                    l[(k, j)] = l[(max_row, j)];
                    l[(max_row, j)] = temp;
                }
            }

            for i in (k + 1)..n {
                l[(i, k)] = u[(i, k)] / u[(k, k)];
                for j in k..n {
                    u[(i, j)] -= l[(i, k)] * u[(k, j)];
                }
            }
        }

        Ok((l, u, pivot))
    }

    pub fn solve_lu(&self, b: &Matrix) -> Result<Matrix, String> {
        self.solve_lu_with_tolerance(b, 1e-12)
    }

    pub fn solve_lu_with_tolerance(
        &self,
        b: &Matrix,
        singular_relative_epsilon: f64,
    ) -> Result<Matrix, String> {
        if self.rows != self.cols {
            return Err("Matrix must be square".to_string());
        }
        if b.rows != self.rows || b.cols != 1 {
            return Err("Invalid dimensions for b".to_string());
        }

        let (l, u, pivot) = self.lu_decomposition_with_tolerance(singular_relative_epsilon)?;

        let mut pb = Matrix::new(b.rows, 1);
        for i in 0..b.rows {
            pb[(i, 0)] = b[(pivot[i], 0)];
        }

        let mut y = Matrix::new(self.rows, 1);
        for i in 0..self.rows {
            y[(i, 0)] = pb[(i, 0)];
            for j in 0..i {
                y[(i, 0)] -= l[(i, j)] * y[(j, 0)];
            }
        }

        let mut x = Matrix::new(self.rows, 1);
        for i in (0..self.rows).rev() {
            x[(i, 0)] = y[(i, 0)];
            for j in (i + 1)..self.rows {
                x[(i, 0)] -= u[(i, j)] * x[(j, 0)];
            }
            x[(i, 0)] /= u[(i, i)];
        }

        Ok(x)
    }

    /// Solve a (possibly non-square) least squares problem using Householder QR.
    ///
    /// Solves `min_x ||A x - b||_2`, where `A` is `self`.
    /// - If `rows >= cols`, returns the standard least-squares solution.
    /// - If `rows < cols`, returns the minimum-norm solution among all minimizers.
    pub fn solve_least_squares_qr(&self, b: &Matrix) -> Result<Matrix, String> {
        self.solve_least_squares_qr_with_info(b)
            .map(|(solution, _)| solution)
    }

    pub fn solve_least_squares_qr_with_parallel(
        &self,
        b: &Matrix,
        parallel: bool,
    ) -> Result<Matrix, String> {
        self.solve_least_squares_qr_with_info_with_parallel(b, parallel)
            .map(|(solution, _)| solution)
    }

    /// Solve a (possibly non-square) least squares problem using Householder QR,
    /// returning diagnostic information about rank and conditioning.
    pub fn solve_least_squares_qr_with_info(
        &self,
        b: &Matrix,
    ) -> Result<(Matrix, LeastSquaresQrInfo), String> {
        self.solve_least_squares_qr_with_info_with_parallel(b, false)
    }

    pub fn solve_least_squares_qr_with_info_with_parallel(
        &self,
        b: &Matrix,
        parallel: bool,
    ) -> Result<(Matrix, LeastSquaresQrInfo), String> {
        if b.cols != 1 {
            return Err("Invalid dimensions for b (expected column vector)".to_string());
        }
        if b.rows != self.rows {
            return Err("Invalid dimensions for b".to_string());
        }

        let m = self.rows;
        let n = self.cols;

        // Trivial cases.
        if n == 0 {
            return Ok((
                Matrix::new(0, 1),
                LeastSquaresQrInfo {
                    rank: 0,
                    cond_est: f64::INFINITY,
                },
            ));
        }

        if m >= n {
            Self::solve_least_squares_qr_tall_with_info(self, b, parallel)
        } else {
            Self::solve_least_squares_qr_wide_with_info(self, b, parallel)
        }
    }

    fn solve_least_squares_qr_tall_with_info(
        a: &Matrix,
        b: &Matrix,
        parallel: bool,
    ) -> Result<(Matrix, LeastSquaresQrInfo), String> {
        let m = a.rows;
        let n = a.cols;

        let mut r = a.clone();
        let mut qt_b = b.clone();
        let mut taus = Vec::with_capacity(n);

        // Householder QR factorization (thin): apply reflectors to `r` and `qt_b`.
        for k in 0..n {
            let mut col_norm: f64 = 0.0;
            for i in k..m {
                col_norm = col_norm.hypot(r[(i, k)]);
            }

            if col_norm == 0.0 {
                taus.push(0.0);
                continue;
            }

            let x0 = r[(k, k)];
            let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
            let alpha = -sign * col_norm;
            let v0 = x0 - alpha;

            // Store Householder vector v in-place in column k (v0 is implicit 1.0).
            for i in (k + 1)..m {
                r[(i, k)] /= v0;
            }

            let mut v_sq = 1.0;
            for i in (k + 1)..m {
                let vi = r[(i, k)];
                v_sq += vi * vi;
            }
            let tau = 2.0 / v_sq;
            taus.push(tau);

            // Apply reflector to remaining columns.
            let cols_left = n.saturating_sub(k + 1);
            if cols_left > 0 {
                let use_parallel = parallel && should_parallelize(m - (k + 1), cols_left);
                let dots: Vec<f64> = if use_parallel {
                    (0..cols_left)
                        .into_par_iter()
                        .map(|offset| {
                            let j = k + 1 + offset;
                            let mut dot = r[(k, j)];
                            for i in (k + 1)..m {
                                dot += r[(i, k)] * r[(i, j)];
                            }
                            dot * tau
                        })
                        .collect()
                } else {
                    let mut dots = Vec::with_capacity(cols_left);
                    for j in (k + 1)..n {
                        let mut dot = r[(k, j)];
                        for i in (k + 1)..m {
                            dot += r[(i, k)] * r[(i, j)];
                        }
                        dots.push(dot * tau);
                    }
                    dots
                };

                for (offset, dot) in dots.iter().enumerate() {
                    let j = k + 1 + offset;
                    r[(k, j)] -= dot;
                }

                if use_parallel {
                    let cols = r.cols;
                    let j_start = k + 1;
                    let k_col = k;
                    r.data[(k + 1) * cols..]
                        .par_chunks_mut(cols)
                        .for_each(|row| {
                            let vik = row[k_col];
                            if vik != 0.0 {
                                for (offset, dot) in dots.iter().enumerate() {
                                    row[j_start + offset] -= vik * dot;
                                }
                            }
                        });
                } else {
                    for i in (k + 1)..m {
                        let vik = r[(i, k)];
                        if vik != 0.0 {
                            for (offset, dot) in dots.iter().enumerate() {
                                let j = k + 1 + offset;
                                r[(i, j)] -= vik * dot;
                            }
                        }
                    }
                }
            }

            // Apply reflector to b: qt_b = Q^T b
            let mut dot = qt_b[(k, 0)];
            for i in (k + 1)..m {
                dot += r[(i, k)] * qt_b[(i, 0)];
            }
            dot *= tau;
            qt_b[(k, 0)] -= dot;

            let use_parallel = parallel && should_parallelize(m - (k + 1), 1);
            if use_parallel {
                let cols = r.cols;
                let k_col = k;
                let r_data = &r.data;
                qt_b.data[(k + 1)..]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, val)| {
                        let i = k + 1 + idx;
                        let vik = r_data[i * cols + k_col];
                        *val -= vik * dot;
                    });
            } else {
                for i in (k + 1)..m {
                    qt_b[(i, 0)] -= r[(i, k)] * dot;
                }
            }

            r[(k, k)] = alpha;
        }

        let mut max_diag: f64 = 0.0;
        for i in 0..n {
            max_diag = max_diag.max(r[(i, i)].abs());
        }
        let tol = 1e-12 * max_diag.max(1.0);
        let mut rank = 0;
        let mut min_diag = f64::INFINITY;
        for i in 0..n {
            let diag = r[(i, i)].abs();
            if diag > tol {
                rank += 1;
                if diag < min_diag {
                    min_diag = diag;
                }
            }
        }
        let cond_est = if rank == 0 || !min_diag.is_finite() {
            f64::INFINITY
        } else {
            max_diag / min_diag
        };

        // Back-substitute R x = Q^T b, using the top n entries.
        let mut x = Matrix::new(n, 1);
        for i in (0..n).rev() {
            let mut sum = qt_b[(i, 0)];
            for j in (i + 1)..n {
                sum -= r[(i, j)] * x[(j, 0)];
            }

            let diag = r[(i, i)];
            if !diag.is_finite() {
                return Err("Least squares solve failed: non-finite diagonal in R".to_string());
            }

            let mut row_norm: f64 = 0.0;
            for j in i..n {
                row_norm = row_norm.max(r[(i, j)].abs());
            }

            if diag.abs() <= 1e-12 * row_norm {
                return Err("Least squares solve failed: matrix is rank deficient".to_string());
            }

            x[(i, 0)] = sum / diag;
        }

        Ok((
            x,
            LeastSquaresQrInfo {
                rank,
                cond_est,
            },
        ))
    }

    fn solve_least_squares_qr_wide_with_info(
        a: &Matrix,
        b: &Matrix,
        parallel: bool,
    ) -> Result<(Matrix, LeastSquaresQrInfo), String> {
        // Use QR on A^T to compute the minimum-norm least squares solution.
        //
        // Let A be m x n with m < n. Compute QR of A^T (n x m):
        // A^T = Q R, where Q is n x n orthogonal and R has top-left m x m upper-triangular.
        // The minimum-norm solution is x = Q [y; 0], where y solves R^T y = b.
        let m = a.rows;
        let n = a.cols;

        let mut r = a.transpose(); // n x m, tall
        let mut taus = Vec::with_capacity(m);

        // Householder QR on r (n x m), producing R in the top m x m block.
        for k in 0..m {
            let mut col_norm: f64 = 0.0;
            for i in k..n {
                col_norm = col_norm.hypot(r[(i, k)]);
            }

            if col_norm == 0.0 {
                taus.push(0.0);
                continue;
            }

            let x0 = r[(k, k)];
            let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
            let alpha = -sign * col_norm;
            let v0 = x0 - alpha;

            for i in (k + 1)..n {
                r[(i, k)] /= v0;
            }

            let mut v_sq = 1.0;
            for i in (k + 1)..n {
                let vi = r[(i, k)];
                v_sq += vi * vi;
            }
            let tau = 2.0 / v_sq;
            taus.push(tau);

            let cols_left = m.saturating_sub(k + 1);
            if cols_left > 0 {
                let use_parallel = parallel && should_parallelize(n - (k + 1), cols_left);
                let dots: Vec<f64> = if use_parallel {
                    (0..cols_left)
                        .into_par_iter()
                        .map(|offset| {
                            let j = k + 1 + offset;
                            let mut dot = r[(k, j)];
                            for i in (k + 1)..n {
                                dot += r[(i, k)] * r[(i, j)];
                            }
                            dot * tau
                        })
                        .collect()
                } else {
                    let mut dots = Vec::with_capacity(cols_left);
                    for j in (k + 1)..m {
                        let mut dot = r[(k, j)];
                        for i in (k + 1)..n {
                            dot += r[(i, k)] * r[(i, j)];
                        }
                        dots.push(dot * tau);
                    }
                    dots
                };

                for (offset, dot) in dots.iter().enumerate() {
                    let j = k + 1 + offset;
                    r[(k, j)] -= dot;
                }

                if use_parallel {
                    let cols = r.cols;
                    let j_start = k + 1;
                    let k_col = k;
                    r.data[(k + 1) * cols..]
                        .par_chunks_mut(cols)
                        .for_each(|row| {
                            let vik = row[k_col];
                            if vik != 0.0 {
                                for (offset, dot) in dots.iter().enumerate() {
                                    row[j_start + offset] -= vik * dot;
                                }
                            }
                        });
                } else {
                    for i in (k + 1)..n {
                        let vik = r[(i, k)];
                        if vik != 0.0 {
                            for (offset, dot) in dots.iter().enumerate() {
                                let j = k + 1 + offset;
                                r[(i, j)] -= vik * dot;
                            }
                        }
                    }
                }
            }

            r[(k, k)] = alpha;
        }

        let mut max_diag: f64 = 0.0;
        for i in 0..m {
            max_diag = max_diag.max(r[(i, i)].abs());
        }
        let tol = 1e-12 * max_diag.max(1.0);
        let mut rank = 0;
        let mut min_diag = f64::INFINITY;
        for i in 0..m {
            let diag = r[(i, i)].abs();
            if diag > tol {
                rank += 1;
                if diag < min_diag {
                    min_diag = diag;
                }
            }
        }
        let cond_est = if rank == 0 || !min_diag.is_finite() {
            f64::INFINITY
        } else {
            max_diag / min_diag
        };

        // Solve R^T y = b (R is m x m upper triangular, so R^T is lower triangular).
        let mut y = vec![0.0; m];
        for i in 0..m {
            let mut sum = b[(i, 0)];
            for j in 0..i {
                sum -= r[(j, i)] * y[j];
            }

            let diag = r[(i, i)];
            if !diag.is_finite() {
                return Err("Least squares solve failed: non-finite diagonal in R".to_string());
            }

            let mut col_norm: f64 = 0.0;
            for j in 0..=i {
                col_norm = col_norm.max(r[(j, i)].abs());
            }

            if diag.abs() <= 1e-12 * col_norm {
                return Err("Least squares solve failed: matrix is rank deficient".to_string());
            }

            y[i] = sum / diag;
        }

        // Form w = [y; 0] in R^n.
        let mut w = vec![0.0; n];
        w[..m].copy_from_slice(&y[..m]);

        // Apply Q to w: Q = H0 H1 ... H_{m-1}.
        // To compute Q w, apply reflectors in reverse order.
        for k in (0..m).rev() {
            let tau = taus[k];
            if tau == 0.0 {
                continue;
            }

            let mut dot = w[k];
            for i in (k + 1)..n {
                dot += r[(i, k)] * w[i];
            }
            dot *= tau;

            w[k] -= dot;

            let use_parallel = parallel && should_parallelize(n - (k + 1), 1);
            if use_parallel {
                let cols = r.cols;
                let k_col = k;
                let r_data = &r.data;
                w[(k + 1)..]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, val)| {
                        let i = k + 1 + idx;
                        let vik = r_data[i * cols + k_col];
                        *val -= vik * dot;
                    });
            } else {
                for i in (k + 1)..n {
                    w[i] -= r[(i, k)] * dot;
                }
            }
        }

        Matrix::from_vec(w, n, 1).map(|solution| {
            (
                solution,
                LeastSquaresQrInfo {
                    rank,
                    cond_est,
                },
            )
        })
    }

    /// Return the Frobenius norm without introducing avoidable intermediate
    /// overflow or underflow.
    ///
    /// A direct `sqrt(sum(value * value))` calculation can overflow while the
    /// final norm is still representable, and it can underflow small non-zero
    /// matrices to zero. The scaled sum-of-squares algorithm used here is the
    /// same numerical strategy traditionally used by BLAS `nrm2` routines.
    /// Non-finite inputs remain visible to callers: a NaN element produces NaN,
    /// while an infinite element produces infinity.
    pub fn norm(&self) -> f64 {
        // `scale` tracks the largest finite magnitude seen so far. `scaled_sum`
        // stores the sum of squares after every element has been divided by
        // that scale, keeping the intermediate values close to unity.
        let mut scale = 0.0;
        let mut scaled_sum = 1.0;

        for &value in &self.data {
            let magnitude = value.abs();

            // Preserve NaN rather than allowing comparisons below to silently
            // ignore it. Solver-level validation can then report the numerical
            // failure instead of mistaking the norm for a converged residual.
            if magnitude.is_nan() {
                return f64::NAN;
            }

            // Any infinite component makes the mathematical norm infinite.
            // Handling it explicitly also avoids the indeterminate `inf / inf`
            // ratios that the scaling branches would otherwise compute.
            if magnitude.is_infinite() {
                return f64::INFINITY;
            }

            // Zero contributes nothing and is skipped so the initial
            // `scaled_sum` sentinel does not affect an all-zero matrix.
            if magnitude == 0.0 {
                continue;
            }

            if scale < magnitude {
                // Rescale all previous contributions to the new, larger
                // magnitude before adding the current element's unit square.
                let ratio = scale / magnitude;
                scaled_sum = 1.0 + scaled_sum * ratio * ratio;
                scale = magnitude;
            } else {
                // The current value is no larger than `scale`, so its ratio is
                // bounded by one and squaring it cannot overflow.
                let ratio = magnitude / scale;
                scaled_sum += ratio * ratio;
            }
        }

        // `scale == 0` means every matrix element was zero (or the matrix was
        // empty). Otherwise, restore the common scale only once at the end.
        if scale == 0.0 {
            0.0
        } else {
            scale * scaled_sum.sqrt()
        }
    }

    pub fn try_add(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "add",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + rhs.data[i];
        }
        Ok(result)
    }

    pub fn try_sub(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "sub",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - rhs.data[i];
        }
        Ok(result)
    }

    pub fn try_mul(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        self.try_mul_with_parallel(rhs, false)
    }

    pub fn try_mul_with_parallel(&self, rhs: &Matrix, parallel: bool) -> Result<Matrix, MatrixError> {
        if self.cols != rhs.rows {
            return Err(MatrixError::DimensionMismatch {
                operation: "mul",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }

        let mut result = Matrix::new(self.rows, rhs.cols);
        if parallel && should_parallelize(self.rows, rhs.cols) {
            let rhs_cols = rhs.cols;
            result
                .data
                .par_chunks_mut(rhs_cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for k in 0..self.cols {
                        let a = self[(i, k)];
                        let rhs_row = &rhs.data[k * rhs_cols..(k + 1) * rhs_cols];
                        for j in 0..rhs_cols {
                            row[j] += a * rhs_row[j];
                        }
                    }
                });
            return Ok(result);
        }

        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                for j in 0..rhs.cols {
                    result[(i, j)] += a * rhs[(k, j)];
                }
            }
        }
        Ok(result)
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(rhs).unwrap_or_else(|err| panic!("{}", err))
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        self.try_sub(rhs).unwrap_or_else(|err| panic!("{}", err))
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        self.try_mul(rhs).unwrap_or_else(|err| panic!("{}", err))
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Self::Output {
        let mut result = self.clone();
        for val in &mut result.data {
            *val *= scalar;
        }
        result
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.4}", self[(i, j)])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (self.state >> 32) as u32
        }

        fn next_f64(&mut self) -> f64 {
            let v = self.next_u32() as f64 / u32::MAX as f64;
            2.0 * v - 1.0
        }
    }

    fn random_matrix(rows: usize, cols: usize, rng: &mut TestRng) -> Matrix {
        let mut m = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m[(i, j)] = rng.next_f64();
            }
        }
        m
    }

    fn assert_matrix_close(a: &Matrix, b: &Matrix, tol: f64) {
        assert_eq!(a.rows(), b.rows());
        assert_eq!(a.cols(), b.cols());
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                let diff = (a[(i, j)] - b[(i, j)]).abs();
                assert!(diff <= tol, "mismatch at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 0)], 0.0);
    }

    /// Verify that the norm retains finite magnitudes that a direct sum of
    /// squares would overflow or underflow before taking the square root.
    #[test]
    fn test_norm_is_stable_across_extreme_finite_scales() {
        // Squaring either large component would overflow, although the expected
        // norm itself remains far below `f64::MAX`.
        let large = Matrix::from_vec(vec![1e200, 1e200], 2, 1).unwrap();
        let large_expected = 2.0_f64.sqrt() * 1e200;
        let large_relative_error = (large.norm() - large_expected).abs() / large_expected;
        assert!(large.norm().is_finite());
        assert!(large_relative_error < 1e-15);

        // Squaring either small component would underflow to zero. The scaled
        // algorithm must nevertheless preserve their non-zero norm.
        let small = Matrix::from_vec(vec![1e-200, 1e-200], 2, 1).unwrap();
        let small_expected = 2.0_f64.sqrt() * 1e-200;
        let small_relative_error = (small.norm() - small_expected).abs() / small_expected;
        assert!(small.norm() > 0.0);
        assert!(small_relative_error < 1e-15);
    }

    /// Confirm that non-finite matrix entries remain observable through the
    /// norm API instead of being accidentally converted to a finite value.
    #[test]
    fn test_norm_propagates_non_finite_components() {
        let with_nan = Matrix::from_vec(vec![1.0, f64::NAN], 2, 1).unwrap();
        let with_infinity = Matrix::from_vec(vec![1.0, f64::INFINITY], 2, 1).unwrap();

        assert!(with_nan.norm().is_nan());
        assert_eq!(with_infinity.norm(), f64::INFINITY);
    }

    #[test]
    fn test_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 1.0);
        assert_eq!(m[(2, 2)], 1.0);
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn test_transpose() {
        let mut m = Matrix::new(2, 3);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(0, 2)] = 3.0;
        m[(1, 0)] = 4.0;
        m[(1, 1)] = 5.0;
        m[(1, 2)] = 6.0;

        let mt_serial = m.transpose_with_parallel(false);
        let mt_parallel = m.transpose_with_parallel(true);
        assert_matrix_close(&mt_serial, &mt_parallel, 0.0);
        assert_eq!(mt_serial.rows(), 3);
        assert_eq!(mt_serial.cols(), 2);
        assert_eq!(mt_serial[(0, 0)], 1.0);
        assert_eq!(mt_serial[(1, 0)], 2.0);
        assert_eq!(mt_serial[(2, 0)], 3.0);
        assert_eq!(mt_serial[(0, 1)], 4.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut a = Matrix::new(2, 3);
        a[(0, 0)] = 1.0;
        a[(0, 1)] = 2.0;
        a[(0, 2)] = 3.0;
        a[(1, 0)] = 4.0;
        a[(1, 1)] = 5.0;
        a[(1, 2)] = 6.0;

        let mut b = Matrix::new(3, 2);
        b[(0, 0)] = 7.0;
        b[(0, 1)] = 8.0;
        b[(1, 0)] = 9.0;
        b[(1, 1)] = 10.0;
        b[(2, 0)] = 11.0;
        b[(2, 1)] = 12.0;

        let c_serial = a.try_mul_with_parallel(&b, false).unwrap();
        let c_parallel = a.try_mul_with_parallel(&b, true).unwrap();
        assert_matrix_close(&c_serial, &c_parallel, 0.0);
        assert_eq!(c_serial.rows(), 2);
        assert_eq!(c_serial.cols(), 2);
        assert_eq!(c_serial[(0, 0)], 58.0);
        assert_eq!(c_serial[(0, 1)], 64.0);
        assert_eq!(c_serial[(1, 0)], 139.0);
        assert_eq!(c_serial[(1, 1)], 154.0);
    }

    #[test]
    fn test_parallel_equivalence_threshold_size() {
        let size = 128;
        let mut rng = TestRng::new(0x3e5a_9f21_d00d_cafe);

        let a = random_matrix(size, size, &mut rng);
        let b = random_matrix(size, size, &mut rng);

        let c_serial = a.try_mul_with_parallel(&b, false).unwrap();
        let c_parallel = a.try_mul_with_parallel(&b, true).unwrap();
        assert_matrix_close(&c_serial, &c_parallel, 1e-10);

        let t_serial = a.transpose_with_parallel(false);
        let t_parallel = a.transpose_with_parallel(true);
        assert_matrix_close(&t_serial, &t_parallel, 0.0);

        let m = size * 4;
        let n = size;
        let tall = random_matrix(m, n, &mut rng);
        let tall_b = random_matrix(m, 1, &mut rng);
        let (x_serial, info_serial) =
            tall.solve_least_squares_qr_with_info_with_parallel(&tall_b, false)
                .unwrap();
        let (x_parallel, info_parallel) =
            tall.solve_least_squares_qr_with_info_with_parallel(&tall_b, true)
                .unwrap();
        assert_eq!(info_serial.rank, info_parallel.rank);
        assert!(info_serial.cond_est.is_finite());
        assert!(info_parallel.cond_est.is_finite());
        assert_matrix_close(&x_serial, &x_parallel, 1e-7);

        let wide = random_matrix(n, m, &mut rng);
        let wide_b = random_matrix(n, 1, &mut rng);
        let (x_serial, info_serial) =
            wide.solve_least_squares_qr_with_info_with_parallel(&wide_b, false)
                .unwrap();
        let (x_parallel, info_parallel) =
            wide.solve_least_squares_qr_with_info_with_parallel(&wide_b, true)
                .unwrap();
        assert_eq!(info_serial.rank, info_parallel.rank);
        assert!(info_serial.cond_est.is_finite());
        assert!(info_parallel.cond_est.is_finite());
        assert_matrix_close(&x_serial, &x_parallel, 1e-7);
    }

    #[test]
    fn test_lu_solve() {
        let a = Matrix::from_vec(vec![2.0, 1.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![5.0, 11.0], 2, 1).unwrap();

        let x = a.solve_lu(&b).unwrap();

        let verify_serial = a.try_mul_with_parallel(&x, false).unwrap();
        let verify_parallel = a.try_mul_with_parallel(&x, true).unwrap();
        assert_matrix_close(&verify_serial, &verify_parallel, 0.0);

        assert!((verify_serial[(0, 0)] - b[(0, 0)]).abs() < 1e-10);
        assert!((verify_serial[(1, 0)] - b[(1, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_lu_solve_is_scale_invariant() {
        let a = Matrix::from_vec(vec![2.0, 1.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![5.0, 11.0], 2, 1).unwrap();

        let x = a.solve_lu(&b).unwrap();

        let scale = 1e-12;
        let a_scaled = &a * scale;
        let b_scaled = &b * scale;

        let x_scaled = a_scaled.solve_lu(&b_scaled).unwrap();
        assert!((x_scaled[(0, 0)] - x[(0, 0)]).abs() < 1e-8);
        assert!((x_scaled[(1, 0)] - x[(1, 0)]).abs() < 1e-8);
    }

    #[test]
    fn test_try_add_dimension_mismatch() {
        let a = Matrix::new(2, 2);
        let b = Matrix::new(2, 3);

        let err = a.try_add(&b).expect_err("expected dimension mismatch");
        assert_eq!(
            err,
            MatrixError::DimensionMismatch {
                operation: "add",
                left: (2, 2),
                right: (2, 3),
            }
        );
    }

    #[test]
    fn test_try_sub_dimension_mismatch() {
        let a = Matrix::new(3, 2);
        let b = Matrix::new(2, 2);

        let err = a.try_sub(&b).expect_err("expected dimension mismatch");
        assert_eq!(
            err,
            MatrixError::DimensionMismatch {
                operation: "sub",
                left: (3, 2),
                right: (2, 2),
            }
        );
    }

    #[test]
    fn test_try_mul_dimension_mismatch() {
        let a = Matrix::new(2, 3);
        let b = Matrix::new(2, 2);

        let err_serial = a
            .try_mul_with_parallel(&b, false)
            .expect_err("expected dimension mismatch");
        let err_parallel = a
            .try_mul_with_parallel(&b, true)
            .expect_err("expected dimension mismatch");
        assert_eq!(
            err_serial,
            MatrixError::DimensionMismatch {
                operation: "mul",
                left: (2, 3),
                right: (2, 2),
            }
        );
        assert_eq!(err_serial, err_parallel);
    }

    #[test]
    fn test_solve_least_squares_qr_overdetermined_exact() {
        // A is 3x2, consistent system with exact solution x=[1,2].
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();

        let x_serial = a.solve_least_squares_qr_with_parallel(&b, false).unwrap();
        let x_parallel = a.solve_least_squares_qr_with_parallel(&b, true).unwrap();
        assert_matrix_close(&x_serial, &x_parallel, 1e-12);
        assert!((x_serial[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((x_serial[(1, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_qr_underdetermined_min_norm() {
        // A is 1x2: x0 + x1 = 1. The minimum-norm solution is [0.5, 0.5].
        let a = Matrix::from_vec(vec![1.0, 1.0], 1, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0], 1, 1).unwrap();

        let x_serial = a.solve_least_squares_qr_with_parallel(&b, false).unwrap();
        let x_parallel = a.solve_least_squares_qr_with_parallel(&b, true).unwrap();
        assert_matrix_close(&x_serial, &x_parallel, 1e-12);
        assert!((x_serial[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((x_serial[(1, 0)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_qr_with_info_random_tall_full_rank() {
        let mut rng = TestRng::new(0x5eed_1234_5678_9abc);
        for _ in 0..5 {
            let mut a = random_matrix(6, 3, &mut rng);
            for i in 0..3 {
                a[(i, i)] += 2.0;
            }
            let x_true = vec![rng.next_f64(), rng.next_f64(), rng.next_f64()];
            let x_mat = Matrix::from_vec(x_true.clone(), 3, 1).unwrap();
            let b = &a * &x_mat;

            let (x_serial, info_serial) =
                a.solve_least_squares_qr_with_info_with_parallel(&b, false)
                    .unwrap();
            let (x_parallel, info_parallel) =
                a.solve_least_squares_qr_with_info_with_parallel(&b, true)
                    .unwrap();
            assert_eq!(info_serial.rank, 3);
            assert_eq!(info_parallel.rank, 3);
            assert!(info_serial.cond_est.is_finite());
            assert!(info_parallel.cond_est.is_finite());
            assert_matrix_close(&x_serial, &x_parallel, 1e-8);
            for i in 0..3 {
                assert!((x_serial[(i, 0)] - x_true[i]).abs() < 1e-8);
                assert!((x_parallel[(i, 0)] - x_true[i]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_solve_least_squares_qr_with_info_random_wide_min_norm() {
        let mut rng = TestRng::new(0x1234_5678_9abc_def0);
        let mut a = Matrix::new(2, 4);
        a[(0, 0)] = 1.0;
        a[(1, 1)] = 1.0;

        for _ in 0..5 {
            let b0 = rng.next_f64();
            let b1 = rng.next_f64();
            let b = Matrix::from_vec(vec![b0, b1], 2, 1).unwrap();

            let (x_serial, info_serial) =
                a.solve_least_squares_qr_with_info_with_parallel(&b, false)
                    .unwrap();
            let (x_parallel, info_parallel) =
                a.solve_least_squares_qr_with_info_with_parallel(&b, true)
                    .unwrap();
            assert_eq!(info_serial.rank, 2);
            assert_eq!(info_parallel.rank, 2);
            assert!(info_serial.cond_est.is_finite());
            assert!(info_parallel.cond_est.is_finite());
            assert_matrix_close(&x_serial, &x_parallel, 1e-12);
            assert!((x_serial[(0, 0)] - b0).abs() < 1e-12);
            assert!((x_serial[(1, 0)] - b1).abs() < 1e-12);
            assert!((x_serial[(2, 0)]).abs() < 1e-12);
            assert!((x_serial[(3, 0)]).abs() < 1e-12);
            assert!((x_parallel[(0, 0)] - b0).abs() < 1e-12);
            assert!((x_parallel[(1, 0)] - b1).abs() < 1e-12);
            assert!((x_parallel[(2, 0)]).abs() < 1e-12);
            assert!((x_parallel[(3, 0)]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_solve_least_squares_qr_with_info_detects_ill_conditioning() {
        let mut a = Matrix::identity(3);
        for i in 0..3 {
            a[(i, 2)] *= 1e-8;
        }
        let b = Matrix::from_vec(vec![0.0, 0.0, 0.0], 3, 1).unwrap();

        let (_x_serial, info_serial) =
            a.solve_least_squares_qr_with_info_with_parallel(&b, false)
                .unwrap();
        let (_x_parallel, info_parallel) =
            a.solve_least_squares_qr_with_info_with_parallel(&b, true)
                .unwrap();
        assert_eq!(info_serial.rank, 3);
        assert_eq!(info_parallel.rank, 3);
        assert!(info_serial.cond_est > 1e6, "cond_est was {}", info_serial.cond_est);
        assert!(info_parallel.cond_est > 1e6, "cond_est was {}", info_parallel.cond_est);
    }

    #[test]
    fn test_solve_least_squares_qr_with_info_tall_full_rank() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();

        let (x_serial, info_serial) =
            a.solve_least_squares_qr_with_info_with_parallel(&b, false)
                .unwrap();
        let (x_parallel, info_parallel) =
            a.solve_least_squares_qr_with_info_with_parallel(&b, true)
                .unwrap();
        assert_eq!(info_serial.rank, 2);
        assert_eq!(info_parallel.rank, 2);
        assert!(info_serial.cond_est.is_finite());
        assert!(info_parallel.cond_est.is_finite());
        assert_matrix_close(&x_serial, &x_parallel, 1e-12);
        assert!((x_serial[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((x_serial[(1, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_qr_with_info_wide_full_rank() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        let (x_serial, info_serial) =
            a.solve_least_squares_qr_with_info_with_parallel(&b, false)
                .unwrap();
        let (x_parallel, info_parallel) =
            a.solve_least_squares_qr_with_info_with_parallel(&b, true)
                .unwrap();
        assert_eq!(info_serial.rank, 2);
        assert_eq!(info_parallel.rank, 2);
        assert!(info_serial.cond_est.is_finite());
        assert!(info_parallel.cond_est.is_finite());
        assert_matrix_close(&x_serial, &x_parallel, 1e-12);
        assert!((x_serial[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((x_serial[(1, 0)] - 2.0).abs() < 1e-12);
        assert!((x_serial[(2, 0)] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_qr_with_info_rank_deficient() {
        let a = Matrix::from_vec(vec![1.0, 1.0, 2.0, 2.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        let err_serial = a
            .solve_least_squares_qr_with_info_with_parallel(&b, false)
            .expect_err("expected rank-deficient QR solve to fail");
        let err_parallel = a
            .solve_least_squares_qr_with_info_with_parallel(&b, true)
            .expect_err("expected rank-deficient QR solve to fail");
        assert!(err_serial.contains("rank deficient"), "{err_serial}");
        assert!(err_parallel.contains("rank deficient"), "{err_parallel}");
    }
}
