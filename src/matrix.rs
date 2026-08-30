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

//! Checked dense matrix operations and permutation-invariant Jacobi-SVD
//! least-squares solving.

use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

use crate::scaled::product_ratio;
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 16_384;

/// Identifies which matrix operand contained a non-finite value before a
/// checked linear solve began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixOperand {
    /// Coefficient matrix being factorized or applied by the operation.
    CoefficientMatrix,
    /// Right-hand-side matrix supplied to a linear solve.
    RightHandSide,
}

impl fmt::Display for MatrixOperand {
    /// Render a stable human-readable operand name for structured error output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixOperand::CoefficientMatrix => write!(f, "coefficient matrix"),
            MatrixOperand::RightHandSide => write!(f, "right-hand side"),
        }
    }
}

/// Decide whether an operation contains enough independent scalar work to
/// amortize Rayon scheduling overhead.
fn should_parallelize_work(estimated_scalar_work: usize) -> bool {
    // Callers use saturating arithmetic when deriving work estimates, so an
    // unrepresentably large operation is conservatively classified as large.
    estimated_scalar_work >= PARALLEL_THRESHOLD
}

/// Convert the relative numerical-rank threshold into the scale of a singular-
/// value spectrum.
///
/// A zero factor produces a zero tolerance and therefore rank zero because all
/// comparisons use strict inequality. Non-finite factors are rejected by the
/// factorization before this helper's result is consumed.
fn numerical_rank_tolerance(max_factor_value: f64, relative_tolerance: f64) -> f64 {
    // Multiplication, rather than `max_factor_value.max(1.0)`, preserves rank
    // under uniform scaling of the input matrix.
    relative_tolerance * max_factor_value
}

/// Derive the relative numerical-rank threshold from floating-point precision
/// and matrix size.
///
/// Multiplying machine epsilon by the larger dimension accounts for roundoff
/// accumulated across one factorization dimension without imposing a fixed
/// application-scale cutoff. The resulting threshold is then multiplied by the
/// factor scale through [`numerical_rank_tolerance`], preserving uniform-scale
/// invariance.
fn numerical_rank_relative_tolerance(rows: usize, cols: usize) -> f64 {
    f64::EPSILON * rows.max(cols) as f64
}

/// Structured failure returned by all fallible matrix construction and linear
/// algebra operations.
///
/// Variants retain machine-readable dimensions and operation names so callers
/// no longer need to parse unrelated ad-hoc strings from factorization routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    /// Two matrix operands do not have the shapes required by an operation.
    DimensionMismatch {
        /// Stable name of the operation that rejected the operands.
        operation: &'static str,
        /// Shape of the primary or left-hand matrix as `(rows, columns)`.
        left: (usize, usize),
        /// Shape of the secondary or right-hand matrix as `(rows, columns)`.
        right: (usize, usize),
    },
    /// A flat data buffer does not contain exactly one element per requested
    /// matrix position.
    InvalidDataLength {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        cols: usize,
        /// Required data length for the requested shape.
        expected: usize,
        /// Actual number of supplied elements.
        actual: usize,
    },
    /// Multiplying the requested dimensions overflowed `usize`, so the shape
    /// cannot be represented by the contiguous matrix storage.
    ElementCountOverflow {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        cols: usize,
    },
    /// The element count fit in `usize`, but contiguous storage could not be
    /// reserved for the requested matrix.
    AllocationFailed {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        cols: usize,
        /// Number of `f64` elements that the allocation would contain.
        elements: usize,
    },
    /// Adding logical dimensions overflowed `usize` before a matrix shape could
    /// be constructed.
    DimensionSumOverflow {
        /// Stable name of the operation assembling the combined dimension.
        operation: &'static str,
        /// First dimension participating in the overflowing addition.
        left: usize,
        /// Second dimension participating in the overflowing addition.
        right: usize,
    },
    /// A coefficient or right-hand-side input contained NaN or infinity before
    /// factorization began.
    NonFiniteInput {
        /// Stable name of the operation that validated the operand.
        operation: &'static str,
        /// Operand in which the invalid value was observed.
        operand: MatrixOperand,
    },
    /// Factorization arithmetic produced or encountered a non-finite diagonal
    /// value.
    NonFiniteFactor {
        /// Stable name of the factorization or solve that detected the value.
        operation: &'static str,
    },
    /// Finite inputs produced a non-finite solution through arithmetic overflow
    /// or invalid intermediate cancellation.
    NonFiniteResult {
        /// Stable name of the solve that produced the invalid solution.
        operation: &'static str,
    },
    /// An iterative factorization exhausted its bounded iteration budget before
    /// reaching the requested numerical orthogonality.
    FactorizationDidNotConverge {
        /// Stable name of the iterative factorization.
        operation: &'static str,
        /// Number of complete sweeps attempted before returning the error.
        sweeps: usize,
    },
}

/// Numerical diagnostics produced alongside a rectangular least-squares
/// solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresInfo {
    /// Numerical rank using a threshold relative to the largest singular value.
    pub rank: usize,
    /// Ratio of the largest to smallest retained singular value. Infinity means
    /// the matrix has no retained non-zero singular subspace.
    pub cond_est: f64,
}

/// Checked least-squares solution paired with the factorization diagnostics
/// that justify its numerical interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresSolution {
    /// Full-column, least-squares, or minimum-norm solution vector.
    pub solution: Matrix,
    /// Rank and retained-subspace condition estimate from the canonical SVD.
    pub info: LeastSquaresInfo,
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
            MatrixError::InvalidDataLength {
                rows,
                cols,
                expected,
                actual,
            } => write!(
                f,
                "Matrix data length mismatch for shape {rows}x{cols}: expected {expected}, got {actual}"
            ),
            MatrixError::ElementCountOverflow { rows, cols } => write!(
                f,
                "Matrix shape {rows}x{cols} exceeds the addressable element count"
            ),
            MatrixError::AllocationFailed {
                rows,
                cols,
                elements,
            } => write!(
                f,
                "Could not allocate {elements} elements for matrix shape {rows}x{cols}"
            ),
            MatrixError::DimensionSumOverflow {
                operation,
                left,
                right,
            } => write!(
                f,
                "Matrix dimension addition overflow during {operation}: {left} + {right}"
            ),
            MatrixError::NonFiniteInput { operation, operand } => {
                write!(f, "Non-finite {operand} supplied to {operation}")
            }
            MatrixError::NonFiniteFactor { operation } => {
                write!(f, "Non-finite factor encountered during {operation}")
            }
            MatrixError::NonFiniteResult { operation } => {
                write!(f, "Non-finite result produced during {operation}")
            }
            MatrixError::FactorizationDidNotConverge { operation, sweeps } => write!(
                f,
                "Factorization {operation} did not converge after {sweeps} sweeps"
            ),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Dense row-major matrix of `f64` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Contiguous row-major element storage.
    data: Vec<f64>,
    /// Number of logical rows represented by `data`.
    rows: usize,
    /// Number of logical columns represented by `data`.
    cols: usize,
}

/// Internal one-sided Jacobi SVD representation for a matrix with at least as
/// many rows as columns.
///
/// If the input is `A`, the fields satisfy `A * right_vectors =
/// orthogonal_columns`. Each non-zero orthogonal column is a left singular
/// vector scaled by the corresponding singular value. Keeping this compact
/// representation avoids allocating a separate dense left-singular matrix.
struct JacobiSvd {
    /// Rotated columns whose pairwise inner products are numerically zero.
    orthogonal_columns: Matrix,
    /// Accumulated orthogonal right-singular-vector matrix.
    right_vectors: Matrix,
    /// Euclidean norm of each rotated column.
    singular_values: Vec<f64>,
}

impl Matrix {
    /// Translate one logical coordinate into its row-major storage offset.
    ///
    /// Keeping both axis checks here prevents an invalid column from aliasing
    /// the next row and prevents unchecked row multiplication from wrapping to
    /// valid storage in optimized builds.
    ///
    /// # Panics
    ///
    /// Panics when `row >= self.rows` or `col >= self.cols`.
    fn offset(&self, row: usize, col: usize) -> usize {
        // Validate each logical axis before flattening. Checking only the final
        // vector offset cannot distinguish an invalid column from a valid cell
        // in the following row.
        assert!(
            row < self.rows,
            "matrix row index {row} is out of bounds for {} rows",
            self.rows
        );
        assert!(
            col < self.cols,
            "matrix column index {col} is out of bounds for {} columns",
            self.cols
        );

        // Every constructor proves that rows * columns fits in usize. With an
        // in-range row and column, this strictly smaller offset therefore
        // cannot overflow and always addresses the requested logical cell.
        row * self.cols + col
    }

    /// Allocate a zero-filled matrix with the requested shape.
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols` cannot be represented by `usize`, or if the
    /// allocator cannot satisfy the resulting request.
    pub fn new(rows: usize, cols: usize) -> Self {
        // Preserve the convenient infallible constructor for statically known
        // internal shapes while routing all validation through `try_new`.
        // Callers handling untrusted dimensions can use the fallible API below.
        Self::try_new(rows, cols).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Allocate a zero-filled matrix with a fully validated shape.
    ///
    /// Unlike [`Matrix::new`], this constructor reports both dimension overflow
    /// and storage-reservation failure as structured errors.
    pub fn try_new(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        // Detect dimension arithmetic overflow explicitly; allowing release-mode
        // wrapping would create storage inconsistent with the public shape.
        let element_count = rows
            .checked_mul(cols)
            .ok_or(MatrixError::ElementCountOverflow { rows, cols })?;

        // Reserve fallibly before initializing elements. `vec![value; len]`
        // uses an infallible capacity path and therefore panics for shapes such
        // as `usize::MAX x 1`, even though their element-count multiplication
        // itself does not overflow.
        let mut data = Vec::new();
        data.try_reserve_exact(element_count)
            .map_err(|_| MatrixError::AllocationFailed {
                rows,
                cols,
                elements: element_count,
            })?;
        // The exact reservation above guarantees that zero initialization does
        // not need another allocation attempt.
        data.resize(element_count, 0.0);

        Ok(Matrix { data, rows, cols })
    }

    /// Construct a row-major matrix from an owned element buffer.
    ///
    /// Returns structured errors when the requested element count overflows or
    /// the supplied buffer length does not exactly match the shape.
    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, MatrixError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(MatrixError::ElementCountOverflow { rows, cols })?;
        if data.len() != expected {
            return Err(MatrixError::InvalidDataLength {
                rows,
                cols,
                expected,
                actual: data.len(),
            });
        }
        Ok(Matrix { data, rows, cols })
    }

    /// Construct a square identity matrix of the requested size.
    pub fn identity(size: usize) -> Self {
        // Begin with zero storage, then set exactly the main diagonal to one.
        let mut m = Matrix::new(size, size);
        for i in 0..size {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Return the logical row count.
    pub fn rows(&self) -> usize {
        // Shape metadata is immutable after construction except inside checked
        // workspace resizing, so returning it by value requires no validation.
        self.rows
    }

    /// Return the logical column count.
    pub fn cols(&self) -> usize {
        // The column count participates in every row-major index calculation.
        self.cols
    }

    /// Return a serially computed transpose of this matrix.
    pub fn transpose(&self) -> Matrix {
        // Route through the mode-aware implementation so the two paths share
        // allocation, indexing, and empty-shape semantics.
        self.transpose_with_parallel(false)
    }

    /// Return the transpose while optionally parallelizing a sufficiently large
    /// element copy.
    pub fn transpose_with_parallel(&self, parallel: bool) -> Matrix {
        // The transposed shape reverses the logical dimensions while preserving
        // the same total element count.
        let mut result = Matrix::new(self.cols, self.rows);
        if result.data.is_empty() {
            // A valid shape such as `usize::MAX x 0` has no coordinates to
            // copy. Returning from storage cardinality avoids walking the huge
            // logical row dimension even though the operation is already
            // complete, and also avoids zero-sized parallel chunks.
            return result;
        }
        let element_count = self.rows.saturating_mul(self.cols);
        if parallel && should_parallelize_work(element_count) {
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

    /// Solve a possibly rectangular least-squares problem by canonical SVD.
    ///
    /// A one-sided Jacobi decomposition classifies numerical rank from singular
    /// values and computes the Moore-Penrose minimum-norm solution for square,
    /// tall, and wide systems. Using the same decomposition for classification
    /// and solution makes rank invariant under column permutation and avoids a
    /// competing factorization policy whose intermediate arithmetic could fail
    /// on otherwise representable systems.
    pub fn solve_least_squares(&self, b: &Matrix) -> Result<LeastSquaresSolution, MatrixError> {
        // Validate the vector shape before the factorization reads it. The
        // solve API intentionally accepts one right-hand side so its
        // minimum-norm and diagnostic contract remains unambiguous.
        if b.cols != 1 {
            return Err(MatrixError::DimensionMismatch {
                operation: "solve_least_squares",
                left: (self.rows, 1),
                right: (b.rows, b.cols),
            });
        }
        if b.rows != self.rows {
            return Err(MatrixError::DimensionMismatch {
                operation: "solve_least_squares",
                left: (self.rows, 1),
                right: (b.rows, b.cols),
            });
        }

        // Reject invalid floating-point inputs before SVD can turn them into
        // misleading rank or convergence diagnostics.
        if !self.all_finite() {
            return Err(MatrixError::NonFiniteInput {
                operation: "solve_least_squares",
                operand: MatrixOperand::CoefficientMatrix,
            });
        }
        if !b.all_finite() {
            return Err(MatrixError::NonFiniteInput {
                operation: "solve_least_squares",
                operand: MatrixOperand::RightHandSide,
            });
        }

        let m = self.rows;
        let n = self.cols;
        // One factorization-derived policy governs SVD truncation. Callers
        // cannot accidentally discard every direction or encode application
        // unit assumptions in a matrix-level option.
        let relative_rank_tolerance = numerical_rank_relative_tolerance(m, n);

        // Trivial cases have a unique zero-length or minimum-norm zero solution
        // and require no numerical factorization. The returned variable vector
        // can still be impossibly large when the coefficient matrix has zero
        // rows, so preserve the fallible solve contract while allocating it.
        if n == 0 || m == 0 {
            return Ok(LeastSquaresSolution {
                solution: Matrix::try_new(n, 1)?,
                info: LeastSquaresInfo {
                    rank: 0,
                    cond_est: f64::INFINITY,
                },
            });
        }

        // The decomposition supplies both the returned correction and the rank
        // diagnostics, so classification cannot disagree with solution policy.
        let result = Self::solve_least_squares_svd_with_info(self, b, relative_rank_tolerance)?;

        // A successful checked solve must never smuggle NaN or infinity through
        // its `Ok` channel, even when finite inputs overflow during substitution.
        if !result.0.all_finite() {
            return Err(MatrixError::NonFiniteResult {
                operation: "solve_least_squares",
            });
        }
        Ok(LeastSquaresSolution {
            solution: result.0,
            info: result.1,
        })
    }

    /// Compute a Moore-Penrose pseudoinverse solution with a one-sided Jacobi
    /// singular-value decomposition.
    ///
    /// The Jacobi kernel operates on a matrix with at least as many rows as
    /// columns. Tall inputs are decomposed directly; wide inputs decompose the
    /// transpose and apply the transposed SVD identities. In both orientations,
    /// singular directions below the relative rank tolerance are omitted,
    /// which selects the unique minimum-Euclidean-norm least-squares solution.
    fn solve_least_squares_svd_with_info(
        a: &Matrix,
        b: &Matrix,
        relative_rank_tolerance: f64,
    ) -> Result<(Matrix, LeastSquaresInfo), MatrixError> {
        let rows = a.rows;
        let cols = a.cols;
        let tall_orientation = rows >= cols;

        // Normalize the complete coefficient matrix before factorization. A
        // matrix can contain only finite entries while its largest singular
        // value exceeds `f64::MAX` (for example, a 2x2 matrix filled with
        // `1e308`). Uniform normalization preserves singular vectors, relative
        // rank, condition, and the exact least-squares problem while keeping
        // every factorization norm representable. The all-zero matrix uses a
        // neutral scale because it has no non-zero direction to normalize.
        let coefficient_scale = a
            .data
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let coefficient_scale = if coefficient_scale == 0.0 {
            1.0
        } else {
            coefficient_scale
        };

        // Decomposing A^T for a wide system keeps the Jacobi kernel compact and
        // avoids manufacturing zero singular columns merely to make A tall.
        // Build either orientation through the checked constructor instead of
        // using `Clone` or the infallible public transpose operation: every
        // allocation beneath this `Result`-returning API must remain observable
        // as `MatrixError::AllocationFailed`.
        let factor_rows = rows.max(cols);
        let factor_cols = rows.min(cols);
        let mut factor_input = Matrix::try_new(factor_rows, factor_cols)?;
        if tall_orientation {
            // Combine the copy and coefficient normalization into one pass so
            // the owned Jacobi workspace is the only duplicate of A.
            for (target, source) in factor_input.data.iter_mut().zip(&a.data) {
                *target = *source / coefficient_scale;
            }
        } else {
            // Materialize the normalized transpose directly. Both source axes
            // are known to be non-empty here, so every logical coordinate maps
            // to one physically allocated destination element.
            for row in 0..rows {
                for column in 0..cols {
                    factor_input[(column, row)] = a[(row, column)] / coefficient_scale;
                }
            }
        }
        let decomposition = Self::one_sided_jacobi_svd_columns(factor_input)?;

        // A relative threshold preserves the numerical rank under uniform unit
        // changes. The strict comparison deliberately classifies an all-zero
        // factor as rank zero when both the maximum and tolerance are zero.
        let max_singular_value = decomposition
            .singular_values
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let rank_tolerance = numerical_rank_tolerance(max_singular_value, relative_rank_tolerance);
        let mut rank = 0usize;
        let mut min_retained_singular_value = f64::INFINITY;
        for &singular_value in &decomposition.singular_values {
            if singular_value > rank_tolerance {
                rank += 1;
                min_retained_singular_value = min_retained_singular_value.min(singular_value);
            }
        }

        // Report conditioning only on the retained image of A. Discarded null
        // directions do not make the pseudoinverse solution itself ambiguous.
        let cond_est = if rank == 0 {
            f64::INFINITY
        } else {
            max_singular_value / min_retained_singular_value
        };
        // Allocate the public result through the same checked path as every
        // factorization workspace so memory failure cannot escape by panic.
        let mut solution = Matrix::try_new(cols, 1)?;

        for component in 0..decomposition.singular_values.len() {
            let singular_value = decomposition.singular_values[component];
            if singular_value <= rank_tolerance {
                // Omitting this singular direction is exactly the pseudoinverse
                // rule that removes unsupported null-space components.
                continue;
            }

            if tall_orientation {
                // A*V=B=U*Sigma, so this coefficient is
                // (U_component^T*b)/sigma. Normalizing B before the dot product
                // prevents squaring a large or tiny singular value explicitly.
                let projection = Self::compensated_scaled_column_dot(
                    &decomposition.orthogonal_columns,
                    component,
                    singular_value,
                    b,
                    "solve_least_squares",
                )?;
                // `singular_value` belongs to A/coefficient_scale. Restore the
                // original scale through exponent arithmetic so neither the
                // unscaled singular value nor a fragile intermediate quotient
                // has to be materialized.
                let coefficient = product_ratio([projection], [singular_value, coefficient_scale]);
                if !coefficient.is_finite() {
                    return Err(MatrixError::NonFiniteResult {
                        operation: "solve_least_squares",
                    });
                }

                // V maps each retained singular-coordinate coefficient back to
                // the original variable space.
                for row in 0..cols {
                    let contribution = decomposition.right_vectors[(row, component)] * coefficient;
                    solution[(row, 0)] += contribution;
                    if !solution[(row, 0)].is_finite() {
                        return Err(MatrixError::NonFiniteResult {
                            operation: "solve_least_squares",
                        });
                    }
                }
            } else {
                // A^T*V=B=U*Sigma implies A=V*Sigma*U^T. Project the right-hand
                // side onto V, then map the scaled coefficient into U to obtain
                // the minimum-norm variable vector.
                let projection = Self::compensated_scaled_column_dot(
                    &decomposition.right_vectors,
                    component,
                    1.0,
                    b,
                    "solve_least_squares",
                )?;
                // Restore the coefficient scale through the same exponent-safe
                // ratio used by the tall orientation above.
                let coefficient = product_ratio([projection], [singular_value, coefficient_scale]);
                if !coefficient.is_finite() {
                    return Err(MatrixError::NonFiniteResult {
                        operation: "solve_least_squares",
                    });
                }

                // A column of B divided by sigma is the corresponding left
                // singular vector U in the original variable space.
                for row in 0..cols {
                    let variable_direction =
                        decomposition.orthogonal_columns[(row, component)] / singular_value;
                    let contribution = variable_direction * coefficient;
                    solution[(row, 0)] += contribution;
                    if !solution[(row, 0)].is_finite() {
                        return Err(MatrixError::NonFiniteResult {
                            operation: "solve_least_squares",
                        });
                    }
                }
            }
        }

        Ok((solution, LeastSquaresInfo { rank, cond_est }))
    }

    /// Orthogonalize the columns of a tall-or-square matrix with cyclic Jacobi
    /// rotations while accumulating its right singular vectors.
    ///
    /// Pairwise Gram entries are evaluated after scaling both columns by their
    /// largest magnitude. That scaling prevents the convergence test and
    /// rotation angle from overflowing for otherwise finite input matrices.
    fn one_sided_jacobi_svd_columns(
        mut orthogonal_columns: Matrix,
    ) -> Result<JacobiSvd, MatrixError> {
        debug_assert!(orthogonal_columns.rows >= orthogonal_columns.cols);

        // Take ownership of the normalized factor input so the Jacobi kernel
        // can rotate it in place without a second full-matrix clone. The right
        // singular-vector basis is also allocated fallibly because an existing
        // source matrix does not guarantee enough memory for another workspace.
        let row_count = orthogonal_columns.rows;
        let column_count = orthogonal_columns.cols;
        let mut right_vectors = Matrix::try_new(column_count, column_count)?;
        for diagonal in 0..column_count {
            right_vectors[(diagonal, diagonal)] = 1.0;
        }

        // Cyclic Jacobi converges rapidly for the small dense systems targeted
        // by this crate. The dimension-aware cap is finite so pathological
        // arithmetic returns a structured error instead of looping forever.
        let max_sweeps = 30usize.saturating_add(column_count.saturating_mul(2));
        let mut converged = false;

        for _sweep in 0..max_sweeps {
            let mut rotated_any_pair = false;

            for left_column in 0..column_count {
                for right_column in (left_column + 1)..column_count {
                    // Find a shared scale for the two columns. A pair of zero
                    // columns is already orthogonal and requires no rotation.
                    let mut pair_scale = 0.0_f64;
                    for row in 0..row_count {
                        pair_scale = pair_scale
                            .max(orthogonal_columns[(row, left_column)].abs())
                            .max(orthogonal_columns[(row, right_column)].abs());
                    }
                    if pair_scale == 0.0 {
                        continue;
                    }

                    // Form the two-by-two Gram matrix in scaled coordinates.
                    // Each summand is bounded, keeping the angle computation
                    // finite whenever the number of rows is representable.
                    let mut left_norm_squared = 0.0;
                    let mut right_norm_squared = 0.0;
                    let mut cross_product = 0.0;
                    for row in 0..row_count {
                        let left = orthogonal_columns[(row, left_column)] / pair_scale;
                        let right = orthogonal_columns[(row, right_column)] / pair_scale;
                        left_norm_squared += left * left;
                        right_norm_squared += right * right;
                        cross_product += left * right;
                    }

                    if !left_norm_squared.is_finite()
                        || !right_norm_squared.is_finite()
                        || !cross_product.is_finite()
                    {
                        return Err(MatrixError::NonFiniteFactor {
                            operation: "one_sided_jacobi_svd",
                        });
                    }

                    // Skip pairs whose correlation is at floating-point noise
                    // level. The product is non-negative, so its square root is
                    // a scale-aware bound for the cross term.
                    let orthogonality_threshold =
                        8.0 * f64::EPSILON * (left_norm_squared * right_norm_squared).sqrt();
                    if cross_product.abs() <= orthogonality_threshold {
                        continue;
                    }

                    // Choose the stable root of the Jacobi tangent equation.
                    // This form avoids cancellation when the two column norms
                    // differ substantially and bounds |tangent| by one.
                    let zeta = (right_norm_squared - left_norm_squared) / (2.0 * cross_product);
                    let tangent = if zeta >= 0.0 {
                        1.0 / (zeta + zeta.hypot(1.0))
                    } else {
                        -1.0 / (-zeta + zeta.hypot(1.0))
                    };
                    let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                    let sine = cosine * tangent;

                    // Apply the rotation to A's working columns. Saving both old
                    // entries before either write preserves the exact paired
                    // transformation at every row.
                    for row in 0..row_count {
                        let old_left = orthogonal_columns[(row, left_column)];
                        let old_right = orthogonal_columns[(row, right_column)];
                        orthogonal_columns[(row, left_column)] =
                            cosine * old_left - sine * old_right;
                        orthogonal_columns[(row, right_column)] =
                            sine * old_left + cosine * old_right;
                    }

                    // Accumulate the same rotation in V so the invariant
                    // source*V=orthogonal_columns remains true.
                    for row in 0..column_count {
                        let old_left = right_vectors[(row, left_column)];
                        let old_right = right_vectors[(row, right_column)];
                        right_vectors[(row, left_column)] = cosine * old_left - sine * old_right;
                        right_vectors[(row, right_column)] = sine * old_left + cosine * old_right;
                    }

                    rotated_any_pair = true;
                }
            }

            // A full sweep without a material rotation means every column pair
            // met the relative orthogonality criterion.
            if !rotated_any_pair {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(MatrixError::FactorizationDidNotConverge {
                operation: "one_sided_jacobi_svd",
                sweeps: max_sweeps,
            });
        }
        if !orthogonal_columns.all_finite() || !right_vectors.all_finite() {
            return Err(MatrixError::NonFiniteFactor {
                operation: "one_sided_jacobi_svd",
            });
        }

        // Column norms are the singular values after orthogonalization. Repeated
        // `hypot` calls avoid overflow and underflow in the sum of squares.
        // A `Vec::with_capacity` panic would violate the surrounding fallible
        // factorization contract. Reserve the compact singular-value buffer
        // explicitly and map any capacity or allocator failure to the same
        // structured matrix allocation error used by dense workspaces.
        let mut singular_values = Vec::new();
        singular_values
            .try_reserve_exact(column_count)
            .map_err(|_| MatrixError::AllocationFailed {
                rows: column_count,
                cols: 1,
                elements: column_count,
            })?;
        for column in 0..column_count {
            let mut singular_value = 0.0_f64;
            for row in 0..row_count {
                singular_value = singular_value.hypot(orthogonal_columns[(row, column)]);
            }
            if !singular_value.is_finite() {
                return Err(MatrixError::NonFiniteFactor {
                    operation: "one_sided_jacobi_svd",
                });
            }
            singular_values.push(singular_value);
        }

        Ok(JacobiSvd {
            orthogonal_columns,
            right_vectors,
            singular_values,
        })
    }

    /// Compute a compensated dot product between one normalized matrix column
    /// and a single-column right-hand side.
    ///
    /// `column_scale` is normally the column's singular value, producing a unit
    /// left singular vector without allocating it. Passing one computes an
    /// ordinary column dot product for an already normalized orthogonal matrix.
    fn compensated_scaled_column_dot(
        left: &Matrix,
        left_column: usize,
        column_scale: f64,
        right: &Matrix,
        operation: &'static str,
    ) -> Result<f64, MatrixError> {
        debug_assert_eq!(left.rows, right.rows);
        debug_assert_eq!(right.cols, 1);
        debug_assert!(column_scale > 0.0 && column_scale.is_finite());

        let mut sum = 0.0;
        let mut correction = 0.0;
        for row in 0..left.rows {
            let product = (left[(row, left_column)] / column_scale) * right[(row, 0)];
            if !product.is_finite() {
                return Err(MatrixError::NonFiniteResult { operation });
            }

            // Neumaier compensation retains low-order terms even when the next
            // product is much larger than the running sum.
            let next_sum = sum + product;
            if sum.abs() >= product.abs() {
                correction += (sum - next_sum) + product;
            } else {
                correction += (product - next_sum) + sum;
            }
            sum = next_sum;
            if !sum.is_finite() || !correction.is_finite() {
                return Err(MatrixError::NonFiniteResult { operation });
            }
        }

        let result = sum + correction;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(MatrixError::NonFiniteResult { operation })
        }
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

    /// Return whether every stored matrix element is a finite floating-point
    /// value.
    ///
    /// Solver code uses this check at numerical boundaries so NaN and infinity
    /// are reported as evaluation failures rather than flowing into rank,
    /// convergence, or regularization logic.
    pub fn all_finite(&self) -> bool {
        // `Iterator::all` stops at the first invalid element and does not expose
        // the matrix's private storage to callers.
        self.data.iter().all(|value| value.is_finite())
    }

    /// Add two identically shaped matrices element by element.
    pub fn try_add(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        // Elementwise arithmetic has no meaningful broadcasting in this dense
        // API, so require both dimensions to match exactly.
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "add",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }

        // Preserve the fallible contract through result allocation. This can
        // fail under memory pressure even though both existing operands have a
        // valid and identical shape.
        let mut result = Matrix::try_new(self.rows, self.cols)?;
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + rhs.data[i];
        }
        Ok(result)
    }

    /// Subtract an identically shaped right-hand matrix element by element.
    pub fn try_sub(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        // Preserve the same strict shape contract as addition before allocating
        // the result buffer.
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "sub",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }

        // Allocate through the checked constructor for the same reason as
        // addition: a fallible arithmetic method must not panic while creating
        // its output buffer.
        let mut result = Matrix::try_new(self.rows, self.cols)?;
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - rhs.data[i];
        }
        Ok(result)
    }

    /// Multiply two shape-compatible matrices serially.
    pub fn try_mul(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        // Delegate validation and arithmetic to the mode-aware implementation so
        // both public entry points retain identical error behavior.
        self.try_mul_with_parallel(rhs, false)
    }

    /// Multiply two matrices while optionally parallelizing sufficiently large
    /// row computations.
    pub fn try_mul_with_parallel(
        &self,
        rhs: &Matrix,
        parallel: bool,
    ) -> Result<Matrix, MatrixError> {
        // The shared dimension must match before any row slice is constructed.
        if self.cols != rhs.rows {
            return Err(MatrixError::DimensionMismatch {
                operation: "mul",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }

        // The output combines dimensions from two independently valid inputs.
        // Zero-sized storage permits shapes such as `usize::MAX x 0`, so input
        // construction alone cannot prove that `self.rows * rhs.cols` fits.
        // Preserve this method's fallible contract by validating the derived
        // shape instead of routing it through the panicking convenience
        // constructor.
        let mut result = Matrix::try_new(self.rows, rhs.cols)?;
        if result.data.is_empty() {
            // No output cell exists when either output dimension is zero. In
            // particular, a valid `usize::MAX x 0` result must not execute an
            // effectively unbounded outer-row loop whose body can never write.
            return Ok(result);
        }
        // Each output cell performs `self.cols` multiply-adds. Including that
        // inner dimension recognizes skinny products with few output cells but
        // substantial arithmetic, which the former output-size test missed.
        let scalar_work = self.rows.saturating_mul(rhs.cols).saturating_mul(self.cols);
        if parallel && should_parallelize_work(scalar_work) {
            let rhs_cols = rhs.cols;
            // Assign complete output rows to workers. Each row is disjoint, so
            // accumulation needs no locks and retains serial summation order
            // within every individual result cell.
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

    /// Borrow the element at one validated logical matrix coordinate.
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        // Use the shared axis-aware translation so immutable and mutable
        // indexing have identical bounds and overflow behavior.
        let offset = self.offset(row, col);
        &self.data[offset]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    /// Mutably borrow the element at one validated logical matrix coordinate.
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        // Compute the checked offset before borrowing storage mutably; this
        // guarantees a failed coordinate cannot modify an aliased valid cell.
        let offset = self.offset(row, col);
        &mut self.data[offset]
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
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
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

    /// Run the canonical least-squares API with its automatic singular-value
    /// rank policy.
    fn solve_default(coefficients: &Matrix, rhs: &Matrix) -> LeastSquaresSolution {
        // Unwrap here so individual numerical tests can focus on their expected
        // solution and diagnostic fields.
        coefficients
            .solve_least_squares(rhs)
            .expect("least-squares solve failed")
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

    /// Cover finite-state detection for empty, ordinary, NaN, and infinite
    /// matrices.
    #[test]
    fn test_all_finite_identifies_invalid_components() {
        let empty = Matrix::new(0, 0);
        let finite = Matrix::from_vec(vec![-1.0, 0.0, 2.5], 3, 1).unwrap();
        let with_nan = Matrix::from_vec(vec![0.0, f64::NAN], 2, 1).unwrap();
        let with_infinity = Matrix::from_vec(vec![f64::NEG_INFINITY], 1, 1).unwrap();

        assert!(empty.all_finite());
        assert!(finite.all_finite());
        assert!(!with_nan.all_finite());
        assert!(!with_infinity.all_finite());
    }

    /// Ensure the checked factorization API rejects invalid operands through
    /// its error channel rather than returning a solution containing NaN.
    #[test]
    fn test_least_squares_rejects_non_finite_inputs() {
        let identity = Matrix::identity(2);
        let nan_rhs = Matrix::from_vec(vec![f64::NAN, 1.0], 2, 1).unwrap();

        assert_eq!(
            identity.solve_least_squares(&nan_rhs).unwrap_err(),
            MatrixError::NonFiniteInput {
                operation: "solve_least_squares",
                operand: MatrixOperand::RightHandSide,
            }
        );

        let nan_coefficient = Matrix::from_vec(vec![1.0, f64::NAN, 0.0, 1.0], 2, 2).unwrap();
        let finite_rhs = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();
        assert_eq!(
            nan_coefficient
                .solve_least_squares(&finite_rhs)
                .unwrap_err(),
            MatrixError::NonFiniteInput {
                operation: "solve_least_squares",
                operand: MatrixOperand::CoefficientMatrix,
            }
        );
    }

    /// Confirm that arithmetic overflow from otherwise finite solve operands is
    /// reported explicitly instead of escaping inside an `Ok(Matrix)` value.
    #[test]
    fn test_least_squares_rejects_non_finite_results() {
        let tiny_coefficient = Matrix::from_vec(vec![1e-308], 1, 1).unwrap();
        let huge_rhs = Matrix::from_vec(vec![f64::MAX], 1, 1).unwrap();

        assert_eq!(
            tiny_coefficient.solve_least_squares(&huge_rhs).unwrap_err(),
            MatrixError::NonFiniteResult {
                operation: "solve_least_squares",
            }
        );
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

    /// Complete zero-storage operations from physical cardinality rather than
    /// iterating over enormous logical dimensions that contain no coordinates.
    #[test]
    fn test_zero_storage_transpose_and_multiplication_return_immediately() {
        let huge_empty = Matrix::from_vec(Vec::new(), usize::MAX, 0)
            .expect("a zero-column matrix requires no storage");
        let empty_right =
            Matrix::from_vec(Vec::new(), 0, 0).expect("a zero-by-zero matrix is valid");

        for parallel in [false, true] {
            let transposed = huge_empty.transpose_with_parallel(parallel);
            assert_eq!(transposed.rows(), 0);
            assert_eq!(transposed.cols(), usize::MAX);
            assert_eq!(transposed.norm(), 0.0);

            let product = huge_empty
                .try_mul_with_parallel(&empty_right, parallel)
                .expect("a product with zero output columns is already complete");
            assert_eq!(product.rows(), usize::MAX);
            assert_eq!(product.cols(), 0);
            assert_eq!(product.norm(), 0.0);
        }
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

    /// Verify that a product with few outputs but a long shared dimension is
    /// classified by its actual arithmetic and remains numerically equivalent.
    #[test]
    fn test_parallel_matmul_counts_inner_dimension_work() {
        let rows = 4usize;
        let inner = 4_096usize;
        let cols = 4usize;

        // Sixteen output cells alone fall below the scheduling threshold, while
        // their 65,536 multiply-adds correctly qualify as substantial work.
        let estimated_work = rows.saturating_mul(cols).saturating_mul(inner);
        assert!(should_parallelize_work(estimated_work));

        let a = Matrix::from_vec(vec![1.0; rows * inner], rows, inner).unwrap();
        let b = Matrix::from_vec(vec![1.0; inner * cols], inner, cols).unwrap();
        let serial = a.try_mul_with_parallel(&b, false).unwrap();
        let parallel = a.try_mul_with_parallel(&b, true).unwrap();

        assert_matrix_close(&serial, &parallel, 0.0);
        for row in 0..rows {
            for column in 0..cols {
                assert_eq!(serial[(row, column)], inner as f64);
            }
        }
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

    /// Ensure fallible multiplication reports overflow in its derived output
    /// shape rather than panicking after both zero-storage inputs were accepted.
    #[test]
    fn test_try_mul_reports_derived_element_count_overflow() {
        // Each input has zero elements and is independently representable. Their
        // compatible shared zero dimension nevertheless produces an impossible
        // `usize::MAX x usize::MAX` output shape.
        let left = Matrix::from_vec(Vec::new(), usize::MAX, 0).unwrap();
        let right = Matrix::from_vec(Vec::new(), 0, usize::MAX).unwrap();

        for parallel in [false, true] {
            let error = left
                .try_mul_with_parallel(&right, parallel)
                .expect_err("derived output overflow must remain fallible");
            assert_eq!(
                error,
                MatrixError::ElementCountOverflow {
                    rows: usize::MAX,
                    cols: usize::MAX,
                }
            );
        }
    }

    #[test]
    fn test_solve_least_squares_overdetermined_exact() {
        // A is 3x2, consistent system with exact solution x=[1,2].
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();

        let result = solve_default(&a, &b);
        assert!((result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_underdetermined_min_norm() {
        // A is 1x2: x0 + x1 = 1. The minimum-norm solution is [0.5, 0.5].
        let a = Matrix::from_vec(vec![1.0, 1.0], 1, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0], 1, 1).unwrap();

        let result = solve_default(&a, &b);
        assert!((result.solution[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 0.5).abs() < 1e-12);
    }

    /// Ensure that uniformly scaling a tall matrix does not change its
    /// numerical rank or condition estimate.
    #[test]
    fn test_solve_least_squares_tall_rank_is_scale_invariant() {
        // This matrix is exactly the small-scale case that the former absolute
        // tolerance floor misreported as rank zero.
        let tiny = Matrix::from_vec(vec![1e-15, 2e-15], 2, 1).unwrap();
        let tiny_b = Matrix::from_vec(vec![1e-15, 2e-15], 2, 1).unwrap();
        let tiny_result = solve_default(&tiny, &tiny_b);

        let unit = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();
        let unit_b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();
        let unit_result = solve_default(&unit, &unit_b);

        assert_eq!(tiny_result.info.rank, 1);
        assert_eq!(tiny_result.info.rank, unit_result.info.rank);
        assert!((tiny_result.info.cond_est - unit_result.info.cond_est).abs() < 1e-12);
        assert!((tiny_result.solution[(0, 0)] - 1.0).abs() < 1e-12);
    }

    /// Ensure that the wide SVD orientation applies the same scale-relative
    /// rank rule as the tall orientation.
    #[test]
    fn test_solve_least_squares_wide_rank_is_scale_invariant() {
        // The equation 1e-15*x0 + 2e-15*x1 = 1e-15 is full row rank and has
        // minimum-norm solution [0.2, 0.4].
        let tiny = Matrix::from_vec(vec![1e-15, 2e-15], 1, 2).unwrap();
        let tiny_b = Matrix::from_vec(vec![1e-15], 1, 1).unwrap();
        let result = solve_default(&tiny, &tiny_b);

        assert_eq!(result.info.rank, 1);
        assert!(result.info.cond_est.is_finite());
        assert!((result.solution[(0, 0)] - 0.2).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 0.4).abs() < 1e-12);
    }

    /// Keep algebraic directions that are small in application units but remain
    /// distinguishable from factorization roundoff.
    #[test]
    fn test_solve_least_squares_uses_precision_derived_rank_threshold() {
        // For a 2x2 factor, eps*2 is far below 1e-13. A fixed application-level
        // threshold of 1e-12 incorrectly discarded this independent direction.
        let coefficients = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1e-13], 2, 2).unwrap();
        let rhs = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();
        let result = solve_default(&coefficients, &rhs);

        assert_eq!(result.info.rank, 2);
        assert!((result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 1e13).abs() < 1e-3);
    }

    /// Keep numerical rank and the Moore-Penrose solution invariant when a
    /// caller merely reorders the variables represented by matrix columns.
    #[test]
    fn test_solve_least_squares_is_invariant_under_column_permutation() {
        // The tiny determinant makes this matrix numerically rank one under the
        // precision-derived SVD threshold. The former unpivoted QR dispatcher
        // classified the original ordering as rank two but the permutation as
        // rank one, so equivalent variable orderings followed different solve
        // policies and produced unrelated answers.
        let original = Matrix::from_vec(vec![1e-8, 1.0, 0.0, 1e-8], 2, 2).unwrap();
        let permuted = Matrix::from_vec(vec![1.0, 1e-8, 1e-8, 0.0], 2, 2).unwrap();
        let right_hand_side = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();

        let original_result = solve_default(&original, &right_hand_side);
        let permuted_result = solve_default(&permuted, &right_hand_side);

        assert_eq!(original_result.info.rank, 1);
        assert_eq!(permuted_result.info.rank, original_result.info.rank);
        assert_eq!(permuted_result.info.cond_est, original_result.info.cond_est);
        assert!(
            (original_result.solution[(0, 0)] - permuted_result.solution[(1, 0)]).abs() < 1e-12
        );
        assert!(
            (original_result.solution[(1, 0)] - permuted_result.solution[(0, 0)]).abs() < 1e-12
        );
    }

    /// Verify that SVD projection remains representable for a maximum-scale
    /// tall system whose naïve squared column norm would overflow.
    #[test]
    fn test_tall_svd_handles_extreme_finite_scale() {
        let coefficient = Matrix::from_vec(vec![1e308, 1e308], 2, 1).unwrap();
        let right_hand_side = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();

        let result = solve_default(&coefficient, &right_hand_side);

        // Both equations are identical after scaling, so x=1e-308 is the exact
        // least-squares solution. Compare by ratio because the expected value is
        // subnormal enough that an absolute tolerance would hide a large error.
        let relative_error = (result.solution[(0, 0)] / 1e-308 - 1.0).abs();
        assert!(relative_error < 1e-12, "relative error: {relative_error}");
        assert_eq!(result.info.rank, 1);
    }

    /// Solve a maximum-scale full-column-rank system without overflowing an
    /// intermediate reflector application or residual projection.
    #[test]
    fn test_tall_svd_handles_extreme_finite_coefficients_and_rhs() {
        let coefficient = Matrix::from_vec(vec![1e308, 1e308], 2, 1).unwrap();
        let right_hand_side = Matrix::from_vec(vec![1e308, 1e308], 2, 1).unwrap();

        let result = solve_default(&coefficient, &right_hand_side);

        assert!((result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(result.info.rank, 1);
        assert!((result.info.cond_est - 1.0).abs() < 1e-12);
    }

    /// Exercise the same scale-safe SVD arithmetic through the wide orientation
    /// used for minimum-norm systems.
    #[test]
    fn test_wide_svd_handles_extreme_finite_scale() {
        let coefficient = Matrix::from_vec(vec![1e308, 1e308], 1, 2).unwrap();
        let right_hand_side = Matrix::from_vec(vec![1.0], 1, 1).unwrap();

        let result = solve_default(&coefficient, &right_hand_side);

        // Symmetry gives the minimum-norm solution [5e-309, 5e-309]. Check each
        // component relatively so the regression remains sensitive at this scale.
        for row in 0..2 {
            let relative_error = (result.solution[(row, 0)] / 5e-309 - 1.0).abs();
            assert!(relative_error < 1e-12, "relative error: {relative_error}");
        }
        assert_eq!(result.info.rank, 1);
    }

    #[test]
    fn test_solve_least_squares_random_tall_full_rank() {
        let mut rng = TestRng::new(0x5eed_1234_5678_9abc);
        for _ in 0..5 {
            let mut a = random_matrix(6, 3, &mut rng);
            for i in 0..3 {
                a[(i, i)] += 2.0;
            }
            let x_true = vec![rng.next_f64(), rng.next_f64(), rng.next_f64()];
            let x_mat = Matrix::from_vec(x_true.clone(), 3, 1).unwrap();
            let b = &a * &x_mat;

            let result = solve_default(&a, &b);
            assert_eq!(result.info.rank, 3);
            assert!(result.info.cond_est.is_finite());
            for (row, expected) in x_true.iter().enumerate() {
                assert!((result.solution[(row, 0)] - expected).abs() < 1e-8);
            }
        }
    }

    /// Verify that callers with untrusted dimensions can observe shape overflow
    /// as structured data instead of triggering the infallible constructor's
    /// documented panic.
    #[test]
    fn test_try_new_reports_element_count_overflow() {
        let error = Matrix::try_new(usize::MAX, 2)
            .expect_err("overflowing element counts must be rejected");

        assert_eq!(
            error,
            MatrixError::ElementCountOverflow {
                rows: usize::MAX,
                cols: 2,
            }
        );
    }

    /// Verify that a representable element count with an impossible byte
    /// capacity is returned as an allocation error instead of panicking.
    #[test]
    fn test_try_new_reports_capacity_overflow() {
        let error = Matrix::try_new(usize::MAX, 1)
            .expect_err("an impossible contiguous allocation must be rejected");

        assert_eq!(
            error,
            MatrixError::AllocationFailed {
                rows: usize::MAX,
                cols: 1,
                elements: usize::MAX,
            }
        );
    }

    /// Preserve the fallible least-squares contract for a valid zero-storage
    /// coefficient shape whose minimum-norm output cannot be allocated.
    #[test]
    fn test_zero_row_least_squares_reports_solution_allocation_failure() {
        // Both input matrices are valid because zero rows require no storage,
        // even though the coefficient matrix describes usize::MAX variables.
        // The requested solution is usize::MAX by one and must therefore fail
        // through `Result` instead of panicking in the trivial-system branch.
        let coefficients = Matrix::from_vec(Vec::new(), 0, usize::MAX)
            .expect("zero rows make the coefficient shape storage-free");
        let right_hand_side = Matrix::from_vec(Vec::new(), 0, 1)
            .expect("a zero-row right-hand side requires no storage");

        let error = coefficients
            .solve_least_squares(&right_hand_side)
            .expect_err("the impossible solution allocation must be reported");
        assert_eq!(
            error,
            MatrixError::AllocationFailed {
                rows: usize::MAX,
                cols: 1,
                elements: usize::MAX,
            }
        );
    }

    /// Compare random full-row-rank wide solves with a construction whose
    /// Moore-Penrose solution is known independently of the SVD implementation.
    #[test]
    fn test_solve_least_squares_random_wide_minimum_norm_oracle() {
        let mut rng = TestRng::new(0x1234_5678_9abc_def0);

        for _case in 0..32 {
            let mut coefficients = random_matrix(3, 6, &mut rng);
            for diagonal in 0..3 {
                // The leading 3x3 block is strictly diagonally dominant because
                // every random entry lies in [-1, 1]. It is therefore invertible
                // and proves that all three rows are linearly independent.
                coefficients[(diagonal, diagonal)] += 4.0;
            }

            // Choose x=A^T*z manually, then b=A*x. This x lies in A's row space
            // and satisfies Ax=b. Every other solution differs by a vector in
            // null(A), which is orthogonal to x, so this construction makes x
            // the unique minimum-Euclidean-norm solution without using a second
            // pseudoinverse implementation as an oracle.
            let row_coordinates = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
            let mut expected_variables = [0.0; 6];
            for variable in 0..6 {
                for equation in 0..3 {
                    expected_variables[variable] +=
                        coefficients[(equation, variable)] * row_coordinates[equation];
                }
            }
            let mut expected_rhs = [0.0; 3];
            for equation in 0..3 {
                for variable in 0..6 {
                    expected_rhs[equation] +=
                        coefficients[(equation, variable)] * expected_variables[variable];
                }
            }
            let right_hand_side = Matrix::from_vec(expected_rhs.to_vec(), 3, 1)
                .expect("the oracle always contains three right-hand-side values");

            let result = solve_default(&coefficients, &right_hand_side);

            assert_eq!(result.info.rank, 3);
            assert!(result.info.cond_est.is_finite());
            for (variable, expected) in expected_variables.iter().enumerate() {
                let error = (result.solution[(variable, 0)] - expected).abs();
                assert!(
                    error < 1e-8,
                    "minimum-norm mismatch at variable {variable}: {error}"
                );
            }
        }
    }

    #[test]
    fn test_solve_least_squares_detects_ill_conditioning() {
        let mut a = Matrix::identity(3);
        for i in 0..3 {
            a[(i, 2)] *= 1e-8;
        }
        let b = Matrix::from_vec(vec![0.0, 0.0, 0.0], 3, 1).unwrap();

        let result = solve_default(&a, &b);
        assert_eq!(result.info.rank, 3);
        assert!(
            result.info.cond_est > 1e6,
            "cond_est was {}",
            result.info.cond_est
        );
    }

    /// Verify that condition diagnostics account for amplification caused by
    /// off-diagonal entries, not only disparities between diagonal elements.
    #[test]
    fn test_solve_least_squares_condition_detects_off_diagonal_growth() {
        // Both diagonal entries are one, so the former max-diagonal/min-diagonal
        // heuristic reported condition one. The singular-value ratio of this
        // matrix is approximately 1e12 while both directions remain above the
        // precision-derived rank threshold.
        let a = Matrix::from_vec(vec![1.0, 1e6, 0.0, 1.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![0.0, 0.0], 2, 1).unwrap();
        let result = solve_default(&a, &b);

        assert_eq!(result.info.rank, 2);
        assert!(
            result.info.cond_est > 1e11,
            "off-diagonal condition estimate was {}",
            result.info.cond_est
        );
    }

    #[test]
    fn test_solve_least_squares_tall_full_rank() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();

        let result = solve_default(&a, &b);
        assert_eq!(result.info.rank, 2);
        assert!(result.info.cond_est.is_finite());
        assert!((result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_wide_full_rank() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        let result = solve_default(&a, &b);
        assert_eq!(result.info.rank, 2);
        assert!(result.info.cond_est.is_finite());
        assert!((result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 2.0).abs() < 1e-12);
        assert!((result.solution[(2, 0)] - 0.0).abs() < 1e-12);
    }

    /// Verify that a square rank-deficient system returns its unique
    /// minimum-norm pseudoinverse solution through the canonical SVD.
    #[test]
    fn test_solve_least_squares_square_rank_deficient() {
        let a = Matrix::from_vec(vec![1.0, 1.0, 2.0, 2.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        let result = solve_default(&a, &b);

        assert!((result.solution[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 0.5).abs() < 1e-12);
        assert_eq!(result.info.rank, 1);
        assert!((result.info.cond_est - 1.0).abs() < 1e-12);
    }

    /// Ensure SVD rank classification and its pseudoinverse solution remain
    /// invariant across very small and very large finite unit scales.
    #[test]
    fn test_solve_least_squares_rank_deficient_scale_invariance() {
        for scale in [1e-200_f64, 1.0, 1e200] {
            let a = Matrix::from_vec(vec![scale, scale, 2.0 * scale, 2.0 * scale], 2, 2).unwrap();
            let b = Matrix::from_vec(vec![scale, 2.0 * scale], 2, 1).unwrap();

            let result = solve_default(&a, &b);

            // Uniformly scaling both operands leaves the Moore-Penrose answer
            // unchanged, including at magnitudes where naïve squared norms
            // would underflow or overflow.
            assert!((result.solution[(0, 0)] - 0.5).abs() < 1e-12);
            assert!((result.solution[(1, 0)] - 0.5).abs() < 1e-12);
            assert_eq!(result.info.rank, 1);
            assert!((result.info.cond_est - 1.0).abs() < 1e-12);
        }
    }

    /// Return the finite minimum-norm answer for a rank-deficient system at the
    /// largest practical coefficient scale.
    #[test]
    fn test_solve_least_squares_rank_deficient_extreme_scale() {
        // Every row constrains x0+x1=1. A factorization that applies a
        // Householder reflector to these finite entries can overflow while
        // forming its transformed right-hand side, even though the canonical
        // pseudoinverse answer is the ordinary finite vector [0.5, 0.5].
        let coefficients = Matrix::from_vec(vec![-1e308, -1e308, -1e308, -1e308], 2, 2).unwrap();
        let right_hand_side = Matrix::from_vec(vec![-1e308, -1e308], 2, 1).unwrap();

        let result = solve_default(&coefficients, &right_hand_side);

        assert!((result.solution[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 0.5).abs() < 1e-12);
        assert_eq!(result.info.rank, 1);
        assert!((result.info.cond_est - 1.0).abs() < 1e-12);
    }

    /// Verify that SVD minimizes residual norm for an inconsistent tall system
    /// and then selects the minimum-norm point among its minimizers.
    #[test]
    fn test_solve_least_squares_tall_rank_deficient_inconsistent() {
        let a = Matrix::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();

        let result = solve_default(&a, &b);

        // The best fitted shared row value is mean(b)=2. Splitting that value
        // equally across duplicate columns minimizes the variable norm.
        assert!((result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(result.info.rank, 1);
        let residual = &(&a * &result.solution) - &b;
        assert!((residual.norm() - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    /// Verify the transpose-oriented Jacobi SVD for a wide matrix whose rows
    /// are themselves linearly dependent.
    #[test]
    fn test_solve_least_squares_wide_rank_deficient_minimum_norm() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 1.0, 2.0, 0.0, 2.0], 2, 3).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        let result = solve_default(&a, &b);

        // The constrained sum x0+x2=1 is divided equally; the unconstrained x1
        // component is zero in the Moore-Penrose solution.
        assert!((result.solution[(0, 0)] - 0.5).abs() < 1e-12);
        assert!(result.solution[(1, 0)].abs() < 1e-12);
        assert!((result.solution[(2, 0)] - 0.5).abs() < 1e-12);
        assert_eq!(result.info.rank, 1);
        let reconstructed = &a * &result.solution;
        assert!((&reconstructed - &b).norm() < 1e-12);
    }

    /// Verify that fallible construction reports both ordinary length mismatch
    /// and dimension-product overflow through structured matrix errors.
    #[test]
    fn test_from_vec_reports_structured_shape_errors() {
        let length_error = Matrix::from_vec(vec![1.0, 2.0], 2, 2).unwrap_err();
        assert_eq!(
            length_error,
            MatrixError::InvalidDataLength {
                rows: 2,
                cols: 2,
                expected: 4,
                actual: 2,
            }
        );

        let overflow_error = Matrix::from_vec(Vec::new(), usize::MAX, 2).unwrap_err();
        assert_eq!(
            overflow_error,
            MatrixError::ElementCountOverflow {
                rows: usize::MAX,
                cols: 2,
            }
        );
    }

    /// Reject invalid row and column coordinates independently before their
    /// values can alias or overflow a row-major storage offset.
    #[test]
    fn test_coordinate_indexing_checks_both_axes_before_flattening() {
        let mut matrix = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2)
            .expect("the fixture has exactly four elements");

        // `(0, 2)` previously flattened to offset two and silently read `(1,
        // 0)`. The very large row also wrapped to offset zero in optimized
        // builds even though debug builds happened to panic on multiplication.
        let invalid_coordinates = [(0, 2), (2, 0), (usize::MAX / 2 + 1, 0)];
        for coordinate in invalid_coordinates {
            let result = std::panic::catch_unwind(|| matrix[coordinate]);
            assert!(
                result.is_err(),
                "immutable indexing accepted invalid coordinate {coordinate:?}"
            );
        }

        // Mutable indexing must enforce the identical invariant and leave the
        // valid cell that used to be aliased completely unchanged.
        let mutation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matrix[(0, 2)] = 99.0;
        }));
        assert!(mutation.is_err());
        assert_eq!(matrix[(1, 0)], 3.0);
    }
}
