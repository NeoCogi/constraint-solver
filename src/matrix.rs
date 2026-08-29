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

//! Checked dense matrix operations, LU factorization, full-rank Householder QR,
//! and rank-deficient Jacobi-SVD least-squares solving.

use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 16_384;

/// Relative factor threshold used to classify numerical rank after QR or SVD.
///
/// The threshold is intentionally relative to the largest QR diagonal or
/// singular value. An absolute floor would make rank depend on the units or
/// scalar normalization chosen by the caller.
const DEFAULT_NUMERICAL_RANK_RELATIVE_TOLERANCE: f64 = 1e-12;

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

/// Convert the relative numerical-rank threshold into the scale of a particular
/// QR diagonal or singular-value spectrum.
///
/// A zero factor produces a zero tolerance and therefore rank zero because all
/// comparisons use strict inequality. Non-finite factors are rejected by the
/// factorization before this helper's result is consumed.
fn numerical_rank_tolerance(max_factor_value: f64, relative_tolerance: f64) -> f64 {
    // Multiplication, rather than `max_factor_value.max(1.0)`, preserves rank
    // under uniform scaling of the input matrix.
    relative_tolerance * max_factor_value
}

/// Structured failure returned by all fallible matrix construction and linear
/// algebra operations.
///
/// Variants retain machine-readable dimensions and operation names so callers
/// no longer need to parse unrelated ad-hoc strings from LU and QR routines.
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
    /// An algorithm that requires a square coefficient matrix received a
    /// rectangular matrix.
    NonSquare {
        /// Stable name of the rejecting operation.
        operation: &'static str,
        /// Actual matrix shape as `(rows, columns)`.
        matrix: (usize, usize),
    },
    /// A numerical tolerance was negative, NaN, or infinite.
    InvalidTolerance {
        /// Stable name of the operation whose tolerance was invalid.
        operation: &'static str,
    },
    /// A coefficient or right-hand-side input contained NaN or infinity before
    /// factorization began.
    NonFiniteInput {
        /// Stable name of the operation that validated the operand.
        operation: &'static str,
        /// Operand in which the invalid value was observed.
        operand: MatrixOperand,
    },
    /// A factorization encountered a pivot too small relative to its complete
    /// coefficient-matrix scale to support a stable solve.
    Singular {
        /// Stable name of the factorization or solve that detected singularity.
        operation: &'static str,
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

/// Factorization method that produced a checked least-squares solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeastSquaresMethod {
    /// Householder QR solved a full-rank rectangular system directly.
    HouseholderQr,
    /// One-sided Jacobi SVD produced a pseudoinverse solution after QR detected
    /// numerical rank loss.
    JacobiSvd,
    /// No factorization was required because one matrix dimension was zero.
    Empty,
}

/// Numerical diagnostics produced alongside a rectangular least-squares
/// solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresInfo {
    /// Estimated numerical rank using a threshold relative to the largest QR
    /// diagonal or singular value, depending on [`LeastSquaresInfo::method`].
    pub rank: usize,
    /// Condition estimate of the retained independent subspace. QR reports an
    /// estimated one-norm condition number; SVD reports the ratio of the largest
    /// to smallest retained singular value. Infinity means no non-zero singular
    /// subspace or unavailable finite arithmetic.
    pub cond_est: f64,
    /// Numerical method that produced the returned solution and diagnostics.
    pub method: LeastSquaresMethod,
}

/// Checked least-squares solution paired with the factorization diagnostics
/// that justify its numerical interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresSolution {
    /// Full-column, least-squares, or minimum-norm solution vector.
    pub solution: Matrix,
    /// Rank, retained-subspace condition estimate, and factorization method.
    pub info: LeastSquaresInfo,
}

/// Numerical policy shared by QR rank detection and the SVD pseudoinverse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresOptions {
    /// Non-negative finite threshold relative to the largest QR diagonal or
    /// singular value. Directions at or below this fraction are treated as
    /// numerically rank deficient and omitted from the pseudoinverse.
    pub relative_rank_tolerance: f64,
}

impl Default for LeastSquaresOptions {
    /// Return the crate's balanced default rank policy.
    fn default() -> Self {
        // Keep the historical threshold explicit in the public option value so
        // callers can inspect and replace it without relying on a hidden const.
        Self {
            relative_rank_tolerance: DEFAULT_NUMERICAL_RANK_RELATIVE_TOLERANCE,
        }
    }
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
            MatrixError::NonSquare { operation, matrix } => write!(
                f,
                "Matrix must be square for {operation}: received {}x{}",
                matrix.0, matrix.1
            ),
            MatrixError::InvalidTolerance { operation } => {
                write!(f, "Invalid numerical tolerance for {operation}")
            }
            MatrixError::NonFiniteInput { operation, operand } => {
                write!(f, "Non-finite {operand} supplied to {operation}")
            }
            MatrixError::Singular { operation } => {
                write!(f, "Matrix is singular during {operation}")
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

/// Compact parameters for one in-place Householder reflector.
///
/// Entries below the active diagonal are stored directly in the factor matrix
/// with an implicit leading one. These two scalars complete the representation
/// needed by both tall and transpose-oriented wide QR paths.
#[derive(Debug, Clone, Copy)]
struct HouseholderReflector {
    /// Signed diagonal value written to the upper-triangular factor after the
    /// reflector has been applied to all remaining operands.
    diagonal: f64,
    /// Multiplier `2 / (v^T v)` for the implicit-one reflector vector.
    tau: f64,
}

/// Internal disposition of a completed Householder QR attempt.
///
/// Numerical rank loss is an instruction for the public least-squares
/// dispatcher to retry with the Moore-Penrose SVD path, not a terminal matrix
/// error exposed to callers.
enum QrLeastSquaresOutcome {
    /// QR produced a stable triangular solution for every independent column or
    /// row required by the system orientation.
    FullRank {
        /// Full-rank least-squares or minimum-norm solution.
        solution: Matrix,
        /// Rank, conditioning, and factorization diagnostics for the solution.
        diagnostics: LeastSquaresInfo,
    },
    /// QR found a diagonal too small for stable triangular substitution, so the
    /// dispatcher must compute a pseudoinverse instead.
    RankDeficient,
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

    /// Build one scale-safe Householder reflector in a matrix column.
    ///
    /// The active column begins at `diagonal_index` and extends to the matrix's
    /// final row. Entries below the diagonal are replaced with the tail of a
    /// reflector vector whose leading component is implicitly one. `None`
    /// represents an already-zero active column.
    fn prepare_householder_column(
        factor: &mut Matrix,
        diagonal_index: usize,
        column: usize,
        operation: &'static str,
    ) -> Result<Option<HouseholderReflector>, MatrixError> {
        // Accumulate the norm with `hypot` so finite columns do not overflow or
        // underflow merely because their squared entries are outside f64 range.
        let mut column_norm = 0.0_f64;
        for row in diagonal_index..factor.rows {
            column_norm = column_norm.hypot(factor[(row, column)]);
        }
        if !column_norm.is_finite() {
            return Err(MatrixError::NonFiniteFactor { operation });
        }
        if column_norm == 0.0 {
            return Ok(None);
        }

        let leading_value = factor[(diagonal_index, column)];
        let sign = if leading_value >= 0.0 { 1.0 } else { -1.0 };
        let diagonal = -sign * column_norm;

        // The conventional denominator `leading_value - diagonal` can overflow
        // when two same-scale finite magnitudes are added. Divide the complete
        // expression by `column_norm` first: both numerator terms are bounded by
        // one, while the normalized denominator has magnitude in [1, 2].
        let normalized_leading_value = leading_value / column_norm;
        let normalized_denominator = normalized_leading_value + sign;
        if normalized_denominator == 0.0 || !normalized_denominator.is_finite() {
            return Err(MatrixError::NonFiniteFactor { operation });
        }

        let mut vector_norm_squared = 1.0;
        for row in (diagonal_index + 1)..factor.rows {
            let normalized_entry = factor[(row, column)] / column_norm;
            let reflector_entry = normalized_entry / normalized_denominator;
            if !reflector_entry.is_finite() {
                return Err(MatrixError::NonFiniteFactor { operation });
            }
            factor[(row, column)] = reflector_entry;
            vector_norm_squared += reflector_entry * reflector_entry;
        }
        if !vector_norm_squared.is_finite() {
            return Err(MatrixError::NonFiniteFactor { operation });
        }

        let tau = 2.0 / vector_norm_squared;
        if !tau.is_finite() {
            return Err(MatrixError::NonFiniteFactor { operation });
        }

        Ok(Some(HouseholderReflector { diagonal, tau }))
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

    /// Compute a partial-pivoting LU factorization using the default relative
    /// singularity threshold.
    pub fn lu_decomposition(&self) -> Result<(Matrix, Matrix, Vec<usize>), MatrixError> {
        self.lu_decomposition_with_tolerance(1e-12)
    }

    /// Compute a partial-pivoting LU factorization with a caller-selected
    /// non-negative finite relative pivot threshold.
    pub fn lu_decomposition_with_tolerance(
        &self,
        singular_relative_epsilon: f64,
    ) -> Result<(Matrix, Matrix, Vec<usize>), MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NonSquare {
                operation: "lu_decomposition",
                matrix: (self.rows, self.cols),
            });
        }
        if !singular_relative_epsilon.is_finite() || singular_relative_epsilon < 0.0 {
            return Err(MatrixError::InvalidTolerance {
                operation: "lu_decomposition",
            });
        }
        if !self.all_finite() {
            return Err(MatrixError::NonFiniteInput {
                operation: "lu_decomposition",
                operand: MatrixOperand::CoefficientMatrix,
            });
        }

        let n = self.rows;
        let mut l = Matrix::identity(n);
        let mut u = self.clone();
        let mut pivot = (0..n).collect::<Vec<_>>();

        // Use one factorization-wide reference scale for every pivot. Comparing
        // a pivot only with its own trailing row makes the last non-zero pivot
        // pass for every tolerance below one, even when it is negligible next
        // to the rest of the coefficient matrix.
        let matrix_scale = self
            .data
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let singular_threshold = singular_relative_epsilon * matrix_scale;

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

            if max_val <= singular_threshold {
                return Err(MatrixError::Singular {
                    operation: "lu_decomposition",
                });
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
                if !l[(i, k)].is_finite() {
                    return Err(MatrixError::NonFiniteFactor {
                        operation: "lu_decomposition",
                    });
                }
                for j in k..n {
                    u[(i, j)] -= l[(i, k)] * u[(k, j)];
                    if !u[(i, j)].is_finite() {
                        return Err(MatrixError::NonFiniteFactor {
                            operation: "lu_decomposition",
                        });
                    }
                }
            }
        }

        Ok((l, u, pivot))
    }

    /// Solve a square system with partial-pivoting LU and the default relative
    /// singularity threshold.
    pub fn solve_lu(&self, b: &Matrix) -> Result<Matrix, MatrixError> {
        self.solve_lu_with_tolerance(b, 1e-12)
    }

    /// Solve a square system against one column-vector right-hand side using a
    /// caller-selected non-negative finite relative pivot threshold.
    pub fn solve_lu_with_tolerance(
        &self,
        b: &Matrix,
        singular_relative_epsilon: f64,
    ) -> Result<Matrix, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NonSquare {
                operation: "solve_lu",
                matrix: (self.rows, self.cols),
            });
        }
        if b.rows != self.rows || b.cols != 1 {
            return Err(MatrixError::DimensionMismatch {
                operation: "solve_lu",
                left: (self.rows, 1),
                right: (b.rows, b.cols),
            });
        }
        if !b.all_finite() {
            return Err(MatrixError::NonFiniteInput {
                operation: "solve_lu",
                operand: MatrixOperand::RightHandSide,
            });
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
            if !x[(i, 0)].is_finite() {
                return Err(MatrixError::NonFiniteResult {
                    operation: "solve_lu",
                });
            }
        }

        Ok(x)
    }

    /// Solve a possibly rectangular least-squares problem with explicit rank
    /// and parallelism policy.
    ///
    /// Full-rank systems use Householder QR. If QR detects numerical rank loss,
    /// a one-sided Jacobi SVD computes the Moore-Penrose minimum-norm solution.
    /// The named return type always preserves factorization diagnostics so
    /// callers cannot silently discard rank and conditioning by choosing a
    /// less-informative overload.
    pub fn solve_least_squares(
        &self,
        b: &Matrix,
        options: LeastSquaresOptions,
        parallel: bool,
    ) -> Result<LeastSquaresSolution, MatrixError> {
        // Rank tolerance participates in every QR/SVD classification, so reject
        // invalid values once before either factorization can observe them.
        if !options.relative_rank_tolerance.is_finite() || options.relative_rank_tolerance < 0.0 {
            return Err(MatrixError::InvalidTolerance {
                operation: "solve_least_squares",
            });
        }

        // Validate the vector shape before either factorization reads it. The
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

        // Reject invalid floating-point inputs before QR or SVD can turn them
        // into misleading rank or convergence diagnostics.
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

        // Trivial cases have a unique zero-length or minimum-norm zero solution
        // and require no numerical factorization.
        if n == 0 || m == 0 {
            return Ok(LeastSquaresSolution {
                solution: Matrix::new(n, 1),
                info: LeastSquaresInfo {
                    rank: 0,
                    cond_est: f64::INFINITY,
                    method: LeastSquaresMethod::Empty,
                },
            });
        }

        let qr_result = if m >= n {
            Self::solve_least_squares_qr_tall_with_info(
                self,
                b,
                options.relative_rank_tolerance,
                parallel,
            )
        } else {
            Self::solve_least_squares_qr_wide_with_info(
                self,
                b,
                options.relative_rank_tolerance,
                parallel,
            )
        };

        // QR remains the efficient full-rank path. Numerical rank loss is not a
        // failure of the least-squares problem itself, so route only that case to
        // the SVD pseudoinverse while preserving unrelated validation and
        // arithmetic errors unchanged.
        let result = match qr_result {
            Ok(QrLeastSquaresOutcome::FullRank {
                solution,
                diagnostics,
            }) => (solution, diagnostics),
            Ok(QrLeastSquaresOutcome::RankDeficient) => {
                Self::solve_least_squares_svd_with_info(self, b, options.relative_rank_tolerance)?
            }
            Err(error) => return Err(error),
        };

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

        // Decomposing A^T for a wide system keeps the Jacobi kernel compact and
        // avoids manufacturing zero singular columns merely to make A tall.
        let factor_input = if tall_orientation {
            a.clone()
        } else {
            a.transpose()
        };
        let decomposition = Self::one_sided_jacobi_svd_columns(&factor_input)?;

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
        let mut solution = Matrix::new(cols, 1);

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
                    "solve_least_squares_svd",
                )?;
                let coefficient = projection / singular_value;
                if !coefficient.is_finite() {
                    return Err(MatrixError::NonFiniteResult {
                        operation: "solve_least_squares_svd",
                    });
                }

                // V maps each retained singular-coordinate coefficient back to
                // the original variable space.
                for row in 0..cols {
                    let contribution = decomposition.right_vectors[(row, component)] * coefficient;
                    solution[(row, 0)] += contribution;
                    if !solution[(row, 0)].is_finite() {
                        return Err(MatrixError::NonFiniteResult {
                            operation: "solve_least_squares_svd",
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
                    "solve_least_squares_svd",
                )?;
                let coefficient = projection / singular_value;
                if !coefficient.is_finite() {
                    return Err(MatrixError::NonFiniteResult {
                        operation: "solve_least_squares_svd",
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
                            operation: "solve_least_squares_svd",
                        });
                    }
                }
            }
        }

        Ok((
            solution,
            LeastSquaresInfo {
                rank,
                cond_est,
                method: LeastSquaresMethod::JacobiSvd,
            },
        ))
    }

    /// Orthogonalize the columns of a tall-or-square matrix with cyclic Jacobi
    /// rotations while accumulating its right singular vectors.
    ///
    /// Pairwise Gram entries are evaluated after scaling both columns by their
    /// largest magnitude. That scaling prevents the convergence test and
    /// rotation angle from overflowing for otherwise finite input matrices.
    fn one_sided_jacobi_svd_columns(source: &Matrix) -> Result<JacobiSvd, MatrixError> {
        debug_assert!(source.rows >= source.cols);

        let mut orthogonal_columns = source.clone();
        let mut right_vectors = Matrix::identity(source.cols);
        let column_count = source.cols;

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
                    for row in 0..source.rows {
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
                    for row in 0..source.rows {
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
                    for row in 0..source.rows {
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
        let mut singular_values = Vec::with_capacity(column_count);
        for column in 0..column_count {
            let mut singular_value = 0.0_f64;
            for row in 0..source.rows {
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

    /// Estimate the one-norm condition number of an upper-triangular leading
    /// block without explicitly forming its inverse.
    ///
    /// The previous diagonal-ratio heuristic missed ill-conditioning caused by
    /// large off-diagonal entries. This routine uses a bounded Hager-style
    /// iteration to estimate `||R^-1||_1`, then multiplies it by the exact
    /// `||R||_1`. Each iteration requires triangular solves rather than an
    /// expensive and less stable explicit inverse.
    fn estimate_upper_triangular_condition(r: &Matrix, size: usize) -> f64 {
        // The empty factor has no meaningful inverse condition. This case is
        // represented as infinity consistently with the public zero-column QR
        // result.
        if size == 0 {
            return f64::INFINITY;
        }

        // Compute the exact matrix one-norm: the largest absolute column sum of
        // the independent upper-triangular block.
        let mut r_one_norm: f64 = 0.0;
        for column in 0..size {
            let mut column_sum = 0.0;
            for row in 0..=column {
                column_sum += r[(row, column)].abs();
            }
            r_one_norm = r_one_norm.max(column_sum);
        }

        // A zero or non-finite factor cannot have a useful finite condition
        // estimate. The subsequent solve will provide the detailed failure.
        if r_one_norm == 0.0 || !r_one_norm.is_finite() {
            return f64::INFINITY;
        }

        // Hager's estimator starts with a normalized positive vector. Limiting
        // the iteration count keeps diagnostics O(n^2) with a small constant,
        // preserving QR's overall complexity for tall systems.
        const MAX_CONDITION_ESTIMATE_ITERATIONS: usize = 8;
        let mut probe = vec![1.0 / size as f64; size];
        let mut inverse_one_norm_estimate: f64 = 0.0;
        let mut previous_index = None;

        for _ in 0..MAX_CONDITION_ESTIMATE_ITERATIONS {
            // Applying R^-1 to the probe exposes a direction amplified by the
            // inverse. Failure means the factor is singular or non-finite.
            let Some(inverse_times_probe) =
                Self::solve_upper_triangular_vector(r, size, &probe, false)
            else {
                return f64::INFINITY;
            };

            let candidate_estimate: f64 = inverse_times_probe.iter().map(|v| v.abs()).sum();
            if !candidate_estimate.is_finite() {
                return f64::INFINITY;
            }
            inverse_one_norm_estimate = inverse_one_norm_estimate.max(candidate_estimate);

            // The sign vector is a subgradient of the one-norm. Applying
            // R^-T identifies the coordinate direction that can most improve
            // the current inverse-norm estimate.
            let sign_vector: Vec<f64> = inverse_times_probe
                .iter()
                .map(|&value| if value < 0.0 { -1.0 } else { 1.0 })
                .collect();
            let Some(transpose_inverse_times_sign) =
                Self::solve_upper_triangular_vector(r, size, &sign_vector, true)
            else {
                return f64::INFINITY;
            };

            let next_index = transpose_inverse_times_sign
                .iter()
                .enumerate()
                .max_by(|(_, lhs), (_, rhs)| lhs.abs().total_cmp(&rhs.abs()))
                .map(|(index, _)| index)
                .unwrap_or(0);

            // Repeating the same maximizing coordinate means the estimator has
            // reached a fixed point. Otherwise, probe that basis direction on
            // the next iteration.
            if previous_index == Some(next_index) {
                break;
            }
            probe.fill(0.0);
            probe[next_index] = 1.0;
            previous_index = Some(next_index);
        }

        r_one_norm * inverse_one_norm_estimate
    }

    /// Solve either `R*x=rhs` or `R^T*x=rhs` for the leading upper-triangular
    /// block used by the condition estimator.
    ///
    /// Returning `None` keeps non-finite or singular diagnostic arithmetic out
    /// of the main QR solution while allowing the caller to report an infinite
    /// condition estimate.
    fn solve_upper_triangular_vector(
        r: &Matrix,
        size: usize,
        rhs: &[f64],
        transpose: bool,
    ) -> Option<Vec<f64>> {
        // A mismatched internal vector would indicate a programming error. It
        // is treated as an unavailable estimate rather than indexing unsafely.
        if rhs.len() != size {
            return None;
        }

        let mut solution = vec![0.0; size];
        if transpose {
            // R^T is lower triangular, so solve from the first row downward.
            for row in 0..size {
                let mut value = rhs[row];
                for column in 0..row {
                    value -= r[(column, row)] * solution[column];
                }
                let diagonal = r[(row, row)];
                if diagonal == 0.0 || !diagonal.is_finite() {
                    return None;
                }
                solution[row] = value / diagonal;
                if !solution[row].is_finite() {
                    return None;
                }
            }
        } else {
            // R is upper triangular, so ordinary back-substitution proceeds
            // from the final row upward.
            for row in (0..size).rev() {
                let mut value = rhs[row];
                for column in (row + 1)..size {
                    value -= r[(row, column)] * solution[column];
                }
                let diagonal = r[(row, row)];
                if diagonal == 0.0 || !diagonal.is_finite() {
                    return None;
                }
                solution[row] = value / diagonal;
                if !solution[row].is_finite() {
                    return None;
                }
            }
        }

        Some(solution)
    }

    /// Solve a tall or square full-column-rank system by thin Householder QR.
    ///
    /// Rank loss is returned to the public dispatcher as an internal structured
    /// signal so it can retry with SVD. Reflectors are stored below R's diagonal
    /// in place, minimizing allocation while preserving the upper factor.
    fn solve_least_squares_qr_tall_with_info(
        a: &Matrix,
        b: &Matrix,
        relative_rank_tolerance: f64,
        parallel: bool,
    ) -> Result<QrLeastSquaresOutcome, MatrixError> {
        let m = a.rows;
        let n = a.cols;

        let mut r = a.clone();
        let mut qt_b = b.clone();

        // Householder QR factorization (thin): apply reflectors to `r` and `qt_b`.
        for k in 0..n {
            let Some(reflector) =
                Self::prepare_householder_column(&mut r, k, k, "solve_least_squares")?
            else {
                continue;
            };
            let tau = reflector.tau;

            // Apply reflector to remaining columns.
            let cols_left = n.saturating_sub(k + 1);
            if cols_left > 0 {
                let reflector_work = (m - (k + 1)).saturating_mul(cols_left);
                let use_parallel = parallel && should_parallelize_work(reflector_work);
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

            let use_parallel = parallel && should_parallelize_work(m - (k + 1));
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

            r[(k, k)] = reflector.diagonal;
        }

        // Householder updates can overflow even when every input is finite.
        // Validate the complete factor and transformed right-hand side before
        // rank classification so overflow is not mislabeled as rank loss.
        if !r.all_finite() {
            return Err(MatrixError::NonFiniteFactor {
                operation: "solve_least_squares",
            });
        }
        if !qt_b.all_finite() {
            return Err(MatrixError::NonFiniteResult {
                operation: "solve_least_squares",
            });
        }

        let mut max_diag: f64 = 0.0;
        for i in 0..n {
            max_diag = max_diag.max(r[(i, i)].abs());
        }
        // Base numerical rank only on the scale of `R`; an absolute floor here
        // would incorrectly classify small, full-rank systems as rank zero.
        let tol = numerical_rank_tolerance(max_diag, relative_rank_tolerance);
        let mut rank = 0;
        for i in 0..n {
            let diag = r[(i, i)].abs();
            if diag > tol {
                rank += 1;
            }
        }
        if rank < n {
            // Rank loss is recoverable through the SVD dispatcher and therefore
            // must not escape through the public matrix error type.
            return Ok(QrLeastSquaresOutcome::RankDeficient);
        }
        let cond_est = Self::estimate_upper_triangular_condition(&r, n);

        // Back-substitute R x = Q^T b, using the top n entries.
        let mut x = Matrix::new(n, 1);
        for i in (0..n).rev() {
            let mut sum = qt_b[(i, 0)];
            for j in (i + 1)..n {
                sum -= r[(i, j)] * x[(j, 0)];
            }

            let diag = r[(i, i)];
            if !diag.is_finite() {
                return Err(MatrixError::NonFiniteFactor {
                    operation: "solve_least_squares",
                });
            }

            let mut row_norm: f64 = 0.0;
            for j in i..n {
                row_norm = row_norm.max(r[(i, j)].abs());
            }

            if diag.abs() <= relative_rank_tolerance * row_norm {
                // A diagonal can pass the global rank threshold yet remain too
                // small relative to its own row for stable substitution.
                return Ok(QrLeastSquaresOutcome::RankDeficient);
            }

            x[(i, 0)] = sum / diag;
        }

        Ok(QrLeastSquaresOutcome::FullRank {
            solution: x,
            diagnostics: LeastSquaresInfo {
                rank,
                cond_est,
                method: LeastSquaresMethod::HouseholderQr,
            },
        })
    }

    /// Solve a wide full-row-rank system by Householder QR of its transpose.
    ///
    /// Factoring `A^T=Q*R` and solving `R^T*y=b`, followed by `x=Q*y`, produces
    /// the minimum-norm solution without forming a normal equation. Numerical
    /// rank loss is delegated to the public dispatcher's SVD fallback.
    fn solve_least_squares_qr_wide_with_info(
        a: &Matrix,
        b: &Matrix,
        relative_rank_tolerance: f64,
        parallel: bool,
    ) -> Result<QrLeastSquaresOutcome, MatrixError> {
        // Use QR on A^T to compute the minimum-norm least squares solution.
        //
        // Let A be m x n with m < n. Compute QR of A^T (n x m):
        // A^T = Q R, where Q is n x n orthogonal and R has top-left m x m upper-triangular.
        // The minimum-norm solution is x = Q [y; 0], where y solves R^T y = b.
        let m = a.rows;
        let n = a.cols;

        // Transposition is part of the wide QR workload and must respect the
        // same explicit parallel-mode decision as the reflector updates.
        let mut r = a.transpose_with_parallel(parallel); // n x m, tall
        let mut taus = Vec::with_capacity(m);

        // Householder QR on r (n x m), producing R in the top m x m block.
        for k in 0..m {
            let Some(reflector) =
                Self::prepare_householder_column(&mut r, k, k, "solve_least_squares")?
            else {
                taus.push(0.0);
                continue;
            };
            let tau = reflector.tau;
            taus.push(tau);

            let cols_left = m.saturating_sub(k + 1);
            if cols_left > 0 {
                let reflector_work = (n - (k + 1)).saturating_mul(cols_left);
                let use_parallel = parallel && should_parallelize_work(reflector_work);
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

            r[(k, k)] = reflector.diagonal;
        }

        // Do not let overflow in the transposed Householder factor masquerade as
        // a rank-deficient matrix during the diagonal threshold checks below.
        if !r.all_finite() {
            return Err(MatrixError::NonFiniteFactor {
                operation: "solve_least_squares",
            });
        }

        let mut max_diag: f64 = 0.0;
        for i in 0..m {
            max_diag = max_diag.max(r[(i, i)].abs());
        }
        // Use the same scale-relative criterion as the tall QR path so wide and
        // tall systems report rank consistently under unit changes.
        let tol = numerical_rank_tolerance(max_diag, relative_rank_tolerance);
        let mut rank = 0;
        for i in 0..m {
            let diag = r[(i, i)].abs();
            if diag > tol {
                rank += 1;
            }
        }
        if rank < m {
            // Let SVD determine the retained singular subspace and unique
            // minimum-norm result for this recoverable rank-loss case.
            return Ok(QrLeastSquaresOutcome::RankDeficient);
        }
        let cond_est = Self::estimate_upper_triangular_condition(&r, m);

        // Solve R^T y = b (R is m x m upper triangular, so R^T is lower triangular).
        let mut y = vec![0.0; m];
        for i in 0..m {
            let mut sum = b[(i, 0)];
            for j in 0..i {
                sum -= r[(j, i)] * y[j];
            }

            let diag = r[(i, i)];
            if !diag.is_finite() {
                return Err(MatrixError::NonFiniteFactor {
                    operation: "solve_least_squares",
                });
            }

            let mut col_norm: f64 = 0.0;
            for j in 0..=i {
                col_norm = col_norm.max(r[(j, i)].abs());
            }

            if diag.abs() <= relative_rank_tolerance * col_norm {
                // Local triangular scaling can reveal instability not captured
                // by the factor-wide diagonal maximum used for rank estimation.
                return Ok(QrLeastSquaresOutcome::RankDeficient);
            }

            y[i] = sum / diag;
            if !y[i].is_finite() {
                return Err(MatrixError::NonFiniteResult {
                    operation: "solve_least_squares",
                });
            }
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

            let use_parallel = parallel && should_parallelize_work(n - (k + 1));
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

        Matrix::from_vec(w, n, 1).map(|solution| QrLeastSquaresOutcome::FullRank {
            solution,
            diagnostics: LeastSquaresInfo {
                rank,
                cond_est,
                method: LeastSquaresMethod::HouseholderQr,
            },
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

    /// Run the canonical least-squares API with default rank policy for tests
    /// that vary only serial versus parallel QR execution.
    fn solve_default(coefficients: &Matrix, rhs: &Matrix, parallel: bool) -> LeastSquaresSolution {
        // Unwrap here so individual numerical tests can focus on their expected
        // solution and diagnostic fields.
        coefficients
            .solve_least_squares(rhs, LeastSquaresOptions::default(), parallel)
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

    /// Ensure checked factorization APIs reject invalid operands through their
    /// error channel rather than returning successful matrices containing NaN.
    #[test]
    fn test_linear_solvers_reject_non_finite_inputs() {
        let identity = Matrix::identity(2);
        let nan_rhs = Matrix::from_vec(vec![f64::NAN, 1.0], 2, 1).unwrap();

        assert_eq!(
            identity.solve_lu(&nan_rhs).unwrap_err(),
            MatrixError::NonFiniteInput {
                operation: "solve_lu",
                operand: MatrixOperand::RightHandSide,
            }
        );
        assert_eq!(
            identity
                .solve_least_squares(&nan_rhs, LeastSquaresOptions::default(), false)
                .unwrap_err(),
            MatrixError::NonFiniteInput {
                operation: "solve_least_squares",
                operand: MatrixOperand::RightHandSide,
            }
        );

        let nan_coefficient = Matrix::from_vec(vec![1.0, f64::NAN, 0.0, 1.0], 2, 2).unwrap();
        let finite_rhs = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();
        assert_eq!(
            nan_coefficient.solve_lu(&finite_rhs).unwrap_err(),
            MatrixError::NonFiniteInput {
                operation: "lu_decomposition",
                operand: MatrixOperand::CoefficientMatrix,
            }
        );
        assert_eq!(
            nan_coefficient
                .solve_least_squares(&finite_rhs, LeastSquaresOptions::default(), false)
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
    fn test_linear_solvers_reject_non_finite_results() {
        let tiny_coefficient = Matrix::from_vec(vec![1e-308], 1, 1).unwrap();
        let huge_rhs = Matrix::from_vec(vec![f64::MAX], 1, 1).unwrap();

        assert_eq!(
            tiny_coefficient
                .solve_lu_with_tolerance(&huge_rhs, 0.0)
                .unwrap_err(),
            MatrixError::NonFiniteResult {
                operation: "solve_lu",
            }
        );
        assert_eq!(
            tiny_coefficient
                .solve_least_squares(&huge_rhs, LeastSquaresOptions::default(), false)
                .unwrap_err(),
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

        let m = size * 4;
        let n = size;
        let tall = random_matrix(m, n, &mut rng);
        let tall_b = random_matrix(m, 1, &mut rng);
        let serial = solve_default(&tall, &tall_b, false);
        let parallel = solve_default(&tall, &tall_b, true);
        assert_eq!(serial.info.rank, parallel.info.rank);
        assert!(serial.info.cond_est.is_finite());
        assert!(parallel.info.cond_est.is_finite());
        assert_matrix_close(&serial.solution, &parallel.solution, 1e-7);

        let wide = random_matrix(n, m, &mut rng);
        let wide_b = random_matrix(n, 1, &mut rng);
        let serial = solve_default(&wide, &wide_b, false);
        let parallel = solve_default(&wide, &wide_b, true);
        assert_eq!(serial.info.rank, parallel.info.rank);
        assert!(serial.info.cond_est.is_finite());
        assert!(parallel.info.cond_est.is_finite());
        assert_matrix_close(&serial.solution, &parallel.solution, 1e-7);
    }

    /// Verify that LU performs a row interchange when the leading pivot is
    /// zero and reconstructs the right-hand side in either execution mode.
    #[test]
    fn test_lu_solve() {
        // A zero leading coefficient makes a row interchange mandatory before
        // elimination can proceed.
        let a = Matrix::from_vec(vec![0.0, 1.0, 1.0, 0.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![2.0, 1.0], 2, 1).unwrap();

        let x = a.solve_lu(&b).unwrap();

        // Check the known coordinates before reconstructing b so compensating
        // errors cannot satisfy only the product assertion.
        assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 2.0).abs() < 1e-10);
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

    /// Accept a non-zero final pivot when it remains above the default
    /// matrix-relative LU threshold.
    #[test]
    fn test_lu_solves_nearly_singular_system_above_tolerance() {
        // Construct b from the exact solution [1, 1]. The small determinant
        // exercises tolerance handling without asking LU to classify the
        // matrix as singular under its documented default.
        let a = Matrix::from_vec(vec![1.0, 1.0, 1.0, 1.0 + 1e-12], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![2.0, 2.0 + 1e-12], 2, 1).unwrap();
        let x = a
            .solve_lu(&b)
            .expect("a finite pivot above the relative threshold is solvable");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-6);
        let reconstructed = a.try_mul(&x).expect("matrix shapes are compatible");
        assert_matrix_close(&reconstructed, &b, 1e-12);
    }

    /// Verify that the caller's LU tolerance applies to every pivot relative to
    /// the complete coefficient scale, including the final diagonal.
    #[test]
    fn test_lu_tolerance_rejects_small_final_pivot() {
        // The old trailing-row comparison divided the last pivot by itself, so
        // this matrix was accepted for every epsilon below one despite its
        // twenty-order diagonal scale separation.
        let matrix = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1e-20], 2, 2).unwrap();
        let error = matrix
            .lu_decomposition_with_tolerance(0.5)
            .expect_err("the final pivot is small relative to the matrix scale");

        assert_eq!(
            error,
            MatrixError::Singular {
                operation: "lu_decomposition",
            }
        );
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

        let serial = solve_default(&a, &b, false);
        let parallel = solve_default(&a, &b, true);
        assert_matrix_close(&serial.solution, &parallel.solution, 1e-12);
        assert!((serial.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((serial.solution[(1, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_underdetermined_min_norm() {
        // A is 1x2: x0 + x1 = 1. The minimum-norm solution is [0.5, 0.5].
        let a = Matrix::from_vec(vec![1.0, 1.0], 1, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0], 1, 1).unwrap();

        let serial = solve_default(&a, &b, false);
        let parallel = solve_default(&a, &b, true);
        assert_matrix_close(&serial.solution, &parallel.solution, 1e-12);
        assert!((serial.solution[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((serial.solution[(1, 0)] - 0.5).abs() < 1e-12);
    }

    /// Ensure that uniformly scaling a tall matrix does not change its
    /// numerical rank or condition estimate.
    #[test]
    fn test_solve_least_squares_tall_rank_is_scale_invariant() {
        // This matrix is exactly the small-scale case that the former absolute
        // tolerance floor misreported as rank zero.
        let tiny = Matrix::from_vec(vec![1e-15, 2e-15], 2, 1).unwrap();
        let tiny_b = Matrix::from_vec(vec![1e-15, 2e-15], 2, 1).unwrap();
        let tiny_result = solve_default(&tiny, &tiny_b, false);

        let unit = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();
        let unit_b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();
        let unit_result = solve_default(&unit, &unit_b, false);

        assert_eq!(tiny_result.info.rank, 1);
        assert_eq!(tiny_result.info.rank, unit_result.info.rank);
        assert!((tiny_result.info.cond_est - unit_result.info.cond_est).abs() < 1e-12);
        assert!((tiny_result.solution[(0, 0)] - 1.0).abs() < 1e-12);
    }

    /// Ensure that the transpose-based wide QR path applies the same
    /// scale-relative rank rule as the tall path.
    #[test]
    fn test_solve_least_squares_wide_rank_is_scale_invariant() {
        // The equation 1e-15*x0 + 2e-15*x1 = 1e-15 is full row rank and has
        // minimum-norm solution [0.2, 0.4].
        let tiny = Matrix::from_vec(vec![1e-15, 2e-15], 1, 2).unwrap();
        let tiny_b = Matrix::from_vec(vec![1e-15], 1, 1).unwrap();
        let result = solve_default(&tiny, &tiny_b, false);

        assert_eq!(result.info.rank, 1);
        assert!(result.info.cond_est.is_finite());
        assert!((result.solution[(0, 0)] - 0.2).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 0.4).abs() < 1e-12);
    }

    /// Verify that callers can deliberately change numerical-rank policy
    /// instead of depending on an inaccessible crate constant.
    #[test]
    fn test_solve_least_squares_honors_explicit_rank_tolerance() {
        // The second diagonal is retained by the default 1e-12 threshold but
        // discarded by the explicit 1e-3 policy.
        let coefficients = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1e-6], 2, 2).unwrap();
        let rhs = Matrix::from_vec(vec![1.0, 1e-6], 2, 1).unwrap();
        let default_result = solve_default(&coefficients, &rhs, false);
        let truncated_result = coefficients
            .solve_least_squares(
                &rhs,
                LeastSquaresOptions {
                    relative_rank_tolerance: 1e-3,
                },
                false,
            )
            .unwrap();

        assert_eq!(default_result.info.rank, 2);
        assert_eq!(truncated_result.info.rank, 1);
        assert_eq!(truncated_result.info.method, LeastSquaresMethod::JacobiSvd);
        assert!((truncated_result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(truncated_result.solution[(1, 0)], 0.0);
    }

    /// Ensure invalid public rank policy is rejected before factorization.
    #[test]
    fn test_solve_least_squares_rejects_invalid_rank_tolerance() {
        let coefficients = Matrix::identity(1);
        let rhs = Matrix::from_vec(vec![1.0], 1, 1).unwrap();
        let error = coefficients
            .solve_least_squares(
                &rhs,
                LeastSquaresOptions {
                    relative_rank_tolerance: f64::NAN,
                },
                false,
            )
            .expect_err("NaN cannot define an ordered rank threshold");

        assert_eq!(
            error,
            MatrixError::InvalidTolerance {
                operation: "solve_least_squares",
            }
        );
    }

    /// Verify that reflector normalization does not overflow when a finite
    /// leading value and finite column norm would have an unrepresentable sum.
    #[test]
    fn test_tall_qr_handles_extreme_finite_reflector_scale() {
        let coefficient = Matrix::from_vec(vec![1e308, 1e308], 2, 1).unwrap();
        let right_hand_side = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();

        let result = solve_default(&coefficient, &right_hand_side, false);

        // Both equations are identical after scaling, so x=1e-308 is the exact
        // least-squares solution. Compare by ratio because the expected value is
        // subnormal enough that an absolute tolerance would hide a large error.
        let relative_error = (result.solution[(0, 0)] / 1e-308 - 1.0).abs();
        assert!(relative_error < 1e-12, "relative error: {relative_error}");
        assert_eq!(result.info.rank, 1);
        assert_eq!(result.info.method, LeastSquaresMethod::HouseholderQr);
    }

    /// Exercise the same scale-safe reflector construction through the
    /// transpose-oriented QR path used for wide minimum-norm systems.
    #[test]
    fn test_wide_qr_handles_extreme_finite_reflector_scale() {
        let coefficient = Matrix::from_vec(vec![1e308, 1e308], 1, 2).unwrap();
        let right_hand_side = Matrix::from_vec(vec![1.0], 1, 1).unwrap();

        let result = solve_default(&coefficient, &right_hand_side, false);

        // Symmetry gives the minimum-norm solution [5e-309, 5e-309]. Check each
        // component relatively so the regression remains sensitive at this scale.
        for row in 0..2 {
            let relative_error = (result.solution[(row, 0)] / 5e-309 - 1.0).abs();
            assert!(relative_error < 1e-12, "relative error: {relative_error}");
        }
        assert_eq!(result.info.rank, 1);
        assert_eq!(result.info.method, LeastSquaresMethod::HouseholderQr);
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

            let serial = solve_default(&a, &b, false);
            let parallel = solve_default(&a, &b, true);
            assert_eq!(serial.info.rank, 3);
            assert_eq!(parallel.info.rank, 3);
            assert!(serial.info.cond_est.is_finite());
            assert!(parallel.info.cond_est.is_finite());
            assert_matrix_close(&serial.solution, &parallel.solution, 1e-8);
            for (row, expected) in x_true.iter().enumerate() {
                assert!((serial.solution[(row, 0)] - expected).abs() < 1e-8);
                assert!((parallel.solution[(row, 0)] - expected).abs() < 1e-8);
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

    #[test]
    fn test_solve_least_squares_random_wide_min_norm() {
        let mut rng = TestRng::new(0x1234_5678_9abc_def0);
        let mut a = Matrix::new(2, 4);
        a[(0, 0)] = 1.0;
        a[(1, 1)] = 1.0;

        for _ in 0..5 {
            let b0 = rng.next_f64();
            let b1 = rng.next_f64();
            let b = Matrix::from_vec(vec![b0, b1], 2, 1).unwrap();

            let serial = solve_default(&a, &b, false);
            let parallel = solve_default(&a, &b, true);
            assert_eq!(serial.info.rank, 2);
            assert_eq!(parallel.info.rank, 2);
            assert!(serial.info.cond_est.is_finite());
            assert!(parallel.info.cond_est.is_finite());
            assert_matrix_close(&serial.solution, &parallel.solution, 1e-12);
            assert!((serial.solution[(0, 0)] - b0).abs() < 1e-12);
            assert!((serial.solution[(1, 0)] - b1).abs() < 1e-12);
            assert!((serial.solution[(2, 0)]).abs() < 1e-12);
            assert!((serial.solution[(3, 0)]).abs() < 1e-12);
            assert!((parallel.solution[(0, 0)] - b0).abs() < 1e-12);
            assert!((parallel.solution[(1, 0)] - b1).abs() < 1e-12);
            assert!((parallel.solution[(2, 0)]).abs() < 1e-12);
            assert!((parallel.solution[(3, 0)]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_solve_least_squares_detects_ill_conditioning() {
        let mut a = Matrix::identity(3);
        for i in 0..3 {
            a[(i, 2)] *= 1e-8;
        }
        let b = Matrix::from_vec(vec![0.0, 0.0, 0.0], 3, 1).unwrap();

        let serial = solve_default(&a, &b, false);
        let parallel = solve_default(&a, &b, true);
        assert_eq!(serial.info.rank, 3);
        assert_eq!(parallel.info.rank, 3);
        assert!(
            serial.info.cond_est > 1e6,
            "cond_est was {}",
            serial.info.cond_est
        );
        assert!(
            parallel.info.cond_est > 1e6,
            "cond_est was {}",
            parallel.info.cond_est
        );
    }

    /// Verify that condition diagnostics account for amplification caused by
    /// off-diagonal entries, not only disparities between diagonal elements.
    #[test]
    fn test_solve_least_squares_condition_detects_off_diagonal_growth() {
        // Both diagonal entries are one, so the former max-diagonal/min-diagonal
        // heuristic reported condition one. The actual one-norm condition of
        // this triangular matrix is approximately 1e16.
        let a = Matrix::from_vec(vec![1.0, 1e8, 0.0, 1.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![0.0, 0.0], 2, 1).unwrap();
        let result = solve_default(&a, &b, false);

        assert_eq!(result.info.rank, 2);
        assert!(
            result.info.cond_est > 1e15,
            "off-diagonal condition estimate was {}",
            result.info.cond_est
        );
    }

    #[test]
    fn test_solve_least_squares_tall_full_rank() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();

        let serial = solve_default(&a, &b, false);
        let parallel = solve_default(&a, &b, true);
        assert_eq!(serial.info.rank, 2);
        assert_eq!(parallel.info.rank, 2);
        assert_eq!(serial.info.method, LeastSquaresMethod::HouseholderQr);
        assert_eq!(parallel.info.method, LeastSquaresMethod::HouseholderQr);
        assert!(serial.info.cond_est.is_finite());
        assert!(parallel.info.cond_est.is_finite());
        assert_matrix_close(&serial.solution, &parallel.solution, 1e-12);
        assert!((serial.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((serial.solution[(1, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_least_squares_wide_full_rank() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        let serial = solve_default(&a, &b, false);
        let parallel = solve_default(&a, &b, true);
        assert_eq!(serial.info.rank, 2);
        assert_eq!(parallel.info.rank, 2);
        assert_eq!(serial.info.method, LeastSquaresMethod::HouseholderQr);
        assert_eq!(parallel.info.method, LeastSquaresMethod::HouseholderQr);
        assert!(serial.info.cond_est.is_finite());
        assert!(parallel.info.cond_est.is_finite());
        assert_matrix_close(&serial.solution, &parallel.solution, 1e-12);
        assert!((serial.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((serial.solution[(1, 0)] - 2.0).abs() < 1e-12);
        assert!((serial.solution[(2, 0)] - 0.0).abs() < 1e-12);
    }

    /// Verify that a square rank-deficient system returns its unique
    /// minimum-norm pseudoinverse solution through the SVD fallback.
    #[test]
    fn test_solve_least_squares_square_rank_deficient() {
        let a = Matrix::from_vec(vec![1.0, 1.0, 2.0, 2.0], 2, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        // Both parallel flags share the serial Jacobi fallback and must expose
        // identical numerical and method diagnostics.
        let serial = solve_default(&a, &b, false);
        let parallel = solve_default(&a, &b, true);

        assert_matrix_close(&serial.solution, &parallel.solution, 1e-12);
        assert!((serial.solution[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((serial.solution[(1, 0)] - 0.5).abs() < 1e-12);
        assert_eq!(serial.info.rank, 1);
        assert_eq!(serial.info.method, LeastSquaresMethod::JacobiSvd);
        assert_eq!(parallel.info, serial.info);
        assert!((serial.info.cond_est - 1.0).abs() < 1e-12);
    }

    /// Ensure SVD rank classification and its pseudoinverse solution remain
    /// invariant across very small and very large finite unit scales.
    #[test]
    fn test_solve_least_squares_rank_deficient_scale_invariance() {
        for scale in [1e-200_f64, 1.0, 1e200] {
            let a = Matrix::from_vec(vec![scale, scale, 2.0 * scale, 2.0 * scale], 2, 2).unwrap();
            let b = Matrix::from_vec(vec![scale, 2.0 * scale], 2, 1).unwrap();

            let result = solve_default(&a, &b, false);

            // Uniformly scaling both operands leaves the Moore-Penrose answer
            // unchanged, including at magnitudes where naïve squared norms
            // would underflow or overflow.
            assert!((result.solution[(0, 0)] - 0.5).abs() < 1e-12);
            assert!((result.solution[(1, 0)] - 0.5).abs() < 1e-12);
            assert_eq!(result.info.rank, 1);
            assert_eq!(result.info.method, LeastSquaresMethod::JacobiSvd);
            assert!((result.info.cond_est - 1.0).abs() < 1e-12);
        }
    }

    /// Verify that SVD minimizes residual norm for an inconsistent tall system
    /// and then selects the minimum-norm point among its minimizers.
    #[test]
    fn test_solve_least_squares_tall_rank_deficient_inconsistent() {
        let a = Matrix::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();

        let result = solve_default(&a, &b, false);

        // The best fitted shared row value is mean(b)=2. Splitting that value
        // equally across duplicate columns minimizes the variable norm.
        assert!((result.solution[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.solution[(1, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(result.info.rank, 1);
        assert_eq!(result.info.method, LeastSquaresMethod::JacobiSvd);
        let residual = &(&a * &result.solution) - &b;
        assert!((residual.norm() - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    /// Verify the transpose-oriented Jacobi SVD for a wide matrix whose rows
    /// are themselves linearly dependent.
    #[test]
    fn test_solve_least_squares_wide_rank_deficient_minimum_norm() {
        let a = Matrix::from_vec(vec![1.0, 0.0, 1.0, 2.0, 0.0, 2.0], 2, 3).unwrap();
        let b = Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap();

        let result = solve_default(&a, &b, false);

        // The constrained sum x0+x2=1 is divided equally; the unconstrained x1
        // component is zero in the Moore-Penrose solution.
        assert!((result.solution[(0, 0)] - 0.5).abs() < 1e-12);
        assert!(result.solution[(1, 0)].abs() < 1e-12);
        assert!((result.solution[(2, 0)] - 0.5).abs() < 1e-12);
        assert_eq!(result.info.rank, 1);
        assert_eq!(result.info.method, LeastSquaresMethod::JacobiSvd);
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

    /// Confirm that LU shape, tolerance, and singularity failures use distinct
    /// structured variants rather than unrelated error strings.
    #[test]
    fn test_lu_reports_structured_errors() {
        let rectangular = Matrix::new(2, 3);
        assert_eq!(
            rectangular.lu_decomposition().unwrap_err(),
            MatrixError::NonSquare {
                operation: "lu_decomposition",
                matrix: (2, 3),
            }
        );

        let identity = Matrix::identity(2);
        assert_eq!(
            identity
                .lu_decomposition_with_tolerance(f64::NAN)
                .unwrap_err(),
            MatrixError::InvalidTolerance {
                operation: "lu_decomposition",
            }
        );

        let singular = Matrix::from_vec(vec![1.0, 2.0, 2.0, 4.0], 2, 2).unwrap();
        assert_eq!(
            singular.lu_decomposition().unwrap_err(),
            MatrixError::Singular {
                operation: "lu_decomposition",
            }
        );
    }
}
