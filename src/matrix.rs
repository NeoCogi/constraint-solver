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

//! Dense matrix operations with checked recoverable shapes and one Jacobi-SVD
//! least-squares path for square, tall, and wide systems.
//!
//! Arithmetic writes into exact-shape caller-provided buffers, rejects
//! non-finite operands, and returns an error when finite operands produce an
//! unrepresentable `f64` result. Allocation is confined to constructors and
//! explicit workspaces. The least-squares solver applies the same finite-state
//! boundary to its inputs, factors, and completed result.

use std::fmt;
use std::ops::{Index, IndexMut};

use crate::scaled::{CompensatedSum, product_ratio};

/// Identifies which matrix operand contained a non-finite value before a
/// checked arithmetic or linear-solve operation began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixOperand {
    /// Left-hand matrix supplied to elementwise or multiplicative arithmetic.
    LeftHandSide,
    /// Coefficient matrix being factorized or applied by the operation.
    CoefficientMatrix,
    /// Right-hand-side matrix supplied to a linear solve.
    RightHandSide,
}

impl fmt::Display for MatrixOperand {
    /// Render a stable human-readable operand name for structured error output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixOperand::LeftHandSide => write!(f, "left-hand side"),
            MatrixOperand::CoefficientMatrix => write!(f, "coefficient matrix"),
            MatrixOperand::RightHandSide => write!(f, "right-hand side"),
        }
    }
}

/// Convert the relative numerical-rank threshold into the scale of a singular-
/// value spectrum.
///
/// A zero factor produces a zero tolerance and therefore rank zero because all
/// comparisons use strict inequality. Non-finite factors are rejected by the
/// factorization before this helper's result is consumed.
fn numerical_rank_tolerance(max_factor_value: f64, relative_tolerance: f64) -> f64 {
    // Multiplication, rather than `max_factor_value.max(1.0)`, keeps the cutoff
    // proportional to the computed spectrum instead of injecting a fixed
    // application-unit floor.
    relative_tolerance * max_factor_value
}

/// Derive the relative numerical-rank threshold from floating-point precision
/// and matrix size.
///
/// Multiplying machine epsilon by the larger dimension accounts for roundoff
/// accumulated across one factorization dimension without imposing a fixed
/// application-scale cutoff. The resulting threshold is then multiplied by the
/// computed factor scale through [`numerical_rank_tolerance`].
fn numerical_rank_relative_tolerance(rows: usize, cols: usize) -> f64 {
    f64::EPSILON * rows.max(cols) as f64
}

/// Structured recoverable failure returned by matrix and linear-algebra
/// operations.
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
    /// A caller-provided output buffer does not have the exact result shape.
    OutputShapeMismatch {
        /// Stable name of the operation that rejected the output buffer.
        operation: &'static str,
        /// Shape the operation requires as `(rows, columns)`.
        expected: (usize, usize),
        /// Actual output-buffer shape as `(rows, columns)`.
        actual: (usize, usize),
    },
    /// A least-squares workspace was constructed for another coefficient shape.
    WorkspaceShapeMismatch {
        /// Stable name of the operation that rejected the workspace.
        operation: &'static str,
        /// Coefficient shape required by the current operation.
        expected: (usize, usize),
        /// Coefficient shape accepted by the supplied workspace.
        actual: (usize, usize),
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
    /// A matrix input contained NaN or infinity before checked arithmetic or
    /// factorization began.
    NonFiniteInput {
        /// Stable name of the operation that validated the operand.
        operation: &'static str,
        /// Operand in which the invalid value was observed.
        operand: MatrixOperand,
    },
    /// A scalar arithmetic operand was NaN or infinity.
    NonFiniteScalar {
        /// Stable name of the operation that validated the scalar.
        operation: &'static str,
    },
    /// Factorization arithmetic produced or encountered a non-finite internal
    /// value.
    NonFiniteFactor {
        /// Stable name of the factorization or solve that detected the value.
        operation: &'static str,
    },
    /// Finite inputs produced a non-finite arithmetic or solve result.
    NonFiniteResult {
        /// Stable name of the operation that produced the invalid result.
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
    ///
    /// This is a floating-point classification, not an exact algebraic rank
    /// proof. A singular value close to the cutoff can cross it after ordinary
    /// input rounding or a different Jacobi column traversal.
    pub rank: usize,
    /// Ratio of the largest to smallest retained singular value. Infinity means
    /// the matrix has no retained non-zero singular subspace.
    pub cond_est: f64,
    /// Fraction of the right-hand-side norm captured by retained directions.
    ///
    /// For right-hand side `b` and its orthogonal projection `P*b` onto the
    /// retained column space, this is `||P*b|| / ||b||`. The value lies in
    /// `[0, 1]` after rounding. It is zero when `b` is zero or when the retained
    /// factorization contains no representable component of `b`.
    ///
    /// This dimensionless value is the factorization's stationarity result. Zero
    /// is reserved for an absent retained component; when an exact non-zero
    /// fraction is smaller than the `f64` subnormal range, the smallest positive
    /// `f64` is returned. Keeping that distinction lets callers separate model
    /// stationarity from an underflowed physical slope.
    pub retained_rhs_projection_fraction: f64,
    /// Initial residual-norm slope along the returned solution.
    ///
    /// If `x` is the returned solution, this is the derivative at `alpha = 0`
    /// of `||alpha*A*x - b||`: `-||P*b||² / ||b||`. It is zero when `b` is zero,
    /// when no retained direction can reduce it, or when the negative physical
    /// slope is smaller than the `f64` subnormal range. The value has the same
    /// units as `b`, is never positive, and may be negative infinity only when
    /// the completed mathematical quantity exceeds `f64` range.
    ///
    /// For a least-squares Newton correction solving `J*p = -f`, this is exactly
    /// the retained linear model's directional derivative of `||f||`. Returning
    /// it from the factorization avoids reconstructing `J*p` with range-sensitive
    /// products or squaring a tiny dimensionless ratio before a large residual
    /// norm can restore a representable slope.
    pub residual_norm_slope: f64,
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
            MatrixError::OutputShapeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "Output shape mismatch for {operation}: expected {}x{}, got {}x{}",
                expected.0, expected.1, actual.0, actual.1
            ),
            MatrixError::WorkspaceShapeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "Workspace shape mismatch for {operation}: expected {}x{}, got {}x{}",
                expected.0, expected.1, actual.0, actual.1
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
            MatrixError::NonFiniteInput { operation, operand } => {
                write!(f, "Non-finite {operand} supplied to {operation}")
            }
            MatrixError::NonFiniteScalar { operation } => {
                write!(f, "Non-finite scalar supplied to {operation}")
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

/// Reusable storage for one fixed-shape Jacobi-SVD least-squares problem.
///
/// Construct this workspace once for coefficient shape `rows x cols`, then pass
/// it to every [`Matrix::solve_least_squares_into`] call with that shape. Its
/// fields are private so callers cannot invalidate factorization dimensions or
/// accidentally depend on transient decomposition values.
pub struct LeastSquaresWorkspace {
    /// Coefficient row count this workspace accepts.
    rows: usize,
    /// Coefficient column count and solution row count this workspace accepts.
    cols: usize,
    /// Normalized coefficient matrix, transposed for wide input, rotated in
    /// place until its columns are mutually orthogonal.
    orthogonal_columns: Matrix,
    /// Orthogonal right-singular vectors accumulated during Jacobi rotations.
    right_vectors: Matrix,
    /// Euclidean norms of the rotated columns.
    singular_values: Vec<f64>,
    /// Right-hand-side projections formed after maximum-magnitude normalization.
    projections: Vec<f64>,
}

impl LeastSquaresWorkspace {
    /// Allocate all numerical storage required for a coefficient matrix shape.
    ///
    /// The workspace accepts both tall and wide matrices by storing the taller
    /// orientation as `max(rows, cols) x min(rows, cols)`. No solve using this
    /// workspace allocates, resizes, or replaces these buffers.
    ///
    /// # Panics
    ///
    /// Panics with the requested factor shape if its dimensions or storage are
    /// impossible to represent.
    #[track_caller]
    pub fn new(rows: usize, cols: usize) -> Self {
        let factor_rows = rows.max(cols);
        let component_count = rows.min(cols);

        // Allocate every buffer at its final length. Solve calls overwrite all
        // active entries, so reuse never depends on a preceding decomposition.
        Self {
            rows,
            cols,
            orthogonal_columns: Matrix::new(factor_rows, component_count),
            right_vectors: Matrix::new(component_count, component_count),
            singular_values: vec![0.0; component_count],
            projections: vec![0.0; component_count],
        }
    }
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
    /// Panics with the requested shape if `rows * cols` cannot be represented by
    /// `usize` or contiguous storage cannot be reserved. Such failures are fatal
    /// resource errors rather than recoverable numerical outcomes.
    #[track_caller]
    pub fn new(rows: usize, cols: usize) -> Self {
        // Check the logical shape before touching the allocator so the panic
        // explains the caller's dimensions instead of reporting a later vector
        // capacity failure without matrix context.
        let element_count = rows
            .checked_mul(cols)
            .unwrap_or_else(|| panic!("matrix shape {rows}x{cols} overflows usize element count"));

        // Use `try_reserve_exact` only to replace opaque allocator diagnostics
        // with the requested shape and concrete reservation cause. The public
        // operation still panics because no useful matrix result exists.
        let mut data = Vec::new();
        data.try_reserve_exact(element_count)
            .unwrap_or_else(|error| {
                panic!(
                    "cannot allocate {element_count} f64 elements for matrix shape {rows}x{cols}: {error}"
                )
            });
        // The exact reservation above guarantees that zero initialization does
        // not need another allocation attempt.
        data.resize(element_count, 0.0);

        Matrix { data, rows, cols }
    }

    /// Construct a row-major matrix from an owned element buffer.
    ///
    /// Returns a structured error when the supplied buffer length does not
    /// exactly match the shape.
    ///
    /// # Panics
    ///
    /// Panics when `rows * cols` overflows `usize`. The owned buffer cannot
    /// represent such a logical shape, so there is no recoverable matrix value.
    #[track_caller]
    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, MatrixError> {
        let expected = rows
            .checked_mul(cols)
            .unwrap_or_else(|| panic!("matrix shape {rows}x{cols} overflows usize element count"));
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
    ///
    /// # Panics
    ///
    /// Panics with the requested square shape when its element count or storage
    /// cannot be represented.
    #[track_caller]
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

    /// Return the backing allocation address for storage-reuse unit tests.
    ///
    /// This is crate-visible only in test builds; numerical callers must reason
    /// through the documented no-allocation contract rather than storage layout.
    #[cfg(test)]
    pub(crate) fn storage_address(&self) -> *const f64 {
        self.data.as_ptr()
    }

    /// Verify that a caller-owned output matrix has one required shape.
    ///
    /// Centralizing this check keeps every allocation-free operation consistent:
    /// output buffers are never resized, replaced, or implicitly allocated.
    fn require_output_shape(
        output: &Matrix,
        expected: (usize, usize),
        operation: &'static str,
    ) -> Result<(), MatrixError> {
        // Shape metadata is sufficient because every constructor has already
        // proved that its storage length matches its logical dimensions.
        let actual = (output.rows, output.cols);
        if actual != expected {
            return Err(MatrixError::OutputShapeMismatch {
                operation,
                expected,
                actual,
            });
        }
        Ok(())
    }

    /// Reject a non-finite matrix operand before an arithmetic output is touched.
    ///
    /// Every checked arithmetic and factorization path assumes finite accepted
    /// state. Enforcing that invariant at the public boundary removes build-
    /// profile-dependent assertions and gives callers a retryable error.
    fn require_finite_input(
        input: &Matrix,
        operation: &'static str,
        operand: MatrixOperand,
    ) -> Result<(), MatrixError> {
        // A complete scan before the first write preserves the output buffer for
        // operand errors; only an arithmetic result error can leave scratch data.
        if !input.all_finite() {
            return Err(MatrixError::NonFiniteInput { operation, operand });
        }
        Ok(())
    }

    /// Copy this matrix's transpose into an exact-shape caller-owned buffer.
    ///
    /// This operation performs no allocation and preserves every stored `f64`
    /// bit pattern, including NaN and infinity, because transposition performs
    /// no arithmetic. `output` must have shape `self.cols() x self.rows()`.
    pub fn transpose_into(&self, output: &mut Matrix) -> Result<(), MatrixError> {
        Self::require_output_shape(output, (self.cols, self.rows), "transpose")?;
        if output.data.is_empty() {
            // A valid shape such as `usize::MAX x 0` has no coordinates to
            // copy. Returning from storage cardinality avoids walking the huge
            // logical row dimension even though the operation is already
            // complete.
            return Ok(());
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                output[(j, i)] = self[(i, j)];
            }
        }
        Ok(())
    }

    /// Solve a rectangular least-squares problem into reusable caller storage.
    ///
    /// A one-sided Jacobi decomposition classifies numerical rank from singular
    /// values and computes the truncated Moore-Penrose minimum-norm solution for
    /// square, tall, and wide systems. Using the same decomposition for both
    /// decisions removes the former order-sensitive QR pre-classification and
    /// avoids a competing factorization path. Column permutations have the same
    /// mathematical singular spectrum, but finite Jacobi sweeps can round
    /// differently; rank is not guaranteed identical when a singular value lies
    /// near the numerical cutoff.
    ///
    /// The coefficient matrix, right-hand side, solution, and workspace must
    /// respectively have shapes `m x n`, `m x 1`, `n x 1`, and a workspace
    /// constructed by [`LeastSquaresWorkspace::new(m, n)`]. The method performs
    /// no allocation and overwrites the solution only after all input and shape
    /// validation succeeds. A later numerical error can leave it partially
    /// overwritten, so it remains scratch until `Ok` is returned.
    ///
    /// On success, [`LeastSquaresInfo`] reports rank, retained conditioning, the
    /// fraction of `b` captured by the same retained singular directions, and
    /// the residual-norm slope along the returned solution. No secondary matrix
    /// multiplication is used to reconstruct those model diagnostics.
    ///
    /// Each singular solution contribution is formed through exponent-safe
    /// factor arithmetic and rounded to `f64` before component-order Neumaier
    /// accumulation. A non-finite contribution or partial compensated total
    /// returns [`MatrixError::NonFiniteResult`], even if a later singular
    /// contribution could have cancelled it to a finite value. Finite underflow
    /// follows ordinary `f64` rounding. This explicit boundary keeps the solver
    /// auditable without a hidden wider-range reduction type.
    pub fn solve_least_squares_into(
        &self,
        b: &Matrix,
        solution: &mut Matrix,
        workspace: &mut LeastSquaresWorkspace,
    ) -> Result<LeastSquaresInfo, MatrixError> {
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
        Self::require_output_shape(solution, (self.cols, 1), "solve_least_squares")?;
        let workspace_shape = (workspace.rows, workspace.cols);
        if workspace_shape != (self.rows, self.cols) {
            return Err(MatrixError::WorkspaceShapeMismatch {
                operation: "solve_least_squares",
                expected: (self.rows, self.cols),
                actual: workspace_shape,
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
        // and require no numerical factorization. Overwrite a non-empty
        // solution because the caller may be reusing values from a prior solve.
        if n == 0 || m == 0 {
            solution.data.fill(0.0);
            return Ok(LeastSquaresInfo {
                rank: 0,
                cond_est: f64::INFINITY,
                retained_rhs_projection_fraction: 0.0,
                residual_norm_slope: -0.0,
            });
        }

        // The decomposition supplies both the returned correction and the rank
        // diagnostics, so classification cannot disagree with solution policy.
        let info = Self::solve_least_squares_svd_into(
            self,
            b,
            solution,
            workspace,
            relative_rank_tolerance,
        )?;

        // A successful checked solve must never smuggle NaN or infinity through
        // its `Ok` channel, even when finite inputs overflow during substitution.
        if !solution.all_finite() {
            return Err(MatrixError::NonFiniteResult {
                operation: "solve_least_squares",
            });
        }
        Ok(info)
    }

    /// Compute a Moore-Penrose pseudoinverse solution with a one-sided Jacobi
    /// singular-value decomposition.
    ///
    /// The Jacobi kernel operates on a matrix with at least as many rows as
    /// columns. Tall inputs are decomposed directly; wide inputs decompose the
    /// transpose and apply the transposed SVD identities. In both orientations,
    /// singular directions below the relative rank tolerance are omitted,
    /// which selects the unique minimum-Euclidean-norm least-squares solution.
    fn solve_least_squares_svd_into(
        a: &Matrix,
        b: &Matrix,
        solution: &mut Matrix,
        workspace: &mut LeastSquaresWorkspace,
        relative_rank_tolerance: f64,
    ) -> Result<LeastSquaresInfo, MatrixError> {
        let rows = a.rows;
        let cols = a.cols;
        let tall_orientation = rows >= cols;

        // Normalize the complete coefficient matrix before factorization. A
        // matrix can contain only finite entries while its largest singular
        // value exceeds `f64::MAX` (for example, a 2x2 matrix filled with
        // `1e308`). In exact arithmetic, uniform normalization preserves
        // singular vectors, relative rank, condition, and—after restoring this
        // scale during reconstruction—the least-squares problem. Division and
        // Jacobi sweeps still round in `f64`, so a singular value near the rank
        // cutoff can change classification. The all-zero matrix uses a neutral
        // scale because it has no non-zero direction to normalize.
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

        // Normalize the right-hand side independently before projection. The
        // singular vectors have unit-scale entries, so every projection term
        // then stays in ordinary `f64` range. Its physical magnitude is restored
        // exactly once during solution reconstruction. An all-zero right-hand
        // side uses a neutral scale and therefore reconstructs exact zero.
        let rhs_scale = b
            .data
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let rhs_scale = if rhs_scale == 0.0 { 1.0 } else { rhs_scale };

        // Decomposing A^T for a wide system keeps the Jacobi kernel compact and
        // avoids manufacturing zero singular columns merely to make A tall.
        // Populate the workspace orientation directly so normalization,
        // transposition, and reuse remain one pass over caller-owned storage.
        let factor_input = &mut workspace.orthogonal_columns;
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
        Self::one_sided_jacobi_svd_columns(
            &mut workspace.orthogonal_columns,
            &mut workspace.right_vectors,
            &mut workspace.singular_values,
            relative_rank_tolerance,
        )?;

        // A relative threshold keeps the cutoff tied to the computed spectrum
        // under uniform coefficient scaling instead of imposing an
        // application-scale floor. The strict comparison deliberately
        // classifies an all-zero factor as rank zero when both the maximum and
        // tolerance are zero.
        let max_singular_value = workspace
            .singular_values
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let rank_tolerance = numerical_rank_tolerance(max_singular_value, relative_rank_tolerance);
        let mut rank = 0usize;
        let mut min_retained_singular_value = f64::INFINITY;
        for &singular_value in &workspace.singular_values {
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
        // Project the normalized right-hand side onto every retained singular
        // direction. These bounded dot products use ordinary compensated
        // `f64`; exponent-safe factor formation is deferred until physical
        // scales participate in the slope and solution reconstruction.
        workspace.projections.fill(0.0);
        for (component, projection) in workspace.projections.iter_mut().enumerate() {
            let singular_value = workspace.singular_values[component];
            if singular_value <= rank_tolerance {
                // Omitting this singular direction is exactly the pseudoinverse
                // rule that removes unsupported null-space components.
                continue;
            }

            let projected_rhs = if tall_orientation {
                // A*V=B=U*Sigma, so this coefficient is
                // (U_component^T*b)/sigma. Normalizing B before the dot product
                // prevents squaring a large or tiny singular value explicitly.
                Self::normalized_column_dot(
                    &workspace.orthogonal_columns,
                    component,
                    singular_value,
                    b,
                    rhs_scale,
                )
            } else {
                // A^T*V=B=U*Sigma implies A=V*Sigma*U^T. Project the right-hand
                // side onto V, then map the scaled coefficient into U to obtain
                // the minimum-norm variable vector.
                Self::normalized_column_dot(&workspace.right_vectors, component, 1.0, b, rhs_scale)
            };
            if !projected_rhs.is_finite() {
                // `product_ratio` relies on finite private operands. Reject an
                // unrepresentable compensated projection here so debug and
                // optimized builds share the same checked boundary.
                return Err(MatrixError::NonFiniteResult {
                    operation: "solve_least_squares",
                });
            }
            *projection = projected_rhs;
        }

        // Measure the retained projection and complete right-hand-side norm in
        // the independently normalized coordinates used above. Their bounded
        // norms are combined with `rhs_scale` only once below, avoiding either a
        // squared physical norm or a range-sensitive divide-then-multiply order.
        let mut normalized_rhs_norm = 0.0_f64;
        for row in 0..b.rows {
            normalized_rhs_norm = normalized_rhs_norm.hypot(b[(row, 0)] / rhs_scale);
        }
        let retained_projection_norm = workspace
            .projections
            .iter()
            .fold(0.0_f64, |norm, projection| norm.hypot(*projection));
        let (retained_rhs_projection_fraction, residual_norm_slope) = if normalized_rhs_norm == 0.0
        {
            // The zero vector has no direction to project and needs no linear
            // correction. Return both additive identities rather than undefined
            // zero-over-zero quotients.
            (0.0, -0.0)
        } else {
            // The norm fraction answers only whether the retained SVD model
            // can reduce this right hand side. Reserve exact zero for no
            // retained component, saturating a smaller-than-subnormal ratio
            // upward only when that distinction would otherwise be lost.
            let rounded_fraction = retained_projection_norm / normalized_rhs_norm;
            let fraction = if retained_projection_norm > 0.0 && rounded_fraction == 0.0 {
                f64::from_bits(1)
            } else {
                // Clamp a possible final-rounding overshoot so the public
                // quantity keeps its documented unit interval.
                rounded_fraction.min(1.0)
            };

            // Form `rhs_scale * projection_norm^2 / normalized_rhs_norm`
            // through exponent arithmetic. Squaring the normalized projection
            // first could underflow even when multiplication by `rhs_scale`
            // makes the completed model slope representable.
            let slope = -product_ratio(
                [
                    rhs_scale,
                    retained_projection_norm,
                    retained_projection_norm,
                ],
                [normalized_rhs_norm],
            );
            (fraction, slope)
        };

        // Reconstruct each output coordinate from individually range-safe
        // singular contributions. Each completed contribution is rounded to
        // `f64`, checked, and accumulated in component order through ordinary
        // Neumaier compensation. This intentionally does not rescue a partial
        // overflow through a later cancelling singular direction.
        for row in 0..cols {
            let mut reconstructed = CompensatedSum::default();
            for (component, &projection) in workspace.projections.iter().enumerate() {
                let singular_value = workspace.singular_values[component];
                if singular_value <= rank_tolerance {
                    continue;
                }

                let contribution = if tall_orientation {
                    // V maps the projected coordinate back to the original
                    // variable space. The normalized singular value and the
                    // global coefficient scale together restore A's units.
                    product_ratio(
                        [
                            projection,
                            workspace.right_vectors[(row, component)],
                            rhs_scale,
                        ],
                        [singular_value, coefficient_scale],
                    )
                } else {
                    // In the transposed orientation, the stored orthogonal
                    // column equals U*sigma. Dividing by sigma once recovers U,
                    // and dividing the projection by the original sigma adds a
                    // second normalized singular-value denominator.
                    product_ratio(
                        [
                            projection,
                            workspace.orthogonal_columns[(row, component)],
                            rhs_scale,
                        ],
                        [singular_value, singular_value, coefficient_scale],
                    )
                };
                if !contribution.is_finite() {
                    // An individually unrepresentable singular contribution is
                    // outside the documented ordinary-f64 reduction contract.
                    return Err(MatrixError::NonFiniteResult {
                        operation: "solve_least_squares",
                    });
                }

                reconstructed.add(contribution);
                if !reconstructed.total().is_finite() {
                    // Inspect the Copy accumulator after every component so a
                    // partial overflow is rejected at the operation that caused
                    // it instead of being hidden by a later cancellation.
                    return Err(MatrixError::NonFiniteResult {
                        operation: "solve_least_squares",
                    });
                }
            }

            let value = reconstructed.total();
            if !value.is_finite() {
                return Err(MatrixError::NonFiniteResult {
                    operation: "solve_least_squares",
                });
            }
            solution[(row, 0)] = value;
        }

        Ok(LeastSquaresInfo {
            rank,
            cond_est,
            retained_rhs_projection_fraction,
            residual_norm_slope,
        })
    }

    /// Orthogonalize the columns of a tall-or-square matrix with cyclic Jacobi
    /// rotations while accumulating its right singular vectors.
    ///
    /// Pairwise Gram entries are evaluated after scaling both columns by their
    /// largest magnitude. That scaling prevents the convergence test and
    /// rotation angle from overflowing for otherwise finite input matrices.
    /// `relative_rank_tolerance` is the caller's eventual pseudoinverse cutoff,
    /// allowing the factorization and solution phases to share one definition of
    /// a numerical null direction.
    fn one_sided_jacobi_svd_columns(
        orthogonal_columns: &mut Matrix,
        right_vectors: &mut Matrix,
        singular_values: &mut [f64],
        relative_rank_tolerance: f64,
    ) -> Result<(), MatrixError> {
        debug_assert!(orthogonal_columns.rows >= orthogonal_columns.cols);

        // Rotate the normalized factor in place and reset the reused right-vector
        // matrix to the identity that begins every independent decomposition.
        let row_count = orthogonal_columns.rows;
        let column_count = orthogonal_columns.cols;
        debug_assert_eq!(
            (right_vectors.rows, right_vectors.cols),
            (column_count, column_count)
        );
        debug_assert_eq!(singular_values.len(), column_count);
        right_vectors.data.fill(0.0);
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

            // Rank-deficient inputs converge by driving one or more column norms
            // into the numerical null space. Requiring relative orthogonality
            // from such a roundoff-sized column is ill-posed: its tiny residual
            // can remain almost parallel to a retained column and trigger the
            // same meaningless rotations forever. Derive the deflation scale
            // from the largest current column norm and the exact tolerance that
            // the caller will later use to truncate singular values.
            let mut largest_column_norm = 0.0_f64;
            for column in 0..column_count {
                let mut column_norm = 0.0_f64;
                for row in 0..row_count {
                    column_norm = column_norm.hypot(orthogonal_columns[(row, column)]);
                }
                largest_column_norm = largest_column_norm.max(column_norm);
            }
            let deflation_threshold =
                numerical_rank_tolerance(largest_column_norm, relative_rank_tolerance);

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
                    let mut left_norm_squared = CompensatedSum::default();
                    let mut right_norm_squared = CompensatedSum::default();
                    let mut cross_product = CompensatedSum::default();
                    for row in 0..row_count {
                        let left = orthogonal_columns[(row, left_column)] / pair_scale;
                        let right = orthogonal_columns[(row, right_column)] / pair_scale;
                        left_norm_squared.add(left * left);
                        right_norm_squared.add(right * right);
                        cross_product.add(left * right);
                    }
                    let left_norm_squared = left_norm_squared.total();
                    let right_norm_squared = right_norm_squared.total();
                    let cross_product = cross_product.total();

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

                    // Do not rotate a direction already below the numerical-rank
                    // boundary. Its singular value will be omitted by the shared
                    // pseudoinverse policy, while rotating it can only amplify
                    // relative roundoff and prevent a zero-rotation sweep.
                    let left_norm = pair_scale * left_norm_squared.sqrt();
                    let right_norm = pair_scale * right_norm_squared.sqrt();
                    if left_norm <= deflation_threshold || right_norm <= deflation_threshold {
                        continue;
                    }
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
            singular_values[column] = singular_value;
        }

        Ok(())
    }

    /// Compute a compensated dot product between two explicitly normalized
    /// columns.
    ///
    /// `column_scale` is normally the column's singular value, producing a unit
    /// left singular vector without allocating it. Passing one computes an
    /// ordinary column dot product for an already normalized orthogonal matrix.
    /// `right_scale` similarly bounds every right-hand-side entry. Consequently
    /// no term needs an exponent-extended representation.
    fn normalized_column_dot(
        left: &Matrix,
        left_column: usize,
        column_scale: f64,
        right: &Matrix,
        right_scale: f64,
    ) -> f64 {
        debug_assert_eq!(left.rows, right.rows);
        debug_assert_eq!(right.cols, 1);
        debug_assert!(column_scale > 0.0 && column_scale.is_finite());
        debug_assert!(right_scale > 0.0 && right_scale.is_finite());

        // Both factors have magnitude at most one apart from harmless singular-
        // vector roundoff. Neumaier compensation improves the finite reduction
        // without pretending that arbitrary dot products have a wider range.
        let mut dot = CompensatedSum::default();
        for row in 0..left.rows {
            let left_value = left[(row, left_column)] / column_scale;
            let right_value = right[(row, 0)] / right_scale;
            dot.add(left_value * right_value);
        }
        dot.total()
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
    /// are reported as evaluation failures rather than flowing into rank or
    /// convergence logic.
    pub fn all_finite(&self) -> bool {
        // `Iterator::all` stops at the first invalid element and does not expose
        // the matrix's private storage to callers.
        self.data.iter().all(|value| value.is_finite())
    }

    /// Add two matrices into an exact-shape caller-owned output buffer.
    ///
    /// The operation allocates nothing. Both inputs must have identical shapes
    /// and finite elements, and `output` must have that same shape. A
    /// `NonFiniteResult` error can leave `output` partially overwritten, so
    /// callers should treat it as scratch until the method returns `Ok(())`.
    pub fn add_into(&self, rhs: &Matrix, output: &mut Matrix) -> Result<(), MatrixError> {
        // Elementwise arithmetic has no meaningful broadcasting in this dense
        // API, so validate both input dimensions before inspecting the output.
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "add",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }
        Self::require_output_shape(output, (self.rows, self.cols), "add")?;
        Self::require_finite_input(self, "add", MatrixOperand::LeftHandSide)?;
        Self::require_finite_input(rhs, "add", MatrixOperand::RightHandSide)?;

        for i in 0..self.data.len() {
            let value = self.data[i] + rhs.data[i];
            if !value.is_finite() {
                return Err(MatrixError::NonFiniteResult { operation: "add" });
            }
            output.data[i] = value;
        }
        Ok(())
    }

    /// Subtract two matrices into an exact-shape caller-owned output buffer.
    ///
    /// The operation allocates nothing and applies the same finite-input,
    /// finite-result, exact-shape, and scratch-on-result-error contract as
    /// [`Matrix::add_into`].
    pub fn sub_into(&self, rhs: &Matrix, output: &mut Matrix) -> Result<(), MatrixError> {
        // Preserve the same strict shape contract as addition before inspecting
        // the caller-owned output or modifying its storage.
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "sub",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }
        Self::require_output_shape(output, (self.rows, self.cols), "sub")?;
        Self::require_finite_input(self, "sub", MatrixOperand::LeftHandSide)?;
        Self::require_finite_input(rhs, "sub", MatrixOperand::RightHandSide)?;

        for i in 0..self.data.len() {
            let value = self.data[i] - rhs.data[i];
            if !value.is_finite() {
                return Err(MatrixError::NonFiniteResult { operation: "sub" });
            }
            output.data[i] = value;
        }
        Ok(())
    }

    /// Multiply two matrices into an exact-shape caller-owned output buffer.
    ///
    /// The operation allocates nothing. Inputs must be finite and shape
    /// compatible; `output` must have shape `self.rows() x rhs.cols()`. Each
    /// completed dot product is checked before it is stored. A
    /// `NonFiniteResult` error can leave earlier cells overwritten, so callers
    /// should treat `output` as scratch until the method returns `Ok(())`.
    pub fn mul_into(&self, rhs: &Matrix, output: &mut Matrix) -> Result<(), MatrixError> {
        // The shared dimension must match before any row slice is constructed.
        if self.cols != rhs.rows {
            return Err(MatrixError::DimensionMismatch {
                operation: "mul",
                left: (self.rows, self.cols),
                right: (rhs.rows, rhs.cols),
            });
        }
        Self::require_output_shape(output, (self.rows, rhs.cols), "mul")?;
        Self::require_finite_input(self, "mul", MatrixOperand::LeftHandSide)?;
        Self::require_finite_input(rhs, "mul", MatrixOperand::RightHandSide)?;

        if output.data.is_empty() {
            // No output cell exists when either output dimension is zero. In
            // particular, a valid `usize::MAX x 0` result must not execute an
            // effectively unbounded outer-row loop whose body can never write.
            return Ok(());
        }
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                // Ordinary matrix multiplication deliberately observes `f64`
                // range. Compensation improves rounding of finite terms but
                // does not rescue a product or partial sum that overflowed;
                // callers receive `NonFiniteResult` and can rescale or retry.
                let mut dot = CompensatedSum::default();
                for k in 0..self.cols {
                    dot.add(self[(i, k)] * rhs[(k, j)]);
                }
                let value = dot.total();
                if !value.is_finite() {
                    return Err(MatrixError::NonFiniteResult { operation: "mul" });
                }
                output[(i, j)] = value;
            }
        }
        Ok(())
    }

    /// Multiply every element by a scalar into an exact-shape output buffer.
    ///
    /// The matrix, scalar, and completed products must be finite. The operation
    /// allocates nothing; `output` must match `self` and is scratch after a
    /// `NonFiniteResult` error.
    pub fn scale_into(&self, scalar: f64, output: &mut Matrix) -> Result<(), MatrixError> {
        Self::require_output_shape(output, (self.rows, self.cols), "scale")?;
        Self::require_finite_input(self, "scale", MatrixOperand::LeftHandSide)?;
        if !scalar.is_finite() {
            return Err(MatrixError::NonFiniteScalar { operation: "scale" });
        }

        for i in 0..self.data.len() {
            let value = self.data[i] * scalar;
            if !value.is_finite() {
                return Err(MatrixError::NonFiniteResult { operation: "scale" });
            }
            output.data[i] = value;
        }
        Ok(())
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

impl fmt::Display for Matrix {
    /// Render non-empty matrices by row and empty matrices as one shape token.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A zero-column matrix can have an arbitrarily large logical row count
        // while owning no elements. Formatting one empty row per logical row
        // would turn a constant-size value into an effectively unbounded write,
        // so every zero-axis shape receives one explicit constant-size form.
        if self.rows == 0 || self.cols == 0 {
            return write!(f, "Matrix({}x{})", self.rows, self.cols);
        }

        // Keep the established row-oriented layout for physically populated
        // matrices; all coordinates are valid under the constructor invariant.
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

    /// Owned test view pairing a caller output buffer with returned diagnostics.
    struct TestLeastSquaresSolution {
        /// Solution buffer populated by the allocation-free public operation.
        solution: Matrix,
        /// Rank and condition metadata returned independently from storage.
        info: LeastSquaresInfo,
    }

    /// Run the canonical least-squares API with freshly constructed test storage.
    fn solve_default(coefficients: &Matrix, rhs: &Matrix) -> TestLeastSquaresSolution {
        let mut solution = Matrix::new(coefficients.cols(), 1);
        let mut workspace = LeastSquaresWorkspace::new(coefficients.rows(), coefficients.cols());
        let info = coefficients
            .solve_least_squares_into(rhs, &mut solution, &mut workspace)
            .expect("least-squares solve failed");
        TestLeastSquaresSolution { solution, info }
    }

    /// Return the structured failure from one checked least-squares invocation.
    fn solve_error(coefficients: &Matrix, rhs: &Matrix) -> MatrixError {
        let mut solution = Matrix::new(coefficients.cols(), 1);
        let mut workspace = LeastSquaresWorkspace::new(coefficients.rows(), coefficients.cols());
        coefficients
            .solve_least_squares_into(rhs, &mut solution, &mut workspace)
            .expect_err("least-squares fixture must fail")
    }

    /// Report retained projection geometry and residual slope from the same
    /// singular subspace used to construct the pseudoinverse solution.
    #[test]
    fn least_squares_reports_retained_projection_and_residual_slope() {
        // Only the first coordinate belongs to this rank-one matrix's column
        // space. For b=[3,4], ||P*b||^2/||b|| is 9/5.
        let coefficients = Matrix::from_vec(vec![1.0, 0.0, 0.0, 0.0], 2, 2)
            .expect("the rank-one fixture has four entries");
        let rhs =
            Matrix::from_vec(vec![3.0, 4.0], 2, 1).expect("the right-hand side has two entries");
        let result = solve_default(&coefficients, &rhs);

        assert_eq!(result.info.rank, 1);
        assert!((result.info.retained_rhs_projection_fraction - 0.6).abs() < 1e-12);
        assert!((result.info.residual_norm_slope + 1.8).abs() < 1e-12);

        // A true wide matrix follows the transposed SVD path. It must report the
        // same projection measures as the equivalent tall residual subspace.
        let wide_coefficients = Matrix::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3)
            .expect("the wide rank-one fixture has six entries");
        let wide_result = solve_default(&wide_coefficients, &rhs);
        assert_eq!(wide_result.info.rank, 1);
        assert!((wide_result.info.retained_rhs_projection_fraction - 0.6).abs() < 1e-12);
        assert!((wide_result.info.residual_norm_slope + 1.8).abs() < 1e-12);

        // A non-axis-aligned column protects the projection identity used by
        // every effective correction system. For A=[1,sqrt(3)]^T and b=[2,0],
        // the normalized retained projection has fraction 1/2 and slope -1/2.
        let angled_coefficients = Matrix::from_vec(vec![1.0, 3.0_f64.sqrt()], 2, 1)
            .expect("the angled single-column fixture has two entries");
        let angled_rhs = Matrix::from_vec(vec![2.0, 0.0], 2, 1)
            .expect("the angled right-hand side has two entries");
        let angled_result = solve_default(&angled_coefficients, &angled_rhs);
        assert!((angled_result.info.retained_rhs_projection_fraction - 0.5).abs() < 1e-12);
        assert!((angled_result.info.residual_norm_slope + 0.5).abs() < 1e-12);

        // A zero right-hand side has no direction and therefore reports the
        // documented zero fraction and slope instead of undefined quotients.
        let zero_rhs = Matrix::new(2, 1);
        let zero_result = solve_default(&coefficients, &zero_rhs);
        assert_eq!(zero_result.info.retained_rhs_projection_fraction, 0.0);
        assert_eq!(zero_result.info.residual_norm_slope, 0.0);
        assert!(zero_result.info.residual_norm_slope.is_sign_negative());

        // The dimensionless projection ratio here is 1e-200. Squaring it first
        // underflows, but ||P*b||^2/||b|| is the representable value 1e-100 once
        // the right-hand side's 1e300 physical scale participates.
        let narrow_coefficients = Matrix::from_vec(vec![1.0, 0.0], 2, 1)
            .expect("the single-column fixture has two entries");
        let wide_scale_rhs = Matrix::from_vec(vec![1e100, 1e300], 2, 1)
            .expect("the wide-scale right-hand side has two entries");
        let wide_scale_result = solve_default(&narrow_coefficients, &wide_scale_rhs);
        let fraction_relative_error =
            (wide_scale_result.info.retained_rhs_projection_fraction / 1e-200 - 1.0).abs();
        let relative_error = (wide_scale_result.info.residual_norm_slope / -1e-100 - 1.0).abs();
        assert!(
            fraction_relative_error < 1e-12,
            "projection-fraction relative error: {fraction_relative_error:e}"
        );
        assert!(relative_error < 1e-12, "relative error: {relative_error:e}");

        // Keep model stationarity separate from physical-slope representability.
        // This retained component is visible in normalized coordinates even
        // though its completed physical slope rounds to negative zero.
        let underflow_scale_rhs = Matrix::from_vec(vec![1e-300, 1.0], 2, 1)
            .expect("the underflow-scale right-hand side has two entries");
        let underflow_scale_result = solve_default(&narrow_coefficients, &underflow_scale_rhs);
        assert!(underflow_scale_result.info.retained_rhs_projection_fraction > 0.0);
        assert_eq!(underflow_scale_result.info.residual_norm_slope, 0.0);
        assert!(
            underflow_scale_result
                .info
                .residual_norm_slope
                .is_sign_negative()
        );

        // Zero in the fraction field is a semantic stationarity result, so a
        // retained projection smaller than the fraction's representable range
        // saturates at the least positive subnormal instead of becoming zero.
        let least_positive = f64::from_bits(1);
        let saturation_coefficients = Matrix::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 1)
            .expect("the six-row single-column fixture has six entries");
        let saturation_rhs = Matrix::from_vec(vec![least_positive, 1.0, 1.0, 1.0, 1.0, 1.0], 6, 1)
            .expect("the fraction-saturation right-hand side has six entries");
        let saturation_result = solve_default(&saturation_coefficients, &saturation_rhs);
        assert_eq!(
            saturation_result.info.retained_rhs_projection_fraction,
            least_positive
        );
        assert_eq!(saturation_result.info.residual_norm_slope, 0.0);
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
            solve_error(&identity, &nan_rhs),
            MatrixError::NonFiniteInput {
                operation: "solve_least_squares",
                operand: MatrixOperand::RightHandSide,
            }
        );

        let nan_coefficient = Matrix::from_vec(vec![1.0, f64::NAN, 0.0, 1.0], 2, 2).unwrap();
        let finite_rhs = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();
        assert_eq!(
            solve_error(&nan_coefficient, &finite_rhs),
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
            solve_error(&tiny_coefficient, &huge_rhs),
            MatrixError::NonFiniteResult {
                operation: "solve_least_squares",
            }
        );
    }

    /// Reuse the complete factorization, projection, and output storage across
    /// independent right-hand sides without retaining stale numerical state.
    #[test]
    fn test_least_squares_into_reuses_all_allocations() {
        let coefficients = Matrix::from_vec(vec![2.0, 0.0, 0.0, 4.0], 2, 2).unwrap();
        let first_rhs = Matrix::from_vec(vec![2.0, 8.0], 2, 1).unwrap();
        let second_rhs = Matrix::from_vec(vec![-4.0, 4.0], 2, 1).unwrap();
        let mut solution = Matrix::new(2, 1);
        let mut workspace = LeastSquaresWorkspace::new(2, 2);
        let addresses = (
            solution.data.as_ptr(),
            workspace.orthogonal_columns.data.as_ptr(),
            workspace.right_vectors.data.as_ptr(),
            workspace.singular_values.as_ptr(),
            workspace.projections.as_ptr(),
        );

        let first_info = coefficients
            .solve_least_squares_into(&first_rhs, &mut solution, &mut workspace)
            .unwrap();
        assert_eq!(solution, Matrix::from_vec(vec![1.0, 2.0], 2, 1).unwrap());
        assert_eq!(first_info.rank, 2);

        let second_info = coefficients
            .solve_least_squares_into(&second_rhs, &mut solution, &mut workspace)
            .unwrap();
        assert_eq!(solution, Matrix::from_vec(vec![-2.0, 1.0], 2, 1).unwrap());
        assert_eq!(second_info.rank, 2);
        assert_eq!(
            addresses,
            (
                solution.data.as_ptr(),
                workspace.orthogonal_columns.data.as_ptr(),
                workspace.right_vectors.data.as_ptr(),
                workspace.singular_values.as_ptr(),
                workspace.projections.as_ptr(),
            )
        );
    }

    /// Reject output and workspace mismatches with their exact expected and
    /// actual shapes before factorization touches reusable scratch storage.
    #[test]
    fn test_least_squares_into_rejects_mismatched_caller_storage() {
        let coefficients = Matrix::identity(2);
        let rhs = Matrix::new(2, 1);
        let mut wrong_solution = Matrix::new(1, 1);
        let mut matching_workspace = LeastSquaresWorkspace::new(2, 2);
        assert_eq!(
            coefficients
                .solve_least_squares_into(&rhs, &mut wrong_solution, &mut matching_workspace,)
                .unwrap_err(),
            MatrixError::OutputShapeMismatch {
                operation: "solve_least_squares",
                expected: (2, 1),
                actual: (1, 1),
            }
        );

        let mut solution = Matrix::new(2, 1);
        let mut wrong_workspace = LeastSquaresWorkspace::new(3, 2);
        assert_eq!(
            coefficients
                .solve_least_squares_into(&rhs, &mut solution, &mut wrong_workspace)
                .unwrap_err(),
            MatrixError::WorkspaceShapeMismatch {
                operation: "solve_least_squares",
                expected: (2, 2),
                actual: (3, 2),
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

        let mut transposed = Matrix::new(3, 2);
        let allocation = transposed.data.as_ptr();
        m.transpose_into(&mut transposed).unwrap();
        assert_eq!(transposed.data.as_ptr(), allocation);
        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);
        assert_eq!(transposed[(0, 0)], 1.0);
        assert_eq!(transposed[(1, 0)], 2.0);
        assert_eq!(transposed[(2, 0)], 3.0);
        assert_eq!(transposed[(0, 1)], 4.0);
    }

    /// Complete zero-storage operations from physical cardinality rather than
    /// iterating over enormous logical dimensions that contain no coordinates.
    #[test]
    fn test_zero_storage_transpose_and_multiplication_return_immediately() {
        let huge_empty = Matrix::from_vec(Vec::new(), usize::MAX, 0)
            .expect("a zero-column matrix requires no storage");
        let empty_right =
            Matrix::from_vec(Vec::new(), 0, 0).expect("a zero-by-zero matrix is valid");

        let mut transposed = Matrix::from_vec(Vec::new(), 0, usize::MAX).unwrap();
        huge_empty.transpose_into(&mut transposed).unwrap();
        assert_eq!(transposed.rows(), 0);
        assert_eq!(transposed.cols(), usize::MAX);
        assert_eq!(transposed.norm(), 0.0);

        let mut product = Matrix::from_vec(Vec::new(), usize::MAX, 0).unwrap();
        huge_empty
            .mul_into(&empty_right, &mut product)
            .expect("a product with zero output columns is already complete");
        assert_eq!(product.rows(), usize::MAX);
        assert_eq!(product.cols(), 0);
        assert_eq!(product.norm(), 0.0);
    }

    /// Format every empty shape in constant work while retaining both logical
    /// dimensions in the rendered value.
    #[test]
    fn test_display_is_bounded_and_shape_explicit_for_empty_matrices() {
        // Zero storage does not imply zero logical dimensions. In particular,
        // the maximum-axis cases must not iterate their enormous empty ranges.
        let empty_shapes = [
            (0, 0, "Matrix(0x0)".to_string()),
            (0, 3, "Matrix(0x3)".to_string()),
            (3, 0, "Matrix(3x0)".to_string()),
            (usize::MAX, 0, format!("Matrix({}x0)", usize::MAX)),
            (0, usize::MAX, format!("Matrix(0x{})", usize::MAX)),
        ];
        for (rows, cols, expected) in empty_shapes {
            let matrix = Matrix::from_vec(Vec::new(), rows, cols)
                .expect("a zero-axis shape requires no element storage");
            assert_eq!(matrix.to_string(), expected);
        }

        // The bounded empty branch must not change the established row layout
        // used for ordinary numerical debugging output.
        let populated = Matrix::from_vec(vec![1.0, -2.0], 1, 2)
            .expect("two values exactly fill a one-by-two matrix");
        assert_eq!(populated.to_string(), "[  1.0000,  -2.0000]\n");
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

        let mut product = Matrix::new(2, 2);
        let allocation = product.data.as_ptr();
        a.mul_into(&b, &mut product).unwrap();
        assert_eq!(product.data.as_ptr(), allocation);
        assert_eq!(product.rows(), 2);
        assert_eq!(product.cols(), 2);
        assert_eq!(product[(0, 0)], 58.0);
        assert_eq!(product[(0, 1)], 64.0);
        assert_eq!(product[(1, 0)], 139.0);
        assert_eq!(product[(1, 1)], 154.0);
    }

    #[test]
    fn test_add_into_dimension_mismatch() {
        let a = Matrix::new(2, 2);
        let b = Matrix::new(2, 3);
        let mut output = Matrix::new(2, 2);

        let err = a
            .add_into(&b, &mut output)
            .expect_err("expected dimension mismatch");
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
    fn test_sub_into_dimension_mismatch() {
        let a = Matrix::new(3, 2);
        let b = Matrix::new(2, 2);
        let mut output = Matrix::new(3, 2);

        let err = a
            .sub_into(&b, &mut output)
            .expect_err("expected dimension mismatch");
        assert_eq!(
            err,
            MatrixError::DimensionMismatch {
                operation: "sub",
                left: (3, 2),
                right: (2, 2),
            }
        );
    }

    /// Exercise successful explicit elementwise arithmetic instead of covering
    /// only the recoverable shape-error branches.
    #[test]
    fn test_add_subtract_and_scale_into_reuse_output_storage() {
        let left = Matrix::from_vec(vec![1.0, -2.0, 3.5, 4.0], 2, 2)
            .expect("left fixture has four elements");
        let right = Matrix::from_vec(vec![0.5, 2.0, -1.5, 6.0], 2, 2)
            .expect("right fixture has four elements");

        let mut output = Matrix::new(2, 2);
        let allocation = output.data.as_ptr();
        left.add_into(&right, &mut output)
            .expect("matching shapes must add");

        assert_eq!(
            output,
            Matrix::from_vec(vec![1.5, 0.0, 2.0, 10.0], 2, 2).unwrap()
        );
        left.sub_into(&right, &mut output)
            .expect("matching shapes must subtract");
        assert_eq!(
            output,
            Matrix::from_vec(vec![0.5, -4.0, 5.0, -2.0], 2, 2).unwrap()
        );
        left.scale_into(2.0, &mut output)
            .expect("finite products must be written");
        assert_eq!(
            output,
            Matrix::from_vec(vec![2.0, -4.0, 7.0, 8.0], 2, 2).unwrap()
        );
        assert_eq!(output.data.as_ptr(), allocation);
    }

    #[test]
    fn test_mul_into_dimension_mismatch() {
        let a = Matrix::new(2, 3);
        let b = Matrix::new(2, 2);
        let mut output = Matrix::new(2, 2);

        let error = a
            .mul_into(&b, &mut output)
            .expect_err("expected dimension mismatch");
        assert_eq!(
            error,
            MatrixError::DimensionMismatch {
                operation: "mul",
                left: (2, 3),
                right: (2, 2),
            }
        );
    }

    /// Reject every invalid public arithmetic buffer shape before performing a
    /// write, including the independently derived multiplication and transpose
    /// shapes.
    #[test]
    fn test_into_operations_reject_output_shape_mismatches_without_writing() {
        let left = Matrix::new(2, 2);
        let right = Matrix::new(2, 2);
        let mut wrong = Matrix::from_vec(vec![7.0, 7.0], 2, 1).unwrap();

        for (operation, error) in [
            ("add", left.add_into(&right, &mut wrong).unwrap_err()),
            ("sub", left.sub_into(&right, &mut wrong).unwrap_err()),
            ("mul", left.mul_into(&right, &mut wrong).unwrap_err()),
            ("scale", left.scale_into(2.0, &mut wrong).unwrap_err()),
            ("transpose", left.transpose_into(&mut wrong).unwrap_err()),
        ] {
            assert_eq!(
                error,
                MatrixError::OutputShapeMismatch {
                    operation,
                    expected: (2, 2),
                    actual: (2, 1),
                }
            );
            assert_eq!(wrong, Matrix::from_vec(vec![7.0, 7.0], 2, 1).unwrap());
        }
    }

    /// Establish one profile-independent finite boundary for matrix arithmetic.
    /// These are the special values that previously reached the internal scaled
    /// accumulator and changed behavior between debug and release builds.
    #[test]
    fn test_arithmetic_rejects_non_finite_operands_before_writing_output() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid_matrix = Matrix::from_vec(vec![invalid], 1, 1).unwrap();
            let finite_matrix = Matrix::from_vec(vec![1.0], 1, 1).unwrap();
            let zero_matrix = Matrix::from_vec(vec![0.0], 1, 1).unwrap();
            let mut output = Matrix::from_vec(vec![42.0], 1, 1).unwrap();

            assert_eq!(
                invalid_matrix
                    .mul_into(&zero_matrix, &mut output)
                    .unwrap_err(),
                MatrixError::NonFiniteInput {
                    operation: "mul",
                    operand: MatrixOperand::LeftHandSide,
                }
            );
            assert_eq!(output[(0, 0)], 42.0);

            assert_eq!(
                finite_matrix
                    .mul_into(&invalid_matrix, &mut output)
                    .unwrap_err(),
                MatrixError::NonFiniteInput {
                    operation: "mul",
                    operand: MatrixOperand::RightHandSide,
                }
            );
            assert_eq!(output[(0, 0)], 42.0);

            assert_eq!(
                invalid_matrix
                    .add_into(&finite_matrix, &mut output)
                    .unwrap_err(),
                MatrixError::NonFiniteInput {
                    operation: "add",
                    operand: MatrixOperand::LeftHandSide,
                }
            );
            assert_eq!(output[(0, 0)], 42.0);

            assert_eq!(
                finite_matrix
                    .sub_into(&invalid_matrix, &mut output)
                    .unwrap_err(),
                MatrixError::NonFiniteInput {
                    operation: "sub",
                    operand: MatrixOperand::RightHandSide,
                }
            );
            assert_eq!(output[(0, 0)], 42.0);

            assert_eq!(
                invalid_matrix.scale_into(1.0, &mut output).unwrap_err(),
                MatrixError::NonFiniteInput {
                    operation: "scale",
                    operand: MatrixOperand::LeftHandSide,
                }
            );
            assert_eq!(output[(0, 0)], 42.0);
        }
    }

    /// Distinguish invalid inputs from finite arithmetic whose completed `f64`
    /// result lies outside the representable range.
    #[test]
    fn test_arithmetic_reports_non_finite_completed_results() {
        let maximum = Matrix::from_vec(vec![f64::MAX], 1, 1).unwrap();
        let negative_maximum = Matrix::from_vec(vec![-f64::MAX], 1, 1).unwrap();
        let two = Matrix::from_vec(vec![2.0], 1, 1).unwrap();
        let mut output = Matrix::new(1, 1);

        assert_eq!(
            maximum.add_into(&maximum, &mut output).unwrap_err(),
            MatrixError::NonFiniteResult { operation: "add" }
        );
        assert_eq!(
            maximum
                .sub_into(&negative_maximum, &mut output)
                .unwrap_err(),
            MatrixError::NonFiniteResult { operation: "sub" }
        );
        assert_eq!(
            maximum.mul_into(&two, &mut output).unwrap_err(),
            MatrixError::NonFiniteResult { operation: "mul" }
        );
        assert_eq!(
            maximum.scale_into(2.0, &mut output).unwrap_err(),
            MatrixError::NonFiniteResult { operation: "scale" }
        );
        assert_eq!(
            maximum.scale_into(f64::NAN, &mut output).unwrap_err(),
            MatrixError::NonFiniteScalar { operation: "scale" }
        );

        // General multiplication is intentionally ordinary `f64` arithmetic,
        // not a hidden arbitrary-range number system. Although these two exact
        // products cancel mathematically, each product first exceeds the scalar
        // range, so the checked operation asks its caller to rescale or retry.
        let cancelling_left = Matrix::from_vec(vec![f64::MAX, f64::MAX], 1, 2).unwrap();
        let cancelling_right = Matrix::from_vec(vec![2.0, -2.0], 2, 1).unwrap();
        assert_eq!(
            cancelling_left
                .mul_into(&cancelling_right, &mut output)
                .unwrap_err(),
            MatrixError::NonFiniteResult { operation: "mul" }
        );
    }

    /// Transposition is a bitwise data movement operation rather than
    /// arithmetic, so it deliberately preserves non-finite diagnostic values.
    #[test]
    fn test_transpose_into_preserves_ieee_special_values() {
        let input = Matrix::from_vec(vec![f64::NAN, f64::INFINITY], 1, 2).unwrap();
        let mut output = Matrix::new(2, 1);

        input.transpose_into(&mut output).unwrap();

        assert!(output[(0, 0)].is_nan());
        assert_eq!(output[(1, 0)], f64::INFINITY);
    }

    /// Preserve a finite least-squares solution when same-sign projection terms
    /// overflow in source order before later terms cancel them.
    #[test]
    fn test_solve_least_squares_scales_large_cancelling_projection() {
        let coefficients = Matrix::from_vec(vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0], 6, 1)
            .expect("the coefficient fixture has six rows");
        let right_hand_side = Matrix::from_vec(
            vec![f64::MAX, f64::MAX, f64::MAX, f64::MAX, f64::MAX, 0.0],
            6,
            1,
        )
        .expect("the right-hand-side fixture has six rows");

        // A^T*b is exactly MAX after three positive and two negative MAX terms
        // cancel, while A^T*A is six. The old source-order partial sum overflowed
        // on its third positive term and rejected this representable result.
        let result = solve_default(&coefficients, &right_hand_side);
        let expected = f64::MAX / 6.0;
        let relative_error = (result.solution[(0, 0)] / expected - 1.0).abs();

        assert_eq!(result.info.rank, 1);
        assert!(relative_error < 1e-15, "relative error: {relative_error}");
    }

    /// Reject reconstruction whose ordinary component-order partial total
    /// exceeds `f64` before a later singular direction could cancel it.
    #[test]
    fn test_solve_least_squares_rejects_cancelling_singular_reconstruction_overflow() {
        let coefficients = Matrix::from_vec(
            vec![
                -7.565e-309,
                6.027e-309,
                2.538e-309,
                -7.022e-309,
                -5.405e-309,
                -8.092e-309,
                -4.089e-309,
                -9.221e-309,
                9.708e-309,
            ],
            3,
            3,
        )
        .expect("the coefficient fixture is square");
        let right_hand_side = Matrix::from_vec(vec![-1.3260, -1.6941, 2.1713], 3, 1)
            .expect("the right-hand-side fixture has three rows");

        // Two same-sign singular contributions exceed `f64::MAX` before a later
        // direction would cancel them. Although the analytical final vector is
        // finite, supporting that cancellation requires a hidden wider-range
        // accumulator, so the checked solver now returns its documented range
        // failure instead.
        assert_eq!(
            solve_error(&coefficients, &right_hand_side),
            MatrixError::NonFiniteResult {
                operation: "solve_least_squares",
            }
        );
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

    /// Keep a clearly retained tall direction across two representable unit
    /// scales far from the numerical-rank cutoff.
    #[test]
    fn test_solve_least_squares_tall_rank_matches_away_from_cutoff() {
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

    /// Retain a tiny wide-system direction that remains clearly above the
    /// scale-relative numerical-rank cutoff.
    #[test]
    fn test_solve_least_squares_retains_tiny_wide_direction_above_cutoff() {
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

    /// Keep the historical variable-ordering regression fixture consistent
    /// without promoting one observed case into a universal cutoff guarantee.
    #[test]
    fn test_solve_least_squares_matches_column_permutation_fixture() {
        // The tiny determinant makes this matrix numerically rank one under the
        // precision-derived SVD threshold. The former unpivoted QR dispatcher
        // classified the original ordering as rank two but the permutation as
        // rank one, so equivalent variable orderings followed different solve
        // policies and produced unrelated answers. This fixture checks that
        // removed dispatcher bug; singular values at the current cutoff remain
        // subject to ordinary floating-point rounding.
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
            let mut b = Matrix::new(6, 1);
            a.mul_into(&x_mat, &mut b).unwrap();

            let result = solve_default(&a, &b);
            assert_eq!(result.info.rank, 3);
            assert!(result.info.cond_est.is_finite());
            for (row, expected) in x_true.iter().enumerate() {
                assert!((result.solution[(row, 0)] - expected).abs() < 1e-8);
            }
        }
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

    /// Match one exactly rank-deficient fixture across very small and very large
    /// finite unit scales.
    #[test]
    fn test_solve_least_squares_matches_rank_deficient_scale_fixture() {
        for scale in [1e-200_f64, 1.0, 1e200] {
            let a = Matrix::from_vec(vec![scale, scale, 2.0 * scale, 2.0 * scale], 2, 2).unwrap();
            let b = Matrix::from_vec(vec![scale, 2.0 * scale], 2, 1).unwrap();

            let result = solve_default(&a, &b);

            // In exact arithmetic, scaling both operands by the same non-zero
            // value leaves the Moore-Penrose answer unchanged. This fixture
            // verifies that practical range handling retains that answer at
            // magnitudes where naïve squared norms would underflow or overflow.
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
        let mut reconstructed = Matrix::new(3, 1);
        let mut residual = Matrix::new(3, 1);
        a.mul_into(&result.solution, &mut reconstructed).unwrap();
        reconstructed.sub_into(&b, &mut residual).unwrap();
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
        let mut reconstructed = Matrix::new(2, 1);
        let mut residual = Matrix::new(2, 1);
        a.mul_into(&result.solution, &mut reconstructed).unwrap();
        reconstructed.sub_into(&b, &mut residual).unwrap();
        assert!(residual.norm() < 1e-12);
    }

    /// Solve the exact rank-deficient Jacobian produced by the graphical
    /// four-link IK example when its initial chain is fully extended.
    #[test]
    fn test_solve_least_squares_handles_straight_four_link_ik_jacobian() {
        let coefficients = Matrix::from_vec(
            vec![
                280.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // link 1
                -240.0, 0.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, // link 2
                0.0, 0.0, -200.0, 0.0, 200.0, 0.0, 0.0, 0.0, // link 3
                0.0, 0.0, 0.0, 0.0, -160.0, 0.0, 160.0, 0.0, // link 4
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // target x
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // target y
            ],
            6,
            8,
        )
        .expect("the IK Jacobian fixture has six rows and eight columns");
        let right_hand_side = Matrix::from_vec(vec![0.0, 0.0, 0.0, 0.0, -300.0, 200.0], 6, 1)
            .expect("the IK residual fixture has six entries");

        let result = solve_default(&coefficients, &right_hand_side);

        assert_eq!(result.info.rank, 5);
        assert!(result.solution.all_finite());
    }

    /// Verify that buffer construction reports an ordinary length mismatch with
    /// both the requested shape and actual storage length.
    #[test]
    fn test_from_vec_reports_data_length_mismatch() {
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
