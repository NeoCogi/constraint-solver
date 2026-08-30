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

//! Symbolic construction and numerical solution of small dense nonlinear
//! constraint systems.
//!
//! Build residual expressions with [`Exp`], compile their variable names once
//! with [`Compiler`], and solve square, underdetermined, or overdetermined
//! systems with [`NewtonRaphsonSolver`]. Rectangular linearizations use QR for
//! full-rank systems and a Jacobi-SVD pseudoinverse for rank-deficient systems.
//!
//! The solver differentiates the supplied expression tree symbolically. It
//! does not estimate or repair derivatives with finite differences. At a
//! domain boundary, a finite residual can therefore have a non-finite symbolic
//! derivative when the chosen expression contains an IEEE-754 indeterminate
//! form such as `0 * infinity`. In that case solving returns
//! [`SolverError::NonFiniteEvaluation`]; callers should start inside the
//! expression's differentiable domain or rewrite the expression into a form
//! whose derivative is finite at the intended starting point.
//!
//! # Quick start
//!
//! ```
//! use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Residual zero means x = 2. Arithmetic operators build expression nodes.
//! let equation = Exp::var("x") - 2.0;
//! let compiled = Compiler::compile(&[equation])?;
//! let solver = NewtonRaphsonSolver::new(compiled);
//! let initial = HashMap::from([("x".to_string(), 0.0)]);
//!
//! let solution = solver.solve(initial)?;
//! assert!(solution.error < 1e-10);
//! assert!((solution.values["x"] - 2.0).abs() < 1e-10);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod compiler;
pub mod exp;
mod jacobian;
pub mod matrix;
pub mod mode;
pub mod solver;

pub use compiler::{CompileError, CompiledSystem, Compiler};
pub use exp::Exp;
pub use matrix::{
    LeastSquaresInfo, LeastSquaresMethod, LeastSquaresSolution, Matrix, MatrixError, MatrixOperand,
};
pub use mode::Mode;
pub use solver::{
    EquationDiagnostic, EquationTrace, LinearSolveDiagnostic, NewtonRaphsonSolver, Solution,
    SolverError, SolverOptions, SolverRunDiagnostic,
};

/// Documentation-only container that makes every Rust block in the repository
/// README part of `cargo test --doc`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {
    // The included Markdown supplies this module's doctests. Keeping the module
    // empty avoids shipping duplicate documentation or executable code in
    // ordinary library builds.
}
