# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-31

### Added

- Added one-sided Jacobi-SVD least-squares solving through
  `Matrix::solve_least_squares_into` for square, tall, and wide systems. The
  caller supplies `LeastSquaresWorkspace`, and `LeastSquaresInfo` reports the
  retained rank, condition estimate, right-hand-side projection fraction, and
  residual-norm slope.
- Added caller-owned `SolverWorkspace` storage and made the public `solve` API
  use it explicitly. Residual, Jacobian, candidate, correction, and SVD buffers
  are allocated once and reused across iterations and sequential solves.
- Added `SolverOptions` with the three current policy fields:
  `max_iterations`, `residual_tolerance`, and `max_backtracks`.
- Added positive equation and variable characteristic scales. Equation scales
  define residual weighting and success; variable scales define the
  minimum-norm correction metric in normalized coordinates before corrections
  are converted back to caller units.
- Added compiled-system metadata, optional equation traces, raw and scaled
  per-equation failure diagnostics, and direct `LeastSquaresInfo` attempt
  provenance on successful and failed runs.
- Added structured matrix and solver errors for recoverable shape, finite-state,
  factorization, and nonlinear-run failures. Impossible `Matrix` shapes and
  `Matrix` storage requests remain explicit fatal panics.
- Added arithmetic operators for owned and borrowed `Exp` values and `f64`
  constants, plus quoted and escaped expression-variable diagnostics.
- Added executable crate documentation, compiled README examples, complete
  public API documentation, an explicit package allowlist, and a `missing_docs`
  lint.
- Added 100-sample KCL-based electronics and geometric continuation fixtures,
  five headless inverse-kinematics model tests, and self-checking CAD and circuit
  examples.
- Added locked CI gates for Rust 1.86 on Linux, Windows, and macOS; RustSec;
  formatting; all-feature Clippy; debug and optimized tests; the optional GLFW
  model; headless examples; doctests and warning-free public docs; benchmark
  fixtures; and package verification.

### Changed

- Replaced the former LU/QR dispatcher, damping and ridge paths, and overlapping
  least-squares entry points with one normalized Jacobi-SVD pseudoinverse path.
  Rank and condition are diagnostics and do not select another equation or
  factorization policy.
- Made nonlinear success strict and shape-independent: `Ok(Solution)` certifies
  only that the equation-scaled residual norm is at or below
  `residual_tolerance`. A non-root with no representable right-hand-side
  projection into the retained SVD column space returns
  `SolverError::StationaryNonRoot`.
- Derived the nonlinear correction, retained-model reducibility decision, and
  Armijo residual-norm slope from the same SVD attempt. Gradient norms remain
  descriptive diagnostics rather than an independently tuned termination rule.
- Replaced competing nonlinear update modes with one transactional Armijo
  policy: test the full correction, then permit at most `max_backtracks`
  deterministic halvings. Rejected candidates never mutate the accepted state
  or consume an accepted-update count.
- Removed `Mode`, dedicated Rayon pools, parallel matrix entry points, and
  mode-specific constructors. Solver arithmetic now has one serial traversal;
  applications can still run independent solves on separate threads with
  distinct workspaces.
- Replaced allocating matrix arithmetic and infallible operator traits with
  checked `add_into`, `sub_into`, `mul_into`, `scale_into`, and
  `transpose_into` operations over exact-shape caller buffers.
- Removed unbounded convergence history and alternate solver entry points.
  Final residual, gradient, accepted-update count, equation state, and last SVD
  attempt remain available as concrete diagnostics.
- Replaced the caller-selected rank cutoff with one spectrum-relative policy.
  With `factor = f64::EPSILON * max(rows, columns)`, a direction is retained only
  when `sigma_i > sigma_max * factor`. The reported rank is a floating-point
  diagnostic: column permutations preserve the exact singular spectrum and
  uniform coefficient scaling multiplies it by one common factor, but rounding
  can change classification near the cutoff.
- Declared Rust 1.86 as the MSRV, made Criterion benchmarks and the native GLFW
  example opt-in features, and removed machine-specific benchmark timings from
  the README.

### Fixed

- Prevented avoidable intermediate overflow or underflow caused by association
  in scale-heavy Jacobian, update, slope, and individual SVD reconstruction
  products through binary-exponent factor formation. SVD reconstruction then
  uses ordinary compensated `f64` across components and rejects a non-finite
  contribution or partial total even if later cancellation could recover a
  finite mathematical value.
- Preserved typed `MatrixError` causes and every already-computed gradient and
  SVD diagnostic at solver failure boundaries without recomputing through an
  invalid state.
- Included fixed parameters in reusable solver-workspace compatibility, so a
  same-Jacobian-shape workspace cannot silently grow its variable maps.
- Corrected accepted-update accounting and state reuse so returned values,
  residuals, gradients, traces, and attempt provenance describe the same
  accepted state.
- Checked matrix axes and element counts before flattening or allocation, and
  made zero-storage transpose, multiplication, and display complete from
  physical storage rather than enormous empty logical ranges.
- Rejected non-finite completed residuals and Jacobians, non-finite solver
  updates, invalid checked add/subtract/multiply/scale operands or results, and
  invalid factorization inputs or results. Expression intermediates retain their
  direct written-order IEEE-754 behavior.
- Documented exact symbolic differentiation at domain boundaries and the lack of
  a hidden finite-difference fallback. Expression display now quotes every
  variable name and escapes controls instead of allowing names to impersonate
  expression syntax.
- Updated the locked `crossbeam-epoch` dependency to resolve
  RUSTSEC-2026-0204.

## [0.1.1] - 2026-02-12

### Added

- Added runtime serial and Rayon-backed parallel execution modes with numerical
  equivalence tests and Criterion matrix benchmarks.
- Added the interactive GLFW/OpenGL two-dimensional inverse-kinematics example.

### Changed

- Updated the graphics, math, and benchmarking development dependencies and
  added repository and crates.io package metadata.

## [0.1.0] - 2026-02-01

### Added

- Introduced symbolic residual expressions, name-based compilation, cached
  symbolic Jacobians, and checked numerical evaluation.
- Introduced dense matrix arithmetic with LU and QR least-squares solving.
- Introduced damped Newton-Raphson solving for square, underdetermined, and
  overdetermined systems with optional line search and adaptive regularization.
- Added CAD and circuit command-line examples and the initial unit-test suite.

[Unreleased]: https://github.com/NeoCogi/constraint-solver/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/NeoCogi/constraint-solver/releases/tag/v0.3.0
[0.1.1]: https://github.com/NeoCogi/constraint-solver/tree/47403f77f99ed122bc9a052f4a4a53f071bdd33e
[0.1.0]: https://github.com/NeoCogi/constraint-solver/tree/f95a5e74d24d76a1b08464b078b91318ee2e2331
