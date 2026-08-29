# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-29

### Added

- One-sided Jacobi SVD fallback for rank-deficient square, tall, and wide
  least-squares systems, including minimum-norm Moore-Penrose solutions and
  retained-subspace rank and condition diagnostics.
- Structured matrix errors for overflowing derived dimensions, non-finite
  operands and results, and iterative-factorization non-convergence.
- Explicit least-squares stationarity as a successful convergence reason for
  inconsistent overdetermined systems.
- Crate-level executable documentation, complete public API documentation, and
  an enforced missing-documentation lint.
- Continuous integration gates for formatting, clippy, unit and example tests,
  documentation tests, benchmark compilation, and crate packaging.

### Changed

- Replaced the QR-named public least-squares methods and diagnostics with the
  factorization-independent `solve_least_squares*`, `LeastSquaresInfo`, and
  `LeastSquaresMethod` API. Backward compatibility with the pre-0.3 API is not
  retained.
- Changed line search to a slope-based Armijo condition for the squared-residual
  objective and kept its configured initial trial size authoritative.
- Made minimum-norm-point projection a separate accepted update so damping does
  not reintroduce an initial Jacobian-null-space component.
- Based parallel multiplication scheduling on full scalar work, including the
  shared inner dimension, and allowed wide QR transposition to use the selected
  parallel mode.
- Made the Criterion benchmark an explicit `benchmarks` feature so ordinary
  all-target test runs never execute performance measurements.
- Declared Rust 1.85 as the minimum supported toolchain for Edition 2024.

### Fixed

- Corrected diagnostic iteration counts so rejected line-search candidates and
  pre-update failures do not masquerade as accepted Newton updates.
- Prevented inconsistent overdetermined problems from being reported as stalled
  solely because their optimal residual is nonzero.
- Prevented damping adaptation from reacting to artificial residual-history
  sentinels or treating deliberately damped progress as stagnation.
- Rejected NaN, infinity, and finite-input arithmetic overflow at matrix solve
  boundaries instead of returning invalid successful matrices.
- Checked matrix element products and augmented row sums before allocation.
- Removed redundant initial-variable and Jacobian copies, unused Householder
  storage, and repeated quadratic clearing of immutable regularization zeros.
- Made the GLFW inverse-kinematics example report changed solver failures,
  announce recovery, synchronize buffer swaps, and release OpenGL objects before
  context destruction.

[0.3.0]: https://github.com/NeoCogi/constraint-solver/releases/tag/v0.3.0
