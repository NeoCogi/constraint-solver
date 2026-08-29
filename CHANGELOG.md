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
- README Rust examples compiled and executed by the documentation-test suite so
  published usage snippets cannot drift away from the public API.
- Continuous integration gates for RustSec advisories, formatting, clippy,
  unit and example tests, documentation tests, benchmark compilation, and crate
  packaging.

### Changed

- Replaced the QR-named public least-squares methods and diagnostics with the
  factorization-independent `solve_least_squares`, `LeastSquaresSolution`, and
  `LeastSquaresMethod` API. Backward compatibility with the pre-0.3 API is not
  retained.
- Restricted the compiled-expression `Jacobian` implementation to crate scope;
  its module, constructor inputs, and evaluation methods were already private,
  so its nominally public declarations could not form a usable external API.
- Removed the unreachable public `MatrixError::RankDeficient` variant. QR rank
  loss is now an explicitly private dispatch outcome that selects the SVD
  pseudoinverse; it was never returned by public least-squares operations.
- Changed line search to a slope-based Armijo condition for the residual norm,
  using `(f^T J delta) / ||f||` as its directional derivative, and kept its
  configured initial trial size authoritative.
- Made minimum-norm-point projection a separate accepted update so damping does
  not reintroduce an initial Jacobian-null-space component.
- Based parallel multiplication scheduling on full scalar work, including the
  shared inner dimension, and allowed wide QR transposition to use the selected
  parallel mode.
- Made the Criterion benchmark an explicit `benchmarks` feature so ordinary
  all-target test runs never execute performance measurements.
- Declared Rust 1.86 as the minimum supported toolchain for Edition 2024 and
  the locked development dependency graph.

### Fixed

- Corrected diagnostic iteration counts so rejected line-search candidates and
  pre-update failures do not masquerade as accepted Newton updates.
- Prevented inconsistent overdetermined problems from being reported as stalled
  solely because their optimal residual is nonzero.
- Prevented absolute equation scaling and zero-Jacobian non-root points from
  being accepted as least-squares stationary solutions by using a normalized
  first-order gradient measure.
- Prevented damping adaptation from reacting to artificial residual-history
  sentinels or treating deliberately damped progress as stagnation.
- Corrected tolerance documentation to cover residual and normalized-gradient
  termination only; deliberately damped small updates are not convergence.
- Rejected NaN, infinity, and finite-input arithmetic overflow at matrix solve
  boundaries instead of returning invalid successful matrices.
- Updated locked `crossbeam-epoch` from 0.9.18 to 0.9.20 to resolve
  RUSTSEC-2026-0204 in the repository's development and release test graph.
- Normalized Householder reflector construction before subtracting its leading
  value, preventing large finite full-rank systems from returning inaccurate QR
  solutions after an intermediate sum overflowed.
- Routed square Jacobians that fail LU through the same QR/SVD pseudoinverse and
  augmented ridge fallback as rectangular systems, replacing the unrelated
  direct diagonal shift and allowing consistent rank-deficient square systems
  to solve without regularization.
- Cached simplified symbolic Jacobian expressions when constructing a solver,
  so repeated solve calls allocate fresh numerical workspaces without cloning
  and differentiating the entire equation system again.
- Checked matrix element products, including multiplication output shapes
  derived from independently valid zero-sized operands, and augmented row sums
  before allocation.
- Removed redundant initial-variable and Jacobian copies, unused Householder
  storage, and repeated quadratic clearing of immutable regularization zeros.
- Made the GLFW inverse-kinematics example report changed solver failures,
  announce recovery, synchronize buffer swaps, and release OpenGL objects before
  context destruction.

[0.3.0]: https://github.com/NeoCogi/constraint-solver/releases/tag/v0.3.0
