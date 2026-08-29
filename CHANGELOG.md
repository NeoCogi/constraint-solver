# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-29

### Added

- Added a one-sided Jacobi SVD fallback for rank-deficient square, tall, and
  wide least-squares systems, including minimum-norm Moore-Penrose solutions.
- Added rank, condition estimate, and QR/SVD method metadata to the canonical
  `LeastSquaresSolution` result.
- Added `LinearSolveDiagnostic` to successful solutions and failed-run
  diagnostics, preserving both the effective factorization and the original
  unregularized classification when ridge regularization is selected.
- Added `SolverOptions` and `LeastSquaresOptions` as the documented sources of
  nonlinear iteration, line-search, regularization, and rank policy.
- Added structured equation diagnostics and optional caller-provided equation
  traces to make failed constraints identifiable without parsing messages.
- Added structured matrix errors for invalid shapes and tolerances, overflowing
  dimensions, allocation failure, non-finite states, and failed convergence.
- Added executable crate documentation, compiled README examples, complete
  public API documentation, and the `missing_docs` lint.
- Added CI gates for Rust 1.86, current stable Rust, Windows and macOS library
  builds, RustSec advisories, formatting, Clippy, tests, headless examples,
  documentation tests, benchmark compilation, and package verification.

### Changed

- Made success strict and shape-independent: `Ok(Solution)` now certifies only
  `residual_norm <= residual_tolerance`; every stationary non-root returns
  `SolverError::StationaryNonRoot`, including square and underdetermined cases.
- Replaced the overlapping QR-named least-squares entry points with one
  `solve_least_squares` operation returning a diagnostic result. Compatibility
  aliases from the pre-alpha API were intentionally removed.
- Unified nonlinear Jacobian correction policy across all shapes. Full-rank
  systems use Householder QR, rank-deficient systems use the SVD pseudoinverse,
  and ill-conditioned retained subspaces use augmented ridge regularization.
- Centralized numerical defaults and made selected-path validation constant-time
  instead of scanning retry budgets or rejecting inactive numerical policies.
- Changed Armijo line search to operate directly on the residual norm with the
  scale-safe directional derivative `(f / ||f||)^T (J delta)`.
- Removed origin-based point projection from nonlinear iteration; every wide
  system now uses one minimum-norm Newton correction per accepted update.
- Cached simplified symbolic derivatives at solver construction while keeping
  independent reusable numerical workspaces for every solve invocation.
- Restricted the compiled-expression Jacobian to crate scope and removed the
  unreachable public `MatrixError::RankDeficient` variant.
- Based parallel multiplication scheduling on complete scalar work and allowed
  wide least-squares transposition to honor the selected execution mode.
- Made Criterion benchmarks opt-in through the `benchmarks` feature and removed
  machine-specific timing tables from the README.
- Declared Rust 1.86 as the MSRV, refreshed all compatible locked dependencies,
  and required CI builds to use the committed dependency graph.

### Fixed

- Prevented norm, column-scaled stationarity, and line-search slope calculations
  from overflowing or underflowing at extreme finite scales.
- Corrected Householder reflector scaling and triangular condition estimation,
  including off-diagonal growth and uniform matrix rescaling.
- Corrected LU pivot classification so every pivot, including the final one,
  is compared against the complete coefficient-matrix scale.
- Returned `MatrixError::AllocationFailed` from fallible construction instead
  of panicking when an otherwise valid shape cannot reserve storage.
- Stopped preallocating convergence history from an untrusted iteration limit;
  history now grows only as accepted updates are observed.
- Checked all derived matrix element counts and augmented-system row counts
  before allocation, including products involving zero-sized operands.
- Rejected NaN, infinity, and finite-input arithmetic overflow at expression,
  Jacobian, update, and matrix-solve boundaries.
- Corrected accepted-iteration accounting and refreshed residuals after every
  update so returned values, errors, histories, and diagnostics describe the
  same numerical state.
- Prevented damping adaptation from reacting to artificial history sentinels or
  treating deliberately damped progress as convergence.
- Made line-search trial limits and initial step sizes authoritative and stopped
  rejected candidates from being counted as accepted solver updates.
- Documented exact symbolic differentiation at domain boundaries and added a
  regression distinguishing a non-finite written derivative from a safe
  algebraic rewrite; no hidden finite-difference fallback is performed.
- Made every headless example assert its known numerical outcome and removed a
  duplicated legacy matrix-test module with stale nonlinear-LU assumptions.
- Hardened the GLFW inverse-kinematics example's failure reporting, recovery,
  buffer synchronization, target geometry, and OpenGL resource cleanup.
- Updated the locked `crossbeam-epoch` dependency to resolve RUSTSEC-2026-0204.

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

[0.3.0]: https://github.com/NeoCogi/constraint-solver/releases/tag/v0.3.0
[0.1.1]: https://github.com/NeoCogi/constraint-solver/tree/47403f77f99ed122bc9a052f4a4a53f071bdd33e
[0.1.0]: https://github.com/NeoCogi/constraint-solver/tree/f95a5e74d24d76a1b08464b078b91318ee2e2331
