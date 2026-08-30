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
  diagnostics, preserving the effective factorization and explicit ridge
  strength used for the accepted correction.
- Added `SolverOptions` as the documented source of nonlinear iteration,
  Armijo, and explicit regularization policy.
- Added structured equation diagnostics and optional caller-provided equation
  traces to make failed constraints identifiable without parsing messages.
- Added explicit positive equation and variable characteristic scales. Equation
  scales define dimensionless residual weighting and success, while variable
  scales normalize Jacobian columns, corrections, and ridge penalties without
  changing caller-visible values.
- Added structured matrix errors for invalid shapes and tolerances, overflowing
  dimensions, allocation failure, non-finite states, and failed convergence.
- Added executable crate documentation, compiled README examples, complete
  public API documentation, and the `missing_docs` lint.
- Added an explicit package allowlist so repository workflows, ignore rules,
  and future local tooling files are not published with the crate.
- Added arithmetic operator implementations for owned and borrowed `Exp`
  values and `f64` constants so equations can mirror their mathematical form.
- Added shared modified-nodal-analysis integration infrastructure and 100-step
  diode, bipolar, MOS, RC, and amplifier waveform validations.
- Added CI gates for Rust 1.86, current stable Rust, Windows and macOS library
  builds, RustSec advisories, formatting, Clippy, debug and optimized tests,
  headless examples, documentation tests, benchmark compilation, and package
  verification.

### Changed

- Made success strict and shape-independent: `Ok(Solution)` now certifies only
  `residual_norm <= residual_tolerance`; every stationary non-root returns
  `SolverError::StationaryNonRoot`, including square and underdetermined cases.
- Replaced the overlapping QR-named least-squares entry points with one
  `solve_least_squares` operation returning a diagnostic result. Compatibility
  aliases from the pre-alpha API were intentionally removed.
- Unified nonlinear Jacobian correction policy across all shapes. Full-rank
  systems use Householder QR and rank-deficient systems use the SVD
  pseudoinverse; ridge regularization is caller-selected.
- Replaced the public fixed rank cutoff with the factorization-derived relative
  threshold `f64::EPSILON * max(rows, columns)`. `solve_least_squares` no longer
  accepts `LeastSquaresOptions`; QR and SVD share one automatic policy.
- Replaced condition-triggered regularization and retry schedules with one
  explicit policy: zero performs a direct QR/SVD solve and a positive value
  performs one augmented ridge solve at exactly the configured strength.
- Replaced the ordinary/adaptive-damping and optional-line-search split with one
  transactional Armijo update path for every `solve` call. Removed
  `solve_with_line_search`, `with_damping`, and `min_damping`; the explicit
  `initial_step_size` now caps the first candidate tested on each correction.
- Applied one numerical default policy to square, tall, and wide systems instead
  of silently changing iteration, step-size, and regularization defaults based
  on Jacobian shape.
- Changed Armijo line search to operate directly on the scaled residual norm
  with a scale-safe directional derivative in normalized coordinates.
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
- Added an asserted rank-deficient Jacobi-SVD fallback benchmark and separated
  bounded factorization sizes from larger dense-operation sizes, avoiding the
  former 4096x1024 QR workload in the default benchmark command.
- Declared Rust 1.86 as the MSRV, refreshed all compatible locked dependencies,
  and required CI builds to use the committed dependency graph.
- Extended the Rust 1.86 all-target gate to Windows and macOS so the MSRV claim
  covers platform-specific examples and development dependencies rather than a
  Linux-only repository graph.

### Fixed

- Corrected the documented `f64` scalar scope, factorization allocation
  behavior, underdetermined policy, line-search example, and LU pivot scale.
- Replaced self-comparison in randomized Jacobian coverage with an analytical
  oracle and added a headless end-to-end solve for the graphical IK model.
- Preserved structured `MatrixError` values through `SolverError` and retained
  already-computed gradient measures in every `NoConvergence` diagnostic.
- Prevented norm, column-scaled stationarity, and line-search slope calculations
  from overflowing or underflowing at extreme finite scales.
- Bounded the dimensionless stationarity tolerance to `(0, 1]` and retained
  algebraically independent small-coefficient directions above factorization
  roundoff, preventing repeated zero corrections on solvable systems.
- Applied equation and variable scaling consistently to residual success,
  stationarity, Armijo acceptance, rank and condition diagnostics, the
  linearized correction, and explicit ridge regularization. Failure diagnostics
  retain both raw and scaled per-equation residuals.
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
- Reused accepted Armijo residuals as the next solver state, eliminating
  repeated evaluation and an unreachable duplicate success branch. Corrections
  that cannot change any variable now terminate without consuming updates.
- Allocated augmented ridge matrices only when a positive regularization value
  actually requests that alternate linear problem; initial roots and default
  direct solves no longer allocate unused `(m+n) x n` storage.
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
