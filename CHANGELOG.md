# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-29

### Added

- Added a one-sided Jacobi SVD for square, tall, and wide least-squares systems,
  including minimum-norm Moore-Penrose solutions.
- Added singular-value rank and condition metadata to the canonical
  `LeastSquaresSolution` result.
- Added `LinearSolveDiagnostic` to successful solutions and failed-run
  diagnostics, preserving the effective factorization and explicit ridge
  strength used for the most recent successful solve attempt, even when line
  search later rejected its correction.
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
- Added shared geometric entities and tangency constraints with 100-step common
  tangent-line and rigid six-circle packing validations.
- Added CI gates for Rust 1.86, current stable Rust, Windows and macOS library
  builds, RustSec advisories, formatting, Clippy, debug and optimized tests,
  headless examples, documentation tests, benchmark compilation, and package
  verification.

### Changed

- Made success strict and shape-independent: `Ok(Solution)` now certifies only
  `residual_norm <= residual_tolerance`; every stationary non-root returns
  `SolverError::StationaryNonRoot`, including square and underdetermined cases.
- Replaced the overlapping least-squares entry points and competing QR/SVD
  dispatcher with one canonical `solve_least_squares` SVD operation returning a
  diagnostic result. Compatibility aliases from the pre-alpha API were
  intentionally removed.
- Unified nonlinear Jacobian correction policy across every shape and rank.
  The SVD pseudoinverse supplies direct corrections, while ridge regularization
  remains explicitly caller-selected.
- Replaced the public fixed rank cutoff with the factorization-derived relative
  threshold `f64::EPSILON * max(rows, columns)`. `solve_least_squares` no longer
  accepts `LeastSquaresOptions`; SVD rank classification has one automatic policy.
- Replaced condition-triggered regularization and retry schedules with one
  explicit policy: zero performs a direct SVD solve and a positive value
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
- Based parallel multiplication scheduling on complete scalar work. The compact
  Jacobi SVD is deterministic and serial in both solver execution modes.
- Made Criterion benchmarks opt-in through the `benchmarks` feature and removed
  machine-specific timing tables from the README.
- Added bounded canonical Jacobi-SVD benchmarks for full-rank square, tall,
  wide, and rank-deficient systems, separate from larger dense-operation sizes.
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
- Preserved structured `MatrixError` values through contextual
  `SolverError::Matrix` failures, including the complete accepted variable,
  residual, gradient, equation-trace, iteration, and prior-attempt state.
  Already-computed gradient measures are also retained in every
  `NoConvergence` diagnostic.
- Prevented norm, column-scaled stationarity, line-search slope, Jacobian
  normalization, physical update, and SVD scale-restoration calculations from
  overflowing or underflowing at extreme finite scales. Products and quotients
  now share one exponent-scaled arithmetic policy instead of choosing ad-hoc
  source-code associations.
- Bounded the dimensionless stationarity tolerance to `(0, 1]` and retained
  algebraically independent small-coefficient directions above factorization
  roundoff, preventing repeated zero corrections on solvable systems.
- Applied equation and variable scaling consistently to residual success,
  stationarity, Armijo acceptance, rank and condition diagnostics, the
  linearized correction, and explicit ridge regularization. Failure diagnostics
  retain both raw and scaled per-equation residuals.
- Made rank classification and Moore-Penrose solutions invariant under column
  permutation by removing the unpivoted QR pre-classification path. Uniform
  coefficient normalization also prevents finite extreme-scale systems from
  overflowing during factorization, including cases whose unscaled largest
  singular value exceeds `f64::MAX` while the final solution remains finite.
- Corrected LU pivot classification so every pivot, including the final one,
  is compared against the complete coefficient-matrix scale.
- Returned `MatrixError::AllocationFailed` from fallible construction instead
  of panicking when an otherwise valid shape cannot reserve storage.
- Stopped preallocating convergence history from an untrusted iteration limit;
  history now grows only as accepted updates are observed.
- Checked all derived matrix element counts and augmented-system row counts
  before allocation, including products involving zero-sized operands.
- Returned immediately from transpose and multiplication when their checked
  result has zero storage, so valid shapes such as `usize::MAX x 0` do not loop
  over logical dimensions that contain no elements.
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
