# constraint-solver
[![crates.io](https://img.shields.io/crates/v/constraint-solver?style=flat-square)](https://crates.io/crates/constraint-solver)

A small `f64` constraint-solving core for nonlinear systems. This crate provides:

- A symbolic expression tree (`Exp`) with ordinary arithmetic operators for
  building equation systems.
- A compiler that maps variable names to internal IDs (`Compiler`).
- A dense matrix type with checked LU and Jacobi-SVD least squares (`Matrix`).
- A Newton-Raphson solver with transactional Armijo backtracking and explicit
  regularization (`NewtonRaphsonSolver`).

This crate is intentionally CAD-agnostic and focuses on the math/solver core. Architected by humans and coded with LLM.

## Usage

Add the 0.3 release to your `Cargo.toml`:

```toml
[dependencies]
constraint-solver = "0.3"
```

## Toolchain support

`constraint-solver` requires Rust 1.86 or newer. Rust 1.86 is the crate's
minimum supported Rust version (MSRV) for the complete locked repository graph,
including development dependencies. Rust 2024 edition syntax itself is
available in Rust 1.85; the 1.86 floor reflects the graph this project actually
builds and tests, not the edition alone. CI runs `cargo test --locked
--all-targets` with Rust 1.86 on Linux, Windows, and macOS in addition to
validating the repository with the current stable toolchain.

## System Types (Square, Under-, Over-Constrained)

The solver supports all three system shapes:

- Square systems (equations == variables): solved by the same SVD pseudoinverse
  policy as rectangular Jacobians; ridge regularization occurs only when
  explicitly set.
- Under-constrained systems (equations < variables): solved via SVD least squares.
  The Moore-Penrose minimum-norm Newton correction preserves the current local
  Jacobian-null-space component for continuity.
- Over-constrained systems (equations > variables): solved via SVD least
  squares. `Ok(Solution)` still requires the residual tolerance; an inconsistent
  system whose residual is orthogonal to every Jacobian column returns
  `SolverError::StationaryNonRoot` with its residual and gradient diagnostics.

Every system shape has the same success contract: `Ok(Solution)` means the
returned equation-scale-normalized residual norm is less than or equal to
`SolverOptions::residual_tolerance`. All scales default to one, so this is the
ordinary residual norm unless scaling is explicitly configured. A first-order
stationary point whose scaled residual remains too large is an error even for a
square or under-constrained system.

## Scaling and units

Use positive characteristic scales when equations or variables use different
physical units. For equation scale `s_i` and variable scale `t_j`, the solver
uses

```text
f_scaled[i]    = f[i] / s_i
J_scaled[i,j]  = J[i,j] * t_j / s_i
delta_x[j]     = t_j * delta_scaled[j]
```

Products and quotients in these transformations are combined through binary
exponent scaling, so a representable normalized derivative or caller-unit
update is not lost merely because one intermediate multiplication order would
overflow or underflow.

The scaled residual controls convergence, stationarity, Armijo acceptance, and
least-squares weighting. A larger equation scale therefore reduces that
equation's weight. Variable scaling changes numerical coordinates without
changing caller-visible values. Explicit ridge regularization penalizes the
normalized `delta_scaled`, so its meaning is stable across variable units.

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x");
let current = Exp::var("current");
let equations = vec![
    Exp::sub(x, Exp::val(5.0)),
    Exp::sub(Exp::mul(Exp::val(1e-3), current), Exp::val(1.0)),
];
let compiled = Compiler::compile(&equations).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled)
    .with_equation_scales(vec![5.0, 1.0])
    .expect("one positive scale per equation")
    .with_variable_scales(HashMap::from([("current".to_string(), 1e3)]))
    .expect("current is a solve variable");

let initial = HashMap::from([
    ("x".to_string(), 0.0),
    ("current".to_string(), 0.0),
]);
let solution = solver.solve(initial).expect("solve failed");
assert!((solution.values["current"] - 1e3).abs() < 1e-8);
```

### Square system example

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x");
let y = Exp::var("y");

let f1 = Exp::sub(
    Exp::add(Exp::power(x.clone(), 2.0), Exp::power(y.clone(), 2.0)),
    Exp::val(1.0),
);
let f2 = Exp::sub(Exp::mul(x.clone(), y.clone()), Exp::val(0.25));

let compiled = Compiler::compile(&[f1, f2]).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled);

let mut initial = HashMap::new();
initial.insert("x".to_string(), 0.5);
initial.insert("y".to_string(), 0.866);

let solution = solver.solve(initial).expect("solve failed");
assert!(solution.error < 1e-10);
```

### Under-constrained example (minimum-norm correction)

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x");
let y = Exp::var("y");

// x + y = 1 has infinitely many solutions. The minimum-norm Newton correction
// preserves the initial [1, -1] null-space component.
let eq = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(1.0));
let compiled = Compiler::compile(&[eq]).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled);

let mut initial = HashMap::new();
initial.insert("x".to_string(), 10.0);
initial.insert("y".to_string(), -10.0);
let solution = solver.solve(initial).expect("solve failed");
assert!(solution.error < 1e-10);
assert!((solution.values["x"] - 10.5).abs() < 1e-10);
assert!((solution.values["y"] + 9.5).abs() < 1e-10);
```

### Over-constrained example (least squares)

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x");

// Consistent over-determined system: x = 1 and 2x = 2.
let eq1 = Exp::sub(x.clone(), Exp::val(1.0));
let eq2 = Exp::sub(Exp::mul(x.clone(), Exp::val(2.0)), Exp::val(2.0));

let compiled = Compiler::compile(&[eq1, eq2]).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled);

let mut initial = HashMap::new();
initial.insert("x".to_string(), 0.0);
let solution = solver.solve(initial).expect("solve failed");
assert!(solution.error < 1e-10);
```

### Globalized nonlinear solve

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x");
let y = Exp::var("y");
let f1 = Exp::sub(Exp::sin(x.clone()), Exp::mul(y.clone(), Exp::val(2.0)));
let f2 = Exp::sub(Exp::cos(y.clone()), x.clone());

let compiled = Compiler::compile(&[f1, f2]).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled)
    .with_residual_tolerance(1e-8)
    .with_max_iterations(50);

let mut initial = HashMap::new();
initial.insert("x".to_string(), 0.5);
initial.insert("y".to_string(), 0.25);

// Every solve uses Armijo backtracking; accepted candidates are evaluated
// before they replace the current state.
let solution = solver.solve(initial).expect("solve failed");
assert!(solution.error < 1e-8);
```

### Execution mode (serial vs parallel)

```rust
use constraint_solver::{Compiler, Exp, Mode, NewtonRaphsonSolver};

let equation = Exp::sub(Exp::var("x"), Exp::val(1.0));
let compiled = Compiler::compile(&[equation]).expect("compile failed");
let _solver = NewtonRaphsonSolver::new_with_mode(compiled, Mode::Parallel { thread_count: 8 })
    .expect("solver");
```

### Introspection and failure diagnostics

`CompiledSystem` exposes equation counts and stable variable names before it is
moved into a solver. Optional equation traces are positional and must include
exactly one entry per equation; attaching them is fallible. On failure,
`SolverRunDiagnostic::equations` pairs every raw and scaled signed residual with
its equation index and optional trace. Successful solutions and run diagnostics also expose
`last_linear_attempt`, which reports the effective SVD rank and condition
estimate and whether the caller explicitly selected ridge regularization. It is
an attempt diagnostic: a later Armijo failure can reject the resulting
correction without erasing the successfully completed factorization. A
`SolverError::Matrix` retains both its typed `MatrixError` source and the
complete `SolverRunDiagnostic` at the accepted state where the attempt failed.

```rust
use constraint_solver::{Compiler, EquationTrace, Exp, NewtonRaphsonSolver};

let equations = [Exp::sub(Exp::var("x"), Exp::val(1.0))];
let compiled = Compiler::compile(&equations).expect("compile failed");
assert_eq!(compiled.equation_count(), 1);
assert_eq!(compiled.variable_names(), &["x"]);

let solver = NewtonRaphsonSolver::new(compiled)
    .with_equation_traces(vec![Some(EquationTrace {
        constraint_id: 7,
        description: "x position".to_string(),
    })])
    .expect("one trace entry is required for one equation");
```

## Examples

The `examples/` folder contains small end-to-end programs that show how to build
constraints and run the solver:

- `cad_circle_intersection`: two circle constraints; finds one of the intersection points.
  Run with `cargo run --example cad_circle_intersection`.
- `cad_segment_horizontal`: fixed-length segment with a horizontal constraint; solves for B
  while A is fixed. Run with `cargo run --example cad_segment_horizontal`.
- `circuit_voltage_divider`: resistive divider with explicit currents and KCL.
  Run with `cargo run --example circuit_voltage_divider`.
- `circuit_diode`: diode + resistor using the Shockley equation (nonlinear).
  Run with `cargo run --example circuit_diode`.
- `ik_2d_glfw`: 2D inverse kinematics chain that follows the mouse (GLFW + OpenGL + glow).
  Uses small joint/target constraint structs and `Vec2d` for joint positions.
  Run with `cargo run --features glfw-example --example ik_2d_glfw`. This
  explicit feature may require GLFW/X11 development packages on Linux; ordinary
  library builds and tests do not compile those native dependencies.

## Testing

Run the unit tests with:

```bash
cargo test
```

### Real-world validation traces

The integration suite reuses shared electronic and geometric builders for 100
warm-started samples per scenario. Electronic coverage includes an AC diode
clipper, diode logic, saturated bipolar inverter/NAND logic, CMOS inverter/NAND
logic, an RC low-pass filter, and a common-emitter amplifier. MOS devices are
described as CMOS rather than TTL because TTL specifically denotes bipolar
transistor logic. The compact smooth device models are solver validation
fixtures, not a replacement for a production SPICE implementation.

Geometry coverage tracks a common line tangent to two moving circles and a
rigid six-circle triangular packing. Six planar circles cannot all be pairwise
tangent; the packing realizes nine contact edges and checks every other pair
for non-overlap.

Successful output is captured during ordinary test runs. Show each complete
CSV time series with:

```bash
cargo test --test real_world_electronics -- --show-output
cargo test --test real_world_geometry -- --show-output
```

## Benchmarks

Benchmarks are in `benches/linear_algebra.rs` and compare the core matrix
operations used by the solver. Dense multiplication and transpose compare
serial and parallel variants; bounded square, tall, wide, and rank-deficient
groups measure the canonical Jacobi SVD.

```bash
# Use all available cores
cargo bench --features benchmarks

# Pin the parallel thread count
BENCH_THREADS=16 cargo bench --features benchmarks
```

The parallel path uses Rayon for matrix multiplication and transpose when the
working set is large enough. The compact Jacobi SVD is deterministic and
serial in both solver execution modes.

Benchmark timings are intentionally not checked into this README: results vary
substantially with CPU topology, memory bandwidth, compiler version, and Rayon
thread count. The harness verifies serial/parallel result equivalence before it
records timings, so run it on the target machine when making performance
decisions.

## Notes

- The solver supports square, under-constrained, and over-constrained systems.
- Underdetermined systems use the Moore-Penrose minimum-norm Newton correction,
  preserving the current Jacobian-null-space component for continuity.
- Least squares solving uses one one-sided Jacobi SVD pseudoinverse for every
  matrix shape and rank. This avoids normal equations and makes numerical rank
  independent of variable ordering. Rank uses the scale-relative threshold
  `f64::EPSILON * max(rows, columns)` rather than a caller-provided application-
  level cutoff.
- Regularization uses the augmented ridge system
  `[J; sqrt(lambda) I] * delta ~= [-f; 0]`. Its policy is explicit and defaults
  to zero: zero solves the original Jacobian once, while a positive value solves
  one augmented system using exactly that ridge strength. Condition estimates
  are diagnostic and never silently replace the requested linear problem.
- Line search applies a slope-based Armijo condition directly to the scaled
  residual norm `||f_scaled||`. It evaluates the equivalent scale-safe
  directional derivative without first forming a potentially overflowing raw
  dot product, and reports failure without counting rejected candidates as
  accepted solver updates.
- Expressions are built by variable name, then compiled into a solver-friendly
  representation. Provide all variables referenced by equations in the initial
  guess map; missing variables are treated as errors.
- Internally, the solver differentiates and simplifies its symbolic Jacobian
  once during construction. Each solve creates independent numerical storage
  and reuses residual, Jacobian, and regularization workspaces between
  iterations. Matrix factorizations still allocate their own working copies and
  temporary vectors.
- Derivatives are evaluated from the exact symbolic tree; the solver does not
  replace non-finite derivatives with finite differences. A finite residual at
  a domain boundary may therefore still return
  `SolverError::NonFiniteEvaluation` when the written derivative contains an
  indeterminate form such as `0 * infinity`. Start inside the differentiable
  domain or rewrite the expression—for example, use `x^1.5` instead of
  `x * sqrt(x)` when evaluating at `x = 0` is required.

## License

MIT. See `LICENSE` for details.
