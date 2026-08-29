# constraint-solver
[![crates.io](https://img.shields.io/crates/v/constraint-solver?style=flat-square)](https://crates.io/crates/constraint-solver)

A small, generic constraint-solving core for nonlinear systems. This crate provides:

- A symbolic expression tree (`Exp`) for building equation systems.
- A compiler that maps variable names to internal IDs (`Compiler`).
- A dense matrix type with checked LU and QR/SVD least squares (`Matrix`).
- A Newton-Raphson solver with damping, regularization, and optional line search (`NewtonRaphsonSolver`).

This crate is intentionally CAD-agnostic and focuses on the math/solver core. Architected by humans and coded with LLM.

## Usage

Add the 0.3 release to your `Cargo.toml`:

```toml
[dependencies]
constraint-solver = "0.3"
```

## System Types (Square, Under-, Over-Constrained)

The solver supports all three system shapes:

- Square systems (equations == variables): solved via LU on the Jacobian, with
  QR/SVD pseudoinverse and augmented regularization fallback when LU cannot
  produce a stable correction.
- Under-constrained systems (equations < variables): solved via QR/SVD least squares.
  Callers explicitly choose between a minimum-norm Newton correction, which
  preserves null-space continuity, and the minimum-norm solution of each local
  linearization.
- Over-constrained systems (equations > variables): solved via QR/SVD least
  squares, minimizing the residual error. Inconsistent systems can converge at
  a nonzero residual when the scale-independent least-squares gradient is
  stationary. This is a first-order condition, not a guarantee of a global
  minimum.

### Square system example

```rust
use constraint_solver::{Compiler, ConvergenceReason, Exp, NewtonRaphsonSolver};
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
assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
```

### Under-constrained example (minimum-norm linearized solution)

```rust
use constraint_solver::{
    Compiler, ConvergenceReason, Exp, NewtonRaphsonSolver, UnderdeterminedPolicy,
};
use std::collections::HashMap;

let x = Exp::var("x");
let y = Exp::var("y");

// x + y = 1 has infinitely many solutions. The point policy removes any
// null-space component inherited from the initial guess.
let eq = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(1.0));
let compiled = Compiler::compile(&[eq]).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled).with_underdetermined_policy(
    UnderdeterminedPolicy::MinimumNormLinearizedSolution,
);

let mut initial = HashMap::new();
initial.insert("x".to_string(), 10.0);
initial.insert("y".to_string(), -10.0);
let solution = solver.solve(initial).expect("solve failed");
assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
assert!((solution.values["x"] - 0.5).abs() < 1e-10);
assert!((solution.values["y"] - 0.5).abs() < 1e-10);
```

### Over-constrained example (least squares)

```rust
use constraint_solver::{Compiler, ConvergenceReason, Exp, NewtonRaphsonSolver};
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
assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
```

### Line search and regularization

```rust
use constraint_solver::{Compiler, ConvergenceReason, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x");
let y = Exp::var("y");
let f1 = Exp::sub(Exp::sin(x.clone()), Exp::mul(y.clone(), Exp::val(2.0)));
let f2 = Exp::sub(Exp::cos(y.clone()), x.clone());

let compiled = Compiler::compile(&[f1, f2]).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled)
    .with_tolerance(1e-8)
    .with_max_iterations(50)
    .with_regularization(1e-8);

let mut initial = HashMap::new();
initial.insert("x".to_string(), 0.5);
initial.insert("y".to_string(), 0.25);

let solution = solver
    .solve_with_line_search(initial)
    .expect("solve failed");
assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
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
`SolverRunDiagnostic::equations` pairs every signed residual with its equation
index and optional trace.

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
  Run with `cargo run --example ik_2d_glfw`.

## Testing

Run the unit tests with:

```bash
cargo test
```

## Benchmarks

Benchmarks are in `benches/linear_algebra.rs` and compare the core matrix
operations used by the solver. They run both serial and parallel variants
in the same benchmark group.

```bash
# Use all available cores
cargo bench --features benchmarks

# Pin the parallel thread count
BENCH_THREADS=16 cargo bench --features benchmarks
```

The parallel path uses Rayon for matrix multiply, transpose, and QR
reflector updates when the working set is large enough.

Benchmark timings are intentionally not checked into this README: results vary
substantially with CPU topology, memory bandwidth, compiler version, and Rayon
thread count. The harness verifies serial/parallel result equivalence before it
records timings, so run it on the target machine when making performance
decisions.

## Notes

- The solver supports square, under-constrained, and over-constrained systems.
- The default underdetermined policy minimizes each Newton correction and
  preserves the initial null-space component. Use
  `UnderdeterminedPolicy::MinimumNormLinearizedSolution` when each local
  linearization should instead choose its origin-based minimum-norm point.
- Least squares solving uses Householder QR for full-rank systems and a
  one-sided Jacobi SVD pseudoinverse for rank-deficient systems. Both paths
  avoid forming normal equations.
- Regularization uses the augmented ridge system
  `[J; sqrt(lambda) I] * delta ~= [-f; 0]` and is applied adaptively when the
  retained solve subspace is ill-conditioned; use `.with_regularization(0.0)`
  to disable it. Rank loss by itself does not force damping when the retained
  SVD subspace is well conditioned.
- Line search applies a slope-based Armijo condition directly to the residual
  norm `||f||`, using directional derivative `(f^T J delta) / ||f||`. This
  avoids overflowing by squaring a large finite norm and reports failure
  without counting rejected candidates as accepted solver updates.
- Expressions are built by variable name, then compiled into a solver-friendly
  representation. Provide all variables referenced by equations in the initial
  guess map; missing variables are treated as errors.
- Internally, the solver differentiates and simplifies its symbolic Jacobian
  once during construction. Each solve creates independent numerical storage
  and reuses that storage between iterations, so solver reuse avoids both
  repeated symbolic work and per-iteration matrix allocation.

## License

MIT. See `LICENSE` for details.
