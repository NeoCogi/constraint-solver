# constraint-solver
[![crates.io](https://img.shields.io/crates/v/constraint-solver?style=flat-square)](https://crates.io/crates/constraint-solver)

A small, generic constraint-solving core for nonlinear systems. This crate provides:

- A symbolic expression tree (`Exp`) for building equation systems.
- A compiler that maps variable names to internal IDs (`Compiler`).
- A dense matrix type with LU and QR-based least squares (`Matrix`).
- A Newton-Raphson solver with damping, regularization, and optional line search (`NewtonRaphsonSolver`).

This crate is intentionally CAD-agnostic and focuses on the math/solver core. Architected by humans and coded with LLM.

## Usage

Add the crate to your `Cargo.toml` (path or git dependency depending on your setup).

## System Types (Square, Under-, Over-Constrained)

The solver supports all three system shapes:

- Square systems (equations == variables): solved via LU on the Jacobian.
- Under-constrained systems (equations < variables): solved via QR least squares,
  returning the minimum-norm solution among all valid solutions.
- Over-constrained systems (equations > variables): solved via QR least squares,
  minimizing the residual error.

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
assert!(solution.converged);
```

### Under-constrained example (minimum-norm)

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x");
let y = Exp::var("y");

// x + y = 1 has infinitely many solutions; the solver returns x = y = 0.5.
let eq = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(1.0));
let compiled = Compiler::compile(&[eq]).expect("compile failed");
let solver = NewtonRaphsonSolver::new(compiled);

let mut initial = HashMap::new();
initial.insert("x".to_string(), 0.0);
initial.insert("y".to_string(), 0.0);
let solution = solver.solve(initial).expect("solve failed");
assert!(solution.converged);
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
assert!(solution.converged);
```

### Example: solve a simple system

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

fn main() {
    // System:
    // f1 = x^2 + y^2 - 1
    // f2 = x*y - 0.25
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
    assert!(solution.converged);

    let x_sol = solution.values.get("x").copied().unwrap();
    let y_sol = solution.values.get("y").copied().unwrap();
    println!("x={:.6}, y={:.6}", x_sol, y_sol);
}
```

### Line search and regularization

```rust
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
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
assert!(solution.converged);
```

### Execution mode (serial vs parallel)

```rust
use constraint_solver::{Mode, NewtonRaphsonSolver};

// `compiled` from any of the examples above.
let solver = NewtonRaphsonSolver::new_with_mode(compiled, Mode::Parallel { thread_count: 8 })
    .expect("solver");
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
cargo bench

# Pin the parallel thread count
BENCH_THREADS=16 cargo bench
```

The parallel path uses Rayon for matrix multiply, transpose, and QR
reflector updates when the working set is large enough.

**Sample Results**
Below are median times from your recent run (Rayon threads: 16). Times are in
milliseconds, and speedup is `serial / parallel`. The benchmark harness also
verifies that serial and parallel results match for each size before timing.

Matmul (square):
| Size | Serial (ms) | Parallel (ms) | Speedup |
| --- | --- | --- | --- |
| 64 | 0.0439 | 0.0518 | 0.85x |
| 128 | 0.3044 | 0.0787 | 3.87x |
| 256 | 2.441 | 0.4181 | 5.84x |
| 512 | 18.88 | 3.178 | 5.94x |
| 1024 | 150.61 | 24.69 | 6.10x |

Transpose (square):
| Size | Serial (ms) | Parallel (ms) | Speedup |
| --- | --- | --- | --- |
| 64 | 0.0033 | 0.0091 | 0.37x |
| 128 | 0.0259 | 0.0178 | 1.46x |
| 256 | 0.1525 | 0.0293 | 5.21x |
| 512 | 1.620 | 0.1606 | 10.09x |
| 1024 | 6.824 | 0.6079 | 11.23x |

QR tall (m = 4n):
| Size | Serial (ms) | Parallel (ms) | Speedup |
| --- | --- | --- | --- |
| 256x64 | 1.584 | 1.642 | 0.96x |
| 512x128 | 12.24 | 5.340 | 2.29x |
| 1024x256 | 116.66 | 32.87 | 3.55x |
| 2048x512 | 1688.50 | 351.22 | 4.81x |
| 4096x1024 | 15418.00 | 3001.70 | 5.14x |

QR wide (n = 4m):
| Size | Serial (ms) | Parallel (ms) | Speedup |
| --- | --- | --- | --- |
| 64x256 | 1.584 | 1.651 | 0.96x |
| 128x512 | 12.29 | 5.235 | 2.35x |
| 256x1024 | 115.35 | 32.08 | 3.60x |
| 512x2048 | 1685.70 | 365.65 | 4.61x |
| 1024x4096 | 15391.00 | 3069.30 | 5.01x |

Why this shape:
Parallel wins grow with size because the fixed Rayon overhead is amortized.
Small matrices can regress (<1x) due to thread scheduling and cache effects,
while larger matrices scale until they hit memory bandwidth limits. Matmul now
uses the same `i-k-j` loop order in both serial and parallel paths, so the
speedups reflect parallelism rather than a cache-friendly loop order change.
QR shows lower speedups than matmul because parts of the factorization remain
serial and memory-bound (Amdahl’s law), so the maximum speedup is naturally
limited.

## Notes

- The solver supports square, under-constrained, and over-constrained systems.
- Least squares solving uses QR to improve numerical stability.
- Regularization is applied adaptively when the QR solve is ill-conditioned; use
  `.with_regularization(0.0)` to disable it.
- Expressions are built by variable name, then compiled into a solver-friendly
  representation. Provide all variables referenced by equations in the initial
  guess map; missing variables are treated as errors.
- Internally, the solver caches Jacobian storage between iterations to reduce
  allocations; this is an implementation detail and does not affect the public API.

## License

MIT. See `LICENSE` for details.
