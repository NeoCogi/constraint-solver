# constraint-solver

A small, generic constraint-solving core for nonlinear systems. This crate provides:

- A symbolic expression tree (`Exp`) with evaluation and differentiation.
- A compiler that maps variable names to internal IDs (`Compiler`).
- A dense matrix type with LU and QR-based least squares (`Matrix`).
- A Newton-Raphson solver with damping, regularization, and optional line search (`NewtonRaphsonSolver`).

This crate is intentionally CAD-agnostic and focuses on the math/solver core.

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

## Testing

Run the unit tests with:

```bash
cargo test
```

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
