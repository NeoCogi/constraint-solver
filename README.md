# constraint-solver

A small, generic constraint-solving core for nonlinear systems. This crate provides:

- A symbolic expression tree (`Exp`) with evaluation and differentiation.
- A Jacobian builder/evaluator (`Jacobian`).
- A dense matrix type with LU and QR-based least squares (`Matrix`).
- A Newton-Raphson solver with damping, regularization, and optional line search (`NewtonRaphsonSolver`).

This crate is intentionally CAD-agnostic and focuses on the math/solver core.

## Usage

Add the crate to your `Cargo.toml` (path or git dependency depending on your setup).

### Example: solve a simple system

```rust
use constraint_solver::{Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

fn main() {
    // System:
    // f1 = x^2 + y^2 - 1
    // f2 = x*y - 0.25
    let x = Exp::var("x".to_string(), 0);
    let y = Exp::var("y".to_string(), 1);

    let f1 = Exp::sub(
        Exp::add(Exp::power(x.clone(), 2.0), Exp::power(y.clone(), 2.0)),
        Exp::val(1.0),
    );
    let f2 = Exp::sub(Exp::mul(x.clone(), y.clone()), Exp::val(0.25));

    let solver = NewtonRaphsonSolver::new(vec![f1, f2], vec![0, 1]);

    let mut initial = HashMap::new();
    initial.insert(0, 0.5);
    initial.insert(1, 0.866);

    let solution = solver.solve(initial).expect("solve failed");
    assert!(solution.converged);

    let x_sol = solution.values[&0];
    let y_sol = solution.values[&1];
    println!("x={:.6}, y={:.6}", x_sol, y_sol);
}
```

### Line search and regularization

```rust
use constraint_solver::{Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

let x = Exp::var("x".to_string(), 0);
let y = Exp::var("y".to_string(), 1);
let f1 = Exp::sub(Exp::sin(x.clone()), Exp::mul(y.clone(), Exp::val(2.0)));
let f2 = Exp::sub(Exp::cos(y.clone()), x.clone());

let solver = NewtonRaphsonSolver::new(vec![f1, f2], vec![0, 1])
    .with_tolerance(1e-8)
    .with_max_iterations(50)
    .with_regularization(1e-8);

let mut initial = HashMap::new();
initial.insert(0, 0.5);
initial.insert(1, 0.25);

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
- Provide all variables referenced by equations in the initial guess map; missing variables are treated as errors.

## License

MIT. See `LICENSE` for details.
