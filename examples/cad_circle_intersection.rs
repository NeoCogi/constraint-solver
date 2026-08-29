/*
MIT License

Copyright (c) 2026 Raja Lehtihet & Wael El Oraiby

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

/// Build the zero-valued residual for one circle equation.
fn circle_eq(x: &Exp, y: &Exp, cx: f64, cy: f64, r: f64) -> Exp {
    // Express squared coordinate offsets separately so the resulting tree
    // mirrors (x-cx)^2 + (y-cy)^2 - r^2 and remains easy to inspect.
    let dx = Exp::sub(x.clone(), Exp::val(cx));
    let dy = Exp::sub(y.clone(), Exp::val(cy));
    Exp::sub(
        Exp::add(Exp::power(dx, 2.0), Exp::power(dy, 2.0)),
        Exp::val(r * r),
    )
}

fn main() {
    // Two circles with radius 3 centered at (0, 0) and (4, 0).
    let x = Exp::var("x");
    let y = Exp::var("y");

    let eq1 = circle_eq(&x, &y, 0.0, 0.0, 3.0);
    let eq2 = circle_eq(&x, &y, 4.0, 0.0, 3.0);

    let compiled = Compiler::compile(&[eq1, eq2]).expect("compile failed");
    let solver = NewtonRaphsonSolver::new(compiled);

    let mut initial = HashMap::new();
    initial.insert("x".to_string(), 2.0);
    initial.insert("y".to_string(), 1.0);

    let solution = solver.solve(initial).expect("solve failed");
    let x_sol = solution.values.get("x").copied().unwrap();
    let y_sol = solution.values.get("y").copied().unwrap();

    // The initial positive y coordinate selects the upper intersection. Check
    // the known geometry so this executable example is also a CI smoke test.
    assert!(solution.error < 1e-10);
    assert!(solution.iterations > 0);
    assert!((x_sol - 2.0).abs() < 1e-10);
    assert!((y_sol - 5.0_f64.sqrt()).abs() < 1e-10);
    println!("intersection: x={:.6}, y={:.6}", x_sol, y_sol);
}
