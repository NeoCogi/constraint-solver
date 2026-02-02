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

fn main() {
    // Find point B given fixed A, fixed length, and horizontal constraint.
    let ax = Exp::var("ax");
    let ay = Exp::var("ay");
    let bx = Exp::var("bx");
    let by = Exp::var("by");

    let length = 5.0;
    let y_target = 2.0;

    let dx = Exp::sub(bx.clone(), ax.clone());
    let dy = Exp::sub(by.clone(), ay.clone());
    let length_eq = Exp::sub(
        Exp::add(Exp::power(dx, 2.0), Exp::power(dy, 2.0)),
        Exp::val(length * length),
    );
    let horizontal_eq = Exp::sub(by.clone(), Exp::val(y_target));

    let compiled = Compiler::compile(&[length_eq, horizontal_eq]).expect("compile failed");
    let solver = NewtonRaphsonSolver::new_with_variables(compiled, &["bx", "by"])
        .expect("solver init failed");

    let mut initial = HashMap::new();
    initial.insert("ax".to_string(), 1.0);
    initial.insert("ay".to_string(), 2.0);
    initial.insert("bx".to_string(), 6.0);
    initial.insert("by".to_string(), 2.0);

    let solution = solver.solve(initial).expect("solve failed");
    let bx_sol = solution.values.get("bx").copied().unwrap();
    let by_sol = solution.values.get("by").copied().unwrap();

    println!("B = ({:.6}, {:.6})", bx_sol, by_sol);
}
