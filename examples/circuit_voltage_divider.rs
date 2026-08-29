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

use constraint_solver::{Compiler, ConvergenceReason, Exp, NewtonRaphsonSolver};
use std::collections::HashMap;

fn main() {
    // Simple resistive divider with explicit currents.
    let vout = Exp::var("vout");
    let i1 = Exp::var("i1");
    let i2 = Exp::var("i2");

    let vin = 12.0;
    let r1 = 1_000.0;
    let r2 = 2_000.0;

    // (Vin - Vout) / R1 = I1
    let eq1 = Exp::sub(
        Exp::div(Exp::sub(Exp::val(vin), vout.clone()), Exp::val(r1)),
        i1.clone(),
    );
    // Vout / R2 = I2
    let eq2 = Exp::sub(Exp::div(vout.clone(), Exp::val(r2)), i2.clone());
    // KCL: I1 = I2
    let eq3 = Exp::sub(i1.clone(), i2.clone());

    let compiled = Compiler::compile(&[eq1, eq2, eq3]).expect("compile failed");
    let solver = NewtonRaphsonSolver::new(compiled);

    // Deliberately violate both Ohm's-law equations and KCL so this example
    // demonstrates the solver rather than merely validating an exact initial
    // root.
    let mut initial = HashMap::new();
    initial.insert("vout".to_string(), 6.0);
    initial.insert("i1".to_string(), 0.005);
    initial.insert("i2".to_string(), 0.002);

    let solution = solver.solve(initial).expect("solve failed");
    let vout_sol = solution.values.get("vout").copied().unwrap();
    let i_sol = solution.values.get("i1").copied().unwrap();

    // A runnable example doubles as a release smoke test when it verifies the
    // known divider solution and confirms that a numerical update occurred.
    assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
    assert!(solution.iterations > 0);
    assert!((vout_sol - 8.0).abs() < 1e-8);
    assert!((i_sol - 0.004).abs() < 1e-10);
    assert!((solution.values["i2"] - i_sol).abs() < 1e-10);
    println!("vout={:.6} V, current={:.6} A", vout_sol, i_sol);
}
