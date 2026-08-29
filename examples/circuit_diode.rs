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
    // Series resistor + diode with Shockley equation.
    let i = Exp::var("i");
    let vd = Exp::var("vd");

    let vs = 5.0;
    let r = 1_000.0;
    let isat = 1e-12;
    let n = 1.0;
    let vt = 0.02585;

    // KVL: Vs - I*R - Vd = 0
    let kvl = Exp::sub(
        Exp::sub(Exp::val(vs), Exp::mul(i.clone(), Exp::val(r))),
        vd.clone(),
    );

    // I - Is * (exp(Vd / (n*Vt)) - 1) = 0
    let exp_arg = Exp::div(vd.clone(), Exp::val(n * vt));
    let diode_i = Exp::mul(Exp::val(isat), Exp::sub(Exp::exp(exp_arg), Exp::val(1.0)));
    let diode_eq = Exp::sub(i.clone(), diode_i);

    let compiled = Compiler::compile(&[kvl, diode_eq]).expect("compile failed");
    let solver = NewtonRaphsonSolver::new(compiled);

    let mut initial = HashMap::new();
    initial.insert("i".to_string(), 0.005);
    initial.insert("vd".to_string(), 0.7);

    let solution = solver.solve(initial).expect("solve failed");
    let i_sol = solution.values.get("i").copied().unwrap();
    let vd_sol = solution.values.get("vd").copied().unwrap();

    println!("diode: i={:.6} A, vd={:.6} V", i_sol, vd_sol);
}
