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

use crate::compiler::{CompiledExp, EvaluationError, VarId};
use crate::matrix::Matrix;
use std::collections::HashMap;

/// Crate-internal Jacobian evaluator for compiled symbolic expressions.
///
/// The Jacobian matrix J is the matrix of all first-order partial derivatives
/// of a vector-valued function f(x). For a system with m equations and n variables:
///
/// J[i,j] = df_i/dx_j
///
/// The nonlinear solver owns this type because its expressions and variable IDs
/// are tied to one crate-private compiled-system registry. Exposing the type
/// publicly would not expose a constructible or evaluable public abstraction.
pub(crate) struct Jacobian {
    /// Vector of expressions f_i(x), each representing one equation in the system
    expressions: Vec<CompiledExp>,
    /// Variable IDs that correspond to the unknowns x_j in the system
    variables: Vec<VarId>,
    /// Cached symbolic partial derivatives df_i/dx_j (simplified)
    symbolic_jacobian: Vec<Vec<CompiledExp>>,
}

pub(crate) struct JacobianWorkspace {
    /// Dense derivative values reused across successive solver iterations.
    jacobian: Matrix,
}

impl JacobianWorkspace {
    /// Allocate a zero-filled workspace with the derivative matrix's expected
    /// equation-by-variable shape.
    pub(crate) fn new(num_equations: usize, num_variables: usize) -> Self {
        // Constructing the correct shape once lets ordinary evaluations replace
        // only element values rather than reallocating the matrix buffer.
        Self {
            jacobian: Matrix::new(num_equations, num_variables),
        }
    }

    /// Borrow the reusable derivative matrix for checked numerical solves.
    pub(crate) fn jacobian(&mut self) -> &mut Matrix {
        // The workspace retains ownership so the same allocation remains
        // available after the caller's temporary mutable borrow ends.
        &mut self.jacobian
    }
}

impl Jacobian {
    /// Create an internal Jacobian cache for one compiled system.
    ///
    /// # Arguments
    /// * `expressions` - Vector of symbolic expressions representing the system equations
    /// * `variables` - Vector of variable IDs that appear as unknowns in the expressions
    ///
    /// # Example
    /// For system: f1 = x^2 + y^2 - 1, f2 = xy - 0.25
    /// The Jacobian would be:
    /// J = [df1/dx  df1/dy] = [2x   2y]
    ///     [df2/dx  df2/dy]   [y    x ]
    pub(crate) fn new(expressions: Vec<CompiledExp>, variables: Vec<VarId>) -> Self {
        // Pre-compute symbolic derivatives once since Newton iterations repeatedly
        // evaluate J at different points.
        let mut symbolic_jacobian = Vec::new();

        for expr in &expressions {
            let mut row = Vec::new();
            for &var_id in &variables {
                row.push(expr.differentiate(var_id).simplify());
            }
            symbolic_jacobian.push(row);
        }

        Jacobian {
            expressions,
            variables,
            symbolic_jacobian,
        }
    }

    /// Return the number of scalar residual expressions represented by this
    /// symbolic Jacobian.
    pub(crate) fn equation_count(&self) -> usize {
        // Each stored expression owns exactly one row of cached partial
        // derivatives, so the expression count is the authoritative row count.
        self.expressions.len()
    }

    /// Allocate numerical derivative storage with the cached symbolic
    /// Jacobian's equation-by-variable shape.
    pub(crate) fn workspace(&self) -> JacobianWorkspace {
        // The symbolic cache is immutable; each solve receives independent
        // mutable numeric storage so a reusable solver remains thread-safe.
        JacobianWorkspace::new(self.expressions.len(), self.variables.len())
    }

    /// Evaluate every symbolic derivative into caller-owned reusable storage.
    pub(crate) fn evaluate_checked_in_workspace<'a>(
        &self,
        vars: &HashMap<VarId, f64>,
        workspace: &'a mut JacobianWorkspace,
    ) -> Result<&'a mut Matrix, EvaluationError> {
        self.evaluate_checked_into(vars, &mut workspace.jacobian)?;
        Ok(&mut workspace.jacobian)
    }

    pub(crate) fn evaluate_checked_into(
        &self,
        vars: &HashMap<VarId, f64>,
        out: &mut Matrix,
    ) -> Result<(), EvaluationError> {
        let rows = self.expressions.len();
        let cols = self.variables.len();
        if out.rows() != rows || out.cols() != cols {
            *out = Matrix::new(rows, cols);
        }

        for (i, row) in self.symbolic_jacobian.iter().enumerate() {
            for (j, expr) in row.iter().enumerate() {
                out[(i, j)] = expr.evaluate_checked(vars)?;
            }
        }
        Ok(())
    }

    pub(crate) fn evaluate_functions_checked_into(
        &self,
        vars: &HashMap<VarId, f64>,
        out: &mut Matrix,
    ) -> Result<(), EvaluationError> {
        let rows = self.expressions.len();
        if out.rows() != rows || out.cols() != 1 {
            *out = Matrix::new(rows, 1);
        }

        for i in 0..rows {
            out[(i, 0)] = self.expressions[i].evaluate_checked(vars)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::Compiler;
    use crate::exp::Exp;
    use crate::matrix::Matrix;

    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 32) as u32
        }

        fn next_f64(&mut self) -> f64 {
            let v = self.next_u32() as f64 / u32::MAX as f64;
            2.0 * v - 1.0
        }

        fn next_f64_range(&mut self, min: f64, max: f64) -> f64 {
            min + (max - min) * (self.next_u32() as f64 / u32::MAX as f64)
        }
    }

    /// Build `sum(coefficients[i] * variables[i]^2) - rhs` for a Jacobian
    /// fixture whose derivatives have a simple independent closed form.
    fn nonlinear_equation(coeffs: &[f64], vars: &[Exp], rhs: f64) -> Exp {
        // Assemble the expression through the public symbolic constructors so
        // the test exercises the same compilation path as crate users.
        let mut sum = Exp::val(0.0);
        for (coeff, var) in coeffs.iter().zip(vars.iter()) {
            let term = Exp::mul(Exp::val(*coeff), Exp::mul(var.clone(), var.clone()));
            sum = Exp::add(sum, term);
        }
        Exp::sub(sum, Exp::val(rhs))
    }

    #[test]
    fn test_jacobian_2x2() {
        // Test Jacobian computation for a simple 2x2 system
        // f1 = x^2 + y^2 - 1
        // f2 = xy - 0.25
        //
        // Expected Jacobian:
        // J = [2x   2y]
        //     [y    x ]
        let x = Exp::var("x");
        let y = Exp::var("y");

        let f1 = Exp::sub(
            Exp::add(Exp::power(x.clone(), 2.0), Exp::power(y.clone(), 2.0)),
            Exp::val(1.0),
        );

        let f2 = Exp::sub(Exp::mul(x.clone(), y.clone()), Exp::val(0.25));

        let compiled = Compiler::compile(&[f1, f2]).expect("compile failed");
        let jacobian = Jacobian::new(compiled.equations.clone(), compiled.var_table.all_var_ids());

        // Evaluate at point (0.5, 0.5)
        let mut vars = HashMap::new();
        let x_id = compiled.var_table.get_id("x").unwrap();
        let y_id = compiled.var_table.get_id("y").unwrap();
        vars.insert(x_id, 0.5); // x = 0.5
        vars.insert(y_id, 0.5); // y = 0.5

        let mut workspace = jacobian.workspace();
        let j_matrix = jacobian
            .evaluate_checked_in_workspace(&vars, &mut workspace)
            .expect("jacobian evaluate failed");
        assert_eq!(j_matrix.rows(), 2);
        assert_eq!(j_matrix.cols(), 2);

        // Expected values at (0.5, 0.5):
        // df1/dx = 2x = 1.0
        // df1/dy = 2y = 1.0
        // df2/dx = y = 0.5
        // df2/dy = x = 0.5
        assert!((j_matrix[(0, 0)] - 1.0).abs() < 1e-10); // df1/dx
        assert!((j_matrix[(0, 1)] - 1.0).abs() < 1e-10); // df1/dy
        assert!((j_matrix[(1, 0)] - 0.5).abs() < 1e-10); // df2/dx
        assert!((j_matrix[(1, 1)] - 0.5).abs() < 1e-10); // df2/dy
    }

    #[test]
    fn test_jacobian_transcendental() {
        // Test Jacobian computation for transcendental functions
        // f1 = sin(x) - y
        // f2 = cos(y) - x
        //
        // Expected Jacobian:
        // J = [cos(x)   -1   ]
        //     [-1       -sin(y)]
        let x = Exp::var("x");
        let y = Exp::var("y");

        let f1 = Exp::sub(Exp::sin(x.clone()), y.clone());
        let f2 = Exp::sub(Exp::cos(y.clone()), x.clone());

        let compiled = Compiler::compile(&[f1, f2]).expect("compile failed");
        let jacobian = Jacobian::new(compiled.equations.clone(), compiled.var_table.all_var_ids());

        // Evaluate at origin (0, 0)
        let mut vars = HashMap::new();
        let x_id = compiled.var_table.get_id("x").unwrap();
        let y_id = compiled.var_table.get_id("y").unwrap();
        vars.insert(x_id, 0.0); // x = 0
        vars.insert(y_id, 0.0); // y = 0

        let mut workspace = jacobian.workspace();
        let j_matrix = jacobian
            .evaluate_checked_in_workspace(&vars, &mut workspace)
            .expect("jacobian evaluate failed");

        // Expected values at (0, 0):
        // df1/dx = cos(0) = 1.0
        // df1/dy = -1
        // df2/dx = -1
        // df2/dy = -sin(0) = 0.0
        assert!((j_matrix[(0, 0)] - 1.0).abs() < 1e-10); // cos(0)
        assert!((j_matrix[(0, 1)] - (-1.0)).abs() < 1e-10); // -1
        assert!((j_matrix[(1, 0)] - (-1.0)).abs() < 1e-10); // -1
        assert!((j_matrix[(1, 1)] - 0.0).abs() < 1e-10); // -sin(0)
    }

    #[test]
    fn test_workspace_new_dimensions() {
        let workspace = JacobianWorkspace::new(2, 3);
        assert_eq!(workspace.jacobian.rows(), 2);
        assert_eq!(workspace.jacobian.cols(), 3);
    }

    /// Verify that workspace evaluation repairs an unexpected matrix shape and
    /// continues reusing the repaired allocation on later evaluations.
    #[test]
    fn test_evaluate_checked_in_workspace_updates_and_resizes() {
        let x = Exp::var("x");
        let y = Exp::var("y");
        let f1 = Exp::add(Exp::mul(x.clone(), x.clone()), y.clone()); // x^2 + y
        let f2 = Exp::mul(x.clone(), y.clone()); // x*y

        let compiled = Compiler::compile(&[f1, f2]).expect("compile failed");
        let jacobian = Jacobian::new(compiled.equations.clone(), compiled.var_table.all_var_ids());
        let mut workspace = jacobian.workspace();
        // Tests are descendants of this module and can deliberately corrupt the
        // private shape without retaining a production-only replacement API.
        workspace.jacobian = Matrix::new(1, 1);

        let mut vars = HashMap::new();
        let x_id = compiled.var_table.get_id("x").unwrap();
        let y_id = compiled.var_table.get_id("y").unwrap();
        vars.insert(x_id, 2.0);
        vars.insert(y_id, 3.0);

        let j1 = jacobian
            .evaluate_checked_in_workspace(&vars, &mut workspace)
            .expect("jacobian evaluate failed");
        assert_eq!(j1.rows(), 2);
        assert_eq!(j1.cols(), 2);
        assert!((j1[(0, 0)] - 4.0).abs() < 1e-12); // df1/dx = 2x
        assert!((j1[(0, 1)] - 1.0).abs() < 1e-12); // df1/dy = 1
        assert!((j1[(1, 0)] - 3.0).abs() < 1e-12); // df2/dx = y
        assert!((j1[(1, 1)] - 2.0).abs() < 1e-12); // df2/dy = x

        vars.insert(x_id, -1.0);
        vars.insert(y_id, 0.5);
        let j2 = jacobian
            .evaluate_checked_in_workspace(&vars, &mut workspace)
            .expect("jacobian evaluate failed");
        assert_eq!(j2.rows(), 2);
        assert_eq!(j2.cols(), 2);
        assert!((j2[(0, 0)] + 2.0).abs() < 1e-12); // df1/dx = 2x
        assert!((j2[(0, 1)] - 1.0).abs() < 1e-12); // df1/dy = 1
        assert!((j2[(1, 0)] - 0.5).abs() < 1e-12); // df2/dx = y
        assert!((j2[(1, 1)] + 1.0).abs() < 1e-12); // df2/dy = x
    }

    /// Compare reused nonlinear Jacobian evaluation with its analytical
    /// derivative rather than a second invocation of the same implementation.
    #[test]
    fn test_reused_workspace_matches_analytic_random_nonlinear_jacobian() {
        let mut rng = TestRng::new(0x9e37_79b9_7f4a_7c15);
        let vars: Vec<Exp> = (0..3).map(|i| Exp::var(format!("x{i}"))).collect();
        let mut equations = Vec::new();
        let mut equation_coefficients = Vec::new();

        for eq_idx in 0..3 {
            let mut coeffs = Vec::with_capacity(vars.len());
            for var_idx in 0..vars.len() {
                let mut coeff = rng.next_f64_range(-2.0, 2.0);
                if eq_idx == var_idx {
                    coeff += 1.0;
                }
                coeffs.push(coeff);
            }
            let rhs = rng.next_f64_range(-1.0, 1.0);
            equations.push(nonlinear_equation(&coeffs, &vars, rhs));
            equation_coefficients.push(coeffs);
        }

        let compiled = Compiler::compile(&equations).expect("compile failed");
        let jacobian = Jacobian::new(compiled.equations.clone(), compiled.var_table.all_var_ids());
        let mut workspace = jacobian.workspace();
        let variable_ids = compiled.var_table.all_var_ids();

        for _ in 0..25 {
            let mut values = HashMap::new();
            let mut point = Vec::with_capacity(variable_ids.len());
            for &variable_id in &variable_ids {
                let value = rng.next_f64();
                values.insert(variable_id, value);
                point.push(value);
            }

            // For f_i = sum_j(a_ij * x_j^2) - rhs_i, the independent analytical
            // oracle is df_i/dx_j = 2 * a_ij * x_j. Repeated points exercise
            // workspace reuse while this closed form detects shared compiler or
            // evaluator mistakes that a fresh workspace could not reveal.
            let evaluated = jacobian
                .evaluate_checked_in_workspace(&values, &mut workspace)
                .expect("jacobian workspace evaluate failed");

            assert_eq!(evaluated.rows(), equation_coefficients.len());
            assert_eq!(evaluated.cols(), point.len());
            for equation_index in 0..evaluated.rows() {
                for variable_index in 0..evaluated.cols() {
                    let expected = 2.0
                        * equation_coefficients[equation_index][variable_index]
                        * point[variable_index];
                    assert!(
                        (evaluated[(equation_index, variable_index)] - expected).abs() < 1e-12,
                        "equation {equation_index}, variable {variable_index}: expected {expected}, got {}",
                        evaluated[(equation_index, variable_index)]
                    );
                }
            }
        }
    }
}
