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

/// Jacobian matrix calculator for systems of symbolic expressions
///
/// The Jacobian matrix J is the matrix of all first-order partial derivatives
/// of a vector-valued function f(x). For a system with m equations and n variables:
///
/// J[i,j] = df_i/dx_j
///
/// This is essential for Newton-Raphson iteration: x_{k+1} = x_k - J^{-1} * f(x_k)
pub struct Jacobian {
    /// Vector of expressions f_i(x), each representing one equation in the system
    expressions: Vec<CompiledExp>,
    /// Variable IDs that correspond to the unknowns x_j in the system
    variables: Vec<VarId>,
    /// Cached symbolic partial derivatives df_i/dx_j (simplified)
    symbolic_jacobian: Vec<Vec<CompiledExp>>,
}

pub(crate) struct JacobianWorkspace {
    jacobian: Matrix,
}

impl JacobianWorkspace {
    pub(crate) fn new(num_equations: usize, num_variables: usize) -> Self {
        Self {
            jacobian: Matrix::new(num_equations, num_variables),
        }
    }

    pub(crate) fn replace_jacobian(&mut self, jacobian: Matrix) {
        self.jacobian = jacobian;
    }

    pub(crate) fn jacobian(&mut self) -> &mut Matrix {
        &mut self.jacobian
    }
}

impl Jacobian {
    /// Create a new Jacobian calculator
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
    pub fn new(expressions: Vec<CompiledExp>, variables: Vec<VarId>) -> Self {
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
    pub(crate) fn workspace(&self) -> JacobianWorkspace {
        JacobianWorkspace::new(self.expressions.len(), self.variables.len())
    }

    pub(crate) fn evaluate_checked(
        &self,
        vars: &HashMap<VarId, f64>,
    ) -> Result<Matrix, EvaluationError> {
        let rows = self.expressions.len();
        let cols = self.variables.len();
        let mut result = Matrix::new(rows, cols);
        self.evaluate_checked_into(vars, &mut result)?;

        Ok(result)
    }

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

    pub(crate) fn evaluate_functions_checked(
        &self,
        vars: &HashMap<VarId, f64>,
    ) -> Result<Matrix, EvaluationError> {
        let rows = self.expressions.len();
        let mut result = Matrix::new(rows, 1);

        self.evaluate_functions_checked_into(vars, &mut result)?;

        Ok(result)
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
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
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

    fn nonlinear_equation(coeffs: &[f64], vars: &[Exp], rhs: f64) -> Exp {
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
        let jacobian = Jacobian::new(
            compiled.equations.clone(),
            compiled.var_table.all_var_ids(),
        );

        // Evaluate at point (0.5, 0.5)
        let mut vars = HashMap::new();
        let x_id = compiled.var_table.get_id("x").unwrap();
        let y_id = compiled.var_table.get_id("y").unwrap();
        vars.insert(x_id, 0.5); // x = 0.5
        vars.insert(y_id, 0.5); // y = 0.5

        let j_matrix = jacobian.evaluate_checked(&vars).expect("jacobian evaluate failed");
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
        let jacobian = Jacobian::new(
            compiled.equations.clone(),
            compiled.var_table.all_var_ids(),
        );

        // Evaluate at origin (0, 0)
        let mut vars = HashMap::new();
        let x_id = compiled.var_table.get_id("x").unwrap();
        let y_id = compiled.var_table.get_id("y").unwrap();
        vars.insert(x_id, 0.0); // x = 0
        vars.insert(y_id, 0.0); // y = 0

        let j_matrix = jacobian.evaluate_checked(&vars).expect("jacobian evaluate failed");

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

    #[test]
    fn test_workspace_replace_jacobian() {
        let mut workspace = JacobianWorkspace::new(2, 2);
        let mut matrix = Matrix::new(2, 2);
        matrix[(0, 0)] = 1.0;
        matrix[(0, 1)] = 2.0;
        matrix[(1, 0)] = -3.0;
        matrix[(1, 1)] = 4.5;
        workspace.replace_jacobian(matrix);

        assert!((workspace.jacobian[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((workspace.jacobian[(0, 1)] - 2.0).abs() < 1e-12);
        assert!((workspace.jacobian[(1, 0)] + 3.0).abs() < 1e-12);
        assert!((workspace.jacobian[(1, 1)] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_checked_in_workspace_updates_and_resizes() {
        let x = Exp::var("x");
        let y = Exp::var("y");
        let f1 = Exp::add(Exp::mul(x.clone(), x.clone()), y.clone()); // x^2 + y
        let f2 = Exp::mul(x.clone(), y.clone()); // x*y

        let compiled = Compiler::compile(&[f1, f2]).expect("compile failed");
        let jacobian = Jacobian::new(
            compiled.equations.clone(),
            compiled.var_table.all_var_ids(),
        );
        let mut workspace = jacobian.workspace();
        workspace.replace_jacobian(Matrix::new(1, 1)); // force resize on first evaluate

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

    #[test]
    fn test_workspace_matches_allocating_random_nonlinear() {
        let mut rng = TestRng::new(0x9e37_79b9_7f4a_7c15);
        let vars: Vec<Exp> = (0..3).map(|i| Exp::var(format!("x{i}"))).collect();
        let mut equations = Vec::new();

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
        }

        let compiled = Compiler::compile(&equations).expect("compile failed");
        let jacobian = Jacobian::new(
            compiled.equations.clone(),
            compiled.var_table.all_var_ids(),
        );
        let mut workspace = jacobian.workspace();

        for _ in 0..25 {
            let mut values = HashMap::new();
            for name in compiled.var_table.names().iter() {
                let id = compiled.var_table.get_id(name).expect("missing id");
                values.insert(id, rng.next_f64());
            }

            let alloc = jacobian
                .evaluate_checked(&values)
                .expect("jacobian evaluate failed");
            let in_ws = jacobian
                .evaluate_checked_in_workspace(&values, &mut workspace)
                .expect("jacobian workspace evaluate failed");

            assert_eq!(alloc.rows(), in_ws.rows());
            assert_eq!(alloc.cols(), in_ws.cols());
            for i in 0..alloc.rows() {
                for j in 0..alloc.cols() {
                    assert!((alloc[(i, j)] - in_ws[(i, j)]).abs() < 1e-12);
                }
            }
        }
    }
}
