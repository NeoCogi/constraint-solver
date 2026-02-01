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

use crate::exp::{Exp, MissingVarError};
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
    expressions: Vec<Exp>,
    /// Variable IDs that correspond to the unknowns x_j in the system
    variables: Vec<usize>,
    /// Cached symbolic partial derivatives df_i/dx_j (simplified)
    symbolic_jacobian: Vec<Vec<Exp>>,
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
    pub fn new(expressions: Vec<Exp>, variables: Vec<usize>) -> Self {
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

    /// Compute the symbolic Jacobian matrix
    ///
    /// This performs symbolic differentiation of each expression with respect
    /// to each variable, creating a matrix of symbolic partial derivatives.
    /// The result is simplified algebraically but not numerically evaluated.
    ///
    /// # Returns
    /// 2D vector where result[i][j] = d(expressions[i]) / d(variables[j]) (simplified)
    ///
    /// # Performance Note
    /// This is computationally expensive as it performs symbolic differentiation
    /// and simplification. `Jacobian::new()` caches the result.
    pub fn compute_symbolic(&self) -> Vec<Vec<Exp>> {
        self.symbolic_jacobian.clone()
    }

    /// Evaluate the Jacobian matrix at specific variable values
    ///
    /// This computes the symbolic Jacobian and then evaluates it numerically
    /// at the given point, producing a numerical matrix suitable for linear algebra.
    ///
    /// # Arguments
    /// * `vars` - HashMap mapping variable IDs to their current numerical values
    ///
    /// # Returns
    /// Matrix with dimensions (num_expressions x num_variables) containing
    /// the numerical values of all partial derivatives at the given point
    ///
    /// # Example
    /// For f1 = x^2 + y^2, f2 = xy at point (x=1, y=2):
    /// J = [2x   2y] = [2   4]
    ///     [y    x ]   [2   1]
    pub fn evaluate(&self, vars: &HashMap<usize, f64>) -> Matrix {
        let rows = self.expressions.len(); // Number of equations
        let cols = self.variables.len(); // Number of variables
        let mut result = Matrix::new(rows, cols);
        self.evaluate_into(vars, &mut result);

        result
    }

    pub fn evaluate_checked(&self, vars: &HashMap<usize, f64>) -> Result<Matrix, MissingVarError> {
        let rows = self.expressions.len();
        let cols = self.variables.len();
        let mut result = Matrix::new(rows, cols);
        self.evaluate_checked_into(vars, &mut result)?;

        Ok(result)
    }

    pub fn evaluate_into(&self, vars: &HashMap<usize, f64>, out: &mut Matrix) {
        let rows = self.expressions.len();
        let cols = self.variables.len();
        if out.rows() != rows || out.cols() != cols {
            *out = Matrix::new(rows, cols);
        }

        for (i, row) in self.symbolic_jacobian.iter().enumerate() {
            for (j, expr) in row.iter().enumerate() {
                out[(i, j)] = expr.evaluate(vars);
            }
        }
    }

    pub fn evaluate_checked_into(
        &self,
        vars: &HashMap<usize, f64>,
        out: &mut Matrix,
    ) -> Result<(), MissingVarError> {
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

    /// Evaluate the function values f(x) at specific variable values
    ///
    /// This evaluates all expressions in the system at the given point,
    /// producing the function vector f(x) used in Newton-Raphson iteration.
    ///
    /// # Arguments
    /// * `vars` - HashMap mapping variable IDs to their current numerical values
    ///
    /// # Returns
    /// Column matrix (vector) with dimensions (num_expressions x 1) containing
    /// the numerical values of all functions at the given point
    ///
    /// # Usage in Newton-Raphson
    /// This is the "f(x)" in the Newton-Raphson formula: x_{k+1} = x_k - J^{-1} * f(x_k)
    /// We seek the root where f(x) = 0, so this tells us how close we are to the solution.
    pub fn evaluate_functions(&self, vars: &HashMap<usize, f64>) -> Matrix {
        let rows = self.expressions.len();
        let mut result = Matrix::new(rows, 1); // Column vector

        self.evaluate_functions_into(vars, &mut result);

        result
    }

    pub fn evaluate_functions_checked(
        &self,
        vars: &HashMap<usize, f64>,
    ) -> Result<Matrix, MissingVarError> {
        let rows = self.expressions.len();
        let mut result = Matrix::new(rows, 1);

        self.evaluate_functions_checked_into(vars, &mut result)?;

        Ok(result)
    }

    pub fn evaluate_functions_into(&self, vars: &HashMap<usize, f64>, out: &mut Matrix) {
        let rows = self.expressions.len();
        if out.rows() != rows || out.cols() != 1 {
            *out = Matrix::new(rows, 1);
        }

        for i in 0..rows {
            out[(i, 0)] = self.expressions[i].evaluate(vars);
        }
    }

    pub fn evaluate_functions_checked_into(
        &self,
        vars: &HashMap<usize, f64>,
        out: &mut Matrix,
    ) -> Result<(), MissingVarError> {
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
    use crate::exp::Exp;

    #[test]
    fn test_jacobian_2x2() {
        // Test Jacobian computation for a simple 2x2 system
        // f1 = x^2 + y^2 - 1
        // f2 = xy - 0.25
        //
        // Expected Jacobian:
        // J = [2x   2y]
        //     [y    x ]
        let x = Exp::var("x".to_string(), 0);
        let y = Exp::var("y".to_string(), 1);

        let f1 = Exp::sub(
            Exp::add(Exp::power(x.clone(), 2.0), Exp::power(y.clone(), 2.0)),
            Exp::val(1.0),
        );

        let f2 = Exp::sub(Exp::mul(x.clone(), y.clone()), Exp::val(0.25));

        let jacobian = Jacobian::new(vec![f1, f2], vec![0, 1]);

        // Evaluate at point (0.5, 0.5)
        let mut vars = HashMap::new();
        vars.insert(0, 0.5); // x = 0.5
        vars.insert(1, 0.5); // y = 0.5

        let j_matrix = jacobian.evaluate(&vars);
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
        let x = Exp::var("x".to_string(), 0);
        let y = Exp::var("y".to_string(), 1);

        let f1 = Exp::sub(Exp::sin(x.clone()), y.clone());
        let f2 = Exp::sub(Exp::cos(y.clone()), x.clone());

        let jacobian = Jacobian::new(vec![f1, f2], vec![0, 1]);

        // Evaluate at origin (0, 0)
        let mut vars = HashMap::new();
        vars.insert(0, 0.0); // x = 0
        vars.insert(1, 0.0); // y = 0

        let j_matrix = jacobian.evaluate(&vars);

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
}
