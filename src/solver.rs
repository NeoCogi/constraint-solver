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
use crate::jacobian::Jacobian;
use crate::matrix::{Matrix, MatrixError};
use std::collections::HashMap;

struct LeastSquaresWorkspace {
    augmented_j: Matrix,
    augmented_b: Matrix,
    num_equations: usize,
    num_variables: usize,
}

impl LeastSquaresWorkspace {
    fn new(num_equations: usize, num_variables: usize, sqrt_reg: f64) -> Self {
        let mut augmented_j = Matrix::new(num_equations + num_variables, num_variables);
        for i in 0..num_variables {
            augmented_j[(num_equations + i, i)] = sqrt_reg;
        }

        LeastSquaresWorkspace {
            augmented_j,
            augmented_b: Matrix::new(num_equations + num_variables, 1),
            num_equations,
            num_variables,
        }
    }
}

enum LeastSquaresSolveError {
    InvalidInput(MatrixError),
    Singular { qr_err: String, normal_eq_err: String },
}

/// Newton-Raphson solver for systems of nonlinear equations
///
/// This solver can handle:
/// - Square systems (equations == variables)
/// - Under-constrained systems (equations < variables) - uses least squares
/// - Over-constrained systems (equations > variables) - uses least squares
///
/// The solver uses adaptive damping and regularization for numerical stability.
pub struct NewtonRaphsonSolver {
    /// The system of equations to solve (each equation should equal zero)
    equations: Vec<Exp>,
    /// Variable IDs that correspond to unknowns in the system
    variables: Vec<usize>,
    /// Maximum number of iterations before giving up
    max_iterations: usize,
    /// Convergence tolerance (solution found when |f(x)| < tolerance)
    tolerance: f64,
    /// Initial damping factor (step size multiplier, 0 < damping <= 1)
    /// Lower values make convergence more stable but slower
    damping_factor: f64,
    /// Minimum allowed damping before declaring convergence failure
    min_damping: f64,
    /// Regularization parameter added to diagonal elements for numerical stability
    regularization: f64,
    /// Optional metadata describing the source of each equation in the system
    equation_traces: Vec<Option<EquationTrace>>,
}

/// Solution result containing the solved variable values and convergence info
#[derive(Debug, Clone)]
pub struct Solution {
    /// Final values of all variables (variable_id -> value)
    pub values: HashMap<usize, f64>,
    /// Number of iterations taken to converge
    pub iterations: usize,
    /// Final residual error |f(x)|
    pub error: f64,
    /// Whether the solver successfully converged
    pub converged: bool,
    /// History of error values throughout the iteration process
    pub convergence_history: Vec<f64>,
}

/// Provides traceability information back to the constraint that generated an equation
#[derive(Debug, Clone)]
pub struct EquationTrace {
    pub constraint_id: usize,
    pub description: String,
}

/// Diagnostic data captured when the solver fails
#[derive(Debug, Clone)]
pub struct SolverRunDiagnostic {
    pub message: String,
    pub iterations: usize,
    pub error: f64,
    pub values: HashMap<usize, f64>,
    pub residuals: Vec<f64>,
}

/// Possible solver error conditions
#[derive(Debug, Clone)]
pub enum SolverError {
    /// Jacobian matrix became singular and couldn't be inverted
    SingularMatrix(SolverRunDiagnostic),
    /// Failed to converge within the maximum number of iterations
    NoConvergence(SolverRunDiagnostic),
    /// A required variable was missing during evaluation
    MissingVariable(MissingVarError),
    /// Invalid input parameters or system setup
    InvalidInput(String),
}

impl From<MissingVarError> for SolverError {
    fn from(value: MissingVarError) -> Self {
        SolverError::MissingVariable(value)
    }
}

impl From<MatrixError> for SolverError {
    fn from(value: MatrixError) -> Self {
        SolverError::InvalidInput(value.to_string())
    }
}

impl NewtonRaphsonSolver {
    /// Create a new Newton-Raphson solver with adaptive parameters
    ///
    /// Parameters are automatically adjusted based on system type:
    /// - Over-constrained systems get more iterations, relaxed tolerance, conservative damping
    /// - Normal systems use standard parameters for fast convergence
    ///
    /// # Arguments
    /// * `equations` - System of equations (each should evaluate to 0 at solution)
    /// * `variables` - Variable IDs that appear in the equations
    pub fn new(equations: Vec<Exp>, variables: Vec<usize>) -> Self {
        // Allow both under-constrained and over-constrained systems
        // We'll use least squares to solve them

        let is_over_constrained = equations.len() > variables.len();

        NewtonRaphsonSolver {
            equations,
            variables,
            // Over-constrained systems are harder to converge, so give them more iterations
            max_iterations: if is_over_constrained { 200 } else { 100 },
            // Over-constrained systems use relaxed tolerance since exact solutions may not exist
            tolerance: if is_over_constrained { 1e-8 } else { 1e-10 },
            // Over-constrained systems use conservative damping for stability
            damping_factor: if is_over_constrained { 0.7 } else { 1.0 },
            min_damping: 0.001,
            // Over-constrained systems need more regularization for numerical stability
            regularization: if is_over_constrained { 1e-4 } else { 1e-8 },
            equation_traces: Vec::new(),
        }
    }

    /// Attach metadata describing the origin of each equation in the system
    pub fn try_with_equation_traces(
        mut self,
        traces: Vec<Option<EquationTrace>>,
    ) -> Result<Self, SolverError> {
        if !self.equations.is_empty() && traces.len() != self.equations.len() {
            return Err(SolverError::InvalidInput(format!(
                "Equation trace length ({}) does not match number of equations ({})",
                traces.len(),
                self.equations.len()
            )));
        }
        self.equation_traces = traces;
        Ok(self)
    }

    /// Attach metadata describing the origin of each equation in the system
    pub fn with_equation_traces(self, traces: Vec<Option<EquationTrace>>) -> Self {
        self.try_with_equation_traces(traces)
            .unwrap_or_else(|err| match err {
                SolverError::InvalidInput(msg) => panic!("{}", msg),
                other => panic!("{other:?}"),
            })
    }

    /// Fetch trace metadata for a specific equation, if available
    pub fn trace_for_equation(&self, equation_index: usize) -> Option<&EquationTrace> {
        self.equation_traces
            .get(equation_index)
            .and_then(|entry| entry.as_ref())
    }

    /// Override the maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Override the convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Override the damping factor (clamped to [0.1, 1.0] for stability)
    pub fn with_damping(mut self, damping_factor: f64) -> Self {
        self.damping_factor = damping_factor.clamp(0.1, 1.0);
        self
    }

    /// Override the regularization parameter
    pub fn with_regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization.max(0.0);
        self
    }

    /// Solve the system using standard Newton-Raphson method
    pub fn solve(&self, initial_guess: HashMap<usize, f64>) -> Result<Solution, SolverError> {
        self.solve_modified(initial_guess, false)
    }

    /// Core Newton-Raphson solver implementation with optional line search
    ///
    /// # Arguments
    /// * `initial_guess` - Starting values for all variables
    /// * `use_line_search` - Whether to use backtracking line search for step size
    ///
    /// # Algorithm
    /// 1. Evaluate f(x) and check for convergence
    /// 2. Compute Jacobian J = df/dx
    /// 3. Solve linear system: J * delta = -f(x)
    ///    - Square systems: Direct LU decomposition
    ///    - Under/over-constrained: Least squares via J^T * J * delta = -J^T * f
    /// 4. Update: x_new = x_old + damping * delta
    /// 5. Adapt damping based on error change
    pub fn solve_modified(
        &self,
        initial_guess: HashMap<usize, f64>,
        use_line_search: bool,
    ) -> Result<Solution, SolverError> {
        self.validate_initial_guess(&initial_guess)?;

        let mut vars = initial_guess.clone();
        let jacobian = Jacobian::new(self.equations.clone(), self.variables.clone());
        let mut convergence_history = Vec::with_capacity(self.max_iterations);
        // Start with configured damping factor, will be adapted during iterations
        let mut damping = self.damping_factor;
        let mut last_error = f64::INFINITY;

        let num_equations = self.equations.len();
        let num_variables = self.variables.len();

        let mut f_vals = Matrix::new(num_equations, 1);
        let mut f_neg = Matrix::new(num_equations, 1);
        let mut j_matrix = Matrix::new(num_equations, num_variables);

        let mut line_search_f_vals = Matrix::new(num_equations, 1);

        let mut least_squares_workspace =
            if self.regularization > 0.0 && num_equations != num_variables {
                Some(LeastSquaresWorkspace::new(
                    num_equations,
                    num_variables,
                    self.regularization.sqrt(),
                ))
            } else {
                None
            };

        for iter in 0..self.max_iterations {
            // Step 1: Evaluate function values f(x)
            jacobian.evaluate_functions_checked_into(&vars, &mut f_vals)?;
            let error = f_vals.norm();
            convergence_history.push(error);

            // Check for convergence: |f(x)| < tolerance
            if error < self.tolerance {
                return Ok(Solution {
                    values: vars,
                    iterations: iter + 1,
                    error,
                    converged: true,
                    convergence_history,
                });
            }

            // Check for stagnation (error not decreasing significantly)
            if iter > 0 && (error - last_error).abs() < self.tolerance * 0.1 {
                damping *= 0.5; // Reduce step size
                if damping < self.min_damping {
                    let diag = self.build_diagnostic(
                        format!(
                            "Solver stagnated at iteration {} with error {:.2e}",
                            iter + 1,
                            error
                        ),
                        iter + 1,
                        &vars,
                        &f_vals,
                    );
                    return Err(SolverError::NoConvergence(diag));
                }
            }

            // Adaptive damping based on error change
            if error > last_error * 1.5 {
                // Error increased significantly - reduce damping for stability
                damping *= 0.5;
            } else if error < last_error * 0.5 {
                // Error decreased significantly - can increase damping for faster convergence
                damping = (damping * 1.2).min(1.0);
            }

            last_error = error;

            // Step 2: Compute Jacobian matrix J = df/dx
            jacobian.evaluate_checked_into(&vars, &mut j_matrix)?;

            // Add regularization to diagonal for numerical stability
            // For non-square matrices, only regularize the available diagonal elements
            if self.regularization > 0.0 {
                let diag_size = j_matrix.rows().min(j_matrix.cols());
                for i in 0..diag_size {
                    j_matrix[(i, i)] += self.regularization;
                }
            }

            // Step 3: Solve linear system J * delta = -f(x)
            for i in 0..num_equations {
                f_neg[(i, 0)] = -f_vals[(i, 0)];
            }

            // Handle different system types with appropriate solving methods
            let delta = if j_matrix.rows() == j_matrix.cols() {
                // Square system (equations == variables)
                // Use standard LU decomposition: J * delta = -f
                match j_matrix.solve_lu(&f_neg) {
                    Ok(d) => d,
                    Err(_) => {
                        // Matrix is singular - try with increased regularization
                        let diag_size = j_matrix.rows().min(j_matrix.cols());
                        for i in 0..diag_size {
                            j_matrix[(i, i)] += self.regularization * 100.0;
                        }
                        match j_matrix.solve_lu(&f_neg) {
                            Ok(d) => d,
                            Err(e) => {
                                let diag = self.build_diagnostic(
                                    format!("Matrix is singular even with regularization: {e}"),
                                    iter + 1,
                                    &vars,
                                    &f_vals,
                                );
                                return Err(SolverError::SingularMatrix(diag));
                            }
                        }
                    }
                }
            } else {
                match self.solve_least_squares_delta(
                    &j_matrix,
                    &f_neg,
                    least_squares_workspace.as_mut(),
                ) {
                    Ok(delta) => delta,
                    Err(LeastSquaresSolveError::InvalidInput(err)) => {
                        return Err(SolverError::InvalidInput(err.to_string()));
                    }
                    Err(LeastSquaresSolveError::Singular {
                        qr_err,
                        normal_eq_err,
                    }) => {
                        let diag = self.build_diagnostic(
                            format!(
                                "Least squares system is singular (qr_err: {qr_err}; normal_eq_err: {normal_eq_err})"
                            ),
                            iter + 1,
                            &vars,
                            &f_vals,
                        );
                        return Err(SolverError::SingularMatrix(diag));
                    }
                }
            };

            // Step 4: Determine step size (damping or line search)
            let step_size = if use_line_search {
                // Use backtracking line search to find optimal step size
                self.line_search(&jacobian, &vars, &delta, error, &mut line_search_f_vals)?
            } else {
                // Use current damping factor as step size
                damping
            };

            // Step 5: Update variables: x_new = x_old + step_size * delta
            for (i, &var_id) in self.variables.iter().enumerate() {
                vars.insert(var_id, vars[&var_id] + step_size * delta[(i, 0)]);
            }

            // Additional convergence check: small step size indicates convergence
            if delta.norm() * step_size < self.tolerance {
                return Ok(Solution {
                    values: vars,
                    iterations: iter + 1,
                    error,
                    converged: true,
                    convergence_history,
                });
            }
        }

        // Failed to converge within max iterations
        let residuals = jacobian.evaluate_functions_checked(&vars)?;
        let diag = self.build_diagnostic(
            format!(
                "Failed to converge after {} iterations. Final error: {:.2e}",
                self.max_iterations,
                residuals.norm()
            ),
            self.max_iterations,
            &vars,
            &residuals,
        );
        Err(SolverError::NoConvergence(diag))
    }

    fn solve_least_squares_delta(
        &self,
        j_matrix: &Matrix,
        f_neg: &Matrix,
        workspace: Option<&mut LeastSquaresWorkspace>,
    ) -> Result<Matrix, LeastSquaresSolveError> {
        // Prefer QR-based least squares to avoid forming normal equations (J^T J), which can
        // severely amplify conditioning issues. When regularization is enabled, solve the
        // augmented system [J; sqrt(lambda) I] * delta ~= [-f; 0], matching (J^T J + lambda I) behavior.
        let qr_result = if self.regularization > 0.0 {
            match workspace {
                Some(workspace) => {
                    if workspace.num_equations != j_matrix.rows()
                        || workspace.num_variables != j_matrix.cols()
                    {
                        return Err(LeastSquaresSolveError::InvalidInput(
                            MatrixError::DimensionMismatch {
                                operation: "least_squares",
                                left: (workspace.augmented_j.rows(), workspace.augmented_j.cols()),
                                right: (j_matrix.rows(), j_matrix.cols()),
                            },
                        ));
                    }

                    for i in 0..workspace.num_equations {
                        for j in 0..workspace.num_variables {
                            workspace.augmented_j[(i, j)] = j_matrix[(i, j)];
                        }
                        workspace.augmented_b[(i, 0)] = f_neg[(i, 0)];
                    }
                    for i in 0..workspace.num_variables {
                        workspace.augmented_b[(workspace.num_equations + i, 0)] = 0.0;
                    }

                    workspace
                        .augmented_j
                        .solve_least_squares_qr(&workspace.augmented_b)
                }
                None => j_matrix.solve_least_squares_qr(f_neg),
            }
        } else {
            j_matrix.solve_least_squares_qr(f_neg)
        };

        match qr_result {
            Ok(delta) => Ok(delta),
            Err(qr_err) => {
                // Fallback to the legacy normal-equation path for compatibility.
                let j_transpose = j_matrix.transpose();
                let jtj = j_transpose
                    .try_mul(j_matrix)
                    .map_err(LeastSquaresSolveError::InvalidInput)?;
                let jtf = j_transpose
                    .try_mul(f_neg)
                    .map_err(LeastSquaresSolveError::InvalidInput)?;

                let mut jtj_reg = jtj;
                if self.regularization > 0.0 {
                    for i in 0..jtj_reg.rows() {
                        jtj_reg[(i, i)] += self.regularization;
                    }
                }

                match jtj_reg.solve_lu(&jtf) {
                    Ok(d) => Ok(d),
                    Err(e) => {
                        if self.regularization > 0.0 {
                            for i in 0..jtj_reg.rows() {
                                jtj_reg[(i, i)] += self.regularization * 100.0;
                            }

                            match jtj_reg.solve_lu(&jtf) {
                                Ok(d) => Ok(d),
                                Err(e2) => Err(LeastSquaresSolveError::Singular {
                                    qr_err,
                                    normal_eq_err: e2,
                                }),
                            }
                        } else {
                            Err(LeastSquaresSolveError::Singular {
                                qr_err,
                                normal_eq_err: e,
                            })
                        }
                    }
                }
            }
        }
    }

    fn validate_initial_guess(&self, vars: &HashMap<usize, f64>) -> Result<(), SolverError> {
        let mut missing: Vec<usize> = self
            .variables
            .iter()
            .copied()
            .filter(|var_id| !vars.contains_key(var_id))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }

        missing.sort_unstable();
        Err(SolverError::InvalidInput(format!(
            "Initial guess missing values for variable IDs: {missing:?}"
        )))
    }

    fn build_diagnostic(
        &self,
        message: String,
        iterations: usize,
        vars: &HashMap<usize, f64>,
        residuals_matrix: &Matrix,
    ) -> SolverRunDiagnostic {
        let mut residuals = Vec::with_capacity(residuals_matrix.rows());
        for i in 0..residuals_matrix.rows() {
            residuals.push(residuals_matrix[(i, 0)]);
        }

        SolverRunDiagnostic {
            message,
            iterations,
            error: residuals_matrix.norm(),
            values: vars.clone(),
            residuals,
        }
    }

    /// Backtracking line search to find optimal step size
    ///
    /// Tries progressively smaller step sizes until one is found that reduces the error.
    /// This helps prevent oscillation and improves convergence robustness.
    ///
    /// # Arguments
    /// * `jacobian` - Jacobian evaluator for computing function values
    /// * `vars` - Current variable values
    /// * `delta` - Newton step direction
    /// * `current_error` - Current function error |f(x)|
    ///
    /// # Returns
    /// Optimal step size alpha such that x_new = x + alpha * delta reduces error
    fn line_search(
        &self,
        jacobian: &Jacobian,
        vars: &HashMap<usize, f64>,
        delta: &Matrix,
        current_error: f64,
        f_vals: &mut Matrix,
    ) -> Result<f64, SolverError> {
        let mut alpha = 1.0; // Start with full Newton step
        let mut new_vars = vars.clone();

        // Try up to 20 different step sizes
        for _ in 0..20 {
            new_vars.clone_from(vars);
            // Compute new variable values: x_new = x + alpha * delta
            for (i, &var_id) in self.variables.iter().enumerate() {
                new_vars.insert(var_id, vars[&var_id] + alpha * delta[(i, 0)]);
            }

            // Check if this step size reduces the error sufficiently
            jacobian.evaluate_functions_checked_into(&new_vars, f_vals)?;
            let new_error = f_vals.norm();
            // Armijo condition: require sufficient decrease
            if new_error < current_error * (1.0 - 0.5 * alpha) {
                return Ok(alpha);
            }
            // Reduce step size and try again
            alpha *= 0.5;

            // Give up if step size becomes too small
            if alpha < 1e-10 {
                break;
            }
        }

        // Return at least the minimum damping to avoid complete stagnation
        Ok(alpha.max(self.min_damping))
    }

    /// Solve the system using Newton-Raphson with line search
    ///
    /// Line search helps improve convergence robustness by automatically
    /// finding good step sizes, especially useful for difficult systems.
    pub fn solve_with_line_search(
        &self,
        initial_guess: HashMap<usize, f64>,
    ) -> Result<Solution, SolverError> {
        self.solve_modified(initial_guess, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exp::Exp;

    #[test]
    fn test_simple_system() {
        // Test system: x^2 + y^2 = 1, xy = 0.25
        // Solution should be approximately x ~= 0.5, y ~= 0.866 (or vice versa)
        let x = Exp::var("x".to_string(), 0);
        let y = Exp::var("y".to_string(), 1);

        let eq1 = Exp::sub(
            Exp::add(Exp::power(x.clone(), 2.0), Exp::power(y.clone(), 2.0)),
            Exp::val(1.0),
        );
        let eq2 = Exp::sub(Exp::mul(x.clone(), y.clone()), Exp::val(0.25));

        let solver = NewtonRaphsonSolver::new(vec![eq1, eq2], vec![0, 1]);

        let mut initial = HashMap::new();
        initial.insert(0, 0.5);
        initial.insert(1, 0.866);

        let solution = match solver.solve(initial.clone()) {
            Ok(sol) => sol,
            Err(_) => match solver.solve_with_line_search(initial) {
                Ok(sol) => sol,
                Err(e) => panic!("Failed to solve: {:?}", e),
            },
        };
        assert!(solution.converged);
        assert!(solution.error < 1e-10);

        let x_sol = solution.values[&0];
        let y_sol = solution.values[&1];
        assert!((x_sol * x_sol + y_sol * y_sol - 1.0).abs() < 1e-10);
        assert!((x_sol * y_sol - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_transcendental_system() {
        // Test transcendental system: sin(x) = 2y, cos(y) = x
        let x = Exp::var("x".to_string(), 0);
        let y = Exp::var("y".to_string(), 1);

        let eq1 = Exp::sub(Exp::sin(x.clone()), Exp::mul(y.clone(), Exp::val(2.0)));
        let eq2 = Exp::sub(Exp::cos(y.clone()), x.clone());

        let solver = NewtonRaphsonSolver::new(vec![eq1, eq2], vec![0, 1])
            .with_tolerance(1e-8)
            .with_max_iterations(50)
            .with_regularization(1e-8);

        let mut initial = HashMap::new();
        initial.insert(0, 0.5);
        initial.insert(1, 0.25);

        let solution = solver
            .solve_with_line_search(initial)
            .expect("Failed to solve transcendental system");

        assert!(solution.converged);
        let x_sol = solution.values[&0];
        let y_sol = solution.values[&1];
        assert!((x_sol.sin() - 2.0 * y_sol).abs() < 1e-8);
        assert!((y_sol.cos() - x_sol).abs() < 1e-8);
    }

    #[test]
    fn test_solver_errors_on_missing_variable() {
        // System: x - a = 0, solving for x with a treated as a fixed parameter.
        // Missing `a` should be an error (not implicitly treated as 0).
        let x = Exp::var("x".to_string(), 0);
        let a = Exp::var("a".to_string(), 1);
        let eq = Exp::sub(x.clone(), a.clone());

        let solver = NewtonRaphsonSolver::new(vec![eq], vec![0]);

        let mut initial = HashMap::new();
        initial.insert(0, 1.0);

        let err = solver.solve(initial).expect_err("expected missing variable error");
        match err {
            SolverError::MissingVariable(missing) => {
                assert_eq!(missing.var_id, 1);
                assert_eq!(missing.var_name, "a");
            }
            other => panic!("expected MissingVariable, got {:?}", other),
        }
    }

    #[test]
    fn test_solver_invalid_input_on_missing_initial_guess_variable() {
        let x = Exp::var("x".to_string(), 0);
        let y = Exp::var("y".to_string(), 1);
        let eq = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(1.0));

        let solver = NewtonRaphsonSolver::new(vec![eq], vec![0, 1]);

        let mut initial = HashMap::new();
        initial.insert(0, 0.0);

        let err = solver
            .solve(initial)
            .expect_err("expected invalid input due to missing y");
        match err {
            SolverError::InvalidInput(msg) => {
                assert!(msg.contains("[1]"), "unexpected message: {}", msg);
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn test_try_with_equation_traces_invalid_length() {
        let x = Exp::var("x".to_string(), 0);
        let eq = Exp::sub(x.clone(), Exp::val(1.0));

        let solver = NewtonRaphsonSolver::new(vec![eq], vec![0]);

        let traces = vec![
            Some(EquationTrace {
                constraint_id: 0,
                description: "eq0".to_string(),
            }),
            None,
        ];
        let err = match solver.try_with_equation_traces(traces) {
            Ok(_) => panic!("expected invalid input for trace length mismatch"),
            Err(err) => err,
        };
        match err {
            SolverError::InvalidInput(msg) => {
                assert!(msg.contains("Equation trace length (2)"), "{}", msg);
                assert!(msg.contains("number of equations (1)"), "{}", msg);
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn test_line_search_preserves_fixed_variables() {
        // Regression test: line search must preserve "fixed parameter" variables that are
        // not part of `self.variables` but are referenced by equations.
        //
        // If line search drops them, evaluation errors and the solver fails.
        let x = Exp::var("x".to_string(), 0);
        let a = Exp::var("a".to_string(), 1);
        let eq = Exp::sub(x.clone(), a.clone());

        let solver = NewtonRaphsonSolver::new(vec![eq], vec![0]);

        let mut initial = HashMap::new();
        initial.insert(0, 0.0);
        initial.insert(1, 2.0);

        let solution = solver
            .solve_with_line_search(initial)
            .expect("solver should converge when fixed variables are provided");
        assert!(solution.converged);
        assert!((solution.values[&0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_least_squares_qr_handles_ill_conditioned_overdetermined_system_without_regularization()
    {
        // This system is overdetermined (3 equations, 2 unknowns) with an ill-conditioned
        // Jacobian whose columns are nearly linearly dependent. Normal equations (J^T J) can
        // lose the tiny distinguishing term and become singular in floating point. QR-based
        // least squares should still solve it.
        let eps = 2f64.powi(-27); // exactly representable; eps^2 is below 1 ulp at ~3.0

        let x = Exp::var("x".to_string(), 0);
        let y = Exp::var("y".to_string(), 1);

        // A = [[1, 1], [1, 1+eps], [1, 1-eps]]
        // b corresponds to solution (x,y) = (1,1)
        let eq1 = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(2.0));
        let eq2 = Exp::sub(
            Exp::add(x.clone(), Exp::mul(y.clone(), Exp::val(1.0 + eps))),
            Exp::val(2.0 + eps),
        );
        let eq3 = Exp::sub(
            Exp::add(x.clone(), Exp::mul(y.clone(), Exp::val(1.0 - eps))),
            Exp::val(2.0 - eps),
        );

        let solver = NewtonRaphsonSolver::new(vec![eq1, eq2, eq3], vec![0, 1])
            .with_regularization(0.0)
            .with_damping(1.0)
            .with_max_iterations(10)
            .with_tolerance(1e-10);

        let mut initial = HashMap::new();
        initial.insert(0, 0.0);
        initial.insert(1, 0.0);

        let solution = solver
            .solve(initial)
            .expect("expected solver to converge with QR least squares");

        assert!(solution.converged, "{:?}", solution);
        assert!((solution.values[&0] - 1.0).abs() < 1e-8);
        assert!((solution.values[&1] - 1.0).abs() < 1e-8);
    }
}
