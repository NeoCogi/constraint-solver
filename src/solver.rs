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

use crate::compiler::{CompiledExp, CompiledSystem, VarId, VarTable};
use crate::exp::MissingVarError;
use crate::jacobian::Jacobian;
use crate::matrix::{LeastSquaresQrInfo, Matrix, MatrixError};
use crate::mode::{build_thread_pool, Mode};
use rayon::ThreadPool;
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
    equations: Vec<CompiledExp>,
    /// Variable IDs that correspond to unknowns in the system
    variables: Vec<VarId>,
    /// Registry of all variables referenced by the compiled system
    var_table: VarTable,
    /// Maximum number of iterations before giving up
    max_iterations: usize,
    /// Convergence tolerance (solution found when |f(x)| < tolerance)
    tolerance: f64,
    /// Initial damping factor (step size multiplier, 0 < damping <= 1)
    /// Lower values make convergence more stable but slower
    damping_factor: f64,
    /// Minimum allowed damping before declaring convergence failure
    min_damping: f64,
    /// Base regularization parameter used when the system is ill-conditioned
    regularization: f64,
    /// Optional metadata describing the source of each equation in the system
    equation_traces: Vec<Option<EquationTrace>>,
    /// Execution mode for linear algebra (serial vs parallel)
    mode: Mode,
    /// Optional thread pool for parallel execution
    pool: Option<ThreadPool>,
}

/// Solution result containing the solved variable values and convergence info
#[derive(Debug, Clone)]
pub struct Solution {
    /// Final values of all variables (variable name -> value)
    pub values: HashMap<String, f64>,
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
    /// Variable values at the end of the run (variable name -> value)
    pub values: HashMap<String, f64>,
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
    /// * `compiled` - Compiled system of equations (each should evaluate to 0 at solution)
    pub fn new(compiled: CompiledSystem) -> Self {
        let variables = compiled.var_table.all_var_ids();
        Self::build(compiled, variables)
    }

    /// Create a new solver with an explicit execution mode.
    pub fn new_with_mode(compiled: CompiledSystem, mode: Mode) -> Result<Self, SolverError> {
        let variables = compiled.var_table.all_var_ids();
        Self::build_with_mode(compiled, variables, mode)
    }

    /// Create a solver while specifying which variables to solve for.
    ///
    /// Variables not listed here are treated as fixed parameters and must be
    /// provided in the initial guess.
    pub fn new_with_variables(
        compiled: CompiledSystem,
        variables: &[&str],
    ) -> Result<Self, SolverError> {
        let mut ids = Vec::with_capacity(variables.len());
        let mut seen = std::collections::HashSet::new();
        for name in variables {
            let id = compiled
                .var_table
                .get_id(name)
                .ok_or_else(|| SolverError::InvalidInput(format!("Unknown variable '{name}'")))?;
            if seen.insert(id) {
                ids.push(id);
            }
        }
        Ok(Self::build(compiled, ids))
    }

    /// Create a solver with explicit variables and execution mode.
    pub fn new_with_variables_and_mode(
        compiled: CompiledSystem,
        variables: &[&str],
        mode: Mode,
    ) -> Result<Self, SolverError> {
        let mut ids = Vec::with_capacity(variables.len());
        let mut seen = std::collections::HashSet::new();
        for name in variables {
            let id = compiled
                .var_table
                .get_id(name)
                .ok_or_else(|| SolverError::InvalidInput(format!("Unknown variable '{name}'")))?;
            if seen.insert(id) {
                ids.push(id);
            }
        }

        Self::build_with_mode(compiled, ids, mode)
    }

    fn build(compiled: CompiledSystem, variables: Vec<VarId>) -> Self {
        // Allow both under-constrained and over-constrained systems
        // We'll use least squares to solve them
        let is_over_constrained = compiled.equations.len() > variables.len();

        NewtonRaphsonSolver {
            equations: compiled.equations,
            variables,
            var_table: compiled.var_table,
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
            mode: Mode::Serial,
            pool: None,
        }
    }

    fn build_with_mode(
        compiled: CompiledSystem,
        variables: Vec<VarId>,
        mode: Mode,
    ) -> Result<Self, SolverError> {
        let mut solver = Self::build(compiled, variables);
        solver.set_mode(mode)?;
        Ok(solver)
    }

    /// Update the execution mode for this solver.
    pub fn with_mode(mut self, mode: Mode) -> Result<Self, SolverError> {
        self.set_mode(mode)?;
        Ok(self)
    }

    /// Returns the current execution mode.
    pub fn mode(&self) -> &Mode {
        &self.mode
    }

    fn set_mode(&mut self, mode: Mode) -> Result<(), SolverError> {
        let pool = build_thread_pool(&mode).map_err(SolverError::InvalidInput)?;
        self.mode = mode;
        self.pool = pool;
        Ok(())
    }

    fn parallel_enabled(&self) -> bool {
        self.pool.is_some()
    }

    fn run_in_mode<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        if let Some(pool) = &self.pool {
            pool.install(f)
        } else {
            f()
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

    /// Override the base regularization parameter used when the system is ill-conditioned
    pub fn with_regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization.max(0.0);
        self
    }

    /// Solve the system using standard Newton-Raphson method
    pub fn solve(&self, initial_guess: HashMap<String, f64>) -> Result<Solution, SolverError> {
        let vars = self.map_initial_guess(initial_guess)?;
        self.run_in_mode(|| self.solve_modified_internal(vars, false))
    }

    /// Core Newton-Raphson solver implementation with optional line search
    ///
    /// # Arguments
    /// * `initial_guess` - Starting values for all variables (by name)
    /// * `use_line_search` - Whether to use backtracking line search for step size
    ///
    /// # Algorithm
    /// 1. Evaluate f(x) and check for convergence
    /// 2. Compute Jacobian J = df/dx
    /// 3. Solve linear system: J * delta = -f(x)
    ///    - Square systems: Direct LU decomposition
    ///    - Under/over-constrained: QR least squares (with adaptive regularization if needed)
    /// 4. Update: x_new = x_old + damping * delta
    /// 5. Adapt damping based on error change
    fn solve_modified_internal(
        &self,
        initial_guess: HashMap<VarId, f64>,
        use_line_search: bool,
    ) -> Result<Solution, SolverError> {
        self.validate_initial_guess(&initial_guess)?;

        let mut vars = initial_guess.clone();
        let jacobian = Jacobian::new(self.equations.clone(), self.variables.clone());
        // Cache Jacobian storage to avoid per-iteration allocations.
        let mut jacobian_workspace = jacobian.workspace();
        let mut convergence_history = Vec::with_capacity(self.max_iterations);
        // Start with configured damping factor, will be adapted during iterations
        let mut damping = self.damping_factor;
        let mut last_error = f64::INFINITY;

        let num_equations = self.equations.len();
        let num_variables = self.variables.len();

        let mut f_vals = Matrix::new(num_equations, 1);
        let mut f_neg = Matrix::new(num_equations, 1);
        jacobian_workspace.replace_jacobian(jacobian.evaluate_checked(&vars)?);

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
        let parallel = self.parallel_enabled();

        for iter in 0..self.max_iterations {
            // Step 1: Evaluate function values f(x)
            jacobian.evaluate_functions_checked_into(&vars, &mut f_vals)?;
            let error = f_vals.norm();
            convergence_history.push(error);

            // Check for convergence: |f(x)| < tolerance
            if error < self.tolerance {
                return Ok(Solution {
                    values: self.values_by_name(&vars),
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

            for i in 0..num_equations {
                f_neg[(i, 0)] = -f_vals[(i, 0)];
            }

            // Step 2: Solve linear system J * delta = -f(x)
            let delta = {
                let j_matrix = jacobian_workspace.jacobian();
                // Handle different system types with appropriate solving methods
                if j_matrix.rows() == j_matrix.cols() {
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
                                        format!(
                                            "Matrix is singular even with regularization: {e}"
                                        ),
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
                        j_matrix,
                        &f_neg,
                        least_squares_workspace.as_mut(),
                        parallel,
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
                    values: self.values_by_name(&vars),
                    iterations: iter + 1,
                    error,
                    converged: true,
                    convergence_history,
                });
            }

            // Refresh Jacobian for next iteration
            jacobian.evaluate_checked_in_workspace(&vars, &mut jacobian_workspace)?;
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
        parallel: bool,
    ) -> Result<Matrix, LeastSquaresSolveError> {
        let mut workspace = workspace;
        // Prefer QR-based least squares to avoid forming normal equations (J^T J), which can
        // severely amplify conditioning issues. If the QR solve is rank deficient or
        // ill-conditioned, fall back to a regularized augmented system
        // [J; sqrt(lambda) I] * delta ~= [-f; 0].
        const COND_LIMIT: f64 = 1e12;

        let full_rank = j_matrix.rows().min(j_matrix.cols());
        let is_ill_conditioned = |info: &LeastSquaresQrInfo| -> bool {
            info.rank < full_rank
                || !info.cond_est.is_finite()
                || info.cond_est > COND_LIMIT
        };

        let mut qr_err: Option<String> = None;
        let mut unreg_delta: Option<Matrix> = None;
        let mut unreg_info: Option<LeastSquaresQrInfo> = None;

        match j_matrix.solve_least_squares_qr_with_info_with_parallel(f_neg, parallel) {
            Ok((delta, info)) => {
                if !is_ill_conditioned(&info) {
                    return Ok(delta);
                }
                unreg_delta = Some(delta);
                unreg_info = Some(info);
            }
            Err(err) => {
                qr_err = Some(err);
            }
        }

        if self.regularization <= 0.0 {
            if let Some(delta) = unreg_delta {
                return Ok(delta);
            }
            return Err(LeastSquaresSolveError::Singular {
                qr_err: qr_err.unwrap_or_else(|| "QR least squares failed".to_string()),
                normal_eq_err: "Regularization disabled".to_string(),
            });
        }

        let reg_factors = [1.0, 10.0, 100.0];
        let mut last_err: Option<String> = None;
        let mut last_solution: Option<Matrix> = None;

        for factor in reg_factors {
            let lambda = self.regularization * factor;
            if lambda <= 0.0 {
                continue;
            }
            let sqrt_reg = lambda.sqrt();

            let reg_result = match workspace.as_deref_mut() {
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
                        for j in 0..workspace.num_variables {
                            workspace.augmented_j[(workspace.num_equations + i, j)] = 0.0;
                        }
                        workspace.augmented_j[(workspace.num_equations + i, i)] = sqrt_reg;
                        workspace.augmented_b[(workspace.num_equations + i, 0)] = 0.0;
                    }

                    workspace
                        .augmented_j
                        .solve_least_squares_qr_with_info_with_parallel(
                            &workspace.augmented_b,
                            parallel,
                        )
                }
                None => {
                    let mut augmented_j =
                        Matrix::new(j_matrix.rows() + j_matrix.cols(), j_matrix.cols());
                    let mut augmented_b = Matrix::new(j_matrix.rows() + j_matrix.cols(), 1);
                    for i in 0..j_matrix.rows() {
                        for j in 0..j_matrix.cols() {
                            augmented_j[(i, j)] = j_matrix[(i, j)];
                        }
                        augmented_b[(i, 0)] = f_neg[(i, 0)];
                    }
                    for i in 0..j_matrix.cols() {
                        augmented_j[(j_matrix.rows() + i, i)] = sqrt_reg;
                        augmented_b[(j_matrix.rows() + i, 0)] = 0.0;
                    }
                    augmented_j.solve_least_squares_qr_with_info_with_parallel(
                        &augmented_b,
                        parallel,
                    )
                }
            };

            match reg_result {
                Ok((delta, info)) => {
                    last_solution = Some(delta);
                    if !is_ill_conditioned(&info) {
                        return Ok(last_solution.unwrap());
                    }
                }
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }

        if let Some(delta) = last_solution {
            return Ok(delta);
        }

        Err(LeastSquaresSolveError::Singular {
            qr_err: qr_err
                .or_else(|| {
                    unreg_info
                        .map(|info| format!("QR ill-conditioned (rank {}, cond {:.2e})", info.rank, info.cond_est))
                })
                .unwrap_or_else(|| "QR least squares failed".to_string()),
            normal_eq_err: last_err.unwrap_or_else(|| "Regularized QR failed".to_string()),
        })
    }

    fn map_initial_guess(
        &self,
        initial_guess: HashMap<String, f64>,
    ) -> Result<HashMap<VarId, f64>, SolverError> {
        let mut unknown = Vec::new();
        let mut vars = HashMap::new();

        for (name, value) in initial_guess {
            if let Some(id) = self.var_table.get_id(&name) {
                vars.insert(id, value);
            } else {
                unknown.push(name);
            }
        }

        if !unknown.is_empty() {
            unknown.sort();
            return Err(SolverError::InvalidInput(format!(
                "Initial guess includes unknown variables: {unknown:?}"
            )));
        }

        self.validate_initial_guess(&vars)?;
        Ok(vars)
    }

    fn validate_initial_guess(&self, vars: &HashMap<VarId, f64>) -> Result<(), SolverError> {
        let mut missing = Vec::new();
        for (idx, name) in self.var_table.names().iter().enumerate() {
            if !vars.contains_key(&VarId::new(idx)) {
                missing.push(name.clone());
            }
        }
        if missing.is_empty() {
            return Ok(());
        }

        missing.sort();
        Err(SolverError::InvalidInput(format!(
            "Initial guess missing values for variables: {missing:?}"
        )))
    }

    fn values_by_name(&self, vars: &HashMap<VarId, f64>) -> HashMap<String, f64> {
        let mut values = HashMap::with_capacity(self.var_table.len());
        for (idx, name) in self.var_table.names().iter().enumerate() {
            if let Some(value) = vars.get(&VarId::new(idx)) {
                values.insert(name.clone(), *value);
            }
        }
        values
    }

    fn build_diagnostic(
        &self,
        message: String,
        iterations: usize,
        vars: &HashMap<VarId, f64>,
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
            values: self.values_by_name(vars),
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
        vars: &HashMap<VarId, f64>,
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
        initial_guess: HashMap<String, f64>,
    ) -> Result<Solution, SolverError> {
        let vars = self.map_initial_guess(initial_guess)?;
        self.run_in_mode(|| self.solve_modified_internal(vars, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::Compiler;
    use crate::exp::Exp;
    use crate::Mode;

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
    }

    fn linear_equation(coeffs: &[f64], vars: &[Exp], rhs: f64) -> Exp {
        let mut sum = Exp::val(0.0);
        for (coeff, var) in coeffs.iter().zip(vars.iter()) {
            let term = Exp::mul(Exp::val(*coeff), var.clone());
            sum = Exp::add(sum, term);
        }
        Exp::sub(sum, Exp::val(rhs))
    }

    fn test_modes() -> Vec<Mode> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(4)
            .max(1);
        vec![Mode::Serial, Mode::Parallel { thread_count: threads }]
    }

    fn assert_solution_close(serial: &Solution, parallel: &Solution, tol: f64) {
        assert_eq!(serial.values.len(), parallel.values.len());
        for (name, value) in &serial.values {
            let other = parallel.values.get(name).expect("missing variable");
            let diff = (value - other).abs();
            assert!(diff <= tol, "var {name} mismatch: {value} vs {other}");
        }
    }

    fn assert_error_matches(serial: &SolverError, parallel: &SolverError) {
        assert_eq!(
            std::mem::discriminant(serial),
            std::mem::discriminant(parallel)
        );
        match (serial, parallel) {
            (SolverError::InvalidInput(a), SolverError::InvalidInput(b)) => assert_eq!(a, b),
            (SolverError::MissingVariable(a), SolverError::MissingVariable(b)) => {
                assert_eq!(a, b)
            }
            _ => {}
        }
    }

    fn solver_for_mode(equations: Vec<Exp>, mode: Mode) -> NewtonRaphsonSolver {
        let compiled = Compiler::compile(&equations).expect("compile failed");
        NewtonRaphsonSolver::new_with_mode(compiled, mode).expect("failed to set solver mode")
    }

    fn solver_for_with_vars_mode(
        equations: Vec<Exp>,
        variables: &[&str],
        mode: Mode,
    ) -> NewtonRaphsonSolver {
        let compiled = Compiler::compile(&equations).expect("compile failed");
        NewtonRaphsonSolver::new_with_variables_and_mode(compiled, variables, mode)
            .expect("failed to select solve variables or set mode")
    }

    #[test]
    fn test_simple_system() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Test system: x^2 + y^2 = 1, xy = 0.25
            // Solution should be approximately x ~= 0.5, y ~= 0.866 (or vice versa)
            let x = Exp::var("x");
            let y = Exp::var("y");

            let eq1 = Exp::sub(
                Exp::add(Exp::power(x.clone(), 2.0), Exp::power(y.clone(), 2.0)),
                Exp::val(1.0),
            );
            let eq2 = Exp::sub(Exp::mul(x.clone(), y.clone()), Exp::val(0.25));

            let solver = solver_for_mode(vec![eq1, eq2], mode);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.5);
            initial.insert("y".to_string(), 0.866);

            let solution = match solver.solve(initial.clone()) {
                Ok(sol) => sol,
                Err(_) => match solver.solve_with_line_search(initial) {
                    Ok(sol) => sol,
                    Err(e) => panic!("Failed to solve: {:?}", e),
                },
            };
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-8);
            }
            assert!(solution.converged);
            assert!(solution.error < 1e-10);

            let x_sol = solution.values.get("x").copied().unwrap();
            let y_sol = solution.values.get("y").copied().unwrap();
            assert!((x_sol * x_sol + y_sol * y_sol - 1.0).abs() < 1e-10);
            assert!((x_sol * y_sol - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transcendental_system() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Test transcendental system: sin(x) = 2y, cos(y) = x
            let x = Exp::var("x");
            let y = Exp::var("y");

            let eq1 = Exp::sub(Exp::sin(x.clone()), Exp::mul(y.clone(), Exp::val(2.0)));
            let eq2 = Exp::sub(Exp::cos(y.clone()), x.clone());

            let solver = solver_for_mode(vec![eq1, eq2], mode)
                .with_tolerance(1e-8)
                .with_max_iterations(50)
                .with_regularization(1e-8);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.5);
            initial.insert("y".to_string(), 0.25);

            let solution = solver
                .solve_with_line_search(initial)
                .expect("Failed to solve transcendental system");

            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-8);
            }
            assert!(solution.converged);
            let x_sol = solution.values.get("x").copied().unwrap();
            let y_sol = solution.values.get("y").copied().unwrap();
            assert!((x_sol.sin() - 2.0 * y_sol).abs() < 1e-8);
            assert!((y_sol.cos() - x_sol).abs() < 1e-8);
        }
    }

    #[test]
    fn test_solver_errors_on_missing_variable() {
        let mut serial_error: Option<SolverError> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // System: x - a = 0, solving for x with a treated as a fixed parameter.
            // Missing `a` should be an error (not implicitly treated as 0).
            let x = Exp::var("x");
            let a = Exp::var("a");
            let eq = Exp::sub(x.clone(), a.clone());

            let solver = solver_for_with_vars_mode(vec![eq], &["x"], mode);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 1.0);

            let err = solver.solve(initial).expect_err("expected missing variable error");
            if is_serial {
                serial_error = Some(err.clone());
            } else {
                let serial = serial_error.as_ref().expect("serial error missing");
                assert_error_matches(serial, &err);
            }
            match err {
                SolverError::InvalidInput(msg) => {
                    assert!(msg.contains("a"), "unexpected message: {msg}");
                }
                other => panic!("expected InvalidInput, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_solver_invalid_input_on_missing_initial_guess_variable() {
        let mut serial_error: Option<SolverError> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            let x = Exp::var("x");
            let y = Exp::var("y");
            let eq = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(1.0));

            let solver = solver_for_mode(vec![eq], mode);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);

            let err = solver
                .solve(initial)
                .expect_err("expected invalid input due to missing y");
            if is_serial {
                serial_error = Some(err.clone());
            } else {
                let serial = serial_error.as_ref().expect("serial error missing");
                assert_error_matches(serial, &err);
            }
            match err {
                SolverError::InvalidInput(msg) => {
                    assert!(msg.contains("y"), "unexpected message: {msg}");
                }
                other => panic!("expected InvalidInput, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_try_with_equation_traces_invalid_length() {
        let mut serial_error: Option<SolverError> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            let x = Exp::var("x");
            let eq = Exp::sub(x.clone(), Exp::val(1.0));

            let solver = solver_for_mode(vec![eq], mode);

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
            if is_serial {
                serial_error = Some(err.clone());
            } else {
                let serial = serial_error.as_ref().expect("serial error missing");
                assert_error_matches(serial, &err);
            }
            match err {
                SolverError::InvalidInput(msg) => {
                    assert!(msg.contains("Equation trace length (2)"), "{}", msg);
                    assert!(msg.contains("number of equations (1)"), "{}", msg);
                }
                other => panic!("expected InvalidInput, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_line_search_preserves_fixed_variables() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Regression test: line search must preserve "fixed parameter" variables that are
            // not part of `self.variables` but are referenced by equations.
            //
            // If line search drops them, evaluation errors and the solver fails.
            let x = Exp::var("x");
            let a = Exp::var("a");
            let eq = Exp::sub(x.clone(), a.clone());

            let solver = solver_for_with_vars_mode(vec![eq], &["x"], mode);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);
            initial.insert("a".to_string(), 2.0);

            let solution = solver
                .solve_with_line_search(initial)
                .expect("solver should converge when fixed variables are provided");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert!(solution.converged);
            assert!((solution.values.get("x").copied().unwrap() - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_least_squares_qr_handles_ill_conditioned_overdetermined_system_without_regularization()
    {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // This system is overdetermined (3 equations, 2 unknowns) with an ill-conditioned
            // Jacobian whose columns are nearly linearly dependent. Normal equations (J^T J) can
            // lose the tiny distinguishing term and become singular in floating point. QR-based
            // least squares should still solve it.
            let eps = 2f64.powi(-27); // exactly representable; eps^2 is below 1 ulp at ~3.0

            let x = Exp::var("x");
            let y = Exp::var("y");

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

            let solver = solver_for_mode(vec![eq1, eq2, eq3], mode)
                .with_regularization(0.0)
                .with_damping(1.0)
                .with_max_iterations(10)
                .with_tolerance(1e-10);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);
            initial.insert("y".to_string(), 0.0);

            let solution = solver
                .solve(initial)
                .expect("expected solver to converge with QR least squares");

            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-8);
            }
            assert!(solution.converged, "{:?}", solution);
            assert!((solution.values.get("x").copied().unwrap() - 1.0).abs() < 1e-8);
            assert!((solution.values.get("y").copied().unwrap() - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_underconstrained_system_returns_min_norm_solution() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Underdetermined system: x + y = 1 has infinitely many solutions.
            // The solver should return the minimum-norm solution: x = 0.5, y = 0.5.
            let x = Exp::var("x");
            let y = Exp::var("y");

            let eq = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(1.0));
            let solver = solver_for_mode(vec![eq], mode)
                .with_tolerance(1e-12)
                .with_max_iterations(10);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);
            initial.insert("y".to_string(), 0.0);

            let solution = solver.solve(initial).expect("expected solver to converge");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert!(solution.converged, "{:?}", solution);
            assert!((solution.values.get("x").copied().unwrap() - 0.5).abs() < 1e-10);
            assert!((solution.values.get("y").copied().unwrap() - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rank_deficient_overconstrained_recovers_with_regularization() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Equations depend only on x; y is unconstrained. The Jacobian is rank deficient.
            let x = Exp::var("x");
            let y = Exp::var("y");
            let y_zero = Exp::mul(y.clone(), Exp::val(0.0));

            let eq1 = Exp::sub(Exp::add(x.clone(), y_zero.clone()), Exp::val(1.0));
            let eq2 = Exp::sub(
                Exp::add(Exp::mul(x.clone(), Exp::val(2.0)), y_zero.clone()),
                Exp::val(2.0),
            );
            let eq3 = Exp::sub(
                Exp::add(Exp::mul(x.clone(), Exp::val(3.0)), y_zero.clone()),
                Exp::val(3.0),
            );

            let solver = solver_for_mode(vec![eq1, eq2, eq3], mode)
                .with_tolerance(1e-12)
                .with_max_iterations(10);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);
            initial.insert("y".to_string(), 0.0);

            let solution = solver.solve(initial).expect("expected solver to converge");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert!(solution.converged, "{:?}", solution);
            assert!((solution.values.get("x").copied().unwrap() - 1.0).abs() < 1e-10);
            assert!((solution.values.get("y").copied().unwrap() - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rank_deficient_overconstrained_fails_without_regularization() {
        let mut serial_error: Option<SolverError> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            let x = Exp::var("x");
            let y = Exp::var("y");
            let y_zero = Exp::mul(y.clone(), Exp::val(0.0));

            let eq1 = Exp::sub(Exp::add(x.clone(), y_zero.clone()), Exp::val(1.0));
            let eq2 = Exp::sub(
                Exp::add(Exp::mul(x.clone(), Exp::val(2.0)), y_zero.clone()),
                Exp::val(2.0),
            );
            let eq3 = Exp::sub(
                Exp::add(Exp::mul(x.clone(), Exp::val(3.0)), y_zero.clone()),
                Exp::val(3.0),
            );

            let solver = solver_for_mode(vec![eq1, eq2, eq3], mode)
                .with_regularization(0.0)
                .with_tolerance(1e-12)
                .with_max_iterations(10);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);
            initial.insert("y".to_string(), 0.0);

            let err = solver.solve(initial).expect_err("expected singular matrix error");
            if is_serial {
                serial_error = Some(err.clone());
            } else {
                let serial = serial_error.as_ref().expect("serial error missing");
                assert_error_matches(serial, &err);
            }
            assert!(matches!(err, SolverError::SingularMatrix(_)), "{:?}", err);
        }
    }

    #[test]
    fn test_overconstrained_consistent_system_solves() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Overdetermined but consistent system: x = 1 and 2x = 2.
            let x = Exp::var("x");
            let eq1 = Exp::sub(x.clone(), Exp::val(1.0));
            let eq2 = Exp::sub(Exp::mul(x.clone(), Exp::val(2.0)), Exp::val(2.0));

            let solver = solver_for_mode(vec![eq1, eq2], mode)
                .with_tolerance(1e-12)
                .with_max_iterations(10);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);

            let solution = solver.solve(initial).expect("expected solver to converge");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert!(solution.converged, "{:?}", solution);
            assert!((solution.values.get("x").copied().unwrap() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_random_square_linear_systems() {
        let mut serial_solutions: Vec<Solution> = Vec::new();
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            let mut rng = TestRng::new(0x51ab_1e55_cafe_f00d);
            for case_index in 0..5 {
                let n = 3;
                let mut a = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        a[i][j] = rng.next_f64();
                    }
                    a[i][i] += 2.0;
                }

                let x_true = vec![rng.next_f64(), rng.next_f64(), rng.next_f64()];
                let mut b = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        b[i] += a[i][j] * x_true[j];
                    }
                }

                let vars: Vec<Exp> = (0..n).map(|i| Exp::var(format!("x{i}"))).collect();
                let equations: Vec<Exp> = (0..n)
                    .map(|i| linear_equation(&a[i], &vars, b[i]))
                    .collect();

                let solver = solver_for_mode(equations, mode.clone())
                    .with_tolerance(1e-12)
                    .with_max_iterations(20);

                let mut initial = HashMap::new();
                for i in 0..n {
                    initial.insert(format!("x{i}"), 0.0);
                }

                let solution = solver.solve(initial).expect("expected to solve");
                if is_serial {
                    serial_solutions.push(solution.clone());
                } else {
                    assert_solution_close(&serial_solutions[case_index], &solution, 1e-8);
                }
                for i in 0..n {
                    let value = solution.values.get(&format!("x{i}")).copied().unwrap();
                    assert!((value - x_true[i]).abs() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn test_random_overconstrained_linear_systems() {
        let mut serial_solutions: Vec<Solution> = Vec::new();
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            let mut rng = TestRng::new(0xa11c_e551_dead_beef);
            let m = 5;
            let n = 3;

            for case_index in 0..5 {
                let mut a = vec![vec![0.0; n]; m];
                for i in 0..m {
                    for j in 0..n {
                        a[i][j] = rng.next_f64();
                    }
                }
                for i in 0..n {
                    a[i][i] += 2.0;
                }

                let x_true = vec![rng.next_f64(), rng.next_f64(), rng.next_f64()];
                let mut b = vec![0.0; m];
                for i in 0..m {
                    for j in 0..n {
                        b[i] += a[i][j] * x_true[j];
                    }
                }

                let vars: Vec<Exp> = (0..n).map(|i| Exp::var(format!("x{i}"))).collect();
                let equations: Vec<Exp> = (0..m)
                    .map(|i| linear_equation(&a[i], &vars, b[i]))
                    .collect();

                let solver = solver_for_mode(equations, mode.clone())
                    .with_tolerance(1e-12)
                    .with_max_iterations(20);

                let mut initial = HashMap::new();
                for i in 0..n {
                    initial.insert(format!("x{i}"), 0.0);
                }

                let solution = solver.solve(initial).expect("expected to solve");
                if is_serial {
                    serial_solutions.push(solution.clone());
                } else {
                    assert_solution_close(&serial_solutions[case_index], &solution, 1e-8);
                }
                for i in 0..n {
                    let value = solution.values.get(&format!("x{i}")).copied().unwrap();
                    assert!((value - x_true[i]).abs() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn test_random_underdetermined_min_norm_structure() {
        let mut serial_solutions: Vec<Solution> = Vec::new();
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            let mut rng = TestRng::new(0x0ddc_affe_fade_bead);
            let _m = 2;
            let n = 4;
            let vars: Vec<Exp> = (0..n).map(|i| Exp::var(format!("x{i}"))).collect();

            for case_index in 0..5 {
                let b0 = rng.next_f64();
                let b1 = rng.next_f64();
                let x2_zero = Exp::mul(vars[2].clone(), Exp::val(0.0));
                let x3_zero = Exp::mul(vars[3].clone(), Exp::val(0.0));
                let eq1 = Exp::sub(Exp::add(vars[0].clone(), x2_zero.clone()), Exp::val(b0));
                let eq2 = Exp::sub(Exp::add(vars[1].clone(), x3_zero.clone()), Exp::val(b1));

                let solver = solver_for_mode(vec![eq1, eq2], mode.clone())
                    .with_tolerance(1e-12)
                    .with_max_iterations(20);

                let mut initial = HashMap::new();
                for i in 0..n {
                    initial.insert(format!("x{i}"), 0.0);
                }

                let solution = solver.solve(initial).expect("expected to solve");
                if is_serial {
                    serial_solutions.push(solution.clone());
                } else {
                    assert_solution_close(&serial_solutions[case_index], &solution, 1e-10);
                }
                assert!((solution.values.get("x0").copied().unwrap() - b0).abs() < 1e-10);
                assert!((solution.values.get("x1").copied().unwrap() - b1).abs() < 1e-10);
                assert!(solution.values.get("x2").copied().unwrap().abs() < 1e-10);
                assert!(solution.values.get("x3").copied().unwrap().abs() < 1e-10);
            }
        }
    }
}
