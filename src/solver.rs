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

/// Selects how an underdetermined Newton iteration resolves its null-space
/// freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnderdeterminedPolicy {
    /// Solve `J * delta = -f` for the minimum-norm correction. This preserves
    /// the current point's null-space component and is usually the smoothest
    /// policy for interactive CAD and inverse-kinematics updates.
    MinimumNormStep,
    /// Solve `J * x_next = J * x_current - f` for the minimum-norm next point.
    /// For a consistent linear system this returns the global minimum-norm
    /// solution independently of the initial guess. For nonlinear systems the
    /// guarantee applies to each local linearization, not the global manifold.
    MinimumNormLinearizedSolution,
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
    /// Policy used to resolve null-space freedom when equations are fewer than
    /// solve variables.
    underdetermined_policy: UnderdeterminedPolicy,
    /// Optional metadata describing the source of each equation in the system
    equation_traces: Vec<Option<EquationTrace>>,
    /// Execution mode for linear algebra (serial vs parallel)
    mode: Mode,
    /// Optional thread pool for parallel execution
    pool: Option<ThreadPool>,
}

/// Describes the mathematical condition that caused a successful solve to
/// terminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// The Euclidean residual norm satisfied the requested root tolerance.
    ResidualTolerance,
    /// An overdetermined system reached first-order least-squares stationarity
    /// with a small step, even though its residual could not be driven to zero.
    LeastSquaresStationary,
}

/// Successful solver result with values and explicit convergence diagnostics.
///
/// The solver returns failures as `SolverError`, so there is no redundant
/// boolean success field. Callers can inspect `reason` to distinguish an
/// approximate root from a stationary overdetermined least-squares result.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Final values of every compiled variable, including fixed parameters.
    pub values: HashMap<String, f64>,
    /// Number of accepted Newton updates. An already-solved initial guess
    /// therefore reports zero iterations.
    pub iterations: usize,
    /// Residual norm evaluated at exactly the values returned in `values`.
    pub error: f64,
    /// Norm of `J^T f` at the returned values, measuring first-order
    /// least-squares stationarity.
    pub gradient_norm: f64,
    /// Mathematical convergence test that accepted the returned values.
    pub reason: ConvergenceReason,
    /// Residual norms for the initial point and each accepted update through the
    /// returned point.
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
    /// Expression, Jacobian, update, or residual-gradient evaluation produced
    /// NaN or infinity.
    NonFiniteEvaluation(SolverRunDiagnostic),
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
            // Preserve the prior continuity-oriented behavior unless the caller
            // explicitly requests origin-based minimum-norm linearizations.
            underdetermined_policy: UnderdeterminedPolicy::MinimumNormStep,
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

    /// Override the maximum number of accepted Newton updates.
    ///
    /// Configuration is validated when a solve begins; zero is rejected as
    /// `SolverError::InvalidInput`.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        // Preserve the caller's exact value so validation can report invalid
        // input rather than silently substituting a different iteration limit.
        self.max_iterations = max_iterations;
        self
    }

    /// Override the positive, finite residual/step/stationarity tolerance.
    ///
    /// Configuration is validated when a solve begins.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        // Store without clamping so NaN, infinity, zero, and negative values are
        // observable to the common configuration validator.
        self.tolerance = tolerance;
        self
    }

    /// Override the finite initial damping factor in the interval `(0, 1]`.
    ///
    /// Standard solves use this value for their first accepted update and adapt
    /// it only after observing a later residual. Line-search solves use it as
    /// the largest initial trial step on every iteration. Configuration is
    /// validated when a solve begins; invalid values are not silently clamped.
    pub fn with_damping(mut self, damping_factor: f64) -> Self {
        // Preserve the requested value for deterministic validation and error
        // reporting at the shared solve entry point.
        self.damping_factor = damping_factor;
        self
    }

    /// Override the finite non-negative regularization parameter used when the
    /// system is ill-conditioned.
    ///
    /// Configuration is validated when a solve begins; negative values are not
    /// converted to zero implicitly.
    pub fn with_regularization(mut self, regularization: f64) -> Self {
        // Retain non-finite and negative values until validation so the caller
        // receives an explicit configuration error.
        self.regularization = regularization;
        self
    }

    /// Select how underdetermined iterations resolve their Jacobian null space.
    ///
    /// This setting has no effect on square or overdetermined systems.
    pub fn with_underdetermined_policy(
        mut self,
        policy: UnderdeterminedPolicy,
    ) -> Self {
        // Store the policy on the reusable solver so every solve invocation has
        // deterministic null-space behavior.
        self.underdetermined_policy = policy;
        self
    }

    /// Solve the system using standard Newton-Raphson method
    pub fn solve(&self, initial_guess: HashMap<String, f64>) -> Result<Solution, SolverError> {
        // Reject invalid reusable settings before translating or validating the
        // caller's variable map.
        self.validate_configuration()?;
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
        // Defensively repeat validation at the internal boundary so future
        // private entry points cannot bypass the public checks.
        self.validate_configuration()?;
        self.validate_initial_guess(&initial_guess)?;

        let mut vars = initial_guess.clone();
        let jacobian = Jacobian::new(self.equations.clone(), self.variables.clone());
        // Cache Jacobian storage to avoid per-iteration allocations.
        let mut jacobian_workspace = jacobian.workspace();
        // Store the initial residual plus at most one residual for every
        // accepted update.
        let mut convergence_history =
            Vec::with_capacity(self.max_iterations.saturating_add(1));
        // Standard solves begin with exactly the configured damping factor. A
        // previous residual is optional so the first iteration cannot pretend
        // it observed an improvement from an artificial infinity sentinel.
        let mut damping = self.damping_factor;
        let mut last_error: Option<f64> = None;

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
            if !f_vals.all_finite() {
                return Err(self.non_finite_evaluation_error(
                    "Residual evaluation produced NaN or infinity",
                    iter,
                    &vars,
                    &f_vals,
                ));
            }
            if !jacobian_workspace.jacobian().all_finite() {
                return Err(self.non_finite_evaluation_error(
                    "Jacobian evaluation produced NaN or infinity",
                    iter,
                    &vars,
                    &f_vals,
                ));
            }
            let error = f_vals.norm();
            convergence_history.push(error);

            // Check for convergence: |f(x)| < tolerance
            if error < self.tolerance {
                // The Jacobian workspace already corresponds to `vars`, so the
                // reported gradient norm is evaluated at the same point as the
                // residual and returned values.
                let gradient_norm = self.residual_gradient_norm(
                    jacobian_workspace.jacobian(),
                    &f_vals,
                    parallel,
                    &vars,
                    iter,
                )?;
                return Ok(self.build_solution(
                    &vars,
                    iter,
                    error,
                    gradient_norm,
                    ConvergenceReason::ResidualTolerance,
                    convergence_history,
                ));
            }

            // Adaptive damping belongs only to the standard solver. Line search
            // has its own evaluated acceptance and exhaustion rules, so mutating
            // an unused damping value must not influence its stagnation outcome.
            if !use_line_search {
                if let Some(previous_error) = last_error {
                    // A negligible residual change reduces the next standard
                    // step. Repeated reductions below the supported minimum are
                    // reported as stagnation.
                    if (error - previous_error).abs() < self.tolerance * 0.1 {
                        damping *= 0.5;
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

                    // Compare only two real consecutive residuals. This keeps
                    // the caller's configured damping unchanged for the first
                    // update and adapts it for subsequent updates.
                    if error > previous_error * 1.5 {
                        damping *= 0.5;
                    } else if error < previous_error * 0.5 {
                        damping = (damping * 1.2).min(1.0);
                    }
                }

                // Retain the current residual for a genuine consecutive
                // comparison on the next standard iteration. Line search does
                // not maintain this unused state.
                last_error = Some(error);
            }

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
                    // Wide systems need an explicit null-space policy; tall
                    // systems always solve the ordinary least-squares Newton
                    // correction.
                    let least_squares_result = if j_matrix.rows() < j_matrix.cols() {
                        self.solve_underdetermined_delta(
                            j_matrix,
                            &f_vals,
                            &f_neg,
                            &vars,
                            least_squares_workspace.as_mut(),
                            parallel,
                        )
                    } else {
                        self.solve_least_squares_system(
                            j_matrix,
                            &f_neg,
                            least_squares_workspace.as_mut(),
                            parallel,
                        )
                    };

                    match least_squares_result {
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

            if !delta.all_finite() {
                return Err(self.non_finite_evaluation_error(
                    "Linear solve produced a non-finite Newton correction",
                    iter + 1,
                    &vars,
                    &f_vals,
                ));
            }

            // Step 4: Determine step size (damping or line search)
            let step_size = if use_line_search {
                // Use backtracking line search to find a step that has actually
                // demonstrated sufficient residual reduction. Passing the
                // current residuals lets an exhausted search report the state
                // at which it failed rather than the final rejected candidate.
                self.line_search(
                    &jacobian,
                    &vars,
                    &delta,
                    error,
                    iter + 1,
                    &f_vals,
                    &mut line_search_f_vals,
                )?
            } else {
                // Use current damping factor as step size
                damping
            };

            // Step 5: Update variables: x_new = x_old + step_size * delta.
            // Retain the applied step norm before refreshing numerical state so
            // a tiny update can be diagnosed without being treated as success
            // on its own.
            let applied_step_norm = delta.norm() * step_size;
            for (i, &var_id) in self.variables.iter().enumerate() {
                vars.insert(var_id, vars[&var_id] + step_size * delta[(i, 0)]);
            }

            // Refresh both the residual and Jacobian at the accepted point.
            // Any convergence result below must describe this new state, not
            // the residual that preceded the update.
            jacobian.evaluate_functions_checked_into(&vars, &mut f_vals)?;
            if !f_vals.all_finite() {
                return Err(self.non_finite_evaluation_error(
                    "Accepted Newton update produced non-finite residuals",
                    iter + 1,
                    &vars,
                    &f_vals,
                ));
            }
            jacobian.evaluate_checked_in_workspace(&vars, &mut jacobian_workspace)?;
            if !jacobian_workspace.jacobian().all_finite() {
                return Err(self.non_finite_evaluation_error(
                    "Accepted Newton update produced a non-finite Jacobian",
                    iter + 1,
                    &vars,
                    &f_vals,
                ));
            }
            let updated_error = f_vals.norm();

            if updated_error < self.tolerance || applied_step_norm < self.tolerance {
                // A terminating candidate belongs in the history even though a
                // subsequent loop iteration will not push it.
                convergence_history.push(updated_error);
                let gradient_norm = self.residual_gradient_norm(
                    jacobian_workspace.jacobian(),
                    &f_vals,
                    parallel,
                    &vars,
                    iter + 1,
                )?;

                if updated_error < self.tolerance {
                    return Ok(self.build_solution(
                        &vars,
                        iter + 1,
                        updated_error,
                        gradient_norm,
                        ConvergenceReason::ResidualTolerance,
                        convergence_history,
                    ));
                }

                // A non-zero least-squares residual is an expected outcome only
                // for overdetermined systems. Require both a small accepted step
                // and a small J^T f norm so regularization or poor scaling alone
                // cannot manufacture convergence.
                if num_equations > num_variables && gradient_norm < self.tolerance {
                    return Ok(self.build_solution(
                        &vars,
                        iter + 1,
                        updated_error,
                        gradient_norm,
                        ConvergenceReason::LeastSquaresStationary,
                        convergence_history,
                    ));
                }

                let diagnostic = self.build_diagnostic(
                    format!(
                        "Newton step fell below tolerance at iteration {} without satisfying residual or least-squares stationarity criteria",
                        iter + 1
                    ),
                    iter + 1,
                    &vars,
                    &f_vals,
                );
                return Err(SolverError::NoConvergence(diagnostic));
            }
        }

        // Failed to converge within max iterations
        let residuals = jacobian.evaluate_functions_checked(&vars)?;
        if !residuals.all_finite() {
            return Err(self.non_finite_evaluation_error(
                "Final residual evaluation produced NaN or infinity",
                self.max_iterations,
                &vars,
                &residuals,
            ));
        }
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

    /// Compute the Newton correction for a wide Jacobian according to the
    /// configured underdetermined null-space policy.
    ///
    /// Both policies use the same QR and regularization machinery; they differ
    /// only in whether QR minimizes the correction vector or the next solution
    /// vector of the local affine constraint.
    fn solve_underdetermined_delta(
        &self,
        j_matrix: &Matrix,
        residuals: &Matrix,
        negative_residuals: &Matrix,
        vars: &HashMap<VarId, f64>,
        workspace: Option<&mut LeastSquaresWorkspace>,
        parallel: bool,
    ) -> Result<Matrix, LeastSquaresSolveError> {
        match self.underdetermined_policy {
            UnderdeterminedPolicy::MinimumNormStep => {
                // The wide QR solver returns the smallest correction satisfying
                // the linearized equation J*delta=-f. Adding it to the current
                // point preserves all existing Jacobian-null-space components.
                self.solve_least_squares_system(
                    j_matrix,
                    negative_residuals,
                    workspace,
                    parallel,
                )
            }
            UnderdeterminedPolicy::MinimumNormLinearizedSolution => {
                // Assemble the current solve-variable vector in exactly the
                // column order used by the Jacobian.
                let mut current_values = Matrix::new(self.variables.len(), 1);
                for (index, &var_id) in self.variables.iter().enumerate() {
                    current_values[(index, 0)] = vars[&var_id];
                }

                // Linearizing f(x_next)=0 about x_current gives
                // J*x_next = J*x_current - f. Solving this wide system directly
                // removes arbitrary null-space content from the next point.
                let jacobian_times_current = j_matrix
                    .try_mul_with_parallel(&current_values, parallel)
                    .map_err(LeastSquaresSolveError::InvalidInput)?;
                let projected_rhs = jacobian_times_current
                    .try_sub(residuals)
                    .map_err(LeastSquaresSolveError::InvalidInput)?;
                let next_values = self.solve_least_squares_system(
                    j_matrix,
                    &projected_rhs,
                    workspace,
                    parallel,
                )?;

                // The surrounding update loop consumes a correction, so convert
                // the projected next point back into delta form.
                next_values
                    .try_sub(&current_values)
                    .map_err(LeastSquaresSolveError::InvalidInput)
            }
        }
    }

    /// Solve a possibly rectangular linear system against one right-hand side,
    /// applying adaptive regularization when QR diagnostics indicate numerical
    /// rank loss or excessive conditioning.
    ///
    /// The returned vector is intentionally named generically: callers use this
    /// routine for both Newton corrections and projected next-point values.
    fn solve_least_squares_system(
        &self,
        j_matrix: &Matrix,
        rhs: &Matrix,
        workspace: Option<&mut LeastSquaresWorkspace>,
        parallel: bool,
    ) -> Result<Matrix, LeastSquaresSolveError> {
        let mut workspace = workspace;
        // Prefer QR-based least squares to avoid forming normal equations
        // (J^T J), which can severely amplify conditioning issues. If the QR
        // solve is rank deficient or ill-conditioned, fall back to the generic
        // regularized augmented system [J; sqrt(lambda) I] * x ~= [rhs; 0].
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

        match j_matrix.solve_least_squares_qr_with_info_with_parallel(rhs, parallel) {
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
                        workspace.augmented_b[(i, 0)] = rhs[(i, 0)];
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
                        augmented_b[(i, 0)] = rhs[(i, 0)];
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
        let mut non_finite = Vec::new();
        let mut vars = HashMap::new();

        for (name, value) in initial_guess {
            if let Some(id) = self.var_table.get_id(&name) {
                if value.is_finite() {
                    // Only validated finite values are allowed into the compact
                    // internal variable map used by expression evaluation.
                    vars.insert(id, value);
                } else {
                    non_finite.push(name);
                }
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

        if !non_finite.is_empty() {
            // Sort names to make diagnostics deterministic despite HashMap's
            // intentionally unspecified iteration order.
            non_finite.sort();
            return Err(SolverError::InvalidInput(format!(
                "Initial guess includes non-finite values for variables: {non_finite:?}"
            )));
        }

        self.validate_initial_guess(&vars)?;
        Ok(vars)
    }

    /// Validate solver settings that must be finite and meaningful before
    /// expression evaluation or iteration begins.
    fn validate_configuration(&self) -> Result<(), SolverError> {
        // A zero update budget cannot even test an accepted Newton step and used
        // to turn exact roots into misleading no-convergence diagnostics.
        if self.max_iterations == 0 {
            return Err(SolverError::InvalidInput(
                "max_iterations must be greater than zero".to_string(),
            ));
        }

        // The same tolerance controls residual, step, and stationarity tests;
        // it must therefore be strictly positive and finite.
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(SolverError::InvalidInput(
                "tolerance must be finite and greater than zero".to_string(),
            ));
        }

        // Damping represents a fraction of a Newton correction. Values outside
        // `(0, 1]` either reverse/amplify the step or eliminate it entirely.
        if !self.damping_factor.is_finite()
            || self.damping_factor <= 0.0
            || self.damping_factor > 1.0
        {
            return Err(SolverError::InvalidInput(
                "damping_factor must be finite and in the interval (0, 1]".to_string(),
            ));
        }

        // Regularization is later square-rooted for augmented QR, so negative or
        // non-finite values must not reach workspace construction.
        if !self.regularization.is_finite() || self.regularization < 0.0 {
            return Err(SolverError::InvalidInput(
                "regularization must be finite and non-negative".to_string(),
            ));
        }

        Ok(())
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

    /// Construct a successful result from numerical quantities that have all
    /// been evaluated at the same accepted variable state.
    ///
    /// Centralizing result construction prevents future convergence branches
    /// from accidentally pairing updated variables with stale residual data.
    fn build_solution(
        &self,
        vars: &HashMap<VarId, f64>,
        iterations: usize,
        error: f64,
        gradient_norm: f64,
        reason: ConvergenceReason,
        convergence_history: Vec<f64>,
    ) -> Solution {
        // Convert internal IDs only after all numerical convergence checks have
        // succeeded, keeping the hot iteration path in its compact ID form.
        Solution {
            values: self.values_by_name(vars),
            iterations,
            error,
            gradient_norm,
            reason,
            convergence_history,
        }
    }

    /// Compute the first-order least-squares optimality measure `||J^T f||`.
    ///
    /// A small Newton step is not sufficient evidence of convergence: it can
    /// also result from regularization, scaling, or numerical cancellation.
    /// The residual gradient supplies the additional stationarity condition
    /// needed for a non-zero overdetermined least-squares residual.
    fn residual_gradient_norm(
        &self,
        jacobian: &Matrix,
        residuals: &Matrix,
        parallel: bool,
        vars: &HashMap<VarId, f64>,
        iterations: usize,
    ) -> Result<f64, SolverError> {
        // Form J^T using the same execution policy as the surrounding solve,
        // then multiply it by the current residual column vector.
        let jacobian_transpose = jacobian.transpose_with_parallel(parallel);
        let gradient = jacobian_transpose.try_mul_with_parallel(residuals, parallel)?;
        if !gradient.all_finite() {
            return Err(self.non_finite_evaluation_error(
                "Residual-gradient evaluation produced NaN or infinity",
                iterations,
                vars,
                residuals,
            ));
        }
        Ok(gradient.norm())
    }

    /// Build the dedicated error variant for non-finite numerical states while
    /// preserving the current values and residual vector for diagnosis.
    fn non_finite_evaluation_error(
        &self,
        message: impl Into<String>,
        iterations: usize,
        vars: &HashMap<VarId, f64>,
        residuals: &Matrix,
    ) -> SolverError {
        // `build_diagnostic` intentionally preserves a NaN or infinite residual
        // norm; hiding it behind a finite sentinel would discard useful failure
        // evidence.
        SolverError::NonFiniteEvaluation(self.build_diagnostic(
            message.into(),
            iterations,
            vars,
            residuals,
        ))
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
    /// * `iteration` - One-based iteration number used in failure diagnostics
    /// * `current_residuals` - Residual vector at the unmodified current point
    /// * `candidate_residuals` - Reusable storage for each trial point
    ///
    /// # Returns
    /// An accepted step size alpha such that `x_new = x + alpha * delta`
    /// satisfies the sufficient-decrease condition. If no tested step is
    /// acceptable, the method returns `NoConvergence`; it never substitutes an
    /// untested or previously rejected fallback step.
    fn line_search(
        &self,
        jacobian: &Jacobian,
        vars: &HashMap<VarId, f64>,
        delta: &Matrix,
        current_error: f64,
        iteration: usize,
        current_residuals: &Matrix,
        candidate_residuals: &mut Matrix,
    ) -> Result<f64, SolverError> {
        // A difficult but recoverable Newton direction can require far more
        // than twenty halvings. Sixty-four trials cover the useful binary range
        // down to the explicit minimum without making the search unbounded.
        const MAX_LINE_SEARCH_TRIALS: usize = 64;
        const MIN_LINE_SEARCH_STEP: f64 = 1e-12;
        const SUFFICIENT_DECREASE_FRACTION: f64 = 0.5;

        // Begin with the caller's configured maximum step and reuse a cloned
        // variable map for all candidates so fixed parameters remain present on
        // every trial.
        let mut alpha = self.damping_factor;
        let mut new_vars = vars.clone();
        let mut attempted_trials = 0;

        for _ in 0..MAX_LINE_SEARCH_TRIALS {
            attempted_trials += 1;
            new_vars.clone_from(vars);

            // Compute the candidate values without mutating the accepted solver
            // state. Only solve variables move; fixed parameters are copied.
            for (i, &var_id) in self.variables.iter().enumerate() {
                new_vars.insert(var_id, vars[&var_id] + alpha * delta[(i, 0)]);
            }

            // Non-finite candidate residuals are rejected by the finite Armijo
            // comparison and backtracked just like any other unusable trial.
            // This permits a large exponential step to recover at a smaller
            // alpha instead of aborting at its first overflow.
            jacobian.evaluate_functions_checked_into(&new_vars, candidate_residuals)?;
            let new_error = candidate_residuals.norm();
            let required_error =
                current_error * (1.0 - SUFFICIENT_DECREASE_FRACTION * alpha);
            if new_error.is_finite() && new_error < required_error {
                return Ok(alpha);
            }

            // Stop only after the current minimum-sized candidate has actually
            // been tested and rejected. This ensures every returned success is
            // backed by an evaluated candidate.
            if alpha <= MIN_LINE_SEARCH_STEP {
                break;
            }
            alpha = (alpha * 0.5).max(MIN_LINE_SEARCH_STEP);
        }

        // Applying any rejected fallback could increase the residual or leave
        // the expression domain. Preserve the current point and provide a
        // diagnostic that identifies line-search exhaustion instead.
        let diagnostic = self.build_diagnostic(
            format!(
                "Line search failed after {attempted_trials} rejected trial steps; smallest tested step was {alpha:.2e}"
            ),
            iteration,
            vars,
            current_residuals,
        );
        Err(SolverError::NoConvergence(diagnostic))
    }

    /// Solve the system using Newton-Raphson with line search
    ///
    /// Line search helps improve convergence robustness by automatically
    /// finding good step sizes, especially useful for difficult systems.
    pub fn solve_with_line_search(
        &self,
        initial_guess: HashMap<String, f64>,
    ) -> Result<Solution, SolverError> {
        // Apply exactly the same configuration contract as the standard solve
        // before mapping caller-visible variable names to internal IDs.
        self.validate_configuration()?;
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
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
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
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
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

    /// Verify that every externally configurable numerical parameter enforces
    /// its documented finite range before iteration begins.
    #[test]
    fn test_solver_rejects_invalid_configuration() {
        // Each case constructs a fresh solver because builder methods consume
        // and return the solver by value.
        let invalid_cases: Vec<(&str, Box<dyn Fn(NewtonRaphsonSolver) -> NewtonRaphsonSolver>)> =
            vec![
                ("max_iterations", Box::new(|solver| solver.with_max_iterations(0))),
                ("tolerance", Box::new(|solver| solver.with_tolerance(0.0))),
                ("tolerance", Box::new(|solver| solver.with_tolerance(f64::NAN))),
                ("damping_factor", Box::new(|solver| solver.with_damping(0.0))),
                ("damping_factor", Box::new(|solver| solver.with_damping(1.1))),
                (
                    "damping_factor",
                    Box::new(|solver| solver.with_damping(f64::INFINITY)),
                ),
                (
                    "regularization",
                    Box::new(|solver| solver.with_regularization(-1.0)),
                ),
                (
                    "regularization",
                    Box::new(|solver| solver.with_regularization(f64::INFINITY)),
                ),
            ];

        for (expected_parameter, configure) in invalid_cases {
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let compiled = Compiler::compile(&[equation]).expect("compile failed");
            let solver = configure(NewtonRaphsonSolver::new(compiled));
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("invalid configuration must be rejected");
            match error {
                SolverError::InvalidInput(message) => {
                    assert!(message.contains(expected_parameter), "{message}");
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    /// Reject NaN and infinity in the caller's initial map before expression
    /// evaluation can obscure their origin.
    #[test]
    fn test_solver_rejects_non_finite_initial_values() {
        for invalid_value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let compiled = Compiler::compile(&[equation]).expect("compile failed");
            let solver = NewtonRaphsonSolver::new(compiled);
            let initial = HashMap::from([("x".to_string(), invalid_value)]);

            let error = solver
                .solve(initial)
                .expect_err("non-finite initial values must be rejected");
            match error {
                SolverError::InvalidInput(message) => {
                    assert!(message.contains("non-finite"), "{message}");
                    assert!(message.contains("x"), "{message}");
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    /// Distinguish an invalid expression domain from singularity or ordinary
    /// iteration exhaustion.
    #[test]
    fn test_solver_reports_non_finite_residual_evaluation() {
        for mode in test_modes() {
            let x = Exp::var("x");
            let equation = Exp::ln(x);
            let solver = solver_for_mode(vec![equation], mode);
            let initial = HashMap::from([("x".to_string(), -1.0)]);

            let error = solver
                .solve(initial)
                .expect_err("ln of a negative value is outside the real domain");
            match error {
                SolverError::NonFiniteEvaluation(diagnostic) => {
                    assert!(diagnostic.message.contains("Residual evaluation"));
                    assert_eq!(diagnostic.iterations, 0);
                    assert!(diagnostic.error.is_nan());
                }
                other => panic!("expected NonFiniteEvaluation, got {other:?}"),
            }
        }
    }

    /// Detect a finite residual paired with a non-finite symbolic derivative.
    #[test]
    fn test_solver_reports_non_finite_jacobian_evaluation() {
        for mode in test_modes() {
            // sqrt(x)-1 is finite at x=0, but its derivative is positive
            // infinity there.
            let x = Exp::var("x");
            let equation = Exp::sub(Exp::power(x, 0.5), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("non-finite derivative must be reported explicitly");
            match error {
                SolverError::NonFiniteEvaluation(diagnostic) => {
                    assert!(diagnostic.message.contains("Jacobian evaluation"));
                    assert_eq!(diagnostic.error, 1.0);
                }
                other => panic!("expected NonFiniteEvaluation, got {other:?}"),
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
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert!((solution.values.get("x").copied().unwrap() - 2.0).abs() < 1e-10);
        }
    }

    /// Exercise a recoverable exponential Newton step that needs more than the
    /// former twenty backtracking trials.
    #[test]
    fn test_line_search_can_backtrack_beyond_twenty_trials() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);

            // At x=-20, exp(x)-1 has a Newton step of roughly 4.85e8. Steps as
            // large as 2^-19 overflow or increase the residual, while a step of
            // roughly 2^-25 enters a useful region and must be accepted.
            let x = Exp::var("x");
            let equation = Exp::sub(Exp::exp(x), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_max_iterations(100)
                .with_tolerance(1e-10);
            let initial = HashMap::from([("x".to_string(), -20.0)]);

            let solution = solver
                .solve_with_line_search(initial)
                .expect("deep backtracking should recover a finite step");

            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-8);
            }
            assert!(solution.error < 1e-10);
            assert!(solution.values["x"].abs() < 1e-8);
        }
    }

    /// Verify that exhaustion preserves the current state and reports failure
    /// instead of applying an untested minimum-damping fallback.
    #[test]
    fn test_line_search_rejects_every_step_for_constant_residual() {
        for mode in test_modes() {
            // Multiplying x by zero registers it as a solve variable while the
            // residual remains exactly one for every possible candidate.
            let x = Exp::var("x");
            let equation = Exp::add(Exp::mul(x, Exp::val(0.0)), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode).with_max_iterations(2);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve_with_line_search(initial)
                .expect_err("a constant residual has no acceptable descent step");
            match error {
                SolverError::NoConvergence(diagnostic) => {
                    assert!(diagnostic.message.contains("Line search failed"));
                    assert_eq!(diagnostic.values["x"], 0.0);
                    assert_eq!(diagnostic.error, 1.0);
                }
                other => panic!("expected line-search NoConvergence, got {other:?}"),
            }
        }
    }

    /// Verify that a standard solve uses the configured damping value unchanged
    /// for its first Newton update.
    #[test]
    fn test_standard_solver_honors_initial_damping_on_first_step() {
        for mode in test_modes() {
            // The undamped Newton correction from x=0 is exactly one. With a
            // single allowed update and damping 0.5, the final diagnostic must
            // therefore contain x=0.5 rather than the formerly inflated x=0.6.
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_damping(0.5)
                .with_max_iterations(1);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("one half-step cannot yet reach the root");
            match error {
                SolverError::NoConvergence(diagnostic) => {
                    assert!((diagnostic.values["x"] - 0.5).abs() < 1e-12);
                    assert!((diagnostic.error - 0.5).abs() < 1e-12);
                }
                other => panic!("expected NoConvergence, got {other:?}"),
            }
        }
    }

    /// Confirm that damping is an effective maximum trial size for line-search
    /// solves rather than an ignored setting.
    #[test]
    fn test_line_search_uses_configured_initial_trial_size() {
        for mode in test_modes() {
            // The full Newton step would solve this equation immediately. A
            // configured maximum alpha of 0.25 must instead stop at x=0.25 after
            // the single allowed update.
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_damping(0.25)
                .with_max_iterations(1);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve_with_line_search(initial)
                .expect_err("one quarter-step cannot yet reach the root");
            match error {
                SolverError::NoConvergence(diagnostic) => {
                    assert!((diagnostic.values["x"] - 0.25).abs() < 1e-12);
                    assert!((diagnostic.error - 0.75).abs() < 1e-12);
                }
                other => panic!("expected NoConvergence, got {other:?}"),
            }
        }
    }

    /// Confirm that a tiny Newton update reports diagnostics from the returned
    /// point rather than retaining the residual from before the update.
    #[test]
    fn test_small_step_convergence_recomputes_returned_residual() {
        for mode in test_modes() {
            // The exact update is only 1e-20, below the default tolerance, but
            // it moves x from residual one to the exact root.
            let x = Exp::var("x");
            let equation = Exp::add(Exp::mul(Exp::val(1e20), x), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let solution = solver.solve(initial).expect("scaled linear root should solve");

            let returned_residual = 1e20 * solution.values["x"] + 1.0;
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert_eq!(solution.iterations, 1);
            assert_eq!(solution.error, returned_residual.abs());
            assert_eq!(solution.error, 0.0);
            assert_eq!(solution.convergence_history.last(), Some(&0.0));
        }
    }

    /// Distinguish a non-zero stationary least-squares result from an
    /// approximate root in an inconsistent overdetermined system.
    #[test]
    fn test_inconsistent_overdetermined_system_reports_stationarity() {
        for mode in test_modes() {
            // No x satisfies both equations. Their least-squares objective is
            // stationary at x=0 with residual norm sqrt(2).
            let x = Exp::var("x");
            let equations = vec![
                Exp::sub(x.clone(), Exp::val(1.0)),
                Exp::add(x, Exp::val(1.0)),
            ];
            let solver = solver_for_mode(equations, mode);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("stationary least-squares result should be explicit");

            assert_eq!(
                solution.reason,
                ConvergenceReason::LeastSquaresStationary
            );
            assert!((solution.error - 2.0_f64.sqrt()).abs() < 1e-12);
            assert!(solution.gradient_norm < 1e-10);
        }
    }

    /// Verify that the iteration count measures accepted updates rather than
    /// residual evaluations.
    #[test]
    fn test_exact_initial_root_reports_zero_iterations() {
        for mode in test_modes() {
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode);
            let initial = HashMap::from([("x".to_string(), 1.0)]);

            let solution = solver.solve(initial).expect("initial point is a root");

            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert_eq!(solution.iterations, 0);
            assert_eq!(solution.convergence_history, vec![0.0]);
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
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert!((solution.values.get("x").copied().unwrap() - 1.0).abs() < 1e-8);
            assert!((solution.values.get("y").copied().unwrap() - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_underconstrained_linearized_solution_policy_returns_minimum_norm_point() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Underdetermined system: x + y = 1 has infinitely many solutions.
            // The explicit point policy should discard the initial null-space
            // component and return x=y=0.5.
            let x = Exp::var("x");
            let y = Exp::var("y");

            let eq = Exp::sub(Exp::add(x.clone(), y.clone()), Exp::val(1.0));
            let solver = solver_for_mode(vec![eq], mode)
                .with_tolerance(1e-12)
                .with_max_iterations(10)
                .with_underdetermined_policy(
                    UnderdeterminedPolicy::MinimumNormLinearizedSolution,
                );

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 10.0);
            initial.insert("y".to_string(), -10.0);

            let solution = solver.solve(initial).expect("expected solver to converge");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert!((solution.values.get("x").copied().unwrap() - 0.5).abs() < 1e-10);
            assert!((solution.values.get("y").copied().unwrap() - 0.5).abs() < 1e-10);
        }
    }

    /// Confirm that the default minimum-step policy deliberately preserves the
    /// initial Jacobian-null-space component for continuity-sensitive clients.
    #[test]
    fn test_underconstrained_minimum_step_policy_preserves_null_space_component() {
        for mode in test_modes() {
            // The initial point has x+y=0 and a large component in the [1,-1]
            // null-space direction. The minimum correction is [0.5,0.5].
            let x = Exp::var("x");
            let y = Exp::var("y");
            let equation = Exp::sub(Exp::add(x, y), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_underdetermined_policy(UnderdeterminedPolicy::MinimumNormStep);
            let initial = HashMap::from([
                ("x".to_string(), 10.0),
                ("y".to_string(), -10.0),
            ]);

            let solution = solver.solve(initial).expect("linear constraint should solve");

            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert!((solution.values["x"] - 10.5).abs() < 1e-10);
            assert!((solution.values["y"] + 9.5).abs() < 1e-10);
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
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
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
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
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
