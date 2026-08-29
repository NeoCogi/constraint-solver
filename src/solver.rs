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

//! Damped Newton-Raphson iteration, line search, regularized least squares,
//! convergence results, and structured failure diagnostics.

use crate::compiler::{CompiledExp, CompiledSystem, EvaluationError, VarId, VarTable};
use crate::jacobian::Jacobian;
use crate::matrix::{LeastSquaresInfo, Matrix, MatrixError};
use crate::mode::{Mode, build_thread_pool};
use rayon::ThreadPool;
use std::collections::HashMap;
use std::fmt;

/// Reusable storage for the augmented system used by regularized QR solves.
struct LeastSquaresWorkspace {
    /// Coefficient matrix `[J; sqrt(lambda) I]` populated for each solve.
    augmented_j: Matrix,
    /// Right-hand side `[rhs; 0]` paired with `augmented_j`.
    augmented_b: Matrix,
    /// Number of residual rows copied from the unregularized Jacobian.
    num_equations: usize,
    /// Number of solve variables and regularization rows.
    num_variables: usize,
}

/// Borrowed numerical state required to test line-search candidates for one
/// Newton iteration.
///
/// Grouping these related values keeps the line-search call site explicit while
/// avoiding an error-prone positional argument list.
struct LineSearchContext<'a> {
    /// Symbolic evaluator used to compute residuals at candidate points.
    jacobian: &'a Jacobian,
    /// Current accepted variables, including solve variables and fixed
    /// parameters.
    vars: &'a HashMap<VarId, f64>,
    /// Newton correction direction whose scalar multiplier is being searched.
    delta: &'a Matrix,
    /// Residual norm at the current accepted point.
    current_residual_norm: f64,
    /// Directional derivative of `0.5 * ||f||^2` along `delta`.
    ///
    /// For residual vector `f`, Jacobian `J`, and candidate direction `p`, this
    /// value is `f^T J p`. A negative value proves that sufficiently small
    /// positive steps descend the least-squares objective.
    objective_directional_derivative: f64,
    /// Number of Newton updates accepted before this line search began.
    ///
    /// Keeping this as an accepted-update count, rather than a one-based attempt
    /// number, preserves the public `SolverRunDiagnostic::iterations` contract
    /// when every candidate is rejected and the current point is left unchanged.
    accepted_updates: usize,
    /// Residual vector at the current accepted point.
    current_residuals: &'a Matrix,
    /// Reusable output storage for candidate residual evaluations.
    candidate_residuals: &'a mut Matrix,
}

impl LeastSquaresWorkspace {
    /// Allocate zero-filled augmented matrices for repeated regularized solves.
    fn new(num_equations: usize, num_variables: usize) -> Result<Self, MatrixError> {
        // The augmented row count is caller-derived dimension arithmetic and
        // must be checked before allocation so release builds cannot wrap to a
        // smaller, internally inconsistent matrix.
        let augmented_rows =
            num_equations
                .checked_add(num_variables)
                .ok_or(MatrixError::DimensionSumOverflow {
                    operation: "regularized least-squares workspace construction",
                    left: num_equations,
                    right: num_variables,
                })?;

        // Both augmented matrices start at zero. Each solve overwrites the
        // residual block and regularization diagonal; every lower off-diagonal
        // coefficient and lower right-hand-side value remains zero forever.
        Ok(LeastSquaresWorkspace {
            augmented_j: Matrix::try_new(augmented_rows, num_variables)?,
            augmented_b: Matrix::try_new(augmented_rows, 1)?,
            num_equations,
            num_variables,
        })
    }
}

/// Internal distinction between invalid linear-algebra inputs and exhaustion of
/// both the direct and regularized least-squares paths.
enum LeastSquaresSolveError {
    /// A shape or matrix-operation precondition failed before a numerical solve
    /// could be attempted.
    InvalidInput(MatrixError),
    /// Neither the unregularized solve nor any configured augmented attempt
    /// produced an acceptable solution.
    Singular {
        /// Explanation from the direct, unregularized path or its condition
        /// diagnostics.
        unregularized_error: String,
        /// Explanation from the augmented regularized fallback.
        regularized_error: String,
    },
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
    /// Lower bound applied when repeated residual growth reduces adaptive damping.
    ///
    /// Reaching this bound is not itself a convergence failure: a non-zero
    /// least-squares residual can change very little near a valid stationary
    /// point, so termination must be decided from `J^T f`, not damping size.
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

/// Metadata connecting one compiled residual equation to its source constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationTrace {
    /// Caller-defined stable identifier for the source constraint.
    pub constraint_id: usize,
    /// Human-readable description suitable for logs and failure reports.
    pub description: String,
}

/// Residual and optional source metadata for one equation at a failed solver
/// state.
#[derive(Debug, Clone, PartialEq)]
pub struct EquationDiagnostic {
    /// Zero-based equation index matching compilation order.
    pub equation_index: usize,
    /// Signed scalar residual evaluated at the diagnostic's `values`.
    pub residual: f64,
    /// Caller-provided source metadata, or `None` when no trace was attached for
    /// this equation.
    pub trace: Option<EquationTrace>,
}

/// Complete numerical state captured when a solver run fails.
#[derive(Debug, Clone)]
pub struct SolverRunDiagnostic {
    /// Human-readable summary of the terminating failure.
    pub message: String,
    /// Number of accepted Newton updates represented by the diagnostic state.
    pub iterations: usize,
    /// Euclidean norm of all entries in `equations`.
    pub error: f64,
    /// Variable values at the end of the run, keyed by compiled variable name.
    pub values: HashMap<String, f64>,
    /// Per-equation residuals enriched with any attached source metadata.
    pub equations: Vec<EquationDiagnostic>,
}

/// Failures returned by solver construction, validation, and iteration.
#[derive(Debug, Clone)]
pub enum SolverError {
    /// The Jacobian or regularized least-squares system could not be solved.
    SingularMatrix(SolverRunDiagnostic),
    /// The configured update budget or line search ended without convergence.
    NoConvergence(SolverRunDiagnostic),
    /// Expression, Jacobian, update, or residual-gradient evaluation produced
    /// NaN or infinity.
    NonFiniteEvaluation(SolverRunDiagnostic),
    /// Invalid input parameters or system setup.
    InvalidInput(String),
}

impl fmt::Display for SolverError {
    /// Format the concise diagnostic summary while preserving structured state
    /// for callers that pattern-match the error.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::SingularMatrix(diagnostic) => {
                write!(f, "Singular matrix: {}", diagnostic.message)
            }
            SolverError::NoConvergence(diagnostic) => {
                write!(f, "Solver did not converge: {}", diagnostic.message)
            }
            SolverError::NonFiniteEvaluation(diagnostic) => {
                write!(f, "Non-finite solver evaluation: {}", diagnostic.message)
            }
            SolverError::InvalidInput(message) => write!(f, "Invalid solver input: {message}"),
        }
    }
}

impl std::error::Error for SolverError {}

impl From<EvaluationError> for SolverError {
    /// Convert an impossible-after-validation evaluation mismatch into an
    /// explicit invariant failure without exposing an unreachable public error
    /// variant.
    fn from(value: EvaluationError) -> Self {
        SolverError::InvalidInput(format!(
            "Internal compiled-system invariant failed: {value}"
        ))
    }
}

impl From<MatrixError> for SolverError {
    /// Preserve matrix validation context in the solver's public input-error
    /// channel.
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

    /// Attach optional source metadata for every equation in compilation order.
    ///
    /// The trace vector must contain exactly one entry per equation, including
    /// for an empty system. Use `None` for an equation whose source is unknown.
    /// This fallible builder avoids the former API's input-dependent panic.
    pub fn with_equation_traces(
        mut self,
        traces: Vec<Option<EquationTrace>>,
    ) -> Result<Self, SolverError> {
        // Exact cardinality preserves the positional equation-to-trace mapping;
        // accepting arbitrary entries for an empty system would create metadata
        // that could never appear in a residual diagnostic.
        if traces.len() != self.equations.len() {
            return Err(SolverError::InvalidInput(format!(
                "Equation trace length ({}) does not match number of equations ({})",
                traces.len(),
                self.equations.len()
            )));
        }
        self.equation_traces = traces;
        Ok(self)
    }

    /// Fetch source metadata for a specific equation, if an entry was attached.
    ///
    /// Returns `None` both for an out-of-range index and for an explicitly
    /// untraced equation.
    pub fn trace_for_equation(&self, equation_index: usize) -> Option<&EquationTrace> {
        // Preserve borrowing so callers can inspect potentially long
        // descriptions without cloning them.
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
    pub fn with_underdetermined_policy(mut self, policy: UnderdeterminedPolicy) -> Self {
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

        // This internal entry point owns the validated map, so move it directly
        // into mutable solver state instead of cloning every variable entry.
        let mut vars = initial_guess;
        let jacobian = Jacobian::new(self.equations.clone(), self.variables.clone());
        // Cache Jacobian storage to avoid per-iteration allocations.
        let mut jacobian_workspace = jacobian.workspace();
        // Store the initial residual plus at most one residual for every
        // accepted update.
        let mut convergence_history = Vec::with_capacity(self.max_iterations.saturating_add(1));
        // Standard solves begin with exactly the configured damping factor. A
        // previous residual is optional so the first iteration cannot pretend
        // it observed an improvement from an artificial infinity sentinel.
        let mut damping = self.damping_factor;
        let mut last_error: Option<f64> = None;

        let num_equations = self.equations.len();
        let num_variables = self.variables.len();

        let mut f_vals = Matrix::new(num_equations, 1);
        let mut f_neg = Matrix::new(num_equations, 1);
        // Populate the workspace in place. Allocating an independent Jacobian
        // and immediately replacing this correctly sized buffer would double
        // initial derivative storage for no benefit.
        jacobian.evaluate_checked_in_workspace(&vars, &mut jacobian_workspace)?;

        let mut line_search_f_vals = Matrix::new(num_equations, 1);

        let mut least_squares_workspace =
            if self.regularization > 0.0 && num_equations != num_variables {
                Some(LeastSquaresWorkspace::new(num_equations, num_variables)?)
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

            // The origin-based point policy cannot damp a vector directly from
            // an arbitrary current point to the projected Newton point: doing so
            // retains the same fraction of the current Jacobian-null-space
            // component. Remove that component as its own accepted update first.
            // Subsequent damped corrections then remain entirely in the row
            // space and are minimum-norm solutions of their damped linearization.
            if num_equations < num_variables
                && self.underdetermined_policy
                    == UnderdeterminedPolicy::MinimumNormLinearizedSolution
            {
                let projection_result = self.project_underdetermined_current_point(
                    jacobian_workspace.jacobian(),
                    &vars,
                    least_squares_workspace.as_mut(),
                    parallel,
                );
                let projected_values = match projection_result {
                    Ok(projected_values) => projected_values,
                    Err(LeastSquaresSolveError::InvalidInput(err)) => {
                        return Err(SolverError::InvalidInput(err.to_string()));
                    }
                    Err(LeastSquaresSolveError::Singular {
                        unregularized_error,
                        regularized_error,
                    }) => {
                        let diagnostic = self.build_diagnostic(
                            format!(
                                "Could not project the current point out of the Jacobian null space (unregularized_error: {unregularized_error}; regularized_error: {regularized_error})"
                            ),
                            iter,
                            &vars,
                            &f_vals,
                        );
                        return Err(SolverError::SingularMatrix(diagnostic));
                    }
                };

                if let Some(projected_values) = projected_values {
                    // Apply the full null-space projection. Partial application
                    // would preserve null-space content and violate the selected
                    // point policy; damping is reserved for the residual-space
                    // Newton correction on the following iteration.
                    for (index, &var_id) in self.variables.iter().enumerate() {
                        vars.insert(var_id, projected_values[(index, 0)]);
                    }

                    // A projection is an accepted variable update and receives
                    // the same finite-state validation as an ordinary Newton
                    // update before the next linearization is constructed.
                    jacobian.evaluate_functions_checked_into(&vars, &mut f_vals)?;
                    if !f_vals.all_finite() {
                        return Err(self.non_finite_evaluation_error(
                            "Minimum-norm point projection produced non-finite residuals",
                            iter + 1,
                            &vars,
                            &f_vals,
                        ));
                    }
                    jacobian.evaluate_checked_in_workspace(&vars, &mut jacobian_workspace)?;
                    if !jacobian_workspace.jacobian().all_finite() {
                        return Err(self.non_finite_evaluation_error(
                            "Minimum-norm point projection produced a non-finite Jacobian",
                            iter + 1,
                            &vars,
                            &f_vals,
                        ));
                    }

                    // Restart from the projected state so convergence, line
                    // search, and damping all use residuals and derivatives from
                    // exactly the same accepted point.
                    continue;
                }
            }

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

            // A non-zero residual can still be the mathematically correct result
            // of an inconsistent overdetermined system. Test first-order
            // least-squares stationarity before damping adaptation so the small
            // objective changes near a residual floor cannot be misclassified as
            // solver stagnation.
            if num_equations > num_variables {
                let gradient_norm = self.residual_gradient_norm(
                    jacobian_workspace.jacobian(),
                    &f_vals,
                    parallel,
                    &vars,
                    iter,
                )?;
                if gradient_norm < self.tolerance {
                    return Ok(self.build_solution(
                        &vars,
                        iter,
                        error,
                        gradient_norm,
                        ConvergenceReason::LeastSquaresStationary,
                        convergence_history,
                    ));
                }
            }

            // Adaptive damping belongs only to the standard solver. Line search
            // has its own evaluated acceptance and exhaustion rules, so mutating
            // an unused damping value must not influence its stagnation outcome.
            if !use_line_search {
                if let Some(previous_error) = last_error {
                    // Compare only two real consecutive residuals. This keeps
                    // the caller's configured damping unchanged for the first
                    // update and adapts it for subsequent updates. Only genuine
                    // residual growth reduces damping; a small improvement is
                    // expected near a non-zero least-squares optimum and is not
                    // evidence of stagnation.
                    if error > previous_error * 1.5 {
                        damping = (damping * 0.5).max(self.min_damping);
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
                            // A singular square Jacobian gets one regularized
                            // fallback. Work on a clone so the cached matrix
                            // remains the derivative of the actual residuals;
                            // line-search slope calculations must use J, not the
                            // numerically modified system used only to obtain p.
                            let mut regularized_jacobian = j_matrix.clone();
                            let diag_size =
                                regularized_jacobian.rows().min(regularized_jacobian.cols());
                            for i in 0..diag_size {
                                regularized_jacobian[(i, i)] += self.regularization * 100.0;
                            }
                            match regularized_jacobian.solve_lu(&f_neg) {
                                Ok(d) => d,
                                Err(e) => {
                                    let diag = self.build_diagnostic(
                                        format!("Matrix is singular even with regularization: {e}"),
                                        iter,
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
                            unregularized_error,
                            regularized_error,
                        }) => {
                            let diag = self.build_diagnostic(
                                format!(
                                    "Least squares system is singular (unregularized_error: {unregularized_error}; regularized_error: {regularized_error})"
                                ),
                                iter,
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
                    iter,
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
                let objective_directional_derivative = self.least_squares_directional_derivative(
                    jacobian_workspace.jacobian(),
                    &f_vals,
                    &delta,
                    parallel,
                    &vars,
                    iter,
                )?;
                self.line_search(LineSearchContext {
                    jacobian: &jacobian,
                    vars: &vars,
                    delta: &delta,
                    current_residual_norm: error,
                    objective_directional_derivative,
                    accepted_updates: iter,
                    current_residuals: &f_vals,
                    candidate_residuals: &mut line_search_f_vals,
                })?
            } else {
                // Use current damping factor as step size
                damping
            };

            // Step 5: Update variables: x_new = x_old + step_size * delta.
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

            if updated_error < self.tolerance {
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

                return Ok(self.build_solution(
                    &vars,
                    iter + 1,
                    updated_error,
                    gradient_norm,
                    ConvergenceReason::ResidualTolerance,
                    convergence_history,
                ));
            }

            // Least-squares stationarity is checked once at the beginning of the
            // next iteration. Deferring that check avoids computing `J^T f`
            // twice for every accepted state while still testing the final state
            // explicitly after the update budget is exhausted below.
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
        let final_error = residuals.norm();
        if final_error < self.tolerance {
            // A minimum-norm null-space projection can consume the final allowed
            // update and land directly on a root. There is no following loop
            // iteration to run the ordinary residual check, so accept that final
            // state here with diagnostics evaluated at the returned values.
            let gradient_norm = self.residual_gradient_norm(
                jacobian_workspace.jacobian(),
                &residuals,
                parallel,
                &vars,
                self.max_iterations,
            )?;
            convergence_history.push(final_error);
            return Ok(self.build_solution(
                &vars,
                self.max_iterations,
                final_error,
                gradient_norm,
                ConvergenceReason::ResidualTolerance,
                convergence_history,
            ));
        }
        if num_equations > num_variables {
            // The loop normally checks stationarity at the start of the next
            // iteration. When the final allowed update lands on an optimum there
            // is no next iteration, so perform the same test here before
            // reporting budget exhaustion.
            let gradient_norm = self.residual_gradient_norm(
                jacobian_workspace.jacobian(),
                &residuals,
                parallel,
                &vars,
                self.max_iterations,
            )?;
            if gradient_norm < self.tolerance {
                convergence_history.push(final_error);
                return Ok(self.build_solution(
                    &vars,
                    self.max_iterations,
                    final_error,
                    gradient_norm,
                    ConvergenceReason::LeastSquaresStationary,
                    convergence_history,
                ));
            }
        }

        let diag = self.build_diagnostic(
            format!(
                "Failed to converge after {} iterations. Final error: {:.2e}",
                self.max_iterations, final_error
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
                self.solve_least_squares_system(j_matrix, negative_residuals, workspace, parallel)
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
                let next_values =
                    self.solve_least_squares_system(j_matrix, &projected_rhs, workspace, parallel)?;

                // The surrounding update loop consumes a correction, so convert
                // the projected next point back into delta form.
                next_values
                    .try_sub(&current_values)
                    .map_err(LeastSquaresSolveError::InvalidInput)
            }
        }
    }

    /// Project the current solve-variable vector onto the row space of a wide
    /// Jacobian when origin-based minimum-norm point semantics are requested.
    ///
    /// Solving `J * x_projected = J * x_current` for its minimum-norm solution
    /// removes exactly the component of `x_current` in the null space of `J`.
    /// The returned `None` means the current point is already projected to
    /// floating-point accuracy; `Some` contains a full replacement vector whose
    /// application must not be damped.
    fn project_underdetermined_current_point(
        &self,
        j_matrix: &Matrix,
        vars: &HashMap<VarId, f64>,
        workspace: Option<&mut LeastSquaresWorkspace>,
        parallel: bool,
    ) -> Result<Option<Matrix>, LeastSquaresSolveError> {
        // Assemble solve variables in Jacobian-column order. Fixed parameters do
        // not appear in this vector and remain untouched by the projection.
        let mut current_values = Matrix::new(self.variables.len(), 1);
        for (index, &var_id) in self.variables.iter().enumerate() {
            current_values[(index, 0)] = vars[&var_id];
        }

        // J*x is invariant under removal of any Jacobian-null-space component.
        // Re-solving this right-hand side with the wide minimum-norm solver
        // therefore produces the row-space representative of the same local
        // linearized state.
        let projected_rhs = j_matrix
            .try_mul_with_parallel(&current_values, parallel)
            .map_err(LeastSquaresSolveError::InvalidInput)?;
        let projected_values =
            self.solve_least_squares_system(j_matrix, &projected_rhs, workspace, parallel)?;
        let projection_delta = projected_values
            .try_sub(&current_values)
            .map_err(LeastSquaresSolveError::InvalidInput)?;

        // Avoid consuming iterations on roundoff-sized reprojections. Scaling by
        // the current vector norm makes the threshold invariant under unit
        // changes while the epsilon multiplier allows ordinary QR noise.
        const PROJECTION_EPSILON_MULTIPLIER: f64 = 64.0;
        let projection_threshold =
            PROJECTION_EPSILON_MULTIPLIER * f64::EPSILON * current_values.norm().max(1.0);
        if projection_delta.norm() <= projection_threshold {
            Ok(None)
        } else {
            Ok(Some(projected_values))
        }
    }

    /// Solve a possibly rectangular linear system against one right-hand side,
    /// applying adaptive regularization when factorization diagnostics indicate
    /// excessive conditioning.
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
        // Prefer direct least squares to avoid forming normal equations (J^T J),
        // which can severely amplify conditioning issues. Full-rank systems use
        // QR, while rank-deficient systems use an SVD pseudoinverse. If the
        // retained singular subspace is still ill-conditioned, fall back to the
        // regularized augmented system [J; sqrt(lambda) I] * x ~= [rhs; 0].
        const COND_LIMIT: f64 = 1e12;

        // Numerical rank loss alone is valid for a Moore-Penrose solve. The SVD
        // condition estimate describes only its retained image, so finite,
        // well-conditioned rank-deficient systems need no artificial damping.
        let is_ill_conditioned = |info: &LeastSquaresInfo| -> bool {
            !info.cond_est.is_finite() || info.cond_est > COND_LIMIT
        };

        let mut unregularized_error: Option<String> = None;
        let mut unreg_delta: Option<Matrix> = None;
        let mut unreg_info: Option<LeastSquaresInfo> = None;

        match j_matrix.solve_least_squares_with_info_with_parallel(rhs, parallel) {
            Ok((delta, info)) => {
                if !is_ill_conditioned(&info) {
                    return Ok(delta);
                }
                unreg_delta = Some(delta);
                unreg_info = Some(info);
            }
            Err(err) => {
                unregularized_error = Some(err.to_string());
            }
        }

        if self.regularization <= 0.0 {
            if let Some(delta) = unreg_delta {
                return Ok(delta);
            }
            return Err(LeastSquaresSolveError::Singular {
                unregularized_error: unregularized_error
                    .unwrap_or_else(|| "Unregularized least squares failed".to_string()),
                regularized_error: "Regularization disabled".to_string(),
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

                    // Construction permanently zeroes every lower
                    // off-diagonal coefficient and lower RHS value. Only the
                    // diagonal changes with each adaptive regularization retry,
                    // so rewriting the entire square block would be dead work.
                    for i in 0..workspace.num_variables {
                        workspace.augmented_j[(workspace.num_equations + i, i)] = sqrt_reg;
                    }

                    workspace
                        .augmented_j
                        .solve_least_squares_with_info_with_parallel(
                            &workspace.augmented_b,
                            parallel,
                        )
                }
                None => {
                    // This allocation fallback is normally avoided by the
                    // reusable workspace, but it remains independently safe for
                    // future callers that do not provide one.
                    let augmented_rows = j_matrix.rows().checked_add(j_matrix.cols()).ok_or(
                        LeastSquaresSolveError::InvalidInput(MatrixError::DimensionSumOverflow {
                            operation: "regularized least-squares allocation",
                            left: j_matrix.rows(),
                            right: j_matrix.cols(),
                        }),
                    )?;
                    let mut augmented_j = Matrix::try_new(augmented_rows, j_matrix.cols())
                        .map_err(LeastSquaresSolveError::InvalidInput)?;
                    let mut augmented_b = Matrix::try_new(augmented_rows, 1)
                        .map_err(LeastSquaresSolveError::InvalidInput)?;
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
                    augmented_j.solve_least_squares_with_info_with_parallel(&augmented_b, parallel)
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
                    last_err = Some(err.to_string());
                }
            }
        }

        if let Some(delta) = last_solution {
            return Ok(delta);
        }

        Err(LeastSquaresSolveError::Singular {
            unregularized_error: unregularized_error
                .or_else(|| {
                    unreg_info.map(|info| {
                        format!(
                            "Least squares ill-conditioned (rank {}, cond {:.2e})",
                            info.rank, info.cond_est
                        )
                    })
                })
                .unwrap_or_else(|| "Unregularized least squares failed".to_string()),
            regularized_error: last_err
                .unwrap_or_else(|| "Regularized least squares failed".to_string()),
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

    /// Compute the directional derivative of the nonlinear least-squares
    /// objective `0.5 * ||f||^2` along a proposed solver direction.
    ///
    /// The derivative is `f^T J p`, where `p` is the Newton or Gauss-Newton
    /// correction. Keeping this calculation in one checked helper ensures the
    /// line search never applies an Armijo test to a non-finite or mismatched
    /// numerical state.
    fn least_squares_directional_derivative(
        &self,
        jacobian: &Matrix,
        residuals: &Matrix,
        direction: &Matrix,
        parallel: bool,
        vars: &HashMap<VarId, f64>,
        accepted_updates: usize,
    ) -> Result<f64, SolverError> {
        // Multiplying J by the candidate direction produces the residual-space
        // velocity used by the chain rule. The multiplication honors the
        // solver's selected execution mode and retains structured shape errors.
        let residual_velocity = jacobian.try_mul_with_parallel(direction, parallel)?;
        if !residual_velocity.all_finite() {
            return Err(self.non_finite_evaluation_error(
                "Line-search directional derivative produced NaN or infinity",
                accepted_updates,
                vars,
                residuals,
            ));
        }

        // Accumulate the dot product with Neumaier compensation. Compensation
        // reduces cancellation error when signed residual contributions differ
        // greatly in magnitude, which is common near a least-squares optimum.
        let mut sum = 0.0;
        let mut compensation = 0.0;
        for row in 0..residuals.rows() {
            let product = residuals[(row, 0)] * residual_velocity[(row, 0)];
            if !product.is_finite() {
                return Err(self.non_finite_evaluation_error(
                    "Line-search directional derivative overflowed",
                    accepted_updates,
                    vars,
                    residuals,
                ));
            }

            let next = sum + product;
            if sum.abs() >= product.abs() {
                compensation += (sum - next) + product;
            } else {
                compensation += (product - next) + sum;
            }
            sum = next;
        }

        let derivative = sum + compensation;
        if !derivative.is_finite() {
            return Err(self.non_finite_evaluation_error(
                "Line-search directional derivative accumulation overflowed",
                accepted_updates,
                vars,
                residuals,
            ));
        }
        Ok(derivative)
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

    /// Capture a failure message together with the exact variable and
    /// per-equation residual state at which it occurred.
    fn build_diagnostic(
        &self,
        message: String,
        iterations: usize,
        vars: &HashMap<VarId, f64>,
        residuals_matrix: &Matrix,
    ) -> SolverRunDiagnostic {
        // Residual row order matches compiled equation order. Clone optional
        // trace metadata into the owned diagnostic so it remains useful after
        // the solver itself has been dropped.
        let mut equations = Vec::with_capacity(residuals_matrix.rows());
        for equation_index in 0..residuals_matrix.rows() {
            equations.push(EquationDiagnostic {
                equation_index,
                residual: residuals_matrix[(equation_index, 0)],
                trace: self.trace_for_equation(equation_index).cloned(),
            });
        }

        SolverRunDiagnostic {
            message,
            iterations,
            error: residuals_matrix.norm(),
            values: self.values_by_name(vars),
            equations,
        }
    }

    /// Backtracking line search to find optimal step size
    ///
    /// Tries progressively smaller step sizes until one is found that reduces the error.
    /// This helps prevent oscillation and improves convergence robustness.
    ///
    /// # Context fields
    /// * `jacobian` - Jacobian evaluator for computing function values
    /// * `vars` - Current variable values
    /// * `delta` - Newton step direction
    /// * `current_residual_norm` - Current function error `||f(x)||`
    /// * `objective_directional_derivative` - Value of `f^T J delta`
    /// * `accepted_updates` - Number of updates accepted before the search began
    /// * `current_residuals` - Residual vector at the unmodified current point
    /// * `candidate_residuals` - Reusable storage for each trial point
    ///
    /// # Returns
    /// An accepted step size alpha such that `x_new = x + alpha * delta`
    /// satisfies the sufficient-decrease condition. If no tested step is
    /// acceptable, the method returns `NoConvergence`; it never substitutes an
    /// untested or previously rejected fallback step.
    fn line_search(&self, context: LineSearchContext<'_>) -> Result<f64, SolverError> {
        // Give the context fields concise numerical names for the candidate loop
        // while preserving their documented grouping at the call boundary.
        let LineSearchContext {
            jacobian,
            vars,
            delta,
            current_residual_norm,
            objective_directional_derivative,
            accepted_updates,
            current_residuals,
            candidate_residuals,
        } = context;

        // A difficult but recoverable Newton direction can require far more
        // than twenty halvings. Sixty-four trials cover the useful binary range
        // down to the explicit minimum without making the search unbounded.
        const MAX_LINE_SEARCH_TRIALS: usize = 64;
        const MIN_LINE_SEARCH_STEP: f64 = 1e-12;
        // A small Armijo coefficient asks only for a decrease consistent with
        // the local objective slope. Unlike the previous fixed 50% residual
        // reduction, this remains attainable near a non-zero residual floor.
        const ARMIJO_COEFFICIENT: f64 = 1e-4;

        // A non-negative derivative means the proposed correction is not a
        // descent direction for the least-squares objective. Backtracking cannot
        // repair such a direction, so preserve the current state and explain the
        // numerical cause without evaluating misleading trial points.
        if objective_directional_derivative >= 0.0 {
            let diagnostic = self.build_diagnostic(
                format!(
                    "Line search cannot proceed because the proposed correction is not a descent direction (f^T J delta = {objective_directional_derivative:.2e})"
                ),
                accepted_updates,
                vars,
                current_residuals,
            );
            return Err(SolverError::NoConvergence(diagnostic));
        }

        // Applying Armijo to the residual norm avoids squaring very large but
        // finite norms. The derivative of `||f||` is `(f^T J p) / ||f||` away
        // from a root; roots have already terminated before line search begins.
        let residual_norm_directional_derivative =
            objective_directional_derivative / current_residual_norm;
        if !residual_norm_directional_derivative.is_finite() {
            return Err(self.non_finite_evaluation_error(
                "Line-search residual-norm slope produced NaN or infinity",
                accepted_updates,
                vars,
                current_residuals,
            ));
        }

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
            let required_error = current_residual_norm
                + ARMIJO_COEFFICIENT * alpha * residual_norm_directional_derivative;
            if new_error.is_finite() && new_error <= required_error {
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
            accepted_updates,
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
    use crate::Mode;
    use crate::compiler::Compiler;
    use crate::exp::Exp;

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
            .clamp(1, 4);
        vec![
            Mode::Serial,
            Mode::Parallel {
                thread_count: threads,
            },
        ]
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
        if let (SolverError::InvalidInput(a), SolverError::InvalidInput(b)) = (serial, parallel) {
            assert_eq!(a, b);
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
    fn test_solver_rejects_missing_fixed_parameter() {
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

            let err = solver
                .solve(initial)
                .expect_err("expected missing variable error");
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
        /// Boxed builder mutation used to exercise one invalid configuration
        /// while constructing a fresh solver for each table entry.
        type SolverConfigurator = Box<dyn Fn(NewtonRaphsonSolver) -> NewtonRaphsonSolver>;

        // Each case constructs a fresh solver because builder methods consume
        // and return the solver by value.
        let invalid_cases: Vec<(&str, SolverConfigurator)> = vec![
            (
                "max_iterations",
                Box::new(|solver| solver.with_max_iterations(0)),
            ),
            ("tolerance", Box::new(|solver| solver.with_tolerance(0.0))),
            (
                "tolerance",
                Box::new(|solver| solver.with_tolerance(f64::NAN)),
            ),
            (
                "damping_factor",
                Box::new(|solver| solver.with_damping(0.0)),
            ),
            (
                "damping_factor",
                Box::new(|solver| solver.with_damping(1.1)),
            ),
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
    fn test_with_equation_traces_rejects_invalid_length() {
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
            let err = match solver.with_equation_traces(traces) {
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

        // Empty systems must reject unreachable trace entries as strictly as
        // non-empty systems reject too many or too few entries.
        let empty = NewtonRaphsonSolver::new(
            Compiler::compile(&[]).expect("empty equation system should compile"),
        );
        let error = match empty.with_equation_traces(vec![None]) {
            Ok(_) => panic!("empty system must reject an unreachable trace"),
            Err(error) => error,
        };
        assert!(matches!(error, SolverError::InvalidInput(_)));
    }

    /// Ensure failure diagnostics pair each residual with its positional source
    /// metadata and retain explicit `None` entries for untraced equations.
    #[test]
    fn test_diagnostics_include_equation_traces() {
        let equations = vec![
            Exp::sub(Exp::var("x"), Exp::val(1.0)),
            Exp::sub(Exp::var("x"), Exp::val(2.0)),
        ];
        let trace = EquationTrace {
            constraint_id: 41,
            description: "horizontal alignment".to_string(),
        };
        let solver = NewtonRaphsonSolver::new(
            Compiler::compile(&equations).expect("equations should compile"),
        )
        .with_equation_traces(vec![Some(trace.clone()), None])
        .expect("trace cardinality matches equation cardinality");
        let vars = solver
            .map_initial_guess(HashMap::from([("x".to_string(), 0.0)]))
            .expect("initial guess should map to the compiled variable");
        let residuals = Matrix::from_vec(vec![-1.0, -2.0], 2, 1)
            .expect("residual vector shape should be valid");

        let diagnostic = solver.build_diagnostic("test failure".to_string(), 0, &vars, &residuals);

        assert_eq!(diagnostic.equations.len(), 2);
        assert_eq!(
            diagnostic.equations[0],
            EquationDiagnostic {
                equation_index: 0,
                residual: -1.0,
                trace: Some(trace),
            }
        );
        assert_eq!(diagnostic.equations[1].equation_index, 1);
        assert_eq!(diagnostic.equations[1].residual, -2.0);
        assert_eq!(diagnostic.equations[1].trace, None);
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
                    // Every trial was rejected, so the public count must describe
                    // the unchanged initial state rather than the first attempted
                    // Newton update.
                    assert_eq!(diagnostic.iterations, 0);
                    assert!(
                        diagnostic.message.contains("not a descent direction"),
                        "{}",
                        diagnostic.message
                    );
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

    /// Ensure that intentionally small damping does not become a false
    /// small-step failure before the residual tolerance is satisfied.
    #[test]
    fn test_heavily_damped_linear_solve_reaches_residual_tolerance() {
        for use_line_search in [false, true] {
            for mode in test_modes() {
                // Damping 0.1 requires roughly 220 geometric updates to reduce
                // the residual below 1e-10. The applied update becomes smaller
                // than tolerance earlier, but that is a caller-selected step
                // size rather than evidence of numerical stagnation.
                let x = Exp::var("x");
                let equation = Exp::sub(x, Exp::val(1.0));
                let solver = solver_for_mode(vec![equation], mode)
                    .with_damping(0.1)
                    .with_tolerance(1e-10)
                    .with_max_iterations(300);
                let initial = HashMap::from([("x".to_string(), 0.0)]);

                let solution = if use_line_search {
                    solver.solve_with_line_search(initial)
                } else {
                    solver.solve(initial)
                }
                .expect("damped linear updates should eventually reach the root");

                assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
                assert!(solution.error < 1e-10);
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

            let solution = solver
                .solve(initial)
                .expect("scaled linear root should solve");

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

            assert_eq!(solution.reason, ConvergenceReason::LeastSquaresStationary);
            assert!((solution.error - 2.0_f64.sqrt()).abs() < 1e-12);
            assert!(solution.gradient_norm < 1e-10);
        }
    }

    /// Ensure the default damping policy can approach a non-zero residual floor
    /// instead of treating the naturally shrinking objective changes as failure.
    #[test]
    fn test_inconsistent_overdetermined_system_reaches_stationarity_from_away() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);

            // The least-squares objective for these incompatible equations has
            // its unique stationary point at x=0 and residual norm sqrt(2). The
            // previous absolute residual-change stagnation rule stopped around
            // x=3e-6 under the default 0.7 damping, before `J^T f` met tolerance.
            let x = Exp::var("x");
            let equations = vec![
                Exp::sub(x.clone(), Exp::val(1.0)),
                Exp::add(x, Exp::val(1.0)),
            ];
            let solver = solver_for_mode(equations, mode);
            let initial = HashMap::from([("x".to_string(), 0.1)]);

            let solution = solver
                .solve(initial)
                .expect("default damping should reach least-squares stationarity");

            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert_eq!(solution.reason, ConvergenceReason::LeastSquaresStationary);
            assert!(solution.values["x"].abs() < 1e-8);
            assert!((solution.error - 2.0_f64.sqrt()).abs() < 1e-10);
            assert!(solution.gradient_norm < 1e-8);
        }
    }

    /// Confirm that line search uses the local least-squares slope rather than
    /// demanding an impossible fixed reduction near a non-zero residual floor.
    #[test]
    fn test_line_search_reaches_inconsistent_least_squares_stationarity() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);

            // Moving from x=0.1 to x=0 only improves the residual norm by about
            // half a percent because sqrt(2) is unavoidable. A valid Armijo test
            // accepts that slope-consistent improvement, whereas the former 50%
            // requirement rejected every possible positive step.
            let x = Exp::var("x");
            let equations = vec![
                Exp::sub(x.clone(), Exp::val(1.0)),
                Exp::add(x, Exp::val(1.0)),
            ];
            let solver = solver_for_mode(equations, mode);
            let initial = HashMap::from([("x".to_string(), 0.1)]);

            let solution = solver
                .solve_with_line_search(initial)
                .expect("Armijo line search should reach the least-squares optimum");

            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert_eq!(solution.reason, ConvergenceReason::LeastSquaresStationary);
            assert!(solution.values["x"].abs() < 1e-8);
            assert!((solution.error - 2.0_f64.sqrt()).abs() < 1e-10);
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
                .with_underdetermined_policy(UnderdeterminedPolicy::MinimumNormLinearizedSolution);

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

    /// Verify that damping changes only residual-space progress and never leaves
    /// a fraction of an inherited Jacobian-null-space component in the result.
    #[test]
    fn test_damped_linearized_solution_policy_remains_minimum_norm() {
        for use_line_search in [false, true] {
            let mut serial_solution: Option<Solution> = None;
            for mode in test_modes() {
                let is_serial = matches!(mode, Mode::Serial);

                // The initial values have a null-space component of magnitude
                // 1e8. Damping the old current-to-target delta left about 0.0058
                // of that component when residual tolerance terminated, yielding
                // visibly unequal coordinates despite the point policy.
                let x = Exp::var("x");
                let y = Exp::var("y");
                let equation = Exp::sub(Exp::add(x, y), Exp::val(1.0));
                let solver = solver_for_mode(vec![equation], mode)
                    .with_underdetermined_policy(
                        UnderdeterminedPolicy::MinimumNormLinearizedSolution,
                    )
                    .with_damping(0.5)
                    .with_tolerance(1e-10)
                    .with_max_iterations(100);
                let initial = HashMap::from([("x".to_string(), 1.0e8), ("y".to_string(), -1.0e8)]);

                let solution = if use_line_search {
                    solver.solve_with_line_search(initial)
                } else {
                    solver.solve(initial)
                }
                .expect("damped point policy should retain minimum-norm semantics");

                if is_serial {
                    serial_solution = Some(solution.clone());
                } else {
                    let serial = serial_solution.as_ref().expect("serial solution missing");
                    assert_solution_close(serial, &solution, 1e-10);
                }
                assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
                assert!((solution.values["x"] - 0.5).abs() < 1e-9);
                assert!((solution.values["y"] - 0.5).abs() < 1e-9);
                assert!((solution.values["x"] - solution.values["y"]).abs() < 1e-10);
            }
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
            let initial = HashMap::from([("x".to_string(), 10.0), ("y".to_string(), -10.0)]);

            let solution = solver
                .solve(initial)
                .expect("linear constraint should solve");

            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert!((solution.values["x"] - 10.5).abs() < 1e-10);
            assert!((solution.values["y"] + 9.5).abs() < 1e-10);
        }
    }

    /// Confirm that default solver settings accept a well-conditioned retained
    /// SVD subspace instead of damping merely because a variable is unused.
    #[test]
    fn test_rank_deficient_overconstrained_converges_with_default_settings() {
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

    /// Confirm that a rank-deficient Jacobian is directly solvable without
    /// regularization when its Moore-Penrose correction is well conditioned.
    #[test]
    fn test_rank_deficient_overconstrained_converges_without_regularization() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // Every equation constrains x while multiplication by zero keeps y
            // present in the compiled variable table but absent from J's image.
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

            let solution = solver
                .solve(initial)
                .expect("the SVD pseudoinverse should solve without damping");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert_eq!(solution.reason, ConvergenceReason::ResidualTolerance);
            assert!((solution.values["x"] - 1.0).abs() < 1e-10);
            assert!(solution.values["y"].abs() < 1e-10);
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
                for (diagonal_index, row) in a.iter_mut().enumerate() {
                    for coefficient in row.iter_mut() {
                        *coefficient = rng.next_f64();
                    }
                    row[diagonal_index] += 2.0;
                }

                let x_true = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
                let mut b = vec![0.0; n];
                for (rhs, row) in b.iter_mut().zip(&a) {
                    *rhs = row
                        .iter()
                        .zip(x_true.iter())
                        .map(|(coefficient, value)| coefficient * value)
                        .sum();
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
                for (i, expected_value) in x_true.iter().enumerate() {
                    let value = solution.values.get(&format!("x{i}")).copied().unwrap();
                    assert!((value - expected_value).abs() < 1e-8);
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
                for row in &mut a {
                    for coefficient in row {
                        *coefficient = rng.next_f64();
                    }
                }
                for (diagonal_index, row) in a.iter_mut().take(n).enumerate() {
                    row[diagonal_index] += 2.0;
                }

                let x_true = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
                let mut b = vec![0.0; m];
                for (rhs, row) in b.iter_mut().zip(&a) {
                    *rhs = row
                        .iter()
                        .zip(x_true.iter())
                        .map(|(coefficient, value)| coefficient * value)
                        .sum();
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
                for (i, expected_value) in x_true.iter().enumerate() {
                    let value = solution.values.get(&format!("x{i}")).copied().unwrap();
                    assert!((value - expected_value).abs() < 1e-8);
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
