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

//! Globalized Newton-Raphson iteration, regularized least squares, convergence
//! results, and structured failure diagnostics.

use crate::compiler::{CompiledSystem, EvaluationError, VarId, VarTable};
use crate::jacobian::Jacobian;
use crate::matrix::{LeastSquaresInfo, Matrix, MatrixError};
use crate::mode::{Mode, build_thread_pool};
use crate::scaled::product_ratio;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::fmt;

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
    /// Reusable transaction buffer populated with each candidate state.
    candidate_vars: &'a mut HashMap<VarId, f64>,
    /// Newton correction direction whose scalar multiplier is being searched.
    delta: &'a Matrix,
    /// Equation-scale-normalized residual norm at the current accepted point.
    current_residual_norm: f64,
    /// Directional derivative of `||f_scaled||_2` along normalized `delta`.
    ///
    /// For non-zero residual vector `f`, Jacobian `J`, and candidate direction
    /// `p`, this value is `(f_scaled / ||f_scaled||_2)^T J_scaled p`.
    /// Computing this normalized form directly avoids overflowing the equivalent
    /// unnormalized dot product before division can reduce its scale.
    residual_norm_directional_derivative: f64,
    /// Number of Newton updates accepted before this line search began.
    ///
    /// Keeping this as an accepted-update count, rather than a one-based attempt
    /// number, preserves the public `SolverRunDiagnostic::iterations` contract
    /// when every candidate is rejected and the current point is left unchanged.
    accepted_updates: usize,
    /// Residual vector at the current accepted point.
    current_residuals: &'a Matrix,
    /// Gradient measures already evaluated at the current accepted point.
    gradient: ResidualGradientMeasure,
    /// Reusable output storage for candidate residual evaluations.
    candidate_residuals: &'a mut Matrix,
    /// Reusable output storage for scaled candidate residuals used by Armijo.
    candidate_scaled_residuals: &'a mut Matrix,
}

/// Borrowed normalized state used to compute one Armijo slope while retaining
/// raw residuals for any caller-facing failure diagnostic.
struct DirectionalDerivativeContext<'a> {
    /// Equation- and variable-scaled Jacobian at the current accepted point.
    jacobian: &'a Matrix,
    /// Equation-scaled residual vector at the same point.
    scaled_residuals: &'a Matrix,
    /// Newton correction expressed in normalized variable coordinates.
    direction: &'a Matrix,
    /// Whether matrix multiplication may use the configured Rayon pool.
    parallel: bool,
    /// Accepted variables preserved for a possible numerical failure report.
    vars: &'a HashMap<VarId, f64>,
    /// Raw residuals preserved for per-equation diagnostics.
    diagnostic_residuals: &'a Matrix,
    /// Number of corrections accepted before this slope calculation.
    accepted_updates: usize,
    /// Gradient measures already evaluated at the current accepted point.
    gradient: ResidualGradientMeasure,
}

/// Metadata for a candidate that satisfied the solver's acceptance policy.
///
/// Candidate variables and residuals remain in the mutable buffers supplied to
/// [`NewtonRaphsonSolver::line_search`]. Keeping only scalar metadata here makes
/// acceptance explicit without allocating or copying a second owned state.
struct AcceptedStep {
    /// Scaled residual norm evaluated at the accepted candidate variables.
    residual_norm: f64,
}

/// Absolute and column-scaled measures of the nonlinear least-squares gradient
/// at one accepted solver state.
///
/// Keeping the two values together prevents convergence checks from using the
/// scale-sensitive absolute gradient while result diagnostics accidentally
/// report a differently evaluated quantity.
#[derive(Debug, Clone, Copy)]
struct ResidualGradientMeasure {
    /// Euclidean norm of `J_scaled^T f_scaled` in normalized variable units.
    absolute_norm: f64,
    /// Largest absolute cosine between `f` and a non-zero column of `J`.
    ///
    /// The value is absent only when `f` is zero. Zero Jacobian columns
    /// contribute zero, so an entirely zero Jacobian at a non-root produces
    /// `Some(0.0)` and is classified as first-order stationary without implying
    /// a local minimum.
    relative_norm: Option<f64>,
}

/// Internal correction vector paired with the public diagnostic that describes
/// how its linearized system was solved.
struct LinearSolveResult {
    /// Minimum-norm Newton correction produced by the solve.
    solution: Matrix,
    /// Factorization and regularization details for caller-facing results.
    diagnostic: LinearSolveDiagnostic,
}

impl ResidualGradientMeasure {
    /// Return whether the dimensionless gradient meets the configured
    /// first-order stationarity tolerance.
    fn is_stationary(self, tolerance: f64) -> bool {
        // A zero residual is handled by the root-success checks before
        // stationarity is queried. Every non-root therefore has a defined
        // column-scaled measure, including the zero-Jacobian case.
        self.relative_norm.is_some_and(|norm| norm < tolerance)
    }
}

/// Complete numerical policy for one reusable nonlinear solver.
///
/// Fields are public so pre-alpha callers can start from [`SolverOptions::default`]
/// or from [`NewtonRaphsonSolver::options`], change only the relevant values,
/// and install the result with [`NewtonRaphsonSolver::with_options`]. Fields used
/// by the selected solve path are validated when that solve begins.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverOptions {
    /// Maximum number of accepted Newton updates before `NoConvergence`.
    pub max_iterations: usize,
    /// Inclusive maximum norm of equation-scale-normalized residuals for a
    /// successful [`Solution`].
    pub residual_tolerance: f64,
    /// Maximum residual/Jacobian-column cosine used for `StationaryNonRoot`.
    pub stationarity_tolerance: f64,
    /// First and largest step multiplier tested for each Newton correction.
    pub initial_step_size: f64,
    /// Explicit ridge parameter on the normalized variable correction; zero
    /// selects the unregularized SVD solve.
    pub regularization: f64,
    /// Armijo sufficient-decrease coefficient for every accepted update.
    pub armijo_coefficient: f64,
    /// Maximum number of candidate steps evaluated by one line search.
    pub line_search_max_trials: usize,
    /// Smallest positive candidate step evaluated before reporting stagnation.
    pub line_search_min_step: f64,
}

impl Default for SolverOptions {
    /// Return the crate's single globalized Newton policy.
    fn default() -> Self {
        // Keep every numerical constant visible in one value rather than
        // scattering hidden policy through iteration and factorization bodies.
        Self {
            max_iterations: 100,
            residual_tolerance: 1e-10,
            stationarity_tolerance: 1e-10,
            initial_step_size: 1.0,
            regularization: 0.0,
            armijo_coefficient: 1e-4,
            line_search_max_trials: 64,
            line_search_min_step: 1e-12,
        }
    }
}

/// Newton-Raphson solver for systems of nonlinear equations
///
/// This solver can handle:
/// - Square systems (equations == variables)
/// - Under-constrained systems (equations < variables) - uses least squares
/// - Over-constrained systems (equations > variables) - uses least squares
///
/// Every correction is accepted transactionally by Armijo backtracking, and
/// callers may explicitly select ridge regularization in normalized units.
pub struct NewtonRaphsonSolver {
    /// Residual expressions and their simplified symbolic partial derivatives.
    ///
    /// Keeping this cache on the reusable solver avoids repeating symbolic
    /// differentiation every time [`NewtonRaphsonSolver::solve`] is called.
    jacobian: Jacobian,
    /// Variable IDs that correspond to unknowns in the system
    variables: Vec<VarId>,
    /// Registry of all variables referenced by the compiled system
    var_table: VarTable,
    /// Positive characteristic magnitude for each residual equation.
    ///
    /// Entry order matches compiled equations. The solver divides residual row
    /// `i` and Jacobian row `i` by this value, so larger scales reduce an
    /// equation's weight in convergence, stationarity, Armijo, and least
    /// squares without changing raw diagnostic residuals.
    equation_scales: Vec<f64>,
    /// Positive characteristic magnitude for each solve variable.
    ///
    /// Entry order matches `variables`. Jacobian column `j` is multiplied by
    /// this value, and accepted normalized corrections are converted back to
    /// caller units with the same factor.
    variable_scales: Vec<f64>,
    /// Numerical policy validated for the selected path at each solve boundary.
    options: SolverOptions,
    /// Optional metadata describing the source of each equation in the system
    equation_traces: Vec<Option<EquationTrace>>,
    /// Execution mode for linear algebra (serial vs parallel)
    mode: Mode,
    /// Optional thread pool for parallel execution
    pool: Option<ThreadPool>,
}

/// Successful solver result with values and explicit convergence diagnostics.
///
/// Constructing this value certifies that the scaled Euclidean residual norm is
/// at most the configured root tolerance. Stationary non-roots are returned through
/// [`SolverError::StationaryNonRoot`] and can never produce `Solution`.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Finite final values of every compiled variable, including fixed
    /// parameters.
    pub values: HashMap<String, f64>,
    /// Number of accepted Newton updates. An already-solved initial guess
    /// therefore reports zero iterations.
    pub iterations: usize,
    /// Euclidean norm of `raw_residual_i / equation_scale_i`, evaluated at
    /// exactly the values returned in `values`.
    pub error: f64,
    /// Norm of the normalized gradient `J_scaled^T f_scaled` at the returned
    /// values, measuring first-order least-squares stationarity.
    pub gradient_norm: f64,
    /// Largest absolute cosine between the residual and a Jacobian column.
    ///
    /// `None` means the residual was zero, so its direction was undefined. A
    /// successful exact root therefore reports `None` here.
    pub relative_gradient_norm: Option<f64>,
    /// Successful linearized solve attempt that produced the most recent
    /// accepted update represented by this solution.
    ///
    /// `None` means the initial point required no factorization or the first
    /// attempted factorization failed before producing diagnostics.
    pub last_linear_attempt: Option<LinearSolveDiagnostic>,
    /// Scaled residual norms for the initial point and each accepted update
    /// through the returned point.
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
    /// Dimensionless residual `residual / equation_scale` used by convergence,
    /// stationarity, Armijo acceptance, and the linearized solve.
    pub scaled_residual: f64,
    /// Caller-provided source metadata, or `None` when no trace was attached for
    /// this equation.
    pub trace: Option<EquationTrace>,
}

/// Numerical provenance for one successful linearized solve attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSolveDiagnostic {
    /// Rank and condition estimate from the SVD of the equation- and variable-
    /// scaled Jacobian that produced the normalized correction.
    pub effective: LeastSquaresInfo,
    /// Explicit ridge parameter `lambda` used in `[J; sqrt(lambda) I]`, or
    /// `None` when the direct Jacobian was solved.
    pub regularization: Option<f64>,
}

/// Complete numerical state captured when a solver run fails.
#[derive(Debug, Clone)]
pub struct SolverRunDiagnostic {
    /// Human-readable summary of the terminating failure.
    pub message: String,
    /// Number of accepted Newton updates represented by the diagnostic state.
    pub iterations: usize,
    /// Euclidean norm of all `EquationDiagnostic::scaled_residual` entries.
    pub error: f64,
    /// Norm of `J_scaled^T f_scaled` when the terminating path evaluated
    /// stationarity.
    ///
    /// Other failures report `None` because computing a fresh Jacobian merely
    /// for diagnostics could itself fail or obscure the original error.
    pub gradient_norm: Option<f64>,
    /// Largest absolute residual/Jacobian-column cosine when it was defined and
    /// evaluated at the diagnostic state.
    pub relative_gradient_norm: Option<f64>,
    /// Most recent successful linearized solve attempt represented by this
    /// failure state, whether or not line search accepted its correction.
    ///
    /// `None` means failure occurred before any correction was produced, such as
    /// an invalid initial evaluation or an initially stationary non-root.
    pub last_linear_attempt: Option<LinearSolveDiagnostic>,
    /// Variable values at the end of the run, keyed by compiled variable name.
    pub values: HashMap<String, f64>,
    /// Per-equation residuals enriched with any attached source metadata.
    pub equations: Vec<EquationDiagnostic>,
}

/// Failures returned by solver construction, validation, and iteration.
#[derive(Debug, Clone)]
pub enum SolverError {
    /// The configured update budget or line search ended without convergence.
    /// The diagnostic is boxed to keep the error enum compact.
    NoConvergence(Box<SolverRunDiagnostic>),
    /// Expression, Jacobian, update, or residual-gradient evaluation produced
    /// NaN or infinity.
    /// The diagnostic is boxed to keep the error enum compact.
    NonFiniteEvaluation(Box<SolverRunDiagnostic>),
    /// The residual remains above its root tolerance while the first-order
    /// residual gradient satisfies the stationarity tolerance.
    ///
    /// This applies uniformly to square, underdetermined, and overdetermined
    /// systems. It does not claim the state is a local minimum; stationary
    /// maxima and saddles use the same non-success result.
    StationaryNonRoot(Box<SolverRunDiagnostic>),
    /// A matrix operation failed while constructing or solving a nonlinear
    /// correction.
    Matrix {
        /// Original typed matrix failure, including machine-readable shape,
        /// operand, operation, or arithmetic metadata.
        source: MatrixError,
        /// Complete accepted nonlinear state at which the matrix failure
        /// occurred.
        diagnostic: Box<SolverRunDiagnostic>,
    },
    /// Invalid input parameters or system setup.
    InvalidInput(String),
}

impl fmt::Display for SolverError {
    /// Format the concise diagnostic summary while preserving structured state
    /// for callers that pattern-match the error.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::NoConvergence(diagnostic) => {
                write!(f, "Solver did not converge: {}", diagnostic.message)
            }
            SolverError::NonFiniteEvaluation(diagnostic) => {
                write!(f, "Non-finite solver evaluation: {}", diagnostic.message)
            }
            SolverError::StationaryNonRoot(diagnostic) => {
                write!(
                    f,
                    "Stationary point is not a constraint root: {}",
                    diagnostic.message
                )
            }
            SolverError::Matrix { source, diagnostic } => write!(
                f,
                "Correction matrix operation failed after {} accepted updates: {source}",
                diagnostic.iterations
            ),
            SolverError::InvalidInput(message) => write!(f, "Invalid solver input: {message}"),
        }
    }
}

impl std::error::Error for SolverError {
    /// Expose nested structured causes through the standard error chain.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Run diagnostics and input messages originate at this solver layer.
        // Only a retained matrix failure has a lower-level source object.
        match self {
            SolverError::Matrix { source, .. } => Some(source),
            SolverError::NoConvergence(_)
            | SolverError::NonFiniteEvaluation(_)
            | SolverError::StationaryNonRoot(_)
            | SolverError::InvalidInput(_) => None,
        }
    }
}

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

impl NewtonRaphsonSolver {
    /// Create a Newton-Raphson solver with one shape-independent policy.
    ///
    /// Square, tall, and wide systems begin with identical iteration,
    /// backtracking, tolerance, and regularization settings. Callers can replace
    /// those explicit defaults through [`NewtonRaphsonSolver::with_options`].
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
        // Symbolic differentiation depends only on the compiled equations and
        // selected solve variables, both of which are immutable for the
        // solver's lifetime. Construct and simplify every derivative here so
        // repeated numerical solves share the result.
        let jacobian = Jacobian::new(compiled.equations, variables.clone());
        let equation_scales = vec![1.0; jacobian.equation_count()];
        let variable_scales = vec![1.0; variables.len()];

        NewtonRaphsonSolver {
            jacobian,
            variables,
            var_table: compiled.var_table,
            // Unit scales preserve historical unscaled equations and variable
            // coordinates until a caller explicitly supplies characteristic
            // magnitudes.
            equation_scales,
            variable_scales,
            // One numerical policy applies to square, tall, and wide systems.
            // Shape-specific hidden defaults would make otherwise identical
            // APIs change iteration budgets and regularization unexpectedly.
            options: SolverOptions::default(),
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

    /// Borrow the complete numerical policy currently installed on the solver.
    pub fn options(&self) -> &SolverOptions {
        // Returning a shared reference makes inspection allocation-free while
        // preventing validation from being bypassed during an active solve.
        &self.options
    }

    /// Replace the complete numerical policy used by subsequent solves.
    pub fn with_options(mut self, options: SolverOptions) -> Self {
        // Preserve caller values verbatim. The solve boundary validates the
        // fields that can affect its selected numerical path.
        self.options = options;
        self
    }

    /// Set one positive characteristic scale per compiled residual equation.
    ///
    /// Residual `f_i` is normalized as `f_i / scale_i`. Consequently the
    /// configured residual tolerance is dimensionless, and choosing a larger
    /// characteristic scale gives that equation less least-squares weight. The
    /// vector must follow compilation order and contain exactly one entry per
    /// equation.
    pub fn with_equation_scales(mut self, scales: Vec<f64>) -> Result<Self, SolverError> {
        if scales.len() != self.jacobian.equation_count() {
            return Err(SolverError::InvalidInput(format!(
                "Equation scale length ({}) does not match number of equations ({})",
                scales.len(),
                self.jacobian.equation_count()
            )));
        }
        if let Some((index, _)) = scales
            .iter()
            .enumerate()
            .find(|(_, scale)| !scale.is_finite() || **scale <= 0.0)
        {
            return Err(SolverError::InvalidInput(format!(
                "Equation scale at index {index} must be finite and greater than zero"
            )));
        }
        self.equation_scales = scales;
        Ok(self)
    }

    /// Override characteristic scales for named solve variables.
    ///
    /// Variable `x_j` is normalized as `x_j / scale_j`, so the linearized
    /// Jacobian column is multiplied by `scale_j` and a normalized correction is
    /// multiplied by the same value before updating caller-visible variables.
    /// Names omitted from the map retain their current scale, initially one.
    /// Unknown names and fixed parameters are rejected rather than ignored.
    pub fn with_variable_scales(
        mut self,
        scales: HashMap<String, f64>,
    ) -> Result<Self, SolverError> {
        // Sort the caller's unordered map so the first reported invalid entry is
        // deterministic across processes and hash seeds.
        let mut entries: Vec<_> = scales.into_iter().collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));

        for (name, scale) in entries {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(SolverError::InvalidInput(format!(
                    "Variable scale for '{name}' must be finite and greater than zero"
                )));
            }
            let variable_id = self.var_table.get_id(&name).ok_or_else(|| {
                SolverError::InvalidInput(format!("Unknown variable scale name '{name}'"))
            })?;
            let column = self
                .variables
                .iter()
                .position(|candidate| *candidate == variable_id)
                .ok_or_else(|| {
                    SolverError::InvalidInput(format!(
                        "Variable scale '{name}' refers to a fixed parameter, not a solve variable"
                    ))
                })?;
            self.variable_scales[column] = scale;
        }
        Ok(self)
    }

    /// Borrow equation characteristic scales in compilation order.
    pub fn equation_scales(&self) -> &[f64] {
        &self.equation_scales
    }

    /// Return the characteristic scale of a named solve variable.
    ///
    /// `None` indicates that the name is unknown or belongs to a fixed
    /// parameter rather than a Jacobian column.
    pub fn variable_scale(&self, name: &str) -> Option<f64> {
        let variable_id = self.var_table.get_id(name)?;
        let column = self
            .variables
            .iter()
            .position(|candidate| *candidate == variable_id)?;
        Some(self.variable_scales[column])
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
        if traces.len() != self.jacobian.equation_count() {
            return Err(SolverError::InvalidInput(format!(
                "Equation trace length ({}) does not match number of equations ({})",
                traces.len(),
                self.jacobian.equation_count()
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
        self.options.max_iterations = max_iterations;
        self
    }

    /// Override the positive, finite scaled residual norm required for success.
    ///
    /// The solver deliberately does not treat a small applied update as
    /// convergence because backtracking can make a useful correction
    /// arbitrarily small. Configuration is validated when a solve begins.
    pub fn with_residual_tolerance(mut self, residual_tolerance: f64) -> Self {
        // Store without clamping so NaN, infinity, zero, and negative values are
        // observable to the common configuration validator.
        self.options.residual_tolerance = residual_tolerance;
        self
    }

    /// Override the positive, finite column-scaled threshold used to
    /// identify stationary non-roots.
    ///
    /// This value does not permit a successful least-squares result. It controls
    /// when the solver stops with [`SolverError::StationaryNonRoot`] instead of
    /// spending the remaining update budget at a first-order fixed point.
    pub fn with_stationarity_tolerance(mut self, stationarity_tolerance: f64) -> Self {
        // Preserve invalid values for deterministic validation at the shared
        // solve boundary rather than silently clamping numerical policy.
        self.options.stationarity_tolerance = stationarity_tolerance;
        self
    }

    /// Override the finite first Armijo trial multiplier in `(0, 1]`.
    ///
    /// A value below one deliberately caps every correction before
    /// backtracking. Configuration is validated when a solve begins; invalid
    /// values are retained so the caller receives a deterministic error.
    pub fn with_initial_step_size(mut self, initial_step_size: f64) -> Self {
        // Store the exact policy value rather than silently clamping it into the
        // valid interval and hiding configuration mistakes.
        self.options.initial_step_size = initial_step_size;
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
        self.options.regularization = regularization;
        self
    }

    /// Solve the system using Newton corrections accepted by Armijo backtracking.
    pub fn solve(&self, initial_guess: HashMap<String, f64>) -> Result<Solution, SolverError> {
        // Reject invalid reusable settings before translating or validating the
        // caller's variable map.
        self.validate_configuration()?;
        let vars = self.map_initial_guess(initial_guess)?;
        self.run_in_mode(|| self.solve_internal(vars))
    }

    /// Run the solver's single accepted-state Newton iteration.
    ///
    /// # Arguments
    /// * `initial_guess` - Starting values for all variables (by name)
    /// # Algorithm
    /// 1. Evaluate `f(x)` and `J(x)` for the current accepted state
    /// 2. Check root and first-order stationarity termination
    /// 3. Solve the normalized `J * delta = -f(x)` system by SVD, augmenting
    ///    it once with the caller's exact ridge strength when requested
    /// 4. Backtrack transactionally until Armijo sufficient decrease holds
    /// 5. Commit the already-evaluated candidate as the next accepted state
    fn solve_internal(&self, initial_guess: HashMap<VarId, f64>) -> Result<Solution, SolverError> {
        // The public entry point has validated both the selected policy and this
        // owned internal map, so move it directly into mutable solver state.
        let mut vars = initial_guess;
        // Borrow the construction-time symbolic cache and allocate only the
        // mutable numerical derivative storage owned by this solve invocation.
        let jacobian = &self.jacobian;
        let mut jacobian_workspace = jacobian.workspace();
        // Grow history with actual progress instead of reserving the caller's
        // entire theoretical update budget. A very large valid budget must not
        // panic before an already-solved initial point can be inspected.
        let mut convergence_history = Vec::new();
        // Retain the most recent successful linear solve attempt independently
        // from line-search acceptance. This is attempt provenance, while `iter`
        // and convergence history continue to count accepted state changes.
        let mut last_linear_attempt: Option<LinearSolveDiagnostic> = None;

        let num_equations = jacobian.equation_count();
        let mut f_vals = Matrix::new(num_equations, 1);
        let mut scaled_f_vals = Matrix::new(num_equations, 1);
        let mut f_neg = Matrix::new(num_equations, 1);
        let mut scaled_jacobian = Matrix::new(num_equations, self.variables.len());
        // Populate the complete initial accepted state once. Subsequent
        // residuals arrive from successful line-search candidates, while each
        // newly accepted point receives exactly one Jacobian evaluation.
        jacobian.evaluate_functions_checked_into(&vars, &mut f_vals)?;
        jacobian.evaluate_checked_in_workspace(&vars, &mut jacobian_workspace)?;
        self.scale_residuals_into(&f_vals, &mut scaled_f_vals);
        self.scale_jacobian_into(jacobian_workspace.jacobian(), &mut scaled_jacobian);

        // Candidate buffers are separate from accepted state until Armijo
        // succeeds. Swapping them on acceptance avoids both rollback logic and
        // reevaluation of an already-tested residual vector.
        let mut candidate_vars = vars.clone();
        let mut candidate_f_vals = Matrix::new(num_equations, 1);
        let mut candidate_scaled_f_vals = Matrix::new(num_equations, 1);

        let parallel = self.parallel_enabled();

        if !f_vals.all_finite() {
            return Err(self.non_finite_evaluation_error(
                "Residual evaluation produced NaN or infinity",
                0,
                &vars,
                &f_vals,
            ));
        }
        if !jacobian_workspace.jacobian().all_finite() {
            return Err(self.non_finite_evaluation_error(
                "Jacobian evaluation produced NaN or infinity",
                0,
                &vars,
                &f_vals,
            ));
        }
        if !scaled_f_vals.all_finite() || !scaled_jacobian.all_finite() {
            return Err(self.non_finite_evaluation_error(
                "Equation or variable scaling produced NaN or infinity",
                0,
                &vars,
                &f_vals,
            ));
        }
        convergence_history.push(scaled_f_vals.norm());

        // The current state is always fully evaluated at loop entry. A mutable
        // accepted-update count lets the budget-exhausted state receive the same
        // root and stationarity checks without a special final evaluation path.
        let mut iter = 0;
        loop {
            let error = scaled_f_vals.norm();

            // A successful result requires the actual constraints to satisfy
            // the residual threshold before any mutation is attempted. Every
            // accepted iteration below is therefore one residual correction;
            // no secondary point-selection policy can move or starve a root.
            if error <= self.options.residual_tolerance {
                // The Jacobian workspace already corresponds to `vars`, so the
                // reported gradient norm is evaluated at the same point as the
                // residual and returned values.
                let gradient = Self::residual_gradient_measure(&scaled_jacobian, &scaled_f_vals);
                return Ok(self.build_solution(
                    &vars,
                    iter,
                    error,
                    gradient,
                    last_linear_attempt,
                    convergence_history,
                ));
            }

            // Every system shape can reach a first-order fixed point that is not
            // a root, particularly after rank truncation or regularization.
            // Detect it before another identical linear solve or backtracking
            // attempt and return a non-success result at the stationary state.
            let gradient = Self::residual_gradient_measure(&scaled_jacobian, &scaled_f_vals);
            if gradient.is_stationary(self.options.stationarity_tolerance) {
                return Err(self.stationary_non_root_error(
                    iter,
                    &vars,
                    &f_vals,
                    gradient,
                    last_linear_attempt,
                ));
            }

            // Root and stationarity checks deliberately precede this boundary:
            // the final permitted update may legitimately land on either. Only
            // a still-active state is classified as budget exhaustion.
            if iter == self.options.max_iterations {
                let diagnostic = self.build_diagnostic(
                    format!(
                        "Failed to converge after {} accepted updates. Final error: {:.2e}",
                        self.options.max_iterations, error
                    ),
                    iter,
                    &vars,
                    &f_vals,
                    Some(gradient),
                    last_linear_attempt,
                );
                return Err(SolverError::NoConvergence(Box::new(diagnostic)));
            }

            for i in 0..num_equations {
                f_neg[(i, 0)] = -scaled_f_vals[(i, 0)];
            }

            // Step 2: Solve J * delta = -f(x) through one shape-independent
            // least-squares path. A wide Jacobian naturally yields the
            // Moore-Penrose minimum-norm correction, which preserves the
            // current null-space component without introducing a second point
            // objective into nonlinear root finding.
            let linear_solve = self
                .solve_least_squares_system(&scaled_jacobian, &f_neg)
                .map_err(|source| {
                    self.matrix_error(
                        source,
                        iter,
                        &vars,
                        &f_vals,
                        gradient,
                        last_linear_attempt.clone(),
                    )
                })?;
            last_linear_attempt = Some(linear_solve.diagnostic.clone());
            let delta = linear_solve.solution;

            if !delta.all_finite() {
                return Err(Self::attach_linear_attempt(
                    self.non_finite_evaluation_error(
                        "Linear solve produced a non-finite Newton correction",
                        iter,
                        &vars,
                        &f_vals,
                    ),
                    last_linear_attempt,
                ));
            }

            // Armijo is the only update policy. It leaves the accepted state
            // untouched while populating candidate buffers, so failure cannot
            // require rollback or consume an update count.
            let residual_norm_directional_derivative = self
                .residual_norm_directional_derivative(DirectionalDerivativeContext {
                    jacobian: &scaled_jacobian,
                    scaled_residuals: &scaled_f_vals,
                    direction: &delta,
                    parallel,
                    vars: &vars,
                    diagnostic_residuals: &f_vals,
                    accepted_updates: iter,
                    gradient,
                })
                .map_err(|error| Self::attach_linear_attempt(error, last_linear_attempt.clone()))?;
            let accepted_step = self
                .line_search(LineSearchContext {
                    jacobian,
                    vars: &vars,
                    candidate_vars: &mut candidate_vars,
                    delta: &delta,
                    current_residual_norm: error,
                    residual_norm_directional_derivative,
                    accepted_updates: iter,
                    current_residuals: &f_vals,
                    gradient,
                    candidate_residuals: &mut candidate_f_vals,
                    candidate_scaled_residuals: &mut candidate_scaled_f_vals,
                })
                .map_err(|error| Self::attach_linear_attempt(error, last_linear_attempt.clone()))?;

            // The candidate residual has already passed finite and Armijo
            // checks. Swap both buffers together so variables and residuals
            // become one atomic accepted state without evaluating f(x) again.
            std::mem::swap(&mut vars, &mut candidate_vars);
            std::mem::swap(&mut f_vals, &mut candidate_f_vals);
            std::mem::swap(&mut scaled_f_vals, &mut candidate_scaled_f_vals);

            // Only the Jacobian was not needed by candidate acceptance. Evaluate
            // it once for the newly committed state and retain it through the
            // next iteration's convergence and correction calculations.
            jacobian.evaluate_checked_in_workspace(&vars, &mut jacobian_workspace)?;
            if !jacobian_workspace.jacobian().all_finite() {
                return Err(Self::attach_linear_attempt(
                    self.non_finite_evaluation_error(
                        "Accepted Newton update produced a non-finite Jacobian",
                        iter + 1,
                        &vars,
                        &f_vals,
                    ),
                    last_linear_attempt,
                ));
            }
            self.scale_jacobian_into(jacobian_workspace.jacobian(), &mut scaled_jacobian);
            if !scaled_jacobian.all_finite() {
                return Err(Self::attach_linear_attempt(
                    self.non_finite_evaluation_error(
                        "Variable or equation scaling produced a non-finite Jacobian",
                        iter + 1,
                        &vars,
                        &f_vals,
                    ),
                    last_linear_attempt,
                ));
            }
            let updated_error = accepted_step.residual_norm;
            convergence_history.push(updated_error);
            iter += 1;

            // Root and stationarity checks occur once at the next loop entry,
            // using the residual and Jacobian for exactly this accepted state.
        }
    }

    /// Solve one linearized correction using the caller's explicit ridge policy.
    ///
    /// A zero ridge parameter solves the Jacobian directly. A positive value
    /// solves the single augmented system `[J; sqrt(lambda) I] delta ~= [rhs; 0]`
    /// without first computing and discarding an unregularized correction.
    /// Both routes use the same canonical SVD implementation and preserve its
    /// exact structured error on failure.
    fn solve_least_squares_system(
        &self,
        j_matrix: &Matrix,
        rhs: &Matrix,
    ) -> Result<LinearSolveResult, MatrixError> {
        if self.options.regularization == 0.0 {
            // The direct SVD path returns the minimum-norm correction for every
            // rank. Conditioning remains diagnostic data; it never silently
            // changes the equation being solved.
            let result = j_matrix.solve_least_squares(rhs)?;
            return Ok(LinearSolveResult {
                solution: result.solution,
                diagnostic: LinearSolveDiagnostic {
                    effective: result.info,
                    regularization: None,
                },
            });
        }

        // Ridge regularization is an explicit alternate linear problem. Build
        // its storage only when the caller selected it; exact roots and default
        // unregularized solves never allocate an augmented matrix.
        let augmented_rows = j_matrix.rows().checked_add(j_matrix.cols()).ok_or(
            MatrixError::DimensionSumOverflow {
                operation: "regularized least-squares allocation",
                left: j_matrix.rows(),
                right: j_matrix.cols(),
            },
        )?;
        let mut augmented_j = Matrix::try_new(augmented_rows, j_matrix.cols())?;
        let mut augmented_b = Matrix::try_new(augmented_rows, 1)?;

        // Copy the nonlinear linearization into the upper block. Fresh
        // zero-filled matrices already contain the lower off-diagonal and
        // right-hand-side zeros required by the ridge formulation.
        for row in 0..j_matrix.rows() {
            for column in 0..j_matrix.cols() {
                augmented_j[(row, column)] = j_matrix[(row, column)];
            }
            augmented_b[(row, 0)] = rhs[(row, 0)];
        }

        // sqrt(lambda) on the appended diagonal makes the augmented
        // least-squares objective exactly ||J delta-rhs||^2 + lambda||delta||^2.
        let sqrt_regularization = self.options.regularization.sqrt();
        for variable in 0..j_matrix.cols() {
            augmented_j[(j_matrix.rows() + variable, variable)] = sqrt_regularization;
        }

        let result = augmented_j.solve_least_squares(&augmented_b)?;
        Ok(LinearSolveResult {
            solution: result.solution,
            diagnostic: LinearSolveDiagnostic {
                effective: result.info,
                regularization: Some(self.options.regularization),
            },
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

    /// Convert one option predicate into the solver's consistent configuration
    /// error representation.
    fn require_valid_option(valid: bool, message: &'static str) -> Result<(), SolverError> {
        // Allocate the caller-facing message only on the invalid path; successful
        // validation remains a branch with no heap work.
        valid
            .then_some(())
            .ok_or_else(|| SolverError::InvalidInput(message.to_string()))
    }

    /// Validate the solver's complete numerical policy before evaluation.
    fn validate_configuration(&self) -> Result<(), SolverError> {
        // One update path consumes every field, so validation no longer depends
        // on which public entry point happened to be selected.
        Self::require_valid_option(
            self.options.max_iterations > 0,
            "max_iterations must be greater than zero",
        )?;
        Self::require_valid_option(
            self.options.residual_tolerance.is_finite() && self.options.residual_tolerance > 0.0,
            "residual_tolerance must be finite and greater than zero",
        )?;
        Self::require_valid_option(
            self.options.stationarity_tolerance.is_finite()
                && self.options.stationarity_tolerance > 0.0
                && self.options.stationarity_tolerance <= 1.0,
            "stationarity_tolerance must be finite and in the interval (0, 1]",
        )?;
        Self::require_valid_option(
            self.options.initial_step_size.is_finite()
                && self.options.initial_step_size > 0.0
                && self.options.initial_step_size <= 1.0,
            "initial_step_size must be finite and in the interval (0, 1]",
        )?;
        Self::require_valid_option(
            self.options.regularization.is_finite() && self.options.regularization >= 0.0,
            "regularization must be finite and non-negative",
        )?;
        Self::require_valid_option(
            self.options.armijo_coefficient.is_finite()
                && self.options.armijo_coefficient > 0.0
                && self.options.armijo_coefficient < 1.0,
            "armijo_coefficient must be finite and in the interval (0, 1)",
        )?;
        Self::require_valid_option(
            self.options.line_search_max_trials > 0,
            "line_search_max_trials must be greater than zero",
        )?;
        Self::require_valid_option(
            self.options.line_search_min_step.is_finite()
                && self.options.line_search_min_step > 0.0
                && self.options.line_search_min_step <= self.options.initial_step_size,
            "line_search_min_step must be finite, positive, and no greater than initial_step_size",
        )?;

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

    /// Normalize raw residuals by their positive equation characteristic scales.
    fn scale_residuals_into(&self, raw: &Matrix, scaled: &mut Matrix) {
        if scaled.rows() != raw.rows() || scaled.cols() != 1 {
            *scaled = Matrix::new(raw.rows(), 1);
        }
        for row in 0..raw.rows() {
            scaled[(row, 0)] = raw[(row, 0)] / self.equation_scales[row];
        }
    }

    /// Return the norm used by convergence and failure diagnostics without
    /// allocating a temporary scaled residual vector.
    fn scaled_residual_norm(&self, raw: &Matrix) -> f64 {
        let mut norm = 0.0_f64;
        for row in 0..raw.rows() {
            norm = norm.hypot(raw[(row, 0)] / self.equation_scales[row]);
        }
        norm
    }

    /// Normalize a raw Jacobian for equation and variable characteristic scales.
    ///
    /// The normalized derivative is `J_ij * variable_scale_j /
    /// equation_scale_i`. The shared exponent-scaled product policy prevents
    /// either source-code association from manufacturing zero or infinity when
    /// the final normalized derivative remains representable.
    fn scale_jacobian_into(&self, raw: &Matrix, scaled: &mut Matrix) {
        if scaled.rows() != raw.rows() || scaled.cols() != raw.cols() {
            *scaled = Matrix::new(raw.rows(), raw.cols());
        }
        for row in 0..raw.rows() {
            let equation_scale = self.equation_scales[row];
            for column in 0..raw.cols() {
                let variable_scale = self.variable_scales[column];
                let derivative = raw[(row, column)];
                if !derivative.is_finite() {
                    // Preserve the original invalid derivative verbatim. The
                    // caller's post-scaling finite check then reports the
                    // established Jacobian evaluation error instead of asking
                    // finite-only normalization arithmetic to classify it.
                    scaled[(row, column)] = derivative;
                    continue;
                }
                // Decomposing all three values into bounded significands and
                // integer exponents makes the result independent of an
                // arbitrary multiplication/division association.
                scaled[(row, column)] =
                    product_ratio([derivative, variable_scale], [equation_scale]);
            }
        }
    }

    /// Multiply a normalized correction by its step and variable scale without
    /// order-dependent intermediate overflow or underflow.
    fn scaled_update_component(delta: f64, step_size: f64, variable_scale: f64) -> f64 {
        // One exponent accumulation now serves both this update path and
        // Jacobian normalization instead of maintaining association heuristics
        // that cover only selected scale orderings.
        product_ratio([delta, step_size, variable_scale], [])
    }

    /// Apply one normalized correction atomically in caller variable units.
    ///
    /// The first pass validates the complete candidate without changing
    /// `vars`. The second pass commits the same arithmetic only after success,
    /// so a failure cannot leave a mixture of old and partially updated values.
    /// Exponent-scaled products let backtracking and variable scaling recover
    /// representable updates without depending on one multiplication order.
    /// Fixed parameters are absent from `self.variables` and remain untouched.
    fn apply_finite_step(
        &self,
        vars: &mut HashMap<VarId, f64>,
        delta: &Matrix,
        step_size: f64,
    ) -> bool {
        if self
            .variables
            .iter()
            .copied()
            .enumerate()
            .any(|(index, var_id)| {
                let update = Self::scaled_update_component(
                    delta[(index, 0)],
                    step_size,
                    self.variable_scales[index],
                );
                let value = vars[&var_id] + update;
                !value.is_finite()
            })
        {
            return false;
        }

        for (index, &var_id) in self.variables.iter().enumerate() {
            let update = Self::scaled_update_component(
                delta[(index, 0)],
                step_size,
                self.variable_scales[index],
            );
            let value = vars[&var_id] + update;
            vars.insert(var_id, value);
        }
        true
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
        gradient: ResidualGradientMeasure,
        last_linear_attempt: Option<LinearSolveDiagnostic>,
        convergence_history: Vec<f64>,
    ) -> Solution {
        // Convert internal IDs only after all numerical convergence checks have
        // succeeded, keeping the hot iteration path in its compact ID form.
        Solution {
            values: self.values_by_name(vars),
            iterations,
            error,
            gradient_norm: gradient.absolute_norm,
            relative_gradient_norm: gradient.relative_norm,
            last_linear_attempt,
            convergence_history,
        }
    }

    /// Construct the shape-independent error returned when a non-root satisfies
    /// the configured first-order stationarity threshold.
    fn stationary_non_root_error(
        &self,
        iterations: usize,
        vars: &HashMap<VarId, f64>,
        residuals: &Matrix,
        gradient: ResidualGradientMeasure,
        last_linear_attempt: Option<LinearSolveDiagnostic>,
    ) -> SolverError {
        // Pass the already-computed gradient and linear attempt into the common
        // constructor so no second evaluation or later field mutation can
        // disagree with the terminating state.
        let diagnostic = self.build_diagnostic(
            format!(
                "Residual norm {:.2e} exceeds tolerance {:.2e}, while the normalized residual gradient satisfies stationarity tolerance {:.2e}",
                self.scaled_residual_norm(residuals),
                self.options.residual_tolerance,
                self.options.stationarity_tolerance,
            ),
            iterations,
            vars,
            residuals,
            Some(gradient),
            last_linear_attempt,
        );
        SolverError::StationaryNonRoot(Box::new(diagnostic))
    }

    /// Combine a typed matrix failure with the complete accepted nonlinear
    /// state at which correction construction or related linear algebra failed.
    fn matrix_error(
        &self,
        source: MatrixError,
        iterations: usize,
        vars: &HashMap<VarId, f64>,
        residuals: &Matrix,
        gradient: ResidualGradientMeasure,
        last_linear_attempt: Option<LinearSolveDiagnostic>,
    ) -> SolverError {
        // The current loop state has already evaluated both residual and
        // gradient. Preserve them rather than forcing callers to choose between
        // the nonlinear context and the machine-readable matrix cause.
        let diagnostic = self.build_diagnostic(
            format!("Correction matrix operation failed: {source}"),
            iterations,
            vars,
            residuals,
            Some(gradient),
            last_linear_attempt,
        );
        SolverError::Matrix {
            source,
            diagnostic: Box::new(diagnostic),
        }
    }

    /// Attach the last successful linear attempt to any run-state error
    /// produced after that correction was available.
    fn attach_linear_attempt(
        mut error: SolverError,
        last_linear_attempt: Option<LinearSolveDiagnostic>,
    ) -> SolverError {
        // Every numerical run-state variant owns the same boxed diagnostic
        // shape. Invalid configuration errors occur before correction and
        // therefore have no attachment point.
        let diagnostic = match &mut error {
            SolverError::NoConvergence(diagnostic)
            | SolverError::NonFiniteEvaluation(diagnostic)
            | SolverError::StationaryNonRoot(diagnostic) => Some(diagnostic),
            SolverError::Matrix { diagnostic, .. } => Some(diagnostic),
            SolverError::InvalidInput(_) => None,
        };
        if let Some(diagnostic) = diagnostic {
            diagnostic.last_linear_attempt = last_linear_attempt;
        }
        error
    }

    /// Compute the normalized least-squares gradient norm and its column-scaled
    /// first-order stationarity measure.
    ///
    /// The decision quantity is `max_j |J_j^T f| / (||J_j||_2 ||f||_2)`, the
    /// largest absolute cosine between the residual and any non-zero Jacobian
    /// column. Normalizing each column independently prevents an unrelated
    /// high-scale variable from hiding a strong descent direction in another
    /// column, which a single Frobenius normalization of the whole Jacobian can
    /// do. Entries are divided by finite maxima before products are formed so
    /// the criterion also remains defined when the raw norms exceed `f64`.
    fn residual_gradient_measure(jacobian: &Matrix, residuals: &Matrix) -> ResidualGradientMeasure {
        // A zero residual has no direction and is already handled as a root by
        // every caller. All residual entries were checked for finiteness before
        // this helper is reached.
        let residual_scale = (0..residuals.rows())
            .map(|row| residuals[(row, 0)].abs())
            .fold(0.0_f64, f64::max);
        if residual_scale == 0.0 {
            return ResidualGradientMeasure {
                absolute_norm: 0.0,
                relative_norm: None,
            };
        }

        // The norm of the max-scaled residual is finite and non-zero. `hypot`
        // keeps the accumulation stable without allocating a normalized vector.
        let mut scaled_residual_norm = 0.0_f64;
        for row in 0..residuals.rows() {
            scaled_residual_norm = scaled_residual_norm.hypot(residuals[(row, 0)] / residual_scale);
        }

        let mut absolute_norm = 0.0_f64;
        let mut maximum_cosine = 0.0_f64;
        for column in 0..jacobian.cols() {
            let column_scale = (0..jacobian.rows())
                .map(|row| jacobian[(row, column)].abs())
                .fold(0.0_f64, f64::max);
            if column_scale == 0.0 {
                // A zero column contributes neither a gradient component nor
                // a residual alignment. An entirely zero Jacobian consequently
                // yields the mathematically stationary value `Some(0.0)`.
                continue;
            }

            let mut scaled_column_norm = 0.0_f64;
            let mut dot = 0.0;
            let mut compensation = 0.0;
            for row in 0..jacobian.rows() {
                let scaled_jacobian = jacobian[(row, column)] / column_scale;
                let scaled_residual = residuals[(row, 0)] / residual_scale;
                scaled_column_norm = scaled_column_norm.hypot(scaled_jacobian);
                let product = scaled_jacobian * scaled_residual;

                // Neumaier compensation preserves genuine cancellation at a
                // stationary least-squares point without reconstructing raw,
                // potentially overflowing products first.
                let next = dot + product;
                compensation += if dot.abs() >= product.abs() {
                    (dot - next) + product
                } else {
                    (product - next) + dot
                };
                dot = next;
            }

            let scaled_dot = dot + compensation;
            let cosine = (scaled_dot.abs() / scaled_column_norm / scaled_residual_norm).min(1.0);
            maximum_cosine = maximum_cosine.max(cosine);

            // Reconstruct the normalized gradient component for diagnostics,
            // multiplying by the smaller internal scale first to avoid needless
            // overflow. An infinite norm remains valid when its mathematical
            // magnitude exceeds `f64`; it does not affect the finite cosine.
            let gradient_component = if column_scale < residual_scale {
                (scaled_dot * column_scale) * residual_scale
            } else {
                (scaled_dot * residual_scale) * column_scale
            };
            absolute_norm = absolute_norm.hypot(gradient_component);
        }

        ResidualGradientMeasure {
            absolute_norm,
            relative_norm: Some(maximum_cosine),
        }
    }

    /// Compute the directional derivative of the scaled residual norm along a
    /// proposed normalized solver direction.
    ///
    /// Away from a root the derivative is
    /// `(f_scaled / ||f_scaled||_2)^T J_scaled p`. Normalizing the residual
    /// before the dot product prevents large finite values and velocities from
    /// overflowing the mathematically equivalent unnormalized expression.
    fn residual_norm_directional_derivative(
        &self,
        context: DirectionalDerivativeContext<'_>,
    ) -> Result<f64, SolverError> {
        let DirectionalDerivativeContext {
            jacobian,
            scaled_residuals,
            direction,
            parallel,
            vars,
            diagnostic_residuals,
            accepted_updates,
            gradient,
        } = context;

        // Multiplying J by the candidate direction produces the residual-space
        // velocity used by the chain rule. The multiplication honors the
        // solver's selected execution mode and retains structured shape errors.
        let residual_velocity = jacobian
            .try_mul_with_parallel(direction, parallel)
            .map_err(|source| {
                self.matrix_error(
                    source,
                    accepted_updates,
                    vars,
                    diagnostic_residuals,
                    gradient,
                    None,
                )
            })?;
        if !residual_velocity.all_finite() {
            return Err(self.non_finite_evaluation_error(
                "Line-search directional derivative produced NaN or infinity",
                accepted_updates,
                vars,
                diagnostic_residuals,
            ));
        }

        // The iteration requests a slope only for non-roots, so a positive
        // finite norm must exist. Rejecting an unexpected zero or non-finite
        // norm here keeps this private helper safe if future call sites change.
        let residual_norm = scaled_residuals.norm();
        if !residual_norm.is_finite() || residual_norm <= 0.0 {
            return Err(self.non_finite_evaluation_error(
                "Line-search residual norm was zero or non-finite",
                accepted_updates,
                vars,
                diagnostic_residuals,
            ));
        }

        // Accumulate the normalized dot product with Neumaier compensation.
        // Dividing each residual first keeps every product representable in
        // cases where the old raw residual-times-velocity product overflowed.
        let mut sum = 0.0;
        let mut compensation = 0.0;
        for row in 0..scaled_residuals.rows() {
            let normalized_residual = scaled_residuals[(row, 0)] / residual_norm;
            let product = normalized_residual * residual_velocity[(row, 0)];
            if !product.is_finite() {
                return Err(self.non_finite_evaluation_error(
                    "Line-search residual-norm slope overflowed",
                    accepted_updates,
                    vars,
                    diagnostic_residuals,
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

        let residual_norm_derivative = sum + compensation;
        if !residual_norm_derivative.is_finite() {
            return Err(self.non_finite_evaluation_error(
                "Line-search residual-norm slope accumulation overflowed",
                accepted_updates,
                vars,
                diagnostic_residuals,
            ));
        }
        Ok(residual_norm_derivative)
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
        SolverError::NonFiniteEvaluation(Box::new(self.build_diagnostic(
            message.into(),
            iterations,
            vars,
            residuals,
            None,
            None,
        )))
    }

    /// Capture a failure message together with the exact variable and
    /// per-equation residual state at which it occurred.
    ///
    /// Optional measures and linear provenance must already describe this same
    /// state. Accepting them here prevents terminal branches from constructing
    /// a partial diagnostic and forgetting to attach data they computed.
    fn build_diagnostic(
        &self,
        message: String,
        iterations: usize,
        vars: &HashMap<VarId, f64>,
        residuals_matrix: &Matrix,
        gradient: Option<ResidualGradientMeasure>,
        last_linear_attempt: Option<LinearSolveDiagnostic>,
    ) -> SolverRunDiagnostic {
        // Residual row order matches compiled equation order. Clone optional
        // trace metadata into the owned diagnostic so it remains useful after
        // the solver itself has been dropped.
        let mut equations = Vec::with_capacity(residuals_matrix.rows());
        for equation_index in 0..residuals_matrix.rows() {
            equations.push(EquationDiagnostic {
                equation_index,
                residual: residuals_matrix[(equation_index, 0)],
                scaled_residual: residuals_matrix[(equation_index, 0)]
                    / self.equation_scales[equation_index],
                trace: self.trace_for_equation(equation_index).cloned(),
            });
        }

        SolverRunDiagnostic {
            message,
            iterations,
            error: self.scaled_residual_norm(residuals_matrix),
            gradient_norm: gradient.map(|measure| measure.absolute_norm),
            relative_gradient_norm: gradient.and_then(|measure| measure.relative_norm),
            last_linear_attempt,
            values: self.values_by_name(vars),
            equations,
        }
    }

    /// Backtrack until a trial step satisfies sufficient decrease in `||f||`.
    ///
    /// The Armijo slope is supplied directly for the residual-norm merit
    /// function. Testing `||f||` directly avoids overflow from squaring a large
    /// but finite norm or forming the unnormalized product `f^T J delta`.
    ///
    /// # Context fields
    /// * `jacobian` - Jacobian evaluator for computing function values
    /// * `vars` - Current variable values
    /// * `candidate_vars` - Reusable transaction buffer for proposed values
    /// * `delta` - Newton step direction
    /// * `current_residual_norm` - Current scaled error `||f_scaled(x)||`
    /// * `residual_norm_directional_derivative` - Scale-safe value of
    ///   `(f_scaled / ||f_scaled||)^T J_scaled delta`
    /// * `accepted_updates` - Number of updates accepted before the search began
    /// * `current_residuals` - Residual vector at the unmodified current point
    /// * `gradient` - Gradient measures at that same current point
    /// * `candidate_residuals` - Reusable storage for each trial point
    /// * `candidate_scaled_residuals` - Reusable normalized candidate residuals
    ///
    /// # Returns
    /// Metadata for an accepted candidate whose variables and residuals remain
    /// in the supplied transaction buffers. If no tested step is acceptable,
    /// the method returns `NoConvergence`; it never substitutes an untested or
    /// previously rejected fallback step.
    fn line_search(&self, context: LineSearchContext<'_>) -> Result<AcceptedStep, SolverError> {
        // Give the context fields concise numerical names for the candidate loop
        // while preserving their documented grouping at the call boundary.
        let LineSearchContext {
            jacobian,
            vars,
            candidate_vars,
            delta,
            current_residual_norm,
            residual_norm_directional_derivative,
            accepted_updates,
            current_residuals,
            gradient,
            candidate_residuals,
            candidate_scaled_residuals,
        } = context;

        // A non-negative derivative means the proposed correction is not a
        // descent direction for the least-squares objective. Backtracking cannot
        // repair such a direction, so preserve the current state and explain the
        // numerical cause without evaluating misleading trial points.
        if residual_norm_directional_derivative >= 0.0 {
            let diagnostic = self.build_diagnostic(
                format!(
                    "Line search cannot proceed because the proposed correction is not a descent direction (d||f||/d alpha = {residual_norm_directional_derivative:.2e})"
                ),
                accepted_updates,
                vars,
                current_residuals,
                Some(gradient),
                None,
            );
            return Err(SolverError::NoConvergence(Box::new(diagnostic)));
        }

        // Roots terminate before line search begins, and the helper has already
        // verified that the supplied residual-norm slope is finite.
        if !residual_norm_directional_derivative.is_finite() {
            return Err(self.non_finite_evaluation_error(
                "Line-search residual-norm slope produced NaN or infinity",
                accepted_updates,
                vars,
                current_residuals,
            ));
        }

        // Begin with the caller's configured maximum step. The reusable
        // candidate map is restored from accepted state for every trial so fixed
        // parameters remain present and rejected values never leak forward.
        let mut alpha = self.options.initial_step_size;
        let mut attempted_trials = 0;

        for _ in 0..self.options.line_search_max_trials {
            attempted_trials += 1;
            candidate_vars.clone_from(vars);

            // Build the candidate transactionally. Variable overflow is rejected
            // before expression evaluation, while a finite smaller step remains
            // eligible on the next backtracking trial.
            if self.apply_finite_step(candidate_vars, delta, alpha) {
                // If even the largest remaining multiplier cannot change any
                // variable at floating-point precision, every smaller candidate
                // will be identical. Stop immediately instead of counting a
                // no-op as an accepted update or repeating it to the budget.
                if candidate_vars == vars {
                    let diagnostic = self.build_diagnostic(
                        format!(
                            "Newton correction cannot change the variable state at the tested step size {alpha:.2e}"
                        ),
                        accepted_updates,
                        vars,
                        current_residuals,
                        Some(gradient),
                        None,
                    );
                    return Err(SolverError::NoConvergence(Box::new(diagnostic)));
                }

                // Non-finite residuals are likewise rejected by the finite Armijo
                // comparison. This permits a large exponential step to recover at
                // a smaller alpha without accepting an invalid candidate.
                jacobian.evaluate_functions_checked_into(candidate_vars, candidate_residuals)?;
                self.scale_residuals_into(candidate_residuals, candidate_scaled_residuals);
                let new_error = candidate_scaled_residuals.norm();
                let required_error = current_residual_norm
                    + self.options.armijo_coefficient
                        * alpha
                        * residual_norm_directional_derivative;
                if new_error.is_finite() && new_error <= required_error {
                    return Ok(AcceptedStep {
                        residual_norm: new_error,
                    });
                }
            }

            // Stop only after the current minimum-sized candidate has actually
            // been tested and rejected. This ensures every returned success is
            // backed by an evaluated candidate.
            if alpha <= self.options.line_search_min_step {
                break;
            }
            alpha = (alpha * 0.5).max(self.options.line_search_min_step);
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
            Some(gradient),
            None,
        );
        Err(SolverError::NoConvergence(Box::new(diagnostic)))
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
        match (serial, parallel) {
            // Compare the structured payloads for variants whose data is not a
            // floating-point run diagnostic covered by their focused tests.
            (
                SolverError::Matrix {
                    source: left_source,
                    diagnostic: left_diagnostic,
                },
                SolverError::Matrix {
                    source: right_source,
                    diagnostic: right_diagnostic,
                },
            ) => {
                assert_eq!(left_source, right_source);
                assert_eq!(left_diagnostic.iterations, right_diagnostic.iterations);
                assert_eq!(left_diagnostic.values, right_diagnostic.values);
            }
            (SolverError::InvalidInput(left), SolverError::InvalidInput(right)) => {
                assert_eq!(left, right);
            }
            _ => {}
        }
    }

    /// Extract the structured state from the strict non-root stationarity error
    /// while producing a focused test failure for any other variant.
    fn expect_stationary_non_root(error: SolverError) -> SolverRunDiagnostic {
        // Centralizing this match keeps stationarity regression tests focused on
        // their numerical state rather than repeating variant boilerplate.
        match error {
            SolverError::StationaryNonRoot(diagnostic) => *diagnostic,
            other => panic!("expected StationaryNonRoot, got {other:?}"),
        }
    }

    /// Clone, mutate, and reinstall a solver's shape-adjusted numerical policy
    /// for focused configuration tests.
    fn configure_options(
        solver: NewtonRaphsonSolver,
        configure: impl FnOnce(&mut SolverOptions),
    ) -> NewtonRaphsonSolver {
        // Begin from the solver's actual policy rather than `Default`, because
        // overdetermined construction intentionally adjusts three workload and
        // stabilization values.
        let mut options = solver.options().clone();
        configure(&mut options);
        solver.with_options(options)
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

    /// Preserve every machine-readable matrix field and the complete nonlinear
    /// run state when an actual correction solve fails.
    #[test]
    fn test_linear_solve_error_retains_matrix_cause_and_run_diagnostic() {
        // The derivative is the smallest positive subnormal. Its mathematical
        // correction is larger than `f64::MAX`, forcing a real public solve to
        // return a typed matrix result error after residuals and gradient have
        // already been evaluated at the accepted initial state.
        let x = Exp::var("x");
        let equation = Exp::sub(Exp::mul(Exp::val(f64::from_bits(1)), x), Exp::val(1.0));
        let solver = solver_for_mode(vec![equation], Mode::Serial)
            .with_equation_traces(vec![Some(EquationTrace {
                constraint_id: 41,
                description: "unrepresentable correction fixture".to_string(),
            })])
            .expect("one trace is supplied for one equation");
        let initial = HashMap::from([("x".to_string(), 0.0)]);

        let solver_error = solver
            .solve(initial)
            .expect_err("the required correction lies outside the f64 range");
        let expected_source = MatrixError::NonFiniteResult {
            operation: "solve_least_squares",
        };

        match &solver_error {
            SolverError::Matrix { source, diagnostic } => {
                assert_eq!(source, &expected_source);
                assert_eq!(diagnostic.iterations, 0);
                assert_eq!(diagnostic.values["x"], 0.0);
                assert_eq!(diagnostic.error, 1.0);
                assert_eq!(diagnostic.last_linear_attempt, None);
                assert_eq!(diagnostic.equations.len(), 1);
                assert_eq!(diagnostic.equations[0].residual, -1.0);
                assert_eq!(diagnostic.equations[0].scaled_residual, -1.0);
                assert_eq!(
                    diagnostic.equations[0]
                        .trace
                        .as_ref()
                        .map(|trace| trace.constraint_id),
                    Some(41)
                );
            }
            other => panic!("expected contextual Matrix error, got {other:?}"),
        }

        // Generic error-reporting integrations must observe the identical typed
        // value as the lower-level source without parsing display text.
        let source = std::error::Error::source(&solver_error)
            .expect("a structured matrix failure must remain in the error chain");
        assert_eq!(source.downcast_ref::<MatrixError>(), Some(&expected_source));
    }

    /// Exercise the public nonlinear solver at an extreme finite scale so the
    /// canonical factorization cannot reject a representable pseudoinverse
    /// correction because of an overflowing intermediate value.
    #[test]
    fn test_solver_solves_extreme_rank_deficient_linearization() {
        for mode in test_modes() {
            // Every coefficient and residual is finite at the initial point,
            // and the duplicated 1e308-scale Jacobian columns have the finite
            // minimum-norm root x=y=-5e-309. The coefficient normalization in
            // the SVD must keep that ordinary answer representable even though
            // the unscaled largest singular value exceeds `f64::MAX`.
            let x = Exp::var("x");
            let y = Exp::var("y");
            let huge_linear_form =
                Exp::add(Exp::mul(Exp::val(-1e308), x), Exp::mul(Exp::val(-1e308), y));
            let equations = vec![
                Exp::sub(huge_linear_form.clone(), Exp::val(1.0)),
                Exp::sub(huge_linear_form, Exp::val(1.0)),
            ];
            let solver = solver_for_mode(equations, mode);
            let initial = HashMap::from([("x".to_string(), 0.0), ("y".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("the finite extreme-scale pseudoinverse root must solve");
            for variable in ["x", "y"] {
                let relative_error = (solution.values[variable] / -5e-309 - 1.0).abs();
                assert!(relative_error < 1e-12, "relative error: {relative_error}");
            }
            assert_eq!(solution.iterations, 1);
            assert!(solution.error < 1e-12);
        }
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

            let solution = solver.solve(initial).expect("Failed to solve system");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-8);
            }
            assert!(solution.error < 1e-10);

            let x_sol = solution.values.get("x").copied().unwrap();
            let y_sol = solution.values.get("y").copied().unwrap();
            assert!((x_sol * x_sol + y_sol * y_sol - 1.0).abs() < 1e-10);
            assert!((x_sol * y_sol - 0.25).abs() < 1e-10);
        }
    }

    /// Exercise one solver across multiple runs so its cached symbolic
    /// derivatives are shared while each run retains independent numeric state.
    #[test]
    fn test_reused_solver_handles_distinct_initial_guesses() {
        for mode in test_modes() {
            // The two initial guesses converge to opposite roots of the same
            // nonlinear equation. Reusing the solver proves that caching the
            // derivative does not retain values from the first numeric run.
            let x = Exp::var("x");
            let equation = Exp::sub(Exp::power(x, 2.0), Exp::val(4.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_residual_tolerance(1e-12)
                .with_max_iterations(20);

            // Both calls borrow the same construction-time symbolic Jacobian
            // but allocate their own residual and derivative matrices.
            let positive_solution = solver
                .solve(HashMap::from([("x".to_string(), 3.0)]))
                .expect("positive initial guess should converge");
            let negative_solution = solver
                .solve(HashMap::from([("x".to_string(), -3.0)]))
                .expect("negative initial guess should converge");

            assert!((positive_solution.values["x"] - 2.0).abs() < 1e-10);
            assert!((negative_solution.values["x"] + 2.0).abs() < 1e-10);
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
                .with_residual_tolerance(1e-8)
                .with_max_iterations(50)
                .with_regularization(1e-8);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.5);
            initial.insert("y".to_string(), 0.25);

            let solution = solver
                .solve(initial)
                .expect("Failed to solve transcendental system");

            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-8);
            }
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
            (
                "residual_tolerance",
                Box::new(|solver| solver.with_residual_tolerance(0.0)),
            ),
            (
                "residual_tolerance",
                Box::new(|solver| solver.with_residual_tolerance(f64::NAN)),
            ),
            (
                "stationarity_tolerance",
                Box::new(|solver| solver.with_stationarity_tolerance(0.0)),
            ),
            (
                "stationarity_tolerance",
                Box::new(|solver| solver.with_stationarity_tolerance(f64::INFINITY)),
            ),
            (
                "stationarity_tolerance",
                Box::new(|solver| solver.with_stationarity_tolerance(2.0)),
            ),
            (
                "initial_step_size",
                Box::new(|solver| solver.with_initial_step_size(0.0)),
            ),
            (
                "initial_step_size",
                Box::new(|solver| solver.with_initial_step_size(1.1)),
            ),
            (
                "initial_step_size",
                Box::new(|solver| solver.with_initial_step_size(f64::INFINITY)),
            ),
            (
                "regularization",
                Box::new(|solver| solver.with_regularization(-1.0)),
            ),
            (
                "regularization",
                Box::new(|solver| solver.with_regularization(f64::INFINITY)),
            ),
            (
                "armijo_coefficient",
                Box::new(|solver| {
                    configure_options(solver, |options| options.armijo_coefficient = 1.0)
                }),
            ),
            (
                "line_search_max_trials",
                Box::new(|solver| {
                    configure_options(solver, |options| options.line_search_max_trials = 0)
                }),
            ),
            (
                "line_search_min_step",
                Box::new(|solver| {
                    configure_options(solver, |options| options.line_search_min_step = 0.0)
                }),
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
                .expect_err("invalid active configuration must be rejected");
            match error {
                SolverError::InvalidInput(message) => {
                    assert!(message.contains(expected_parameter), "{message}");
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    /// Reject malformed physical scaling at the builder boundary where names
    /// and equation cardinality are still available for precise diagnostics.
    #[test]
    fn test_solver_rejects_invalid_scales() {
        let x = Exp::var("x");
        let equation = Exp::sub(x.clone(), Exp::val(1.0));

        let solver = NewtonRaphsonSolver::new(
            Compiler::compile(std::slice::from_ref(&equation)).expect("equation should compile"),
        );
        let error = solver
            .with_equation_scales(Vec::new())
            .err()
            .expect("one equation requires one scale");
        assert!(matches!(error, SolverError::InvalidInput(message) if message.contains("length")));

        let solver = NewtonRaphsonSolver::new(
            Compiler::compile(std::slice::from_ref(&equation)).expect("equation should compile"),
        );
        let error = solver
            .with_equation_scales(vec![0.0])
            .err()
            .expect("zero cannot be a characteristic scale");
        assert!(
            matches!(error, SolverError::InvalidInput(message) if message.contains("greater than zero"))
        );

        let solver = NewtonRaphsonSolver::new(
            Compiler::compile(std::slice::from_ref(&equation)).expect("equation should compile"),
        );
        let error = solver
            .with_variable_scales(HashMap::from([("unknown".to_string(), 1.0)]))
            .err()
            .expect("unknown variable scale names must be rejected");
        assert!(matches!(error, SolverError::InvalidInput(message) if message.contains("Unknown")));

        let solver = NewtonRaphsonSolver::new(
            Compiler::compile(&[equation]).expect("equation should compile"),
        );
        let error = solver
            .with_variable_scales(HashMap::from([("x".to_string(), f64::NAN)]))
            .err()
            .expect("non-finite variable scales must be rejected");
        assert!(
            matches!(error, SolverError::InvalidInput(message) if message.contains("greater than zero"))
        );

        // A compiled variable can be a fixed parameter rather than a Jacobian
        // column. Scaling it would have no mathematical effect and is therefore
        // rejected instead of silently stored.
        let parameter = Exp::var("a");
        let parameterized = Exp::sub(x, parameter);
        let solver = NewtonRaphsonSolver::new_with_variables(
            Compiler::compile(&[parameterized]).expect("equation should compile"),
            &["x"],
        )
        .expect("x is a solve variable");
        let error = solver
            .with_variable_scales(HashMap::from([("a".to_string(), 1.0)]))
            .err()
            .expect("fixed parameter scales must be rejected");
        assert!(
            matches!(error, SolverError::InvalidInput(message) if message.contains("fixed parameter"))
        );
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

    /// Prevent a finite residual from hiding overflow in the variable state
    /// produced by either solver update path.
    #[test]
    fn test_solver_never_commits_non_finite_variable_updates() {
        for mode in test_modes() {
            // At x=1e308 the reciprocal expression and its derivative are both
            // finite, but the undamped Newton correction is another 1e308. The
            // resulting x would overflow to infinity while the reciprocal of
            // that invalid value evaluates to the deceptively finite value zero.
            let x = Exp::var("x");
            let scaled_x = Exp::mul(Exp::val(1e-308), x);
            let equation = Exp::div(Exp::val(1.0), scaled_x);
            let initial = HashMap::from([("x".to_string(), 1e308)]);

            // Candidate construction rejects the overflowing full step before
            // expression evaluation. Limiting the search to that one trial
            // makes preservation of the initial accepted state deterministic.
            let solver = configure_options(
                solver_for_mode(vec![equation], mode).with_regularization(0.0),
                |options| options.line_search_max_trials = 1,
            );
            let error = solver
                .solve(initial)
                .expect_err("an overflowing line-search candidate must be rejected");
            match error {
                SolverError::NoConvergence(diagnostic) => {
                    assert!(diagnostic.message.contains("Line search failed"));
                    assert_eq!(diagnostic.iterations, 0);
                    assert_eq!(diagnostic.values["x"], 1e308);
                    assert!(diagnostic.values.values().all(|value| value.is_finite()));
                }
                other => panic!("expected NoConvergence, got {other:?}"),
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

    /// Preserve exact symbolic-derivative behavior at a removable boundary
    /// singularity and demonstrate the caller-controlled algebraic rewrite.
    #[test]
    fn test_solver_requires_boundary_safe_expression_form() {
        for mode in test_modes() {
            let x = Exp::var("x");

            // Differentiating x*sqrt(x) with the product rule retains
            // x*(0.5*x^-0.5). At x=0 that term evaluates as 0*infinity, so the
            // exact symbolic Jacobian is NaN even though the residual is -1.
            let boundary_singular = Exp::sub(
                Exp::add(Exp::mul(x.clone(), Exp::power(x.clone(), 0.5)), x.clone()),
                Exp::val(1.0),
            );
            let solver = solver_for_mode(vec![boundary_singular], mode.clone());
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial.clone())
                .expect_err("the written product has a non-finite derivative at zero");
            match error {
                SolverError::NonFiniteEvaluation(diagnostic) => {
                    assert!(diagnostic.message.contains("Jacobian evaluation"));
                    assert_eq!(diagnostic.error, 1.0);
                }
                other => panic!("expected NonFiniteEvaluation, got {other:?}"),
            }

            // x^(3/2) is algebraically identical for x>=0, but its symbolic
            // derivative is finite at zero. This explicit model rewrite lets
            // Newton iteration leave the domain boundary and certify the root.
            let boundary_safe = Exp::sub(Exp::add(Exp::power(x.clone(), 1.5), x), Exp::val(1.0));
            let solution = solver_for_mode(vec![boundary_safe], mode)
                .solve(initial)
                .expect("the rewritten expression has a finite boundary derivative");
            assert!(solution.error < 1e-10);
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

        let diagnostic =
            solver.build_diagnostic("test failure".to_string(), 0, &vars, &residuals, None, None);

        assert_eq!(diagnostic.equations.len(), 2);
        assert_eq!(
            diagnostic.equations[0],
            EquationDiagnostic {
                equation_index: 0,
                residual: -1.0,
                scaled_residual: -1.0,
                trace: Some(trace),
            }
        );
        assert_eq!(diagnostic.equations[1].equation_index, 1);
        assert_eq!(diagnostic.equations[1].residual, -2.0);
        assert_eq!(diagnostic.equations[1].scaled_residual, -2.0);
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
                .solve(initial)
                .expect("solver should converge when fixed variables are provided");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
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
                .with_residual_tolerance(1e-10);
            let initial = HashMap::from([("x".to_string(), -20.0)]);

            let solution = solver
                .solve(initial)
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

    /// Verify that the public line-search trial budget controls candidate
    /// exhaustion rather than serving as documentation-only metadata.
    #[test]
    fn test_line_search_honors_configured_trial_budget() {
        for mode in test_modes() {
            // The same exponential problem used by the deep-backtracking test
            // cannot accept its full Newton step. Restricting the search to one
            // candidate must therefore preserve the initial point and fail.
            let x = Exp::var("x");
            let equation = Exp::sub(Exp::exp(x), Exp::val(1.0));
            let solver = configure_options(solver_for_mode(vec![equation], mode), |options| {
                options.line_search_max_trials = 1
            });
            let initial = HashMap::from([("x".to_string(), -20.0)]);

            let error = solver
                .solve(initial)
                .expect_err("one rejected candidate must exhaust the configured search");

            match error {
                SolverError::NoConvergence(diagnostic) => {
                    assert_eq!(diagnostic.iterations, 0);
                    assert!(diagnostic.message.contains("after 1 rejected trial steps"));
                    assert_eq!(diagnostic.values["x"], -20.0);
                    assert!(diagnostic.gradient_norm.is_some());
                    assert_eq!(diagnostic.relative_gradient_norm, Some(1.0));
                    let attempt = diagnostic
                        .last_linear_attempt
                        .as_ref()
                        .expect("the rejected candidate still followed a successful SVD attempt");
                    assert_eq!(attempt.effective.rank, 1);
                    assert_eq!(attempt.regularization, None);
                }
                other => panic!("expected NoConvergence, got {other:?}"),
            }
        }
    }

    /// Ensure Armijo line search normalizes the residual before forming its
    /// directional derivative.
    #[test]
    fn test_line_search_avoids_raw_directional_derivative_overflow() {
        for mode in test_modes() {
            // The raw objective derivative at x=0 is -1e400, but the derivative
            // of the residual norm is the representable value -1e200. The full
            // Newton step reaches the exact root and must not fail merely because
            // the unused squared-residual objective has an unrepresentable slope.
            let x = Exp::var("x");
            let equation = Exp::sub(Exp::mul(Exp::val(1e200), x), Exp::val(1e200));
            let solver = solver_for_mode(vec![equation], mode);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("scale-safe residual-norm slope should permit the Newton step");

            assert_eq!(solution.values["x"], 1.0);
            assert_eq!(solution.error, 0.0);
        }
    }

    /// Verify that exhaustion preserves the current state and reports failure
    /// instead of applying an untested minimum-step fallback.
    #[test]
    fn test_line_search_reports_constant_residual_as_stationary_non_root() {
        for mode in test_modes() {
            // Multiplying x by zero registers it as a solve variable while the
            // residual remains exactly one for every possible candidate.
            let x = Exp::var("x");
            let equation = Exp::add(Exp::mul(x, Exp::val(0.0)), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode).with_max_iterations(2);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("a constant residual is stationary but not a root");
            let diagnostic = expect_stationary_non_root(error);

            // Stationarity is visible before line search attempts an unusable
            // zero correction, so the unchanged initial state has zero updates.
            assert_eq!(diagnostic.iterations, 0);
            assert_eq!(diagnostic.values["x"], 0.0);
            assert_eq!(diagnostic.error, 1.0);
            assert_eq!(diagnostic.gradient_norm, Some(0.0));
            assert_eq!(diagnostic.relative_gradient_norm, Some(0.0));
        }
    }

    /// Verify that the configured initial trial cap applies to the first update.
    #[test]
    fn test_solver_honors_initial_step_size() {
        for mode in test_modes() {
            // The undamped Newton correction from x=0 is exactly one. With a
            // single allowed update and a 0.5 cap, the final diagnostic must
            // contain x=0.5.
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_initial_step_size(0.5)
                .with_max_iterations(1);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("one half-step cannot yet reach the root");
            match error {
                SolverError::NoConvergence(diagnostic) => {
                    assert!((diagnostic.values["x"] - 0.5).abs() < 1e-12);
                    assert!((diagnostic.error - 0.5).abs() < 1e-12);
                    assert_eq!(diagnostic.gradient_norm, Some(0.5));
                    assert_eq!(diagnostic.relative_gradient_norm, Some(1.0));
                }
                other => panic!("expected NoConvergence, got {other:?}"),
            }
        }
    }

    /// Ensure that an intentionally small trial cap does not become a false
    /// small-step failure before the residual tolerance is satisfied.
    #[test]
    fn test_small_initial_step_reaches_residual_tolerance() {
        for mode in test_modes() {
            // A 0.1 cap requires roughly 220 geometric updates to reduce the
            // residual below 1e-10. The applied correction becomes small before
            // then, but that is explicit policy rather than stagnation.
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_initial_step_size(0.1)
                .with_residual_tolerance(1e-10)
                .with_max_iterations(300);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("capped linear updates should eventually reach the root");

            assert!(solution.error < 1e-10);
        }
    }

    /// Confirm that a tiny Newton update reports diagnostics from the returned
    /// point rather than retaining the residual from before the update.
    #[test]
    fn test_tiny_update_recomputes_returned_residual() {
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
            assert_eq!(solution.iterations, 1);
            assert_eq!(solution.error, returned_residual.abs());
            assert_eq!(solution.error, 0.0);
            assert_eq!(solution.convergence_history.last(), Some(&0.0));
        }
    }

    /// Distinguish a stationary non-root error from a successful constraint root
    /// in an inconsistent overdetermined system.
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

            let error = solver
                .solve(initial)
                .expect_err("stationary least-squares state is not a constraint root");
            let diagnostic = expect_stationary_non_root(error);

            assert_eq!(diagnostic.iterations, 0);
            assert!((diagnostic.error - 2.0_f64.sqrt()).abs() < 1e-12);
            assert!(diagnostic.gradient_norm.is_some_and(|norm| norm < 1e-10));
            assert!(
                diagnostic
                    .relative_gradient_norm
                    .is_some_and(|norm| norm < 1e-10)
            );
        }
    }

    /// Verify strict stationary-non-root semantics do not depend on the number
    /// of equations relative to solve variables.
    #[test]
    fn test_stationary_non_root_error_is_shape_independent() {
        for mode in test_modes() {
            // Square case: two incompatible equations constrain x while y is a
            // registered but unused solve variable. Their gradient cancels at
            // the initial point even though neither residual is zero.
            let square_x = Exp::var("square_x");
            let square_y = Exp::var("square_y");
            let square_zero_y = Exp::mul(square_y, Exp::val(0.0));
            let square_equations = vec![
                Exp::add(Exp::sub(square_x.clone(), Exp::val(1.0)), square_zero_y),
                Exp::add(square_x, Exp::val(1.0)),
            ];
            let square_initial =
                HashMap::from([("square_x".to_string(), 0.0), ("square_y".to_string(), 0.0)]);

            // Underdetermined case: the positive residual x^2 + 1 has a zero
            // derivative at x=0, and multiplication by zero registers a second
            // solve variable without changing the equation.
            let wide_x = Exp::var("wide_x");
            let wide_y = Exp::var("wide_y");
            let wide_equations = vec![Exp::add(
                Exp::add(Exp::power(wide_x, 2.0), Exp::val(1.0)),
                Exp::mul(wide_y, Exp::val(0.0)),
            )];
            let wide_initial =
                HashMap::from([("wide_x".to_string(), 0.0), ("wide_y".to_string(), 0.0)]);

            for (shape, equations, initial) in [
                ("square", square_equations, square_initial),
                ("underdetermined", wide_equations, wide_initial),
            ] {
                let error = solver_for_mode(equations, mode.clone())
                    .solve(initial)
                    .expect_err("a stationary non-root must never be successful");
                let diagnostic = expect_stationary_non_root(error);

                assert_eq!(diagnostic.iterations, 0, "unexpected {shape} update");
                assert!(diagnostic.error > 0.0, "missing {shape} residual");
                assert_eq!(diagnostic.gradient_norm, Some(0.0));
            }
        }
    }

    /// Ensure column-scaled stationarity remains observable when both raw norms
    /// and residual-gradient products exceed the finite `f64` range.
    #[test]
    fn test_scaled_stationary_system_avoids_raw_gradient_overflow() {
        for mode in test_modes() {
            // At x=0 the residual and Jacobian norms are both infinite in f64,
            // while their residual-gradient contributions are -1e616 and
            // +1e616. Max-scaling the entries first exposes their exact finite
            // cancellation instead of falling back from an undefined norm.
            let x = Exp::var("x");
            let scaled_x = Exp::mul(Exp::val(1.3e308), x);
            let equations = vec![
                Exp::sub(scaled_x.clone(), Exp::val(1.3e308)),
                Exp::add(scaled_x, Exp::val(1.3e308)),
            ];
            let solver = solver_for_mode(equations, mode);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("normalized stationary non-root should survive raw overflow");
            let diagnostic = expect_stationary_non_root(error);

            assert_eq!(diagnostic.gradient_norm, Some(0.0));
            assert_eq!(diagnostic.relative_gradient_norm, Some(0.0));
            assert!(diagnostic.error.is_infinite());
        }
    }

    /// Ensure one high-scale variable cannot hide a descent direction belonging
    /// to another variable before an exactly solvable system's first update.
    #[test]
    fn test_scaled_consistent_overdetermined_system_does_not_false_converge() {
        for mode in test_modes() {
            // At the origin the x column is exactly anti-parallel to the
            // residual, but two unrelated y equations dominate the Jacobian's
            // global Frobenius norm by nine orders of magnitude. A global
            // normalization falls below the configured 1e-8 threshold even
            // though the x=1 root is one ordinary Newton correction away.
            let x = Exp::var("x");
            let y = Exp::var("y");
            let x_residual = Exp::sub(x, Exp::val(1.0));
            let y_residual = Exp::mul(Exp::val(1e9), y);
            let solver = solver_for_mode(vec![x_residual, y_residual.clone(), y_residual], mode)
                .with_initial_step_size(1.0)
                .with_stationarity_tolerance(1e-8)
                .with_regularization(0.0);
            let initial = HashMap::from([("x".to_string(), 0.0), ("y".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("a scaled consistent linear system should reach its root");

            assert_eq!(solution.iterations, 1);
            assert!((solution.values["x"] - 1.0).abs() < 1e-12);
            assert_eq!(solution.values["y"], 0.0);
            assert!(solution.error < 1e-12);
        }
    }

    /// Classify a non-root zero-Jacobian point as stationary without claiming it
    /// is a successful least-squares minimum.
    #[test]
    fn test_zero_jacobian_nonminimum_is_stationary_non_root() {
        for mode in test_modes() {
            // Both equations have exact roots at +/-1. At x=0 their residual is
            // non-zero and the squared-residual objective has a local maximum,
            // even though the raw first derivative is zero.
            let x = Exp::var("x");
            let residual = Exp::sub(Exp::power(x, 2.0), Exp::val(1.0));
            let solver =
                solver_for_mode(vec![residual.clone(), residual], mode).with_max_iterations(1);
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("a zero-Jacobian local maximum must not be successful");

            let diagnostic = expect_stationary_non_root(error);
            assert_eq!(diagnostic.iterations, 0);
            assert!((diagnostic.error - 2.0_f64.sqrt()).abs() < 1e-12);
            assert_eq!(diagnostic.values["x"], 0.0);
            assert_eq!(diagnostic.gradient_norm, Some(0.0));
            assert_eq!(diagnostic.relative_gradient_norm, Some(0.0));
        }
    }

    /// Ensure Armijo reaches a non-zero residual floor without confusing small
    /// objective changes with failure.
    #[test]
    fn test_inconsistent_overdetermined_system_reaches_stationarity_from_away() {
        let mut serial_diagnostic: Option<SolverRunDiagnostic> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);

            // The least-squares objective for these incompatible equations has
            // its unique stationary point at x=0 and residual norm sqrt(2). The
            // A former absolute residual-change rule stopped before `J^T f`
            // met tolerance even though valid descent remained.
            let x = Exp::var("x");
            let equations = vec![
                Exp::sub(x.clone(), Exp::val(1.0)),
                Exp::add(x, Exp::val(1.0)),
            ];
            let solver = solver_for_mode(equations, mode);
            let initial = HashMap::from([("x".to_string(), 0.1)]);

            let error = solver
                .solve(initial)
                .expect_err("Armijo should identify a stationary non-root");
            let diagnostic = expect_stationary_non_root(error);

            if is_serial {
                serial_diagnostic = Some(diagnostic.clone());
            } else {
                let serial = serial_diagnostic
                    .as_ref()
                    .expect("serial diagnostic missing");
                assert!((serial.values["x"] - diagnostic.values["x"]).abs() < 1e-10);
                assert!((serial.error - diagnostic.error).abs() < 1e-10);
            }
            assert!(diagnostic.values["x"].abs() < 1e-8);
            assert!((diagnostic.error - 2.0_f64.sqrt()).abs() < 1e-10);
            assert!(diagnostic.gradient_norm.is_some_and(|norm| norm < 1e-8));
            assert!(
                diagnostic
                    .relative_gradient_norm
                    .is_some_and(|norm| norm < 1e-8)
            );
            let linear = diagnostic
                .last_linear_attempt
                .as_ref()
                .expect("stationarity reached after an SVD correction");
            assert_eq!(linear.regularization, None);
        }
    }

    /// Confirm that line search uses the local least-squares slope rather than
    /// demanding an impossible fixed reduction near a non-zero residual floor.
    #[test]
    fn test_line_search_reaches_inconsistent_least_squares_stationarity() {
        let mut serial_diagnostic: Option<SolverRunDiagnostic> = None;
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

            let error = solver
                .solve(initial)
                .expect_err("Armijo line search should reach a stationary non-root");
            let diagnostic = expect_stationary_non_root(error);

            if is_serial {
                serial_diagnostic = Some(diagnostic.clone());
            } else {
                let serial = serial_diagnostic
                    .as_ref()
                    .expect("serial diagnostic missing");
                assert!((serial.values["x"] - diagnostic.values["x"]).abs() < 1e-10);
                assert!((serial.error - diagnostic.error).abs() < 1e-10);
            }
            assert!(diagnostic.values["x"].abs() < 1e-8);
            assert!((diagnostic.error - 2.0_f64.sqrt()).abs() < 1e-10);
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

            assert_eq!(solution.iterations, 0);
            assert_eq!(solution.convergence_history, vec![0.0]);
            assert_eq!(solution.last_linear_attempt, None);
        }
    }

    /// Enforce the documented inclusive residual threshold both before and
    /// after a Newton correction.
    #[test]
    fn test_residual_tolerance_boundary_is_successful() {
        for mode in test_modes() {
            // The initial residual norm is exactly one. Equality with the
            // configured tolerance must certify this point without a linear
            // solve or an accepted update.
            let x = Exp::var("x");
            let solver =
                solver_for_mode(vec![x.clone()], mode.clone()).with_residual_tolerance(1.0);
            let initial = HashMap::from([("x".to_string(), 1.0)]);
            let solution = solver
                .solve(initial)
                .expect("a residual equal to tolerance must be successful");

            assert_eq!(solution.error, 1.0);
            assert_eq!(solution.iterations, 0);
            assert_eq!(solution.last_linear_attempt, None);

            // Starting at residual two with a half-sized Newton step lands
            // exactly on the same boundary. The next accepted-state check must
            // use the identical inclusive contract.
            let solver = solver_for_mode(vec![x], mode)
                .with_residual_tolerance(1.0)
                .with_initial_step_size(0.5)
                .with_max_iterations(1);
            let initial = HashMap::from([("x".to_string(), 2.0)]);
            let solution = solver
                .solve(initial)
                .expect("an accepted update on the tolerance boundary must succeed");

            assert_eq!(solution.values["x"], 1.0);
            assert_eq!(solution.error, 1.0);
            assert_eq!(solution.iterations, 1);
        }
    }

    /// Ensure an intentionally unbounded-looking iteration budget does not
    /// become an eager allocation request before convergence is evaluated.
    #[test]
    fn test_large_iteration_budget_does_not_preallocate_history() {
        for mode in test_modes() {
            // The initial point is an exact root, so only one history entry is
            // required regardless of the permitted maximum number of updates.
            let x = Exp::var("x");
            let equation = Exp::sub(x, Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode).with_max_iterations(usize::MAX);
            let initial = HashMap::from([("x".to_string(), 1.0)]);

            let solution = solver
                .solve(initial)
                .expect("an exact root must not reserve the full update budget");

            assert_eq!(solution.iterations, 0);
            assert_eq!(solution.convergence_history, vec![0.0]);
        }
    }

    #[test]
    fn test_least_squares_svd_handles_ill_conditioned_overdetermined_system_without_regularization()
    {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);
            // This system is overdetermined (3 equations, 2 unknowns) with an ill-conditioned
            // Jacobian whose columns are nearly linearly dependent. Normal equations (J^T J) can
            // lose the tiny distinguishing term and become singular in floating point. The
            // canonical SVD should still retain and solve both singular directions.
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
                .with_initial_step_size(1.0)
                .with_max_iterations(10)
                .with_residual_tolerance(1e-10);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);
            initial.insert("y".to_string(), 0.0);

            let solution = solver
                .solve(initial)
                .expect("expected solver to converge with SVD least squares");

            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-8);
            }
            // The matrix condition is roughly 2/eps, so a few ulps in the
            // factorization can move the individual coordinates by more than
            // the residual tolerance. Require accurate variables at the scale
            // justified by that conditioning and retain the stricter root test
            // through the solver's residual assertion.
            assert!((solution.values.get("x").copied().unwrap() - 1.0).abs() < 1e-7);
            assert!((solution.values.get("y").copied().unwrap() - 1.0).abs() < 1e-7);
            assert!(solution.error < 1e-10);
        }
    }

    /// Retain an independent direction whose coefficient is small because of
    /// caller units rather than floating-point rank loss.
    #[test]
    fn test_solver_does_not_truncate_solvable_small_coefficient_direction() {
        for mode in test_modes() {
            // The exact root is x=1, y=1e13. The former fixed 1e-12 relative
            // cutoff dropped the y direction, accepted 100 zero corrections,
            // and reported NoConvergence with y still at zero.
            let x = Exp::var("x");
            let y = Exp::var("y");
            let equations = vec![
                Exp::sub(x, Exp::val(1.0)),
                Exp::sub(Exp::mul(Exp::val(1e-13), y), Exp::val(1.0)),
            ];
            let solver = solver_for_mode(equations, mode);
            let initial = HashMap::from([("x".to_string(), 0.0), ("y".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("a direction above factorization roundoff must be retained");

            assert_eq!(solution.iterations, 1);
            assert_eq!(solution.values["x"], 1.0);
            assert!((solution.values["y"] - 1e13).abs() < 1e-3);
            assert!(solution.error <= solver.options().residual_tolerance);
            assert_eq!(
                solution
                    .last_linear_attempt
                    .as_ref()
                    .expect("one correction must retain factorization diagnostics")
                    .effective
                    .rank,
                2
            );
        }
    }

    /// Normalize variable coordinates so condition diagnostics and optional
    /// ridge penalties do not depend on the caller's chosen units.
    #[test]
    fn test_variable_scales_normalize_jacobian_columns() {
        for mode in test_modes() {
            let x = Exp::var("x");
            let y = Exp::var("y");
            let equations = vec![
                Exp::sub(x, Exp::val(1.0)),
                Exp::sub(Exp::mul(Exp::val(1e-13), y), Exp::val(1.0)),
            ];
            let solver = solver_for_mode(equations, mode)
                .with_variable_scales(HashMap::from([("y".to_string(), 1e13)]))
                .expect("y has a positive finite characteristic scale");
            let initial = HashMap::from([("x".to_string(), 0.0), ("y".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("normalized variable coordinates should solve");
            let linear = solution
                .last_linear_attempt
                .expect("one correction must report its scaled factorization");

            assert_eq!(solution.values["x"], 1.0);
            assert!((solution.values["y"] - 1e13).abs() < 1e-3);
            assert_eq!(linear.effective.rank, 2);
            assert!((linear.effective.cond_est - 1.0).abs() < 1e-12);
            assert_eq!(solver.variable_scale("x"), Some(1.0));
            assert_eq!(solver.variable_scale("y"), Some(1e13));
        }
    }

    /// Preserve finite caller-unit corrections across extreme scaling and
    /// backtracking factors.
    #[test]
    fn test_scaled_update_preserves_representable_products() {
        // Applying the step size first underflows 1e-320 * 1e-12, while
        // applying the large variable scale first preserves a result near 1e-24.
        let recovered_from_underflow =
            NewtonRaphsonSolver::scaled_update_component(1e-320, 1e-12, 1e308);
        assert!(recovered_from_underflow.is_finite());
        assert!(recovered_from_underflow > 0.0);
        assert!((recovered_from_underflow.log10() + 24.0).abs() < 0.01);

        // Scaling first overflows 1e308 * 10, while applying the step size
        // first preserves the finite result 1e307.
        let recovered_from_overflow =
            NewtonRaphsonSolver::scaled_update_component(1e308, 1e-2, 10.0);
        assert!(recovered_from_overflow.is_finite());
        assert!((recovered_from_overflow / 1e307 - 1.0).abs() < 1e-12);
    }

    /// Preserve a non-zero normalized derivative when coefficient and unit
    /// scales span the complete finite exponent range.
    #[test]
    fn test_jacobian_scaling_does_not_underflow_representable_derivative() {
        for mode in test_modes() {
            // The normalized derivative is
            // 1e308 * 1e-308 / 1e308 = 1e-308. Forming the scale ratio first
            // underflows it to zero and falsely classifies x=0 as stationary;
            // the normalized Newton correction is 1e308 and restores the
            // ordinary caller-unit update x=1.
            let x = Exp::var("x");
            let equation = Exp::sub(Exp::mul(Exp::val(1e308), x), Exp::val(1e308));
            let solver = solver_for_mode(vec![equation], mode)
                .with_equation_scales(vec![1e308])
                .expect("the equation has one positive characteristic scale")
                .with_variable_scales(HashMap::from([("x".to_string(), 1e-308)]))
                .expect("the solve variable has one positive characteristic scale");
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let solution = solver
                .solve(initial)
                .expect("the representable normalized derivative must reach the root");

            assert_eq!(solution.iterations, 1);
            assert!((solution.values["x"] - 1.0).abs() < 1e-12);
            assert!(solution.error < 1e-12);
        }
    }

    /// Use equation characteristic scales consistently for least-squares
    /// weighting, convergence, stationarity, Armijo, and diagnostics.
    #[test]
    fn test_equation_scales_define_dimensionless_residual_objective() {
        for mode in test_modes() {
            // Raw least squares would let the second equation's factor 1000
            // dominate and place x near two. Dividing it by its characteristic
            // scale gives both physical constraints equal dimensionless weight,
            // whose stationary point is x=1.5.
            let x = Exp::var("x");
            let equations = vec![
                Exp::sub(x.clone(), Exp::val(1.0)),
                Exp::mul(Exp::val(1000.0), Exp::sub(x, Exp::val(2.0))),
            ];
            let solver = solver_for_mode(equations, mode)
                .with_equation_scales(vec![1.0, 1000.0])
                .expect("one positive scale is supplied per equation");
            let initial = HashMap::from([("x".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("the equally weighted equations are inconsistent");
            let diagnostic = expect_stationary_non_root(error);

            assert!((diagnostic.values["x"] - 1.5).abs() < 1e-12);
            assert!((diagnostic.error - 0.5_f64.sqrt()).abs() < 1e-12);
            assert!((diagnostic.equations[0].residual - 0.5).abs() < 1e-12);
            assert!((diagnostic.equations[0].scaled_residual - 0.5).abs() < 1e-12);
            assert!((diagnostic.equations[1].residual + 500.0).abs() < 1e-9);
            assert!((diagnostic.equations[1].scaled_residual + 0.5).abs() < 1e-12);
            assert_eq!(solver.equation_scales(), &[1.0, 1000.0]);
        }
    }

    /// Ensure a nonlinear wide system certifies roots before mutation and still
    /// makes genuine minimum-norm residual corrections away from a root.
    #[test]
    fn test_nonlinear_underdetermined_solver_never_starves_corrections() {
        for mode in test_modes() {
            // The Jacobian row [y, x] rotates as either coordinate changes. The
            // removed point-projection phase chased that row space and could
            // starve residual corrections indefinitely.
            let x = Exp::var("x");
            let y = Exp::var("y");
            let equation = Exp::sub(Exp::mul(x, y), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode)
                .with_residual_tolerance(1e-12)
                .with_regularization(0.0);

            let exact_root = HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 0.5)]);
            let solution = solver
                .solve(exact_root)
                .expect("an exact nonlinear root must remain unchanged");
            assert_eq!(solution.iterations, 0);
            assert_eq!(solution.values["x"], 2.0);
            assert_eq!(solution.values["y"], 0.5);

            // Reuse the same compiled solver away from the root. Every accepted
            // iteration must be a residual-reducing Newton correction.
            let initial = HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 1.0)]);
            let solution = solver
                .solve(initial)
                .expect("a nonlinear wide system must execute Newton corrections");
            assert!(solution.iterations > 0);
            assert!((solution.values["x"] * solution.values["y"] - 1.0).abs() < 1e-12);
        }
    }

    /// Confirm that a wide solve preserves the initial Jacobian-null-space
    /// component by applying the unique minimum-norm Newton correction.
    #[test]
    fn test_underconstrained_solver_preserves_null_space_component() {
        for mode in test_modes() {
            // The initial point has x+y=0 and a large component in the [1,-1]
            // null-space direction. The minimum correction is [0.5,0.5].
            let x = Exp::var("x");
            let y = Exp::var("y");
            let equation = Exp::sub(Exp::add(x, y), Exp::val(1.0));
            let solver = solver_for_mode(vec![equation], mode);
            let initial = HashMap::from([("x".to_string(), 10.0), ("y".to_string(), -10.0)]);

            let solution = solver
                .solve(initial)
                .expect("linear constraint should solve");

            assert!((solution.values["x"] - 10.5).abs() < 1e-10);
            assert!((solution.values["y"] + 9.5).abs() < 1e-10);
        }
    }

    /// Confirm that default settings accept a well-conditioned retained SVD
    /// subspace without changing policy merely because a variable is unused.
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
                .with_residual_tolerance(1e-12)
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
                .with_residual_tolerance(1e-12)
                .with_max_iterations(10);

            let mut initial = HashMap::new();
            initial.insert("x".to_string(), 0.0);
            initial.insert("y".to_string(), 0.0);

            let solution = solver
                .solve(initial)
                .expect("the SVD pseudoinverse should solve without implicit regularization");
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert!((solution.values["x"] - 1.0).abs() < 1e-10);
            assert!(solution.values["y"].abs() < 1e-10);
        }
    }

    /// Confirm that a dependent square system uses the canonical SVD directly,
    /// even when augmented regularization is disabled.
    #[test]
    fn test_rank_deficient_square_system_converges_without_regularization() {
        let mut serial_solution: Option<Solution> = None;
        for mode in test_modes() {
            let is_serial = matches!(mode, Mode::Serial);

            // Both equations describe the same line. Their 2-by-2 Jacobian is
            // exactly rank one, so LU must reject it while the Moore-Penrose
            // correction selects x = y = 1/2 from the zero initial point.
            let x = Exp::var("x");
            let y = Exp::var("y");
            let first_equation = Exp::sub(Exp::add(x, y), Exp::val(1.0));
            let second_equation = Exp::mul(Exp::val(2.0), first_equation.clone());
            let solver = solver_for_mode(vec![first_equation, second_equation], mode)
                .with_regularization(0.0)
                .with_residual_tolerance(1e-12)
                .with_max_iterations(10);

            // A symmetric initial point makes the pseudoinverse's minimum-norm
            // choice deterministic and easy to distinguish from an arbitrary
            // diagonal perturbation of the singular Jacobian.
            let initial = HashMap::from([("x".to_string(), 0.0), ("y".to_string(), 0.0)]);
            let solution = solver
                .solve(initial)
                .expect("the canonical SVD should solve dependent equations");

            // Both execution modes must select the same pseudoinverse solution
            // and terminate by satisfying the actual residual equations.
            if is_serial {
                serial_solution = Some(solution.clone());
            } else {
                let serial = serial_solution.as_ref().expect("serial solution missing");
                assert_solution_close(serial, &solution, 1e-10);
            }
            assert!(solution.error < 1e-12);
            assert!((solution.values["x"] - 0.5).abs() < 1e-10);
            assert!((solution.values["y"] - 0.5).abs() < 1e-10);
            let linear = solution
                .last_linear_attempt
                .expect("the accepted correction must retain SVD diagnostics");
            assert_eq!(linear.effective.rank, 1);
            assert_eq!(linear.regularization, None);
        }
    }

    /// Confirm that an explicitly regularized square solve uses the augmented
    /// ridge system rather than first applying its direct correction.
    #[test]
    fn test_ill_conditioned_square_system_uses_regularized_least_squares() {
        for mode in test_modes() {
            // This upper-triangular Jacobian has unit diagonal pivots but a
            // condition estimate around 1e16. LU previously accepted it and
            // jumped to the exact root, bypassing the configured ridge policy.
            let x = Exp::var("x");
            let y = Exp::var("y");
            let first_equation = Exp::sub(
                Exp::add(x, Exp::mul(Exp::val(1e8), y.clone())),
                Exp::val(1e8 + 1.0),
            );
            let second_equation = Exp::sub(y, Exp::val(1.0));
            let solver = solver_for_mode(vec![first_equation, second_equation], mode)
                .with_residual_tolerance(1e-12)
                .with_regularization(1e-8)
                .with_max_iterations(1);
            let initial = HashMap::from([("x".to_string(), 0.0), ("y".to_string(), 0.0)]);

            let error = solver
                .solve(initial)
                .expect_err("one regularized update should not masquerade as an exact root");

            match error {
                SolverError::NoConvergence(diagnostic) => {
                    assert_eq!(diagnostic.iterations, 1);
                    assert!(diagnostic.values["x"].abs() < 0.1);
                    assert!(diagnostic.error > 1e-12);
                    let linear = diagnostic
                        .last_linear_attempt
                        .as_ref()
                        .expect("regularized correction diagnostics must be retained");
                    assert_eq!(linear.regularization, Some(1e-8));
                }
                other => panic!("expected NoConvergence, got {other:?}"),
            }
        }
    }

    /// Verify that explicit regularization performs one predictable augmented
    /// solve with exactly the caller's ridge strength.
    #[test]
    fn test_regularization_uses_exact_configured_strength() {
        let x = Exp::var("x");
        let y = Exp::var("y");
        let equation = Exp::add(x, y);
        let solver = solver_for_mode(vec![equation], Mode::Serial).with_regularization(1e-8);

        // The fixture is deliberately ill-conditioned, but conditioning no
        // longer selects or changes policy: positive regularization means the
        // requested augmented problem is solved exactly once.
        let jacobian = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1e-8], 2, 2)
            .expect("the fixture has a valid square shape");
        let rhs = Matrix::from_vec(vec![1.0, 1.0], 2, 1)
            .expect("the fixture has a valid right-hand-side shape");
        let result = match solver.solve_least_squares_system(&jacobian, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("finite ridge systems must remain solvable"),
        };

        let used_regularization = result
            .diagnostic
            .regularization
            .expect("positive regularization must select an augmented solve");
        assert_eq!(used_regularization, 1e-8);
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
                .with_residual_tolerance(1e-12)
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
                    .with_residual_tolerance(1e-12)
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
                    .with_residual_tolerance(1e-12)
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
                    .with_residual_tolerance(1e-12)
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
