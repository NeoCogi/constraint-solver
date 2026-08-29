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

//! Compilation of name-based symbolic expressions into compact, indexed
//! equation systems suitable for repeated numerical evaluation.

use std::collections::HashMap;
use std::fmt;

use crate::exp::Exp;

/// Error returned when a symbolic expression cannot be compiled into the
/// solver's compact variable-indexed representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// A variable used an empty name, which cannot be addressed unambiguously
    /// by the name-based public solver API.
    EmptyVariableName,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::EmptyVariableName => write!(f, "Variable name cannot be empty"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Dense identifier assigned to a named variable during compilation.
///
/// Keeping this identifier crate-private prevents callers from depending on
/// compilation order while allowing evaluation to use integer-indexed maps.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct VarId(usize);

impl VarId {
    /// Wrap an index from a specific compiled system as its internal variable
    /// identifier.
    pub(crate) const fn new(id: usize) -> Self {
        // `VarId` is deliberately a thin newtype: range validation belongs to
        // the owning `VarTable`, which is the only source of public IDs.
        Self(id)
    }
}

/// Bidirectional variable registry owned by one compiled equation system.
#[derive(Debug, Clone)]
pub(crate) struct VarTable {
    /// Names in stable compilation order; each position is the corresponding
    /// `VarId` payload.
    names: Vec<String>,
    /// Reverse lookup used when translating name-keyed initial guesses.
    name_to_id: HashMap<String, VarId>,
}

impl VarTable {
    fn new() -> Self {
        Self {
            names: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    fn register(&mut self, name: &str) -> Result<VarId, CompileError> {
        if name.is_empty() {
            return Err(CompileError::EmptyVariableName);
        }
        if let Some(id) = self.name_to_id.get(name) {
            return Ok(*id);
        }
        let id = VarId::new(self.names.len());
        self.names.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        Ok(id)
    }

    pub(crate) fn get_id(&self, name: &str) -> Option<VarId> {
        self.name_to_id.get(name).copied()
    }

    pub(crate) fn len(&self) -> usize {
        self.names.len()
    }

    pub(crate) fn all_var_ids(&self) -> Vec<VarId> {
        (0..self.names.len()).map(VarId::new).collect()
    }

    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }
}

/// Internal evaluation failure for a compiled expression whose variable map
/// does not contain one of the IDs embedded by the compiler.
///
/// Solver entry points validate complete initial guesses before evaluation, so
/// this error indicates an internal state mismatch rather than a recoverable
/// public "missing variable" condition. Keeping it crate-private removes the
/// previously unreachable public error variant without making internal
/// evaluators infallible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvaluationError {
    /// Human-readable variable name retained in compiled variable nodes for a
    /// useful invariant-violation message.
    variable_name: String,
}

impl EvaluationError {
    /// Construct an evaluation error for a compiled variable that could not be
    /// found in the supplied internal value map.
    fn missing_variable(variable_name: &str) -> Self {
        // Clone only on the exceptional path; ordinary expression evaluation
        // continues to borrow the name stored in the compiled node.
        Self {
            variable_name: variable_name.to_owned(),
        }
    }
}

impl fmt::Display for EvaluationError {
    /// Describe the violated compiled-system invariant without presenting it as
    /// a user-correctable public lookup error.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compiled evaluation map is missing variable '{}'",
            self.variable_name
        )
    }
}

impl std::error::Error for EvaluationError {}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CompiledExp {
    Val(f64),
    Var(String, VarId),
    Add(Box<CompiledExp>, Box<CompiledExp>),
    Sub(Box<CompiledExp>, Box<CompiledExp>),
    Mul(Box<CompiledExp>, Box<CompiledExp>),
    Div(Box<CompiledExp>, Box<CompiledExp>),
    Power(Box<CompiledExp>, f64),
    Neg(Box<CompiledExp>),
    Sin(Box<CompiledExp>),
    Cos(Box<CompiledExp>),
    Ln(Box<CompiledExp>),
    Exp(Box<CompiledExp>),
}

impl CompiledExp {
    pub(crate) fn evaluate_checked(
        &self,
        vars: &HashMap<VarId, f64>,
    ) -> Result<f64, EvaluationError> {
        match self {
            CompiledExp::Val(v) => Ok(*v),
            CompiledExp::Var(name, id) => vars
                .get(id)
                .copied()
                .ok_or_else(|| EvaluationError::missing_variable(name)),
            CompiledExp::Add(l, r) => Ok(l.evaluate_checked(vars)? + r.evaluate_checked(vars)?),
            CompiledExp::Sub(l, r) => Ok(l.evaluate_checked(vars)? - r.evaluate_checked(vars)?),
            CompiledExp::Mul(l, r) => Ok(l.evaluate_checked(vars)? * r.evaluate_checked(vars)?),
            CompiledExp::Div(l, r) => Ok(l.evaluate_checked(vars)? / r.evaluate_checked(vars)?),
            CompiledExp::Power(base, exp) => Ok(base.evaluate_checked(vars)?.powf(*exp)),
            CompiledExp::Neg(e) => Ok(-e.evaluate_checked(vars)?),
            CompiledExp::Sin(e) => Ok(e.evaluate_checked(vars)?.sin()),
            CompiledExp::Cos(e) => Ok(e.evaluate_checked(vars)?.cos()),
            CompiledExp::Ln(e) => Ok(e.evaluate_checked(vars)?.ln()),
            CompiledExp::Exp(e) => Ok(e.evaluate_checked(vars)?.exp()),
        }
    }

    pub(crate) fn differentiate(&self, var_id: VarId) -> CompiledExp {
        match self {
            CompiledExp::Val(_) => CompiledExp::Val(0.0),
            CompiledExp::Var(_, id) => {
                if *id == var_id {
                    CompiledExp::Val(1.0)
                } else {
                    CompiledExp::Val(0.0)
                }
            }
            CompiledExp::Add(l, r) => CompiledExp::Add(
                Box::new(l.differentiate(var_id)),
                Box::new(r.differentiate(var_id)),
            ),
            CompiledExp::Sub(l, r) => CompiledExp::Sub(
                Box::new(l.differentiate(var_id)),
                Box::new(r.differentiate(var_id)),
            ),
            CompiledExp::Mul(l, r) => {
                let dl = l.differentiate(var_id);
                let dr = r.differentiate(var_id);
                CompiledExp::Add(
                    Box::new(CompiledExp::Mul(Box::new(dl), r.clone())),
                    Box::new(CompiledExp::Mul(l.clone(), Box::new(dr))),
                )
            }
            CompiledExp::Div(l, r) => {
                let dl = l.differentiate(var_id);
                let dr = r.differentiate(var_id);
                CompiledExp::Div(
                    Box::new(CompiledExp::Sub(
                        Box::new(CompiledExp::Mul(Box::new(dl), r.clone())),
                        Box::new(CompiledExp::Mul(l.clone(), Box::new(dr))),
                    )),
                    Box::new(CompiledExp::Power(r.clone(), 2.0)),
                )
            }
            CompiledExp::Power(base, exp) => {
                let db = base.differentiate(var_id);
                CompiledExp::Mul(
                    Box::new(CompiledExp::Mul(
                        Box::new(CompiledExp::Val(*exp)),
                        Box::new(CompiledExp::Power(base.clone(), exp - 1.0)),
                    )),
                    Box::new(db),
                )
            }
            CompiledExp::Neg(e) => CompiledExp::Neg(Box::new(e.differentiate(var_id))),
            CompiledExp::Sin(e) => {
                let de = e.differentiate(var_id);
                CompiledExp::Mul(Box::new(CompiledExp::Cos(e.clone())), Box::new(de))
            }
            CompiledExp::Cos(e) => {
                let de = e.differentiate(var_id);
                CompiledExp::Neg(Box::new(CompiledExp::Mul(
                    Box::new(CompiledExp::Sin(e.clone())),
                    Box::new(de),
                )))
            }
            CompiledExp::Ln(e) => {
                let de = e.differentiate(var_id);
                CompiledExp::Div(Box::new(de), e.clone())
            }
            CompiledExp::Exp(e) => {
                let de = e.differentiate(var_id);
                CompiledExp::Mul(Box::new(CompiledExp::Exp(e.clone())), Box::new(de))
            }
        }
    }

    pub(crate) fn simplify(&self) -> CompiledExp {
        match self {
            CompiledExp::Add(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (CompiledExp::Val(lv), CompiledExp::Val(rv)) => CompiledExp::Val(lv + rv),
                    (CompiledExp::Val(0.0), _) => rs,
                    (_, CompiledExp::Val(0.0)) => ls,
                    _ => CompiledExp::Add(Box::new(ls), Box::new(rs)),
                }
            }
            CompiledExp::Sub(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (CompiledExp::Val(lv), CompiledExp::Val(rv)) => CompiledExp::Val(lv - rv),
                    (_, CompiledExp::Val(0.0)) => ls,
                    _ => CompiledExp::Sub(Box::new(ls), Box::new(rs)),
                }
            }
            CompiledExp::Mul(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (CompiledExp::Val(lv), CompiledExp::Val(rv)) => CompiledExp::Val(lv * rv),
                    (CompiledExp::Val(0.0), _) | (_, CompiledExp::Val(0.0)) => {
                        CompiledExp::Val(0.0)
                    }
                    (CompiledExp::Val(1.0), _) => rs,
                    (_, CompiledExp::Val(1.0)) => ls,
                    _ => CompiledExp::Mul(Box::new(ls), Box::new(rs)),
                }
            }
            CompiledExp::Div(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (CompiledExp::Val(lv), CompiledExp::Val(rv)) if *rv != 0.0 => {
                        CompiledExp::Val(lv / rv)
                    }
                    (CompiledExp::Val(0.0), _) => CompiledExp::Val(0.0),
                    (_, CompiledExp::Val(1.0)) => ls,
                    _ => CompiledExp::Div(Box::new(ls), Box::new(rs)),
                }
            }
            CompiledExp::Power(base, exp) => {
                let bs = base.simplify();
                match &bs {
                    CompiledExp::Val(v) => CompiledExp::Val(v.powf(*exp)),
                    _ if *exp == 0.0 => CompiledExp::Val(1.0),
                    _ if *exp == 1.0 => bs,
                    _ => CompiledExp::Power(Box::new(bs), *exp),
                }
            }
            CompiledExp::Neg(e) => {
                let es = e.simplify();
                match &es {
                    CompiledExp::Val(v) => CompiledExp::Val(-v),
                    _ => CompiledExp::Neg(Box::new(es)),
                }
            }
            _ => self.clone(),
        }
    }
}

/// Immutable compiled representation consumed by `NewtonRaphsonSolver`.
///
/// Expression storage remains private so callers cannot invalidate the
/// compiler's variable IDs, while the introspection methods expose the stable
/// metadata needed to prepare solver inputs.
#[derive(Debug, Clone)]
pub struct CompiledSystem {
    /// Equations in the same order as the source slice passed to `compile`.
    pub(crate) equations: Vec<CompiledExp>,
    /// Variable names and IDs shared by all compiled equations.
    pub(crate) var_table: VarTable,
}

impl CompiledSystem {
    /// Return the number of compiled scalar equations.
    pub fn equation_count(&self) -> usize {
        // Every entry represents one residual row in the eventual solver.
        self.equations.len()
    }

    /// Return the number of distinct named variables referenced by the system.
    pub fn variable_count(&self) -> usize {
        // `VarTable` deduplicates repeated occurrences during compilation.
        self.var_table.len()
    }

    /// Return variable names in the stable order used for Jacobian columns.
    pub fn variable_names(&self) -> &[String] {
        // Borrow the table's storage so introspection performs no allocation and
        // cannot mutate compiler-owned identifiers.
        self.var_table.names()
    }

    /// Report whether the compiled system contains a variable with `name`.
    pub fn contains_variable(&self, name: &str) -> bool {
        // Use the registry's reverse index instead of scanning the ordered name
        // slice; this remains constant-time for large systems.
        self.var_table.get_id(name).is_some()
    }

    /// Report whether the system contains no equations.
    pub fn is_empty(&self) -> bool {
        // Variables originate from equations, so an empty equation list also
        // implies an empty variable table for compiler-created values.
        self.equations.is_empty()
    }
}

/// Stateless entry point that translates symbolic `Exp` trees into a validated
/// `CompiledSystem` with shared dense variable identifiers.
pub struct Compiler;

impl Compiler {
    /// Compile symbolic equations and assign stable dense IDs to their named
    /// variables.
    pub fn compile(equations: &[Exp]) -> Result<CompiledSystem, CompileError> {
        // One shared table ensures repeated names in different equations refer
        // to the same solver variable and Jacobian column.
        let mut var_table = VarTable::new();
        let mut compiled = Vec::with_capacity(equations.len());

        for eq in equations {
            compiled.push(Self::compile_exp(eq, &mut var_table)?);
        }

        Ok(CompiledSystem {
            equations: compiled,
            var_table,
        })
    }

    fn compile_exp(exp: &Exp, var_table: &mut VarTable) -> Result<CompiledExp, CompileError> {
        match exp {
            Exp::Val(v) => Ok(CompiledExp::Val(*v)),
            Exp::Var(name) => {
                let id = var_table.register(name)?;
                Ok(CompiledExp::Var(name.clone(), id))
            }
            Exp::Add(l, r) => Ok(CompiledExp::Add(
                Box::new(Self::compile_exp(l, var_table)?),
                Box::new(Self::compile_exp(r, var_table)?),
            )),
            Exp::Sub(l, r) => Ok(CompiledExp::Sub(
                Box::new(Self::compile_exp(l, var_table)?),
                Box::new(Self::compile_exp(r, var_table)?),
            )),
            Exp::Mul(l, r) => Ok(CompiledExp::Mul(
                Box::new(Self::compile_exp(l, var_table)?),
                Box::new(Self::compile_exp(r, var_table)?),
            )),
            Exp::Div(l, r) => Ok(CompiledExp::Div(
                Box::new(Self::compile_exp(l, var_table)?),
                Box::new(Self::compile_exp(r, var_table)?),
            )),
            Exp::Power(base, exp) => Ok(CompiledExp::Power(
                Box::new(Self::compile_exp(base, var_table)?),
                *exp,
            )),
            Exp::Neg(e) => Ok(CompiledExp::Neg(Box::new(Self::compile_exp(e, var_table)?))),
            Exp::Sin(e) => Ok(CompiledExp::Sin(Box::new(Self::compile_exp(e, var_table)?))),
            Exp::Cos(e) => Ok(CompiledExp::Cos(Box::new(Self::compile_exp(e, var_table)?))),
            Exp::Ln(e) => Ok(CompiledExp::Ln(Box::new(Self::compile_exp(e, var_table)?))),
            Exp::Exp(e) => Ok(CompiledExp::Exp(Box::new(Self::compile_exp(e, var_table)?))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure the public compiled-system API reports deduplicated variables in
    /// the same first-seen order used by internal Jacobian construction.
    #[test]
    fn compiled_system_exposes_stable_metadata() {
        let equations = vec![
            Exp::sub(Exp::var("y"), Exp::var("x")),
            Exp::add(Exp::var("x"), Exp::var("z")),
        ];

        let compiled = Compiler::compile(&equations).expect("valid system should compile");

        assert_eq!(compiled.equation_count(), 2);
        assert_eq!(compiled.variable_count(), 3);
        assert_eq!(compiled.variable_names(), &["y", "x", "z"]);
        assert!(compiled.contains_variable("x"));
        assert!(!compiled.contains_variable("missing"));
        assert!(!compiled.is_empty());
    }

    /// Verify that an empty source system has coherent public metadata rather
    /// than exposing an otherwise opaque zero-sized compiled value.
    #[test]
    fn empty_compiled_system_reports_no_equations_or_variables() {
        let compiled = Compiler::compile(&[]).expect("empty systems are valid to compile");

        assert_eq!(compiled.equation_count(), 0);
        assert_eq!(compiled.variable_count(), 0);
        assert!(compiled.variable_names().is_empty());
        assert!(compiled.is_empty());
    }
}
