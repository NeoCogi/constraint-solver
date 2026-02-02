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

use std::collections::HashMap;
use std::fmt;

use crate::exp::{Exp, MissingVarError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct VarId(usize);

impl VarId {
    pub(crate) const fn new(id: usize) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VarTable {
    names: Vec<String>,
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
    ) -> Result<f64, MissingVarError> {
        match self {
            CompiledExp::Val(v) => Ok(*v),
            CompiledExp::Var(name, id) => vars.get(id).copied().ok_or_else(|| MissingVarError {
                var_name: name.clone(),
            }),
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

#[derive(Debug, Clone)]
pub struct CompiledSystem {
    pub(crate) equations: Vec<CompiledExp>,
    pub(crate) var_table: VarTable,
}

impl CompiledSystem {
}

pub struct Compiler;

impl Compiler {
    pub fn compile(equations: &[Exp]) -> Result<CompiledSystem, CompileError> {
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
