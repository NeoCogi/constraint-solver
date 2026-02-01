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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(usize);

impl VarId {
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingVarError {
    pub var_name: String,
    pub var_id: VarId,
}

impl fmt::Display for MissingVarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Missing variable '{}' (id {})",
            self.var_name, self.var_id
        )
    }
}

impl std::error::Error for MissingVarError {}

#[derive(Debug, Clone, PartialEq)]
pub enum Exp {
    Val(f64),
    Var(String, VarId),
    Add(Box<Exp>, Box<Exp>),
    Sub(Box<Exp>, Box<Exp>),
    Mul(Box<Exp>, Box<Exp>),
    Div(Box<Exp>, Box<Exp>),
    Power(Box<Exp>, f64),
    Neg(Box<Exp>),
    Sin(Box<Exp>),
    Cos(Box<Exp>),
    Ln(Box<Exp>),
    Exp(Box<Exp>),
}

#[allow(clippy::self_named_constructors, clippy::should_implement_trait)]
impl Exp {
    pub fn var(name: String, id: VarId) -> Self {
        Exp::Var(name, id)
    }

    pub fn val(v: f64) -> Self {
        Exp::Val(v)
    }

    pub fn add(lhs: Exp, rhs: Exp) -> Self {
        Exp::Add(Box::new(lhs), Box::new(rhs))
    }

    pub fn sub(lhs: Exp, rhs: Exp) -> Self {
        Exp::Sub(Box::new(lhs), Box::new(rhs))
    }

    pub fn mul(lhs: Exp, rhs: Exp) -> Self {
        Exp::Mul(Box::new(lhs), Box::new(rhs))
    }

    pub fn div(lhs: Exp, rhs: Exp) -> Self {
        Exp::Div(Box::new(lhs), Box::new(rhs))
    }

    pub fn power(base: Exp, exp: f64) -> Self {
        Exp::Power(Box::new(base), exp)
    }

    pub fn neg(exp: Exp) -> Self {
        Exp::Neg(Box::new(exp))
    }

    pub fn sin(exp: Exp) -> Self {
        Exp::Sin(Box::new(exp))
    }

    pub fn cos(exp: Exp) -> Self {
        Exp::Cos(Box::new(exp))
    }

    pub fn ln(exp: Exp) -> Self {
        Exp::Ln(Box::new(exp))
    }

    pub fn exp(exp: Exp) -> Self {
        Exp::Exp(Box::new(exp))
    }

    pub fn evaluate_checked(
        &self,
        vars: &HashMap<VarId, f64>,
    ) -> Result<f64, MissingVarError> {
        match self {
            Exp::Val(v) => Ok(*v),
            Exp::Var(name, id) => vars.get(id).copied().ok_or_else(|| MissingVarError {
                var_name: name.clone(),
                var_id: *id,
            }),
            Exp::Add(l, r) => Ok(l.evaluate_checked(vars)? + r.evaluate_checked(vars)?),
            Exp::Sub(l, r) => Ok(l.evaluate_checked(vars)? - r.evaluate_checked(vars)?),
            Exp::Mul(l, r) => Ok(l.evaluate_checked(vars)? * r.evaluate_checked(vars)?),
            Exp::Div(l, r) => Ok(l.evaluate_checked(vars)? / r.evaluate_checked(vars)?),
            Exp::Power(base, exp) => Ok(base.evaluate_checked(vars)?.powf(*exp)),
            Exp::Neg(e) => Ok(-e.evaluate_checked(vars)?),
            Exp::Sin(e) => Ok(e.evaluate_checked(vars)?.sin()),
            Exp::Cos(e) => Ok(e.evaluate_checked(vars)?.cos()),
            Exp::Ln(e) => Ok(e.evaluate_checked(vars)?.ln()),
            Exp::Exp(e) => Ok(e.evaluate_checked(vars)?.exp()),
        }
    }

    pub fn evaluate(&self, vars: &HashMap<VarId, f64>) -> f64 {
        match self {
            Exp::Val(v) => *v,
            Exp::Var(_, id) => *vars.get(id).unwrap_or(&0.0),
            Exp::Add(l, r) => l.evaluate(vars) + r.evaluate(vars),
            Exp::Sub(l, r) => l.evaluate(vars) - r.evaluate(vars),
            Exp::Mul(l, r) => l.evaluate(vars) * r.evaluate(vars),
            Exp::Div(l, r) => l.evaluate(vars) / r.evaluate(vars),
            Exp::Power(base, exp) => base.evaluate(vars).powf(*exp),
            Exp::Neg(e) => -e.evaluate(vars),
            Exp::Sin(e) => e.evaluate(vars).sin(),
            Exp::Cos(e) => e.evaluate(vars).cos(),
            Exp::Ln(e) => e.evaluate(vars).ln(),
            Exp::Exp(e) => e.evaluate(vars).exp(),
        }
    }

    pub fn differentiate(&self, var_id: VarId) -> Exp {
        match self {
            Exp::Val(_) => Exp::Val(0.0),
            Exp::Var(_, id) => {
                if *id == var_id {
                    Exp::Val(1.0)
                } else {
                    Exp::Val(0.0)
                }
            }
            Exp::Add(l, r) => Exp::add(l.differentiate(var_id), r.differentiate(var_id)),
            Exp::Sub(l, r) => Exp::sub(l.differentiate(var_id), r.differentiate(var_id)),
            Exp::Mul(l, r) => {
                let dl = l.differentiate(var_id);
                let dr = r.differentiate(var_id);
                Exp::add(Exp::mul(dl, (**r).clone()), Exp::mul((**l).clone(), dr))
            }
            Exp::Div(l, r) => {
                let dl = l.differentiate(var_id);
                let dr = r.differentiate(var_id);
                Exp::div(
                    Exp::sub(Exp::mul(dl, (**r).clone()), Exp::mul((**l).clone(), dr)),
                    Exp::power((**r).clone(), 2.0),
                )
            }
            Exp::Power(base, exp) => {
                let db = base.differentiate(var_id);
                Exp::mul(
                    Exp::mul(Exp::val(*exp), Exp::power((**base).clone(), exp - 1.0)),
                    db,
                )
            }
            Exp::Neg(e) => Exp::neg(e.differentiate(var_id)),
            Exp::Sin(e) => {
                let de = e.differentiate(var_id);
                Exp::mul(Exp::cos((**e).clone()), de)
            }
            Exp::Cos(e) => {
                let de = e.differentiate(var_id);
                Exp::neg(Exp::mul(Exp::sin((**e).clone()), de))
            }
            Exp::Ln(e) => {
                let de = e.differentiate(var_id);
                Exp::div(de, (**e).clone())
            }
            Exp::Exp(e) => {
                let de = e.differentiate(var_id);
                Exp::mul(Exp::exp((**e).clone()), de)
            }
        }
    }

    pub fn simplify(&self) -> Exp {
        match self {
            Exp::Add(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Exp::Val(lv), Exp::Val(rv)) => Exp::Val(lv + rv),
                    (Exp::Val(0.0), _) => rs,
                    (_, Exp::Val(0.0)) => ls,
                    _ => Exp::add(ls, rs),
                }
            }
            Exp::Sub(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Exp::Val(lv), Exp::Val(rv)) => Exp::Val(lv - rv),
                    (_, Exp::Val(0.0)) => ls,
                    _ => Exp::sub(ls, rs),
                }
            }
            Exp::Mul(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Exp::Val(lv), Exp::Val(rv)) => Exp::Val(lv * rv),
                    (Exp::Val(0.0), _) | (_, Exp::Val(0.0)) => Exp::Val(0.0),
                    (Exp::Val(1.0), _) => rs,
                    (_, Exp::Val(1.0)) => ls,
                    _ => Exp::mul(ls, rs),
                }
            }
            Exp::Div(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Exp::Val(lv), Exp::Val(rv)) if *rv != 0.0 => Exp::Val(lv / rv),
                    (Exp::Val(0.0), _) => Exp::Val(0.0),
                    (_, Exp::Val(1.0)) => ls,
                    _ => Exp::div(ls, rs),
                }
            }
            Exp::Power(base, exp) => {
                let bs = base.simplify();
                match &bs {
                    Exp::Val(v) => Exp::Val(v.powf(*exp)),
                    _ if *exp == 0.0 => Exp::Val(1.0),
                    _ if *exp == 1.0 => bs,
                    _ => Exp::power(bs, *exp),
                }
            }
            Exp::Neg(e) => {
                let es = e.simplify();
                match &es {
                    Exp::Val(v) => Exp::Val(-v),
                    _ => Exp::neg(es),
                }
            }
            _ => self.clone(),
        }
    }
}

impl fmt::Display for Exp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exp::Val(v) => write!(f, "{v}"),
            Exp::Var(name, _) => write!(f, "{name}"),
            Exp::Add(l, r) => write!(f, "({l} + {r})"),
            Exp::Sub(l, r) => write!(f, "({l} - {r})"),
            Exp::Mul(l, r) => write!(f, "({l} * {r})"),
            Exp::Div(l, r) => write!(f, "({l} / {r})"),
            Exp::Power(base, exp) => write!(f, "({base}^{exp})"),
            Exp::Neg(e) => write!(f, "(-{e})"),
            Exp::Sin(e) => write!(f, "sin({e})"),
            Exp::Cos(e) => write!(f, "cos({e})"),
            Exp::Ln(e) => write!(f, "ln({e})"),
            Exp::Exp(e) => write!(f, "exp({e})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate() {
        let mut vars = HashMap::new();
        vars.insert(VarId::new(0), 2.0);
        vars.insert(VarId::new(1), 3.0);

        let x = Exp::var("x".to_string(), VarId::new(0));
        let y = Exp::var("y".to_string(), VarId::new(1));

        let expr = Exp::add(Exp::mul(x.clone(), y.clone()), Exp::val(5.0));
        assert_eq!(expr.evaluate(&vars), 11.0);

        let expr2 = Exp::power(x.clone(), 2.0);
        assert_eq!(expr2.evaluate(&vars), 4.0);
    }

    #[test]
    fn test_evaluate_checked_missing_var() {
        let vars = HashMap::new();
        let x = Exp::var("x".to_string(), VarId::new(0));
        let err = x.evaluate_checked(&vars).expect_err("expected missing variable error");
        assert_eq!(err.var_id, VarId::new(0));
        assert_eq!(err.var_name, "x");
    }

    #[test]
    fn test_differentiate() {
        let x = Exp::var("x".to_string(), VarId::new(0));
        let y = Exp::var("y".to_string(), VarId::new(1));

        let expr = Exp::mul(x.clone(), y.clone());
        let dx = expr.differentiate(VarId::new(0));
        let dy = expr.differentiate(VarId::new(1));

        let mut vars = HashMap::new();
        vars.insert(VarId::new(0), 2.0);
        vars.insert(VarId::new(1), 3.0);

        assert_eq!(dx.evaluate(&vars), 3.0);
        assert_eq!(dy.evaluate(&vars), 2.0);

        let expr2 = Exp::power(x.clone(), 3.0);
        let dx2 = expr2.differentiate(VarId::new(0));
        assert_eq!(dx2.evaluate(&vars), 12.0);
    }

    #[test]
    fn test_simplify() {
        let expr = Exp::add(Exp::val(2.0), Exp::val(3.0));
        assert_eq!(expr.simplify(), Exp::val(5.0));

        let x = Exp::var("x".to_string(), VarId::new(0));
        let expr2 = Exp::mul(x.clone(), Exp::val(0.0));
        assert_eq!(expr2.simplify(), Exp::val(0.0));

        let expr3 = Exp::add(x.clone(), Exp::val(0.0));
        assert_eq!(expr3.simplify(), x);
    }
}
