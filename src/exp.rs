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

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Exp {
    Val(f64),
    Var(String),
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
    pub fn var(name: impl Into<String>) -> Self {
        Exp::Var(name.into())
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
}

impl fmt::Display for Exp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exp::Val(v) => write!(f, "{v}"),
            Exp::Var(name) => write!(f, "{name}"),
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
    fn test_constructors_match_variants() {
        assert_eq!(Exp::var("x"), Exp::Var("x".to_string()));
        assert_eq!(Exp::val(2.5), Exp::Val(2.5));

        let expr = Exp::add(Exp::val(1.0), Exp::val(2.0));
        assert_eq!(
            expr,
            Exp::Add(Box::new(Exp::Val(1.0)), Box::new(Exp::Val(2.0)))
        );
    }

    #[test]
    fn test_display_formats_expression() {
        let x = Exp::var("x");
        let y = Exp::var("y");
        let expr = Exp::sub(
            Exp::add(
                Exp::power(x.clone(), 2.0),
                Exp::mul(Exp::val(3.0), y.clone()),
            ),
            Exp::val(1.0),
        );

        assert_eq!(format!("{expr}"), "(((x^2) + (3 * y)) - 1)");
    }
}
