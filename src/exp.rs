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

//! Symbolic scalar expression nodes and ergonomic constructors used to define
//! nonlinear equation residuals.

use std::fmt;

/// Immutable symbolic scalar expression tree.
///
/// Expressions represent residual functions rather than equations with a
/// separate equality operator: a constraint is satisfied when its expression
/// evaluates to zero. The solver differentiates the tree as written and does
/// not substitute finite-difference derivatives. Algebraically equivalent
/// trees can consequently behave differently at domain boundaries where one
/// symbolic derivative contains an indeterminate floating-point operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Exp {
    /// Literal floating-point constant.
    Val(f64),
    /// Variable referenced by its pre-compilation name.
    Var(String),
    /// Sum of two child expressions.
    Add(Box<Exp>, Box<Exp>),
    /// Difference between left and right child expressions.
    Sub(Box<Exp>, Box<Exp>),
    /// Product of two child expressions.
    Mul(Box<Exp>, Box<Exp>),
    /// Quotient of the left child by the right child.
    Div(Box<Exp>, Box<Exp>),
    /// Real-valued constant power of a base expression.
    Power(Box<Exp>, f64),
    /// Arithmetic negation of a child expression.
    Neg(Box<Exp>),
    /// Sine of a child expression, measured in radians.
    Sin(Box<Exp>),
    /// Cosine of a child expression, measured in radians.
    Cos(Box<Exp>),
    /// Natural logarithm of a child expression.
    Ln(Box<Exp>),
    /// Natural exponential of a child expression.
    Exp(Box<Exp>),
}

#[allow(clippy::self_named_constructors, clippy::should_implement_trait)]
impl Exp {
    /// Construct a named symbolic variable.
    pub fn var(name: impl Into<String>) -> Self {
        // Retain the caller-facing name until compilation assigns a dense ID.
        Exp::Var(name.into())
    }

    /// Construct a literal floating-point constant.
    pub fn val(v: f64) -> Self {
        // Constants are stored without normalization so IEEE-754 behavior is
        // preserved during checked evaluation.
        Exp::Val(v)
    }

    /// Construct the sum `lhs + rhs`.
    pub fn add(lhs: Exp, rhs: Exp) -> Self {
        // Box recursive children so every enum value has a fixed stack size.
        Exp::Add(Box::new(lhs), Box::new(rhs))
    }

    /// Construct the difference `lhs - rhs`.
    pub fn sub(lhs: Exp, rhs: Exp) -> Self {
        // Preserve operand order because subtraction is not commutative.
        Exp::Sub(Box::new(lhs), Box::new(rhs))
    }

    /// Construct the product `lhs * rhs`.
    pub fn mul(lhs: Exp, rhs: Exp) -> Self {
        // Symbolic simplification occurs during compilation, not construction.
        Exp::Mul(Box::new(lhs), Box::new(rhs))
    }

    /// Construct the quotient `lhs / rhs`.
    pub fn div(lhs: Exp, rhs: Exp) -> Self {
        // Domain and division-by-zero checks remain evaluation concerns because
        // the right expression can depend on variables.
        Exp::Div(Box::new(lhs), Box::new(rhs))
    }

    /// Construct `base` raised to the constant real exponent `exp`.
    pub fn power(base: Exp, exp: f64) -> Self {
        // The exponent is scalar metadata rather than another expression,
        // matching the derivative rules supported by the compiler.
        Exp::Power(Box::new(base), exp)
    }

    /// Construct arithmetic negation of `exp`.
    pub fn neg(exp: Exp) -> Self {
        // Keep negation explicit so differentiation and formatting can preserve
        // its unary structure.
        Exp::Neg(Box::new(exp))
    }

    /// Construct the sine of `exp`, interpreted in radians.
    pub fn sin(exp: Exp) -> Self {
        // Trigonometric evaluation and differentiation operate on this owned
        // boxed child.
        Exp::Sin(Box::new(exp))
    }

    /// Construct the cosine of `exp`, interpreted in radians.
    pub fn cos(exp: Exp) -> Self {
        // Retain cosine as a distinct node so its derivative can introduce the
        // correct negated sine expression.
        Exp::Cos(Box::new(exp))
    }

    /// Construct the natural logarithm of `exp`.
    pub fn ln(exp: Exp) -> Self {
        // The positive-domain requirement is checked when concrete variable
        // values are evaluated.
        Exp::Ln(Box::new(exp))
    }

    /// Construct the natural exponential of `exp`.
    pub fn exp(exp: Exp) -> Self {
        // Use the enum-qualified name to disambiguate the node from this
        // convenience constructor.
        Exp::Exp(Box::new(exp))
    }
}

impl fmt::Display for Exp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Parenthesize binary operations consistently so the rendered tree is
        // unambiguous without relying on precedence rules.
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
