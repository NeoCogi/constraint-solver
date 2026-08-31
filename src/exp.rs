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
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Immutable symbolic scalar expression tree.
///
/// Expressions represent residual functions rather than equations with a
/// separate equality operator: a constraint is satisfied when its expression
/// evaluates to zero. The solver differentiates the tree as written and does
/// not substitute finite-difference derivatives. Algebraically equivalent
/// trees can consequently behave differently at domain boundaries where one
/// symbolic derivative contains an indeterminate floating-point operation.
///
/// The ordinary arithmetic operators accept owned or borrowed expressions and
/// `f64` constants. Operator evaluation builds a new symbolic tree; it does not
/// numerically evaluate either operand.
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

impl From<f64> for Exp {
    /// Convert a floating-point scalar into a symbolic constant.
    fn from(value: f64) -> Self {
        // Preserve the exact scalar payload just as `Exp::val` does.
        Self::Val(value)
    }
}

impl From<&Exp> for Exp {
    /// Clone a borrowed expression when an operator needs owned tree nodes.
    fn from(value: &Exp) -> Self {
        // Expression trees are immutable values, so cloning cannot change the
        // source operand or introduce shared mutable state.
        value.clone()
    }
}

/// Implement one binary operator for owned and borrowed expression left-hand
/// sides, plus scalar left-hand sides whose operand order must be preserved.
///
/// Keeping these mechanically identical implementations in one macro makes the
/// supported ownership combinations consistent across all four operators.
macro_rules! impl_binary_operator {
    ($trait:ident, $method:ident, $constructor:ident) => {
        impl<Rhs> $trait<Rhs> for Exp
        where
            Rhs: Into<Exp>,
        {
            type Output = Exp;

            fn $method(self, rhs: Rhs) -> Self::Output {
                // Move the owned left tree and normalize the right operand into
                // an expression before constructing the requested binary node.
                Exp::$constructor(self, rhs.into())
            }
        }

        impl<Rhs> $trait<Rhs> for &Exp
        where
            Rhs: Into<Exp>,
        {
            type Output = Exp;

            fn $method(self, rhs: Rhs) -> Self::Output {
                // Clone only the borrowed left tree; an owned right operand can
                // still move directly into the result.
                Exp::$constructor(self.clone(), rhs.into())
            }
        }

        impl $trait<Exp> for f64 {
            type Output = Exp;

            fn $method(self, rhs: Exp) -> Self::Output {
                // Scalar-left subtraction and division are order-sensitive, so
                // retain the scalar as the first constructor argument.
                Exp::$constructor(Exp::Val(self), rhs)
            }
        }

        impl $trait<&Exp> for f64 {
            type Output = Exp;

            fn $method(self, rhs: &Exp) -> Self::Output {
                // Borrowed right operands are cloned into the new immutable
                // tree while the scalar becomes its left constant node.
                Exp::$constructor(Exp::Val(self), rhs.clone())
            }
        }
    };
}

impl_binary_operator!(Add, add, add);
impl_binary_operator!(Sub, sub, sub);
impl_binary_operator!(Mul, mul, mul);
impl_binary_operator!(Div, div, div);

impl Neg for Exp {
    type Output = Exp;

    fn neg(self) -> Self::Output {
        // Move the owned tree directly under a unary negation node.
        Exp::neg(self)
    }
}

impl Neg for &Exp {
    type Output = Exp;

    fn neg(self) -> Self::Output {
        // Preserve the borrowed source by cloning it into the new unary node.
        Exp::neg(self.clone())
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

    /// Exercise every ownership and scalar position supported by the binary
    /// operator implementations, including order-sensitive scalar-left forms.
    #[test]
    fn test_arithmetic_operators_preserve_expression_structure() {
        let x = Exp::var("x");
        let y = Exp::var("y");

        let borrowed_sum = &x + &y;
        assert_eq!(borrowed_sum, Exp::add(x.clone(), y.clone()));

        let owned_product = x.clone() * y.clone();
        assert_eq!(owned_product, Exp::mul(x.clone(), y.clone()));

        let scalar_right = x.clone() - 2.0;
        assert_eq!(scalar_right, Exp::sub(x.clone(), Exp::val(2.0)));

        let scalar_left = 2.0 / &x;
        assert_eq!(scalar_left, Exp::div(Exp::val(2.0), x.clone()));

        let borrowed_negation = -&y;
        assert_eq!(borrowed_negation, Exp::neg(y));
    }

    /// Compile and solve a mixed borrowed/scalar operator expression through
    /// the public pipeline so ergonomic syntax is covered beyond enum shapes.
    #[test]
    fn test_operator_expression_compiles_and_solves() {
        use crate::compiler::Compiler;
        use crate::solver::NewtonRaphsonSolver;
        use std::collections::HashMap;

        let x = Exp::var("x");
        let equation = 2.0 / (&x + 1.0) - 1.0;
        let compiled = Compiler::compile(&[equation]).expect("operator tree must compile");
        let solver = NewtonRaphsonSolver::new(compiled);
        let initial = HashMap::from([("x".to_string(), 0.0)]);

        let solution = solver
            .solve_once(initial)
            .expect("operator tree must solve");
        assert!((solution.values["x"] - 1.0).abs() < 1e-10);
    }

    /// Carry a logarithm through public construction, compilation, symbolic
    /// differentiation, and nonlinear solving on its valid real domain.
    #[test]
    fn test_logarithm_expression_compiles_differentiates_and_solves() {
        use crate::compiler::Compiler;
        use crate::solver::NewtonRaphsonSolver;
        use std::collections::HashMap;

        // ln(x)-1 has the unique positive root e and derivative 1/x. Beginning
        // at two keeps every Newton and Armijo candidate inside the real domain.
        let x = Exp::var("x");
        let equation = Exp::ln(x) - 1.0;
        let compiled = Compiler::compile(&[equation]).expect("logarithm tree must compile");
        let solver = NewtonRaphsonSolver::new(compiled).with_residual_tolerance(1e-12);
        let initial = HashMap::from([("x".to_string(), 2.0)]);

        let solution = solver
            .solve_once(initial)
            .expect("logarithm equation must solve");

        assert!((solution.values["x"] - std::f64::consts::E).abs() < 1e-10);
        assert!(solution.error < 1e-12);
    }
}
