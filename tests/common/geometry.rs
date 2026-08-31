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

//! Reusable geometric entities, tangency constraints, and warm-started solve
//! state for the animated geometry integration tests.

use crate::common::Sample;
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver, SolverError, SolverWorkspace};
use std::collections::HashMap;

/// Characteristic coordinate size used for normalized geometric corrections.
const COORDINATE_SCALE: f64 = 5.0;

/// Two-dimensional point represented by symbolic coordinate expressions.
#[derive(Clone)]
pub struct Point {
    /// Horizontal coordinate expression.
    pub x: Exp,
    /// Vertical coordinate expression.
    pub y: Exp,
}

/// Circle represented by a symbolic center and radius.
#[derive(Clone)]
pub struct Circle {
    /// Symbolic center point.
    pub center: Point,
    /// Symbolic positive radius supplied by the test scenario.
    pub radius: Exp,
}

/// Normal-form line `a*x + b*y + c = 0` with symbolic coefficients.
#[derive(Clone)]
pub struct Line {
    /// Horizontal unit-normal coefficient.
    pub a: Exp,
    /// Vertical unit-normal coefficient.
    pub b: Exp,
    /// Signed line offset.
    pub c: Exp,
}

/// Builder for a consistent set of named geometric equations and scales.
pub struct Geometry {
    /// Zero-valued residual expressions in declaration order.
    equations: Vec<Exp>,
    /// Positive characteristic scale corresponding to every residual.
    equation_scales: Vec<f64>,
    /// Unknown names selected as solver variables.
    solve_variables: Vec<String>,
    /// Initial values for unknowns before the first animated sample.
    initial_values: HashMap<String, f64>,
    /// Characteristic scale for each declared unknown.
    variable_scales: HashMap<String, f64>,
}

impl Geometry {
    /// Construct an empty geometric constraint system.
    pub fn new() -> Self {
        // All entities and constraints are explicit; the builder adds no hidden
        // origin, orientation, or regularization assumptions.
        Self {
            equations: Vec::new(),
            equation_scales: Vec::new(),
            solve_variables: Vec::new(),
            initial_values: HashMap::new(),
            variable_scales: HashMap::new(),
        }
    }

    /// Declare an unknown point with a named coordinate pair and initial value.
    pub fn unknown_point(&mut self, name: &str, x: f64, y: f64) -> Point {
        // Stable `_x`/`_y` names make captured traces self-explanatory and avoid
        // a parallel coordinate-index registry in test code.
        let x_name = format!("{name}_x");
        let y_name = format!("{name}_y");
        self.add_unknown(&x_name, x, COORDINATE_SCALE);
        self.add_unknown(&y_name, y, COORDINATE_SCALE);
        Point {
            x: Exp::var(x_name),
            y: Exp::var(y_name),
        }
    }

    /// Construct a point whose coordinates are supplied at every sample.
    pub fn known_point(&self, x_name: &str, y_name: &str) -> Point {
        // Known coordinates appear in expressions but are excluded from the
        // solver-variable list, so animation cannot move them accidentally.
        Point {
            x: Exp::var(x_name),
            y: Exp::var(y_name),
        }
    }

    /// Declare an unknown normalized line and its initial coefficients.
    pub fn unknown_line(&mut self, name: &str, a: f64, b: f64, c: f64) -> Line {
        // Normal components are dimensionless while the offset shares the
        // coordinate scale; preserving that distinction improves conditioning.
        let a_name = format!("{name}_a");
        let b_name = format!("{name}_b");
        let c_name = format!("{name}_c");
        self.add_unknown(&a_name, a, 1.0);
        self.add_unknown(&b_name, b, 1.0);
        self.add_unknown(&c_name, c, COORDINATE_SCALE);
        Line {
            a: Exp::var(a_name),
            b: Exp::var(b_name),
            c: Exp::var(c_name),
        }
    }

    /// Construct a circle from an existing center and symbolic radius.
    pub fn circle(&self, center: Point, radius: Exp) -> Circle {
        // Circles remain lightweight expression groupings; all mathematical
        // restrictions are added explicitly by constraint methods.
        Circle { center, radius }
    }

    /// Add one residual with its positive characteristic magnitude.
    pub fn constrain(&mut self, residual: Exp, scale: f64) {
        // Tests provide fixed positive scales chosen from their geometry units;
        // the public solver validates them again during builder configuration.
        self.equations.push(residual);
        self.equation_scales.push(scale);
    }

    /// Require a line's `(a,b)` normal to have unit Euclidean length.
    pub fn constrain_normalized(&mut self, line: &Line) {
        // Unit normalization makes signed point-to-line distance equal to the
        // raw normal-form expression used by tangency constraints.
        self.constrain(&line.a * &line.a + &line.b * &line.b - 1.0, 1.0);
    }

    /// Require a normalized line to be tangent to one signed side of a circle.
    pub fn constrain_line_circle_tangent(&mut self, line: &Line, circle: &Circle, side: f64) {
        // A side of +1 places the center one positive radius from the line;
        // -1 selects the reflected tangent branch.
        let signed_distance = &line.a * &circle.center.x + &line.b * &circle.center.y + &line.c;
        self.constrain(signed_distance - side * &circle.radius, COORDINATE_SCALE);
    }

    /// Require two circles to be externally tangent without selecting direction.
    pub fn constrain_external_tangency(&mut self, first: &Circle, second: &Circle) {
        // Squared distance avoids a square root while preserving the complete
        // tangency locus for positive radii.
        let dx = &first.center.x - &second.center.x;
        let dy = &first.center.y - &second.center.y;
        let radius_sum = &first.radius + &second.radius;
        self.constrain(
            &dx * &dx + &dy * &dy - &radius_sum * &radius_sum,
            COORDINATE_SCALE * COORDINATE_SCALE,
        );
    }

    /// Compile the system and retain its declared initial geometry.
    pub fn finish(self) -> GeometrySimulation {
        // Solver selection and scaling are centralized here so both animated
        // scenarios exercise identical numerical policy.
        let compiled = Compiler::compile(&self.equations)
            .expect("geometric constraint expressions must compile");
        let solve_variable_refs: Vec<&str> =
            self.solve_variables.iter().map(String::as_str).collect();
        let solver = NewtonRaphsonSolver::new_with_variables(compiled, &solve_variable_refs)
            .expect("every geometric unknown must occur in a constraint")
            .with_equation_scales(self.equation_scales)
            .expect("every geometric equation has a positive scale")
            .with_variable_scales(self.variable_scales)
            .expect("geometric scales name only declared unknowns")
            .with_residual_tolerance(1e-9);
        let workspace = solver.workspace();
        GeometrySimulation {
            solver,
            workspace,
            state: self.initial_values,
        }
    }

    /// Register one scalar unknown and its numerical scale.
    fn add_unknown(&mut self, name: &str, value: f64, scale: f64) {
        // Entity constructors generate unique names within each fixture; an
        // assertion catches accidental reuse instead of silently aliasing points.
        assert!(
            self.initial_values
                .insert(name.to_string(), value)
                .is_none(),
            "duplicate geometric unknown {name}"
        );
        self.solve_variables.push(name.to_string());
        self.variable_scales.insert(name.to_string(), scale);
    }
}

/// Warm-started reusable solver for a moving geometric system.
pub struct GeometrySimulation {
    /// Compiled constraint solver reused for all animation samples.
    solver: NewtonRaphsonSolver,
    /// Numerical buffers reused across every warm-started animation sample.
    workspace: SolverWorkspace,
    /// Complete most-recent accepted values and known parameters.
    state: HashMap<String, f64>,
}

impl GeometrySimulation {
    /// Solve one animation sample after updating its known geometry parameters.
    pub fn step(
        &mut self,
        time: f64,
        parameters: impl IntoIterator<Item = (String, f64)>,
    ) -> Result<Sample, SolverError> {
        // The preceding solution supplies a continuous branch initial guess,
        // while explicit parameters replace only externally animated values.
        self.state.extend(parameters);
        let solution = self.solver.solve(self.state.clone(), &mut self.workspace)?;
        self.state = solution.values.clone();
        Ok(Sample::new(time, solution.values))
    }
}
