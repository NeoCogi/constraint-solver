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

//! Small modified-nodal-analysis helpers shared by electronic integration
//! tests. These models favor smooth, deterministic solver validation over the
//! exhaustive device behavior of a production circuit simulator.

use crate::common::Sample;
use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver, SolverError, SolverWorkspace};
use std::collections::{BTreeMap, HashMap};

/// Characteristic KCL scale used to make ampere residuals dimensionless.
const KCL_EQUATION_SCALE: f64 = 1e-3;

/// Characteristic node-voltage scale used for every electrical unknown.
const NODE_VOLTAGE_SCALE: f64 = 5.0;

/// One electrical terminal with a voltage expression and optional KCL row.
#[derive(Clone)]
pub struct Terminal {
    /// Symbolic terminal voltage in volts.
    voltage: Exp,
    /// Unknown-node name receiving branch-current stamps; known voltages and
    /// ground deliberately have no row.
    unknown_node: Option<String>,
}

impl Terminal {
    /// Borrow the symbolic terminal voltage for controlled-device equations.
    pub fn voltage(&self) -> &Exp {
        // The expression is immutable, so callers can safely build derived
        // trees without gaining access to the circuit's KCL storage.
        &self.voltage
    }
}

/// Smooth Shockley diode parameters used by diode and circuit tests.
#[derive(Clone, Copy)]
pub struct DiodeModel {
    /// Reverse saturation current in amperes.
    pub saturation_current: f64,
    /// Dimensionless emission coefficient.
    pub emission_coefficient: f64,
    /// Thermal voltage in volts.
    pub thermal_voltage: f64,
}

impl DiodeModel {
    /// Construct the conventional small-signal silicon model used in tests.
    pub fn silicon() -> Self {
        // The selected values produce realistic sub-volt forward drops while
        // keeping all exponentials well within `f64` for the tested supplies.
        Self {
            saturation_current: 1e-12,
            emission_coefficient: 1.0,
            thermal_voltage: 0.025_85,
        }
    }

    /// Return current flowing from anode to cathode for a terminal voltage.
    fn current(&self, voltage: Exp) -> Exp {
        // Shockley's equation remains smooth in both forward and reverse bias,
        // making it suitable for symbolic differentiation and Armijo search.
        self.saturation_current
            * (Exp::exp(voltage / (self.emission_coefficient * self.thermal_voltage)) - 1.0)
    }
}

/// Compact Ebers–Moll parameters for an NPN bipolar transistor.
#[derive(Clone, Copy)]
pub struct BjtModel {
    /// Junction saturation current in amperes.
    pub saturation_current: f64,
    /// Forward common-emitter current gain.
    pub forward_beta: f64,
    /// Reverse common-emitter current gain used in saturation.
    pub reverse_beta: f64,
    /// Thermal voltage in volts.
    pub thermal_voltage: f64,
}

impl BjtModel {
    /// Construct a moderate-gain silicon transistor model.
    pub fn silicon_npn() -> Self {
        // A finite reverse gain lets the same smooth equations represent both
        // forward-active and saturated logic operation.
        Self {
            saturation_current: 1e-15,
            forward_beta: 100.0,
            reverse_beta: 2.0,
            thermal_voltage: 0.025_85,
        }
    }
}

/// Smooth voltage-controlled switch parameters for MOS logic validation.
#[derive(Clone, Copy)]
pub struct MosSwitchModel {
    /// Conductance in siemens when the channel is strongly enhanced.
    pub on_conductance: f64,
    /// Residual off-state conductance in siemens.
    pub off_conductance: f64,
    /// Gate overdrive at the middle of the smooth transition, in volts.
    pub threshold_voltage: f64,
    /// Voltage width of the logistic transition between off and on.
    pub transition_voltage: f64,
}

impl MosSwitchModel {
    /// Construct a symmetric digital MOS channel model.
    pub fn logic() -> Self {
        // The off/on ratio yields decisive logic levels without an ideal switch
        // discontinuity that would obscure nonlinear solver behavior.
        Self {
            on_conductance: 1e-2,
            off_conductance: 1e-9,
            threshold_voltage: 1.5,
            transition_voltage: 0.12,
        }
    }

    /// Smoothly interpolate channel conductance from an overdrive voltage.
    fn conductance(&self, overdrive: Exp) -> Exp {
        // Logistic activation is bounded between zero and one. Tested gate
        // levels keep its exponential argument far from overflow.
        let activation = 1.0 / (1.0 + Exp::exp(-overdrive / self.transition_voltage));
        self.off_conductance + (self.on_conductance - self.off_conductance) * activation
    }
}

/// Builder that accumulates one KCL residual per unknown voltage node.
pub struct Circuit {
    /// Deterministically ordered node equations; each expression is the sum of
    /// currents leaving that node.
    kcl: BTreeMap<String, Exp>,
    /// Unknown node names in declaration order for solver-variable selection.
    unknown_nodes: Vec<String>,
}

impl Circuit {
    /// Create an empty circuit with no unknown KCL rows.
    pub fn new() -> Self {
        // Starting without implicit ground avoids a redundant zero equation;
        // ground is represented by a constant-voltage terminal instead.
        Self {
            kcl: BTreeMap::new(),
            unknown_nodes: Vec::new(),
        }
    }

    /// Declare an unknown voltage node and its initially empty KCL equation.
    pub fn unknown(&mut self, name: &str) -> Terminal {
        // Duplicate declarations refer to the same node and must not add a
        // second equation or solver variable.
        if !self.kcl.contains_key(name) {
            self.kcl.insert(name.to_string(), Exp::val(0.0));
            self.unknown_nodes.push(name.to_string());
        }
        Terminal {
            voltage: Exp::var(name),
            unknown_node: Some(name.to_string()),
        }
    }

    /// Refer to a time-varying known voltage supplied with every sample.
    pub fn known(&self, name: &str) -> Terminal {
        // Known terminals participate in device expressions but intentionally
        // receive no KCL equation because an ideal source supplies their current.
        Terminal {
            voltage: Exp::var(name),
            unknown_node: None,
        }
    }

    /// Construct a fixed scalar-voltage terminal.
    pub fn constant(&self, voltage: f64) -> Terminal {
        // Constants avoid adding needless fixed variables to every solve map.
        Terminal {
            voltage: Exp::val(voltage),
            unknown_node: None,
        }
    }

    /// Construct the conventional zero-volt reference terminal.
    pub fn ground(&self) -> Terminal {
        // Ground is simply the most common fixed scalar terminal.
        self.constant(0.0)
    }

    /// Stamp a branch current flowing from `from` to `to` into both KCL rows.
    pub fn branch_current(&mut self, from: &Terminal, to: &Terminal, current: Exp) {
        if let Some(node) = &from.unknown_node {
            // A current defined as leaving `from` enters its KCL sum positively.
            let existing = self.kcl.get(node).expect("declared node must have KCL");
            self.kcl.insert(node.clone(), existing + &current);
        }
        if let Some(node) = &to.unknown_node {
            // The identical branch enters `to`, so its leaving-current sum gets
            // the opposite sign.
            let existing = self.kcl.get(node).expect("declared node must have KCL");
            self.kcl.insert(node.clone(), existing - current);
        }
    }

    /// Stamp an ideal resistor between two terminals.
    pub fn resistor(&mut self, first: &Terminal, second: &Terminal, resistance: f64) {
        // Ohm's law defines positive branch current from the first terminal to
        // the second; test fixtures provide strictly positive resistances.
        let current = (first.voltage() - second.voltage()) / resistance;
        self.branch_current(first, second, current);
    }

    /// Stamp a Shockley diode from anode to cathode.
    pub fn diode(&mut self, anode: &Terminal, cathode: &Terminal, model: DiodeModel) {
        // The model current orientation matches the branch stamping convention.
        let voltage = anode.voltage() - cathode.voltage();
        let current = model.current(voltage);
        self.branch_current(anode, cathode, current);
    }

    /// Stamp a backward-Euler capacitor using a known previous branch voltage.
    pub fn capacitor_backward_euler(
        &mut self,
        first: &Terminal,
        second: &Terminal,
        capacitance: f64,
        previous_voltage_name: &str,
        time_step: f64,
    ) {
        // Backward Euler turns the dynamic device into a conductance plus a
        // known history source for each nonlinear operating-point solve.
        let voltage = first.voltage() - second.voltage();
        let previous_voltage = Exp::var(previous_voltage_name);
        let current = (capacitance / time_step) * (voltage - previous_voltage);
        self.branch_current(first, second, current);
    }

    /// Stamp a three-terminal NPN transistor using smooth Ebers–Moll currents.
    pub fn npn(
        &mut self,
        collector: &Terminal,
        base: &Terminal,
        emitter: &Terminal,
        model: BjtModel,
    ) {
        // Junction currents use the same Shockley law, while alpha factors split
        // them into base recombination and bidirectional transport components.
        let vbe = base.voltage() - emitter.voltage();
        let vbc = base.voltage() - collector.voltage();
        let i_be = model.saturation_current * (Exp::exp(vbe / model.thermal_voltage) - 1.0);
        let i_bc = model.saturation_current * (Exp::exp(vbc / model.thermal_voltage) - 1.0);
        let alpha_forward = model.forward_beta / (model.forward_beta + 1.0);
        let alpha_reverse = model.reverse_beta / (model.reverse_beta + 1.0);

        self.branch_current(base, emitter, (1.0 - alpha_forward) * &i_be);
        self.branch_current(base, collector, (1.0 - alpha_reverse) * &i_bc);
        self.branch_current(
            collector,
            emitter,
            alpha_forward * i_be - alpha_reverse * i_bc,
        );
    }

    /// Stamp an n-channel MOS switch between drain and source.
    pub fn nmos(
        &mut self,
        drain: &Terminal,
        gate: &Terminal,
        source: &Terminal,
        model: MosSwitchModel,
    ) {
        // Positive gate-to-source overdrive smoothly enables a symmetric
        // resistive channel, sufficient for static CMOS truth-table validation.
        let overdrive = gate.voltage() - source.voltage() - model.threshold_voltage;
        let current = model.conductance(overdrive) * (drain.voltage() - source.voltage());
        self.branch_current(drain, source, current);
    }

    /// Stamp a p-channel MOS switch between source and drain.
    pub fn pmos(
        &mut self,
        source: &Terminal,
        gate: &Terminal,
        drain: &Terminal,
        model: MosSwitchModel,
    ) {
        // Positive source-to-gate overdrive enables the complementary channel.
        // Current orientation follows the usual high-side source-to-drain path.
        let overdrive = source.voltage() - gate.voltage() - model.threshold_voltage;
        let current = model.conductance(overdrive) * (source.voltage() - drain.voltage());
        self.branch_current(source, drain, current);
    }

    /// Compile all KCL rows and create a warm-started transient simulation.
    pub fn finish(self) -> CircuitSimulation {
        // BTreeMap order makes equation/scaling order reproducible even when
        // component declaration order changes.
        let equations: Vec<Exp> = self.kcl.into_values().collect();
        let equation_count = equations.len();
        let compiled = Compiler::compile(&equations).expect("circuit expressions must compile");
        let solve_variable_refs: Vec<&str> =
            self.unknown_nodes.iter().map(String::as_str).collect();
        let variable_scales: HashMap<String, f64> = self
            .unknown_nodes
            .iter()
            .map(|name| (name.clone(), NODE_VOLTAGE_SCALE))
            .collect();
        let solver = NewtonRaphsonSolver::new_with_variables(compiled, &solve_variable_refs)
            .expect("every circuit node must appear in a KCL equation")
            .with_equation_scales(vec![KCL_EQUATION_SCALE; equation_count])
            .expect("one positive scale is supplied for every KCL row")
            .with_variable_scales(variable_scales)
            .expect("voltage scales name only declared unknown nodes")
            .with_residual_tolerance(1e-8);

        // Zero volts is a neutral warm start for the first operating point;
        // later samples reuse the complete previously accepted state.
        let state = self
            .unknown_nodes
            .into_iter()
            .map(|name| (name, 0.0))
            .collect();
        let workspace = solver.workspace();
        CircuitSimulation {
            solver,
            workspace,
            state,
        }
    }
}

/// Reusable circuit solver whose accepted state becomes the next warm start.
pub struct CircuitSimulation {
    /// Compiled nonlinear KCL system shared across every time sample.
    solver: NewtonRaphsonSolver,
    /// Numerical buffers reused across every warm-started operating point.
    workspace: SolverWorkspace,
    /// Complete variable map from the most recently accepted operating point.
    state: HashMap<String, f64>,
}

impl CircuitSimulation {
    /// Solve one time sample after replacing its known parameter values.
    pub fn step(
        &mut self,
        time: f64,
        parameters: impl IntoIterator<Item = (String, f64)>,
    ) -> Result<Sample, SolverError> {
        // Keep solved node values as the nonlinear initial guess while known
        // sources and history terms are overwritten for the new time point.
        self.state.extend(parameters);
        let solution = self.solver.solve(self.state.clone(), &mut self.workspace)?;
        self.state = solution.values.clone();
        Ok(Sample::new(time, solution.values))
    }
}
