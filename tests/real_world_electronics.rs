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

//! Time-varying electronic systems built on the shared smooth MNA device layer.

mod common;

use common::electronics::{BjtModel, Circuit, DiodeModel, MosSwitchModel, Trace};
use common::{TRANSIENT_STEPS, sample_time};
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Convert a compact `(name, value)` slice into one simulation-step map.
fn parameters(values: &[(&str, f64)]) -> HashMap<String, f64> {
    // Owning names matches the solver's public input contract while keeping
    // individual transient loops concise.
    values
        .iter()
        .map(|(name, value)| ((*name).to_string(), *value))
        .collect()
}

/// Return four logic combinations with a twenty-sample slew before each state.
fn logic_inputs(step: usize) -> (f64, f64) {
    // Source stepping is the physical analogue of finite edge rise time and
    // prevents discontinuous five-volt jumps from dominating diode Newton work.
    let states = [(0.0, 0.0), (0.0, 5.0), (5.0, 5.0), (5.0, 0.0)];
    let segment_length = TRANSIENT_STEPS / states.len();
    let segment = step / segment_length;
    let within_segment = step % segment_length;
    if segment == 0 {
        return states[0];
    }
    let fraction = (within_segment as f64 / 20.0).min(1.0);
    let previous = states[segment - 1];
    let target = states[segment];
    (
        previous.0 + fraction * (target.0 - previous.0),
        previous.1 + fraction * (target.1 - previous.1),
    )
}

/// Classify only settled rail voltages, leaving transition samples unasserted.
fn settled_logic_level(voltage: f64) -> Option<bool> {
    // Exact low/high plateaus validate logic behavior, while intermediate input
    // voltages remain useful waveform samples without pretending to be digital.
    if voltage <= 0.1 {
        Some(false)
    } else if voltage >= 4.9 {
        Some(true)
    } else {
        None
    }
}

/// Validate a resistor-fed diode clipper across positive and negative AC bias.
#[test]
fn ac_source_resistor_diode_clipper_tracks_one_hundred_steps() {
    let mut circuit = Circuit::new();
    let source = circuit.known("source");
    let output = circuit.unknown("output");
    let ground = circuit.ground();
    let diode = DiodeModel::silicon();
    circuit.resistor(&source, &output, 1_000.0);
    circuit.diode(&output, &ground, diode);
    let mut simulation = circuit.finish();
    let mut trace = Trace::default();

    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, 1.0 / 60.0);
        let source_voltage = 5.0 * (TAU * 60.0 * time).sin();
        let sample = simulation
            .step(time, parameters(&[("source", source_voltage)]))
            .expect("every diode clipper operating point must converge");
        trace.push(sample);
    }

    trace.print("diode_clipper", &["source", "output"]);
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    let positive_peak = trace
        .samples()
        .iter()
        .max_by(|left, right| left.value("source").total_cmp(&right.value("source")))
        .expect("trace must contain a positive peak");
    assert!(positive_peak.value("output") > 0.5);
    assert!(positive_peak.value("output") < 0.8);
    assert!(positive_peak.value("output") < positive_peak.value("source"));
    let negative_peak = trace
        .samples()
        .iter()
        .min_by(|left, right| left.value("source").total_cmp(&right.value("source")))
        .expect("trace must contain a negative peak");
    assert!((negative_peak.value("output") - negative_peak.value("source")).abs() < 1e-6);
    assert!(trace.span("output") > 5.0);
}

/// Exercise diode OR and AND gates through every two-input logic combination.
#[test]
fn diode_logic_gates_follow_transient_truth_tables() {
    let mut circuit = Circuit::new();
    let input_a = circuit.known("input_a");
    let input_b = circuit.known("input_b");
    let supply = circuit.constant(5.0);
    let ground = circuit.ground();
    let or_output = circuit.unknown("or_output");
    let and_output = circuit.unknown("and_output");
    let diode = DiodeModel::silicon();

    circuit.resistor(&or_output, &ground, 10_000.0);
    circuit.diode(&input_a, &or_output, diode);
    circuit.diode(&input_b, &or_output, diode);
    circuit.resistor(&supply, &and_output, 10_000.0);
    circuit.diode(&and_output, &input_a, diode);
    circuit.diode(&and_output, &input_b, diode);

    let mut simulation = circuit.finish();
    let mut trace = Trace::default();
    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, 1e-3);
        let (a, b) = logic_inputs(step);
        let sample = simulation
            .step(time, parameters(&[("input_a", a), ("input_b", b)]))
            .expect("every diode-logic operating point must converge");
        if let (Some(a_high), Some(b_high)) = (settled_logic_level(a), settled_logic_level(b)) {
            assert_eq!(sample.value("or_output") > 2.5, a_high || b_high);
            assert_eq!(sample.value("and_output") > 2.5, a_high && b_high);
        }
        trace.push(sample);
    }

    trace.print(
        "diode_logic",
        &["input_a", "input_b", "or_output", "and_output"],
    );
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    assert!(trace.span("or_output") > 3.0);
    assert!(trace.span("and_output") > 3.0);
}

/// Validate bipolar inverter and series-transistor NAND behavior over time.
#[test]
fn bipolar_ttl_style_gates_follow_transient_truth_tables() {
    let mut circuit = Circuit::new();
    let input_a = circuit.known("input_a");
    let input_b = circuit.known("input_b");
    let supply = circuit.constant(5.0);
    let ground = circuit.ground();
    let inverter_base = circuit.unknown("inverter_base");
    let inverter_output = circuit.unknown("inverter_output");
    let nand_base_a = circuit.unknown("nand_base_a");
    let nand_base_b = circuit.unknown("nand_base_b");
    let nand_output = circuit.unknown("nand_output");
    let nand_middle = circuit.unknown("nand_middle");
    let transistor = BjtModel::silicon_npn();

    circuit.resistor(&input_a, &inverter_base, 10_000.0);
    circuit.resistor(&supply, &inverter_output, 1_000.0);
    circuit.npn(&inverter_output, &inverter_base, &ground, transistor);

    circuit.resistor(&input_a, &nand_base_a, 10_000.0);
    circuit.resistor(&input_b, &nand_base_b, 10_000.0);
    circuit.resistor(&supply, &nand_output, 1_000.0);
    circuit.resistor(&nand_middle, &ground, 10_000_000.0);
    circuit.npn(&nand_output, &nand_base_a, &nand_middle, transistor);
    circuit.npn(&nand_middle, &nand_base_b, &ground, transistor);

    let mut simulation = circuit.finish();
    let mut trace = Trace::default();
    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, 1e-3);
        let (a, b) = logic_inputs(step);
        let sample = simulation
            .step(time, parameters(&[("input_a", a), ("input_b", b)]))
            .expect("every bipolar-logic operating point must converge");
        if let (Some(a_high), Some(b_high)) = (settled_logic_level(a), settled_logic_level(b)) {
            assert_eq!(sample.value("inverter_output") > 2.5, !a_high);
            assert_eq!(sample.value("nand_output") > 2.5, !a_high || !b_high);
        }
        trace.push(sample);
    }

    trace.print(
        "bipolar_logic",
        &["input_a", "input_b", "inverter_output", "nand_output"],
    );
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    assert!(trace.span("inverter_output") > 3.0);
    assert!(trace.span("nand_output") > 3.0);
}

/// Validate complementary MOS inverter and NAND networks across all inputs.
#[test]
fn cmos_logic_gates_follow_transient_truth_tables() {
    let mut circuit = Circuit::new();
    let input_a = circuit.known("input_a");
    let input_b = circuit.known("input_b");
    let supply = circuit.constant(5.0);
    let ground = circuit.ground();
    let inverter_output = circuit.unknown("inverter_output");
    let nand_output = circuit.unknown("nand_output");
    let nand_middle = circuit.unknown("nand_middle");
    let mosfet = MosSwitchModel::logic();

    circuit.pmos(&supply, &input_a, &inverter_output, mosfet);
    circuit.nmos(&inverter_output, &input_a, &ground, mosfet);

    circuit.pmos(&supply, &input_a, &nand_output, mosfet);
    circuit.pmos(&supply, &input_b, &nand_output, mosfet);
    circuit.nmos(&nand_output, &input_a, &nand_middle, mosfet);
    circuit.nmos(&nand_middle, &input_b, &ground, mosfet);

    let mut simulation = circuit.finish();
    let mut trace = Trace::default();
    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, 1e-3);
        let (a, b) = logic_inputs(step);
        let sample = simulation
            .step(time, parameters(&[("input_a", a), ("input_b", b)]))
            .expect("every CMOS operating point must converge");
        if let (Some(a_high), Some(b_high)) = (settled_logic_level(a), settled_logic_level(b)) {
            assert_eq!(sample.value("inverter_output") > 2.5, !a_high);
            assert_eq!(sample.value("nand_output") > 2.5, !a_high || !b_high);
        }
        trace.push(sample);
    }

    trace.print(
        "cmos_logic",
        &["input_a", "input_b", "inverter_output", "nand_output"],
    );
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    assert!(trace.span("inverter_output") > 4.5);
    assert!(trace.span("nand_output") > 4.5);
}

/// Simulate a backward-Euler RC low-pass filter driven by a sine wave.
#[test]
fn rc_low_pass_filter_attenuates_over_one_hundred_steps() {
    let duration = 0.2;
    let time_step = duration / (TRANSIENT_STEPS - 1) as f64;
    let mut circuit = Circuit::new();
    let source = circuit.known("source");
    let previous_output = "previous_output";
    let output = circuit.unknown("output");
    let ground = circuit.ground();
    circuit.resistor(&source, &output, 1_000.0);
    circuit.capacitor_backward_euler(&output, &ground, 10e-6, previous_output, time_step);
    let mut simulation = circuit.finish();
    let mut trace = Trace::default();
    let mut previous_voltage = 0.0;

    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, duration);
        let source_voltage = (TAU * 10.0 * time).sin();
        let sample = simulation
            .step(
                time,
                parameters(&[
                    ("source", source_voltage),
                    (previous_output, previous_voltage),
                ]),
            )
            .expect("every RC backward-Euler step must converge");
        previous_voltage = sample.value("output");
        trace.push(sample);
    }

    trace.print("rc_low_pass", &["source", "output"]);
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    assert!(trace.span("source") > 1.9);
    assert!(trace.span("output") > 1.0);
    assert!(trace.span("output") < trace.span("source"));
}

/// Simulate a nonlinear common-emitter voltage amplifier over two AC cycles.
#[test]
fn common_emitter_amplifier_inverts_a_small_signal() {
    let mut circuit = Circuit::new();
    let source = circuit.known("source");
    let supply = circuit.constant(5.0);
    let ground = circuit.ground();
    let base = circuit.unknown("base");
    let output = circuit.unknown("output");
    let transistor = BjtModel::silicon_npn();
    circuit.resistor(&source, &base, 50_000.0);
    circuit.resistor(&supply, &output, 2_000.0);
    circuit.npn(&output, &base, &ground, transistor);
    let mut simulation = circuit.finish();
    let mut trace = Trace::default();

    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, 2e-3);
        let source_voltage = 0.78 + 0.04 * (TAU * 1_000.0 * time).sin();
        let sample = simulation
            .step(time, parameters(&[("source", source_voltage)]))
            .expect("every common-emitter operating point must converge");
        assert!((0.0..=5.0).contains(&sample.value("output")));
        trace.push(sample);
    }

    trace.print("common_emitter", &["source", "base", "output"]);
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    assert!(trace.span("output") > 0.1);
    assert!(trace.covariance("source", "output") < 0.0);
}
