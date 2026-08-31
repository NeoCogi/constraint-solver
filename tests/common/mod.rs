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

//! Shared infrastructure for the integration-level real-world validations.

use std::collections::HashMap;

/// Number of operating points used by every time-varying validation scenario.
///
/// A single shared count makes accidental one-step "transient" tests visible
/// and keeps the total release-test cost predictable.
pub const TRANSIENT_STEPS: usize = 100;

/// Return a uniformly spaced time in the inclusive interval `[0, duration]`.
pub fn sample_time(step: usize, duration: f64) -> f64 {
    // All callers iterate exactly `TRANSIENT_STEPS`, so the final sample uses
    // the requested duration and the first sample remains exactly zero.
    duration * step as f64 / (TRANSIENT_STEPS - 1) as f64
}

/// One accepted state in a time-varying validation trace.
pub struct Sample {
    /// Simulation time or normalized animation time.
    pub time: f64,
    /// Known parameters and solved values at this sample.
    values: HashMap<String, f64>,
}

impl Sample {
    /// Construct a sample from the complete accepted solver state.
    pub fn new(time: f64, values: HashMap<String, f64>) -> Self {
        // Keeping construction here gives electrical and geometric simulations
        // one identical immutable trace representation.
        Self { time, values }
    }

    /// Return one named parameter or solved value from this sample.
    pub fn value(&self, name: &str) -> f64 {
        // Fixtures use only compiled names, so absence identifies a malformed
        // trace immediately instead of becoming an ignored optional value.
        self.values[name]
    }
}

/// Ordered collection of accepted time-varying samples.
#[derive(Default)]
pub struct Trace {
    /// Samples in monotonically increasing time order.
    samples: Vec<Sample>,
}

impl Trace {
    /// Append one accepted state.
    pub fn push(&mut self, sample: Sample) {
        // Callers generate time monotonically; insertion order therefore remains
        // the natural waveform or animation order.
        self.samples.push(sample);
    }

    /// Borrow every sample in chronological order.
    pub fn samples(&self) -> &[Sample] {
        // Immutable access prevents assertions from invalidating trace order.
        &self.samples
    }

    /// Return the peak-to-peak range of one named quantity.
    pub fn span(&self, name: &str) -> f64 {
        // Every scenario produces 100 points, so infinite initial extrema make
        // an accidental empty trace conspicuous instead of hiding it.
        let minimum = self
            .samples
            .iter()
            .map(|sample| sample.value(name))
            .fold(f64::INFINITY, f64::min);
        let maximum = self
            .samples
            .iter()
            .map(|sample| sample.value(name))
            .fold(f64::NEG_INFINITY, f64::max);
        maximum - minimum
    }

    /// Return the ordinary covariance between two sampled quantities.
    pub fn covariance(&self, first: &str, second: &str) -> f64 {
        // Covariance sign provides a robust phase-polarity assertion without
        // depending on the exact value at one sample.
        let count = self.samples.len() as f64;
        let first_mean = self
            .samples
            .iter()
            .map(|sample| sample.value(first))
            .sum::<f64>()
            / count;
        let second_mean = self
            .samples
            .iter()
            .map(|sample| sample.value(second))
            .sum::<f64>()
            / count;
        self.samples
            .iter()
            .map(|sample| (sample.value(first) - first_mean) * (sample.value(second) - second_mean))
            .sum::<f64>()
            / count
    }

    /// Print requested quantities as CSV when tests run with output shown.
    pub fn print(&self, label: &str, names: &[&str]) {
        // Successful output is captured by default; `-- --show-output` turns
        // these records into directly inspectable 100-step traces.
        println!("{label}: time,{}", names.join(","));
        for sample in &self.samples {
            let values = names
                .iter()
                .map(|name| format!("{:.9}", sample.value(name)))
                .collect::<Vec<_>>()
                .join(",");
            println!("{:.9},{values}", sample.time);
        }
    }
}
