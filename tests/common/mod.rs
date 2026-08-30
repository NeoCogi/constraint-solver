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

pub mod electronics;

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
