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

//! Animated line/circle tangency and rigid six-circle packing validations.

mod common;
#[path = "common/geometry.rs"]
mod geometry;

use common::{TRANSIENT_STEPS, Trace, sample_time};
use constraint_solver::Exp;
use geometry::Geometry;
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

/// Convert named scalar pairs into one animation parameter map.
fn parameters(values: &[(&str, f64)]) -> HashMap<String, f64> {
    // Geometry simulations use the same owned name/value contract as public
    // solver calls and electronic samples.
    values
        .iter()
        .map(|(name, value)| ((*name).to_string(), *value))
        .collect()
}

/// Compute Euclidean distance between two named points in one accepted sample.
fn sample_distance(sample: &common::Sample, first: &str, second: &str) -> f64 {
    // Point names follow the shared `_x`/`_y` convention established by the
    // geometry builder.
    let dx = sample.value(&format!("{first}_x")) - sample.value(&format!("{second}_x"));
    let dy = sample.value(&format!("{first}_y")) - sample.value(&format!("{second}_y"));
    dx.hypot(dy)
}

/// Track one continuously selected common tangent of two moving circles.
#[test]
fn common_tangent_line_tracks_two_circles_for_one_hundred_steps() {
    let first_center_x = 0.0;
    let first_center_y = 0.0;
    let initial_second_x: f64 = 4.0;
    let initial_second_y: f64 = 1.0;
    let center_distance = initial_second_x.hypot(initial_second_y);
    let initial_a = -initial_second_y / center_distance;
    let initial_b = initial_second_x / center_distance;

    let mut geometry = Geometry::new();
    let first_center = geometry.known_point("first_x", "first_y");
    let second_center = geometry.known_point("second_x", "second_y");
    let first_circle = geometry.circle(first_center, Exp::var("first_radius"));
    let second_circle = geometry.circle(second_center, Exp::var("second_radius"));
    let tangent = geometry.unknown_line("tangent", initial_a, initial_b, 1.0);
    geometry.constrain_normalized(&tangent);
    geometry.constrain_line_circle_tangent(&tangent, &first_circle, 1.0);
    geometry.constrain_line_circle_tangent(&tangent, &second_circle, 1.0);
    let mut simulation = geometry.finish();
    let mut trace = Trace::default();

    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, 1.0);
        let second_x = 4.0 + 0.5 * (TAU * time).sin();
        let second_y = 1.0 + 0.4 * (TAU * time).sin();
        let sample = simulation
            .step(
                time,
                parameters(&[
                    ("first_x", first_center_x),
                    ("first_y", first_center_y),
                    ("second_x", second_x),
                    ("second_y", second_y),
                    ("first_radius", 1.0),
                    ("second_radius", 1.0),
                ]),
            )
            .expect("each common-tangent animation sample must converge");
        let a = sample.value("tangent_a");
        let b = sample.value("tangent_b");
        let c = sample.value("tangent_c");
        assert!((a.hypot(b) - 1.0).abs() < 1e-8);
        assert!((a * first_center_x + b * first_center_y + c - 1.0).abs() < 1e-8);
        assert!((a * second_x + b * second_y + c - 1.0).abs() < 1e-8);
        trace.push(sample);
    }

    trace.print(
        "common_tangent",
        &[
            "second_x",
            "second_y",
            "tangent_a",
            "tangent_b",
            "tangent_c",
        ],
    );
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    assert!(trace.span("tangent_a") > 0.02);
    assert!(trace.span("tangent_b") > 0.005);
    assert!(trace.covariance("second_y", "tangent_a") < 0.0);
}

/// Animate a rigid triangular packing of six mutually contacting circles.
#[test]
fn six_circle_tangent_packing_tracks_one_hundred_steps() {
    // Six circles cannot all be pairwise tangent in the plane because circle
    // contact graphs are planar. This triangular packing realizes nine contact
    // edges and verifies that every remaining pair stays non-overlapping.
    let initial_centers = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (1.0, 3.0_f64.sqrt()),
        (3.0, 3.0_f64.sqrt()),
        (2.0, 2.0 * 3.0_f64.sqrt()),
    ];
    let contact_pairs = [
        (0, 1),
        (1, 2),
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
    ];

    let mut geometry = Geometry::new();
    let points: Vec<_> = initial_centers
        .iter()
        .enumerate()
        .map(|(index, (x, y))| geometry.unknown_point(&format!("circle_{index}"), *x, *y))
        .collect();
    let circles: Vec<_> = points
        .into_iter()
        .map(|point| geometry.circle(point, Exp::var("radius")))
        .collect();
    for (first, second) in contact_pairs {
        geometry.constrain_external_tangency(&circles[first], &circles[second]);
    }
    geometry.constrain(&circles[0].center.x - Exp::var("center_x"), 5.0);
    geometry.constrain(&circles[0].center.y - Exp::var("center_y"), 5.0);
    geometry.constrain(&circles[1].center.y - Exp::var("center_y"), 5.0);
    let mut simulation = geometry.finish();
    let mut trace = Trace::default();

    for step in 0..TRANSIENT_STEPS {
        let time = sample_time(step, 1.0);
        let center_x = 0.5 * (TAU * time).sin();
        let center_y = 0.25 * (TAU * time).sin();
        let radius = 1.0 + 0.1 * (2.0 * PI * time).sin();
        let sample = simulation
            .step(
                time,
                parameters(&[
                    ("center_x", center_x),
                    ("center_y", center_y),
                    ("radius", radius),
                ]),
            )
            .expect("each six-circle packing sample must converge");

        for (first, second) in contact_pairs {
            let distance = sample_distance(
                &sample,
                &format!("circle_{first}"),
                &format!("circle_{second}"),
            );
            assert!((distance - 2.0 * radius).abs() < 1e-7);
        }
        for first in 0..circles.len() {
            for second in (first + 1)..circles.len() {
                assert!(
                    sample_distance(
                        &sample,
                        &format!("circle_{first}"),
                        &format!("circle_{second}")
                    ) >= 2.0 * radius - 1e-7
                );
            }
        }
        trace.push(sample);
    }

    trace.print(
        "six_circle_packing",
        &[
            "radius",
            "circle_0_x",
            "circle_0_y",
            "circle_5_x",
            "circle_5_y",
        ],
    );
    assert_eq!(trace.samples().len(), TRANSIENT_STEPS);
    assert!(trace.span("circle_0_x") > 0.9);
    assert!(trace.span("circle_5_y") > 0.5);
}
