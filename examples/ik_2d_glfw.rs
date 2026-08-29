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

use constraint_solver::{Compiler, Exp, NewtonRaphsonSolver, UnderdeterminedPolicy};
use glfw::{Action, Context, Key, WindowEvent};
use glow::HasContext;
use rs_math3d::Vec2d;
use std::collections::HashMap;
use std::ffi::c_void;
use std::f64::consts::PI;

// Link lengths and draw sizes (world units == screen pixels).
const LINK_1: f64 = 140.0;
const LINK_2: f64 = 120.0;
const LINK_3: f64 = 100.0;
const LINK_4: f64 = 80.0;
const JOINT_RADIUS: f64 = 6.0;
const TARGET_RADIUS: f64 = 8.0;

// Current IK solution in local (base-centered) coordinates.
#[derive(Clone)]
struct IkState {
    joint1: Vec2d,
    joint2: Vec2d,
    joint3: Vec2d,
    effector: Vec2d,
}

// Small Vec2d helpers (kept explicit to avoid relying on operator overloads).
fn vec2(x: f64, y: f64) -> Vec2d {
    Vec2d::new(x, y)
}

fn vec2_len(v: &Vec2d) -> f64 {
    (v.x * v.x + v.y * v.y).sqrt()
}

fn vec2_add(a: &Vec2d, b: &Vec2d) -> Vec2d {
    vec2(a.x + b.x, a.y + b.y)
}

fn vec2_scale(v: &Vec2d, s: f64) -> Vec2d {
    vec2(v.x * s, v.y * s)
}

// Constraint helper: squared distance between two points equals length^2.
fn length_eq(ax: &Exp, ay: &Exp, bx: &Exp, by: &Exp, length: f64) -> Exp {
    let dx = Exp::sub(ax.clone(), bx.clone());
    let dy = Exp::sub(ay.clone(), by.clone());
    let dist_sq = Exp::add(Exp::power(dx, 2.0), Exp::power(dy, 2.0));
    Exp::sub(dist_sq, Exp::val(length * length))
}

// Clamp target into the reachable annulus of the chain.
fn clamp_target(target: &Vec2d, lengths: &[f64]) -> (Vec2d, bool) {
    let dist = vec2_len(target);
    let max_reach: f64 = lengths.iter().sum();
    let mut min_reach = 0.0;
    if let Some(max_len) = lengths.iter().copied().fold(None, |acc, v| {
        Some(acc.map_or(v, |m| if v > m { v } else { m }))
    }) {
        let sum_other = max_reach - max_len;
        min_reach = (max_len - sum_other).max(0.0);
    }

    if dist > max_reach {
        let scale = max_reach / dist;
        (vec2_scale(target, scale), true)
    } else if min_reach > 0.0 && dist < min_reach {
        if dist < 1e-6 {
            (vec2(min_reach, 0.0), true)
        } else {
            let scale = min_reach / dist;
            (vec2_scale(target, scale), true)
        }
    } else {
        (vec2(target.x, target.y), false)
    }
}

// Minimal GL state for line rendering.
struct GlState {
    program: glow::Program,
    vbo: glow::Buffer,
    a_pos_loc: u32,
    u_view_loc: Option<glow::UniformLocation>,
    u_color_loc: Option<glow::UniformLocation>,
}

fn circle_vertices(cx: f64, cy: f64, r: f64, segments: i32) -> Vec<f32> {
    let mut vertices = Vec::with_capacity(segments.max(0) as usize * 2);
    for i in 0..segments {
        let t = (i as f64) * 2.0 * PI / (segments as f64);
        let x = cx + r * t.cos();
        let y = cy + r * t.sin();
        vertices.push(x as f32);
        vertices.push(y as f32);
    }
    vertices
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn build_gl_state(gl: &glow::Context) -> GlState {
    let vertex_src = r#"
        #version 120
        attribute vec2 a_pos;
        uniform vec2 u_view;
        void main() {
            vec2 ndc = (a_pos / u_view) * 2.0 - 1.0;
            gl_Position = vec4(ndc, 0.0, 1.0);
        }
    "#;

    let fragment_src = r#"
        #version 120
        uniform vec3 u_color;
        void main() {
            gl_FragColor = vec4(u_color, 1.0);
        }
    "#;

    let program = gl.create_program().expect("create program");
    let vertex = gl.create_shader(glow::VERTEX_SHADER).expect("create vertex shader");
    gl.shader_source(vertex, vertex_src);
    gl.compile_shader(vertex);
    if !gl.get_shader_compile_status(vertex) {
        panic!("vertex shader error: {}", gl.get_shader_info_log(vertex));
    }

    let fragment = gl
        .create_shader(glow::FRAGMENT_SHADER)
        .expect("create fragment shader");
    gl.shader_source(fragment, fragment_src);
    gl.compile_shader(fragment);
    if !gl.get_shader_compile_status(fragment) {
        panic!("fragment shader error: {}", gl.get_shader_info_log(fragment));
    }

    gl.attach_shader(program, vertex);
    gl.attach_shader(program, fragment);
    gl.link_program(program);
    if !gl.get_program_link_status(program) {
        panic!("program link error: {}", gl.get_program_info_log(program));
    }

    gl.delete_shader(vertex);
    gl.delete_shader(fragment);

    let a_pos_loc = gl
        .get_attrib_location(program, "a_pos")
        .expect("attribute a_pos not found");
    let u_view_loc = gl.get_uniform_location(program, "u_view");
    let u_color_loc = gl.get_uniform_location(program, "u_color");

    let vbo = gl.create_buffer().expect("create vbo");

    GlState {
        program,
        vbo,
        a_pos_loc,
        u_view_loc,
        u_color_loc,
    }
}

fn draw_vertices(
    gl: &glow::Context,
    state: &GlState,
    mode: u32,
    vertices: &[f32],
    color: [f32; 3],
    line_width: f32,
) {
    unsafe {
        gl.uniform_3_f32(state.u_color_loc.as_ref(), color[0], color[1], color[2]);
        gl.line_width(line_width);

        gl.bind_buffer(glow::ARRAY_BUFFER, Some(state.vbo));
        let byte_len = vertices.len() * std::mem::size_of::<f32>();
        let bytes = std::slice::from_raw_parts(vertices.as_ptr() as *const u8, byte_len);
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STREAM_DRAW);

        gl.enable_vertex_attrib_array(state.a_pos_loc);
        gl.vertex_attrib_pointer_f32(state.a_pos_loc, 2, glow::FLOAT, false, 2 * 4, 0);
        gl.draw_arrays(mode, 0, (vertices.len() / 2) as i32);
        gl.disable_vertex_attrib_array(state.a_pos_loc);
    }
}

// Symbolic point wrapper used to build constraints.
#[derive(Clone)]
struct PointExpr {
    x: Exp,
    y: Exp,
}

impl PointExpr {
    // Fixed point (used for the origin).
    fn constant(x: f64, y: f64) -> Self {
        Self {
            x: Exp::val(x),
            y: Exp::val(y),
        }
    }

    // Live point backed by solver variables.
    fn from_vars(vars: &JointVars) -> Self {
        Self {
            x: vars.x.clone(),
            y: vars.y.clone(),
        }
    }
}

// Constraint interface used by joints and the target.
trait IKConstraint {
    fn equations(&self) -> Vec<Exp>;
    fn variables(&self) -> &Vec<String>;
}

// Names + expressions for a 2D point in the solver.
struct JointVars {
    x_name: String,
    y_name: String,
    var_names: Vec<String>,
    x: Exp,
    y: Exp,
}

impl JointVars {
    // Create named variables (e.g., j1x/j1y).
    fn new(name: &str) -> Self {
        let x_name = format!("{name}x");
        let y_name = format!("{name}y");
        let var_names = vec![x_name.clone(), y_name.clone()];
        Self {
            x: Exp::var(&x_name),
            y: Exp::var(&y_name),
            x_name,
            y_name,
            var_names,
        }
    }

    fn variables(&self) -> &Vec<String> {
        &self.var_names
    }

    // Insert a Vec2d into the solver input map.
    fn insert_values(&self, map: &mut HashMap<String, f64>, value: &Vec2d) {
        map.insert(self.x_name.clone(), value.x);
        map.insert(self.y_name.clone(), value.y);
    }

    // Read a Vec2d back from the solver output.
    fn read_values(&self, values: &HashMap<String, f64>) -> Option<Vec2d> {
        let x = values.get(&self.x_name)?;
        let y = values.get(&self.y_name)?;
        Some(vec2(*x, *y))
    }
}

// Joint constraint: child point stays a fixed distance from parent.
struct Joint {
    vars: JointVars,
    parent: PointExpr,
    length: f64,
}

impl Joint {
    // name: variable prefix, parent: point expression, length: fixed segment length.
    fn new(name: &str, parent: PointExpr, length: f64) -> Self {
        Self {
            vars: JointVars::new(name),
            parent,
            length,
        }
    }
}

impl IKConstraint for Joint {
    fn equations(&self) -> Vec<Exp> {
        vec![length_eq(
            &self.vars.x,
            &self.vars.y,
            &self.parent.x,
            &self.parent.y,
            self.length,
        )]
    }

    fn variables(&self) -> &Vec<String> {
        self.vars.variables()
    }
}

// Target constraint: effector should match a target point.
struct Target {
    vars: JointVars,
    effector: PointExpr,
}

impl Target {
    // name: variable prefix, effector: point expression for the end effector.
    fn new(name: &str, effector: PointExpr) -> Self {
        Self {
            vars: JointVars::new(name),
            effector,
        }
    }
}

impl IKConstraint for Target {
    fn equations(&self) -> Vec<Exp> {
        vec![
            Exp::sub(self.effector.x.clone(), self.vars.x.clone()),
            Exp::sub(self.effector.y.clone(), self.vars.y.clone()),
        ]
    }

    fn variables(&self) -> &Vec<String> {
        self.vars.variables()
    }
}

fn main() {
    let mut glfw = glfw::init(glfw::fail_on_errors).expect("failed to init glfw");
    glfw.window_hint(glfw::WindowHint::ContextVersion(2, 1));

    let (mut window, events) = glfw
        .create_window(900, 700, "2D IK (constraint-solver)", glfw::WindowMode::Windowed)
        .expect("failed to create window");

    window.make_current();
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_framebuffer_size_polling(true);

    let gl = unsafe {
        glow::Context::from_loader_function(|symbol| {
            window
                .get_proc_address(symbol)
                .map_or(std::ptr::null(), |proc| proc as *const () as *const c_void)
        })
    };
    let gl_state = unsafe { build_gl_state(&gl) };
    unsafe {
        gl.disable(glow::DEPTH_TEST);
        gl.clear_color(0.06, 0.07, 0.09, 1.0);
    }

    // Build the IK chain as constraint objects.
    let origin = PointExpr::constant(0.0, 0.0);
    let joint1 = Joint::new("j1", origin.clone(), LINK_1);
    let joint2 = Joint::new("j2", PointExpr::from_vars(&joint1.vars), LINK_2);
    let joint3 = Joint::new("j3", PointExpr::from_vars(&joint2.vars), LINK_3);
    let joint4 = Joint::new("j4", PointExpr::from_vars(&joint3.vars), LINK_4);
    let target = Target::new("t", PointExpr::from_vars(&joint4.vars));

    // Collect equations and compile them into the solver.
    let constraints: [&dyn IKConstraint; 5] = [&joint1, &joint2, &joint3, &joint4, &target];
    let mut equations = Vec::new();
    for constraint in constraints {
        equations.extend(constraint.equations());
    }

    let compiled = Compiler::compile(&equations).expect("compile");
    // Only the joints are solved for; the target is provided as a parameter.
    let joints: [&dyn IKConstraint; 4] = [&joint1, &joint2, &joint3, &joint4];
    let mut solve_var_names = Vec::with_capacity(joints.len() * 2);
    for joint in joints {
        solve_var_names.extend(joint.variables().iter().cloned());
    }
    let solve_var_refs: Vec<&str> = solve_var_names.iter().map(|s| s.as_str()).collect();
    let solver = NewtonRaphsonSolver::new_with_variables(compiled, &solve_var_refs)
        .expect("solver")
        // Preserve the previous frame's null-space component so the
        // underdetermined chain moves continuously instead of being projected
        // toward the origin on every frame.
        .with_underdetermined_policy(UnderdeterminedPolicy::MinimumNormStep);

    // Initial pose: straight chain along +X axis.
    let mut state = IkState {
        joint1: vec2(LINK_1, 0.0),
        joint2: vec2(LINK_1 + LINK_2, 0.0),
        joint3: vec2(LINK_1 + LINK_2 + LINK_3, 0.0),
        effector: vec2(LINK_1 + LINK_2 + LINK_3 + LINK_4, 0.0),
    };

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            if let WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                window.set_should_close(true);
            }
        }

        let (win_w, win_h) = window.get_size();
        let (fb_w, fb_h) = window.get_framebuffer_size();
        let win_w = win_w.max(1) as f64;
        let win_h = win_h.max(1) as f64;
        let fb_w = fb_w.max(1) as f64;
        let fb_h = fb_h.max(1) as f64;

        let scale_x = fb_w / win_w;
        let scale_y = fb_h / win_h;

        let (cursor_x, cursor_y) = window.get_cursor_pos();
        let mouse_x = cursor_x * scale_x;
        let mouse_y = cursor_y * scale_y;

        // Place the chain base at the window center.
        let base = vec2(fb_w * 0.5, fb_h * 0.5);

        // Mouse position in base-centered coordinates.
        let raw_target = vec2(mouse_x - base.x, (fb_h - mouse_y) - base.y);

        let lengths = [LINK_1, LINK_2, LINK_3, LINK_4];
        let (clamped_target, was_clamped) = clamp_target(&raw_target, &lengths);

        // Provide current state + target as the solver initial guess.
        let mut initial = HashMap::with_capacity(10);
        let joint_values = [
            (&joint1, &state.joint1),
            (&joint2, &state.joint2),
            (&joint3, &state.joint3),
            (&joint4, &state.effector),
        ];
        for (joint, value) in joint_values {
            joint.vars.insert_values(&mut initial, value);
        }
        target.vars.insert_values(&mut initial, &clamped_target);

        // Solve for the joint positions.
        if let Ok(solution) = solver.solve(initial) {
            // Every successful result now carries an explicit convergence reason;
            // failures are represented only by `Err`, so no redundant boolean
            // success check is needed here.
            if let (Some(j1), Some(j2), Some(j3), Some(j4)) = (
                joint1.vars.read_values(&solution.values),
                joint2.vars.read_values(&solution.values),
                joint3.vars.read_values(&solution.values),
                joint4.vars.read_values(&solution.values),
            ) {
                state.joint1 = j1;
                state.joint2 = j2;
                state.joint3 = j3;
                state.effector = j4;
            }
        }

        // Convert to world (window) coordinates for drawing.
        let joint1_world = vec2_add(&base, &state.joint1);
        let joint2_world = vec2_add(&base, &state.joint2);
        let joint3_world = vec2_add(&base, &state.joint3);
        let eff_world = vec2_add(&base, &state.effector);

        let raw_world = vec2_add(&base, &raw_target);
        let clamped_world = vec2_add(&base, &clamped_target);

        // Render chain + target markers.
        unsafe {
            gl.viewport(0, 0, fb_w as i32, fb_h as i32);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(gl_state.program));
            gl.uniform_2_f32(
                gl_state.u_view_loc.as_ref(),
                fb_w as f32,
                fb_h as f32,
            );

            let mut line_vertices = Vec::with_capacity(16);
            let points = [
                base,
                joint1_world,
                joint2_world,
                joint3_world,
                eff_world,
            ];
            for i in 0..points.len() - 1 {
                let a = &points[i];
                let b = &points[i + 1];
                line_vertices.push(a.x as f32);
                line_vertices.push(a.y as f32);
                line_vertices.push(b.x as f32);
                line_vertices.push(b.y as f32);
            }
            draw_vertices(
                &gl,
                &gl_state,
                glow::LINES,
                &line_vertices,
                [0.85, 0.87, 0.92],
                4.0,
            );

            for point in points {
                let circle = circle_vertices(point.x, point.y, JOINT_RADIUS, 24);
                draw_vertices(
                    &gl,
                    &gl_state,
                    glow::LINE_LOOP,
                    &circle,
                    [0.32, 0.8, 0.68],
                    2.0,
                );
            }

            if was_clamped {
                let raw_target = circle_vertices(raw_world.x, raw_world.y, TARGET_RADIUS, 24);
                draw_vertices(
                    &gl,
                    &gl_state,
                    glow::LINE_LOOP,
                    &raw_target,
                    [0.9, 0.35, 0.35],
                    2.0,
                );
            }

            let clamped_target =
                circle_vertices(clamped_world.x, clamped_world.y, TARGET_RADIUS, 32);
            draw_vertices(
                &gl,
                &gl_state,
                glow::LINE_LOOP,
                &clamped_target,
                [0.95, 0.9, 0.35],
                2.0,
            );

            gl.use_program(None);
        }

        window.swap_buffers();
    }
}
