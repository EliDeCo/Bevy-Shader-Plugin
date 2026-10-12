//! Rust port of `assets/shaders/solar_system.wgsl`, used by `examples/solar_system_rust.rs`.
//!   set(0) uniform:            resolution (no time; all motion is computed on the CPU)
//!   set(1) binding(0) storage: planet UV positions, fully re-uploaded every frame
//!   set(1) binding(1) storage: average planet position, written by `average_positions`
//!                              and read back to the CPU every frame
//!   set(2) array:              per-planet RGBA colors, updated only when a planet completes an orbit
//!
//! `descriptor_set` is the group index. Bindings must match the registrations in the
//! example, and `&` (read) vs `&mut` (read_write) must match how each was registered.

#![cfg_attr(target_arch = "spirv", no_std)]
#![deny(warnings)]

use spirv_std::glam::{Vec2, Vec4, vec2, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

pub const N: usize = 8;

const STAR_CENTER: Vec2 = vec2(0.5, 0.5);
const STAR_RADIUS: f32 = 0.03;
const STAR_COLOR: Vec4 = vec4(1.0, 0.95, 0.4, 1.0);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FrameUniform {
    pub resolution: Vec2,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AveragePosition {
    pub value: Vec2,
}

fn hash(p: Vec2) -> f32 {
    ((p.x * 127.1 + p.y * 311.7).sin() * 43758.5453).fract()
}

// Returns brightness of a background star at the given pixel coordinate.
// Works in pixel space so cells are square regardless of window aspect ratio.
fn star_field(pixel: Vec2) -> f32 {
    let cell = (pixel / 30.0).floor();
    let local = (pixel / 30.0).fract();
    if hash(cell) > 0.18 {
        return 0.0;
    }
    let pos = vec2(hash(cell + vec2(7.3, 2.1)), hash(cell + vec2(3.7, 8.5)));
    let brightness = 0.5 + 0.5 * hash(cell + vec2(5.0, 5.0));
    if local.distance(pos) <= 0.06 {
        brightness
    } else {
        0.0
    }
}

// One thread is plenty for 8 planets.
#[spirv(compute(threads(1)))]
pub fn average_positions(
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] positions: &[Vec2; N],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] average: &mut AveragePosition,
) {
    let mut sum = Vec2::ZERO;
    let mut i = 0usize;
    while i < N {
        sum += positions[i];
        i += 1;
    }
    average.value = sum / N as f32;
}

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] u: &FrameUniform,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] positions: &[Vec2; N],
    #[spirv(storage_buffer, descriptor_set = 2, binding = 0)] colors: &[Vec4; N],
    output: &mut Vec4,
) {
    let frag = vec2(frag_coord.x, frag_coord.y);
    let uv = frag / u.resolution;

    // Star is always drawn on top.
    if uv.distance(STAR_CENTER) < STAR_RADIUS {
        *output = STAR_COLOR;
        return;
    }

    let mut i = 0usize;
    while i < N {
        let pos = positions[i];
        // Planet display radius scales with orbital distance: outer planets are larger.
        let radius = 0.006 + 0.006 * pos.distance(STAR_CENTER);
        if uv.distance(pos) < radius {
            let c = colors[i];
            *output = vec4(c.x, c.y, c.z, 1.0);
            return;
        }
        i += 1;
    }

    let bg = star_field(frag);
    *output = vec4(bg, bg, bg, 1.0);
}
