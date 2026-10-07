//! Fragment shader for the solar system example.
//! Bindings must match the registrations in ../../../main.rs.

#![no_std]

use spirv_std::glam::{Vec2, Vec4, vec2, vec4};
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

fn hash(p: Vec2) -> f32 {
    ((p.x * 127.1 + p.y * 311.7).sin() * 43758.5453).fract()
}

fn star_field(pixel: Vec2) -> f32 {
    let cell = (pixel / 30.0).floor();
    let local = (pixel / 30.0).fract();
    if hash(cell) > 0.18 {
        return 0.0;
    }
    let pos = vec2(hash(cell + vec2(7.3, 2.1)), hash(cell + vec2(3.7, 8.5)));
    let brightness = 0.5 + 0.5 * hash(cell + vec2(5.0, 5.0));
    if local.distance(pos) <= 0.06 { brightness } else { 0.0 }
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

    if uv.distance(STAR_CENTER) < STAR_RADIUS {
        *output = STAR_COLOR;
        return;
    }

    let mut i = 0usize;
    while i < N {
        let pos = positions[i];
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
