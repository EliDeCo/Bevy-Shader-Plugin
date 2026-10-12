//! Particles example, rust-gpu version: the shader, used by `app.rs` next to this file.
//! It's the shader crate's `lib.rs` (`shader/src/lib.rs` in bevy-shader-plugin-rust-template),
//! and a port of `../wgsl/shader.wgsl`.
//!   init       (compute, once):          scatters particles across the top of the window
//!   collide    (compute, every substep): works out each particle's collision response
//!   integrate  (compute, every substep): applies gravity + collisions, moves, bounces off walls
//!   draw       (fragment):               white circles on black
//!
//! Everything is in physical pixels with y pointing down, matching `frag_coord`.
//!
//! GPU buffers are registered `read_write`, and wgpu requires every entry point that
//! uses a binding to declare the same access, so every entry point takes them as `&mut`
//! — even `draw`, which only reads.

#![cfg_attr(target_arch = "spirv", no_std)]
#![deny(warnings)]

use spirv_std::glam::{UVec3, Vec2, Vec4, vec2, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SimParams {
    pub resolution: Vec2,
    pub gravity: f32,
    pub dt: f32,
    pub radius: f32,
    pub restitution: f32,
    pub seed: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Particle {
    pub pos: Vec2,
    pub vel: Vec2,
}

// Collision response for one particle, computed by `collide` and applied by `integrate`.
// Kept separate so `collide` only reads `particles` and each thread writes only its own
// slot — no thread ever reads a value another thread is writing.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Correction {
    pub dpos: Vec2,
    pub dvel: Vec2,
}

// --- Tuning ----------------------------------------------------------------------------

// Fraction of each overlap pushed apart per substep. Higher separates faster but jitters
// more in tall piles.
const RELAXATION: f32 = 0.5;
// Fraction of the window height particles spawn in, measured from the top.
const SPAWN_HEIGHT: f32 = 0.6;
// Largest initial speed on each axis, in pixels per second.
const SPAWN_SPEED: f32 = 150.0;

// --- Random numbers --------------------------------------------------------------------

fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

// Uniform random value in [0, 1] for particle `i`, stream `k`.
fn rand(params: &SimParams, i: u32, k: u32) -> f32 {
    pcg(pcg(i.wrapping_mul(4).wrapping_add(k)) ^ params.seed) as f32 / 4294967295.0
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// --- Compute passes --------------------------------------------------------------------

#[spirv(compute(threads(64)))]
pub fn init(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] particles: &mut [Particle],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] corrections: &mut [Correction],
) {
    let i = id.x as usize;
    if i >= particles.len() {
        return;
    }

    let r = params.radius;
    let area = vec2(params.resolution.x, params.resolution.y * SPAWN_HEIGHT) - 2.0 * r;
    let pos =
        Vec2::splat(r) + vec2(rand(params, id.x, 0), rand(params, id.x, 1)) * area.max(Vec2::ZERO);
    let vel = (vec2(rand(params, id.x, 2), rand(params, id.x, 3)) * 2.0 - 1.0) * SPAWN_SPEED;

    particles[i] = Particle { pos, vel };
    corrections[i] = Correction {
        dpos: Vec2::ZERO,
        dvel: Vec2::ZERO,
    };
}

#[spirv(compute(threads(64)))]
pub fn collide(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] particles: &mut [Particle],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] corrections: &mut [Correction],
) {
    let n = particles.len();
    let i = id.x as usize;
    if i >= n {
        return;
    }

    let me = particles[i];
    let min_dist = 2.0 * params.radius;
    let mut dpos = Vec2::ZERO;
    let mut dvel = Vec2::ZERO;

    let mut j = 0usize;
    while j < n {
        if j != i {
            let other = particles[j];
            let d = me.pos - other.pos;
            let dist2 = d.dot(d);
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let normal = if dist > 1e-4 {
                    d / dist
                } else {
                    // Exactly on top of each other: pick a direction from the pair's
                    // indices, pointing opposite ways for the two particles.
                    let angle = i.min(j) as f32 * 2.399963;
                    let base = vec2(angle.cos(), angle.sin());
                    if i < j { base } else { -base }
                };

                // Push apart by our half of the overlap.
                dpos += normal * (min_dist - dist) * 0.5 * RELAXATION;

                // Equal-mass impulse along the contact normal, only if the pair is
                // approaching.
                let approach = (me.vel - other.vel).dot(normal);
                if approach < 0.0 {
                    dvel -= normal * approach * 0.5 * (1.0 + params.restitution);
                }
            }
        }
        j += 1;
    }

    corrections[i] = Correction { dpos, dvel };
}

#[spirv(compute(threads(64)))]
pub fn integrate(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] particles: &mut [Particle],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] corrections: &mut [Correction],
) {
    let i = id.x as usize;
    if i >= particles.len() {
        return;
    }

    let mut p = particles[i];
    let c = corrections[i];

    p.vel += c.dvel + vec2(0.0, params.gravity) * params.dt;
    p.pos += c.dpos + p.vel * params.dt;

    // Walls: clamp inside the window and reflect the velocity component into the wall.
    let lo = Vec2::splat(params.radius);
    let hi = (params.resolution - params.radius).max(lo);
    if p.pos.x < lo.x {
        p.pos.x = lo.x;
        p.vel.x = p.vel.x.abs() * params.restitution;
    }
    if p.pos.x > hi.x {
        p.pos.x = hi.x;
        p.vel.x = -p.vel.x.abs() * params.restitution;
    }
    if p.pos.y < lo.y {
        p.pos.y = lo.y;
        p.vel.y = p.vel.y.abs() * params.restitution;
    }
    if p.pos.y > hi.y {
        p.pos.y = hi.y;
        p.vel.y = -p.vel.y.abs() * params.restitution;
    }

    particles[i] = p;
}

// --- Drawing ---------------------------------------------------------------------------

#[spirv(fragment)]
pub fn draw(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] particles: &mut [Particle],
    output: &mut Vec4,
) {
    let frag = vec2(frag_coord.x, frag_coord.y);
    let r = params.radius;
    let mut brightness = 0.0f32;
    let n = particles.len();
    let mut i = 0usize;
    while i < n {
        let dist = frag.distance(particles[i].pos);
        // One-pixel antialiased edge.
        brightness = brightness.max(1.0 - smoothstep(r - 1.0, r, dist));
        i += 1;
    }
    *output = vec4(brightness, brightness, brightness, 1.0);
}
