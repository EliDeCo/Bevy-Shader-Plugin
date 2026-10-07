// Particles — a compute simulation drawn by a fullscreen fragment shader.
//   init       (compute, once):         scatters particles across the top of the window
//   collide    (compute, every substep): works out each particle's collision response
//   integrate  (compute, every substep): applies gravity + collisions, moves, bounces off walls
//   draw       (fragment):               white circles on black
//
// Everything is in physical pixels with y pointing down, matching @builtin(position).

struct SimParams {
    resolution: vec2<f32>,
    gravity: f32,
    dt: f32,
    radius: f32,
    restitution: f32,
    seed: u32,
}

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
}

// Collision response for one particle, computed by `collide` and applied by `integrate`.
// Kept separate so `collide` only reads `particles` and each thread writes only its own
// slot — no thread ever reads a value another thread is writing.
struct Correction {
    dpos: vec2<f32>,
    dvel: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
// GPU-owned buffers are always read_write, and every entry point in every shader that
// uses a binding must declare the same access mode.
@group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(1) @binding(1) var<storage, read_write> corrections: array<Correction>;

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
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform random value in [0, 1] for particle `i`, stream `k`.
fn rand(i: u32, k: u32) -> f32 {
    return f32(pcg(pcg(i * 4u + k) ^ params.seed)) / 4294967295.0;
}

// --- Compute passes --------------------------------------------------------------------

@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&particles) { return; }

    let r = params.radius;
    let area = vec2(params.resolution.x, params.resolution.y * SPAWN_HEIGHT) - 2.0 * r;
    let pos = vec2(r) + vec2(rand(i, 0u), rand(i, 1u)) * max(area, vec2(0.0));
    let vel = (vec2(rand(i, 2u), rand(i, 3u)) * 2.0 - 1.0) * SPAWN_SPEED;

    particles[i] = Particle(pos, vel);
    corrections[i] = Correction(vec2(0.0), vec2(0.0));
}

@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = arrayLength(&particles);
    let i = id.x;
    if i >= n { return; }

    let me = particles[i];
    let min_dist = 2.0 * params.radius;
    var dpos = vec2(0.0);
    var dvel = vec2(0.0);

    for (var j = 0u; j < n; j++) {
        if j == i { continue; }
        let other = particles[j];
        let d = me.pos - other.pos;
        let dist2 = dot(d, d);
        if dist2 >= min_dist * min_dist { continue; }

        let dist = sqrt(dist2);
        var normal: vec2<f32>;
        if dist > 1e-4 {
            normal = d / dist;
        } else {
            // Exactly on top of each other: pick a direction from the pair's indices,
            // pointing opposite ways for the two particles.
            let angle = f32(min(i, j)) * 2.399963;
            let base = vec2(cos(angle), sin(angle));
            normal = select(-base, base, i < j);
        }

        // Push apart by our half of the overlap.
        dpos += normal * (min_dist - dist) * 0.5 * RELAXATION;

        // Equal-mass impulse along the contact normal, only if the pair is approaching.
        let approach = dot(me.vel - other.vel, normal);
        if approach < 0.0 {
            dvel -= normal * approach * 0.5 * (1.0 + params.restitution);
        }
    }

    corrections[i] = Correction(dpos, dvel);
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&particles) { return; }

    var p = particles[i];
    let c = corrections[i];

    p.vel += c.dvel + vec2(0.0, params.gravity) * params.dt;
    p.pos += c.dpos + p.vel * params.dt;

    // Walls: clamp inside the window and reflect the velocity component into the wall.
    let lo = vec2(params.radius);
    let hi = max(params.resolution - params.radius, lo);
    if p.pos.x < lo.x { p.pos.x = lo.x; p.vel.x = abs(p.vel.x) * params.restitution; }
    if p.pos.x > hi.x { p.pos.x = hi.x; p.vel.x = -abs(p.vel.x) * params.restitution; }
    if p.pos.y < lo.y { p.pos.y = lo.y; p.vel.y = abs(p.vel.y) * params.restitution; }
    if p.pos.y > hi.y { p.pos.y = hi.y; p.vel.y = -abs(p.vel.y) * params.restitution; }

    particles[i] = p;
}

// --- Drawing ---------------------------------------------------------------------------

@fragment
fn draw(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let r = params.radius;
    var brightness = 0.0;
    let n = arrayLength(&particles);
    for (var i = 0u; i < n; i++) {
        let dist = distance(frag_coord.xy, particles[i].pos);
        // One-pixel antialiased edge.
        brightness = max(brightness, 1.0 - smoothstep(r - 1.0, r, dist));
    }
    return vec4(vec3(brightness), 1.0);
}
