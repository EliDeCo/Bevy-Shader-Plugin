// Solar system — demonstrates all three buffer types from bevy_fragment_shader:
//   group(0) uniform:  resolution (no time; all motion is computed on the CPU)
//   group(1) storage:  planet UV positions, fully re-uploaded every frame
//   group(2) array:    per-planet RGBA colors, updated only when a planet completes an orbit

struct FrameUniform {
    resolution: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: FrameUniform;
@group(1) @binding(0) var<storage, read> positions: array<vec2<f32>, 8>;
@group(2) @binding(0) var<storage, read> colors: array<vec4<f32>, 8>;

const STAR_CENTER: vec2<f32> = vec2<f32>(0.5, 0.5);
const STAR_RADIUS: f32 = 0.03;
const STAR_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.4);

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// Returns brightness of a background star at the given pixel coordinate.
// Works in pixel space so cells are square regardless of window aspect ratio.
fn star_field(pixel: vec2<f32>) -> f32 {
    let cell = floor(pixel / 30.0);
    let local = fract(pixel / 30.0);
    if hash(cell) > 0.18 { return 0.0; }
    let pos = vec2<f32>(hash(cell + vec2<f32>(7.3, 2.1)), hash(cell + vec2<f32>(3.7, 8.5)));
    let brightness = 0.5 + 0.5 * hash(cell + vec2<f32>(5.0, 5.0));
    return brightness * step(distance(local, pos), 0.06);
}

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coords.xy / u.resolution;

    // Star is always drawn on top.
    if distance(uv, STAR_CENTER) < STAR_RADIUS {
        return vec4<f32>(STAR_COLOR, 1.0);
    }

    for (var i = 0u; i < 8u; i++) {
        let pos = positions[i];
        // Planet display radius scales with orbital distance: outer planets are larger.
        let orbit_dist = distance(pos, STAR_CENTER);
        let planet_radius = 0.006 + 0.006 * orbit_dist;
        if distance(uv, pos) < planet_radius {
            return vec4<f32>(colors[i].rgb, 1.0);
        }
    }

    let bg = star_field(frag_coords.xy);
    return vec4<f32>(vec3<f32>(bg), 1.0);
}
