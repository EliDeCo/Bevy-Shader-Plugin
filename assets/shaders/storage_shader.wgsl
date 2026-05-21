struct ExampleUniform {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> u: ExampleUniform;

@group(1) @binding(0)
var<storage, read> particles: array<vec4<f32>>;

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coords.xy / u.resolution;

    // Map the horizontal UV coordinate to a particle index, creating vertical
    // color bands — one band per entry in the storage buffer.
    let count = arrayLength(&particles);
    let idx = min(u32(uv.x * f32(count)), count - 1u);
    return particles[idx];
}
