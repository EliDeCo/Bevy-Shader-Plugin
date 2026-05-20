// Fullscreen animated color effect demonstrating the bevy_fragment_shader library.


struct ExampleUniform {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> u: ExampleUniform;

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coords.xy / u.resolution;
    let t  = u.time;

    // Three out-of-phase sine waves — one per colour channel.
    let r = 0.5 + 0.5 * sin(t          + uv.x * 6.2832);
    let g = 0.5 + 0.5 * sin(t * 0.7    + uv.y * 6.2832 + 2.094);
    let b = 0.5 + 0.5 * sin(t * 0.5    + length(uv - vec2<f32>(0.5)) * 12.566 + 4.189);

    return vec4<f32>(r, g, b, 1.0);
}
