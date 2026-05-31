struct FrameUniform {
    resolution: vec2<f32>,
    time: f32,
}

@group(0) @binding(0) var<uniform> u: FrameUniform;

// colors[i]:       hue-cycling color for color columns; zero elsewhere.
// brightnesses[i]: greyscale brightness for brightness columns; zero elsewhere.
// The shader adds both contributions so each set is fully independent.
@group(1) @binding(0) var<storage, read> colors:       array<vec4<f32>, 64>;
@group(1) @binding(1) var<storage, read> brightnesses: array<vec4<f32>, 64>;

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coords.xy / u.resolution;
    let idx = min(u32(uv.x * 64.0), 63u);

    // Color columns contribute their hue; brightness columns contribute grey.
    // Columns that are neither stay black (both buffers default to zero).
    let out = saturate(colors[idx].rgb + brightnesses[idx].rgb);
    return vec4<f32>(out, 1.0);
}
