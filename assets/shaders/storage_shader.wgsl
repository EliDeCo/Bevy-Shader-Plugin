struct ExampleUniform {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> u: ExampleUniform;

struct ColorData {
    colors: array<vec4<f32>, 64>,
}

@group(1) @binding(0)
var<storage, read> color_data: ColorData;

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coords.xy / u.resolution;

    // Map the horizontal UV coordinate to a color index, creating vertical
    // color bands — one band per entry in the storage buffer.
    let idx = min(u32(uv.x * 64.0), 63u);
    return color_data.colors[idx];
}
