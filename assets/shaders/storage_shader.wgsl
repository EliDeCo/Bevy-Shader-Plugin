struct ExampleUniform {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> u: ExampleUniform;

struct RedChannel   { values: array<f32, 64> }
struct GreenChannel { values: array<f32, 64> }
struct BlueChannel  { values: array<f32, 64> }

@group(1) @binding(0) var<storage, read> red:   RedChannel;
@group(2) @binding(0) var<storage, read> green: GreenChannel;
@group(3) @binding(0) var<storage, read> blue:  BlueChannel;

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coords.xy / u.resolution;
    let idx = min(u32(uv.x * 64.0), 63u);
    return vec4(red.values[idx], green.values[idx], blue.values[idx], 1.0);
}
