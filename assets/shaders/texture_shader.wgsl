struct ExampleUniform {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> u: ExampleUniform;

@group(1) @binding(0)
var scene_texture: texture_2d<f32>;

@group(1) @binding(1)
var scene_sampler: sampler;

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(scene_texture));

    // Map fragment position to UV so the texture is centered at its native pixel size.
    // Pixels outside the texture area return a dark background.
    let uv = (frag_coords.xy - (u.resolution - tex_size) * 0.5) / tex_size;
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return vec4<f32>(0.05, 0.05, 0.05, 1.0);
    }

    let tex_color = textureSample(scene_texture, scene_sampler, uv);
    let tint = vec3<f32>(
        0.5 + 0.5 * sin(u.time),
        0.5 + 0.5 * cos(u.time * 0.7),
        0.5 + 0.5 * sin(u.time * 0.5 + 1.0),
    );
    return vec4<f32>(tex_color.rgb * tint, 1.0);
}
