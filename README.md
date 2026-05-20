# bevy_fragment_shader

A Bevy library for rendering fullscreen fragment shaders. Handles the render graph wiring, pipeline creation, and uniform extraction so you only need to define your data and write WGSL.

## Bevy compatibility

| `bevy_fragment_shader` | Bevy |
|---|---|
| 0.1 | 0.18 |

## Usage

### 1. Add the dependency

```toml
[dependencies]
bevy_fragment_shader = { path = "..." }
```

### 2. Define your uniform struct

The struct is your shader's per-frame data. It must derive `Resource`, `ShaderType`, `Clone`, and `Default`, and its layout must match the WGSL struct exactly.

WebGPU requires uniform buffers to be a multiple of 16 bytes. Use a padding field (`_pad: f32`) if needed to reach the next multiple.

```rust
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

#[derive(Resource, ShaderType, Clone, Default)]
struct MyUniform {
    resolution: Vec2,  // 8 bytes
    time: f32,         // 4 bytes
    _pad: f32,         // 4 bytes — total: 16 bytes
}
```

### 3. Add the plugin

```rust
use bevy::prelude::*;
use bevy_fragment_shader::FullscreenFragmentPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<MyUniform>::new("shaders/my_shader.wgsl"))
        .init_resource::<MyUniform>()
        .add_systems(Startup, setup)
        .add_systems(Update, update_uniform)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Msaa::Off));
}
```

> **Note:** This plugin is not compatible with MSAA. Spawn your camera with `Msaa::Off`.

### 4. Update the uniform each frame

```rust
use bevy::{prelude::*, window::PrimaryWindow};

fn update_uniform(
    mut uniform: ResMut<MyUniform>,
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let Ok(window) = windows.single() else { return };
    uniform.resolution = Vec2::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
    );
    uniform.time = time.elapsed_secs();
}
```

### 5. Write the WGSL shader

Place your shader in `assets/shaders/my_shader.wgsl`. Group 0, binding 0 is always the uniform.

```wgsl
struct MyUniform {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> u: MyUniform;

@fragment
fn frag_main(@builtin(position) frag_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coords.xy / u.resolution;
    let r = 0.5 + 0.5 * sin(u.time + uv.x * 6.28);
    let g = 0.5 + 0.5 * sin(u.time * 0.7 + uv.y * 6.28);
    let b = 0.5 + 0.5 * sin(u.time * 0.5 + length(uv - 0.5) * 12.56);
    return vec4<f32>(r, g, b, 1.0);
}
```

---

## Extra bind groups (storage buffers, textures, etc.)

The library manages **group 0** (your uniform). You can add groups 1..n for storage buffers, textures, samplers, or any other data using two resources:

| Resource | Purpose | When to populate |
|---|---|---|
| `FragmentExtraLayouts` | `BindGroupLayoutDescriptor` for each extra group | Once, in `RenderStartup` **before** `FragmentSystems::InitPipeline` |
| `FragmentExtraBindGroups` | `BindGroup` for each extra group | Each frame, in `RenderSystems::PrepareBindGroups` |

### Storage buffer example

```rust
use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            Buffer, BufferDescriptor, BufferUsages, ShaderStages,
            binding_types::storage_buffer_read_only_sized,
        },
        renderer::RenderDevice,
    },
};
use bevy_fragment_shader::{FragmentExtraBindGroups, FragmentExtraLayouts, FragmentSystems};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<MyUniform>::new("shaders/my_shader.wgsl"))
        .init_resource::<MyUniform>();

    // Register render-world systems in the RenderApp
    let render_app = app.sub_app_mut(RenderApp);
    render_app
        .add_systems(
            RenderStartup,
            setup_storage_layout.before(FragmentSystems::InitPipeline),
        )
        .add_systems(
            Render,
            update_storage_bind_group.in_set(RenderSystems::PrepareBindGroups),
        );

    app.run();
}

fn setup_storage_layout(
    mut extra_layouts: ResMut<FragmentExtraLayouts>,
    render_device: Res<RenderDevice>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "my_storage_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (storage_buffer_read_only_sized(false, None),),
        ),
    );
    extra_layouts.0.push(layout);
}

fn update_storage_bind_group(
    mut extra_bind_groups: ResMut<FragmentExtraBindGroups>,
    render_device: Res<RenderDevice>,
    // ... your render-world resources holding the buffer
) {
    // Build or retrieve your BindGroup for group 1
    let bind_group: BindGroup = /* ... */;
    extra_bind_groups.0 = vec![bind_group];
}
```

The corresponding WGSL declares the buffer at group 1:

```wgsl
@group(0) @binding(0) var<uniform> u: MyUniform;
@group(1) @binding(0) var<storage, read> my_data: array<f32>;
```

### Texture example

```rust
use bevy::{
    prelude::*,
    render::render_resource::{
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, ShaderStages,
        binding_types::{sampler, texture_2d},
        SamplerBindingType, TextureSampleType,
    },
};
use bevy_fragment_shader::{FragmentExtraLayouts, FragmentSystems};

fn setup_texture_layout(mut extra_layouts: ResMut<FragmentExtraLayouts>) {
    let layout = BindGroupLayoutDescriptor::new(
        "my_texture_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    extra_layouts.0.push(layout);
}
```

```wgsl
@group(0) @binding(0) var<uniform> u: MyUniform;
@group(1) @binding(0) var my_texture: texture_2d<f32>;
@group(1) @binding(1) var my_sampler: sampler;
```

---

## Running the built-in example

```sh
cargo run --example shader_example
```

This renders an animated RGB gradient that exercises the full pipeline end-to-end.

---

## How it works

```
Main World                   Render World                 GPU
──────────────────────       ─────────────────────────    ──────────────────
MyUniform (Resource)    ───► MyUniform (extracted)   ───► UniformBuffer (group 0)
                             FullscreenPipeline<U>         CachedRenderPipeline
                             FullscreenBindGroup<U>         BindGroup 0
FragmentExtraBindGroups ───► bind groups 1..n         ───► BindGroups 1..n
                             FullscreenNode<U>::run         fullscreen triangle draw
```

The render node runs before `Node3d::StartMainPass` in the `Core3d` graph, so the shader output appears as the scene background.
