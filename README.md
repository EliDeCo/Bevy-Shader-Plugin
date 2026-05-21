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

The library manages **group 0** (your uniform). You can add groups 1..n for storage buffers, textures, samplers, or any other data.

Use `FragmentAppExt::register_extra_bind_group` to register both the layout-setup and per-frame bind-group systems in a single call — ordering is handled for you automatically.

```rust
use bevy_fragment_shader::FragmentAppExt;

app.register_extra_bind_group(setup_my_layout, prepare_my_bind_group);
```

| System argument | Schedule | Purpose |
|---|---|---|
| `setup_my_layout` | `RenderStartup` (before pipeline init) | Push layout descriptors into `FragmentExtraLayouts` |
| `prepare_my_bind_group` | `RenderSystems::PrepareBindGroups` | Build and push `BindGroup` values into `FragmentExtraBindGroups` |

Both systems run in the render world. If you need a main-world resource available there, register an extraction system separately:

```rust
app.sub_app_mut(RenderApp).add_systems(ExtractSchedule, my_extract_system);
```

### Layout helpers — `FragmentExtraLayouts`

Use the built-in methods instead of constructing `BindGroupLayoutDescriptor` by hand:

```rust
fn setup_my_layout(mut extra_layouts: ResMut<FragmentExtraLayouts>) {
    // Filterable 2D texture at binding 0 + filtering sampler at binding 1:
    extra_layouts.texture_2d_and_sampler("my_texture_layout");

    // Read-only storage buffer at binding 0:
    extra_layouts.storage_buffer_read_only("my_storage_layout");

    // Read-write storage buffer at binding 0:
    extra_layouts.storage_buffer_read_write("my_rw_layout");

    // Arbitrary layout via the fragment_layout! macro:
    extra_layouts.push(fragment_layout!(
        "custom_layout",
        texture_2d(TextureSampleType::Float { filterable: true }),
        sampler(SamplerBindingType::Filtering),
        storage_buffer_read_only_sized(false, None),
    ));
}
```

### Bind group helpers — `FragmentExtraBindGroups`

The compiled layouts for your extra groups are stored on `FullscreenPipeline::extra_layouts` (index 0 = GPU group 1) so you don't need to go back to `PipelineCache`.

```rust
// One-shot helper for a GpuImage (texture view + sampler):
extra_bind_groups
    .clear()
    .push_gpu_image("label", &pipeline, &render_device, 0, gpu_image);

// Fluent builder for custom resources:
let bg = FragmentBindGroupBuilder::new(&pipeline.extra_layouts[0], &render_device)
    .label("my_bg")
    .texture_view(&my_view)
    .sampler(&my_sampler)
    .buffer(&my_buffer)
    .build();
extra_bind_groups.clear().push(bg);
```

---

### Storage buffer example

```rust
use bevy::{
    prelude::*,
    render::{
        render_resource::{ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
    },
};
use bevy_fragment_shader::{
    FragmentAppExt, FragmentBindGroupBuilder, FragmentExtraBindGroups,
    FragmentExtraLayouts, FullscreenFragmentPlugin, FullscreenPipeline,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<MyUniform>::new("shaders/my_shader.wgsl"))
        .init_resource::<MyUniform>()
        .register_extra_bind_group(setup_storage_layout, prepare_storage_bind_group)
        .add_systems(Startup, setup)
        .add_systems(Update, update_uniform)
        .run();
}

// Render-world resource holding the GPU storage buffer
#[derive(Resource)]
struct MyStorageBuffer(StorageBuffer<Vec<Vec4>>);

fn setup_storage_layout(
    mut extra_layouts: ResMut<FragmentExtraLayouts>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    extra_layouts.storage_buffer_read_only("my_storage_layout");

    // Create the initial buffer in RenderStartup so it's ready on frame 1.
    let mut buf = StorageBuffer::default();
    buf.set(vec![Vec4::ONE; 64]);
    buf.write_buffer(&render_device, &render_queue);
    commands.insert_resource(MyStorageBuffer(buf));
}

fn prepare_storage_bind_group(
    mut extra_bind_groups: ResMut<FragmentExtraBindGroups>,
    render_device: Res<RenderDevice>,
    pipeline: Option<Res<FullscreenPipeline<MyUniform>>>,
    storage: Option<Res<MyStorageBuffer>>,
) {
    let (Some(pipeline), Some(storage)) = (pipeline, storage) else { return };
    let Some(raw_buf) = storage.0.buffer() else { return };

    let bg = FragmentBindGroupBuilder::new(&pipeline.extra_layouts[0], &render_device)
        .label("storage_bg")
        .buffer(raw_buf)
        .build();

    extra_bind_groups.clear().push(bg);
}
```

The corresponding WGSL declares the buffer at group 1:

```wgsl
@group(0) @binding(0) var<uniform> u: MyUniform;
@group(1) @binding(0) var<storage, read> my_data: array<vec4<f32>>;
```

### Texture example

```rust
use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, RenderApp,
        render_asset::RenderAssets,
        render_resource::ShaderType,
        renderer::RenderDevice,
        texture::GpuImage,
    },
    window::PrimaryWindow,
};
use bevy_fragment_shader::{
    FragmentAppExt, FragmentExtraBindGroups, FragmentExtraLayouts,
    FullscreenFragmentPlugin, FullscreenPipeline,
};

#[derive(Resource, Clone)]
struct SceneTexture(Handle<Image>);

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<MyUniform>::new("shaders/my_shader.wgsl"))
        .init_resource::<MyUniform>()
        .register_extra_bind_group(setup_texture_layout, prepare_texture_bind_group)
        .add_systems(Startup, setup)
        .add_systems(Update, update_uniform);

    app.sub_app_mut(RenderApp)
        .add_systems(ExtractSchedule, extract_scene_texture);

    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((Camera3d::default(), Msaa::Off));
    commands.insert_resource(SceneTexture(asset_server.load("textures/my_texture.png")));
}

fn extract_scene_texture(
    mut commands: Commands,
    texture: Extract<Option<Res<SceneTexture>>>,
) {
    if let Some(t) = texture.as_deref() {
        commands.insert_resource(SceneTexture(t.0.clone()));
    }
}

fn setup_texture_layout(mut extra_layouts: ResMut<FragmentExtraLayouts>) {
    extra_layouts.texture_2d_and_sampler("scene_texture_layout");
}

fn prepare_texture_bind_group(
    mut extra_bind_groups: ResMut<FragmentExtraBindGroups>,
    render_device: Res<RenderDevice>,
    pipeline: Option<Res<FullscreenPipeline<MyUniform>>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    texture: Option<Res<SceneTexture>>,
) {
    let (Some(pipeline), Some(texture)) = (pipeline, texture) else { return };
    let Some(gpu_image) = gpu_images.get(&texture.0) else { return };

    extra_bind_groups
        .clear()
        .push_gpu_image("scene_texture_bg", &pipeline, &render_device, 0, gpu_image);
}
```

The corresponding WGSL declares the texture and sampler at group 1:

```wgsl
@group(0) @binding(0) var<uniform> u: MyUniform;
@group(1) @binding(0) var my_texture: texture_2d<f32>;
@group(1) @binding(1) var my_sampler: sampler;
```

---

## Running the examples

```sh
# Animated RGB gradient — no extra bind groups
cargo run --example simple_shader

# Bevy logo sampled via a texture bind group, tinted with a time-varying color
cargo run --example texture_shader

# 64-entry rainbow color palette passed via a storage buffer, displayed as vertical bands
cargo run --example storage_shader
```

---