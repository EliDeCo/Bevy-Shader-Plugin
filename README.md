# bevy_fragment_shader

A Bevy plugin for rendering fullscreen fragment shaders. Handles render graph wiring, pipeline creation, and buffer management.

The plugin is not compatible with MSAA, make sure to disable it on all cameras
```rust
commands.spawn((Camera3d::default(), Msaa::Off));
```

## Bevy compatibility

| `bevy_fragment_shader` | Bevy |
|---|---|
| 0.1 | 0.18 |

## Setup

```toml
[dependencies]
bevy_fragment_shader = { path = "..." }
```

Import everything you need in two lines:

```rust
use bevy::prelude::*;
use bevy_fragment_shader::prelude::*;
```

## Quick start

### 1. Define your Uniform struct

```rust
#[derive(Resource, ShaderType, Clone, Default)]
struct MyUniform {
    resolution: Vec2,
    time: f32,
}
```

Padding to 16 bytes is handled automatically — no `_pad` fields needed.

### 2. Register the plugin and buffer

```rust
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(FullscreenFragmentPlugin::new("shaders/my_shader.wgsl"))
    .register_uniform_buffer::<MyUniform>(0, 0)
    .init_resource::<MyUniform>()
    // ...
```

Any changes to the MyUniform resource will be reflected in the associated buffer on the next frame.

### 3. Write the shader

```wgsl
struct MyUniform { resolution: vec2<f32>, time: f32 }
@group(0) @binding(0) var<uniform> u: MyUniform;

@fragment
fn frag_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / u.resolution;
    return vec4(uv, 0.5 + 0.5 * sin(u.time), 1.0);
}
```

See [`examples/simple_shader.rs`](examples/simple_shader.rs) for the complete setup.

---

## Buffer registration

All three methods are on the `FragmentAppExt` trait (included in the prelude). The `group_index` and `binding_index` arguments map directly to `@group(n) @binding(n)` in WGSL.

### Uniform buffer

```rust
app.register_uniform_buffer::<MyUniform>(0, 0);
```

The resource is extracted from the main world and uploaded every frame. WGSL: `var<uniform>`.

### Storage buffer

```rust
app.register_storage_buffer::<MyData>(1, 0, false); // false = read-only, true = read_write
```

Multiple bindings sharing the same `group_index` are packed into one bind group:

```rust
app.register_storage_buffer::<Red>(1, 0, false)
   .register_storage_buffer::<Green>(1, 1, false)
   .register_storage_buffer::<Blue>(1, 2, false);
```

See [`examples/storage_shader.rs`](examples/storage_shader.rs).

### Fixed-size array buffer

For fixed length arrays that benefit for per element updates (rather than full resend each frame), use the register_array_buffer function

```rust
struct Colors;
//                          <Name, Type, Capacity>
app.register_array_buffer::<Colors, Vec4, 64>(1, 0, false);
```

Update elements each frame via `ArrayBufferChanges<Tag>` and the `set`,`set_many`, and `set_all` functions, supplying the index to change and the value to assign for each. Only changed elements are uploaded, batched into contiguous `write_buffer` runs:

```rust
fn animate(mut changes: ResMut<ArrayBufferChanges<Colors>>, time: Res<Time>) {
    changes.set(0, Vec4::splat(time.elapsed_secs().sin()));
    changes.set_many([(1, Vec4::ONE), (2, Vec4::ZERO)]);
}
```

See [`examples/array_shader.rs`](examples/array_shader.rs).

---

## Running the examples

```sh
cargo run --example simple_shader   # animated gradient via uniform buffer
cargo run --example storage_shader  # per-band RGB via storage buffers
cargo run --example array_shader    # per-element color and brightness arrays
```
