# bevy_fragment_shader_plugin

A Bevy plugin for rendering fullscreen fragment shaders. Handles render graph wiring, pipeline creation, and buffer management.

The plugin renders through Bevy's 3D pipeline, so it needs a `Camera3d`. It is not compatible with MSAA, make sure to disable it on all cameras:
```rust
commands.spawn((Camera3d::default(), Msaa::Off));
```

## Compatibility

| `bevy-fragment-shader-plugin` | Bevy | rust-gpu |
|---|---|---|
| 0.2 | 0.18 | 0.10.0-alpha.1 |
| 0.1 | 0.18 | — |

## Setup

Import everything you need in two lines:

```rust
use bevy::prelude::*;
use bevy_fragment_shader_plugin::prelude::*;
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

Any changes to the `MyUniform` resource will be reflected in the associated buffer on the next frame.

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

See [`examples/solar_system.rs`](examples/solar_system.rs) for a complete example combining uniform, storage, and array buffers.

Shaders can also be written in Rust instead of WGSL — see [Using rust-gpu shaders](#using-rust-gpu-shaders).

---

## Buffer registration

All three methods are on the `FragmentAppExt` trait (included in the prelude). The `group_index` and `binding_index` arguments map directly to `@group(n) @binding(n)` in WGSL.

- Call them after adding `DefaultPlugins`.
- Group indices must start at 0 with no gaps (e.g. 0, 1, 2 — not 1, 2 or 0, 2).
- WebGPU only guarantees 4 bind groups per pipeline, counting registered groups and manual extra groups together. Prefer putting several bindings in one group (different `binding_index` values under the same `group_index`) over giving each buffer its own group. Native backends usually allow more, but browsers and some mobile GPUs enforce the limit.

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

### Fixed-size array buffer

For fixed-length arrays that benefit from per-element updates (rather than a full resend each frame), use `register_array_buffer`:

```rust
struct Colors;
//                          <Tag, Type, Capacity>
app.register_array_buffer::<Colors, Vec4, 64>(1, 0, false);
```

Update elements each frame via `ArrayBufferChanges<Tag>` using `set`, `set_many`, or `set_all`. Only changed elements are uploaded, batched into contiguous `write_buffer` runs:

```rust
fn animate(mut changes: ResMut<ArrayBufferChanges<Colors>>, time: Res<Time>) {
    changes.set(0, Vec4::splat(time.elapsed_secs().sin())); // single element
    changes.set_many([(1, Vec4::ONE), (2, Vec4::ZERO)]);    // multiple elements
    changes.set_all(Vec4::ZERO);                            // every element
}
```

Values must be the `Type` given at registration (`Vec4` above). This isn't checked.

---

## Using rust-gpu shaders

Fragment shaders can be written in Rust with [rust-gpu](https://github.com/Rust-GPU/rust-gpu) instead of WGSL.

Enable the `spirv` feature:

```toml
bevy-fragment-shader-plugin = { version = "0.2", features = ["spirv"] }
```

Point the plugin at a `.spv`. Naming the entry point is only required if the module has more than one:

```rust
FullscreenFragmentPlugin::new("shaders/my_shader.spv")
    .with_entry_point("main_fs")
```

In the shader, `descriptor_set` is the group index and `binding` is the binding index, so they line up with the registration functions above:

```rust
#[spirv(fragment)]
pub fn main_fs(
    #[spirv(uniform, descriptor_set = 0, binding = 0)] u: &FrameUniform,
    output: &mut Vec4,
) { /* ... */ }
```

Not supported on web — leave the feature off for wasm builds.

### Buffer layout

Stick to `Vec2`, `Vec4`, `f32` and arrays of them. Avoid `Vec3`. Keep arrays out of uniform buffers unless the element is 16 bytes — use a storage buffer instead.

### Building the shader

See [`examples/solar_system_rust/`](examples/solar_system_rust/) for a complete project. The compiled shader is committed at `assets/shaders/solar_system_rust.spv`, so the example runs on stable. Rebuilding it after editing the shader needs `nightly-2026-05-22` with the `rust-src`, `rustc-dev` and `llvm-tools` components:

```sh
cd examples/solar_system_rust/shader-build
cargo build   # writes assets/shaders/solar_system_rust.spv
```

The first build takes a few minutes, later ones a few seconds.

Keep the shader crate inside `shader-build/`. As a sibling it builds with the wrong toolchain and fails with `the -Z flag is only accepted on the nightly channel`.

`WARN naga::front::spv: Unknown decoration Block` on startup is harmless.

---

## Running the examples

```sh
cargo run --example solar_system        # orbital simulation using all three buffer types
cargo run --example solar_system_rust   # the same, with the shader written in Rust (native only)
```
