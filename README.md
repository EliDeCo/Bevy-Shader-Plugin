# bevy_shader_plugin

A Bevy plugin for fullscreen fragment shaders and the compute shaders that feed them. Handles render scheduling, pipeline creation, and buffer management.

The plugin renders through Bevy's 3D pipeline, so it needs a `Camera3d`. It is not compatible with MSAA, make sure to disable it on all cameras:
```rust
commands.spawn((Camera3d::default(), Msaa::Off));
```

## Compatibility

| `bevy-shader-plugin` | Bevy | rust-gpu |
|---|---|---|
| 0.3 | 0.19 | 0.10.0-alpha.1 |
| 0.2 | 0.18 | 0.10.0-alpha.1 |
| 0.1 | 0.18 | — |

Versions 0.1 and 0.2 were published as `bevy-fragment-shader-plugin`.

## Setup

Import everything you need in two lines:

```rust
use bevy::prelude::*;
use bevy_shader_plugin::prelude::*;
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

To run a simulation on the GPU and draw it, see [Compute shaders](#compute-shaders).

Shaders can also be written in Rust instead of WGSL — see [Using rust-gpu shaders](#using-rust-gpu-shaders).

---

## Buffer registration

All registration methods are on the `ShaderAppExt` trait (included in the prelude). The `group_index` and `binding_index` arguments map directly to `@group(n) @binding(n)` in WGSL. Every registered buffer is visible to both the fragment shader and the compute shader.

- Call them after adding `DefaultPlugins`.
- Group indices must start at 0 with no gaps (e.g. 0, 1, 2 — not 1, 2 or 0, 2).
- Declare each binding with the same access mode (`read` or `read_write`) in every shader that uses it. wgpu rejects a shader that declares `read` for a binding registered as `read_write`.
- WebGPU only guarantees 4 bind groups per pipeline, counting registered groups and manual extra groups together. Prefer putting several bindings in one group (different `binding_index` values under the same `group_index`) over giving each buffer its own group. Native backends usually allow more, but browsers and some mobile GPUs enforce the limit.

### Uniform buffer

```rust
app.register_uniform_buffer::<MyUniform>(0, 0);
```

The resource is extracted from the main world and uploaded whenever it changes. WGSL: `var<uniform>`.

### Storage buffer

```rust
app.register_storage_buffer::<MyData>(1, 0, false); // false = read-only, true = read_write
```

Like uniforms, the resource is uploaded whenever it changes, into a buffer that persists between frames. With `read_write`, anything a shader writes stays until the resource next changes, then the CPU upload replaces the whole buffer. For data only the GPU updates, use a [GPU buffer](#gpu-buffer) instead.

Multiple bindings sharing the same `group_index` are packed into one bind group:

```rust
app.register_storage_buffer::<Red>(1, 0, false)
   .register_storage_buffer::<Green>(1, 1, false)
   .register_storage_buffer::<Blue>(1, 2, false);
```

### Fixed-size array buffer

For fixed-length arrays that benefit from per-element updates (rather than resending the whole array whenever one element changes), use `register_array_buffer`:

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

### GPU buffer

For data that lives on the GPU, such as simulation state that compute passes update every frame, use `register_gpu_buffer`:

```rust
#[derive(ShaderType, Clone, Copy, Default)]
struct Particle { pos: Vec2, vel: Vec2 }
struct Particles;
//                        <Tag, Type>
app.register_gpu_buffer::<Particles, Particle>(1, 0, 1024); // 1024 elements
```

```wgsl
@group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
```

The buffer starts filled with `Type::default()` and is never uploaded from the CPU again, so whatever shaders write persists across frames. It is always `read_write`. The tag names the buffer and sizes compute dispatches (see [`Workgroups::over`](#dispatch-sizes)).

### Reading buffers back

Storage, array and GPU buffers can be copied back to the CPU, e.g. to read results a shader wrote. Enable readback after registering the buffer, using the same tag and type:

```rust
app.register_gpu_buffer::<Particles, Particle>(1, 0, 1024)
   .read_back_gpu_buffer::<Particles, Particle>(ReadbackMode::OnRequest);     // → GpuReadback<Particles, Particle>

app.register_array_buffer::<Colors, Vec4, 64>(2, 0, true)
   .read_back_array_buffer::<Colors, Vec4>(ReadbackMode::OnRequest);         // → GpuReadback<Colors, Vec4>

app.register_storage_buffer::<Stats>(1, 1, true)
   .read_back_storage_buffer::<Stats>(ReadbackMode::OnRequest);              // → StorageReadback<Stats>
```

Array and GPU buffers read back as a slice of elements (`latest()` returns `&[Type]`). A storage buffer reads back as the resource type it was registered with (`latest()` returns `&Stats`), which can itself contain arrays.

With `ReadbackMode::OnRequest`, call `request()` on the readback resource to ask for a copy. The resource only changes when new data comes in:

```rust
fn request(readback: Res<GpuReadback<Particles, Particle>>) {
    readback.request();
}

fn print(readback: Res<GpuReadback<Particles, Particle>>) {
    if !readback.is_changed() { return; }
    let Some(particles) = readback.latest() else { return }; // &[Particle]
    info!("first particle at {}", particles[0].pos);
}
```

With `ReadbackMode::EveryFrame`, you don't need to call `request()` or check `is_changed()`. Just read `latest()`, which is replaced with a new copy about once per frame.

- Results arrive 1–3 frames after the copy, and show the buffer as it was at the end of that frame. Use them for UI, stats or logging, not for logic that needs the exact current GPU state.
- Every readback copies the whole buffer. That's negligible for thousands of elements, but `EveryFrame` on very large buffers has a real cost.
- If a storage type is registered at several bindings, the first storage registration is read. Uniform buffers can't be read back.

---

## Compute shaders

`ComputeShaderPlugin` runs `@compute` entry points once per frame, before any camera renders. Compute and fragment shaders share every registered buffer, so a compute pass can update a GPU buffer and the fragment shader can draw it in the same frame. Both plugins can load the same `.wgsl` file.

```rust
struct Particles;

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins((
        ComputeShaderPlugin::new("shaders/particles.wgsl")
            .init_pass("init", Workgroups::over::<Particles>())    // once, when ready
            .pass("update", Workgroups::over::<Particles>()),      // every frame
        FullscreenFragmentPlugin::new("shaders/particles.wgsl"),
    ))
    .register_gpu_buffer::<Particles, Particle>(0, 0, 1024)
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn((Camera3d::default(), Msaa::Off));
    })
    .run();
```

```wgsl
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&particles) { return; }
    // ...
}
```

- `pass` runs every frame. Passes run in the order they're added, and each one sees what the passes before it wrote. The same entry point can be added several times, e.g. for simulation substeps.
- `init_pass` runs once, on the first frame everything is ready. Every-frame passes wait until all init passes have run. Run one again with `ComputePasses`:

  ```rust
  fn reset(keys: Res<ButtonInput<KeyCode>>, mut passes: ResMut<ComputePasses>) {
      if keys.just_pressed(KeyCode::KeyR) {
          passes.rerun("init");
      }
  }
  ```
- Only one `ComputeShaderPlugin` is supported per app.
- To order your own render systems around this crate's GPU work, use the `ShaderPassSystems` sets (`Compute`, `Fullscreen`, `ReadbackCopy`).

### Dispatch sizes

`Workgroups` says how much work a pass covers. The plugin reads each entry point's `@workgroup_size` from the WGSL source and works out the workgroup count, so the size is only written in the shader.

| | Threads |
|---|---|
| `Workgroups::over::<Tag>()` | One per element of a GPU buffer, along x. Use `global_invocation_id.x` as the index and a 1D `@workgroup_size(N)`. |
| `Workgroups::window()` | One per physical pixel of the primary window, on x and y. |
| `Workgroups::threads(UVec3)` | An explicit thread count. |
| `Workgroups::exact(UVec3)` | Raw workgroup counts, with no reflection. |

Reflection only supports plain WGSL. Shaders that use Bevy's `#import`, or a `@workgroup_size` built from `override` constants, can't be reflected; use `Workgroups::exact` for those passes.

See [`examples/particles.rs`](examples/particles.rs) for a complete simulation: gravity and collisions computed on the GPU, drawn by the fragment shader.

---

## Using rust-gpu shaders

Fragment shaders can be written in Rust with [rust-gpu](https://github.com/Rust-GPU/rust-gpu) instead of WGSL.

Enable the `spirv` feature:

```toml
bevy-shader-plugin = { version = "0.3", features = ["spirv"] }
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
cargo run --example solar_system        # orbital simulation using all three CPU buffer types, plus a compute pass read back every frame
cargo run --example particles           # compute-shader particle simulation with CPU readback (press R to respawn)
cargo run --example solar_system_rust   # the orbital simulation (without compute), with the shader written in Rust (native only)
```
