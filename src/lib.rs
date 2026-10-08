mod auto_array;
mod bindings;
mod compute;
mod cpu_buffer;
mod extra_bind_group;
mod gpu_buffer;
mod node;
mod pipeline;
mod readback;

use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        render_resource::{
            BindGroup, BindGroupLayoutDescriptor, BindGroupLayoutEntries, ShaderStages, ShaderType,
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        },
    },
};
use encase::{
    ShaderSize,
    internal::{CreateFrom, WriteInto},
};

pub use auto_array::{ArrayBufferChanges, ArrayBufferState};
pub use bindings::{
    AutoBufferBindGroups, AutoBufferCompiledLayouts, AutoBufferKind, AutoBufferLayoutDescriptors,
    AutoBufferLayouts, BindingTable, BoundResource,
};
pub use compute::{ComputePasses, ComputeShaderPlugin, Workgroups};
pub use extra_bind_group::FragmentBindGroupBuilder;
pub use gpu_buffer::GpuBuffer;
pub use node::fullscreen_pass;
pub use pipeline::{FullscreenPipeline, FullscreenPipelineConfig};
pub use readback::{GpuReadback, ReadbackMode, StorageReadback};

pub mod prelude {
    pub use crate::{
        ArrayBufferChanges, ComputePasses, ComputeShaderPlugin, FullscreenFragmentPlugin,
        GpuBuffer, GpuReadback, ReadbackMode, ShaderAppExt, StorageReadback, Workgroups,
    };
    pub use bevy::render::render_resource::ShaderType;
    pub use bevy::window::PrimaryWindow;
}

// ---------------------------------------------------------------------------
// Re-exports for the fragment_layout! macro
// ---------------------------------------------------------------------------

#[doc(hidden)]
pub mod __private {
    pub use bevy::render::render_resource::{
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, ShaderStages,
    };
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// System sets for the binding machinery shared by the fragment and compute plugins.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShaderSystems {
    /// Runs in `RenderStartup`. Compiles the bind group layouts of all registered buffers.
    /// Pipelines are created after this set.
    CompileLayouts,
    /// Runs in `Render`, inside [`RenderSystems::PrepareBindGroups`]. Uploads changed
    /// buffers and records their handles in the [`BindingTable`].
    PrepareBindings,
    /// Runs in `Render`, inside [`RenderSystems::PrepareBindGroups`], after
    /// [`PrepareBindings`](Self::PrepareBindings). Rebuilds bind groups whose bindings
    /// changed.
    FinalizeBindGroups,
}

/// System set label for the pipeline initialisation system so users can order
/// their static-resource init systems before it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FragmentSystems {
    /// Runs in `RenderStartup`. All user systems that populate
    /// `FragmentExtraLayouts` must be ordered before this set.
    InitPipeline,
}

/// Bind group layout descriptors for the manual extra bind groups.
///
/// Populate this resource in a layout system registered in `RenderStartup`
/// before [`FragmentSystems::InitPipeline`].
#[derive(Resource, Default)]
pub struct FragmentExtraLayouts(pub Vec<BindGroupLayoutDescriptor>);

impl FragmentExtraLayouts {
    /// Push an arbitrary [`BindGroupLayoutDescriptor`].
    pub fn push(&mut self, desc: BindGroupLayoutDescriptor) -> &mut Self {
        self.0.push(desc);
        self
    }

    /// Add a group with a single read-only storage buffer at binding 0.
    pub fn storage_buffer_read_only(&mut self, label: &'static str) -> &mut Self {
        self.0.push(BindGroupLayoutDescriptor::new(
            label,
            &BindGroupLayoutEntries::single(
                ShaderStages::FRAGMENT,
                storage_buffer_read_only_sized(false, None),
            ),
        ));
        self
    }

    /// Add a group with a single read-write storage buffer at binding 0.
    pub fn storage_buffer_read_write(&mut self, label: &'static str) -> &mut Self {
        self.0.push(BindGroupLayoutDescriptor::new(
            label,
            &BindGroupLayoutEntries::single(
                ShaderStages::FRAGMENT,
                storage_buffer_sized(false, None),
            ),
        ));
        self
    }
}

/// Per-frame bind groups for manual extra groups.
///
/// Call [`clear`](Self::clear) at the start of your `PrepareBindGroups` system,
/// then use [`push`](Self::push) or [`FragmentBindGroupBuilder`] to populate each
/// group in order.
#[derive(Resource, Default)]
pub struct FragmentExtraBindGroups(pub Vec<BindGroup>);

impl FragmentExtraBindGroups {
    /// Push a pre-built bind group.
    pub fn push(&mut self, bind_group: BindGroup) -> &mut Self {
        self.0.push(bind_group);
        self
    }

    /// Remove all bind groups. Call at the start of your `PrepareBindGroups` system.
    pub fn clear(&mut self) -> &mut Self {
        self.0.clear();
        self
    }
}

/// System sets for the GPU passes this crate records, so your own render systems can be
/// ordered around them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShaderPassSystems {
    /// The [`ComputeShaderPlugin`] passes. Runs in the root
    /// [`RenderGraph`] schedule, before any camera
    /// renders.
    Compute,
    /// The [`FullscreenFragmentPlugin`] draw. Runs in each 3D camera's
    /// [`Core3d`] schedule, after the prepass and before the main pass.
    Fullscreen,
    /// Buffer copies for readback. Runs in the root
    /// [`RenderGraph`] schedule, after every camera
    /// has rendered.
    ReadbackCopy,
}

// ---------------------------------------------------------------------------
// App extension trait
// ---------------------------------------------------------------------------

/// Extension methods on [`App`] for registering buffers shared by the fullscreen
/// fragment shader and the compute shader.
///
/// Every registered buffer is visible to both stages at the same `@group`/`@binding`.
/// Declare each binding with the same access mode (`read` or `read_write`) in every
/// shader that uses it — wgpu rejects a mismatch.
pub trait ShaderAppExt {
    /// Register an auto-managed uniform buffer at `@group(group_index) @binding(binding_index)`.
    ///
    /// `U` must be inserted as a [`Resource`] in the main world. The library extracts
    /// it to the render world and uploads it to a persistent uniform buffer whenever
    /// the resource changes.
    ///
    /// Multiple calls with the same `group_index` but different `binding_index` values
    /// pack several uniform bindings into one bind group.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// #[derive(Resource, Default, ShaderType, Clone)]
    /// struct TimeUniforms { elapsed: f32 }
    ///
    /// app.insert_resource(TimeUniforms::default())
    ///    .register_uniform_buffer::<TimeUniforms>(0, 0);
    /// // WGSL: @group(0) @binding(0) var<uniform> time_uniforms: TimeUniforms;
    /// ```
    fn register_uniform_buffer<U>(&mut self, group_index: u32, binding_index: u32) -> &mut Self
    where
        U: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static;

    /// Register an auto-managed storage buffer at `@group(group_index) @binding(binding_index)`.
    ///
    /// `S` must be inserted as a [`Resource`] in the main world. The library extracts
    /// it to the render world and uploads it to a persistent storage buffer whenever
    /// the resource changes.
    ///
    /// Multiple calls with the same `group_index` but different `binding_index` values
    /// pack several storage bindings into one bind group.
    /// `read_write`: `false` → `var<storage, read>`, `true` → `var<storage, read_write>`.
    ///
    /// The CPU owns this buffer: shader writes persist until the resource next changes,
    /// then the CPU upload replaces the whole buffer. For data the GPU owns, use
    /// [`register_gpu_buffer`](Self::register_gpu_buffer).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// #[derive(Resource, Default, ShaderType, Clone)]
    /// struct Palette { colors: [Vec4; 16] }
    ///
    /// app.insert_resource(Palette::default())
    ///    .register_storage_buffer::<Palette>(1, 0, false);
    /// // WGSL: @group(1) @binding(0) var<storage, read> palette: Palette;
    /// ```
    fn register_storage_buffer<S>(
        &mut self,
        group_index: u32,
        binding_index: u32,
        read_write: bool,
    ) -> &mut Self
    where
        S: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static;

    /// Register a fixed-size array buffer at `@group(group_index) @binding(binding_index)`.
    ///
    /// Maps to WGSL `array<T, N>`. The buffer is initialized once with `T::default()` values
    /// and persists across frames — only elements explicitly changed via [`ArrayBufferChanges`]
    /// are uploaded each frame, batched into contiguous `write_buffer` runs.
    ///
    /// `Tag` is a user-defined zero-sized marker type. Define one per registration so that
    /// multiple buffers of the same element type and length can coexist:
    ///
    /// ```rust,ignore
    /// struct Colors;
    /// struct Positions;
    /// app.register_array_buffer::<Colors, Vec4, 64>(1, 0, false);
    /// app.register_array_buffer::<Positions, Vec4, 64>(2, 0, false);
    ///
    /// // Each system names only its tag — no T or N required:
    /// fn update_colors(mut changes: ResMut<ArrayBufferChanges<Colors>>) {
    ///     changes.set(0, Vec4::ONE);
    /// }
    /// ```
    ///
    /// `read_write`: `false` → `var<storage, read>`, `true` → `var<storage, read_write>`.
    fn register_array_buffer<Tag, T, const N: usize>(
        &mut self,
        group_index: u32,
        binding_index: u32,
        read_write: bool,
    ) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + WriteInto + Default + Send + Sync + 'static;

    /// Register a GPU-owned buffer of `len` elements at
    /// `@group(group_index) @binding(binding_index)`.
    ///
    /// Maps to WGSL `var<storage, read_write> name: array<T>` (always `read_write`). The
    /// buffer starts filled with `T::default()` and is never uploaded from the CPU again,
    /// so shader writes persist across frames — use it for simulation state that compute
    /// passes update. Its handle is available in the main world as [`GpuBuffer<Tag>`].
    ///
    /// `Tag` is a user-defined marker type naming the buffer, also used to size
    /// dispatches with [`Workgroups::over`]:
    ///
    /// ```rust,ignore
    /// #[derive(ShaderType, Clone, Copy, Default)]
    /// struct Particle { pos: Vec2, vel: Vec2 }
    /// struct Particles;
    ///
    /// app.register_gpu_buffer::<Particles, Particle>(1, 0, 1024);
    /// // WGSL: @group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
    /// ```
    fn register_gpu_buffer<Tag, T>(
        &mut self,
        group_index: u32,
        binding_index: u32,
        len: u32,
    ) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + WriteInto + Default + Send + Sync + 'static;

    /// Enable reading the GPU buffer tagged `Tag` back to the CPU, decoded as `T` values
    /// into the [`GpuReadback<Tag, T>`] resource. Call after
    /// [`register_gpu_buffer`](Self::register_gpu_buffer), with the same `T`.
    ///
    /// With [`ReadbackMode::OnRequest`], call [`GpuReadback::request`] whenever you want a
    /// copy. With [`ReadbackMode::EveryFrame`], the buffer is copied every frame. Results
    /// arrive 1–3 frames after the copy is made.
    ///
    /// ```rust,ignore
    /// app.register_gpu_buffer::<Particles, Particle>(1, 0, 1024)
    ///    .read_back_gpu_buffer::<Particles, Particle>(ReadbackMode::OnRequest);
    ///
    /// fn request(readback: Res<GpuReadback<Particles, Particle>>) {
    ///     readback.request();
    /// }
    ///
    /// fn print(readback: Res<GpuReadback<Particles, Particle>>) {
    ///     if !readback.is_changed() { return; }
    ///     let Some(particles) = readback.latest() else { return };
    ///     info!("{} particles", particles.len());
    /// }
    /// ```
    fn read_back_gpu_buffer<Tag, T>(&mut self, mode: ReadbackMode) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + CreateFrom + Send + Sync + 'static;

    /// Enable reading the array buffer tagged `Tag` back to the CPU, decoded as `T` values
    /// into the [`GpuReadback<Tag, T>`] resource. Call after
    /// [`register_array_buffer`](Self::register_array_buffer), with the same `T`.
    ///
    /// Works exactly like [`read_back_gpu_buffer`](Self::read_back_gpu_buffer). Useful
    /// when a shader writes to a `read_write` array buffer.
    fn read_back_array_buffer<Tag, T>(&mut self, mode: ReadbackMode) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + CreateFrom + Send + Sync + 'static;

    /// Enable reading the storage buffer of resource type `S` back to the CPU, decoded as
    /// an `S` into the [`StorageReadback<S>`] resource. Call after
    /// [`register_storage_buffer`](Self::register_storage_buffer). If `S` is registered at
    /// several bindings, the first storage registration is read.
    ///
    /// Useful when a shader writes to a `read_write` storage buffer. With
    /// [`ReadbackMode::OnRequest`], call [`StorageReadback::request`] whenever you want a
    /// copy. With [`ReadbackMode::EveryFrame`], the buffer is copied every frame.
    ///
    /// ```rust,ignore
    /// app.register_storage_buffer::<Stats>(1, 0, true)
    ///    .init_resource::<Stats>()
    ///    .read_back_storage_buffer::<Stats>(ReadbackMode::EveryFrame);
    ///
    /// fn print(readback: Res<StorageReadback<Stats>>) {
    ///     let Some(stats) = readback.latest() else { return };
    ///     info!("{:?}", stats.total);
    /// }
    /// ```
    fn read_back_storage_buffer<S>(&mut self, mode: ReadbackMode) -> &mut Self
    where
        S: ShaderType + WriteInto + CreateFrom + Default + Resource + Clone + Send + Sync + 'static;
}

impl ShaderAppExt for App {
    fn register_uniform_buffer<U>(&mut self, group_index: u32, binding_index: u32) -> &mut Self
    where
        U: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static,
    {
        cpu_buffer::register::<U>(self, group_index, binding_index, AutoBufferKind::Uniform);
        self
    }

    fn register_storage_buffer<S>(
        &mut self,
        group_index: u32,
        binding_index: u32,
        read_write: bool,
    ) -> &mut Self
    where
        S: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static,
    {
        cpu_buffer::register::<S>(
            self,
            group_index,
            binding_index,
            AutoBufferKind::Storage {
                read_only: !read_write,
            },
        );
        self
    }

    fn register_array_buffer<Tag, T, const N: usize>(
        &mut self,
        group_index: u32,
        binding_index: u32,
        read_write: bool,
    ) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + WriteInto + Default + Send + Sync + 'static,
    {
        auto_array::register::<Tag, T, N>(self, group_index, binding_index, read_write);
        self
    }

    fn register_gpu_buffer<Tag, T>(
        &mut self,
        group_index: u32,
        binding_index: u32,
        len: u32,
    ) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + WriteInto + Default + Send + Sync + 'static,
    {
        gpu_buffer::register::<Tag, T>(self, group_index, binding_index, len);
        self
    }

    fn read_back_gpu_buffer<Tag, T>(&mut self, mode: ReadbackMode) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + CreateFrom + Send + Sync + 'static,
    {
        readback::register_gpu::<Tag, T>(self, mode);
        self
    }

    fn read_back_array_buffer<Tag, T>(&mut self, mode: ReadbackMode) -> &mut Self
    where
        Tag: Send + Sync + 'static,
        T: ShaderSize + CreateFrom + Send + Sync + 'static,
    {
        readback::register_array::<Tag, T>(self, mode);
        self
    }

    fn read_back_storage_buffer<S>(&mut self, mode: ReadbackMode) -> &mut Self
    where
        S: ShaderType + WriteInto + CreateFrom + Default + Resource + Clone + Send + Sync + 'static,
    {
        readback::register_storage::<S>(self, mode);
        self
    }
}

// ---------------------------------------------------------------------------
// fragment_layout! macro
// ---------------------------------------------------------------------------

/// Build a [`BindGroupLayoutDescriptor`] with sequential fragment-stage bindings.
#[macro_export]
macro_rules! fragment_layout {
    ($label:expr, $($entry:expr),+ $(,)?) => {
        $crate::__private::BindGroupLayoutDescriptor::new(
            $label,
            &$crate::__private::BindGroupLayoutEntries::sequential(
                $crate::__private::ShaderStages::FRAGMENT,
                ($($entry,)+),
            ),
        )
    };
}

// ---------------------------------------------------------------------------
// Plugins
// ---------------------------------------------------------------------------

/// Binding machinery shared by [`FullscreenFragmentPlugin`] and [`ComputeShaderPlugin`]:
/// layout compilation and bind group assembly for every registered buffer.
struct ShaderCorePlugin;

impl Plugin for ShaderCorePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        bindings::init_resources(render_app);

        render_app
            .configure_sets(
                Render,
                (
                    ShaderSystems::PrepareBindings,
                    ShaderSystems::FinalizeBindGroups,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                bindings::finalize_bind_groups.in_set(ShaderSystems::FinalizeBindGroups),
            )
            .add_systems(
                RenderStartup,
                bindings::compile_layouts.in_set(ShaderSystems::CompileLayouts),
            );
    }
}

/// Adds [`ShaderCorePlugin`] unless another plugin already did.
fn add_core_plugin(app: &mut App) {
    if !app.is_plugin_added::<ShaderCorePlugin>() {
        app.add_plugins(ShaderCorePlugin);
    }
}

/// Bevy plugin that wires up a fullscreen fragment shader pipeline.
///
/// Call the [`ShaderAppExt`] registration methods on the [`App`] to bind data to your
/// shader.
///
/// This plugin is not compatible with MSAA. Disable MSAA on all cameras.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Resource, Default, ShaderType, Clone)]
/// struct MyUniforms { time: f32 }
///
/// App::new()
///     .add_plugins(FullscreenFragmentPlugin::new("shaders/my_effect.wgsl"))
///     .insert_resource(MyUniforms::default())
///     .register_uniform_buffer::<MyUniforms>(0, 0)
///     .run();
/// ```
pub struct FullscreenFragmentPlugin {
    /// Path to the fragment shader asset (e.g. `"shaders/effect.wgsl"` or
    /// `"shaders/effect.spv"`).
    pub shader_path: &'static str,
    /// Fragment entry point name. `None` auto-detects the module's only one.
    pub entry_point: Option<&'static str>,
}

impl FullscreenFragmentPlugin {
    /// Creates a new plugin that renders the shader at the given asset path.
    pub fn new(shader_path: &'static str) -> Self {
        Self {
            shader_path,
            entry_point: None,
        }
    }

    /// Names the fragment entry point. Required when the shader module has more
    /// than one.
    ///
    /// ```rust,ignore
    /// FullscreenFragmentPlugin::new("shaders/my_shader.spv")
    ///     .with_entry_point("main_fs")
    /// ```
    pub fn with_entry_point(mut self, entry_point: &'static str) -> Self {
        self.entry_point = Some(entry_point);
        self
    }
}

impl Plugin for FullscreenFragmentPlugin {
    fn build(&self, app: &mut App) {
        add_core_plugin(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.insert_resource(FullscreenPipelineConfig {
            shader_path: self.shader_path,
            entry_point: self.entry_point,
        });

        render_app.init_resource::<FragmentExtraLayouts>();
        render_app.init_resource::<FragmentExtraBindGroups>();

        render_app
            .configure_sets(
                RenderStartup,
                FragmentSystems::InitPipeline.after(ShaderSystems::CompileLayouts),
            )
            .add_systems(
                RenderStartup,
                pipeline::init_pipeline.in_set(FragmentSystems::InitPipeline),
            )
            .add_systems(
                Render,
                pipeline::queue_fullscreen_pipelines.in_set(RenderSystems::Prepare),
            );

        // After the prepass and before the main pass, which draws on top. Only one
        // `FullscreenFragmentPlugin` is supported per app.
        render_app.add_systems(
            Core3d,
            fullscreen_pass
                .in_set(ShaderPassSystems::Fullscreen)
                .after(Core3dSystems::Prepass)
                .before(Core3dSystems::MainPass),
        );
    }
}
