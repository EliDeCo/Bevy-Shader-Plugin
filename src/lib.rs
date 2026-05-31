mod auto_array;
mod auto_buffer;
mod extra_bind_group;
mod node;
mod pipeline;

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        render_graph::{RenderGraphExt, RenderLabel, ViewNodeRunner},
        render_resource::{
            BindGroup, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, ShaderStages,
            ShaderType, StorageBuffer, UniformBuffer,
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use encase::{ShaderSize, internal::WriteInto};
use wgpu::{BufferUsages, util::BufferInitDescriptor};

fn init_auto_buffer_resources(render_app: &mut bevy::app::SubApp) {
    render_app.init_resource::<AutoBufferLayouts>();
    render_app.init_resource::<AutoBufferBindGroups>();
    render_app.init_resource::<AutoBufferCompiledLayouts>();
    render_app.init_resource::<PendingBufferBindings>();
}

fn uniform_write<T: ShaderType + WriteInto + Default>(
    value: T,
    device: &RenderDevice,
    queue: &RenderQueue,
) -> Option<Buffer> {
    let mut buf = UniformBuffer::default();
    buf.set(value);
    buf.write_buffer(device, queue);
    buf.buffer().cloned()
}

fn storage_write<T: ShaderType + WriteInto + Default>(
    value: T,
    device: &RenderDevice,
    queue: &RenderQueue,
) -> Option<Buffer> {
    let mut buf = StorageBuffer::default();
    buf.set(value);
    buf.write_buffer(device, queue);
    buf.buffer().cloned()
}

fn register_buffer_impl<T, F>(
    render_app: &mut bevy::app::SubApp,
    group_index: u32,
    binding_index: u32,
    kind: AutoBufferKind,
    write_fn: F,
) where
    T: Resource + Clone + Send + Sync + 'static,
    F: Fn(T, &RenderDevice, &RenderQueue) -> Option<Buffer> + Send + Sync + 'static,
{
    init_auto_buffer_resources(render_app);

    {
        let world = render_app.world_mut();
        let mut layouts = world.get_resource_mut::<AutoBufferLayouts>().unwrap();
        layouts
            .0
            .entry(group_index)
            .or_default()
            .insert(binding_index, kind);
    }

    render_app.add_systems(ExtractSchedule, auto_buffer::extract_buffer::<T>);

    render_app.add_systems(
        Render,
        (move |mut pending: ResMut<PendingBufferBindings>,
               render_device: Res<RenderDevice>,
               render_queue: Res<RenderQueue>,
               resource: Option<Res<T>>| {
            let Some(resource) = resource else { return };
            if let Some(buf) = write_fn((*resource).clone(), &render_device, &render_queue) {
                pending.0.insert((group_index, binding_index), buf);
            }
        })
        .in_set(RenderSystems::PrepareBindGroups),
    );
}

pub use auto_array::{ArrayBufferChanges, ArrayBufferState};
pub use auto_buffer::{
    AutoBufferBindGroups, AutoBufferCompiledLayouts, AutoBufferKind, AutoBufferLayouts,
    PendingBufferBindings,
};
pub use extra_bind_group::FragmentBindGroupBuilder;
pub use node::FullscreenNode;
pub use pipeline::{FullscreenPipeline, FullscreenPipelineConfig};

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

/// The render graph node label. Only one `FullscreenFragmentPlugin` instance
/// is supported per app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FullscreenShaderNode;
impl RenderLabel for FullscreenShaderNode {
    fn dyn_clone(&self) -> Box<dyn RenderLabel> {
        Box::new(*self)
    }
}

// ---------------------------------------------------------------------------
// App extension trait
// ---------------------------------------------------------------------------

/// Extension methods on [`App`] for registering buffer bindings and manual bind groups.
pub trait FragmentAppExt {
    /// Register an auto-managed uniform buffer at `@group(group_index) @binding(binding_index)`.
    ///
    /// `U` must be inserted as a [`Resource`] in the main world. The library extracts
    /// it to the render world and uploads it to a uniform buffer every frame.
    ///
    /// Multiple calls with the same `group_index` but different `binding_index` values
    /// pack several uniform bindings into one bind group.
    fn register_uniform_buffer<U>(&mut self, group_index: u32, binding_index: u32) -> &mut Self
    where
        U: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static;

    /// Register an auto-managed storage buffer at `@group(group_index) @binding(binding_index)`.
    ///
    /// `S` must be inserted as a [`Resource`] in the main world. The library extracts
    /// it to the render world and uploads it to a storage buffer every frame.
    ///
    /// Multiple calls with the same `group_index` but different `binding_index` values
    /// pack several storage bindings into one bind group.
    /// `read_write`: `false` → `var<storage, read>`, `true` → `var<storage, read_write>`.
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
}

impl FragmentAppExt for App {
    fn register_uniform_buffer<U>(&mut self, group_index: u32, binding_index: u32) -> &mut Self
    where
        U: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static,
    {
        register_buffer_impl::<U, _>(
            self.sub_app_mut(RenderApp),
            group_index,
            binding_index,
            AutoBufferKind::Uniform,
            uniform_write::<U>,
        );
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
        register_buffer_impl::<S, _>(
            self.sub_app_mut(RenderApp),
            group_index,
            binding_index,
            AutoBufferKind::Storage {
                read_only: !read_write,
            },
            storage_write::<S>,
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
        self.insert_resource(ArrayBufferChanges::<Tag> {
            changes: Vec::new(),
            len: N,
            _marker: std::marker::PhantomData,
        });
        self.add_systems(First, auto_array::clear_array_changes::<Tag>);

        let render_app = self.sub_app_mut(RenderApp);

        init_auto_buffer_resources(render_app);

        {
            let world = render_app.world_mut();
            let mut layouts = world.get_resource_mut::<AutoBufferLayouts>().unwrap();
            layouts.0.entry(group_index).or_default().insert(
                binding_index,
                AutoBufferKind::Storage {
                    read_only: !read_write,
                },
            );
        }

        // RenderStartup: create the persistent GPU buffer pre-filled with T::default().
        render_app.add_systems(
            RenderStartup,
            move |mut commands: Commands, render_device: Res<RenderDevice>| {
                // [T; 1]::SHADER_SIZE equals the WGSL array element stride for any N.
                let stride = <[T; 1] as ShaderSize>::SHADER_SIZE.get() as usize;
                let el_size = T::SHADER_SIZE.get() as usize;

                let default_val = T::default();
                let mut el_buf = vec![0u8; el_size];
                encase::StorageBuffer::new(&mut el_buf[..])
                    .write(&default_val)
                    .unwrap();

                let mut bytes = vec![0u8; stride * N];
                for i in 0..N {
                    bytes[i * stride..i * stride + el_size].copy_from_slice(&el_buf);
                }

                let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("array_buffer"),
                    contents: &bytes,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });

                commands.insert_resource(ArrayBufferState::<Tag> {
                    buffer,
                    stride,
                    _marker: std::marker::PhantomData,
                });
            },
        );

        render_app.add_systems(ExtractSchedule, auto_array::extract_array_changes::<Tag>);

        render_app.add_systems(
            Render,
            (move |state: Option<Res<ArrayBufferState<Tag>>>,
                   changes: Option<Res<ArrayBufferChanges<Tag>>>,
                   render_queue: Res<RenderQueue>,
                   mut pending: ResMut<PendingBufferBindings>| {
                let (Some(state), Some(changes)) = (state, changes) else {
                    return;
                };
                auto_array::apply_array_buffer_changes(
                    &*state,
                    &*changes,
                    &render_queue,
                    &mut pending,
                    group_index,
                    binding_index,
                );
            })
            .in_set(RenderSystems::PrepareBindGroups),
        );

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
// Plugin
// ---------------------------------------------------------------------------

/// Bevy plugin that wires up a fullscreen fragment shader pipeline.
///
/// Call [`register_uniform_buffer`](FragmentAppExt::register_uniform_buffer) and
/// [`register_storage_buffer`](FragmentAppExt::register_storage_buffer) on the
/// [`App`] to bind data to your shader.
///
/// This plugin is not compatible with MSAA. Disable MSAA on all cameras.
pub struct FullscreenFragmentPlugin {
    /// Path to the fragment shader asset (e.g. `"shaders/effect.wgsl"`).
    pub shader_path: &'static str,
}

impl FullscreenFragmentPlugin {
    pub fn new(shader_path: &'static str) -> Self {
        Self { shader_path }
    }
}

impl Plugin for FullscreenFragmentPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.insert_resource(FullscreenPipelineConfig {
            shader_path: self.shader_path,
        });

        render_app.init_resource::<FragmentExtraLayouts>();
        render_app.init_resource::<FragmentExtraBindGroups>();
        init_auto_buffer_resources(render_app);

        render_app.add_systems(
            Render,
            auto_buffer::finalize_buffer_bind_groups.after(RenderSystems::PrepareBindGroups),
        );

        render_app.add_systems(
            RenderStartup,
            pipeline::init_pipeline.in_set(FragmentSystems::InitPipeline),
        );

        render_app
            .add_render_graph_node::<ViewNodeRunner<FullscreenNode>>(Core3d, FullscreenShaderNode)
            .add_render_graph_edges(Core3d, (FullscreenShaderNode, Node3d::StartMainPass));
    }
}
