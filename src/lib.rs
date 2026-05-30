mod auto_storage;
mod bind_group;
mod extra_bind_group;
mod node;
mod pipeline;

use std::marker::PhantomData;

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    ecs::system::ScheduleSystem,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        render_graph::{RenderGraphExt, RenderLabel, ViewNodeRunner},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            ShaderStages, ShaderType, StorageBuffer,
            binding_types::{
                sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
            },
        },
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};
use encase::internal::WriteInto;

pub use auto_storage::{
    AutoStorageBindGroups, AutoStorageCompiledLayouts, AutoStorageLayouts,
    PendingStorageBindings,
};
pub use bind_group::FullscreenBindGroup;
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

/// Bind group layout descriptors for the extra bind groups (groups 1..n).
///
/// Prefer the helper methods ([`texture_2d_and_sampler`](Self::texture_2d_and_sampler),
/// [`storage_buffer_read_only`](Self::storage_buffer_read_only), etc.) over
/// pushing raw descriptors via `.0.push(...)`.
///
/// Populate this resource in the layout system passed to
/// [`FragmentAppExt::register_extra_bind_group`].
#[derive(Resource, Default)]
pub struct FragmentExtraLayouts(pub Vec<BindGroupLayoutDescriptor>);

impl FragmentExtraLayouts {
    /// Push an arbitrary [`BindGroupLayoutDescriptor`].
    pub fn push(&mut self, desc: BindGroupLayoutDescriptor) -> &mut Self {
        self.0.push(desc);
        self
    }

    /// Add a group with a filterable 2D float texture at binding 0 and a
    /// filtering sampler at binding 1, both visible to fragment shaders.
    pub fn texture_2d_and_sampler(&mut self, label: &'static str) -> &mut Self {
        self.0.push(BindGroupLayoutDescriptor::new(
            label,
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(bevy::render::render_resource::TextureSampleType::Float {
                        filterable: true,
                    }),
                    sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
                ),
            ),
        ));
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

/// Per-frame bind groups for groups 1..n.
///
/// Call [`clear`](Self::clear) at the start of your `PrepareBindGroups` system,
/// then use [`push`](Self::push), [`push_gpu_image`](Self::push_gpu_image), or
/// [`FragmentBindGroupBuilder`] to populate each group in order.
#[derive(Resource, Default)]
pub struct FragmentExtraBindGroups(pub Vec<BindGroup>);

impl FragmentExtraBindGroups {
    /// Push a pre-built bind group. Corresponds to GPU group `self.0.len() + 1`.
    pub fn push(&mut self, bind_group: BindGroup) -> &mut Self {
        self.0.push(bind_group);
        self
    }

    /// Remove all bind groups. Call this at the start of your
    /// `PrepareBindGroups` system before re-populating.
    pub fn clear(&mut self) -> &mut Self {
        self.0.clear();
        self
    }

    /// Create and push a bind group for a [`GpuImage`] (texture view + sampler)
    /// using the compiled layout at `group_index` in [`FullscreenPipeline::extra_layouts`].
    ///
    /// `group_index` is 0-based into the extra layouts (so index 0 → GPU group 1).
    pub fn push_gpu_image<U: 'static>(
        &mut self,
        label: &str,
        pipeline: &FullscreenPipeline<U>,
        render_device: &RenderDevice,
        group_index: usize,
        gpu_image: &GpuImage,
    ) -> &mut Self {
        let layout = &pipeline.extra_layouts[group_index];
        let bind_group = render_device.create_bind_group(
            label,
            layout,
            &BindGroupEntries::sequential((&gpu_image.texture_view, &gpu_image.sampler)),
        );
        self.0.push(bind_group);
        self
    }
}

/// The render graph node label. Only one `FullscreenFragmentPlugin` instance
/// is supported per app (adding two would cause a label conflict here).
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

/// Extension methods on [`App`] for registering extra bind group systems.
pub trait FragmentAppExt {
    /// Register a layout-setup system and a per-frame bind-group system for
    /// extra bind groups (groups 1..n).
    ///
    /// - `layout_system` runs in `RenderStartup` **before**
    ///   [`FragmentSystems::InitPipeline`] — ordering is handled automatically.
    /// - `prepare_system` runs in `RenderSystems::PrepareBindGroups`.
    ///
    /// Both systems run in the render world. If you need main-world resources
    /// available there, register an extraction system separately via
    /// `app.sub_app_mut(RenderApp).add_systems(ExtractSchedule, ...)`.
    fn register_extra_bind_group<LM, PM>(
        &mut self,
        layout_system: impl IntoScheduleConfigs<ScheduleSystem, LM>,
        prepare_system: impl IntoScheduleConfigs<ScheduleSystem, PM>,
    ) -> &mut Self;

    /// Register an auto-managed read-only storage buffer at the given WGSL
    /// `@group(group_index) @binding(binding_index)` location.
    ///
    /// `group_index` must be ≥ 1 (group 0 is the uniform) and all registered
    /// group indices must be contiguous starting at 1. Multiple calls with the
    /// same `group_index` but different `binding_index` values pack several
    /// buffers into one bind group.
    ///
    /// `S` must be inserted as a `Resource` in the main world. The library
    /// extracts it to the render world and uploads it to a storage buffer every
    /// frame, identical to how the uniform is handled.
    fn register_storage_buffer<S>(&mut self, group_index: u32, binding_index: u32) -> &mut Self
    where
        S: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static;
}

impl FragmentAppExt for App {
    fn register_extra_bind_group<LM, PM>(
        &mut self,
        layout_system: impl IntoScheduleConfigs<ScheduleSystem, LM>,
        prepare_system: impl IntoScheduleConfigs<ScheduleSystem, PM>,
    ) -> &mut Self {
        let render_app = self.sub_app_mut(RenderApp);
        render_app
            .add_systems(
                RenderStartup,
                layout_system.before(FragmentSystems::InitPipeline),
            )
            .add_systems(
                Render,
                prepare_system.in_set(RenderSystems::PrepareBindGroups),
            );
        self
    }

    fn register_storage_buffer<S>(&mut self, group_index: u32, binding_index: u32) -> &mut Self
    where
        S: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static,
    {
        let render_app = self.sub_app_mut(RenderApp);

        render_app.init_resource::<AutoStorageLayouts>();
        render_app.init_resource::<AutoStorageBindGroups>();
        render_app.init_resource::<AutoStorageCompiledLayouts>();
        render_app.init_resource::<PendingStorageBindings>();

        // Eagerly record (group, binding) so init_pipeline can build the layout.
        {
            let world = render_app.world_mut();
            let mut layouts = world.get_resource_mut::<AutoStorageLayouts>().unwrap();
            layouts.0.entry(group_index).or_default().insert(binding_index);
        }

        // Extract S from main world → render world each frame.
        render_app.add_systems(ExtractSchedule, auto_storage::extract_storage::<S>);

        // Write S to GPU each frame and stash the raw buffer handle for finalization.
        render_app.add_systems(
            Render,
            (move |mut pending: ResMut<PendingStorageBindings>,
                   render_device: Res<RenderDevice>,
                   render_queue: Res<RenderQueue>,
                   resource: Option<Res<S>>| {
                let Some(resource) = resource else { return };
                let mut buf = StorageBuffer::default();
                buf.set((*resource).clone());
                buf.write_buffer(&render_device, &render_queue);
                if let Some(raw_buf) = buf.buffer() {
                    pending.0.insert((group_index, binding_index), raw_buf.clone());
                }
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
///
/// Shorthand for the common `BindGroupLayoutDescriptor::new + BindGroupLayoutEntries::sequential`
/// pattern when all bindings are visible only to the fragment shader.
///
/// # Example
/// ```rust,ignore
/// use bevy::render::render_resource::binding_types::{texture_2d, sampler};
/// use bevy::render::render_resource::{TextureSampleType, SamplerBindingType};
///
/// extra_layouts.push(fragment_layout!(
///     "my_layout",
///     texture_2d(TextureSampleType::Float { filterable: true }),
///     sampler(SamplerBindingType::Filtering),
/// ));
/// ```
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
/// `U` is the uniform type, which must be inserted as a `Resource` in the
/// main world. The library extracts it to the render world automatically.
///
/// This plugin is not compatible with MSAA. Make sure to disable MSAA on all cameras.
pub struct FullscreenFragmentPlugin<U> {
    /// Path to the fragment shader asset (e.g. `"shaders/effect.wgsl"`).
    pub shader_path: &'static str,
    _phantom: PhantomData<fn() -> U>,
}

impl<U> FullscreenFragmentPlugin<U> {
    pub fn new(shader_path: &'static str) -> Self {
        Self {
            shader_path,
            _phantom: PhantomData,
        }
    }
}

impl<U> Plugin for FullscreenFragmentPlugin<U>
where
    U: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static,
{
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.insert_resource(FullscreenPipelineConfig {
            shader_path: self.shader_path,
        });

        render_app.init_resource::<FragmentExtraLayouts>();
        render_app.init_resource::<FragmentExtraBindGroups>();
        render_app.init_resource::<AutoStorageLayouts>();
        render_app.init_resource::<AutoStorageBindGroups>();
        render_app.init_resource::<AutoStorageCompiledLayouts>();
        render_app.init_resource::<PendingStorageBindings>();

        render_app.add_systems(
            Render,
            auto_storage::finalize_storage_bind_groups
                .after(RenderSystems::PrepareBindGroups),
        );

        render_app.add_systems(
            RenderStartup,
            pipeline::init_pipeline::<U>.in_set(FragmentSystems::InitPipeline),
        );

        render_app.add_systems(ExtractSchedule, extract_uniform::<U>);

        render_app.add_systems(
            Render,
            bind_group::prepare_bind_group::<U>.in_set(RenderSystems::PrepareBindGroups),
        );

        render_app
            .add_render_graph_node::<ViewNodeRunner<FullscreenNode<U>>>(
                Core3d,
                FullscreenShaderNode,
            )
            .add_render_graph_edges(Core3d, (FullscreenShaderNode, Node3d::StartMainPass));
    }
}

// ---------------------------------------------------------------------------
// Extract system
// ---------------------------------------------------------------------------

/// Copies the `U` resource from the main world into the render world each frame.
fn extract_uniform<U: Resource + Clone>(mut commands: Commands, uniform: Extract<Option<Res<U>>>) {
    if let Some(u) = uniform.as_deref() {
        commands.insert_resource(u.clone());
    }
}
