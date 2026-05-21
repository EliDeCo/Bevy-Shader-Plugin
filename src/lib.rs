//! A reusable Bevy library for fullscreen fragment shader effects.
//!
//! # Quickstart
//!
//! 1. Define your uniform type:
//!    ```rust,ignore
//!    #[derive(Resource, ShaderType, Clone, Default)]
//!    struct MyUniform { time: f32, resolution: Vec2, _pad: f32 }
//!    ```
//! 2. Add the plugin and insert the uniform resource:
//!    ```rust,ignore
//!    app.add_plugins(FullscreenFragmentPlugin::<MyUniform>::new("shaders/effect.wgsl"))
//!       .init_resource::<MyUniform>();
//!    ```
//! 3. Spawn a `Camera3d` with Msaa::Off and update `MyUniform` each frame.
//!
//! # Extra bind groups (groups 1..n)
//!
//! To pass storage buffers, textures, or other data to the shader beyond the
//! uniform, create your bind group layout descriptors in a `RenderStartup`
//! system ordered **before** `FragmentSystems::InitPipeline`, push them into
//! `FragmentExtraLayouts`, and update `FragmentExtraBindGroups` each frame in
//! a `RenderSystems::PrepareBindGroups` system.

mod bind_group;
mod node;
mod pipeline;

use std::marker::PhantomData;

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        render_graph::{RenderGraphExt, RenderLabel, ViewNodeRunner},
        render_resource::{BindGroup, BindGroupLayoutDescriptor, ShaderType},
    },
};
use encase::internal::WriteInto;

pub use bind_group::FullscreenBindGroup;
pub use node::FullscreenNode;
pub use pipeline::{FullscreenPipeline, FullscreenPipelineConfig};

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
/// Insert your `BindGroupLayoutDescriptor` values here in a `RenderStartup`
/// system **before** `FragmentSystems::InitPipeline`. The library includes
/// these descriptors in the pipeline layout when queueing the render pipeline.
#[derive(Resource, Default)]
pub struct FragmentExtraLayouts(pub Vec<BindGroupLayoutDescriptor>);

/// Per-frame bind groups for groups 1..n.
///
/// Populate this resource each frame in a `RenderSystems::PrepareBindGroups`
/// system. The library's render node sets these as bind groups 1..n before
/// issuing the draw call.
#[derive(Resource, Default)]
pub struct FragmentExtraBindGroups(pub Vec<BindGroup>);

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
        Self { shader_path, _phantom: PhantomData }
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
fn extract_uniform<U: Resource + Clone>(
    mut commands: Commands,
    uniform: Extract<Option<Res<U>>>,
) {
    if let Some(u) = uniform.as_deref() {
        commands.insert_resource(u.clone());
    }
}
