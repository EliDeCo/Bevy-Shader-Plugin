use std::marker::PhantomData;

use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, PipelineCache, ShaderType, UniformBuffer,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use encase::internal::WriteInto;

use crate::pipeline::FullscreenPipeline;

/// Render-world resource holding the per-frame bind group for group 0
/// (the uniform buffer). Re-created each frame by `prepare_bind_group`.
#[derive(Resource)]
pub struct FullscreenBindGroup<U: 'static> {
    pub bind_group: BindGroup,
    _phantom: PhantomData<U>,
}

/// `RenderSystems::PrepareBindGroups` system. Reads `U` from the render world
/// (extracted from the main world each frame), writes it into a `UniformBuffer`,
/// and creates the bind group for group 0.
///
/// `U` must implement `WriteInto` in addition to `ShaderType`. All types that
/// `#[derive(ShaderType)]` automatically satisfy this bound.
pub(crate) fn prepare_bind_group<
    U: ShaderType + WriteInto + Default + Resource + Clone + Send + Sync + 'static,
>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<Res<FullscreenPipeline<U>>>,
    uniform: Option<Res<U>>,
) {
    let Some(pipeline) = pipeline else { return };
    let Some(uniform) = uniform else { return };

    let layout = pipeline_cache.get_bind_group_layout(&pipeline.per_frame_layout);

    let mut u_buffer = UniformBuffer::default();
    u_buffer.set((*uniform).clone());
    u_buffer.write_buffer(&render_device, &render_queue);

    let Some(binding) = u_buffer.binding() else { return };

    let bind_group = render_device.create_bind_group(
        "fullscreen_uniform_bind_group",
        &layout,
        &BindGroupEntries::sequential((binding,)),
    );

    commands.insert_resource(FullscreenBindGroup::<U> {
        bind_group,
        _phantom: PhantomData,
    });
}
