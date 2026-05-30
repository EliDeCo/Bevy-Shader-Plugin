use std::marker::PhantomData;

use bevy::{
    ecs::query::QueryItem,
    prelude::*,
    render::{
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_resource::{PipelineCache, ShaderType},
        renderer::RenderContext,
        view::ViewTarget,
    },
};

use crate::{FragmentExtraBindGroups, auto_storage::AutoStorageBindGroups, bind_group::FullscreenBindGroup, pipeline::FullscreenPipeline};

/// The render graph node that executes the fullscreen fragment shader.
///
/// Draws a single fullscreen triangle (vertices 0–2, one instance) using the
/// vertex shader provided by Bevy's `FullscreenShader`. All bind groups are
/// set before the draw call: group 0 is the uniform managed by this library,
/// groups 1..n are taken from `FragmentExtraBindGroups`.
pub struct FullscreenNode<U: 'static> {
    _phantom: PhantomData<U>,
}

impl<U> Default for FullscreenNode<U> {
    fn default() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<U> ViewNode for FullscreenNode<U>
where
    U: ShaderType + encase::internal::WriteInto + Default + Resource + Clone + Send + Sync + 'static,
{
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(pipeline_res) = world.get_resource::<FullscreenPipeline<U>>() else {
            return Ok(());
        };
        let Some(bind_group_res) = world.get_resource::<FullscreenBindGroup<U>>() else {
            return Ok(());
        };
        let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
            return Ok(());
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_res.pipeline_id) else {
            return Ok(());
        };

        let mut render_pass = render_context.begin_tracked_render_pass(
            bevy::render::render_resource::RenderPassDescriptor {
                label: Some("fullscreen_fragment_pass".into()),
                color_attachments: &[Some(view_target.get_color_attachment())],
                ..default()
            },
        );

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group_res.bind_group, &[]);

        if let Some(auto) = world.get_resource::<AutoStorageBindGroups>() {
            for (group_index, bind_group) in auto.0.iter() {
                render_pass.set_bind_group(*group_index as usize, bind_group, &[]);
            }
        }

        if let Some(extra) = world.get_resource::<FragmentExtraBindGroups>() {
            for (i, bind_group) in extra.0.iter().enumerate() {
                render_pass.set_bind_group(1 + i, bind_group, &[]);
            }
        }

        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}
