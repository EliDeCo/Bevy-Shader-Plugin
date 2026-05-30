use bevy::{
    ecs::query::QueryItem,
    prelude::*,
    render::{
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_resource::PipelineCache,
        renderer::RenderContext,
        view::ViewTarget,
    },
};

use crate::{
    FragmentExtraBindGroups, auto_buffer::AutoBufferBindGroups, pipeline::FullscreenPipeline,
};

/// The render graph node that executes the fullscreen fragment shader.
///
/// All bind groups — both auto-managed (uniform + storage) and manual extra — are
/// set before the draw call. Auto-managed groups are bound at their registered group
/// index; manual extra groups follow sequentially.
#[derive(Default)]
pub struct FullscreenNode;

impl ViewNode for FullscreenNode {
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(pipeline_res) = world.get_resource::<FullscreenPipeline>() else {
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

        let auto_count = if let Some(auto) = world.get_resource::<AutoBufferBindGroups>() {
            for (group_index, bind_group) in auto.0.iter() {
                render_pass.set_bind_group(*group_index as usize, bind_group, &[]);
            }
            auto.0.len()
        } else {
            0
        };

        if let Some(extra) = world.get_resource::<FragmentExtraBindGroups>() {
            for (i, bind_group) in extra.0.iter().enumerate() {
                render_pass.set_bind_group(auto_count + i, bind_group, &[]);
            }
        }

        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}
