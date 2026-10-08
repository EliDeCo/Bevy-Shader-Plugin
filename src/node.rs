use bevy::{
    prelude::*,
    render::{
        render_resource::{PipelineCache, RenderPassDescriptor},
        renderer::{RenderContext, ViewQuery},
        view::ViewTarget,
    },
};

use crate::{
    FragmentExtraBindGroups,
    bindings::{AutoBufferBindGroups, AutoBufferLayouts, all_groups_bound},
    pipeline::FullscreenPipeline,
};

/// The render system that draws the fullscreen fragment shader for each 3D camera. It
/// runs in the [`Core3d`](bevy::core_pipeline::Core3d) schedule, in
/// [`ShaderPassSystems::Fullscreen`](crate::ShaderPassSystems::Fullscreen), just before
/// the main pass.
///
/// All bind groups — both auto-managed (uniform + storage) and manual extra — are
/// set before the draw call. Auto-managed groups are bound at their registered group
/// index; manual extra groups follow sequentially.
pub fn fullscreen_pass(
    view: ViewQuery<&ViewTarget>,
    pipeline_res: Option<Res<FullscreenPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<AutoBufferLayouts>,
    auto: Res<AutoBufferBindGroups>,
    extra: Option<Res<FragmentExtraBindGroups>>,
    mut ctx: RenderContext,
) {
    let view_target = view.into_inner();
    let Some(pipeline_res) = pipeline_res else {
        return;
    };
    // Each camera format has its own pipeline (see `queue_fullscreen_pipelines`).
    let Some(pipeline) = pipeline_res
        .pipeline_id(view_target.main_texture_format())
        .and_then(|id| pipeline_cache.get_render_pipeline(id))
    else {
        return;
    };
    // Drawing with an unset group is a validation error, so wait until every
    // registered group has its bind group.
    if !all_groups_bound(&layouts, &auto) {
        return;
    }

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("fullscreen_fragment_pass"),
        color_attachments: &[Some(view_target.get_color_attachment())],
        ..default()
    });

    render_pass.set_render_pipeline(pipeline);

    for (group_index, bind_group) in auto.0.iter() {
        render_pass.set_bind_group(*group_index as usize, bind_group, &[]);
    }
    let auto_count = auto.0.len();

    if let Some(extra) = &extra {
        for (i, bind_group) in extra.0.iter().enumerate() {
            render_pass.set_bind_group(auto_count + i, bind_group, &[]);
        }
    }

    render_pass.draw(0..3, 0..1);
}
