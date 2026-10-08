use bevy::{
    prelude::*,
    render::{
        render_resource::{ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
};

use super::pipeline::ComputeRunList;
use crate::bindings::AutoBufferBindGroups;

/// Records every dispatch in [`ComputeRunList`] into a single compute pass. Runs in the
/// root [`RenderGraph`](bevy::render::renderer::RenderGraph) schedule before
/// [`camera_driver`](bevy::core_pipeline::schedule::camera_driver), so compute work is
/// submitted once per frame, before any camera renders. wgpu makes each dispatch's
/// writes visible to the dispatches after it.
pub(crate) fn compute_pass(
    run_list: Option<Res<ComputeRunList>>,
    pipeline_cache: Res<PipelineCache>,
    bind_groups: Res<AutoBufferBindGroups>,
    mut ctx: RenderContext,
) {
    let Some(run_list) = run_list else {
        return;
    };
    if run_list.0.is_empty() {
        return;
    }

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("compute_shader_pass"),
            timestamp_writes: None,
        });

    for (group_index, bind_group) in &bind_groups.0 {
        pass.set_bind_group(*group_index, &**bind_group, &[]);
    }

    for (pipeline_id, count) in &run_list.0 {
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(*pipeline_id) else {
            continue;
        };
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(count.x, count.y, count.z);
    }
}
