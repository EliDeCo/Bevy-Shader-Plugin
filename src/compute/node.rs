use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
};

use super::pipeline::ComputeRunList;
use crate::bindings::AutoBufferBindGroups;

/// The render graph label of the compute node. It runs in the root render graph before
/// [`CameraDriverLabel`](bevy::render::graph::CameraDriverLabel), so compute work happens
/// once per frame, before any camera renders.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct ComputeShaderNode;

/// Records every dispatch in [`ComputeRunList`] into a single compute pass. wgpu makes
/// each dispatch's writes visible to the dispatches after it.
#[derive(Default)]
pub(crate) struct ComputeNode;

impl Node for ComputeNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(run_list) = world.get_resource::<ComputeRunList>() else {
            return Ok(());
        };
        if run_list.0.is_empty() {
            return Ok(());
        }
        let pipeline_cache = world.resource::<PipelineCache>();
        let bind_groups = world.resource::<AutoBufferBindGroups>();

        let mut pass =
            render_context
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

        Ok(())
    }
}
