use std::collections::HashMap;

use bevy::{
    prelude::*,
    render::render_resource::{
        CachedComputePipelineId, CachedPipelineState, ComputePipelineDescriptor, PipelineCache,
    },
};

use super::{ComputeFrame, PassConfig};
use crate::bindings::{
    AutoBufferBindGroups, AutoBufferLayoutDescriptors, AutoBufferLayouts, all_groups_bound,
};

/// Render-world copy of the compute configuration.
#[derive(Resource)]
pub(crate) struct ComputeRenderConfig {
    pub shader_path: &'static str,
    pub passes: Vec<PassConfig>,
}

/// One cached pipeline id per configured pass. Passes that share an entry point share a
/// pipeline.
#[derive(Resource)]
pub(crate) struct ComputePipelines {
    pub per_pass: Vec<CachedComputePipelineId>,
}

/// Which init passes have already run.
#[derive(Resource)]
pub(crate) struct ComputeRunState {
    pub init_done: Vec<bool>,
}

/// The dispatches the compute node records this frame, in order.
#[derive(Resource, Default)]
pub(crate) struct ComputeRunList(pub Vec<(CachedComputePipelineId, UVec3)>);

/// `RenderStartup` system. Queues one compute pipeline per distinct entry point, using
/// the same group layouts as the fragment pipeline.
pub(crate) fn init_compute_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    config: Res<ComputeRenderConfig>,
    descriptors: Res<AutoBufferLayoutDescriptors>,
) {
    let shader = asset_server.load(config.shader_path);
    let mut by_entry: HashMap<&'static str, CachedComputePipelineId> = HashMap::new();

    let per_pass = config
        .passes
        .iter()
        .map(|pass| {
            *by_entry.entry(pass.entry).or_insert_with(|| {
                pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(format!("compute_pipeline_{}", pass.entry).into()),
                    layout: descriptors.0.clone(),
                    immediate_size: 0,
                    shader: shader.clone(),
                    shader_defs: Vec::new(),
                    entry_point: Some(pass.entry.into()),
                    zero_initialize_workgroup_memory: true,
                })
            })
        })
        .collect();

    commands.insert_resource(ComputePipelines { per_pass });
}

/// Decides which passes the compute node dispatches this frame. Nothing runs until every
/// pipeline has compiled and every registered group has a bind group; every-frame passes
/// additionally wait for all init passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_compute_run(
    config: Res<ComputeRenderConfig>,
    pipelines: Option<Res<ComputePipelines>>,
    frame: Option<Res<ComputeFrame>>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<AutoBufferLayouts>,
    bind_groups: Res<AutoBufferBindGroups>,
    mut state: ResMut<ComputeRunState>,
    mut run_list: ResMut<ComputeRunList>,
) {
    run_list.0.clear();
    let (Some(pipelines), Some(frame)) = (pipelines, frame) else {
        return;
    };

    for &index in &frame.reruns {
        state.init_done[index] = false;
    }

    // Compile errors are logged by Bevy's pipeline cache.
    let pipelines_ready = pipelines.per_pass.iter().all(|id| {
        matches!(
            pipeline_cache.get_compute_pipeline_state(*id),
            CachedPipelineState::Ok(_)
        )
    });
    if !pipelines_ready || !all_groups_bound(&layouts, &bind_groups) {
        return;
    }

    for (index, pass) in config.passes.iter().enumerate() {
        if pass.init
            && !state.init_done[index]
            && let Some(count) = frame.counts[index]
        {
            run_list.0.push((pipelines.per_pass[index], count));
            state.init_done[index] = true;
        }
    }

    let all_init_done = config
        .passes
        .iter()
        .zip(&state.init_done)
        .all(|(pass, done)| !pass.init || *done);
    if !all_init_done {
        return;
    }

    for (index, pass) in config.passes.iter().enumerate() {
        if !pass.init
            && let Some(count) = frame.counts[index]
        {
            run_list.0.push((pipelines.per_pass[index], count));
        }
    }
}
