use std::collections::HashMap;

use bevy::{
    asset::AssetServer,
    core_pipeline::FullscreenShader,
    prelude::*,
    render::{
        render_resource::{
            BindGroupLayout, BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            FragmentState, MultisampleState, PipelineCache, RenderPipelineDescriptor,
            TextureFormat,
        },
        view::ExtractedView,
    },
};

use crate::{FragmentExtraLayouts, bindings::AutoBufferLayoutDescriptors};

/// Inserted during `Plugin::build` so `init_pipeline` can read the shader path
/// and entry point.
#[derive(Resource)]
pub struct FullscreenPipelineConfig {
    pub shader_path: &'static str,
    pub entry_point: Option<&'static str>,
}

/// Render-world resource created by `init_pipeline`.
///
/// A camera's texture format depends on its window surface (and on HDR), so one
/// pipeline is queued per distinct format the cameras use.
#[derive(Resource)]
pub struct FullscreenPipeline {
    /// Pipeline description shared by every format; only the color target differs.
    descriptor: RenderPipelineDescriptor,
    pipelines: HashMap<TextureFormat, CachedRenderPipelineId>,
    /// Compiled [`BindGroupLayout`] for each manual extra group registered via
    /// [`FragmentExtraLayouts`]. Index 0 corresponds to the first manual extra group.
    pub extra_layouts: Vec<BindGroupLayout>,
}

impl FullscreenPipeline {
    /// The pipeline for views rendering to `format`, once it has been queued.
    pub fn pipeline_id(&self, format: TextureFormat) -> Option<CachedRenderPipelineId> {
        self.pipelines.get(&format).copied()
    }
}

/// `RenderStartup` system. Prepares the pipeline description using the auto-buffer group
/// layouts (compiled by the shared core) followed by the manual extra groups. Pipelines
/// are queued per camera format by [`queue_fullscreen_pipelines`].
pub(crate) fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    config: Res<FullscreenPipelineConfig>,
    extra_layouts: Res<FragmentExtraLayouts>,
    auto_buffer_descriptors: Res<AutoBufferLayoutDescriptors>,
) {
    let mut all_layouts = auto_buffer_descriptors.0.clone();
    all_layouts.extend(extra_layouts.0.iter().cloned());

    let descriptor = RenderPipelineDescriptor {
        label: Some("fullscreen_fragment_pipeline".into()),
        layout: all_layouts,
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load(config.shader_path),
            entry_point: config.entry_point.map(Into::into),
            // Filled in per camera format by `queue_fullscreen_pipelines`.
            targets: Vec::new(),
            ..default()
        }),
        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        ..default()
    };

    let extra_compiled: Vec<BindGroupLayout> = extra_layouts
        .0
        .iter()
        .map(|desc| pipeline_cache.get_bind_group_layout(desc))
        .collect();

    commands.insert_resource(FullscreenPipeline {
        descriptor,
        pipelines: HashMap::new(),
        extra_layouts: extra_compiled,
    });
}

/// Queues a pipeline for every camera texture format that doesn't have one yet.
pub(crate) fn queue_fullscreen_pipelines(
    pipeline: Option<ResMut<FullscreenPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<&ExtractedView>,
) {
    let Some(mut pipeline) = pipeline else {
        return;
    };
    for view in &views {
        let format = view.target_format;
        if pipeline.pipelines.contains_key(&format) {
            continue;
        }
        let mut descriptor = pipeline.descriptor.clone();
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.targets = vec![Some(ColorTargetState {
                format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })];
        }
        let id = pipeline_cache.queue_render_pipeline(descriptor);
        pipeline.pipelines.insert(format, id);
    }
}
