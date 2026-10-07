use bevy::{
    asset::AssetServer,
    core_pipeline::FullscreenShader,
    prelude::*,
    render::render_resource::{
        BindGroupLayout, BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites,
        FragmentState, MultisampleState, PipelineCache, RenderPipelineDescriptor, TextureFormat,
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
#[derive(Resource)]
pub struct FullscreenPipeline {
    pub pipeline_id: CachedRenderPipelineId,
    /// Compiled [`BindGroupLayout`] for each manual extra group registered via
    /// [`FragmentExtraLayouts`]. Index 0 corresponds to the first manual extra group.
    pub extra_layouts: Vec<BindGroupLayout>,
}

/// `RenderStartup` system. Queues the render pipeline using the auto-buffer group layouts
/// (compiled by the shared core) followed by the manual extra groups.
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

    let shader = asset_server.load(config.shader_path);
    let vertex_state = fullscreen_shader.to_vertex_state();

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("fullscreen_fragment_pipeline".into()),
        layout: all_layouts,
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader,
            entry_point: config.entry_point.map(Into::into),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::bevy_default(),
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        ..default()
    });

    let extra_compiled: Vec<BindGroupLayout> = extra_layouts
        .0
        .iter()
        .map(|desc| pipeline_cache.get_bind_group_layout(desc))
        .collect();

    commands.insert_resource(FullscreenPipeline {
        pipeline_id,
        extra_layouts: extra_compiled,
    });
}
