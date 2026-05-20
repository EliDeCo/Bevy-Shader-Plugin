use std::marker::PhantomData;

use bevy::{
    asset::AssetServer,
    core_pipeline::FullscreenShader,
    prelude::*,
    render::render_resource::{
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, FragmentState, MultisampleState, PipelineCache,
        RenderPipelineDescriptor, ShaderStages, ShaderType, TextureFormat,
        binding_types::uniform_buffer,
    },
};

use crate::FragmentExtraLayouts;

/// Inserted during `Plugin::build` so `init_pipeline` can read the shader path
/// without being generic over U.
#[derive(Resource)]
pub struct FullscreenPipelineConfig {
    pub shader_path: &'static str,
}

/// Render-world resource created by `init_pipeline`. Holds the cached pipeline
/// ID and the per-frame bind group layout descriptor so `prepare_bind_group`
/// can retrieve the layout from the cache each frame.
#[derive(Resource)]
pub struct FullscreenPipeline<U: 'static> {
    pub pipeline_id: CachedRenderPipelineId,
    pub per_frame_layout: BindGroupLayoutDescriptor,
    _phantom: PhantomData<U>,
}

/// `RenderStartup` system. Reads `FragmentExtraLayouts` to build the full
/// pipeline bind group layout, then queues the render pipeline.
///
/// User systems that insert `FragmentExtraLayouts` must be ordered before this
/// system (`.before(FragmentSystems::InitPipeline)`).
pub(crate) fn init_pipeline<U: ShaderType + encase::internal::WriteInto + Send + Sync + 'static>(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    config: Res<FullscreenPipelineConfig>,
    extra_layouts: Res<FragmentExtraLayouts>,
) {
    let per_frame_layout = BindGroupLayoutDescriptor::new(
        "fullscreen_per_frame_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (uniform_buffer::<U>(false),),
        ),
    );

    // Group 0 is always the per-frame uniform. Groups 1..n are user-supplied.
    let mut all_layouts = vec![per_frame_layout.clone()];
    all_layouts.extend(extra_layouts.0.iter().cloned());

    let shader = asset_server.load(config.shader_path);
    let vertex_state = fullscreen_shader.to_vertex_state();

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("fullscreen_fragment_pipeline".into()),
        layout: all_layouts,
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader,
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

    commands.insert_resource(FullscreenPipeline::<U> {
        pipeline_id,
        per_frame_layout,
        _phantom: PhantomData,
    });
}
