use bevy::{
    asset::AssetServer,
    core_pipeline::FullscreenShader,
    prelude::*,
    render::render_resource::{
        BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState,
        BufferBindingType, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
        MultisampleState, PipelineCache, RenderPipelineDescriptor, ShaderStages, TextureFormat,
    },
};

use crate::{
    FragmentExtraLayouts,
    auto_buffer::{AutoBufferCompiledLayouts, AutoBufferKind, AutoBufferLayouts},
};

/// Inserted during `Plugin::build` so `init_pipeline` can read the shader path.
#[derive(Resource)]
pub struct FullscreenPipelineConfig {
    pub shader_path: &'static str,
}

/// Render-world resource created by `init_pipeline`.
#[derive(Resource)]
pub struct FullscreenPipeline {
    pub pipeline_id: CachedRenderPipelineId,
    /// Compiled [`BindGroupLayout`] for each manual extra group registered via
    /// [`FragmentExtraLayouts`]. Index 0 corresponds to the first manual extra group.
    pub extra_layouts: Vec<BindGroupLayout>,
}

/// `RenderStartup` system. Builds bind group layouts for all registered auto-buffer
/// groups and manual extra groups, then queues the render pipeline.
pub(crate) fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    config: Res<FullscreenPipelineConfig>,
    extra_layouts: Res<FragmentExtraLayouts>,
    auto_buffer_layouts: Res<AutoBufferLayouts>,
    mut compiled_layouts: ResMut<AutoBufferCompiledLayouts>,
) {
    // Validate: registered group indices must be contiguous (no gaps).
    let keys: Vec<u32> = auto_buffer_layouts.0.keys().cloned().collect();
    debug_assert!(
        keys.windows(2).all(|w| w[1] == w[0] + 1),
        "register_uniform_buffer/register_storage_buffer group indices must be contiguous (no gaps)"
    );

    // Build one BindGroupLayoutDescriptor per auto-buffer group.
    let mut all_layouts: Vec<BindGroupLayoutDescriptor> = Vec::new();
    for (&group_index, binding_map) in auto_buffer_layouts.0.iter() {
        let entries: Vec<BindGroupLayoutEntry> = binding_map
            .iter()
            .map(|(&binding, &kind)| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::FRAGMENT,
                ty: match kind {
                    AutoBufferKind::Uniform => BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    AutoBufferKind::StorageRead => BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                count: None,
            })
            .collect();
        let desc = BindGroupLayoutDescriptor::new("auto_buffer_layout", &entries);
        compiled_layouts
            .0
            .insert(group_index, pipeline_cache.get_bind_group_layout(&desc));
        all_layouts.push(desc);
    }
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
