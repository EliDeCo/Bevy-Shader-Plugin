use std::marker::PhantomData;

use bevy::{
    asset::AssetServer,
    core_pipeline::FullscreenShader,
    prelude::*,
    render::{
        render_resource::{
            BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BindGroupLayoutEntry, BindingType, BlendState, BufferBindingType,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
            MultisampleState, PipelineCache, RenderPipelineDescriptor, ShaderStages,
            ShaderType, TextureFormat,
            binding_types::uniform_buffer,
        },
    },
};

use crate::{
    FragmentExtraLayouts,
    auto_storage::{AutoStorageCompiledLayouts, AutoStorageLayouts},
};

/// Inserted during `Plugin::build` so `init_pipeline` can read the shader path
/// without being generic over U.
#[derive(Resource)]
pub struct FullscreenPipelineConfig {
    pub shader_path: &'static str,
}

/// Render-world resource created by `init_pipeline`. Holds the cached pipeline
/// ID, the per-frame bind group layout descriptor, and compiled layouts for any
/// extra bind groups (groups 1..n) registered via `FragmentExtraLayouts`.
#[derive(Resource)]
pub struct FullscreenPipeline<U: 'static> {
    pub pipeline_id: CachedRenderPipelineId,
    pub per_frame_layout: BindGroupLayoutDescriptor,
    /// Compiled `BindGroupLayout` for each extra group in the order they were
    /// pushed into `FragmentExtraLayouts`. Index 0 corresponds to GPU group 1.
    pub extra_layouts: Vec<BindGroupLayout>,
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
    auto_storage_layouts: Res<AutoStorageLayouts>,
    mut compiled_layouts: ResMut<AutoStorageCompiledLayouts>,
) {
    let per_frame_layout = BindGroupLayoutDescriptor::new(
        "fullscreen_per_frame_layout",
        &BindGroupLayoutEntries::sequential(ShaderStages::FRAGMENT, (uniform_buffer::<U>(false),)),
    );

    // Group 0 is the per-frame uniform. Auto-storage buffers occupy the next
    // groups in ascending key order, followed by any manual extra layouts.
    debug_assert!(
        auto_storage_layouts
            .0
            .keys()
            .enumerate()
            .all(|(i, &k)| k == i as u32 + 1),
        "register_storage_buffer group indices must be contiguous starting at 1"
    );

    // Build one BindGroupLayoutDescriptor per auto-storage group and compile it.
    // `BindGroupLayoutDescriptor::new` takes `&[BindGroupLayoutEntry]`, so runtime
    // binding sets work directly.
    let mut all_layouts = vec![per_frame_layout.clone()];
    for (&group_index, binding_set) in auto_storage_layouts.0.iter() {
        let entries: Vec<BindGroupLayoutEntry> = binding_set
            .iter()
            .map(|&binding| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let desc = BindGroupLayoutDescriptor::new("auto_storage_layout", &entries);
        compiled_layouts.0.insert(group_index, pipeline_cache.get_bind_group_layout(&desc));
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

    // FullscreenPipeline::extra_layouts: compiled auto-storage first, then manual.
    let extra_compiled: Vec<BindGroupLayout> = compiled_layouts
        .0
        .values()
        .cloned()
        .chain(extra_layouts.0.iter().map(|desc| pipeline_cache.get_bind_group_layout(desc)))
        .collect();

    commands.insert_resource(FullscreenPipeline::<U> {
        pipeline_id,
        per_frame_layout,
        extra_layouts: extra_compiled,
        _phantom: PhantomData,
    });
}
