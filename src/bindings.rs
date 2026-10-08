use std::collections::{BTreeMap, BTreeSet};

use bevy::{
    app::SubApp,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
            PipelineCache, ShaderStages,
        },
        renderer::RenderDevice,
    },
};

/// Shader stages that can see auto-managed bindings. Every registered binding is shared
/// by the fullscreen fragment pipeline and all compute pipelines.
pub(crate) const AUTO_BINDING_VISIBILITY: ShaderStages =
    ShaderStages::FRAGMENT.union(ShaderStages::COMPUTE);

/// Whether a registered auto-managed buffer is a uniform or a storage buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoBufferKind {
    Uniform,
    Storage { read_only: bool },
}

/// (group → binding → kind) for all auto-managed buffers.
///
/// Populated eagerly at app-build time by the [`ShaderAppExt`](crate::ShaderAppExt)
/// registration methods. Read at startup to compile per-group layouts.
#[derive(Resource, Default)]
pub struct AutoBufferLayouts(pub BTreeMap<u32, BTreeMap<u32, AutoBufferKind>>);

/// Compiled [`BindGroupLayout`] per group, populated at startup.
///
/// Used in [`ShaderSystems::FinalizeBindGroups`](crate::ShaderSystems::FinalizeBindGroups) to create bind groups.
#[derive(Resource, Default)]
pub struct AutoBufferCompiledLayouts(pub BTreeMap<u32, BindGroupLayout>);

/// Layout descriptors for every auto-managed group, in group order, populated at startup.
///
/// Both the fragment pipeline and the compute pipelines build their pipeline layouts
/// from this list, so they agree on every group.
#[derive(Resource, Default)]
pub struct AutoBufferLayoutDescriptors(pub Vec<BindGroupLayoutDescriptor>);

/// Assembled bind groups for all auto-managed buffers, keyed by WGSL group index.
///
/// Rebuilt in [`ShaderSystems::FinalizeBindGroups`](crate::ShaderSystems::FinalizeBindGroups) only when a binding in the group changes.
/// Read by [`fullscreen_pass`](crate::fullscreen_pass) and the compute pass.
#[derive(Resource, Default)]
pub struct AutoBufferBindGroups(pub BTreeMap<u32, BindGroup>);

/// A GPU resource bound at one `(group, binding)` slot.
#[derive(Clone)]
pub enum BoundResource {
    Buffer(Buffer),
    // Storage textures will add a `TextureView` variant here.
}

impl BoundResource {
    fn binding(&self) -> BindingResource<'_> {
        match self {
            BoundResource::Buffer(buffer) => buffer.as_entire_binding(),
        }
    }
}

/// Persistent `(group, binding)` → GPU resource table for all auto-managed bindings.
///
/// Each registered buffer records its GPU handle here every frame. A group is only marked
/// dirty (and its bind group rebuilt) when one of its handles actually changes, e.g. the
/// first frame or after a buffer is reallocated to a larger size.
#[derive(Resource, Default)]
pub struct BindingTable {
    entries: BTreeMap<(u32, u32), BoundResource>,
    dirty_groups: BTreeSet<u32>,
}

impl BindingTable {
    /// Record `buffer` at `(group, binding)`. Does nothing if that exact buffer is
    /// already recorded there.
    pub fn set_buffer(&mut self, group: u32, binding: u32, buffer: &Buffer) {
        let unchanged = matches!(
            self.entries.get(&(group, binding)),
            Some(BoundResource::Buffer(existing)) if existing.id() == buffer.id()
        );
        if !unchanged {
            self.entries
                .insert((group, binding), BoundResource::Buffer(buffer.clone()));
            self.dirty_groups.insert(group);
        }
    }
}

/// Initialise all binding resources in the render world. Idempotent.
pub(crate) fn init_resources(render_app: &mut SubApp) {
    render_app.init_resource::<AutoBufferLayouts>();
    render_app.init_resource::<AutoBufferCompiledLayouts>();
    render_app.init_resource::<AutoBufferLayoutDescriptors>();
    render_app.init_resource::<AutoBufferBindGroups>();
    render_app.init_resource::<BindingTable>();
}

/// Record that `(group_index, binding_index)` holds a buffer of `kind`.
pub(crate) fn register_binding(
    render_app: &mut SubApp,
    group_index: u32,
    binding_index: u32,
    kind: AutoBufferKind,
) {
    init_resources(render_app);
    render_app
        .world_mut()
        .resource_mut::<AutoBufferLayouts>()
        .0
        .entry(group_index)
        .or_default()
        .insert(binding_index, kind);
}

/// `RenderStartup` system. Builds one bind group layout per registered group.
pub(crate) fn compile_layouts(
    pipeline_cache: Res<PipelineCache>,
    auto_buffer_layouts: Res<AutoBufferLayouts>,
    mut compiled_layouts: ResMut<AutoBufferCompiledLayouts>,
    mut descriptors: ResMut<AutoBufferLayoutDescriptors>,
) {
    // Validate: registered group indices must be contiguous (no gaps).
    let keys: Vec<u32> = auto_buffer_layouts.0.keys().cloned().collect();
    debug_assert!(
        keys.windows(2).all(|w| w[1] == w[0] + 1),
        "register_uniform_buffer/register_storage_buffer/register_array_buffer/register_gpu_buffer group indices must be contiguous (no gaps)"
    );

    for (&group_index, binding_map) in auto_buffer_layouts.0.iter() {
        let entries: Vec<BindGroupLayoutEntry> = binding_map
            .iter()
            .map(|(&binding, &kind)| BindGroupLayoutEntry {
                binding,
                visibility: AUTO_BINDING_VISIBILITY,
                ty: match kind {
                    AutoBufferKind::Uniform => BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    AutoBufferKind::Storage { read_only } => BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only },
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
        descriptors.0.push(desc);
    }
}

/// Rebuilds the bind group of every dirty group whose registered bindings are all present.
/// Groups that are still missing a binding stay dirty and are retried next frame.
pub(crate) fn finalize_bind_groups(
    mut auto_bind_groups: ResMut<AutoBufferBindGroups>,
    mut table: ResMut<BindingTable>,
    compiled_layouts: Res<AutoBufferCompiledLayouts>,
    auto_layouts: Res<AutoBufferLayouts>,
    render_device: Res<RenderDevice>,
) {
    if table.dirty_groups.is_empty() {
        return;
    }
    let BindingTable {
        entries,
        dirty_groups,
    } = &mut *table;

    dirty_groups.retain(|&group_index| {
        let (Some(expected), Some(layout)) = (
            auto_layouts.0.get(&group_index),
            compiled_layouts.0.get(&group_index),
        ) else {
            return true;
        };
        let Some(bind_entries) = expected
            .keys()
            .map(|&binding| {
                entries
                    .get(&(group_index, binding))
                    .map(|resource| BindGroupEntry {
                        binding,
                        resource: resource.binding(),
                    })
            })
            .collect::<Option<Vec<_>>>()
        else {
            return true;
        };

        let bind_group =
            render_device.create_bind_group("auto_buffer_bind_group", layout, &bind_entries);
        auto_bind_groups.0.insert(group_index, bind_group);
        false
    });
}

/// `true` once every registered group has a bind group.
pub(crate) fn all_groups_bound(
    layouts: &AutoBufferLayouts,
    bind_groups: &AutoBufferBindGroups,
) -> bool {
    layouts
        .0
        .keys()
        .all(|group| bind_groups.0.contains_key(group))
}
