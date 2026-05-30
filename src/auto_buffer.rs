use std::collections::BTreeMap;

use bevy::{
    prelude::*,
    render::{
        Extract,
        render_resource::{
            BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer,
        },
        renderer::RenderDevice,
    },
};

/// Whether a registered auto-managed buffer is a uniform or read-only storage buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoBufferKind {
    Uniform,
    StorageRead,
}

/// (group → binding → kind) for all auto-managed buffers.
///
/// Populated eagerly at app-build time by [`register_uniform_buffer`](crate::FragmentAppExt::register_uniform_buffer)
/// and [`register_storage_buffer`](crate::FragmentAppExt::register_storage_buffer).
/// Read by `init_pipeline` at startup to compile per-group layouts.
#[derive(Resource, Default)]
pub struct AutoBufferLayouts(pub BTreeMap<u32, BTreeMap<u32, AutoBufferKind>>);

/// Compiled [`BindGroupLayout`] per group, populated by `init_pipeline` at startup.
///
/// Used by [`finalize_buffer_bind_groups`] each frame to create bind groups.
#[derive(Resource, Default)]
pub struct AutoBufferCompiledLayouts(pub BTreeMap<u32, BindGroupLayout>);

/// Per-frame staging: raw GPU buffer handles written by per-type prepare systems.
///
/// Keyed by `(group_index, binding_index)`. Drained and cleared by
/// [`finalize_buffer_bind_groups`] after bind groups are assembled.
#[derive(Resource, Default)]
pub struct PendingBufferBindings(pub BTreeMap<(u32, u32), Buffer>);

/// Assembled bind groups for all auto-managed buffers, keyed by WGSL group index.
///
/// Written by [`finalize_buffer_bind_groups`] each frame. Read by [`FullscreenNode`](crate::FullscreenNode).
#[derive(Resource, Default)]
pub struct AutoBufferBindGroups(pub BTreeMap<u32, BindGroup>);

/// Extraction system: copies `U` from the main world into the render world each frame.
pub(crate) fn extract_buffer<U: Resource + Clone>(
    mut commands: Commands,
    resource: Extract<Option<Res<U>>>,
) {
    if let Some(r) = resource.as_deref() {
        commands.insert_resource(r.clone());
    }
}

/// Assembles one [`BindGroup`] per registered group from the per-type buffer handles
/// stashed in [`PendingBufferBindings`]. Runs after all `PrepareBindGroups` systems.
pub(crate) fn finalize_buffer_bind_groups(
    mut auto_bind_groups: ResMut<AutoBufferBindGroups>,
    mut pending: ResMut<PendingBufferBindings>,
    compiled_layouts: Res<AutoBufferCompiledLayouts>,
    auto_layouts: Res<AutoBufferLayouts>,
    render_device: Res<RenderDevice>,
) {
    // Take all pending entries (leaves pending.0 empty for next frame).
    let taken = std::mem::take(&mut pending.0);

    // Group by group_index.
    let mut by_group: BTreeMap<u32, BTreeMap<u32, Buffer>> = BTreeMap::new();
    for ((group, binding), buffer) in taken {
        by_group.entry(group).or_default().insert(binding, buffer);
    }

    for (group_index, group_buffers) in by_group {
        // Skip if any registered binding for this group is missing.
        let Some(expected) = auto_layouts.0.get(&group_index) else {
            continue;
        };
        if !expected.keys().all(|b| group_buffers.contains_key(b)) {
            continue;
        }

        let Some(layout) = compiled_layouts.0.get(&group_index) else {
            continue;
        };

        let mut sorted: Vec<(u32, Buffer)> = group_buffers.into_iter().collect();
        sorted.sort_by_key(|(b, _)| *b);
        let entries: Vec<BindGroupEntry> = sorted
            .iter()
            .map(|(binding, buf)| BindGroupEntry {
                binding: *binding,
                resource: buf.as_entire_binding(),
            })
            .collect();

        let bind_group = render_device
            .wgpu_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("auto_buffer_bind_group"),
                layout: &*layout,
                entries: &entries,
            });

        auto_bind_groups.0.insert(group_index, bind_group.into());
    }
}
