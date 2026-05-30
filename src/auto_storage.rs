use std::collections::{BTreeMap, BTreeSet};

use bevy::{
    prelude::*,
    render::{
        Extract,
        render_resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer},
        renderer::RenderDevice,
    },
};

/// Registered binding indices per group, keyed by WGSL group index.
///
/// Populated eagerly at app-build time by [`register_storage_buffer`](crate::FragmentAppExt::register_storage_buffer).
/// Read by `init_pipeline` to compile per-group [`BindGroupLayout`]s.
#[derive(Resource, Default)]
pub struct AutoStorageLayouts(pub BTreeMap<u32, BTreeSet<u32>>);

/// Compiled [`BindGroupLayout`] per group, populated by `init_pipeline` at startup.
///
/// Used by [`finalize_storage_bind_groups`] each frame to create bind groups.
#[derive(Resource, Default)]
pub struct AutoStorageCompiledLayouts(pub BTreeMap<u32, BindGroupLayout>);

/// Per-frame staging: raw GPU buffer handles written by per-type prepare systems.
///
/// Keyed by `(group_index, binding_index)`. Drained and cleared by
/// [`finalize_storage_bind_groups`] after bind groups are assembled.
#[derive(Resource, Default)]
pub struct PendingStorageBindings(pub BTreeMap<(u32, u32), Buffer>);

/// Assembled bind groups for auto-managed storage buffers, keyed by WGSL group index.
///
/// Written by [`finalize_storage_bind_groups`] each frame. Read by [`FullscreenNode`](crate::FullscreenNode).
#[derive(Resource, Default)]
pub struct AutoStorageBindGroups(pub BTreeMap<u32, BindGroup>);

/// Extraction system: copies `S` from the main world into the render world each frame.
pub(crate) fn extract_storage<S: Resource + Clone>(
    mut commands: Commands,
    resource: Extract<Option<Res<S>>>,
) {
    if let Some(r) = resource.as_deref() {
        commands.insert_resource(r.clone());
    }
}

/// Assembles one [`BindGroup`] per registered group from the per-type buffer handles
/// stashed in [`PendingStorageBindings`]. Runs after all `PrepareBindGroups` systems.
pub(crate) fn finalize_storage_bind_groups(
    mut auto_bind_groups: ResMut<AutoStorageBindGroups>,
    mut pending: ResMut<PendingStorageBindings>,
    compiled_layouts: Res<AutoStorageCompiledLayouts>,
    auto_layouts: Res<AutoStorageLayouts>,
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
        // Skip if any registered binding for this group is missing (resource not yet inserted).
        let Some(expected) = auto_layouts.0.get(&group_index) else { continue };
        if !expected.iter().all(|b| group_buffers.contains_key(b)) { continue }

        let Some(layout) = compiled_layouts.0.get(&group_index) else { continue };

        // Build wgpu entries sorted by binding index.
        let mut sorted: Vec<(u32, Buffer)> = group_buffers.into_iter().collect();
        sorted.sort_by_key(|(b, _)| *b);
        let entries: Vec<BindGroupEntry> = sorted
            .iter()
            .map(|(binding, buf)| BindGroupEntry {
                binding: *binding,
                resource: buf.as_entire_binding(),
            })
            .collect();

        // Use the raw wgpu device so we can pass a dynamic slice of entries.
        // BindGroupLayout: Deref<Target = wgpu::BindGroupLayout>, so &*layout coerces.
        let bind_group = render_device
            .wgpu_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("auto_storage_bind_group"),
                layout: &*layout,
                entries: &entries,
            });

        auto_bind_groups.0.insert(group_index, bind_group.into());
    }
}
