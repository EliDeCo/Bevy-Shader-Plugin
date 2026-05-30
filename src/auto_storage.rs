use std::collections::BTreeMap;

use bevy::{
    prelude::*,
    render::{Extract, render_resource::{BindGroup, BindGroupLayoutDescriptor}},
};

/// Layout descriptors for auto-managed storage buffers, keyed by WGSL group index.
///
/// Populated eagerly at app-build time by [`register_storage_buffer`](crate::FragmentAppExt::register_storage_buffer).
/// Read by `init_pipeline` in `RenderStartup` before the pipeline is compiled.
#[derive(Resource, Default)]
pub struct AutoStorageLayouts(pub BTreeMap<u32, BindGroupLayoutDescriptor>);

/// Compiled bind groups for auto-managed storage buffers, keyed by WGSL group index.
///
/// Written each frame by the per-type prepare closures registered via
/// [`register_storage_buffer`](crate::FragmentAppExt::register_storage_buffer).
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
