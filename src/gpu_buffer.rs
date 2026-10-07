use std::{
    any::{TypeId, type_name},
    collections::HashMap,
    marker::PhantomData,
};

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        Render, RenderApp,
        render_asset::RenderAssets,
        render_resource::BufferUsages,
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
    },
};
use encase::{ShaderSize, internal::WriteInto};

use crate::{
    ShaderSystems,
    auto_array::default_filled_bytes,
    bindings::{AutoBufferKind, BindingTable, register_binding},
};

/// Main-world handle to a GPU-owned buffer registered with
/// [`register_gpu_buffer`](crate::ShaderAppExt::register_gpu_buffer).
///
/// The buffer lives entirely on the GPU: it starts filled with `T::default()` and is
/// never re-uploaded from the CPU, so whatever shaders write to it persists across frames.
#[derive(Resource)]
pub struct GpuBuffer<Tag> {
    handle: Handle<ShaderStorageBuffer>,
    len: u32,
    _marker: PhantomData<Tag>,
}

impl<Tag> GpuBuffer<Tag> {
    /// The storage buffer asset backing this buffer.
    pub fn handle(&self) -> &Handle<ShaderStorageBuffer> {
        &self.handle
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// `true` if the buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Element count of every registered GPU buffer, keyed by its tag type.
/// Used to size [`Workgroups::over`](crate::Workgroups::over) dispatches.
#[derive(Resource, Default)]
pub(crate) struct GpuBufferLengths(pub HashMap<TypeId, u32>);

/// Register a GPU-owned buffer. See
/// [`ShaderAppExt::register_gpu_buffer`](crate::ShaderAppExt::register_gpu_buffer).
pub(crate) fn register<Tag, T>(app: &mut App, group_index: u32, binding_index: u32, len: u32)
where
    Tag: Send + Sync + 'static,
    T: ShaderSize + WriteInto + Default + Send + Sync + 'static,
{
    let mut storage = ShaderStorageBuffer::new(
        &default_filled_bytes::<T>(len as usize),
        RenderAssetUsages::default(),
    );
    storage.buffer_description.label = Some("gpu_buffer");
    // COPY_SRC lets readback (ours, or Bevy's `Readback::buffer`) copy out of this buffer.
    storage.buffer_description.usage |= BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let handle = app
        .world_mut()
        .resource_mut::<Assets<ShaderStorageBuffer>>()
        .add(storage);
    let asset_id = handle.id();

    let previous = app
        .world_mut()
        .get_resource_or_insert_with(GpuBufferLengths::default)
        .0
        .insert(TypeId::of::<Tag>(), len);
    assert!(
        previous.is_none(),
        "register_gpu_buffer: tag `{}` is already registered; use a distinct tag per buffer",
        type_name::<Tag>()
    );

    app.insert_resource(GpuBuffer::<Tag> {
        handle,
        len,
        _marker: PhantomData,
    });

    let render_app = app.sub_app_mut(RenderApp);
    register_binding(
        render_app,
        group_index,
        binding_index,
        AutoBufferKind::Storage { read_only: false },
    );

    render_app.add_systems(
        Render,
        (move |buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
               mut table: ResMut<BindingTable>| {
            if let Some(gpu_buffer) = buffers.get(asset_id) {
                table.set_buffer(group_index, binding_index, &gpu_buffer.buffer);
            }
        })
        .in_set(ShaderSystems::PrepareBindings),
    );
}
