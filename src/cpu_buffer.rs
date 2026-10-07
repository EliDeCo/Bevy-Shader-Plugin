use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp,
        render_resource::{Buffer, BufferUsages, ShaderType, StorageBuffer, UniformBuffer},
        renderer::{RenderDevice, RenderQueue},
    },
};
use encase::internal::WriteInto;

use crate::{
    ShaderSystems,
    bindings::{AutoBufferKind, BindingTable, register_binding},
};

/// The persistent GPU buffer behind one registration of a CPU-owned resource.
enum SlotBuffer<T: ShaderType> {
    Uniform(UniformBuffer<T>),
    Storage(StorageBuffer<T>),
}

struct Slot<T: ShaderType> {
    group_index: u32,
    binding_index: u32,
    kind: AutoBufferKind,
    buffer: SlotBuffer<T>,
    /// Size in bytes of the last value written. The GPU buffer can be larger after the
    /// value shrinks, so readback copies only this many bytes.
    written_size: u64,
}

impl<T: ShaderType + WriteInto + Default> Slot<T> {
    fn new(group_index: u32, binding_index: u32, kind: AutoBufferKind) -> Self {
        let buffer = match kind {
            AutoBufferKind::Uniform => {
                let mut buffer = UniformBuffer::default();
                buffer.set_label(Some("auto_uniform_buffer"));
                SlotBuffer::Uniform(buffer)
            }
            AutoBufferKind::Storage { .. } => {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("auto_storage_buffer"));
                // COPY_SRC lets a future GPU→CPU readback copy out of this buffer.
                // Set before the first write: changing usages forces a reallocation.
                buffer.add_usages(BufferUsages::COPY_SRC);
                SlotBuffer::Storage(buffer)
            }
        };
        Self {
            group_index,
            binding_index,
            kind,
            buffer,
            written_size: 0,
        }
    }

    fn write(&mut self, value: T, device: &RenderDevice, queue: &RenderQueue) {
        self.written_size = value.size().get();
        match &mut self.buffer {
            SlotBuffer::Uniform(buffer) => {
                buffer.set(value);
                buffer.write_buffer(device, queue);
            }
            SlotBuffer::Storage(buffer) => {
                buffer.set(value);
                buffer.write_buffer(device, queue);
            }
        }
    }

    fn gpu_buffer(&self) -> Option<&Buffer> {
        match &self.buffer {
            SlotBuffer::Uniform(buffer) => buffer.buffer(),
            SlotBuffer::Storage(buffer) => buffer.buffer(),
        }
    }
}

/// Render-world state for a CPU-owned resource `T`: the latest extracted value and one
/// persistent GPU buffer per registration of `T`.
#[derive(Resource)]
pub(crate) struct CpuBuffer<T: ShaderType + Send + Sync + 'static> {
    value: Option<T>,
    dirty: bool,
    slots: Vec<Slot<T>>,
}

impl<T: ShaderType + WriteInto + Default + Send + Sync + 'static> CpuBuffer<T> {
    /// The kind of every registration of `T`.
    pub(crate) fn slot_kinds(&self) -> impl Iterator<Item = AutoBufferKind> + '_ {
        self.slots.iter().map(|slot| slot.kind)
    }

    /// The buffer and byte count to read back: the first storage registration of `T`,
    /// once it has been written.
    pub(crate) fn readback_source(&self) -> Option<(&Buffer, u64)> {
        let slot = self
            .slots
            .iter()
            .find(|slot| matches!(slot.kind, AutoBufferKind::Storage { .. }))?;
        let buffer = slot.gpu_buffer()?;
        (slot.written_size > 0).then_some((buffer, slot.written_size))
    }
}

impl<T: ShaderType + Send + Sync + 'static> Default for CpuBuffer<T> {
    fn default() -> Self {
        Self {
            value: None,
            dirty: false,
            slots: Vec::new(),
        }
    }
}

/// Register main-world resource `T` as a CPU-owned buffer at `(group_index, binding_index)`.
pub(crate) fn register<T>(app: &mut App, group_index: u32, binding_index: u32, kind: AutoBufferKind)
where
    T: ShaderType + WriteInto + Default + Resource + Clone,
{
    let render_app = app.sub_app_mut(RenderApp);
    register_binding(render_app, group_index, binding_index, kind);

    let world = render_app.world_mut();
    let first_registration = !world.contains_resource::<CpuBuffer<T>>();
    world
        .get_resource_or_insert_with(CpuBuffer::<T>::default)
        .slots
        .push(Slot::new(group_index, binding_index, kind));

    // One extract/prepare pair per type serves every slot registered for it.
    if first_registration {
        render_app.add_systems(ExtractSchedule, extract_cpu_buffer::<T>);
        render_app.add_systems(
            Render,
            prepare_cpu_buffer::<T>.in_set(ShaderSystems::PrepareBindings),
        );
    }
}

/// Copies `T` into the render world only when it changed in the main world
/// (including the first frame it exists).
fn extract_cpu_buffer<T>(main_resource: Extract<Option<Res<T>>>, mut state: ResMut<CpuBuffer<T>>)
where
    T: ShaderType + Resource + Clone,
{
    if let Some(resource) = main_resource.as_ref()
        && resource.is_changed()
    {
        let value: &T = resource;
        state.value = Some(value.clone());
        state.dirty = true;
    }
}

/// Uploads `T` into its persistent buffers when it changed, then records the buffers in
/// the [`BindingTable`] (a no-op unless a buffer was reallocated).
fn prepare_cpu_buffer<T>(
    mut state: ResMut<CpuBuffer<T>>,
    mut table: ResMut<BindingTable>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) where
    T: ShaderType + WriteInto + Default + Resource + Clone,
{
    let state = &mut *state;
    if state.dirty {
        if let Some(value) = &state.value {
            for slot in &mut state.slots {
                slot.write(value.clone(), &render_device, &render_queue);
            }
        }
        state.dirty = false;
    }

    for slot in &state.slots {
        if let Some(buffer) = slot.gpu_buffer() {
            table.set_buffer(slot.group_index, slot.binding_index, buffer);
        }
    }
}
