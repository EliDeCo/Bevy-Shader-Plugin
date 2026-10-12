use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp,
        render_resource::{
            Buffer, BufferUsages, ShaderType, StorageBuffer, UniformBuffer,
            encase::internal::WriteInto,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

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

    /// The value uploaded by the next [`upload`](Self::upload).
    fn value_mut(&mut self) -> &mut T {
        match &mut self.buffer {
            SlotBuffer::Uniform(buffer) => buffer.get_mut(),
            SlotBuffer::Storage(buffer) => buffer.get_mut(),
        }
    }

    fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.written_size = match &mut self.buffer {
            SlotBuffer::Uniform(buffer) => {
                buffer.write_buffer(device, queue);
                buffer.get().size().get()
            }
            SlotBuffer::Storage(buffer) => {
                buffer.write_buffer(device, queue);
                buffer.get().size().get()
            }
        };
    }

    fn gpu_buffer(&self) -> Option<&Buffer> {
        match &self.buffer {
            SlotBuffer::Uniform(buffer) => buffer.buffer(),
            SlotBuffer::Storage(buffer) => buffer.buffer(),
        }
    }
}

/// Render-world state for a CPU-owned resource `T`: one persistent GPU buffer per
/// registration of `T`, each holding its own copy of the latest extracted value.
#[derive(Resource)]
pub(crate) struct CpuBuffer<T: ShaderType + Send + Sync + 'static> {
    /// Set when a new value was extracted and hasn't been uploaded yet.
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
    register_binding(
        render_app,
        group_index,
        binding_index,
        kind,
        Some(T::min_size()),
    );

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
/// (including the first frame it exists). It's copied straight into each slot's buffer
/// value with `clone_from`, which can reuse memory the previous value held.
fn extract_cpu_buffer<T>(main_resource: Extract<Option<Res<T>>>, mut state: ResMut<CpuBuffer<T>>)
where
    T: ShaderType + WriteInto + Default + Resource + Clone,
{
    if let Some(resource) = main_resource.as_ref()
        && resource.is_changed()
    {
        let value: &T = resource;
        let state = &mut *state;
        for slot in &mut state.slots {
            slot.value_mut().clone_from(value);
        }
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
        for slot in &mut state.slots {
            slot.upload(&render_device, &render_queue);
        }
        state.dirty = false;
    }

    for slot in &state.slots {
        if let Some(buffer) = slot.gpu_buffer() {
            table.set_buffer(slot.group_index, slot.binding_index, buffer);
        }
    }
}
