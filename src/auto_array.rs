use std::{collections::BTreeMap, marker::PhantomData};

use bevy::{
    prelude::*,
    render::{MainWorld, render_resource::Buffer, renderer::RenderQueue},
};
use encase::ShaderSize;
use encase::internal::WriteInto;

use crate::auto_buffer::PendingBufferBindings;

/// Main-world resource for queuing index-value changes to a fixed-size array buffer.
///
/// `Tag` is a zero-sized marker type that uniquely identifies this buffer, allowing
/// multiple buffers of the same element type and length to coexist. The element type
/// and length are specified at registration and do not appear in this type.
///
/// Changes are serialized to bytes immediately in [`set`](Self::set), then batched
/// into contiguous `write_buffer` runs each frame. Indices need not be sorted or
/// unique — duplicates use last-write-wins.
///
/// # Example
///
/// ```rust,ignore
/// struct Colors;
/// // registered via: app.register_array_buffer::<Colors, Vec4, 64>(0, 0, false)
///
/// fn update(mut changes: ResMut<ArrayBufferChanges<Colors>>) {
///     changes.set(0, Vec4::new(1.0, 0.0, 0.0, 1.0)); // red at index 0
///     changes.set_many([(1, Vec4::Y), (2, Vec4::Z)]); // green, blue
/// }
/// ```
#[derive(Resource)]
pub struct ArrayBufferChanges<Tag> {
    pub(crate) changes: Vec<(usize, Vec<u8>)>,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<Tag>,
}

impl<Tag> Default for ArrayBufferChanges<Tag> {
    fn default() -> Self {
        Self {
            changes: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }
}

fn serialize_element<T: ShaderSize + WriteInto>(value: &T) -> Vec<u8> {
    let el_size = T::SHADER_SIZE.get() as usize;
    let mut bytes = vec![0u8; el_size];
    encase::StorageBuffer::new(&mut bytes[..])
        .write(value)
        .unwrap();
    bytes
}

impl<Tag> ArrayBufferChanges<Tag> {
    /// Queue a change: element at `index` will be updated to `value` this frame.
    /// The value is serialized to bytes immediately.
    pub fn set<T: ShaderSize + WriteInto>(&mut self, index: usize, value: T) {
        if index >= self.len {
            panic!(
                "ArrayBufferChanges::set index out of bounds: {} >= {}",
                index, self.len
            );
        }
        let bytes = serialize_element(&value);
        self.changes.push((index, bytes));
    }

    /// Queue multiple changes at once from any iterable of `(index, value)` pairs.
    ///
    /// Equivalent to calling [`set`](Self::set) for each pair. Accepts arrays, vecs,
    /// iterators, or anything that implements `IntoIterator<Item = (usize, T)>`:
    ///
    /// ```rust,ignore
    /// changes.set_many([(8, color_a), (9, color_a), (24, color_b), (25, color_b)]);
    /// changes.set_many((0..4).map(|i| (i, Vec4::ONE)));
    /// ```
    pub fn set_many<T, I>(&mut self, pairs: I)
    where
        T: ShaderSize + WriteInto,
        I: IntoIterator<Item = (usize, T)>,
    {
        for (index, value) in pairs {
            self.set(index, value);
        }
    }

    /// Queue a change to set every element in the buffer to `value`.
    /// The value is serialized once and the bytes are reused for each element.
    pub fn set_all<T: ShaderSize + WriteInto>(&mut self, value: T) {
        let bytes = serialize_element(&value);
        self.changes.reserve(self.len);
        for i in 0..self.len {
            self.changes.push((i, bytes.clone()));
        }
    }
}

/// Render-world resource holding the persistent GPU buffer and element stride.
#[derive(Resource)]
pub struct ArrayBufferState<Tag> {
    pub(crate) buffer: Buffer,
    /// WGSL array element stride in bytes (includes any alignment padding).
    pub(crate) stride: usize,
    pub(crate) _marker: PhantomData<Tag>,
}

/// `ExtractSchedule` system: moves queued changes from main world to render world.
///
/// Uses `mem::take` so the main-world vec is left empty after extraction — no
/// separate clear system is needed, and changes set during `Startup` survive to
/// the first extraction without being wiped by a `First`-schedule clear.
pub(crate) fn extract_array_changes<Tag: Send + Sync + 'static>(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
) {
    let extracted = {
        let Some(mut changes) = main_world.get_resource_mut::<ArrayBufferChanges<Tag>>() else {
            return;
        };
        (std::mem::take(&mut changes.changes), changes.len)
    };
    commands.insert_resource(ArrayBufferChanges::<Tag> {
        changes: extracted.0,
        len: extracted.1,
        _marker: PhantomData,
    });
}

/// Applies pending changes to the GPU buffer via contiguous-run `write_buffer` batching,
/// then inserts the persistent buffer handle into `PendingBufferBindings`.
pub(crate) fn apply_array_buffer_changes<Tag: Send + Sync + 'static>(
    state: &ArrayBufferState<Tag>,
    changes: &ArrayBufferChanges<Tag>,
    render_queue: &RenderQueue,
    pending: &mut PendingBufferBindings,
    group_index: u32,
    binding_index: u32,
) {
    let stride = state.stride;

    // Deduplicate: last .set() wins per index; BTreeMap gives sorted order.
    let mut by_index: BTreeMap<usize, &[u8]> = BTreeMap::new();
    for (i, bytes) in &changes.changes {
        by_index.insert(*i, bytes);
    }

    // Group consecutive indices into runs; each run becomes one write_buffer call.
    let mut runs: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut run_bytes: Vec<u8> = Vec::new();
    let mut prev: Option<usize> = None;

    for (&index, &el_bytes) in &by_index {
        // Pad element bytes to stride (tail bytes remain zero).
        let mut padded = vec![0u8; stride];
        padded[..el_bytes.len()].copy_from_slice(el_bytes);

        match prev {
            Some(p) if index == p + 1 => {
                run_bytes.extend_from_slice(&padded);
            }
            _ => {
                if let Some(start) = run_start {
                    runs.push((start as u64 * stride as u64, std::mem::take(&mut run_bytes)));
                }
                run_start = Some(index);
                run_bytes = padded;
            }
        }
        prev = Some(index);
    }
    if let Some(start) = run_start {
        runs.push((start as u64 * stride as u64, run_bytes));
    }

    for (offset, bytes) in runs {
        render_queue.write_buffer(&state.buffer, offset, &bytes);
    }

    pending
        .0
        .insert((group_index, binding_index), state.buffer.clone());
}
