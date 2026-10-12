use std::{
    marker::PhantomData,
    num::NonZeroU64,
    sync::{Mutex, PoisonError},
};

use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup,
        render_resource::{
            Buffer, BufferInitDescriptor, BufferUsages,
            encase::{self, ShaderSize, internal::WriteInto},
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

use crate::{
    ShaderSystems,
    bindings::{AutoBufferKind, BindingTable, register_binding},
};

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
    /// Behind a mutex only so extraction can take the queue through a shared reference.
    /// The setters have `&mut self` and use `get_mut`, which doesn't lock.
    queue: Mutex<ChangeQueue>,
    len: usize,
    stride: usize,
    _marker: PhantomData<Tag>,
}

impl<Tag> Default for ArrayBufferChanges<Tag> {
    fn default() -> Self {
        Self {
            queue: Mutex::default(),
            len: 0,
            stride: 0,
            _marker: PhantomData,
        }
    }
}

/// Changes queued for one array buffer, with every element serialized to `stride` bytes.
#[derive(Default)]
struct ChangeQueue {
    /// Set by `set_all`: every element becomes `fill` before `indices` are applied.
    has_fill: bool,
    fill: Vec<u8>,
    /// Element indices in the order they were set. Element `i` of this list has its bytes
    /// at `bytes[i * stride..(i + 1) * stride]`.
    indices: Vec<usize>,
    bytes: Vec<u8>,
}

impl ChangeQueue {
    fn is_empty(&self) -> bool {
        !self.has_fill && self.indices.is_empty()
    }

    /// Empties the queue, keeping its allocations.
    fn clear(&mut self) {
        self.has_fill = false;
        self.fill.clear();
        self.indices.clear();
        self.bytes.clear();
    }

    /// Serializes `value` into a new `stride`-byte slot at the end of `bytes`.
    fn push<T: ShaderSize + WriteInto>(&mut self, index: usize, value: &T, stride: usize) {
        let start = self.bytes.len();
        self.bytes.resize(start + stride, 0);
        write_element(value, &mut self.bytes[start..]);
        self.indices.push(index);
    }

    /// Moves everything queued in `newer` onto the end of `self`, leaving `newer` empty.
    fn append(&mut self, newer: &mut ChangeQueue) {
        if self.is_empty() || newer.has_fill {
            // Nothing to keep: a fill replaces every earlier change.
            std::mem::swap(self, newer);
        } else {
            self.indices.append(&mut newer.indices);
            self.bytes.append(&mut newer.bytes);
        }
        newer.clear();
    }
}

/// Serializes `value` into the start of `out`, which must be at least `T::SHADER_SIZE` bytes.
fn write_element<T: ShaderSize + WriteInto>(value: &T, out: &mut [u8]) {
    encase::StorageBuffer::new(out).write(value).unwrap();
}

/// WGSL array element stride of `T` in bytes (element size plus alignment padding).
pub(crate) fn array_stride<T: ShaderSize>() -> usize {
    // [T; 1]::SHADER_SIZE equals the WGSL array element stride for any length.
    <[T; 1] as ShaderSize>::SHADER_SIZE.get() as usize
}

/// Bytes of a WGSL `array<T, len>` with every element set to `T::default()`.
pub(crate) fn default_filled_bytes<T: ShaderSize + WriteInto + Default>(len: usize) -> Vec<u8> {
    let stride = array_stride::<T>();
    let mut element = vec![0u8; stride];
    write_element(&T::default(), &mut element);
    element.repeat(len)
}

/// Register a fixed-size array buffer. See
/// [`ShaderAppExt::register_array_buffer`](crate::ShaderAppExt::register_array_buffer).
pub(crate) fn register<Tag, T, const N: usize>(
    app: &mut App,
    group_index: u32,
    binding_index: u32,
    read_write: bool,
) where
    Tag: Send + Sync + 'static,
    T: ShaderSize + WriteInto + Default + Send + Sync + 'static,
{
    let stride = array_stride::<T>();
    app.insert_resource(ArrayBufferChanges::<Tag> {
        queue: Mutex::default(),
        len: N,
        stride,
        _marker: PhantomData,
    });

    let render_app = app.sub_app_mut(RenderApp);
    register_binding(
        render_app,
        group_index,
        binding_index,
        AutoBufferKind::Storage {
            read_only: !read_write,
        },
        NonZeroU64::new((stride * N) as u64),
    );

    // RenderStartup: create the persistent GPU buffer pre-filled with T::default().
    render_app.add_systems(
        RenderStartup,
        move |mut commands: Commands, render_device: Res<RenderDevice>| {
            let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("array_buffer"),
                contents: &default_filled_bytes::<T>(N),
                // COPY_SRC lets a future GPU→CPU readback copy out of this buffer.
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            commands.insert_resource(ArrayBufferState::<Tag> {
                buffer,
                stride,
                len: N,
                _marker: PhantomData,
            });
        },
    );

    render_app.init_resource::<PendingArrayChanges<Tag>>();
    render_app.add_systems(ExtractSchedule, extract_array_changes::<Tag>);

    render_app.add_systems(
        Render,
        (move |state: Option<Res<ArrayBufferState<Tag>>>,
               mut pending: ResMut<PendingArrayChanges<Tag>>,
               render_queue: Res<RenderQueue>,
               mut table: ResMut<BindingTable>| {
            let Some(state) = state else {
                return;
            };
            let PendingArrayChanges { queue, scratch, .. } = &mut *pending;
            plan_writes(queue, state.stride, state.len, scratch, |offset, bytes| {
                render_queue.write_buffer(&state.buffer, offset, bytes);
            });
            queue.clear();
            table.set_buffer(group_index, binding_index, &state.buffer);
        })
        .in_set(ShaderSystems::PrepareBindings),
    );
}

impl<Tag> ArrayBufferChanges<Tag> {
    /// Queue a change: element at `index` will be updated to `value` this frame.
    /// The value is serialized to bytes immediately.
    pub fn set<T: ShaderSize + WriteInto>(&mut self, index: usize, value: T) {
        self.check_index(index);
        let stride = self.stride;
        self.queue_mut().push(index, &value, stride);
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
        let stride = self.stride;
        let queue = self.queue_mut();
        // Every change queued so far is overwritten.
        queue.clear();
        queue.has_fill = true;
        queue.fill.resize(stride, 0);
        write_element(&value, &mut queue.fill);
    }

    fn check_index(&self, index: usize) {
        if index >= self.len {
            panic!(
                "ArrayBufferChanges::set index out of bounds: {} >= {}",
                index, self.len
            );
        }
    }

    fn queue_mut(&mut self) -> &mut ChangeQueue {
        self.queue.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Render-world resource holding the persistent GPU buffer and element stride.
#[derive(Resource)]
pub struct ArrayBufferState<Tag> {
    pub(crate) buffer: Buffer,
    /// WGSL array element stride in bytes (includes any alignment padding).
    pub(crate) stride: usize,
    /// Number of elements.
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<Tag>,
}

/// Render-world changes waiting to be uploaded, plus reusable working memory.
#[derive(Resource)]
struct PendingArrayChanges<Tag> {
    queue: ChangeQueue,
    scratch: WriteScratch,
    _marker: PhantomData<Tag>,
}

impl<Tag> Default for PendingArrayChanges<Tag> {
    fn default() -> Self {
        Self {
            queue: ChangeQueue::default(),
            scratch: WriteScratch::default(),
            _marker: PhantomData,
        }
    }
}

/// `ExtractSchedule` system: moves queued changes from the main world to the render world.
///
/// Both sides keep their allocations: when nothing is pending (the usual case) the two
/// queues are swapped, so the main world gets back the render world's emptied one.
/// Taking the queue here, rather than clearing it in a main-world schedule, means
/// changes set during `Startup` survive to the first extraction.
fn extract_array_changes<Tag: Send + Sync + 'static>(
    changes: Extract<Option<Res<ArrayBufferChanges<Tag>>>>,
    mut pending: ResMut<PendingArrayChanges<Tag>>,
) {
    let Some(changes) = changes.as_ref() else {
        return;
    };
    let mut queue = changes.queue.lock().unwrap_or_else(PoisonError::into_inner);
    if !queue.is_empty() {
        pending.queue.append(&mut queue);
    }
}

/// Reusable memory for [`plan_writes`].
#[derive(Default)]
struct WriteScratch {
    /// Positions in the queue, sorted by element index.
    order: Vec<usize>,
    /// Bytes of the write being assembled.
    bytes: Vec<u8>,
}

/// Turns `queue` into as few writes as possible and passes each to `write` as
/// `(byte offset, bytes)`. Later changes to an index win over earlier ones.
fn plan_writes(
    queue: &ChangeQueue,
    stride: usize,
    len: usize,
    scratch: &mut WriteScratch,
    mut write: impl FnMut(u64, &[u8]),
) {
    let element = |position: usize| &queue.bytes[position * stride..(position + 1) * stride];
    let WriteScratch { order, bytes } = scratch;

    if queue.has_fill {
        // One write covering the whole buffer.
        bytes.clear();
        for _ in 0..len {
            bytes.extend_from_slice(&queue.fill);
        }
        for (position, &index) in queue.indices.iter().enumerate() {
            bytes[index * stride..(index + 1) * stride].copy_from_slice(element(position));
        }
        write(0, bytes);
        return;
    }

    // Sort queue positions by index. The sort is stable, so for a repeated index the
    // last position in each group is the newest change.
    order.clear();
    order.extend(0..queue.indices.len());
    order.sort_by_key(|&position| queue.indices[position]);

    let mut run_start = 0;
    bytes.clear();
    let mut iter = order.iter().copied().peekable();
    while let Some(position) = iter.next() {
        let index = queue.indices[position];
        if iter
            .peek()
            .is_some_and(|&next| queue.indices[next] == index)
        {
            continue; // superseded by a newer change to the same index
        }
        if !bytes.is_empty() && index != run_start + bytes.len() / stride {
            write((run_start * stride) as u64, bytes);
            bytes.clear();
        }
        if bytes.is_empty() {
            run_start = index;
        }
        bytes.extend_from_slice(element(position));
    }
    if !bytes.is_empty() {
        write((run_start * stride) as u64, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: usize = 16;
    const LEN: usize = 32;

    /// One queued operation, applied the obvious way by `naive`.
    #[derive(Clone, Copy, Debug)]
    enum Op {
        Set(usize, u32),
        SetAll(u32),
    }

    fn element_bytes(value: u32) -> Vec<u8> {
        let mut bytes = vec![0u8; STRIDE];
        write_element(&UVec4::splat(value), &mut bytes);
        bytes
    }

    fn queue_from(ops: &[Op]) -> ChangeQueue {
        let mut queue = ChangeQueue::default();
        for &op in ops {
            match op {
                Op::Set(index, value) => queue.push(index, &UVec4::splat(value), STRIDE),
                Op::SetAll(value) => {
                    queue.clear();
                    queue.has_fill = true;
                    queue.fill = element_bytes(value);
                }
            }
        }
        queue
    }

    fn naive(initial: &[u8], ops: &[Op]) -> Vec<u8> {
        let mut buffer = initial.to_vec();
        for &op in ops {
            match op {
                Op::Set(index, value) => buffer[index * STRIDE..(index + 1) * STRIDE]
                    .copy_from_slice(&element_bytes(value)),
                Op::SetAll(value) => buffer = element_bytes(value).repeat(LEN),
            }
        }
        buffer
    }

    /// Applies `queue` to `initial` through `plan_writes`, returning the result and the
    /// number of writes.
    fn apply(initial: &[u8], queue: &ChangeQueue) -> (Vec<u8>, usize) {
        let mut buffer = initial.to_vec();
        let mut writes = 0;
        plan_writes(
            queue,
            STRIDE,
            LEN,
            &mut WriteScratch::default(),
            |offset, bytes| {
                let offset = offset as usize;
                buffer[offset..offset + bytes.len()].copy_from_slice(bytes);
                writes += 1;
            },
        );
        (buffer, writes)
    }

    fn initial() -> Vec<u8> {
        (0..LEN as u32)
            .flat_map(|i| element_bytes(1000 + i))
            .collect()
    }

    #[test]
    fn empty_queue_writes_nothing() {
        assert_eq!(apply(&initial(), &ChangeQueue::default()), (initial(), 0));
    }

    #[test]
    fn contiguous_indices_become_one_write() {
        let ops = [Op::Set(5, 1), Op::Set(3, 2), Op::Set(4, 3)];
        let (buffer, writes) = apply(&initial(), &queue_from(&ops));
        assert_eq!(buffer, naive(&initial(), &ops));
        assert_eq!(writes, 1);
    }

    #[test]
    fn gaps_split_writes() {
        let ops = [Op::Set(0, 1), Op::Set(1, 2), Op::Set(9, 3), Op::Set(31, 4)];
        let (buffer, writes) = apply(&initial(), &queue_from(&ops));
        assert_eq!(buffer, naive(&initial(), &ops));
        assert_eq!(writes, 3);
    }

    #[test]
    fn last_write_wins() {
        let ops = [Op::Set(7, 1), Op::Set(8, 2), Op::Set(7, 3), Op::Set(7, 4)];
        let (buffer, writes) = apply(&initial(), &queue_from(&ops));
        assert_eq!(buffer, naive(&initial(), &ops));
        assert_eq!(writes, 1);
    }

    #[test]
    fn set_all_then_set() {
        let ops = [Op::Set(2, 9), Op::SetAll(5), Op::Set(3, 6)];
        let (buffer, writes) = apply(&initial(), &queue_from(&ops));
        assert_eq!(buffer, naive(&initial(), &ops));
        assert_eq!(writes, 1);
    }

    #[test]
    fn matches_naive_on_random_sequences() {
        // Small deterministic LCG, so the test needs no extra dependencies.
        let mut seed = 0x2545_f491_u64;
        let mut next = |bound: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % bound) as usize
        };
        for _ in 0..500 {
            let ops: Vec<Op> = (0..next(40))
                .map(|_| match next(20) {
                    0 => Op::SetAll(next(100) as u32),
                    _ => Op::Set(next(LEN as u64), next(100) as u32),
                })
                .collect();
            assert_eq!(
                apply(&initial(), &queue_from(&ops)).0,
                naive(&initial(), &ops),
                "{ops:?}"
            );
        }
    }

    #[test]
    fn append_keeps_order_and_fill_semantics() {
        // Render world still has unapplied changes when new ones arrive.
        let older = [Op::Set(1, 1), Op::Set(2, 2)];
        let newer = [Op::Set(1, 3), Op::Set(4, 4)];
        let mut pending = queue_from(&older);
        pending.append(&mut queue_from(&newer));
        let all: Vec<Op> = older.iter().chain(&newer).copied().collect();
        assert_eq!(apply(&initial(), &pending).0, naive(&initial(), &all));

        // A fill in the newer batch replaces everything older.
        let newer = [Op::SetAll(8), Op::Set(0, 9)];
        let mut pending = queue_from(&older);
        let mut incoming = queue_from(&newer);
        pending.append(&mut incoming);
        assert!(incoming.is_empty());
        let all: Vec<Op> = older.iter().chain(&newer).copied().collect();
        assert_eq!(apply(&initial(), &pending).0, naive(&initial(), &all));
    }
}
