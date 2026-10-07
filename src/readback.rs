use std::{
    any::type_name,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        graph::CameraDriverLabel,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{Buffer, BufferDescriptor, BufferUsages, MapMode},
        renderer::{RenderContext, RenderDevice, render_system},
        storage::GpuShaderStorageBuffer,
    },
};
use encase::{
    ShaderSize, ShaderType,
    internal::{CreateFrom, WriteInto},
};

use crate::{
    ArrayBufferChanges, ShaderSystems,
    auto_array::{ArrayBufferState, array_stride},
    bindings::AutoBufferKind,
    cpu_buffer::CpuBuffer,
    gpu_buffer::GpuBuffer,
};

/// When a read-back buffer is copied to the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackMode {
    /// Copy the buffer every frame (skipping frames while a previous copy is still
    /// being mapped), so `latest()` keeps updating. Costs one buffer copy per frame.
    EveryFrame,
    /// Copy the buffer once per `request()` call.
    OnRequest,
}

/// State shared between a main-world readback resource and the render world. Held in an
/// `Arc` by both, so no extraction is needed and it works with pipelined rendering.
#[derive(Default)]
struct ReadbackShared {
    /// Set by `request()`, consumed when a copy is queued.
    requested: AtomicBool,
    /// A copy has been queued and its bytes haven't arrived yet.
    in_flight: AtomicBool,
    /// Bytes delivered by the map callback, taken by the main world.
    result: Mutex<Option<Vec<u8>>>,
}

impl ReadbackShared {
    fn take_result(&self) -> Option<Vec<u8>> {
        self.result.lock().ok()?.take()
    }
}

/// The most recent CPU copy of an array buffer or GPU buffer, decoded into `T` values.
///
/// This resource only changes when new data arrives, so check `is_changed()` to react to
/// fresh data. [`request`](Self::request) takes `&self`, so requesting through
/// `Res<GpuReadback<..>>` doesn't count as a change:
///
/// ```rust,ignore
/// fn request(readback: Res<GpuReadback<Particles, Particle>>) {
///     readback.request();
/// }
///
/// fn print(readback: Res<GpuReadback<Particles, Particle>>) {
///     if !readback.is_changed() { return; }
///     let Some(particles) = readback.latest() else { return };
///     info!("first particle at {}", particles[0].pos);
/// }
/// ```
///
/// The data shows the buffer at the end of the frame it was copied in, and arrives
/// 1–3 frames later. Use it for UI, stats, or logging — not for logic that needs the
/// exact current GPU state.
#[derive(Resource)]
pub struct GpuReadback<Tag, T> {
    shared: Arc<ReadbackShared>,
    latest: Option<Vec<T>>,
    _marker: PhantomData<Tag>,
}

impl<Tag, T> GpuReadback<Tag, T> {
    /// The most recently read-back contents of the buffer, or `None` until the first
    /// readback arrives.
    pub fn latest(&self) -> Option<&[T]> {
        self.latest.as_deref()
    }

    /// Ask for the buffer to be copied back. Only needed with
    /// [`ReadbackMode::OnRequest`]. Requests made while a copy is in flight are merged
    /// into one follow-up copy.
    pub fn request(&self) {
        self.shared.requested.store(true, Ordering::Release);
    }
}

/// The most recent CPU copy of a storage buffer, decoded as the registered resource type
/// `S` (which may itself contain arrays).
///
/// Works like [`GpuReadback`]: check `is_changed()` for fresh data, and call
/// [`request`](Self::request) in [`ReadbackMode::OnRequest`] mode. The data arrives
/// 1–3 frames after the copy.
#[derive(Resource)]
pub struct StorageReadback<S> {
    shared: Arc<ReadbackShared>,
    latest: Option<S>,
}

impl<S> StorageReadback<S> {
    /// The most recently read-back contents of the buffer, or `None` until the first
    /// readback arrives.
    pub fn latest(&self) -> Option<&S> {
        self.latest.as_ref()
    }

    /// Ask for the buffer to be copied back. Only needed with
    /// [`ReadbackMode::OnRequest`]. Requests made while a copy is in flight are merged
    /// into one follow-up copy.
    pub fn request(&self) {
        self.shared.requested.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Render world
// ---------------------------------------------------------------------------

/// One buffer copy recorded by [`ReadbackNode`] this frame.
struct QueuedCopy {
    source: Buffer,
    staging: Buffer,
    size: u64,
    shared: Arc<ReadbackShared>,
}

/// Copies queued this frame. Filled during `PrepareBindGroups`, recorded by
/// [`ReadbackNode`], then mapped and cleared by [`map_readbacks`].
#[derive(Resource, Default)]
struct ReadbackCopies(Vec<QueuedCopy>);

/// Render-world state for one read-back registration. `Key` distinguishes registrations.
#[derive(Resource)]
struct ReadbackJob<Key> {
    shared: Arc<ReadbackShared>,
    mode: ReadbackMode,
    staging: Option<Buffer>,
    _marker: PhantomData<Key>,
}

// Keys keep registrations of different buffer kinds apart.
struct GpuKey<Tag>(PhantomData<Tag>);
struct ArrayKey<Tag>(PhantomData<Tag>);
struct StorageKey<S>(PhantomData<S>);

/// Whether to queue a copy this frame. Never copies while a previous copy is in flight;
/// an on-request copy consumes the request, which otherwise stays pending.
fn should_copy(mode: ReadbackMode, in_flight: bool, requested: &AtomicBool) -> bool {
    if in_flight {
        return false;
    }
    match mode {
        ReadbackMode::EveryFrame => true,
        ReadbackMode::OnRequest => requested.swap(false, Ordering::AcqRel),
    }
}

impl<Key> ReadbackJob<Key> {
    fn new(shared: Arc<ReadbackShared>, mode: ReadbackMode) -> Self {
        Self {
            shared,
            mode,
            staging: None,
            _marker: PhantomData,
        }
    }

    /// Queue a copy of the first `size` bytes of `source`, if this job wants one now.
    fn queue_copy(
        &mut self,
        source: &Buffer,
        size: u64,
        render_device: &RenderDevice,
        copies: &mut ReadbackCopies,
    ) {
        if size == 0
            || !should_copy(
                self.mode,
                self.shared.in_flight.load(Ordering::Acquire),
                &self.shared.requested,
            )
        {
            return;
        }

        // Reuse the staging buffer unless the size changed. It is never still mapped
        // here: `in_flight` is only cleared after unmapping.
        let staging = match &self.staging {
            Some(staging) if staging.size() == size => staging.clone(),
            _ => {
                let staging = render_device.create_buffer(&BufferDescriptor {
                    label: Some("readback_staging_buffer"),
                    size,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.staging = Some(staging.clone());
                staging
            }
        };

        self.shared.in_flight.store(true, Ordering::Release);
        copies.0.push(QueuedCopy {
            source: source.clone(),
            staging,
            size,
            shared: self.shared.clone(),
        });
    }
}

/// Render graph label of the readback copy node. It runs in the root render graph after
/// [`CameraDriverLabel`], so copies see everything the frame's compute passes and
/// cameras wrote.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct ReadbackCopyNode;

#[derive(Default)]
struct ReadbackNode;

impl Node for ReadbackNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(copies) = world.get_resource::<ReadbackCopies>() else {
            return Ok(());
        };
        if copies.0.is_empty() {
            return Ok(());
        }
        let encoder = render_context.command_encoder();
        for copy in &copies.0 {
            encoder.copy_buffer_to_buffer(&copy.source, 0, &copy.staging, 0, copy.size);
        }
        Ok(())
    }
}

/// After the frame is submitted, asks wgpu to map every staging buffer copied into this
/// frame. The callback hands the bytes to the main world.
fn map_readbacks(mut copies: ResMut<ReadbackCopies>) {
    for copy in copies.0.drain(..) {
        let QueuedCopy {
            staging,
            size,
            shared,
            ..
        } = copy;
        let mapped = staging.clone();
        staging
            .slice(..size)
            .map_async(MapMode::Read, move |result| {
                match result {
                    Ok(()) => {
                        let bytes = mapped.slice(..size).get_mapped_range().to_vec();
                        mapped.unmap();
                        if let Ok(mut slot) = shared.result.lock() {
                            *slot = Some(bytes);
                        }
                    }
                    Err(error) => warn!("buffer readback failed: {error}"),
                }
                shared.in_flight.store(false, Ordering::Release);
            });
    }
}

/// Copy-and-map machinery shared by every readback registration.
struct ReadbackPlugin;

impl Plugin for ReadbackPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<ReadbackCopies>().add_systems(
            Render,
            map_readbacks
                .after(render_system)
                .in_set(RenderSystems::Render),
        );
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ReadbackCopyNode, ReadbackNode);
        render_graph.add_node_edge(CameraDriverLabel, ReadbackCopyNode);
    }
}

fn add_readback_plugin(app: &mut App) {
    if !app.is_plugin_added::<ReadbackPlugin>() {
        app.add_plugins(ReadbackPlugin);
    }
}

/// Adds the render-world job for `Key` and a system that resolves its source buffer each
/// frame and queues copies.
fn add_render_job<Key, M>(
    app: &mut App,
    shared: Arc<ReadbackShared>,
    mode: ReadbackMode,
    queue_system: impl IntoScheduleConfigs<bevy::ecs::system::ScheduleSystem, M>,
) where
    Key: Send + Sync + 'static,
{
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .insert_resource(ReadbackJob::<Key>::new(shared, mode))
        .add_systems(
            Render,
            queue_system
                .in_set(RenderSystems::PrepareBindGroups)
                .after(ShaderSystems::PrepareBindings),
        );
}

fn decode<D: ShaderType + CreateFrom>(bytes: &[u8]) -> Option<D> {
    match encase::StorageBuffer::new(bytes).create() {
        Ok(value) => Some(value),
        Err(error) => {
            error!("could not decode read-back {}: {error}", type_name::<D>());
            None
        }
    }
}

/// Main world: moves newly arrived bytes into the [`GpuReadback`] resource.
fn deliver_slice<Tag, T>(mut readback: ResMut<GpuReadback<Tag, T>>)
where
    Tag: Send + Sync + 'static,
    T: ShaderSize + CreateFrom + Send + Sync + 'static,
{
    if let Some(bytes) = readback.shared.take_result()
        && let Some(values) = decode::<Vec<T>>(&bytes)
    {
        readback.latest = Some(values);
    }
}

/// Main world: moves newly arrived bytes into the [`StorageReadback`] resource.
fn deliver_storage<S>(mut readback: ResMut<StorageReadback<S>>)
where
    S: ShaderType + CreateFrom + Send + Sync + 'static,
{
    if let Some(bytes) = readback.shared.take_result()
        && let Some(value) = decode::<S>(&bytes)
    {
        readback.latest = Some(value);
    }
}

/// Inserts `GpuReadback<Tag, T>` and its delivery system, returning the shared state.
fn insert_slice_readback<Tag, T>(app: &mut App) -> Arc<ReadbackShared>
where
    Tag: Send + Sync + 'static,
    T: ShaderSize + CreateFrom + Send + Sync + 'static,
{
    assert!(
        !app.world().contains_resource::<GpuReadback<Tag, T>>(),
        "readback is already enabled for tag `{}`",
        type_name::<Tag>()
    );
    let shared = Arc::new(ReadbackShared::default());
    app.insert_resource(GpuReadback::<Tag, T> {
        shared: shared.clone(),
        latest: None,
        _marker: PhantomData,
    })
    .add_systems(PreUpdate, deliver_slice::<Tag, T>);
    shared
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// See [`ShaderAppExt::read_back_gpu_buffer`](crate::ShaderAppExt::read_back_gpu_buffer).
pub(crate) fn register_gpu<Tag, T>(app: &mut App, mode: ReadbackMode)
where
    Tag: Send + Sync + 'static,
    T: ShaderSize + CreateFrom + Send + Sync + 'static,
{
    let Some(gpu_buffer) = app.world().get_resource::<GpuBuffer<Tag>>() else {
        panic!(
            "read_back_gpu_buffer::<{}, _>: call register_gpu_buffer for this tag first",
            type_name::<Tag>()
        );
    };
    let asset_id = gpu_buffer.handle().id();

    add_readback_plugin(app);
    let shared = insert_slice_readback::<Tag, T>(app);
    add_render_job::<GpuKey<Tag>, _>(
        app,
        shared,
        mode,
        move |buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
              mut job: ResMut<ReadbackJob<GpuKey<Tag>>>,
              render_device: Res<RenderDevice>,
              mut copies: ResMut<ReadbackCopies>| {
            if let Some(gpu_buffer) = buffers.get(asset_id) {
                let size = gpu_buffer.buffer.size();
                job.queue_copy(&gpu_buffer.buffer, size, &render_device, &mut copies);
            }
        },
    );
}

/// See [`ShaderAppExt::read_back_array_buffer`](crate::ShaderAppExt::read_back_array_buffer).
pub(crate) fn register_array<Tag, T>(app: &mut App, mode: ReadbackMode)
where
    Tag: Send + Sync + 'static,
    T: ShaderSize + CreateFrom + Send + Sync + 'static,
{
    assert!(
        app.world().contains_resource::<ArrayBufferChanges<Tag>>(),
        "read_back_array_buffer::<{}, _>: call register_array_buffer for this tag first",
        type_name::<Tag>()
    );

    add_readback_plugin(app);
    let shared = insert_slice_readback::<Tag, T>(app);
    add_render_job::<ArrayKey<Tag>, _>(
        app,
        shared,
        mode,
        |state: Option<Res<ArrayBufferState<Tag>>>,
         mut job: ResMut<ReadbackJob<ArrayKey<Tag>>>,
         render_device: Res<RenderDevice>,
         mut copies: ResMut<ReadbackCopies>| {
            let Some(state) = state else { return };
            if state.stride != array_stride::<T>() {
                error_once!(
                    "read_back_array_buffer::<{}, {}>: element type doesn't match the one passed to register_array_buffer",
                    type_name::<Tag>(),
                    type_name::<T>()
                );
                return;
            }
            let size = state.buffer.size();
            job.queue_copy(&state.buffer, size, &render_device, &mut copies);
        },
    );
}

/// See [`ShaderAppExt::read_back_storage_buffer`](crate::ShaderAppExt::read_back_storage_buffer).
pub(crate) fn register_storage<S>(app: &mut App, mode: ReadbackMode)
where
    S: ShaderType + WriteInto + Default + CreateFrom + Resource,
{
    let has_storage_slot = app
        .get_sub_app(RenderApp)
        .and_then(|render_app| render_app.world().get_resource::<CpuBuffer<S>>())
        .is_some_and(|state| {
            state
                .slot_kinds()
                .any(|kind| matches!(kind, AutoBufferKind::Storage { .. }))
        });
    assert!(
        has_storage_slot,
        "read_back_storage_buffer::<{}>: call register_storage_buffer for this type first",
        type_name::<S>()
    );
    assert!(
        !app.world().contains_resource::<StorageReadback<S>>(),
        "read_back_storage_buffer::<{}>: readback is already enabled for this type",
        type_name::<S>()
    );

    add_readback_plugin(app);
    let shared = Arc::new(ReadbackShared::default());
    app.insert_resource(StorageReadback::<S> {
        shared: shared.clone(),
        latest: None,
    })
    .add_systems(PreUpdate, deliver_storage::<S>);

    add_render_job::<StorageKey<S>, _>(
        app,
        shared,
        mode,
        |state: Res<CpuBuffer<S>>,
         mut job: ResMut<ReadbackJob<StorageKey<S>>>,
         render_device: Res<RenderDevice>,
         mut copies: ResMut<ReadbackCopies>| {
            if let Some((buffer, size)) = state.readback_source() {
                job.queue_copy(buffer, size, &render_device, &mut copies);
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_frame_copies_unless_in_flight() {
        let requested = AtomicBool::new(false);
        assert!(should_copy(ReadbackMode::EveryFrame, false, &requested));
        assert!(!should_copy(ReadbackMode::EveryFrame, true, &requested));
    }

    #[test]
    fn on_request_consumes_the_request() {
        let requested = AtomicBool::new(true);
        assert!(should_copy(ReadbackMode::OnRequest, false, &requested));
        assert!(!requested.load(Ordering::Acquire));
        assert!(!should_copy(ReadbackMode::OnRequest, false, &requested));
    }

    #[test]
    fn on_request_stays_pending_while_in_flight() {
        let requested = AtomicBool::new(true);
        assert!(!should_copy(ReadbackMode::OnRequest, true, &requested));
        assert!(requested.load(Ordering::Acquire));
    }

    #[test]
    fn decodes_what_encase_wrote() {
        #[derive(ShaderType, Debug, PartialEq)]
        struct Sample {
            a: Vec2,
            b: f32,
        }
        let values = vec![
            Sample {
                a: Vec2::new(1.0, 2.0),
                b: 3.0,
            },
            Sample {
                a: Vec2::ZERO,
                b: -1.0,
            },
        ];
        let mut bytes = encase::StorageBuffer::new(Vec::new());
        bytes.write(&values).unwrap();
        assert_eq!(decode::<Vec<Sample>>(&bytes.into_inner()), Some(values));
    }
}
