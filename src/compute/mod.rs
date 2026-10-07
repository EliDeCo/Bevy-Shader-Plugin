mod node;
mod pipeline;
mod reflect;

use std::{
    any::{TypeId, type_name},
    collections::HashSet,
};

use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        graph::CameraDriverLabel, render_graph::RenderGraph,
    },
    shader::Shader,
    window::PrimaryWindow,
};

use crate::{ShaderSystems, add_core_plugin, gpu_buffer::GpuBufferLengths};

pub use node::ComputeShaderNode;
use reflect::{ComputeReflection, WorkgroupSize};

/// WebGPU's guaranteed maximum workgroup count per dispatch dimension.
const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;

/// How many workgroups a compute pass dispatches.
///
/// Every variant except [`exact`](Self::exact) is expressed as a number of *threads*;
/// the plugin divides it by the entry point's `@workgroup_size`, read from the WGSL
/// source, so the size is never repeated on the Rust side.
#[derive(Clone, Debug)]
pub struct Workgroups(WorkgroupsKind);

#[derive(Clone, Debug)]
enum WorkgroupsKind {
    Window,
    Buffer { tag: TypeId, tag_name: &'static str },
    Threads(UVec3),
    Exact(UVec3),
}

impl Workgroups {
    /// One thread per physical pixel of the primary window, on x and y.
    /// Use `global_invocation_id.xy` as the pixel coordinate.
    pub fn window() -> Self {
        Self(WorkgroupsKind::Window)
    }

    /// One thread per element of the GPU buffer registered with tag `Tag` via
    /// [`register_gpu_buffer`](crate::ShaderAppExt::register_gpu_buffer).
    /// Dispatches along x only; use `global_invocation_id.x` as the element index.
    pub fn over<Tag: 'static>() -> Self {
        Self(WorkgroupsKind::Buffer {
            tag: TypeId::of::<Tag>(),
            tag_name: type_name::<Tag>(),
        })
    }

    /// An explicit total number of threads on each axis.
    pub fn threads(threads: UVec3) -> Self {
        Self(WorkgroupsKind::Threads(threads))
    }

    /// Raw workgroup counts, passed straight to `dispatch_workgroups`. No reflection is
    /// needed, so this also works for shaders whose workgroup size can't be reflected.
    pub fn exact(workgroups: UVec3) -> Self {
        Self(WorkgroupsKind::Exact(workgroups))
    }
}

/// One configured compute pass: an entry point and how to size its dispatch.
#[derive(Clone, Debug)]
pub(crate) struct PassConfig {
    pub entry: &'static str,
    pub workgroups: Workgroups,
    /// Init passes run once (and again on [`ComputePasses::rerun`]) instead of every frame.
    pub init: bool,
}

/// Bevy plugin that runs compute shader passes once per frame, before any camera renders.
///
/// Every buffer registered through [`ShaderAppExt`](crate::ShaderAppExt) is visible to
/// both the compute passes and the [`FullscreenFragmentPlugin`](crate::FullscreenFragmentPlugin)
/// shader, at the same `@group`/`@binding`. Each binding must be declared with the same
/// access mode (`read` or `read_write`) in every shader that uses it.
///
/// Only one `ComputeShaderPlugin` is supported per app. Add it after `DefaultPlugins`.
///
/// # Example
///
/// ```rust,ignore
/// struct Particles;
///
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(
///         ComputeShaderPlugin::new("shaders/particles.wgsl")
///             .init_pass("init", Workgroups::over::<Particles>())
///             .pass("update", Workgroups::over::<Particles>()),
///     )
///     .register_gpu_buffer::<Particles, Particle>(0, 0, 1024)
///     .run();
/// ```
pub struct ComputeShaderPlugin {
    /// Path to the compute shader asset (e.g. `"shaders/simulation.wgsl"`).
    pub shader_path: &'static str,
    passes: Vec<PassConfig>,
}

impl ComputeShaderPlugin {
    /// Creates a plugin for the compute shader at the given asset path. Add passes with
    /// [`pass`](Self::pass) and [`init_pass`](Self::init_pass).
    pub fn new(shader_path: &'static str) -> Self {
        Self {
            shader_path,
            passes: Vec::new(),
        }
    }

    /// Run the `@compute` entry point `entry` every frame. Passes run in the order they
    /// are added, and each sees the writes of the passes before it. The same entry point
    /// may be added more than once (e.g. for simulation substeps).
    pub fn pass(mut self, entry: &'static str, workgroups: Workgroups) -> Self {
        self.passes.push(PassConfig {
            entry,
            workgroups,
            init: false,
        });
        self
    }

    /// Run the `@compute` entry point `entry` once, on the first frame everything is
    /// ready. Every-frame passes don't start until all init passes have run. Use
    /// [`ComputePasses::rerun`] to run it again later.
    pub fn init_pass(mut self, entry: &'static str, workgroups: Workgroups) -> Self {
        self.passes.push(PassConfig {
            entry,
            workgroups,
            init: true,
        });
        self
    }
}

/// Main-world control over the compute passes.
///
/// ```rust,ignore
/// fn reset(keys: Res<ButtonInput<KeyCode>>, mut passes: ResMut<ComputePasses>) {
///     if keys.just_pressed(KeyCode::KeyR) {
///         passes.rerun("init");
///     }
/// }
/// ```
#[derive(Resource, Default)]
pub struct ComputePasses {
    reruns: Vec<String>,
}

impl ComputePasses {
    /// Run the init pass(es) for entry point `entry` again on the next frame.
    pub fn rerun(&mut self, entry: &str) {
        self.reruns.push(entry.to_owned());
    }
}

/// Main-world configuration, inserted by [`ComputeShaderPlugin`].
#[derive(Resource)]
pub(crate) struct ComputeConfig {
    pub shader_path: &'static str,
    pub shader: Handle<Shader>,
    pub passes: Vec<PassConfig>,
}

/// This frame's dispatch plan, built in the main world and extracted to the render world.
#[derive(Resource, Default, Clone)]
pub(crate) struct ComputeFrame {
    /// Workgroup count per pass (same order as the configured passes), or `None` if the
    /// pass can't be dispatched this frame.
    pub counts: Vec<Option<UVec3>>,
    /// Indices of init passes to run again.
    pub reruns: Vec<usize>,
}

impl Plugin for ComputeShaderPlugin {
    fn build(&self, app: &mut App) {
        add_core_plugin(app);

        let shader = app.world().resource::<AssetServer>().load(self.shader_path);

        app.init_resource::<GpuBufferLengths>()
            .init_resource::<ComputePasses>()
            .init_resource::<ComputeReflection>()
            .init_resource::<ComputeFrame>()
            .insert_resource(ComputeConfig {
                shader_path: self.shader_path,
                shader,
                passes: self.passes.clone(),
            })
            .add_systems(
                PostUpdate,
                (reflect::reflect_workgroup_sizes, plan_dispatches).chain(),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .insert_resource(pipeline::ComputeRenderConfig {
                shader_path: self.shader_path,
                passes: self.passes.clone(),
            })
            .insert_resource(pipeline::ComputeRunState {
                init_done: vec![false; self.passes.len()],
            })
            .init_resource::<pipeline::ComputeRunList>()
            .add_systems(ExtractSchedule, extract_compute_frame)
            .add_systems(
                RenderStartup,
                pipeline::init_compute_pipelines.after(ShaderSystems::CompileLayouts),
            )
            .add_systems(
                Render,
                pipeline::prepare_compute_run
                    .in_set(RenderSystems::PrepareBindGroups)
                    .after(ShaderSystems::FinalizeBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ComputeShaderNode, node::ComputeNode);
        render_graph.add_node_edge(ComputeShaderNode, CameraDriverLabel);
    }
}

/// Converts a thread count to a workgroup count for the given workgroup size.
/// Returns `Ok(None)` when there is nothing to dispatch (e.g. a minimised window).
fn workgroup_count(threads: UVec3, size: UVec3) -> Result<Option<UVec3>, String> {
    if size.cmpeq(UVec3::ZERO).any() {
        return Err(format!("invalid workgroup size {size}"));
    }
    let count = UVec3::new(
        threads.x.div_ceil(size.x),
        threads.y.div_ceil(size.y),
        threads.z.div_ceil(size.z),
    );
    check_count(count)
}

fn check_count(count: UVec3) -> Result<Option<UVec3>, String> {
    if count.max_element() > MAX_WORKGROUPS_PER_DIMENSION {
        return Err(format!(
            "dispatch of {count} workgroups exceeds the limit of {MAX_WORKGROUPS_PER_DIMENSION} per dimension"
        ));
    }
    if count.cmpeq(UVec3::ZERO).any() {
        return Ok(None);
    }
    Ok(Some(count))
}

/// Works out this frame's workgroup counts for every pass. Problems are logged once per
/// pass and message, and that pass is skipped until they're resolved.
fn plan_dispatches(
    config: Res<ComputeConfig>,
    reflection: Res<ComputeReflection>,
    lengths: Res<GpuBufferLengths>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut passes: ResMut<ComputePasses>,
    mut frame: ResMut<ComputeFrame>,
    mut reported: Local<HashSet<(usize, String)>>,
) {
    let window_size = windows
        .single()
        .ok()
        .map(|window| UVec3::new(window.physical_width(), window.physical_height(), 1));

    let mut report = |index: usize, message: String| {
        if reported.insert((index, message.clone())) {
            warn!(
                "compute pass `{}` skipped: {message}",
                config.passes[index].entry
            );
        }
    };

    frame.counts.clear();
    for (index, pass) in config.passes.iter().enumerate() {
        let reflected_size = || -> Result<Option<UVec3>, String> {
            let Some(sizes) = &reflection.sizes else {
                // Shader not loaded yet (or failed to parse, which is logged separately).
                return Ok(None);
            };
            match sizes.get(pass.entry) {
                Some(WorkgroupSize::Fixed(size)) => Ok(Some(*size)),
                Some(WorkgroupSize::Overridden) => Err(
                    "its @workgroup_size uses override constants; use `Workgroups::exact`".into(),
                ),
                None => Err("no compute entry point with this name".into()),
            }
        };

        let count = match &pass.workgroups.0 {
            WorkgroupsKind::Exact(count) => check_count(*count),
            kind => reflected_size().and_then(|size| {
                let Some(size) = size else { return Ok(None) };
                let threads = match kind {
                    WorkgroupsKind::Window => window_size.unwrap_or(UVec3::ZERO),
                    WorkgroupsKind::Buffer { tag, tag_name } => {
                        let Some(&len) = lengths.0.get(tag) else {
                            return Err(format!(
                                "no GPU buffer is registered with tag `{tag_name}`"
                            ));
                        };
                        if size.y != 1 || size.z != 1 {
                            return Err(format!(
                                "`Workgroups::over` dispatches along x only, but the workgroup size is {size}; use @workgroup_size(N)"
                            ));
                        }
                        UVec3::new(len, 1, 1)
                    }
                    WorkgroupsKind::Threads(threads) => *threads,
                    WorkgroupsKind::Exact(_) => unreachable!(),
                };
                workgroup_count(threads, size)
            }),
        };

        frame.counts.push(count.unwrap_or_else(|message| {
            report(index, message);
            None
        }));
    }

    frame.reruns.clear();
    for entry in passes.reruns.drain(..) {
        let matching: Vec<usize> = config
            .passes
            .iter()
            .enumerate()
            .filter(|(_, pass)| pass.init && pass.entry == entry)
            .map(|(index, _)| index)
            .collect();
        if matching.is_empty() {
            warn!("ComputePasses::rerun: `{entry}` is not an init pass");
        }
        frame.reruns.extend(matching);
    }
}

fn extract_compute_frame(mut commands: Commands, frame: Extract<Res<ComputeFrame>>) {
    commands.insert_resource(frame.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_thread_counts_up() {
        assert_eq!(
            workgroup_count(UVec3::new(500, 1, 1), UVec3::new(64, 1, 1)),
            Ok(Some(UVec3::new(8, 1, 1)))
        );
        assert_eq!(
            workgroup_count(UVec3::new(1920, 1080, 1), UVec3::new(8, 8, 1)),
            Ok(Some(UVec3::new(240, 135, 1)))
        );
    }

    #[test]
    fn skips_empty_dispatches() {
        assert_eq!(
            workgroup_count(UVec3::new(0, 0, 1), UVec3::new(8, 8, 1)),
            Ok(None)
        );
    }

    #[test]
    fn rejects_oversized_dispatches() {
        assert!(workgroup_count(UVec3::new(65536 * 64 + 1, 1, 1), UVec3::new(64, 1, 1)).is_err());
        assert!(check_count(UVec3::new(65536, 1, 1)).is_err());
    }

    #[test]
    fn rejects_zero_workgroup_size() {
        assert!(workgroup_count(UVec3::ONE, UVec3::new(0, 1, 1)).is_err());
    }
}
