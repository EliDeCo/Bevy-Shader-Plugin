use bevy::{
    prelude::*,
    render::{
        render_resource::{ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
    },
    window::PrimaryWindow,
};
use bevy_fragment_shader::{
    FragmentAppExt, FragmentBindGroupBuilder, FragmentExtraBindGroups, FragmentExtraLayouts,
    FullscreenFragmentPlugin, FullscreenPipeline,
};

const SHADER_PATH: &str = "shaders/storage_shader.wgsl";

#[derive(Resource, ShaderType, Clone, Default)]
struct ExampleUniform {
    resolution: Vec2,
    time: f32,
    _pad: f32,
}

/// Render-world resource holding a GPU storage buffer of color data.
#[derive(Resource)]
struct ParticleBuffer(StorageBuffer<Vec<Vec4>>);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<ExampleUniform>::new(SHADER_PATH))
        .init_resource::<ExampleUniform>()
        .register_fragment_extra_bind_group(setup_storage_layout, prepare_storage_bind_group)
        .add_systems(Startup, setup)
        .add_systems(Update, update_uniform)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Msaa::Off));
}

fn update_uniform(
    mut uniform: ResMut<ExampleUniform>,
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let Ok(window) = windows.single() else { return };
    uniform.resolution = Vec2::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
    );
    uniform.time = time.elapsed_secs();
}

// --- Render world systems ---

fn setup_storage_layout(
    mut extra_layouts: ResMut<FragmentExtraLayouts>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    extra_layouts.storage_buffer_read_only("particle_layout");

    // Build color data: 64 entries with hue-varying colors.
    let colors: Vec<Vec4> = (0..64)
        .map(|i| {
            let t = i as f32 / 64.0;
            let tau = std::f32::consts::TAU;
            Vec4::new(
                0.5 + 0.5 * (t * tau).sin(),
                0.5 + 0.5 * (t * tau * 2.0 + 2.094).sin(),
                0.5 + 0.5 * (t * tau * 3.0 + 4.189).sin(),
                1.0,
            )
        })
        .collect();

    let mut buf = StorageBuffer::default();
    buf.set(colors);
    buf.write_buffer(&render_device, &render_queue);
    commands.insert_resource(ParticleBuffer(buf));
}

fn prepare_storage_bind_group(
    mut extra_bind_groups: ResMut<FragmentExtraBindGroups>,
    render_device: Res<RenderDevice>,
    pipeline: Option<Res<FullscreenPipeline<ExampleUniform>>>,
    particle_buf: Option<Res<ParticleBuffer>>,
) {
    let (Some(pipeline), Some(particle_buf)) = (pipeline, particle_buf) else { return };
    let Some(raw_buf) = particle_buf.0.buffer() else { return };

    let bg = FragmentBindGroupBuilder::new(&pipeline.extra_layouts[0], &render_device)
        .label("particle_bg")
        .buffer(raw_buf)
        .build();

    extra_bind_groups.clear().push(bg);
}
