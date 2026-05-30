use bevy::{prelude::*, render::render_resource::ShaderType, window::PrimaryWindow};
use bevy_fragment_shader::{FragmentAppExt, FullscreenFragmentPlugin};

const SHADER_PATH: &str = "shaders/storage_shader.wgsl";

#[derive(Resource, ShaderType, Clone, Default)]
struct ExampleUniform {
    resolution: Vec2,
    time: f32,
    _pad: f32,
}

/// Red channel values for each of the 64 vertical bands, updated every frame.
#[derive(Resource, ShaderType, Clone)]
struct RedChannel {
    values: [f32; 64],
}

/// Green channel values for each of the 64 vertical bands, updated every frame.
#[derive(Resource, ShaderType, Clone)]
struct GreenChannel {
    values: [f32; 64],
}

/// Blue channel values for each of the 64 vertical bands, updated every frame.
#[derive(Resource, ShaderType, Clone)]
struct BlueChannel {
    values: [f32; 64],
}

impl Default for RedChannel {
    fn default() -> Self {
        Self { values: [0.0; 64] }
    }
}
impl Default for GreenChannel {
    fn default() -> Self {
        Self { values: [0.0; 64] }
    }
}
impl Default for BlueChannel {
    fn default() -> Self {
        Self { values: [0.0; 64] }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::new(SHADER_PATH))
        .register_uniform_buffer::<ExampleUniform>(0, 0)
        .init_resource::<ExampleUniform>()
        .register_storage_buffer::<RedChannel>(1, 0, false)
        .register_storage_buffer::<GreenChannel>(1, 1, false)
        .register_storage_buffer::<BlueChannel>(1, 2, false)
        .init_resource::<RedChannel>()
        .init_resource::<GreenChannel>()
        .init_resource::<BlueChannel>()
        .add_systems(Startup, setup)
        .add_systems(Update, (update_uniform, update_channels))
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

fn update_channels(
    mut red: ResMut<RedChannel>,
    mut green: ResMut<GreenChannel>,
    mut blue: ResMut<BlueChannel>,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();
    let tau = std::f32::consts::TAU;
    for i in 0..64usize {
        let phase = i as f32 / 64.0 * tau;
        red.values[i] = 0.5 + 0.5 * (t + phase).sin();
        green.values[i] = 0.5 + 0.5 * (t * 1.3 + phase + 2.094).sin();
        blue.values[i] = 0.5 + 0.5 * (t * 0.7 + phase + 4.189).sin();
    }
}
