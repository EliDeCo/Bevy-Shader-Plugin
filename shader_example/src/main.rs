use bevy::{prelude::*, render::render_resource::ShaderType, window::PrimaryWindow};
use bevy_fragment_shader::FullscreenFragmentPlugin;

const SHADER_PATH: &str = "shaders/example.wgsl";

/// Uniform data sent to the fragment shader each frame.
///
/// Layout must match the WGSL struct in `example.wgsl`.
/// Field order: resolution (vec2) then time (f32) then padding keeps the
/// struct at 16 bytes — the minimum for a WebGPU uniform buffer.
#[derive(Resource, ShaderType, Clone, Default)]
struct ExampleUniform {
    resolution: Vec2,
    time: f32,
    _pad: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<ExampleUniform>::new(SHADER_PATH))
        .init_resource::<ExampleUniform>()
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
