use bevy::prelude::*;
use bevy_fragment_shader::prelude::*;

const SHADER_PATH: &str = "shaders/simple_shader.wgsl";

/// Uniform data sent to the fragment shader each frame.
///
/// Layout must match the WGSL struct in the linked .wgsl file.
#[derive(Resource, ShaderType, Clone, Default)]
struct ExampleUniform {
    resolution: Vec2,
    time: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::new(SHADER_PATH))
        .register_uniform_buffer::<ExampleUniform>(0, 0)
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
