use bevy::{prelude::*, render::render_resource::ShaderType, window::PrimaryWindow};
use bevy_fragment_shader::{FragmentAppExt, FullscreenFragmentPlugin};

const SHADER_PATH: &str = "shaders/storage_shader.wgsl";

#[derive(Resource, ShaderType, Clone, Default)]
struct ExampleUniform {
    resolution: Vec2,
    time: f32,
    _pad: f32,
}

/// Color data uploaded to the GPU as a read-only storage buffer at @group(1).
#[derive(Resource, ShaderType, Clone)]
struct ColorData {
    colors: [Vec4; 64],
}

impl Default for ColorData {
    fn default() -> Self {
        Self { colors: [Vec4::ZERO; 64] }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<ExampleUniform>::new(SHADER_PATH))
        .init_resource::<ExampleUniform>()
        .register_storage_buffer::<ColorData>(1)
        .init_resource::<ColorData>()
        .add_systems(Startup, setup)
        .add_systems(Update, update_uniform)
        .add_systems(Startup, init_colors)
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

fn init_colors(mut color_data: ResMut<ColorData>) {
    let tau = std::f32::consts::TAU;
    for i in 0..64 {
        let t = i as f32 / 64.0;
        color_data.colors[i] = Vec4::new(
            0.5 + 0.5 * (t * tau).sin(),
            0.5 + 0.5 * (t * tau * 2.0 + 2.094).sin(),
            0.5 + 0.5 * (t * tau * 3.0 + 4.189).sin(),
            1.0,
        );
    }
}
