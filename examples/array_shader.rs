use bevy::{prelude::*, render::render_resource::ShaderType, window::PrimaryWindow};
use bevy_fragment_shader::{ArrayBufferChanges, FragmentAppExt, FullscreenFragmentPlugin};

const SHADER_PATH: &str = "shaders/array_shader.wgsl";

// Zero-sized tag types — one per buffer registration, no impl required.
struct Colors;
struct Brightnesses;

#[derive(Resource, ShaderType, Clone, Default)]
struct FrameUniform {
    resolution: Vec2,
    time: f32,
    _pad: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::new(SHADER_PATH))
        .register_uniform_buffer::<FrameUniform>(0, 0)
        .init_resource::<FrameUniform>()
        .register_array_buffer::<Colors, Vec4, 64>(1, 0, false)
        .register_array_buffer::<Brightnesses, Vec4, 64>(1, 1, false)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (update_uniform, animate_colors, animate_brightnesses),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Msaa::Off));
}

fn update_uniform(
    mut uniform: ResMut<FrameUniform>,
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

/// Updates two fixed pairs of color columns (8–9 and 24–25) each frame.
/// Both pairs are consecutive, so each becomes one GPU write call.
/// The non-adjacent groups produce two separate write_buffer calls.
fn animate_colors(mut changes: ResMut<ArrayBufferChanges<Colors>>, time: Res<Time>) {
    let t = time.elapsed_secs();

    // Group A: columns 8–9, hue cycles slowly.
    let color_a = hsl_to_rgba(t * 0.15, 1.0, 0.5);
    changes.set(8, color_a);
    changes.set(9, color_a);

    // Group B: columns 24–25, hue cycles at a different rate and offset.
    let color_b = hsl_to_rgba(t * 0.25 + 0.5, 1.0, 0.5);
    changes.set(24, color_b);
    changes.set(25, color_b);
}

/// Updates two fixed pairs of brightness columns (16–17 and 40–41) each frame.
/// These are disjoint from the color columns and from each other.
fn animate_brightnesses(
    mut changes: ResMut<ArrayBufferChanges<Brightnesses>>,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();
    let tau = std::f32::consts::TAU;

    // Group C: columns 16–17, brightness pulses at 0.4 Hz.
    let b1 = 0.5 + 0.5 * (t * tau * 0.4).sin();
    changes.set(16, Vec4::splat(b1));
    changes.set(17, Vec4::splat(b1));

    // Group D: columns 40–41, brightness pulses at 0.6 Hz, opposite phase.
    let b2 = 0.5 + 0.5 * (t * tau * 0.6 + std::f32::consts::PI).sin();
    changes.set(40, Vec4::splat(b2));
    changes.set(41, Vec4::splat(b2));
}

fn hsl_to_rgba(h: f32, s: f32, l: f32) -> Vec4 {
    let h = h.fract();
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Vec4::new(r + m, g + m, b + m, 1.0)
}
