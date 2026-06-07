use bevy::prelude::*;
use bevy_fragment_shader::prelude::*;

const SHADER_PATH: &str = "shaders/solar_system.wgsl";
const N: usize = 8;

// Tag type for the per-planet color array buffer.
struct PlanetColors;

// group(0) binding(0) — resolution only; all motion is computed on the CPU.
#[derive(Resource, ShaderType, Clone, Default)]
struct FrameUniform {
    resolution: Vec2,
}

// group(1) binding(0) — UV positions for all 8 planets, re-uploaded every frame
// because every planet moves every frame.
#[derive(Resource, ShaderType, Clone, Default)]
struct PlanetPositions {
    positions: [Vec2; N],
}

// CPU-only resource: tracks each planet's current orbital angle and how many
// full orbits it has completed. Not a GPU buffer.
#[derive(Resource)]
struct OrbitalState {
    angles: [f32; N],
    orbit_counts: [u32; N],
}

impl Default for OrbitalState {
    fn default() -> Self {
        let mut s = Self { angles: [0.0; N], orbit_counts: [0; N] };
        // Spread starting positions so planets don't all begin at the same point.
        for i in 0..N {
            s.angles[i] = i as f32 / N as f32 * std::f32::consts::TAU;
        }
        s
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::new(SHADER_PATH))
        // Uniform buffer — frame-level resolution
        .register_uniform_buffer::<FrameUniform>(0, 0)
        .init_resource::<FrameUniform>()
        // Storage buffer — all planet positions, fully re-uploaded every frame
        .register_storage_buffer::<PlanetPositions>(1, 0, false)
        .init_resource::<PlanetPositions>()
        // Array buffer — per-planet colors; only updated when a planet completes an orbit
        .register_array_buffer::<PlanetColors, Vec4, N>(2, 0, false)
        .init_resource::<OrbitalState>()
        .add_systems(Startup, setup)
        .add_systems(Update, (update_resolution, update_planets))
        .run();
}

fn setup(mut commands: Commands, mut changes: ResMut<ArrayBufferChanges<PlanetColors>>) {
    commands.spawn((Camera3d::default(), Msaa::Off));
    for i in 0..N {
        changes.set(i, planet_color(i, 0));
    }
}

fn update_resolution(
    mut uniform: ResMut<FrameUniform>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let Ok(window) = windows.single() else { return };
    uniform.resolution = Vec2::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
    );
}

fn update_planets(
    mut positions: ResMut<PlanetPositions>,
    mut orbital: ResMut<OrbitalState>,
    mut changes: ResMut<ArrayBufferChanges<PlanetColors>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    let tau = std::f32::consts::TAU;
    for i in 0..N {
        // Orbital radii spread evenly from 0.10 to 0.40 (UV units, 0..1 range).
        let r = 0.10 + 0.30 * (i as f32 / (N - 1) as f32);
        // Kepler-like angular speed: inner planets orbit faster (ω ∝ r^-1.5).
        // With this constant, the innermost planet takes ~10 s and outermost ~80 s.
        let speed = 0.02 / r.powf(1.5);
        let prev = orbital.angles[i];
        orbital.angles[i] += speed * dt;
        // Detect when an orbit completes and upload a new color (sparse array update);
        // floor math fires exactly once per revolution regardless of frame rate.
        if (orbital.angles[i] / tau).floor() > (prev / tau).floor() {
            orbital.orbit_counts[i] = orbital.orbit_counts[i].wrapping_add(1);
            changes.set(i, planet_color(i, orbital.orbit_counts[i]));
        }
        let a = orbital.angles[i];
        positions.positions[i] = Vec2::new(0.5 + r * a.cos(), 0.5 + r * a.sin());
    }
}

fn planet_color(planet_idx: usize, orbit_count: u32) -> Vec4 {
    let palette = [
        Vec4::new(0.9, 0.3, 0.2, 1.0), // red
        Vec4::new(1.0, 0.6, 0.1, 1.0), // orange
        Vec4::new(0.9, 0.9, 0.2, 1.0), // yellow
        Vec4::new(0.2, 0.8, 0.3, 1.0), // green
        Vec4::new(0.2, 0.7, 1.0, 1.0), // sky blue
        Vec4::new(0.3, 0.3, 1.0, 1.0), // deep blue
        Vec4::new(0.7, 0.2, 1.0, 1.0), // violet
        Vec4::new(1.0, 0.2, 0.7, 1.0), // pink
    ];
    palette[(planet_idx + orbit_count as usize * 3) % palette.len()]
}
