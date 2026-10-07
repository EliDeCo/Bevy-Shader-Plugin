use bevy::prelude::*;
use bevy_shader_plugin::prelude::*;

// One file holds both the compute passes and the fragment shader that draws the result.
const SHADER_PATH: &str = "shaders/particles.wgsl";
const N: u32 = 500;
// Simulation steps per frame. More substeps keep piles steadier at a higher GPU cost.
const SUBSTEPS: usize = 4;

// Tuning, in physical pixels and seconds.
const RADIUS: f32 = 8.0;
const GRAVITY: f32 = 1500.0;
const RESTITUTION: f32 = 0.5;

// Tags naming the two GPU-owned buffers.
struct Particles;
struct Corrections;

// group(1) binding(0) — particle state. Lives only on the GPU; `init` fills it.
// Read back to the CPU once a second to print averages.
#[derive(ShaderType, Clone, Copy, Default)]
struct Particle {
    pos: Vec2,
    vel: Vec2,
}

// group(1) binding(1) — scratch space `collide` writes and `integrate` reads.
#[derive(ShaderType, Clone, Copy, Default)]
struct Correction {
    dpos: Vec2,
    dvel: Vec2,
}

// group(0) binding(0) — CPU-owned simulation parameters, uploaded whenever they change.
#[derive(Resource, ShaderType, Clone, Default)]
struct SimParams {
    resolution: Vec2,
    gravity: f32,
    dt: f32,
    radius: f32,
    restitution: f32,
    seed: u32,
}

fn main() {
    // `init` runs once; then each frame runs `SUBSTEPS` rounds of collide → integrate.
    // Workgroup counts come from N and the shader's @workgroup_size.
    let mut compute =
        ComputeShaderPlugin::new(SHADER_PATH).init_pass("init", Workgroups::over::<Particles>());
    for _ in 0..SUBSTEPS {
        compute = compute
            .pass("collide", Workgroups::over::<Particles>())
            .pass("integrate", Workgroups::over::<Particles>());
    }

    App::new()
        // Half of Bevy's default 1280x720.
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (640, 360).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((compute, FullscreenFragmentPlugin::new(SHADER_PATH)))
        .register_uniform_buffer::<SimParams>(0, 0)
        .insert_resource(SimParams {
            gravity: GRAVITY,
            radius: RADIUS,
            restitution: RESTITUTION,
            ..default()
        })
        .register_gpu_buffer::<Particles, Particle>(1, 0, N)
        .register_gpu_buffer::<Corrections, Correction>(1, 1, N)
        .read_back_gpu_buffer::<Particles, Particle>(ReadbackMode::OnRequest)
        .insert_resource(StatsTimer(Timer::from_seconds(1.0, TimerMode::Repeating)))
        .add_systems(Startup, setup)
        .add_systems(Update, (update_params, reset, request_stats, print_stats))
        .run();
}

// How often the particle buffer is read back for `print_stats`.
#[derive(Resource)]
struct StatsTimer(Timer);

fn setup(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Msaa::Off));
    info!("Press R to respawn the particles");
}

fn update_params(
    mut params: ResMut<SimParams>,
    windows: Query<&Window, With<PrimaryWindow>>,
    time: Res<Time>,
) {
    let Ok(window) = windows.single() else { return };
    params.resolution = Vec2::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
    );
    // Cap the step so a hitch (e.g. dragging the window) can't tunnel particles.
    params.dt = time.delta_secs().min(1.0 / 30.0) / SUBSTEPS as f32;
    params.seed = time.elapsed().as_nanos() as u32;
}

fn reset(keys: Res<ButtonInput<KeyCode>>, mut passes: ResMut<ComputePasses>) {
    if keys.just_pressed(KeyCode::KeyR) {
        passes.rerun("init");
    }
}

fn request_stats(
    time: Res<Time>,
    mut timer: ResMut<StatsTimer>,
    readback: Res<GpuReadback<Particles, Particle>>,
) {
    if timer.0.tick(time.delta()).just_finished() {
        readback.request();
    }
}

// Runs when a readback arrives, a few frames after the request.
fn print_stats(readback: Res<GpuReadback<Particles, Particle>>) {
    if !readback.is_changed() {
        return;
    }
    let Some(particles) = readback.latest() else {
        return;
    };
    let n = particles.len() as f32;
    let avg_pos = particles.iter().map(|p| p.pos).sum::<Vec2>() / n;
    let avg_vel = particles.iter().map(|p| p.vel).sum::<Vec2>() / n;
    info!(
        "average position ({:.1}, {:.1}) px, average velocity ({:.1}, {:.1}) px/s",
        avg_pos.x, avg_pos.y, avg_vel.x, avg_vel.y
    );
}
