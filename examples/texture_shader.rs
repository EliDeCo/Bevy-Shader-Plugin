use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, RenderApp,
        render_asset::RenderAssets,
        render_resource::{Extent3d, ShaderType, TextureDimension, TextureFormat},
        renderer::RenderDevice,
        texture::GpuImage,
    },
    window::PrimaryWindow,
};
use bevy_fragment_shader::{
    FragmentAppExt, FragmentExtraBindGroups, FragmentExtraLayouts, FullscreenFragmentPlugin,
    FullscreenPipeline,
};

const SHADER_PATH: &str = "shaders/texture_shader.wgsl";

#[derive(Resource, ShaderType, Clone, Default)]
struct ExampleUniform {
    resolution: Vec2,
    time: f32,
    _pad: f32,
}

/// Main-world handle to the procedurally generated texture.
#[derive(Resource, Clone)]
struct SceneTexture(Handle<Image>);

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .add_plugins(FullscreenFragmentPlugin::<ExampleUniform>::new(SHADER_PATH))
        .init_resource::<ExampleUniform>()
        .register_fragment_extra_bind_group(setup_texture_layout, prepare_texture_bind_group)
        .add_systems(Startup, setup)
        .add_systems(Update, update_uniform);

    // Extraction for SceneTexture is wired up in the render app separately from
    // register_fragment_extra_bind_group, which only handles the layout/prepare pair.
    app.sub_app_mut(RenderApp)
        .add_systems(ExtractSchedule, extract_scene_texture);

    app.run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((Camera3d::default(), Msaa::Off));

    // Build a 64×64 gradient image procedurally so no texture file is needed.
    let size = 64u32;
    let mut data = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..size {
            let i = ((y * size + x) * 4) as usize;
            data[i] = ((x * 255) / size) as u8;      // R: horizontal gradient
            data[i + 1] = ((y * 255) / size) as u8;  // G: vertical gradient
            data[i + 2] = 128;                         // B: constant mid-blue
            data[i + 3] = 255;                         // A: opaque
        }
    }
    let image = Image::new(
        Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        default(),
    );
    let handle = images.add(image);
    commands.insert_resource(SceneTexture(handle));
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

fn extract_scene_texture(
    mut commands: Commands,
    texture: Extract<Option<Res<SceneTexture>>>,
) {
    if let Some(t) = texture.as_deref() {
        commands.insert_resource(SceneTexture(t.0.clone()));
    }
}

fn setup_texture_layout(mut extra_layouts: ResMut<FragmentExtraLayouts>) {
    extra_layouts.texture_2d_and_sampler("scene_texture_layout");
}

fn prepare_texture_bind_group(
    mut extra_bind_groups: ResMut<FragmentExtraBindGroups>,
    render_device: Res<RenderDevice>,
    pipeline: Option<Res<FullscreenPipeline<ExampleUniform>>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    texture: Option<Res<SceneTexture>>,
) {
    let (Some(pipeline), Some(texture)) = (pipeline, texture) else { return };
    let Some(gpu_image) = gpu_images.get(&texture.0) else { return };

    extra_bind_groups
        .clear()
        .push_gpu_image("scene_texture_bg", &pipeline, &render_device, 0, gpu_image);
}
