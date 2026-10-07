use spirv_builder::SpirvBuilder;
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    // shader-build -> solar_system_rust -> examples -> repo root
    let project = manifest.ancestors().nth(3).unwrap();

    let mut builder = SpirvBuilder::new(manifest.join("shader"), "spirv-unknown-vulkan1.2");
    builder.build_script.defaults = true;
    let result = builder.build()?;

    let dest = project.join("assets/shaders/solar_system_rust.spv");
    fs::create_dir_all(dest.parent().unwrap())?;
    fs::copy(result.module.unwrap_single(), &dest)?;

    println!("cargo::warning=entry points: {:?}", result.entry_points);
    println!("cargo::warning=wrote {}", dest.display());

    Ok(())
}
