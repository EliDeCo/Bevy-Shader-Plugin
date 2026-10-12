use spirv_builder::SpirvBuilder;
use std::{env, fs, path::PathBuf};

// (shader crate under `shaders/`, output file in `assets/shaders/`)
const SHADERS: &[(&str, &str)] = &[
    ("solar_system", "solar_system_rust.spv"),
    ("particles", "particles_rust.spv"),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    // shader-build -> rust_gpu -> examples -> repo root
    let assets = manifest.ancestors().nth(3).unwrap().join("assets/shaders");
    fs::create_dir_all(&assets)?;

    for (crate_name, output) in SHADERS {
        let mut builder = SpirvBuilder::new(
            manifest.join("shaders").join(crate_name),
            "spirv-unknown-vulkan1.4",
        );
        builder.build_script.defaults = true;
        let result = builder.build()?;

        let dest = assets.join(output);
        fs::copy(result.module.unwrap_single(), &dest)?;
        println!(
            "cargo::warning={crate_name} entry points: {:?}",
            result.entry_points
        );
        println!("cargo::warning=wrote {}", dest.display());
    }

    Ok(())
}
