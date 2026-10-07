use std::collections::HashMap;

use bevy::{
    prelude::*,
    shader::{Shader, Source},
};

use super::ComputeConfig;

/// Workgroup size of one compute entry point, as declared by `@workgroup_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkgroupSize {
    Fixed(UVec3),
    /// The size depends on pipeline-overridable constants, so it can't be known from the
    /// source alone.
    Overridden,
}

/// Reflected workgroup sizes for the compute shader, keyed by entry point name.
///
/// `sizes` is `None` until the shader has loaded and parsed successfully.
#[derive(Resource, Default)]
pub(crate) struct ComputeReflection {
    pub sizes: Option<HashMap<String, WorkgroupSize>>,
}

/// Reads the workgroup size of every compute entry point in `source`.
pub(crate) fn compute_workgroup_sizes(
    source: &Source,
) -> Result<HashMap<String, WorkgroupSize>, String> {
    match source {
        Source::Wgsl(code) => wgsl_workgroup_sizes(code),
        // rust-gpu support will parse SPIR-V here with naga's `spv-in` frontend.
        _ => Err(
            "workgroup size reflection only supports WGSL shaders; use `Workgroups::exact` instead"
                .into(),
        ),
    }
}

pub(crate) fn wgsl_workgroup_sizes(code: &str) -> Result<HashMap<String, WorkgroupSize>, String> {
    let module = naga::front::wgsl::parse_str(code).map_err(|e| e.emit_to_string(code))?;
    Ok(module
        .entry_points
        .iter()
        .filter(|entry| entry.stage == naga::ShaderStage::Compute)
        .map(|entry| {
            let size = if entry.workgroup_size_overrides.is_some() {
                WorkgroupSize::Overridden
            } else {
                WorkgroupSize::Fixed(UVec3::from_array(entry.workgroup_size))
            };
            (entry.name.clone(), size)
        })
        .collect())
}

/// Re-reflects the compute shader whenever it finishes loading or is hot-reloaded.
pub(crate) fn reflect_workgroup_sizes(
    mut events: MessageReader<AssetEvent<Shader>>,
    shaders: Res<Assets<Shader>>,
    config: Res<ComputeConfig>,
    mut reflection: ResMut<ComputeReflection>,
) {
    let id = config.shader.id();
    let reload = events.read().any(|event| {
        matches!(
            event,
            AssetEvent::LoadedWithDependencies { id: event_id } | AssetEvent::Modified { id: event_id }
                if *event_id == id
        )
    });
    if !reload {
        return;
    }
    let Some(shader) = shaders.get(id) else {
        return;
    };

    match compute_workgroup_sizes(&shader.source) {
        Ok(sizes) => {
            for pass in &config.passes {
                if !sizes.contains_key(pass.entry) {
                    let mut available: Vec<&str> = sizes.keys().map(String::as_str).collect();
                    available.sort_unstable();
                    error!(
                        "compute shader `{}` has no compute entry point `{}` (available: {:?})",
                        config.shader_path, pass.entry, available
                    );
                }
            }
            reflection.sizes = Some(sizes);
        }
        Err(message) => {
            error!(
                "could not reflect workgroup sizes from `{}`: {message}",
                config.shader_path
            );
            reflection.sizes = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflects_compute_entry_points_only() {
        let sizes = wgsl_workgroup_sizes(
            "
            @compute @workgroup_size(64) fn a() {}
            @compute @workgroup_size(8, 8) fn b() {}
            @fragment fn f() -> @location(0) vec4<f32> { return vec4(1.0); }
            ",
        )
        .unwrap();
        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes["a"], WorkgroupSize::Fixed(UVec3::new(64, 1, 1)));
        assert_eq!(sizes["b"], WorkgroupSize::Fixed(UVec3::new(8, 8, 1)));
    }

    #[test]
    fn detects_override_sized_workgroups() {
        let sizes = wgsl_workgroup_sizes(
            "
            override SIZE: u32 = 64;
            @compute @workgroup_size(SIZE) fn a() {}
            ",
        )
        .unwrap();
        assert_eq!(sizes["a"], WorkgroupSize::Overridden);
    }

    #[test]
    fn reports_parse_errors() {
        assert!(
            wgsl_workgroup_sizes("#import bevy_pbr::foo\n@compute @workgroup_size(1) fn a() {}")
                .is_err()
        );
    }
}
