use bevy::render::{
    render_resource::{BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, Buffer, Sampler, TextureView},
    renderer::RenderDevice,
};

/// Fluent builder for creating a single extra bind group.
///
/// Bindings are assigned sequentially starting at 0 for each call to
/// [`texture_view`](Self::texture_view), [`sampler`](Self::sampler), or
/// [`buffer`](Self::buffer). Use [`at`](Self::at) to set an explicit index.
pub struct FragmentBindGroupBuilder<'a> {
    label: Option<&'a str>,
    layout: &'a BindGroupLayout,
    render_device: &'a RenderDevice,
    entries: Vec<BindGroupEntry<'a>>,
}

impl<'a> FragmentBindGroupBuilder<'a> {
    pub fn new(layout: &'a BindGroupLayout, render_device: &'a RenderDevice) -> Self {
        Self {
            label: None,
            layout,
            render_device,
            entries: Vec::new(),
        }
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn texture_view(mut self, view: &'a TextureView) -> Self {
        let binding = self.entries.len() as u32;
        self.entries.push(BindGroupEntry {
            binding,
            resource: BindingResource::TextureView(view),
        });
        self
    }

    pub fn sampler(mut self, sampler: &'a Sampler) -> Self {
        let binding = self.entries.len() as u32;
        self.entries.push(BindGroupEntry {
            binding,
            resource: BindingResource::Sampler(sampler),
        });
        self
    }

    pub fn buffer(mut self, buffer: &'a Buffer) -> Self {
        let binding = self.entries.len() as u32;
        self.entries.push(BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        });
        self
    }

    /// Bind an arbitrary [`BindingResource`] at an explicit binding index.
    pub fn at(mut self, binding: u32, resource: BindingResource<'a>) -> Self {
        self.entries.push(BindGroupEntry { binding, resource });
        self
    }

    pub fn build(self) -> BindGroup {
        self.render_device.create_bind_group(
            self.label.unwrap_or("fragment_extra_bind_group"),
            self.layout,
            &self.entries,
        )
    }
}
