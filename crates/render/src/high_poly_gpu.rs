/// The high poly bake mode draws over the low poly
pub(crate) struct HighPolyGpu {
    pub(crate) bounds: Option<(texturecraft_geom::glam::Vec3, texturecraft_geom::glam::Vec3)>,
    /// Which mesh the buffers hold
    pub(crate) id: u64,
    pub(crate) vertices: wgpu::Buffer,
    pub(crate) vertex_count: usize,
    pub(crate) indices: wgpu::Buffer,
    pub(crate) index_count: u32,
    /// One value per vertex: 1 where the bake's rays miss it (drawn red), 0 where they reach it
    pub(crate) missed: wgpu::Buffer,
}
