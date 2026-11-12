/// One texture set's cage on the GPU: three corners per triangle and the wireframe's corner pairs
pub(crate) struct CageGpu {
    pub(crate) bounds: Option<(texturecraft_geom::glam::Vec3, texturecraft_geom::glam::Vec3)>,
    /// Identifies the corners' order and the edges: a new distance with the same topology only rewrites `corners`
    pub(crate) topology: u64,
    pub(crate) corners: wgpu::Buffer,
    pub(crate) corner_count: u32,
    pub(crate) edges: wgpu::Buffer,
    pub(crate) edge_index_count: u32,
}
