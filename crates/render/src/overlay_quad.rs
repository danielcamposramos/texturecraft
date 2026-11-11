//! See-through quads drawn into the 3D view and hidden where the mesh is in front (the symmetry plane and lines)

use texturecraft_geom::glam::Vec3;

/// A flat quad with its corners in order, in a screen color (linear RGB) and opacity
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayQuad {
    pub corners: [Vec3; 4],
    pub color: [f32; 3],
    pub opacity: f32,
}
