//! Where a symmetry copy of a stamp comes from

use texturecraft_geom::glam::{Mat3, Vec3};

/// The map from a symmetry copy back to its source stamp: a copy paints where its source would be painted
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StampSource {
    /// The point the symmetry maps turn and mirror about
    pub origin: Vec3,
    /// The inverse of the copy's linear map (mirrors and turns, so its transpose)
    pub to_source: Mat3,
}

impl StampSource {
    /// The source's point for a point of the copy
    pub fn point(&self, copy_point: Vec3) -> Vec3 {
        self.origin + self.to_source * (copy_point - self.origin)
    }

    /// The source's direction for a direction of the copy
    pub fn direction(&self, copy_direction: Vec3) -> Vec3 {
        self.to_source * copy_direction
    }
}
