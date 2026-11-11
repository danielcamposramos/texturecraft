//! Where a mirror plane cuts the mesh, kept between frames

use std::sync::Arc;

use texturecraft_engine::geom::Mesh;
use texturecraft_engine::geom::glam::Vec3;

/// The segments where the plane `p[axis] = position` crosses the mesh's triangles
#[derive(Clone, Debug)]
pub(crate) struct SymmetryIntersection {
    mesh_id: u64,
    axis: usize,
    position: f32,
    pub(crate) segments: Arc<Vec<[Vec3; 2]>>,
}

impl SymmetryIntersection {
    /// Every triangle's crossing with the plane (mesh `mesh_id`, the plane `p[axis] = position`)
    pub(crate) fn compute(mesh: &Mesh, mesh_id: u64, axis: usize, position: f32) -> Self {
        let mut segments = Vec::new();
        for triangle in 0..mesh.triangle_count() {
            let Some([first, second, third]) = mesh.tri(triangle) else { continue };
            let (Some(&first), Some(&second), Some(&third)) = (mesh.positions.get(first), mesh.positions.get(second), mesh.positions.get(third)) else {
                continue;
            };

            let corners = [first, second, third];

            // Each edge whose ends lie on opposite sides crosses the plane once
            let side = |corner: usize| corners[corner][axis] - position;
            let mut crossings = [Vec3::ZERO; 2];
            let mut count = 0;
            for (start, end) in [(0, 1), (1, 2), (2, 0)] {
                if (side(start) < 0.0) != (side(end) < 0.0) && count < crossings.len() {
                    crossings[count] = corners[start].lerp(corners[end], side(start) / (side(start) - side(end)));
                    count += 1;
                }
            }
            if count == crossings.len() {
                segments.push(crossings);
            }
        }
        SymmetryIntersection { mesh_id, axis, position, segments: Arc::new(segments) }
    }

    /// Whether this is the cut of mesh `mesh_id` by the plane `p[axis] = position`
    pub(crate) fn is_for(&self, mesh_id: u64, axis: usize, position: f32) -> bool {
        self.mesh_id == mesh_id && self.axis == axis && self.position == position
    }
}
