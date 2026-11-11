//! How the 3D view shows symmetry: the settings popup's Display section

use serde::{Deserialize, Serialize};

/// The manipulator size the popup starts at; the manipulator's measurements are given at this size
pub(crate) const DEFAULT_MANIPULATOR_SIZE: f32 = 0.4;

/// Display settings for the symmetry plane, axis, intersection, cursors and manipulator
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SymmetryDisplay {
    /// Linear RGB of everything drawn for symmetry
    pub color: [f32; 3],
    /// Mirror: a see-through plane between the two sides
    pub show_plane: bool,
    /// Radial: a see-through line along the axis
    pub show_axis: bool,
    /// Where the plane cuts the mesh (a line) or the axis meets it (a point)
    pub show_intersection: bool,
    /// A brush cursor at every copy of the cursor
    pub show_cursor: bool,
    pub hide_while_painting: bool,
    pub show_manipulator: bool,
    /// Manipulator scale; [`DEFAULT_MANIPULATOR_SIZE`] draws it at its base size
    pub manipulator_size: f32,
}

impl Default for SymmetryDisplay {
    fn default() -> Self {
        SymmetryDisplay {
            color: [1.0, 0.0, 0.0],
            show_plane: false,
            show_axis: false,
            show_intersection: true,
            show_cursor: true,
            hide_while_painting: true,
            show_manipulator: false,
            manipulator_size: DEFAULT_MANIPULATOR_SIZE,
        }
    }
}
