//! What a pointer position in a view is over, for the Path tool

use egui::Pos2;
use texturecraft_engine::core::LayerId;

/// What a pointer position in a view is over, for the Path tool
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PathHit {
    /// A control point of the path being edited
    Point(usize),
    /// The edited path's curve: the segment starting at this point, and the nearest screen position on it
    Curve(usize, Pos2),
    /// Another stored path of the selected layer: (path sublayer, index)
    Stored(LayerId, usize),
    /// Nothing of a path
    Miss,
}
