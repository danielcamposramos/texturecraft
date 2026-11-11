//! A symmetry manipulator drag in progress

use egui::Pos2;

/// Kept in egui's memory between the frames of one drag
#[derive(Clone, Copy, Debug)]
pub(crate) struct SymmetryDrag {
    /// The world axes the grabbed handle moves along
    pub(crate) axes: [bool; 3],
    pub(crate) from: Pos2,
    pub(crate) start_offset: [f32; 3],
}
