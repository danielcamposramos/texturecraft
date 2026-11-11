//! One arrow of the symmetry manipulator

use egui::Pos2;
use texturecraft_engine::geom::glam::Vec3;

/// An arrow along one world axis, in 3D (so it foreshortens) and where it lands on screen
#[derive(Clone, Copy, Debug)]
pub(crate) struct SymmetryManipulatorArrow {
    pub(crate) axis: usize,
    /// The world axis, pointing toward the tip
    pub(crate) direction: Vec3,
    /// Where the shaft starts, just off the center, and where it meets the cone
    pub(crate) start: Pos2,
    pub(crate) base: Pos2,
    pub(crate) tip: Pos2,
    pub(crate) base_world: Vec3,
    pub(crate) tip_world: Vec3,
}
