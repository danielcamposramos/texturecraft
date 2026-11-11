//! The axis position manipulator: arrows at the symmetry origin that move it

use egui::{Pos2, Rect, Vec2};
use texturecraft_engine::geom::glam::Vec3;

use super::symmetry_display::DEFAULT_MANIPULATOR_SIZE;
use super::symmetry_manipulator_arrow::SymmetryManipulatorArrow;
use crate::TextureCraftApp;

/// Arrow length at the default manipulator size, as a fraction of the view height
const ARROW_LENGTH: f32 = 0.1;

/// Gap between the center and where the shafts start (points at the default size)
const SHAFT_GAP: f32 = 5.0;

/// Cone length and base radius (points at the default size, at the origin's depth)
pub(crate) const CONE_LENGTH: f32 = 15.0;
pub(crate) const CONE_RADIUS: f32 = 5.0;

/// Radial: the square handle spans this part of the arrows, clear of the shafts
const SQUARE_FROM: f32 = 0.16;
const SQUARE_TO: f32 = 0.46;

/// How far (fraction of the mesh diagonal, at least a minimum) the screen scale is measured along each axis
const PROBE_FRACTION: f32 = 0.01;
const MIN_PROBE: f32 = 1e-4;

/// Shortest on-screen axis (points per world unit) the manipulator offers a handle for
const MIN_SCREEN_SCALE: f32 = 1e-3;

/// Grab radius around an arrow and around the center (points, at least this at any size)
const GRAB_RADIUS: f32 = 7.0;

/// The manipulator as the 3D view shows it this frame
#[derive(Clone, Debug)]
pub(crate) struct SymmetryManipulator {
    pub(crate) center: Pos2,
    pub(crate) arrows: Vec<SymmetryManipulatorArrow>,
    /// Radial: a square between the two arrows that moves the axis along both
    pub(crate) square: Option<[Pos2; 4]>,
    /// World units per screen point at the origin
    pub(crate) world_per_point: f32,
    /// The manipulator size relative to the default: every measurement scales with it
    pub(crate) scale: f32,
    /// Screen points per world unit along X, Y and Z at the origin
    screen_per_unit: [Vec2; 3],
}

impl SymmetryManipulator {
    /// Handles along the axes the plane or axis can move on, or None when the manipulator is hidden
    pub(crate) fn of(app: &TextureCraftApp, rect: Rect) -> Option<SymmetryManipulator> {
        let symmetry = app.session.tool.brush.symmetry;
        if !app.ui.symmetry.show_manipulator || !super::shown(app) {
            return None;
        }

        let project = |point: Vec3| crate::viewport::project(app, rect, point);
        let (lo, hi) = app.session.mesh.bounds();
        let origin = symmetry.origin((lo + hi) * 0.5);
        let center = project(origin)?;
        let step = ((hi - lo).length() * PROBE_FRACTION).max(MIN_PROBE);
        let screen_per_unit = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| project(origin + axis * step).map_or(Vec2::ZERO, |point| (point - center) / step));

        // Sized in points at the origin's depth, so the arrows keep their size on screen but still turn in 3D
        let right = app.session.camera.forward().cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        let points_per_unit = project(origin + right * step).map_or(0.0, |point| (point - center).length() / step);
        if points_per_unit <= MIN_SCREEN_SCALE {
            return None;
        }

        let world_per_point = 1.0 / points_per_unit;
        let scale = app.ui.symmetry.manipulator_size / DEFAULT_MANIPULATOR_SIZE;
        let length = rect.height() * ARROW_LENGTH * scale * world_per_point;

        let movable = super::movable_axes(&symmetry);
        let signs: &[f32] = if symmetry.radial_mode() { &[1.0] } else { &[1.0, -1.0] };
        let arrows: Vec<SymmetryManipulatorArrow> = (0..3)
            .filter(|axis| movable[*axis] && screen_per_unit[*axis].length() > MIN_SCREEN_SCALE)
            .flat_map(|axis| signs.iter().map(move |sign| (axis, [Vec3::X, Vec3::Y, Vec3::Z][axis] * *sign)))
            .filter_map(|(axis, direction)| {
                let tip_world = origin + direction * length;
                let base_world = origin + direction * (length - CONE_LENGTH * scale * world_per_point);
                let start = project(origin + direction * SHAFT_GAP * scale * world_per_point)?;
                Some(SymmetryManipulatorArrow { axis, direction, start, base: project(base_world)?, tip: project(tip_world)?, base_world, tip_world })
            })
            .collect();

        let square = match movable_pair(&movable) {
            Some((first, second)) if symmetry.radial_mode() => {
                let corner = |along_first: f32, along_second: f32| {
                    let mut point = origin;
                    (point[first], point[second]) = (origin[first] + along_first * length, origin[second] + along_second * length);
                    project(point)
                };
                let (from, to) = (SQUARE_FROM, SQUARE_TO);
                match [corner(from, from), corner(to, from), corner(to, to), corner(from, to)] {
                    [Some(near), Some(along_first), Some(far), Some(along_second)] => Some([near, along_first, far, along_second]),
                    _ => None,
                }
            }
            _ => None,
        };
        Some(SymmetryManipulator { center, arrows, square, world_per_point, scale, screen_per_unit })
    }

    /// The axes the handle under `point` moves along: the center and the square move along all movable axes (two at most)
    pub(crate) fn grab(&self, point: Pos2) -> Option<[bool; 3]> {
        let mut all = [false; 3];
        self.arrows.iter().for_each(|arrow| all[arrow.axis] = true);
        let grab_radius = GRAB_RADIUS * self.scale.max(1.0);

        // The center moves along all movable axes, which a screen drag can only do for two of them
        let center_grabs = all.iter().filter(|axis| **axis).count() <= 2 && point.distance(self.center) <= grab_radius;
        if center_grabs || self.square.is_some_and(|square| inside(&square, point)) {
            return Some(all);
        }

        let arrow = self.arrows.iter().find(|arrow| distance_to_segment(point, arrow.start, arrow.tip) <= grab_radius)?;
        let mut axes = [false; 3];
        axes[arrow.axis] = true;
        Some(axes)
    }

    /// The world offset change that follows a pointer move of `moved` along `axes`
    pub(crate) fn world_delta(&self, axes: [bool; 3], moved: Vec2) -> Vec3 {
        let picked: Vec<usize> = (0..3).filter(|axis| axes[*axis]).collect();
        let mut delta = Vec3::ZERO;
        match picked.as_slice() {
            [axis] => {
                let along = self.screen_per_unit[*axis];
                delta[*axis] = moved.dot(along) / along.length_sq().max(MIN_SCREEN_SCALE);
            }
            [first, second] => {
                // Solve moved = a * first_axis + b * second_axis on screen
                let (first_axis, second_axis) = (self.screen_per_unit[*first], self.screen_per_unit[*second]);
                let determinant = first_axis.x * second_axis.y - first_axis.y * second_axis.x;
                if determinant.abs() > MIN_SCREEN_SCALE {
                    delta[*first] = (moved.x * second_axis.y - moved.y * second_axis.x) / determinant;
                    delta[*second] = (first_axis.x * moved.y - first_axis.y * moved.x) / determinant;
                }
            }
            _ => {}
        }
        delta
    }
}

/// The two movable axes, when there are two
fn movable_pair(movable: &[bool; 3]) -> Option<(usize, usize)> {
    let picked: Vec<usize> = (0..3).filter(|axis| movable[*axis]).collect();
    match picked.as_slice() {
        [first, second] => Some((*first, *second)),
        _ => None,
    }
}

fn distance_to_segment(point: Pos2, from: Pos2, to: Pos2) -> f32 {
    let along = to - from;
    let fraction = ((point - from).dot(along) / along.length_sq().max(f32::EPSILON)).clamp(0.0, 1.0);
    point.distance(from + along * fraction)
}

/// Whether `point` is inside the convex quad `corners`
fn inside(corners: &[Pos2; 4], point: Pos2) -> bool {
    let side = |index: usize| {
        let (start, end) = (corners[index], corners[(index + 1) % 4]);
        (end - start).x * (point - start).y - (end - start).y * (point - start).x
    };
    let sides = [side(0), side(1), side(2), side(3)];
    sides.iter().all(|side| *side >= 0.0) || sides.iter().all(|side| *side <= 0.0)
}

#[cfg(test)]
mod tests {
    use egui::pos2;

    use super::*;

    #[test]
    fn a_two_axis_drag_splits_the_move_between_both_axes() {
        let manipulator = SymmetryManipulator {
            center: pos2(0.0, 0.0),
            arrows: Vec::new(),
            square: None,
            world_per_point: 1.0,
            scale: 1.0,
            screen_per_unit: [Vec2::new(10.0, 0.0), Vec2::new(0.0, -10.0), Vec2::ZERO],
        };

        let delta = manipulator.world_delta([true, true, false], Vec2::new(20.0, -30.0));

        assert!((delta - Vec3::new(2.0, 3.0, 0.0)).length() < 1e-5, "{delta}");
    }
}
