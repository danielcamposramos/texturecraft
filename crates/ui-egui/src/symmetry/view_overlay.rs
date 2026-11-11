//! What the 3D view draws for symmetry: the plane or axis, where they meet the mesh, the cursor copies and the manipulator

use egui::{Color32, PointerButton, Pos2, Rect, Response, Stroke, Ui};
use texturecraft_engine::geom::glam::Vec3;
use texturecraft_render::OverlayQuad;

use super::ribbons::Ribbons;
use super::shown;
use super::symmetry_drag::SymmetryDrag;
use super::symmetry_intersection::SymmetryIntersection;
use super::symmetry_manipulator::SymmetryManipulator;
use super::symmetry_manipulator_arrow::SymmetryManipulatorArrow;
use crate::TextureCraftApp;

/// The plane reaches this far past the mesh, as a fraction of its size
const PLANE_MARGIN: f32 = 0.05;
const PLANE_OPACITY: f32 = 0.18;

/// The radial axis line's width (points) and opacity
const AXIS_WIDTH: f32 = 1.5;
const AXIS_OPACITY: f32 = 0.85;

/// The intersection line's opacity
const LINE_OPACITY: f32 = 0.9;

/// A point counts as visible when nothing is in front of it by more than this fraction of the mesh diagonal
const VISIBILITY_TOLERANCE: f32 = 2e-3;

/// Width of the cut line and radius of the radial axis points (points)
const INTERSECTION_WIDTH: f32 = 1.5;
const INTERSECTION_POINT_RADIUS: f32 = 3.0;

/// Manipulator arrows along X, Y and Z (the view-corner axis colors, a little softer)
const AXIS_COLORS: [Color32; 3] = [Color32::from_rgb(0xd8, 0x60, 0x4f), Color32::from_rgb(0x6c, 0xbc, 0x5a), Color32::from_rgb(0x5a, 0x8c, 0xe0)];

/// Arrow shaft width, its dark halo and the halo's extra width (points at the default size)
const SHAFT_WIDTH: f32 = 2.2;
const SHAFT_HALO: Color32 = Color32::from_black_alpha(128);
const SHAFT_HALO_EXTRA_WIDTH: f32 = 3.0;

/// Faces around a cone
const CONE_SEGMENTS: usize = 20;

/// Cone faces facing the camera more than this (cosine) are tinted toward white, the others toward black
const CONE_LIT_FROM: f32 = 0.55;

/// How strongly the cone faces' tint follows their facing, toward white and toward black
const CONE_LIGHTEN: f32 = 0.9;
const CONE_DARKEN: f32 = 0.8;

/// The center: a small disc in the arrows' material, its rim shaded and its middle lit like the cones
const CENTER_RADIUS: f32 = 4.0;
const CENTER_LIT_RADIUS: f32 = 0.6;
const CENTER_SHADE: f32 = 0.3;
const CENTER_LIGHT: f32 = 0.25;

/// The radial square at rest and the opacity of its accent fill when active
const SQUARE_EDGE_WIDTH: f32 = 1.3;
const SQUARE_FILL: Color32 = Color32::from_rgba_premultiplied(40, 40, 40, 40);
const SQUARE_EDGE: Color32 = Color32::from_rgba_premultiplied(190, 190, 190, 190);
const SQUARE_ACTIVE_OPACITY: f32 = 0.45;

/// `linear` RGB as a screen color with `alpha`
pub(super) fn color(linear: [f32; 3], alpha: u8) -> Color32 {
    let [r, g, b] = linear.map(|channel| (texturecraft_engine::texel::linear_to_srgb(channel.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8);
    Color32::from_rgba_unmultiplied(r, g, b, alpha)
}

/// The planes, the axis and where the planes cut the mesh, drawn by the renderer: the mesh hides them pixel by pixel where it is in front
pub(crate) fn scene_overlay(app: &TextureCraftApp, ui: &Ui, rect: Rect) -> Vec<OverlayQuad> {
    if !shown(app) {
        return Vec::new();
    }

    let symmetry = app.session.tool.brush.symmetry;
    let display = app.ui.symmetry;
    let (lo, hi) = app.session.mesh.bounds();
    let origin = symmetry.origin((lo + hi) * 0.5);
    let ribbon = Ribbons::new(app, rect);
    let mut quads = Vec::new();

    for (axis, on) in [symmetry.x, symmetry.y, symmetry.z].into_iter().enumerate() {
        if !on {
            continue;
        }

        if display.show_plane {
            let margin = (hi - lo) * PLANE_MARGIN;
            let (lo, hi) = (lo - margin, hi + margin);
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let corner = |at_u: f32, at_v: f32| {
                let mut point = origin;
                (point[u], point[v]) = (at_u, at_v);
                point
            };
            let corners = [corner(lo[u], lo[v]), corner(hi[u], lo[v]), corner(hi[u], hi[v]), corner(lo[u], hi[v])];
            quads.push(OverlayQuad { corners, color: display.color, opacity: PLANE_OPACITY });
        }
        if display.show_intersection {
            let segments = intersection(app, ui, axis, origin[axis]);
            quads.extend(segments.iter().map(|[from, to]| ribbon.quad(*from, *to, INTERSECTION_WIDTH, display.color, LINE_OPACITY)));
        }
    }

    if display.show_axis && symmetry.radial_mode() {
        let direction = symmetry.radial_axis();
        let reach = (hi - lo).length() * 0.5;
        quads.push(ribbon.quad(origin - direction * reach, origin + direction * reach, AXIS_WIDTH, display.color, AXIS_OPACITY));
    }

    quads
}

/// Whether `point` is on the mesh's surface where the camera sees it: in the air (a copy that missed the mesh) or hidden behind the mesh, it is not
fn on_visible_surface(app: &TextureCraftApp, rect: Rect, point: Vec3, tolerance: f32) -> bool {
    let camera = &app.session.camera;
    let aspect = rect.width() / rect.height().max(1.0);
    let Some(ndc) = camera.project(point, aspect) else { return false };

    let ray = camera.ray(ndc.truncate(), aspect);
    let distance = (point - ray.origin).dot(ray.dir);
    app.session.bvh.closest(ray.origin, ray.dir, f32::MAX).is_some_and(|hit| (hit.t - distance).abs() <= tolerance)
}

/// Over the 3D view: where the planes or axis meet the mesh, copies of the brush `cursor` (position, normal) and the manipulator
pub(crate) fn overlay(app: &TextureCraftApp, ui: &Ui, rect: Rect, cursor: Option<(Vec3, Vec3)>) {
    if !shown(app) {
        return;
    }

    let symmetry = app.session.tool.brush.symmetry;
    let display = app.ui.symmetry;
    let (lo, hi) = app.session.mesh.bounds();
    let origin = symmetry.origin((lo + hi) * 0.5);

    // Bounds scan the mesh's vertices; every cursor copy shares these measurements
    let diagonal = (hi - lo).length();
    let tolerance = diagonal * VISIBILITY_TOLERANCE;
    let brush_radius = app.session.tool.brush.size * diagonal;

    let painter = ui.painter_at(rect);
    let line = Stroke::new(INTERSECTION_WIDTH, color(display.color, u8::MAX));
    let project = |point: Vec3| crate::viewport::project(app, rect, point);

    // The planes' lines are drawn in the scene (see scene_overlay); where the radial axis enters the mesh, from either end, is drawn here
    if display.show_intersection && symmetry.radial_mode() {
        let axis = symmetry.radial_axis();
        let reach = diagonal;
        for (start, direction) in [(origin - axis * reach, axis), (origin + axis * reach, -axis)] {
            if let Some(hit) = app.session.bvh.closest(start, direction, reach * 2.0)
                && let point = start + direction * hit.t
                && on_visible_surface(app, rect, point, tolerance)
                && let Some(at) = project(point)
            {
                painter.circle_filled(at, INTERSECTION_POINT_RADIUS, line.color);
            }
        }
    }

    let painting = app.session.stroke_active();
    if display.show_cursor
        && !(display.hide_while_painting && painting)
        && let Some((position, normal)) = cursor
    {
        for map in symmetry.transforms().iter().skip(1) {
            let copy = origin + *map * (position - origin);
            if on_visible_surface(app, rect, copy, tolerance) {
                crate::viewport::brush_cursor(app, ui, rect, copy, (*map * normal).normalize_or_zero(), brush_radius);
            }
        }
    }

    if let Some(manipulator) = SymmetryManipulator::of(app, rect) {
        draw_manipulator(app, ui, rect, &painter, &manipulator);
    }
}

/// Where the mirror plane `p[axis] = position` cuts the mesh, recomputed only when it moves
fn intersection(app: &TextureCraftApp, ui: &Ui, axis: usize, position: f32) -> std::sync::Arc<Vec<[Vec3; 2]>> {
    let id = egui::Id::unique("symmetry_intersection").with(axis);
    let mesh_id = app.session.mesh_id;
    let cached = ui.ctx().data(|data| data.get_temp::<SymmetryIntersection>(id)).filter(|cached| cached.is_for(mesh_id, axis, position));
    let current = cached.unwrap_or_else(|| {
        let computed = SymmetryIntersection::compute(&app.session.mesh, mesh_id, axis, position);
        ui.ctx().data_mut(|data| data.insert_temp(id, computed.clone()));
        computed
    });
    current.segments
}

/// The manipulator: the radial square, the arrows with their cones and the center; copper where hovered or dragged
fn draw_manipulator(app: &TextureCraftApp, ui: &Ui, rect: Rect, painter: &egui::Painter, manipulator: &SymmetryManipulator) {
    let accent = crate::theme::Tokens::get(ui.ctx()).viewport_accent;
    let hovered = ui.ctx().pointer_hover_pos().and_then(|pointer| manipulator.grab(pointer));
    let dragging = ui.ctx().data(|data| data.get_temp::<SymmetryDrag>(drag_id())).map(|drag| drag.axes);
    let highlighted = dragging.or(hovered).unwrap_or_default();
    let all_highlighted = manipulator.arrows.iter().all(|arrow| highlighted[arrow.axis]);
    let tint = |axis: usize| if highlighted[axis] { accent } else { AXIS_COLORS[axis] };

    if let Some(square) = manipulator.square {
        let fill = if all_highlighted { accent.gamma_multiply(SQUARE_ACTIVE_OPACITY) } else { SQUARE_FILL };
        let edge = if all_highlighted { accent } else { SQUARE_EDGE };
        painter.add(egui::Shape::convex_polygon(square.to_vec(), fill, Stroke::new(SQUARE_EDGE_WIDTH * manipulator.scale, edge)));
    }

    for arrow in &manipulator.arrows {
        let color = tint(arrow.axis);
        painter.line_segment([arrow.start, arrow.base], Stroke::new((SHAFT_WIDTH + SHAFT_HALO_EXTRA_WIDTH) * manipulator.scale, SHAFT_HALO));
        painter.line_segment([arrow.start, arrow.base], Stroke::new(SHAFT_WIDTH * manipulator.scale, color));
        cone(app, rect, painter, arrow, manipulator.world_per_point * super::symmetry_manipulator::CONE_RADIUS * manipulator.scale, color);
    }

    let center_color = if all_highlighted { accent } else { manipulator.arrows.first().map_or(AXIS_COLORS[0], |arrow| AXIS_COLORS[arrow.axis]) };
    painter.circle_filled(manipulator.center, CENTER_RADIUS * manipulator.scale, center_color.lerp_to_gamma(Color32::BLACK, CENTER_SHADE));
    painter.circle_filled(manipulator.center, CENTER_RADIUS * CENTER_LIT_RADIUS * manipulator.scale, center_color.lerp_to_gamma(Color32::WHITE, CENTER_LIGHT));
}

/// A flat-shaded 3D cone over `arrow`'s head: faces facing the camera lighter, back to front
fn cone(app: &TextureCraftApp, rect: Rect, painter: &egui::Painter, arrow: &SymmetryManipulatorArrow, radius: f32, color: Color32) {
    let camera = &app.session.camera;
    let (eye, forward) = (camera.eye(), camera.forward());
    let side = arrow.direction.any_orthonormal_vector();
    let up = arrow.direction.cross(side);
    let rim: Vec<Vec3> = (0..CONE_SEGMENTS)
        .map(|segment| {
            let angle = segment as f32 / CONE_SEGMENTS as f32 * std::f32::consts::TAU;
            arrow.base_world + (side * angle.cos() + up * angle.sin()) * radius
        })
        .collect();

    // Each face: its corners, its depth and how much it faces the camera
    let mut faces: Vec<(Vec<Vec3>, f32, f32)> = (0..CONE_SEGMENTS)
        .map(|segment| {
            let (rim_start, rim_end) = (rim[segment], rim[(segment + 1) % CONE_SEGMENTS]);
            let middle = (rim_start + rim_end + arrow.tip_world) / 3.0;
            let normal = (rim_end - rim_start).cross(arrow.tip_world - rim_start).normalize_or_zero();
            (vec![arrow.tip_world, rim_start, rim_end], (middle - eye).dot(forward), normal.dot((eye - middle).normalize_or_zero()))
        })
        .collect();
    faces.push((rim.clone(), (arrow.base_world - eye).dot(forward), (-arrow.direction).dot((eye - arrow.base_world).normalize_or_zero())));
    faces.sort_by(|(_, nearer, _), (_, farther, _)| farther.total_cmp(nearer));

    // One plain triangle mesh: anti-aliased polygons smear thin edge-on faces into long hairlines
    let mut mesh = egui::Mesh::default();
    for (corners, _, facing) in faces {
        let Some(points) = corners.iter().map(|corner| crate::viewport::project(app, rect, *corner)).collect::<Option<Vec<Pos2>>>() else {
            continue;
        };

        let facing = facing.max(0.0);
        let fill = if facing > CONE_LIT_FROM {
            color.lerp_to_gamma(Color32::WHITE, (facing - CONE_LIT_FROM) * CONE_LIGHTEN)
        } else {
            color.lerp_to_gamma(Color32::BLACK, (CONE_LIT_FROM - facing) * CONE_DARKEN)
        };
        let first = mesh.vertices.len() as u32;
        points.iter().for_each(|point| mesh.colored_vertex(*point, fill));
        for corner in 1..points.len().saturating_sub(1) as u32 {
            mesh.add_triangle(first, first + corner, first + corner + 1);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

fn drag_id() -> egui::Id {
    egui::Id::unique("symmetry_manipulator_drag")
}

/// Whether the pointer is on a manipulator handle or dragging one (the brush cursor hides then)
pub(crate) fn manipulator_hovered(app: &TextureCraftApp, ui: &Ui, rect: Rect) -> bool {
    let dragging = ui.ctx().data(|data| data.get_temp::<SymmetryDrag>(drag_id())).is_some();
    let hovered =
        ui.ctx().pointer_hover_pos().is_some_and(|pointer| SymmetryManipulator::of(app, rect).is_some_and(|manipulator| manipulator.grab(pointer).is_some()));
    dragging || hovered
}

/// Drags on the manipulator move the symmetry origin. Returns whether a drag owns the pointer
pub(crate) fn manipulator_input(app: &mut TextureCraftApp, ui: &Ui, response: &Response, rect: Rect) -> bool {
    let ctx = ui.ctx();
    let manipulator = SymmetryManipulator::of(app, rect);
    let mut drag = ctx.data(|data| data.get_temp::<SymmetryDrag>(drag_id()));
    if response.drag_started_by(PointerButton::Primary)
        && let (Some(manipulator), Some(from)) = (&manipulator, ctx.input(|input| input.pointer.press_origin()))
        && let Some(axes) = manipulator.grab(from)
    {
        drag = Some(SymmetryDrag { axes, from, start_offset: app.session.tool.brush.symmetry.offset });
    }

    let dragging = response.dragged_by(PointerButton::Primary) || response.drag_started_by(PointerButton::Primary);
    let (Some(manipulator), Some(drag)) = (manipulator, drag.filter(|_| dragging)) else {
        ctx.data_mut(|data| data.remove::<SymmetryDrag>(drag_id()));
        if let (Some(manipulator), Some(hover)) = (SymmetryManipulator::of(app, rect), response.hover_pos())
            && manipulator.grab(hover).is_some()
        {
            ctx.set_cursor_icon(egui::CursorIcon::Move);
        }

        return false;
    };

    if let Some(pointer) = response.interact_pointer_pos() {
        let (lo, hi) = app.session.mesh.bounds();
        let half_extent = (hi - lo) * 0.5;
        let moved = Vec3::from(drag.start_offset) + manipulator.world_delta(drag.axes, pointer - drag.from);
        let offset = moved.clamp(-half_extent, half_extent).to_array();
        if offset != app.session.tool.brush.symmetry.offset {
            app.run("tool.brush", serde_json::json!({"symmetryOffset": offset}));
        }
    }

    ctx.data_mut(|data| data.insert_temp(drag_id(), drag));
    ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
    true
}
