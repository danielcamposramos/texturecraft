//! The 3D viewport and the 2D UV view: navigation, painting input, brush cursor and the GPU
//! callbacks that render them.

use std::sync::{Arc, Mutex};

use egui::emath::GuiRounding as _;
use egui::{Color32, PointerButton, Pos2, Rect, Sense, Stroke, Ui, pos2, vec2};
use serde_json::json;
use texturecraft_engine::camera_cmds::DEFAULT_CAMERA;
use texturecraft_engine::core::BakedMap;
use texturecraft_engine::geom::glam::{Vec2, Vec3};
use texturecraft_engine::{Session, Tool, ViewMode};
use texturecraft_render::{Frame, SetHighlight, Shade, ViewKind};

use crate::TextureCraftApp;
use crate::decal_drop::DecalDrop;
use crate::gpu::ViewCallback;
use crate::i18n::{tr, trf};
use crate::path_hit::PathHit;
use crate::path_surface_cache::{PathSurfaceCache, SurfaceCurve};
use crate::state::ViewLayout;
use crate::theme::Tokens;
use crate::toolbar::{MAX_BRUSH_SIZE, MIN_BRUSH_SIZE};

pub const VIEW_3D: u64 = 1;
pub const VIEW_2D: u64 = 2;

/// While on, a click in the 3D view focuses the depth of field there (a transient UI mode kept in
/// egui's memory).
fn focus_pick_id() -> egui::Id {
    egui::Id::unique("tc_focus_pick")
}

pub fn focus_picking(ctx: &egui::Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(focus_pick_id())).unwrap_or(false)
}

pub fn set_focus_picking(ctx: &egui::Context, on: bool) {
    ctx.data_mut(|d| d.insert_temp(focus_pick_id(), on));
}

/// The viewport dock tab: the paint bar, then the 3D and 2D views in the chosen layout with the mode's overlays
pub fn central(app: &mut TextureCraftApp, ui: &mut Ui) {
    let mode = app.ui.mode;
    let bake_painting = mode == crate::state::Mode::Baking
        && app.ui.bake_paint_brush.on
        && !app.session.bake_meshes.high.is_empty()
        && app.session.doc.project.texture_sets.get(app.session.ts()).is_some_and(|texture_set| {
            !texture_set.bake.high_poly.use_low_as_high && texture_set.bake.cage.kind == texturecraft_engine::core::CageType::DistanceBased
        });
    app.ui.bake_paint_brush.on = bake_painting;

    if !app.ui.hide_ui {
        crate::toolbar::paint_bar(app, ui);
    }

    crate::id_color_pick::IdColorPick::end_when_stale(app);
    let tokens = Tokens::get(ui.ctx());
    let aux_view = match app.session.view {
        _ if mode == crate::state::Mode::Painting && app.session.quick_mask.editing && app.picking_id_color.is_none() => Some(crate::gpu::AuxView::QuickMask),
        ViewMode::MeshMap(map) => Some(crate::gpu::AuxView::MeshMap(map)),
        ViewMode::Mask(id) => Some(crate::gpu::AuxView::Mask(id)),
        _ => None,
    };
    let mut sync = PendingSync(Some(aux_view));
    let full = ui.available_rect_before_wrap();
    path_keys(app, ui);

    // Skew painting needs the 2D view alongside the 3D Bake preview
    let layout = if mode == crate::state::Mode::Painting || bake_painting { app.ui.layout } else { ViewLayout::Only3D };
    match layout {
        ViewLayout::Only3D => view_3d(app, ui, full, &mut sync),
        ViewLayout::Only2D => view_2d(app, ui, full, &mut sync),
        ViewLayout::Split => {
            // The 2D view's edges sit on whole pixels, so rounding can't nudge a texture its zoom keeps at one size
            let pixels_per_point = ui.ctx().pixels_per_point();
            let full = full.with_max_x(full.right().round_to_pixels(pixels_per_point));
            let divider_x = split_divider_x(app, ui, full);
            let left = Rect::from_min_max(full.min, pos2(divider_x - SPLIT_LINE_HALF_WIDTH, full.max.y));
            let right = Rect::from_min_max(pos2(divider_x + SPLIT_LINE_HALF_WIDTH, full.min.y), full.max);
            // Each view draws only on its side, so overlays such as the texture set name stay out of a narrow neighbour
            let clip = ui.clip_rect();
            ui.set_clip_rect(left.intersect(clip));
            view_3d(app, ui, left, &mut sync);
            ui.set_clip_rect(right.intersect(clip));
            view_2d(app, ui, right, &mut sync);
            ui.set_clip_rect(clip);
            split_divider_grab(ui, full, divider_x);
            let line = Stroke::new(SPLIT_LINE_HALF_WIDTH * 2.0, tokens.chrome);
            ui.painter().line_segment([pos2(divider_x, full.top()), pos2(divider_x, full.bottom())], line);
        }
    }

    crate::properties_popup::show(app, ui.ctx());

    if mode == crate::state::Mode::Rendering {
        crate::render_mode::show(app, ui, full);
    } else if app.render.is_active() {
        crate::render_mode::stop(app);
    }
    if mode == crate::state::Mode::Baking {
        crate::bake_mode::bake_bar(app, ui.ctx(), full);
    }

    // Keep drawing until the 3D view finishes accumulating anti-aliasing samples
    let accumulating = layout != ViewLayout::Only2D
        && app.render_state.as_ref().is_some_and(|render_state| {
            render_state.renderer.read().callback_resources.get::<crate::gpu::GpuState>().is_some_and(|gpu| gpu.renderer.accumulating(VIEW_3D))
        });
    if app.session.stroke_active() || accumulating {
        ui.ctx().request_repaint();
    }
}

/// Half the split divider's line width: each view starts this far from the line's centre
const SPLIT_LINE_HALF_WIDTH: f32 = 1.0;

/// Narrowest a split view may get: a sliver that is still easy to grab and drag back out
const MIN_SPLIT_VIEW_WIDTH: f32 = 16.0;

/// Width of the split divider's grab zone, centred on its line
const SPLIT_DIVIDER_GRAB_WIDTH: f32 = 8.0;

/// The split divider's widget id, shared by its drag read and its grab zone
fn split_divider_id(ui: &Ui) -> egui::Id {
    ui.scope_id().with("split_divider")
}

/// The 3D view's share of `width`, keeping both views at least [`MIN_SPLIT_VIEW_WIDTH`] wide when they fit
fn clamp_split_ratio(ratio: f32, width: f32) -> f32 {
    let min_ratio = (MIN_SPLIT_VIEW_WIDTH / width).min(0.5);
    let ratio = if ratio.is_finite() { ratio } else { 0.5 };
    ratio.clamp(min_ratio, 1.0 - min_ratio)
}

/// The split divider's x position: while it is dragged it follows the pointer and changes the 2D zoom so the texture keeps its size on screen
fn split_divider_x(app: &mut TextureCraftApp, ui: &Ui, full: Rect) -> f32 {
    // The 2D view's left edge, beside the line, sits on a whole pixel
    let pixels_per_point = ui.ctx().pixels_per_point();
    let divider_x = |ratio: f32| {
        let unsnapped = full.left() + full.width() * clamp_split_ratio(ratio, full.width());
        (unsnapped + SPLIT_LINE_HALF_WIDTH).round_to_pixels(pixels_per_point) - SPLIT_LINE_HALF_WIDTH
    };
    let old_x = divider_x(app.ui.split_ratio);

    // Read the drag before the views lay out so they follow the pointer in the same frame
    let ctx = ui.ctx();
    if ctx.is_being_dragged(split_divider_id(ui))
        && ctx.input(|i| i.pointer.primary_down())
        && let Some(pointer) = ctx.pointer_interact_pos()
    {
        // `max` and `min` rather than `clamp`, so a NaN bound from a corrupt saved zoom can't panic
        let (min_x, max_x) = divider_range_keeping_uv_size(&app.ui, full, old_x, pixels_per_point);
        let target_x = pointer.x.max(min_x).min(max_x);
        app.ui.split_ratio = clamp_split_ratio((target_x - full.left()) / full.width(), full.width());
        let new_x = divider_x(app.ui.split_ratio);
        keep_uv_view_on_screen(&mut app.ui, full, old_x, new_x);
    }

    divider_x(app.ui.split_ratio)
}

/// Lowest 2D zoom, for the wheel and the divider (inside the renderer's own limit of 0.05)
const UV_VIEW_MIN_ZOOM: f32 = 0.1;

/// Highest 2D zoom, for the wheel and the divider (inside the renderer's own limit of 64)
const UV_VIEW_MAX_ZOOM: f32 = 40.0;

/// UV zoom change per point of mouse wheel, as a fraction of the zoom
const UV_WHEEL_ZOOM_PER_POINT: f32 = 0.002;

/// Dash length of the straight-line preview, in points
const RUBBER_BAND_DASH: f32 = 6.0;

/// Gap between the straight-line preview's dashes, in points
const RUBBER_BAND_GAP: f32 = 4.0;

/// Opacity of the UV tile outlines in the 2D view
const UV_TILE_OUTLINE_ALPHA: u8 = 40;

/// Smallest skew brush circle in the 2D view, in points, so a tiny brush stays visible
const BAKE_PAINT_CURSOR_MIN_RADIUS: f32 = 2.0;

/// The 2D view's width when the split divider is at `divider_x`
fn uv_view_width(full: Rect, divider_x: f32) -> f32 {
    (full.right() - divider_x - SPLIT_LINE_HALF_WIDTH).max(1.0)
}

/// The share of the 2D view's height that a zoom of 1 fills at `width`: the view fits the texture to its narrower side
fn uv_view_fit(full: Rect, width: f32) -> f32 {
    (width / full.height().max(1.0)).min(1.0)
}

/// The divider positions where the 2D zoom that keeps the texture's screen size stays inside the 2D zoom range
fn divider_range_keeping_uv_size(ui_state: &crate::state::UiState, full: Rect, old_x: f32, pixels_per_point: f32) -> (f32, f32) {
    // The zoom times the fit is the texture's screen size, which a divider drag keeps
    let screen_zoom = ui_state.uv_zoom * uv_view_fit(full, uv_view_width(full, old_x));

    // Whole pixels, rounded inward, so snapping the divider to a pixel can't step past the range
    let narrowest_view = (full.height() * screen_zoom / UV_VIEW_MAX_ZOOM * pixels_per_point).ceil() / pixels_per_point;
    let widest_view = (full.height() * screen_zoom / UV_VIEW_MIN_ZOOM * pixels_per_point).floor() / pixels_per_point;

    let edge = full.right() - SPLIT_LINE_HALF_WIDTH;
    (edge - widest_view, edge - narrowest_view)
}

/// Keeps the 2D view's texture at the same size and offset from the view's centre when the divider moves its left edge from `old_x` to `new_x`
fn keep_uv_view_on_screen(ui_state: &mut crate::state::UiState, full: Rect, old_x: f32, new_x: f32) {
    let (old_width, new_width) = (uv_view_width(full, old_x), uv_view_width(full, new_x));
    ui_state.uv_zoom *= uv_view_fit(full, old_width) / uv_view_fit(full, new_width);

    // The horizontal pan is in half view widths
    ui_state.uv_pan[0] *= old_width / new_width;
}

/// The split divider's grab zone with a resize cursor, claimed after the views so it wins the pointer over them
fn split_divider_grab(ui: &Ui, full: Rect, divider_x: f32) {
    let zone = Rect::from_x_y_ranges(divider_x - SPLIT_DIVIDER_GRAB_WIDTH * 0.5..=divider_x + SPLIT_DIVIDER_GRAB_WIDTH * 0.5, full.y_range());
    ui.interact(zone, split_divider_id(ui), Sense::drag()).on_hover_and_drag_cursor(egui::CursorIcon::ResizeHorizontal);
}

fn shade(app: &TextureCraftApp) -> Shade {
    let (view_mode, channel) = match app.session.view {
        _ if app.ui.mode == crate::state::Mode::Painting && app.session.quick_mask.editing && app.picking_id_color.is_none() => (2, 0),
        ViewMode::Material => (0, 0),
        ViewMode::Channel(selected_channel) => (1, selected_channel.index() as u32),
        ViewMode::MeshMap(_) | ViewMode::Mask(_) => (2, 0),
        ViewMode::UvChecker => (3, 0),
    };
    let set_index = app.session.ts();
    let height_strength = app.session.doc.project.texture_sets.get(set_index).map(|texture_set| texture_set.atlas_height_strength()).unwrap_or(1.0);
    // Polygon fill shows the polygons it fills without changing the saved wireframe toggles
    let polygon_fill_tool = app.ui.mode == crate::state::Mode::Painting && app.session.tool.tool == Tool::PolygonFill;
    Shade {
        height_strength,
        view_mode,
        channel,
        wireframe: app.ui.uv_wireframe || polygon_fill_tool,
        wireframe_3d: app.ui.wireframe_3d || polygon_fill_tool,
        ..Shade::from_project(&app.session.doc.project)
    }
}

/// The texture set a shelf material or image dragged over the 3D view `response` would drop on, its edges drawn in the accent color
fn drop_highlight(app: &TextureCraftApp, ui: &Ui, response: &egui::Response) -> Option<SetHighlight> {
    let set = crate::shelf::drop_set(app, ui, response)?;
    let accent = egui::Rgba::from(Tokens::get(ui.ctx()).viewport_accent);

    Some(SetHighlight { set, color: [accent.r(), accent.g(), accent.b()] })
}

/// Show what the GPU side reported (each message once while its toast is up).
fn gpu_report_toasts(app: &mut TextureCraftApp, reports: crate::gpu::GpuReports) {
    let names: Vec<String> = reports.memory.iter().filter_map(|ts| app.session.doc.project.texture_sets.get(*ts).map(|t| t.name.clone())).collect();
    if !names.is_empty() {
        // One sticky notice for every set the GPU could not hold (they are not retried, so the
        // notice stays until dismissed); after a dismissal, a new one names only the new sets.
        let (sets, shown) = &mut app.gpu_memory;
        let current = shown.take().filter(|text| app.toasts.iter().any(|t| &t.text == text));
        match &current {
            Some(text) => app.toasts.retain(|t| &t.text != text),
            None => sets.clear(),
        }
        for n in names {
            if !sets.contains(&n) {
                sets.push(n);
            }
        }
        let text = match sets.as_slice() {
            [one] => trf(
                "Not enough GPU memory to show texture set \"{name}\" fully: lower the viewport Texture resolution in Settings. Painting and export are not affected.",
                &[("name", one)],
            ),
            many => {
                let short = |n: &String| match n.char_indices().nth(MAX_NAME_CHARS) {
                    Some((cut, _)) => format!("\"{}...\"", &n[..cut]),
                    None => format!("\"{n}\""),
                };
                let mut list = many.iter().take(MAX_NAMED_SETS).map(short).collect::<Vec<_>>().join(", ");
                if many.len() > MAX_NAMED_SETS {
                    list.push_str(", ...");
                }
                trf(
                    "Not enough GPU memory to show {count} texture sets fully ({names}): lower the viewport Texture resolution in Settings. Painting and export are not affected.",
                    &[("count", &many.len()), ("names", &list)],
                )
            }
        };
        app.gpu_memory.1 = Some(text.clone());
        app.toast_sticky(text);
    }
    for e in &reports.errors {
        let text = trf("GPU error: {error}", &[("error", e)]);
        if !app.toasts.iter().any(|t| t.text == text) {
            app.toast(text, true);
        }
    }
}

/// Texture sets a GPU memory notice names before "...".
const MAX_NAMED_SETS: usize = 5;
/// Characters of a set name in that notice (long names are cut, with "...").
const MAX_NAME_CHARS: usize = 32;

/// This frame's scene sync (compositing and GPU uploads), run by the first view drawn, after
/// that view handled its input: a stroke shows in the frame it is painted, not the next.
pub struct PendingSync(Option<Option<crate::gpu::AuxView>>);

/// A view's render size in pixels, rounded as egui rounds the rect it draws the frame into, so the frame is never stretched
fn frame_size_in_pixels(ui: &Ui, rect: Rect) -> [u32; 2] {
    let pixels_per_point = ui.ctx().pixels_per_point();
    let pixels = |from: f32, to: f32| ((to * pixels_per_point).round() - (from * pixels_per_point).round()).max(1.0) as u32;
    [pixels(rect.left(), rect.right()), pixels(rect.top(), rect.bottom())]
}

fn add_callback(app: &mut TextureCraftApp, ui: &Ui, rect: Rect, mut frame: Frame, sync: &mut PendingSync) {
    let tokens = Tokens::get(ui.ctx());
    let bake = bake_visualization(app);
    frame.shade.bake = bake.filter(|_| frame.kind == ViewKind::Scene);
    let scene = match (sync.0.take(), app.gpu.as_mut()) {
        (Some(aux), Some(gpu_link)) => {
            let sync_start_ms = crate::now_ms();
            let scene = gpu_link.sync(&mut app.session, aux, bake, u32::from(app.ui.prefs.viewport_texture.min(2)), f64::from(app.profiler.logic_ms));
            let reports = gpu_link.take_reports();
            app.profiler.sync_ms += (crate::now_ms() - sync_start_ms) as f32;
            gpu_report_toasts(app, reports);

            // Display tiles left for later frames (the frame budget ran out), or the bake visualization still being worked out
            if app.session.caches.iter().any(|cache| !cache.upload.is_empty()) || app.gpu.as_ref().is_some_and(crate::gpu::GpuLink::bake_busy) {
                ui.ctx().request_repaint();
            }
            Some(scene)
        }
        _ => None,
    };

    let Some(gpu) = &app.gpu else {
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            tr("No GPU renderer available"),
            egui::FontId::proportional(NO_GPU_TEXT_SIZE),
            tokens.viewport_text,
        );
        return;
    };

    let callback = ViewCallback { frame, sync: Mutex::new(scene), rect, lost: gpu.lost.clone() };
    ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(rect, callback));
}

/// Text size of the notice a view shows without a GPU renderer
const NO_GPU_TEXT_SIZE: f32 = 14.0;

/// Width of the camera and view pickers centred at the top of a view
const VIEW_HEADER_WIDTH: f32 = 300.0;

/// Height of the view pickers' row
const VIEW_HEADER_HEIGHT: f32 = 22.0;

/// Gap between the view's top edge and its pickers
const VIEW_HEADER_TOP_GAP: f32 = 8.0;

/// Width of each of the two pickers
const VIEW_PICKER_WIDTH: f32 = 130.0;

/// The camera and view pickers at the top of a view, left out when the view is too narrow to hold them
fn view_header(app: &mut TextureCraftApp, ui: &mut Ui, rect: Rect, id: &str) {
    if rect.width() < VIEW_HEADER_WIDTH {
        return;
    }

    let header_min = rect.min + vec2((rect.width() - VIEW_HEADER_WIDTH) * 0.5, VIEW_HEADER_TOP_GAP);
    let header = Rect::from_min_size(header_min, vec2(VIEW_HEADER_WIDTH, VIEW_HEADER_HEIGHT));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(header).layout(egui::Layout::left_to_right(egui::Align::Center)));

    let current = texturecraft_engine::camera_cmds::current(&app.session);
    let names: Vec<String> = app.session.doc.project.cameras.iter().map(|camera| camera.name.clone()).collect();
    let shown = current.and_then(|index| names.get(index)).map_or(tr(DEFAULT_CAMERA), String::as_str);
    egui::ComboBox::from_id_salt(format!("{id}_cam")).width(VIEW_PICKER_WIDTH).selected_text(shown).show_ui(&mut child, |ui| {
        if ui.selectable_label(current.is_none(), tr(DEFAULT_CAMERA)).clicked() {
            app.run("camera.select", json!({}));
        }
        for (index, name) in names.iter().enumerate() {
            if ui.selectable_label(current == Some(index), name).clicked() {
                app.run("camera.select", json!({"index": index}));
            }
        }
        ui.separator();
        if ui.selectable_label(false, tr("Save view as camera")).clicked() {
            app.run("camera.add", json!({}));
        }
    });

    let label = match app.session.view {
        ViewMode::MeshMap(map) => crate::i18n::baked_map_label(map).to_owned(),
        _ => app.session.view.label(),
    };
    let channels = app.session.doc.project.texture_set(app.session.ts()).map(|texture_set| texture_set.channel_list()).unwrap_or_default();
    egui::ComboBox::from_id_salt(format!("{id}_mode")).width(VIEW_PICKER_WIDTH).selected_text(tr(&label)).show_ui(&mut child, |ui| {
        if ui.selectable_label(app.session.view == ViewMode::Material, tr("Material")).clicked() {
            app.run("viewport.view", json!({"mode": "material"}));
        }
        ui.separator();
        for channel in channels {
            if ui.selectable_label(app.session.view == ViewMode::Channel(channel), tr(channel.label())).clicked() {
                app.run("viewport.view", json!({"mode": channel.id()}));
            }
        }
        ui.separator();
        for map in BakedMap::ALL {
            let map_label = trf("Mesh map: {map}", &[("map", &tr(crate::i18n::baked_map_label(map)))]);
            if ui.selectable_label(app.session.view == ViewMode::MeshMap(map), map_label).clicked() {
                app.run("viewport.view", json!({"mode": format!("map:{}", map.id())}));
            }
        }
        ui.separator();
        if ui.selectable_label(app.session.view == ViewMode::UvChecker, tr("UV checker")).clicked() {
            app.run("viewport.view", json!({"mode": "uvChecker"}));
        }
    });
}

fn axis_gizmo(ui: &Ui, rect: Rect, app: &TextureCraftApp, uv: bool) {
    let tk = Tokens::get(ui.ctx());
    let o = pos2(rect.left() + 28.0, rect.bottom() - 28.0);
    let p = ui.painter();
    if uv {
        p.line_segment([o, o + vec2(16.0, 0.0)], Stroke::new(1.5, Color32::from_rgb(0xd8, 0x4a, 0x3a)));
        p.line_segment([o, o - vec2(0.0, 16.0)], Stroke::new(1.5, Color32::from_rgb(0x5c, 0xc0, 0x4a)));
        p.text(o + vec2(20.0, 0.0), egui::Align2::LEFT_CENTER, "U", egui::FontId::proportional(9.0), tk.viewport_text);
        p.text(o - vec2(0.0, 22.0), egui::Align2::CENTER_CENTER, "V", egui::FontId::proportional(9.0), tk.viewport_text);
        return;
    }
    let view = app.session.camera.view();
    for (axis, color, name) in [
        (Vec3::X, Color32::from_rgb(0xd8, 0x4a, 0x3a), "X"),
        (Vec3::Y, Color32::from_rgb(0x5c, 0xc0, 0x4a), "Y"),
        (Vec3::Z, Color32::from_rgb(0x4a, 0x80, 0xe0), "Z"),
    ] {
        let d = view.transform_vector3(axis);
        let end = o + vec2(d.x, -d.y) * 16.0;
        p.line_segment([o, end], Stroke::new(1.5, color));
        p.text(end + vec2(d.x, -d.y) * 6.0, egui::Align2::CENTER_CENTER, name, egui::FontId::proportional(9.0), tk.viewport_text);
    }
}

/// Camera turn per point of a drag (radians)
const ORBIT_RADIANS_PER_POINT: f32 = 0.008;

/// Camera zoom per point of an Alt+right drag
const DRAG_ZOOM_PER_POINT: f32 = 0.01;

/// Camera zoom per point of mouse wheel
const WHEEL_ZOOM_PER_POINT: f32 = 0.0025;

/// Environment light turn per point of a Shift+right drag (degrees)
const LIGHT_DEGREES_PER_POINT: f32 = 0.5;

/// Brush size change per point of Shift+wheel, as a fraction of the size
const WHEEL_BRUSH_SIZE_PER_POINT: f32 = 0.002;

/// Stencil turn per point of an S+left drag (degrees)
const STENCIL_DEGREES_PER_POINT: f32 = 0.4;

/// Stencil scale change per point of an S+right drag, as a fraction of the scale
const STENCIL_SCALE_PER_POINT: f32 = 0.005;

/// Smallest stencil scale an S+right drag reaches
const MIN_STENCIL_SCALE: f32 = 0.05;

/// Largest stencil scale an S+right drag reaches
const MAX_STENCIL_SCALE: f32 = 10.0;

/// Smallest viewport the camera projects into, so a collapsed view keeps a usable aspect (points)
const MIN_CAMERA_VIEWPORT: f32 = 16.0;

fn view_3d(app: &mut TextureCraftApp, ui: &mut Ui, rect: Rect, sync: &mut PendingSync) {
    let response = ui.interact(rect, ui.scope_id().with("view3d"), Sense::click_and_drag());
    app.viewport_rect = rect;
    let size = rect.size();
    app.session.viewport = Vec2::new(size.x.max(MIN_CAMERA_VIEWPORT), size.y.max(MIN_CAMERA_VIEWPORT));
    let (alt, shift) = ui.input(|input| (input.modifiers.alt, input.modifiers.shift));
    let local = |pointer: egui::Pos2| Vec2::new(pointer.x - rect.left(), pointer.y - rect.top());
    let adjusting = brush_drag(app, ui, &response);
    let navigating = !adjusting && (alt || ui.input(|input| input.pointer.middle_down() || input.pointer.secondary_down()));

    // Navigation; S + drag moves the projection stencil instead, below
    let delta = ui.input(|input| input.pointer.delta());
    let s_held = app.ui.mode == crate::state::Mode::Painting && ui.input(|input| input.key_down(egui::Key::S)) && !ui.ctx().egui_wants_keyboard_input();
    let stencil_drag = s_held && app.session.tool.stencil.is_some();
    if stencil_drag || adjusting {
        // The stencil or the brush changes instead of the camera
    } else if response.dragged_by(PointerButton::Middle) {
        app.session.camera.pan(delta.x / size.y, delta.y / size.y);
    } else if alt && response.dragged_by(PointerButton::Primary) {
        app.session.camera.orbit(delta.x * ORBIT_RADIANS_PER_POINT, delta.y * ORBIT_RADIANS_PER_POINT);
    } else if response.dragged_by(PointerButton::Secondary) {
        if alt {
            app.session.camera.zoom(-delta.y * DRAG_ZOOM_PER_POINT);
        } else if shift {
            let display = &mut app.session.doc.project.display;
            display.env_rotation = (display.env_rotation + delta.x * LIGHT_DEGREES_PER_POINT).rem_euclid(360.0);
        } else {
            app.session.camera.orbit(delta.x * ORBIT_RADIANS_PER_POINT, delta.y * ORBIT_RADIANS_PER_POINT);
        }
    }

    if response.hovered() {
        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
        if scroll != 0.0 {
            if shift && app.ui.mode == crate::state::Mode::Painting {
                let brush = &mut app.session.tool.brush;
                brush.size = (brush.size * (1.0 + scroll * WHEEL_BRUSH_SIZE_PER_POINT)).clamp(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE);
            } else {
                app.session.camera.zoom(scroll * WHEEL_ZOOM_PER_POINT);
            }
        }
    }

    // Touch: two fingers orbit and pinch zooms, three pan; a second finger cancels the first one's stroke
    let touch = ui.input(|input| input.multi_touch()).filter(|_| response.contains_pointer() || response.dragged());
    if let Some(gesture) = touch {
        if app.session.stroke_active() {
            app.session.cancel_stroke();
        }

        let moved = gesture.translation_delta;
        if gesture.num_touches >= 3 {
            app.session.camera.pan(moved.x / size.y, moved.y / size.y);
        } else {
            app.session.camera.orbit(moved.x * ORBIT_RADIANS_PER_POINT, moved.y * ORBIT_RADIANS_PER_POINT);
        }
        if gesture.zoom_delta.is_finite() && gesture.zoom_delta > 0.0 {
            app.session.camera.zoom(gesture.zoom_delta.ln());
        }
    }

    // Stencil: S + left drag turns it, S + right drag scales it, S + middle drag moves it
    if s_held && let Some(stencil) = app.session.tool.stencil.as_mut() {
        if response.dragged_by(PointerButton::Primary) {
            stencil.rotation = (stencil.rotation + delta.x * STENCIL_DEGREES_PER_POINT).rem_euclid(360.0);
        } else if response.dragged_by(PointerButton::Secondary) {
            stencil.scale = (stencil.scale * (1.0 + delta.x * STENCIL_SCALE_PER_POINT)).clamp(MIN_STENCIL_SCALE, MAX_STENCIL_SCALE);
        } else if response.dragged_by(PointerButton::Middle) {
            stencil.center[0] += delta.x / size.x * 2.0;
            stencil.center[1] -= delta.y / size.y * 2.0;
        }
    }

    // Clone source: V + click
    let v_held = clone_source_key(app, ui);
    if v_held
        && !alt
        && (response.clicked_by(PointerButton::Primary) || response.drag_started_by(PointerButton::Primary))
        && let Some(pointer) = response.interact_pointer_pos()
    {
        let (at, viewport) = (local(pointer), app.session.viewport);
        app.run("tool.cloneSource", json!({"x": at.x, "y": at.y, "viewport": [viewport.x, viewport.y]}));
    }

    // Depth of field focus pick: a click focuses there instead of painting
    let focus_pick = focus_picking(ui.ctx());
    if focus_pick {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        if !alt
            && response.clicked_by(PointerButton::Primary)
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let (at, viewport) = (local(pointer), app.session.viewport);
            app.run("viewport.focusAt", json!({"x": at.x, "y": at.y, "viewport": [viewport.x, viewport.y]}));
            set_focus_picking(ui.ctx(), false);
        }
    }

    // A dismissing click cannot paint or place a decal through the rail popup
    let rail_popup = crate::rail_popup::blocks_paint_input(app, ui.ctx());

    // Decal placement: a click moves the decal being placed there, unless it dismisses a waiting image drop's menu
    let placing = app.placing_decal.filter(|_| app.ui.mode == crate::state::Mode::Painting);
    if let Some(layer) = placing {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        if !alt
            && !rail_popup
            && !DecalDrop::blocks_input(ui.ctx())
            && response.clicked_by(PointerButton::Primary)
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let (at, viewport) = (local(pointer), app.session.viewport);
            app.run("layer.decal.set", json!({"layer": layer.0, "x": at.x, "y": at.y, "viewport": [viewport.x, viewport.y]}));
            app.placing_decal = None;
        }
    }

    crate::properties_popup::open_on_right_click(app, ui, &response, rect, s_held || v_held);
    let popup = app.properties_popup.is_some() || rail_popup;

    // A pending ID color pick takes the next click instead of the brush
    let picking_id_color = app.picking_id_color.is_some();
    if picking_id_color {
        crate::id_color_pick::IdColorPick::draw_cursor(ui, rect);
        if !alt
            && response.clicked_by(PointerButton::Primary)
            && let Some(position) = response.interact_pointer_pos()
        {
            let (pixel, viewport) = (local(position), app.session.viewport);
            crate::id_color_pick::IdColorPick::pick_at(app, json!({"x": pixel.x, "y": pixel.y, "viewport": [viewport.x, viewport.y]}));
        }
    }

    // The selected decal's frame and handles: drag to move, scale or turn it
    let decal_drag = !alt
        && !s_held
        && !v_held
        && !focus_pick
        && !popup
        && placing.is_none()
        && !picking_id_color
        && touch.is_none()
        && !DecalDrop::blocks_input(ui.ctx())
        && decal_input(app, ui, &response, rect);
    let symmetry_drag = !alt
        && !adjusting
        && !s_held
        && !v_held
        && !focus_pick
        && !popup
        && placing.is_none()
        && !picking_id_color
        && touch.is_none()
        && !DecalDrop::blocks_input(ui.ctx())
        && !decal_drag
        && crate::symmetry::manipulator_input(app, ui, &response, rect);

    // A handle owns a new click, but a stroke already under way must still finish over it
    let symmetry_handle = !app.session.stroke_active() && crate::symmetry::manipulator_hovered(app, ui, rect);

    // Painting
    let painting = app.ui.mode == crate::state::Mode::Painting
        && !popup
        && !alt
        && !adjusting
        && !s_held
        && !v_held
        && !focus_pick
        && placing.is_none()
        && !picking_id_color
        && touch.is_none()
        && !decal_drag
        && !symmetry_drag
        && !symmetry_handle
        && !DecalDrop::blocks_input(ui.ctx());
    let path_tool = app.session.tool.tool == Tool::Path;
    let viewport = app.session.viewport;
    let show = |app: &TextureCraftApp, point: Vec3| project(app, rect, point);
    let mut fill_box = None;
    if painting && path_tool {
        path_input(app, ui, &response, (rect, false), |app, pointer| app.session.pick(local(pointer), viewport).map(|hit| hit.pos), show);
    } else if painting && app.session.tool.tool == Tool::PolygonFill {
        fill_box = polygon_fill_input(
            app,
            ui,
            &response,
            |session, point| session.polygon_fill(local(point), viewport),
            |session, corner, opposite| session.drag_polygon_fill_box(local(corner), local(opposite), viewport),
        );
    } else if painting {
        let pressure = pen_pressure(ui);
        if response.drag_started_by(PointerButton::Primary) || response.clicked_by(PointerButton::Primary) {
            // Shift+click paints a straight line from where the last stroke ended
            let line_from = if shift { app.last_paint_point } else { None };
            if let Some(pointer) = response.interact_pointer_pos() {
                match app.session.begin_stroke(line_from.unwrap_or(local(pointer)), app.session.viewport, pressure) {
                    Ok(_) => {
                        if line_from.is_some() {
                            app.session.continue_stroke(local(pointer), pressure);
                        }
                        app.last_paint_point = Some(local(pointer));
                    }
                    Err(error) => app.toast(tr(&error.to_string()).to_string(), true),
                }
            }
            if response.clicked_by(PointerButton::Primary)
                && let Err(error) = app.session.end_stroke()
            {
                app.toast(error.to_string(), true);
            }
        } else if response.dragged_by(PointerButton::Primary)
            && app.session.stroke_active()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            app.session.continue_stroke(local(pointer), pressure);
            app.last_paint_point = Some(local(pointer));
        }
        if response.drag_stopped_by(PointerButton::Primary)
            && let Err(error) = app.session.end_stroke()
        {
            app.toast(error.to_string(), true);
        }
    }

    let frame = Frame {
        view_id: crate::viewport::VIEW_3D,
        size: frame_size_in_pixels(ui, rect),
        kind: ViewKind::Scene,
        camera: app.session.camera,
        shade: Shade { highlight: drop_highlight(app, ui, &response), ..shade(app) },
        visible: app.session.doc.project.texture_sets.iter().map(|texture_set| texture_set.visible).collect(),
        overlay: crate::symmetry::scene_overlay(app, ui, rect),
    };
    add_callback(app, ui, rect, frame, sync);
    polygon_fill_box_overlay(ui, rect, fill_box);

    // Shelf drops: a material goes above the selection of the set under the pointer, an image opens the decal menu there, an alpha goes on the brush
    crate::shelf::drop_target(app, ui, &response, crate::shelf::DropZone::View3d);

    // Brush cursor; while a Ctrl+drag changes the brush it stays where the drag began
    let mut symmetry_cursor = None;
    if app.ui.mode == crate::state::Mode::Painting
        && !navigating
        && response.hovered()
        && brush_tool(app.session.tool.tool)
        && (app.session.tool.tool != Tool::Path || adjusting)
        && !crate::symmetry::manipulator_hovered(app, ui, rect)
        && app.picking_id_color.is_none()
        && let Some(hover_pos) = if adjusting { ui.input(|input| input.pointer.press_origin()) } else { response.hover_pos() }
        && let Some(hit) = app.session.pick(local(hover_pos), app.session.viewport)
    {
        ui.ctx().set_cursor_icon(egui::CursorIcon::None);
        let (bounds_min, bounds_max) = app.session.mesh.bounds();
        let radius = app.session.tool.brush.size * (bounds_max - bounds_min).length();
        brush_cursor(app, ui, rect, hit.pos, hit.nrm, radius);
        symmetry_cursor = Some((hit.pos, hit.nrm));
        if app.session.tool.tool == Tool::Clone
            && let Some(source) = app.session.clone_source_point(Some(hit.uv))
        {
            clone_marker(app, ui, rect, source);
        }

        // Shift: preview the straight line Shift+click paints from the last stroke
        if shift
            && !path_tool
            && let Some(from) = app.last_paint_point
        {
            rubber_band(ui, rect, &[pos2(rect.left() + from.x, rect.top() + from.y), hover_pos]);
        }
    }

    if app.ui.mode == crate::state::Mode::Painting && path_tool {
        path_overlay(app, ui, rect, &response, false, show);
    }
    crate::symmetry::overlay(app, ui, rect, symmetry_cursor);
    stencil_overlay(app, ui, rect);
    decal_overlay(app, ui, rect);
    brush_readout(app, ui, &response, adjusting);
    if app.session.quick_mask.editing && app.picking_id_color.is_none() {
        ui.painter_at(rect).text(
            rect.left_top() + QUICK_MASK_NOTICE_OFFSET,
            egui::Align2::LEFT_TOP,
            tr("Editing quick mask: paint where painting is allowed (U to finish)"),
            egui::FontId::proportional(QUICK_MASK_NOTICE_TEXT_SIZE),
            QUICK_MASK_NOTICE_COLOR,
        );
    }
    if app.ui.mode == crate::state::Mode::Baking {
        crate::bake_mode::bake_visualization_panel(app, ui, rect);
    }
    view_header(app, ui, rect, "v3d");
    axis_gizmo(ui, rect, app, false);
}

/// Where the quick-mask editing notice sits from the 3D view's top-left corner, its text size and its warning red
const QUICK_MASK_NOTICE_OFFSET: egui::Vec2 = vec2(12.0, 40.0);
const QUICK_MASK_NOTICE_TEXT_SIZE: f32 = 12.0;
const QUICK_MASK_NOTICE_COLOR: Color32 = Color32::from_rgb(0xe8, 0x6a, 0x6a);

/// What bake mode's 3D view draws besides the low poly (nothing in the other modes)
fn bake_visualization(app: &TextureCraftApp) -> Option<texturecraft_render::BakeVisualization> {
    // Setup meshes can cover the low-poly map preview; keep the switches for when setup resumes
    if app.ui.mode != crate::state::Mode::Baking || app.session.bake_preview_active() {
        return None;
    }

    app.ui.bake_visualization.visualization(app.session.ts())
}

/// What a drag on the selected decal's frame does.
#[derive(Clone, Copy, Debug, PartialEq)]
enum DecalGrab {
    /// Inside the frame: slide it over the surface.
    Move,
    /// A corner: scale it about its center.
    Scale,
    /// The round handle above its top edge: turn it.
    Rotate,
}

impl DecalGrab {
    fn cursor(self, dragging: bool) -> egui::CursorIcon {
        match self {
            DecalGrab::Move => egui::CursorIcon::Move,
            DecalGrab::Scale => egui::CursorIcon::ResizeNwSe,
            DecalGrab::Rotate if dragging => egui::CursorIcon::Grabbing,
            DecalGrab::Rotate => egui::CursorIcon::Grab,
        }
    }
}

/// A decal handle being dragged (kept in egui's memory between frames)
#[derive(Clone, Copy, Debug)]
struct DecalDrag {
    layer: texturecraft_engine::core::LayerId,
    /// The mask fill effect holding the decal, when it was placed as a mask
    effect: Option<texturecraft_engine::core::LayerId>,
    grab: DecalGrab,
    /// Where the drag started, the decal's center on screen and the decal then
    from: Pos2,
    center: Pos2,
    start: texturecraft_engine::core::Decal,
    /// The drag made its undo step: later updates merge into it
    stepped: bool,
}

fn decal_drag_id() -> egui::Id {
    egui::Id::unique("tc_decal_drag")
}

/// Screen distance from the top edge to the rotation handle (points).
const DECAL_KNOB: f32 = 24.0;

/// The selected decal (a decal layer's content, or a mask decal effect) as the 3D view shows it
struct DecalFrame {
    layer: texturecraft_engine::core::LayerId,
    /// The selected mask fill effect holding the decal, when it was placed as a mask
    effect: Option<texturecraft_engine::core::LayerId>,
    decal: texturecraft_engine::core::Decal,
    /// Top left, top right, bottom right, bottom left
    corners: [Pos2; 4],
    center: Pos2,
    /// The middle of the top edge and the rotation handle above it
    top: Pos2,
    knob: Pos2,
}

impl DecalFrame {
    fn of(app: &TextureCraftApp, rect: Rect) -> Option<DecalFrame> {
        use texturecraft_engine::StackTarget;
        use texturecraft_engine::compose::pattern::{decal_axes, decal_corners};
        use texturecraft_engine::core::{EffectKind, Fill, LayerKind, Projection};

        if app.ui.mode != crate::state::Mode::Painting {
            return None;
        }

        let layers = &app.session.doc.project;
        let selected = app.session.selected_layer()?;
        let (layer, effect, decal) = match app.session.target {
            StackTarget::Content => {
                let layer = layers.content_layer(selected)?;
                let LayerKind::Fill(Fill { projection: Projection::Decal(decal), .. }) = &layer.kind else { return None };

                (layer.id, None, decal)
            }
            StackTarget::Effect(effect) => {
                // Only the selected layer's effective mask may supply a decal frame, including an instance's inherited mask
                let owner = layers.mask_owner(selected);
                let mask = layers.layer(owner).ok()?.mask.as_ref()?;
                let mask_effect = mask.effects.iter().find(|candidate| candidate.id == effect)?;
                let EffectKind::Fill { projection: Projection::Decal(decal), .. } = &mask_effect.kind else { return None };

                (owner, Some(effect), decal)
            }
            StackTarget::Mask => return None,
        };
        let [top_left, top_right, bottom_right, bottom_left] = decal_corners(decal)?.map(|corner| project(app, rect, corner));
        let corners = [top_left?, top_right?, bottom_right?, bottom_left?];
        let center = project(app, rect, decal_axes(decal)?.0)?;
        let top = corners[0].lerp(corners[1], 0.5);
        let knob = top + (top - center).normalized() * DECAL_KNOB;
        Some(DecalFrame { layer, effect, decal: *decal, corners, center, top, knob })
    }

    /// The handle under screen point `p`.
    fn grab(&self, p: Pos2) -> Option<DecalGrab> {
        if self.knob.distance(p) <= HANDLE_GRAB {
            return Some(DecalGrab::Rotate);
        }
        if self.corners.iter().any(|c| c.distance(p) <= HANDLE_GRAB) {
            return Some(DecalGrab::Scale);
        }
        // Inside the (convex) frame: on the same side of every edge.
        let side = |i: usize| {
            let (a, b) = (self.corners[i], self.corners[(i + 1) % 4]);
            (b - a).x * (p - a).y - (b - a).y * (p - a).x
        };
        let s = [side(0), side(1), side(2), side(3)];
        (s.iter().all(|v| *v >= 0.0) || s.iter().all(|v| *v <= 0.0)).then_some(DecalGrab::Move)
    }
}

/// Drags on the selected decal's frame and handles (one undo step per drag, the decal updates live); returns whether a decal drag owns the pointer
fn decal_input(app: &mut TextureCraftApp, ui: &Ui, resp: &egui::Response, rect: Rect) -> bool {
    let ctx = ui.ctx();
    let frame = DecalFrame::of(app, rect);
    let on_frame = |drag: &DecalDrag| frame.as_ref().is_some_and(|frame| frame.layer == drag.layer && frame.effect == drag.effect);
    let mut drag = ctx.data(|data| data.get_temp::<DecalDrag>(decal_drag_id())).filter(on_frame);
    if resp.drag_started_by(PointerButton::Primary)
        && let (Some(frame), Some(from)) = (&frame, ctx.input(|input| input.pointer.press_origin()))
        && let Some(grab) = frame.grab(from)
    {
        drag = Some(DecalDrag { layer: frame.layer, effect: frame.effect, grab, from, center: frame.center, start: frame.decal, stepped: false });
    }

    let Some(mut drag) = drag.filter(|_| resp.dragged_by(PointerButton::Primary) || resp.drag_started_by(PointerButton::Primary)) else {
        ctx.data_mut(|data| data.remove::<DecalDrag>(decal_drag_id()));
        if let (Some(frame), Some(hover)) = (&frame, resp.hover_pos())
            && let Some(grab) = frame.grab(hover)
        {
            ctx.set_cursor_icon(grab.cursor(false));
        }
        return false;
    };

    ctx.set_cursor_icon(drag.grab.cursor(true));
    if let Some(pointer) = resp.interact_pointer_pos().filter(|_| ctx.input(|input| input.pointer.delta() != egui::Vec2::ZERO)) {
        let mut params = match drag.grab {
            DecalGrab::Move => {
                let target = drag.center + (pointer - drag.from) - rect.left_top();
                let viewport = app.session.viewport;
                json!({"x": target.x, "y": target.y, "viewport": [viewport.x, viewport.y]})
            }
            DecalGrab::Scale => {
                let scale = (pointer - drag.center).length() / (drag.from - drag.center).length().max(1.0);
                json!({"size": [drag.start.size[0] * scale, drag.start.size[1] * scale]})
            }
            DecalGrab::Rotate => {
                // Screen y points down: counter-clockwise angles turn the decal's right toward its up
                let angle = |offset: egui::Vec2| (-offset.y).atan2(offset.x).to_degrees();
                json!({"rotation": drag.start.rotation + angle(pointer - drag.center) - angle(drag.from - drag.center)})
            }
        };
        params["layer"] = json!(drag.layer.0);
        if let Some(effect) = drag.effect {
            params["effect"] = json!(effect.0);
        }
        params["merge"] = json!(drag.stepped);
        // Off the mesh while moving: the decal waits where it is
        drag.stepped |= texturecraft_engine::execute(&mut app.session, "layer.decal.set", &params).is_ok();
    }
    ctx.data_mut(|data| data.insert_temp(decal_drag_id(), drag));
    true
}

/// The frame of the selected decal layer over the 3D view with its handles (and a hint while
/// placing it): outline, corner handles, the rotation handle and the projection direction.
fn decal_overlay(app: &mut TextureCraftApp, ui: &Ui, rect: Rect) {
    if app.ui.mode != crate::state::Mode::Painting {
        return;
    }

    let tk = Tokens::get(ui.ctx());
    use texturecraft_engine::core::{Fill, LayerKind, Projection};
    let p = &app.session.doc.project;
    let sel = app.session.selected_layer().and_then(|id| p.content_layer(id));
    let is_decal = matches!(sel.map(|l| &l.kind), Some(LayerKind::Fill(Fill { projection: Projection::Decal(_), .. })));
    // Placing follows the selection: another layer selected stops it.
    if app.placing_decal.is_some() && app.placing_decal != sel.map(|l| l.id).filter(|_| is_decal) {
        app.placing_decal = None;
    }
    let painter = ui.painter_at(rect);
    let hint_at = rect.left_top() + vec2(12.0, 40.0);
    if app.placing_decal.is_some() {
        let hint = tr("Click the mesh to place the decal");
        painter.text(hint_at, egui::Align2::LEFT_TOP, hint, egui::FontId::proportional(12.0), tk.viewport_accent);
    }
    let Some(f) = DecalFrame::of(app, rect) else { return };
    let active = ui.ctx().data(|d| d.get_temp::<DecalDrag>(decal_drag_id())).map(|d| d.grab);
    let hovered = ui.ctx().pointer_hover_pos().filter(|h| rect.contains(*h)).and_then(|h| f.grab(h));
    if app.placing_decal.is_none() && (hovered.is_some() || active.is_some()) {
        let hint = tr("Drag the frame to move the decal, a corner to scale it, the round handle to turn it");
        painter.text(hint_at, egui::Align2::LEFT_TOP, hint, egui::FontId::proportional(12.0), tk.viewport_text);
    }
    let shadow = Stroke::new(3.0, Color32::from_black_alpha(140));
    let mut outline = f.corners.to_vec();
    outline.push(f.corners[0]);
    let frame_color = if active == Some(DecalGrab::Move) || hovered == Some(DecalGrab::Move) { tk.viewport_accent } else { Color32::from_white_alpha(220) };
    for stroke in [shadow, Stroke::new(1.2, frame_color)] {
        painter.add(egui::Shape::dashed_line(&outline, stroke, 6.0, 4.0));
    }
    // The projection direction from the center.
    if let Some((o, _, _, n)) = texturecraft_engine::compose::pattern::decal_axes(&f.decal)
        && let Some(tip) = project(app, rect, o + n * f.decal.size[0].max(f.decal.size[1]) * 0.25)
    {
        painter.line_segment([f.center, tip], shadow);
        painter.line_segment([f.center, tip], Stroke::new(1.4, tk.viewport_accent));
    }
    painter.circle(f.center, 3.5, tk.viewport_accent, Stroke::new(1.0, Color32::from_black_alpha(180)));
    // Handles: white, the one under the pointer or being dragged in the accent color.
    let fill = |g: DecalGrab| if active == Some(g) || (active.is_none() && hovered == Some(g)) { tk.viewport_accent } else { Color32::WHITE };
    let edge = Stroke::new(1.0, Color32::from_black_alpha(200));
    painter.line_segment([f.top, f.knob], shadow);
    painter.line_segment([f.top, f.knob], Stroke::new(1.4, Color32::from_white_alpha(220)));
    painter.circle(f.knob, 5.0, fill(DecalGrab::Rotate), edge);
    for c in f.corners {
        painter.rect(Rect::from_center_size(c, vec2(8.0, 8.0)), 1.0, fill(DecalGrab::Scale), edge, egui::StrokeKind::Middle);
    }
}

/// Draw the projection stencil in Paint Mode
fn stencil_overlay(app: &mut TextureCraftApp, ui: &Ui, rect: Rect) {
    if app.ui.mode != crate::state::Mode::Painting || app.session.tool.tool != Tool::Projection {
        return;
    }
    let Some(stencil) = app.session.tool.stencil.clone() else { return };
    let Some(image) = app.session.images.get(&stencil.image).cloned() else { return };

    let texture = app.thumbs.stencil(ui.ctx(), &stencil.image, &image);
    let aspect = rect.width() / rect.height().max(1.0);
    let (center, half_size, rotation) = stencil.placement(image.width as f32 / image.height.max(1) as f32, aspect);
    let to_screen = |x: f32, y: f32| pos2(rect.left() + (x * 0.5 + 0.5) * rect.width(), rect.top() + (0.5 - y * 0.5) * rect.height());
    let (sine, cosine) = rotation.sin_cos();
    let corner = |x: f32, y: f32| {
        let (x, y) = (x * half_size[0], y * half_size[1]);
        to_screen(center[0] + x * cosine - y * sine, center[1] + x * sine + y * cosine)
    };
    let tint = Color32::from_white_alpha((stencil.opacity * 255.0) as u8);
    let mut mesh = egui::Mesh::with_texture(texture.id());
    for (x, y, u, v) in [(-1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0), (1.0, -1.0, 1.0, 1.0), (-1.0, -1.0, 0.0, 1.0)] {
        mesh.vertices.push(egui::epaint::Vertex { pos: corner(x, y), uv: pos2(u, v), color: tint });
    }
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    let painter = ui.painter_at(rect);
    painter.add(egui::Shape::mesh(mesh));
    let outline = vec![corner(-1.0, 1.0), corner(1.0, 1.0), corner(1.0, -1.0), corner(-1.0, -1.0), corner(-1.0, 1.0)];
    painter.add(egui::Shape::dashed_line(&outline, Stroke::new(1.0, Color32::from_white_alpha(120)), 5.0, 4.0));
}

/// A world point on screen in the 3D view `rect` (None behind the camera)
pub(crate) fn project(app: &TextureCraftApp, rect: Rect, point: Vec3) -> Option<Pos2> {
    let ndc = app.session.camera.project(point, rect.width() / rect.height().max(1.0))?;
    Some(pos2(rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(), rect.top() + (0.5 - ndc.y * 0.5) * rect.height()))
}

/// Control point and curve grab radius (points)
const HANDLE_GRAB: f32 = 8.0;
/// Opacity of the selected layer's other stored paths
const PATH_STORED_OPACITY: f32 = 0.55;
/// Opacity of a curve where the mesh hides it
const PATH_HIDDEN_OPACITY: f32 = 0.5;
/// A control point counts as hidden when the surface under it is this much closer (relative)
const PATH_HIDDEN_DEPTH: f32 = 0.01;
/// The Path tool's pointer glyph size (points)
const PATH_CURSOR_SIZE: f32 = 18.0;
/// A path's line and the dark outline under it (points)
const PATH_LINE_WIDTH: f32 = 1.4;
const PATH_OUTLINE_WIDTH: f32 = 3.0;
/// Where the view hint sits from the view's bottom left corner, and its text size (points)
const PATH_HINT_OFFSET: egui::Vec2 = vec2(56.0, -12.0);
const PATH_HINT_FONT_SIZE: f32 = 12.0;
/// The pen pointer's drop shadow, and the space before its insert plus (points)
const PATH_CURSOR_SHADOW_OFFSET: egui::Vec2 = vec2(1.0, 1.0);
const PATH_CURSOR_PLUS_GAP: f32 = 1.0;
/// The insert plus beside the pen pointer, relative to the pen's size
const PATH_CURSOR_PLUS_SCALE: f32 = 0.6;
/// Space kept right of the view hint, and its narrowest wrap width (points)
const PATH_HINT_MARGIN: f32 = 8.0;
const PATH_HINT_MIN_WIDTH: f32 = 80.0;
/// How much a control point marker grows under the pointer, and when selected (points)
const PATH_POINT_HOVER_GROWTH: f32 = 1.5;
const PATH_POINT_SELECTED_GROWTH: f32 = 0.5;
/// Control point marker radius (points)
const PATH_POINT_RADIUS: f32 = 3.0;
/// Curve sampling step on screen (points)
const PATH_SCREEN_STEP: f32 = 3.0;

/// Path tool keys: Enter applies the path, Backspace removes the selected (or last) point; Esc is a global shortcut
fn path_keys(app: &mut TextureCraftApp, ui: &Ui) {
    #[cfg(not(target_arch = "wasm32"))]
    if crate::updates_ui::blocks_input(app) {
        return;
    }

    if app.ui.mode != crate::state::Mode::Painting
        || app.session.tool.tool != Tool::Path
        || app.session.tool.path.points.is_empty()
        || ui.ctx().egui_wants_keyboard_input()
    {
        return;
    }

    let (enter, back) = ui.input_mut(|i| (i.consume_key(egui::Modifiers::NONE, egui::Key::Enter), i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace)));
    if enter {
        app.run("tool.path", json!({"commit": true}));
    }
    if back {
        app.session.remove_selected_path_point();
    }
}

/// The selected layer's stored paths except the edited one: (path sublayer, index, path)
fn other_paths(app: &TextureCraftApp) -> Vec<(texturecraft_engine::core::LayerId, usize, texturecraft_engine::PathState)> {
    let s = &app.session;
    let Some(layer) = s.selected_layer().and_then(|id| s.doc.project.layer(id).ok()) else { return Vec::new() };
    let effects = layer.content_effects.iter().chain(layer.mask.iter().flat_map(|m| m.effects.iter()));
    effects
        .flat_map(|e| s.stored_path_states(e.id).into_iter().enumerate().map(move |(i, path)| (e.id, i, path)))
        .filter(|(e, i, _)| s.tool.path.editing != Some((*e, *i)))
        .collect()
}

/// The screen curve through `points` per segment, each starting on its first point
fn screen_segments(points: &[Pos2], closed: bool) -> Vec<Vec<Pos2>> {
    let pts: Vec<Vec2> = points.iter().map(|p| Vec2::new(p.x, p.y)).collect();
    texturecraft_engine::paint::spline_segments(&pts, closed, PATH_SCREEN_STEP)
        .into_iter()
        .zip(points)
        .map(|(samples, first)| std::iter::once(*first).chain(samples.into_iter().map(|v| pos2(v.x, v.y))).collect())
        .collect()
}

/// A path's curve in a view per segment: screen position and whether the mesh leaves it visible (None off the mesh or screen)
type ScreenCurve = Vec<Vec<Option<(Pos2, bool)>>>;

/// The nearest segment of `curve` within grab distance of `at`, with the nearest position on it
fn segment_near(curve: &ScreenCurve, at: Pos2) -> Option<(usize, Pos2)> {
    let nearest = |segment: &Vec<Option<(Pos2, bool)>>| {
        segment.windows(2).filter_map(|w| Some(closest_on_line(w[0]?.0, w[1]?.0, at))).min_by(|a, b| a.distance(at).total_cmp(&b.distance(at)))
    };
    curve
        .iter()
        .enumerate()
        .filter_map(|(i, segment)| nearest(segment).map(|p| (i, p)))
        .filter(|(_, p)| p.distance(at) < HANDLE_GRAB)
        .min_by(|a, b| a.1.distance(at).total_cmp(&b.1.distance(at)))
}

/// `path`'s curve in a view; a 3D curve lies where its paint lands, and without `depth_test` (hit tests) every sample counts as visible
fn curve_on_screen(
    app: &TextureCraftApp,
    ui: &Ui,
    (rect, uv, depth_test): (Rect, bool, bool),
    path: &texturecraft_engine::PathState,
    stored: Option<(texturecraft_engine::core::LayerId, usize)>,
    show: &impl Fn(&TextureCraftApp, Vec3) -> Option<Pos2>,
) -> ScreenCurve {
    if uv {
        let Some(points) = path.points.iter().map(|p| show(app, *p)).collect::<Option<Vec<_>>>() else { return Vec::new() };
        return screen_segments(&points, path.closed).into_iter().map(|segment| segment.into_iter().map(|at| Some((at, true))).collect()).collect();
    }
    let s = &app.session;
    let surface = stored
        .and_then(|(eid, index)| stored_surface(app, ui, eid, index))
        .unwrap_or_else(|| Arc::new(s.path_surface(&path.points, path.closed, s.camera, s.viewport)));
    surface
        .iter()
        .map(|segment| segment.iter().map(|p| p.and_then(|p| show(app, p).map(|at| (at, !depth_test || path_point_visible(app, rect, p, at))))).collect())
        .collect()
}

/// Sample the stored curve once while visible, invalidating it when the path or mesh changes
fn stored_surface(app: &TextureCraftApp, ui: &Ui, eid: texturecraft_engine::core::LayerId, index: usize) -> Option<SurfaceCurve> {
    ui.memory_mut(|memory| memory.caches.cache::<PathSurfaceCache>().get(&app.session, eid, index))
}

/// The point of the segment `a`–`b` nearest to `p`
fn closest_on_line(a: Pos2, b: Pos2, p: Pos2) -> Pos2 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    a + ab * t
}

/// What `at` is over in a view (`uv`: the 2D view); `show` maps a path point to the screen
fn path_hit(app: &TextureCraftApp, ui: &Ui, (rect, uv): (Rect, bool), at: Pos2, show: &impl Fn(&TextureCraftApp, Vec3) -> Option<Pos2>) -> PathHit {
    let path = &app.session.tool.path;
    if path.uv == uv {
        if let Some(i) = path.points.iter().position(|p| show(app, *p).is_some_and(|s| s.distance(at) < HANDLE_GRAB)) {
            return PathHit::Point(i);
        }
        if let Some((i, on)) = segment_near(&curve_on_screen(app, ui, (rect, uv, false), path, path.editing, show), at) {
            return PathHit::Curve(i, on);
        }
    }
    for (eid, i, other) in other_paths(app) {
        if other.uv == uv
            && (other.points.iter().any(|p| show(app, *p).is_some_and(|s| s.distance(at) < HANDLE_GRAB))
                || segment_near(&curve_on_screen(app, ui, (rect, uv, false), &other, Some((eid, i)), show), at).is_some())
        {
            return PathHit::Stored(eid, i);
        }
    }
    PathHit::Miss
}

/// Path tool presses in a view: select and drag a point, insert on the curve, open another path, close on the first point, or add a point
fn path_input(
    app: &mut TextureCraftApp,
    ui: &Ui,
    resp: &egui::Response,
    (rect, uv): (Rect, bool),
    place: impl Fn(&TextureCraftApp, Pos2) -> Option<Vec3>,
    show: impl Fn(&TextureCraftApp, Vec3) -> Option<Pos2>,
) {
    let drag_id = resp.id.with("path_drag");
    let clicked = resp.clicked_by(PointerButton::Primary);
    if (clicked || resp.drag_started_by(PointerButton::Primary))
        && let Some(at) = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos())
    {
        let path = &app.session.tool.path;
        let (closed, len) = (path.closed, path.points.len());
        let hit = path_hit(app, ui, (rect, uv), at, &show);

        let grab = match hit {
            PathHit::Point(0) if clicked && !closed && len >= 3 => {
                app.session.set_path_closed(true);
                None
            }
            PathHit::Point(i) => {
                app.session.select_path_point(i);
                Some(i)
            }
            PathHit::Stored(eid, index) => {
                app.run("path.edit", json!({"effect": eid.0, "index": index}));
                None
            }
            hit => match place(app, if let PathHit::Curve(_, on) = hit { on } else { at }) {
                Some(p) => match hit {
                    PathHit::Curve(i, _) => app.session.insert_path_point(i + 1, p),
                    _ => app.session.append_path_point(p, uv),
                },
                None => {
                    app.toast(tr("Place path points on the mesh"), false);
                    None
                }
            },
        };
        if let Some(i) = grab {
            ui.data_mut(|d| d.insert_temp(drag_id, (i, true)));
        }
    }

    if resp.dragged_by(PointerButton::Primary)
        && let Some((index, begin_drag)) = ui.data(|data| data.get_temp::<(usize, bool)>(drag_id))
        && let Some(p) = resp.interact_pointer_pos().and_then(|pos| place(app, pos))
        && app.session.move_path_point(index, p, begin_drag)
    {
        ui.data_mut(|data| data.insert_temp(drag_id, (index, false)));
    }

    if !resp.is_pointer_button_down_on() {
        ui.data_mut(|data| data.remove::<(usize, bool)>(drag_id));
    }
}

/// Is path point `p` (on screen at `at`) in front of the mesh in the 3D view `rect`?
fn path_point_visible(app: &TextureCraftApp, rect: Rect, p: Vec3, at: Pos2) -> bool {
    let eye = app.session.camera.eye();
    let local = Vec2::new(at.x - rect.left(), at.y - rect.top());
    app.session.pick(local, app.session.viewport).is_none_or(|hit| hit.pos.distance(eye) >= p.distance(eye) * (1.0 - PATH_HIDDEN_DEPTH))
}

/// Draw `curve` in `color` over a dark outline, as a faint line where the mesh hides it
fn draw_path_curve(painter: &egui::Painter, curve: &ScreenCurve, color: Color32) {
    let draw = |run: &mut Vec<Pos2>, shown: bool| {
        if run.len() > 1 {
            // Hidden parts are a ghost line: no outline, half strength
            if shown {
                painter.add(egui::Shape::line(run.clone(), Stroke::new(PATH_OUTLINE_WIDTH, Color32::from_black_alpha(140))));
            }
            painter.add(egui::Shape::line(
                std::mem::take(run),
                Stroke::new(PATH_LINE_WIDTH, if shown { color } else { color.gamma_multiply(PATH_HIDDEN_OPACITY) }),
            ));
        }
        run.clear();
    };
    for segment in curve {
        // Runs of samples the view shows (or hides) alike, joined at the sample where that changes
        let (mut run, mut shown) = (Vec::new(), true);
        for sample in segment {
            match *sample {
                Some((at, visible)) => {
                    if visible != shown && !run.is_empty() {
                        run.push(at);
                        draw(&mut run, shown);
                    }
                    shown = visible;
                    run.push(at);
                }
                None => draw(&mut run, shown),
            }
        }
        draw(&mut run, shown);
    }
}

/// The layer's other paths (lit under the pointer), then the edited path with its points, the next segment's preview and the pen cursor
fn path_overlay(app: &TextureCraftApp, ui: &Ui, rect: Rect, resp: &egui::Response, uv: bool, show: impl Fn(&TextureCraftApp, Vec3) -> Option<Pos2>) {
    let tk = Tokens::get(ui.ctx());
    let painter = ui.painter_at(rect);
    let hint = tr("Path: click adds a point, click the curve inserts one, drag moves it, Enter applies, Esc cancels");
    // Wrapped to the view: the hint is longer than a narrow split view
    let at = rect.left_bottom() + PATH_HINT_OFFSET;
    let wrap = (rect.right() - at.x - PATH_HINT_MARGIN).max(PATH_HINT_MIN_WIDTH);
    let hint = painter.layout(hint.to_string(), egui::FontId::proportional(PATH_HINT_FONT_SIZE), tk.viewport_text, wrap);
    painter.galley(at - vec2(0.0, hint.size().y), hint, tk.viewport_text);

    let hover = resp.hover_pos();
    let hit = hover.map(|at| (at, path_hit(app, ui, (rect, uv), at, &show)));
    for (eid, i, other) in other_paths(app).iter().filter(|(_, _, other)| other.uv == uv) {
        let curve = curve_on_screen(app, ui, (rect, uv, true), other, Some((*eid, *i)), &show);
        // The path a click opens lights up
        let opened = matches!(hit, Some((_, PathHit::Stored(e, n))) if (e, n) == (*eid, *i));
        draw_path_curve(&painter, &curve, if opened { Color32::WHITE } else { Color32::WHITE.gamma_multiply(PATH_STORED_OPACITY) });
    }

    let path = &app.session.tool.path;
    if path.uv == uv {
        draw_path_curve(&painter, &curve_on_screen(app, ui, (rect, uv, true), path, path.editing, &show), tk.viewport_accent);
        let handles: Vec<Pos2> = path.points.iter().filter_map(|q| show(app, *q)).collect();
        if !path.closed
            && !handles.is_empty()
            && let Some((to, PathHit::Miss)) = hit
        {
            let mut next = handles.clone();
            next.push(to);
            if let Some(segment) = screen_segments(&next, false).last() {
                rubber_band(ui, rect, segment);
            }
        }

        for (index, point) in path.points.iter().enumerate() {
            let Some(handle) = show(app, *point) else { continue };

            // Keep the point's original index when earlier points project behind the camera
            let hovered = hover.is_some_and(|m| m.distance(handle) < HANDLE_GRAB);
            let radius = if hovered { PATH_POINT_RADIUS + PATH_POINT_HOVER_GROWTH } else { PATH_POINT_RADIUS };
            if path.selected == Some(index) {
                painter.circle(handle, radius + PATH_POINT_SELECTED_GROWTH, tk.viewport_accent, Stroke::new(1.0, Color32::from_black_alpha(160)));
            } else {
                painter.circle(handle, radius, Color32::from_black_alpha(110), Stroke::new(1.0, Color32::from_white_alpha(if hovered { 230 } else { 150 })));
            }
        }
    }

    if let Some((at, hit)) = hit
        && !ui.input(|i| i.modifiers.command)
    {
        if let PathHit::Curve(_, on) = hit {
            // Where the click inserts its point
            painter.circle(on, PATH_POINT_RADIUS, Color32::from_black_alpha(60), Stroke::new(1.0, Color32::from_white_alpha(110)));
        }
        path_cursor(ui, &painter, at, matches!(hit, PathHit::Curve(..)));
    }
}

/// The Path tool's pointer: a pen with its nib at `at`, and a plus when a click inserts a point
fn path_cursor(ui: &Ui, painter: &egui::Painter, at: Pos2, insert: bool) {
    ui.ctx().set_cursor_icon(egui::CursorIcon::None);
    let glyph = |icon: &str, size: f32, color: Color32| painter.layout_no_wrap(icon.to_string(), egui::FontId::proportional(size), color);
    for (offset, color) in [(PATH_CURSOR_SHADOW_OFFSET, Color32::from_black_alpha(200)), (egui::Vec2::ZERO, Color32::WHITE)] {
        // The pen's nib is the bottom left corner of the drawn glyph
        let pen = glyph(egui_phosphor::regular::PEN_NIB, PATH_CURSOR_SIZE, color);
        let origin = at + offset - pen.mesh_bounds.left_bottom().to_vec2();
        let right = origin + pen.mesh_bounds.right_bottom().to_vec2();
        painter.galley(origin, pen, color);
        if insert {
            let plus = glyph(egui_phosphor::regular::PLUS, PATH_CURSOR_SIZE * PATH_CURSOR_PLUS_SCALE, color);
            painter.galley(right + vec2(PATH_CURSOR_PLUS_GAP, 0.0) - plus.mesh_bounds.left_top().to_vec2(), plus, color);
        }
    }
}

/// Dashed curve preview clipped segment by segment to bound offscreen dash allocation
fn rubber_band(ui: &Ui, rect: Rect, line: &[Pos2]) {
    let painter = ui.painter_at(rect);
    let period = RUBBER_BAND_DASH + RUBBER_BAND_GAP;
    let mut distance = 0.0;
    for pair in line.windows(2) {
        if let Some(visible) = visible_dashes(rect, pair[0], pair[1]) {
            // Continue the dash pattern across sampled curve segments and clipped portions
            let phase = (distance + pair[0].distance(visible[0])) % period;
            for stroke in [Stroke::new(2.5, Color32::from_black_alpha(120)), Stroke::new(1.0, Color32::from_white_alpha(200))] {
                // Draw any dash already in progress without extending behind this curve segment
                if phase > 0.0 && phase < RUBBER_BAND_DASH {
                    let direction = visible[1] - visible[0];
                    let end = visible[0] + direction.normalized() * (RUBBER_BAND_DASH - phase).min(direction.length());
                    painter.line_segment([visible[0], end], stroke);
                }
                let offset = if phase == 0.0 { 0.0 } else { period - phase };
                painter.add(egui::Shape::dashed_line_with_offset(&visible, stroke, &[RUBBER_BAND_DASH], &[RUBBER_BAND_GAP], offset));
            }
        }
        distance = (distance + pair[0].distance(pair[1])) % period;
    }
}

/// The part of `from`→`to` inside `rect`, its start moved back to a whole dash period so the dashes keep their place.
fn visible_dashes(rect: Rect, from: Pos2, to: Pos2) -> Option<[Pos2; 2]> {
    let direction = to - from;
    let length = direction.length();
    if length == 0.0 {
        return rect.contains(from).then_some([from, to]);
    }

    // Clip the segment's parameter range against each axis's slab
    let (mut enter, mut exit) = (0.0_f32, 1.0_f32);
    for (delta, origin, low, high) in [(direction.x, from.x, rect.left(), rect.right()), (direction.y, from.y, rect.top(), rect.bottom())] {
        if delta == 0.0 {
            if origin < low || origin > high {
                return None;
            }
            continue;
        }
        let (at_low, at_high) = ((low - origin) / delta, (high - origin) / delta);
        enter = enter.max(at_low.min(at_high));
        exit = exit.min(at_low.max(at_high));
    }
    if enter > exit {
        return None;
    }

    let period = RUBBER_BAND_DASH + RUBBER_BAND_GAP;
    let start = (enter * length / period).floor() * period / length;
    Some([from + direction * start, from + direction * exit])
}

/// Polygon fill in a view: a click fills one target, a drag fills its box live (one undo step); returns the box to draw over the view image
fn polygon_fill_input(
    app: &mut TextureCraftApp,
    ui: &Ui,
    response: &egui::Response,
    fill_at: impl Fn(&mut Session, Pos2) -> texturecraft_engine::Result<usize>,
    drag_box: impl Fn(&mut Session, Pos2, Pos2) -> texturecraft_engine::Result<usize>,
) -> Option<Rect> {
    let origin_id = response.id.with("polygon_fill_box");
    let dragging = response.dragged_by(PointerButton::Primary);
    let released = response.drag_stopped_by(PointerButton::Primary);
    let held = ui.input(|input| input.pointer.primary_down());
    if !held && !released {
        // A drag this view started and cut short (Alt, a second finger) keeps what it filled as one step
        if ui.data_mut(|data| data.remove_temp::<Pos2>(origin_id)).is_some() {
            app.session.end_polygon_fill_box();
        }
    } else if dragging && let Some(press) = ui.input(|input| input.pointer.press_origin()) {
        ui.data_mut(|data| data.insert_temp(origin_id, press));
    }

    // Consume the origin on release so a new press cannot resume the previous rectangle
    let origin = if released { ui.data_mut(|data| data.remove_temp::<Pos2>(origin_id)) } else { ui.data(|data| data.get_temp::<Pos2>(origin_id)) };
    let pointer = ui.input(|input| input.pointer.latest_pos());
    let (filled, dragged_box) = match (origin, pointer) {
        _ if response.clicked_by(PointerButton::Primary) => (response.interact_pointer_pos().map(|point| fill_at(&mut app.session, point)), None),
        (Some(origin), Some(pointer)) if released => {
            // The drag ends on release even when its last move failed
            let shown = drag_box(&mut app.session, origin, pointer);
            let filled = app.session.end_polygon_fill_box();
            (Some(shown.map(|_| filled)), None)
        }
        _ if released => {
            app.session.end_polygon_fill_box();
            (None, None)
        }
        (Some(origin), Some(pointer)) => (Some(drag_box(&mut app.session, origin, pointer)), Some(Rect::from_two_pos(origin, pointer))),
        _ => (None, None),
    };

    if let Some(Err(error)) = filled {
        app.toast(error.to_string(), true);
    }

    dragged_box
}

/// How strongly the polygon fill box tints what it covers (0..1)
const FILL_BOX_TINT: f32 = 0.15;

/// The polygon fill box being dragged, over view `rect`
fn polygon_fill_box_overlay(ui: &Ui, rect: Rect, fill_box: Option<Rect>) {
    let Some(fill_box) = fill_box else { return };

    let accent = Tokens::get(ui.ctx()).viewport_accent;
    ui.painter_at(rect).rect(fill_box, 0.0, accent.gamma_multiply(FILL_BOX_TINT), Stroke::new(1.0, accent), egui::StrokeKind::Inside);
}

/// V held with the Clone tool in painting mode: clicks set the clone source.
fn clone_source_key(app: &TextureCraftApp, ui: &Ui) -> bool {
    app.ui.mode == crate::state::Mode::Painting
        && app.session.tool.tool == Tool::Clone
        && ui.input(|i| i.key_down(egui::Key::V))
        && !ui.ctx().egui_wants_keyboard_input()
}

/// Crosshair where the Clone tool reads from.
fn clone_marker(app: &TextureCraftApp, ui: &Ui, rect: Rect, pos: Vec3) {
    let tk = Tokens::get(ui.ctx());
    let Some(c) = project(app, rect, pos) else { return };
    let p = ui.painter_at(rect);
    for (w, color) in [(3.0, Color32::from_black_alpha(140)), (1.2, tk.viewport_accent)] {
        p.circle_stroke(c, 7.0, Stroke::new(w, color));
        p.line_segment([c - vec2(11.0, 0.0), c + vec2(11.0, 0.0)], Stroke::new(w, color));
        p.line_segment([c - vec2(0.0, 11.0), c + vec2(0.0, 11.0)], Stroke::new(w, color));
    }
}

/// Tools that paint with the brush (and show its cursor).
fn brush_tool(t: Tool) -> bool {
    matches!(t, Tool::Paint | Tool::Eraser | Tool::Projection | Tool::Smudge | Tool::Clone | Tool::Path)
}

/// Ctrl (Cmd on macOS) + drag in a view changes the brush instead of painting or navigating: a
/// right drag its size (left/right) and hardness (up/down), a left drag its flow and angle. True
/// while Ctrl is held with a brush tool and no stroke is under way.
fn brush_drag(app: &mut TextureCraftApp, ui: &Ui, resp: &egui::Response) -> bool {
    let held =
        app.ui.mode == crate::state::Mode::Painting && brush_tool(app.session.tool.tool) && !app.session.stroke_active() && ui.input(|i| i.modifiers.command);
    if held && let Some(right) = brush_drag_button(resp) {
        adjust_brush(&mut app.session.tool.brush, right, ui.input(|i| i.pointer.delta()));
    }
    held
}

/// The button of a drag on `resp`: Some(true) right, Some(false) left.
fn brush_drag_button(resp: &egui::Response) -> Option<bool> {
    let right = resp.dragged_by(PointerButton::Secondary);
    (right || resp.dragged_by(PointerButton::Primary)).then_some(right)
}

/// While a Ctrl+drag changes the brush, its values beside where the drag began (drawn over the
/// view; the tool bar and Properties show them too).
fn brush_readout(app: &TextureCraftApp, ui: &Ui, resp: &egui::Response, adjusting: bool) {
    let Some(right) = brush_drag_button(resp).filter(|_| adjusting) else { return };
    let Some(at) = ui.input(|i| i.pointer.press_origin()) else { return };
    // The pointer would cover the values; the brush cursor stays where the drag began.
    ui.ctx().set_cursor_icon(egui::CursorIcon::None);
    let b = &app.session.tool.brush;
    let text = if right {
        format!("{} {:.1}   {} {:.0}", tr("Size"), b.size / 0.25 * 100.0, tr("Hardness"), b.hardness * 100.0)
    } else {
        format!("{} {:.0}   {} {:.0}°", tr("Flow"), b.flow * 100.0, tr("Angle"), b.angle)
    };
    let tk = Tokens::get(ui.ctx());
    let p = ui.painter_at(resp.rect);
    let g = p.layout_no_wrap(text, egui::FontId::proportional(12.0), tk.text_strong);
    let r = Rect::from_min_size(at + vec2(18.0, -30.0), g.size()).expand2(vec2(6.0, 3.0));
    p.rect_filled(r, 3.0, tk.panel);
    p.galley(r.min + vec2(6.0, 3.0), g, tk.text_strong);
}

/// Size fraction, hardness and flow change per point of a Ctrl+drag
const CTRL_DRAG_RATE_PER_POINT: f32 = 0.005;

/// Brush angle change per point of a Ctrl+left drag (degrees)
const CTRL_DRAG_DEGREES_PER_POINT: f32 = 0.5;

/// Highest hardness a Ctrl+right drag sets
const CTRL_DRAG_MAX_HARDNESS: f32 = 0.999;

/// Brush change for a Ctrl+drag by `drag` points: a right drag sets the size (x) and hardness (y), a left drag the flow (x) and angle (y); up raises hardness and angle
fn adjust_brush(brush: &mut texturecraft_engine::paint::BrushSettings, right: bool, drag: egui::Vec2) {
    if right {
        brush.size = (brush.size * (1.0 + drag.x * CTRL_DRAG_RATE_PER_POINT)).clamp(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE);
        brush.hardness = (brush.hardness - drag.y * CTRL_DRAG_RATE_PER_POINT).clamp(0.0, CTRL_DRAG_MAX_HARDNESS);
    } else {
        brush.flow = (brush.flow + drag.x * CTRL_DRAG_RATE_PER_POINT).clamp(0.0, 1.0);
        brush.angle = (brush.angle - drag.y * CTRL_DRAG_DEGREES_PER_POINT + 180.0).rem_euclid(360.0) - 180.0;
    }
}

pub(crate) fn brush_cursor(app: &TextureCraftApp, ui: &Ui, rect: Rect, position: Vec3, normal: Vec3, radius: f32) {
    const CIRCLE_SEGMENTS: usize = 48;

    let tangent = normal.any_orthonormal_vector();
    let bitangent = normal.cross(tangent);
    let to_screen = |point: Vec3| project(app, rect, point);
    let mut outline = Vec::with_capacity(CIRCLE_SEGMENTS + 1);
    for segment in 0..=CIRCLE_SEGMENTS {
        let angle = segment as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        if let Some(screen) = to_screen(position + (tangent * angle.cos() + bitangent * angle.sin()) * radius) {
            outline.push(screen);
        }
    }

    let painter = ui.painter_at(rect);
    painter.add(egui::Shape::line(outline.clone(), Stroke::new(2.5, Color32::from_black_alpha(140))));
    painter.add(egui::Shape::line(outline, Stroke::new(1.2, Color32::from_white_alpha(230))));

    let hardness = app.session.tool.brush.hardness;
    if hardness > 0.05 {
        let mut inner = Vec::new();
        for segment in 0..=CIRCLE_SEGMENTS {
            let angle = segment as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
            if let Some(screen) = to_screen(position + (tangent * angle.cos() + bitangent * angle.sin()) * radius * hardness) {
                inner.push(screen);
            }
        }
        painter.add(egui::Shape::dashed_line(&inner, Stroke::new(1.0, Color32::from_white_alpha(120)), 3.0, 3.0));
    }

    if let Some(center) = to_screen(position) {
        painter.circle_filled(center, 1.5, Color32::WHITE);
    }
}

/// This frame's pen pressure (0..1), or 1 without a pen. Pressure arrives as touch force: the
/// windowing layer reports it for Windows Ink pens, and in the browser for pens that send touch
/// events (tablets and phones); other pens come through as a mouse.
/// shortcut: the windowing layer reports no pen tilt, so tilt can't drive the brush angle yet.
fn pen_pressure(ui: &Ui) -> f32 {
    let id = egui::Id::unique("pen_pressure");
    let touch = ui.input(|i| {
        i.raw.events.iter().rev().find_map(|e| match e {
            egui::Event::Touch { phase, force, .. } => Some((*phase, force.filter(|f| f.is_finite()))),
            _ => None,
        })
    });
    match touch {
        Some((_, Some(f))) => {
            let f = f.clamp(0.0, 1.0);
            ui.data_mut(|d| d.insert_temp(id, f));
            f
        }
        // A pen lifts with no pressure, and a tap paints on that frame: keep its contact pressure.
        Some((egui::TouchPhase::End, None)) => ui.data(|d| d.get_temp(id)).unwrap_or(1.0),
        // A touch without pressure (a finger) must not inherit an earlier pen's.
        Some((egui::TouchPhase::Start, None)) => {
            ui.data_mut(|d| d.remove_temp::<f32>(id));
            1.0
        }
        _ => 1.0,
    }
}

/// The 2D (UV) view: navigation, painting in texture coordinates and the overlays drawn over the texture.
fn view_2d(app: &mut TextureCraftApp, ui: &mut Ui, rect: Rect, sync: &mut PendingSync) {
    let tokens = Tokens::get(ui.ctx());
    let response = ui.interact(rect, ui.scope_id().with("view2d"), Sense::click_and_drag());
    let size = rect.size();
    let (alt, shift) = ui.input(|input| (input.modifiers.alt, input.modifiers.shift));
    let adjusting = brush_drag(app, ui, &response);

    // Pan with a middle drag or Alt+left drag, zoom with the wheel about the cursor
    let delta = ui.input(|input| input.pointer.delta());
    if response.dragged_by(PointerButton::Middle) || (alt && response.dragged_by(PointerButton::Primary)) {
        app.ui.uv_pan[0] += delta.x / size.x * 2.0;
        app.ui.uv_pan[1] -= delta.y / size.y * 2.0;
    }
    if response.hovered() {
        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let old_zoom = app.ui.uv_zoom;
            let new_zoom = (old_zoom * (1.0 + scroll * UV_WHEEL_ZOOM_PER_POINT)).clamp(UV_VIEW_MIN_ZOOM, UV_VIEW_MAX_ZOOM);
            if let Some(hover_pos) = response.hover_pos() {
                let cursor = [(hover_pos.x - rect.left()) / size.x * 2.0 - 1.0, 1.0 - (hover_pos.y - rect.top()) / size.y * 2.0];
                for axis in 0..2 {
                    app.ui.uv_pan[axis] = cursor[axis] - (cursor[axis] - app.ui.uv_pan[axis]) * new_zoom / old_zoom;
                }
            }
            app.ui.uv_zoom = new_zoom;
        }
    }

    // Conversions between the view's points and the texture set's UV space
    let ts = app.session.ts();
    let set = app.session.doc.project.texture_sets.get(ts);
    let [scale_x, scale_y] = texturecraft_render::uv_view_scale(size.x / size.y.max(1.0), app.ui.uv_zoom, set.map_or((1, 1), |texture_set| texture_set.size()));
    let pan = app.ui.uv_pan;
    let to_uv = |pointer: egui::Pos2| {
        let clip_x = (pointer.x - rect.left()) / size.x * 2.0 - 1.0;
        let clip_y = 1.0 - (pointer.y - rect.top()) / size.y * 2.0;
        Vec2::new(((clip_x - pan[0]) / scale_x + 1.0) * 0.5, (1.0 - (clip_y - pan[1]) / scale_y) * 0.5)
    };
    let to_screen = |u: f32, v: f32| {
        let clip_x = (u * 2.0 - 1.0) * scale_x + pan[0];
        let clip_y = (1.0 - v * 2.0) * scale_y + pan[1];
        pos2(rect.left() + (clip_x * 0.5 + 0.5) * size.x, rect.top() + (0.5 - clip_y * 0.5) * size.y)
    };
    let show = |_: &TextureCraftApp, point: Vec3| Some(to_screen(point.x, point.y));
    let path_tool = app.session.tool.tool == Tool::Path;

    // Clone source: V + click
    let v_held = clone_source_key(app, ui);
    if v_held
        && (response.clicked_by(PointerButton::Primary) || response.drag_started_by(PointerButton::Primary))
        && let Some(pointer) = response.interact_pointer_pos()
    {
        let uv = to_uv(pointer);
        app.run("tool.cloneSource", json!({"uv": [uv.x, uv.y]}));
    }

    let bake_paint_on = app.ui.mode == crate::state::Mode::Baking && app.ui.bake_paint_brush.on && !alt;
    if bake_paint_on {
        bake_paint_input(app, ui, &response, &to_uv);
    }

    crate::properties_popup::open_on_right_click(app, ui, &response, rect, v_held);

    // A pending ID color pick takes the next click instead of the brush
    let picking_id_color = app.picking_id_color.is_some();
    if picking_id_color {
        crate::id_color_pick::IdColorPick::draw_cursor(ui, rect);
        if !alt
            && response.clicked_by(PointerButton::Primary)
            && let Some(position) = response.interact_pointer_pos()
        {
            let uv = to_uv(position);
            crate::id_color_pick::IdColorPick::pick_at(app, json!({"uv": [uv.x, uv.y]}));
        }
    }

    // Painting in UV space
    let painting = app.ui.mode == crate::state::Mode::Painting
        && app.properties_popup.is_none()
        && !crate::rail_popup::blocks_paint_input(app, ui.ctx())
        && !alt
        && !adjusting
        && !v_held
        && !picking_id_color
        && !DecalDrop::blocks_input(ui.ctx());
    let mut fill_box = None;
    if painting && path_tool {
        path_input(app, ui, &response, (rect, true), |_, pointer| Some(to_uv(pointer).extend(0.0)), show);
    } else if painting && app.session.tool.tool == Tool::PolygonFill {
        fill_box = polygon_fill_input(
            app,
            ui,
            &response,
            |session, point| session.polygon_fill_uv(to_uv(point)),
            |session, corner, opposite| session.drag_polygon_fill_box_uv(to_uv(corner), to_uv(opposite)),
        );
    } else if painting {
        paint_2d_input(app, ui, &response, shift, &to_uv);
    }
    if response.double_clicked() && !(painting && path_tool) {
        app.ui.uv_zoom = 1.0;
        app.ui.uv_pan = [0.0, 0.0];
    }

    let frame = Frame {
        view_id: VIEW_2D,
        size: frame_size_in_pixels(ui, rect),
        kind: ViewKind::Uv { ts, zoom: app.ui.uv_zoom, pan: app.ui.uv_pan },
        camera: app.session.camera,
        shade: shade(app),
        visible: app.session.doc.project.texture_sets.iter().map(|texture_set| texture_set.visible).collect(),
        overlay: Vec::new(),
    };
    add_callback(app, ui, rect, frame, sync);
    polygon_fill_box_overlay(ui, rect, fill_box);
    crate::shelf::drop_target(app, ui, &response, crate::shelf::DropZone::View2d);

    // The UV tile outlines (numbered for UDIM sets) and the path being edited
    let painter = ui.painter_at(rect);
    let set = app.session.doc.project.texture_sets.get(ts);
    let outline = |tile: Rect| painter.rect_stroke(tile, 0.0, Stroke::new(1.0, Color32::from_white_alpha(UV_TILE_OUTLINE_ALPHA)), egui::StrokeKind::Outside);
    match set.and_then(|texture_set| Some((texture_set, texture_set.tile_grid()?))) {
        Some((texture_set, grid)) => {
            for (udim, (block_x, block_y)) in texture_set.uv_tiles.iter().filter_map(|udim| Some((*udim, grid.block(*udim)?))) {
                // A tile smaller than its block fills the block's top-left corner
                let (u0, v0) = (block_x as f32 / grid.cols as f32, block_y as f32 / grid.rows as f32);
                let tile_scale = texture_set.tile_scale(udim);
                let tile = Rect::from_two_pos(to_screen(u0, v0), to_screen(u0 + tile_scale / grid.cols as f32, v0 + tile_scale / grid.rows as f32));
                outline(tile);
                painter.text(
                    tile.left_top() + vec2(6.0, 4.0),
                    egui::Align2::LEFT_TOP,
                    udim.to_string(),
                    egui::FontId::proportional(11.0),
                    tokens.viewport_text,
                );
            }
        }
        None => {
            outline(Rect::from_two_pos(to_screen(0.0, 0.0), to_screen(1.0, 1.0)));
        }
    }
    if app.ui.mode == crate::state::Mode::Painting && path_tool {
        path_overlay(app, ui, rect, &response, true, show);
    }

    // Shift: preview the straight line Shift+click paints from the last stroke
    if painting
        && shift
        && !path_tool
        && brush_tool(app.session.tool.tool)
        && let Some(from) = app.last_paint_uv.and_then(|point| point.uv_for(&app.session))
        && let Some(hover) = response.hover_pos()
    {
        rubber_band(ui, rect, &[to_screen(from.x, from.y), hover]);
    }

    if app.ui.mode == crate::state::Mode::Baking {
        bake_paint_overlay(app, ui, &painter, to_screen(0.0, 1.0), to_screen(1.0, 0.0));
        if bake_paint_on && let Some(hover) = response.hover_pos() {
            let radius = (to_screen(app.ui.bake_paint_brush.radius, 0.0).x - to_screen(0.0, 0.0).x).abs().max(BAKE_PAINT_CURSOR_MIN_RADIUS);
            painter.circle_stroke(hover, radius, Stroke::new(1.0, Color32::from_white_alpha(200)));
        }
    }

    brush_readout(app, ui, &response, adjusting);
    view_header(app, ui, rect, "v2d");
    axis_gizmo(ui, rect, app, true);
    let name = app.session.doc.project.texture_sets.get(ts).map(|texture_set| texture_set.name.clone()).unwrap_or_default();
    ui.painter().text(rect.right_bottom() + vec2(-10.0, -10.0), egui::Align2::RIGHT_BOTTOM, name, egui::FontId::proportional(11.0), tokens.viewport_text);
}

/// Paint in the 2D view: a press starts a stroke, a drag continues it, Shift+click draws a line from the last stroke.
fn paint_2d_input(app: &mut TextureCraftApp, ui: &Ui, response: &egui::Response, shift: bool, to_uv: &dyn Fn(egui::Pos2) -> Vec2) {
    let pressure = pen_pressure(ui);

    // Where the stroke under way is; it becomes the line start only when the stroke finishes
    let stroke_end = ui.scope_id().with("paint_2d_stroke_end");

    if response.drag_started_by(PointerButton::Primary) || response.clicked_by(PointerButton::Primary) {
        let line_start = app.last_paint_uv.filter(|_| shift).and_then(|point| point.uv_for(&app.session));
        if let Some(pointer) = response.interact_pointer_pos() {
            let uv = to_uv(pointer);
            match app.session.begin_stroke_uv(line_start.unwrap_or(uv), pressure) {
                Ok(()) => {
                    if line_start.is_some() {
                        app.session.continue_stroke_uv(uv, pressure);
                    }
                    ui.data_mut(|data| data.insert_temp(stroke_end, crate::uv_paint_point::UvPaintPoint::new(&app.session, uv)));
                }
                Err(error) => app.toast(error.to_string(), true),
            }
        }
    } else if response.dragged_by(PointerButton::Primary)
        && app.session.stroke_active()
        && let Some(pointer) = response.interact_pointer_pos()
    {
        let uv = to_uv(pointer);
        app.session.continue_stroke_uv(uv, pressure);
        ui.data_mut(|data| data.insert_temp(stroke_end, crate::uv_paint_point::UvPaintPoint::new(&app.session, uv)));
    }

    // Escape cancels the stroke and ends the drag without a primary release, so a cancelled stroke never gets here
    if response.clicked_by(PointerButton::Primary) || response.drag_stopped_by(PointerButton::Primary) {
        let finishing = app.session.stroke_active();
        match app.session.end_stroke() {
            Ok(()) if finishing => app.last_paint_uv = ui.data(|data| data.get_temp(stroke_end)),
            Ok(()) => {}
            Err(error) => app.toast(error.to_string(), true),
        }
    }
}

/// Paint the skew map under the pointer in the 2D view (a click is one dab, a drag a stroke).
fn bake_paint_input(app: &mut TextureCraftApp, ui: &Ui, resp: &egui::Response, to_uv: &dyn Fn(egui::Pos2) -> Vec2) {
    let last = ui.scope_id().with(("bake_paint_last", app.session.ts(), app.ui.bake_paint_brush.target as u8));
    let brush = app.ui.bake_paint_brush;
    let started = resp.drag_started_by(PointerButton::Primary) || resp.clicked_by(PointerButton::Primary);
    if !(started || resp.dragged_by(PointerButton::Primary)) {
        return;
    }
    let Some(p) = resp.interact_pointer_pos() else { return };
    let uv = to_uv(p);
    let previous: Option<[f32; 2]> = if started { None } else { ui.data(|d| d.get_temp(last)) };
    let points: Vec<[f32; 2]> = previous.into_iter().chain([[uv.x, uv.y]]).collect();
    let command = match brush.target {
        texturecraft_engine::core::BakePaintTarget::Skew => "bake.skew.paint",
        texturecraft_engine::core::BakePaintTarget::Cage => "bake.cage.paint",
    };
    if app.run(command, json!({"points": points, "radius": brush.radius, "value": brush.value, "hardness": brush.hardness, "continue": !started})).is_some() {
        ui.data_mut(|d| d.insert_temp(last, [uv.x, uv.y]));
    }
}

/// The skew map as a translucent orange layer over the UV square from `top_left` to `bottom_right`.
fn bake_paint_overlay(app: &TextureCraftApp, ui: &Ui, painter: &egui::Painter, top_left: egui::Pos2, bottom_right: egui::Pos2) {
    let Some(k) = app.session.doc.bake_paint.get(&(app.session.ts(), app.ui.bake_paint_brush.target)).filter(|k| k.values.len() > 1) else { return };
    let key = (app.session.revision, std::sync::Arc::as_ptr(k) as usize);
    let id = ui.scope_id().with("bake_paint_overlay_texture");
    let cached: Option<((u64, usize), egui::TextureHandle)> = ui.data(|d| d.get_temp(id));
    let tex = match cached {
        Some((c, t)) if c == key => t,
        _ => {
            // At most 256 texels a side: the overlay is a guide, not the map.
            let (w, h) = (k.width as usize, k.height as usize);
            let (ow, oh) = (w.min(256), h.min(256));
            let mut rgba = Vec::with_capacity(ow * oh * 4);
            for y in 0..oh {
                for x in 0..ow {
                    let v = k.values.get((y * h / oh) * w + x * w / ow).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    let v = match app.ui.bake_paint_brush.target {
                        texturecraft_engine::core::BakePaintTarget::Skew => v,
                        texturecraft_engine::core::BakePaintTarget::Cage => 1.0 - v,
                    };
                    rgba.extend_from_slice(&[255, 140, 40, (v * 170.0) as u8]);
                }
            }
            let tex = ui.ctx().load_texture("bake_paint_overlay", egui::ColorImage::from_rgba_unmultiplied([ow, oh], &rgba), egui::TextureOptions::NEAREST);
            ui.data_mut(|d| d.insert_temp(id, (key, tex.clone())));
            tex
        }
    };
    painter.image(tex.id(), Rect::from_two_pos(top_left, bottom_right), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_surfaces_release_after_a_frame_without_paths() {
        use texturecraft_engine::{core::EffectKind, execute, geom::primitives};

        let mut session = Session::new_project("Path cache", primitives::sphere(1.0, 48, 24), "builtin:sphere", None, 256).unwrap();
        session.viewport = Vec2::splat(400.0);
        session.camera.yaw = 0.0;
        session.camera.pitch = 0.0;
        let result = execute(&mut session, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
        let effect = texturecraft_engine::core::LayerId(result["committed"]["effect"].as_u64().unwrap());
        let EffectKind::Path(paths) = &mut session.doc.project.effect_mut(effect).unwrap().kind else { panic!("path effect") };
        paths.resize(8, paths[0].clone());
        let mut harness = egui_kittest::Harness::builder().build_eframe(|context| TextureCraftApp::new(context, session));
        let context = egui::Context::default();
        let mut previous = Vec::<std::sync::Weak<Vec<Vec<Option<Vec3>>>>>::new();

        // Every path in the selection stays warm, and leaving it releases every sampled buffer
        for _ in 0..4 {
            let output = context.run_ui(Default::default(), |ui| {
                for index in 0..8 {
                    let surface = stored_surface(harness.state(), ui, effect, index).unwrap();
                    assert!(surface.iter().map(Vec::len).sum::<usize>() > 10);
                    if let Some(cached) = previous.get(index) {
                        assert!(Arc::ptr_eq(&cached.upgrade().unwrap(), &surface), "unchanged paths must not be resampled");
                    } else {
                        previous.push(Arc::downgrade(&surface));
                    }
                }
            });

            // This cache lifecycle test has no renderer to consume texture updates
            output.drop_without_applying_deltas();
        }

        let EffectKind::Path(paths) = &mut harness.state_mut().session.doc.project.effect_mut(effect).unwrap().kind else { panic!("path effect") };
        for path in paths {
            path.viewport = [800.0, 800.0];
        }
        let output = context.run_ui(Default::default(), |ui| {
            for (index, previous) in previous.iter_mut().enumerate() {
                let surface = stored_surface(harness.state(), ui, effect, index).unwrap();
                assert!(previous.upgrade().is_none(), "editing a path releases its obsolete samples");
                *previous = Arc::downgrade(&surface);
            }
        });
        output.drop_without_applying_deltas();

        for _ in 0..2 {
            context.run_pass(Default::default(), |_| {}).drop_without_applying_deltas();
        }

        assert!(previous.iter().all(|surface| surface.upgrade().is_none()), "unused path surfaces must be freed while the window stays open");
    }

    /// A selected mask decal on the ball, with another texture set available for selection tests
    fn decal_frame_app(render: bool) -> egui_kittest::Harness<'static, TextureCraftApp> {
        let session =
            Session::new_project("Decal selection", texturecraft_engine::geom::primitives::material_ball(), "builtin:material_ball", None, 256).unwrap();
        let builder = egui_kittest::Harness::builder().with_size(vec2(1600.0, 960.0));
        let builder = if render { builder.wgpu() } else { builder };
        let mut harness = builder.build_eframe(|context| TextureCraftApp::new(context, session));
        harness.run_steps(4);
        let mut png = Vec::new();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 255, 255, 255])).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let image = harness.state_mut().session.import_resource_bytes("decal.png", png).unwrap();
        harness.state_mut().run("layer.add.decal", json!({"image": image, "channel": "mask", "size": 0.5})).unwrap();
        harness.run_steps(2);
        assert_eq!(harness.state().session.ts(), 0);
        assert!(DecalFrame::of(harness.state(), harness.state().viewport_rect).is_some());
        harness
    }

    /// Move a primary-button drag through enough frames to reach the viewport's decal handles
    fn drag_decal_handle(harness: &mut egui_kittest::Harness<'_, TextureCraftApp>, from: Pos2, to: Pos2) {
        let modifiers = egui::Modifiers::NONE;
        harness.hover_at(from);
        harness.event(egui::Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers });
        harness.run_steps(2);
        for step in 1..=4 {
            harness.hover_at(from.lerp(to, step as f32 / 4.0));
            harness.run_steps(1);
        }
        harness.event(egui::Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers });
        harness.run_steps(2);
    }

    #[test]
    fn decal_frame_rejects_a_stale_effect_after_failed_layer_selection() {
        use texturecraft_engine::{StackTarget, execute};

        for texture_set in [0, 1] {
            let mut harness = decal_frame_app(false);
            let original = DecalFrame::of(harness.state(), harness.state().viewport_rect).unwrap();
            let effect = original.effect.unwrap();
            let session = &mut harness.state_mut().session;
            execute(session, "textureSet.select", &json!({"ts": texture_set})).unwrap();
            let other = execute(session, "layer.add.paint", &json!({"name": "Other layer"})).unwrap()["layer"].clone();
            execute(session, "layer.select", &json!({"layer": original.layer.0, "target": effect.0})).unwrap();

            let selected = execute(session, "layer.select", &json!({"layer": other, "target": effect.0}));

            assert!(selected.is_err());
            assert_eq!(session.ts(), texture_set);
            assert_eq!(session.selected_layer().unwrap().0, other.as_u64().unwrap());
            assert_eq!(session.target, StackTarget::Effect(effect), "the failed selection leaves the previous effect target");
            assert!(DecalFrame::of(harness.state(), harness.state().viewport_rect).is_none(), "a stale effect must not show a decal frame");

            drag_decal_handle(&mut harness, original.corners[1], original.center + (original.corners[1] - original.center) * 1.5);
            drag_decal_handle(&mut harness, original.knob, original.knob + vec2(40.0, 0.0));

            let effect = &harness.state().session.doc.project.effect(effect).unwrap().1.kind;
            assert!(
                matches!(effect, texturecraft_engine::core::EffectKind::Fill { projection: texturecraft_engine::core::Projection::Decal(decal), .. } if *decal == original.decal)
            );
        }
    }

    #[test]
    fn decal_frame_keeps_an_instances_inherited_mask_effect_editable() {
        use texturecraft_engine::execute;

        let mut harness = decal_frame_app(false);
        let original = DecalFrame::of(harness.state(), harness.state().viewport_rect).unwrap();
        let effect = original.effect.unwrap();
        let session = &mut harness.state_mut().session;
        let instance = execute(session, "layer.instance", &json!({"layer": original.layer.0, "ts": 1})).unwrap()["layer"].clone();

        // The source effect remains a valid inherited target when selection changes to its instance
        assert!(execute(session, "layer.select", &json!({"layer": instance, "target": effect.0})).is_err());

        assert_eq!(session.ts(), 1);
        assert_eq!(session.selected_layer().unwrap().0, instance.as_u64().unwrap());
        let inherited = DecalFrame::of(harness.state(), harness.state().viewport_rect).unwrap();
        assert_eq!((inherited.layer, inherited.effect, inherited.decal), (original.layer, original.effect, original.decal));

        drag_decal_handle(&mut harness, inherited.corners[1], inherited.center + (inherited.corners[1] - inherited.center) * 1.5);

        let edited = DecalFrame::of(harness.state(), harness.state().viewport_rect).unwrap();
        assert!(edited.decal.size[0] > original.decal.size[0], "scaling an inherited decal edits its source");
    }

    #[test]
    fn decal_frame_rejects_a_source_mask_overridden_by_the_selected_instance() {
        use texturecraft_engine::execute;

        let mut harness = decal_frame_app(false);
        let original = DecalFrame::of(harness.state(), harness.state().viewport_rect).unwrap();
        let effect = original.effect.unwrap();
        let session = &mut harness.state_mut().session;
        let instance = execute(session, "layer.instance", &json!({"layer": original.layer.0, "ts": 1})).unwrap()["layer"].clone();
        execute(session, "mask.add.black", &json!({"layer": instance})).unwrap();
        execute(session, "layer.select", &json!({"layer": original.layer.0, "target": effect.0})).unwrap();

        assert!(execute(session, "layer.select", &json!({"layer": instance, "target": effect.0})).is_err());

        assert!(DecalFrame::of(harness.state(), harness.state().viewport_rect).is_none(), "an override hides the source's mask decal");
    }

    #[test]
    #[ignore = "requires an offscreen GPU adapter and TC_UI_SNAPSHOTS"]
    fn decal_frame_selection_snapshots() {
        use texturecraft_engine::execute;

        let Some(directory) = std::env::var_os("TC_UI_SNAPSHOTS").map(std::path::PathBuf::from) else { return };

        std::fs::create_dir_all(&directory).unwrap();
        let mut harness = decal_frame_app(true);
        let original = DecalFrame::of(harness.state(), harness.state().viewport_rect).unwrap();
        let effect = original.effect.unwrap();
        let session = &mut harness.state_mut().session;
        let instance = execute(session, "layer.instance", &json!({"layer": original.layer.0, "ts": 1})).unwrap()["layer"].clone();
        assert!(execute(session, "layer.select", &json!({"layer": instance, "target": effect.0})).is_err());
        harness.run_steps(6);

        assert!(DecalFrame::of(harness.state(), harness.state().viewport_rect).is_some());
        harness.render().unwrap().save(directory.join("decal-frame-inherited.png")).unwrap();

        let session = &mut harness.state_mut().session;
        let other = execute(session, "layer.add.paint", &json!({"name": "Other layer"})).unwrap()["layer"].clone();
        execute(session, "layer.select", &json!({"layer": original.layer.0, "target": effect.0})).unwrap();
        assert!(execute(session, "layer.select", &json!({"layer": other, "target": effect.0})).is_err());
        harness.run_steps(6);

        assert!(DecalFrame::of(harness.state(), harness.state().viewport_rect).is_none());
        harness.render().unwrap().save(directory.join("decal-frame-stale.png")).unwrap();
    }

    #[test]
    fn live_bake_preview_hides_setup_overlays_and_restores_their_preferences() {
        let session = texturecraft_engine::sample_session(256, 64).unwrap();
        let mut harness = egui_kittest::Harness::builder().build_eframe(|context| TextureCraftApp::new(context, session));
        let app = harness.state_mut();
        app.ui.mode = crate::state::Mode::Baking;
        let before = bake_visualization(app).unwrap();
        assert!(before.high_poly && before.cage);

        app.start_bake(vec![0]);

        assert!(bake_visualization(app).is_none(), "setup overlays must not obscure the finished map");
        for show in [texturecraft_engine::PreviewShow::Pin(BakedMap::Normal), texturecraft_engine::PreviewShow::Material] {
            app.session.bake_preview_show(show).unwrap();
            assert!(bake_visualization(app).is_none());
        }

        app.cancel_bake();

        assert_eq!(bake_visualization(app), Some(before));
    }

    const VIEW: Rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 100.0));
    const DASH_PERIOD: f32 = RUBBER_BAND_DASH + RUBBER_BAND_GAP;

    #[test]
    fn a_far_start_is_clipped_to_the_view_on_a_whole_dash_period() {
        let far_start = pos2(-10_003.0, 50.0);

        let [start, end] = visible_dashes(VIEW, far_start, pos2(50.0, 50.0)).unwrap();

        assert!(start.x <= VIEW.left() && start.x > VIEW.left() - DASH_PERIOD, "starts at the view's edge: {start:?}");
        let periods = (start.x - far_start.x) / DASH_PERIOD;
        assert!((periods - periods.round()).abs() < 1e-3, "a whole number of dash periods from the real start: {periods}");
        assert_eq!(end, pos2(50.0, 50.0));
    }

    #[test]
    fn a_segment_inside_the_view_is_kept_whole() {
        assert_eq!(visible_dashes(VIEW, pos2(10.0, 20.0), pos2(90.0, 80.0)), Some([pos2(10.0, 20.0), pos2(90.0, 80.0)]));
    }

    #[test]
    fn a_segment_outside_the_view_draws_nothing() {
        assert_eq!(visible_dashes(VIEW, pos2(-50.0, -10.0), pos2(150.0, -10.0)), None);
        assert_eq!(visible_dashes(VIEW, pos2(-50.0, 120.0), pos2(120.0, 300.0)), None);
    }
}
