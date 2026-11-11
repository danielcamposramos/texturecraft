//! Context controls, view and mode buttons, tool shelf, icon rail and status bar

use egui::{RichText, vec2};
use egui_phosphor::regular as ic;
use serde_json::json;
use texturecraft_engine::Tool;

use crate::TextureCraftApp;
use crate::docking::Panel;
use crate::i18n::{tr, trf};
use crate::keymap::{shortcut_of, tip};
use crate::state::{Mode, ViewLayout};
use crate::theme::{GUTTER, Tokens, card};
use crate::widgets::{bar_slider, icon_button, icon_button_without_tooltip, icon_dropdown_button, icon_dropdown_button_width};

const BAR_SPACING: f32 = 10.0;
const COMPACT_BAR_SPACING: f32 = 4.0;
const BAR_SEPARATOR_WIDTH: f32 = 6.0;

pub fn tool_icon(t: Tool) -> &'static str {
    match t {
        Tool::Paint => ic::PAINT_BRUSH,
        Tool::Eraser => ic::ERASER,
        Tool::Projection => ic::PROJECTOR_SCREEN,
        Tool::PolygonFill => ic::POLYGON,
        Tool::Smudge => ic::FINGERPRINT,
        Tool::Clone => ic::STAMP,
        Tool::ColorPicker => ic::EYEDROPPER,
        Tool::Path => ic::PATH,
        Tool::GeometryMask => ic::CUBE_FOCUS,
    }
}

/// Keep mode buttons visible, with painting controls scrolling as space runs out
pub fn paint_bar(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let tokens = Tokens::get(ui.ctx());
    egui::Panel::top("paint_bar").exact_size(40.0).show_separator_line(false).frame(egui::Frame::NONE).show(ui, |ui| {
        card(ui, &tokens, egui::Margin { bottom: GUTTER, ..Default::default() }, egui::Margin::symmetric(8, 0), |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let button_size = ui.spacing().interact_size.y;
                let mode_width = 3.0 * button_size + 2.0 * BAR_SPACING;
                let separator_width = BAR_SEPARATOR_WIDTH + 2.0 * BAR_SPACING;
                // Leave room for both scroll arrows and at least one button of content
                let minimum_strip_width = 3.0 * button_size + 2.0 * BAR_SPACING;
                let scroll_views =
                    ui.available_width() < mode_width + view_buttons_width(button_size, BAR_SPACING) + 2.0 * separator_width + minimum_strip_width;
                ui.spacing_mut().item_spacing.x = if scroll_views { COMPACT_BAR_SPACING } else { BAR_SPACING };
                mode_buttons(app, ui);

                if app.ui.mode != Mode::Painting {
                    return;
                }

                ui.add(egui::Separator::default().spacing(BAR_SEPARATOR_WIDTH));
                if !scroll_views {
                    view_buttons(app, ui);
                    ui.add(egui::Separator::default().spacing(BAR_SEPARATOR_WIDTH));
                }

                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    scrolling_tool_controls(app, ui, &tokens, scroll_views);
                });
            });
        });
    });
}

/// Width of `view_buttons`, so the bar knows when to move them into the scrolling strip
fn view_buttons_width(button_size: f32, spacing: f32) -> f32 {
    // Two icons, two dropdowns, two separators and the five gaps between them
    2.0 * button_size + 2.0 * icon_dropdown_button_width(button_size) + 2.0 * BAR_SEPARATOR_WIDTH + 5.0 * spacing
}

/// View buttons in a right-to-left layout, with the screenshot button at the end
fn view_buttons(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let button_size = ui.spacing().interact_size.y;
    if icon_button(ui, ic::CAMERA, tr("Screenshot of the viewport"), false, button_size).clicked() {
        app.toast(tr("Use the control channel or `texturecraft-cli render` for screenshots"), false);
    }

    ui.add(egui::Separator::default().spacing(BAR_SEPARATOR_WIDTH));
    projection_dropdown(app, ui);
    layout_dropdown(app, ui);

    ui.add(egui::Separator::default().spacing(BAR_SEPARATOR_WIDTH));
    let checker = app.session.view == texturecraft_engine::ViewMode::UvChecker;
    if icon_button(ui, ic::CHECKERBOARD, &tip(&app.ui.keymap, tr("UV checker"), "ui.uvChecker"), checker, button_size).clicked() {
        let mode = if checker { "material" } else { "uvChecker" };
        app.run("viewport.view", json!({"mode": mode}));
    }
}

/// The 3D/2D, 3D-only and 2D-only layouts, showing the current one's icon
fn layout_dropdown(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    const LAYOUTS: [(ViewLayout, &str, &str, &str); 3] = [
        (ViewLayout::Split, ic::SQUARE_SPLIT_HORIZONTAL, "3D/2D views", "ui.layout.split"),
        (ViewLayout::Only3D, ic::SPHERE, "3D view only", "ui.layout.3d"),
        (ViewLayout::Only2D, ic::SQUARE, "2D view only", "ui.layout.2d"),
    ];

    let (_, icon, label, command) = LAYOUTS.iter().find(|(layout, ..)| *layout == app.ui.layout).unwrap_or(&LAYOUTS[0]);
    let tooltip = tip(&app.ui.keymap, tr(label), command);
    let mut chosen = None;
    icon_dropdown(ui, "view_layout", icon, &tooltip, |ui| {
        for (layout, icon, label, command) in LAYOUTS {
            if dropdown_row(ui, icon, tr(label), shortcut_of(&app.ui.keymap, command), app.ui.layout == layout) {
                chosen = Some(command);
            }
        }
    });

    if let Some(command) = chosen {
        crate::menus::invoke(app, &ui.ctx().clone(), command);
    }
}

/// Perspective or orthographic projection, the same setting as the Display Settings checkbox
fn projection_dropdown(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let projections = [(false, ic::PERSPECTIVE, tr("Perspective")), (true, ic::CUBE, tr("Orthographic"))];
    let ortho = app.session.camera.ortho;
    let (_, icon, label) = projections.iter().find(|(choice, ..)| *choice == ortho).unwrap_or(&projections[0]);
    let mut chosen = None;
    icon_dropdown(ui, "view_projection", icon, label, |ui| {
        for (choice, icon, label) in projections {
            if dropdown_row(ui, icon, label, None, ortho == choice) {
                chosen = Some(choice);
            }
        }
    });

    if let Some(choice) = chosen.filter(|choice| *choice != ortho) {
        app.run("camera.set", json!({"ortho": choice}));
    }
}

/// An icon button with a caret that opens `menu` below it, highlighted while open
fn icon_dropdown(ui: &mut egui::Ui, id_salt: &str, icon: &str, tooltip: &str, menu: impl FnOnce(&mut egui::Ui)) {
    let button_size = ui.spacing().interact_size.y;
    let popup_id = ui.make_persistent_id(id_salt);
    let open = egui::Popup::is_id_open(ui.ctx(), popup_id);
    let mut response = icon_dropdown_button(ui, icon, tooltip, open, button_size);
    if !open {
        response = response.on_hover_text(tooltip);
    }

    egui::Popup::menu(&response).id(popup_id).show(menu);
}

/// One dropdown entry: icon, label and shortcut. Returns true when clicked
fn dropdown_row(ui: &mut egui::Ui, icon: &str, label: &str, shortcut: Option<&str>, selected: bool) -> bool {
    let mut button = egui::Button::new(format!("{icon}  {label}")).selected(selected);
    if let Some(shortcut) = shortcut {
        button = button.shortcut_text(shortcut);
    }

    ui.add(button).clicked()
}

/// Scroll tool controls and, in narrow viewports, the view buttons
fn scrolling_tool_controls(app: &mut TextureCraftApp, ui: &mut egui::Ui, tokens: &Tokens, scroll_views: bool) {
    let button_size = ui.spacing().interact_size.y;
    let strip_width = ui.available_width();
    let overflow_id = egui::Id::unique("tool_controls_overflow");
    let overflowing = ui.data(|data| data.get_temp::<bool>(overflow_id)).unwrap_or(false);

    // A positive delta moves the content right, revealing its start
    let arrow_scroll_step = strip_width / 2.0;
    let mut scroll_delta = 0.0;
    if overflowing && icon_button_without_tooltip(ui, ic::CARET_DOUBLE_LEFT, "Scroll left", false, button_size).clicked() {
        scroll_delta = arrow_scroll_step;
    }

    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if overflowing && icon_button_without_tooltip(ui, ic::CARET_DOUBLE_RIGHT, "Scroll right", false, button_size).clicked() {
            scroll_delta = -arrow_scroll_step;
        }

        let scroll_area = egui::ScrollArea::horizontal()
            .id_salt("tool_controls")
            .min_scrolled_width(0.0)
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden);
        let output = ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
            scroll_area.show(ui, |ui| {
                ui.scroll_with_delta(vec2(scroll_delta, 0.0));
                ui.horizontal_centered(|ui| {
                    tool_controls(app, ui, tokens);
                    if scroll_views {
                        ui.add(egui::Separator::default().spacing(BAR_SEPARATOR_WIDTH));
                        let size = vec2(view_buttons_width(button_size, ui.spacing().item_spacing.x), ui.available_height());
                        ui.allocate_ui_with_layout(size, egui::Layout::right_to_left(egui::Align::Center), |ui| view_buttons(app, ui));
                    }
                });
            })
        });

        // The arrows take room from the strip, so measure the overflow against its full width
        let overflows_now = output.inner.content_size.x > strip_width;
        if overflows_now != overflowing {
            ui.data_mut(|data| data.insert_temp(overflow_id, overflows_now));
            ui.ctx().request_repaint();
        }
    });
}

/// Largest brush radius as a fraction of the mesh diagonal; size sliders show it as 100
pub(crate) const MAX_BRUSH_SIZE: f32 = 0.25;

/// Smallest brush radius as a fraction of the mesh diagonal
pub(crate) const MIN_BRUSH_SIZE: f32 = 0.0005;

/// The selected tool's icon and its settings
fn tool_controls(app: &mut TextureCraftApp, ui: &mut egui::Ui, tokens: &Tokens) {
    const DEFAULT_LAZY_DISTANCE_PX: f32 = 20.0;
    const TOOL_ICON_SIZE: f32 = 15.0;

    let button_size = ui.spacing().interact_size.y;
    let tool = app.session.tool.tool;
    ui.label(RichText::new(tool_icon(tool)).size(TOOL_ICON_SIZE).color(tokens.text_strong)).on_hover_text(tr(tool.label()));

    let brush = &mut app.session.tool.brush;
    match tool {
        Tool::Paint | Tool::Eraser | Tool::Projection | Tool::Smudge | Tool::Clone | Tool::Path => {
            let mut size = brush.size / MAX_BRUSH_SIZE * 100.0;
            if bar_slider(ui, tr("Size"), &mut size, 0.05..=100.0, 110.0).changed() {
                brush.size = (size / 100.0 * MAX_BRUSH_SIZE).clamp(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE);
            }

            let mut flow = brush.flow * 100.0;
            if bar_slider(ui, tr("Flow"), &mut flow, 0.0..=100.0, 100.0).changed() {
                brush.flow = flow / 100.0;
            }

            let mut opacity = brush.opacity * 100.0;
            if bar_slider(ui, tr("Stroke opacity"), &mut opacity, 0.0..=100.0, 110.0).changed() {
                brush.opacity = opacity / 100.0;
            }

            let mut spacing = brush.spacing * 100.0;
            if bar_slider(ui, tr("Spacing"), &mut spacing, 1.0..=200.0, 100.0).changed() {
                brush.spacing = spacing / 100.0;
            }

            ui.separator();
            let lazy_on = brush.lazy > 0.0;
            if icon_button(ui, ic::SCRIBBLE, tr("Lazy mouse"), lazy_on, button_size).clicked() {
                brush.lazy = if lazy_on { 0.0 } else { DEFAULT_LAZY_DISTANCE_PX };
            }
            if lazy_on {
                let mut distance = brush.lazy;
                if bar_slider(ui, tr("Distance"), &mut distance, 1.0..=200.0, 90.0).changed() {
                    brush.lazy = distance;
                }
            }

            ui.separator();
            if icon_button(ui, ic::CLOUD_RAIN, tr("Particle brush: particles slide down the surface"), brush.particles.enabled, button_size).clicked() {
                brush.particles.enabled = !brush.particles.enabled;
            }
            if crate::symmetry::applies_to(tool) {
                crate::symmetry::toolbar_controls(app, ui);
            }

            if tool == Tool::Smudge {
                ui.separator();
                let mut strength = app.session.tool.smudge_strength * 100.0;
                if bar_slider(ui, tr("Strength"), &mut strength, 0.0..=100.0, 100.0).changed() {
                    app.session.tool.smudge_strength = (strength / 100.0).clamp(0.0, 1.0);
                }
            }

            if tool == Tool::Clone && app.session.tool.clone_source.is_none() {
                ui.separator();
                ui.label(RichText::new(tr("Hold V and click the mesh to set the clone source")).color(tokens.text_dim));
            }

            if tool == Tool::Path {
                ui.separator();
                path_controls(app, ui);
            }
        }
        Tool::PolygonFill => fill_mode_buttons(app, ui),
        Tool::ColorPicker => {
            ui.label(RichText::new(tr("Click the mesh to pick its material into the brush")).color(tokens.text_dim));
        }
        Tool::GeometryMask => {
            ui.label(RichText::new(tr("Click a mesh part to show or hide the selected layer on it.")).color(tokens.text_dim));
        }
    }
}

/// Bake, Painting and Rendering mode buttons for a right-to-left layout, so Rendering is added first
fn mode_buttons(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let button_size = ui.spacing().interact_size.y;
    for (mode, icon, label, command) in [
        (Mode::Rendering, ic::APERTURE, tr("Rendering (path traced)"), "ui.mode.render"),
        (Mode::Painting, ic::PALETTE, tr("Painting"), "ui.mode.paint"),
        (Mode::Baking, ic::ATOM, tr("Bake Mesh Maps"), "ui.mode.bake"),
    ] {
        if icon_button(ui, icon, &tip(&app.ui.keymap, label, command), app.ui.mode == mode, button_size).clicked() {
            crate::menus::invoke(app, &ui.ctx().clone(), command);
        }
    }
}

/// Polygon fill's fill modes (top bar and properties)
pub fn fill_mode_buttons(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    use texturecraft_engine::FillMode;

    let button_size = ui.spacing().interact_size.y;
    for (fill_mode, icon, tooltip) in [
        (FillMode::Triangle, ic::TRIANGLE, tr("Triangle fill")),
        (FillMode::Polygon, ic::PARALLELOGRAM, tr("Polygon fill")),
        (FillMode::Mesh, ic::CUBE, tr("Mesh fill")),
        (FillMode::UvIsland, ic::CHECKERBOARD, tr("UV island fill")),
    ] {
        if icon_button(ui, icon, tooltip, app.session.tool.fill_mode == fill_mode, button_size).clicked() {
            app.session.tool.fill_mode = fill_mode;
        }
    }
}

/// Path tool options and actions (top bar and properties)
pub fn path_controls(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let button_size = ui.spacing().interact_size.y;
    let path = &app.session.tool.path;
    let (closed, fill) = (path.closed, path.fill);
    let selected = path.selected.is_some_and(|i| i < path.points.len());
    let has_points = !path.points.is_empty();
    if icon_button(ui, ic::POLYGON, tr("Closed path"), closed, button_size).clicked() {
        app.session.set_path_closed(!closed);
    }

    let fill_tip = if closed { tr("Fill the closed path (lasso)") } else { tr("Fill the path (close it first)") };
    if icon_button(ui, ic::PAINT_BUCKET, fill_tip, fill, button_size).clicked() {
        app.session.set_path_fill(!fill);
    }

    if ui.add_enabled_ui(selected, |ui| icon_button(ui, ic::BACKSPACE, tr("Remove point"), false, button_size)).inner.clicked() {
        app.session.remove_selected_path_point();
    }

    if ui.add_enabled_ui(has_points, |ui| icon_button(ui, ic::CHECK, tr("Apply"), false, button_size)).inner.clicked() {
        app.run("tool.path", json!({"commit": true}));
    }
}

pub fn tool_shelf(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let tokens = Tokens::get(ui.ctx());
    egui::Panel::left("tool_shelf").resizable(false).exact_size(40.0).show_separator_line(false).frame(egui::Frame::NONE).show(ui, |ui| {
        card(ui, &tokens, egui::Margin { left: GUTTER, right: 0, top: GUTTER, bottom: GUTTER }, egui::Margin::symmetric(3, 6), |ui| {
            ui.spacing_mut().item_spacing.y = 3.0;
            ui.add_enabled_ui(app.ui.mode == Mode::Painting, |ui| {
                for tool in Tool::ALL {
                    let selected = app.ui.mode == Mode::Painting && app.session.tool.tool == tool;
                    let hint = tip(&app.ui.keymap, tr(tool.label()), &format!("tool.{}", tool.id()));
                    if icon_button(ui, tool_icon(tool), &hint, selected, 28.0).clicked() {
                        app.run("tool.select", json!({"tool": tool.id()}));
                    }
                }

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);
                let quick_mask = &app.session.quick_mask;
                for (command, icon, label, selected) in [
                    ("viewport.quickMask", ic::SELECTION_BACKGROUND, tr("Quick mask"), quick_mask.enabled),
                    ("viewport.quickMask.edit", ic::SELECTION, tr("Edit quick mask"), quick_mask.editing),
                    ("viewport.quickMask.invert", ic::SELECTION_INVERSE, tr("Invert quick mask"), quick_mask.inverted),
                ] {
                    if icon_button(ui, icon, &tip(&app.ui.keymap, label, command), selected, 28.0).clicked() {
                        app.run(command, json!({}));
                    }
                }

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);
                if icon_button(ui, ic::SPHERE, tr("Add smart material from the shelf"), false, 28.0).clicked() {
                    app.ui.dock.focus(Panel::Assets);
                }
            });
        });
    });
}

/// Width of the icon rail beside the dock
const RAIL_WIDTH: f32 = 38.0;

/// Size of the rail's icons
const RAIL_ICON_SIZE: f32 = 26.0;

/// Gap between the rail's icons
const RAIL_ICON_GAP: f32 = 4.0;

/// Padding inside the rail's card around its icons, as the tool shelf's
const RAIL_CARD_PADDING: egui::Margin = egui::Margin::symmetric(3, 6);

/// The icon rail: Export, then the rail panels that are not in the dock, each opening its popup
pub fn rail(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let tokens = Tokens::get(ui.ctx());
    egui::Panel::right("rail").resizable(false).exact_size(RAIL_WIDTH).show_separator_line(false).frame(egui::Frame::NONE).show(ui, |ui| {
        card(ui, &tokens, egui::Margin { left: 0, right: GUTTER, top: GUTTER, bottom: GUTTER }, RAIL_CARD_PADDING, |ui| {
            ui.spacing_mut().item_spacing.y = RAIL_ICON_GAP;
            crate::rail_popup::begin_rail(ui.ctx(), ui.max_rect());
            if icon_button(ui, ic::EXPORT, &tip(&app.ui.keymap, tr("Export textures"), "ui.export"), app.ui.windows.export, RAIL_ICON_SIZE).clicked() {
                app.ui.windows.export = !app.ui.windows.export;
            }

            // A panel in the dock, docked or floating, leaves the rail until it is closed there
            for (panel, icon, id) in crate::rail_popup::RAIL_PANELS {
                if app.ui.dock.is_open(panel) {
                    continue;
                }

                let response = icon_button(ui, icon, &tip(&app.ui.keymap, tr(panel.name()), id), app.ui.rail_panel == Some(panel), RAIL_ICON_SIZE);
                crate::rail_popup::remember_icon(ui.ctx(), panel, response.rect);
                if response.clicked() {
                    crate::rail_popup::toggle(app, panel);
                }
            }
        });
    });
}

pub fn status_bar(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let tk = Tokens::get(ui.ctx());
    egui::Panel::bottom("status_bar")
        .exact_size(22.0)
        .show_separator_line(false)
        .frame(egui::Frame::NONE.fill(tk.chrome).inner_margin(egui::Margin::symmetric(10, 0)))
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                if let Some(p) = app.bake_progress() {
                    ui.add(egui::ProgressBar::new(p).desired_width(160.0).desired_height(10.0).text(RichText::new(tr("Baking")).size(10.0)));
                    if ui.small_button(tr("Cancel")).clicked() {
                        app.cancel_bake();
                    }
                } else if let Some(job) = &app.resize {
                    let bar = egui::ProgressBar::new(job.progress.fraction()).desired_width(160.0).desired_height(10.0);
                    ui.add(bar.animate(true).text(RichText::new(tr("Resizing")).size(10.0)));
                } else if let Some(p) = app.gpu.as_ref().and_then(|g| g.composite_progress(&app.session)) {
                    ui.add(egui::ProgressBar::new(p).desired_width(160.0).desired_height(10.0).text(RichText::new(tr("Compositing")).size(10.0)));
                    ui.ctx().request_repaint();
                } else if let Some(msg) = app.session.log.last() {
                    ui.label(RichText::new(msg).size(11.0).color(tk.text_dim));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let gpu = app.gpu.as_ref().map(|g| g.adapter.as_str()).unwrap_or(tr("no GPU"));
                    ui.label(
                        RichText::new(trf("Version: {version} ({gpu})", &[("version", &texturecraft_engine::VERSION), ("gpu", &gpu)]))
                            .size(11.0)
                            .color(tk.text_dim),
                    );
                    ui.separator();
                    let m = app.session.memory();
                    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
                    let total = mb(m.total());
                    let decimals = if total < 10.0 { 1 } else { 0 };
                    let mb1 = |b: usize| format!("{:.1}", mb(b));
                    ui.label(RichText::new(trf("Layer memory: {mb} MB", &[("mb", &format!("{total:.decimals$}"))])).size(11.0).color(tk.text_dim)).on_hover_ui(
                        |ui| {
                            ui.label(trf(
                                "Painted layer pixels {pixels} MB, composited channels {composites} MB, compositing cache {cache} MB",
                                &[("pixels", &mb1(m.pixels)), ("composites", &mb1(m.composites)), ("cache", &mb1(m.cache))],
                            ));
                        },
                    );
                    ui.separator();
                    ui.label(RichText::new(trf("UI {ms} ms", &[("ms", &format!("{:.1}", app.ui_ms))])).size(11.0).color(tk.text_dim));
                    ui.add_space(vec2(0.0, 0.0).x);
                });
            });
        });
}
